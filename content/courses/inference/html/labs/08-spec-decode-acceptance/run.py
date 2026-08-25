#!/usr/bin/env python3
"""Lab 08 — speculative decoding: acceptance, and where it stops paying.

Two measurements that people routinely conflate:

  A. ACCEPT   Mean acceptance length E per workload type, holding the model and
              kappa fixed. This is a property of the workload, not of the
              engine, and it is the only input to the speedup model you cannot
              compute on paper.

  B. SWEEP    The batch size at which speculation stops paying for itself: run
              the same concurrency sweep against a speculative server and a
              plain one, and find where the OUTPUT-TOKEN THROUGHPUT ratio
              crosses 1.0. Latency improves at every batch size; that is not
              the question.

  C. MODEL    Print the 6.2 arithmetic that predicts B from E, so you have a
              prediction before you measure. `model` runs anywhere: no GPU, no
              engine, no network.

Acceptance is never recomputed here from token counts. It is read from what the
engine reports, because both engines include the bonus token in the length and
exclude it from the rate, and reimplementing that convention is a reliable way
to be off by exactly one.

    python3 run.py model --alpha 0.5 0.6 0.7 0.8 --kappa 1 2 3 5
    python3 run.py accept --categories summarization translation qa writing
    python3 run.py sweep  --concurrency 8 16 32 64 128 192 256 --dry-run
    python3 run.py read   --results ./results

Reference points at the SHAs this book pins:
  vLLM   counter names                 vllm/v1/spec_decode/metrics.py:L227-L232
  vLLM   PromQL for every derived stat vllm/v1/spec_decode/metrics.py:L178-L196
  vLLM   harness diffs them for you    vllm/benchmarks/serve.py:L1107-L1119
  vLLM   result JSON field names       vllm/benchmarks/serve.py:L1300-L1308
  vLLM   per-request opt-in            vllm/config/observability.py:L48-L58
  vLLM   synthetic acceptance          vllm/config/speculative.py:L219-L241
  SGLang per-request field             python/sglang/srt/entrypoints/openai/protocol.py:L418-L427
  SGLang decode-log accept len/rate    .../scheduler_components/metrics_reporter.py:L818-L830
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Roofline ridge in token positions for bf16 on an H100 SXM (0.4). In bf16 a
# parameter costs two bytes and buys two FLOPs, so arithmetic intensity is
# numerically the number of token positions in the step -- which is why a ridge
# in FLOP/byte reads directly as a ridge in positions.
DEFAULT_RIDGE = 295.0

# Draft step cost as a fraction of the target step, derived in 6.2 from the
# ratio of streamed weight bytes. Llama-3.2-1B against Llama-3-8B.
DRAFT_COST = {"1b-draft": 0.165, "ngram": 0.0}

# Result-JSON keys written by vllm/benchmarks/serve.py:L1300-L1308.
VLLM_KEYS = {
    "length": "spec_decode_acceptance_length",
    "rate": "spec_decode_acceptance_rate",
    "per_pos": "spec_decode_per_position_acceptance_rates",
    "out_tput": "output_throughput",
    "req_tput": "request_throughput",
}


def table(rows, title):
    if not rows:
        return
    cols = len(rows[0])
    w = [max(len(str(r[i])) for r in rows) for i in range(cols)]
    print(f"\n{title}")
    print("-" * (sum(w) + 2 * (cols - 1)))
    for n, row in enumerate(rows):
        print("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i])
                        for i, c in enumerate(row)))
        if n == 0:
            print("-" * (sum(w) + 2 * (cols - 1)))


def print_environment():
    import platform  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    rows = [("field", "value"),
            ("host", platform.node()),
            ("python", platform.python_version()),
            ("vllm on PATH", shutil.which("vllm") or "no")]
    try:
        import torch  # noqa: PLC0415
        rows.append(("torch", torch.__version__))
        rows.append(("gpu", torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "none visible"))
    except ImportError:
        rows.append(("torch", "not installed (fine for `model`)"))
    table(rows, "Environment")


# ------------------------------------------------------------------ C: model


def expected_tokens(alpha: float, kappa: int) -> float:
    """E = sum_{i=0..kappa} alpha^i, the i.i.d. mean acceptance length,
    bonus token included. 6.2 derives it; it is capped at kappa + 1."""
    if alpha >= 1.0:
        return float(kappa + 1)
    return (1.0 - alpha ** (kappa + 1)) / (1.0 - alpha)


def alpha_from_E(E: float, kappa: int) -> float:
    """Invert the above numerically. Useful for turning a MEASURED E into the
    alpha the i.i.d. model would have needed -- and the gap between that alpha
    and the measured per-position vector is exercise 2."""
    lo, hi = 0.0, 0.999999
    if E <= 1.0:
        return 0.0
    if E >= kappa + 1:
        return 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expected_tokens(mid, kappa) < E:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def speedup_at_batch(B, E, kappa, c, ridge):
    """S(B) from 6.2: the drafted positions are free only while the whole step
    is bandwidth-bound. Past the ridge every max() cancels and S < 1 always."""
    plain = max(1.0, B / ridge)
    verify = max(1.0, B * (kappa + 1) / ridge)
    draft = kappa * c * max(1.0, B / ridge)
    return E * plain / (verify + draft)


def breakeven_batch(E, kappa, c, ridge):
    return ridge * (E - kappa * c) / (kappa + 1)


def cmd_model(args):
    c = args.draft_cost if args.draft_cost is not None else DRAFT_COST[args.proposer]
    rows = [("alpha", "kappa", "E", "S at batch 1", "S asymptotic", "break-even B")]
    for alpha in args.alpha:
        for k in args.kappa:
            E = expected_tokens(alpha, k)
            s1 = E / (1.0 + k * c)
            sinf = E / ((k + 1) + k * c)
            rows.append((f"{alpha:.2f}", str(k), f"{E:.3f}", f"{s1:.2f}x",
                         f"{sinf:.2f}x", f"{breakeven_batch(E, k, c, args.ridge):.0f}"))
    table(rows, f"Derived from the 6.2 model. proposer={args.proposer} (c={c}), "
                f"ridge T*={args.ridge:g} positions. alpha is SWEPT, not measured.")

    if args.measured_E is not None:
        k = args.kappa[0]
        a = alpha_from_E(args.measured_E, k)
        B = breakeven_batch(args.measured_E, k, c, args.ridge)
        table([("field", "value"),
               ("measured E", f"{args.measured_E:.3f}"),
               ("kappa", str(k)),
               ("implied alpha (i.i.d.)", f"{a:.3f}"),
               ("predicted break-even B", f"{B:.0f} concurrent sequences"),
               ("speedup at batch 1", f"{args.measured_E / (1 + k * c):.2f}x")],
              "Your measured E, run through the model")
        print("\nThe implied alpha is what the i.i.d. model would need. Compare it")
        print("against the per-position vector the engine reports: real acceptance")
        print("decays faster than geometric, and the gap is why the optimal kappa")
        print("is smaller than the geometric table suggests.")

    print("\nEverything above is arithmetic. Nothing was measured.")
    print("Costs the model does NOT charge for, all of which move the crossing LEFT:")
    for line in [
        "  - KV slots reserved for drafting, per request, per step (6.2)",
        "  - the sampler's top-k/top-p sort, multiplied by kappa+1 rows (6.2)",
        "  - the proposer's own time, which the model folds into a single c",
        "  - CUDA-graph bucketing across the concurrency sweep (8.1)",
    ]:
        print(line)
    return 0


# ------------------------------------------------------------------ A: accept


def bench_serve_cmd(args, extra, out):
    cmd = ["vllm", "bench", "serve",
           "--model", args.model,
           "--base-url", args.base_url,
           "--backend", args.backend,
           "--num-prompts", str(args.num_prompts),
           "--seed", str(args.seed),
           "--save-result", "--result-filename", str(out)]
    return cmd + extra + args.extra_bench_args


def run_or_show(cmd, dry):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry:
        return True
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0:
        print(f"  FAILED with exit code {rc}")
    return rc == 0


def cmd_accept(args):
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if not args.dataset_path and not args.dry_run:
        sys.exit("`accept` needs --dataset-path: the Spec-Bench question.jsonl. "
                 "See vllm/docs/benchmarking/cli.md for the download line.")
    for cat in args.categories:
        out = work / f"accept-{cat}.json"
        extra = ["--dataset-name", args.dataset_name,
                 "--dataset-path", args.dataset_path or "<question.jsonl>",
                 "--request-rate", str(args.request_rate),
                 "--temperature", str(args.temperature)]
        if args.dataset_name == "spec_bench":
            extra += ["--spec-bench-category", cat]
        else:
            extra += ["--speed-bench-category", cat]
        run_or_show(bench_serve_cmd(args, extra, out), args.dry_run)
    if args.dry_run:
        print("\n--dry-run: nothing ran. Nothing was measured.")
        return 0
    return read_results(work, "accept-", args)


# ------------------------------------------------------------------ B: sweep


def cmd_sweep(args):
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    for c in args.concurrency:
        out = work / f"{args.arm}-c{c}.json"
        extra = ["--dataset-name", "random",
                 "--random-input-len", str(args.input_len),
                 "--random-output-len", str(args.output_len),
                 "--max-concurrency", str(c),
                 "--num-prompts", str(c * args.prompts_per_slot),
                 "--temperature", str(args.temperature)]
        # --num-prompts appears twice if we do not strip the default one.
        base = [x for x in bench_serve_cmd(args, extra, out)]
        i = base.index("--num-prompts")
        del base[i:i + 2]
        run_or_show(base, args.dry_run)
    if args.dry_run:
        print("\n--dry-run: nothing ran.")
        print("Run this twice -- once with --arm spec against a speculative server,")
        print("once with --arm base against a plain one -- then `read` the pair.")
        return 0
    return read_results(work, f"{args.arm}-", args)


# ------------------------------------------------------------------ read


def load(path):
    d = json.loads(Path(path).read_text())
    return {k: d.get(v) for k, v in VLLM_KEYS.items()} | {"_file": path.name}


def read_results(work, prefix, args):
    files = sorted(Path(work).glob(f"{prefix}*.json"))
    if not files:
        print(f"\nNo {prefix}*.json under {work}.")
        return 1
    rows = [("result file", "E (accept len)", "accept rate %", "out tok/s", "per-position r_i")]
    for f in files:
        r = load(f)
        pp = r["per_pos"] or []
        rows.append((f.name,
                     f"{r['length']:.3f}" if r["length"] is not None else "n/a",
                     f"{r['rate']:.1f}" if r["rate"] is not None else "n/a",
                     f"{r['out_tput']:.1f}" if r["out_tput"] is not None else "n/a",
                     ", ".join(f"{x:.3f}" for x in pp) if pp else "n/a"))
    table(rows, "Measured - read out of the harness result JSON, not recomputed")
    print("\n'E' includes the bonus token; 'accept rate' does not. Do not compare")
    print("the RATE column against an SGLang run: SGLang divides correct drafts by")
    print("speculative_num_draft_tokens - 1 per round, which is a different ratio.")
    print("The LENGTH column is comparable across engines.")
    return 0


def cmd_read(args):
    work = Path(args.results)
    rc = 0
    for prefix in ("accept-", "spec-", "base-"):
        if list(work.glob(f"{prefix}*.json")):
            rc |= read_results(work, prefix, args)
    # If both arms of the sweep are present, print the crossing.
    spec = {int(f.stem.split("-c")[1]): load(f) for f in work.glob("spec-c*.json")}
    base = {int(f.stem.split("-c")[1]): load(f) for f in work.glob("base-c*.json")}
    both = sorted(set(spec) & set(base))
    if both:
        rows = [("concurrency", "base tok/s", "spec tok/s", "ratio", "verdict")]
        crossing = None
        prev = None
        for c in both:
            b, s = base[c]["out_tput"], spec[c]["out_tput"]
            if not b or not s:
                continue
            ratio = s / b
            rows.append((str(c), f"{b:.1f}", f"{s:.1f}", f"{ratio:.3f}",
                         "spec wins" if ratio > 1 else "spec LOSES"))
            if prev and prev[1] > 1.0 >= ratio:
                # linear interpolation between the two bracketing points
                c0, r0 = prev
                crossing = c0 + (c - c0) * (r0 - 1.0) / (r0 - ratio)
            prev = (c, ratio)
        table(rows, "Break-even sweep - measured output-token throughput, both arms")
        if crossing:
            print(f"\nCrossing interpolated at B* = {crossing:.0f} concurrent sequences.")
            print("Compare against `run.py model --measured-E <your E>`. If the measured")
            print("crossing sits LEFT of the derived one, the difference is a cost the")
            print("6.2 model does not charge for -- see the list it prints.")
        else:
            print("\nNo crossing inside the swept range. Either speculation still pays at")
            print("your largest concurrency (extend the sweep) or it never paid (check E).")
    return rc


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure speculative-decoding acceptance, and find where it stops paying.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="model id, passed to the harness as --model")
        p.add_argument("--base-url", default="http://127.0.0.1:8000",
                       help="server to point the harness at (default: http://127.0.0.1:8000)")
        p.add_argument("--backend", default="vllm",
                       help="harness backend name; 'vllm' or 'openai' (default: vllm)")
        p.add_argument("--workdir", default="./results",
                       help="where result JSON lands (default: ./results). Nothing is "
                            "ever written into the engine checkouts.")
        p.add_argument("--num-prompts", type=int, default=200,
                       help="requests per point (default: 200)")
        p.add_argument("--seed", type=int, default=1, help="dataset seed (default: 1)")
        p.add_argument("--temperature", type=float, default=0.0,
                       help="sampling temperature; raising it lowers acceptance because "
                            "alpha = 1 - D_TV(p, q). Default 0.")
        p.add_argument("--dry-run", action="store_true",
                       help="print every command that would run, run none of them")
        p.add_argument("--extra-bench-args", nargs=argparse.REMAINDER, default=[],
                       help="everything after this flag is passed to vllm bench serve verbatim")

    m = sub.add_parser("model", help="Part C: the 6.2 arithmetic. No GPU, no engine.")
    m.add_argument("--alpha", type=float, nargs="+",
                   default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   help="per-token acceptance rates to sweep (default: 0.4 .. 0.9)")
    m.add_argument("--kappa", type=int, nargs="+", default=[3],
                   help="drafted tokens per step (default: 3)")
    m.add_argument("--proposer", choices=sorted(DRAFT_COST), default="ngram",
                   help="sets the draft cost c: ngram=0, 1b-draft=0.165 (default: ngram)")
    m.add_argument("--draft-cost", type=float, default=None,
                   help="override c directly, as a fraction of one target step")
    m.add_argument("--ridge", type=float, default=DEFAULT_RIDGE,
                   help="roofline ridge in token positions (default: 295, H100 SXM bf16)")
    m.add_argument("--measured-E", type=float, default=None,
                   help="feed a measured acceptance length back through the model")
    m.set_defaults(fn=cmd_model)

    a = sub.add_parser("accept", help="Part A: acceptance length per workload category.")
    common(a)
    a.add_argument("--dataset-name", default="spec_bench",
                   choices=("spec_bench", "speed_bench"),
                   help="spec_bench filters by task, speed_bench by output entropy")
    a.add_argument("--dataset-path", default="",
                   help="path to the downloaded question.jsonl / SPEED-Bench jsonl")
    a.add_argument("--categories", nargs="+",
                   default=["summarization", "translation", "qa", "writing"],
                   help="one run per category (default: summarization translation qa writing)")
    a.add_argument("--request-rate", type=float, default=4.0,
                   help="open-loop arrival rate; keep it well below saturation so the "
                        "acceptance figure is not confounded by batch size (default: 4)")
    a.set_defaults(fn=cmd_accept)

    s = sub.add_parser("sweep", help="Part B: concurrency sweep for one arm.")
    common(s)
    s.add_argument("--arm", choices=("spec", "base"), default="spec",
                   help="which server this sweep is pointed at; run both (default: spec)")
    s.add_argument("--concurrency", type=int, nargs="+",
                   default=[8, 16, 32, 64, 96, 128, 160, 192, 256],
                   help="--max-concurrency values to sweep")
    s.add_argument("--input-len", type=int, default=2048,
                   help="--random-input-len (default: 2048). Longer context keeps the "
                        "step bandwidth-bound and moves the crossing right.")
    s.add_argument("--output-len", type=int, default=256,
                   help="--random-output-len (default: 256)")
    s.add_argument("--prompts-per-slot", type=int, default=8,
                   help="requests per concurrency slot, so every point runs long "
                        "enough to reach steady state (default: 8)")
    s.set_defaults(fn=cmd_sweep)

    r = sub.add_parser("read", help="Read result JSON already on disk and tabulate it.")
    r.add_argument("--results", default="./results", help="directory of result JSON")
    r.set_defaults(fn=cmd_read)

    args = ap.parse_args()
    if args.cmd != "model":
        print_environment()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
