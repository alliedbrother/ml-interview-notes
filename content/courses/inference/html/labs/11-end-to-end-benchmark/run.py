#!/usr/bin/env python3
"""Lab 11 — an end-to-end benchmark you can defend.

Two engines, one harness, one arrival schedule, one seed, a warmup sized from
queueing theory, and a headline number that is GOODPUT: completed requests that
met an SLO stated in advance.

Three subcommands:

  plan    Size the experiment before you burn an afternoon. Queue relaxation
          time from the M/M/1 result, pipeline fill from Little's Law, and the
          minimum detectable effect for a given number of runs per arm. Pure
          arithmetic -- no GPU, no engine, no network.

  sweep   Emit and (optionally) run the full protocol for one arm: warm at the
          load you will measure, flush and CHECK THE BODY, then measure. One
          command per (load, repetition), in randomised order.

  report  Read the result JSON from both arms and print the comparison table,
          plus every reason the comparison might be invalid that can be checked
          from the files themselves.

This script never implements a load generator. It drives `vllm bench serve`,
which is the only one of the two projects' harnesses that computes goodput at
all (grep python/sglang/benchmark/ for "goodput" at 7d89325 and you get
nothing) -- and the same client drives an SGLang server through
`--backend openai`, because SGLang serves /v1/completions.

Nothing here writes into the pinned engine checkouts. Results land in --workdir.

    python3 run.py plan --mu 25 --rho 0.9
    python3 run.py plan --cv 0.03 --mde 0.04
    python3 run.py sweep --arm vllm --base-url http://127.0.0.1:8000 \
        --rates 8 12 16 20 22 24 --runs 3 --dry-run
    python3 run.py report --workdir ./results

Reference points at the SHAs this book pins:
  --goodput and its SLO names       vllm/benchmarks/serve.py:L1778-L1789, L1432-L1451
  goodput predicate                 vllm/benchmarks/serve.py:L622-L644
  result JSON keys                  vllm/benchmarks/serve.py:L1260-L1280, L1346-L1358
  arrival schedule + rescale        vllm/benchmarks/serve.py:L471-L484
  seeding                           vllm/benchmarks/serve.py:L1968-L1969
  backend dispatch table            vllm/benchmarks/lib/endpoint_request_func.py:L873-L877
  SGLang /v1/completions            python/sglang/srt/entrypoints/http_server.py:L1714-L1719
  harness warmup defaults           vllm/benchmarks/serve.py:L1692-L1697
                                    python/sglang/benchmark/serving.py:L2598-L2603
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# 3.96 = (z_0.975 + z_0.80) * sqrt(2); see 10.3.
MDE_CONST = 3.96

# Keys written by vllm/benchmarks/serve.py into --result-filename.
KEYS = {
    "duration": "duration",
    "completed": "completed",
    "failed": "failed",
    "req_tput": "request_throughput",
    "goodput": "request_goodput",
    "out_tput": "output_throughput",
    "ttft_p99": "p99_ttft_ms",
    "tpot_p99": "p99_tpot_ms",
    "e2el_p99": "p99_e2el_ms",
    "max_conc": "max_concurrent_requests",
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
    rows = [("field", "value"),
            ("host", platform.node()),
            ("python", platform.python_version()),
            ("platform", platform.platform()),
            ("vllm on PATH", shutil.which("vllm") or "NOT ON PATH"),
            ("curl on PATH", shutil.which("curl") or "NOT ON PATH"),
            ("nvidia-smi", shutil.which("nvidia-smi") or "not present")]
    try:
        import torch  # noqa: PLC0415
        rows.append(("torch", torch.__version__))
    except ImportError:
        rows.append(("torch", "not installed (fine for `plan`)"))
    table(rows, "Environment - client side. The SERVER's environment is the one "
                "that matters and neither harness records it; see `report`.")


# ------------------------------------------------------------------ plan


def relaxation(mu: float, rho: float):
    """M/M/1 approach to stationarity from an empty queue (10.3):

        t_rel = S / (1 - sqrt(rho))^2,     N_rel = lambda * t_rel

    S is the mean service time, 1/mu. This is the term that makes warmup
    expensive, and it blows up as rho -> 1."""
    S = 1.0 / mu
    denom = (1.0 - math.sqrt(rho)) ** 2
    t_rel = S / denom
    lam = rho * mu
    return S, lam, t_rel, lam * t_rel


def cmd_plan(args):
    if args.mu:
        rows = [("rho", "lambda", "t_rel", "3 t_rel", "requests in 3 t_rel", "mean wait W")]
        for rho in args.rho:
            S, lam, t_rel, n_rel = relaxation(args.mu, rho)
            W = S * rho / (1.0 - rho) + S  # M/M/1 sojourn time
            rows.append((f"{rho:.2f}", f"{lam:.1f} req/s", f"{t_rel:.1f} s",
                         f"{3 * t_rel:.0f} s", f"{3 * n_rel:,.0f}", f"{W * 1000:.0f} ms"))
        table(rows, f"Derived - warmup sizing at mu = {args.mu:g} req/s. "
                    "M/M/1 relaxation from an empty queue; nothing measured.")
        print("\nWarm AT the load you will measure, for at least 3 t_rel, THEN flush,")
        print("THEN discard L-bar = lambda*W more requests to refill the batch the")
        print("flush emptied. Both harnesses default to 0 or 1 warmup requests.")

        rho_hi = max(args.rho)
        _, lam_hi, _, _ = relaxation(args.mu, rho_hi)
        W_hi = (1.0 / args.mu) * rho_hi / (1.0 - rho_hi) + 1.0 / args.mu
        print(f"\nAt rho = {rho_hi:g}: L-bar = {lam_hi * W_hi:.0f} requests of pipeline refill,")
        print(f"then a measured window of at least {args.min_completions:,} completions for a p99")
        print(f"(a p99 from fewer sits on the tenth-worst sample; a p999 needs ~10x that).")

    if args.cv is not None:
        rows = [("runs per arm n", "MDE at your CV", "verdict")]
        for n in (3, 5, 10, 20):
            mde = MDE_CONST * args.cv / math.sqrt(n)
            verdict = "resolves your target" if args.mde and mde <= args.mde else ""
            rows.append((str(n), f"{100 * mde:.1f}%", verdict))
        table(rows, f"Derived - minimum detectable effect at CV = {100 * args.cv:.1f}%, "
                    "alpha = 0.05 two-sided, 80% power. There is no n=1 row: with one "
                    "run you have no sigma and therefore no claim.")
        if args.mde:
            n = math.ceil((MDE_CONST * args.cv / args.mde) ** 2)
            print(f"\nTo resolve {100 * args.mde:.1f}% at CV = {100 * args.cv:.1f}% you need "
                  f"n = {n} runs per arm.")
            print("Cheaper than more runs: drive the CV down. Flush between runs,")
            print("randomise the sweep order, and pin the server flags on both arms.")

    if not args.mu and args.cv is None:
        print("\nNothing to plan. Pass --mu (warmup sizing) or --cv (runs per arm).")
    print("\nEverything above is arithmetic. Nothing was measured.")
    return 0


# ------------------------------------------------------------------ sweep


def bench_cmd(args, rate, num_prompts, out, seed):
    cmd = ["vllm", "bench", "serve",
           "--model", args.model,
           "--backend", args.backend,
           "--base-url", args.base_url,
           "--dataset-name", args.dataset_name,
           "--random-input-len", str(args.input_len),
           "--random-output-len", str(args.output_len),
           "--request-rate", str(rate),
           "--burstiness", str(args.burstiness),
           "--num-prompts", str(num_prompts),
           "--seed", str(seed),
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,90,99",
           "--ignore-eos"]
    if args.goodput:
        cmd += ["--goodput"] + args.goodput
    if args.max_concurrency is not None:
        cmd += ["--max-concurrency", str(args.max_concurrency)]
    if out is not None:
        cmd += ["--metadata"] + args.metadata + [f"arm={args.arm}", f"rate={rate}"]
        cmd += ["--save-result", "--result-filename", str(out)]
    return cmd + args.extra_bench_args


def flush_cmd(args):
    """SGLang exposes /flush_cache as a first-class route and returns HTTP 400
    while requests are in flight. vLLM's /reset_prefix_cache exists only under
    VLLM_SERVER_DEV_MODE=1 and returns 200 with {"success": false} when it
    refuses. Both look like success to a script that checks only the exit code,
    which is why this prints the body."""
    path = "/flush_cache" if args.arm.startswith("sglang") else "/reset_prefix_cache"
    return ["curl", "-sS", "-w", "\\nHTTP %{http_code}\\n",
            "-X", "POST", args.base_url.rstrip("/") + path]


def run_or_show(cmd, dry, label=""):
    print(f"\n# {label}" if label else "")
    print("$ " + " ".join(str(c) for c in cmd))
    if dry:
        return True
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0:
        print(f"  FAILED, exit {rc}")
    return rc == 0


def cmd_sweep(args):
    if args.max_concurrency is not None and not args.allow_closed_loop:
        sys.exit("--max-concurrency turns this into a closed loop: the reported clock "
                 "starts after a client-side semaphore, so client queueing vanishes "
                 "from every latency percentile and therefore from goodput, and the "
                 "run can never reach rho > 1. If you mean it, pass "
                 "--allow-closed-loop; the limit is then stamped into every result "
                 "via --metadata.")
    if args.max_concurrency is not None:
        args.metadata = list(args.metadata) + [f"max_concurrency={args.max_concurrency}"]

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    clock_log = None
    if args.log_clocks:
        # Neither repository records clock, temperature or power in its result
        # file; see the Unverified callout in the README. So log it here.
        clock_log = work / f"{args.arm}-clocks.csv"
        cmd = ["nvidia-smi",
               "--query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw",
               "--format=csv", "-l", "1", "-f", str(clock_log)]
        print("\n# start this in another shell for the duration of the sweep:")
        print("$ " + " ".join(cmd))

    points = [(rate, rep) for rate in args.rates for rep in range(args.runs)]
    rng = random.Random(args.order_seed)
    rng.shuffle(points)  # so thermal drift does not alias onto the swept variable
    print(f"\n{len(points)} measured runs, order randomised with seed {args.order_seed}.")

    for rate, rep in points:
        seed_warm = args.seed * 1000 + rep * 2
        seed_meas = args.seed * 1000 + rep * 2 + 1
        # 2. warm AT the load about to be measured
        warm = bench_cmd(args, rate, args.warmup_prompts, None, seed_warm)
        run_or_show(warm, args.dry_run,
                    f"rate={rate} rep={rep}: WARM at the measured load "
                    f"({args.warmup_prompts} requests, result discarded)")
        # 3. flush, and read the body
        run_or_show(flush_cmd(args), args.dry_run,
                    "FLUSH - read the body, not just the exit code")
        # 4+5. measure
        out = work / f"{args.arm}-rate{rate}-run{rep}.json"
        run_or_show(bench_cmd(args, rate, args.num_prompts, out, seed_meas),
                    args.dry_run, "MEASURE")

    if args.dry_run:
        print("\n--dry-run: nothing ran, nothing was measured.")
        print("Read the commands above before letting them loose. In particular:")
        print("  - the warm run uses the SAME rate as the measured run;")
        print("  - the flush sits BETWEEN them, not before both;")
        print("  - the two arms must share --seed, --num-prompts, --request-rate")
        print("    and --burstiness, or they are not the same experiment.")
    return 0


# ------------------------------------------------------------------ report


def load(path):
    d = json.loads(Path(path).read_text())
    out = {k: d.get(v) for k, v in KEYS.items()}
    out["_file"] = Path(path).name
    return out


def parse_name(name):
    """<arm>-rate<r>-run<n>.json"""
    stem = Path(name).stem
    try:
        arm, rate, run = stem.split("-rate")[0], stem.split("-rate")[1].split("-run")[0], \
            stem.split("-run")[1]
        return arm, float(rate), int(run)
    except (IndexError, ValueError):
        return None, None, None


def cmd_report(args):
    files = sorted(Path(args.workdir).glob("*-rate*-run*.json"))
    if not files:
        print(f"No <arm>-rate<r>-run<n>.json under {args.workdir}. Run `sweep` first.")
        return 1

    by_point = {}
    for f in files:
        arm, rate, _ = parse_name(f)
        if arm is None:
            continue
        by_point.setdefault((arm, rate), []).append(load(f))

    rows = [("engine / arm", "lambda", "runs", "completed", "failed", "tput req/s",
             "goodput req/s", "good/total", "TTFT p99 ms", "TPOT p99 ms",
             "window s", "CV of tput")]
    no_goodput = []
    for (arm, rate) in sorted(by_point):
        rs = by_point[(arm, rate)]
        def col(k):
            vals = [r[k] for r in rs if r[k] is not None]
            return vals

        tputs = col("req_tput")
        goods = col("goodput")
        if not goods:
            no_goodput.append((arm, rate))
        mean = statistics.mean
        cv = (statistics.stdev(tputs) / mean(tputs)) if len(tputs) > 1 and mean(tputs) else None
        ratio = (mean(goods) / mean(tputs)) if goods and tputs else None
        rows.append((
            arm, f"{rate:g}", str(len(rs)),
            f"{mean(col('completed')):.0f}" if col("completed") else "-",
            f"{mean(col('failed')):.0f}" if col("failed") else "-",
            f"{mean(tputs):.2f}" if tputs else "-",
            f"{mean(goods):.2f}" if goods else "NONE",
            f"{100 * ratio:.1f}%" if ratio is not None else "-",
            f"{mean(col('ttft_p99')):.0f}" if col("ttft_p99") else "-",
            f"{mean(col('tpot_p99')):.0f}" if col("tpot_p99") else "-",
            f"{mean(col('duration')):.0f}" if col("duration") else "-",
            f"{100 * cv:.1f}%" if cv is not None else "n/a (n=1)",
        ))
    table(rows, "Measured - mean over runs per point, read from the harness result JSON")

    # What you may and may not claim.
    cvs = []
    for (arm, rate), rs in by_point.items():
        t = [r["req_tput"] for r in rs if r["req_tput"]]
        if len(t) > 1 and statistics.mean(t):
            cvs.append(statistics.stdev(t) / statistics.mean(t))
    n_min = min(len(v) for v in by_point.values())
    if cvs:
        cv = max(cvs)
        mde = MDE_CONST * cv / math.sqrt(n_min)
        print(f"\nWorst-case CV across points: {100 * cv:.1f}%. With n = {n_min} runs per")
        print(f"point that is an MDE of {100 * mde:.1f}%: you cannot defend any claim")
        print(f"smaller than that. To halve it you need 4x the runs, or a lower CV.")
    else:
        print("\nOnly one run per point, so there is no sigma and therefore no claim.")
        print("Re-run with --runs 3 at minimum.")

    if no_goodput:
        print("\nGOODPUT MISSING for: " + ", ".join(f"{a}@{r:g}" for a, r in no_goodput))
        print("request_goodput is null unless --goodput was passed. A throughput number")
        print("without the latency it was achieved at is a coordinate with one axis")
        print("missing; re-run with --goodput ttft:<ms> tpot:<ms>.")

    # Peak goodput per arm -- the one number off each curve.
    print()
    for arm in sorted({a for a, _ in by_point}):
        pts = [(r, statistics.mean([x["goodput"] for x in by_point[(a, r)]
                                    if x["goodput"] is not None] or [0]))
               for a, r in sorted(by_point) if a == arm]
        pts = [(r, g) for r, g in pts if g]
        if not pts:
            continue
        best = max(pts, key=lambda p: p[1])
        peak_t = max(((r, statistics.mean([x["req_tput"] for x in by_point[(arm, r)]
                                           if x["req_tput"]])) for r, _ in pts),
                     key=lambda p: p[1])
        print(f"{arm}: peak goodput {best[1]:.2f} req/s at lambda = {best[0]:g}; "
              f"peak throughput {peak_t[1]:.2f} req/s at lambda = {peak_t[0]:g}.")
        if best[0] < peak_t[0]:
            print("   Goodput peaks BELOW the throughput peak, as it should: past the")
            print("   knee the queue adds latency to requests that still complete.")

    print("\nBefore you put this in a document, fill in the provenance block from 10.3:")
    for line in [
        "  IDENTITY   engine + commit SHA, model + revision, harness + SHA",
        "  HARDWARE   GPU SKU and count, driver, CUDA, torch, client host and NIC",
        "  SERVER     the full launch command for BOTH arms, every non-default flag",
        "  WORKLOAD   input/output length DISTRIBUTIONS, prefix-sharing rate, seed",
        "  LOAD       arrival process, burstiness, ramp-up, concurrency cap if any",
        "  PROTOCOL   warmup length, flush endpoint and its response BODY, window",
        "  RESULTS    the curve, the SLO used for goodput, sigma, GPU clock range",
    ]:
        print(line)
    print("\nNeither harness records the server side, and neither records GPU clocks.")
    print("A result table without that block is a screenshot.")
    return 0


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a two-engine serving benchmark you can defend, and report goodput.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Size warmup and runs-per-arm. No GPU needed.")
    p.add_argument("--mu", type=float, default=None,
                   help="measured saturation throughput in req/s; drives warmup sizing")
    p.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95],
                   help="utilisations to size for (default: 0.5 0.8 0.9 0.95)")
    p.add_argument("--min-completions", type=int, default=1000,
                   help="completions the measured window needs for a usable p99 (default: 1000)")
    p.add_argument("--cv", type=float, default=None,
                   help="run-to-run coefficient of variation, as a fraction, e.g. 0.03")
    p.add_argument("--mde", type=float, default=None,
                   help="the difference you want to be able to defend, as a fraction")
    p.add_argument("--slo-ttft", type=float, default=None,
                   help="TTFT SLO in ms, echoed back as the --goodput argument to use")
    p.add_argument("--slo-tpot", type=float, default=None,
                   help="TPOT SLO in ms, echoed back as the --goodput argument to use")
    p.set_defaults(fn=cmd_plan)

    s = sub.add_parser("sweep", help="Run the full protocol for one arm.")
    s.add_argument("--arm", default="vllm",
                   help="a label for this server; anything starting with 'sglang' "
                        "makes the flush use /flush_cache (default: vllm)")
    s.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    s.add_argument("--backend", default="vllm",
                   help="harness backend name. Use 'openai' to drive an SGLang "
                        "server through /v1/completions (default: vllm)")
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--dataset-name", default="random",
                   help="passed straight through (default: random)")
    s.add_argument("--input-len", type=int, default=2048, help="--random-input-len")
    s.add_argument("--output-len", type=int, default=256, help="--random-output-len")
    s.add_argument("--rates", type=float, nargs="+",
                   default=[8, 12, 16, 20, 22, 24],
                   help="offered loads in req/s to sweep (default: 8 12 16 20 22 24)")
    s.add_argument("--burstiness", type=float, default=1.0,
                   help="1.0 is Poisson; below 1 is burstier (default: 1.0)")
    s.add_argument("--num-prompts", type=int, default=2000,
                   help="requests in the MEASURED window (default: 2000)")
    s.add_argument("--warmup-prompts", type=int, default=1200,
                   help="requests in the warm run, at the same rate (default: 1200). "
                        "Size it with `plan`, not with a round number.")
    s.add_argument("--runs", type=int, default=3,
                   help="repetitions per load point; 1 gives you no sigma (default: 3)")
    s.add_argument("--seed", type=int, default=1,
                   help="dataset and arrival seed. BOTH ARMS MUST USE THE SAME VALUE.")
    s.add_argument("--order-seed", type=int, default=7,
                   help="seed for randomising sweep order against thermal drift")
    s.add_argument("--goodput", nargs="+", default=["ttft:500", "tpot:50"],
                   help="SLOs as KEY:VALUE in ms; only ttft, tpot and e2el are "
                        "accepted by the harness (default: ttft:500 tpot:50)")
    s.add_argument("--max-concurrency", type=int, default=None,
                   help="closed-loop cap. Refused unless --allow-closed-loop.")
    s.add_argument("--allow-closed-loop", action="store_true",
                   help="acknowledge that a concurrency cap excludes client queueing "
                        "from every latency percentile and cannot observe overload")
    s.add_argument("--log-clocks", action="store_true",
                   help="print the nvidia-smi logging command to run alongside the sweep")
    s.add_argument("--metadata", nargs="+", default=[],
                   help="extra KEY=VALUE pairs stamped into every result file")
    s.add_argument("--workdir", default="./results")
    s.add_argument("--dry-run", action="store_true",
                   help="print every command, run none of them")
    s.add_argument("--extra-bench-args", nargs=argparse.REMAINDER, default=[],
                   help="everything after this flag goes to vllm bench serve verbatim")
    s.set_defaults(fn=cmd_sweep)

    r = sub.add_parser("report", help="Tabulate both arms and say what you may claim.")
    r.add_argument("--workdir", default="./results")
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    if args.cmd == "plan":
        if args.slo_ttft or args.slo_tpot:
            parts = ([f"ttft:{args.slo_ttft:g}"] if args.slo_ttft else []) + \
                    ([f"tpot:{args.slo_tpot:g}"] if args.slo_tpot else [])
            print("Pass this to every measured run, on both arms:\n"
                  "    --goodput " + " ".join(parts))
    else:
        print_environment()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
