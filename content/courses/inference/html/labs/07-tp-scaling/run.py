#!/usr/bin/env python3
"""Lab 07 — tensor-parallel scaling.

Sweep TP = 1, 2, 4, 8 on one node, measure the speedup, and attribute the
shortfall to collectives instead of guessing at it.

The lab has three parts and this script does all three:

  A. PREDICT   Derive the step time at each TP from the roofline plus a ring
               all-reduce model, with the per-collective latency lambda swept
               as a free parameter. No GPU, no network, no engine.

  B. MEASURE   Drive each engine's own single-batch harness once per TP size
               and collect the latency it reports. vLLM: `vllm bench latency`
               (--output-json). SGLang: `python -m sglang.benchmark.one_batch`
               (--result-filename, jsonl). This script never reimplements the
               timing loop, because the harnesses already handle warmup and
               CUDA-graph capture correctly.

  C. ATTRIBUTE Fit lambda to the measured curve, and tell you what the residual
               would have to be blamed on. The profile itself is an nsys
               capture -- see the README; both repos ship the kernel-name map
               that turns a trace into per-bucket time.

Nothing here writes into the pinned engine checkouts. Results land in --workdir.

    python3 run.py --predict-only --model-preset llama-3-8b --tp 1 2 4 8
    python3 run.py --engine vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 2 4 8 --batch-size 1 --workdir ./results
    python3 run.py --engine sglang --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tp 1 2 4 8 --dry-run

Reference points at the SHAs this book pins:
  vLLM   `vllm bench latency` flags       vllm/benchmarks/latency.py:L35-L66
  vLLM   result JSON shape                vllm/benchmarks/latency.py:L170-L177
  SGLang median_decode_latency            python/sglang/benchmark/one_batch.py:L857-L864
  vLLM   custom all-reduce size ceiling   .../device_communicators/all_reduce_utils.py:L31-L37
  SGLang custom all-reduce size ceiling   .../device_communicators/custom_all_reduce.py:L41-L42
  kernel-name -> bucket map               tools/profiler/nsys_profile_tools/vllm_engine_model.json
                                          examples/profiler/nsys_profile_tools/sglang_engine_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

KIB = 1024
MIB = 1024 * 1024

# Shapes are published config values, not measurements. head_dim is derived
# where the config does not state it.
PRESETS = {
    # name            L    d      h   h_kv  streamed GB (bf16, embedding gathered)
    "llama-3-8b":   dict(layers=32, hidden=4096, heads=32, kv_heads=8, streamed_gb=15.01),
    "llama-3-70b":  dict(layers=80, hidden=8192, heads=64, kv_heads=8, streamed_gb=140.6),
    "qwen2-72b":    dict(layers=80, hidden=8192, heads=64, kv_heads=8, streamed_gb=144.0),
}

# vllm/distributed/device_communicators/all_reduce_utils.py:L31-L37 --- the
# largest message the custom all-reduce kernel will accept, per compute
# capability and world size. Anything larger falls through to NCCL.
VLLM_CUSTOM_AR_MAX = {
    "9.0": {2: 64 * MIB, 4: 32 * MIB, 6: MIB // 2, 8: MIB // 4},
    "10.0": {2: 2 * MIB, 4: 2 * MIB, 6: 1 * MIB, 8: 1 * MIB},
    "10.3": {2: 4 * MIB, 4: 4 * MIB, 6: 8 * MIB, 8: 4 * MIB},
    "10.7": {2: 4 * MIB, 4: 4 * MIB, 6: 8 * MIB, 8: 4 * MIB},
}
# python/sglang/srt/distributed/device_communicators/custom_all_reduce.py:L41-L42
# --- one flat constant, at every world size and every architecture.
SGLANG_CUSTOM_AR_MAX = 8192 * 1024


# ------------------------------------------------------------------ printing


def table(rows, title, widths=None):
    if not rows:
        return
    cols = len(rows[0])
    widths = widths or [max(len(str(r[i])) for r in rows) for i in range(cols)]
    print(f"\n{title}")
    print("-" * (sum(widths) + 2 * (cols - 1)))
    for n, row in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i])
                         for i, c in enumerate(row))
        print(line)
        if n == 0:
            print("-" * (sum(widths) + 2 * (cols - 1)))


def print_environment(args):
    """Every run prints its environment, so a pasted result is reproducible."""
    rows = [("field", "value"),
            ("host", platform.node()),
            ("python", platform.python_version()),
            ("platform", platform.platform())]
    for tool in ("nvidia-smi", "nsys", "vllm"):
        rows.append((tool, shutil.which(tool) or "NOT ON PATH"))
    try:  # imported lazily: --help and --predict-only must work without torch
        import torch  # noqa: PLC0415
        rows.append(("torch", torch.__version__))
        if torch.cuda.is_available():
            rows.append(("gpu", torch.cuda.get_device_name(0)))
            rows.append(("gpu count", str(torch.cuda.device_count())))
            cap = torch.cuda.get_device_capability(0)
            rows.append(("compute capability", f"{cap[0]}.{cap[1]}"))
        else:
            rows.append(("gpu", "none visible"))
    except ImportError:
        rows.append(("torch", "not installed (fine for --predict-only)"))
    rows.append(("compute cap assumed", args.compute_capability))
    table(rows, "Environment")


# ------------------------------------------------------------------ part A


def residual_bytes_per_token(shape, elem_bytes):
    """Bytes of one token's residual slice: the tensor every all-reduce moves."""
    return shape["hidden"] * elem_bytes


def collectives_per_forward(shape):
    """Two all-reduces per transformer block (attention out-proj, MLP
    down-proj). The vocab-parallel embedding adds one all-reduce and the LM
    head one all-gather per pass; both are counted separately because they are
    a different message size. See 5.1."""
    return 2 * shape["layers"]


def predict(args, shape):
    elem = args.dtype_bytes
    per_tok = residual_bytes_per_token(shape, elem)
    ncoll = collectives_per_forward(shape)
    beta = args.hbm_tb_s * 1e12
    bnet = args.nvlink_gb_s * 1e9
    floor1_s = shape["streamed_gb"] * 1e9 / beta

    table([("field", "value"),
           ("layers L", f"{shape['layers']}"),
           ("hidden d", f"{shape['hidden']}"),
           ("kv heads", f"{shape['kv_heads']}"),
           ("bytes/elem", f"{elem}"),
           ("streamed weights", f"{shape['streamed_gb']:.2f} GB"),
           ("residual slice", f"{per_tok / KIB:.1f} KiB per token"),
           ("collectives / forward", f"{ncoll}  (2L)"),
           ("HBM bandwidth", f"{args.hbm_tb_s} TB/s"),
           ("interconnect", f"{args.nvlink_gb_s} GB/s per GPU"),
           ("TP=1 step floor", f"{floor1_s * 1e6:.0f} us")],
          "Part A - inputs. Published shapes and vendor specs; no measurement.")

    # Bandwidth term. A ring all-reduce moves 2(p-1)/p of the tensor per rank.
    rows = [("TP", "ring 2(p-1)/p", "B/rank/token", "us/token",
             f"% floor @ b={args.batch_size}", "custom AR up to")]
    for p in args.tp:
        if p == 1:
            rows.append(("1", "-", "-", "-", "0.000%", "-"))
            continue
        ring = 2 * (p - 1) / p
        vol = ncoll * ring * per_tok
        t_bw = vol / bnet
        floor_p = floor1_s / p
        pct = 100.0 * args.batch_size * t_bw / floor_p
        cap = VLLM_CUSTOM_AR_MAX.get(args.compute_capability, {}).get(p)
        if args.engine == "sglang":
            cap = SGLANG_CUSTOM_AR_MAX
        cap_tok = f"{cap // per_tok:,} tok" if cap else "n/a"
        rows.append((str(p), f"{ring:.2f}", f"{vol / KIB:,.0f} KiB",
                     f"{t_bw * 1e6:.3f}", f"{pct:.3f}%", cap_tok))
    table(rows, "Part A - bandwidth term (derived). The last column is the "
                "largest step that still uses the custom all-reduce kernel.")

    # Latency term. lambda is swept, not known.
    header = ["TP", "floor"]
    for lam in args.lambda_us:
        header += [f"t @ L={lam:g}us", "eff"]
    rows = [tuple(header)]
    for p in args.tp:
        floor_p = floor1_s / p
        row = [str(p), f"{floor_p * 1e6:.0f} us"]
        for lam in args.lambda_us:
            t = floor_p + (0.0 if p == 1 else ncoll * lam * 1e-6)
            row += [f"{t * 1e6:.0f} us", f"{100.0 * (floor1_s / t) / p:.0f}%"]
        rows.append(tuple(row))
    table(rows, f"Part A - predicted step time, t(p) = floor/p + {ncoll}*lambda "
                "(derived). lambda is a swept parameter, NOT a measurement.")

    print("\nPredict before you measure. At batch 1 the bandwidth term above is a")
    print("fraction of a percent, so anything you lose is per-collective latency,")
    print("and the whole sweep reduces to fitting one constant.")
    return dict(floor1_s=floor1_s, ncoll=ncoll, per_tok=per_tok)


# ------------------------------------------------------------------ part B


def vllm_cmd(args, tp, out_json):
    return ["vllm", "bench", "latency",
            "--model", args.model,
            "--tensor-parallel-size", str(tp),
            "--batch-size", str(args.batch_size),
            "--input-len", str(args.input_len),
            "--output-len", str(args.output_len),
            "--num-iters-warmup", str(args.warmup_iters),
            "--num-iters", str(args.iters),
            "--output-json", str(out_json)] + args.extra_engine_args


def sglang_cmd(args, tp, out_jsonl):
    return [sys.executable, "-m", "sglang.benchmark.one_batch",
            "--model-path", args.model,
            "--tp-size", str(tp),
            "--batch-size", str(args.batch_size),
            "--input-len", str(args.input_len),
            "--output-len", str(args.output_len),
            "--result-filename", str(out_jsonl)] + args.extra_engine_args


def read_vllm(out_json, args):
    """vllm/benchmarks/latency.py:L170-L177 writes avg_latency / latencies /
    percentiles. avg_latency covers one prefill plus output_len decode steps for
    the whole batch, so the per-step figure needs the prefill removed by hand --
    keep --input-len small and --output-len large so the residue is negligible."""
    d = json.loads(Path(out_json).read_text())
    return d["avg_latency"] / args.output_len, d


def read_sglang(out_jsonl, args):
    """python/sglang/benchmark/one_batch.py:L857-L864 records
    median_decode_latency, which already excludes prefill and the first token.
    That is the cleaner number for this lab. The file is jsonl and is appended
    to, so take the last record."""
    recs = [json.loads(x) for x in Path(out_jsonl).read_text().splitlines() if x.strip()]
    if not recs:
        raise RuntimeError(f"{out_jsonl} is empty; did the harness reach rank 0?")
    last = recs[-1]
    if "median_decode_latency" not in last:
        raise RuntimeError("no median_decode_latency in the result; "
                           "--output-len must be > 1")
    return last["median_decode_latency"], last


def measure(args):
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for p in args.tp:
        if args.engine == "vllm":
            out = work / f"tp{p}.json"
            cmd = vllm_cmd(args, p, out)
        else:
            out = work / f"tp{p}.jsonl"
            if out.exists():
                out.unlink()  # the harness appends
            cmd = sglang_cmd(args, p, out)

        print(f"\n$ {' '.join(cmd)}")
        if args.dry_run:
            continue
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(p)))
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f"  TP={p} FAILED with exit code {rc}. Recording it as a failure "
                  f"rather than dropping it -- a sweep with a hole in it is a "
                  f"different experiment.")
            results[p] = None
            continue
        reader = read_vllm if args.engine == "vllm" else read_sglang
        step_s, raw = reader(out, args)
        results[p] = step_s
        print(f"  TP={p}: {step_s * 1e6:.0f} us per decode step")
        (work / f"tp{p}.parsed.json").write_text(
            json.dumps({"tp": p, "step_s": step_s, "cmd": cmd, "raw": raw}, indent=2))
    return results


# ------------------------------------------------------------------ part C


def attribute(args, model, results):
    ok = {p: t for p, t in results.items() if t}
    if 1 not in ok:
        print("\nNo TP=1 point, so no speedup. Re-run with 1 in --tp, or pass "
              "--baseline-us to supply it.")
        if args.baseline_us:
            ok[1] = args.baseline_us * 1e-6
        else:
            return
    t1 = ok[1]
    floor1, ncoll = model["floor1_s"], model["ncoll"]

    rows = [("TP", "step", "speedup", "efficiency", "floor", "residual", "implied lambda")]
    for p in sorted(ok):
        t = ok[p]
        floor_p = floor1 / p
        resid = t - floor_p
        lam = resid / ncoll if p > 1 else 0.0
        rows.append((str(p), f"{t * 1e6:.0f} us", f"{t1 / t:.2f}x",
                     f"{100.0 * (t1 / t) / p:.0f}%", f"{floor_p * 1e6:.0f} us",
                     f"{resid * 1e6:+.0f} us",
                     f"{lam * 1e6:.2f} us" if p > 1 else "-"))
    table(rows, "Part C - measured, against the derived floor. 'implied lambda' "
                "is residual / (2L); it is only meaningful if it is CONSTANT.")

    lams = [(t - floor1 / p) / ncoll for p, t in sorted(ok.items()) if p > 1]
    if len(lams) >= 2:
        spread = (max(lams) - min(lams)) / max(min(lams), 1e-12)
        print(f"\nImplied lambda spans {min(lams) * 1e6:.2f} to {max(lams) * 1e6:.2f} us "
              f"({spread * 100:.0f}% spread).")
        if spread < 0.25:
            print("One constant explains the whole curve. The shortfall is fixed")
            print("per-collective latency and nothing size-dependent is happening.")
        else:
            print("One constant does NOT explain the curve. Work down this list:")
            for line in [
                "  1. Did you cross the custom all-reduce size ceiling? Part A's last",
                "     column gives the token count; the profile shows the kernel name",
                "     changing from cross_device_reduce_* to ncclDevKernel*.",
                "  2. Is the TP=1 baseline actually memory-bound at this batch? If not,",
                "     t(1) is not the floor and every ratio above is wrong.",
                "  3. Rank skew: one throttled or badly-placed GPU makes p-1 ranks wait.",
                "     Only an nsys capture across ranks shows this.",
                "  4. CUDA-graph bucketing, if you also swept batch size (see 10.3).",
                "  5. KV heads: at TP > h_kv the cache replicates, so throughput and",
                "     latency stop agreeing (see 5.1).",
            ]:
                print(line)

    print("\nEvery 'floor' above is arithmetic from published shapes. Every 'step'")
    print("came from the engine's own harness. The residual is the lab: name its")
    print("cause with a profile before you move on.")


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep tensor-parallel size, then attribute the scaling shortfall.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    g = ap.add_argument_group("what to run")
    g.add_argument("--engine", choices=("vllm", "sglang"), default="vllm",
                   help="which engine's single-batch harness to drive (default: vllm)")
    g.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="model id or local path passed to the engine")
    g.add_argument("--model-preset", choices=sorted(PRESETS), default="llama-3-8b",
                   help="published shapes used for the PREDICTION (default: llama-3-8b)")
    g.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="tensor-parallel sizes to sweep (default: 1 2 4 8)")
    g.add_argument("--predict-only", action="store_true",
                   help="Part A only: derive the prediction, run no engine, need no GPU")
    g.add_argument("--dry-run", action="store_true",
                   help="print every engine command that would run, run none of them")

    g = ap.add_argument_group("workload")
    g.add_argument("--batch-size", type=int, default=1,
                   help="sequences in the step (default: 1, where the bandwidth term vanishes)")
    g.add_argument("--input-len", type=int, default=128,
                   help="prompt tokens (default: 128; keep it small so prefill barely counts)")
    g.add_argument("--output-len", type=int, default=128,
                   help="decode steps to time (default: 128)")
    g.add_argument("--iters", type=int, default=30,
                   help="vLLM --num-iters (default: 30)")
    g.add_argument("--warmup-iters", type=int, default=10,
                   help="vLLM --num-iters-warmup (default: 10)")

    g = ap.add_argument_group("machine model - change these for your hardware")
    g.add_argument("--hbm-tb-s", type=float, default=3.35,
                   help="HBM bandwidth in TB/s; H100 SXM = 3.35 (default: 3.35)")
    g.add_argument("--nvlink-gb-s", type=float, default=900.0,
                   help="per-GPU interconnect bandwidth in GB/s; H100 NVLink4 = 900")
    g.add_argument("--dtype-bytes", type=int, default=2,
                   help="bytes per activation element (default: 2 for bf16)")
    g.add_argument("--streamed-gb", type=float, default=None,
                   help="override the preset's streamed-weight figure, in GB. The "
                        "preset uses 0.4's convention (input embedding gathered, "
                        "not streamed); 5.1's exercises use the cruder 2P form, "
                        "which is about 7 percent higher for Llama-3-8B.")
    g.add_argument("--compute-capability", default="9.0",
                   choices=sorted(VLLM_CUSTOM_AR_MAX),
                   help="used to look up vLLM's custom all-reduce size ceiling "
                        "(default: 9.0, i.e. H100)")
    g.add_argument("--lambda-us", type=float, nargs="+", default=[2.0, 5.0, 10.0],
                   help="per-collective latencies to sweep in the prediction, in "
                        "microseconds (default: 2 5 10)")

    g = ap.add_argument_group("output")
    g.add_argument("--workdir", default="./results",
                   help="where result files land (default: ./results). Nothing is "
                        "ever written into the engine checkouts.")
    g.add_argument("--baseline-us", type=float, default=None,
                   help="supply the TP=1 step time in microseconds if you cannot run it")
    g.add_argument("--extra-engine-args", nargs=argparse.REMAINDER, default=[],
                   help="everything after this flag is passed through to the engine "
                        "harness verbatim, e.g. --extra-engine-args --enforce-eager")

    args = ap.parse_args()
    shape = dict(PRESETS[args.model_preset])
    if args.streamed_gb is not None:
        shape["streamed_gb"] = args.streamed_gb
    args.tp = sorted(set(args.tp))

    print_environment(args)
    model = predict(args, shape)

    if args.predict_only:
        print("\n--predict-only: stopping before any engine runs. Nothing was measured.")
        return 0

    results = measure(args)
    if args.dry_run:
        print("\n--dry-run: no engine was started, so there is nothing to attribute.")
        return 0
    attribute(args, model, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
