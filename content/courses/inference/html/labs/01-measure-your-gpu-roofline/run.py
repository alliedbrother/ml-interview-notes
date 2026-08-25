#!/usr/bin/env python3
"""Lab 01 — measure your GPU's roofline.

Two numbers describe a GPU well enough to predict almost everything a serving
engine does with it: how fast it can multiply (pi, FLOP/s) and how fast it can
read (beta, bytes/s). Their ratio is the ridge point I* = pi / beta, the
arithmetic intensity at which a kernel stops being bandwidth-bound and starts
being compute-bound.

Vendor datasheets publish peaks that assume perfect tiling, no clock throttling
and no tail effects. This script measures what your card actually delivers:

  * a GEMM sweep     -> achieved TFLOP/s, and the M at which it saturates
  * a streaming loop -> achieved GB/s of HBM read+write
  * the ridge point  -> I* from your two measured numbers, not the datasheet

Every later prediction in this book is a function of pi and beta. Re-derive them
here and substitute; the book's own numbers use H100 SXM published peaks
(989.4 TFLOP/s dense bf16, 3.35 TB/s) and will be optimistic on real silicon.

    python3 run.py                                   # full sweep, autodetect
    python3 run.py --mode gemm --dtype float16
    python3 run.py --mode bandwidth --buffer-mib 2048
    python3 run.py --peak-tflops 989.4 --peak-gbps 3350 --csv roofline.csv
    python3 run.py --help                            # every knob

Reference points at the SHAs this book pins:
  vLLM's own GEMM sweep, TFLOP/s from 2*M*N*K
        benchmarks/kernels/benchmark_fp8_gemm.py:L89-L97, L123-L124
  vLLM's own GB/s idiom, bytes_moved / latency
        benchmarks/kernels/benchmark_cp_gather.py:L193-L201
  The Llama-3.1-8B weight shapes this script sweeps
        benchmarks/kernels/weight_shapes.py:L52-L57
  vLLM's runtime version of the same two numbers, under --enable-mfu-metrics
        vllm/v1/metrics/perf.py:L1521-L1534
  SGLang's runtime version, under --enable-mfu-metrics
        python/sglang/srt/managers/scheduler_components/metrics_reporter.py:L892-L903

This script reads nothing from and writes nothing to either checkout.
"""

from __future__ import annotations

import argparse
import json
import sys

# Bytes per element, for the FLOP and byte accounting. Keys match torch dtype
# names so --dtype maps straight through.
#
# fp8 is deliberately absent. A plain torch.nn.functional.linear cannot run it;
# measuring an fp8 roofline means going through an engine's scaled-mm path, and
# vLLM already ships that sweep at benchmarks/kernels/benchmark_fp8_gemm.py.
# Faking it here with a bf16 GEMM relabelled fp8 would produce a wrong ridge.
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}

# Published dense peaks, no structured sparsity, for the two cards this book
# already cites. CITED, not measured: NVIDIA datasheets, as used throughout the
# book (see FORMULAS). Anything else must come from --peak-tflops/--peak-gbps,
# because a number nobody in this repo can check is worse than no number.
SPEC = {
    "H100": {"bf16_tflops": 989.4, "hbm_gbps": 3350.0,
             "note": "H100 SXM, dense bf16 (the 1979 headline is the 2:4 sparsity figure)"},
    "A100": {"bf16_tflops": 312.0, "hbm_gbps": 2039.0,
             "note": "A100 SXM 80GB, dense bf16"},
}

# Real Llama-3.1-8B-Instruct projection shapes as [K, N], transcribed from
# benchmarks/kernels/weight_shapes.py:L52-L57 in the pinned vLLM tree.
LLAMA3_8B_SHAPES = [
    (4096, 6144, "fused QKV"),
    (4096, 4096, "o_proj"),
    (4096, 28672, "gate+up"),
    (14336, 4096, "down_proj"),
]

DEFAULT_M_SWEEP = [1, 8, 32, 128, 256, 345, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_SQUARE = [512, 1024, 2048, 4096, 8192, 16384]


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def describe_env() -> dict:
    """Everything a pasted result needs to be reproducible."""
    import torch

    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info.update({
        "device": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "sms": props.multi_processor_count,
        "total_mem_gib": round(props.total_memory / 1024 ** 3, 1),
        "cuda_runtime": torch.version.cuda,
    })
    return info


def guess_spec(device_name: str) -> tuple[float | None, float | None, str]:
    """Match a device name against the two cards the book cites peaks for."""
    up = (device_name or "").upper()
    for key, row in SPEC.items():
        if key in up:
            return row["bf16_tflops"], row["hbm_gbps"], row["note"]
    return None, None, "no published peak on file for this device — pass --peak-tflops/--peak-gbps"


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def time_ms(fn, warmup: int, reps: int) -> float:
    """Median wall time of fn() in ms, measured with CUDA events.

    CUDA launches are asynchronous, so a bare perf_counter around fn() measures
    how fast Python enqueues work — often faster than the work itself. Events are
    recorded on the stream and read back after a synchronize, so the number is
    device time. See the pitfall in the roofline chapter (00-04).
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


# --------------------------------------------------------------------------
# GEMM sweep
# --------------------------------------------------------------------------

def gemm_point(M: int, K: int, N: int, dtype_name: str, warmup: int, reps: int) -> dict:
    """One [M,K] x [K,N] GEMM: achieved TFLOP/s and its arithmetic intensity.

    FLOPs = 2*M*N*K, one multiply and one add per MAC — the same accounting as
    vLLM's own sweep (benchmarks/kernels/benchmark_fp8_gemm.py:L123).

    Bytes are COMPULSORY traffic: read A, read B, write C, each once. A real
    kernel that spills tiles moves more, so the intensity below is an upper
    bound on what this shape can achieve, which is exactly what a roofline
    x-coordinate is supposed to be.
    """
    import torch

    dtype = getattr(torch, dtype_name)
    dev = "cuda"
    a = torch.randn((M, K), device=dev, dtype=dtype)
    # torch.nn.functional.linear takes weights as [N, K], matching how both
    # engines store projection weights.
    b = torch.randn((N, K), device=dev, dtype=dtype)

    ms = time_ms(lambda: torch.nn.functional.linear(a, b), warmup, reps)
    del a, b
    torch.cuda.empty_cache()

    elem = DTYPE_BYTES[dtype_name]
    flops = 2.0 * M * N * K
    qbytes = float(elem) * (M * K + K * N + M * N)
    return {
        "M": M, "K": K, "N": N,
        "ms": ms,
        "tflops": flops * 1e-12 / (ms * 1e-3),
        "intensity": flops / qbytes,
    }


def sweep_gemm(args) -> tuple[list[dict], list[dict]]:
    square, skinny = [], []

    if args.mode in ("gemm", "all"):
        for n in args.square:
            try:
                square.append(gemm_point(n, n, n, args.dtype, args.warmup, args.reps))
            except RuntimeError as exc:          # OOM on a small card is expected
                print(f"  skipped square {n}: {exc.__class__.__name__}", file=sys.stderr)
        for K, N, label in LLAMA3_8B_SHAPES:
            for M in args.m_sweep:
                try:
                    row = gemm_point(M, K, N, args.dtype, args.warmup, args.reps)
                except RuntimeError as exc:
                    print(f"  skipped M={M} {K}x{N}: {exc.__class__.__name__}", file=sys.stderr)
                    continue
                row["label"] = label
                skinny.append(row)
    return square, skinny


# --------------------------------------------------------------------------
# bandwidth
# --------------------------------------------------------------------------

def sweep_bandwidth(args) -> list[dict]:
    """Achieved HBM bandwidth from two streaming kernels.

    'copy'  reads one buffer and writes another: 2 bytes of traffic per element.
    'triad' reads two and writes one: 3 bytes per element, and is the harder of
    the two because it needs two read streams in flight.

    Bytes over median latency, the same idiom as vLLM's cache-gather benchmark
    at benchmarks/kernels/benchmark_cp_gather.py:L196.
    """
    import torch

    if args.mode not in ("bandwidth", "all"):
        return []

    dtype = getattr(torch, args.dtype)
    elem = torch.tensor([], dtype=dtype).element_size()
    n = (args.buffer_mib * 1024 * 1024) // elem

    rows = []
    try:
        src = torch.empty(n, device="cuda", dtype=dtype).uniform_(-1, 1)
        dst = torch.empty_like(src)
        ms = time_ms(lambda: dst.copy_(src), args.warmup, args.reps)
        rows.append({"kernel": "copy (1 read + 1 write)", "bytes": 2 * n * elem,
                     "ms": ms, "gbps": 2 * n * elem / (ms * 1e-3) / 1e9})

        other = torch.empty_like(src).uniform_(-1, 1)
        ms = time_ms(lambda: torch.add(src, other, out=dst), args.warmup, args.reps)
        rows.append({"kernel": "triad (2 reads + 1 write)", "bytes": 3 * n * elem,
                     "ms": ms, "gbps": 3 * n * elem / (ms * 1e-3) / 1e9})
        del src, dst, other
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        print(f"  bandwidth sweep failed ({exc.__class__.__name__}); "
              f"lower --buffer-mib", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def kv(rows: list[tuple[str, str]], title: str) -> None:
    w = max((len(k) for k, _ in rows), default=0)
    print(f"\n{title}\n" + "-" * (w + 30))
    for k_, v in rows:
        print(f"{k_:<{w}}  {v}")


def grid(headers: list[str], rows: list[list[str]], title: str) -> None:
    if not rows:
        return
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print(f"\n{title}")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i])
                        for i, c in enumerate(r)))


def report(args, env, square, skinny, bw) -> dict:
    kv([(k, str(v)) for k, v in env.items()], "Environment — paste this with any result")

    grid(["M=N=K", "ms", "TFLOP/s", "I (FLOP/byte)"],
         [[str(r["M"]), f"{r['ms']:.3f}", f"{r['tflops']:.1f}", f"{r['intensity']:.0f}"]
          for r in square],
         "Square GEMM sweep — measured on this card")

    grid(["shape", "K", "N", "M", "ms", "TFLOP/s", "I"],
         [[r["label"], str(r["K"]), str(r["N"]), str(r["M"]),
           f"{r['ms']:.3f}", f"{r['tflops']:.1f}", f"{r['intensity']:.1f}"]
          for r in skinny],
         "Llama-3.1-8B projection shapes, M swept — measured on this card")

    grid(["kernel", "GiB moved", "ms", "GB/s"],
         [[r["kernel"], f"{r['bytes'] / 1024 ** 3:.2f}", f"{r['ms']:.3f}", f"{r['gbps']:.1f}"]
          for r in bw],
         "Streaming kernels — measured on this card")

    achieved_tflops = max((r["tflops"] for r in square + skinny), default=0.0)
    achieved_gbps = max((r["gbps"] for r in bw), default=0.0)

    out = {"env": env, "square": square, "skinny": skinny, "bandwidth": bw,
           "achieved_tflops": achieved_tflops, "achieved_gbps": achieved_gbps}

    if not (achieved_tflops and achieved_gbps):
        print("\nRun with --mode all to get both halves; the ridge point needs "
              "compute AND bandwidth.")
        return out

    ridge = achieved_tflops * 1e12 / (achieved_gbps * 1e9)
    out["ridge"] = ridge

    peak_t, peak_b = args.peak_tflops, args.peak_gbps
    note = "supplied on the command line"
    if peak_t is None or peak_b is None:
        gt, gb, note = guess_spec(env.get("device", ""))
        peak_t = peak_t if peak_t is not None else gt
        peak_b = peak_b if peak_b is not None else gb

    rows = [
        ("achieved pi  (TFLOP/s)", f"{achieved_tflops:.1f}   measured, best shape in the sweep"),
        ("achieved beta (GB/s)", f"{achieved_gbps:.1f}   measured, best streaming kernel"),
        ("measured ridge I*", f"{ridge:.0f} FLOP/byte   derived: pi / beta"),
    ]
    if peak_t and peak_b:
        spec_ridge = peak_t * 1e12 / (peak_b * 1e9)
        out.update({"spec_tflops": peak_t, "spec_gbps": peak_b, "spec_ridge": spec_ridge})
        rows += [
            ("published pi", f"{peak_t:.1f} TFLOP/s   cited — {note}"),
            ("published beta", f"{peak_b:.1f} GB/s"),
            ("published ridge", f"{spec_ridge:.0f} FLOP/byte"),
            ("compute efficiency", f"{100 * achieved_tflops / peak_t:.1f}% of published peak"),
            ("bandwidth efficiency", f"{100 * achieved_gbps / peak_b:.1f}% of published peak"),
        ]
    else:
        rows.append(("published peaks", f"unknown — {note}"))
    kv(rows, "Roofline — the two numbers every later prediction depends on")

    elem = DTYPE_BYTES[args.dtype]
    d = args.hidden_size
    # Reference intensities, all from 0.4:
    #   dense weight GEMV at one token           I = 2/b
    #   decode attention against the KV cache    I = 2g/b  (context length cancels)
    #   square prefill projection over T tokens  I = 2Td / (b(2T + d))
    t_ref = 2048
    placements = [
        ("batch-1 decode weight GEMV", 2.0 / elem),
        (f"decode attention, GQA g={args.gqa_group}", 2.0 * args.gqa_group / elem),
        (f"prefill GEMM, T={t_ref}, d={d}",
         2.0 * t_ref * d / (elem * (2 * t_ref + d))),
    ]
    grid(["workload", "I (FLOP/byte)", "bound by", "achievable TFLOP/s"],
         [[name, f"{i:.1f}",
           "memory" if i < ridge else "compute",
           f"{min(achieved_tflops, achieved_gbps * i / 1000):.1f}"]
          for name, i in placements],
         "Where the book's three reference workloads land on YOUR ridge — derived")

    denom = d - elem * ridge
    if denom > 0:
        t_star = ridge * d / denom
        print(f"\nA square [T,{d}] x [{d},{d}] {args.dtype} GEMM reaches your ridge at "
              f"T* = {t_star:.0f} tokens.")
        print("  Derived: T* = I* d / (d - b I*). Compare it against your engine's "
              "batched-token budget.")
    else:
        print(f"\nAt d={d} and b={elem}, b*I* = {elem * ridge:.0f} >= d: no batch size "
              "makes a square GEMM of this width compute-bound on this card.")
    return out


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure achieved TFLOP/s and HBM GB/s on this GPU, and report the ridge point.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["gemm", "bandwidth", "all"], default="all",
                    help="which half to run (default: all)")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_BYTES),
                    help="compute dtype for the GEMM sweep (default: bfloat16)")
    ap.add_argument("--square", type=int, nargs="+", default=DEFAULT_SQUARE,
                    metavar="N", help=f"square GEMM sizes M=N=K (default: {DEFAULT_SQUARE})")
    ap.add_argument("--m-sweep", type=int, nargs="+", default=DEFAULT_M_SWEEP,
                    metavar="M",
                    help="token counts to sweep against the Llama-3.1-8B weight shapes "
                         f"(default: {DEFAULT_M_SWEEP})")
    ap.add_argument("--buffer-mib", type=int, default=1024,
                    help="per-buffer size for the streaming kernels, MiB (default: 1024). "
                         "Must be much larger than L2 or you are measuring cache.")
    ap.add_argument("--warmup", type=int, default=10,
                    help="untimed iterations before each measurement (default: 10)")
    ap.add_argument("--reps", type=int, default=25,
                    help="timed iterations; the median is reported (default: 25)")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="published dense peak for this dtype, TFLOP/s. Autodetected for "
                         "H100/A100 from the two figures this book cites; required otherwise.")
    ap.add_argument("--peak-gbps", type=float, default=None,
                    help="published HBM bandwidth, GB/s (H100 SXM: 3350)")
    ap.add_argument("--hidden-size", type=int, default=4096,
                    help="d, for the T* calculation (default: 4096, Llama-3-8B)")
    ap.add_argument("--gqa-group", type=int, default=4,
                    help="g = h / h_kv, for placing decode attention (default: 4, Llama-3-8B)")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="also write every measured row to this CSV")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the full result object to this JSON file")
    args = ap.parse_args()

    try:
        import torch                                  # noqa: F401
    except ImportError:
        sys.exit("This lab needs PyTorch with CUDA. --help works without it; nothing else does.")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible. This lab measures silicon; there is no CPU fallback.\n"
                 "Read the README's 'What to expect' section and come back with a GPU.")

    env = describe_env()
    square, skinny = sweep_gemm(args)
    bw = sweep_bandwidth(args)
    out = report(args, env, square, skinny, bw)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "label", "M", "K", "N", "ms", "tflops", "intensity", "gbps"])
            for r in square:
                w.writerow(["square", "", r["M"], r["K"], r["N"],
                            f"{r['ms']:.6f}", f"{r['tflops']:.3f}", f"{r['intensity']:.3f}", ""])
            for r in skinny:
                w.writerow(["skinny", r["label"], r["M"], r["K"], r["N"],
                            f"{r['ms']:.6f}", f"{r['tflops']:.3f}", f"{r['intensity']:.3f}", ""])
            for r in bw:
                w.writerow(["bandwidth", r["kernel"], "", "", "",
                            f"{r['ms']:.6f}", "", "", f"{r['gbps']:.3f}"])
        print(f"\nwrote {args.csv}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")

    print("\nCarry pi, beta and I* forward. Every derived number in this book was "
          "computed against published peaks; substitute yours and re-check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
