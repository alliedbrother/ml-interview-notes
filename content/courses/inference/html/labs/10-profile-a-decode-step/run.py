#!/usr/bin/env python3
"""Lab 10 — Profile a decode step.

Account for every microsecond of one decode iteration, including the microseconds
that are not inside any kernel. Four things, in order:

  1. floor        derive the compulsory-traffic floor for your operating point
                  (pure arithmetic, no GPU, no network)
  2. capture      arm the engine's own torch.profiler over HTTP and pull a trace
  3. analyze      read the trace: top kernels, kernel-busy time, wall time, GAP
  4. launch-probe measure YOUR box's per-launch driver cost with a null kernel,
                  instead of citing someone else's 2.374 us

The gap is the lab. Kernel time you can see in any profiler summary table; the
time between kernels is what tells you whether the step is launch-bound, and it
is the number the engine's own `self_cuda_time_total` table cannot show you.

    python3 run.py --mode floor --batch 32 --seq 2048
    python3 run.py --mode capture --engine vllm --url http://localhost:8000
    python3 run.py --mode analyze --trace ./traces/rank0.pt.trace.json.gz
    python3 run.py --mode analyze --trace t.json.gz --span "execute_" --top 15
    python3 run.py --mode launch-probe --iters 2000      # needs torch + a GPU
    python3 run.py --help

Only `launch-probe` needs torch. `capture` uses urllib. `analyze` and `floor`
are stdlib-only and run on a laptop against a trace someone else captured.

Reference points at the SHAs this book pins:
  vLLM   ProfilerConfig fields            vllm/config/profiler.py:L42-L147
  vLLM   POST /start_profile, /stop_profile
                                          vllm/entrypoints/serve/profile/api_router.py:L21-L34
  vLLM   step span name                   vllm/v1/worker/gpu_worker.py:L1000-L1041
  SGLang POST body for a capture          python/sglang/profiler.py:L51-L68
  SGLang step span name                   python/sglang/srt/utils/profile_utils.py:L476-L487

NOTHING HERE WAS RUN AGAINST A GPU BY THE AUTHOR. Every number this script
prints that is not read out of your trace file is labelled derived or cited.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

# Categories torch.profiler puts on the device timeline. Kernel-busy time is the
# union of these intervals; a plain sum double-counts overlapping streams.
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
# Host-side CUDA runtime calls. cudaLaunchKernel is the one the launch-overhead
# budget is denominated in; cudaGraphLaunch is what replaces N of them when
# CUDA graphs are on (see chapter 08-01).
LAUNCH_NAMES = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cudaGraphLaunch",
)
ANNOT_CATS = {"user_annotation", "gpu_user_annotation"}

# Cited vendor constants. Not measured here; substitute your own card.
HBM_TB_S = {"h100-sxm": 3.35, "h100-pcie": 2.0, "a100-80": 2.039, "l40s": 0.864}
# Cited: Vellaisamy et al., "Characterizing GPU Kernel Launch Overheads",
# ISPASS 2025, arXiv:2504.11750, Table V — driver-only cudaLaunchKernel on
# H100 / CUDA 12.6. Replace it with your own number via --mode launch-probe.
CITED_LAUNCH_US = 2.374


# ---------------------------------------------------------------- 1. the floor


def mode_floor(a: argparse.Namespace) -> int:
    """Compulsory HBM traffic for one decode step, then t = Q / beta.

    This is the same arithmetic as chapter 10-05 section 3, parameterised. It is
    a floor, not a prediction: it counts only bytes that MUST move.
    """
    beta = a.hbm_tb_s if a.hbm_tb_s else HBM_TB_S[a.card]
    # KV cell size: one K and one V entry per KV head per layer, per token.
    k_bytes = 2 * a.layers * a.kv_heads * a.head_dim * a.kv_dtype_bytes
    kv_read = k_bytes * a.batch * a.seq
    kv_write = k_bytes * a.batch
    weights = a.weights_gb * 1e9
    total = weights + kv_read + kv_write
    t_ms = total / (beta * 1e12) * 1e3

    rows = [
        ("card / HBM bandwidth", f"{a.card if not a.hbm_tb_s else 'custom'}  {beta} TB/s  [cited]"),
        ("batch B", f"{a.batch}"),
        ("context s (tokens/seq)", f"{a.seq:,}"),
        ("KV cell k = 2*L*h_kv*d_h*b", f"{k_bytes:,} bytes  ({k_bytes / 1024:.0f} KiB/token)"),
        ("weights streamed", f"{weights / 1e9:.2f} GB   ({100 * weights / total:.1f}%)"),
        ("KV read  = k*B*s", f"{kv_read / 1e9:.3f} GB   ({100 * kv_read / total:.1f}%)"),
        ("KV write = k*B", f"{kv_write / 1e9:.4f} GB   ({100 * kv_write / total:.2f}%)"),
        ("total compulsory traffic Q", f"{total / 1e9:.2f} GB"),
        ("STEP FLOOR t = Q / beta", f"{t_ms:.3f} ms"),
        ("implied decode ceiling", f"{a.batch / (t_ms / 1e3):,.0f} tok/s"),
    ]
    table(rows, "Derived floor — arithmetic only, nothing measured")

    launchcost = a.launchcost if a.launchcost else CITED_LAUNCH_US
    src = "measured by --mode launch-probe" if a.launchcost else "cited, arXiv:2504.11750 Table V"
    budget_ms = a.launches * launchcost / 1e3
    table(
        [
            ("kernel launches per step", f"{a.launches}  (chapter 08-01, TP=1 Llama-3-8B)"),
            ("per-launch driver cost", f"{launchcost:.3f} us   [{src}]"),
            ("launch-overhead budget", f"{budget_ms:.3f} ms"),
            ("share of the floor", f"{100 * budget_ms / t_ms:.1f}%"),
        ],
        "Derived launch budget — what the gap in your trace must not exceed",
    )
    print(
        "\nPredict before you capture. Write down these two numbers:\n"
        f"  kernel-busy time should land near {t_ms:.2f} ms\n"
        f"  gap (wall minus busy)  should land under {budget_ms:.2f} ms\n"
        "Then run --mode analyze and see which one you got wrong."
    )
    return 0


# -------------------------------------------------------------- 2. the capture


def _post(url: str, body: dict | None, timeout: float) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")[:400]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400]
    except Exception as e:  # connection refused, timeout, ...
        return 0, f"{type(e).__name__}: {e}"


def mode_capture(a: argparse.Namespace) -> int:
    """Arm the engine's profiler, drive decode load, disarm it.

    Both engines expose /start_profile and /stop_profile; they differ in where
    the capture window is configured. vLLM binds it at launch through
    --profiler-config.* and takes an empty POST. SGLang takes the window in the
    POST body and blocks until the trace is flushed.
    """
    import threading

    base = a.url.rstrip("/")

    if a.engine == "vllm":
        print("vLLM: the capture window is set at LAUNCH, not here. The server must")
        print("have been started with, at minimum:")
        print("  --profiler-config.profiler=torch")
        print("  --profiler-config.torch_profiler_dir=/abs/path")
        print("Otherwise the worker raises 'Profiling is not enabled.'")
        print("  (vllm/v1/worker/gpu_worker.py:L1146-L1153)\n")
        start_body: dict | None = None
    else:
        start_body = {
            "output_dir": a.out,
            "num_steps": str(a.steps),
            "activities": ["CPU", "GPU"],
            "profile_by_stage": a.by_stage,
        }
        print(f"SGLang: capture window travels in the POST body: {start_body}\n")

    # Fire the load first so the engine is in steady-state decode when the
    # profiler arms. A capture that begins on an idle server records warmup.
    stop_load = threading.Event()
    if a.drive:
        t = threading.Thread(target=_drive, args=(a, stop_load), daemon=True)
        t.start()
        print(f"driving {a.concurrency} concurrent request(s) at {base} ...")
        time.sleep(a.warmup_s)

    code, body = _post(f"{base}/start_profile", start_body, a.timeout)
    print(f"POST /start_profile -> {code} {body}")
    if code != 200:
        stop_load.set()
        print("\nProfiler did not arm. Read the body above; the usual causes are:")
        print("  vLLM  : --profiler-config was never passed, so the route is not attached")
        print("          (vllm/entrypoints/serve/profile/api_router.py:L37-L46)")
        print("  SGLang: SGLANG_TORCH_PROFILER_DIR unset and output_dir not writable")
        return 1

    if a.engine == "vllm":
        # vLLM's endpoint returns immediately; the window is bounded by
        # active_iterations / max_iterations, or by your stop call.
        time.sleep(a.hold_s)
        code, body = _post(f"{base}/stop_profile", None, a.timeout)
        print(f"POST /stop_profile  -> {code} {body}")

    stop_load.set()
    print(f"\nTrace directory: {a.out}")
    print("Next: python3 run.py --mode analyze --trace <that file>")
    return 0


def _drive(a: argparse.Namespace, stop: "object") -> None:
    """Keep `concurrency` decode requests in flight until told to stop."""
    import threading
    import urllib.request

    base = a.url.rstrip("/")
    if a.engine == "vllm":
        url, payload = f"{base}/v1/completions", {
            "model": a.model, "prompt": "Count slowly: ", "max_tokens": a.gen_tokens,
            "temperature": 0.0, "stream": False,
        }
    else:
        url, payload = f"{base}/generate", {
            "text": "Count slowly: ",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": a.gen_tokens},
        }

    def one() -> None:
        while not stop.is_set():  # type: ignore[attr-defined]
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=a.timeout).read()
            except Exception:
                return

    threads = [threading.Thread(target=one, daemon=True) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# -------------------------------------------------------------- 3. the analysis


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        blob = json.load(f)
    if isinstance(blob, dict):
        return blob.get("traceEvents", [])
    return blob


def union_us(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by at least one interval. Overlapping streams count once."""
    if not intervals:
        return 0.0
    intervals.sort()
    total, cur_a, cur_b = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = s, e
        else:
            cur_b = max(cur_b, e)
    return total + (cur_b - cur_a)


def mode_analyze(a: argparse.Namespace) -> int:
    path = Path(a.trace)
    if not path.exists():
        sys.exit(f"No such trace: {path}")
    events = load_trace(path)
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    print(f"{path.name}: {len(events):,} events, {len(complete):,} with a duration")

    annots = [e for e in complete if e.get("cat") in ANNOT_CATS]
    if a.spans:
        seen: dict[str, int] = {}
        for e in annots:
            seen[f"[{e.get('cat')}] {e.get('name')}"] = seen.get(
                f"[{e.get('cat')}] {e.get('name')}", 0) + 1
        for name, n in sorted(seen.items(), key=lambda kv: -kv[1])[: a.top]:
            print(f"  {n:6d}x  {name}")
        print("\nPick one with --span and re-run. vLLM's decode spans start "
              "'execute_'; SGLang's start 'step[DECODE'.")
        return 0

    # Window selection. Prefer a GPU-projected annotation, because a CPU
    # annotation's window does not contain the kernels it launched — the host
    # runs ahead. If only the CPU side exists, say so, loudly.
    window = None
    window_src = "whole trace"
    if a.span:
        gpu_side = [e for e in annots
                    if e.get("cat") == "gpu_user_annotation" and a.span in str(e.get("name"))]
        cpu_side = [e for e in annots
                    if e.get("cat") == "user_annotation" and a.span in str(e.get("name"))]
        picked = gpu_side or cpu_side
        if not picked:
            sys.exit(f"No annotation matching {a.span!r}. Re-run with --spans.")
        picked.sort(key=lambda e: e["ts"])
        e = picked[min(a.span_index, len(picked) - 1)]
        window = (e["ts"], e["ts"] + e["dur"])
        window_src = f"{e.get('cat')} {e.get('name')!r} (#{a.span_index} of {len(picked)})"
        if not gpu_side:
            print("\n  WARNING: only a CPU-side user_annotation matched. The host runs")
            print("  ahead of the device, so kernels for this step may fall outside")
            print("  this window. Treat the gap below as unusable and re-capture with")
            print("  GPU activities enabled.\n")

    def inside(e: dict) -> bool:
        if window is None:
            return True
        s = e["ts"]
        return window[0] <= s <= window[1]

    gpu = [e for e in complete if str(e.get("cat")) in GPU_CATS and inside(e)]
    launches = [e for e in complete
                if str(e.get("cat")) == "cuda_runtime"
                and str(e.get("name")).startswith(LAUNCH_NAMES) and inside(e)]

    if not gpu:
        sys.exit("No GPU events in the window. Was the trace captured with GPU "
                 "activities on? (vLLM: default; SGLang: activities must include 'GPU')")

    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    busy_us = union_us(list(intervals))
    sum_us = sum(e["dur"] for e in gpu)
    first, last = min(i[0] for i in intervals), max(i[1] for i in intervals)
    wall_us = (window[1] - window[0]) if window else (last - first)
    gap_us = wall_us - busy_us

    by_name: dict[str, list[float]] = {}
    for e in gpu:
        by_name.setdefault(str(e.get("name")), []).append(e["dur"])

    table(
        [
            ("window", window_src),
            ("wall time", f"{wall_us / 1e3:9.3f} ms"),
            ("kernel-busy (union)", f"{busy_us / 1e3:9.3f} ms   {100 * busy_us / wall_us:5.1f}% of wall"),
            ("kernel-sum (overlap counted twice)", f"{sum_us / 1e3:9.3f} ms"),
            ("GAP  = wall - busy", f"{gap_us / 1e3:9.3f} ms   {100 * gap_us / wall_us:5.1f}% of wall"),
            ("device events", f"{len(gpu):,}  in {len(set(e.get('tid') for e in gpu))} stream(s)"),
            ("host launch calls", f"{len(launches):,}"),
        ],
        "Measured — read out of your trace file",
    )

    launchcost = a.launchcost if a.launchcost else CITED_LAUNCH_US
    src = "measured, --mode launch-probe" if a.launchcost else "cited, arXiv:2504.11750"
    if launches:
        host_us = sum(e["dur"] for e in launches)
        driver_us = len(launches) * launchcost
        table(
            [
                ("launch calls in window", f"{len(launches):,}"),
                ("host time inside those calls", f"{host_us:9.1f} us   (from the trace)"),
                ("driver floor at %.3f us/launch" % launchcost, f"{driver_us:9.1f} us   [{src}]"),
                ("framework cost above the floor", f"{host_us - driver_us:9.1f} us"),
                ("host launch time vs the gap", f"{host_us / gap_us:9.2f}x" if gap_us > 0 else "n/a"),
            ],
            "Launch attribution — derived from the counts above",
        )
        print("\nA ratio above 1.0 means the host spent more time issuing launches than")
        print("the device spent idle: the host is running ahead and hiding it. Below 1.0")
        print("means launches cannot explain the whole gap — look for a synchronize.")

    rows = []
    for name, durs in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[: a.top]:
        tot = sum(durs)
        rows.append((name[: a.name_width],
                     f"{tot / 1e3:8.3f} ms  {100 * tot / busy_us:5.1f}%  n={len(durs):<5d} "
                     f"mean {tot / len(durs):7.1f} us"))
    table(rows, f"Top {a.top} kernels by total device time — measured")

    if a.floor:
        resid = wall_us / 1e3 - a.floor
        table(
            [
                ("derived floor", f"{a.floor:8.3f} ms   [--mode floor]"),
                ("measured wall", f"{wall_us / 1e3:8.3f} ms"),
                ("residual", f"{resid:+8.3f} ms   ({100 * resid / a.floor:+.1f}%)"),
                ("of which gap", f"{gap_us / 1e3:8.3f} ms"),
                ("of which kernel excess", f"{(busy_us / 1e3) - a.floor:+8.3f} ms"),
            ],
            "Attribution — every microsecond must land in one of these rows",
        )
        if wall_us / 1e3 < a.floor:
            print("\n  Measured below the floor. The byte model is wrong, not the machine:")
            print("  prefix caching, an L2-resident weight slice, or a KV dtype you did")
            print("  not pass. Fix the floor before you interpret anything else.")

    print("\nDiagnosis, mechanically (chapter 10-05, Figure 3):")
    frac = gap_us / wall_us if wall_us else 0
    if frac > 0.20:
        print(f"  gap is {100 * frac:.0f}% of wall  ->  SHAPE 1, launch-bound.")
        print("  Nothing about the kernels matters until you fix this. Re-capture with")
        print("  CUDA graphs on (vLLM: drop --enforce-eager; SGLang: drop")
        print("  --cuda-graph-backend-decode=disabled) and diff the gap. See chapter 08-01.")
    else:
        top_share = max(sum(d) for d in by_name.values()) / busy_us
        if top_share > 0.5:
            print(f"  one kernel is {100 * top_share:.0f}% of device time  ->  SHAPE 3, compute-bound.")
            print("  Take that kernel name to Nsight Compute; this tool cannot tell you why.")
        else:
            print("  kernels are back to back  ->  SHAPE 2 or SHAPE 4, and a timeline")
            print("  cannot tell them apart. Only achieved DRAM throughput can. Compare")
            print("  the byte model's bytes against the kernel duration, per kernel.")
    return 0


# --------------------------------------------------------- 4. the launch probe


def mode_launch_probe(a: argparse.Namespace) -> int:
    """Measure this box's per-launch cost with a kernel that does nothing.

    Chapter 08-01 had to cite 2.374 us from a paper because the book has no GPU.
    This is where you replace it. An empty kernel's wall time IS the launch cost,
    as long as the host stays ahead of the device — which it does not, once the
    queue is deep, so we also report the async-submit rate separately.
    """
    try:
        import torch
    except ImportError:
        sys.exit("--mode launch-probe needs torch. The other three modes do not.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Nothing to measure; the cited 2.374 us stands.")

    dev = torch.device("cuda")
    print(f"device : {torch.cuda.get_device_name(dev)}")
    print(f"torch  : {torch.__version__}   CUDA {torch.version.cuda}")

    # The smallest real kernel we can issue from Python without writing CUDA.
    x = torch.empty(1, device=dev)

    def burst(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            x.add_(0.0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6  # us per launch

    burst(200)  # warm the allocator and the JIT
    serial = [burst(a.iters) for _ in range(a.repeats)]

    # Submit-only: how fast the host can push launches with the device behind.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        x.add_(0.0)
    submit_us = (time.perf_counter() - t0) / a.iters * 1e6
    torch.cuda.synchronize()

    table(
        [
            ("iterations per burst", f"{a.iters:,}"),
            ("end-to-end per launch (min of %d)" % a.repeats, f"{min(serial):8.3f} us"),
            ("end-to-end per launch (median)", f"{sorted(serial)[len(serial) // 2]:8.3f} us"),
            ("host submit only", f"{submit_us:8.3f} us"),
            ("book's cited driver figure", f"{CITED_LAUNCH_US:8.3f} us   [arXiv:2504.11750 Table V]"),
        ],
        "Measured on this machine — the one number in this lab you own",
    )
    print("\nThe end-to-end figure includes the PyTorch dispatcher and the Python")
    print("call; the cited figure is driver-only. Yours should be larger. The")
    print("difference is exactly the framework cost chapter 08-01 calls unquantified.")
    print(f"\nFeed it back in:  --launchcost {min(serial):.3f}")
    return 0


# ---------------------------------------------------------------------- shared


def table(rows: list[tuple[str, str]], title: str) -> None:
    if not rows:
        return
    w = max(len(k) for k, _ in rows)
    print(f"\n{title}\n" + "-" * (w + 34))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Capture one decode step and account for every microsecond in it.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["floor", "capture", "analyze", "launch-probe"],
                    help="floor: derive the prediction. capture: arm the engine's "
                         "profiler. analyze: read a trace. launch-probe: measure "
                         "per-launch cost on this box.")

    g = ap.add_argument_group("floor — derive the prediction (no GPU needed)")
    g.add_argument("--batch", type=int, default=32, help="batch size B (default: 32)")
    g.add_argument("--seq", type=int, default=2048, help="context tokens per sequence (default: 2048)")
    g.add_argument("--layers", type=int, default=32, help="L (default: 32, Llama-3-8B)")
    g.add_argument("--kv-heads", type=int, default=8, help="h_kv (default: 8)")
    g.add_argument("--head-dim", type=int, default=128, help="d_h (default: 128)")
    g.add_argument("--kv-dtype-bytes", type=int, default=2, choices=[1, 2, 4],
                   help="bytes per KV element; 1 for fp8 (default: 2)")
    g.add_argument("--weights-gb", type=float, default=15.01,
                   help="weight bytes streamed per step, GB (default: 15.01, "
                        "Llama-3-8B bf16 minus the embedding table)")
    g.add_argument("--card", default="h100-sxm", choices=sorted(HBM_TB_S),
                   help="card whose cited HBM bandwidth to use (default: h100-sxm)")
    g.add_argument("--hbm-tb-s", type=float, default=None,
                   help="override HBM bandwidth in TB/s; use YOUR measured number "
                        "from lab 01 if you have it")
    g.add_argument("--launches", type=int, default=330,
                   help="kernel launches per step (default: 330, chapter 08-01 TP=1)")

    c = ap.add_argument_group("capture — drive a live engine over HTTP")
    c.add_argument("--engine", default="vllm", choices=["vllm", "sglang"],
                   help="which engine is listening (default: vllm)")
    c.add_argument("--url", default="http://localhost:8000",
                   help="server base URL (SGLang usually :30000)")
    c.add_argument("--out", default="/tmp/lab10-traces",
                   help="trace output dir; SGLang reads this from the POST body, "
                        "vLLM from --profiler-config.torch_profiler_dir at launch")
    c.add_argument("--steps", type=int, default=5,
                   help="forward steps to capture, SGLang num_steps (default: 5)")
    c.add_argument("--by-stage", action="store_true",
                   help="SGLang profile_by_stage: separate prefill and decode traces")
    c.add_argument("--drive", action="store_true",
                   help="also generate load, so the capture lands on decode steps")
    c.add_argument("--concurrency", type=int, default=8,
                   help="concurrent requests when --drive (default: 8)")
    c.add_argument("--gen-tokens", type=int, default=128,
                   help="max tokens per driven request (default: 128)")
    c.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="model name for the vLLM completions payload")
    c.add_argument("--warmup-s", type=float, default=5.0,
                   help="seconds of load before arming the profiler (default: 5)")
    c.add_argument("--hold-s", type=float, default=2.0,
                   help="vLLM only: seconds between start and stop (default: 2)")
    c.add_argument("--timeout", type=float, default=600.0,
                   help="HTTP timeout, seconds (default: 600; SGLang's start_profile "
                        "blocks until the trace is flushed)")

    z = ap.add_argument_group("analyze — read a chrome trace")
    z.add_argument("--trace", help="path to a .json or .json.gz torch profiler trace")
    z.add_argument("--span", help="substring of the annotation naming the step to isolate")
    z.add_argument("--span-index", type=int, default=0,
                   help="which matching span, in time order (default: 0)")
    z.add_argument("--spans", action="store_true",
                   help="print every annotation name in the trace and exit")
    z.add_argument("--top", type=int, default=12, help="kernels to list (default: 12)")
    z.add_argument("--name-width", type=int, default=64, help="truncate kernel names")
    z.add_argument("--floor", type=float, default=None,
                   help="derived floor from --mode floor, to attribute the residual against")
    z.add_argument("--launchcost", type=float, default=None,
                   help="per-launch driver cost; omit to use the cited 2.374 us")

    p = ap.add_argument_group("launch-probe — needs torch and a GPU")
    p.add_argument("--iters", type=int, default=2000, help="launches per burst (default: 2000)")
    p.add_argument("--repeats", type=int, default=5, help="bursts (default: 5)")

    a = ap.parse_args()
    if a.mode == "analyze" and not a.trace:
        ap.error("--mode analyze needs --trace")
    return {
        "floor": mode_floor, "capture": mode_capture,
        "analyze": mode_analyze, "launch-probe": mode_launch_probe,
    }[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
