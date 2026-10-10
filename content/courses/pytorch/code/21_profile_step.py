"""Profile one training step on CPU, count its FLOPs, and compute a CPU MFU.

The model is a token-level MLP: ``Embedding(vocab, width)`` followed by
``depth`` blocks of ``Linear(width, width) + GELU`` and a ``Linear(width,
vocab)`` head, trained with AdamW (fused by default; ``--adamw forloop`` is
the CPU default implementation) on random (token, target) pairs. The
embedding output requires grad, as in a language model, so every Linear
computes both a grad-input and a grad-weight matmul and the FLOP count is
exactly ``6 * batch * sum(in * out)``.

The script

1. counts one step's FLOPs with ``torch.utils.flop_counter.FlopCounterMode``
   and compares them with the analytic ``6 * B * sum(in * out)``;
2. profiles six steps with ``torch.profiler`` under
   ``schedule(wait=1, warmup=1, active=3, repeat=1)``, with
   ``record_function`` ranges ``forward``, ``backward`` and ``optimizer``,
   and optionally exports a Chrome/Perfetto trace;
3. times the step with ``torch.utils.benchmark.Timer.blocked_autorange``;
4. measures an empirical matmul ceiling (a large fp32 square matmul timed the
   same way) and reports achieved GFLOP/s and "CPU MFU" against it, or against
   ``--peak-gflops`` if given. The ceiling is a measurement on this machine,
   not a datasheet number.

CPU only. Timings vary by machine and load; the FLOP counts do not.

Run from the repository root:

    python content/courses/pytorch/code/21_profile_step.py --trace-dir /tmp/pt-trace
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.flop_counter import FlopCounterMode

PHASES = ("forward", "backward", "optimizer")


def build_model(vocab: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Embedding(vocab, width)]
    for _ in range(depth):
        layers += [nn.Linear(width, width), nn.GELU()]
    layers.append(nn.Linear(width, vocab))
    return nn.Sequential(*layers)


def analytic_flops(model: nn.Module, batch: int) -> int:
    """6 * B * sum(in * out) over Linear layers: 2 forward + 2 grad-input + 2 grad-weight."""
    return 6 * batch * sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))


def make_step(model, opt, tokens, targets):
    def step():
        with record_function("forward"):
            loss = F.cross_entropy(model(tokens), targets)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss

    return step


def count_step_flops(step) -> int:
    with FlopCounterMode(display=False) as counter:
        step()
    return int(counter.get_total_flops())


def profile_steps(step, steps: int, trace_dir: str | None):
    """Run ``steps`` profiler steps; return (events of the active window, trace path)."""
    captured = {}

    def on_ready(prof):
        captured["events"] = list(prof.events())
        if trace_dir:
            path = Path(trace_dir) / "profile_step_trace.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(path))
            captured["trace"] = str(path)

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=on_ready,
        record_shapes=True,
    ) as prof:
        for _ in range(steps):
            step()
            prof.step()
    return captured["events"], captured.get("trace")


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def summarize(events):
    """Per-step medians, so one preempted step cannot reorder the top five.

    Every event is assigned to the ``ProfilerStep*`` range that contains its
    start; per op name we sum self CPU time and calls within each step and take
    the median over the active steps.
    """
    steps = sorted((e.time_range.start, e.time_range.end) for e in events if e.name.startswith("ProfilerStep"))

    def step_of(e):
        for i, (lo, hi) in enumerate(steps):
            if lo <= e.time_range.start <= hi:
                return i
        return None

    per_op: dict[str, list[list[float]]] = {}
    phase: dict[str, list[float]] = {name: [0.0] * len(steps) for name in PHASES}
    for e in events:
        i = step_of(e)
        if i is None:
            continue
        if e.name in phase:
            phase[e.name][i] += e.cpu_time_total
        elif e.name.startswith("aten::"):
            row = per_op.setdefault(e.name, [[0.0] * len(steps), [0] * len(steps)])
            row[0][i] += e.self_cpu_time_total
            row[1][i] += 1
    ranked = sorted(per_op.items(), key=lambda kv: _median(kv[1][0]), reverse=True)
    top_ops = [[name, round(_median(t), 1), _median(c)] for name, (t, c) in ranked[:5]]
    phase_ms = {name: round(_median(v) / 1000.0, 4) for name, v in phase.items()}
    return phase_ms, top_ops, len(steps)


def matmul_peak_gflops(n: int, threads: int, min_run_time: float) -> float:
    a, b = torch.randn(n, n), torch.randn(n, n)
    m = benchmark.Timer("a @ b", globals={"a": a, "b": b}, num_threads=threads).blocked_autorange(
        min_run_time=min_run_time
    )
    return 2 * n**3 / m.median / 1e9


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6, help="profiler steps (>= 5 for one full cycle)")
    parser.add_argument("--min-run-time", type=float, default=0.3, help="blocked_autorange budget for the step")
    parser.add_argument("--peak-n", type=int, default=1024, help="side of the square matmul used as the ceiling")
    parser.add_argument("--peak-gflops", type=float, default=None, help="override the measured ceiling")
    parser.add_argument("--adamw", choices=("fused", "forloop"), default="fused")
    parser.add_argument("--trace-dir", default=None, help="write a Chrome/Perfetto trace here")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    if args.steps < 5:
        parser.error("--steps must be >= 5 so the wait/warmup/active cycle completes")

    threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)

    model = build_model(args.vocab, args.width, args.depth)
    # fused=True: one aten::_fused_adamw_ call for all parameters. "forloop" is
    # the CPU default (foreach is only auto-selected on CUDA): ~10 small ops per
    # parameter, which dominates the profile of a model this small.
    fused = args.adamw == "fused"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused, foreach=False if not fused else None)
    gen = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(0, args.vocab, (args.batch,), generator=gen)
    targets = torch.randint(0, args.vocab, (args.batch,), generator=gen)
    step = make_step(model, opt, tokens, targets)

    step()  # first call: allocator, AdamW state creation, kernel selection
    flops = count_step_flops(step)
    events, trace = profile_steps(step, args.steps, args.trace_dir)
    phase_ms, top_ops, active_steps = summarize(events)

    timing = benchmark.Timer("step()", globals={"step": step}, num_threads=threads).blocked_autorange(
        min_run_time=args.min_run_time
    )
    measured_peak = matmul_peak_gflops(args.peak_n, threads, min_run_time=0.2)
    peak = args.peak_gflops if args.peak_gflops else measured_peak
    achieved = flops / timing.median / 1e9

    result = {
        "seed": args.seed,
        "threads": threads,
        "shape": {"batch": args.batch, "width": args.width, "depth": args.depth, "vocab": args.vocab},
        "adamw": args.adamw,
        "flops_per_step": flops,
        "analytic_flops_per_step": analytic_flops(model, args.batch),
        "median_step_s": timing.median,
        "iqr_step_s": timing.iqr,
        "achieved_gflops": round(achieved, 3),
        "matmul_peak_gflops": round(measured_peak, 3),
        "peak_used_gflops": round(peak, 3),
        "mfu": round(achieved / peak, 4),
        "top_ops": top_ops,
        "phase_ms": phase_ms,
        "active_steps": active_steps,
        "trace_path": trace,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
