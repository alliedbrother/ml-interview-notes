"""Explore torch.compile capture behaviour: graph breaks, guards, dynamic shapes, regions, training.

Five experiments, each starting from ``torch._dynamo.reset()``:

1. A "messy" two-layer model with three deliberate graph-break sources (a
   ``print``, a Python branch on ``.item()``, a ``torch.compiler.disable``d
   helper) is analysed with ``torch._dynamo.explain``; a "clean" rewrite with
   ``torch.where`` and the helper inlined has zero breaks and equal outputs.
2. A recompilation sweep over sequence lengths 8, 16, 24, 32, 40, 48 (six
   shapes, below the default ``recompile_limit`` of 8) with ``dynamic=False``,
   ``None`` (automatic) and ``True`` counts compiled frames.
3. A guard trap: a Python-float module attribute changed between calls forces
   a recompile; the same value stored as a buffer tensor does not.
4. Regional compilation: a 6-block model compiled whole versus block by block;
   records compile wall time and frame/op counts (times are never asserted).
5. A compiled forward+loss whose backward runs the AOTAutograd backward graph,
   with the optimizer outside, matches an eager twin over 5 SGD steps.

Counting uses ``torch._dynamo.testing.CompileCounterWithBackend`` wrapped around
the chosen backend (default ``aot_eager``; Inductor is never used unless you pass
``--backend inductor``, which needs a working C++ toolchain). This demonstrates
capture behaviour, not speed.

Run from the repository root:

    python content/courses/pytorch/code/23_compile_explorer.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time

import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

LENGTHS = (8, 16, 24, 32, 40, 48)
D = 16


# ---------------------------------------------------------------- experiment 1
@torch.compiler.disable
def host_helper(h: torch.Tensor) -> torch.Tensor:
    """Stands in for code Dynamo must not trace (a C extension, a logger)."""
    return h + 1.0


class MessyModel(nn.Module):
    def __init__(self, d: int = D):
        super().__init__()
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)
        self.log = io.StringIO()

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        print("hidden computed", file=self.log)  # break 1: builtin print
        if h.mean().item() > 0:  # break 2: .item() leaves the graph
            h = h * 2.0
        else:
            h = h * 0.5
        h = host_helper(h)  # break 3: disabled function
        return self.fc2(h)


class CleanModel(MessyModel):
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.where(h.mean() > 0, h * 2.0, h * 0.5)  # stays a tensor op
        h = h + 1.0  # helper inlined
        return self.fc2(h)


def short_reason(reason) -> str:
    return str(getattr(reason, "reason", reason)).strip().splitlines()[0][:120]


def explain_report(model, x) -> dict:
    dynamo.reset()
    out = dynamo.explain(model)(x)
    dynamo.reset()
    return {
        "graphs": out.graph_count,
        "breaks": out.graph_break_count,
        "reasons": [short_reason(r) for r in out.break_reasons],
    }


def experiment_breaks(seed: int):
    torch.manual_seed(seed)
    messy = MessyModel()
    clean = CleanModel()
    clean.load_state_dict(messy.state_dict())
    x = torch.randn(4, D)
    messy_rep = explain_report(messy, x)
    clean_rep = explain_report(clean, x)
    with torch.no_grad():
        match = torch.allclose(messy(x), clean(x), rtol=1e-5, atol=1e-6)
        x_neg = -x.abs() * 10  # exercise the other branch too
        match = match and torch.allclose(messy(x_neg), clean(x_neg), rtol=1e-5, atol=1e-6)
    return messy_rep, clean_rep, bool(match)


# ---------------------------------------------------------------- experiment 2
class TinyBlock(nn.Module):
    def __init__(self, d: int = D):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class SweepBlock(nn.Module):
    """TinyBlock with a hand-written RMS norm: no float ``eps`` attribute to specialise on."""

    def __init__(self, d: int = D):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):  # x: (B, T, d_model)
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return x + self.fc2(F.gelu(self.fc1(h)))


def experiment_dynamic(seed: int, backend: str) -> dict:
    frames = {}
    for label, dyn in (("False", False), ("None", None), ("True", True)):
        dynamo.reset()
        torch.manual_seed(seed)
        block = SweepBlock()
        counter = CompileCounterWithBackend(backend)
        compiled = torch.compile(block, backend=counter, dynamic=dyn)
        with torch.no_grad():
            for t in LENGTHS:
                compiled(torch.randn(2, t, D))
        frames[label] = counter.frame_count
    dynamo.reset()
    return frames


# ---------------------------------------------------------------- experiment 3
class AttrScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, D)
        self.scale = 0.5  # plain Python float: Dynamo guards on its value

    def forward(self, x):
        return self.fc(x) * self.scale


class BufferScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, D)
        self.register_buffer("scale", torch.tensor(0.5))  # tensor: guarded on metadata only

    def forward(self, x):
        return self.fc(x) * self.scale


def experiment_guards(seed: int, backend: str):
    values = (0.5, 0.6, 0.7, 0.8)
    x = torch.randn(2, D)
    results = []
    for cls in (AttrScale, BufferScale):
        dynamo.reset()
        torch.manual_seed(seed)
        model = cls()
        counter = CompileCounterWithBackend(backend)
        compiled = torch.compile(model, backend=counter)
        with torch.no_grad():
            for v in values:
                if cls is AttrScale:
                    model.scale = v
                else:
                    model.scale.fill_(v)
                compiled(x)
        results.append(counter.frame_count - 1)
    dynamo.reset()
    return results[0], results[1]


# ---------------------------------------------------------------- experiment 4
class Stack(nn.Module):
    def __init__(self, n_blocks: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def experiment_regional(seed: int, backend: str) -> dict:
    x = torch.randn(2, 8, D)
    # Warm up the backend's one-off imports so neither timing pays for them.
    dynamo.reset()
    torch.compile(TinyBlock(), backend=backend)(x)

    dynamo.reset()
    torch.manual_seed(seed)
    whole_model = Stack()
    whole_counter = CompileCounterWithBackend(backend)
    t0 = time.perf_counter()
    torch.compile(whole_model, backend=whole_counter)(x)
    whole_s = time.perf_counter() - t0

    dynamo.reset()
    torch.manual_seed(seed)
    regional_model = Stack()
    block_counter = CompileCounterWithBackend(backend)
    t0 = time.perf_counter()
    for block in regional_model.blocks:
        block.compile(backend=block_counter)  # in place: state_dict keys unchanged
    regional_model(x)
    per_block_s = time.perf_counter() - t0
    dynamo.reset()
    return {
        "whole_frames": whole_counter.frame_count,
        "per_block_frames": block_counter.frame_count,
        "whole_ops": whole_counter.op_count,
        "per_block_ops": block_counter.op_count,
        "whole_s": round(whole_s, 4),
        "per_block_s": round(per_block_s, 4),
    }


# ---------------------------------------------------------------- experiment 5
def experiment_training(seed: int, backend: str, steps: int = 5) -> float:
    dynamo.reset()
    torch.manual_seed(seed)
    eager_model = Stack(n_blocks=2)
    compiled_model = Stack(n_blocks=2)
    compiled_model.load_state_dict(eager_model.state_dict())
    head = torch.compile(compiled_model, backend=backend)

    def loss_fn(model, x, y):
        return F.mse_loss(model(x), y)

    compiled_loss = torch.compile(loss_fn, backend=backend)
    opt_e = torch.optim.SGD(eager_model.parameters(), lr=0.05)
    opt_c = torch.optim.SGD(compiled_model.parameters(), lr=0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        x = torch.randn(4, 8, D, generator=gen)
        y = torch.randn(4, 8, D, generator=gen)
        for model, opt, fn in ((eager_model, opt_e, loss_fn), (head, opt_c, compiled_loss)):
            opt.zero_grad(set_to_none=True)
            fn(model, x, y).backward()  # compiled case runs the AOTAutograd backward graph
            opt.step()  # optimizer stays eager
    dynamo.reset()
    return max(
        (a - b).abs().max().item()
        for a, b in zip(eager_model.parameters(), compiled_model.parameters())
    )


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="aot_eager", choices=["eager", "aot_eager", "inductor"])
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    try:
        messy, clean, outputs_match = experiment_breaks(args.seed)
        frames = experiment_dynamic(args.seed, args.backend)
        attr_recompiles, buffer_recompiles = experiment_guards(args.seed, args.backend)
        regional = experiment_regional(args.seed, args.backend)
        train_diff = experiment_training(args.seed, args.backend)
    finally:
        dynamo.reset()

    result = {
        "backend": args.backend,
        "messy": messy,
        "clean": clean,
        "outputs_match": outputs_match,
        "frames_by_dynamic": frames,
        "attr_recompiles": attr_recompiles,
        "buffer_recompiles": buffer_recompiles,
        "regional": regional,
        "train_max_param_diff": train_diff,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
