"""Measure training memory buckets and what activation checkpointing saves.

Builds a stack of ``--layers`` pre-LN Transformer-like blocks (LayerNorm ->
QKV Linear -> causal SDPA -> output Linear, then LayerNorm -> Linear(4x) ->
GELU -> Linear, both with residuals, no dropout) on an input of shape
``(B, T, d_model)`` and reports:

* parameter, gradient and AdamW-state bytes, analytically and by summing tensors;
* bytes saved for backward, measured with ``saved_tensors_hooks`` (unique
  storages, parameter storages excluded) for four variants: no checkpointing,
  every block checkpointed, every second block checkpointed, and selective
  checkpointing that must-save matmul outputs;
* forward+backward time per variant (``torch.utils.benchmark``; informative
  only; ``--skip-timing`` reports null);
* the largest gradient difference between any variant and the plain run;
* the per-layer prediction ``4 * 16 * B * T * d_model`` bytes: Korthikanti et
  al.'s ``34 * s*b*h`` fp16 term with the two 1-byte dropout masks removed
  (32 bytes = 16 elements) and converted to fp32; the ``5*a*s/h`` attention
  matrix term is dropped because SDPA does not store the (B, H, T, T) matrix.

Selective checkpointing keeps its must-save outputs in a private cache that the
outer hook cannot see, so those bytes are added by the policy function, which
sees every op and its input shapes during the forward.

Scope: what autograd keeps alive between forward and backward, on CPU. Not the
allocator peak, temporaries, or GPU fragmentation.

Run from the repository root:

    python content/courses/pytorch/code/22_memory_accounting.py
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import benchmark
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

VARIANTS = ("none", "full", "every_2", "selective")
MATMULS = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, d // self.n_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, T, d_head)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class SaveCounter:
    """Policy for selective checkpointing that also counts what it keeps."""

    def __init__(self):
        self.bytes = 0

    def __call__(self, ctx, op, *args, **kwargs):
        if op in MATMULS:
            if not ctx.is_recompute:
                a, b = args[-2], args[-1]  # mm(a, b) or addmm(bias, a, b)
                self.bytes += a.shape[0] * b.shape[1] * a.element_size()
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE


def forward(blocks, x, mode, counter=None):
    for i, blk in enumerate(blocks):
        if mode == "full" or (mode == "every_2" and i % 2 == 0):
            x = checkpoint(blk, x, use_reentrant=False)
        elif mode == "selective":
            ctx_fn = functools.partial(create_selective_checkpoint_contexts, counter)
            x = checkpoint(blk, x, use_reentrant=False, context_fn=ctx_fn)
        else:
            x = blk(x)
    return x.pow(2).mean()


def saved_bytes(blocks, x, mode):
    """Bytes of unique non-parameter storages autograd keeps, plus the grads."""
    param_ptrs = {p.untyped_storage().data_ptr() for p in blocks.parameters()}
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in param_ptrs:
            seen[s.data_ptr()] = s.nbytes()
        return t

    counter = SaveCounter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = forward(blocks, x, mode, counter)
    total = sum(seen.values()) + counter.bytes
    blocks.zero_grad(set_to_none=True)
    loss.backward()
    return total, [p.grad.clone() for p in blocks.parameters()]


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--skip-timing", action="store_true",
                        help="report time_s as null (the test uses this to stay in budget)")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)

    L, d, B, T = args.layers, args.width, args.batch, args.seq
    blocks = nn.ModuleList(Block(d, args.heads) for _ in range(L))
    x = torch.randn(B, T, d)

    # (a) parameters, gradients, AdamW state
    n_params = sum(p.numel() for p in blocks.parameters())
    assert n_params == L * (12 * d * d + 13 * d)
    param_bytes = sum(p.numel() * p.element_size() for p in blocks.parameters())
    opt = torch.optim.AdamW(blocks.parameters(), lr=1e-3)
    forward(blocks, x, "none").backward()
    grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in blocks.parameters())
    opt.step()
    adamw_state_bytes = sum(
        v.numel() * v.element_size()
        for st in opt.state.values()
        for k, v in st.items()
        if k in ("exp_avg", "exp_avg_sq")
    )
    opt.zero_grad(set_to_none=True)

    # (b) saved activations, (d) gradient equality
    measured = {m: saved_bytes(blocks, x, m) for m in VARIANTS}
    saved = {m: b for m, (b, _) in measured.items()}
    ref = measured["none"][1]
    max_grad_diff = 0.0
    for m in VARIANTS[1:]:
        for g0, g1 in zip(ref, measured[m][1]):
            max_grad_diff = max(max_grad_diff, (g0 - g1).abs().max().item())

    # (c) wall time, informative only: Timer.timeit(1) runs 2 warm-ups + 1 timed
    def step(mode):
        blocks.zero_grad(set_to_none=True)
        forward(blocks, x, mode, SaveCounter()).backward()

    time_s = {m: None for m in VARIANTS}
    if not args.skip_timing:
        for m in VARIANTS:
            timer = benchmark.Timer("step(m)", globals={"step": step, "m": m})
            time_s[m] = round(timer.timeit(1).median, 5)
    blocks.zero_grad(set_to_none=True)

    # (e) prediction
    predicted = L * 4 * 16 * B * T * d
    result = {
        "config": {"layers": L, "width": d, "heads": args.heads, "batch": B, "seq": T},
        "n_params": n_params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "adamw_state_bytes": adamw_state_bytes,
        "saved_bytes": saved,
        "saved_ratio_vs_none": {m: round(saved[m] / saved["none"], 4) for m in VARIANTS},
        "time_s": time_s,
        "max_grad_diff": max_grad_diff,
        "predicted_saved_bytes_none": predicted,
        "measured_over_predicted": round(saved["none"] / predicted, 4),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
