"""Build, test and compile a custom op, then run a FlashAttention-style tiled attention on CPU.

Part (a): ``pytc_lab25::fused_bias_gelu(x, bias)`` is registered with
``torch.library.custom_op``. It has a Python CPU kernel computing
GELU(x + bias), a fake (meta) kernel for tracing, and a hand-written backward
that saves only the inputs and recomputes ``x + bias`` instead of saving
intermediates. ``torch.library.opcheck`` and float64 ``gradcheck`` verify the
registrations; saved-tensor hooks count the bytes autograd keeps for backward,
compared with the same GELU written as a composite of aten ops and with
``F.gelu(x + bias)``.

Part (b): the op sits inside a small MLP compiled with ``backend="aot_eager"``
through a counting backend. The custom op must not break the graph (expect 1
graph) and the compiled forward and backward must match eager.

Part (c): ``tiled_attention(q, k, v, block, causal)`` is the online-softmax
algorithm of FlashAttention written in plain PyTorch: it walks query blocks
and K/V blocks, keeps a running max ``m``, running sum ``l`` and an output
accumulator, and never allocates a ``(T, T)`` score tensor. A
``TorchDispatchMode`` records the numel of every tensor produced inside, and
the result is compared with ``F.scaled_dot_product_attention`` for causal
and full attention at a sequence length that is not a multiple of the block.

Python kernels on CPU prove registration, autograd, compile visibility and the
tiling algorithm. They are not fast; a real kernel would be Triton or CUDA.

Run from the repository root:

    python content/courses/pytorch/code/25_custom_ops.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

INV_SQRT2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- (a) the op
@torch.library.custom_op("pytc_lab25::fused_bias_gelu", mutates_args=())
def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU of x + bias; bias broadcasts over the last dimension."""
    u = x + bias
    return u * 0.5 * (1.0 + torch.erf(u * INV_SQRT2))


@fused_bias_gelu.register_fake
def _(x, bias):
    torch._check(bias.dim() == 1 and bias.shape[0] == x.shape[-1])
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    x, bias = inputs
    ctx.save_for_backward(x, bias)  # inputs only: no intermediate is kept alive


def _backward(ctx, grad):
    x, bias = ctx.saved_tensors
    u = x + bias  # recompute instead of saving
    cdf = 0.5 * (1.0 + torch.erf(u * INV_SQRT2))
    pdf = torch.exp(-0.5 * u * u) * INV_SQRT_2PI
    grad_u = grad * (cdf + u * pdf)
    return grad_u, grad_u.reshape(-1, grad_u.shape[-1]).sum(0)


fused_bias_gelu.register_autograd(_backward, setup_context=_setup_context)


def composite_bias_gelu(x, bias):
    """The same math as separate aten ops; autograd saves each op's needs."""
    u = x + bias
    return 0.5 * u * (1.0 + torch.erf(u * INV_SQRT2))


def saved_bytes(fn, *inputs):
    """Bytes of distinct storages autograd saves for backward while running fn."""
    seen = {}

    def pack(t):
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*inputs)
    return int(sum(seen.values()))


def check_op(gen):
    x = torch.randn(4, 8, dtype=torch.float64, generator=gen, requires_grad=True)
    b = torch.randn(8, dtype=torch.float64, generator=gen, requires_grad=True)
    report = torch.library.opcheck(fused_bias_gelu, (x, b))
    opcheck = {k: (v if isinstance(v, str) else repr(v)) for k, v in report.items()}
    grad_ok = torch.autograd.gradcheck(fused_bias_gelu, (x, b))
    torch.testing.assert_close(fused_bias_gelu(x, b), F.gelu(x + b), atol=1e-12, rtol=0)

    xs = torch.randn(64, 256, generator=gen, requires_grad=True)
    bs = torch.randn(256, generator=gen, requires_grad=True)
    return {
        "opcheck": opcheck,
        "gradcheck": bool(grad_ok),
        "saved_bytes_fused": saved_bytes(fused_bias_gelu, xs, bs),
        "saved_bytes_unfused": saved_bytes(composite_bias_gelu, xs, bs),
        "saved_bytes_aten_gelu": saved_bytes(lambda a, c: F.gelu(a + c), xs, bs),
    }


# --------------------------------------------------------------------------- (b) compile
class BiasGeluMLP(nn.Module):
    def __init__(self, d=16, hidden=32):
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.down = nn.Linear(hidden, d)

    def forward(self, x):
        return self.down(fused_bias_gelu(self.up(x), self.bias))


def check_compile(gen):
    model = BiasGeluMLP()
    with torch.no_grad():
        model.bias.copy_(torch.randn(32, generator=gen) * 0.1)
    x = torch.randn(8, 16, generator=gen)
    graphs = []
    aot_eager = torch._dynamo.lookup_backend("aot_eager")

    def counting_backend(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend=counting_backend)
        out_c = compiled(x)
        out_c.square().mean().backward()
        grads_c = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        out_e = model(x)
        out_e.square().mean().backward()
        grads_e = [p.grad.clone() for p in model.parameters()]
    finally:
        torch._dynamo.reset()
    diffs = [(out_c - out_e).abs().max().item()]
    diffs += [(gc - ge).abs().max().item() for gc, ge in zip(grads_c, grads_e)]
    in_graph = any(n.target is torch.ops.pytc_lab25.fused_bias_gelu.default
                   for gm in graphs for n in gm.graph.nodes)
    return {"compile_graphs": len(graphs), "compile_max_diff": float(max(diffs)),
            "op_in_graph": bool(in_graph)}


# --------------------------------------------------------------------------- (c) tiling
class NumelRecorder(TorchDispatchMode):
    """Records the largest tensor any aten op produces while active."""

    def __init__(self):
        super().__init__()
        self.max_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.max_numel = max(self.max_numel, t.numel())
        return out


def tiled_attention(q, k, v, block, causal=False):
    """FlashAttention-style forward. q, k, v: (B, H, T, d_head). Returns (out, tiles)."""
    T, d_head = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(d_head)
    outs, tiles = [], 0
    for qs in range(0, T, block):
        qi = q[..., qs:qs + block, :] * scale                    # (B, H, bq, d_head)
        bq = qi.shape[-2]
        m = torch.full(qi.shape[:-1], -math.inf, dtype=q.dtype)   # running max   (B, H, bq)
        l = torch.zeros(qi.shape[:-1], dtype=q.dtype)             # running sum   (B, H, bq)
        acc = torch.zeros_like(qi)                                # unnormalised output
        kv_end = min(T, qs + bq) if causal else T                 # skip tiles above the diagonal
        for ks in range(0, kv_end, block):
            kj, vj = k[..., ks:ks + block, :], v[..., ks:ks + block, :]
            s = qi @ kj.transpose(-2, -1)                         # (B, H, bq, bk): one tile
            if causal and ks + kj.shape[-2] > qs:                 # tile touches the diagonal
                qpos = torch.arange(qs, qs + bq).unsqueeze(-1)
                kpos = torch.arange(ks, ks + kj.shape[-2]).unsqueeze(0)
                s = s.masked_fill(kpos > qpos, -math.inf)
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new.unsqueeze(-1))
            alpha = torch.exp(m - m_new)                          # rescale old statistics
            l = l * alpha + p.sum(-1)
            acc = acc * alpha.unsqueeze(-1) + p @ vj
            m = m_new
            tiles += 1
        outs.append(acc / l.unsqueeze(-1))
    return torch.cat(outs, dim=-2), tiles


def check_tiling(gen, block, seq):
    T = seq if seq % block else seq + 1                           # force a ragged last tile
    B, H, d_head = 2, 2, 16
    q, k, v = (torch.randn(B, H, T, d_head, generator=gen) for _ in range(3))
    diffs, tiles, max_numel = {}, {}, 0
    with torch.no_grad():
        for name, causal in (("causal", True), ("full", False)):
            rec = NumelRecorder()
            with rec:
                out, n_tiles = tiled_attention(q, k, v, block, causal)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            diffs[name] = float((out - ref).abs().max().item())
            tiles[name] = n_tiles
            max_numel = max(max_numel, rec.max_numel)
    return {"tiled_max_diff": diffs, "tiles": tiles, "seq_len": T,
            "max_intermediate_numel": int(max_numel), "full_scores_numel": B * H * T * T}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--seq", type=int, default=96)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    gen = torch.Generator().manual_seed(args.seed)

    result = {}
    result.update(check_op(gen))
    result.update(check_compile(gen))
    result.update(check_tiling(gen, args.block, args.seq))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
