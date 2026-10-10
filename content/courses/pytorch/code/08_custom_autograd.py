"""Memory-saving fused Linear+GELU written as a custom autograd.Function.

The reference ``F.gelu(F.linear(x, W, b))`` keeps the (B, d_out) pre-activation
alive for backward. ``LinearGELUFunction`` saves only ``x, W, b`` and recomputes
the pre-activation in backward, using the exact GELU derivative
``Phi(z) + z * phi(z)``. The script checks gradients with ``gradcheck``,
compares outputs and gradients with the reference, measures the bytes each
version saves for backward with ``saved_tensors_hooks``, and trains both
versions for 50 SGD steps on the same seeded regression task.

Run from the repository root:

    python content/courses/pytorch/code/08_custom_autograd.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.autograd.graph import saved_tensors_hooks

INV_SQRT2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """d/dz [z * Phi(z)] = Phi(z) + z * phi(z) for the exact (erf) GELU."""
    cdf = 0.5 * (1.0 + torch.erf(z * INV_SQRT2))
    pdf = torch.exp(-0.5 * z * z) * INV_SQRT_2PI
    return cdf + z * pdf


def _linear_gelu_backward(x, weight, bias, grad_out, needs):
    z = F.linear(x, weight, bias)  # recompute (B, d_out): one extra matmul
    grad_z = grad_out * gelu_grad(z)
    grad_x = grad_z @ weight if needs[0] else None
    grad_w = grad_z.transpose(0, 1) @ x if needs[1] else None
    grad_b = grad_z.sum(0) if needs[2] else None
    return grad_x, grad_w, grad_b


class LinearGELUFunction(torch.autograd.Function):
    """gelu(x @ W.T + b); saves x, W, b and nothing of shape (B, d_out)."""

    @staticmethod
    def forward(x, weight, bias):
        return F.gelu(F.linear(x, weight, bias))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias = inputs
        ctx.save_for_backward(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        # Written with differentiable torch ops, so double backward also works.
        return _linear_gelu_backward(x, weight, bias, grad_out, ctx.needs_input_grad)


class LinearGELUOnce(LinearGELUFunction):
    """Same math, but the backward is declared first-order only."""

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        return _linear_gelu_backward(x, weight, bias, grad_out, ctx.needs_input_grad)


class FusedLinearGELU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearGELUFunction.apply(x, self.linear.weight, self.linear.bias)


class ReferenceLinearGELU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.linear(x))


def saved_bytes(fn: Callable[[], torch.Tensor]) -> tuple[int, torch.Tensor]:
    """Sum of nbytes over unique storages packed for backward while running fn."""
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return sum(storages.values()), out


def expected_fused_bytes(batch: int, d_in: int, d_out: int, itemsize: int = 4) -> int:
    return itemsize * (batch * d_in + d_out * d_in + d_out)


def run_gradcheck(seed: int) -> bool:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, dtype=torch.float64, generator=g, requires_grad=True)
    w = torch.randn(5, 3, dtype=torch.float64, generator=g, requires_grad=True)
    b = torch.randn(5, dtype=torch.float64, generator=g, requires_grad=True)
    first = torch.autograd.gradcheck(LinearGELUFunction.apply, (x, w, b))
    second = torch.autograd.gradgradcheck(LinearGELUFunction.apply, (x, w, b))
    return bool(first and second)


def train(model: nn.Module, head: nn.Linear, xs, ys, steps: int, lr: float) -> list[float]:
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(head(model(xs)), ys)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--d-in", type=int, default=128)
    parser.add_argument("--d-out", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # (a) gradcheck + gradgradcheck in float64 on a 4x3 -> 5 instance.
    ok = run_gradcheck(args.seed)

    # (b) outputs and grads vs the reference, float32, configured shape.
    torch.manual_seed(args.seed)
    ref = ReferenceLinearGELU(args.d_in, args.d_out)
    fused = FusedLinearGELU(args.d_in, args.d_out)
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(args.batch, args.d_in)
    x_ref = x.clone().requires_grad_(True)  # an activation from an earlier layer
    x_fus = x.clone().requires_grad_(True)

    # (c) bytes saved for backward, measured while building each graph.
    ref_bytes, y_ref = saved_bytes(lambda: ref(x_ref))
    fus_bytes, y_fus = saved_bytes(lambda: fused(x_fus))
    max_out = (y_ref - y_fus).abs().max().item()
    upstream = torch.randn_like(y_ref)
    y_ref.backward(upstream)
    y_fus.backward(upstream)
    pairs = [(x_ref.grad, x_fus.grad)] + [
        (p.grad, q.grad) for p, q in zip(ref.parameters(), fused.parameters())
    ]
    max_grad = max((a - b).abs().max().item() for a, b in pairs)

    # (d) 50 SGD steps on a seeded regression task, same init for both.
    g = torch.Generator().manual_seed(args.seed + 1)
    xs = torch.randn(args.batch, args.d_in, generator=g)
    ys = torch.sin(xs[:, :1]) + 0.1 * torch.randn(args.batch, 1, generator=g)
    torch.manual_seed(args.seed + 2)
    m_ref = ReferenceLinearGELU(args.d_in, args.d_out)
    h_ref = nn.Linear(args.d_out, 1)
    m_fus = FusedLinearGELU(args.d_in, args.d_out)
    h_fus = nn.Linear(args.d_out, 1)
    m_fus.load_state_dict(m_ref.state_dict())
    h_fus.load_state_dict(h_ref.state_dict())
    curve_ref = train(m_ref, h_ref, xs, ys, args.steps, lr=0.05)
    curve_fus = train(m_fus, h_fus, xs, ys, args.steps, lr=0.05)
    curve_diff = max(abs(a - b) for a, b in zip(curve_ref, curve_fus))

    result = {
        "gradcheck": ok,
        "max_out_diff": max_out,
        "max_grad_diff": max_grad,
        "saved_bytes_reference": ref_bytes,
        "saved_bytes_fused": fus_bytes,
        "saved_bytes_fused_expected": expected_fused_bytes(args.batch, args.d_in, args.d_out),
        "saved_ratio": fus_bytes / ref_bytes,
        "loss_first": curve_ref[0],
        "loss_last": curve_ref[-1],
        "loss_curve_max_diff": curve_diff,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
