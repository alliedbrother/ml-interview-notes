"""Quantize one trained MLP several ways and measure size, accuracy and logit error.

A 64 -> 128 -> 128 -> 10 ReLU MLP is trained on a seeded synthetic Gaussian
mixture. Every compression scheme below is implemented from scratch with plain
tensors so the arithmetic is visible:

1. ``int8_tensor``      int8 symmetric weight-only, one scale per weight matrix
2. ``int8_channel``     int8 symmetric weight-only, one scale per output row
                        (``Int8WeightOnlyLinear``: int8 buffer + fp32 scales)
3. ``int4_group32``     int4 symmetric weight-only, one scale per 32 inputs,
                        two int4 codes packed per uint8 byte
4. ``ptq_static_int8``  static post-training quantization: uint8 activations
                        with min/max ranges from 8 calibration batches, int8
                        per-channel weights, int32 accumulation, fp rescale
5. ``qat_int8``         5 epochs of fine-tuning with fake quantization and the
                        straight-through estimator, then converted to (4)
6. ``sparse_2_4``       2:4 magnitude mask on every weight + 5 epochs of
                        masked fine-tuning (the sparse layout is simulated;
                        ``to_sparse_semi_structured`` is CUDA-only)
7. ``ao_dynamic``       optional (``--with-ao-dynamic``): the deprecated
                        ``torch.ao.quantization.quantize_dynamic`` with an
                        engine from ``torch.backends.quantized.supported_engines``

Speed is not measured: dequantize-then-matmul in Python is slower than fp32 on
CPU. Byte counts are what a checkpoint would hold (packed codes, scales, zero
points, fp32 biases).

Run from the repository root:

    python content/courses/pytorch/code/29_quantization.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = (64, 128, 128, 10)


# ---------------------------------------------------------------- data, model
def make_data(seed: int, n_train: int = 2048, n_test: int = 1024,
              n_classes: int = 10, d: int = 64, spread: float = 1.0):
    """Gaussian mixture: one random centre per class plus isotropic noise."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n_classes, d, generator=g) * spread * 0.45

    def sample(n):
        y = torch.randint(0, n_classes, (n,), generator=g)
        x = centres[y] + torch.randn(n, d, generator=g)
        return x, y

    return sample(n_train) + sample(n_test)


def make_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(DIMS[0], DIMS[1]), nn.ReLU(),
                         nn.Linear(DIMS[1], DIMS[2]), nn.ReLU(),
                         nn.Linear(DIMS[2], DIMS[3]))


def linears(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def train(model, x, y, *, epochs, lr, seed, batch=128, after_step=None):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if after_step is not None:
                after_step()
    model.eval()
    return model


def fp32_bytes(model) -> int:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


# ---------------------------------------------------------------- primitives
def sym_scale(w: torch.Tensor, qmax: int, dim=None) -> torch.Tensor:
    """Symmetric scale max|w| / qmax, over everything or along ``dim``."""
    amax = w.abs().amax() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    return amax.clamp_min(1e-12) / qmax


def quant_sym(w, s, qmin, qmax):
    return torch.clamp(torch.round(w / s), qmin, qmax)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Two signed int4 codes in [-8, 7] per uint8: low nibble first."""
    u = (q.to(torch.int16) & 0xF).to(torch.uint8).reshape(-1, 2)
    return u[:, 0] | (u[:, 1] << 4)


def unpack_int4(packed: torch.Tensor, shape) -> torch.Tensor:
    lo = (packed & 0xF).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    u = torch.stack([lo, hi], dim=1).reshape(-1)
    return torch.where(u >= 8, u - 16, u).to(torch.int8).reshape(shape)


class Int8WeightOnlyLinear(nn.Module):
    """int8 weight codes + per-output-channel fp32 scales; fp32 activations."""

    def __init__(self, linear: nn.Linear, per_channel: bool = True):
        super().__init__()
        w = linear.weight.detach()
        s = sym_scale(w, 127, dim=1) if per_channel else sym_scale(w, 127).reshape(1, 1)
        self.register_buffer("qweight", quant_sym(w, s, -127, 127).to(torch.int8))
        self.register_buffer("scale", s.reshape(-1).to(torch.float32))
        self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale[:, None]   # dequantize in "kernel"
        return F.linear(x, w, self.bias)


class Int4GroupLinear(nn.Module):
    """int4 symmetric per-group weights, nibble-packed into uint8."""

    def __init__(self, linear: nn.Linear, group: int = 32):
        super().__init__()
        w = linear.weight.detach()
        out_f, in_f = w.shape
        assert in_f % group == 0
        wg = w.reshape(out_f, in_f // group, group)
        s = sym_scale(wg, 7, dim=2)                           # (out, in/group, 1)
        q = quant_sym(wg, s, -8, 7).to(torch.int8)
        self.shape = (out_f, in_f)
        self.group = group
        self.register_buffer("packed", pack_int4(q.reshape(-1)))
        self.register_buffer("scale", s.reshape(out_f, -1).to(torch.float32))
        self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x):
        q = unpack_int4(self.packed, self.shape).to(x.dtype)
        w = (q.reshape(self.shape[0], -1, self.group) * self.scale[..., None]).reshape(self.shape)
        return F.linear(x, w, self.bias)


def swap_linears(model, factory) -> nn.Sequential:
    return nn.Sequential(*[factory(m) if isinstance(m, nn.Linear) else m for m in model])


def buffer_bytes(model) -> int:
    return sum(b.numel() * b.element_size() for b in model.buffers())


# ---------------------------------------------------------------- static PTQ
class MinMaxObserver:
    """Running min/max of a tensor -> asymmetric uint8 (scale, zero_point)."""

    def __init__(self):
        self.lo, self.hi = float("inf"), float("-inf")

    def update(self, x):
        self.lo = min(self.lo, x.min().item())
        self.hi = max(self.hi, x.max().item())

    def qparams(self):
        lo, hi = min(self.lo, 0.0), max(self.hi, 0.0)       # 0.0 must be exact
        s = max(hi - lo, 1e-8) / 255.0
        z = int(min(255, max(0, round(-lo / s))))
        return s, z


def calibrate(model, x, n_batches=8, batch=128):
    """Observe the input of every Linear over ``n_batches`` batches."""
    obs = [MinMaxObserver() for _ in linears(model)]
    hooks = [lin.register_forward_pre_hook(lambda m, a, o=o: o.update(a[0]))
             for lin, o in zip(linears(model), obs)]
    try:
        with torch.no_grad():
            for i in range(n_batches):
                model(x[i * batch:(i + 1) * batch])
    finally:
        for h in hooks:
            h.remove()
    return [o.qparams() for o in obs]


class StaticInt8Linear(nn.Module):
    """uint8 activation x int8 weight -> int32 accumulate -> fp32 rescale.

    acc = sum_k (qx_k - zx) qw_k = qx @ qw^T - zx * sum_k qw_k   (int32)
    y   = sx * sw * acc + b
    """

    def __init__(self, linear: nn.Linear, act_qparams):
        super().__init__()
        w = linear.weight.detach()
        sw = sym_scale(w, 127, dim=1)
        self.register_buffer("qweight", quant_sym(w, sw, -127, 127).to(torch.int8))
        self.register_buffer("w_scale", sw.reshape(-1).to(torch.float32))
        self.register_buffer("bias", linear.bias.detach().clone())
        sx, zx = act_qparams
        self.register_buffer("x_scale", torch.tensor(sx, dtype=torch.float32))
        self.register_buffer("x_zero", torch.tensor(zx, dtype=torch.int32))
        # precomputed zero-point correction term, one int32 per output row
        self.register_buffer("w_rowsum", self.qweight.to(torch.int32).sum(1))

    def forward(self, x):
        qx = torch.clamp(torch.round(x / self.x_scale) + self.x_zero, 0, 255).to(torch.int32)
        acc = qx @ self.qweight.to(torch.int32).T - self.x_zero * self.w_rowsum
        return acc.to(torch.float32) * (self.x_scale * self.w_scale) + self.bias


def to_static_int8(model, x_calib):
    qparams = iter(calibrate(model, x_calib))
    return swap_linears(model, lambda m: StaticInt8Linear(m, next(qparams)))


# ---------------------------------------------------------------- QAT
def fake_quant_ste(x, s, z, qmin, qmax):
    """quantize -> dequantize in fp; gradient 1 inside the range, 0 outside."""
    q = torch.clamp(torch.round(x / s) + z, qmin, qmax)
    fq = (q - z) * s
    inside = ((x / s + z) >= qmin) & ((x / s + z) <= qmax)
    ste = x + (fq - x).detach()                 # straight-through: d ste / d x = 1
    return torch.where(inside, ste, fq)         # clipped: gradient 0


class QATLinear(nn.Module):
    """Linear whose weight (per-channel int8) and input (uint8) are fake-quantized.

    The activation range is an EMA of observed min/max (momentum 0.9), the
    weight scale is recomputed from the current weight every step.
    """

    def __init__(self, linear: nn.Linear, act_qparams):
        super().__init__()
        self.linear = copy.deepcopy(linear)
        s, z = act_qparams
        self.lo, self.hi = -z * s, (255 - z) * s

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.lo = 0.9 * self.lo + 0.1 * min(x.min().item(), 0.0)
                self.hi = 0.9 * self.hi + 0.1 * max(x.max().item(), 0.0)
        s = max(self.hi - self.lo, 1e-8) / 255.0
        z = float(min(255, max(0, round(-self.lo / s))))
        xq = fake_quant_ste(x, s, z, 0, 255)
        w = self.linear.weight
        sw = sym_scale(w.detach(), 127, dim=1)
        wq = fake_quant_ste(w, sw, 0.0, -127, 127)
        return F.linear(xq, wq, self.linear.bias)


def qat(model, x, y, x_calib, *, epochs, seed):
    qparams = iter(calibrate(model, x_calib))
    qmodel = swap_linears(copy.deepcopy(model), lambda m: QATLinear(m, next(qparams)))
    train(qmodel, x, y, epochs=epochs, lr=5e-4, seed=seed + 1)
    plain = nn.Sequential(*[m.linear if isinstance(m, QATLinear) else m for m in qmodel])
    return to_static_int8(plain, x_calib)


# ---------------------------------------------------------------- 2:4 sparsity
def mask_2_4(w: torch.Tensor) -> torch.Tensor:
    """Keep the 2 largest |w| in every contiguous group of 4 along in-features."""
    out_f, in_f = w.shape
    g = w.abs().reshape(out_f, in_f // 4, 4)
    keep = g.topk(2, dim=-1).indices
    m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, keep, True)
    return m.reshape(out_f, in_f)


def sparse_2_4(model, x, y, *, epochs, seed):
    sm = copy.deepcopy(model)
    masks = [mask_2_4(lin.weight.detach()) for lin in linears(sm)]

    def apply():
        with torch.no_grad():
            for lin, m in zip(linears(sm), masks):
                lin.weight.mul_(m)

    apply()
    train(sm, x, y, epochs=epochs, lr=5e-4, seed=seed + 2, after_step=apply)
    zeros = [(lin.weight.detach() == 0).reshape(lin.out_features, -1, 4).sum(-1)
             for lin in linears(sm)]
    counts = torch.cat([z.reshape(-1) for z in zeros])
    total = sum(lin.weight.numel() for lin in linears(sm))
    # 2:4 storage: half the values (fp32) + 2-bit index per kept value
    w_bytes = sum(lin.weight.numel() // 2 * 4 + lin.weight.numel() // 2 * 2 // 8
                  for lin in linears(sm))
    b_bytes = sum(lin.bias.numel() * 4 for lin in linears(sm))
    return sm, {
        "zero_fraction": sum(int(z.sum()) for z in zeros) / total,
        "group_zero_counts": sorted(set(counts.tolist())),
        "bytes": w_bytes + b_bytes,
    }


# ---------------------------------------------------------------- torch.ao
def ao_dynamic(model, x_test):
    """quantize_dynamic with a supported engine; restores the engine."""
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    preferred = [e for e in ("fbgemm", "x86", "qnnpack") if e in engines]
    if not preferred:
        return None, {"skipped": f"no quantized engine in {torch.backends.quantized.supported_engines}"}
    old = torch.backends.quantized.engine
    try:
        torch.backends.quantized.engine = preferred[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from torch.ao.quantization import quantize_dynamic
            qm = quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
        with torch.no_grad():
            logits = qm(x_test)
        n_bytes = 0
        for m in qm.modules():
            if hasattr(m, "weight") and callable(m.weight):
                qw = m.weight()
                n_bytes += qw.int_repr().numel() + 8   # int8 codes + fp32 scale + int32 zero point
                n_bytes += m.bias().numel() * 4
        return logits, {"engine": preferred[0], "bytes": n_bytes}
    except Exception as exc:  # pragma: no cover - platform dependent
        return None, {"skipped": f"{type(exc).__name__}: {exc}"}
    finally:
        torch.backends.quantized.engine = old


# ---------------------------------------------------------------- main
def evaluate(model, x, y, ref_logits=None):
    with torch.no_grad():
        logits = model(x)
    out = {"acc": (logits.argmax(1) == y).float().mean().item()}
    if ref_logits is not None:
        out["logit_mse"] = F.mse_loss(logits, ref_logits).item()
    return out, logits


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--with-ao-dynamic", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    x_tr, y_tr, x_te, y_te = make_data(args.seed)
    model = train(make_model(args.seed), x_tr, y_tr, epochs=args.epochs, lr=3e-3,
                  seed=args.seed)
    x_cal = x_tr[:8 * 128]

    fp32, ref = evaluate(model, x_te, y_te)
    result = {"fp32": {**fp32, "bytes": fp32_bytes(model)}}

    variants = {
        "int8_tensor": swap_linears(model, lambda m: Int8WeightOnlyLinear(m, per_channel=False)),
        "int8_channel": swap_linears(model, Int8WeightOnlyLinear),
        "int4_group32": swap_linears(model, Int4GroupLinear),
        "ptq_static_int8": to_static_int8(model, x_cal),
        "qat_int8": qat(model, x_tr, y_tr, x_cal, epochs=args.finetune_epochs, seed=args.seed),
    }
    for name, qm in variants.items():
        metrics, _ = evaluate(qm, x_te, y_te, ref)
        result[name] = {**metrics, "bytes": buffer_bytes(qm)}

    sm, info = sparse_2_4(model, x_tr, y_tr, epochs=args.finetune_epochs, seed=args.seed)
    metrics, _ = evaluate(sm, x_te, y_te, ref)
    result["sparse_2_4"] = {**metrics, **info}

    if args.with_ao_dynamic:
        logits, info = ao_dynamic(model, x_te)
        if logits is not None:
            info["acc"] = (logits.argmax(1) == y_te).float().mean().item()
            info["logit_mse"] = F.mse_loss(logits, ref).item()
        result["ao_dynamic"] = info
    else:
        result["ao_dynamic"] = {"skipped": "pass --with-ao-dynamic to run torch.ao"}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
