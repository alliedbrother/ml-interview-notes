"""Train one small MLP four ways: fp32, bf16 autocast, fp16 + GradScaler, naive fp16.

Every variant multiplies its loss by ``--grad-multiplier`` (default 1e-6) before
backward and divides the gradients by it again in fp32 afterwards. That models
a network whose true gradients are tiny. fp32 and bf16 have an 8-bit exponent,
so the tiny gradients survive. fp16 has a 5-bit exponent, so without a loss
scaler most of them flush to zero inside the fp16 backward matmuls. With
``torch.amp.GradScaler("cpu")`` they survive. The scaler starts at a
deliberately huge scale (2**40) so the trace shows real overflow, skipped steps
and halving, then growth every ``--growth-interval`` clean steps.

CPU autocast emulates the numerics of mixed precision. It does not show GPU
speedups. TF32 and fp8 are not exercised.

Run from the repository root:

    python content/courses/pytorch/code/17_mixed_precision.py
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

VARIANTS = ("fp32", "bf16", "fp16_scaler", "fp16_naive")


def make_task(seed: int, steps: int, batch: int = 64, n_eval: int = 512,
              d_in: int = 16, noise: float = 0.3):
    """Fresh (batch, d_in) training batches for every step plus a held-out set.

    Targets come from a fixed random tanh "teacher" network plus Gaussian
    noise, so the best possible held-out MSE is about noise**2 = 0.09 and the
    final losses sit near a floor instead of near zero.
    """
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(d_in, 8, generator=g) / math.sqrt(d_in)
    w2 = torch.randn(8, 1, generator=g) / math.sqrt(8)

    def sample(n):
        x = torch.randn(n, d_in, generator=g)
        y = 2.0 * torch.tanh(1.5 * x @ w1) @ w2
        return x, y + noise * torch.randn(n, 1, generator=g)

    x_tr, y_tr = sample(steps * batch)
    x_ev, y_ev = sample(n_eval)
    return x_tr.view(steps, batch, d_in), y_tr.view(steps, batch, 1), x_ev, y_ev


def make_model(seed: int, d_in: int = 16, width: int = 64) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, 1),
    )


def grad_zero_fraction(model: nn.Module) -> float:
    zeros = total = 0
    for p in model.parameters():
        zeros += int((p.grad == 0).sum())
        total += p.grad.numel()
    return zeros / total


def snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def unchanged(model: nn.Module, before: list[torch.Tensor]) -> bool:
    return all(torch.equal(p, b) for p, b in zip(model.parameters(), before))


def train_variant(variant: str, task, *, seed: int, steps: int, lr: float,
                  grad_multiplier: float, init_scale: float,
                  growth_interval: int, clip: float = 1.0) -> dict:
    model = make_model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dtype = {"bf16": torch.bfloat16, "fp16_scaler": torch.float16,
             "fp16_naive": torch.float16}.get(variant)
    scaler = None
    if variant == "fp16_scaler":
        scaler = torch.amp.GradScaler("cpu", init_scale=init_scale,
                                      growth_interval=growth_interval)

    losses, zero_fracs, scale_trace = [], [], []
    skipped = 0
    skipped_params_unchanged = True
    param_dtypes = set()

    xs, ys, x_ev, y_ev = task
    for step in range(steps):
        x, y = xs[step], ys[step]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=dtype or torch.bfloat16,
                            enabled=dtype is not None):
            pred = model(x)
            loss = F.mse_loss(pred, y)  # on CPU's fp32 autocast list
        losses.append(loss.item())
        small_loss = loss * grad_multiplier

        if scaler is None:
            small_loss.backward()
            for p in model.parameters():  # undo the multiplier in fp32
                p.grad.div_(grad_multiplier)
            zero_fracs.append(grad_zero_fraction(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        else:
            scale_before = scaler.get_scale()
            before = snapshot(model)
            scaler.scale(small_loss).backward()
            scaler.unscale_(opt)  # grads now = true grad * grad_multiplier
            for p in model.parameters():
                p.grad.div_(grad_multiplier)
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters())
            if finite:
                zero_fracs.append(grad_zero_fraction(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            scale_after = scaler.get_scale()
            scale_trace.append(scale_before)
            if scale_after < scale_before:  # update() backed off: step was skipped
                skipped += 1
                skipped_params_unchanged &= unchanged(model, before)
        param_dtypes.update(str(p.dtype) for p in model.parameters())

    with torch.no_grad():
        final = F.mse_loss(model(x_ev), y_ev).item()
    out = {
        "losses": losses,
        "final_loss": final,
        "zero_grad_frac": sum(zero_fracs) / max(1, len(zero_fracs)),
        "param_dtypes": sorted(param_dtypes),
    }
    if scaler is not None:
        out.update(scale_trace=scale_trace + [scaler.get_scale()],
                   skipped_steps=skipped,
                   skipped_params_unchanged=skipped_params_unchanged,
                   scaler_state=scaler.state_dict())
    return out


def forced_inf_step(seed: int = 0) -> dict:
    """One GradScaler step with an injected inf gradient, then one clean step."""
    xs, ys, _, _ = make_task(seed, steps=1, batch=32)
    x, y = xs[0], ys[0]
    model = make_model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)

    def one_step(inject_inf: bool):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.float16):
            loss = F.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        if inject_inf:
            next(model.parameters()).grad[0, 0] = float("inf")
        before = snapshot(model)
        scaler.step(opt)
        scaler.update()
        return unchanged(model, before), scaler.get_scale()

    bad_unchanged, scale_after_bad = one_step(True)
    good_unchanged, scale_after_good = one_step(False)
    return {"inf_step_params_unchanged": bad_unchanged,
            "scale_after_inf": scale_after_bad,
            "clean_step_params_unchanged": good_unchanged,
            "scale_after_clean": scale_after_good}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--grad-multiplier", type=float, default=1e-6)
    parser.add_argument("--init-scale", type=float, default=2.0 ** 40)
    parser.add_argument("--growth-interval", type=int, default=50)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    task = make_task(args.seed, args.steps)
    with torch.no_grad():
        initial = F.mse_loss(make_model(args.seed)(task[2]), task[3]).item()
    runs = {v: train_variant(v, task, seed=args.seed, steps=args.steps, lr=args.lr,
                             grad_multiplier=args.grad_multiplier,
                             init_scale=args.init_scale,
                             growth_interval=args.growth_interval)
            for v in VARIANTS}
    sc = runs["fp16_scaler"]
    result = {
        "initial_loss": initial,
        "final_loss": {v: runs[v]["final_loss"] for v in VARIANTS},
        "zero_grad_frac": {v: runs[v]["zero_grad_frac"] for v in VARIANTS},
        "zero_grad_frac_naive": runs["fp16_naive"]["zero_grad_frac"],
        "zero_grad_frac_scaled": sc["zero_grad_frac"],
        "scale_trace": sc["scale_trace"],
        "skipped_steps": sc["skipped_steps"],
        "skipped_params_unchanged": sc["skipped_params_unchanged"],
        "scaler_state": sc["scaler_state"],
        "param_dtypes": sorted({d for v in VARIANTS for d in runs[v]["param_dtypes"]}),
        "loss_curves": {v: runs[v]["losses"] for v in VARIANTS},
        "forced_inf": forced_inf_step(args.seed),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    out = main()
    out.pop("loss_curves")  # keep the printout short; --json-out keeps them
    runs = []  # print the scale trace as [log2(scale), number of steps] runs
    for s in out.pop("scale_trace"):
        e = int(math.log2(s))
        if runs and runs[-1][0] == e:
            runs[-1][1] += 1
        else:
            runs.append([e, 1])
    out["scale_trace_log2_runs"] = runs
    print(json.dumps(out, indent=2))
