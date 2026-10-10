"""SGD, AdamW and a warmup-cosine schedule from scratch, checked against torch.optim.

Module 15 lab. It implements three ``torch.optim.Optimizer`` subclasses:

* ``ScratchSGD``: momentum, dampening, Nesterov and L2 weight decay, written
  in PyTorch's form ``b = mu * b + (1 - dampening) * g; p -= lr * b``;
* ``ScratchAdamW``: bias-corrected Adam moments with decoupled weight decay
  ``p *= 1 - lr * wd`` applied before the Adam step (amsgrad off);
* ``SignMomentum``: sign of a momentum buffer (a Lion-like teaching optimizer);

plus ``warmup_cosine(step)``, a closed-form linear-warmup + cosine schedule.

On a seeded float64 synthetic regression task with a small MLP
(Linear -> LayerNorm -> Tanh -> Linear) it then:

(a) trains scratch and torch.optim side by side for SGD and AdamW under the
    same LambdaLR schedule and reports the largest parameter difference;
(b) compares AdamW ``foreach=False`` with ``foreach=True`` (and ``fused=True``,
    which has a CPU kernel in 2.8) in float32: equal within tolerance, not bitwise;
(c) trains AdamW with decay/no-decay groups vs decay-all and reports final loss
    and weight norms;
(d) keeps an EMA of the weights with ``AveragedModel`` and reports its
    held-out loss next to the raw model's.

Run from the repository root:

    python content/courses/pytorch/code/15_optimizers_from_scratch.py [--seed 0] [--steps 200] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


# ---------------------------------------------------------------- optimizers
class ScratchSGD(torch.optim.Optimizer):
    """SGD with PyTorch's momentum convention (first step: buffer = gradient)."""

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov needs momentum > 0 and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, damp = group["lr"], group["momentum"], group["dampening"]
            wd, nesterov = group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p                      # L2: decay enters the gradient
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:  # lazy init, no dampening on step 1
                        buf = state["momentum_buffer"] = g.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(mu).add_(g, alpha=1 - damp)
                    g = g + mu * buf if nesterov else buf
                p.add_(g, alpha=-lr)
        return loss


class ScratchAdamW(torch.optim.Optimizer):
    """AdamW: decoupled decay, then the bias-corrected Adam step."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                p.mul_(1 - lr * wd)                       # decoupled: never touches m or v
                m.lerp_(g, 1 - b1)                        # m = b1*m + (1-b1)*g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)   # v = b2*v + (1-b2)*g*g
                step_size = lr / (1 - b1 ** t)            # bias correction of m folded into lr
                denom = v.sqrt() / (1 - b2 ** t) ** 0.5 + eps
                p.addcdiv_(m, denom, value=-step_size)
        return loss


class SignMomentum(torch.optim.Optimizer):
    """Lion-like: step along sign(beta1*m + (1-beta1)*g); m tracks g with beta2."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - lr * wd)
                p.add_(torch.sign(m.mul(b1).add(p.grad, alpha=1 - b1)), alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


# ---------------------------------------------------------------- schedule
def warmup_cosine(step, warmup=10, total=200, floor=0.1):
    """Multiplier of the peak lr: linear warmup reaching 1 at step warmup-1, cosine to floor at total."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- model, data, groups
class MLP(nn.Module):
    def __init__(self, d_in=8, d_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.norm(self.fc1(x))))


def make_data(seed, dtype, n_train=256, n_eval=256, d_in=8):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d_in, 1, generator=g, dtype=dtype) / math.sqrt(d_in)
    x = torch.randn(n_train + n_eval, d_in, generator=g, dtype=dtype)
    y = torch.sin(x @ w) + 0.5 * x[:, :1] * x[:, 1:2] + 0.1 * torch.randn(
        n_train + n_eval, 1, generator=g, dtype=dtype)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def make_model(seed, dtype):
    torch.manual_seed(seed)
    return MLP().to(dtype)


def build_param_groups(model, weight_decay):
    """Decay tensors with ndim >= 2 (matmul weights, embeddings); biases and norm gains get 0.

    named_parameters() already drops duplicates (tied weights appear once); the id
    set makes that explicit and survives callers who pass a hand-built list.
    """
    decay, no_decay, seen = [], [], set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def mse(model, x, y):
    return torch.nn.functional.mse_loss(model(x), y)


def train(model, opt, sched, x, y, steps):
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = mse(model, x, y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def max_param_diff(a, b):
    return max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))


def compare(make_scratch, make_torch, seed, steps, lam, x, y, ema_decay=None):
    """Train scratch and torch copies in lockstep; return the worst parameter gap seen.

    With ema_decay, an AveragedModel EMA follows the torch.optim copy.
    """
    m1, m2 = make_model(seed, torch.float64), make_model(seed, torch.float64)
    o1, o2 = make_scratch(m1), make_torch(m2)
    s1, s2 = LambdaLR(o1, lam), LambdaLR(o2, lam)
    ema = None if ema_decay is None else AveragedModel(
        m2, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    worst, losses = 0.0, []
    for _ in range(steps):
        for m, o, s in ((m1, o1, s1), (m2, o2, s2)):
            o.zero_grad(set_to_none=True)
            loss = mse(m, x, y)
            loss.backward()
            o.step()
            s.step()
        losses.append(loss.item())
        if ema is not None:
            ema.update_parameters(m2)
        worst = max(worst, max_param_diff(m1, m2))
    return {"diff": worst, "losses": losses, "model": m2, "ema": ema}


def norms(model):
    weights = [p for p in model.parameters() if p.ndim >= 2]
    others = [p for p in model.parameters() if p.ndim < 2]
    return {"matrix_weights": torch.sqrt(sum((p ** 2).sum() for p in weights)).item(),
            "biases_and_norm_gains": torch.sqrt(sum((p ** 2).sum() for p in others)).item()}


# ---------------------------------------------------------------- main
def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    steps, warmup = args.steps, max(1, args.steps // 20)

    def lam(step):
        return warmup_cosine(step, warmup=warmup, total=steps, floor=0.1)

    x, y, xe, ye = make_data(args.seed, torch.float64)
    init_loss = mse(make_model(args.seed, torch.float64), x, y).item()

    # (a) scratch vs torch.optim, float64 (the AdamW run also feeds (c) and (d))
    sgd_diff, sgd_loss = 0.0, None
    for nesterov in (False, True):
        cfg = dict(lr=0.05, momentum=0.9, nesterov=nesterov, weight_decay=1e-4)
        run = compare(lambda m: ScratchSGD(m.parameters(), **cfg),
                      lambda m: torch.optim.SGD(m.parameters(), foreach=False, **cfg),
                      args.seed, steps, lam, x, y)
        sgd_diff, sgd_loss = max(sgd_diff, run["diff"]), run["losses"][-1]
    adamw_kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    groups_run = compare(
        lambda m: ScratchAdamW(build_param_groups(m, 0.1), **adamw_kw),
        lambda m: torch.optim.AdamW(build_param_groups(m, 0.1), foreach=False, **adamw_kw),
        args.seed, steps, lam, x, y, ema_decay=0.95)

    # (b) for-loop vs foreach vs fused (CPU), float32
    x32, y32 = x.float(), y.float()
    impl = {}
    for name, flags in (("for_loop", dict(foreach=False)), ("foreach", dict(foreach=True)),
                        ("fused", dict(fused=True))):
        m = make_model(args.seed, torch.float32)
        o = torch.optim.AdamW(build_param_groups(m, 0.1), **adamw_kw, **flags)
        train(m, o, LambdaLR(o, lam), x32, y32, steps)
        impl[name] = m
    foreach_diff = max_param_diff(impl["for_loop"], impl["foreach"])
    fused_diff = max_param_diff(impl["for_loop"], impl["fused"])
    foreach_bitwise = all(torch.equal(a, b) for a, b in zip(impl["for_loop"].parameters(),
                                                            impl["foreach"].parameters()))

    # (c) decay-all baseline for the grouped run above
    m = make_model(args.seed, torch.float64)
    o = torch.optim.AdamW(m.parameters(), weight_decay=0.1, **adamw_kw)
    all_losses = train(m, o, LambdaLR(o, lam), x, y, steps)
    runs = {"groups": groups_run, "decay_all": {"model": m, "losses": all_losses}}

    # (d) EMA vs raw weights on held-out data
    with torch.no_grad():
        ema_eval = mse(groups_run["ema"].module, xe, ye).item()
        raw_eval = mse(groups_run["model"], xe, ye).item()

    probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=adamw_kw["lr"])
    sched = LambdaLR(probe, lam)
    lr_trace = []
    for _ in range(steps):
        lr_trace.append(probe.param_groups[0]["lr"])
        probe.step()
        sched.step()

    result = {
        "sgd_max_param_diff": sgd_diff,
        "adamw_max_param_diff": groups_run["diff"],
        "foreach_max_diff": foreach_diff,
        "foreach_bitwise_equal": foreach_bitwise,
        "fused_cpu_max_diff": fused_diff,
        "initial_loss": init_loss,
        "final_loss": {
            "sgd": sgd_loss,
            "adamw_groups": runs["groups"]["losses"][-1],
            "adamw_decay_all": runs["decay_all"]["losses"][-1],
        },
        "weight_norms": {k: norms(v["model"]) for k, v in runs.items()},
        "raw_eval_loss": raw_eval,
        "ema_eval_loss": ema_eval,
        "warmup": warmup,
        "steps": steps,
        "peak_lr": adamw_kw["lr"],
        "lr_trace": lr_trace,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
