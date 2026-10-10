"""Per-sample gradients three ways, then one DP-SGD step (module 09 lab).

A 2-layer MLP classifier (20 -> hidden -> 5) on seeded synthetic data. Per-sample
gradients are computed (a) with a Python loop of B backward passes, (b) with
vmap(grad(...)) over functional_call, and (c) for the first Linear layer
analytically from hooked activations and output gradients (grad_W_i = delta_i a_i^T).
The three must agree, and their mean must equal the ordinary batch gradient.
Then one DP-SGD step: clip each example's gradient to norm C, sum, add seeded
Gaussian noise with std sigma * C, divide by B. No privacy accounting is done.

Run from the repository root:
    python content/courses/pytorch/code/09_per_sample_grads.py [--seed 0] [--batch 32] [--hidden 64] [--json-out PATH]
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.utils import benchmark

IN_FEATURES = 20
NUM_CLASSES = 5


def make_problem(seed, batch, hidden):
    """Seeded model and data; everything float32 on CPU."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, hidden), nn.ReLU(), nn.Linear(hidden, NUM_CLASSES)
    )
    x = torch.randn(batch, IN_FEATURES, generator=gen)
    y = torch.randint(0, NUM_CLASSES, (batch,), generator=gen)
    return model, x, y


def per_sample_loop(model, x, y):
    """(a) B separate backward passes with batch size 1. Returns {name: (B, *shape)}."""
    rows = {name: [] for name, _ in model.named_parameters()}
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[i : i + 1]), y[i : i + 1])
        loss.backward()
        for name, p in model.named_parameters():
            rows[name].append(p.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return {name: torch.stack(g) for name, g in rows.items()}


def per_sample_vmap(model, x, y):
    """(b) vmap(grad(loss of ONE example)) over the batch dimension of x and y."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(params, xi, yi):
        logits = functional_call(model, params, (xi.unsqueeze(0),))
        return F.cross_entropy(logits, yi.unsqueeze(0))

    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)


def per_sample_first_layer_analytic(model, x, y):
    """(c) Outer-product trick for model[0]: grad_W_i = delta_i a_i^T, grad_b_i = delta_i.

    A forward hook on the first Linear captures its input a (B, 20) and puts a
    tensor hook on its output z (B, hidden) to capture delta = d loss / d z.
    One batched forward/backward with reduction="sum", so row i of delta is
    exactly d loss_i / d z_i (no 1/B factor).
    """
    first = model[0]
    captured = {}

    def save_delta(g):
        captured["delta"] = g.detach()

    def fwd_hook(_mod, inputs, out):
        captured["a"] = inputs[0].detach()
        out.register_hook(save_delta)

    handle = first.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y, reduction="sum").backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    a, delta = captured["a"], captured["delta"]  # (B, 20), (B, hidden)
    return {
        "0.weight": torch.einsum("bh,bi->bhi", delta, a),  # (B, hidden, 20)
        "0.bias": delta,
    }


def batch_mean_grad(model, x, y):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    out = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return out


def flatten_per_sample(grads):
    """{name: (B, ...)} -> (B, P) in named_parameters order."""
    return torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)


def dp_sgd_step(per_sample_flat, clip_norm=1.0, noise_multiplier=1.0, seed=0, return_tensors=False):
    """Clip each row to L2 norm <= clip_norm, sum, add N(0, (sigma*C)^2) noise, divide by B."""
    batch = per_sample_flat.shape[0]
    norms = per_sample_flat.norm(dim=1)  # (B,)
    scale = (clip_norm / (norms + 1e-6)).clamp(max=1.0)
    clipped = per_sample_flat * scale.unsqueeze(1)  # (B, P)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.normal(
        0.0, noise_multiplier * clip_norm, size=(per_sample_flat.shape[1],), generator=gen
    )
    update = (clipped.sum(0) + noise) / batch  # (P,)
    result = {
        "clipped_fraction": float((norms > clip_norm).float().mean()),
        "dp_update_norm": float(update.norm()),
    }
    if return_tensors:
        result.update({"norms": norms, "clipped": clipped, "noise": noise, "update": update})
    return result


def max_abs_diff(a, b, keys=None):
    keys = keys if keys is not None else a.keys()
    return max(float((a[k] - b[k]).abs().max()) for k in keys)


def run(seed=0, batch=32, hidden=64, time_it=True, return_tensors=False):
    model, x, y = make_problem(seed, batch, hidden)
    loop = per_sample_loop(model, x, y)
    vec = per_sample_vmap(model, x, y)
    analytic = per_sample_first_layer_analytic(model, x, y)
    batch_grad = batch_mean_grad(model, x, y)
    mean_of_per_sample = {k: v.mean(0) for k, v in vec.items()}

    result = {
        "max_diff_loop_vs_vmap": max_abs_diff(loop, vec),
        "max_diff_analytic_first_layer": max_abs_diff(analytic, vec, ["0.weight", "0.bias"]),
        "mean_vs_batch_grad_diff": max_abs_diff(mean_of_per_sample, batch_grad),
    }
    flat = flatten_per_sample(vec)
    dp = dp_sgd_step(flat, clip_norm=1.0, noise_multiplier=1.0, seed=seed, return_tensors=return_tensors)
    result.update({k: dp[k] for k in ("clipped_fraction", "dp_update_norm")})

    if time_it:
        t_loop = benchmark.Timer(
            stmt="f(model, x, y)", globals={"f": per_sample_loop, "model": model, "x": x, "y": y}
        ).timeit(5)
        t_vmap = benchmark.Timer(
            stmt="f(model, x, y)", globals={"f": per_sample_vmap, "model": model, "x": x, "y": y}
        ).timeit(5)
        result["time_loop_ms"] = t_loop.mean * 1e3
        result["time_vmap_ms"] = t_vmap.mean * 1e3
    else:
        result["time_loop_ms"] = float("nan")
        result["time_vmap_ms"] = float("nan")

    if return_tensors:
        result["tensors"] = {
            "per_sample_flat": flat,
            "norms": dp["norms"],
            "clipped": dp["clipped"],
            "noise": dp["noise"],
            "update": dp["update"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = run(args.seed, args.batch, args.hidden)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
