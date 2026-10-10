"""Hunt five seeded training bugs with cheap automated detectors.

A small MLP classifier is trained on one fixed batch. A ``--bug`` switch plants
one of five classic PyTorch bugs in the recipe (model, loss or training step):

    no_zero_grad       the step never calls ``optimizer.zero_grad()``
    softmax_before_ce  ``cross_entropy(logits.softmax(-1), y)``
    eval_dropout       ``F.dropout(h, p)`` without ``training=self.training``
    broadcast_loss     per-example weights shaped (B, 1) against a (B,) loss
    detached_layer     ``self.fc1(x).detach()`` cuts fc1 out of the graph

A detector suite then runs checks that need no knowledge of which bug is
planted: loss at init vs ln C, the scale of d(loss)/d(logits), a shape contract
on the per-example loss, eval-mode determinism, parameters left with
``grad is None``, a two-step gradient-accumulation probe, and an overfit-one-batch
run. The result is a bug x detector matrix of booleans.

Run from the repository root:

    python content/courses/pytorch/code/18_bug_hunt.py --bug all
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

BUGS = ("no_zero_grad", "softmax_before_ce", "eval_dropout", "broadcast_loss", "detached_layer")
DETECTORS = ("init_loss", "logit_grad_scale", "shape_contract", "eval_determinism",
             "unused_params", "grad_accumulation", "overfit_one_batch")

B, D_IN, HIDDEN, C = 32, 16, 64, 8
OVERFIT_STEPS, LR = 60, 1e-2
OVERFIT_TOL = 0.05


class MLP(nn.Module):
    """(B, D_IN) -> (B, C) classifier: Linear-ReLU-Dropout-Linear-ReLU-Linear."""

    def __init__(self, bug: str = "none", p: float = 0.1):
        super().__init__()
        self.bug, self.p = bug, p
        self.fc1 = nn.Linear(D_IN, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, C)
        with torch.no_grad():  # small last layer: near-uniform logits at init
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if self.bug == "detached_layer":
            h = h.detach()  # BUG: fc1 never receives a gradient
        h = F.relu(h)
        if self.bug == "eval_dropout":
            h = F.dropout(h, self.p)  # BUG: training defaults to True
        else:
            h = F.dropout(h, self.p, training=self.training)
        h = F.relu(self.fc2(h))
        return self.out(h)


class Recipe:
    """Everything a training script decides: model, loss, and one step."""

    def __init__(self, bug: str = "none"):
        if bug != "none" and bug not in BUGS:
            raise ValueError(f"unknown bug {bug!r}")
        self.bug = bug

    def build_model(self, seed: int) -> MLP:
        torch.manual_seed(seed)
        return MLP(self.bug)

    def per_example_loss(self, logits: torch.Tensor, y: torch.Tensor,
                         w: torch.Tensor) -> torch.Tensor:
        """Weighted per-example cross-entropy; contract: shape (B,)."""
        inp = logits.softmax(-1) if self.bug == "softmax_before_ce" else logits
        ce = F.cross_entropy(inp, y, reduction="none")  # (B,)
        if self.bug == "broadcast_loss":
            w = w.unsqueeze(1)  # BUG: (B,) * (B, 1) -> (B, B), no error
        return ce * w

    def loss(self, logits, y, w):
        return self.per_example_loss(logits, y, w).mean()

    def train_step(self, model, opt, x, y, w) -> float:
        if self.bug != "no_zero_grad":
            opt.zero_grad(set_to_none=True)
        loss = self.loss(model(x), y, w)
        loss.backward()
        opt.step()
        return loss.item()


def make_batch(seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, D_IN, generator=g)
    y = torch.randint(0, C, (B,), generator=g)
    w = 0.5 + torch.rand(B, generator=g)
    w = w / w.mean()  # weights average exactly 1, so expected init loss stays ln C
    return x, y, w


# ---------------------------------------------------------------- detectors

def find_unused_params(model: nn.Module) -> list[str]:
    """Names of trainable parameters whose .grad is still None after backward."""
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


def detect_init_loss(recipe, seed, x, y, w) -> dict:
    model = recipe.build_model(seed).train()
    with torch.no_grad():
        loss = recipe.loss(model(x), y, w).item()
    expected = math.log(C)
    return {"fired": abs(loss - expected) > 0.25 * expected, "loss": loss, "expected": expected}


def detect_logit_grad_scale(recipe, seed, x, y, w) -> dict:
    """d(mean CE)/d(logits) is (softmax - onehot)/B; check its norm within 4x."""
    model = recipe.build_model(seed).eval()
    with torch.no_grad():
        logits = model(x)
    z = logits.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(recipe.loss(z, y, w), z)
    expected = (z.detach().softmax(-1) - F.one_hot(y, C)).norm() / B
    ratio = (g.norm() / expected).item()
    return {"fired": not (0.25 < ratio < 4.0), "ratio": ratio}


def detect_shape_contract(recipe, seed, x, y, w) -> dict:
    model = recipe.build_model(seed)
    logits = model(x)
    per_ex = recipe.per_example_loss(logits, y, w)
    ok = tuple(logits.shape) == (B, C) and tuple(per_ex.shape) == (B,)
    return {"fired": not ok, "per_example_shape": list(per_ex.shape)}


def detect_eval_determinism(recipe, seed, x, y, w) -> dict:
    model = recipe.build_model(seed).eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    return {"fired": not torch.equal(a, b)}


def detect_unused_params(recipe, seed, x, y, w) -> dict:
    model = recipe.build_model(seed).train()
    recipe.loss(model(x), y, w).backward()
    unused = find_unused_params(model)
    return {"fired": bool(unused), "unused": unused}


def detect_grad_accumulation(recipe, seed, x, y, w) -> dict:
    """Two train steps at lr=0 on one batch: grads must not double."""
    model = recipe.build_model(seed).eval()  # eval: same forward both times
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    torch.manual_seed(seed + 1)  # same dropout mask in both steps if a bug keeps dropout on
    recipe.train_step(model, opt, x, y, w)
    first = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    torch.manual_seed(seed + 1)
    recipe.train_step(model, opt, x, y, w)
    second = [p.grad for p in model.parameters() if p.grad is not None]
    ratio = (sum(s.norm() for s in second) / sum(f.norm() for f in first)).item()
    return {"fired": abs(ratio - 1.0) > 1e-3, "ratio": ratio}


def detect_overfit_one_batch(recipe, seed, x, y, w) -> dict:
    model = recipe.build_model(seed).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve = []
    for _ in range(OVERFIT_STEPS):
        before = [p.detach().clone() for p in model.parameters()]
        curve.append(recipe.train_step(model, opt, x, y, w))
    # update-to-data ratio |delta p| / |p| per parameter tensor at the last step
    ratios = sorted(((p.detach() - q).norm() / (q.norm() + 1e-12)).item()
                    for p, q in zip(model.parameters(), before))
    model.eval()
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()  # true unweighted CE, eval mode
    return {"fired": not (final < OVERFIT_TOL), "final_loss": final,
            "first_loss": curve[0], "log10_update_ratio_median": math.log10(max(ratios[len(ratios) // 2], 1e-12))}


DETECTOR_FNS = {
    "init_loss": detect_init_loss,
    "logit_grad_scale": detect_logit_grad_scale,
    "shape_contract": detect_shape_contract,
    "eval_determinism": detect_eval_determinism,
    "unused_params": detect_unused_params,
    "grad_accumulation": detect_grad_accumulation,
    "overfit_one_batch": detect_overfit_one_batch,
}


def run_detectors(bug: str, seed: int) -> dict:
    recipe = Recipe(bug)
    x, y, w = make_batch(seed)
    return {name: fn(recipe, seed, x, y, w) for name, fn in DETECTOR_FNS.items()}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bug", default="all", choices=("none",) + BUGS + ("all",))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    bugs = ("none",) + BUGS if args.bug == "all" else (args.bug,)
    details = {bug: run_detectors(bug, args.seed) for bug in bugs}
    result = {
        "matrix": {bug: {d: bool(r[d]["fired"]) for d in DETECTORS} for bug, r in details.items()},
        "clean_overfit_loss": None,
        "details": details,
    }
    if "none" in details:
        result["clean_overfit_loss"] = details["none"]["overfit_one_batch"]["final_loss"]
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
