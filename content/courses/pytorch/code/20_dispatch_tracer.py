"""Trace every aten op of one training step with a dispatch mode and a wrapper subclass.

Part (a): ``OpCounter`` is a ``TorchDispatchMode`` that counts aten overloads
separately for the forward pass, the backward pass and the optimizer step of a
2-layer MLP (16 -> 32 -> 4, ReLU, cross-entropy) trained with SGD. Because the
mode sits below the Autograd dispatch key it sees the backward kernels too.

Part (b): ``LoggingTensor`` is a wrapper tensor subclass built with
``torch.Tensor._make_wrapper_subclass``. Its ``__torch_dispatch__`` unwraps the
inner tensors, runs the aten op, records its name and rewraps the outputs.
Autograd runs above it, so gradients still flow and come back as
``LoggingTensor`` objects.

Part (c): both runs are compared with a plain eager run of the same step.

Op names and counts are those of PyTorch 2.8 CPU kernels; other versions or
backends may decompose differently.

Run from the repository root:

    python content/courses/pytorch/code/20_dispatch_tracer.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

MM_LIKE = ("aten.mm.default", "aten.addmm.default", "aten.bmm.default")


class OpCounter(TorchDispatchMode):
    """Counts aten overloads per phase; set ``self.phase`` before each phase."""

    def __init__(self):
        super().__init__()
        self.phase = "forward"
        self.counts = {"forward": Counter(), "backward": Counter(), "optimizer": Counter()}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[self.phase][str(func)] += 1
        return func(*args, **(kwargs or {}))


class LoggingTensor(torch.Tensor):
    """Wrapper subclass: holds a plain ``elem`` and logs every aten op it reaches."""

    elem: torch.Tensor
    __slots__ = ["elem", "log"]

    @staticmethod
    def __new__(cls, elem, log, requires_grad=False):
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, device=elem.device, requires_grad=requires_grad,
        )
        r.elem = elem
        r.log = log
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem!r})"

    # Needed by torch.compile / export to see through the wrapper (not used here).
    def __tensor_flatten__(self):
        return ["elem"], self.log

    @staticmethod
    def __tensor_unflatten__(inner_tensors, log, outer_size, outer_stride):
        return LoggingTensor(inner_tensors["elem"], log)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        log = None

        def unwrap(t):
            nonlocal log
            if isinstance(t, LoggingTensor):
                log = t.log
                return t.elem
            return t

        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        log.append(str(func))
        return tree_map(lambda t: LoggingTensor(t, log) if isinstance(t, torch.Tensor) else t, out)


def make_batch(seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 16, generator=g)
    y = torch.randint(0, 4, (8,), generator=g)
    return x, y


def make_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def plain_step(seed: int):
    model = make_model(seed)
    x, y = make_batch(seed + 1)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    return logits.detach(), grads, [p.detach().clone() for p in model.parameters()]


def traced_step(seed: int):
    model = make_model(seed)
    x, y = make_batch(seed + 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    counter = OpCounter()
    with counter:
        counter.phase = "forward"
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        counter.phase = "backward"
        loss.backward()
        counter.phase = "optimizer"
        opt.step()
    params = [p.detach().clone() for p in model.parameters()]
    return logits.detach(), counter.counts, params


def subclass_step(seed: int):
    """Functional forward on LoggingTensor parameters; returns logits, grads and the op log."""
    model = make_model(seed)
    x, y = make_batch(seed + 1)
    log: list[str] = []
    params = [LoggingTensor(p.detach().clone(), log, requires_grad=True) for p in model.parameters()]
    w1, b1, w2, b2 = params
    h = F.relu(F.linear(LoggingTensor(x, log), w1, b1))
    logits = F.linear(h, w2, b2)
    loss = F.cross_entropy(logits, LoggingTensor(y, log))
    loss.backward()
    grads = [p.grad.elem for p in params]
    return logits.detach().elem, grads, log, type(params[0].grad).__name__


def mm_like(counts: Counter) -> int:
    return sum(counts[name] for name in MM_LIKE)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    ref_logits, ref_grads, ref_params = plain_step(args.seed)
    tr_logits, counts, tr_params = traced_step(args.seed)
    sub_logits, sub_grads, sub_log, grad_type = subclass_step(args.seed)

    def close(a, b):
        return all(torch.allclose(u, v, rtol=1e-5, atol=1e-6) for u, v in zip(a, b))

    result = {
        "forward_ops": dict(sorted(counts["forward"].items())),
        "backward_ops": dict(sorted(counts["backward"].items())),
        "optimizer_ops": dict(sorted(counts["optimizer"].items())),
        "total_ops": sum(sum(c.values()) for c in counts.values()),
        "mm_like_forward": mm_like(counts["forward"]),
        "mm_like_backward": mm_like(counts["backward"]),
        "traced_step_matches": close([tr_logits] + tr_params, [ref_logits] + ref_params),
        "subclass_output_matches": close([sub_logits], [ref_logits]),
        "subclass_grad_matches": close(sub_grads, ref_grads),
        "subclass_grad_type": grad_type,
        "subclass_logged_ops": len(sub_log),
        "subclass_distinct_ops": len(set(sub_log)),
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
