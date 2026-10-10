"""Write an FX optimization pass and a custom torch.compile backend.

``fusion_backend(gm, example_inputs)`` is a Dynamo backend. It receives the
torch-level FX graph that Dynamo captured, records its op histogram, annotates
shapes with ``ShapeProp``, then applies two graph rewrites before returning
``gm.forward``:

1. ``y * torch.sigmoid(y)`` -> ``F.silu(y)``.
2. Eval-mode ``conv2d -> batch_norm(training=False)`` -> one ``conv2d`` whose
   weight and bias are computed *inside the graph* from the conv and BN
   placeholders (W' = W * gamma / sigma, b' = (b - mu) * gamma / sigma + beta,
   sigma = sqrt(running_var + eps)). Nothing is baked into constants, so the
   compiled graph stays correct if the parameters change.

In PyTorch 2.8 Dynamo inlines built-in nn.Modules, so the graph contains
``torch.conv2d`` and ``F.batch_norm`` calls with parameters lifted to
placeholders, not ``call_module`` nodes. The passes match that form.

A second, train-mode run wraps recording compilers with ``aot_autograd`` to
count forward/backward aten nodes and the tensors the partitioner saves for
backward.

Graph-level rewrites only, validated numerically on CPU: no kernel generation
and no speed claims. Real fusion is Inductor's job.

Run from the repository root:

    python content/courses/pytorch/code/24_fx_backend.py
"""

from __future__ import annotations

import argparse
import json
import operator
import os
from collections import Counter

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.shape_prop import ShapeProp
from functorch.compile import make_boxed_func


class SmallCNN(nn.Module):
    """Conv-BN-SiLU x2 -> global average pool -> Linear; SiLU written as x * sigmoid(x)."""

    def __init__(self, channels: int = 8, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.head = nn.Linear(channels, classes)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = y * torch.sigmoid(y)
        y = self.bn2(self.conv2(y))
        y = y * torch.sigmoid(y)
        y = torch.flatten(F.adaptive_avg_pool2d(y, 1), 1)
        return self.head(y)


def target_name(target) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None) or ""
    return f"{module}.{getattr(target, '__name__', str(target))}".lstrip(".")


def op_histogram(gm: fx.GraphModule) -> dict:
    counts = Counter(target_name(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module"))
    return dict(sorted(counts.items()))


def rewrite_silu(gm: fx.GraphModule) -> int:
    """Replace mul(y, sigmoid(y)) or mul(sigmoid(y), y) with F.silu(y)."""
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in (operator.mul, torch.mul) or len(node.args) != 2:
            continue
        a, b = node.args
        for y, sig in ((a, b), (b, a)):
            if (isinstance(sig, fx.Node) and sig.op == "call_function" and sig.target is torch.sigmoid
                    and sig.args[0] is y):
                with gm.graph.inserting_after(node):
                    silu = gm.graph.call_function(F.silu, (y,))
                node.replace_all_uses_with(silu)
                gm.graph.erase_node(node)
                if not sig.users:
                    gm.graph.erase_node(sig)
                count += 1
                break
    return count


def _fold_weights(w, b, mean, var, gamma, beta, eps: float):
    """Conv-BN folding algebra, traced into the graph as ordinary ops."""
    scale = gamma * torch.rsqrt(var + eps)
    w_fold = w * scale.reshape(-1, 1, 1, 1)
    b0 = b if b is not None else torch.zeros_like(mean)
    b_fold = (b0 - mean) * scale + beta
    return w_fold, b_fold


def rewrite_conv_bn(gm: fx.GraphModule) -> int:
    """Fold eval-mode conv2d -> batch_norm pairs into one conv2d with in-graph folded weights."""
    count = 0
    for bn in list(gm.graph.nodes):
        if bn.op != "call_function" or bn.target is not F.batch_norm:
            continue
        args = list(bn.args) + [None] * (8 - len(bn.args))
        x, mean, var, gamma, beta = args[:5]
        training = bn.kwargs.get("training", args[5] if args[5] is not None else False)
        eps = bn.kwargs.get("eps", args[7] if args[7] is not None else 1e-5)
        conv = x
        if training or not isinstance(conv, fx.Node) or conv.target not in (torch.conv2d, F.conv2d):
            continue
        if len(conv.users) != 1 or gamma is None or beta is None:
            continue
        meta = conv.meta.get("tensor_meta")
        if meta is not None and len(meta.shape) != 4:
            continue
        inp, w = conv.args[0], conv.args[1]
        b = conv.args[2] if len(conv.args) > 2 else conv.kwargs.get("bias")
        rest = tuple(conv.args[3:])
        with gm.graph.inserting_before(bn):
            folded = gm.graph.call_function(_fold_weights, (w, b, mean, var, gamma, beta, eps))
            w_fold = gm.graph.call_function(operator.getitem, (folded, 0))
            b_fold = gm.graph.call_function(operator.getitem, (folded, 1))
            new_conv = gm.graph.call_function(torch.conv2d, (inp, w_fold, b_fold) + rest, dict(conv.kwargs))
        bn.replace_all_uses_with(new_conv)
        gm.graph.erase_node(bn)
        gm.graph.erase_node(conv)
        count += 1
    return count


class FusionBackend:
    """A Dynamo backend: (GraphModule, example_inputs) -> callable. Records what it saw and did."""

    def __init__(self):
        self.graphs = 0
        self.histograms = []
        self.rewrites = {"silu": 0, "conv_bn": 0}

    def __call__(self, gm: fx.GraphModule, example_inputs):
        self.graphs += 1
        self.histograms.append(op_histogram(gm))
        ShapeProp(gm).propagate(*example_inputs)       # node.meta["tensor_meta"] = shape, dtype, stride
        self.rewrites["silu"] += rewrite_silu(gm)
        self.rewrites["conv_bn"] += rewrite_conv_bn(gm)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        self.final_code = gm.code
        return gm.forward


def aot_graph_stats(model: nn.Module, x: torch.Tensor) -> dict:
    """Train-mode run through aot_autograd with recording forward/backward compilers."""
    seen = {}

    def recorder(tag):
        def compiler(gm, example_inputs):
            seen[tag] = gm
            return make_boxed_func(gm.forward)
        return compiler

    backend = aot_autograd(fw_compiler=recorder("fw"), bw_compiler=recorder("bw"))
    compiled = torch.compile(model, backend=backend)
    eager_model = _clone(model)
    out = compiled(x)
    out.square().mean().backward()
    ref = eager_model(x)
    ref.square().mean().backward()
    grads_match = all(
        torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4)
        for p, q in zip(model.parameters(), eager_model.parameters())
    )

    def n_calls(gm):
        return sum(n.op == "call_function" for n in gm.graph.nodes)

    fw, bw = seen["fw"], seen["bw"]
    saved = [n.name for n in bw.graph.nodes if n.op == "placeholder" and not n.name.startswith("tangents")]
    inplace_fw = [str(n.target) for n in fw.graph.nodes
                  if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload)
                  and n.target._schema.name.split("::")[-1].endswith("_")]
    return {
        "fw_nodes": n_calls(fw),
        "bw_nodes": n_calls(bw),
        "saved_for_backward": len(saved),
        "saved_names": saved,
        "fw_inplace_ops": inplace_fw,
        "fw_all_aten": all(isinstance(n.target, torch._ops.OpOverload) or n.target is operator.getitem
                           for n in fw.graph.nodes if n.op == "call_function"),
        "aot_out_matches": bool(torch.allclose(out, ref, atol=1e-5, rtol=1e-4)),
        "aot_grads_match": bool(grads_match),
    }


def _clone(model: nn.Module) -> nn.Module:
    clone = type(model)()
    clone.load_state_dict(model.state_dict())
    clone.train(model.training)
    return clone


def build_model(seed: int) -> SmallCNN:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = SmallCNN()
    with torch.no_grad():                         # non-trivial running stats so folding is tested
        for bn in (model.bn1, model.bn2):
            bn.running_mean.copy_(torch.randn(bn.num_features, generator=gen) * 0.5)
            bn.running_var.copy_(torch.rand(bn.num_features, generator=gen) + 0.5)
            bn.weight.copy_(torch.rand(bn.num_features, generator=gen) + 0.5)
            bn.bias.copy_(torch.randn(bn.num_features, generator=gen) * 0.1)
    return model


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    gen = torch.Generator().manual_seed(args.seed + 1)
    x = torch.randn(4, 3, 8, 8, generator=gen)

    torch._dynamo.reset()
    try:
        model = build_model(args.seed).eval()
        backend = FusionBackend()
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            got = compiled(x)
            want = model(x)
        max_abs_diff = (got - want).abs().max().item()

        torch._dynamo.reset()
        train_model = build_model(args.seed).train()
        stats = aot_graph_stats(train_model, x)
    finally:
        torch._dynamo.reset()

    result = {
        "dynamo_graph_ops": backend.histograms[0] if backend.histograms else {},
        "rewrites": dict(backend.rewrites),
        "max_abs_diff": max_abs_diff,
        "graphs": backend.graphs,
        "rewritten_has_batch_norm": "batch_norm" in backend.final_code,
        "rewritten_has_sigmoid": "sigmoid" in backend.final_code,
        **stats,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
