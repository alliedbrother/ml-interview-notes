"""A scalar reverse-mode autograd engine (micrograd-style), checked against PyTorch.

Module 07 lab. It implements ``Value`` (add, mul, pow with a constant exponent,
neg, sub, truediv, exp, log, tanh, relu) with a topologically ordered backward,
then:

(a) builds five seeded random expressions and compares every leaf gradient with
    PyTorch float64 autograd;
(b) trains a 3 -> 4 -> 4 -> 1 tanh MLP (41 parameters) on Karpathy's four-point
    toy set with plain SGD, and trains an identically initialised float64
    ``torch.nn`` MLP side by side, comparing the loss at every step;
(c) reports the size of one training-step graph and the first ops in backward
    order.

Design after karpathy/micrograd (engine.py at 7bc720e9). One deliberate change:
``backward()`` resets non-leaf gradients before propagating, so leaves
accumulate across calls exactly as PyTorch's ``.grad`` does while intermediates
stay transient.

Run from the repository root:

    python content/courses/pytorch/code/07_micrograd.py [--seed 0] [--steps 60] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch


class Value:
    """One scalar node: data, accumulated grad, parents, and a local backward closure."""

    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)  # ordered, so backward order is deterministic
        self._backward = lambda: None
        self._op = _op

    @property
    def is_leaf(self):
        return not self._prev

    # ---- binary ops -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only constant exponents"
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    # ---- unary ops --------------------------------------------------------
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # saves its output, like torch's ExpBackward0

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data  # saves its input

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ---- derived ops ------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) / self

    def backward(self):
        topo = topological_order(self)
        for node in topo:  # non-leaves are transient: fresh every call
            if not node.is_leaf:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self._op!r})"


def topological_order(root):
    """Every node appears after all of its children (inputs)."""
    order, seen = [], set()

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in node._prev:
            visit(child)
        order.append(node)

    visit(root)
    return order


# ---- (a) random expressions, mirrored in torch ------------------------------
UNARY = ("exp", "log", "tanh", "relu", "pow", "neg")
BINARY = ("add", "sub", "mul", "div")


def make_program(rng, n_inputs=3, n_ops=12):
    """A list of (op, args) over registers; register i < n_inputs is an input."""
    program, n_regs = [], n_inputs
    for _ in range(n_ops):
        if rng.random() < 0.5:
            op = rng.choice(UNARY)
            arg = rng.randrange(n_regs)
            exponent = rng.choice((2, 3, -1, 0.5))
            program.append((op, (arg, exponent)))
        else:
            op = rng.choice(BINARY)
            program.append((op, (rng.randrange(n_regs), rng.randrange(n_regs))))
        n_regs += 1
    return program


def run_program(program, inputs):
    """Evaluate on Value or 0-d torch tensors; safe-guards keep every op in-domain."""
    regs = list(inputs)
    for op, (a, b) in program:
        x = regs[a]
        if op == "exp":
            y = (x.tanh()).exp()
        elif op == "log":
            y = (x * x + 1.0).log()
        elif op == "tanh":
            y = x.tanh()
        elif op == "relu":
            y = x.relu()
        elif op == "neg":
            y = -x
        elif op == "pow":
            y = (x * x + 1.0) ** b  # base >= 1, so any constant exponent is safe
        elif op == "add":
            y = x + regs[b]
        elif op == "sub":
            y = x - regs[b]
        elif op == "mul":
            y = x * regs[b]
        else:  # div
            y = x / (regs[b] * regs[b] + 1.0)
        regs.append(y)
    out = regs[-1]
    for r in regs[len(inputs):-1]:  # fold every intermediate in so all paths matter
        out = out + 0.1 * r
    return out


def compare_expressions(seed, n_expr=5):
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(n_expr):
        program = make_program(rng)
        xs = [rng.uniform(-2.0, 2.0) for _ in range(3)]
        vals = [Value(v) for v in xs]
        run_program(program, vals).backward()
        ts = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in xs]
        run_program(program, ts).backward()
        for v, t in zip(vals, ts):
            g = 0.0 if t.grad is None else t.grad.item()
            worst = max(worst, abs(v.grad - g))
    return worst


# ---- (b) a 3-4-4-1 tanh MLP in Values and in torch.nn -----------------------
XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]
SIZES = [3, 4, 4, 1]


def init_weights(seed):
    """Per layer: W as a list of rows (out x in) and b (out), uniform(-1, 1)."""
    rng = random.Random(seed)
    layers = []
    for nin, nout in zip(SIZES[:-1], SIZES[1:]):
        W = [[rng.uniform(-1, 1) for _ in range(nin)] for _ in range(nout)]
        b = [rng.uniform(-1, 1) for _ in range(nout)]
        layers.append((W, b))
    return layers


class ValueMLP:
    def __init__(self, layers):
        self.layers = [([[Value(w) for w in row] for row in W], [Value(v) for v in b]) for W, b in layers]

    def parameters(self):
        return [p for W, b in self.layers for row, bias in zip(W, b) for p in (*row, bias)]

    def __call__(self, x):
        for W, b in self.layers:
            x = [sum((wi * xi for wi, xi in zip(row, x)), bias).tanh() for row, bias in zip(W, b)]
        return x[0]


def value_loss(model):
    preds = [model([Value(v) for v in x]) for x in XS]
    return sum(((p - y) ** 2 for p, y in zip(preds, YS)), Value(0.0))


def torch_mlp(layers):
    mods = []
    for W, b in layers:
        lin = torch.nn.Linear(len(W[0]), len(W), dtype=torch.float64)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W, dtype=torch.float64))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float64))
        mods += [lin, torch.nn.Tanh()]
    return torch.nn.Sequential(*mods)


def train_both(seed, steps, lr=0.05):
    layers = init_weights(seed)
    vm, tm = ValueMLP(layers), torch_mlp(layers)
    X = torch.tensor(XS, dtype=torch.float64)
    Y = torch.tensor(YS, dtype=torch.float64)
    curve_v, curve_t = [], []
    graph_nodes, order_head = None, None
    for _ in range(steps):
        loss = value_loss(vm)
        for p in vm.parameters():
            p.grad = 0.0
        loss.backward()
        if graph_nodes is None:
            topo = topological_order(loss)
            graph_nodes = len(topo)
            order_head = [n._op or "leaf" for n in reversed(topo)][:8]
        for p in vm.parameters():
            p.data -= lr * p.grad
        curve_v.append(loss.data)

        tm.zero_grad(set_to_none=True)
        tloss = ((tm(X).squeeze(1) - Y) ** 2).sum()
        tloss.backward()
        with torch.no_grad():  # the optimizer step must not be recorded
            for p in tm.parameters():
                p -= lr * p.grad
        curve_t.append(tloss.item())
    n_params = len(vm.parameters())
    return curve_v, curve_t, graph_nodes, order_head, n_params


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)

    max_grad = compare_expressions(args.seed)
    curve_v, curve_t, nodes, order_head, n_params = train_both(args.seed, args.steps)
    result = {
        "max_grad_abs_diff": max_grad,
        "loss_curve_micrograd": curve_v,
        "loss_curve_torch": curve_t,
        "max_loss_diff": max(abs(a - b) for a, b in zip(curve_v, curve_t)),
        "graph_nodes": nodes,
        "backward_order_head": order_head,
        "n_parameters": n_params,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
