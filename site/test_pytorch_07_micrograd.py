"""Contracts for the module 07 scalar autograd lab (micrograd vs PyTorch)."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


mg = load("07_micrograd")
_STATE = {}
_RESULT = {}


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        result = mg.main(["--seed", "0", "--steps", "60", "--json-out", str(path)])
        _RESULT.update(result)
        _RESULT["_json_roundtrip"] = json.loads(path.read_text()) == json.loads(json.dumps(result))


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class MicrogradMatchesTorch(unittest.TestCase):
    def test_result_contract(self):
        for key in ("max_grad_abs_diff", "loss_curve_micrograd", "loss_curve_torch",
                    "max_loss_diff", "graph_nodes"):
            self.assertIn(key, _RESULT)
        self.assertTrue(_RESULT["_json_roundtrip"])
        self.assertEqual(_RESULT["n_parameters"], 41)
        self.assertGreater(_RESULT["graph_nodes"], 41)

    def test_expression_gradients_match_float64_torch(self):
        self.assertLess(_RESULT["max_grad_abs_diff"], 1e-9)

    def test_loss_curves_match_every_step(self):
        cv, ct = _RESULT["loss_curve_micrograd"], _RESULT["loss_curve_torch"]
        self.assertEqual(len(cv), 60)
        self.assertEqual(len(ct), 60)
        for step, (a, b) in enumerate(zip(cv, ct)):
            self.assertLess(abs(a - b), 1e-9, f"step {step}")

    def test_training_reduces_loss(self):
        cv = _RESULT["loss_curve_micrograd"]
        self.assertLess(cv[-1], 0.3 * cv[0])


class EngineSemantics(unittest.TestCase):
    def test_diamond_graph_accumulates(self):
        a, b = mg.Value(2.0), mg.Value(-3.0)
        y = a * b + a
        y.backward()
        self.assertAlmostEqual(a.grad, b.data + 1.0, places=12)
        self.assertAlmostEqual(b.grad, a.data, places=12)

    def test_self_product_counts_both_paths(self):
        a = mg.Value(1.5)
        (a * a).backward()
        self.assertAlmostEqual(a.grad, 3.0, places=12)

    def test_topological_order_children_first(self):
        a, b, c = mg.Value(0.5), mg.Value(-1.2), mg.Value(2.0)
        y = ((a * b).tanh() + c.exp() * a - b / c).relu() + (a ** 2).log()
        topo = mg.topological_order(y)
        position = {id(n): i for i, n in enumerate(topo)}
        self.assertEqual(len(position), len(topo))
        self.assertIs(topo[-1], y)
        for node in topo:
            for child in node._prev:
                self.assertLess(position[id(child)], position[id(node)])

    def test_second_backward_doubles_leaf_grads(self):
        a, b = mg.Value(0.5), mg.Value(-0.7)
        y = (a * b + a).tanh() * b
        y.backward()
        first = (a.grad, b.grad)
        y.backward()
        self.assertAlmostEqual(a.grad, 2 * first[0], places=12)
        self.assertAlmostEqual(b.grad, 2 * first[1], places=12)

    def test_backward_twice_parallels_torch(self):
        a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
        y = (a * b + a).tanh() * b
        y.backward(retain_graph=True)
        y.backward()
        va, vb = mg.Value(0.5), mg.Value(-0.7)
        vy = (va * vb + va).tanh() * vb
        vy.backward()
        vy.backward()
        self.assertAlmostEqual(va.grad, a.grad.item(), places=12)
        self.assertAlmostEqual(vb.grad, b.grad.item(), places=12)


if __name__ == "__main__":
    unittest.main()
