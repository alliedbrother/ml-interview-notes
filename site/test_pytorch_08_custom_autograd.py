"""Contracts for the module 08 lab: a memory-saving fused Linear+GELU Function."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
_STATE = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("08_custom_autograd")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class FusedLinearGELUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            cls.result = lab.main(["--seed", "0", "--json-out", str(path)])
            cls.from_disk = json.loads(path.read_text())

    def test_result_is_json_round_trippable(self):
        self.assertEqual(self.result, self.from_disk)

    def test_gradcheck_and_gradgradcheck_pass(self):
        self.assertIs(self.result["gradcheck"], True)

    def test_outputs_and_gradients_match_reference(self):
        self.assertLess(self.result["max_out_diff"], 1e-5)
        self.assertLess(self.result["max_grad_diff"], 1e-4)

    def test_fused_saves_exactly_inputs_and_parameters(self):
        b, d_in, d_out = 64, 128, 512
        expected_fused = 4 * (b * d_in + d_out * d_in + d_out)
        expected_reference = 4 * (b * d_in + d_out * d_in + b * d_out)
        self.assertEqual(self.result["saved_bytes_fused"], expected_fused)
        self.assertEqual(self.result["saved_bytes_reference"], expected_reference)
        self.assertLess(self.result["saved_bytes_fused"], self.result["saved_bytes_reference"])
        self.assertLessEqual(self.result["saved_ratio"], 0.75)

    def test_training_curves_agree_and_learn(self):
        self.assertLess(self.result["loss_curve_max_diff"], 1e-4)
        self.assertLess(self.result["loss_last"], 0.5 * self.result["loss_first"])

    def test_once_differentiable_variant_refuses_double_backward(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(4, 3, dtype=torch.float64, generator=g, requires_grad=True)
        w = torch.randn(5, 3, dtype=torch.float64, generator=g, requires_grad=True)
        b = torch.randn(5, dtype=torch.float64, generator=g, requires_grad=True)
        y = lab.LinearGELUOnce.apply(x, w, b)
        v = torch.ones_like(y, requires_grad=True)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=v, create_graph=True)
        with self.assertRaisesRegex(RuntimeError, "once_differentiable"):
            gx.sum().backward()
        # The differentiable version supports the same second-order call.
        y2 = lab.LinearGELUFunction.apply(x, w, b)
        (gx2,) = torch.autograd.grad(y2, x, grad_outputs=v, create_graph=True)
        gx2.pow(2).sum().backward()
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_saved_bytes_scale_with_batch_only_for_reference(self):
        small = lab.main(["--batch", "8", "--d-in", "16", "--d-out", "32", "--steps", "2"])
        self.assertEqual(small["saved_bytes_fused"], lab.expected_fused_bytes(8, 16, 32))
        self.assertEqual(small["saved_bytes_reference"] - small["saved_bytes_fused"],
                         4 * (8 * 32 - 32))


if __name__ == "__main__":
    unittest.main()
