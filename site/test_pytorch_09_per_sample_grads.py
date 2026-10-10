"""Contracts for the module 09 lab: per-sample gradients three ways and one DP-SGD step."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"

_saved = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("09_per_sample_grads")


def setUpModule():
    _saved["threads"] = torch.get_num_threads()
    _saved["dtype"] = torch.get_default_dtype()
    _saved["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_saved["threads"])
    torch.set_default_dtype(_saved["dtype"])
    torch.use_deterministic_algorithms(_saved["det"])


class PerSampleGradContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = lab.run(seed=0, batch=16, hidden=32, time_it=False, return_tensors=True)

    def test_three_methods_agree(self):
        self.assertLess(self.result["max_diff_loop_vs_vmap"], 1e-5)
        self.assertLess(self.result["max_diff_analytic_first_layer"], 1e-5)

    def test_mean_of_per_sample_equals_batch_gradient(self):
        self.assertLess(self.result["mean_vs_batch_grad_diff"], 1e-6)

    def test_per_sample_tensor_shape_matches_parameter_count(self):
        model, _, _ = lab.make_problem(0, 16, 32)
        num_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(tuple(self.result["tensors"]["per_sample_flat"].shape), (16, num_params))

    def test_every_clipped_gradient_has_norm_at_most_c(self):
        t = self.result["tensors"]
        clipped_norms = t["clipped"].norm(dim=1)
        self.assertTrue(bool((clipped_norms <= 1.0 + 1e-6).all()))
        # Re-derive: rows already inside the ball are untouched, others land on the sphere.
        flat, norms = t["per_sample_flat"], t["norms"]
        inside = norms <= 1.0
        torch.testing.assert_close(t["clipped"][inside], flat[inside])
        if (~inside).any():
            torch.testing.assert_close(
                clipped_norms[~inside], torch.ones(int((~inside).sum())), atol=1e-5, rtol=0
            )
        expected_fraction = float((norms > 1.0).float().mean())
        self.assertAlmostEqual(self.result["clipped_fraction"], expected_fraction)
        expected_update = (t["clipped"].sum(0) + t["noise"]) / flat.shape[0]
        torch.testing.assert_close(t["update"], expected_update)

    def test_clipping_leaves_small_gradients_alone(self):
        flat = torch.tensor([[0.3, 0.4], [3.0, 4.0]])  # norms 0.5 and 5.0
        out = lab.dp_sgd_step(flat, clip_norm=1.0, noise_multiplier=0.0, seed=0, return_tensors=True)
        torch.testing.assert_close(out["clipped"][0], flat[0])
        torch.testing.assert_close(out["clipped"][1], torch.tensor([0.6, 0.8]), atol=1e-5, rtol=0)
        self.assertEqual(out["clipped_fraction"], 0.5)

    def test_dp_noise_reproducible_for_seed_and_different_across_seeds(self):
        flat = self.result["tensors"]["per_sample_flat"]
        a = lab.dp_sgd_step(flat, seed=7, return_tensors=True)
        b = lab.dp_sgd_step(flat, seed=7, return_tensors=True)
        c = lab.dp_sgd_step(flat, seed=8, return_tensors=True)
        torch.testing.assert_close(a["noise"], b["noise"], atol=0, rtol=0)
        torch.testing.assert_close(a["update"], b["update"], atol=0, rtol=0)
        self.assertFalse(torch.equal(a["noise"], c["noise"]))
        # Noise std is sigma * C = 1.0; loose check over P draws.
        self.assertTrue(0.8 < float(a["noise"].std()) < 1.2)

    def test_main_contract_and_json_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            result = lab.main(["--seed", "1", "--batch", "8", "--hidden", "16", "--json-out", str(path)])
            on_disk = json.loads(path.read_text())
        expected = {
            "max_diff_loop_vs_vmap", "max_diff_analytic_first_layer", "mean_vs_batch_grad_diff",
            "clipped_fraction", "dp_update_norm", "time_loop_ms", "time_vmap_ms",
        }
        self.assertEqual(set(result), expected)
        self.assertEqual(set(on_disk), expected)
        self.assertLess(result["max_diff_loop_vs_vmap"], 1e-5)
        self.assertTrue(0.0 <= result["clipped_fraction"] <= 1.0)
        self.assertTrue(math.isfinite(result["dp_update_norm"]))
        self.assertGreater(result["time_loop_ms"], 0.0)  # timing ratio deliberately not asserted


if __name__ == "__main__":
    unittest.main()
