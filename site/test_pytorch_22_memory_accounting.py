"""Contracts for the module 22 lab: memory buckets and activation checkpointing."""

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


lab = load("22_memory_accounting")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class MemoryAccountingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            cls.result = lab.main(["--seed", "0", "--skip-timing", "--json-out", str(path)])
            cls.from_disk = json.loads(path.read_text())

    def test_result_is_json_round_trippable(self):
        self.assertEqual(self.result, self.from_disk)

    def test_static_buckets_match_bytes_per_parameter(self):
        r = self.result
        self.assertEqual(r["param_bytes"], 4 * r["n_params"])
        self.assertEqual(r["grad_bytes"], r["param_bytes"])
        self.assertEqual(r["adamw_state_bytes"], 2 * r["param_bytes"])

    def test_checkpointing_orders_saved_bytes(self):
        s = self.result["saved_bytes"]
        self.assertLess(s["full"], s["every_2"])
        self.assertLess(s["every_2"], s["none"])
        self.assertLessEqual(s["selective"], s["none"])
        self.assertLess(s["full"], s["selective"])

    def test_full_checkpointing_keeps_only_block_inputs_and_loss_input(self):
        c = self.result["config"]
        unit = 4 * c["batch"] * c["seq"] * c["width"]  # one fp32 (B, T, d_model) tensor
        self.assertEqual(self.result["saved_bytes"]["full"], (c["layers"] + 1) * unit)

    def test_gradients_identical_across_variants(self):
        self.assertLess(self.result["max_grad_diff"], 1e-5)

    def test_measured_within_35_percent_of_prediction(self):
        # The 16-elements-per-token prediction ignores LayerNorm statistics,
        # the SDPA log-sum-exp and the loss input, so allow 35 %.
        pred = self.result["predicted_saved_bytes_none"]
        meas = self.result["saved_bytes"]["none"]
        self.assertLess(abs(meas - pred) / pred, 0.35)

    def test_timing_skipped_not_asserted(self):
        self.assertEqual(set(self.result["time_s"]), {"none", "full", "every_2", "selective"})
        self.assertTrue(all(t is None for t in self.result["time_s"].values()))


if __name__ == "__main__":
    unittest.main()
