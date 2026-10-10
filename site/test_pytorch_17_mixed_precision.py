"""Contracts for the module 17 lab: fp32 vs bf16 autocast vs fp16 with and without GradScaler."""

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
_STATE = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("17_mixed_precision")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class MixedPrecisionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            cls.result = lab.main(["--seed", "0", "--json-out", str(path)])
            cls.from_disk = json.loads(path.read_text())

    def test_result_is_json_round_trippable(self):
        self.assertEqual(self.result, self.from_disk)

    def test_low_precision_runs_match_fp32_and_learn(self):
        final = self.result["final_loss"]
        initial = self.result["initial_loss"]
        self.assertLess(final["fp32"], 0.5 * initial)
        for variant in ("bf16", "fp16_scaler"):
            with self.subTest(variant=variant):
                rel = abs(final[variant] - final["fp32"]) / final["fp32"]
                self.assertLess(rel, 0.10)
                self.assertLess(final[variant], 0.5 * initial)

    def test_naive_fp16_gradients_underflow_and_scaler_recovers_them(self):
        self.assertGreater(self.result["zero_grad_frac_naive"], 0.5)
        self.assertLess(self.result["zero_grad_frac_scaled"], 0.05)
        self.assertLess(self.result["zero_grad_frac"]["fp32"], 0.05)
        self.assertLess(self.result["zero_grad_frac"]["bf16"], 0.05)

    def test_scale_trace_is_powers_of_two_and_backs_off_on_overflow(self):
        trace = self.result["scale_trace"]
        self.assertEqual(len(trace), 151)
        for s in trace:
            self.assertEqual(math.log2(s), round(math.log2(s)))
        drops = [(a, b) for a, b in zip(trace, trace[1:]) if b < a]
        self.assertEqual(len(drops), self.result["skipped_steps"])
        self.assertGreater(self.result["skipped_steps"], 0)
        for a, b in drops:
            self.assertEqual(b, a * 0.5)  # backoff_factor
        for a, b in zip(trace, trace[1:]):
            self.assertIn(b / a, (0.5, 1.0, 2.0))
        self.assertEqual(self.result["scaler_state"]["scale"], trace[-1])

    def test_parameters_stay_fp32(self):
        self.assertEqual(self.result["param_dtypes"], ["torch.float32"])

    def test_skipped_steps_leave_parameters_unchanged(self):
        self.assertIs(self.result["skipped_params_unchanged"], True)
        forced = self.result["forced_inf"]
        self.assertIs(forced["inf_step_params_unchanged"], True)
        self.assertEqual(forced["scale_after_inf"], 2.0 ** 15)
        self.assertIs(forced["clean_step_params_unchanged"], False)


if __name__ == "__main__":
    unittest.main()
