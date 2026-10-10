"""Contracts for the module 21 lab: profile one CPU training step, count FLOPs, compute MFU."""

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


lab = load("21_profile_step")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class ProfileStepContracts(unittest.TestCase):
    BATCH, WIDTH, DEPTH, VOCAB = 64, 256, 3, 512

    @classmethod
    def setUpClass(cls):
        before = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "result.json"
                cls.result = lab.main([
                    "--seed", "0", "--batch", str(cls.BATCH), "--width", str(cls.WIDTH),
                    "--depth", str(cls.DEPTH), "--vocab", str(cls.VOCAB),
                    "--trace-dir", tmp, "--json-out", str(out),
                ])
                cls.from_disk = json.loads(out.read_text())
                trace = Path(cls.result["trace_path"])
                cls.trace_exists = trace.is_file() and trace.parent == Path(tmp)
                cls.trace = json.loads(trace.read_text()) if cls.trace_exists else None
        finally:
            torch.set_num_threads(before)

    def test_result_round_trips_through_json(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)

    def test_flops_match_six_b_sum_in_out(self):
        linears = [(self.WIDTH, self.WIDTH)] * self.DEPTH + [(self.WIDTH, self.VOCAB)]
        expected = 6 * self.BATCH * sum(i * o for i, o in linears)
        self.assertEqual(self.result["analytic_flops_per_step"], expected)
        self.assertEqual(self.result["flops_per_step"], expected)

    def test_mfu_is_a_sane_fraction(self):
        # Sanity, not quality: > 1 could only come from timing noise in the ceiling.
        self.assertGreater(self.result["mfu"], 0.0)
        self.assertLessEqual(self.result["mfu"], 1.5)
        self.assertGreater(self.result["median_step_s"], 0.0)
        self.assertGreater(self.result["matmul_peak_gflops"], 0.0)

    def test_top_ops_contain_a_matmul(self):
        names = [row[0] for row in self.result["top_ops"]]
        self.assertTrue(0 < len(names) <= 5)
        self.assertTrue({"aten::addmm", "aten::mm"} & set(names), names)
        for _, self_us, count in self.result["top_ops"]:
            self.assertGreaterEqual(self_us, 0.0)
            self.assertGreaterEqual(count, 1)

    def test_all_phases_recorded(self):
        self.assertEqual(set(self.result["phase_ms"]), {"forward", "backward", "optimizer"})
        for value in self.result["phase_ms"].values():
            self.assertGreater(value, 0.0)

    def test_trace_is_chrome_json(self):
        self.assertTrue(self.trace_exists)
        self.assertIn("traceEvents", self.trace)
        names = {e.get("name") for e in self.trace["traceEvents"]}
        self.assertTrue({"forward", "backward", "optimizer"} <= names)

    def test_too_few_steps_rejected(self):
        with self.assertRaises(SystemExit):
            lab.main(["--steps", "4"])


if __name__ == "__main__":
    unittest.main()
