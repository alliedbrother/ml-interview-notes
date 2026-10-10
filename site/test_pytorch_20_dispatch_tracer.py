"""Contracts for the module 20 lab: tracing aten ops with a dispatch mode and a wrapper subclass."""

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


lab = load("20_dispatch_tracer")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class DispatchTracerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        before = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "result.json"
                cls.result = lab.main(["--seed", "0", "--json-out", str(path)])
                cls.from_disk = json.loads(path.read_text())
        finally:
            torch.set_num_threads(before)
        cls.threads_after_restore = torch.get_num_threads()
        cls.threads_before = before

    def test_result_is_json_round_trippable(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)

    def test_forward_has_one_addmm_per_linear(self):
        self.assertEqual(self.result["forward_ops"].get("aten.addmm.default"), 2)
        self.assertEqual(self.result["mm_like_forward"], 2)

    def test_backward_is_visible_below_autograd(self):
        # dL/dW2, dL/dh and dL/dW1; the input needs no gradient, so no fourth mm.
        self.assertGreaterEqual(self.result["mm_like_backward"], 3)
        self.assertNotIn("aten.addmm.default", self.result["backward_ops"])
        self.assertTrue(any("backward" in op for op in self.result["backward_ops"]))

    def test_optimizer_step_is_an_add_family(self):
        self.assertTrue(any("add" in op for op in self.result["optimizer_ops"]))
        self.assertFalse(any("mm" in op for op in self.result["optimizer_ops"]))

    def test_total_is_sum_of_phases(self):
        total = sum(sum(self.result[k].values()) for k in ("forward_ops", "backward_ops", "optimizer_ops"))
        self.assertEqual(self.result["total_ops"], total)

    def test_tracing_does_not_change_numerics(self):
        self.assertIs(self.result["traced_step_matches"], True)
        self.assertIs(self.result["subclass_output_matches"], True)
        self.assertIs(self.result["subclass_grad_matches"], True)
        self.assertEqual(self.result["subclass_grad_type"], "LoggingTensor")

    def test_subclass_grads_match_plain_run_directly(self):
        logits, grads, log, _ = lab.subclass_step(3)
        ref_logits, ref_grads, _ = lab.plain_step(3)
        torch.testing.assert_close(logits, ref_logits)
        for g, r in zip(grads, ref_grads):
            torch.testing.assert_close(g, r)
        self.assertIn("aten.addmm.default", log)
        self.assertIn("aten.mm.default", log)

    def test_num_threads_restored(self):
        self.assertEqual(self.threads_after_restore, self.threads_before)


if __name__ == "__main__":
    unittest.main()
