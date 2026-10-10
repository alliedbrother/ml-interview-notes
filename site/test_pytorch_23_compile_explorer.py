"""Contracts for the module 23 lab: graph breaks, guards, dynamic shapes and compiled training (aot_eager only)."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch._dynamo


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
_STATE = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("23_compile_explorer")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch._dynamo.reset()


class CompileExplorerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        before = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "result.json"
                cls.result = lab.main(["--seed", "0", "--backend", "aot_eager", "--json-out", str(path)])
                cls.from_disk = json.loads(path.read_text())
        finally:
            torch.set_num_threads(before)
            torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_result_is_json_round_trippable(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)
        self.assertEqual(self.result["backend"], "aot_eager")

    def test_messy_model_breaks_and_clean_rewrite_does_not(self):
        messy, clean = self.result["messy"], self.result["clean"]
        self.assertGreaterEqual(messy["breaks"], 3)
        self.assertEqual(messy["graphs"], messy["breaks"] + 1)
        self.assertEqual(len(messy["reasons"]), messy["breaks"])
        self.assertEqual(clean["breaks"], 0)
        self.assertEqual(clean["graphs"], 1)
        self.assertIs(self.result["outputs_match"], True)

    def test_dynamic_setting_controls_frame_count(self):
        frames = self.result["frames_by_dynamic"]
        self.assertEqual(frames["False"], len(lab.LENGTHS))  # one static graph per length
        self.assertEqual(frames["True"], 1)
        self.assertLessEqual(frames["None"], 3)
        self.assertGreaterEqual(frames["None"], 1)
        self.assertLess(len(lab.LENGTHS), torch._dynamo.config.recompile_limit)

    def test_float_attribute_recompiles_buffer_does_not(self):
        self.assertGreaterEqual(self.result["attr_recompiles"], 1)
        self.assertEqual(self.result["buffer_recompiles"], 0)

    def test_regional_compilation_counts(self):
        regional = self.result["regional"]
        self.assertGreaterEqual(regional["per_block_frames"], 1)
        self.assertGreaterEqual(regional["whole_frames"], 1)
        # The per-block graph is one block; the whole graph inlines all six.
        self.assertLess(regional["per_block_ops"] / regional["per_block_frames"], regional["whole_ops"] / regional["whole_frames"])
        self.assertGreater(regional["whole_s"], 0.0)
        self.assertGreater(regional["per_block_s"], 0.0)

    def test_compiled_training_matches_eager(self):
        self.assertLess(self.result["train_max_param_diff"], 1e-6)

    def test_explain_report_directly(self):
        torch.manual_seed(0)
        report = lab.explain_report(lab.CleanModel(), torch.randn(3, lab.D))
        self.assertEqual(report["breaks"], 0)

    def test_num_threads_restored(self):
        self.assertEqual(torch.get_num_threads(), _STATE["threads"])


if __name__ == "__main__":
    unittest.main()
