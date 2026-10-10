"""Contracts for the module 28 lab: torch.export round trip, dynamic dims, decompositions, unflatten, draft_export."""

import hashlib
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
    sys.modules[name] = loaded  # torch.cond traces branch lambdas via Dynamo, which re-imports their module
    spec.loader.exec_module(loaded)
    return loaded


lab = load("28_export_roundtrip")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch._dynamo.reset()


class ExportRoundTripContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        before = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "artifacts"
                json_path = Path(tmp) / "result.json"
                cls.result = lab.main(["--seed", "0", "--output-dir", str(out), "--json-out", str(json_path)])
                cls.from_disk = json.loads(json_path.read_text())
                cls.manifest = json.loads((out / "manifest.json").read_text())
                cls.file_sha = hashlib.sha256((out / "model.pt2").read_bytes()).hexdigest()
                cls.file_bytes = (out / "model.pt2").stat().st_size
                cls.tmp_root = Path(tmp)
        finally:
            torch.set_num_threads(before)
            torch._dynamo.reset()

    def test_result_is_json_round_trippable(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)

    def test_reloaded_program_matches_eager_on_shape_grid(self):
        self.assertLess(self.result["max_diff"], 1e-6)
        self.assertGreaterEqual(self.result["shapes_checked"], 6)

    def test_manifest_hash_matches_file(self):
        self.assertEqual(self.manifest["sha256"], self.file_sha)
        self.assertEqual(self.result["sha256"], self.file_sha)
        self.assertEqual(self.manifest["bytes"], self.file_bytes)
        self.assertEqual(self.manifest["dims"], {"B": [1, 32], "T": [2, 64]})
        self.assertEqual(self.manifest["torch"], torch.__version__)

    def test_decompositions_remove_linear(self):
        self.assertIn("aten.linear.default", self.result["ops_training_ir"])
        self.assertNotIn("aten.linear.default", self.result["ops_core_aten"])
        self.assertIn("cond", self.result["ops_core_aten"])  # control flow survives lowering

    def test_contract_violations_raise(self):
        rejections = self.result["rejections"]
        self.assertTrue(rejections["T_out_of_range"])
        self.assertTrue(rejections["wrong_dtype"])

    def test_unflatten_head_swap(self):
        self.assertIs(self.result["unflatten_ok"], True)

    def test_draft_export_reports_item_branch(self):
        self.assertGreaterEqual(self.result["draft_export_issues"], 1)

    def test_timings_reported_not_asserted(self):
        self.assertGreater(self.result["eager_us"], 0.0)
        self.assertGreater(self.result["exported_us"], 0.0)

    def test_artifacts_removed_without_output_dir(self):
        before = set(Path(tempfile.gettempdir()).iterdir())
        threads = torch.get_num_threads()
        try:
            lab.main(["--seed", "1"])
        finally:
            torch.set_num_threads(threads)
        leaked = [p for p in set(Path(tempfile.gettempdir()).iterdir()) - before if (p / "model.pt2").exists()]
        self.assertEqual(leaked, [])

    def test_num_threads_restored(self):
        self.assertEqual(torch.get_num_threads(), _STATE["threads"])


if __name__ == "__main__":
    unittest.main()
