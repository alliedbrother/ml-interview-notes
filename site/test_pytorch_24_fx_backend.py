"""Contracts for the module 24 lab: an FX optimization pass inside a custom torch.compile backend."""

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


lab = load("24_fx_backend")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch._dynamo.reset()


class FxBackendContracts(unittest.TestCase):
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
        cls.threads_before = before
        cls.threads_after = torch.get_num_threads()

    def tearDown(self):
        torch._dynamo.reset()

    def test_result_is_json_round_trippable(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)

    def test_both_passes_fire_twice(self):
        self.assertEqual(self.result["rewrites"]["silu"], 2)
        self.assertEqual(self.result["rewrites"]["conv_bn"], 2)
        self.assertFalse(self.result["rewritten_has_batch_norm"])
        self.assertFalse(self.result["rewritten_has_sigmoid"])

    def test_dynamo_captured_one_graph_with_the_patterns(self):
        self.assertEqual(self.result["graphs"], 1)
        ops = self.result["dynamo_graph_ops"]
        self.assertEqual(ops.get("torch.sigmoid"), 2)
        self.assertEqual(ops.get("torch.nn.functional.batch_norm"), 2)

    def test_transformed_model_matches_eager(self):
        self.assertLessEqual(self.result["max_abs_diff"], 1e-5)

    def test_transformed_model_directly_on_fresh_input(self):
        torch._dynamo.reset()
        model = lab.build_model(3).eval()
        backend = lab.FusionBackend()
        compiled = torch.compile(model, backend=backend)
        x = torch.randn(2, 3, 6, 6, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), atol=1e-5, rtol=1e-5)
        self.assertEqual(backend.rewrites, {"silu": 2, "conv_bn": 2})

    def test_train_mode_is_left_alone_by_conv_bn_pass(self):
        torch._dynamo.reset()
        model = lab.build_model(0).train()
        backend = lab.FusionBackend()
        compiled = torch.compile(model, backend=backend)
        x = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(1))
        ref = lab.build_model(0).train()
        torch.testing.assert_close(compiled(x), ref(x), atol=1e-5, rtol=1e-5)
        self.assertEqual(backend.rewrites["conv_bn"], 0)
        self.assertEqual(backend.rewrites["silu"], 2)

    def test_aot_graphs_are_aten_functional_and_save_tensors(self):
        self.assertGreater(self.result["fw_nodes"], 0)
        self.assertGreater(self.result["bw_nodes"], 0)
        self.assertGreaterEqual(self.result["saved_for_backward"], 1)
        self.assertEqual(self.result["fw_inplace_ops"], [])
        self.assertTrue(self.result["fw_all_aten"])
        self.assertTrue(self.result["aot_out_matches"])
        self.assertTrue(self.result["aot_grads_match"])

    def test_num_threads_restored(self):
        self.assertEqual(self.threads_after, self.threads_before)


if __name__ == "__main__":
    unittest.main()
