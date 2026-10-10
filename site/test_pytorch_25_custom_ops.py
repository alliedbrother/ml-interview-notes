"""Contracts for the module 25 lab: a registered custom op, compile visibility, and tiled attention."""

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


def load(name, alias=None):
    spec = importlib.util.spec_from_file_location(alias or name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[alias or name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# Imported once: the ops live in the pytc_lab25 namespace, which no other test or fence uses.
lab = load("25_custom_ops")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch._dynamo.reset()


class CustomOpContracts(unittest.TestCase):
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

    def test_result_is_json_round_trippable(self):
        self.assertEqual(json.loads(json.dumps(self.result)), self.from_disk)

    def test_opcheck_all_success(self):
        self.assertEqual(set(self.result["opcheck"]), {
            "test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"})
        for name, status in self.result["opcheck"].items():
            self.assertEqual(status, "SUCCESS", name)

    def test_gradcheck(self):
        self.assertIs(self.result["gradcheck"], True)

    def test_recompute_backward_saves_less_than_composite(self):
        self.assertLess(self.result["saved_bytes_fused"], self.result["saved_bytes_unfused"])
        self.assertGreater(self.result["saved_bytes_fused"], 0)

    def test_compiles_to_one_graph_and_matches_eager(self):
        self.assertEqual(self.result["compile_graphs"], 1)
        self.assertLess(self.result["compile_max_diff"], 1e-6)
        self.assertIs(self.result["op_in_graph"], True)

    def test_tiled_attention_matches_sdpa(self):
        self.assertNotEqual(self.result["seq_len"] % 16, 0)  # ragged last tile exercised
        for name in ("causal", "full"):
            self.assertLess(self.result["tiled_max_diff"][name], 1e-5, name)
        self.assertLess(self.result["tiles"]["causal"], self.result["tiles"]["full"])

    def test_no_full_score_matrix(self):
        self.assertLess(self.result["max_intermediate_numel"], self.result["full_scores_numel"])

    def test_tiled_attention_block_sizes_agree(self):
        g = torch.Generator().manual_seed(1)
        q, k, v = (torch.randn(1, 1, 23, 8, generator=g, dtype=torch.float64) for _ in range(3))
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        for block in (1, 5, 23, 64):
            out, _ = lab.tiled_attention(q, k, v, block, causal=True)
            torch.testing.assert_close(out, ref, atol=1e-12, rtol=0)

    def test_op_lives_in_lab_namespace(self):
        op = torch.ops.pytc_lab25.fused_bias_gelu.default
        self.assertEqual(op._schema.name, "pytc_lab25::fused_bias_gelu")
        self.assertEqual(str(op._schema),
                         "pytc_lab25::fused_bias_gelu(Tensor x, Tensor bias) -> Tensor")

    def test_reimport_replaces_registration_without_error(self):
        # In 2.8 a second custom_op with the same name destroys the old Library fragment first.
        again = load("25_custom_ops", alias="_25_custom_ops_again")
        try:
            x, b = torch.randn(3, 4), torch.randn(4)
            torch.testing.assert_close(torch.ops.pytc_lab25.fused_bias_gelu(x, b),
                                       torch.nn.functional.gelu(x + b), atol=1e-6, rtol=1e-6)
            self.assertIsNot(again.fused_bias_gelu, lab.fused_bias_gelu)
        finally:
            sys.modules.pop("_25_custom_ops_again", None)
            load("25_custom_ops")  # leave the canonical registration in place


if __name__ == "__main__":
    unittest.main()
