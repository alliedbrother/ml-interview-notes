"""Contracts for the module 12 lab: attention implementations, mask polarity, offset-chunk trap."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"

_saved = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    # Registered so that Dynamo (used inside eager flex_attention) can resolve the mask_mod globals.
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("12_attention_backends")


def setUpModule():
    _saved["threads"] = torch.get_num_threads()
    _saved["dtype"] = torch.get_default_dtype()
    _saved["det"] = torch.are_deterministic_algorithms_enabled()
    torch._dynamo.reset()


def tearDownModule():
    torch._dynamo.reset()
    torch.set_num_threads(_saved["threads"])
    torch.set_default_dtype(_saved["dtype"])
    torch.use_deterministic_algorithms(_saved["det"])


class AttentionBackendContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = lab.run(seed=0, time_it=False, return_tensors=True)

    def test_every_implementation_and_mask_agrees(self):
        diffs = self.result["max_abs_diff"]
        impls = ("sdpa", "sdpa_math", "mha", "flex")
        masks = ("causal", "window", "document", "padding")
        for impl in impls:
            for mask in masks:
                self.assertIn(f"{impl}_{mask}", diffs)
        for key, value in diffs.items():
            self.assertLess(value, 1e-5, key)

    def test_gradients_match(self):
        self.assertLess(self.result["grad_max_abs_diff"], 1e-4)

    def test_offset_chunk_trap_is_real(self):
        self.assertGreater(self.result["offset_chunk_is_causal_diff"], 1e-3)
        # And the explicit bottom-right mask reproduces the last rows of full causal attention.
        self.assertLess(self.result["max_abs_diff"]["offset_chunk_explicit_vs_full"], 1e-5)

    def test_document_two_equals_running_it_alone(self):
        t = self.result["tensors"]
        len_a, _ = lab.doc_lengths(t["q"].size(-2))
        q, k, v = (x[:, :, len_a:] for x in (t["q"], t["k"], t["v"]))
        alone, _ = lab.manual_attention(q, k, v, torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril())
        torch.testing.assert_close(t["outputs"]["document"][:, :, len_a:], alone, atol=1e-5, rtol=0)

    def test_sliding_window_puts_no_weight_outside_the_window(self):
        t = self.result["tensors"]
        _, weights = lab.manual_attention(t["q"], t["k"], t["v"], t["masks"]["window"])
        seq = weights.size(-1)
        i = torch.arange(seq)[:, None]
        j = torch.arange(seq)[None, :]
        outside = (j > i) | (i - j >= lab.WINDOW)
        self.assertEqual(float(weights[..., outside].abs().max()), 0.0)
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[:-1]), atol=1e-6, rtol=0)

    def test_padding_keys_get_zero_weight_and_mha_polarity_is_inverted(self):
        t = self.result["tensors"]
        _, weights = lab.manual_attention(t["q"], t["k"], t["v"], t["masks"]["padding"])
        for b, length in enumerate(t["lengths"].tolist()):
            if length < weights.size(-1):
                self.assertEqual(float(weights[b, ..., length:].abs().max()), 0.0)
        # Copying MHA polarity (True = padding) into SDPA inverts the meaning: batch item 1 then
        # attends ONLY to its padding keys, so its output changes.
        q, k, v = (x[1:2] for x in (t["q"], t["k"], t["v"]))
        keep = t["masks"]["padding"][1:2]
        right = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        wrong = F.scaled_dot_product_attention(q, k, v, attn_mask=~keep)
        self.assertGreater(float((right - wrong).abs().max()), 1e-3)

    def test_main_contract_and_json_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            result = lab.main(["--seed", "1", "--json-out", str(path)])
            on_disk = json.loads(path.read_text())
        expected = {
            "max_abs_diff", "grad_max_abs_diff", "offset_chunk_is_causal_diff",
            "nested_causal_supported", "timings_us", "torch",
        }
        self.assertEqual(set(result), expected)
        self.assertEqual(set(on_disk), expected)
        self.assertEqual(json.loads(json.dumps(result)), on_disk)
        self.assertIsInstance(result["nested_causal_supported"], bool)
        self.assertTrue(all(value < 1e-5 for value in result["max_abs_diff"].values()))
        self.assertTrue(all(math.isfinite(us) and us > 0 for us in result["timings_us"].values()))
        # Timings are reported, never compared.


if __name__ == "__main__":
    unittest.main()
