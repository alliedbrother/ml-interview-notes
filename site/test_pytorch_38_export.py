"""Export / quantize / serve contracts for PyTorch course module 38 (llm/export.py, 38_export_quantize.py).

Loads the committed checkpoint through llm.train.load_pretrained(); never retrains,
never uses Inductor or torchao. Asserts contracts and relative quality bars against
the fp32 checkpoint, not the exact numbers of one machine.
"""

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
PROMPT = "Sherlock Holmes"

lab = ex = None
RESULT = MODEL = TOK = VAL = None
_TMP = None
_STATE = {}


def setUpModule():
    global lab, ex, RESULT, MODEL, TOK, VAL, _TMP
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  rng=torch.get_rng_state(), path=list(sys.path),
                  engine=torch.backends.quantized.engine)
    spec = importlib.util.spec_from_file_location("pytorch_38_export_quantize", CODE / "38_export_quantize.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.export as export_mod
    from llm.data import encode_all_splits
    from llm.train import load_pretrained
    ex = export_mod
    _TMP = tempfile.TemporaryDirectory()
    RESULT = lab.main(["all", "--try-ao-dynamic", "--output-dir", _TMP.name])
    MODEL, TOK, _ = load_pretrained()
    VAL = encode_all_splits(TOK)["val"]


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    torch.set_rng_state(_STATE["rng"])
    torch.backends.quantized.engine = _STATE["engine"]
    sys.path[:] = _STATE["path"]


class Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(_TMP.name) / "gpt.pt2"
        assert path.is_file(), "the lab must leave gpt.pt2 in --output-dir"
        cls.ep = ex.load_exported(path)
        cls.lm = ex.ExportedLM(cls.ep, MODEL.config)

    def test_signature_and_ranges(self):
        sig = self.ep.graph_signature
        self.assertEqual(list(sig.user_inputs), ["idx"])
        self.assertEqual(sorted(str(v) for v in self.ep.range_constraints.values()),
                         ["VR[1, 64]", "VR[2, 128]"])
        self.assertEqual((self.lm.t_min, self.lm.t_max), (2, 128))
        self.assertGreater(RESULT["export"]["pt2_bytes"], 0)

    @torch.no_grad()
    def test_logits_match_eager(self):
        gen = torch.Generator().manual_seed(380)
        for T in (2, 17, 128):
            x = torch.randint(0, 512, (2, T), generator=gen)
            torch.testing.assert_close(self.lm(x)[0], MODEL(x)[0], rtol=0, atol=1e-5)
        self.assertLessEqual(RESULT["export"]["max_logit_diff"], 1e-5)

    def test_out_of_range_lengths_rejected(self):
        with self.assertRaises(Exception):
            self.lm.gm(torch.zeros(1, 129, dtype=torch.long))      # the program's own check
        with self.assertRaises(ValueError):
            self.lm(torch.zeros(1, 129, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.lm(torch.zeros(1, 1, dtype=torch.long))           # below Dim min=2
        self.assertTrue(RESULT["export"]["rejects_T_129"])

    def test_greedy_generation_equals_eager(self):
        # the lab generated 32 greedy tokens through the reloaded program and through eager
        self.assertTrue(RESULT["export"]["generation_equal"])
        idx = torch.tensor([TOK.encode(PROMPT)])
        self.assertEqual(ex.greedy_generate(self.lm, idx, 32).shape, (1, idx.size(1) + 32))


class Quantize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = ex.quantize_int8_weight_only(MODEL)

    def test_structure_and_tied_head(self):
        qlin = [m for m in self.q.modules() if isinstance(m, ex.Int8WeightOnlyLinear)]
        self.assertEqual(len(qlin), 16)
        self.assertTrue(all(m.weight.dtype == torch.int8 for m in qlin))
        plain = [n for n, m in self.q.named_modules() if isinstance(m, nn.Linear)]
        self.assertEqual(plain, ["lm_head"])
        self.assertIs(self.q.lm_head.weight, self.q.wte.weight)
        self.assertEqual(self.q.wte.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(self.q.wte.weight, MODEL.wte.weight))
        self.assertIsInstance(MODEL.blocks[0].attn.c_attn, nn.Linear, "original model untouched")

    def test_size_and_quality_bars(self):
        r = RESULT["quantize"]
        self.assertLess(r["int8_bytes"], 0.5 * r["fp32_bytes"])
        self.assertEqual(ex.state_dict_bytes(self.q), r["int8_bytes"])
        delta = r["val_nll_int8"] - r["val_nll_fp32"]
        self.assertLess(delta, 0.15)
        self.assertGreaterEqual(delta, -0.02)
        self.assertGreaterEqual(r["agreement"], 0.75)
        self.assertTrue(r["head_still_tied"])

    def test_agreement_is_a_fraction(self):
        idx = torch.tensor([TOK.encode(PROMPT)])
        self.assertEqual(ex.greedy_agreement(MODEL, MODEL, idx, 8), 1.0)
        self.assertTrue(0.75 <= RESULT["quantize"]["agreement"] <= 1.0)  # 64 tokens, in the lab


class Fallbacks(unittest.TestCase):
    def test_fallback_evaluator_matches_upstream(self):
        fb = ex._fallback_val_nll(MODEL, VAL, 128)
        self.assertAlmostEqual(fb, RESULT["quantize"]["val_nll_fp32"], delta=1e-4)

    def test_fallback_generate_matches(self):
        idx = torch.tensor([TOK.encode(PROMPT)])
        self.assertTrue(torch.equal(ex._fallback_generate(MODEL, idx, 16),
                                    ex.greedy_generate(MODEL, idx, 16)))


class Serve(unittest.TestCase):
    def test_deterministic_and_rng_free(self):
        rng = torch.get_rng_state()
        a = ex.complete(MODEL, TOK, PROMPT, max_new_tokens=16)
        b = ex.complete(MODEL, TOK, PROMPT, max_new_tokens=16)
        self.assertEqual(a, b)
        s1 = ex.complete(MODEL, TOK, PROMPT, max_new_tokens=16, temperature=0.8, seed=7)
        s2 = ex.complete(MODEL, TOK, PROMPT, max_new_tokens=16, temperature=0.8, seed=7)
        self.assertEqual(s1, s2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), "complete() must not use the global RNG")
        self.assertEqual(set(a), {"text", "n_tokens", "stop_reason", "prompt_tokens", "model_sha256"})
        self.assertEqual(len(a["model_sha256"]), 64)

    def test_validation_and_stop_reasons(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                ex.complete(MODEL, TOK, PROMPT, max_new_tokens=bad)
        with self.assertRaises(ValueError):
            ex.complete(MODEL, TOK, "", max_new_tokens=4)
        with self.assertRaises(ValueError):
            ex.complete(MODEL, TOK, PROMPT, temperature=-1.0)
        with self.assertRaises(ValueError):
            ex.complete(MODEL, TOK, "word " * 200)                   # prompt beyond the context
        r = ex.complete(MODEL, TOK, PROMPT, max_new_tokens=4)
        self.assertIn(r["stop_reason"], ex.STOP_REASONS)
        capped = RESULT["serve"]["capped"]
        self.assertEqual(capped["stop_reason"] in ("context", "eot"), True)
        if capped["stop_reason"] == "context":
            self.assertEqual(capped["n_tokens"] + capped["prompt_tokens"], 128)
        self.assertIn(RESULT["serve"]["stop_reason"], ex.STOP_REASONS)

    def test_eot_stops_and_is_not_returned(self):
        eot = TOK.eot_id
        fake = lambda idx: (torch.nn.functional.one_hot(  # noqa: E731 - always predicts EOT
            torch.full_like(idx, eot), 512).float(), None)
        fake.config = MODEL.config
        fake.state_dict = lambda: {}
        r = ex.complete(fake, TOK, PROMPT, max_new_tokens=8)
        self.assertEqual((r["stop_reason"], r["n_tokens"], r["text"]), ("eot", 0, ""))


class AoDynamic(unittest.TestCase):
    def test_result_or_none_and_engine_restored(self):
        r = RESULT["ao_dynamic"]
        if r is not None:
            self.assertTrue(math.isfinite(r["val_nll"]))
            self.assertLess(r["val_nll_delta"], 0.15)
        before = torch.backends.quantized.engine
        again = ex.try_ao_dynamic(MODEL)
        self.assertEqual(torch.backends.quantized.engine, before)
        self.assertTrue(again is None or again["bytes"] > 0)

    def test_no_engine_returns_none(self):
        before = torch.backends.quantized.engine
        with mock.patch.object(ex, "pick_quantized_engine", return_value=None):
            self.assertIsNone(ex.try_ao_dynamic(MODEL))
        self.assertEqual(torch.backends.quantized.engine, before)


if __name__ == "__main__":
    unittest.main()
