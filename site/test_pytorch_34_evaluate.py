"""Evaluation contracts for PyTorch course module 34 (llm/evaluate.py, 34_evaluate.py).

Loads the committed checkpoint through llm.train.load_pretrained (never trains)
and evaluates the validation split (story XI). The test split is exercised only
through a 512-token slice to prove the code path runs.
"""

import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
UNIFORM_VAL = math.log(512)
UNIGRAM_VAL = 5.22   # add-one unigram on story XI (documented in module 34)
BIGRAM_VAL = 3.67    # add-0.1 bigram with unigram prior on story XI (documented in module 34)

lab = ev = None
MODEL = TOK = MANIFEST = SPLITS = None
R = {}
BASE = {}
_STATE = {}


def setUpModule():
    global lab, ev, MODEL, TOK, MANIFEST, SPLITS
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  rng=torch.get_rng_state(), path=list(sys.path))
    spec = importlib.util.spec_from_file_location("pytorch_34_evaluate", CODE / "34_evaluate.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.evaluate as evaluate_mod
    from llm.data import encode_all_splits
    from llm.train import load_pretrained
    ev = evaluate_mod
    torch.set_num_threads(min(4, torch.get_num_threads()))
    MODEL, TOK, MANIFEST = load_pretrained()
    SPLITS = encode_all_splits(TOK)
    val, train = SPLITS["val"], SPLITS["train"]
    n_bytes = ev.scored_bytes(TOK, val)
    for stride in (128, 64):
        R[stride] = ev.evaluate_nll(MODEL, val, 128, stride, text_bytes=n_bytes)
    BASE.update(uniform=ev.uniform_nll(512), unigram=ev.unigram_nll(train, val, 512),
                bigram=ev.bigram_nll(train, val, 512, k=0.1))


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    torch.set_rng_state(_STATE["rng"])
    sys.path[:] = _STATE["path"]


class HeldOutNLLTest(unittest.TestCase):
    def test_every_target_scored_exactly_once(self):
        n = SPLITS["val"].numel()
        for stride, r in R.items():
            self.assertEqual(r["n_tokens"], n - 1, stride)
            self.assertEqual(sum(r["window_counts"]), n - 1, stride)
            self.assertEqual(len(r["window_losses"]), r["n_windows"])
            torch.testing.assert_close(
                math.fsum(l * c for l, c in zip(r["window_losses"], r["window_counts"])) / (n - 1),
                r["nll_per_token"], rtol=0, atol=1e-9)

    def test_sliding_window_not_worse_than_non_overlapping(self):
        self.assertLessEqual(R[64]["nll_per_token"], R[128]["nll_per_token"] + 0.01)

    def test_stride_128_reproduces_manifest(self):
        self.assertAlmostEqual(R[128]["nll_per_token"], MANIFEST["final_val_loss"], delta=0.05)
        self.assertAlmostEqual(R[128]["ppl"], math.exp(R[128]["nll_per_token"]), places=9)

    def test_bits_per_byte_consistent(self):
        val = SPLITS["val"]
        n_bytes = ev.scored_bytes(TOK, val)
        story = lab.load_splits()["val"][0]
        # val is one story, so decoding is lossless and only the first token's bytes are unscored
        self.assertEqual(n_bytes, len(story.encode("utf-8")) - len(TOK.token_bytes(int(val[0]))))
        for r in R.values():
            bytes_per_token = n_bytes / r["n_tokens"]
            self.assertAlmostEqual(r["bits_per_byte"],
                                   r["nll_per_token"] / (math.log(2) * bytes_per_token), places=9)
        self.assertIsNone(ev.evaluate_nll(MODEL, val[:300], 128, 128)["bits_per_byte"])

    def test_deterministic_and_restores_mode(self):
        val = SPLITS["val"][:2000]
        MODEL.train()
        try:
            a = ev.evaluate_nll(MODEL, val, 128, 64)
            self.assertTrue(MODEL.training)
            b = ev.evaluate_nll(MODEL, val, 128, 64)
        finally:
            MODEL.eval()
        self.assertEqual(a["nll_per_token"], b["nll_per_token"])
        self.assertEqual(a["window_losses"], b["window_losses"])


class BaselineTest(unittest.TestCase):
    def test_model_beats_bigram_beats_unigram_beats_uniform(self):
        nll = R[64]["nll_per_token"]
        self.assertLess(nll, BASE["bigram"] - 0.3)
        self.assertLess(BASE["bigram"] - 0.3, BASE["unigram"] - 1.0)
        self.assertLess(BASE["unigram"] - 1.0, BASE["uniform"])

    def test_baselines_match_documented_values(self):
        self.assertAlmostEqual(BASE["uniform"], UNIFORM_VAL, places=12)
        self.assertAlmostEqual(BASE["unigram"], UNIGRAM_VAL, delta=0.1)
        self.assertAlmostEqual(BASE["bigram"], BIGRAM_VAL, delta=0.1)

    def test_bigram_interpolation_schemes(self):
        train, val = SPLITS["train"], SPLITS["val"]
        # lam=1 with add-k is the plain add-k bigram; lam=0 is the add-one unigram on pairs
        plain = ev.bigram_nll(train, val, 512, k=0.1, lam=1.0)
        unigram_pairs = ev.bigram_nll(train, val, 512, k=0.1, lam=0.0)
        self.assertAlmostEqual(unigram_pairs, BASE["unigram"], places=9)
        self.assertLess(plain, BASE["unigram"])
        with self.assertRaises(ValueError):
            ev.bigram_nll(train, val, 512, lam=1.5)


class UncertaintyTest(unittest.TestCase):
    def test_standard_error_and_block_bootstrap(self):
        r = R[64]
        se = ev.standard_error(r["window_losses"], r["window_counts"])
        self.assertTrue(0 < se < 0.1)
        b1 = ev.block_bootstrap(r["window_losses"], r["window_counts"], 100, seed=34)
        b2 = ev.block_bootstrap(r["window_losses"], r["window_counts"], 100, seed=34)
        self.assertEqual(b1, b2)
        self.assertEqual(b1["block_len"], round(r["n_windows"] ** (1 / 3)))
        self.assertAlmostEqual(b1["mean"], r["nll_per_token"], places=9)
        self.assertLess(b1["lo"], b1["mean"])
        self.assertLess(b1["mean"], b1["hi"])
        state = torch.get_rng_state()
        ev.block_bootstrap(r["window_losses"], r["window_counts"], 10, seed=1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class BehaviourTest(unittest.TestCase):
    def test_causality_on_loaded_model(self):
        g = torch.Generator().manual_seed(34)
        x = torch.randint(0, 512, (2, 64), generator=g)
        x2 = x.clone()
        x2[:, 40:] = torch.randint(0, 512, (2, 24), generator=g)
        with torch.inference_mode():
            a, _ = MODEL(x)
            b, _ = MODEL(x2)
        torch.testing.assert_close(a[:, :40], b[:, :40], rtol=0, atol=1e-5)
        self.assertGreater((a[:, 40:] - b[:, 40:]).abs().max().item(), 1e-3)

    def test_greedy_twice_identical(self):
        ids = TOK.encode("Holmes said")
        self.assertEqual(ev.greedy_continuation(MODEL, ids, 20), ev.greedy_continuation(MODEL, ids, 20))

    def test_seeded_sampling_from_model(self):
        with torch.inference_mode():
            logits, _ = MODEL(torch.tensor([TOK.encode("It was a")]))
        probs = torch.softmax(logits[0, -1], -1)

        def draw(seed):
            return torch.multinomial(probs, 32, replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(draw(7), draw(7)))
        self.assertFalse(torch.equal(draw(7), draw(8)))

    def test_sherlock_holmes_if_margin_is_safe(self):
        token, margin = ev.greedy_margin(MODEL, TOK, "Sherlock")
        if margin <= 1.0:
            self.skipTest(f"logit margin {margin:.3f} <= 1.0; greedy token {token!r} is not platform-robust")
        self.assertEqual(token, " Holmes")
        table = ev.next_token_table(MODEL, TOK, "Sherlock", k=3)
        self.assertEqual(table[0][0], " Holmes")
        self.assertEqual(len(table), 3)
        self.assertTrue(all(table[i][1] >= table[i + 1][1] for i in range(2)))

    def test_longest_common_substring(self):
        lcs = ev.longest_common_substring_in
        self.assertEqual(lcs("xx Holmes yy", "said Holmes, and"), " Holmes")
        self.assertEqual(lcs("abc", "xyz"), "")


class LabTest(unittest.TestCase):
    def test_lab_val_with_prompts(self):
        out = lab.main(["--strides", "128", "--bootstrap", "20", "--prompts"])
        self.assertEqual(out["split"], "val")
        self.assertEqual(set(out["baselines"]), {"uniform", "unigram", "bigram"})
        self.assertEqual(out["results"]["128"]["n_tokens"], SPLITS["val"].numel() - 1)
        self.assertNotIn("window_losses", out["results"]["128"])
        self.assertEqual(len(out["prompts"]), len(lab.PROMPTS))
        self.assertEqual(len(out["bootstrap_ci"]), 2)
        self.assertIn("lcs_chars", out["memorisation"]["val_story_XI"])

    def test_test_split_path_runs_on_slice(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = lab.main(["--split", "test", "--limit-tokens", "512", "--strides", "128,64",
                            "--bootstrap", "10"])
        self.assertIn("final report only", err.getvalue().lower())
        self.assertEqual(out["split"], "test")
        for stride in ("128", "64"):
            self.assertEqual(out["results"][stride]["n_tokens"], 511)
            self.assertTrue(math.isfinite(out["results"][stride]["nll_per_token"]))


if __name__ == "__main__":
    unittest.main()
