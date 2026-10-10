"""Contracts for capstone generation (module 35): llm/generate.py and code/35_generate.py.

Generated-text checks are gated on logit margins: Mac and Linux CPU kernels can
differ in the last bits, so a token mismatch is only acceptable at a near-tie,
and such a mismatch is reported, never silently passed.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from llm.generate import (ContextLimitError, generate, generate_batch, sample_next,  # noqa: E402
                          stream_decode, top_k_filter, top_p_filter, verify_cache)
from llm.train import load_pretrained  # noqa: E402

NEAR_TIE = 1e-4
_STATE = {}


def setUpModule():
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(min(4, torch.get_num_threads()))


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])


def load_lab():
    spec = importlib.util.spec_from_file_location("pytorch_35_generate", CODE / "35_generate.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    return lab


class SamplingContracts(unittest.TestCase):
    def test_argument_validation(self):
        logits = torch.zeros(1, 8)
        for kwargs in ({"top_p": 0}, {"top_p": 1.5}, {"top_k": 0}, {"temperature": -0.5},
                       {"temperature": float("nan")}, {"top_k": 2.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sample_next(logits, **kwargs)
        with self.assertRaises(ValueError):
            sample_next(torch.zeros(8))

    def test_filters(self):
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.06, 0.04]]))
        kept = lambda t: torch.isfinite(t).nonzero()[:, 1].tolist()  # noqa: E731
        self.assertEqual(kept(top_k_filter(logits, 2)), [0, 1])
        self.assertEqual(kept(top_k_filter(logits, 99)), [0, 1, 2, 3, 4])
        self.assertEqual(kept(top_p_filter(logits, 0.45)), [0])
        self.assertEqual(kept(top_p_filter(logits, 0.85)), [0, 1, 2])
        self.assertEqual(kept(top_p_filter(logits, 1e-9)), [0])          # top token always kept
        self.assertEqual(kept(top_p_filter(logits[:, [3, 0, 4, 2, 1]], 0.85)), [1, 3, 4])  # unsorted input

    def test_temperature_zero_and_top_k_one_are_greedy(self):
        gen = torch.Generator().manual_seed(0)
        logits = torch.randn(4, 512, generator=gen)
        greedy = logits.argmax(-1, keepdim=True)
        self.assertTrue(torch.equal(sample_next(logits, temperature=0.0), greedy))
        self.assertTrue(torch.equal(sample_next(logits, top_k=1, generator=gen), greedy))
        self.assertEqual(sample_next(logits, generator=gen).shape, (4, 1))


class GenerationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok, _ = load_pretrained()
        cls.idx = torch.tensor([cls.tok.encode("Sherlock Holmes")])

    def test_cached_matches_uncached_for_64_greedy_steps(self):
        r = verify_cache(self.model, self.idx, 64)
        self.assertEqual(r["steps"], 64)
        self.assertLess(r["max_abs_logit_diff"], 1e-4)
        if not r["tokens_equal"]:
            print(f"\n[35] cached/uncached diverged at step {r['first_divergence']} "
                  f"(margin {r['margin_at_divergence']:.2e})", file=sys.stderr)
            self.assertLess(r["margin_at_divergence"], NEAR_TIE,
                            "tokens diverged without a near-tie: the cache is wrong")
        # the free-running decoders agree up to the first divergence (all 64 when none)
        n = 64 if r["tokens_equal"] else r["first_divergence"]
        a = generate(self.model, self.idx, n, temperature=0.0, use_cache=True)
        b = generate(self.model, self.idx, n, temperature=0.0, use_cache=False)
        self.assertTrue(torch.equal(a, b))

    def test_uncached_accepts_any_callable(self):
        wrapped = lambda idx: (self.model(idx)[0], None)  # noqa: E731  (module 38 contract)
        a = generate(wrapped, self.idx, 8, temperature=0.0, use_cache=False, block_size=128)
        b = generate(self.model, self.idx, 8, temperature=0.0, use_cache=False)
        self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            generate(wrapped, self.idx, 2, use_cache=True)

    def test_seeded_sampling(self):
        def sample(seed, use_cache=True):
            g = torch.Generator().manual_seed(seed)
            return generate(self.model, self.idx, 32, temperature=1.0, top_k=40, top_p=0.95,
                            generator=g, use_cache=use_cache)
        state = torch.get_rng_state()
        first = sample(1)
        self.assertTrue(torch.equal(first, sample(1)))
        self.assertFalse(torch.equal(first, sample(2)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()), "global RNG must be untouched")

    def test_context_limit(self):
        prompt = torch.tensor([self.tok.encode(" the" * 200)[:120]])
        out, info = generate(self.model, prompt, 20, temperature=0.0, return_info=True)
        self.assertEqual((info["stop_reason"], out.shape[1]), ("context_limit", 129))
        with self.assertRaisesRegex(ContextLimitError, "block_size"):
            generate(self.model, prompt, 20, temperature=0.0, on_context_limit="raise")
        slid = generate(self.model, prompt, 20, temperature=0.0, use_cache=False)
        self.assertEqual(slid.shape[1], 140)                       # uncached slides the window
        if not torch.equal(slid[:, :129], out):                    # same 129 tokens unless a near-tie
            k = int((slid[0, 120:129] != out[0, 120:]).nonzero()[0])
            self.assertLess(float(info["margins"][0, k]), NEAR_TIE)

    def test_eos_stops_rows(self):
        greedy = generate(self.model, self.idx, 6, temperature=0.0)
        eos = int(greedy[0, -3])                                   # a token greedy emits
        first = greedy[0, self.idx.size(1):].tolist().index(eos)
        out, info = generate(self.model, self.idx, 6, temperature=0.0, eos_id=eos, return_info=True)
        self.assertEqual((info["stop_reason"], out.shape[1]), ("eos", self.idx.size(1) + first + 1))
        self.assertEqual(int(out[0, -1]), eos)
        two = torch.cat([self.idx, torch.tensor([self.tok.encode("My dear Watson, said he")[:5]])])
        out = generate(self.model, two, 12, temperature=0.0, eos_id=eos)
        self.assertIn(eos, out[0, 5:].tolist())                   # row 0 finishes as above
        for row in out[:, 5:].tolist():
            if eos in row:                                         # finished rows emit only eos
                self.assertTrue(all(t == eos for t in row[row.index(eos):]))

    def test_left_padded_batch_equals_single_prompts(self):
        prompts = ["Holmes", "It was a dark night and", "My dear Watson, said he, I"]
        lengths = {len(self.tok.encode(p)) for p in prompts}
        self.assertEqual(len(lengths), 3)
        for use_cache in (True, False):
            batched, info = self._batch_ids(prompts, use_cache)
            for row, prompt in enumerate(prompts):
                single, s_info = self._batch_ids([prompt], use_cache)
                if batched[row] != single[0]:
                    k = next(i for i, (a, b) in enumerate(zip(batched[row], single[0])) if a != b)
                    margin = float(s_info["margins"][0, k])
                    print(f"\n[35] batch row {row} diverged at step {k} (margin {margin:.2e})", file=sys.stderr)
                    self.assertLess(margin, NEAR_TIE, "padding changed a decision that was not a near-tie")

    def _batch_ids(self, prompts, use_cache):
        pad = self.tok.eot_id
        enc = [self.tok.encode(p) for p in prompts]
        T = max(map(len, enc))
        idx = torch.tensor([[pad] * (T - len(e)) + e for e in enc])
        mask = torch.tensor([[0] * (T - len(e)) + [1] * len(e) for e in enc])
        out, info = generate(self.model, idx, 24, temperature=0.0, use_cache=use_cache,
                             attention_mask=mask, return_info=True)
        texts = generate_batch(self.model, self.tok, prompts, 24, temperature=0.0, use_cache=use_cache,
                               eos_id=None, return_ids=True)
        self.assertEqual(texts, out[:, T:].tolist())
        return out[:, T:].tolist(), info

    def test_stream_decode(self):
        text = "“I can glanced,” said he — ‘well’. Café."
        ids = self.tok.encode(text)
        self.assertTrue(any(b"\xe2\x80" == self.tok.token_bytes(i) for i in ids))  # a split character
        pieces = list(stream_decode(self.tok, ids))
        self.assertEqual("".join(pieces), self.tok.decode(ids))
        self.assertEqual("".join(pieces), text)
        self.assertNotIn("�", "".join(pieces))
        self.assertIn("�", "".join(self.tok.decode([i]) for i in ids))  # naive per-token decode breaks
        sampled = generate(self.model, self.idx, 48, temperature=1.0,
                           generator=torch.Generator().manual_seed(35))[0].tolist()
        self.assertEqual("".join(stream_decode(self.tok, sampled)), self.tok.decode(sampled))


class LabContracts(unittest.TestCase):
    def test_lab_main(self):
        lab = load_lab()
        r = lab.main(["--max-new-tokens", "16", "--verify", "--batch-prompts", "Holmes|It was a dark night and"])
        self.assertTrue({"prompt", "completion", "verify", "benchmark", "batch_consistent"} <= set(r))
        self.assertIsNone(r["benchmark"])
        self.assertLess(r["verify"]["max_abs_logit_diff"], 1e-4)
        self.assertEqual(r["verify"]["steps"], 64)
        self.assertIsInstance(r["completion"], str)
        self.assertIsInstance(r["batch_consistent"], bool)


if __name__ == "__main__":
    unittest.main()
