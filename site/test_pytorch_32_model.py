"""Contracts for the capstone GPT (module 32): llm/model.py and code/32_gpt_model.py."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from llm.model import GPT, GPTConfig, KVCache, causal_mask  # noqa: E402

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
    spec = importlib.util.spec_from_file_location("pytorch_32_gpt_model", CODE / "32_gpt_model.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    return lab


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(32)
        cls.cfg = GPTConfig()
        cls.model = GPT(cls.cfg).eval()
        gen = torch.Generator().manual_seed(32)
        cls.ids = torch.randint(0, cls.cfg.vocab_size, (4, 33), generator=gen)

    def test_parameter_count_and_table(self):
        self.assertEqual(self.model.num_params(), 875_264)
        self.assertEqual(self.model.num_params(non_embedding=True), 875_264 - 16_384)
        table = self.model.param_table()
        self.assertEqual(sum(row["numel"] for row in table), 875_264)
        head = [row for row in table if row["name"] == "lm_head"][0]
        self.assertEqual((head["numel"], head["tied_to"]), (0, "wte.weight"))

    def test_tied_weight_identity_and_state_dict(self):
        model = self.model
        self.assertIs(model.lm_head.weight, model.wte.weight)
        state = model.state_dict()
        self.assertEqual(state["lm_head.weight"].data_ptr(), state["wte.weight"].data_ptr())
        self.assertEqual(len(state), len(list(model.parameters())) + 1)

    def test_initial_loss_is_ln_vocab(self):
        x, y = self.ids[:, :-1], self.ids[:, 1:]
        with torch.no_grad():
            _, loss = self.model(x, y)
        self.assertLess(abs(loss.item() - math.log(512)), 0.1)
        ignored = y.clone()
        ignored[:, :16] = -100
        with torch.no_grad():
            _, partial = self.model(x, ignored)
            logits, _ = self.model(x)
        reference = torch.nn.functional.cross_entropy(logits[:, 16:].reshape(-1, 512), y[:, 16:].reshape(-1))
        torch.testing.assert_close(partial, reference, atol=1e-6, rtol=0)

    def test_sdpa_matches_manual(self):
        with torch.no_grad():
            a, _ = self.model(self.ids, attn_impl="sdpa")
            b, _ = self.model(self.ids, attn_impl="manual")
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
        with self.assertRaises(ValueError):
            self.model(self.ids, attn_impl="flash")

    def test_causality(self):
        changed = self.ids.clone()
        changed[:, 10] = (changed[:, 10] + 1) % self.cfg.vocab_size
        with torch.no_grad():
            a, _ = self.model(self.ids)
            b, _ = self.model(changed)
        torch.testing.assert_close(a[:, :10], b[:, :10], atol=1e-6, rtol=0)
        self.assertGreater((a[:, 10:] - b[:, 10:]).abs().max().item(), 1e-4)

    def test_batch_independence(self):
        changed = self.ids.clone()
        changed[1] = torch.randint(0, self.cfg.vocab_size, (changed.size(1),),
                                   generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            a, _ = self.model(self.ids)
            b, _ = self.model(changed)
            alone, _ = self.model(self.ids[2:3])
        for row in (0, 2, 3):
            torch.testing.assert_close(a[row], b[row], atol=1e-6, rtol=0)
        torch.testing.assert_close(a[2:3], alone, atol=1e-5, rtol=0)

    def test_kv_cache_one_step_equivalence(self):
        idx = self.ids[:2, :13]
        with torch.no_grad():
            full, _ = self.model(idx)
            for impl in ("sdpa", "manual"):
                cache = KVCache(self.cfg, batch_size=2, max_len=self.cfg.block_size)
                prefix, _ = self.model(idx[:, :12], cache=cache, attn_impl=impl)
                step, _ = self.model(idx[:, 12:13], cache=cache, attn_impl=impl)
                self.assertEqual(cache.length, 13)
                torch.testing.assert_close(prefix, full[:, :12], atol=1e-5, rtol=0)
                torch.testing.assert_close(step[:, -1], full[:, -1], atol=1e-5, rtol=0)
        self.assertEqual(tuple(cache.k.shape), (4, 2, 4, 128, 32))
        self.assertEqual(cache.nbytes, 2 * 4 * 2 * 4 * 128 * 32 * 4)
        cache.reset()
        self.assertEqual(cache.length, 0)

    def test_left_padding_hooks_match_unpadded(self):
        # module 35 batches prompts with left padding via position_ids + key_mask
        prompt = self.ids[0, :7]
        padded = torch.cat([torch.zeros(3, dtype=torch.long), prompt])[None]
        key_mask = torch.tensor([[False] * 3 + [True] * 7])
        position_ids = (key_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            ref, _ = self.model(prompt[None])
            out, _ = self.model(padded, position_ids=position_ids, key_mask=key_mask)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[:, 3:], ref, atol=1e-5, rtol=0)

    def test_length_and_config_errors(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 129, dtype=torch.long))
        cache = KVCache(self.cfg, batch_size=1, max_len=16)
        with torch.no_grad():
            self.model(self.ids[:1, :12], cache=cache)
            with self.assertRaises(ValueError):
                self.model(self.ids[:1, :5], cache=cache)
        for bad in ({"d_model": 130}, {"n_head": 0}, {"dropout": 1.0}, {"vocab_size": 1.5}):
            with self.assertRaises(ValueError):
                GPTConfig(**bad)
        with self.assertRaises(ValueError):
            KVCache(self.cfg, batch_size=1, max_len=129)

    def test_causal_mask_offset(self):
        expected = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(causal_mask(2, 5, 3), expected))
        self.assertTrue(torch.equal(causal_mask(4, 4, 0), torch.ones(4, 4, dtype=torch.bool).tril()))

    def test_optimizer_groups(self):
        opt = self.model.configure_optimizer(weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95))
        decay, no_decay = opt.param_groups
        self.assertEqual(decay["weight_decay"], 0.1)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        seen = [id(p) for group in opt.param_groups for p in group["params"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in self.model.parameters()})
        self.assertTrue(all(p.dim() >= 2 for p in decay["params"]))
        names = {id(p): n for n, p in self.model.named_parameters()}
        for p in no_decay["params"]:
            self.assertTrue(names[id(p)].endswith(".bias") or ".ln_" in names[id(p)]
                            or names[id(p)].startswith("ln_f"))
        self.assertEqual(sum(p.numel() for g in opt.param_groups for p in g["params"]), 875_264)


class LabContracts(unittest.TestCase):
    def test_lab_audit(self):
        result = load_lab().main(["--overfit-steps", "40", "--init-batch", "random", "--seed", "1337"])
        self.assertEqual(result["params"], 875_264)
        self.assertEqual(sum(row["numel"] for row in result["param_table"]), 875_264)
        self.assertLess(abs(result["init_loss"] - math.log(512)), 0.1)
        self.assertLess(result["sdpa_manual_max_diff"], 1e-5)
        self.assertLess(result["overfit_final_loss"], 0.5)
        self.assertEqual(result["shape_trace"][-1], ["lm_head -> logits", [2, 16, 512]])
        # FlopCounterMode counts matmuls; the manual path exposes attention as bmm.
        formula = result["flops_6P"] + result["flops_attention_term"]
        self.assertLess(abs(result["flops_measured"] - formula) / formula, 0.25)
        self.assertEqual(result["flops_measured"] - result["flops_measured_sdpa"],
                         result["flops_attention_term"])


if __name__ == "__main__":
    unittest.main()
