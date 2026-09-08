"""CPU numerical contracts for the Transformer course and its reference decoder."""

import contextlib
import importlib.util
import io
from pathlib import Path
import re
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "content/courses/transformers"
spec = importlib.util.spec_from_file_location("course_decoder", COURSE / "code/modern_decoder.py")
decoder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = decoder
spec.loader.exec_module(decoder)
torch.set_num_threads(1)


def tiny_config(**overrides):
    values = dict(vocab_size=23, d_model=16, n_layers=2, n_heads=4,
                  n_kv_heads=2, d_head=4, d_ff=24, n_experts=3,
                  n_experts_active=1, n_shared_experts=1, d_expert=12,
                  first_k_dense=1, max_seq_len=32)
    return decoder.Config(**(values | overrides))


class DecoderContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_cache_matches_every_position_for_multiple_partitions(self):
        for experts in (0, 3):
            model = decoder.ModernDecoder(tiny_config(n_experts=experts)).eval()
            for batch in (1, 2):
                ids = torch.randint(23, (batch, 9))
                with torch.no_grad():
                    expected, _ = model(ids)
                    for partition in ([9], [1] * 9, [3, 2, 4], [7, 2]):
                        cache = [decoder.KVCache() for _ in model.blocks]
                        offset, chunks = 0, []
                        for length in partition:
                            actual, _ = model(ids[:, offset:offset + length], cache, offset)
                            chunks.append(actual)
                            offset += length
                        torch.testing.assert_close(torch.cat(chunks, 1), expected, atol=2e-6, rtol=2e-5)
                        self.assertTrue(all(c.length == 9 for c in cache))
                        self.assertTrue(all(c.k.shape[1] == 2 for c in cache))

    def test_dense_report_and_real_optimizer_step(self):
        cfg = tiny_config(n_experts=0)
        model = decoder.ModernDecoder(cfg)
        self.assertTrue(all(not block.is_moe for block in model.blocks))
        with contextlib.redirect_stdout(io.StringIO()):
            decoder._report(cfg, model)
        ids = torch.randint(23, (2, 8))
        before = model.embed.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        logits, routers = model(ids)
        self.assertEqual(routers, [])
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 23), ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.embed.weight))

    def test_top_one_router_has_task_gradient(self):
        moe = decoder.MoE(tiny_config())
        out, _ = moe(torch.randn(2, 3, 16))
        out.square().sum().backward()
        self.assertGreater(moe.gate.weight.grad.abs().sum().item(), 0)

    def test_uniform_balancing_is_independent_of_top_k(self):
        logits = torch.zeros(3, 3)
        for k in (1, 2, 3):
            indices = torch.stack([(torch.arange(k) + i) % 3 for i in range(3)])
            loss = decoder.load_balancing_loss(logits, indices, 3)
            self.assertAlmostEqual(loss.item(), 0.01, places=7)

    def test_generation_contracts(self):
        model = decoder.ModernDecoder(tiny_config()).eval()
        prompt = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(model.generate(prompt, 0), prompt)
        self.assertEqual(model.generate(prompt, 4, 0).shape, (1, 7))
        eos = model(prompt)[0][:, -1].argmax(-1).item()
        self.assertEqual(model.generate(prompt, 4, 0, eos).shape, (1, 4))
        for kwargs in ({"max_new_tokens": -1}, {"temperature": -1},
                       {"temperature": float("nan")}, {"max_new_tokens": 30}):
            with self.assertRaises(ValueError):
                model.generate(prompt, **kwargs)
        with self.assertRaises(ValueError):
            model.generate(prompt[:, :0])
        with self.assertRaises(ValueError):
            model(prompt, [decoder.KVCache()], 0)
        with self.assertRaises(ValueError):
            model(prompt, [decoder.KVCache() for _ in model.blocks], 1)

    def test_config_rejects_invalid_geometry(self):
        for kwargs in ({"n_heads": 3}, {"d_head": 3}, {"n_layers": 0},
                       {"n_experts_active": 4}, {"first_k_dense": 3}):
            with self.assertRaises(ValueError):
                tiny_config(**kwargs)


class PublishedTeachingChecks(unittest.TestCase):
    def test_actual_quantizer_and_sampler_excerpts(self):
        text = (COURSE / "14-inference-optimizations.md").read_text()
        namespace = {"torch": torch, "F": torch.nn.functional}
        for code in re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S):
            if "def quantize_int8(" in code or "def sample(" in code:
                exec(compile(code, "inference-excerpts", "exec"), namespace)
        q, scale = namespace["quantize_int8"](torch.zeros(2, 3))
        torch.testing.assert_close(namespace["dequantize"](q, scale), torch.zeros(2, 3))
        sample = namespace["sample"]
        logits = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
        for temperature in (0, 0.8, 1e-300):
            self.assertEqual(sample(logits, temperature=temperature).shape, (2, 1))
        for kwargs in ({"top_p": 0}, {"top_p": 1.1}, {"temperature": -1}):
            with self.assertRaises(ValueError):
                sample(logits, **kwargs)

    def test_independent_numerical_fences(self):
        count = 0
        for path in sorted(COURSE.glob("*.md")):
            for code in re.findall(r"^```python transformer-check\n(.*?)^```\s*$", path.read_text(), re.M | re.S):
                count += 1
                with self.subTest(page=path.name, check=count):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(code, str(path), "exec"), {"__name__": "__main__"})
        self.assertGreaterEqual(count, 10, "retain the independent worked checks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
