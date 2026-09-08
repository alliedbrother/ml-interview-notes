"""CPU training, masking, serialization and BPE trace contracts for course labs."""

import importlib.util
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/transformers/code"


def module(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


bpe = module("train_bpe")
translation = module("train_tiny_translation")
torch.set_num_threads(1)


class BPEContracts(unittest.TestCase):
    def test_complete_frequency_trace_and_deterministic_ties(self):
        texts = ["low", "low", "low", "lower", "lowest", "newer", "newest"]
        tokenizer, trace = bpe.ByteBPE.train(texts)
        other, other_trace = bpe.ByteBPE.train(list(reversed(texts)))
        self.assertEqual(trace, other_trace)
        self.assertEqual(tokenizer.merges, other.merges)
        self.assertEqual([event["frequency"] for event in trace], [5, 5, 2, 2, 2, 2, 2])
        for event in trace:
            counts = {tuple(row["pair"]): row["count"] for row in event["pair_counts"]}
            expected = min(counts, key=lambda pair: (-counts[pair], pair))
            self.assertEqual(tuple(event["pair"]), expected)
            self.assertGreater(event["tokens_before"], event["tokens_after"])

    def test_overlapping_pairs_do_not_double_merge(self):
        tokenizer, trace = bpe.ByteBPE.train(["aaaa"], num_merges=1)
        self.assertEqual(trace[0]["frequency"], 3)
        self.assertEqual(trace[0]["tokens_before"] - trace[0]["tokens_after"], 2)
        self.assertEqual(tokenizer.encode("aaaa"), [256, 256])
        self.assertEqual(bpe.ByteBPE.train(["a", "a"])[1], [])

    def test_unicode_whitespace_empty_and_random_round_trips(self):
        tokenizer, _ = bpe.ByteBPE.train(["low low", "newest"])
        rng = random.Random(2)
        alphabet = "ab \n\t\u00e9\u4f60\U0001f30d\u0301"
        texts = ["", "<BOS>", "  low\n"] + ["".join(rng.choices(alphabet, k=n)) for n in range(70)]
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([195])

    def test_saved_ranks_and_invalid_contracts(self):
        tokenizer, _ = bpe.ByteBPE.train(["low", "lower"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(tokenizer.encode("lower"), bpe.ByteBPE.load(path).encode("lower"))
        for kwargs in ({"num_merges": -1}, {"min_frequency": 0}):
            with self.assertRaises(ValueError):
                bpe.ByteBPE.train(["x"], **kwargs)
        with self.assertRaises(ValueError):
            bpe.ByteBPE([(256, 0)])

    def test_library_byte_coverage_and_disk_round_trip(self):
        from tokenizers import Tokenizer
        tokenizer = bpe.train_library_tokenizer(["low", "low", "lower", "newest"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(str(path))
            restored = Tokenizer.from_file(str(path))
            for text in ("", "  low\n", "caf\u00e9 \U0001f30d", "<BOS>"):
                ids = tokenizer.encode(text).ids
                self.assertEqual(restored.encode(text).ids, ids)
                self.assertEqual(restored.decode(ids), text)


class TranslationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.optimizer, cls.sv, cls.tv, cls.results = translation.train()
        cls.tensors = translation.batch(translation.corpus()[0], cls.sv, cls.tv)

    def test_shift_padding_loss_and_disjoint_combinations(self):
        source, inputs, labels = self.tensors
        self.assertTrue(inputs[:, 0].eq(translation.BOS).all())
        torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
        self.assertTrue(source.eq(translation.PAD).any())
        self.assertTrue(labels.eq(translation.PAD).any())
        train, heldout = translation.corpus()
        self.assertFalse(set(train) & set(heldout))
        logits = self.model(source, inputs)
        valid = labels.ne(translation.PAD)
        torch.testing.assert_close(translation.loss_for(logits, labels), F.cross_entropy(logits[valid], labels[valid]))
        changed = logits.clone()
        changed[~valid] = 12345.
        torch.testing.assert_close(translation.loss_for(logits, labels), translation.loss_for(changed, labels))

    def test_future_and_padding_isolation(self):
        self.model.eval()
        source, inputs, _ = self.tensors
        with torch.no_grad():
            expected = self.model(source, inputs)
            future = inputs.clone()
            future[:, 2:] = 3
            torch.testing.assert_close(expected[:, :2], self.model(source, future)[:, :2], atol=2e-5, rtol=2e-5)
            padded = F.pad(source, (0, 3), value=translation.PAD)
            torch.testing.assert_close(expected, self.model(padded, inputs), atol=2e-5, rtol=2e-5)
            pad_embedding = self.model.source_embedding.weight[translation.PAD].clone()
            try:
                self.model.source_embedding.weight[translation.PAD].fill_(100.)
                torch.testing.assert_close(expected, self.model(source, inputs), atol=2e-5, rtol=2e-5)
            finally:
                self.model.source_embedding.weight[translation.PAD].copy_(pad_embedding)
            # The first item is a single-word phrase; compare only its valid queries.
            torch.testing.assert_close(expected[:1, :2], self.model(source[:1, :2], inputs[:1, :2]), atol=2e-5, rtol=2e-5)

    def test_real_training_and_honest_autoregressive_metrics(self):
        results = self.results
        self.assertLess(results["train"]["teacher_forced_loss"], results["initial_loss"] * .1)
        self.assertEqual(results["train"]["autoregressive_exact_match"], 1.)
        heldout = results["heldout_combinations"]
        self.assertEqual(len(heldout["rows"]), 3)
        self.assertEqual(heldout["autoregressive_exact_match"], sum(row["exact"] for row in heldout["rows"]) / 3)
        # Do not enforce a tiny held-out score or tune training against this split.
        self.assertTrue(all(row["ended"] for row in results["train"]["rows"]))

    def test_disk_reload_optimizer_update_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            translation.save_checkpoint(directory, self.model, self.optimizer, self.sv, self.tv, 250)
            left, left_opt, manifest = translation.load_checkpoint(directory)
            right, right_opt = copy.deepcopy((self.model, self.optimizer))
            left.eval()
            with torch.no_grad():
                torch.testing.assert_close(self.model(*self.tensors[:2]), left(*self.tensors[:2]))
            self.assertEqual(manifest["steps"], 250)
            for model, opt in ((left, left_opt), (right, right_opt)):
                translation.step(model, opt, self.tensors)
            for a, b in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(a, b)
            self.assertGreater(left_opt.state[next(iter(left.parameters()))]["step"].item(), 250)
            path = Path(directory) / "manifest.json"
            manifest["source_vocab"][0] = "wrong-pad"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                translation.load_checkpoint(directory)

    def test_generation_limits_and_unknown_words(self):
        source = self.tensors[0]
        self.assertEqual(self.model.generate(source, max_new_tokens=1).shape, (len(source), 1))
        for limit in (0, 17):
            with self.assertRaises(ValueError):
                self.model.generate(source, max_new_tokens=limit)
        for text in ("", "unseen"):
            with self.assertRaises(ValueError):
                translation.encode(text, self.sv)


if __name__ == "__main__":
    unittest.main()
