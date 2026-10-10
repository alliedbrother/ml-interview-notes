"""Contracts for the module 19 lab: MLP, CNN, char-GRU and CNN transfer on generated data."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"

_saved = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("19_classic_models")


def setUpModule():
    _saved["threads"] = torch.get_num_threads()
    _saved["dtype"] = torch.get_default_dtype()
    _saved["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_saved["threads"])
    torch.set_default_dtype(_saved["dtype"])
    torch.use_deterministic_algorithms(_saved["det"])


def image_hashes(x):
    return {hashlib.sha256(img.numpy().tobytes()).hexdigest() for img in x}


class ClassicModelsLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.json_path = Path(cls.tmp.name) / "out.json"
        cls.result = lab.main(["--seed", "0", "--which", "all", "--json-out", str(cls.json_path)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_json_contract(self):
        on_disk = json.loads(self.json_path.read_text())
        self.assertEqual(set(on_disk), {"mlp", "cnn", "gru", "transfer"})
        self.assertLessEqual({"val_acc"}, set(on_disk["mlp"]))
        self.assertLessEqual({"val_acc", "test_acc", "params"}, set(on_disk["cnn"]))
        self.assertLessEqual({"val_loss", "unigram_loss", "sample"}, set(on_disk["gru"]))
        self.assertLessEqual(
            {"probe_acc", "finetune_acc", "scratch_acc", "trainable_params"}, set(on_disk["transfer"])
        )
        self.assertEqual(on_disk["cnn"]["params"], self.result["cnn"]["params"])

    def test_mlp_learns_spirals(self):
        self.assertGreater(self.result["mlp"]["val_acc"], 0.9)

    def test_cnn_accuracy_and_closed_form_params(self):
        cnn = self.result["cnn"]
        self.assertGreater(cnn["val_acc"], 0.85)
        self.assertGreater(cnn["test_acc"], 0.85)
        # conv weights (no bias) + BN affine + linear head, written out independently
        closed = (1 * 16 + 16 * 16 + 16 * 32 + 32 * 32) * 9 + 2 * (16 + 16 + 32 + 32) + (32 * 4 + 4)
        self.assertEqual(closed, 16_596)
        self.assertEqual(cnn["params"], closed)
        self.assertEqual(lab.cnn_param_formula(), closed)

    def test_gru_beats_unigram_and_samples(self):
        gru = self.result["gru"]
        self.assertLess(gru["val_loss"], gru["unigram_loss"] - 0.3)
        self.assertTrue(gru["sample"].startswith("ada "))
        self.assertEqual(len(gru["sample"]), 4 + 100)

    def test_transfer_trainable_counts_and_quality(self):
        tr = self.result["transfer"]
        counts = tr["trainable_params"]
        self.assertEqual(counts["probe"], 32 * 4 + 4)
        self.assertEqual(counts["probe"], tr["head_params"])
        self.assertEqual(counts["finetune"], 16_596)
        self.assertEqual(counts["scratch"], 16_596)
        self.assertTrue(tr["backbone_unchanged_after_probe"])
        self.assertGreater(tr["source_acc"], 0.85)
        self.assertGreater(tr["probe_acc"], 0.6)
        self.assertGreater(tr["finetune_acc"], 0.6)
        # Over six local seeds the better transfer strategy beat scratch by >= 0.045;
        # we assert only the direction, not a margin.
        self.assertGreater(max(tr["probe_acc"], tr["finetune_acc"]), tr["scratch_acc"])

    def test_procedural_data_deterministic_and_splits_disjoint(self):
        a, ya = lab.make_shapes(64, 5)
        b, yb = lab.make_shapes(64, 5)
        self.assertTrue(torch.equal(a, b) and torch.equal(ya, yb))
        self.assertEqual(tuple(a.shape), (64, 1, 16, 16))
        self.assertEqual(a.dtype, torch.float32)
        s_tr, s_va, s_te = lab.split_seeds(0)
        tr, va, te = (image_hashes(lab.make_shapes(200, s)[0]) for s in (s_tr, s_va, s_te))
        self.assertEqual(len(tr), 200)
        self.assertFalse(tr & va or tr & te or va & te)

    def test_augmentation_preserves_shape_and_flip_is_involution(self):
        x, _ = lab.make_shapes(32, 9)
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        out1, out2 = lab.augment(x, g1), lab.augment(x, g2)
        self.assertEqual(out1.shape, x.shape)
        self.assertEqual(out1.dtype, x.dtype)
        self.assertTrue(torch.equal(out1, out2))
        flipped = lab.random_hflip(x, torch.Generator().manual_seed(0), p=1.0)
        self.assertTrue(torch.equal(lab.random_hflip(flipped, torch.Generator().manual_seed(0), p=1.0), x))
        shifted = lab.random_shift(x, torch.Generator().manual_seed(0), pad=0)
        self.assertTrue(torch.equal(shifted, x))


if __name__ == "__main__":
    unittest.main()
