"""Contracts for the module 16 lab: atomic checkpoints and bitwise-exact resume."""

import importlib.util
import json
from pathlib import Path
import random
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


lab = load("16_train_resume")
KILL_AT, STEPS = 50, 120


def setUpModule():
    _saved["threads"] = torch.get_num_threads()
    _saved["dtype"] = torch.get_default_dtype()
    _saved["det"] = torch.are_deterministic_algorithms_enabled()
    _saved["rng"] = torch.get_rng_state()
    _saved["py_rng"] = random.getstate()
    import numpy as np

    _saved["np_rng"] = np.random.get_state()


def tearDownModule():
    import numpy as np

    torch.set_num_threads(_saved["threads"])
    torch.set_default_dtype(_saved["dtype"])
    torch.use_deterministic_algorithms(_saved["det"])
    torch.set_rng_state(_saved["rng"])
    random.setstate(_saved["py_rng"])
    np.random.set_state(_saved["np_rng"])


class ExactResumeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "run"
        json_path = Path(cls.tmp.name) / "result.json"
        cls.result = lab.main([
            "--seed", "0", "--steps", str(STEPS), "--kill-at", str(KILL_AT), "--accum", "2",
            "--output-dir", str(cls.out), "--json-out", str(json_path),
        ])
        cls.on_disk = json.loads(json_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_resume_is_bitwise_exact(self):
        self.assertTrue(self.result["bitwise_equal_params"])
        self.assertEqual(self.result["max_param_diff_resume"], 0.0)

    def test_negative_control_without_rng_diverges(self):
        self.assertGreater(self.result["max_param_diff_no_rng"], 0.0)

    def test_loss_curves_identical_including_after_resume(self):
        a, b = self.result["loss_curve_a"], self.result["loss_curve_b"]
        self.assertEqual(len(a), STEPS)
        self.assertEqual(len(b), STEPS)
        self.assertEqual(a[KILL_AT:], b[KILL_AT:])
        self.assertEqual(a, b)
        self.assertLess(sum(a[-10:]) / 10, sum(a[:10]) / 10)  # it actually trained

    def test_checkpoint_loads_weights_only_and_matches_manifest(self):
        path = self.out / "last.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("model", "optimizer", "scheduler", "trainer", "rng", "config", "meta"):
            self.assertIn(key, ckpt)
        self.assertEqual(ckpt["trainer"]["step"], KILL_AT)
        self.assertEqual(ckpt["trainer"]["micro"], KILL_AT * 2)
        self.assertEqual(ckpt["rng"]["torch"].dtype, torch.uint8)
        manifest = json.loads((self.out / "last.json").read_text())
        self.assertEqual(manifest["sha256"], lab.sha256(path))
        self.assertEqual(manifest["bytes"], self.result["checkpoint_bytes"])

    def test_atomic_write_leaves_no_tmp_and_survives_failed_save(self):
        self.assertEqual(self.result["checkpoint_files"], ["last.json", "last.pt"])
        self.assertFalse(any(p.suffix == ".tmp" for p in self.out.iterdir()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            lab.atomic_save({"w": torch.ones(3)}, path)
            before = path.read_bytes()
            with self.assertRaises(Exception):
                lab.atomic_save({"w": torch.zeros(3), "bad": lambda: None}, path)  # unpicklable
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["ckpt.pt"])

    def test_corrupted_checkpoint_is_rejected_by_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "last.pt"
            data = bytearray((self.out / "last.pt").read_bytes())
            data[-100] ^= 0xFF
            dst.write_bytes(bytes(data))
            (Path(tmp) / "last.json").write_text((self.out / "last.json").read_text())
            train_ds, _ = lab.make_data(0)
            model, opt, sched, _, _ = lab.build(0, STEPS, train_ds)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                lab.load_checkpoint(dst, model, opt, sched)

    def test_quality_and_json_contract(self):
        self.assertGreater(self.result["final_val_acc"], 0.8)
        self.assertGreaterEqual(self.result["best_val_acc"], self.result["final_val_acc"] - 0.05)
        self.assertEqual(self.result["best_val_acc"], self.result["best_val_acc_resumed"])
        self.assertEqual(set(self.on_disk), set(self.result))

    def test_sampler_resumes_without_replay(self):
        sampler = lab.StepIndexedSampler(64, seed=3, batch_size=8)
        sampler.set_position(epoch=2, start_batch=0)
        full = list(sampler)
        sampler.set_position(epoch=2, start_batch=5)
        self.assertEqual(list(sampler), full[40:])
        sampler.set_position(epoch=3, start_batch=0)
        self.assertNotEqual(list(sampler), full)


if __name__ == "__main__":
    unittest.main()
