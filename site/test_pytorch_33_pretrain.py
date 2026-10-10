"""Pretraining contracts for PyTorch course module 33 (llm/train.py, 33_pretrain.py, the artifact).

Never runs the full 800-step training: it runs the 150-step smoke config and
validates the committed checkpoint code/data/gpt_sherlock.pt + gpt_sherlock.json.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
UNIGRAM_VAL = 5.22   # add-one unigram NLL on story XI (module 34)
BIGRAM_VAL = 3.67    # interpolated add-k bigram NLL on story XI (module 34)
MANIFEST_KEYS = {
    "format_version", "config", "train_config", "params", "steps", "seed", "num_threads",
    "tokens_seen", "tokenizer_file", "tokenizer_sha256", "corpus_sha256", "checkpoint_file",
    "checkpoint_sha256", "checkpoint_bytes", "final_train_loss", "final_val_loss", "history",
    "wall_seconds", "torch_version", "python_version", "machine", "created",
}

lab = train_mod = None
SMOKE = None
_STATE = {}


def setUpModule():
    global lab, train_mod, SMOKE
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  rng=torch.get_rng_state(), path=list(sys.path))
    spec = importlib.util.spec_from_file_location("pytorch_33_pretrain", CODE / "33_pretrain.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.train as tm
    train_mod = tm
    SMOKE = lab.main(["--smoke"])  # 150 steps, eval every 75, full model; sets 4 threads


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    torch.set_rng_state(_STATE["rng"])
    sys.path[:] = _STATE["path"]


def reduced_cfg(**overrides):
    from llm.model import GPTConfig
    model = dict(lab.REDUCED_MODEL, dropout=overrides.pop("dropout", 0.0))
    base = dict(steps=20, batch_size=8, eval_every=0, seed=1337, model=GPTConfig(**model))
    base.update(overrides)
    return train_mod.TrainConfig(**base)


class Schedule(unittest.TestCase):
    def test_lr_at_shape(self):
        cfg = train_mod.TrainConfig()
        self.assertEqual(cfg.warmup, 40)
        self.assertAlmostEqual(train_mod.lr_at(0, cfg), 3e-3 / 40)
        self.assertAlmostEqual(train_mod.lr_at(39, cfg), 3e-3)
        self.assertAlmostEqual(train_mod.lr_at(40, cfg), 3e-3)
        self.assertAlmostEqual(train_mod.lr_at(420, cfg), (3e-3 + 3e-4) / 2)  # cosine midpoint
        self.assertLess(abs(train_mod.lr_at(799, cfg) - 3e-4), 1e-6)
        lrs = [train_mod.lr_at(s, cfg) for s in range(40, 800)]
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine phase must not increase")


class SmokeRun(unittest.TestCase):
    def test_smoke_config_and_quality_bar(self):
        self.assertEqual(SMOKE["steps"], 150)
        self.assertEqual(SMOKE["params"], 875_264)
        self.assertEqual([h["step"] for h in SMOKE["history"]], [0, 75, 150])
        initial = SMOKE["history"][0]["val_loss"]
        self.assertLess(abs(initial - math.log(512)), 0.1)  # untrained model ~ uniform
        final = SMOKE["final_val_loss"]
        self.assertTrue(math.isfinite(final))
        self.assertLess(final, UNIGRAM_VAL - 0.5)
        self.assertLess(final, initial - 1.0)
        self.assertAlmostEqual(final, SMOKE["history"][-1]["val_loss"], places=6)
        for h in SMOKE["history"][1:]:
            self.assertTrue(math.isfinite(h["train_loss"]) and math.isfinite(h["grad_norm"]))

    def test_lr_history_matches_lr_at(self):
        cfg = train_mod.TrainConfig.from_dict(SMOKE["config"])
        self.assertEqual((cfg.steps, cfg.warmup), (150, 7))
        self.assertEqual(SMOKE["lrs"], [train_mod.lr_at(s, cfg) for s in range(150)])
        self.assertEqual(SMOKE["tokens_seen"], 150 * 32 * 128)


class ResumeAndAccumulation(unittest.TestCase):
    def test_exact_resume_restores_params_and_loader(self):
        out = lab.resume_check(seed=1337, steps=12)  # dropout 0.1: RNG restore is load-bearing
        self.assertLessEqual(out["max_abs_diff"], 1e-6)
        self.assertEqual(out["straight_loader"], {"step": 12, "seed": 1337})
        self.assertEqual(out["resumed_loader"], out["straight_loader"])
        sa, sb = out["straight"]["optimizer"].state_dict(), out["resumed"]["optimizer"].state_dict()
        for k in sa["state"]:
            torch.testing.assert_close(sa["state"][k]["exp_avg_sq"], sb["state"][k]["exp_avg_sq"],
                                       atol=1e-6, rtol=0)

    def test_resume_without_rng_restore_is_caught(self):
        with mock.patch.object(torch, "set_rng_state", lambda state: None):
            out = lab.resume_check(seed=1337, steps=12)
        self.assertGreater(out["max_abs_diff"], 1e-6)

    def test_checkpoint_is_weights_only_and_complete(self):
        cfg = reduced_cfg(steps=4)
        run = train_mod.train(cfg, max_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            train_mod.save_checkpoint(path, run["model"], run["optimizer"], run["step"],
                                      run["loader"], cfg, run["tokenizer_sha256"])
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["ckpt.pt"])  # no stray .tmp
            state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertTrue({"model", "optimizer", "step", "rng_cpu", "loader_state", "config",
                         "tokenizer_sha256", "torch_version"} <= set(state))
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["loader_state"], {"step": 2, "seed": 1337})
        self.assertEqual(train_mod.TrainConfig.from_dict(state["config"]), cfg)

    def test_accumulation_matches_big_batch(self):
        def first_grads(batch_size, accum):
            grads = {}

            def grab(step, record, model, optimizer):
                grads.update({n: p.grad.clone() for n, p in model.named_parameters()})

            cfg = reduced_cfg(batch_size=batch_size, accum=accum, grad_clip=0.0)
            run = train_mod.train(cfg, max_steps=1, callback=grab)
            return grads, run["last"]["loss"]

        g1, l1 = first_grads(32, 1)
        g2, l2 = first_grads(16, 2)
        self.assertAlmostEqual(l1, l2, places=5)
        for name in g1:
            torch.testing.assert_close(g2[name], g1[name], atol=1e-5, rtol=0, msg=name)


class CommittedArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok, cls.manifest = train_mod.load_pretrained()

    def test_manifest_and_hashes(self):
        from llm.corpus import sha256_file
        from llm.data import TOKENIZER_PATH
        m = self.manifest
        self.assertTrue(MANIFEST_KEYS <= set(m), MANIFEST_KEYS - set(m))
        self.assertEqual(m["checkpoint_sha256"], sha256_file(train_mod.ARTIFACT_PT))
        self.assertEqual(m["checkpoint_bytes"], train_mod.ARTIFACT_PT.stat().st_size)
        self.assertEqual(m["tokenizer_sha256"], sha256_file(TOKENIZER_PATH))
        self.assertEqual(m["tokenizer_sha256"],
                         "5e38bec016e902e7b098b5e5ebf566e5f66fb7e6ac3ee34858bc132f8e51c815")
        self.assertEqual((m["steps"], m["seed"], m["tokens_seen"]), (800, 1337, 800 * 4096))
        self.assertTrue(m["torch_version"].startswith("2.8"))
        self.assertLess(m["final_val_loss"], BIGRAM_VAL)

    def test_loads_as_expected(self):
        self.assertFalse(self.model.training)
        self.assertEqual(self.model.num_params(), 875_264)
        self.assertIs(self.model.lm_head.weight, self.model.wte.weight)  # tying survives loading
        self.assertEqual(self.tok.vocab_size, 512)
        state = torch.load(train_mod.ARTIFACT_PT, map_location="cpu", weights_only=True)
        self.assertEqual(set(state), set(self.model.state_dict()))
        self.assertTrue(all(t.dtype == torch.float32 for t in state.values()))

    def test_val_loss_reproduces_and_beats_bigram(self):
        from llm.data import encode_all_splits
        val = encode_all_splits(self.tok)["val"]
        loss = train_mod.evaluate_split(self.model, val, self.model.config.block_size)
        self.assertLess(abs(loss - self.manifest["final_val_loss"]), 0.05)
        self.assertLess(loss, BIGRAM_VAL)

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = Path(tmp) / "gpt_sherlock.json"
            m = dict(self.manifest, checkpoint_sha256="0" * 64)
            bad_json.write_text(json.dumps(m), encoding="utf-8")
            with mock.patch.object(train_mod, "ARTIFACT_JSON", bad_json):
                with self.assertRaisesRegex(ValueError, "sha256"):
                    train_mod.load_pretrained()


if __name__ == "__main__":
    unittest.main()
