"""Token pipeline contracts for PyTorch course module 31 (llm/data.py and its lab)."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
# Documented split sizes (module 31 text, measured with the committed tokenizer).
DOCUMENTED_TOKENS = {"train": 217_572, "val": 23_848, "test": 25_216}
STORIES = {"train": 10, "val": 1, "test": 1}

lab = data = None
RESULT = SPLITS = TOK = None
_STATE = {}


def setUpModule():
    global lab, data, RESULT, SPLITS, TOK
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["deterministic"] = torch.are_deterministic_algorithms_enabled()
    _STATE["path"] = list(sys.path)
    spec = importlib.util.spec_from_file_location("pytorch_31_token_pipeline",
                                                  CODE / "31_token_pipeline.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.data as data_mod
    from llm.tokenizer import ByteBPE
    data = data_mod
    RESULT = lab.main([])
    TOK = ByteBPE.load(data.TOKENIZER_PATH)
    SPLITS = data.encode_all_splits(TOK)


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    sys.path[:] = _STATE["path"]


class SplitContracts(unittest.TestCase):
    def test_token_counts_and_eot_per_split(self):
        for split, expected in DOCUMENTED_TOKENS.items():
            t = SPLITS[split]
            self.assertEqual(t.dtype, torch.int64)
            self.assertEqual(t.dim(), 1)
            self.assertLessEqual(abs(t.numel() - expected) / expected, 0.02, split)
            self.assertEqual(int((t == TOK.eot_id).sum()), STORIES[split] - 1, split)
            self.assertEqual(RESULT["n_tokens"][split], t.numel())
            self.assertEqual(RESULT["uint16_bytes"][split], 2 * t.numel())

    def test_encode_all_splits_memoised_but_returns_private_copies(self):
        a = data.encode_all_splits(TOK)
        a["val"][0] = -1
        b = data.encode_all_splits(TOK)
        self.assertTrue(torch.equal(b["val"], SPLITS["val"]))

    def test_uint16_round_trip_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.bin"
            n = data.save_uint16(SPLITS["val"], path)
            self.assertEqual(n, path.stat().st_size)
            back = data.load_uint16(path)
            self.assertEqual(back.dtype, torch.int64)
            self.assertTrue(torch.equal(back, SPLITS["val"]))
            edge = torch.tensor([0, 511, 65_535])
            data.save_uint16(edge, path)
            self.assertTrue(torch.equal(data.load_uint16(path), edge))
            with self.assertRaises(ValueError):
                data.save_uint16(torch.tensor([1, 70_000]), path)
            with self.assertRaises(ValueError):
                data.save_uint16(torch.tensor([-1]), path)


class WindowContracts(unittest.TestCase):
    def test_train_batches_shift_dtype_shape(self):
        loader = data.make_train_loader(SPLITS["train"], 128, 32, seed=1337)
        for _ in range(3):
            x, y = next(loader)
            self.assertEqual((x.dtype, y.dtype), (torch.int64, torch.int64))
            self.assertEqual(tuple(x.shape), (32, 128))
            self.assertEqual(tuple(y.shape), (32, 128))
            self.assertTrue(x.is_contiguous() and y.is_contiguous())
            self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        g = torch.Generator().manual_seed(0)
        x, y = data.get_batch(SPLITS["val"], 16, 4, g)
        self.assertEqual(tuple(x.shape), (4, 16))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_sequential_eval_scores_every_target_once(self):
        val = SPLITS["val"]
        for stride in (128, 64, 32):
            ds = data.SequentialEvalDataset(val, 128, stride)
            self.assertEqual(ds.n_scored, val.numel() - 1)
            hits = torch.zeros(val.numel(), dtype=torch.int64)
            pos = torch.arange(val.numel())
            for i, b in enumerate(ds.starts):
                x, y, mask = ds[i]
                self.assertEqual(tuple(x.shape), (128,))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertTrue(torch.equal(y[:-1], x[1:]))
                self.assertTrue(torch.equal(y, val[b + 1:b + 129]))
                if i > 0:  # after window 0 a scored target at position j has j + 1 >= T - stride + 1 context
                    self.assertGreaterEqual(int(mask.long().argmax()), 128 - stride)
                hits[pos[b + 1:b + 129][mask]] += 1
            self.assertEqual(int(hits[0]), 0)
            self.assertTrue(bool((hits[1:] == 1).all()), stride)
        for entry in RESULT["eval_windows_by_stride"]:
            self.assertTrue(entry["each_target_once"])
            self.assertEqual(entry["n_scored"], val.numel() - 1)

    def test_sequential_eval_short_and_invalid_inputs(self):
        ds = data.SequentialEvalDataset(torch.arange(10), 128, 64)
        self.assertEqual((len(ds), ds.n_scored), (1, 9))
        with self.assertRaises(ValueError):
            data.SequentialEvalDataset(SPLITS["val"], 128, 0)
        with self.assertRaises(ValueError):
            data.SequentialEvalDataset(SPLITS["val"], 128, 129)

    def test_random_window_dataset_epochs(self):
        ds = data.RandomWindowDataset(SPLITS["train"], 64, num_samples=16, seed=5)
        self.assertEqual(len(ds), 16)
        x, y = ds[3]
        self.assertTrue(torch.equal(y[:-1], x[1:]))
        e0 = torch.stack([ds[i][0] for i in range(16)])
        ds.set_epoch(1)
        e1 = torch.stack([ds[i][0] for i in range(16)])
        self.assertFalse(torch.equal(e0, e1))
        ds.set_epoch(0)
        self.assertTrue(torch.equal(e0, torch.stack([ds[i][0] for i in range(16)])))
        twin = data.RandomWindowDataset(SPLITS["train"], 64, num_samples=16, seed=5)
        twin.set_epoch(1)
        self.assertTrue(torch.equal(e1, torch.stack([twin[i][0] for i in range(16)])))


class ResumeContracts(unittest.TestCase):
    def test_resume_at_step_7_matches_uninterrupted(self):
        train = SPLITS["train"]
        straight = data.make_train_loader(train, 128, 32, seed=1337)
        ref = [next(straight) for _ in range(12)]
        state = data.make_train_loader(train, 128, 32, seed=1337, start_step=7).state_dict()
        self.assertEqual(state, {"step": 7, "seed": 1337})
        resumed = data.make_train_loader(train, 128, 32, seed=0)
        resumed.load_state_dict(state)
        for k in range(7, 12):
            x, y = next(resumed)
            self.assertTrue(torch.equal(x, ref[k][0]) and torch.equal(y, ref[k][1]), k)
        self.assertEqual(resumed.state_dict(), straight.state_dict())
        self.assertTrue(RESULT["resume_ok"])

    def test_different_seeds_differ_and_global_rng_untouched(self):
        train = SPLITS["train"]
        before = torch.get_rng_state()
        a = next(data.make_train_loader(train, 128, 32, seed=1337))[0]
        b = next(data.make_train_loader(train, 128, 32, seed=1338))[0]
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class AuditNumbers(unittest.TestCase):
    def test_boundary_fraction_is_small_but_nonzero(self):
        frac = RESULT["boundary_fraction"]
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 0.02)  # 9 boundaries, each inside about T + 1 of ~217k windows
        tiny = torch.tensor([1, 2, 511, 3, 4, 5, 6])
        # windows of 3 tokens: [1,2,511] [2,511,3] [511,3,4] [3,4,5] [4,5,6] -> 3 of 5
        self.assertAlmostEqual(data.boundary_fraction(tiny, 2, 511), 0.6)
        self.assertIsNone(RESULT["bench"])


if __name__ == "__main__":
    unittest.main()
