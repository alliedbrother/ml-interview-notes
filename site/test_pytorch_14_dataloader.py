"""Contracts for the module 14 lab: sharding, seeded workers, bucketing, memmap windows.

Worker processes are exercised by running the lab script in a subprocess (the
D.4 preference): spawn/forkserver workers re-import the script's module by path,
which a digit-named file loaded via importlib inside unittest cannot guarantee.
The in-process checks (memmap windows, collation, bucketing sampler) use
num_workers=0 and the module loaded via importlib. No throughput is asserted.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
SCRIPT = CODE / "14_dataloader.py"

_saved = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("14_dataloader")


def setUpModule():
    _saved["threads"] = torch.get_num_threads()
    _saved["dtype"] = torch.get_default_dtype()
    _saved["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_saved["threads"])
    torch.set_default_dtype(_saved["dtype"])
    torch.use_deterministic_algorithms(_saved["det"])


class WorkerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            log = os.path.join(tmp, "stderr.txt")
            # stderr goes to a file, not a pipe: the forkserver process inherits the
            # child's stderr and outlives it briefly, so reading a pipe to EOF would
            # wait for the server too (measured: about 2 s extra on macOS).
            with open(log, "w") as err:
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT), "--seed", "0", "--num-items", "2000",
                     "--workers", "2", "--json-out", out],
                    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err, timeout=120,
                )
            if proc.returncode != 0:
                raise AssertionError(Path(log).read_text()[-2000:])
            with open(out) as f:
                cls.result = json.load(f)

    def test_unsharded_iterable_duplicates(self):
        # Two workers each replay the whole stream: 2000 duplicates exactly.
        self.assertGreater(self.result["unsharded_duplicates"], 0)
        self.assertEqual(self.result["unsharded_duplicates"], 2000)

    def test_sharded_iterable_exactly_once(self):
        self.assertEqual(self.result["sharded_unique"], 2000)
        self.assertEqual(self.result["sharded_total"], 2000)

    def test_seeded_workers_reproduce_numpy_augmentation(self):
        self.assertIs(self.result["reproducible"], True)

    def test_bucketing_reduces_padding(self):
        self.assertLess(self.result["padding_frac_bucketed"], self.result["padding_frac_random"])

    def test_throughput_reported_not_asserted(self):
        self.assertEqual(set(self.result["throughput"]), {"0", "2"})
        self.assertTrue(all(v > 0 for v in self.result["throughput"].values()))


class InProcessContracts(unittest.TestCase):
    def test_memmap_windows_shift_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.bin")
            lab.write_tokens(path, 5000, 512, seed=1)
            self.assertEqual(os.path.getsize(path), 5000 * 2)  # uint16 on disk
            ds = lab.TokenWindows(path, block_size=16)
            self.assertEqual(len(ds), 5000 - 16)
            raw = np.fromfile(path, dtype=np.uint16)
            loader = DataLoader(ds, batch_size=8, shuffle=True,
                                generator=torch.Generator().manual_seed(0))
            x, y = next(iter(loader))
            self.assertEqual(x.shape, (8, 16))
            self.assertEqual(x.dtype, torch.int64)
            self.assertEqual(y.dtype, torch.int64)
            torch.testing.assert_close(y[:, :-1], x[:, 1:], rtol=0, atol=0)
            x0, y0 = ds[123]
            self.assertTrue(np.array_equal(x0.numpy(), raw[123:139]))
            self.assertTrue(np.array_equal(y0.numpy(), raw[124:140]))
            del ds, loader  # release the memmap before the directory is removed

    def test_pad_collate_contract(self):
        seqs = [torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([9, 10])]
        batch = lab.pad_collate(seqs)
        self.assertEqual(batch["input_ids"].tolist(), [[5, 6, 7], [8, 0, 0], [9, 10, 0]])
        self.assertEqual(batch["attention_mask"].sum(1).tolist(), [3, 1, 2])
        self.assertEqual(batch["labels"].tolist(), [[5, 6, 7], [8, -100, -100], [9, 10, -100]])

    def test_bucket_sampler_is_a_permutation(self):
        text = lab.VarLenText(300, seed=0)
        sampler = lab.BucketBatchSampler(text.lengths, 32, torch.Generator().manual_seed(0), pool_batches=4)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(300)))


if __name__ == "__main__":
    unittest.main()
