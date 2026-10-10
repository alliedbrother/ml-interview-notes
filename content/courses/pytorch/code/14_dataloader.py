"""Build and stress a DataLoader on CPU (module 14 lab).

Four experiments on synthetic data, all inside a temporary directory:

(a) An IterableDataset over the integers 0..N-1 read with num_workers=2, once
    without sharding (every worker yields everything -> duplicates) and once
    sharded with get_worker_info() (every item exactly once).
(b) A memory-mapped uint16 token file read through a random-window map-style
    dataset with a NumPy-based masking augmentation. Each worker builds its own
    np.random.Generator in worker_init_fn from torch.initial_seed(). Two runs with
    the same loader generator seed and num_workers=2 must give identical batches.
(c) A variable-length dataset batched randomly vs with a length-bucketing
    BatchSampler; reports the fraction of padded positions for each.
(d) Throughput (samples/s) of the token loader for num_workers 0 and 2. Reported
    only: the number depends on the machine and is never asserted.

All dataset classes and worker_init_fn live at module level so they pickle under
spawn, forkserver and fork. Workers start with --mp-context, default "forkserver"
where the OS offers it (Linux, macOS) else the OS default (spawn on Windows):
on macOS with PyTorch 2.8, spawn workers of an IterableDataset that return tensor
batches were measured to stall about 10 s at shutdown (two 5 s worker-join
timeouts); forkserver has the same pickling rules as spawn without that stall.
The forkserver is told to preload torch and numpy, so each worker pool forks from
a process that has already paid the ~1 s import instead of re-importing torch.

Run from the repository root:
    python content/courses/pytorch/code/14_dataloader.py [--seed 0] [--num-items 2000] [--workers 2] [--batch-size 32] [--mp-context forkserver] [--json-out PATH]
"""

import argparse
import json
import multiprocessing
import os
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
    get_worker_info,
)

NUM_TOKENS = 50_000
VOCAB = 512
BLOCK_SIZE = 64
MASK_ID = 0
MASK_PROB = 0.15


# ---------------------------------------------------------------- (a) streams
class IntStream(IterableDataset):
    """Yields 0..n-1. With shard=True each worker takes every num_workers-th item."""

    def __init__(self, n, shard):
        self.n = n
        self.shard = shard

    def __iter__(self):
        info = get_worker_info()
        if self.shard and info is not None:
            return iter(range(info.id, self.n, info.num_workers))
        return iter(range(self.n))


def default_context():
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else None


def count_stream(n, shard, workers, batch_size, ctx=None):
    loader = DataLoader(IntStream(n, shard), batch_size=batch_size, num_workers=workers,
                        multiprocessing_context=ctx if workers > 0 else None)
    seen = torch.cat([batch for batch in loader]).tolist()
    return len(seen), len(set(seen))


# ------------------------------------------------------- (b) memmap token data
def write_tokens(path, num_tokens, vocab, seed):
    """A flat uint16 token file, the on-disk format of nanoGPT-style corpora."""
    rng = np.random.default_rng(seed)
    rng.integers(0, vocab, size=num_tokens, dtype=np.uint16).tofile(path)


class TokenWindows(Dataset):
    """Window i is tokens[i : i + block + 1]; returns (x, y) shifted by one.

    The memmap is opened lazily, so pickling the dataset into a worker sends a
    file path, not the corpus. Tokens are uint16 on disk and int64 once read.
    With mask_prob > 0, positions of x are replaced by MASK_ID using self.rng,
    a NumPy Generator that worker_init_fn creates per worker.
    """

    def __init__(self, path, block_size, mask_prob=0.0):
        self.path = path
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.rng = None
        self._tokens = None

    def tokens(self):
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._tokens

    def __len__(self):
        return len(self.tokens()) - self.block_size

    def __getitem__(self, i):
        chunk = torch.from_numpy(self.tokens()[i : i + self.block_size + 1].astype(np.int64))
        x, y = chunk[:-1].clone(), chunk[1:]
        if self.mask_prob > 0:
            if self.rng is None:  # main process (num_workers=0)
                self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
            hit = torch.from_numpy(self.rng.random(self.block_size) < self.mask_prob)
            x[hit] = MASK_ID
        return x, y


def seed_worker(worker_id):
    """Runs in each worker after PyTorch seeded it with base_seed + worker_id."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(seed)


def token_loader(path, seed, workers, batch_size, num_samples, ctx=None):
    dataset = TokenWindows(path, BLOCK_SIZE, mask_prob=MASK_PROB)
    gen = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=gen)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=gen,
        multiprocessing_context=ctx if workers > 0 else None,
    )


def run_tokens(path, seed, workers, batch_size, num_samples, ctx=None):
    """Returns (list of (x, y) batches, samples/s measured after the first batch)."""
    loader = token_loader(path, seed, workers, batch_size, num_samples, ctx)
    batches, t0, n = [], None, 0
    for x, y in loader:
        if t0 is None:
            t0 = time.perf_counter()  # exclude worker start-up from the rate
        else:
            n += x.shape[0]
        batches.append((x, y))
    elapsed = time.perf_counter() - t0
    return batches, n / elapsed if elapsed > 0 else float("inf")


# ------------------------------------------------- (c) variable length + buckets
class VarLenText(Dataset):
    """Token sequences with lengths uniform in [4, 64]; label = next token."""

    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.lengths = torch.randint(4, 65, (n,), generator=g).tolist()
        self.seqs = [torch.randint(1, VOCAB, (L,), generator=g) for L in self.lengths]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def pad_collate(seqs, pad_id=0):
    """List of 1-D int64 tensors -> right-padded dict batch; labels -100 at padding."""
    lengths = torch.tensor([len(s) for s in seqs])
    T = int(lengths.max())
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for row, s in enumerate(seqs):
        input_ids[row, : len(s)] = s
    attention_mask = (torch.arange(T)[None, :] < lengths[:, None]).long()
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class BucketBatchSampler(Sampler):
    """Shuffle, cut into pools of pool_batches * batch_size, sort each pool by length,
    split into batches, then shuffle the batch order. Yields lists of indices."""

    def __init__(self, lengths, batch_size, generator, pool_batches=50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.generator = generator
        self.pool = pool_batches * batch_size

    def __iter__(self):
        order = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        batches = []
        for start in range(0, len(order), self.pool):
            pool = sorted(order[start : start + self.pool], key=self.lengths.__getitem__)
            batches += [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
        for j in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[j]

    def __len__(self):
        return sum(-(-min(self.pool, len(self.lengths) - s) // self.batch_size)
                   for s in range(0, len(self.lengths), self.pool))


def padding_fraction(loader):
    pad = total = 0
    for batch in loader:
        mask = batch["attention_mask"]
        pad += int((mask == 0).sum())
        total += mask.numel()
    return pad / total


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-items", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mp-context", type=str, default=default_context())
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    W, B, ctx = args.workers, args.batch_size, args.mp_context
    if ctx == "forkserver":
        multiprocessing.set_forkserver_preload(["torch", "numpy"])

    # (a) duplicated vs sharded stream
    total, unique = count_stream(args.num_items, shard=False, workers=W, batch_size=B, ctx=ctx)
    sharded_total, sharded_unique = count_stream(args.num_items, shard=True, workers=W, batch_size=B, ctx=ctx)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tokens.bin")
        write_tokens(path, NUM_TOKENS, VOCAB, args.seed)
        # (b) + (d): the same seeded loader twice with workers, once without
        run1, rate_w = run_tokens(path, args.seed, W, B, args.num_items, ctx)
        run2, _ = run_tokens(path, args.seed, W, B, args.num_items, ctx)
        _, rate_0 = run_tokens(path, args.seed, 0, B, args.num_items)
    reproducible = len(run1) == len(run2) and all(
        torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(run1, run2)
    )

    # (c) random batching vs length bucketing, both with num_workers=0
    text = VarLenText(args.num_items, args.seed)
    rand_loader = DataLoader(text, batch_size=B, shuffle=True, collate_fn=pad_collate,
                             generator=torch.Generator().manual_seed(args.seed))
    bucket_loader = DataLoader(text, collate_fn=pad_collate, batch_sampler=BucketBatchSampler(
        text.lengths, B, torch.Generator().manual_seed(args.seed)))

    result = {
        "unsharded_duplicates": total - unique,
        "sharded_unique": sharded_unique,
        "sharded_total": sharded_total,
        "reproducible": bool(reproducible),
        "padding_frac_random": round(padding_fraction(rand_loader), 4),
        "padding_frac_bucketed": round(padding_fraction(bucket_loader), 4),
        "throughput": {"0": round(rate_0, 1), str(W): round(rate_w, 1)},
        "mp_context": ctx or multiprocessing.get_start_method(),
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
