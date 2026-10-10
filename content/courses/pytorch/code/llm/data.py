"""Token datasets and the resumable training data pipeline for the capstone (module 31).

One split = one flat ``int64`` tensor of token ids (stories joined with
``<|endoftext|>``). Training draws random ``(T + 1)``-token windows from it;
evaluation walks it with sequential, optionally overlapping windows and a loss
mask so every target is scored exactly once. Batch ``k`` of the training loader
is a pure function of ``(seed, k)``, so resuming at step ``k`` needs no replay.

Public interface (modules 33, 34, 36, 37 and 38 import it; names are frozen):
    load_split_tokens, encode_all_splits, check_uint16_range, save_uint16,
    load_uint16, get_batch, batch_seed, RandomWindowDataset,
    SequentialEvalDataset, TrainBatchIterator, make_train_loader,
    boundary_fraction
"""

import os
from pathlib import Path

import numpy as np
import torch

from llm.corpus import DATA_DIR, EOT, join_with_eot, load_corpus, load_splits
from llm.tokenizer import ByteBPE

TOKENIZER_PATH = DATA_DIR / "bpe_sherlock_512.json"
SPLITS = ("train", "val", "test")
UINT16_MAX = 65_535
# Per-step generator seed = seed * SEED_STRIDE + step. A prime larger than any
# step count we run keeps (seed, step) pairs from colliding across nearby seeds.
SEED_STRIDE = 1_000_003

_SPLIT_CACHE: dict = {}


# --------------------------------------------------------------------- encoding
def _default_tokenizer() -> ByteBPE:
    return ByteBPE.load(TOKENIZER_PATH)


def load_split_tokens(split: str, tokenizer: ByteBPE, text: str | None = None) -> torch.Tensor:
    """Encode one split (stories joined with EOT) to a 1-D ``torch.int64`` tensor.

    ``text`` is the full corpus text (default: the committed, hash-verified file).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    stories = load_splits(text)[split]
    ids = tokenizer.encode(join_with_eot(stories), allowed_special={EOT})
    return torch.tensor(ids, dtype=torch.int64)


def encode_all_splits(tokenizer: ByteBPE | None = None) -> dict[str, torch.Tensor]:
    """Return ``{"train", "val", "test"}`` int64 token tensors, memoised per process.

    The cache is keyed by the tokenizer's merges and pattern, so a different
    tokenizer never receives another one's ids. Each call returns fresh clones,
    so callers may modify what they get without corrupting the cache.
    """
    if tokenizer is None:
        tokenizer = _default_tokenizer()
    key = (tokenizer.pattern, tuple(tokenizer.merges))
    if key not in _SPLIT_CACHE:
        text = load_corpus()  # read and hash-check once, not once per split
        _SPLIT_CACHE[key] = {s: load_split_tokens(s, tokenizer, text) for s in SPLITS}
    return {s: t.clone() for s, t in _SPLIT_CACHE[key].items()}


# ------------------------------------------------------------------ storage
def check_uint16_range(tokens) -> None:
    """Raise ``ValueError`` unless every id fits in uint16 (0..65,535).

    A plain cast does not check: ``torch.tensor([70000]).to(torch.uint16)`` is 4464.
    """
    t = torch.as_tensor(tokens)
    if t.numel() == 0:
        return
    lo, hi = int(t.min()), int(t.max())
    if lo < 0 or hi > UINT16_MAX:
        raise ValueError(f"token ids span [{lo}, {hi}]; uint16 storage needs 0..{UINT16_MAX}")


def save_uint16(tokens, path) -> int:
    """Write a 1-D id tensor as raw little-endian uint16 (nanoGPT ``.bin`` layout).

    Atomic (temp file + rename). Returns the number of bytes written.
    """
    t = torch.as_tensor(tokens)
    if t.dim() != 1:
        raise ValueError(f"expected a 1-D token tensor, got shape {tuple(t.shape)}")
    check_uint16_range(t)
    arr = t.cpu().numpy().astype("<u2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arr.tofile(tmp)
    os.replace(tmp, path)
    return arr.nbytes


def load_uint16(path) -> torch.Tensor:
    """Read a raw uint16 file back as a 1-D ``torch.int64`` tensor (a copy in RAM).

    For corpora larger than RAM keep the ``np.memmap`` and convert only the
    sampled windows, as nanoGPT's ``get_batch`` does.
    """
    mm = np.memmap(path, dtype="<u2", mode="r")
    out = torch.from_numpy(mm.astype(np.int64))  # astype copies: the memmap can close
    del mm
    return out


# ----------------------------------------------------------------- sampling
def _check_stream(tokens: torch.Tensor, block_size: int) -> None:
    if tokens.dim() != 1 or tokens.dtype != torch.int64:
        raise ValueError(f"tokens must be 1-D int64, got {tuple(tokens.shape)} {tokens.dtype}")
    if block_size < 1 or tokens.numel() < block_size + 1:
        raise ValueError(f"need at least block_size + 1 = {block_size + 1} tokens, "
                         f"have {tokens.numel()}")


def get_batch(tokens: torch.Tensor, block_size: int, batch_size: int,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; return contiguous ``(x, y)``, each ``(B, T)`` int64.

    ``tokens.unfold(0, T + 1, 1)`` is a zero-copy ``(n - T, T + 1)`` view of every
    possible window (stride ``(1, 1)``); indexing it with the sampled offsets
    copies only the ``B`` chosen rows. ``y`` is ``x`` shifted left by one.
    """
    _check_stream(tokens, block_size)
    offsets = torch.randint(0, tokens.numel() - block_size, (batch_size,), generator=generator)
    windows = tokens.unfold(0, block_size + 1, 1)[offsets]  # (B, T + 1), a copy
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def batch_seed(seed: int, step: int) -> int:
    """Seed of the generator that draws training batch ``step`` (pure function)."""
    if seed < 0 or step < 0:
        raise ValueError("seed and step must be non-negative")
    return seed * SEED_STRIDE + step


class RandomWindowDataset(torch.utils.data.Dataset):
    """Map-style dataset of ``num_samples`` random windows per epoch.

    All offsets of an epoch are drawn up front from ``Generator(seed + epoch)``,
    so item ``i`` does not depend on access order, sampler or worker count.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int, num_samples: int, seed: int):
        _check_stream(tokens, block_size)
        self.tokens = tokens
        self.block_size = block_size
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        g = torch.Generator().manual_seed(self.seed + epoch)
        self.offsets = torch.randint(0, self.tokens.numel() - self.block_size,
                                     (self.num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        o = int(self.offsets[i])
        w = self.tokens[o:o + self.block_size + 1]
        return w[:-1], w[1:]


class SequentialEvalDataset(torch.utils.data.Dataset):
    """Sequential evaluation windows; each target index 1..n-1 is scored exactly once.

    Window starts are ``0, stride, 2*stride, ...`` while a full window fits; if
    targets remain, one last window is right-aligned to the end of the stream
    (same length ``T``, more context than a short tail window). In every window
    ``loss_mask`` keeps only targets the previous window did not score, so with
    ``stride < T`` each scored target after the first window has at least
    ``T - stride + 1`` tokens of context. Items are ``(x, y, loss_mask)``:
    int64, int64, bool, each of shape ``(T,)``. A stream shorter than ``T + 1``
    tokens gives a single window of length ``n - 1``.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int | None = None):
        if tokens.dim() != 1 or tokens.dtype != torch.int64 or tokens.numel() < 2:
            raise ValueError("tokens must be 1-D int64 with at least 2 ids")
        stride = block_size if stride is None else stride
        if not 1 <= stride <= block_size:
            raise ValueError(f"stride must be in [1, block_size={block_size}], got {stride}")
        self.tokens, self.block_size, self.stride = tokens, block_size, stride
        n = tokens.numel()
        span = min(block_size, n - 1)            # targets per window
        last_start = n - 1 - span                # right-aligned final window
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        self.starts = starts
        self.span = span
        # first scored position in each window: everything after the previous end
        prev_end = 0  # index of the last target already scored (0 = none; targets start at 1)
        self.first_scored = []
        for b in starts:
            self.first_scored.append(max(0, prev_end - b))  # target index b+1+j > prev_end
            prev_end = b + span
        self.n_scored = sum(span - f for f in self.first_scored)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        b, f = self.starts[i], self.first_scored[i]
        w = self.tokens[b:b + self.span + 1]
        mask = torch.zeros(self.span, dtype=torch.bool)
        mask[f:] = True
        return w[:-1], w[1:], mask


# ------------------------------------------------------- resumable loader
class TrainBatchIterator:
    """Infinite iterator of training batches whose position is one integer.

    Batch ``k`` = ``get_batch(..., Generator().manual_seed(seed * 1_000_003 + k))``.
    ``state_dict()`` is ``{"step": k, "seed": seed}``, where ``k`` is the index of
    the next batch ``__next__`` will return.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int, batch_size: int,
                 seed: int, start_step: int = 0):
        _check_stream(tokens, block_size)
        batch_seed(seed, start_step)  # validates
        self.tokens, self.block_size, self.batch_size = tokens, block_size, batch_size
        self.seed, self.step = seed, start_step

    def __iter__(self):
        return self

    def batch_at(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(batch_seed(self.seed, step))
        return get_batch(self.tokens, self.block_size, self.batch_size, g)

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch_at(self.step)
        self.step += 1
        return batch

    def state_dict(self) -> dict:
        return {"step": self.step, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        step, seed = int(state["step"]), int(state["seed"])
        batch_seed(seed, step)
        self.step, self.seed = step, seed


def make_train_loader(tokens: torch.Tensor, block_size: int, batch_size: int, seed: int,
                      start_step: int = 0) -> TrainBatchIterator:
    """The capstone's training loader: no DataLoader, no workers, exact resume by step."""
    return TrainBatchIterator(tokens, block_size, batch_size, seed, start_step)


# ------------------------------------------------------------ diagnostics
def boundary_fraction(tokens: torch.Tensor, block_size: int, eot_id: int) -> float:
    """Fraction of all ``n - T`` possible training windows (``T + 1`` tokens) containing EOT."""
    _check_stream(tokens, block_size)
    is_eot = (tokens == eot_id).to(torch.int64)
    csum = torch.cat([torch.zeros(1, dtype=torch.int64), is_eot.cumsum(0)])
    n_windows = tokens.numel() - block_size
    starts = torch.arange(n_windows)
    counts = csum[starts + block_size + 1] - csum[starts]
    return float((counts > 0).double().mean())
