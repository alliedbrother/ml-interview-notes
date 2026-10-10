"""Held-out evaluation for the capstone GPT (module 34).

Every number here is a *next-token* quantity over a 1-D token stream
``tokens[0..n-1]``: targets are ``tokens[1:]`` (``n - 1`` of them); the first
token is context only and is never scored. Model and baselines score exactly
the same targets, so their NLLs are directly comparable.

Public interface (syllabus E.6; modules 36 and 38 import ``evaluate_nll``):

    evaluate_nll(model, tokens, block_size, stride, batch_size=16, text_bytes=None) -> dict
    scored_bytes(tokenizer, tokens) -> int
    uniform_nll(V) -> float
    unigram_nll(train_tokens, eval_tokens, V, alpha=1.0) -> float
    bigram_nll(train_tokens, eval_tokens, V, k=0.1, lam=None) -> float
    standard_error(window_losses, window_counts) -> float
    block_bootstrap(window_losses, window_counts, n, seed, block_len=None) -> dict
    next_token_table(model, tokenizer, prompt, k=3) -> list[(str, float)]
    greedy_margin(model, tokenizer, prompt) -> (str, float)
    greedy_continuation(model, ids, max_new_tokens) -> list[int]
    longest_common_substring_in(needle, haystack) -> str

Imports torch and llm.data only; never touches the global RNG.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import SequentialEvalDataset

IGNORE_INDEX = -100
LN2 = math.log(2.0)


# ------------------------------------------------------------------ model NLL

def _set_mode(model, training: bool) -> None:
    """``model.train(training)`` when supported; exported GraphModules raise on it."""
    try:
        model.train(training)
    except (AttributeError, NotImplementedError):
        pass


def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def evaluate_nll(model, tokens: torch.Tensor, block_size: int, stride: int,
                 batch_size: int = 16, text_bytes: int | None = None) -> dict:
    """Token-weighted mean next-token NLL (nats) of ``model`` on ``tokens``.

    Windows come from ``SequentialEvalDataset(tokens, block_size, stride)``:
    ``stride == block_size`` is non-overlapping; ``stride < block_size`` slides,
    scoring only the last ``stride`` targets of each window (the masked overlap
    is context). Either way every target ``tokens[1:]`` is scored exactly once,
    so ``n_tokens == len(tokens) - 1``.

    Per-token losses come from ``F.cross_entropy(..., reduction="none")`` on
    float32 logits and are summed in float64. Runs under ``inference_mode`` and
    restores the model's train/eval mode. ``model`` may be any callable
    returning ``logits`` or ``(logits, ...)`` of shape ``(B, T, V)``, including an
    exported program's ``.module()`` (whose ``train()/eval()`` raise; skipped). An
    export with a static batch dimension needs ``batch_size`` equal to it.

    Returns ``nll_per_token``, ``n_tokens``, ``nll_sum``, ``ppl`` (= exp of the
    mean), ``bits_per_byte`` (``nll_sum / (ln 2 * text_bytes)``; None when
    ``text_bytes`` is None), ``stride``, ``block_size``, ``n_windows``, and
    ``window_losses`` / ``window_counts`` (per-window mean NLL over that window's
    scored targets, and the number of those targets; lists of float / int).
    """
    ds = SequentialEvalDataset(tokens, block_size, stride=stride)
    was_training = getattr(model, "training", False)
    _set_mode(model, False)
    window_sums, window_counts = [], []
    try:
        with torch.inference_mode():
            for i in range(0, len(ds), batch_size):
                items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
                x = torch.stack([it[0] for it in items])
                y = torch.stack([it[1] for it in items])
                m = torch.stack([it[2] for it in items])
                logits = _forward_logits(model, x).float()
                y = y.masked_fill(~m, IGNORE_INDEX)
                tok = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=IGNORE_INDEX,
                                      reduction="none")             # (B, T), 0 where ignored
                window_sums.extend(tok.double().sum(dim=1).tolist())
                window_counts.extend(m.sum(dim=1).tolist())
    finally:
        _set_mode(model, was_training)
    n = int(sum(window_counts))
    assert n == ds.n_scored == tokens.numel() - 1, (n, ds.n_scored, tokens.numel())
    total = math.fsum(window_sums)
    nll = total / n
    return {
        "nll_per_token": nll,
        "n_tokens": n,
        "nll_sum": total,
        "ppl": math.exp(nll),
        "bits_per_byte": None if text_bytes is None else total / (LN2 * text_bytes),
        "stride": stride,
        "block_size": block_size,
        "n_windows": len(ds),
        "window_losses": [s / c for s, c in zip(window_sums, window_counts)],
        "window_counts": [int(c) for c in window_counts],
    }


def scored_bytes(tokenizer, tokens) -> int:
    """UTF-8 bytes covered by the scored targets ``tokens[1:]``.

    Special tokens (``<|endoftext|>``) are not text and count 0 bytes. The first
    token is context only, so its bytes are excluded: this is the denominator
    that makes ``bits_per_byte`` the cost of the bytes the model actually predicted.
    """
    special = set(tokenizer.special_tokens.values())
    lengths = {}
    total = 0
    for t in tokens[1:].tolist():
        if t not in lengths:
            lengths[t] = 0 if t in special else len(tokenizer.token_bytes(t))
        total += lengths[t]
    return total


# ------------------------------------------------------------------ baselines

def uniform_nll(V: int) -> float:
    """NLL of guessing uniformly over V ids: ln V nats per token."""
    return math.log(V)


def _unigram_probs(train_tokens: torch.Tensor, V: int, alpha: float) -> torch.Tensor:
    counts = torch.bincount(train_tokens, minlength=V).double()
    return (counts + alpha) / (counts.sum() + alpha * V)


def unigram_nll(train_tokens: torch.Tensor, eval_tokens: torch.Tensor, V: int,
                alpha: float = 1.0) -> float:
    """Add-alpha unigram fitted on train, scored on ``eval_tokens[1:]``.

    p(b) = (c_b + alpha) / (N + alpha * V).
    """
    p = _unigram_probs(train_tokens, V, alpha)
    return -p[eval_tokens[1:]].log().mean().item()


def bigram_counts(tokens: torch.Tensor, V: int) -> torch.Tensor:
    """(V, V) float64 count matrix C[a, b] of adjacent pairs, via one bincount on a*V+b."""
    pair_ids = tokens[:-1] * V + tokens[1:]
    return torch.bincount(pair_ids, minlength=V * V).double().view(V, V)


def bigram_nll(train_tokens: torch.Tensor, eval_tokens: torch.Tensor, V: int,
               k: float = 0.1, lam: float | None = None) -> float:
    """Bigram fitted on train, scored on pairs ``(eval[i], eval[i+1])``.

    Two interpolation schemes with the add-one unigram ``p_uni``:

    * ``lam=None`` (default): add-k with a *unigram* prior instead of a uniform one,
      ``p(b|a) = (C(a,b) + k*V*p_uni(b)) / (C(a) + k*V)``. Algebraically this is
      ``lam_a * C(a,b)/C(a) + (1 - lam_a) * p_uni(b)`` with
      ``lam_a = C(a) / (C(a) + k*V)``: frequent contexts trust their own counts,
      rare ones fall back to the unigram. No free parameter besides k.
    * ``lam`` a float in [0, 1]: fixed mixture
      ``lam * (C(a,b) + k)/(C(a) + k*V) + (1 - lam) * p_uni(b)``.
    """
    C = bigram_counts(train_tokens, V)
    p_uni = _unigram_probs(train_tokens, V, 1.0)
    ctx = C.sum(dim=1, keepdim=True)
    if lam is None:
        P = (C + k * V * p_uni[None, :]) / (ctx + k * V)
    else:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        P = lam * (C + k) / (ctx + k * V) + (1.0 - lam) * p_uni[None, :]
    return -P[eval_tokens[:-1], eval_tokens[1:]].log().mean().item()


# ------------------------------------------------------------------ uncertainty

def _weighted_mean(losses: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return (losses * counts).sum(-1) / counts.sum(-1)


def standard_error(window_losses, window_counts) -> float:
    """Standard error of the token-weighted mean, treating windows as independent.

    Ratio-estimator form: ``sqrt(n/(n-1) * sum(c_i^2 (l_i - m)^2)) / sum(c_i)``.
    Adjacent windows are correlated (neighbouring text), so this is optimistic;
    ``block_bootstrap`` is the honest alternative.
    """
    loss = torch.as_tensor(window_losses, dtype=torch.float64)
    cnt = torch.as_tensor(window_counts, dtype=torch.float64)
    n = loss.numel()
    if n < 2:
        raise ValueError("need at least 2 windows")
    m = _weighted_mean(loss, cnt)
    var = n / (n - 1) * ((cnt * (loss - m)) ** 2).sum()
    return (var.sqrt() / cnt.sum()).item()


def block_bootstrap(window_losses, window_counts, n: int, seed: int,
                    block_len: int | None = None) -> dict:
    """Moving-block bootstrap of the token-weighted mean NLL.

    Each replicate concatenates ``ceil(W / block_len)`` runs of ``block_len``
    *contiguous* windows (starts drawn with ``torch.randint`` from a dedicated
    ``torch.Generator`` seeded with ``seed``), truncated to W windows, and takes
    the token-weighted mean. Contiguous blocks keep short-range correlation that
    an i.i.d. resample would destroy. Default ``block_len = round(W ** (1/3))``.
    Returns ``mean``, ``lo``/``hi`` (2.5 / 97.5 percentiles), ``se`` (std of the
    replicates), ``n``, ``block_len``.
    """
    loss = torch.as_tensor(window_losses, dtype=torch.float64)
    cnt = torch.as_tensor(window_counts, dtype=torch.float64)
    W = loss.numel()
    if block_len is None:
        block_len = max(1, round(W ** (1.0 / 3.0)))
    if not 1 <= block_len <= W:
        raise ValueError(f"block_len must be in [1, {W}]")
    g = torch.Generator().manual_seed(seed)
    n_blocks = math.ceil(W / block_len)
    starts = torch.randint(0, W - block_len + 1, (n, n_blocks), generator=g)
    idx = (starts[:, :, None] + torch.arange(block_len)).reshape(n, -1)[:, :W]   # (n, W)
    reps = _weighted_mean(loss[idx], cnt[idx])                                    # (n,)
    q = torch.quantile(reps, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"mean": _weighted_mean(loss, cnt).item(), "lo": q[0].item(), "hi": q[1].item(),
            "se": reps.std().item(), "n": n, "block_len": block_len}


# ------------------------------------------------------------------ behaviour

def _last_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("empty prompt")
    block = model.config.block_size if hasattr(model, "config") else len(ids)
    x = torch.tensor([ids[-block:]], dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return _forward_logits(model, x)[0, -1].float()
    finally:
        model.train(was_training)


def next_token_table(model, tokenizer, prompt: str, k: int = 3) -> list[tuple[str, float]]:
    """Top-k next tokens after ``prompt`` as ``(decoded string, probability)``."""
    probs = torch.softmax(_last_logits(model, tokenizer, prompt), dim=-1)
    top = torch.topk(probs, k)
    return [(tokenizer.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)]


def greedy_margin(model, tokenizer, prompt: str) -> tuple[str, float]:
    """Greedy next token after ``prompt`` and its logit margin over the runner-up.

    A small margin means a different CPU's reduction order could flip the argmax;
    assert the greedy token only when the margin is comfortably large.
    """
    top = torch.topk(_last_logits(model, tokenizer, prompt), 2)
    return tokenizer.decode([int(top.indices[0])]), float(top.values[0] - top.values[1])


def greedy_continuation(model, ids, max_new_tokens: int) -> list[int]:
    """Uncached greedy decoding (argmax), cropping context to ``block_size``.

    A minimal loop for the memorisation probe; module 35 owns real generation.
    """
    block = model.config.block_size
    out = list(ids)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                x = torch.tensor([out[-block:]], dtype=torch.long)
                out.append(int(_forward_logits(model, x)[0, -1].argmax()))
    finally:
        model.train(was_training)
    return out[len(ids):]


def longest_common_substring_in(needle: str, haystack: str) -> str:
    """Longest substring of ``needle`` that occurs verbatim in ``haystack``.

    For each start, binary-search the longest match (substring membership is
    monotone in length): O(len(needle) * log len(needle)) ``in`` scans.
    """
    best = ""
    for i in range(len(needle)):
        lo, hi = len(best), len(needle) - i      # only look for something longer
        if hi <= lo:
            break
        if needle[i:i + lo + 1] not in haystack:
            continue
        lo += 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if needle[i:i + mid] in haystack:
                lo = mid
            else:
                hi = mid - 1
        best = needle[i:i + lo]
    return best
