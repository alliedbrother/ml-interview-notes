"""Autoregressive generation for the capstone GPT (module 35).

Sampling (greedy, temperature, top-k, top-p) with explicit torch.Generators,
cached and uncached decode loops, left-padded batched generation, a
cached-vs-uncached equivalence check and byte-safe streaming decode.

Public interface (syllabus E.7; modules 36 and 38 import it):
    sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None) -> LongTensor (B, 1)
    top_k_filter(logits, k) -> logits with everything outside the top k set to -inf
    top_p_filter(logits, p) -> logits with everything outside the nucleus set to -inf
    generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             eos_id=None, generator=None, use_cache=True, *, attention_mask=None,
             block_size=None, on_context_limit="stop", return_info=False) -> LongTensor (B, T + n)
    generate_batch(model, tokenizer, prompts, max_new_tokens, **sampling) -> list[str]
    verify_cache(model, idx, steps) -> dict
    stream_decode(tokenizer, ids) -> Iterator[str]

With use_cache=False and no attention_mask, `model` may be any callable
f(idx) -> (logits, _) (module 38 passes an exported program's module).

Imports torch and llm.model only. Read it next to
content/courses/pytorch/35-capstone-generation-kv-cache.md.
"""

from __future__ import annotations

import codecs
import math
from typing import Iterator

import torch

from llm.model import KVCache

STOP_REASONS = ("max_new_tokens", "eos", "context_limit")


class ContextLimitError(ValueError):
    """Raised by generate(on_context_limit="raise") when a learned-position cache is full."""


# ---------------------------------------------------------------- sampling

def _check_sampling_args(temperature, top_k, top_p) -> None:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or math.isnan(temperature) or temperature < 0:
        raise ValueError(f"temperature must be a number >= 0, got {temperature!r}")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be None or an integer >= 1, got {top_k!r}")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                              or not 0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be None or lie in (0, 1], got {top_p!r}")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per row (ties at the k-th value are kept), -inf elsewhere."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]           # (..., 1)
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter (Holtzman et al. 2019): keep the smallest prefix of the
    descending-sorted distribution whose mass reaches p; the top token always survives."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p!r}")
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs                      # exclusive prefix sum
    drop_sorted = mass_before >= p                                  # prefix already reached p
    drop_sorted[..., 0] = False                                     # never drop the top token
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)  # back to vocab order
    return logits.masked_fill(drop, float("-inf"))


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1) int64.

    temperature 0 is greedy (argmax; top_k/top_p are irrelevant). Otherwise divide
    by temperature, apply top-k then top-p, and draw with torch.multinomial using
    `generator` (pass one: the global RNG is shared with everything else)."""
    _check_sampling_args(temperature, top_k, top_p)
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, V), got shape {tuple(logits.shape)}")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits.float() / temperature
    if top_k is not None:
        scaled = top_k_filter(scaled, top_k)
    if top_p is not None:
        scaled = top_p_filter(scaled, top_p)
    return torch.multinomial(scaled.softmax(dim=-1), num_samples=1, generator=generator)


def _margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 logit per row: how close greedy decoding was to a tie."""
    top2 = torch.topk(logits.float(), 2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


# ---------------------------------------------------------------- the decode loop

def _block_size(model, block_size):
    if block_size is not None:
        return int(block_size)
    cfg = getattr(model, "config", None)
    return None if cfg is None else int(cfg.block_size)


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, T) 0/1 mask -> (B, T) position ids that start at 0 on each row's first real token."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


@torch.inference_mode()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None, eos_id: int | None = None,
             generator: torch.Generator | None = None, use_cache: bool = True, *,
             attention_mask: torch.Tensor | None = None, block_size: int | None = None,
             on_context_limit: str = "stop", return_info: bool = False):
    """Extend idx (B, T) by up to max_new_tokens tokens; returns (B, T + n) int64.

    use_cache=True : prefill the prompt once into a static KVCache, then feed one
                     token per step. Learned positions end at block_size, so when
                     cache.length == block_size the loop stops (stop_reason
                     "context_limit") or raises ContextLimitError.
    use_cache=False: re-run the full (cropped to the last block_size tokens) prefix
                     every step, i.e. a sliding window = full recompute.
    eos_id         : a row that emits it is finished and emits eos_id from then on;
                     the loop ends when every row has finished.
    attention_mask : (B, T) 1 = real token, 0 = left padding (needs a GPT forward that
                     accepts position_ids= and key_mask=).
    return_info    : also return {"stop_reason", "new_tokens", "margins" (B, n)}.
    """
    _check_sampling_args(temperature, top_k, top_p)
    if on_context_limit not in ("stop", "raise"):
        raise ValueError("on_context_limit must be 'stop' or 'raise'")
    if idx.dim() != 2 or idx.size(1) == 0:
        raise ValueError(f"idx must be a non-empty (B, T) tensor, got shape {tuple(idx.shape)}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be an integer >= 0, got {max_new_tokens!r}")
    B, T0 = idx.shape
    limit = _block_size(model, block_size)
    if use_cache and limit is None:
        raise ValueError("use_cache=True needs a model with .config.block_size")
    mask = None
    if attention_mask is not None:
        if attention_mask.shape != idx.shape:
            raise ValueError("attention_mask must have the same shape as idx")
        mask = attention_mask.to(device=idx.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every row needs at least one real token")
    if use_cache and T0 > limit:
        raise ContextLimitError(f"prompt of {T0} tokens exceeds block_size={limit}")

    tokens = torch.empty(B, T0 + max_new_tokens, dtype=torch.long, device=idx.device)
    tokens[:, :T0] = idx
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
    margins = []
    stop_reason, n = "max_new_tokens", 0

    cache = full_mask = None
    if use_cache:
        cache = KVCache(model.config, B, limit, dtype=next(model.parameters()).dtype, device=idx.device)
        if mask is not None:  # preallocated key mask; new tokens are always real
            full_mask = torch.ones(B, limit, dtype=torch.bool, device=idx.device)
            full_mask[:, :T0] = mask
        next_pos = None if mask is None else mask.sum(dim=1, keepdim=True)  # (B, 1)

    for step in range(max_new_tokens):
        cur = T0 + step                                                 # tokens so far
        if use_cache:
            if cache.length == limit:                                   # positions exhausted
                if on_context_limit == "raise":
                    raise ContextLimitError(
                        f"KV cache full at block_size={limit} positions; learned positions "
                        f"cannot extrapolate (use use_cache=False to slide the window)")
                stop_reason = "context_limit"
                break
            if step == 0:                                               # prefill
                feed = tokens[:, :T0]
                kwargs = {} if mask is None else {"position_ids": positions_from_mask(mask),
                                                  "key_mask": full_mask[:, :T0]}
            else:                                                       # decode one token
                feed = tokens[:, cur - 1:cur]
                kwargs = {} if mask is None else {"position_ids": next_pos,
                                                  "key_mask": full_mask[:, :cache.length + 1]}
                if mask is not None:
                    next_pos = next_pos + 1
            logits, _ = model(feed, cache=cache, **kwargs)
        else:
            start = 0 if limit is None else max(0, cur - limit)        # crop to the window
            feed = tokens[:, start:cur]
            if mask is None:
                logits, _ = model(feed)
            else:
                m = torch.cat([mask, torch.ones(B, cur - T0, dtype=torch.bool, device=idx.device)], 1)
                m = m[:, start:cur]
                logits, _ = model(feed, position_ids=positions_from_mask(m), key_mask=m)
        last = logits[:, -1, :]                                         # (B, V)
        if return_info:
            margins.append(_margin(last))
        nxt = sample_next(last, temperature, top_k, top_p, generator).view(B)
        if eos_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            finished |= nxt == eos_id
        tokens[:, cur] = nxt
        n = step + 1
        if eos_id is not None and bool(finished.all()):
            stop_reason = "eos"
            break

    out = tokens[:, :T0 + n]
    if not return_info:
        return out
    info = {"stop_reason": stop_reason, "new_tokens": n,
            "margins": torch.stack(margins, dim=1) if margins else torch.empty(B, 0)}
    return out, info


# ---------------------------------------------------------------- batches of strings

def generate_batch(model, tokenizer, prompts, max_new_tokens: int, *, return_ids: bool = False,
                   **sampling):
    """Left-pad the encoded prompts with the EOT id, generate them as one batch with
    a per-row key mask and mask-derived positions, and return each row's completion
    (new tokens up to, not including, the first EOT) as text, or as id lists if
    return_ids=True. sampling: temperature, top_k, top_p, generator, use_cache, eos_id
    (default tokenizer.eot_id)."""
    if isinstance(prompts, str) or not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    pad = tokenizer.eot_id
    sampling.setdefault("eos_id", pad)
    encoded = [tokenizer.encode(p) for p in prompts]
    if any(len(e) == 0 for e in encoded):
        raise ValueError("every prompt must encode to at least one token")
    T = max(len(e) for e in encoded)
    idx = torch.full((len(encoded), T), pad, dtype=torch.long)
    mask = torch.zeros(len(encoded), T, dtype=torch.long)
    for row, e in enumerate(encoded):
        idx[row, T - len(e):] = torch.tensor(e, dtype=torch.long)
        mask[row, T - len(e):] = 1
    out = generate(model, idx, max_new_tokens, attention_mask=mask, **sampling)
    results = []
    for row in out[:, T:].tolist():
        eos = sampling["eos_id"]
        if eos is not None and eos in row:
            row = row[:row.index(eos)]
        results.append(row if return_ids else tokenizer.decode(row))
    return results


# ---------------------------------------------------------------- verification

@torch.inference_mode()
def verify_cache(model, idx: torch.Tensor, steps: int) -> dict:
    """Greedy-decode `steps` tokens with the cache; at every step also run the uncached
    forward over the same prefix and compare last-position logits.

    tokens_equal is True when every step's argmax agrees, which is the same as the
    two free-running greedy decodes producing identical tokens. When they do not,
    first_divergence and margin_at_divergence say whether it was a near-tie."""
    if idx.dim() != 2:
        raise ValueError("idx must be (B, T)")
    B, T0 = idx.shape
    limit = model.config.block_size
    if T0 + steps - 1 > limit:
        raise ValueError(f"prompt {T0} + steps {steps} needs more than block_size={limit} positions")
    cache = KVCache(model.config, B, limit)
    seq = idx.clone()
    diffs, margins, first_div, margin_div = [], [], None, None
    for t in range(steps):
        feed = seq if t == 0 else seq[:, -1:]
        cached, _ = model(feed, cache=cache)
        full, _ = model(seq)
        c, u = cached[:, -1].float(), full[:, -1].float()
        diffs.append((c - u).abs().max().item())
        margins.append(_margin(u).min().item())
        nxt_c, nxt_u = c.argmax(-1), u.argmax(-1)
        if first_div is None and not torch.equal(nxt_c, nxt_u):
            first_div, margin_div = t, _margin(u).min().item()
        seq = torch.cat([seq, nxt_c[:, None]], dim=1)   # teacher-force the cached path
    return {"steps": steps, "max_abs_logit_diff": max(diffs), "per_step_max_abs_diff": diffs,
            "tokens_equal": first_div is None, "first_divergence": first_div,
            "margin_at_divergence": margin_div, "min_margin": min(margins),
            "tokens": seq[:, T0:].tolist()}


# ---------------------------------------------------------------- streaming text

def stream_decode(tokenizer, ids) -> Iterator[str]:
    """Yield text as soon as it forms complete UTF-8 characters.

    Byte-level tokens can end mid-character (e.g. 0xE2 0x80 before 0x9C = U+201C).
    An incremental decoder buffers the partial bytes instead of emitting U+FFFD; the
    concatenated output equals tokenizer.decode(ids) (both use errors="replace")."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token_id in ids:
        text = decoder.decode(tokenizer.token_bytes(int(token_id)))
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
