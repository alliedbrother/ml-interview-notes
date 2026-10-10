"""Five attention implementations, four masks, one offset-chunk trap (module 12 lab).

On CPU in float32 this builds x of shape (B, T, d_model), projects it with explicit
seeded weights into Q, K, V of shape (B, H, T, d_head), and checks that these agree:

  manual        softmax(Q K^T / sqrt(d_head) + additive mask) V, written out by hand
  sdpa          F.scaled_dot_product_attention with default backend dispatch
  sdpa_math     the same call inside sdpa_kernel([SDPBackend.MATH])
  mha           nn.MultiheadAttention with the explicit projections copied in
  flex          eager flex_attention with a BlockMask built on CPU

Masks: causal, sliding window (w=8, causal), document (two packed documents of
lengths 40 + 24 for T=64, causal inside each), and key padding (per-batch valid
lengths). It also checks the offset-chunk trap (is_causal=True on 4 queries against
64 keys is top-left aligned, which is wrong for a KV cache), compares gradients of
manual vs SDPA, checks a jagged nested tensor against the packed bidirectional
document mask, and times each implementation (reported, never asserted).

Run from the repository root:
    python content/courses/pytorch/code/12_attention_backends.py [--seed 0] [--batch 2] [--heads 4] [--seq 64] [--d-head 16] [--json-out PATH]
"""

import argparse
import json
import logging
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils import benchmark

WINDOW = 8
CHUNK = 4


def doc_lengths(seq):
    """Two packed documents; 40 + 24 when seq == 64."""
    second = (seq * 3) // 8
    return seq - second, second


def make_problem(seed, batch, heads, seq, d_head):
    """Seeded input x, explicit projection weights, and the projected q, k, v."""
    gen = torch.Generator().manual_seed(seed)
    d_model = heads * d_head
    x = torch.randn(batch, seq, d_model, generator=gen)
    std = 1.0 / math.sqrt(d_model)
    w = {name: torch.randn(d_model, d_model, generator=gen) * std for name in ("q", "k", "v", "o")}
    b = {name: torch.randn(d_model, generator=gen) * 0.1 for name in ("q", "k", "v", "o")}

    def split(t):  # (B, T, d_model) -> (B, H, T, d_head)
        return t.view(batch, seq, heads, d_head).transpose(1, 2)

    q = split(F.linear(x, w["q"], b["q"]))
    k = split(F.linear(x, w["k"], b["k"]))
    v = split(F.linear(x, w["v"], b["v"]))
    return {"x": x, "w": w, "b": b, "q": q, "k": k, "v": v}


def build_masks(batch, seq):
    """Boolean masks, True = this (query, key) pair takes part in attention (SDPA polarity).

    Returns a dict of tensors broadcastable to (B, H, T, T), the per-batch valid lengths,
    and the document ids.
    """
    i = torch.arange(seq)[:, None]
    j = torch.arange(seq)[None, :]
    causal = j <= i
    window = causal & (i - j < WINDOW)
    len_a, _ = doc_lengths(seq)
    doc = (torch.arange(seq) >= len_a).long()
    document = causal & (doc[:, None] == doc[None, :])
    lengths = torch.tensor([max(1, seq - (bi * seq) // 4) for bi in range(batch)])
    keep_key = torch.arange(seq)[None, :] < lengths[:, None]  # (B, T)
    padding = keep_key[:, None, None, :]  # (B, 1, 1, T)
    masks = {"causal": causal, "window": window, "document": document, "padding": padding}
    return masks, lengths, doc


def mask_mods(lengths, doc):
    """The same four masks as FlexAttention mask_mod(b, h, q_idx, kv_idx) -> bool."""

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def window(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < WINDOW)

    def document(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc[q_idx] == doc[kv_idx])

    def padding(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return {"causal": causal, "window": window, "document": document, "padding": padding}


def manual_attention(q, k, v, mask=None):
    """Reference: masked_fill(-inf) then softmax, as in nanoGPT and Raschka. Returns (out, weights)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def merge_heads(t):
    """(B, H, T, d_head) -> (B, T, H * d_head)."""
    batch, heads, seq, d_head = t.shape
    return t.transpose(1, 2).reshape(batch, seq, heads * d_head)


def build_mha(prob, heads):
    d_model = prob["x"].size(-1)
    mha = nn.MultiheadAttention(d_model, heads, batch_first=True)
    w, b = prob["w"], prob["b"]
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([w["q"], w["k"], w["v"]], dim=0))  # (3 d_model, d_model)
        mha.in_proj_bias.copy_(torch.cat([b["q"], b["k"], b["v"]], dim=0))
        mha.out_proj.weight.copy_(w["o"])
        mha.out_proj.bias.copy_(b["o"])
    return mha.eval()


def mha_forward(mha, x, name, mask):
    """MHA uses the opposite boolean polarity: True = do NOT attend."""
    if name == "padding":
        key_padding_mask = ~mask[:, 0, 0, :]  # (B, T), True = padding key, ignore it
        return mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)[0]
    return mha(x, x, x, attn_mask=~mask, need_weights=False)[0]


def max_diff(a, b):
    return float((a - b).abs().max())


def run(seed=0, batch=2, heads=4, seq=64, d_head=16, time_it=True, return_tensors=False):
    torch.manual_seed(seed)
    prob = make_problem(seed, batch, heads, seq, d_head)
    q, k, v, x = prob["q"], prob["k"], prob["v"], prob["x"]
    masks, lengths, doc = build_masks(batch, seq)
    mods = mask_mods(lengths, doc)
    mha = build_mha(prob, heads)
    w, b = prob["w"], prob["b"]

    diffs, outputs, block_masks = {}, {}, {}
    with torch.no_grad():
        for name, mask in masks.items():
            ref, _ = manual_attention(q, k, v, mask)
            outputs[name] = ref
            diffs[f"sdpa_{name}"] = max_diff(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), ref)
            with sdpa_kernel([SDPBackend.MATH]):
                diffs[f"sdpa_math_{name}"] = max_diff(
                    F.scaled_dot_product_attention(q, k, v, attn_mask=mask), ref
                )
            mha_ref = F.linear(merge_heads(ref), w["o"], b["o"])  # (B, T, d_model)
            diffs[f"mha_{name}"] = max_diff(mha_forward(mha, x, name, mask), mha_ref)
            bsz = batch if name == "padding" else None
            block_masks[name] = create_block_mask(mods[name], bsz, None, seq, seq, device="cpu")
            diffs[f"flex_{name}"] = max_diff(flex_attention(q, k, v, block_mask=block_masks[name]), ref)
        diffs["sdpa_is_causal"] = max_diff(F.scaled_dot_product_attention(q, k, v, is_causal=True), outputs["causal"])

        # Packed documents must not leak: document 2 equals running document 2 alone.
        len_a, _ = doc_lengths(seq)
        alone = F.scaled_dot_product_attention(q[:, :, len_a:], k[:, :, len_a:], v[:, :, len_a:], is_causal=True)
        diffs["document2_alone_vs_packed"] = max_diff(outputs["document"][:, :, len_a:], alone)

        # Offset chunk: the last CHUNK queries against all seq keys (a KV-cache decode step).
        start = seq - CHUNK
        q_chunk = q[:, :, start:]
        correct_mask = torch.ones(CHUNK, seq, dtype=torch.bool).tril(diagonal=start)  # key j <= start + i
        chunk_ok = F.scaled_dot_product_attention(q_chunk, k, v, attn_mask=correct_mask)
        chunk_trap = F.scaled_dot_product_attention(q_chunk, k, v, is_causal=True)  # top-left aligned
        diffs["offset_chunk_explicit_vs_full"] = max_diff(chunk_ok, outputs["causal"][:, :, start:])
        offset_trap = max_diff(chunk_trap, chunk_ok)

        # Variable length without padding: jagged nested tensor, bidirectional inside each document.
        len_a, _ = doc_lengths(seq)
        q0, k0, v0 = (t[0].transpose(0, 1) for t in (q, k, v))  # (T, H, d_head) for batch item 0

        def nest(t):  # two documents -> (2, H, j, d_head) jagged along the sequence dim
            return torch.nested.nested_tensor([t[:len_a], t[len_a:]], layout=torch.jagged).transpose(1, 2)

        with warnings.catch_warnings():
            # The CPU jagged SDPA path converts internally and warns that nested tensors are prototype.
            warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
            nested_out = F.scaled_dot_product_attention(nest(q0), nest(k0), nest(v0))
            nested_log = logging.getLogger("torch.nested._internal.sdpa")
            old_level = nested_log.level
            nested_log.setLevel(logging.ERROR)
            try:
                F.scaled_dot_product_attention(nest(q0), nest(k0), nest(v0), is_causal=True)
                nested_causal_supported = True
            except RuntimeError:  # 2.8 CPU: "No viable backend" for jagged + is_causal
                nested_causal_supported = False
            finally:
                nested_log.setLevel(old_level)
        bidir_doc = doc[:, None] == doc[None, :]
        packed_bidir, _ = manual_attention(q[:1], k[:1], v[:1], bidir_doc)
        packed_parts = (packed_bidir[0, :, :len_a], packed_bidir[0, :, len_a:])
        diffs["nested_jagged_vs_packed_bidirectional"] = max(
            max_diff(n, p) for n, p in zip(nested_out.unbind(), packed_parts)
        )

    # Gradients through manual vs SDPA (causal), same upstream gradient.
    gen = torch.Generator().manual_seed(seed + 1)
    upstream = torch.randn(q.shape, generator=gen)
    grads = {}
    for label in ("manual", "sdpa"):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        if label == "manual":
            out, _ = manual_attention(*leaves, masks["causal"])
        else:
            out = F.scaled_dot_product_attention(*leaves, is_causal=True)
        (out * upstream).sum().backward()
        grads[label] = [t.grad for t in leaves]
    grad_diff = max(max_diff(a, c) for a, c in zip(grads["manual"], grads["sdpa"]))

    timings = {}
    if time_it:
        causal = masks["causal"]
        cases = {
            "manual": lambda: manual_attention(q, k, v, causal),
            "sdpa_is_causal": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            "sdpa_bool_mask": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=causal),
            "sdpa_math": lambda: _sdpa_math(q, k, v),
            "mha": lambda: mha_forward(mha, x, "causal", causal),
            "flex_eager": lambda: flex_attention(q, k, v, block_mask=block_masks["causal"]),
        }
        with torch.no_grad():
            for name, fn in cases.items():
                fn()  # warm-up
                timer = benchmark.Timer(stmt="fn()", globals={"fn": fn}, num_threads=torch.get_num_threads())
                timings[name] = round(timer.timeit(20).median * 1e6, 1)

    result = {
        "max_abs_diff": diffs,
        "grad_max_abs_diff": grad_diff,
        "offset_chunk_is_causal_diff": offset_trap,
        "nested_causal_supported": nested_causal_supported,
        "timings_us": timings,
        "torch": torch.__version__.split("+")[0],
    }
    if return_tensors:
        result["tensors"] = {"q": q, "k": k, "v": v, "masks": masks, "outputs": outputs, "lengths": lengths}
    return result


def _sdpa_math(q, k, v):
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    if args.seq <= CHUNK + WINDOW:
        parser.error(f"--seq must exceed {CHUNK + WINDOW}")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = run(args.seed, args.batch, args.heads, args.seq, args.d_head)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
