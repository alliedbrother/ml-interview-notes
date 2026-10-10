"""The capstone GPT: a GPT-2-style decoder written for reading (module 32).

Pre-LayerNorm blocks, learned absolute positions, exact GELU, biases on, a tied
output head and a static, preallocated KV cache. The capstone configuration
(V=512, block_size=128, L=4, H=4, d_model=128) has 875,264 parameters.

Frozen interface (syllabus E.4) used by modules 33-38:
    GPTConfig, KVCache, CausalSelfAttention, MLP, Block, GPT, causal_mask
    GPT.forward(idx, targets=None, cache=None, attn_impl="sdpa") -> (logits, loss)
    GPT.num_params, GPT.configure_optimizer, GPT.param_table, GPT.shape_trace
Attribute names wte, wpe, blocks, ln_f, lm_head, c_attn, c_proj, c_fc are frozen.

Imports torch only. Read it next to content/courses/pytorch/32-capstone-gpt-model.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
ATTN_IMPLS = ("sdpa", "manual")


# ---------------------------------------------------------------- config

@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128   # context length; learned positions cannot go past it
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> "GPTConfig":
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "d_model"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout!r}")
        if not isinstance(self.bias, bool):
            raise ValueError("bias must be a bool")
        return self

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


# ---------------------------------------------------------------- masks and cache

def causal_mask(q_len: int, k_len: int, offset: int = 0, device=None) -> torch.Tensor:
    """Boolean (q_len, k_len) mask, True = may attend.

    Query i sits at absolute position offset + i; key j at position j.
    Query i may see key j iff j <= offset + i. With offset = k_len - q_len this is
    the bottom-right-aligned triangle a cached chunk needs; is_causal=True in SDPA
    always builds the top-left-aligned one (offset = 0).
    """
    if q_len <= 0 or k_len <= 0 or offset < 0:
        raise ValueError("q_len and k_len must be positive and offset non-negative")
    q_pos = torch.arange(offset, offset + q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    return k_pos[None, :] <= q_pos[:, None]


class KVCache:
    """Static, preallocated keys/values for every layer: (L, B, H, max_len, d_head).

    `length` counts positions already written. GPT.forward writes the new chunk's
    keys/values at [length : length + T_new] in every layer and advances `length`
    once, after the last layer. Meant for inference (no_grad / inference_mode).
    """

    def __init__(self, cfg: GPTConfig, batch_size: int, max_len: int,
                 dtype: torch.dtype = torch.float32, device="cpu"):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_len, int) or not 0 < max_len <= cfg.block_size:
            raise ValueError(f"max_len must lie in [1, block_size={cfg.block_size}], got {max_len!r}")
        shape = (cfg.n_layer, batch_size, cfg.n_head, max_len, cfg.d_head)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.batch_size = batch_size
        self.max_len = max_len

    def reset(self) -> None:
        self.length = 0  # stale entries beyond `length` are never read

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


# ---------------------------------------------------------------- layers

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head, self.d_head, self.d_model = cfg.n_head, cfg.d_head, cfg.d_model
        self.layer_idx = layer_idx
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)  # fused q, k, v
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)      # residual projection
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None, cache: KVCache | None = None, attn_impl: str = "sdpa"):
        # x: (B, T, d_model). mask: None (plain causal) or bool broadcastable to (B, H, T, T_k).
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)                       # 3 x (B, T, d_model)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)    # (B, H, T, d_head)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        if cache is not None:
            start, end = cache.length, cache.length + T
            cache.k[self.layer_idx, :, :, start:end] = k
            cache.v[self.layer_idx, :, :, start:end] = v
            k = cache.k[self.layer_idx, :, :, :end]                    # (B, H, T_k, d_head)
            v = cache.v[self.layer_idx, :, :, :end]
        p = self.dropout if self.training else 0.0
        if attn_impl == "sdpa":
            if mask is None:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:  # "manual": softmax(q k^T / sqrt(d_head) + mask) v, spelled out
            if mask is None:
                mask = causal_mask(T, k.size(-2), k.size(-2) - T, device=x.device)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))  # (B, H, T, T_k)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.dropout(att.softmax(dim=-1), p=p, training=self.training)
            y = att @ v                                                # (B, H, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)               # merge heads
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.gelu = nn.GELU()  # exact erf form, approximate="none"
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN block: x + attn(ln_1(x)), then x + mlp(ln_2(x))."""

    def __init__(self, cfg: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, mask=None, cache: KVCache | None = None, attn_impl: str = "sdpa"):
        x = x + self.attn(self.ln_1(x), mask=mask, cache=cache, attn_impl=attn_impl)
        return x + self.mlp(self.ln_2(x))


# ---------------------------------------------------------------- model

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: one Parameter, two uses
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)  # two residual adds per block
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=resid_std)

    @property
    def cfg(self) -> GPTConfig:
        return self.config

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache: KVCache | None = None, attn_impl: str = "sdpa",
                *, position_ids=None, key_mask=None):
        """idx (B, T) int64 -> (logits (B, T, V), loss or None).

        cache:        KVCache; positions start at cache.length and the chunk attends to
                      every cached key through an explicit offset mask.
        position_ids: optional (B, T) int64 overriding start + arange(T) (left padding).
        key_mask:     optional (B, T_k) bool, True = real token, T_k = (cache.length) + T.
        """
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {attn_impl!r}")
        if idx.dim() != 2:
            raise ValueError(f"idx must be (B, T), got shape {tuple(idx.shape)}")
        B, T = idx.shape
        start = 0 if cache is None else cache.length
        if start + T > self.config.block_size:
            raise ValueError(f"sequence of {start + T} positions exceeds block_size={self.config.block_size}")
        if cache is not None:
            if start + T > cache.max_len:
                raise ValueError(f"cache overflow: {start} cached + {T} new > max_len={cache.max_len}")
            if B != cache.batch_size:
                raise ValueError(f"batch {B} does not match cache batch_size={cache.batch_size}")

        if position_ids is None:
            pos = torch.arange(start, start + T, device=idx.device)  # (T,)
        else:
            pos = position_ids                                        # (B, T)
        x = self.drop(self.wte(idx) + self.wpe(pos))                  # (B, T, d_model)

        mask = None  # None -> SDPA is_causal=True; valid only for an unoffset square chunk
        if cache is not None or key_mask is not None:
            T_k = start + T
            mask = causal_mask(T, T_k, start, device=idx.device)      # (T, T_k)
            if key_mask is not None:
                if key_mask.shape != (B, T_k):
                    raise ValueError(f"key_mask must be (B, {T_k}), got {tuple(key_mask.shape)}")
                diag = torch.arange(T_k, device=idx.device)[None, :] == torch.arange(start, T_k, device=idx.device)[:, None]
                # a padded query would otherwise see no key at all (softmax of all -inf = NaN)
                mask = (mask & key_mask[:, None, None, :].bool()) | diag  # (B, 1, T, T_k)

        for block in self.blocks:
            x = block(x, mask=mask, cache=cache, attn_impl=attn_impl)
        if cache is not None:
            cache.length = start + T  # advance once, after every layer has written
        logits = self.lm_head(self.ln_f(x))                           # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss

    # ------------------------------------------------------------ accounting

    def num_params(self, non_embedding: bool = False) -> int:
        """Unique parameters (the tied head counts once). non_embedding subtracts wpe only:
        wte is also the output projection, so its matmul FLOPs are real (nanoGPT convention)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def param_table(self) -> list[dict]:
        """One row per parameter-owning leaf module; rows sum to num_params()."""
        rows, seen = [], {}
        for name, module in self.named_modules():
            own = list(module.named_parameters(recurse=False))
            if not own:
                continue
            numel, shapes, tied_to = 0, {}, None
            for pname, p in own:
                shapes[pname] = list(p.shape)
                if id(p) in seen:
                    tied_to = seen[id(p)]
                    continue
                seen[id(p)] = f"{name}.{pname}"
                numel += p.numel()
            row = {"name": name, "type": type(module).__name__, "shapes": shapes, "numel": numel}
            if tied_to:
                row["tied_to"] = tied_to
            rows.append(row)
        return rows

    def configure_optimizer(self, weight_decay: float, lr: float, betas=(0.9, 0.95),
                            eps: float = 1e-8) -> torch.optim.AdamW:
        """AdamW with decay on matrices (dim >= 2) and none on biases / LayerNorm."""
        params = dict(self.named_parameters())  # remove_duplicate=True: tied weight once
        decay = [p for p in params.values() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in params.values() if p.requires_grad and p.dim() < 2]
        grouped = {id(p) for p in decay} | {id(p) for p in no_decay}
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        assert grouped == trainable and len(decay) + len(no_decay) == len(trainable), \
            "every trainable parameter must be in exactly one group"
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps,
                                 foreach=True, fused=False)

    @torch.no_grad()
    def shape_trace(self, idx) -> list[tuple[str, tuple]]:
        """Shapes of every intermediate tensor of an uncached forward (block 0 in detail)."""
        cfg = self.config
        B, T = idx.shape
        out = [("idx", tuple(idx.shape))]
        tok = self.wte(idx)
        pos = self.wpe(torch.arange(T, device=idx.device))
        x = tok + pos
        out += [("wte(idx)", tuple(tok.shape)), ("wpe(arange(T))", tuple(pos.shape)),
                ("x = tok + pos", tuple(x.shape))]
        for i, block in enumerate(self.blocks):
            if i == 0:
                a = block.attn
                h = block.ln_1(x)
                qkv = a.c_attn(h)
                q, k, v = qkv.split(cfg.d_model, dim=2)
                q = q.view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)
                k = k.view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)
                v = v.view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.d_head)
                att = att.masked_fill(~causal_mask(T, T, 0, device=idx.device), float("-inf")).softmax(-1)
                y = att @ v
                merged = y.transpose(1, 2).contiguous().view(B, T, cfg.d_model)
                x = x + a.c_proj(merged)
                fc = block.mlp.c_fc(block.ln_2(x))
                x = x + block.mlp.c_proj(block.mlp.gelu(fc))
                out += [("blocks.0.ln_1(x)", tuple(h.shape)), ("blocks.0.attn.c_attn", tuple(qkv.shape)),
                        ("q, k, v split + view + transpose", tuple(q.shape)),
                        ("scores q @ k^T / sqrt(d_head)", tuple(att.shape)),
                        ("softmax(scores) @ v", tuple(y.shape)),
                        ("heads merged", tuple(merged.shape)),
                        ("x + attn.c_proj(...)", tuple(x.shape)),
                        ("blocks.0.mlp.c_fc / gelu", tuple(fc.shape)),
                        ("x + mlp.c_proj(...)", tuple(x.shape))]
            else:
                x = block(x)
                out.append((f"blocks.{i}(x)", tuple(x.shape)))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        out += [("ln_f(x)", tuple(x.shape)), ("lm_head -> logits", tuple(logits.shape))]
        return out
