"""
A complete, runnable modern decoder-only Transformer.

Combines the components built across the course:
  RMSNorm (module 06) + RoPE (module 05) + GQA (module 09)
  + SwiGLU (module 07) + MoE with shared experts (module 12)
  + pre-norm residual block (module 06) + KV cache (module 11)

Written for clarity, not speed. Run directly:  python modern_decoder.py
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config

@dataclass
class Config:
    vocab_size: int = 1000
    d_model: int = 256          # residual stream width
    n_layers: int = 4
    n_heads: int = 8            # query heads
    n_kv_heads: int = 2         # key/value heads  -> GQA group size 4
    d_head: int = 32
    d_ff: int = 512             # dense FFN width (used in dense layers)
    # --- MoE ---
    n_experts: int = 8
    n_experts_active: int = 2   # top-k
    n_shared_experts: int = 1
    d_expert: int = 128         # fine-grained: smaller than d_ff
    first_k_dense: int = 1      # first N layers are dense (module 12)
    # --- misc ---
    rope_theta: float = 10000.0
    max_seq_len: int = 512
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                     "d_head", "d_ff", "d_expert", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_heads % self.n_kv_heads or self.d_head % 2:
            raise ValueError("query heads must divide into KV groups; RoPE head width must be even")
        if not 0 <= self.first_k_dense <= self.n_layers:
            raise ValueError("first_k_dense must lie between zero and n_layers")
        for name in ("n_experts", "n_experts_active", "n_shared_experts", "first_k_dense"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.n_experts < 0 or self.n_shared_experts < 0:
            raise ValueError("expert counts cannot be negative")
        if self.n_experts and not 1 <= self.n_experts_active <= self.n_experts:
            raise ValueError("active routed experts must lie between one and n_experts")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")


# ---------------------------------------------------------------- module 06

class RMSNorm(nn.Module):
    """Module 06: no mean-centering, no bias, one reduction pass."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                     # fp32 for stability
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


# ---------------------------------------------------------------- module 05

def build_rope_cache(seq_len, d_head, theta=10000.0, device=None):
    """Precompute cos/sin. Built once, shared by every layer."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # (T, d_head/2)
    emb = torch.cat([freqs, freqs], dim=-1)               # (T, d_head)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, H, T, d_head);  cos/sin: (T, d_head)"""
    return (x * cos[None, None]) + (rotate_half(x) * sin[None, None])


# ---------------------------------------------------------------- module 11

class KVCache:
    """Module 11: append-only cache of the SMALL (n_kv_heads) tensors."""

    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k, v):
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError("cache expects matching (batch, heads, tokens, width) tensors")
        if self.k is None:
            self.k, self.v = k, v
        else:
            if (k.shape[:2] + k.shape[3:] != self.k.shape[:2] + self.k.shape[3:]
                    or k.device != self.k.device or k.dtype != self.k.dtype):
                raise ValueError("cache batch, heads, width, device and dtype must match")
            self.k = torch.cat([self.k, k], dim=2)        # grow along T
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self):
        return 0 if self.k is None else self.k.shape[2]


# ---------------------------------------------------------------- module 09

class GroupedQueryAttention(nn.Module):
    """Module 09: fewer K/V heads than Q heads, repeated to match."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.H, self.H_kv = cfg.n_heads, cfg.n_kv_heads
        self.group_size = cfg.n_heads // cfg.n_kv_heads
        self.d_head = cfg.d_head

        self.W_q = nn.Linear(cfg.d_model, self.H    * self.d_head, bias=False)
        self.W_k = nn.Linear(cfg.d_model, self.H_kv * self.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, self.H_kv * self.d_head, bias=False)
        self.W_o = nn.Linear(self.H * self.d_head, cfg.d_model, bias=False)

        # module 06: QK-Norm
        self.q_norm = RMSNorm(self.d_head, cfg.norm_eps)
        self.k_norm = RMSNorm(self.d_head, cfg.norm_eps)

    def forward(self, x, cos, sin, cache: "KVCache | None" = None):
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.H,    self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.H_kv, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.H_kv, self.d_head).transpose(1, 2)

        q = self.q_norm(q)                                # QK-Norm BEFORE RoPE
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)                     # cache SMALL tensors

        # expand K/V to match query heads (module 09)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Query i is at absolute cache position past+i, not position i.
        past = k.shape[2] - T
        allowed = (torch.arange(k.shape[2], device=x.device)[None, :]
                   <= past + torch.arange(T, device=x.device)[:, None])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        out = out.transpose(1, 2).reshape(B, T, self.H * self.d_head)
        return self.W_o(out)


# ---------------------------------------------------------------- module 07

class SwiGLU(nn.Module):
    """Module 07: three matrices, gated."""

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up   = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------- module 12

class MoE(nn.Module):
    """Module 12: top-k routed experts + always-on shared experts."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.n_experts_active
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_shared_experts)]
        )

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)                           # (B*T, D)

        logits = self.gate(flat)                          # (N, n_experts)
        # Full-softmax probabilities retain a task-loss router gradient at k=1.
        topk_w, topk_idx = F.softmax(logits, dim=-1).topk(self.top_k, dim=-1)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (topk_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, topk_w[tok, slot, None] * expert(flat[tok]))

        for expert in self.shared:                        # always active
            out = out + expert(flat)

        return out.view(B, T, D), logits


def load_balancing_loss(logits, topk_idx, n_experts, alpha=0.01):
    """Module 12: couples non-differentiable usage to differentiable prob."""
    P = F.softmax(logits, dim=-1).mean(dim=0)
    f = F.one_hot(topk_idx, n_experts).float().mean(dim=(0, 1))
    return alpha * n_experts * torch.sum(f * P)


# ---------------------------------------------------------------- module 06

class Block(nn.Module):
    """Pre-norm block. Residual stream stays clean."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn  = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.is_moe = cfg.n_experts > 0 and layer_idx >= cfg.first_k_dense
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        if self.is_moe:
            ffn_out, router_logits = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, router_logits
        x = x + self.ffn(self.norm2(x))
        return x, None


# ---------------------------------------------------------------- the model

class ModernDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight           # weight tying (module 02)

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_head, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Standard GPT-style init. Without this, logits at initialization are
        # enormous and the first loss is nowhere near the expected ln(vocab).
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, caches=None, pos_offset=0):
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("ids must be a nonempty, unpadded (batch, tokens) tensor")
        B, T = ids.shape
        if not isinstance(pos_offset, int) or pos_offset < 0 or pos_offset + T > self.cfg.max_seq_len:
            raise ValueError("positions exceed the configured context window")
        if caches is not None:
            if len(caches) != len(self.blocks) or any(c.length != pos_offset for c in caches):
                raise ValueError("one cache per layer is required, each matching pos_offset")
        x = self.embed(ids)                               # (B, T, d_model)

        cos = self.cos[pos_offset:pos_offset + T]
        sin = self.sin[pos_offset:pos_offset + T]

        all_router_logits = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, rl = block(x, cos, sin, cache)
            if rl is not None:
                all_router_logits.append(rl)

        x = self.norm_f(x)
        return self.lm_head(x), all_router_logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=20, temperature=0.8, eos_token_id=None):
        """Unpadded, equal-length prompts; finished batch rows repeat EOS."""
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if prompt_ids.ndim != 2 or min(prompt_ids.shape) <= 0:
            raise ValueError("a nonempty (batch, tokens) prompt is required")
        if prompt_ids.shape[1] + max_new_tokens > self.cfg.max_seq_len:
            raise ValueError("prompt plus requested output exceeds max_seq_len")
        if eos_token_id is not None and not 0 <= eos_token_id < self.cfg.vocab_size:
            raise ValueError("eos_token_id must be a valid vocabulary ID")
        if max_new_tokens == 0:
            return prompt_ids.clone()
        self.eval()
        caches = [KVCache() for _ in self.blocks]

        # --- PREFILL: whole prompt in one pass, causal ---
        logits, _ = self.forward(prompt_ids, caches, pos_offset=0)
        pos = prompt_ids.shape[1]
        out = [prompt_ids]

        finished = torch.zeros(prompt_ids.shape[0], 1, dtype=torch.bool, device=prompt_ids.device)
        for step in range(max_new_tokens):
            next_id = self._sample(logits[:, -1], temperature)
            if eos_token_id is not None:
                next_id = torch.where(finished, eos_token_id, next_id)
                finished |= next_id == eos_token_id
            out.append(next_id)
            if finished.all() or step == max_new_tokens - 1:
                break
            logits, _ = self.forward(next_id, caches, pos_offset=pos)
            pos += 1

        return torch.cat(out, dim=1)

    @staticmethod
    def _sample(logits, temperature):
        if temperature == 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits.double()
        probs = F.softmax((logits - logits.amax(-1, keepdim=True)) / temperature, dim=-1)
        return torch.multinomial(probs, 1)


# ---------------------------------------------------------------- checks

def _report(cfg, model):
    total = sum(p.numel() for p in model.parameters())
    expert_params = sum(
        p.numel() for n, p in model.named_parameters()
        if ".ffn.experts." in n
    )
    shared_params = sum(
        p.numel() for n, p in model.named_parameters()
        if ".ffn.shared." in n
    )
    n_moe_layers = sum(block.is_moe for block in model.blocks)
    per_layer_experts = expert_params / max(n_moe_layers, 1)
    active_experts = per_layer_experts * cfg.n_experts_active / max(cfg.n_experts, 1)
    active = total - expert_params + active_experts * n_moe_layers

    print(f"  total parameters   : {total:,}")
    print(f"  routed-expert params: {expert_params:,}")
    print(f"  shared-expert params: {shared_params:,}")
    print(f"  active per token   : {active:,.0f}  ({100*active/total:.1f}%)")


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = Config()
    model = ModernDecoder(cfg)

    print("=== model ===")
    _report(cfg, model)

    print("\n=== forward pass ===")
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, router_logits = model(ids)
    print(f"  input  {tuple(ids.shape)}  ->  logits {tuple(logits.shape)}")
    assert logits.shape == (2, 16, cfg.vocab_size)
    print(f"  MoE layers producing router logits: {len(router_logits)}"
          f"  (expected {cfg.n_layers - cfg.first_k_dense})")
    assert len(router_logits) == cfg.n_layers - cfg.first_k_dense

    print("\n=== training step ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                         ids[:, 1:].reshape(-1))
    aux = sum(
        load_balancing_loss(rl, rl.topk(cfg.n_experts_active, -1).indices, cfg.n_experts)
        for rl in router_logits
    )
    (ce + aux).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"  cross-entropy {ce.item():.4f}  (expected approx ln(V) = "
          f"{math.log(cfg.vocab_size):.4f} at init)")
    print(f"  aux loss {aux.item():.4f}   grad-norm {grad_norm:.3f}")

    print("\n=== generation with KV cache ===")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    gen = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  prompt {tuple(prompt.shape)}  ->  output {tuple(gen.shape)}")
    assert gen.shape == (1, 15)

    print("\n=== cached vs uncached equivalence ===")
    torch.manual_seed(1)
    ctx = torch.randint(0, cfg.vocab_size, (1, 8))
    # uncached: full sequence in one pass
    full_logits, _ = model(ctx)
    last_uncached = full_logits[:, -1]
    # cached: prefill 7 tokens, then decode the 8th
    caches = [KVCache() for _ in model.blocks]
    model(ctx[:, :7], caches, pos_offset=0)
    step_logits, _ = model(ctx[:, 7:8], caches, pos_offset=7)
    last_cached = step_logits[:, -1]
    diff = (last_uncached - last_cached).abs().max().item()
    print(f"  max abs difference: {diff:.2e}")
    assert diff < 1e-3, "KV cache path diverged from full forward"

    print("\nAll checks passed.")
