"""LoRA from scratch: identity at init, merge equivalence, PEFT parity, and the 4-bit merge trap.

Purpose
    Module 06 (LoRA and QLoRA in Depth), labs L6.1 and L6.4. Implements a
    `LoRALinear` that wraps `nn.Linear`, injects it into a model by module
    name, and checks four contracts:
      1. step-0 identity (B = 0 makes the adapted model equal the base model),
      2. merged == unmerged, and unmerge restores W0,
      3. trainable-parameter count == sum of r * (d_in + d_out),
      4. merging into a 4-bit (NF4) base and re-quantizing loses far more of
         the adapter than merging into a high-precision copy of the base.
    Full mode adds parity with Hugging Face PEFT on a real model's q/v
    projections and the same merge experiment with real bitsandbytes NF4.

Full-mode procedure
    L6.1: load Qwen/Qwen3-0.6B in fp32, attach PEFT LoRA to q_proj/v_proj with
          init_lora_weights=False (random A and B, so parity is not trivially
          zero), copy lora_A/lora_B into LoRALinear layers, assert equal logits
          within 1e-4, then check merge_and_unload(). Record the
          print_trainable_parameters() line and the max logit difference.
    L6.4: (CUDA + bitsandbytes) compare a 4-bit QLoRA model against path A
          (dequantise the NF4 base to bf16, load the adapter, merge_and_unload())
          and path B (merge_and_unload() on the 4-bit model). Record both mean
          logit errors and sample a few generations from each.

Hardware
    --smoke: any CPU, under a minute, stdlib + torch only, no downloads.
    --full:  L6.1 parity runs on CPU or a free T4 (T0) with Qwen/Qwen3-0.6B;
             L6.4 needs a CUDA GPU with bitsandbytes (T0 is enough for 0.6B,
             T1 for larger models).

Pinned versions for --full (as of October 2026)
    torch>=2.8, transformers>=5.0, peft>=0.21.2, bitsandbytes>=0.48, accelerate

Run from the repository root
    python content/courses/finetuning/code/lora_from_scratch.py --smoke
    python content/courses/finetuning/code/lora_from_scratch.py --full --model Qwen/Qwen3-0.6B
"""

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 16 NF4 levels, copied from bitsandbytes `get_4bit_type("nf4")`
# (bitsandbytes/functional.py at commit 833649043474794b8fe7a4136e0c40faf077b2e0, L788-L805).
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W0 x + b + gamma * B (A (dropout(x))), gamma = alpha/r or alpha/sqrt(r)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0,
                 use_rslora: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / math.sqrt(r) if use_rslora else alpha / r
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))    # (r, d_in)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))   # (d_out, r): zero => no-op at step 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same as PEFT's default
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)    # (d_out, d_in)

    def forward(self, x):                                    # x: (..., d_in)
        y = self.base(x)                                     # (..., d_out)
        if self.merged:
            return y
        h = self.dropout(x) @ self.lora_A.T                  # (..., r)
        return y + self.scaling * (h @ self.lora_B.T)        # (..., d_out)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False


def inject_lora(model: nn.Module, targets, r: int, alpha: float, use_rslora: bool = False):
    """Replace every nn.Linear whose attribute name is in `targets` with a LoRALinear."""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r, alpha, use_rslora=use_rslora))
                replaced.append(f"{parent_name}.{child_name}".lstrip("."))
    return replaced


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


# ---------------------------------------------------------------------------
# A toy 2-layer decoder with Llama-style module names
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, d_kv, d_ff, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.kv_heads = d_kv // self.d_head
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d_kv, bias=False)
        self.v_proj = nn.Linear(d, d_kv, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):                                    # x: (B, T, d)
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        rep = self.n_heads // self.kv_heads                  # grouped-query attention
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d_head)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, -1))
        h = self.norm2(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class TinyDecoder(nn.Module):
    def __init__(self, vocab=64, d=64, d_kv=16, d_ff=224, n_layers=2, n_heads=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(Block(d, d_kv, d_ff, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):                                  # ids: (B, T)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))                    # (B, T, V)


ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# NF4 block quantization (simulated, same levels and block size as bitsandbytes)
# ---------------------------------------------------------------------------
def nf4_quant_dequant(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    levels = torch.tensor(NF4_LEVELS, dtype=torch.float32)
    flat = w.detach().float().reshape(-1, block)             # (n_blocks, 64)
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = ((flat / absmax).unsqueeze(-1) - levels).abs().argmin(-1)  # nearest level
    return (levels[idx] * absmax).reshape(w.shape)


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------
def toy_batch(gen, stride=3, vocab=64, batch=16, t=12):
    # A learnable rule: the next token is (current + stride) mod vocab.
    # "Pretraining" uses stride 3; the "fine-tuning task" switches to stride 5.
    start = torch.randint(0, vocab, (batch, 1), generator=gen)
    return (start + stride * torch.arange(t + 1)) % vocab   # (B, T+1)


def train_steps(model, params, steps, lr, gen, stride=3):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        seq = toy_batch(gen, stride)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def smoke():
    t0 = time.time()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(1)
    r, alpha = 4, 8
    # A small adapter LR keeps ||delta W|| at a few percent of ||W0||, as in real fine-tunes.
    # Larger deltas survive re-quantization better; the trap is worst for small, careful updates.
    lora_lr = 3e-4

    # Pretrain the toy base briefly so its weights are not pure noise.
    base = TinyDecoder()
    train_steps(base, base.parameters(), steps=60, lr=3e-3, gen=gen)
    base.eval()
    probe = toy_batch(torch.Generator().manual_seed(2), stride=5)[:, :-1]
    with torch.no_grad():
        base_logits = base(probe)

    # 1. Step-0 identity.
    model = copy.deepcopy(base)
    replaced = inject_lora(model, ALL_LINEAR, r=r, alpha=alpha)
    assert len(replaced) == 2 * len(ALL_LINEAR), replaced
    with torch.no_grad():
        assert torch.equal(model(probe), base_logits), "B=0 must make LoRA a no-op"
    print(f"[1] step-0 identity holds on {len(replaced)} adapted layers")

    # 2. Parameter count matches r * (d_in + d_out) summed over adapted layers.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formula = sum(r * (m.base.in_features + m.base.out_features) for m in lora_layers(model))
    per_layer = r * (2 * (64 + 64) + 2 * (64 + 16) + 3 * (64 + 224))  # q,o + k,v + gate,up,down
    assert trainable == formula == 2 * per_layer, (trainable, formula, per_layer)
    print(f"[2] trainable params {trainable:,} == formula {formula:,}")

    # Fine-tune the adapter only; base weights must not move.
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    lora_params = [p for p in model.parameters() if p.requires_grad]
    losses = train_steps(model, lora_params, steps=60, lr=lora_lr, gen=gen, stride=5)
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, frozen_before[n]), f"frozen weight moved: {n}"
    print(f"[3] adapter-only training: loss {losses[0]:.3f} -> {losses[-1]:.3f}; base untouched")

    # 3. merged == unmerged, unmerge restores W0.
    model.eval()
    with torch.no_grad():
        unmerged = model(probe)
        w0 = {id(m): m.base.weight.clone() for m in lora_layers(model)}
        for m in lora_layers(model):
            m.merge()
        merged = model(probe)
        torch.testing.assert_close(merged, unmerged, atol=1e-5, rtol=1e-5)
        for m in lora_layers(model):
            m.unmerge()
            torch.testing.assert_close(m.base.weight, w0[id(m)], atol=1e-6, rtol=0)
    print(f"[4] merged vs unmerged max |diff| = {(merged - unmerged).abs().max():.2e}; unmerge restores W0")

    # 4. QLoRA-style: base frozen in NF4, adapter trained against the dequantized base.
    qmodel = copy.deepcopy(base)
    with torch.no_grad():
        for name, mod in qmodel.named_modules():
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in ALL_LINEAR:
                mod.weight.copy_(nf4_quant_dequant(mod.weight))
    inject_lora(qmodel, ALL_LINEAR, r=r, alpha=alpha)
    qparams = [p for p in qmodel.parameters() if p.requires_grad]
    train_steps(qmodel, qparams, steps=60, lr=lora_lr, gen=gen, stride=5)
    qmodel.eval()
    with torch.no_grad():
        reference = qmodel(probe)            # what the trained QLoRA model computes

        path_a = copy.deepcopy(qmodel)       # merge into a dequantized copy, store in bf16
        for m in lora_layers(path_a):
            m.merge()
            m.base.weight.copy_(m.base.weight.to(torch.bfloat16).float())
        err_a = (path_a(probe) - reference).abs().max().item()

        path_b = copy.deepcopy(qmodel)       # merge, then re-quantize to NF4
        cos, ratios = [], []
        for m in lora_layers(path_b):
            q_before = m.base.weight.clone()
            delta = m.delta_weight()
            m.merge()
            m.base.weight.copy_(nf4_quant_dequant(m.base.weight))
            kept = m.base.weight - q_before
            cos.append(F.cosine_similarity(kept.flatten(), delta.flatten(), dim=0).item())
            ratios.append((delta.norm() / q_before.norm()).item())
        err_b = (path_b(probe) - reference).abs().max().item()
    assert err_b > 10 * err_a, (err_a, err_b)
    assert sum(cos) / len(cos) < 0.9, cos
    print(f"[5] merge error vs QLoRA reference: bf16 copy {err_a:.2e}, re-quantised NF4 {err_b:.2e} "
          f"({err_b / err_a:.0f}x)\n    mean ||delta||/||W0|| = {sum(ratios) / len(ratios):.3f}; "
          f"mean cos(delta kept after re-quantisation, delta) = {sum(cos) / len(cos):.2f}")

    print(f"smoke OK in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Full mode (imports inside the function so --smoke never needs them)
# ---------------------------------------------------------------------------
def full(model_id: str, r: int, alpha: int, skip_4bit: bool):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    batch = tok(["Customer Ana Ruiz, order 88412, was charged twice and wants a refund.",
                 "Order 1009 arrived broken; Lee is upset but does not want money back."],
                return_tensors="pt", padding=True).to(device)

    # L6.1: PEFT parity on q_proj/v_proj in fp32.
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    mine = copy.deepcopy(base)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, init_lora_weights=False)  # random A and B: non-trivial delta
    peft_model = get_peft_model(base, cfg).eval()
    peft_model.print_trainable_parameters()

    inject_lora(mine, ("q_proj", "v_proj"), r=r, alpha=alpha)
    sd = peft_model.state_dict()
    with torch.no_grad():
        for name, mod in mine.named_modules():
            if isinstance(mod, LoRALinear):
                mod.lora_A.copy_(sd[f"base_model.model.{name}.lora_A.default.weight"])
                mod.lora_B.copy_(sd[f"base_model.model.{name}.lora_B.default.weight"])
        ref = peft_model(**batch).logits                   # (B, T, V)
        out = mine(**batch).logits
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        merged = peft_model.merge_and_unload()             # returns the base model; assign it
        torch.testing.assert_close(merged(**batch).logits, ref, atol=1e-3, rtol=1e-3)
    print("[full L6.1] LoRALinear == PEFT LoRA on q/v; merge_and_unload preserves logits")
    del merged, peft_model, mine, base

    if skip_4bit or device != "cuda":
        print("[full L6.4] skipped (needs CUDA + bitsandbytes)")
        return

    # L6.4: QLoRA reference vs two merge paths, with real bitsandbytes NF4.
    import tempfile

    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules="all-linear", init_lora_weights=False)
    with tempfile.TemporaryDirectory() as tmp:
        q_base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map=device)
        qlora = get_peft_model(q_base, cfg)
        with torch.no_grad():                               # shrink the random adapter to a realistic size
            for n, p in qlora.named_parameters():
                if "lora_B" in n:
                    p.mul_(1e-2)
        qlora.save_pretrained(tmp)                          # adapter_config.json + adapter_model.safetensors
        with torch.no_grad():
            ref = qlora(**batch).logits.float()

            # Path A: dequantize the NF4 base to bf16, then merge (what training actually saw).
            deq = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map=device)
            deq = deq.dequantize(dtype=torch.bfloat16)
            path_a = PeftModel.from_pretrained(deq, tmp).merge_and_unload()
            err_a = (path_a(**batch).logits.float() - ref).abs().mean().item()
            del path_a, deq

            # Path B: merge straight into the 4-bit weights (PEFT dequantizes, adds, re-quantizes).
            path_b = qlora.merge_and_unload()
            err_b = (path_b(**batch).logits.float() - ref).abs().mean().item()
    print(f"[full L6.4] mean |logit error| vs QLoRA reference: bf16 merge {err_a:.3e}, 4-bit merge {err_b:.3e}")
    if err_b <= err_a:
        print("warning: expected the 4-bit merge to be worse; try a smaller adapter scale")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU-only toy run (default)")
    mode.add_argument("--full", action="store_true", help="PEFT parity + bitsandbytes merge on a real model")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--skip-4bit", action="store_true")
    args = ap.parse_args()
    if args.full:
        full(args.model, args.r, args.alpha, args.skip_4bit)
    else:
        smoke()


if __name__ == "__main__":
    main()
