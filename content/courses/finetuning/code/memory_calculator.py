"""Will it fit? A bucket-by-bucket GPU memory estimator for LLM fine-tuning.

Module: 03 — Training Mechanics: Optimizers, Schedules, and Memory Math (Lab L3.1).

What it does
    Estimates weights, gradients, optimizer state, activations, logits and a
    fixed runtime overhead (in GB = 1e9 bytes) for full fine-tuning, LoRA and
    QLoRA, from a model's config.json shapes. Presets carry shapes copied from
    the pinned config.json files listed in PRESETS below.

Hardware tiers
    --smoke : CPU, stdlib only, under a second. Prints the table for the 8B
              worked example and asserts the module's numbers.
    --full  : T1 (one 24 GB Ampere+ GPU). Loads Qwen/Qwen3-1.7B, runs one real
              training step with LoRA (or full FT with --method full on a
              larger GPU) and compares torch.cuda.max_memory_allocated() with
              the estimate.

Pinned versions for --full (as of October 2026)
    torch>=2.8, transformers==5.19.0, peft>=0.21

Run (from the repository root)
    python content/courses/finetuning/code/memory_calculator.py --smoke
    python content/courses/finetuning/code/memory_calculator.py --preset qwen3-4b --method qlora --seq-len 4096
    python content/courses/finetuning/code/memory_calculator.py --full --method lora --seq-len 2048

Assumptions (state them whenever you quote a number from this tool)
    * Full FT: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + optimizer state
      (AdamW: 4 + 4). --grad-accum-buffer adds a 2-byte buffer (the "18" route).
    * LoRA: frozen base 2 bytes/param; adapters carry the full 16 bytes/param.
    * QLoRA: base at 4.127 bits/param (NF4 + double-quantized absmax, module 06);
      embeddings and norms are treated as quantized too (slight underestimate).
    * Activations: Korthikanti et al. (2022) per-layer estimate
      s*b*h*(34 + 5*a*s/h) bytes in 16-bit; FlashAttention drops the 5as/h term.
      With checkpointing we keep 2*s*b*h bytes per layer (block inputs) plus one
      layer's internals during recompute. The constant 34 was derived for a
      GPT-style MLP (4h); SwiGLU models differ, so treat it as +-30%.
    * Logits: B*T*V*4 bytes (fp32 upcast) unless --fused-ce.
    * Fixed overhead: 1.5 GB for CUDA context, allocator fragmentation, buffers.
"""

import argparse
from dataclasses import dataclass


GB = 1e9

# Shapes copied from config.json at the pinned revisions (fetched Oct 2026):
#   Qwen/Qwen3-1.7B   @70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
#   Qwen/Qwen3-4B     @1cfa9a7208912126459214e8b04321603b3df60c
#   unsloth/Meta-Llama-3.1-8B  @e9a141a2091ea561b96483212645a2a05e6f99fc
#   Qwen/Qwen3-32B    @9216db5781bf21249d130ec9da846c4624c16137
#   unsloth/Meta-Llama-3.1-70B @1b7306651142d0cc65d993076a250a6a82cf046c
#   unsloth/Llama-3.2-1B @9535bd9b1d1dea6acafbdc4813b728796aeb28da
# `nominal` is the marketing parameter count used for the course's round-number
# examples; `--exact-params` uses the shape-derived count instead.


@dataclass(frozen=True)
class Config:
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool
    nominal: float


PRESETS = {
    "llama-3.2-1b-like": Config(2048, 8192, 16, 32, 8, 64, 128256, True, 1e9),
    "qwen3-1.7b": Config(2048, 6144, 28, 16, 8, 128, 151936, True, 1.7e9),
    "qwen3-4b": Config(2560, 9728, 36, 32, 8, 128, 151936, True, 4e9),
    "llama-3.1-8b-like": Config(4096, 14336, 32, 32, 8, 128, 128256, False, 8e9),
    "qwen3-32b-like": Config(5120, 25600, 64, 64, 8, 128, 151936, False, 32e9),
    "llama-3.1-70b-like": Config(8192, 28672, 80, 64, 8, 128, 128256, False, 70e9),
}

ALL_LINEAR = ("q", "k", "v", "o", "gate", "up", "down")


def linear_shapes(c):
    """(name, d_in, d_out) for every linear layer in one decoder block."""
    q_out = c.heads * c.head_dim
    kv_out = c.kv_heads * c.head_dim
    return [
        ("q", c.hidden, q_out), ("k", c.hidden, kv_out), ("v", c.hidden, kv_out),
        ("o", q_out, c.hidden), ("gate", c.hidden, c.inter),
        ("up", c.hidden, c.inter), ("down", c.inter, c.hidden),
    ]


def exact_params(c):
    per_layer = sum(i * o for _, i, o in linear_shapes(c)) + 2 * c.hidden
    emb = c.vocab * c.hidden
    head = 0 if c.tied else c.vocab * c.hidden
    return c.layers * per_layer + emb + head + c.hidden


def lora_params(c, r, targets=ALL_LINEAR):
    return c.layers * sum(r * (i + o) for n, i, o in linear_shapes(c) if n in targets)


def optimizer_bytes_per_param(name):
    return {"adamw": 8.0, "adamw_8bit": 2.0, "lion": 4.0, "sgd": 0.0}[name]


def adafactor_state_bytes(c):
    """Factored second moment: fp32 row + column vectors per matrix, no first moment."""
    per_layer = sum(4 * (i + o) for _, i, o in linear_shapes(c))
    return c.layers * per_layer + 4 * (c.vocab + c.hidden) * (1 if c.tied else 2)


def activation_bytes(c, batch, seq, flash=True, checkpointing=True):
    sbh = seq * batch * c.hidden
    per_layer = sbh * 34 + (0 if flash else sbh * 5 * c.heads * seq / c.hidden)
    if checkpointing:
        return c.layers * 2 * sbh + per_layer
    return c.layers * per_layer


def estimate(c, method="lora", r=16, targets=ALL_LINEAR, batch=1, seq=2048,
             optimizer="adamw", checkpointing=True, flash=True, fused_ce=False,
             grad_accum_buffer=False, exact=False, overhead_gb=1.5):
    n = exact_params(c) if exact else c.nominal
    out = {"adapter_weights": 0}
    if method == "full":
        out["weights"] = 2 * n
        out["grads"] = 2 * n + (2 * n if grad_accum_buffer else 0)
        out["master"] = 4 * n
        out["optimizer"] = (adafactor_state_bytes(c) if optimizer == "adafactor"
                            else optimizer_bytes_per_param(optimizer) * n)
        adapters = 0
    else:
        bits = 16 if method == "lora" else 4.127
        out["weights"] = n * bits / 8
        adapters = lora_params(c, r, targets)
        out["grads"] = 2 * adapters
        out["adapter_weights"] = 2 * adapters  # bf16 adapter copy used in forward
        out["master"] = 4 * adapters           # fp32 master
        opt = 8.0 if optimizer == "adafactor" else optimizer_bytes_per_param(optimizer)
        out["optimizer"] = opt * adapters
    out["activations"] = activation_bytes(c, batch, seq, flash, checkpointing)
    out["logits"] = 0 if fused_ce else batch * seq * c.vocab * 4
    out["overhead"] = overhead_gb * GB
    gb = {k: v / GB for k, v in out.items()}
    gb["state_total"] = sum(gb[k] for k in ("weights", "adapter_weights", "grads", "master", "optimizer"))
    gb["total"] = sum(gb[k] for k in out)
    gb["adapter_params"] = adapters
    return gb


def print_table(title, gb):
    print(f"\n{title}")
    for k in ("weights", "adapter_weights", "grads", "master", "optimizer", "state_total",
              "activations", "logits", "overhead", "total"):
        print(f"  {k:<15} {gb[k]:9.3f} GB")


def smoke():
    c = PRESETS["llama-3.1-8b-like"]
    assert lora_params(c, 16) == 41_943_040
    full = estimate(c, "full", seq=2048)
    assert abs(full["state_total"] - 128.0) < 1e-9
    full18 = estimate(c, "full", grad_accum_buffer=True)
    assert abs(full18["state_total"] - 144.0) < 1e-9
    lora = estimate(c, "lora")
    assert abs(lora["weights"] - 16.0) < 1e-9
    assert round(lora["state_total"] - lora["weights"], 3) == 0.671
    qlora = estimate(c, "qlora")
    assert round(qlora["weights"], 3) == 4.127
    assert abs(qlora["state_total"] - 4.798) < 1e-3
    # Logits at 8k context for a Qwen3 vocabulary, one sequence, fp32.
    q4 = PRESETS["qwen3-4b"]
    lg = estimate(q4, "lora", seq=8192)["logits"]
    assert round(lg, 2) == 4.98
    # Activation worked example from section 3.1 (Qwen3-4B dims, T=2048, B=1).
    no_ckpt_fa = activation_bytes(q4, 1, 2048, flash=True, checkpointing=False) / GB
    no_ckpt_eager = activation_bytes(q4, 1, 2048, flash=False, checkpointing=False) / GB
    ckpt = activation_bytes(q4, 1, 2048, flash=True, checkpointing=True) / GB
    assert round(no_ckpt_eager, 1) == 30.6 and round(no_ckpt_fa, 2) == 6.42
    assert round(ckpt, 2) == 0.56
    print_table("llama-3.1-8b-like, full FT, AdamW, B=1, T=2048, checkpointing", full)
    print_table("llama-3.1-8b-like, LoRA r=16 all-linear", lora)
    print_table("llama-3.1-8b-like, QLoRA r=16 all-linear", qlora)
    print("\nPer-size state (weights + adapter + optimizer only, GB):")
    for name in ("llama-3.2-1b-like", "llama-3.1-8b-like", "qwen3-32b-like", "llama-3.1-70b-like"):
        cc = PRESETS[name]
        f, l, q = (estimate(cc, m)["state_total"] for m in ("full", "lora", "qlora"))
        print(f"  {name:<20} full {f:8.1f}  lora {l:7.2f}  qlora {q:6.2f}  "
              f"(adapters {lora_params(cc, 16) / 1e6:.1f}M)")
    print("\nsmoke OK: 8B full 128.0 / 144.0 GB, LoRA 16.67 GB, QLoRA 4.80 GB, logits 4.98 GB")


def full(args):
    """T1 GPU: measure one real step and compare with the estimate."""
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available(), "--full needs a CUDA GPU (T1)"
    model_id = "Qwen/Qwen3-1.7B"
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.method == "qlora":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **kwargs)
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    ids = torch.randint(0, 151936, (args.batch, args.seq_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    measured = torch.cuda.max_memory_allocated() / GB
    est = estimate(PRESETS["qwen3-1.7b"], args.method, r=args.rank, batch=args.batch,
                   seq=args.seq_len, checkpointing=args.checkpointing, flash=False,
                   exact=True, overhead_gb=0.0)
    print_table(f"estimate for Qwen3-1.7B ({args.method}, exact params, no overhead)", est)
    print(f"\nmeasured max_memory_allocated: {measured:.2f} GB "
          f"(excludes CUDA context; compare with nvidia-smi for the full footprint)")
    print("Explain the gap: allocator rounding, temporary buffers in backward, the fp32 "
          "logits upcast inside the loss, and whether SDPA chose a memory-efficient kernel.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--preset", default="llama-3.1-8b-like", choices=sorted(PRESETS))
    p.add_argument("--method", default="lora", choices=["full", "lora", "qlora"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw_8bit", "lion", "adafactor", "sgd"])
    p.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    p.add_argument("--eager-attention", action="store_true", help="no FlashAttention: store attention scores")
    p.add_argument("--fused-ce", action="store_true", help="fused/chunked CE: never materialise logits")
    p.add_argument("--grad-accum-buffer", action="store_true", help="add the 2 B/p buffer (16 -> 18)")
    p.add_argument("--exact-params", action="store_true")
    args = p.parse_args()
    if args.full:
        return full(args)
    if args.smoke:
        return smoke()
    gb = estimate(PRESETS[args.preset], args.method, r=args.rank, batch=args.batch,
                  seq=args.seq_len, optimizer=args.optimizer, checkpointing=args.checkpointing,
                  flash=not args.eager_attention, fused_ce=args.fused_ce,
                  grad_accum_buffer=args.grad_accum_buffer, exact=args.exact_params)
    print_table(f"{args.preset} {args.method} r={args.rank} B={args.batch} T={args.seq_len} "
                f"opt={args.optimizer} ckpt={args.checkpointing}", gb)
    if args.method != "full":
        print(f"  adapter params: {gb['adapter_params']:,}")


if __name__ == "__main__":
    main()
