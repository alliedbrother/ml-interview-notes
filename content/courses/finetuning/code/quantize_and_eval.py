"""Fine-tune, merge, quantize, measure: PTQ vs QAT and token-level KL.

Module: 11 — Quantization Across the Lifecycle: QLoRA, QAT, FP8, and Export
(Lab 11.8, parts L11.1 and L11.2).

What it does
    --smoke (CPU, stdlib + torch only, no downloads, well under a minute):
      1. Implements symmetric and asymmetric int4 group quantization and checks
         the round-trip error bound (half a step per element).
      2. Trains a small MLP regressor in fp32, then compares
         (a) PTQ: round its weights to int4 (g=16) after training, with
         (b) QAT: continue training with fake quantization (quantize-dequantize)
             in the forward pass and a straight-through estimator in backward,
             then convert with the same int4 scheme.
         Asserts the QAT model's post-quantization loss is below PTQ's.
      3. Builds a toy next-token head and measures the token-level
         KL(p_fp32 || p_int4) (mean and p99) and top-1 agreement on held-out
         hidden states. Asserts KL > 0 and KL(p || p) == 0.
      4. Simulates the training/serving base mismatch: adapters (rank 8 and
         full rank) trained against an int4 base are served on that base, on
         the fp32 base and on a different int4 scheme. Asserts the full-rank
         adapter (which absorbs the quantization error) loses accuracy on any
         base it was not trained against, and reports the rank-8 numbers.
    --full (GPU, T1): for a LoRA you trained on Qwen/Qwen3-4B (bf16), merges it
      into a bf16 copy of the base with PEFT, quantizes the merged model to
      4-bit GPTQ with GPTQModel, measures token KL and top-1 agreement between
      the bf16 and GPTQ models on held-out prompts, and prints the llama.cpp
      commands that convert the merged model to GGUF and quantize it to Q4_K_M
      (with an optional imatrix). Run your task eval (module 09) on each artefact.

Hardware tier for --full: T1 (single 24 GB GPU, bf16-capable; RTX 4090/L4/A10G).
Pinned versions for --full: transformers>=5.0 (`dtype=`), peft>=0.21,
    gptqmodel>=7.0 (GPTQConfig/GPTQModel.load/quantize/save as in its v7.5.0
    README), datasets>=3.0, a llama.cpp checkout for the GGUF step.

Run from the repository root:
    python content/courses/finetuning/code/quantize_and_eval.py --smoke
    python content/courses/finetuning/code/quantize_and_eval.py --full \
        --base Qwen/Qwen3-4B --adapter ./out/qwen3-4b-lora \
        --prompts ./data/eval_prompts.jsonl --out ./quant/
"""

import argparse
import json
import math
import os
import time

import torch


# ---------------------------------------------------------------------------
# Quantization primitives (pure torch)
# ---------------------------------------------------------------------------

def quant_sym(x, group=32, bits=4):
    """Symmetric per-group quantize-dequantize. Scale s = max|x| / qmax."""
    qmax = 2 ** (bits - 1) - 1
    xg = x.reshape(-1, group)
    s = (xg.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(xg / s), -qmax - 1, qmax)
    return (q * s).reshape(x.shape)


def quant_asym(x, group=32, bits=4):
    """Asymmetric per-group quantize-dequantize: s=(max-min)/(2^b-1), z=-round(min/s)."""
    levels = 2 ** bits - 1
    xg = x.reshape(-1, group)
    lo = xg.amin(dim=1, keepdim=True)
    hi = xg.amax(dim=1, keepdim=True)
    s = ((hi - lo) / levels).clamp_min(1e-12)
    z = -torch.round(lo / s)
    q = torch.clamp(torch.round(xg / s) + z, 0, levels)
    return (s * (q - z)).reshape(x.shape)


def fake_quant_ste(w, group=16, bits=4):
    """Int4 symmetric fake quantization with a straight-through estimator.

    Forward value: quantize-dequantize(w). Backward: gradient 1 where w/s is
    inside [qmin, qmax], 0 where it was clamped. The scale is treated as a
    constant (detached), as in a weight-only absmax recipe.
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    wg = w.reshape(-1, group)
    s = (wg.detach().abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(wg / s), qmin, qmax)
    inside = ((wg / s) >= qmin) & ((wg / s) <= qmax)
    wm = wg * inside
    return (wm + (q * s - wm).detach()).reshape(w.shape)


def token_kl(logits_p, logits_q):
    """Per-position KL(p || q) from two (N, V) logit tensors."""
    lp = torch.log_softmax(logits_p.float(), dim=-1)
    lq = torch.log_softmax(logits_q.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

def smoke():
    t0 = time.time()
    torch.manual_seed(0)

    # 1. Round-trip bound: |x - x_hat| <= s/2 for every unclamped element.
    x = torch.randn(4096) * 0.02
    xs = quant_sym(x, group=32)
    s = x.reshape(-1, 32).abs().amax(1, keepdim=True) / 7
    err = (xs - x).reshape(-1, 32).abs()
    assert bool((err <= s / 2 + 1e-9).all()), "symmetric round-trip exceeds s/2"
    shifted = x + 0.05
    mse_sym = ((quant_sym(shifted) - shifted) ** 2).mean().item()
    mse_asym = ((quant_asym(shifted) - shifted) ** 2).mean().item()
    assert mse_asym < mse_sym, "asymmetric should win on a shifted tensor"
    print(f"[1] int4 g=32 round-trip ok; shifted tensor MSE sym={mse_sym:.3e} "
          f"asym={mse_asym:.3e}")

    # 2. PTQ vs QAT on a toy MLP regressor.
    g = torch.Generator().manual_seed(0)
    W1t = torch.randn(64, 16, generator=g) / 4
    W2t = torch.randn(1, 64, generator=g) / 8
    X = torch.randn(1024, 16, generator=g)
    y = torch.tanh(X @ W1t.T) @ W2t.T

    def forward(W1, b1, W2, quant):
        if quant:
            W1, W2 = fake_quant_ste(W1), fake_quant_ste(W2)
        return torch.tanh(X @ W1.T + b1) @ W2.T

    torch.manual_seed(1000)
    W1 = (torch.randn(64, 16) / 4).requires_grad_()
    b1 = torch.zeros(64, requires_grad=True)
    W2 = (torch.randn(1, 64) / 8).requires_grad_()
    opt = torch.optim.Adam([W1, b1, W2], lr=0.01)
    for _ in range(800):
        loss = ((forward(W1, b1, W2, False) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    fp32_loss = loss.item()
    with torch.no_grad():
        ptq_loss = ((forward(W1.detach(), b1.detach(), W2.detach(), True) - y) ** 2).mean().item()

    qat_steps = 400
    opt = torch.optim.SGD([W1, b1, W2], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1 - t / qat_steps)
    for _ in range(qat_steps):
        loss = ((forward(W1, b1, W2, True) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        qat_loss = ((forward(W1, b1, W2, True) - y) ** 2).mean().item()
    recovered = (ptq_loss - qat_loss) / max(ptq_loss - fp32_loss, 1e-12)
    print(f"[2] MLP MSE fp32={fp32_loss:.2e} PTQ-int4={ptq_loss:.2e} "
          f"QAT-int4={qat_loss:.2e} (QAT recovers {100 * recovered:.0f}% of the PTQ gap)")
    assert qat_loss < ptq_loss, "QAT should beat PTQ after conversion"

    # 3. Token-level KL between fp32 and int4 heads.
    torch.manual_seed(0)
    V, d, n = 256, 64, 512
    W_head = torch.randn(V, d) * 0.05
    h = torch.randn(n, d) * 2.0
    logits_fp = h @ W_head.T                              # (n, V)
    logits_q = h @ quant_sym(W_head, group=32).T          # (n, V)
    kl = token_kl(logits_fp, logits_q)
    agree = (logits_fp.argmax(-1) == logits_q.argmax(-1)).float().mean().item()
    kl_self = token_kl(logits_fp, logits_fp)
    print(f"[3] token KL mean={kl.mean().item():.4f} p99={torch.quantile(kl, 0.99).item():.4f} "
          f"nats, top-1 agreement={agree:.3f}")
    assert kl.mean().item() > 0 and float(kl_self.abs().max()) < 1e-6
    assert 0.0 <= agree <= 1.0

    # 4. Adapter trained on an int4 base, then served on three bases:
    #    the same int4 base (matched), the fp32 base, and a different int4
    #    scheme (asymmetric, g=64: a stand-in for "trained on NF4, served on AWQ").
    #    A full-rank adapter absorbs the quantization error, so moving it to any
    #    other base hurts; a rank-8 adapter cannot absorb random error, so the
    #    effect is smaller and can even flip sign. Either way, measure it.
    torch.manual_seed(0)
    d_in = d_out = 64
    W0 = torch.randn(d_out, d_in) * 0.1
    W0q = quant_sym(W0, group=32)
    W0q_other = quant_asym(W0, group=64)
    target = W0 + 0.02 * torch.randn(d_out, 4) @ torch.randn(4, d_in)
    Xa = torch.randn(2048, d_in)
    Ya = Xa @ target.T
    results = {}
    for r in (8, 64):
        torch.manual_seed(1)
        A = (torch.randn(r, d_in) * 0.01).requires_grad_()
        B = torch.zeros(d_out, r, requires_grad=True)
        opt = torch.optim.Adam([A, B], lr=0.01)
        for _ in range(600):
            loss = ((Xa @ (W0q + B @ A).T - Ya) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            delta = B @ A
            results[r] = {name: ((Xa @ (base + delta).T - Ya) ** 2).mean().item()
                          for name, base in (("matched int4", W0q), ("fp32", W0),
                                             ("other int4", W0q_other))}
        print(f"[4] rank {r:2d} adapter trained on int4 base, served on: " +
              ", ".join(f"{k} MSE={v:.2e}" for k, v in results[r].items()))
    full_rank = results[64]
    assert full_rank["fp32"] > full_rank["matched int4"]
    assert full_rank["other int4"] > full_rank["matched int4"]
    assert results[8]["fp32"] != results[8]["matched int4"]

    print(f"smoke ok in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Full mode (GPU, T1). Heavy imports stay inside these functions.
# ---------------------------------------------------------------------------

def merge_to_bf16(base, adapter, out_dir):
    """Merge a LoRA into a bf16 copy of the base (never into 4-bit weights)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    return out_dir


def gptq_quantize(merged_dir, out_dir, n_calib=256):
    """4-bit GPTQ, group size 128, via GPTQModel (the backend PEFT uses since v0.19)."""
    from datasets import load_dataset
    from gptqmodel import GPTQConfig, GPTQModel

    calib = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
                         split="train").select(range(n_calib))["text"]
    model = GPTQModel.load(merged_dir, GPTQConfig(bits=4, group_size=128))
    model.quantize(calib, batch_size=1)
    model.save(out_dir)
    return out_dir


@torch.no_grad()
def kl_report(ref_dir, test_dir, prompts, max_len=512):
    """Mean and p99 token KL(ref || test) and top-1 agreement over prompt tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref_dir)
    ref = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.bfloat16, device_map="cuda")
    test = AutoModelForCausalLM.from_pretrained(test_dir, device_map="cuda")
    kls, agree = [], []
    for text in prompts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
        lp = ref(ids).logits[0]       # (T, V)
        lq = test(ids).logits[0]      # (T, V)
        kls.append(token_kl(lp, lq).cpu())
        agree.append((lp.argmax(-1) == lq.argmax(-1)).float().cpu())
    kl = torch.cat(kls)
    return {"kl_mean": kl.mean().item(), "kl_p99": torch.quantile(kl, 0.99).item(),
            "top1_agreement": torch.cat(agree).mean().item(), "tokens": kl.numel()}


def full(args):
    if not torch.cuda.is_available():
        raise SystemExit("--full needs a CUDA GPU (tier T1). Use --smoke on CPU.")
    os.makedirs(args.out, exist_ok=True)
    merged = os.path.join(args.out, "merged-bf16")
    gptq = os.path.join(args.out, "gptq-int4")
    if not os.path.isdir(merged):
        merge_to_bf16(args.base, args.adapter, merged)
    if not os.path.isdir(gptq):
        gptq_quantize(merged, gptq)
    with open(args.prompts) as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    report = kl_report(merged, gptq, prompts)
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.out, "kl_gptq.json"), "w") as f:
        json.dump(report, f, indent=2)

    llama = args.llama_cpp
    gguf_bf16 = os.path.join(args.out, "model-bf16.gguf")
    gguf_q4 = os.path.join(args.out, "model-Q4_K_M.gguf")
    print("\nGGUF export (run these; llama.cpp must be built):")
    print(f"python {llama}/convert_hf_to_gguf.py {merged} --outtype bf16 --outfile {gguf_bf16}")
    print(f"{llama}/build/bin/llama-imatrix -m {gguf_bf16} -f calib.txt -o {args.out}/imatrix.gguf")
    print(f"{llama}/build/bin/llama-quantize --imatrix {args.out}/imatrix.gguf "
          f"{gguf_bf16} {gguf_q4} Q4_K_M")
    print("\nThen run the same task eval (module 09) on merged-bf16, gptq-int4 and the "
          "Q4_K_M GGUF, and record task metric, KL mean/p99 and top-1 agreement.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy run (default without a GPU)")
    mode.add_argument("--full", action="store_true", help="GPU lab (tier T1)")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="./out/qwen3-4b-lora")
    ap.add_argument("--prompts", default="./data/eval_prompts.jsonl",
                    help="JSONL with a 'prompt' field per line (held-out prompts)")
    ap.add_argument("--llama-cpp", default="./llama.cpp")
    ap.add_argument("--out", default="./quant")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
