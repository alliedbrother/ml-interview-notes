"""RoPE scaling lab: position interpolation, NTK-aware scaling and YaRN.

Module: 17 — Long-Context Extension and Long-Sequence Training.

Purpose
    --smoke (CPU; also the default when --full is not given): build the RoPE frequency tables
    for PI, NTK-aware scaling and YaRN for a given (base b, head dim d, factor s,
    original context L), assert the module's worked numbers, and show on a tiny
    attention layer that unscaled RoPE at 4x the trained length produces rotation
    angles and per-pair scores never seen in range, while PI-scaled RoPE stays in
    range. Uses only stdlib + torch.
    --full (tier T1, one 24 GB Ampere+ GPU): extend Qwen/Qwen3-1.7B with YaRN,
    run a short LoRA fine-tune with trainable embeddings and norms on long PG-19
    books, then run a passkey (needle) sweep before and after.

Pinned versions for --full (as of October 2026): transformers>=5.0, trl>=1.15,
peft>=0.18, datasets>=4.0, torch>=2.8 with CUDA. gradient-checkpointing offload
(--offload) needs transformers>=5.17.

Run (from the repository root):
    python content/courses/finetuning/code/rope_scaling_lab.py --smoke
    python content/courses/finetuning/code/rope_scaling_lab.py --full --factor 4 --train-len 32768
"""
from __future__ import annotations

import argparse
import math

import torch


# ---------------------------------------------------------------- frequency math
def rope_freqs(base: float, d: int) -> torch.Tensor:
    """theta_i = base^(-2i/d), i = 0..d/2-1, in float64."""
    i = torch.arange(d // 2, dtype=torch.float64)
    return base ** (-2.0 * i / d)


def pi_freqs(base: float, d: int, s: float) -> torch.Tensor:
    """Position interpolation: positions scaled by 1/s == every frequency / s."""
    return rope_freqs(base, d) / s


def ntk_base(base: float, d: int, s: float) -> float:
    """NTK-aware base change b' = b * s^(d/(d-2))."""
    return base * s ** (d / (d - 2))


def ntk_freqs(base: float, d: int, s: float) -> torch.Tensor:
    return rope_freqs(ntk_base(base, d, s), d)


def yarn_gamma(base: float, d: int, orig_len: int, alpha: float = 1.0, beta: float = 32.0) -> torch.Tensor:
    """YaRN ramp. r_i = L / lambda_i (rotations completed within the trained window).
    gamma = 0 (interpolate fully) if r < alpha, 1 (leave alone) if r > beta, linear between."""
    lam = 2 * math.pi / rope_freqs(base, d)
    r = orig_len / lam
    return ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)


def yarn_freqs(base: float, d: int, s: float, orig_len: int, alpha: float = 1.0, beta: float = 32.0) -> torch.Tensor:
    theta = rope_freqs(base, d)
    g = yarn_gamma(base, d, orig_len, alpha, beta)
    return (1 - g) * theta / s + g * theta


def yarn_attention_factor(s: float) -> float:
    """sqrt(1/t) = 0.1 ln s + 1; q and k are each multiplied by it, so logits scale by its square."""
    return 0.1 * math.log(s) + 1.0


# ---------------------------------------------------------------- tiny attention demo
def out_of_range_report(base: float, d: int, s: float, orig_len: int, seed: int = 0) -> dict:
    """For every frequency pair, compare the per-pair attention-score contribution
    q_i . R(delta * theta_i) k_i over test distances [0, s*L) against the set of
    values reachable inside the trained window (angles in [0, L*theta_i])."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(d // 2, 2, generator=g, dtype=torch.float64)
    k = torch.randn(d // 2, 2, generator=g, dtype=torch.float64)

    def contrib(angle: torch.Tensor) -> torch.Tensor:  # angle: (n, d/2)
        c, si = torch.cos(angle), torch.sin(angle)
        # (R k) = (c*k0 - s*k1, s*k0 + c*k1); dot with q
        rk0 = c * k[:, 0] - si * k[:, 1]
        rk1 = si * k[:, 0] + c * k[:, 1]
        return q[:, 0] * rk0 + q[:, 1] * rk1  # (n, d/2)

    theta = rope_freqs(base, d)
    # dense grid of in-range angles: the continuous set reachable within the trained window
    grid = torch.linspace(0, 1, 20001, dtype=torch.float64)[:, None] * (orig_len * theta)
    train_vals = contrib(grid)
    lo, hi = train_vals.min(0).values, train_vals.max(0).values
    test_pos = torch.arange(int(s * orig_len), dtype=torch.float64)[:, None]
    report = {}
    for name, freqs in (("none", theta), ("pi", theta / s)):
        ang = test_pos * freqs
        vals = contrib(ang)
        tol = 1e-6
        outside = ((vals < lo - tol) | (vals > hi + tol)).any(0)
        max_angle_ratio = (ang.max(0).values / (orig_len * theta)).max().item()
        report[name] = {"pairs_out_of_range": int(outside.sum()), "max_angle_ratio": max_angle_ratio}
    return report


def smoke() -> None:
    torch.manual_seed(0)
    b, d, s, L = 10_000.0, 128, 4.0, 4096
    theta = rope_freqs(b, d)

    # PI divides every frequency by s
    torch.testing.assert_close(pi_freqs(b, d, s), theta / 4, rtol=0, atol=0)
    # NTK: b' = 40,889.9; lowest freq / s, highest unchanged
    bp = ntk_base(b, d, s)
    assert round(bp, 1) == 40889.9, bp
    nt = ntk_freqs(b, d, s)
    assert math.isclose(nt[-1].item(), theta[-1].item() / s, rel_tol=1e-9)
    assert nt[0].item() == theta[0].item() == 1.0
    # YaRN ramp and temperature
    gam = yarn_gamma(b, d, L)
    assert gam[0].item() == 1.0 and gam[-1].item() == 0.0
    assert bool((gam[1:] <= gam[:-1]).all())
    yf = yarn_freqs(b, d, s, L)
    assert bool((yf >= theta / s - 1e-15).all() and (yf <= theta + 1e-15).all())
    assert round(yarn_attention_factor(s), 4) == 1.1386

    print(f"{'i':>3} {'lambda_i':>10} {'r_i=L/lam':>10} {'gamma':>6} {'PI/orig':>8} {'NTK/orig':>9} {'YaRN/orig':>9}")
    for i in (0, 8, 16, 24, 32, 40, 48, 56, 63):
        lam = 2 * math.pi / theta[i].item()
        print(f"{i:>3} {lam:>10.1f} {L / lam:>10.3f} {gam[i].item():>6.3f} "
              f"{(1 / s):>8.3f} {(nt[i] / theta[i]).item():>9.4f} {(yf[i] / theta[i]).item():>9.4f}")
    print(f"NTK base b' = {bp:.1f}; YaRN sqrt(1/t) = {yarn_attention_factor(s):.4f}")

    # tiny attention layer: scores at 4x length, small d for speed
    rep = out_of_range_report(b, 16, s, 256)
    print("tiny attention (d=16, L=256, test to 4L):", rep)
    assert rep["none"]["pairs_out_of_range"] > 0, "unscaled RoPE should leave the trained score range"
    assert rep["none"]["max_angle_ratio"] > 3.9
    assert rep["pi"]["pairs_out_of_range"] == 0, "PI-scaled scores should stay in range"
    assert rep["pi"]["max_angle_ratio"] <= 1.0
    print("smoke OK: PI, NTK and YaRN tables verified; PI keeps 4x-length scores in range")


# ---------------------------------------------------------------- full GPU lab
def passkey_prompt(tok, filler: str, n_tokens: int, depth: float, key: str) -> str:
    ids = tok(filler, add_special_tokens=False)["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    ids = ids[: n_tokens - 64]
    cut = int(len(ids) * depth)
    needle = f" The secret passkey is {key}. Remember it. "
    text = tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:])
    return text + "\nWhat is the secret passkey? The secret passkey is"


def passkey_sweep(model, tok, filler: str, lengths, depths=(0.1, 0.5, 0.9)) -> dict:
    import random
    rng = random.Random(0)
    results = {}
    model.eval()
    for n in lengths:
        hits = 0
        for dpt in depths:
            key = str(rng.randint(10000, 99999))
            enc = tok(passkey_prompt(tok, filler, n, dpt, key), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False)
            hits += key in tok.decode(out[0, enc["input_ids"].shape[1]:])
        results[n] = hits / len(depths)
        print(f"passkey @ {n:>6} tokens: {results[n]:.2f}")
    return results


def full(args) -> None:
    # Imported here so --smoke never needs them.
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "--full needs a CUDA GPU (tier T1)"
    model_id = "Qwen/Qwen3-1.7B"
    rope = {"rope_type": "yarn", "rope_theta": 1_000_000, "factor": args.factor,
            "original_max_position_embeddings": 32768}
    tok = AutoTokenizer.from_pretrained(model_id)
    books = load_dataset("emozilla/pg19", split="train", streaming=True)
    rows = [r["text"] for _, r in zip(range(args.num_books), books)]
    filler = load_dataset("emozilla/pg19", split="validation", streaming=True)
    filler_text = next(iter(filler))["text"]
    eval_lengths = [int(x) for x in args.eval_lengths.split(",")]

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    print("== before: native RoPE ==")
    passkey_sweep(base, tok, filler_text, [n for n in eval_lengths if n <= 32768])
    del base
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                 rope_parameters=rope)
    print("== YaRN only, no fine-tune ==")
    passkey_sweep(model, tok, filler_text, eval_lengths)

    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # LongLoRA's finding: embeddings and norms must be trainable for context extension.
    for name, p in model.named_parameters():
        if "embed_tokens" in name or name.endswith("norm.weight"):
            p.requires_grad_(True)
    model.print_trainable_parameters()

    from datasets import Dataset
    ds = Dataset.from_dict({"text": rows})
    gc_kwargs = {"use_reentrant": False}
    if args.offload:
        gc_kwargs["offload"] = True  # transformers>=5.17
    cfg = SFTConfig(
        output_dir=args.output_dir, max_length=args.train_len, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-5, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        max_steps=args.max_steps, bf16=True, logging_steps=5, save_strategy="no",
        gradient_checkpointing=True, gradient_checkpointing_kwargs=gc_kwargs,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    print("== YaRN + LoRA (embeddings, norms trainable) ==")
    passkey_sweep(trainer.model, tok, filler_text, eval_lengths)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{args.output_dir}/merged")
    tok.save_pretrained(f"{args.output_dir}/merged")
    print(f"saved merged model with rope_parameters={merged.config.rope_parameters}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--train-len", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-books", type=int, default=2000)
    ap.add_argument("--eval-lengths", default="16384,32768,65536,98304")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--output-dir", default="/tmp/qwen3-1.7b-yarn")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
