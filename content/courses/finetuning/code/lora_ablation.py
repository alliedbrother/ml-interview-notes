"""LoRA ablation grid (rank x targets x variant) and QLoRA vs LoRA vs full fine-tuning.

Purpose
    Module 06 (LoRA and QLoRA in Depth), labs L6.2 and L6.3.
    --smoke fits a known rank-8 weight update with LoRA adapters of rank 1, 4
    and 16 under the alpha/r and alpha/sqrt(r) (rsLoRA) scalings, on CPU, and
    asserts the trainable-parameter formula and the capacity ordering.
    --full runs the real grids with TRL's SFTTrainer and PEFT.

Hardware
    --smoke:          any CPU, a few seconds, torch only, no downloads.
    --full grid:      T1 (one 24 GB GPU, bf16) for Qwen/Qwen3-4B-Base, 18 short runs.
    --full compare:   QLoRA and LoRA on T1; full fine-tuning of Qwen/Qwen3-1.7B-Base on T2 (80 GB).

Pinned versions for --full (as of October 2026)
    torch>=2.8, transformers>=5.0, trl>=1.15, peft>=0.21.2, bitsandbytes>=0.48, datasets, matplotlib

Run from the repository root
    python content/courses/finetuning/code/lora_ablation.py --smoke
    python content/courses/finetuning/code/lora_ablation.py --full grid --model Qwen/Qwen3-4B-Base
    python content/courses/finetuning/code/lora_ablation.py --full compare --model Qwen/Qwen3-1.7B-Base
"""

import argparse
import json
import math
import time

import torch


# ---------------------------------------------------------------------------
# Smoke: a toy regression with a known low-rank target update
# ---------------------------------------------------------------------------
def fit_lora(w0, w_star, x_tr, x_te, r, alpha, rslora, steps=300, lr=1e-2, seed=0, early=20):
    g = torch.Generator().manual_seed(seed)
    d_out, d_in = w0.shape
    a = (torch.randn(r, d_in, generator=g) / math.sqrt(d_in)).requires_grad_()  # (r, d_in)
    b = torch.zeros(d_out, r, requires_grad=True)                               # (d_out, r)
    scale = alpha / math.sqrt(r) if rslora else alpha / r
    opt = torch.optim.Adam([a, b], lr=lr)
    y_tr, y_te = x_tr @ w_star.T, x_te @ w_star.T
    test = lambda: ((x_te @ (w0 + scale * b @ a).T - y_te) ** 2).mean().item()  # noqa: E731
    for step in range(steps):
        pred = x_tr @ (w0 + scale * b @ a).T
        loss = ((pred - y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step + 1 == early:
            with torch.no_grad():
                early_mse = test()
    with torch.no_grad():
        return early_mse, test(), a.numel() + b.numel()


def smoke():
    t0 = time.time()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    d, true_rank, alpha = 64, 8, 8
    w0 = torch.randn(d, d) / math.sqrt(d)
    u, v = torch.randn(d, true_rank), torch.randn(true_rank, d)
    w_star = w0 + 0.5 * (u @ v) / d                  # the "task" needs a rank-8 change
    x_tr, x_te = torch.randn(512, d), torch.randn(256, d)

    results = {}
    early_results = {}
    print(f"{'rank':>4} {'scaling':>9} {'params':>7} {'MSE@20':>9} {'MSE@300':>9}")
    for r in (1, 4, 16):
        for rslora in (False, True):
            early, mse, n = fit_lora(w0, w_star, x_tr, x_te, r, alpha, rslora)
            assert n == r * (d + d), (r, n)          # trainable params = r (d_in + d_out)
            results[(r, rslora)], early_results[(r, rslora)] = mse, early
            print(f"{r:>4} {'a/sqrt(r)' if rslora else 'a/r':>9} {n:>7} {early:>9.5f} {mse:>9.5f}")

    # Capacity: a rank-8 target cannot be fit by rank 1 or 4, but rank 16 can.
    for rs in (False, True):
        assert results[(1, rs)] > results[(4, rs)] > results[(16, rs)], results
    # At r=1 the two scalings coincide. At r=16, alpha/r is 4x smaller than alpha/sqrt(r),
    # so with the same alpha and LR the alpha/r adapter learns more slowly early on.
    assert early_results[(1, True)] == early_results[(1, False)]
    assert early_results[(16, True)] < early_results[(16, False)], early_results
    print(f"smoke OK in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Full mode (heavy imports live inside these functions)
# ---------------------------------------------------------------------------
def _data(n_train, n_eval):
    from datasets import load_dataset
    ds = load_dataset("trl-lib/Capybara")
    return ds["train"].shuffle(seed=0).select(range(n_train)), ds["test"].select(range(n_eval))


def _sft_config(out, lr, max_steps, gradient_checkpointing=True):
    from trl import SFTConfig
    return SFTConfig(
        output_dir=out, learning_rate=lr, max_steps=max_steps, max_length=1024,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        eval_strategy="steps", eval_steps=max_steps, logging_steps=10,
        bf16=True, gradient_checkpointing=gradient_checkpointing, report_to="none",
        assistant_only_loss=False, model_init_kwargs={"dtype": torch.bfloat16},
        save_strategy="no", seed=0,
    )


def _run(model, peft_config, cfg, train, evals):
    from trl import SFTTrainer
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=evals,
                         peft_config=peft_config)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainer.train()
    metrics = trainer.evaluate()
    return {"eval_loss": metrics["eval_loss"], "trainable": trainable,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "minutes": (time.time() - start) / 60}


def full_grid(model_id, max_steps):
    from peft import LoraConfig
    train, evals = _data(4000, 200)
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    rows = []
    for r in (8, 32, 128):
        for name, targets in (("attention", attention), ("all-linear", "all-linear")):
            for variant in ("lora", "rslora", "dora"):
                pc = LoraConfig(r=r, lora_alpha=32, target_modules=targets, lora_dropout=0.0,
                                use_rslora=variant == "rslora", use_dora=variant == "dora",
                                task_type="CAUSAL_LM")
                # LoRA LR ~10x a full-FT LR; alpha fixed at 32 as in LoRA Without Regret.
                res = _run(model_id, pc, _sft_config(f"out/grid-{r}-{name}-{variant}", 2e-4, max_steps),
                           train, evals)
                rows.append({"r": r, "targets": name, "variant": variant, **res})
                print(json.dumps(rows[-1]))
    with open("lora_grid.json", "w") as f:
        json.dump(rows, f, indent=2)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for (name, variant) in {(x["targets"], x["variant"]) for x in rows}:
            pts = sorted((x["trainable"], x["eval_loss"]) for x in rows
                         if x["targets"] == name and x["variant"] == variant)
            ax.plot(*zip(*pts), marker="o", label=f"{name} / {variant}")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel("eval loss")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig("lora_grid.png", dpi=150)
    except ImportError:
        print("matplotlib not installed; wrote lora_grid.json only")


def full_compare(model_id, max_steps):
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    train, evals = _data(4000, 200)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    rows = {}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    q_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="cuda")
    rows["qlora"] = _run(q_model, lora, _sft_config("out/qlora", 2e-4, max_steps), train, evals)
    del q_model
    rows["lora"] = _run(model_id, lora, _sft_config("out/lora", 2e-4, max_steps), train, evals)
    if torch.cuda.get_device_properties(0).total_memory > 60e9:
        rows["full"] = _run(model_id, None, _sft_config("out/full", 2e-5, max_steps), train, evals)
    else:
        print("full fine-tuning skipped: needs a T2 (80 GB) GPU")
    for k, v in rows.items():
        print(f"{k:>6}: eval_loss {v['eval_loss']:.4f}  peak {v['peak_gb']:.1f} GB  "
              f"{v['minutes']:.1f} min  trainable {v['trainable']:,}")
    with open("lora_compare.json", "w") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy regression (default)")
    mode.add_argument("--full", choices=["grid", "compare"], help="GPU runs with TRL + PEFT")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
    if args.full == "grid":
        full_grid(args.model or "Qwen/Qwen3-4B-Base", args.max_steps)
    elif args.full == "compare":
        full_compare(args.model or "Qwen/Qwen3-1.7B-Base", args.max_steps)
    else:
        smoke()


if __name__ == "__main__":
    main()
