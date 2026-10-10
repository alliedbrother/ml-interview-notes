"""Throughput ablation and LR sweep for SFT (Labs L3.2 and L3.3).

Module: 03 — Training Mechanics: Optimizers, Schedules, and Memory Math.

What it does
    --smoke (CPU, < 30 s, stdlib + torch only): builds a 2-layer toy decoder,
        times forward+backward tokens/s with and without
        torch.utils.checkpoint, and asserts the gradients are identical
        (checkpointing changes memory and time, never the maths).
    --full (T1: one 24 GB Ampere+ GPU): Qwen/Qwen3-1.7B SFT with TRL on
        trl-lib/Capybara under four configurations
        (a) no checkpointing, (b) checkpointing, (c) packing, (d) Hub kernels
        via model_init_kwargs={"use_kernels": True}; records peak VRAM and
        tokens/s per configuration.
    --full --lr-sweep (T1): LoRA r=16 all-linear at LRs
        {2e-5, 5e-5, 1e-4, 2e-4, 5e-4} for 200 steps each, plus full FT at
        {5e-6, 1e-5, 2e-5} (needs T2 for full FT of 1.7B with AdamW; pass
        --skip-full-ft on T1), and writes final eval loss vs LR to a CSV and PNG.

Pinned versions for --full (as of October 2026)
    torch>=2.8, transformers==5.19.0, trl==1.15.0, peft>=0.21, datasets>=4,
    kernels (for use_kernels), matplotlib (for the plot)

Run (from the repository root)
    python content/courses/finetuning/code/throughput_ablation.py --smoke
    python content/courses/finetuning/code/throughput_ablation.py --full
    python content/courses/finetuning/code/throughput_ablation.py --full --lr-sweep --skip-full-ft
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ToyLM(nn.Module):
    def __init__(self, vocab=256, d=128, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, use_ckpt=False):
        T = ids.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        x = self.emb(ids)                                   # (B, T, d)
        for blk in self.blocks:
            x = checkpoint(blk, x, mask, use_reentrant=False) if use_ckpt else blk(x, mask)
        return self.head(x)                                 # (B, T, V)


def step_grads(model, ids, use_ckpt):
    model.zero_grad(set_to_none=True)
    logits = model(ids[:, :-1], use_ckpt=use_ckpt)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}


def time_tokens_per_s(model, ids, use_ckpt, iters):
    step_grads(model, ids, use_ckpt)  # warm-up
    t0 = time.perf_counter()
    for _ in range(iters):
        step_grads(model, ids, use_ckpt)
    dt = time.perf_counter() - t0
    return iters * ids.shape[0] * (ids.shape[1] - 1) / dt


def smoke():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = ToyLM().eval()  # eval(): no dropout, so recompute is bit-reproducible
    ids = torch.randint(0, 256, (4, 129))
    l0, g0 = step_grads(model, ids, use_ckpt=False)
    l1, g1 = step_grads(model, ids, use_ckpt=True)
    torch.testing.assert_close(l0, l1, rtol=0, atol=0)
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol=1e-6, atol=1e-7)
    # Interleave and keep the best of 3 rounds: CPU timings are noisy.
    plain = ckpt = 0.0
    for _ in range(3):
        plain = max(plain, time_tokens_per_s(model, ids, False, iters=8))
        ckpt = max(ckpt, time_tokens_per_s(model, ids, True, iters=8))
    print(f"tokens/s without checkpointing: {plain:9.0f}")
    print(f"tokens/s with checkpointing:    {ckpt:9.0f}  (ratio {ckpt / plain:.2f})")
    print("Expect a ratio below 1 (one extra forward per block, ~0.75 when backward = 2x forward). "
          "On a GPU the memory saving is what you buy; CPU timings of a toy are noisy.")
    print(f"smoke OK: identical loss and gradients across {len(g0)} parameter tensors")


def sft_run(cfg_name, overrides, max_steps=60):
    import torch as th
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("trl-lib/Capybara", split="train")
    base = dict(output_dir=f"runs/l32-{cfg_name}", max_steps=max_steps, logging_steps=10,
                per_device_train_batch_size=2, gradient_accumulation_steps=4,
                max_length=2048, learning_rate=2e-5, report_to="none", save_strategy="no",
                model_init_kwargs={"dtype": th.bfloat16},
                include_num_input_tokens_seen="non_padding")
    base.update(overrides)
    args = SFTConfig(**base)
    trainer = SFTTrainer(model="Qwen/Qwen3-1.7B", args=args, train_dataset=ds)
    th.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    dt = time.perf_counter() - t0
    tokens = trainer.state.num_input_tokens_seen
    return {"config": cfg_name, "peak_gb": th.cuda.max_memory_allocated() / 1e9,
            "tokens_per_s": tokens / dt}


def full_ablation():
    assert torch.cuda.is_available(), "--full needs a CUDA GPU (T1)"
    configs = {
        "a_no_ckpt": {"gradient_checkpointing": False},
        "b_ckpt": {"gradient_checkpointing": True},
        "c_packing": {"gradient_checkpointing": True, "packing": True,
                      "model_init_kwargs": {"dtype": torch.bfloat16,
                                            "attn_implementation": "kernels-community/flash-attn2"}},
        "d_kernels": {"gradient_checkpointing": True,
                      "model_init_kwargs": {"dtype": torch.bfloat16, "use_kernels": True}},
    }
    rows = []
    for name, ov in configs.items():
        try:
            rows.append(sft_run(name, ov))
        except torch.cuda.OutOfMemoryError:
            rows.append({"config": name, "peak_gb": float("nan"), "tokens_per_s": 0.0})
            torch.cuda.empty_cache()
    for r in rows:
        print(f"{r['config']:<12} peak {r['peak_gb']:6.2f} GB   {r['tokens_per_s']:8.0f} tokens/s")


def lr_sweep(skip_full_ft):
    import csv
    from peft import LoraConfig
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    split = load_dataset("trl-lib/Capybara", split="train").train_test_split(test_size=200, seed=0)
    grid = [("lora", lr) for lr in (2e-5, 5e-5, 1e-4, 2e-4, 5e-4)]
    if not skip_full_ft:
        grid += [("full", lr) for lr in (5e-6, 1e-5, 2e-5)]
    results = []
    for method, lr in grid:
        args = SFTConfig(output_dir=f"runs/l33-{method}-{lr:g}", max_steps=200, learning_rate=lr,
                         lr_scheduler_type="cosine", warmup_steps=0.05, eval_strategy="no",
                         per_device_train_batch_size=2, gradient_accumulation_steps=4,
                         max_length=2048, report_to="none", save_strategy="no", seed=0, data_seed=0,
                         model_init_kwargs={"dtype": torch.bfloat16})
        peft_cfg = (LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
                    if method == "lora" else None)
        trainer = SFTTrainer(model="Qwen/Qwen3-1.7B", args=args, train_dataset=split["train"],
                             eval_dataset=split["test"], peft_config=peft_cfg)
        trainer.train()
        results.append((method, lr, trainer.evaluate()["eval_loss"]))
        print(results[-1])
    with open("runs/l33_lr_sweep.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("method", "lr", "eval_loss"), *results])
    import matplotlib.pyplot as plt
    for method in ("lora", "full"):
        pts = [(lr, loss) for m, lr, loss in results if m == method]
        if pts:
            plt.semilogx(*zip(*pts), marker="o", label=method)
    plt.xlabel("learning rate")
    plt.ylabel("eval loss after 200 steps")
    plt.legend()
    plt.savefig("runs/l33_lr_sweep.png", dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--lr-sweep", action="store_true")
    p.add_argument("--skip-full-ft", action="store_true")
    args = p.parse_args()
    if args.full and args.lr_sweep:
        lr_sweep(args.skip_full_ft)
    elif args.full:
        full_ablation()
    else:
        smoke()


if __name__ == "__main__":
    main()
