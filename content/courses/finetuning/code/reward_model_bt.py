"""Bradley-Terry reward modelling, then RM-ranked pairs for one round of iterative DPO (module 12, L12.3).

Purpose
    --smoke : a linear Bradley-Terry reward model on synthetic feature pairs, in pure torch.
              Pairs are labelled by a hidden weight vector with 5% label noise; the model is fit
              with -log sigmoid(r_w - r_l) plus the optional centering term TRL exposes as
              center_rewards_coefficient. Asserts held-out pairwise accuracy > 0.9 against the
              true ordering, that the BT loss alone never moves a shared reward offset, and
              that centering pulls the mean reward to roughly zero.
    --full  : step "train": TRL RewardTrainer (LoRA) on trl-lib/ultrafeedback_binarized.
              step "rank": sample 4 on-policy completions per prompt from your current policy,
              score them with the trained RM, and write (prompt, chosen=best, rejected=worst)
              pairs to disk for one round of iterative DPO with dpo_from_scratch.py --full.

Module      content/courses/finetuning/12-preference-optimization.md
Hardware    smoke: any CPU, a few seconds.  full: T1 (24 GB) for a Qwen3-0.6B/1.7B reward model
            plus Qwen3-1.7B sampling.
Pinned      smoke: Python 3.11+, torch 2.8 CPU.  full (as of October 2026): trl==1.15.0,
            transformers>=5.2, peft>=0.18, datasets.

Run (from the repository root)
    python content/courses/finetuning/code/reward_model_bt.py --smoke
    python content/courses/finetuning/code/reward_model_bt.py --full --step train --rm-model Qwen/Qwen3-0.6B
    python content/courses/finetuning/code/reward_model_bt.py --full --step rank --rm-path rm-out --policy Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Smoke mode: linear Bradley-Terry RM on synthetic pairs
# --------------------------------------------------------------------------------------


def make_pairs(n: int, w_true: torch.Tensor, gen: torch.Generator, noise: float = 0.05):
    d = w_true.numel()
    a = torch.randn(n, d, generator=gen)
    b = torch.randn(n, d, generator=gen)
    a_better = (a @ w_true) > (b @ w_true)
    flip = torch.rand(n, generator=gen) < noise  # annotator noise
    a_better = a_better ^ flip
    x_w = torch.where(a_better[:, None], a, b)
    x_l = torch.where(a_better[:, None], b, a)
    return x_w, x_l


def fit_bt(x_w, x_l, center_coef: float | None, steps: int = 400, lr: float = 0.1):
    d = x_w.shape[1]
    w = torch.zeros(d, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)  # BT is blind to a shared offset; centering pins it
    with torch.no_grad():
        bias += 3.0  # start off-centre so the centering term has something to do
    opt = torch.optim.Adam([w, bias], lr=lr)
    for _ in range(steps):
        r_w = x_w @ w + bias
        r_l = x_l @ w + bias
        loss = -F.logsigmoid(r_w - r_l).mean()
        if center_coef is not None:
            loss = loss + center_coef * ((r_w + r_l) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), bias.detach()


def smoke() -> None:
    gen = torch.Generator().manual_seed(0)
    d = 8
    w_true = torch.randn(d, generator=gen)
    tr_w, tr_l = make_pairs(2000, w_true, gen)
    # Held-out pairs are judged against the TRUE ordering (noise=0): what we care about.
    te_w, te_l = make_pairs(1000, w_true, gen, noise=0.0)

    w_plain, b_plain = fit_bt(tr_w, tr_l, center_coef=None)
    w_cent, b_cent = fit_bt(tr_w, tr_l, center_coef=0.01)
    for name, w, b in (("plain", w_plain, b_plain), ("centered(0.01)", w_cent, b_cent)):
        acc = ((te_w @ w) > (te_l @ w)).float().mean().item()
        cos = F.cosine_similarity(w, w_true, dim=0).item()
        mean_r = (torch.cat([te_w, te_l]) @ w + b).mean().item()
        print(f"{name:15s} held-out pairwise acc {acc:.3f}  cos(w, w_true) {cos:.3f}  mean reward {mean_r:+.3f}")
        assert acc > 0.9, acc
        assert cos > 0.95, cos
    # The bias gets no gradient from the BT term (it cancels in r_w - r_l); only centering moves it.
    assert abs(b_plain.item() - 3.0) < 1e-6, "BT loss is invariant to a shared offset"
    assert abs(b_cent.item()) < abs(b_plain.item())
    print("\nsmoke OK: held-out pairwise accuracy > 0.9; BT ignores the offset, centering pins it near 0")


# --------------------------------------------------------------------------------------
# Full mode (GPU). Imports stay inside so --smoke never needs them.
# --------------------------------------------------------------------------------------


def full_train(args) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import RewardConfig, RewardTrainer

    ds = load_dataset("trl-lib/ultrafeedback_binarized")
    config = RewardConfig(
        output_dir=args.rm_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_length=1024,
        learning_rate=1e-4,  # the RewardConfig default as of v1.15; suits LoRA
        center_rewards_coefficient=0.01,
        eval_strategy="steps",
        eval_steps=200,
        bf16=True,
        report_to="none",
    )
    trainer = RewardTrainer(
        model=args.rm_model,  # loaded as AutoModelForSequenceClassification with num_labels=1
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"].select(range(1000)),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_CLS"),
    )
    trainer.train()
    trainer.save_model(args.rm_path)


def full_rank(args) -> None:
    import json

    import torch as _torch
    from datasets import load_dataset
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.policy)
    tok.padding_side = "left"  # generation needs left padding (module 02)
    policy = AutoModelForCausalLM.from_pretrained(args.policy, dtype=_torch.bfloat16).to(dev)
    rm = AutoPeftModelForSequenceClassification.from_pretrained(args.rm_path, num_labels=1,
                                                                dtype=_torch.bfloat16).to(dev)
    rm_tok = AutoTokenizer.from_pretrained(args.rm_path)
    prompts = load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(args.n_prompts))
    out = []
    for row in prompts:
        msgs = row["prompt"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * 4, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = policy.generate(**enc, do_sample=True, temperature=0.8, max_new_tokens=512)
        completions = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        convs = [msgs + [{"role": "assistant", "content": c}] for c in completions]
        rm_text = [rm_tok.apply_chat_template(c, tokenize=False) for c in convs]
        rm_enc = rm_tok(rm_text, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with _torch.no_grad():
            scores = rm(**rm_enc).logits.squeeze(-1).float()
        best, worst = int(scores.argmax()), int(scores.argmin())
        if scores[best] - scores[worst] < args.min_gap:
            continue  # near-ties carry no signal
        out.append({"prompt": msgs,
                    "chosen": [{"role": "assistant", "content": completions[best]}],
                    "rejected": [{"role": "assistant", "content": completions[worst]}]})
    with open(args.pairs_out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} on-policy pairs to {args.pairs_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU linear Bradley-Terry (default without a GPU)")
    ap.add_argument("--full", action="store_true", help="TRL RewardTrainer / RM ranking (GPU)")
    ap.add_argument("--step", choices=["train", "rank"], default="train")
    ap.add_argument("--rm-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rm-path", default="rm-out")
    ap.add_argument("--policy", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--min-gap", type=float, default=0.5)
    ap.add_argument("--pairs-out", default="onpolicy_pairs.jsonl")
    args = ap.parse_args()

    run_full = args.full
    if not args.smoke and not args.full:
        run_full = torch.cuda.is_available()
    if not run_full:
        smoke()
    elif args.step == "train":
        full_train(args)
    else:
        full_rank(args)


if __name__ == "__main__":
    main()
