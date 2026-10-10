"""DPO from scratch, plus the TRL DPOTrainer runs for SFT -> DPO and the loss-type ablation (module 12, L12.1/L12.2).

Purpose
    --smoke : pure-torch DPO on a tabular categorical "policy". Each of 16 prompts has 6 fixed
              candidate responses; the policy is a table of logits (16, 6), the reference is a
              frozen copy of the initial table, and preference pairs are labelled by a hidden
              true reward. Logs the four TRL reward metrics (rewards/chosen, rewards/rejected,
              rewards/margins, rewards/accuracies) and asserts that the training-pair accuracy
              reaches 1.0 and the mean margin grows. A second, designed fixture uses a linear
              feature policy in which the chosen and rejected responses share a feature; it
              reproduces "both log-probs fall while the margin grows" (likelihood displacement)
              and asserts it.
    --full  : TRL DPOTrainer (LoRA) on trl-lib/ultrafeedback_binarized, starting either from an
              SFT adapter you trained in module 05 (the reference becomes a frozen copy of that
              adapter) or from a fresh LoRA on the base model (reference = adapter disabled).
              Pass --loss-type sigmoid / ipo / sigmoid_norm / apo_zero for the L12.2 ablation.
              Pass --dataset onpolicy_pairs.jsonl (written by reward_model_bt.py --step rank) to
              train one round of iterative DPO on your own on-policy pairs (L12.3).

Module      content/courses/finetuning/12-preference-optimization.md
Hardware    smoke: any CPU, a few seconds.  full: T1 (24 GB) for Qwen3-1.7B or Qwen3-4B + LoRA.
Pinned      smoke: Python 3.11+, torch 2.8 CPU.  full (as of October 2026): trl==1.15.0,
            transformers>=5.2, peft>=0.18, datasets.

Run (from the repository root)
    python content/courses/finetuning/code/dpo_from_scratch.py --smoke
    python content/courses/finetuning/code/dpo_from_scratch.py --full --model Qwen/Qwen3-1.7B --loss-type sigmoid
    python content/courses/finetuning/code/dpo_from_scratch.py --full --sft-adapter ./sft-out --loss-type ipo --beta 0.5
    python content/courses/finetuning/code/dpo_from_scratch.py --full --dataset onpolicy_pairs.jsonl --output-dir dpo-iter1
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Smoke mode, part 1: tabular DPO with synthetic preferences
# --------------------------------------------------------------------------------------


def dpo_metrics(pol_w, pol_l, ref_w, ref_l, beta):
    """TRL-style metrics from summed sequence log-probs, each of shape (N,)."""
    chosen_rewards = beta * (pol_w - ref_w)
    rejected_rewards = beta * (pol_l - ref_l)
    margins = chosen_rewards - rejected_rewards
    return {
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/margins": margins.mean().item(),
        "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
        "logps/chosen": pol_w.mean().item(),
        "logps/rejected": pol_l.mean().item(),
    }


def tabular_dpo(steps: int = 300, beta: float = 0.1, lr: float = 0.5, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    n_prompts, n_resp = 16, 6
    ref_logits = torch.randn(n_prompts, n_resp, generator=gen)  # (P, K) frozen reference
    true_reward = torch.randn(n_prompts, n_resp, generator=gen)  # (P, K) hidden "human" reward
    # Three preference pairs per prompt, labelled by the true reward (noise-free here).
    pairs = []
    for p in range(n_prompts):
        combos = list(itertools.combinations(range(n_resp), 2))
        idx = torch.randperm(len(combos), generator=gen)[:3]
        for c in idx.tolist():
            i, j = combos[c]
            w, l = (i, j) if true_reward[p, i] > true_reward[p, j] else (j, i)
            pairs.append((p, w, l))
    P = torch.tensor([p for p, _, _ in pairs])
    W = torch.tensor([w for _, w, _ in pairs])
    L = torch.tensor([l for _, _, l in pairs])

    theta = ref_logits.clone().requires_grad_(True)  # policy initialised at the reference
    opt = torch.optim.Adam([theta], lr=lr)
    ref_logp = F.log_softmax(ref_logits, dim=-1)
    history = []
    for step in range(steps + 1):
        logp = F.log_softmax(theta, dim=-1)  # (P, K)
        pol_w, pol_l = logp[P, W], logp[P, L]
        ref_w, ref_l = ref_logp[P, W], ref_logp[P, L]
        h = beta * ((pol_w - ref_w) - (pol_l - ref_l))
        loss = -F.logsigmoid(h).mean()
        if step % 50 == 0:
            m = dpo_metrics(pol_w.detach(), pol_l.detach(), ref_w, ref_l, beta)
            m["loss"] = loss.item()
            history.append((step, m))
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return history


# --------------------------------------------------------------------------------------
# Smoke mode, part 2: the designed "both log-probs fall" fixture
# --------------------------------------------------------------------------------------


def both_fall_fixture(steps: int = 20, beta: float = 0.1, lr: float = 1.0):
    """Linear-feature policy: logits = phi @ theta. Chosen and rejected share feature 0.

    The DPO gradient only moves theta along phi_w - phi_l = [0, 1]. Response 2 loads more
    heavily on that direction than the chosen response does, so it soaks up probability mass:
    the margin grows while both the chosen and the rejected log-probs fall.
    """
    phi = torch.tensor([[1.0, 1.0],   # 0: chosen
                        [1.0, 0.0],   # 1: rejected
                        [0.0, 4.0],   # 2: an unseen response that shares the "preferred" feature
                        [0.0, 0.0]])  # 3: neutral
    theta = torch.zeros(2, requires_grad=True)
    ref = F.log_softmax(phi @ torch.zeros(2), dim=-1)
    opt = torch.optim.SGD([theta], lr=lr)
    rows = []
    for step in range(steps + 1):
        logp = F.log_softmax(phi @ theta, dim=-1)
        h = beta * ((logp[0] - ref[0]) - (logp[1] - ref[1]))
        loss = -F.logsigmoid(h)
        rows.append((step, logp[0].item(), logp[1].item(), logp[2].exp().item(), h.item(), loss.item()))
        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rows


def smoke() -> None:
    torch.manual_seed(0)
    history = tabular_dpo()
    print("step  loss    r/chosen  r/rejected  margin   acc")
    for step, m in history:
        print(f"{step:4d}  {m['loss']:.4f}  {m['rewards/chosen']:+.4f}  {m['rewards/rejected']:+.4f}"
              f"     {m['rewards/margins']:.4f}  {m['rewards/accuracies']:.3f}")
    first, last = history[0][1], history[-1][1]
    assert abs(first["rewards/margins"]) < 1e-6 and abs(first["loss"] - 0.693147) < 1e-5
    assert last["rewards/accuracies"] == 1.0, last
    margins = [m["rewards/margins"] for _, m in history]
    assert all(b > a for a, b in zip(margins, margins[1:])), margins
    assert last["loss"] < first["loss"]

    rows = both_fall_fixture()
    s0, s1 = rows[0], rows[-1]
    print("\nboth-fall fixture: step  logp(chosen)  logp(rejected)  p(unseen)  margin")
    for r in (s0, rows[len(rows) // 2], s1):
        print(f"                   {r[0]:4d}  {r[1]:+.4f}       {r[2]:+.4f}        {r[3]:.4f}     {r[4]:.4f}")
    assert s1[1] < s0[1] and s1[2] < s0[2], "both chosen and rejected log-probs should fall"
    assert s1[4] > s0[4] and s1[5] < s0[5], "margin should grow and loss should fall"
    assert s1[3] > s0[3], "mass should move to the unseen response"
    print("\nsmoke OK: accuracy -> 1.0, margins increase monotonically, both-fall reproduced")


# --------------------------------------------------------------------------------------
# Full mode: TRL DPOTrainer (GPU). Imports stay inside so --smoke never needs them.
# --------------------------------------------------------------------------------------


def full(args) -> None:
    import torch as _torch
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig
    from trl import DPOConfig, DPOTrainer

    if args.dataset.endswith((".jsonl", ".json")):
        # Local preference pairs ({"prompt", "chosen", "rejected"}), e.g. the L12.3 on-policy JSONL.
        dataset = load_dataset("json", data_files=args.dataset, split="train").train_test_split(
            test_size=0.05, seed=0)
    else:
        dataset = load_dataset(args.dataset)
    loss_types = args.loss_type.split(",")
    weights = [float(w) for w in args.loss_weights.split(",")] if args.loss_weights else None
    config = DPOConfig(
        output_dir=args.output_dir or f"dpo-{'-'.join(loss_types)}",
        loss_type=loss_types,
        loss_weights=weights,
        beta=args.beta,
        learning_rate=args.lr,  # LoRA tolerates ~10x the full-FT default of 1e-6
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_steps=args.max_steps,
        max_length=1024,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        model_init_kwargs={"dtype": _torch.bfloat16},
        report_to=args.report_to,
    )
    if args.sft_adapter:
        # Continue the module-05 SFT adapter. TRL loads a frozen copy of it as the "ref" adapter.
        model = AutoPeftModelForCausalLM.from_pretrained(args.sft_adapter, is_trainable=True,
                                                         dtype=_torch.bfloat16)
        peft_config = None
    else:
        # Fresh LoRA: the reference is the same model with the adapter disabled.
        model = args.model
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"].select(range(min(500, len(dataset["test"])))),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    print("Saved to", config.output_dir, "- now generate on your eval prompts and judge them (module 09).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU tabular DPO (default without a GPU)")
    ap.add_argument("--full", action="store_true", help="TRL DPOTrainer + LoRA on UltraFeedback (GPU)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sft-adapter", default=None, help="path to the module-05 SFT LoRA adapter")
    ap.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized",
                    help="Hub preference dataset id, or a local .jsonl of prompt/chosen/rejected rows")
    ap.add_argument("--loss-type", default="sigmoid", help="comma-separated, e.g. sigmoid,bco_pair,sft")
    ap.add_argument("--loss-weights", default=None, help="comma-separated weights, e.g. 0.8,0.2,1.0")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    run_full = args.full
    if not args.smoke and not args.full:
        run_full = torch.cuda.is_available()
    if run_full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
