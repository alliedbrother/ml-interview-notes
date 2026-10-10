"""GRPO from scratch, plus the TRL GRPOTrainer ablation (module 13, labs L13.2 and L13.3).

Purpose
    --smoke : a complete GRPO loop on a tabular softmax policy. The "model" is a table of
              logits indexed by (prompt, position, token); prompts are toy additions a+b,
              completions are short token strings (digits then EOS), rewards come from an
              exact verifier. Implements group/batch/no reward scaling, the grpo, dapo and
              dr_grpo loss aggregations, clip-higher (epsilon_high), multiple updates per
              rollout batch (num_iterations), mask_truncated_completions and an optional k3
              KL penalty to the initial policy. Asserts that the expected reward rises above
              0.80 for every loss type and that a zero-variance group contributes exactly
              zero gradient.
    --full  : TRL GRPOTrainer with vLLM colocate on openai/gsm8k, for the loss_type x
              scale_rewards ablation (L13.2) and the GSPO sequence-level ratio run (L13.3).

Module      content/courses/finetuning/13-rl-with-verifiable-rewards.md
Hardware    smoke: any CPU, about 10-20 s.  full: T1 (24 GB, Qwen3-1.7B + LoRA) or
            T2 (80 GB, Qwen3-4B/8B or a small MoE for L13.3).
Pinned      smoke: Python 3.11+, torch 2.8 CPU.  full (as of October 2026): trl==1.15.0,
            transformers>=5.2, peft>=0.18, vllm within TRL's supported window, datasets.

Run (from the repository root)
    python content/courses/finetuning/code/grpo_from_scratch.py --smoke
    python content/courses/finetuning/code/grpo_from_scratch.py --full --loss-type dr_grpo --scale-rewards none
"""

from __future__ import annotations

import argparse
import math
import re
import time

import torch

# --------------------------------------------------------------------------------------
# Smoke mode: a tabular policy over short token strings
# --------------------------------------------------------------------------------------

DIGITS = 10
EOS = 10
V = 11  # vocabulary: digits 0..9 plus EOS
L_MAX = 4  # max_completion_length in tokens


def make_prompts():
    """All a+b with a, b in 0..4 (25 prompts, answers 0..8)."""
    return [(a, b) for a in range(5) for b in range(5)]


def init_logits(prompts, seed=0):
    """A 'base model': weak prior towards the right digit and towards stopping.

    Prior strength varies by prompt so that some prompts start easy and some hard,
    which is what makes group advantages informative.
    """
    g = torch.Generator().manual_seed(seed)
    logits = 0.5 * torch.randn(len(prompts), L_MAX, V, generator=g)  # (P, L, V)
    for p, (a, b) in enumerate(prompts):
        strength = 0.5 + 2.5 * ((a * 7 + b * 3) % 5) / 4  # in [0.5, 3.0]
        logits[p, 0, a + b] += strength
        logits[p, 0, EOS] -= 2.0  # an empty answer is unlikely
        logits[p, 1:, EOS] += 1.0  # mild prior to stop after the first token
    return logits


def exact_pass_rate(logits, prompts):
    """P(completion == [answer, EOS]) per prompt; positions are independent in this policy."""
    probs = logits.softmax(-1)
    ans = torch.tensor([a + b for a, b in prompts])
    return probs[torch.arange(len(prompts)), 0, ans] * probs[:, 1, EOS]  # (P,)


def sample(logits, prompt_idx, G, gen):
    """Sample G completions for each prompt index. Returns tokens (B, L), mask (B, L), truncated (B,)."""
    probs = logits[prompt_idx].softmax(-1)  # (n, L, V)
    probs = probs.repeat_interleave(G, dim=0)  # (B, L, V) with B = n * G
    B = probs.shape[0]
    tokens = torch.multinomial(probs.reshape(-1, V), 1, generator=gen).reshape(B, L_MAX)
    is_eos = tokens == EOS
    # mask covers tokens up to and including the first EOS
    first_eos = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((B,), L_MAX))
    pos = torch.arange(L_MAX).unsqueeze(0)
    mask = (pos <= first_eos.unsqueeze(1)).float()  # (B, L)
    truncated = ~is_eos.any(1)
    return tokens, mask, truncated


def verify(tokens, mask, truncated, prompt_idx, prompts, G):
    """Exact verifier: reward 1 iff the digits before EOS spell the answer."""
    answers = [prompts[p][0] + prompts[p][1] for p in prompt_idx.repeat_interleave(G).tolist()]
    rewards = []
    for row, m, tr, ans in zip(tokens.tolist(), mask.tolist(), truncated.tolist(), answers):
        if tr:
            rewards.append(0.0)
            continue
        n = int(sum(m)) - 1  # tokens before EOS
        digits = row[:n]
        ok = n > 0 and all(t < DIGITS for t in digits) and int("".join(map(str, digits))) == ans
        rewards.append(1.0 if ok else 0.0)
    return torch.tensor(rewards)


def nanstd_unbiased(x, dim=None):
    """Bessel-corrected std, matching TRL's nanstd for non-NaN input."""
    return x.std(dim=dim, unbiased=True) if dim is not None else x.std(unbiased=True)


def advantages(rewards, G, scale_rewards="group", eps=1e-4):
    grouped = rewards.view(-1, G)
    mean = grouped.mean(1).repeat_interleave(G)
    adv = rewards - mean
    if scale_rewards == "group":
        std = nanstd_unbiased(grouped, dim=1).repeat_interleave(G)
        adv = adv / (std + eps)
    elif scale_rewards == "batch":
        adv = adv / (nanstd_unbiased(rewards) + eps)
    elif scale_rewards != "none":
        raise ValueError(scale_rewards)
    return adv


def token_logps(logits, prompt_idx, tokens, G):
    lp = logits[prompt_idx].log_softmax(-1).repeat_interleave(G, dim=0)  # (B, L, V)
    return lp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, loss_type, eps_low, eps_high, beta):
    ratio = torch.exp(logp - old_logp)  # (B, L)
    clipped = ratio.clamp(1 - eps_low, 1 + eps_high)
    A = adv.unsqueeze(1)
    per_token = -torch.min(ratio * A, clipped * A)
    if beta != 0.0:
        log_r = ref_logp - logp  # r = pi_ref / pi
        per_token = per_token + beta * (log_r.exp() - log_r - 1.0)  # k3 >= 0
    if loss_type == "grpo":
        return ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    if loss_type == "dapo":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    if loss_type == "dr_grpo":
        return (per_token * mask).sum() / (mask.shape[0] * L_MAX)
    raise ValueError(loss_type)


def train_smoke(loss_type, steps=200, G=8, prompts_per_batch=8, num_iterations=2, lr=0.1,
                eps_low=0.2, eps_high=0.28, beta=0.0, scale_rewards="group",
                mask_truncated=True, seed=0):
    prompts = make_prompts()
    init = init_logits(prompts, seed)
    logits = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    gen = torch.Generator().manual_seed(seed + 1)
    start = exact_pass_rate(init, prompts).mean().item()
    zero_std_frac = []
    updates = 0
    while updates < steps:
        prompt_idx = torch.randperm(len(prompts), generator=gen)[:prompts_per_batch]
        with torch.no_grad():
            tokens, mask, truncated = sample(logits, prompt_idx, G, gen)
            rewards = verify(tokens, mask, truncated, prompt_idx, prompts, G)
            adv = advantages(rewards, G, scale_rewards)
            old_logp = token_logps(logits, prompt_idx, tokens, G)
            ref_logp = token_logps(init, prompt_idx, tokens, G)
            zero_std_frac.append((rewards.view(-1, G).std(1) == 0).float().mean().item())
            if mask_truncated:
                mask = mask * (~truncated).float().unsqueeze(1)
        for _ in range(num_iterations):  # mu > 1: ratios move away from 1 on the second pass
            logp = token_logps(logits, prompt_idx, tokens, G)
            loss = grpo_loss(logp, old_logp, ref_logp, adv, mask, loss_type, eps_low, eps_high, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
    end = exact_pass_rate(logits.detach(), prompts).mean().item()
    return start, end, sum(zero_std_frac) / len(zero_std_frac)


def zero_std_group_has_zero_gradient():
    """A group whose rewards are all equal gets advantage 0 and hence zero gradient (beta=0)."""
    prompts = make_prompts()
    logits = init_logits(prompts).requires_grad_(True)
    G = 8
    prompt_idx = torch.tensor([3])
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        tokens, mask, _ = sample(logits, prompt_idx, G, gen)
        old = token_logps(logits, prompt_idx, tokens, G)
    rewards = torch.ones(G)  # all correct: zero variance
    adv = advantages(rewards, G, "group")
    for loss_type in ("grpo", "dapo", "dr_grpo"):
        logits.grad = None
        loss = grpo_loss(token_logps(logits, prompt_idx, tokens, G), old, old, adv, mask,
                         loss_type, 0.2, 0.28, 0.0)
        loss.backward()
        assert torch.count_nonzero(logits.grad) == 0, loss_type
    return True


def smoke():
    torch.manual_seed(0)
    t0 = time.time()
    threshold = 0.80
    assert zero_std_group_has_zero_gradient()
    print("zero-variance group -> zero gradient: ok")
    for loss_type in ("grpo", "dapo", "dr_grpo"):
        start, end, zf = train_smoke(loss_type)
        print(f"{loss_type:8s} expected reward {start:.3f} -> {end:.3f}  "
              f"(mean frac_reward_zero_std {zf:.2f})")
        assert end > threshold, (loss_type, end)
        assert end > start + 0.3, (loss_type, start, end)
    start, end, _ = train_smoke("dapo", beta=0.04)
    print(f"dapo+k3  expected reward {start:.3f} -> {end:.3f}  (beta=0.04 KL to the initial policy)")
    assert end > start
    print(f"smoke passed in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------------------
# Full mode: TRL GRPOTrainer + vLLM (GPU). Imports stay inside so --smoke never needs them.
# --------------------------------------------------------------------------------------

SYSTEM = ("Reason step by step inside <think>...</think>, then give only the final number "
          "inside \\boxed{}.")


def _extract(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = m[-1] if m else None
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def correctness_reward(prompts, completions, answer, **kwargs):
    out = []
    for c, gold in zip(completions, answer):
        text = c[0]["content"] if isinstance(c, list) else c
        pred = _extract(text)
        out.append(1.0 if pred is not None and math.isclose(pred, float(gold)) else 0.0)
    return out


def format_reward(completions, **kwargs):
    pat = re.compile(r"^<think>.+?</think>\s*.*\\boxed\{[^{}]+\}\s*$", re.S)
    return [0.2 if pat.match(c[0]["content"] if isinstance(c, list) else c) else 0.0
            for c in completions]


def full(args):
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": x["question"]}],
        "answer": x["answer"].split("####")[-1].strip().replace(",", ""),
    }, remove_columns=ds.column_names)

    scale = {"none": False}.get(args.scale_rewards, args.scale_rewards)
    loss_type = args.loss_type
    if args.is_level == "sequence" and loss_type not in ("grpo", "luspo", "sapo"):
        # token-summing losses weight each sequence by its length at sequence level; GSPO paper uses "grpo"
        loss_type = "grpo"
    cfg = GRPOConfig(
        output_dir=f"grpo-{loss_type}-{args.scale_rewards}-{args.is_level}",
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=True,
        learning_rate=args.lr,
        loss_type=loss_type,
        scale_rewards=scale,
        importance_sampling_level=args.is_level,
        epsilon=0.2 if args.is_level == "token" else 3e-4,
        epsilon_high=0.28 if args.is_level == "token" else 4e-4,
        num_generations=8,
        per_device_train_batch_size=8,  # completions, not prompts
        gradient_accumulation_steps=4,
        steps_per_generation=8 if args.is_level == "sequence" else 4,  # > GA makes updates off-policy
        max_completion_length=1024,
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        max_steps=args.max_steps,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=2,
        report_to=args.report_to,
    )
    peft_cfg = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward],
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg if args.lora else None,
    )
    trainer.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU tabular GRPO (default without a GPU)")
    ap.add_argument("--full", action="store_true", help="TRL GRPOTrainer + vLLM on GSM8K (GPU)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--loss-type", default="dapo", choices=["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo"])
    ap.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"])
    ap.add_argument("--is-level", default="token", choices=["token", "sequence"])
    ap.add_argument("--lr", type=float, default=1e-5, help="1e-5 suits LoRA; use ~1e-6 for full FT")
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
