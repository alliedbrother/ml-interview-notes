"""Distillation losses and off-policy vs on-policy distillation.

Module: 14 — Distillation and Reasoning Fine-Tuning (LLM Fine-Tuning Deep Dive).

Smoke mode (CPU, stdlib + torch only, < 10 s):
  * numerically checks forward KL, reverse KL, generalised JSD and the T^2
    temperature scaling of classic KD;
  * runs two toy loops against a fixed teacher distribution over 8 tokens:
      - off-policy "SFT": sample from the teacher, minimise cross-entropy
        (a Monte Carlo estimate of forward KL);
      - on-policy distillation: sample from the student, use the per-token
        advantage A = log p_T(y) - log q_S(y) in a REINFORCE update
        (descends reverse KL);
  * asserts the student's reverse KL to the teacher falls by >= 50x under
    on-policy distillation, and forward KL falls by >= 20x under SFT.

Full mode (GPU; imports vllm / transformers / trl / datasets lazily):
  generate  L14.1 step 1, tier T2 for a 32B bf16 teacher (T1 with Qwen3-8B): sample traces from a teacher
            with vLLM, keep the ones whose boxed answer verifies.
  sft       L14.1 step 2: SFT a Qwen3 base model on the kept traces with TRL
            SFTTrainer (prompt-completion format); --lora fits tier T1, full FT T2.
  opd       L14.2, tier T2 (one 80 GB GPU): on-policy distillation with TRL
            DistillationTrainer (stable since TRL v1.10).
  Pinned versions for full mode: TRL 1.15, Transformers 5.x, vLLM >= 0.22,
  PEFT >= 0.18 (only if you add LoRA).

Run from the repository root:
  python content/courses/finetuning/code/kd_losses.py --smoke
  python content/courses/finetuning/code/kd_losses.py --full generate --teacher Qwen/Qwen3-32B
  python content/courses/finetuning/code/kd_losses.py --full sft --student Qwen/Qwen3-1.7B-Base --lora
  python content/courses/finetuning/code/kd_losses.py --full opd --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-1.7B
"""

import argparse
import json
import math
import os
import re

import torch


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def kl(p, q):
    """KL(p || q) for probability vectors on the last dim (0 log 0 = 0)."""
    p = p.to(torch.float64)
    q = q.to(torch.float64)
    terms = torch.where(p > 0, p * (p.clamp_min(1e-300).log() - q.clamp_min(1e-300).log()),
                        torch.zeros_like(p))
    return terms.sum(-1)


def generalised_jsd(p_teacher, q_student, beta):
    """TRL convention: beta=0 forward KL(p_T||q_S), beta=1 reverse KL(q_S||p_T)."""
    if beta == 0.0:
        return kl(p_teacher, q_student)
    if beta == 1.0:
        return kl(q_student, p_teacher)
    m = beta * p_teacher + (1 - beta) * q_student
    return beta * kl(p_teacher, m) + (1 - beta) * kl(q_student, m)


def temperature_kd(student_logits, teacher_logits, T):
    """Hinton et al. (2015): T^2 * KL(softmax(z_T/T) || softmax(z_S/T))."""
    p = torch.softmax(teacher_logits / T, dim=-1)
    log_q = torch.log_softmax(student_logits / T, dim=-1)
    return (T * T) * (p * (p.log() - log_q)).sum(-1)


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

TEACHER = torch.tensor([0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01], dtype=torch.float64)


def check_losses():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        p = torch.softmax(2 * torch.randn(8, generator=g, dtype=torch.float64), 0)
        q = torch.softmax(2 * torch.randn(8, generator=g, dtype=torch.float64), 0)
        assert kl(p, q) >= 0 and kl(q, p) >= 0
        j = generalised_jsd(p, q, 0.5)
        assert 0 <= j <= math.log(2) + 1e-12
        assert abs(float(generalised_jsd(p, q, 0.0) - kl(p, q))) < 1e-12
        assert abs(float(generalised_jsd(p, q, 1.0) - kl(q, p))) < 1e-12
    # T^2 scaling keeps the gradient magnitude comparable across temperatures.
    z_t = torch.tensor([3.0, 1.0, 0.2, -1.0, -2.0], dtype=torch.float64)
    norms = {}
    for T in (1.0, 4.0):
        z_s = torch.tensor([1.0, 2.0, 0.0, 0.5, -1.0], dtype=torch.float64, requires_grad=True)
        temperature_kd(z_s, z_t, T).backward()
        norms[T] = float(z_s.grad.norm())
    assert 0.2 <= norms[4.0] / norms[1.0] <= 5.0, norms
    print(f"losses ok; T^2-scaled grad norm T=1: {norms[1.0]:.3f}, T=4: {norms[4.0]:.3f}")


def reverse_kl_to_teacher(logits):
    q = torch.softmax(logits.detach().to(torch.float64), 0)
    return float(kl(q, TEACHER))


def forward_kl_to_teacher(logits):
    q = torch.softmax(logits.detach().to(torch.float64), 0)
    return float(kl(TEACHER, q))


def init_student():
    # Student starts confidently wrong: most mass on the teacher's least likely token.
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], requires_grad=True)


def toy_sft(steps=300, batch=64, lr=0.05, seed=0):
    """Off-policy: sample y ~ p_T, minimise -log q_S(y)  (MC forward KL)."""
    g = torch.Generator().manual_seed(seed)
    logits = init_student()
    opt = torch.optim.Adam([logits], lr=lr)
    start = forward_kl_to_teacher(logits)
    for _ in range(steps):
        y = torch.multinomial(TEACHER.float(), batch, replacement=True, generator=g)
        loss = -torch.log_softmax(logits, 0)[y].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return start, forward_kl_to_teacher(logits)


def toy_opd(steps=300, batch=64, lr=0.05, seed=0):
    """On-policy: sample y ~ q_S, advantage A = log p_T(y) - log q_S(y), REINFORCE."""
    g = torch.Generator().manual_seed(seed)
    log_p = TEACHER.float().log()
    logits = init_student()
    opt = torch.optim.Adam([logits], lr=lr)
    start = reverse_kl_to_teacher(logits)
    for _ in range(steps):
        log_q = torch.log_softmax(logits, 0)
        y = torch.multinomial(log_q.exp().detach(), batch, replacement=True, generator=g)
        adv = (log_p[y] - log_q[y]).detach()          # per-token negative reverse KL
        adv = adv - adv.mean()                        # baseline: lowers variance, no bias
        loss = -(adv * log_q[y]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return start, reverse_kl_to_teacher(logits)


def smoke():
    torch.manual_seed(0)
    check_losses()
    f0, f1 = toy_sft()
    r0, r1 = toy_opd()
    print(f"off-policy SFT : forward KL {f0:.4f} -> {f1:.5f}  ({f0 / f1:.0f}x)")
    print(f"on-policy OPD  : reverse KL {r0:.4f} -> {r1:.5f}  ({r0 / r1:.0f}x)")
    assert f0 / f1 >= 20, (f0, f1)
    assert r0 / r1 >= 50, (r0, r1)
    print("smoke OK: on-policy distillation cut the student's reverse KL by >= 50x")


# ---------------------------------------------------------------------------
# Full mode (GPU). Heavy imports stay inside these functions.
# ---------------------------------------------------------------------------

BOXED_HINT = "\nPut the final answer in \\boxed{}."
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text):
    hits = BOXED.findall(text)
    return hits[-1].strip() if hits else None


def normalise_answer(ans):
    return None if ans is None else re.sub(r"\s+", "", ans).rstrip(".").replace("\\!", "")


def full_generate(args):
    """L14.1 step 1: teacher traces with vLLM, filtered by answer correctness."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    ds = ds.shuffle(seed=0).select(range(args.n_prompts))
    tok = AutoTokenizer.from_pretrained(args.teacher)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": row["problem"] + BOXED_HINT}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for row in ds
    ]
    llm = LLM(model=args.teacher, dtype="bfloat16", max_model_len=args.max_length)
    params = SamplingParams(n=args.samples_per_prompt, temperature=0.6, top_p=0.95, top_k=20,
                            max_tokens=args.max_length - 512)
    outputs = llm.generate(prompts, params)
    kept, lengths = [], []
    for row, out in zip(ds, outputs):
        gold = normalise_answer(row["answer"])
        for cand in out.outputs:
            text = cand.text
            if "</think>" not in text:                     # truncated mid-thought
                continue
            if normalise_answer(extract_boxed(text.split("</think>")[-1])) != gold:
                continue
            kept.append({
                "prompt": [{"role": "user", "content": row["problem"] + BOXED_HINT}],
                "completion": [{"role": "assistant", "content": text.strip()}],
            })
            lengths.append(len(cand.token_ids))
            break                                          # one trace per prompt
    with open(args.traces, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"kept {len(kept)}/{len(ds)} prompts; median trace length "
          f"{sorted(lengths)[len(lengths) // 2] if lengths else 0} tokens -> {args.traces}")


def full_sft(args):
    """L14.1 step 2: SFT on the filtered traces (prompt-completion, completion-only loss)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train = load_dataset("json", data_files=args.traces, split="train")
    # The base checkpoint shares the Qwen3 vocabulary; use the instruct chat template.
    tok = AutoTokenizer.from_pretrained(args.template_from)
    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,             # default 1024 would cut reasoning traces
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=10,
        save_steps=50,
    )
    peft_config = None
    if args.lora:                               # fits a T1 card; full FT of 1.7B wants T2
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=32, target_modules="all-linear",
                                 task_type="CAUSAL_LM")
        cfg.learning_rate = 1e-4
    trainer = SFTTrainer(model=args.student, args=cfg, train_dataset=train,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output_dir)


def full_opd(args):
    """L14.2: on-policy distillation with TRL DistillationTrainer."""
    from datasets import load_dataset
    from trl import DistillationConfig, DistillationTrainer

    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    ds = ds.shuffle(seed=0).select(range(args.n_prompts))
    train = ds.map(lambda r: {"prompt": [{"role": "user", "content": r["problem"] + BOXED_HINT}]},
                   remove_columns=ds.column_names)
    cfg = DistillationConfig(
        output_dir=args.output_dir + "-opd",
        beta=1.0,                                    # reverse KL
        temperature=1.0,
        max_completion_length=args.max_completion_length,
        chat_template_kwargs={"enable_thinking": True},
        model_init_kwargs={"dtype": "bfloat16"},
        teacher_model_init_kwargs={"dtype": "bfloat16"},
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        logging_steps=5,
        save_steps=50,
        log_completions=True,
    )
    trainer = DistillationTrainer(model=args.student, teacher_model=args.teacher,
                                  args=cfg, train_dataset=train)
    trainer.train()
    trainer.save_model(cfg.output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="CPU checks and toy loops (default)")
    ap.add_argument("--full", choices=["generate", "sft", "opd"], help="GPU lab step")
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B")
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--template-from", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--lora", action="store_true", help="LoRA on all linear layers (sft step)")
    ap.add_argument("--traces", default="l14_traces.jsonl")
    ap.add_argument("--output-dir", default="qwen3-1.7b-distill")
    args = ap.parse_args()

    if args.full is None:
        smoke()
        return
    if not torch.cuda.is_available():
        raise SystemExit("--full needs a GPU; run --smoke on CPU")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"generate": full_generate, "sft": full_sft, "opd": full_opd}[args.full](args)


if __name__ == "__main__":
    main()
