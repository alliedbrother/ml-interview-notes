"""Catastrophic forgetting study: plain fine-tuning vs L2-SP vs replay (toy), and
full FT vs LoRA vs LoRA + replay (real).

Purpose
    Show that forgetting is controllable. Smoke mode pretrains a tiny MLP on task A,
    fine-tunes it on a shifted task B three ways (plain, L2-SP toward the task-A weights,
    replay of task-A data) for a fixed step budget, and asserts that the regularised and
    replay variants keep task A much better while still learning task B. Full mode runs
    lab L19.3: SFT of Qwen/Qwen3-0.6B on the course's support-email extraction data with
    full fine-tuning, LoRA, and LoRA plus general-data replay, then prints the
    lm-evaluation-harness commands that measure IFEval / GSM8K / MMLU-Pro deltas.

Module
    LLM Fine-Tuning Deep Dive, module 19 (Safety, Forgetting, and Responsible
    Fine-Tuning), lab L19.3.

Hardware tier
    --smoke: CPU, torch only, about one second.
    --full : T1 (single 24 GB GPU, bf16). Full FT of a 0.6B model fits; LoRA runs are
             lighter. Expect roughly an hour per variant on an L4-class GPU for a few
             thousand examples (an estimate, not a measurement).

Pinned versions for the full lab (as of October 2026)
    Python 3.11+, torch 2.8+, transformers 5.x, trl 1.15, peft 0.21.x, datasets 4.x,
    lm_eval[hf] 0.4.13 (for the evaluation commands it prints).

Run commands (from the repository root)
    python content/courses/finetuning/code/forgetting_study.py --smoke
    python content/courses/finetuning/code/forgetting_study.py --full \
        --train out/conversational.jsonl --out runs/forgetting
"""

from __future__ import annotations

import argparse
import os

D_IN, HIDDEN = 8, 32


# ----------------------------------------------------------------------------- smoke
def _toy_problem():
    import torch

    def teacher(seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(D_IN, 16, generator=g) / D_IN ** 0.5
        v = torch.randn(16, 1, generator=g) / 4
        return lambda x: torch.tanh(x @ w) @ v

    f_a, f_extra = teacher(1), teacher(2)
    g = torch.Generator().manual_seed(3)
    xa = torch.randn(256, D_IN, generator=g)               # task A inputs
    xb = torch.randn(32, D_IN, generator=g) + 1.0           # task B: shifted inputs
    ya = f_a(xa)
    yb = f_a(xb) + 0.5 * f_extra(xb)                        # related but different target
    return xa, ya, xb, yb


def _mlp():
    import torch

    return torch.nn.Sequential(torch.nn.Linear(D_IN, HIDDEN), torch.nn.Tanh(),
                               torch.nn.Linear(HIDDEN, 1))


def run_toy(steps: int = 60, lam: float = 0.1, replay_weight: float = 1.0):
    import torch

    mse = torch.nn.functional.mse_loss
    xa, ya, xb, yb = _toy_problem()
    torch.manual_seed(0)
    net = _mlp()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(500):                                    # "pretraining" on task A
        opt.zero_grad()
        mse(net(xa), ya).backward()
        opt.step()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    theta_a = [p.detach().clone() for p in net.parameters()]

    def evaluate(model):
        with torch.no_grad():
            return mse(model(xa), ya).item(), mse(model(xb), yb).item()

    base = evaluate(net)
    results = {"before": base}
    for name in ("plain", "l2sp", "replay"):
        m = _mlp()
        m.load_state_dict(state)
        o = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(steps):
            o.zero_grad()
            loss = mse(m(xb), yb)
            if name == "l2sp":                              # lambda * ||theta - theta_A||^2
                loss = loss + lam * sum(((p - q) ** 2).sum() for p, q in zip(m.parameters(), theta_a))
            if name == "replay":                            # 32 task-A examples replayed
                loss = loss + replay_weight * mse(m(xa[:32]), ya[:32])
            loss.backward()
            o.step()
        drift = sum(((p.detach() - q) ** 2).sum().item() for p, q in zip(m.parameters(), theta_a)) ** 0.5
        results[name] = evaluate(m) + (drift,)
    return results


def smoke() -> None:
    r = run_toy()
    a0, b0 = r["before"]
    print(f"before FT      task-A MSE {a0:.4f}  task-B MSE {b0:.4f}")
    for name in ("plain", "l2sp", "replay"):
        a, b, drift = r[name]
        print(f"{name:13s}  task-A MSE {a:.4f}  task-B MSE {b:.4f}  ||theta-theta_A|| {drift:.3f}")
    inc = {k: r[k][0] - a0 for k in ("plain", "l2sp", "replay")}
    gain = {k: b0 - r[k][1] for k in ("plain", "l2sp", "replay")}
    for k in ("l2sp", "replay"):
        assert inc[k] <= 0.7 * inc["plain"], f"{k} should cut forgetting by >= 30%"
        assert gain[k] >= 0.9 * gain["plain"], f"{k} should keep >= 90% of the task-B gain"
    assert r["l2sp"][2] < r["plain"][2], "L2-SP must stay closer to theta_A"
    print(f"forgetting cut: l2sp {1 - inc['l2sp'] / inc['plain']:.0%}, "
          f"replay {1 - inc['replay'] / inc['plain']:.0%}")
    print("smoke OK: L2-SP and replay retain task A while learning task B")


# ----------------------------------------------------------------------------- full
def full(args) -> None:
    import torch
    from datasets import concatenate_datasets, load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    domain = load_dataset("json", data_files=args.train, split="train").select_columns(["messages"])
    general = (load_dataset("allenai/tulu-3-sft-mixture", split="train")
               .shuffle(seed=0).select(range(int(len(domain) * args.replay_frac)))
               .select_columns(["messages"]))
    variants = {
        "full": dict(peft=None, data=domain, lr=1e-5),
        "lora": dict(peft=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"), data=domain, lr=1e-4),
        "lora_replay": dict(peft=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
                            data=concatenate_datasets([domain, general]).shuffle(seed=0), lr=1e-4),
    }
    for name, v in variants.items():
        out_dir = os.path.join(args.out, name)
        cfg = SFTConfig(
            output_dir=out_dir,
            learning_rate=v["lr"],
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            max_length=1024,
            assistant_only_loss=True,
            model_init_kwargs={"dtype": torch.bfloat16},
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )
        trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=v["data"], peft_config=v["peft"])
        trainer.train()
        trainer.save_model(out_dir)
        del trainer
        torch.cuda.empty_cache()

    tasks = "ifeval,gsm8k,mmlu_pro"
    print("\nEvaluate base and each variant with the same harness settings (module 09):")
    print(f"lm_eval run --model hf --model_args pretrained={args.model},dtype=bfloat16 "
          f"--tasks {tasks} --apply_chat_template --log_samples --output_path {args.out}/eval/base")
    print(f"lm_eval run --model hf --model_args pretrained={args.out}/full,dtype=bfloat16 "
          f"--tasks {tasks} --apply_chat_template --log_samples --output_path {args.out}/eval/full")
    for name in ("lora", "lora_replay"):
        print(f"lm_eval run --model hf --model_args pretrained={args.model},peft={args.out}/{name},"
              f"dtype=bfloat16 --tasks {tasks} --apply_chat_template --log_samples "
              f"--output_path {args.out}/eval/{name}")
    print("Then compute paired deltas with CIs using eval_toolkit.py from module 09.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy study (default)")
    mode.add_argument("--full", action="store_true", help="T1 GPU study (L19.3)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", default="out/conversational.jsonl",
                    help="conversational JSONL from module 04's build_sft_dataset.py")
    ap.add_argument("--replay-frac", type=float, default=0.2, help="general rows per domain row")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--out", default="runs/forgetting")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
