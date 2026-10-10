"""Catastrophic forgetting study: plain fine-tuning vs L2-SP vs replay (toy), and
full FT vs LoRA vs LoRA + replay (real).

Purpose
    Show that forgetting is controllable. Smoke mode pretrains a tiny MLP on task A,
    fine-tunes it on a shifted task B three ways (plain, L2-SP toward the task-A weights,
    replay of task-A data) for a fixed step budget, and asserts that the regularised and
    replay variants keep task A much better while still learning task B. With --train it
    also validates a running-example file (module 01 schema) and converts every row to
    the SFT format the full run uses. Full mode runs lab L19.3: SFT of Qwen/Qwen3-0.6B on
    the support-email extraction split (`ft_data/train.jsonl`, written by module 01's
    fit_or_not_baseline.py) with full fine-tuning, LoRA, and LoRA plus general-data
    replay, then prints the lm-evaluation-harness commands that measure IFEval / GSM8K /
    MMLU-Pro deltas.

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
    python content/courses/finetuning/code/fit_or_not_baseline.py --data-only --out ft_data
    python content/courses/finetuning/code/forgetting_study.py --smoke --train ft_data/train.jsonl
    python content/courses/finetuning/code/forgetting_study.py --full \
        --train ft_data/train.jsonl --out runs/forgetting
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re

D_IN, HIDDEN = 8, 32

# Running-example contract (module 01, copied verbatim; see fit_or_not_baseline.py).
FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
ISSUE_TYPES = ["refund", "shipping", "defect", "account", "other"]
SENTIMENTS = ["positive", "neutral", "negative"]
ORDER_ID_RE = r"^ORD-[0-9]{5}$"
SYSTEM_PROMPT = (
    "You extract structured data from customer support emails. Reply with one JSON "
    "object and nothing else, with exactly these keys: "
    '"customer_name" (string), '
    '"order_id" (string: "ORD-" followed by the five-digit order number, e.g. "ORD-04217"), '
    '"issue_type" (one of "refund", "shipping", "defect", "account", "other"), '
    '"refund_requested" (true or false), '
    '"sentiment" (one of "positive", "neutral", "negative").'
)


def load_task_rows(path: str) -> list[dict]:
    """Load module 01 rows {"id", "email", "gold"}; any deviation from the schema is an error."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            r = json.loads(line)
            g = r.get("gold")
            assert set(r) == {"id", "email", "gold"}, f"{path}:{n}: keys {sorted(r)}"
            assert isinstance(g, dict) and list(g) == FIELDS, f"{path}:{n}: gold keys"
            assert isinstance(g["customer_name"], str) and g["customer_name"], f"{path}:{n}"
            assert re.match(ORDER_ID_RE, g["order_id"]), f"{path}:{n}: order_id"
            assert g["issue_type"] in ISSUE_TYPES and g["sentiment"] in SENTIMENTS, f"{path}:{n}"
            assert isinstance(g["refund_requested"], bool), f"{path}:{n}: refund_requested"
            rows.append(r)
    return rows


def to_prompt_completion(row: dict) -> dict:
    """Conversational prompt-completion row: TRL trains on the completion only."""
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": row["email"]}],
            "completion": [{"role": "assistant", "content": json.dumps(row["gold"])}]}


def check_system_prompt() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_or_not_baseline.py")
    spec = importlib.util.spec_from_file_location("fit_or_not_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SYSTEM_PROMPT == SYSTEM_PROMPT, "SYSTEM_PROMPT drifted from module 01"
    assert (mod.FIELDS, mod.ISSUE_TYPES, mod.SENTIMENTS) == (FIELDS, ISSUE_TYPES, SENTIMENTS)


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


def smoke(train: str | None = None) -> None:
    if train:
        check_system_prompt()
        rows = load_task_rows(train)
        sft = [to_prompt_completion(r) for r in rows]
        assert len(sft) == len(rows) > 0
        assert all(json.loads(x["completion"][0]["content"]) == r["gold"] for x, r in zip(sft, rows))
        print(f"{train}: {len(rows)}/{len(rows)} rows valid and converted to prompt-completion")
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

    from datasets import Dataset

    domain = Dataset.from_list([to_prompt_completion(r) for r in load_task_rows(args.train)])

    def split_last_turn(ex):                                # Tulu "messages" -> same format
        return {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}

    general = (load_dataset("allenai/tulu-3-sft-mixture", split="train")
               .filter(lambda ex: ex["messages"][-1]["role"] == "assistant")
               .shuffle(seed=0).select(range(int(len(domain) * args.replay_frac)))
               .map(split_last_turn, remove_columns=["id", "messages", "source"]))
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
            max_length=1024,              # prompt-completion data: completion-only loss by default
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
    print(f"lm_eval --model hf --model_args pretrained={args.model},dtype=bfloat16 "
          f"--tasks {tasks} --apply_chat_template --log_samples --output_path {args.out}/eval/base")
    print(f"lm_eval --model hf --model_args pretrained={args.out}/full,dtype=bfloat16 "
          f"--tasks {tasks} --apply_chat_template --log_samples --output_path {args.out}/eval/full")
    for name in ("lora", "lora_replay"):
        print(f"lm_eval --model hf --model_args pretrained={args.model},peft={args.out}/{name},"
              f"dtype=bfloat16 --tasks {tasks} --apply_chat_template --log_samples "
              f"--output_path {args.out}/eval/{name}")
    print("Then compute paired deltas with CIs using eval_toolkit.py from module 09.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy study (default)")
    mode.add_argument("--full", action="store_true", help="T1 GPU study (L19.3)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", default=None,
                    help="module 01 running-example split (ft_data/train.jsonl); "
                         "--full requires it, --smoke validates it if given")
    ap.add_argument("--replay-frac", type=float, default=0.2, help="general rows per domain row")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--out", default="runs/forgetting")
    args = ap.parse_args()
    if args.full:
        if not args.train:
            ap.error("--full needs --train ft_data/train.jsonl (module 01's generator)")
        full(args)
    else:
        smoke(args.train)


if __name__ == "__main__":
    main()
