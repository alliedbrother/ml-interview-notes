"""One recipe, three tools: config emitter, defaults diff and stale-API linter.

Purpose
    Module 08 (Frameworks: TRL, Unsloth, Axolotl, LLaMA-Factory, and Training
    Services). One recipe dict (Qwen3-1.7B, LoRA r=16 on all linear layers,
    LR 2e-4, one epoch on a trl-lib/Capybara slice) is rendered as
      * a TRL CLI YAML           (run with: trl sft --config trl_sft.yaml)
      * an Axolotl YAML          (run with: axolotl train axolotl_sft.yaml)
      * an Unsloth Python script (run with: python unsloth_sft.py)
    plus a table of the *effective* hyperparameters each tool applies, including
    the defaults you inherit when you leave a knob out. It also computes the
    optimizer-step schedule (effective batch, steps, warmup) the way the
    Transformers Trainer does, and ships a stale-API linter for 2024-era TRL
    scripts.

    --smoke : CPU only, stdlib only (no torch needed, no downloads). Writes all
              three artifacts to a temporary directory, asserts each contains the
              recipe's LR, rank and targets, asserts the schedule arithmetic of
              module 08, section 8.2, and runs the linter on a planted fixture.
    --full  : prints the GPU commands for labs L8.1 (T0/T1) and L8.2 (T1 or
              Hugging Face Jobs). No GPU code runs inside this script.
    --out DIR : write the three artifacts to DIR instead of a temp directory.
    --lint FILE : run the stale-API linter on a Python file and exit non-zero if
              it finds anything.

Module      content/courses/finetuning/08-frameworks-and-tooling.md
Hardware    smoke: any CPU, well under 60 s.  full: T0 (free T4, fp16, QLoRA via
            Unsloth) or T1 (one 24 GB GPU, bf16 LoRA) for each tool.
Versions    smoke: Python 3.11+, stdlib only.
            full (as of October 2026): transformers 5.19, trl 1.15.0,
            peft 0.21.2, accelerate 1.15.0, axolotl 0.20.0, unsloth (2026.10
            release line), datasets >= 4.7.

Run (from the repository root)
    python content/courses/finetuning/code/framework_configs.py --smoke
    python content/courses/finetuning/code/framework_configs.py --full --out runs/l8
    python content/courses/finetuning/code/framework_configs.py --lint my_old_script.py
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# The one recipe
# ---------------------------------------------------------------------------

RECIPE = {
    "model": "Qwen/Qwen3-1.7B",
    "dataset": "trl-lib/Capybara",
    "split": "train[:2000]",
    "lr": 2e-4,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.0,
    "targets": "all-linear",
    "epochs": 1,
    "per_device_batch": 4,
    "grad_accum": 4,
    "world_size": 1,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "scheduler": "linear",
    "packing": False,
    "output_dir": "qwen3-1.7b-capybara-lora",
}

LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def fmt_lr(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-")  # 2e-4 -> "2.0e-4"


def emit_trl_yaml(r: dict) -> str:
    """TRL CLI config. Keys are SFTConfig + ModelConfig + ScriptArguments fields."""
    return "\n".join([
        "# trl sft --config trl_sft.yaml   (TRL v1.15, Transformers v5)",
        f"model_name_or_path: {r['model']}",
        "dtype: bfloat16                 # ModelConfig default is float32",
        "datasets:",
        f"  - path: {r['dataset']}",
        f"    split: \"{r['split']}\"",
        "use_peft: true",
        f"lora_r: {r['rank']}",
        f"lora_alpha: {r['alpha']}",
        f"lora_dropout: {r['dropout']}              # ModelConfig default is 0.05",
        f"lora_target_modules: [{r['targets']}]",
        f"learning_rate: {fmt_lr(r['lr'])}",
        f"num_train_epochs: {r['epochs']}",
        f"per_device_train_batch_size: {r['per_device_batch']}",
        f"gradient_accumulation_steps: {r['grad_accum']}",
        f"warmup_steps: {r['warmup_ratio']}            # float < 1 = ratio (v5)",
        f"lr_scheduler_type: {r['scheduler']}",
        f"max_length: {r['max_length']}",
        f"packing: {str(r['packing']).lower()}",
        f"output_dir: {r['output_dir']}",
        "",
    ])


def emit_axolotl_yaml(r: dict) -> str:
    targets = "lora_target_linear: true" if r["targets"] == "all-linear" else (
        "lora_target_modules: [" + r["targets"] + "]")
    return "\n".join([
        "# axolotl train axolotl_sft.yaml   (Axolotl 0.20)",
        f"base_model: {r['model']}",
        "chat_template: qwen3",
        "datasets:",
        f"  - path: {r['dataset']}",
        "    type: chat_template",
        f"    split: \"{r['split']}\"",
        "    roles_to_train: [assistant]     # the default, made explicit",
        "val_set_size: 0.0",
        f"output_dir: ./outputs/{r['output_dir']}",
        f"sequence_len: {r['max_length']}",
        f"sample_packing: {str(r['packing']).lower()}",
        "adapter: lora",
        f"lora_r: {r['rank']}",
        f"lora_alpha: {r['alpha']}",
        f"lora_dropout: {r['dropout']}",
        targets,
        f"micro_batch_size: {r['per_device_batch']}",
        f"gradient_accumulation_steps: {r['grad_accum']}",
        f"num_epochs: {r['epochs']}",
        f"learning_rate: {fmt_lr(r['lr'])}",
        f"lr_scheduler: {r['scheduler']}          # Axolotl default is cosine",
        f"warmup_ratio: {r['warmup_ratio']}",
        "bf16: auto",
        "gradient_checkpointing: true      # Axolotl default is false",
        "",
    ])


def emit_unsloth_py(r: dict) -> str:
    targets = LINEAR_TARGETS if r["targets"] == "all-linear" else [r["targets"]]
    return "\n".join([
        '"""Unsloth SFT script generated by framework_configs.py (needs a CUDA GPU)."""',
        "from unsloth import FastLanguageModel  # import unsloth before trl/transformers",
        "from datasets import load_dataset",
        "from trl import SFTConfig, SFTTrainer",
        "",
        "model, tokenizer = FastLanguageModel.from_pretrained(",
        f"    model_name={r['model']!r},",
        f"    max_seq_length={r['max_length']},",
        "    load_in_4bit=False,   # True for QLoRA on a T4 (T0)",
        "    dtype=None,           # auto: bf16 on Ampere+, fp16 on T4",
        ")",
        "model = FastLanguageModel.get_peft_model(",
        "    model,",
        f"    r={r['rank']},",
        f"    lora_alpha={r['alpha']},",
        f"    lora_dropout={r['dropout']},",
        f"    target_modules={targets!r},",
        '    use_gradient_checkpointing="unsloth",',
        "    random_state=3407,",
        ")",
        f"ds = load_dataset({r['dataset']!r}, split={r['split']!r})",
        "trainer = SFTTrainer(",
        "    model=model,",
        "    processing_class=tokenizer,",
        "    train_dataset=ds,",
        "    args=SFTConfig(",
        f"        learning_rate={r['lr']},",
        f"        num_train_epochs={r['epochs']},",
        f"        per_device_train_batch_size={r['per_device_batch']},",
        f"        gradient_accumulation_steps={r['grad_accum']},",
        f"        warmup_steps={r['warmup_ratio']},",
        f"        lr_scheduler_type={r['scheduler']!r},",
        f"        max_length={r['max_length']},",
        f"        packing={r['packing']},",
        f"        output_dir={r['output_dir']!r},",
        "    ),",
        ")",
        "trainer.train()",
        f"model.save_pretrained({r['output_dir'] + '-adapter'!r})",
        "",
    ])


# Effective values: what each tool does for the knobs a recipe often forgets.
# Each cell is (value with this recipe's emitted config, default if you omit the key).
DEFAULTS = [
    ("learning rate", ("2e-4", "2e-5 (SFTConfig)"), ("2e-4", "required"), ("2e-4", "2e-5 (SFTConfig)")),
    ("LoRA dropout", ("0.0", "0.05 (ModelConfig)"), ("0.0", "0.0"), ("0.0", "0.0 (get_peft_model)")),
    ("LoRA targets", ("all-linear", "None -> PEFT per-arch default"), ("all linear", "none"), ("7 proj layers", "7 proj layers")),
    ("scheduler", ("linear", "linear"), ("linear", "cosine"), ("linear", "linear")),
    ("warmup", ("ceil(0.03 x steps)", "0"), ("int(0.03 x steps)", "min(int(0.03 x steps), 100)"), ("ceil(0.03 x steps)", "0")),
    ("loss on", ("all tokens", "all tokens (messages data)"), ("assistant turns", "assistant turns"), ("all tokens", "all tokens")),
    ("packing", ("off", "off"), ("off", "off (recommended on)"), ("off", "off")),
    ("grad checkpointing", ("on", "on (SFTConfig)"), ("on", "off"), ("unsloth (offload)", "unsloth (offload)")),
    ("load dtype", ("bfloat16", "float32 (ModelConfig)"), ("auto", "from bf16/fp16 flags"), ("auto", "auto")),
]


def defaults_table() -> str:
    head = "| knob | TRL CLI (this / omitted) | Axolotl (this / omitted) | Unsloth (this / omitted) |"
    sep = "|---|---|---|---|"
    rows = [f"| {k} | {t[0]} / {t[1]} | {a[0]} / {a[1]} | {u[0]} / {u[1]} |" for k, t, a, u in DEFAULTS]
    return "\n".join([head, sep, *rows])


# ---------------------------------------------------------------------------
# Schedule arithmetic (module 08, section 8.2)
# ---------------------------------------------------------------------------

def schedule(n_examples: int, per_device: int, grad_accum: int, world: int, epochs: float,
             warmup_ratio: float, max_steps: int = -1, count_remainder: bool = True,
             warmup_rounding: str = "ceil") -> dict:
    """Mirror of Transformers v5 Trainer.set_initial_training_values + get_warmup_steps.

    count_remainder=False reproduces the older floor (len_dataloader // grad_accum).
    warmup_rounding="floor" reproduces Axolotl's int(warmup_ratio * total_steps).
    """
    effective = per_device * grad_accum * world
    per_rank_examples = math.ceil(n_examples / world)             # DistributedSampler pads
    micro_batches = math.ceil(per_rank_examples / per_device)     # len(dataloader) per rank
    if count_remainder:
        steps_per_epoch = max(micro_batches // grad_accum + int(micro_batches % grad_accum > 0), 1)
    else:
        steps_per_epoch = max(micro_batches // grad_accum, 1)
    total = max_steps if max_steps > 0 else math.ceil(epochs * steps_per_epoch)
    raw = warmup_ratio * total
    warmup = math.ceil(raw) if warmup_rounding == "ceil" else int(raw)
    return {"effective_batch": effective, "micro_batches_per_rank": micro_batches,
            "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup": warmup}


# ---------------------------------------------------------------------------
# Stale-API linter (module 08, section 8.2)
# ---------------------------------------------------------------------------

STALE_KWARGS = {
    "max_seq_length": "max_length",
    "evaluation_strategy": "eval_strategy",
    "torch_dtype": "dtype",
}
STALE_NAMES = {
    "DataCollatorForCompletionOnlyLM": "completion_only_loss / assistant_only_loss",
    "PPOTrainer": "GRPOTrainer or RLOOTrainer (PPOTrainer removed in TRL v1.13)",
}


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def lint(source: str) -> list[tuple[str, str]]:
    """Return sorted unique (stale pattern, replacement) pairs found in source."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            for kw in node.keywords:
                if kw.arg == "tokenizer" and name.endswith("Trainer"):
                    found.add(("tokenizer=", "processing_class="))
                elif kw.arg == "max_seq_length" and not name.endswith(("Config", "Trainer")):
                    continue  # Unsloth's from_pretrained(max_seq_length=...) is current API
                elif kw.arg in STALE_KWARGS:
                    found.add((kw.arg + "=", STALE_KWARGS[kw.arg] + "="))
                elif (kw.arg == "use_liger_kernel" and isinstance(kw.value, ast.Constant)
                      and kw.value.value is True):
                    found.add(("use_liger_kernel=True", 'model_init_kwargs={"use_kernels": True}'))
                elif (kw.arg == "truncation_mode" and isinstance(kw.value, ast.Constant)
                      and kw.value.value == "keep_end"):
                    found.add(('truncation_mode="keep_end"', 'truncation_mode="keep_start"'))
        elif isinstance(node, ast.Name) and node.id in STALE_NAMES:
            found.add((node.id, STALE_NAMES[node.id]))
        elif isinstance(node, ast.alias) and node.name in STALE_NAMES:
            found.add((node.name, STALE_NAMES[node.name]))
    return sorted(found)


FIXTURE_2024 = "\n".join([
    "from trl import SFTTrainer, SFTConfig, DPOConfig, PPOTrainer, DataCollatorForCompletionOnlyLM",
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype="bfloat16")',
    'tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")',
    'collator = DataCollatorForCompletionOnlyLM("### Answer:", tokenizer=tok)',
    'args = SFTConfig(output_dir="out", max_seq_length=2048, evaluation_strategy="steps",',
    '                 truncation_mode="keep_end")',
    'dpo_args = DPOConfig(output_dir="dpo", use_liger_kernel=True)',
    "trainer = SFTTrainer(model=model, args=args, train_dataset=ds, tokenizer=tok,",
    "                     data_collator=collator)",
    "trainer.train()",
    "rlhf = PPOTrainer",
])

FIXTURE_2026 = "\n".join([
    "from trl import SFTTrainer, SFTConfig, DPOConfig",
    "from transformers import AutoTokenizer",
    'tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")',
    "# old code used tokenizer= and PPOTrainer; comments must not trigger the linter",
    'args = SFTConfig(output_dir="out", max_length=2048, eval_strategy="steps",',
    '                 truncation_mode="keep_start", assistant_only_loss=True,',
    '                 model_init_kwargs={"dtype": "bfloat16"})',
    'dpo_args = DPOConfig(output_dir="dpo", model_init_kwargs={"use_kernels": True})',
    'trainer = SFTTrainer(model="Qwen/Qwen3-0.6B", args=args, train_dataset=ds,',
    "                     processing_class=tok)",
    "trainer.train()",
])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def write_artifacts(out_dir: str, r: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "trl": os.path.join(out_dir, "trl_sft.yaml"),
        "axolotl": os.path.join(out_dir, "axolotl_sft.yaml"),
        "unsloth": os.path.join(out_dir, "unsloth_sft.py"),
        "defaults": os.path.join(out_dir, "effective_defaults.md"),
    }
    texts = {"trl": emit_trl_yaml(r), "axolotl": emit_axolotl_yaml(r),
             "unsloth": emit_unsloth_py(r), "defaults": defaults_table() + "\n"}
    for k, p in paths.items():
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(texts[k])
    return paths


def flat_yaml(text: str) -> dict:
    """Tiny parser for the flat 'key: value' lines we emit (no PyYAML needed)."""
    out = {}
    for line in text.splitlines():
        s = line.split("#", 1)[0].rstrip()
        if not s or s.startswith(" ") or s.startswith("-") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def smoke(out_dir: str | None) -> None:
    r = RECIPE
    ctx = tempfile.TemporaryDirectory() if out_dir is None else None
    target = ctx.name if ctx else out_dir
    try:
        paths = write_artifacts(target, r)
        trl_cfg = flat_yaml(open(paths["trl"], encoding="utf-8").read())
        axo_cfg = flat_yaml(open(paths["axolotl"], encoding="utf-8").read())
        uns_src = open(paths["unsloth"], encoding="utf-8").read()

        assert float(trl_cfg["learning_rate"]) == r["lr"]
        assert int(trl_cfg["lora_r"]) == r["rank"]
        assert trl_cfg["lora_target_modules"] == "[all-linear]"
        assert trl_cfg["dtype"] == "bfloat16"
        assert float(axo_cfg["learning_rate"]) == r["lr"]
        assert int(axo_cfg["lora_r"]) == r["rank"]
        assert axo_cfg["lora_target_linear"] == "true"
        assert axo_cfg["adapter"] == "lora"
        ast.parse(uns_src)  # the generated script is valid Python
        assert f"learning_rate={r['lr']}" in uns_src
        assert f"r={r['rank']}," in uns_src
        assert all(t in uns_src for t in LINEAR_TARGETS)
        assert "processing_class=tokenizer" in uns_src
        assert lint(uns_src) == []  # our own output passes our own linter

        # Section 8.2 schedule arithmetic: 10,000 examples, batch 4, accumulation 8, 2 GPUs, 3 epochs.
        s = schedule(10_000, 4, 8, 2, 3, 0.03)
        assert (s["effective_batch"], s["micro_batches_per_rank"]) == (64, 1250)
        assert (s["steps_per_epoch"], s["total_steps"], s["warmup"]) == (157, 471, 15)
        assert schedule(10_000, 4, 8, 2, 3, 0.03, count_remainder=False)["steps_per_epoch"] == 156
        assert schedule(10_000, 4, 8, 2, 3, 0.03, warmup_rounding="floor")["warmup"] == 14
        s200 = schedule(10_000, 4, 8, 2, 3, 0.03, max_steps=200)
        assert (s200["total_steps"], s200["warmup"]) == (200, 6)
        # The lab recipe itself: 2,000 examples, 4 x 4 x 1, one epoch.
        lab = schedule(2_000, r["per_device_batch"], r["grad_accum"], r["world_size"],
                       r["epochs"], r["warmup_ratio"])
        assert (lab["effective_batch"], lab["total_steps"], lab["warmup"]) == (16, 125, 4)

        # 8.2 linter: the linter finds exactly the eight planted issues and nothing in clean code.
        issues = lint(FIXTURE_2024)
        assert len(issues) == 8, issues
        assert lint(FIXTURE_2026) == []

        print(defaults_table())
        print(f"\nartifacts: {', '.join(os.path.basename(p) for p in paths.values())}")
        print(f"lab schedule: {lab}")
        print(f"linter: {len(issues)} stale patterns in the 2024 fixture, 0 in the 2026 fixture")
        print("SMOKE OK")
    finally:
        if ctx:
            ctx.cleanup()


FULL_COMMANDS = """\
L8.1 (T1: one 24 GB GPU, bf16; on a T4 set load_in_4bit=True for Unsloth and fp16 everywhere)
  pip install "trl==1.15.0" "peft>=0.18" "transformers>=5" datasets trackio
  trl sft --config {out}/trl_sft.yaml
  pip install axolotl            # or: docker run --gpus all -it axolotlai/axolotl:main-latest
  axolotl train {out}/axolotl_sft.yaml
  pip install unsloth            # or the unsloth/unsloth Docker image
  python {out}/unsloth_sft.py
  Record for each run: tokens/s, torch.cuda.max_memory_allocated(), final train loss,
  eval loss on a held-out Capybara slice, and diff the resolved configs against
  {out}/effective_defaults.md.

L8.2 (T1 or Hugging Face Jobs)
  trl sft --config {out}/trl_sft.yaml --push_to_hub true --hub_model_id <you>/qwen3-1.7b-capybara-lora
  # or on HF Jobs (pay per second; default timeout 30 min):
  hf jobs run --flavor a10g-large --timeout 2h --secrets HF_TOKEN huggingface/trl -- \\
      trl sft --model_name_or_path Qwen/Qwen3-1.7B --dtype bfloat16 \\
      --dataset_name trl-lib/Capybara --use_peft --lora_r 16 --lora_alpha 32 \\
      --lora_target_modules all-linear --learning_rate 2e-4 --num_train_epochs 1 \\
      --output_dir qwen3-1.7b-capybara-lora --push_to_hub
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU self-test (default)")
    mode.add_argument("--full", action="store_true", help="write artifacts and print GPU commands")
    mode.add_argument("--lint", metavar="FILE", help="lint a Python file for stale TRL/Transformers APIs")
    p.add_argument("--out", default=None, help="directory for the generated artifacts")
    args = p.parse_args()

    if args.lint:
        with open(args.lint, encoding="utf-8") as fh:
            issues = lint(fh.read())
        for pat, rep in issues:
            print(f"{args.lint}: {pat}  ->  {rep}")
        sys.exit(1 if issues else 0)
    if args.full:
        out = args.out or "runs/l8"
        write_artifacts(out, RECIPE)
        print(FULL_COMMANDS.format(out=out))
        return
    smoke(args.out)


if __name__ == "__main__":
    main()
