"""SFT lab for module 05 (Supervised Fine-Tuning and Continued Pretraining).

Purpose
    --smoke : a CPU-only, pure-torch rehearsal of what SFTTrainer does for you.
              A 2-layer toy causal LM is trained for 30 steps on synthetic
              support-email -> JSON strings with completion-only loss masks and
              best-fit-decreasing (BFD) packing with position-id resets. The run
              asserts that (1) the loss falls, (2) masked (prompt) positions get
              exactly zero gradient at the logits, and (3) a packed row produces
              the same logits as running each example alone.
    --full  : the real labs with TRL's SFTTrainer, selected with --task:
                base2instruct  L5.1  Qwen/Qwen3-0.6B-Base -> chat model on
                               trl-lib/Capybara, template cloned from SmolLM3,
                               QLoRA, fp16 (T0: free T4 16 GB)
                task           L5.2  task SFT on the running JSON-extraction task
                               with Qwen/Qwen3-4B-Instruct-2507 + LoRA, with
                               --full-sequence-loss and --packing ablations
                               (T1: one 24 GB GPU)
                reasoning      L5.4  <think>-formatted SFT on
                               FreedomIntelligence/medical-o1-reasoning-SFT
                               (T0)

Module      content/courses/finetuning/05-supervised-fine-tuning.md
Hardware    smoke: any CPU, < 60 s.  full: T0 or T1 as listed above.
Versions    smoke: Python 3.11+, torch 2.x (no other dependency).
            full (as of October 2026): transformers 5.19, trl 1.15.0,
            peft 0.21.2, datasets, bitsandbytes (QLoRA), accelerate.

Run (from the repository root)
    python content/courses/finetuning/code/sft_lab.py --smoke
    python content/courses/finetuning/code/sft_lab.py --full --task base2instruct
    python content/courses/finetuning/code/sft_lab.py --full --task task --train-file ft_data/train.jsonl --eval-file ft_data/eval.jsonl
    python content/courses/finetuning/code/sft_lab.py --full --task task --full-sequence-loss --packing
    python content/courses/finetuning/code/sft_lab.py --full --task reasoning
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Synthetic running-task data (support email -> JSON)
# --------------------------------------------------------------------------

NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana"]
ISSUES = {
    "refund": "I want my money back for",
    "shipping": "Still waiting on delivery of",
    "defect": "The item arrived broken in",
    "account": "I cannot log in to see",
    "other": "Quick question about",
}
SENTIMENT = {"negative": "This is awful.", "neutral": "Thanks.", "positive": "Love your shop!"}
FILLER = [
    "I ordered it last week.",
    "My neighbour got the same thing.",
    "Please reply soon, I am travelling on Friday and need this sorted before then.",
    "I have been a customer for years and never had a problem before this one.",
]


def make_example(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(NAMES)
    order = str(rng.randint(1000, 9999))
    issue = rng.choice(list(ISSUES))
    sentiment = rng.choice(list(SENTIMENT))
    refund = issue == "refund" or rng.random() < 0.2
    extra = " ".join(rng.sample(FILLER, rng.randint(0, len(FILLER))))  # variable lengths
    email = f"Hi, {name} here. {ISSUES[issue]} order {order}. {extra} {SENTIMENT[sentiment]}"
    target = json.dumps(
        {
            "customer_name": name,
            "order_id": order,
            "issue_type": issue,
            "refund_requested": refund,
            "sentiment": sentiment,
        },
        separators=(",", ":"),
    )
    return email, target


class CharTokenizer:
    """Character tokenizer with three special tokens, enough for the smoke run."""

    specials = ["<pad>", "<user>", "<assistant>", "<eos>"]

    def __init__(self) -> None:
        chars = [chr(i) for i in range(32, 127)]
        self.itos = self.specials + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def sid(self, name: str) -> int:
        return self.stoi[name]


def tokenize_prompt_completion(tok: CharTokenizer, email: str, target: str):
    """Return input_ids and already-shifted labels for one example.

    labels[i] is the target for position i (the next token) or -100. Only
    completion tokens (the JSON and <eos>) are targets: completion-only loss.
    The last position of each example has no in-document successor, so it is
    -100; this is what keeps packed rows from training across boundaries.
    """
    prompt = [tok.sid("<user>")] + tok.encode(email) + [tok.sid("<assistant>")]
    completion = tok.encode(target) + [tok.sid("<eos>")]
    ids = prompt + completion
    labels = [-100] * len(ids)
    for i in range(len(ids) - 1):
        if i + 1 >= len(prompt):  # next token belongs to the completion
            labels[i] = ids[i + 1]
    return ids, labels


def bfd_pack(lengths: list[int], capacity: int) -> list[list[int]]:
    """Best-fit decreasing over example indices; returns bins of indices."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        n = lengths[i]
        if n > capacity:
            raise ValueError("bfd would truncate this example; bfd_split would split it")
        fits = [b for b, r in enumerate(room) if r >= n]
        if fits:
            b = min(fits, key=lambda j: room[j])
            bins[b].append(i)
            room[b] -= n
        else:
            bins.append([i])
            room.append(capacity - n)
    return bins


def build_packed_rows(examples, bins, capacity, pad_id):
    """Concatenate each bin into one row; position ids restart at 0 per example."""
    ids_rows, lab_rows, pos_rows = [], [], []
    for b in bins:
        ids, labs, pos = [], [], []
        for i in b:
            x, y = examples[i]
            ids += x
            labs += y
            pos += list(range(len(x)))
        pad = capacity - len(ids)
        # Smoke pads rows to a rectangle for simplicity; padding-free kernels skip this.
        ids += [pad_id] * pad
        labs += [-100] * pad
        pos += [0] * pad
        ids_rows.append(ids)
        lab_rows.append(labs)
        pos_rows.append(pos)
    return (
        torch.tensor(ids_rows),
        torch.tensor(lab_rows),
        torch.tensor(pos_rows),
    )


def block_causal_mask(pos: torch.Tensor) -> torch.Tensor:
    """(B, T) position ids -> (B, T, T) boolean mask, True = may attend.

    A new document starts wherever the position id resets to 0. Tokens attend
    causally and only within their own document, which is what varlen
    FlashAttention achieves with cu_seqlens.
    """
    doc = torch.cumsum((pos == 0).long(), dim=1)
    T = pos.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=pos.device).tril()
    same = doc[:, :, None] == doc[:, None, :]
    return causal[None] & same


# --------------------------------------------------------------------------
# Toy causal LM
# --------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])  # (B, H, T, Dh)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 64, layers: int = 2, heads: int = 4, max_pos: int = 512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)  # indexed by position ids, so resets matter
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids) + self.pos(pos)
        allowed = block_causal_mask(pos)
        for blk in self.blocks:
            x = blk(x, allowed)
        return self.head(self.ln(x))  # (B, T, V)


def run_smoke(seed: int = 0, steps: int = 30) -> None:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = CharTokenizer()
    capacity = 512

    examples = [tokenize_prompt_completion(tok, *make_example(rng)) for _ in range(96)]
    lengths = [len(x) for x, _ in examples]
    bins = bfd_pack(lengths, capacity)
    ids, labels, pos = build_packed_rows(examples, bins, capacity, tok.sid("<pad>"))

    real = sum(lengths)
    pad_longest = len(lengths) * max(lengths) - real
    pad_packed = len(bins) * capacity - real
    trained = int((labels != -100).sum())
    print(f"examples={len(examples)} tokens={real} rows={len(bins)} "
          f"pad(pad-to-longest)={pad_longest} pad(packed rows)={pad_packed} trained_targets={trained}")
    assert pad_packed < pad_longest
    assert sorted(i for b in bins for i in b) == list(range(len(examples)))

    model = TinyLM(tok.vocab_size, max_pos=capacity)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    # (3) Packing isolation: a packed row equals each example run alone.
    model.eval()
    with torch.no_grad():
        row = model(ids[:1], pos[:1])[0]
        start = 0
        for i in bins[0]:
            x = torch.tensor([examples[i][0]])
            alone = model(x, torch.arange(x.shape[1])[None])[0]
            torch.testing.assert_close(row[start:start + x.shape[1]], alone, atol=1e-5, rtol=1e-5)
            start += x.shape[1]
    model.train()

    losses = []
    batch = 4
    for step in range(steps):
        idx = torch.tensor([(step * batch + j) % ids.shape[0] for j in range(batch)])
        logits = model(ids[idx], pos[idx])
        logits.retain_grad()
        lab = labels[idx]
        # Labels are already shifted per example, so no further shift here.
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        if step == 0:
            # (2) Masked positions contribute exactly zero gradient.
            g = logits.grad
            masked = lab == -100
            assert g[masked].abs().max().item() == 0.0
            assert g[~masked].abs().sum().item() > 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:2d} loss {loss.item():.3f}")

    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    # (1) The loss falls: the JSON skeleton is learnable within 30 steps.
    assert last < 0.6 * first, (first, last)
    assert math.isfinite(last)
    print(f"mean loss first3={first:.3f} last3={last:.3f}; elapsed {time.time() - t0:.1f}s")
    print("smoke OK: loss decreased, masked tokens got zero gradient, packed == per-example")


# --------------------------------------------------------------------------
# Full mode (GPU). Library imports live inside these functions.
# --------------------------------------------------------------------------

FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
ENUMS = {
    "issue_type": {"refund", "shipping", "defect", "account", "other"},
    "sentiment": {"positive", "neutral", "negative"},
}
SYSTEM = (
    "Extract customer_name, order_id, issue_type (refund|shipping|defect|account|other), "
    "refund_requested (bool) and sentiment (positive|neutral|negative) from the email. "
    "Reply with JSON only."
)


def score(predictions: list[str], references: list[dict]) -> dict:
    """The four module-01 metrics: parse rate, schema validity, per-field EM, all-fields EM."""
    parsed = valid = all_em = 0
    field_hits = {f: 0 for f in FIELDS}
    for pred, ref in zip(predictions, references):
        try:
            obj = json.loads(pred.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        parsed += 1
        ok = (
            isinstance(obj, dict)
            and set(obj) == set(FIELDS)
            and isinstance(obj["refund_requested"], bool)
            and all(obj[k] in v for k, v in ENUMS.items())
        )
        valid += int(ok)
        hits = [isinstance(obj, dict) and obj.get(f) == ref[f] for f in FIELDS]
        for f, h in zip(FIELDS, hits):
            field_hits[f] += int(h)
        all_em += int(all(hits))
    n = max(len(references), 1)
    return {
        "parse_rate": parsed / n,
        "schema_valid": valid / n,
        "per_field_em": {f: h / n for f, h in field_hits.items()},
        "all_fields_em": all_em / n,
    }


def load_task_rows(path: str) -> list[dict]:
    """Accept module-01 rows: either {prompt, completion} or {email, <fields>}."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "email" in r:
                target = {f: r[f] for f in FIELDS} if all(f in r for f in FIELDS) else r["target"]
                rows.append({"email": r["email"], "target": target})
            else:
                rows.append({"email": r["prompt"], "target": json.loads(r["completion"])})
    return rows


def full_base2instruct(args) -> None:
    import torch as _torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("trl-lib/Capybara", split="train", revision="e235e846458bff3398a88aed812347f7f0756520")
    cfg = SFTConfig(
        output_dir=args.output_dir or "qwen3-0.6b-base-capybara",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",  # clones template + special tokens, sets EOS
        max_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # LoRA-scale LR
        num_train_epochs=1,
        fp16=True,  # T4 has no bf16
        bf16=False,
        packing=True,
        model_init_kwargs={"dtype": _torch.float16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model="Qwen/Qwen3-0.6B-Base",
        args=cfg,
        train_dataset=ds,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_torch.float16
        ),
        # The ChatML tokens SmolLM3's template uses already exist in Qwen3's vocabulary.
        # If a cloned template adds genuinely new tokens, their rows must train too:
        # see module 06 (trainable_token_indices, ensure_weight_tying for tied embeddings).
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model()
    print("saved to", cfg.output_dir, "- now score it with module 01's eval set (--task task --eval-only)")


def generate_and_score(model, tokenizer, rows, enable_thinking=False) -> dict:
    import torch as _torch

    preds = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["email"]}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=enable_thinking,
        ).to(model.device)
        with _torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False)
        preds.append(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return score(preds, [r["target"] for r in rows])


def full_task(args) -> None:
    import torch as _torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    train = load_task_rows(args.train_file)
    evalrows = load_task_rows(args.eval_file)

    def to_pc(r):  # conversational prompt-completion: completion-only loss by default
        return {
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["email"]}],
            "completion": [{"role": "assistant",
                            "content": json.dumps(r["target"], separators=(",", ":"))}],
        }

    ds = Dataset.from_list([to_pc(r) for r in train])
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = SFTConfig(
        output_dir=args.output_dir or "qwen3-4b-json-sft",
        completion_only_loss=not args.full_sequence_loss,
        packing=args.packing,
        max_length=1024,
        learning_rate=1e-4,  # instruct model + LoRA: stay at or below 1e-4
        num_train_epochs=2,
        per_device_train_batch_size=8,
        model_init_kwargs={"dtype": _torch.bfloat16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model_id, args=cfg, train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    )
    trainer.train()
    trainer.model.eval()
    metrics = generate_and_score(trainer.model, tok, evalrows)
    tag = f"completion_only={not args.full_sequence_loss} packing={args.packing}"
    print(tag, json.dumps(metrics, indent=2))
    with open(f"{cfg.output_dir}/task_scores.json", "w", encoding="utf-8") as fh:
        json.dump({"config": tag, **metrics}, fh, indent=2)


def full_reasoning(args) -> None:
    import torch as _torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train",
                      revision="fc2c9e8a37b38f38da6d449564a8c350b244aef4")

    def fmt(ex):
        return {
            "prompt": [{"role": "user", "content": ex["Question"]}],
            "completion": [{"role": "assistant",
                            "content": f"<think>\n{ex['Complex_CoT']}\n</think>\n\n{ex['Response']}"}],
        }

    ds = ds.map(fmt, remove_columns=ds.column_names).select(range(min(5000, len(ds))))
    cfg = SFTConfig(
        output_dir=args.output_dir or "qwen3-0.6b-medical-think",
        max_length=2048,  # traces are long: check truncation rate first (module 04)
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=True,
        bf16=False,
        model_init_kwargs={"dtype": _torch.float16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model="Qwen/Qwen3-0.6B", args=cfg, train_dataset=ds,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=_torch.float16),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy run (default without a GPU)")
    mode.add_argument("--full", action="store_true", help="real TRL run on a GPU")
    p.add_argument("--task", choices=["base2instruct", "task", "reasoning"], default="task")
    p.add_argument("--train-file", default="ft_data/train.jsonl")
    p.add_argument("--eval-file", default="ft_data/eval.jsonl")
    p.add_argument("--full-sequence-loss", action="store_true", help="ablation: train on prompt too")
    p.add_argument("--packing", action="store_true", help="ablation: BFD packing on")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.full and not torch.cuda.is_available():
        raise SystemExit("--full needs a GPU; use --smoke on CPU")
    if not args.full:
        run_smoke(args.seed)
        return
    {"base2instruct": full_base2instruct, "task": full_task, "reasoning": full_reasoning}[args.task](args)


if __name__ == "__main__":
    main()
