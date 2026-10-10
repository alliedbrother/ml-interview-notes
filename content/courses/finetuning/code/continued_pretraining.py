"""Continued-pretraining lab for module 05 (L5.3).

Purpose
    --smoke : CPU-only, pure torch. A tiny causal LM is "pretrained" on general
              text, then its vocabulary is extended with new domain tokens and
              it is trained further on a domain/general mix with two optimizer
              parameter groups: the embedding matrix and LM head at a 10x
              smaller learning rate than the rest. The run asserts that the
              optimizer really holds two learning rates (ratio 10), that the
              rows of the newly added tokens move, that domain loss falls, and
              that general loss is still measured (the forgetting check).
    --full  : continued pretraining of Qwen/Qwen3-1.7B-Base on public-domain US
              case law (common-pile/caselaw_access_project_filtered) mixed with
              general web text (HuggingFaceFW/fineweb-edu), LoRA r=128 on all
              linear layers plus fully trained embed_tokens/lm_head at a smaller
              LR, then domain and general perplexity. Follow with a short SFT
              (sft_lab.py) and an MMLU-Pro subset with lm_eval (module 09).

Module      content/courses/finetuning/05-supervised-fine-tuning.md
Hardware    smoke: any CPU, < 60 s.  full: T1 (24 GB, tight) or T2 (80 GB).
Versions    smoke: Python 3.11+, torch 2.x only.
            full (as of October 2026): transformers 5.19, trl 1.15.0,
            peft 0.21.2, datasets, accelerate.

Run (from the repository root)
    python content/courses/finetuning/code/continued_pretraining.py --smoke
    python content/courses/finetuning/code/continued_pretraining.py --full --max-steps 2000
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

GENERAL = [
    "the cat sat on the mat and looked at the door",
    "we walked to the market and bought bread and milk",
    "the weather was cold so she wore a warm coat",
    "he read the book slowly and then went to sleep",
]
# Domain text uses three new "terms" the base tokenizer does not have.
NEW_TOKENS = ["<tort>", "<plaintiff>", "<estoppel>"]
DOMAIN = [
    "the <plaintiff> filed a <tort> claim and the court applied <estoppel>",
    "under <estoppel> the <plaintiff> could not deny the earlier statement",
    "a <tort> requires duty breach causation and damage to the <plaintiff>",
    "the court held that <estoppel> barred the <tort> claim",
]


class Tok:
    def __init__(self) -> None:
        self.itos = ["<pad>"] + list("abcdefghijklmnopqrstuvwxyz ")
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def add_tokens(self, toks: list[str]) -> list[int]:
        new = []
        for t in toks:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)
            new.append(self.stoi[t])
        return new

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for t in NEW_TOKENS:
                if t in self.stoi and text.startswith(t, i):
                    out.append(self.stoi[t])
                    i += len(t)
                    break
            else:
                ch = text[i]
                if ch in "<>":  # before vocab extension these fragment into characters
                    ch = " "
                out.append(self.stoi.get(ch, self.stoi[" "]))
                i += 1
        return out


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 48, layers: int = 2, heads: int = 4, max_len: int = 128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        x = self.embed_tokens(ids) + self.pos(torch.arange(T))
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        return self.lm_head(self.norm(self.layers(x, mask=mask, is_causal=True)))  # (B, T, V)

    def resize_token_embeddings(self, new_vocab: int) -> None:
        """Grow embed_tokens and lm_head; new rows start at the mean of old rows."""
        old_e, old_h = self.embed_tokens.weight.data, self.lm_head.weight.data
        n_old, d = old_e.shape
        e = nn.Embedding(new_vocab, d)
        h = nn.Linear(d, new_vocab, bias=False)
        e.weight.data[:n_old] = old_e
        e.weight.data[n_old:] = old_e.mean(0, keepdim=True)
        h.weight.data[:n_old] = old_h
        h.weight.data[n_old:] = old_h.mean(0, keepdim=True)
        self.embed_tokens, self.lm_head = e, h


def batch_from(texts: list[str], tok: Tok, rng: random.Random, n: int) -> torch.Tensor:
    rows = [tok.encode(rng.choice(texts)) for _ in range(n)]
    T = max(len(r) for r in rows)
    return torch.tensor([r + [0] * (T - len(r)) for r in rows])


def lm_loss(model: TinyLM, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)


def param_groups(model: nn.Module, lr: float, embedding_lr: float):
    emb, rest = [], []
    for name, p in model.named_parameters():
        (emb if name.startswith(("embed_tokens", "lm_head")) else rest).append(p)
    return [{"params": rest, "lr": lr, "name": "body"},
            {"params": emb, "lr": embedding_lr, "name": "embeddings"}]


def run_smoke(seed: int = 0) -> None:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = Tok()
    model = TinyLM(len(tok.itos))

    # Stage 0: "pretraining" on general text only.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss = lm_loss(model, batch_from(GENERAL, tok, rng, 8))
        opt.zero_grad(); loss.backward(); opt.step()

    # Vocabulary extension for the domain terms.
    new_ids = tok.add_tokens(NEW_TOKENS)
    model.resize_token_embeddings(len(tok.itos))
    before_rows = model.embed_tokens.weight.data[new_ids].clone()

    with torch.no_grad():
        dom0 = lm_loss(model, batch_from(DOMAIN, tok, random.Random(1), 32)).item()
        gen0 = lm_loss(model, batch_from(GENERAL, tok, random.Random(1), 32)).item()

    # Stage 1: continued pretraining, 3:1 domain:general mix, embeddings at lr/10.
    lr, embedding_lr = 2e-3, 2e-4
    opt = torch.optim.AdamW(param_groups(model, lr, embedding_lr), weight_decay=0.0)
    lrs = sorted({g["lr"] for g in opt.param_groups})
    assert lrs == [embedding_lr, lr] and math.isclose(lr / embedding_lr, 10.0)
    emb_group = next(g for g in opt.param_groups if g["name"] == "embeddings")
    assert any(p is model.embed_tokens.weight for p in emb_group["params"])
    assert any(p is model.lm_head.weight for p in emb_group["params"])

    for step in range(80):
        texts = DOMAIN if rng.random() < 0.75 else GENERAL
        loss = lm_loss(model, batch_from(texts, tok, rng, 8))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        dom1 = lm_loss(model, batch_from(DOMAIN, tok, random.Random(1), 32)).item()
        gen1 = lm_loss(model, batch_from(GENERAL, tok, random.Random(1), 32)).item()
    moved = (model.embed_tokens.weight.data[new_ids] - before_rows).norm(dim=1)
    print(f"domain ppl {math.exp(dom0):.2f} -> {math.exp(dom1):.2f}; "
          f"general ppl {math.exp(gen0):.2f} -> {math.exp(gen1):.2f}")
    print("new-token row movement:", [round(x, 4) for x in moved.tolist()])
    assert bool((moved > 0).all()), "new token embeddings must be trained"
    assert dom1 < dom0, (dom0, dom1)
    assert math.isfinite(gen1)
    print(f"smoke OK: two LRs (ratio 10), new rows moved, domain loss fell; {time.time() - t0:.1f}s")


def run_full(args) -> None:
    import torch as _torch
    from datasets import interleave_datasets, load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen3-1.7B-Base"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_torch.bfloat16)

    domain = load_dataset("common-pile/caselaw_access_project_filtered", split="train", streaming=True,
                          revision="50e1961a5ed8fdddcee64c9e66e1338de8953b63").select_columns(["text"])
    general = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True,
                           revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9").select_columns(["text"])
    held_dom = list(domain.skip(1_000_000).take(200))
    held_gen = list(general.skip(1_000_000).take(200))
    mix = interleave_datasets([domain, general], probabilities=[0.8, 0.2], seed=0)

    peft_cfg = LoraConfig(r=128, lora_alpha=128, target_modules="all-linear",
                          modules_to_save=["embed_tokens", "lm_head"])
    model = get_peft_model(model, peft_cfg)
    lr, embedding_lr = 1e-4, 1e-5  # embeddings 10x smaller (Unsloth suggests 2-10x)
    groups = [{"params": [], "lr": lr}, {"params": [], "lr": embedding_lr}]
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[1 if ("embed_tokens" in n or "lm_head" in n) else 0]["params"].append(p)
    opt = _torch.optim.AdamW(groups, weight_decay=0.0)

    cfg = SFTConfig(
        output_dir=args.output_dir or "qwen3-1.7b-caselaw-cpt",
        max_steps=args.max_steps,  # streaming dataset: max_steps is required
        max_length=2048,
        packing=True,
        packing_strategy="bfd_split",  # keep every token of long opinions
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=mix, processing_class=tok,
                         optimizers=(opt, None))

    def ppl(rows):
        model.eval()
        tot, n = 0.0, 0
        for r in rows:
            enc = tok(r["text"], return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with _torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            k = enc["input_ids"].shape[1] - 1
            tot += out.loss.item() * k
            n += k
        model.train()
        return math.exp(tot / n)

    before = (ppl(held_dom), ppl(held_gen))
    trainer.train()
    after = (ppl(held_dom), ppl(held_gen))
    print(f"domain ppl {before[0]:.2f} -> {after[0]:.2f}; general ppl {before[1]:.2f} -> {after[1]:.2f}")
    trainer.save_model()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.full and not torch.cuda.is_available():
        raise SystemExit("--full needs a GPU; use --smoke on CPU")
    if args.full:
        run_full(args)
    else:
        run_smoke(args.seed)


if __name__ == "__main__":
    main()
