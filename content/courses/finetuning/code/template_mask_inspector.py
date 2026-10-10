"""Read a masked, templated batch: chat templates, BOS, EOS, labels and loss.

Module: 02 -- Tokens, Chat Templates, and the Causal-LM Objective (LLM Fine-Tuning Deep Dive).

Purpose
    Render one support-email conversation through chat templates, tokenize it,
    print a per-token table (index, token, label id or -100, in loss?), reproduce
    the double-BOS and pad=eos bugs, fix them, and check a hand-computed
    causal-LM loss against the model's own loss.

Modes
    --smoke  (default when no GPU is found) CPU only, stdlib + torch, no downloads.
             A hand-written ChatML renderer and a toy word-level vocabulary stand in
             for real tokenizers; a tiny random GRU LM stands in for the model. Also
             trains two tiny LMs (labels masked by id vs by position) and asserts that
             only the position-masked one learns to emit EOS. Runs in a few seconds.
    --full   Downloads the tokenizers of Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B-Instruct
             (gated) and google/gemma-3-1b-it (gated) at pinned revisions, plus the
             Qwen/Qwen3-0.6B weights for the loss comparison (works on CPU; a T4 is faster).
             Run `hf auth login` first and accept the Llama/Gemma licenses on the Hub.
    --full --never-stop
             Optional T0 extension (free Colab/Kaggle T4, 16 GB): trains
             Qwen/Qwen3-0.6B-Base for 100 steps with pad=<|im_end|> and id-based
             masking, then with a distinct pad, and reports stop rates.

Hardware tier for full mode: CPU (tokenizers + 0.6B loss check), T0 for --never-stop.
Pinned versions for full mode: Python 3.11+, torch 2.8+, transformers 5.x (tested on 5.5).

Run (from the repository root):
    python content/courses/finetuning/code/template_mask_inspector.py --smoke
    python content/courses/finetuning/code/template_mask_inspector.py --full
    python content/courses/finetuning/code/template_mask_inspector.py --full --never-stop
"""

from __future__ import annotations

import argparse
import re

import torch
import torch.nn.functional as F

CONVO = [
    {"role": "system", "content": "Extract the ticket as JSON."},
    {"role": "user", "content": "Hi, order ORD-01142 arrived broken. Ana"},
    {"role": "assistant", "content": '{"order_id": "ORD-01142", "issue_type": "defect"}'},
    {"role": "user", "content": "Add whether I want a refund: yes."},
    {"role": "assistant", "content": '{"refund_requested": true}'},
]

# Revisions read for Table 2 of module 02.
MODELS = {
    "qwen3": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
              "<|im_start|>assistant\n", "<|im_end|>"),
    "llama3.2": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6",
                 "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
    "gemma3": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
               "<start_of_turn>model\n", "<end_of_turn>"),
}


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def print_table(tokens, labels, title, limit=None):
    print(f"\n{title}")
    print(f"{'idx':>4}  {'token':<34} {'label':>8}  in-loss(shifted)?")
    rows = range(len(tokens)) if limit is None else range(min(limit, len(tokens)))
    for i in rows:
        # The label stored at i is scored at position i-1 (HF shifts internally).
        shown = repr(tokens[i])[:34]
        print(f"{i:>4}  {shown:<34} {labels[i]:>8}  {'yes' if labels[i] != -100 else ''}")


def hf_style_loss(logits, labels):
    """What HF's ForCausalLMLoss does: pad labels with one -100, shift, mean over kept."""
    V = logits.shape[-1]
    tgt = F.pad(labels, (0, 1), value=-100)[:, 1:]
    return F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1), ignore_index=-100)


def hand_loss(logits, labels):
    """log_softmax + gather over shifted, unmasked targets, divided by the count."""
    tgt = labels[:, 1:]
    keep = tgt.ne(-100)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)          # (B, T-1, V)
    nll = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return nll[keep].sum() / keep.sum()


# ----------------------------------------------------------------------------
# Smoke mode: toy renderer, toy vocabulary, tiny models
# ----------------------------------------------------------------------------

SPECIAL = ("<bos>", "<|im_start|>", "<|im_end|>", "<pad>")
SPLIT = re.compile(r"(<bos>|<\|im_start\|>|<\|im_end\|>|<pad>)")


def toy_render(messages, add_generation_prompt=False, bos=False):
    """ChatML renderer returning (text, trainable) pieces, like a {% generation %} template."""
    pieces = [("<bos>", False)] if bos else []
    for m in messages:
        if m["role"] == "assistant":
            pieces.append(("<|im_start|>assistant\n", False))
            pieces.append((m["content"] + "<|im_end|>", True))
            pieces.append(("\n", False))
        else:
            pieces.append(("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n", False))
    if add_generation_prompt:
        pieces.append(("<|im_start|>assistant\n", False))
    return pieces


def toy_tokenize(text, add_special_tokens=False):
    out = ["<bos>"] if add_special_tokens else []
    for part in SPLIT.split(text):
        out.extend([part] if part in SPECIAL else part.split())
    return out


class TinyLM(torch.nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.emb = torch.nn.Embedding(V, d)
        self.rnn = torch.nn.GRU(d, d, batch_first=True)
        self.head = torch.nn.Linear(d, V)

    def forward(self, ids):                                      # (B, T) -> (B, T, V)
        h, _ = self.rnn(self.emb(ids))
        return self.head(h)


def smoke_mask_and_bugs():
    pieces = toy_render(CONVO)
    text = "".join(p for p, _ in pieces)
    tokens = toy_tokenize(text)
    mask = [flag for p, flag in pieces for _ in toy_tokenize(p)]
    assert [t for p, _ in pieces for t in toy_tokenize(p)] == tokens
    vocab = {t: i for i, t in enumerate(sorted(set(tokens) | set(SPECIAL)))}
    ids = [vocab[t] for t in tokens]
    labels = [i if m else -100 for i, m in zip(ids, mask)]
    print_table(tokens, labels, "Step 3 (toy ChatML): per-token labels")
    trained = sum(mask)
    ends = [k for k, t in enumerate(tokens) if t == "<|im_end|>"]
    assert trained == 8, trained                                # (4 + 1) + (2 + 1) with this fixture
    assert sum(labels[k] != -100 for k in ends) == 2
    print(f"trained tokens: {trained} of {len(tokens)}")

    # Step 2: double BOS with a Llama/Gemma-style template that emits <bos> itself
    templated = "".join(p for p, _ in toy_render(CONVO, bos=True))
    n_default = toy_tokenize(templated, add_special_tokens=True).count("<bos>")
    n_fixed = toy_tokenize(templated, add_special_tokens=False).count("<bos>")
    assert (n_default, n_fixed) == (2, 1)
    print(f"Step 2: BOS count add_special_tokens=True -> {n_default}, False -> {n_fixed}")

    # Step 4: pad = eos, a 3-example batch, right padded
    eos = vocab["<|im_end|>"]
    seqs = [ids, ids[:24], ids[:24]]                           # full, and twice cut after turn 1
    T = max(map(len, seqs))
    batch = torch.tensor([s + [eos] * (T - len(s)) for s in seqs])
    attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs])
    lab_mask = torch.tensor([labels[: len(s)] + [-100] * (T - len(s)) for s in seqs])
    by_id = lab_mask.clone()
    by_id[batch == eos] = -100                                  # what an id-masking collator does
    by_pos = lab_mask.clone()
    by_pos[attn == 0] = -100                                    # what TRL's collator does
    n_id, n_pos = int(by_id.eq(eos).sum()), int(by_pos.eq(eos).sum())
    assert n_id == 0 and n_pos == 2 + 1 + 1, (n_id, n_pos)
    print(f"Step 4: turn-end targets kept: masked by id -> {n_id}, masked by position -> {n_pos}")

    # Step 5: hand loss vs HF-style loss on a tiny random model
    torch.manual_seed(0)
    model = TinyLM(len(vocab))
    with torch.no_grad():
        logits = model(batch)
    a, b = hand_loss(logits, by_pos), hf_style_loss(logits, by_pos)
    assert abs(a.item() - b.item()) < 1e-6
    print(f"Step 5: hand loss {a.item():.6f} vs HF-style loss {b.item():.6f}")


def smoke_never_stop(steps=250):
    """Two tiny LMs; only the one whose labels keep EOS learns to stop."""
    BOS, EOS, V, T = 1, 2, 10, 8                                # PAD == EOS == 2: the bug
    g = torch.Generator().manual_seed(0)
    data, attn = [], []
    for _ in range(256):
        n = int(torch.randint(2, 6, (1,), generator=g))
        content = torch.randint(3, V, (n,), generator=g).tolist()
        seq = [BOS] + content + [EOS]
        data.append(seq + [EOS] * (T - len(seq)))
        attn.append([1] * len(seq) + [0] * (T - len(seq)))
    data, attn = torch.tensor(data), torch.tensor(attn)

    def train(labels):
        torch.manual_seed(0)
        m = TinyLM(V)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(steps):
            loss = hf_style_loss(m(data), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m

    by_id = data.clone()
    by_id[data == EOS] = -100
    by_pos = data.clone()
    by_pos[attn == 0] = -100
    results = {}
    for name, labels in (("masked by id", by_id), ("masked by position", by_pos)):
        m = train(labels)
        stops = 0
        for p in range(5):
            ids = torch.tensor([[BOS, 3 + p, 4 + p]])
            with torch.no_grad():
                for _ in range(12):
                    nxt = m(ids)[0, -1].argmax().item()
                    ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
                    if nxt == EOS:
                        stops += 1
                        break
        results[name] = stops
        print(f"never-stop demo, labels {name}: {stops}/5 greedy generations emitted EOS")
    assert results["masked by id"] == 0 and results["masked by position"] == 5, results


def run_smoke():
    smoke_mask_and_bugs()
    smoke_never_stop()
    print("\nsmoke OK: mask, double BOS, pad=eos, hand loss and never-stop verified")


# ----------------------------------------------------------------------------
# Full mode: real tokenizers (and Qwen3-0.6B weights for step 5)
# ----------------------------------------------------------------------------

def assistant_char_spans(text, header, turn_end):
    """Character spans from the end of each assistant header through its turn-end token."""
    spans, cur = [], 0
    while True:
        h = text.find(header, cur)
        if h < 0:
            return spans
        start = h + len(header)
        e = text.find(turn_end, start)
        end = e + len(turn_end)
        spans.append((start, end))
        cur = end


def labels_from_offsets(tok, text, header, turn_end):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = assistant_char_spans(text, header, turn_end)
    labels = []
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        inside = any(a <= s and e <= b and e > s for a, b in spans)
        labels.append(tid if inside else -100)
    return enc["input_ids"], labels


def run_full(never_stop=False):
    from transformers import AutoTokenizer  # imported here so --smoke never needs it

    qwen = None
    for key, (repo, rev, header, turn_end) in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        except Exception as exc:  # gated repo without access, offline, ...
            print(f"\n[{key}] skipped: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        text = tok.apply_chat_template(CONVO, tokenize=False)
        print(f"\n===== {key}: {repo}@{rev[:7]}\nStep 1 rendered:\n{text}")

        bos = tok.bos_token_id
        a = tok(text)["input_ids"]
        b = tok(text, add_special_tokens=False)["input_ids"]
        c = tok.apply_chat_template(CONVO, tokenize=True, return_dict=True)["input_ids"]
        counts = [x.count(bos) if bos is not None else 0 for x in (a, b, c)]
        print(f"Step 2: BOS counts default/no-special/tokenize=True = {counts}; lengths {len(a)}/{len(b)}/{len(c)}")
        if bos is not None:
            assert counts == [2, 1, 1], counts

        out = tok.apply_chat_template(CONVO, tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True)
        print(f"Step 3: return_assistant_tokens_mask sum = {sum(out['assistant_masks'])} "
              "(0 means the template has no {% generation %} markers)")
        ids, labels = labels_from_offsets(tok, text, header, turn_end)
        toks = tok.convert_ids_to_tokens(ids)
        print_table(toks, labels, f"Step 3 ({key}): offsets-based assistant mask")
        print(f"trained tokens: {sum(l != -100 for l in labels)} of {len(ids)}")

        broken = []
        for k in range(1, len(CONVO)):
            short = tok.apply_chat_template(CONVO[:k], tokenize=False)
            longer = tok.apply_chat_template(CONVO[: k + 1], tokenize=False)
            if not longer.startswith(short):
                broken.append(k)
        print(f"prefix-preserving: {'yes' if not broken else 'NO, breaks after turn(s) ' + str(broken)}")
        if key == "qwen3":
            qwen = (tok, header, turn_end)

    if qwen is None:
        print("Qwen tokenizer unavailable; skipping steps 4-5")
        return
    tok, header, turn_end = qwen

    # Step 4: pad = eos with a 3-example batch
    variants = [CONVO, CONVO[:3], [CONVO[0], {"role": "user", "content": "Order ORD-00077 is late."},
                                   {"role": "assistant", "content": '{"order_id": "ORD-00077", "issue_type": "shipping"}'}]]
    rows = [labels_from_offsets(tok, tok.apply_chat_template(v, tokenize=False), header, turn_end)
            for v in variants]
    eos_id = tok.convert_tokens_to_ids(turn_end)
    T = max(len(r[0]) for r in rows)
    input_ids = torch.tensor([r[0] + [eos_id] * (T - len(r[0])) for r in rows])   # pad = eos
    attention_mask = torch.tensor([[1] * len(r[0]) + [0] * (T - len(r[0])) for r in rows])
    lab = torch.tensor([r[1] + [-100] * (T - len(r[1])) for r in rows])
    by_id = lab.clone()
    by_id[input_ids == eos_id] = -100
    n_id, n_pos = int(by_id.eq(eos_id).sum()), int(lab.eq(eos_id).sum())
    print(f"\nStep 4: turn-end targets kept: masked by id -> {n_id}, masked by position -> {n_pos}")
    assert n_id == 0 and n_pos > 0

    # Step 5: hand loss vs model(**batch).loss on Qwen3-0.6B
    from transformers import AutoModelForCausalLM
    repo, rev = MODELS["qwen3"][:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32).to(device)
    model.eval()
    batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
             "labels": lab.to(device)}
    with torch.no_grad():
        out = model(**batch)
    mine = hand_loss(out.logits, batch["labels"])
    print(f"Step 5: model loss {out.loss.item():.6f} vs hand loss {mine.item():.6f}")
    assert abs(out.loss.item() - mine.item()) < 1e-4
    del model

    if never_stop:
        run_never_stop(device)


def run_never_stop(device, steps=100, n_eval=20):
    """T0 extension: id-masked pad=<|im_end|> vs distinct pad, on Qwen3-0.6B-Base."""
    import random

    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

    base = "Qwen/Qwen3-0.6B-Base"
    rng = random.Random(0)
    names = ["Ana", "Bo", "Chen", "Dita", "Eli", "Fay", "Gus", "Hana"]
    issues = ["refund", "shipping", "defect", "account", "other"]   # module 01 enum

    def example():
        oid, who, issue = f"ORD-{rng.randint(0, 99999):05d}", rng.choice(names), rng.choice(issues)
        return [{"role": "user", "content": f"Order {oid} problem: {issue}. {who}"},
                {"role": "assistant",
                 "content": f'{{"customer_name": "{who}", "order_id": "{oid}", "issue_type": "{issue}"}}'}]

    train_convos = [example() for _ in range(400)]
    eval_convos = [example() for _ in range(n_eval)]
    results = {}
    for name, pad in (("pad=<|im_end|>, masked by id", "<|im_end|>"),
                      ("pad=<|endoftext|> (distinct)", "<|endoftext|>")):
        tok = AutoTokenizer.from_pretrained(base)
        tok.pad_token = pad
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable()
        collate = DataCollatorForLanguageModeling(tok, mlm=False)          # masks labels by pad id
        texts = [tok.apply_chat_template(c, tokenize=False) for c in train_convos]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        use_amp = device == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability() >= (8, 0) else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
        model.train()
        for step in range(steps):
            chunk = texts[(step * 4) % len(texts):(step * 4) % len(texts) + 4]
            batch = collate([tok(t, add_special_tokens=False) for t in chunk])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        stop_id = tok.convert_tokens_to_ids("<|im_end|>")
        stops = 0
        for c in eval_convos:
            ids = tok.apply_chat_template(c[:1], add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(device)
            out = model.generate(**ids, max_new_tokens=64, do_sample=False, eos_token_id=stop_id,
                                 pad_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
            new = out[0, ids["input_ids"].shape[1]:].tolist()
            stops += int(stop_id in new)
        results[name] = stops
        print(f"never-stop extension, {name}: {stops}/{n_eval} replies stopped on <|im_end|>")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("Record the gap. The id-masked run never receives a stop signal, so its stop rate")
    print("depends on what the base model already knew about <|im_end|>.")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy run, no downloads (default without GPU)")
    mode.add_argument("--full", action="store_true", help="real tokenizers + Qwen3-0.6B loss check")
    ap.add_argument("--never-stop", action="store_true", help="with --full: T0 training extension")
    args = ap.parse_args()
    full = args.full or (not args.smoke and torch.cuda.is_available())
    if args.never_stop and not full:
        ap.error("--never-stop requires --full")
    if full:
        run_full(never_stop=args.never_stop)
    else:
        run_smoke()


if __name__ == "__main__":
    main()
