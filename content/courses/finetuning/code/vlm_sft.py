"""VLM SFT lab for module 16 (Multimodal Fine-Tuning).

Purpose
    --smoke : a CPU-only, pure-torch rehearsal of what a VLM processor and
              TRL's DataCollatorForVisionLanguageModeling do for you.
              (1) A fake Qwen2.5-VL-style processor resizes each image with the
                  `smart_resize` rule (14-px patches, 2x2 merge, so 28 px per
                  token side) and expands one <|image_pad|> placeholder into N
                  image tokens.
              (2) A collator builds input_ids and labels with assistant-only
                  masking, so image tokens, prompt tokens and padding are -100.
              (3) A truncation guard raises when `max_length` would cut image
                  tokens (the failure `max_length=None` avoids).
              (4) A toy VLM (patch encoder -> 2x2 merger/projector -> tiny
                  causal decoder) is trained for 200 steps with the vision
                  encoder frozen ("LLM + projector" strategy) on a synthetic
                  "read the colour of the receipt stamp" task; the run asserts
                  the loss falls and the frozen encoder did not move.
    --full  : the real lab L16.1 with TRL's SFTTrainer: LoRA SFT of
              Qwen/Qwen2.5-VL-3B-Instruct (T0/T1) or Qwen/Qwen3-VL-8B-Instruct
              (T1, QLoRA) on naver-clova-ix/cord-v2 receipts (CC-BY-4.0).
              It scores the base model first, trains, then scores again:
              micro and per-field P/R/F1, JSON validity, FP on fields absent
              from gold, mean image tokens per receipt, and peak memory.
              --eval-only stops after the base-model score.

Module      content/courses/finetuning/16-multimodal-and-embedding-fine-tuning.md
Hardware    smoke: any CPU, < 60 s.  full: T0 (T4 16 GB, fp16, 3B + QLoRA) or
            T1 (24 GB, bf16, 3B LoRA or 8B QLoRA).
Versions    smoke: Python 3.11+, torch 2.x (no other dependency).
            full (as of October 2026): transformers 5.19, trl 1.15.0,
            peft 0.21.2, datasets, accelerate, bitsandbytes (QLoRA only).

Run (from the repository root)
    python content/courses/finetuning/code/vlm_sft.py --smoke
    python content/courses/finetuning/code/vlm_sft.py --full --model Qwen/Qwen2.5-VL-3B-Instruct --qlora  # T0
    python content/courses/finetuning/code/vlm_sft.py --full --model Qwen/Qwen2.5-VL-3B-Instruct          # T1
    python content/courses/finetuning/code/vlm_sft.py --full --model Qwen/Qwen3-VL-8B-Instruct --qlora    # T1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. Image-token budget: the smart_resize rule (transformers qwen2_vl)
# --------------------------------------------------------------------------

PATCH = 14
MERGE = 2
FACTOR = PATCH * MERGE  # 28 px per image token side
MIN_PIXELS = 3136  # Qwen2.5-VL-3B preprocessor_config.json
MAX_PIXELS = 12845056


def smart_resize(h: int, w: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Same arithmetic as transformers' Qwen2-VL `smart_resize`."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio must be < 200")
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def num_image_tokens(h: int, w: int, **kw) -> int:
    hb, wb = smart_resize(h, w, **kw)
    return (hb // PATCH) * (wb // PATCH) // (MERGE * MERGE)


# --------------------------------------------------------------------------
# 2. A fake processor: chat template + placeholder expansion
# --------------------------------------------------------------------------

SPECIAL = ["<pad>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
VOCAB = 512


class FakeProcessor:
    """Tokenizer + image processor + chat template, like AutoProcessor."""

    def __init__(self):
        self.special = {t: i for i, t in enumerate(SPECIAL)}
        self.pad_id = self.special["<pad>"]
        self.image_id = self.special["<|image_pad|>"]

    def word_id(self, w: str) -> int:
        # stable across processes (never use hash())
        d = hashlib.sha256(w.encode()).digest()
        return len(SPECIAL) + int.from_bytes(d[:4], "little") % (VOCAB - len(SPECIAL))

    def encode_text(self, s: str) -> list[int]:
        return [self.word_id(w) for w in s.split()]

    def __call__(self, messages: list[dict], image_sizes: list[tuple[int, int]]):
        """Render messages; return ids, an assistant mask, and per-image spans."""
        ids, assistant, spans = [], [], []
        img_iter = iter(image_sizes)
        for m in messages:
            is_asst = m["role"] == "assistant"
            head = [self.special["<|im_start|>"]] + self.encode_text(m["role"])
            ids += head
            assistant += [0] * len(head)
            for block in m["content"]:
                if block["type"] == "image":
                    h, w = next(img_iter)
                    n = num_image_tokens(h, w)
                    ids.append(self.special["<|vision_start|>"])
                    start = len(ids)
                    ids += [self.image_id] * n  # one placeholder -> n image tokens
                    spans.append((start, start + n))
                    ids.append(self.special["<|vision_end|>"])
                    assistant += [0] * (n + 2)
                else:
                    t = self.encode_text(block["text"])
                    ids += t
                    assistant += [int(is_asst)] * len(t)
            ids.append(self.special["<|im_end|>"])
            assistant.append(int(is_asst))  # train on the end-of-turn token too
        return ids, assistant, spans


def check_no_image_truncation(seq_len: int, spans: list[tuple[int, int]], max_length: int | None) -> None:
    """Raise if truncating at max_length would cut any image-token span."""
    if max_length is None:
        return
    for start, end in spans:
        if end > max_length:
            raise ValueError(
                f"max_length={max_length} cuts image tokens [{start},{end}) of a {seq_len}-token "
                "sequence; set max_length=None or lower the pixel budget"
            )


def collate(processor: FakeProcessor, examples: list[dict], max_length: int | None = None):
    rows = []
    for ex in examples:
        ids, asst, spans = processor(ex["messages"], ex["image_sizes"])
        check_no_image_truncation(len(ids), spans, max_length)
        if max_length is not None:
            ids, asst = ids[:max_length], asst[:max_length]
        rows.append((ids, asst))
    T = max(len(r[0]) for r in rows)
    input_ids = torch.full((len(rows), T), processor.pad_id, dtype=torch.long)  # (B, T)
    attention = torch.zeros((len(rows), T), dtype=torch.long)
    labels = torch.full((len(rows), T), -100, dtype=torch.long)
    for b, (ids, asst) in enumerate(rows):
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids)
        attention[b, :n] = 1
        lab = torch.tensor(ids)
        lab[torch.tensor(asst) == 0] = -100  # prompt, image tokens -> ignored
        labels[b, :n] = lab
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


# --------------------------------------------------------------------------
# 3. A toy VLM: patch encoder (frozen) -> 2x2 merger/projector -> decoder
# --------------------------------------------------------------------------

COLOURS = ["red", "green", "blue", "black"]
RGB = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.0, 0, 0]])


class ToyVLM(nn.Module):
    def __init__(self, d: int = 64, patch: int = 4):
        super().__init__()
        self.patch = patch
        self.vision = nn.Sequential(nn.Linear(3 * patch * patch, d), nn.GELU(), nn.Linear(d, d))
        self.projector = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))  # 2x2 merge
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(64, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, 2)
        self.lm_head = nn.Linear(d, VOCAB)

    def image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (3, H, W) -> patches (Hp*Wp, 3*p*p) -> merged tokens (Hp*Wp/4, d)
        p = self.patch
        _, H, W = pixels.shape
        x = pixels.unfold(1, p, p).unfold(2, p, p)  # (3, Hp, Wp, p, p)
        Hp, Wp = x.shape[1], x.shape[2]
        x = x.permute(1, 2, 0, 3, 4).reshape(Hp, Wp, 3 * p * p)
        x = self.vision(x)  # (Hp, Wp, d)
        x = x.reshape(Hp // 2, 2, Wp // 2, 2, -1).permute(0, 2, 1, 3, 4).reshape((Hp // 2) * (Wp // 2), -1)
        return self.projector(x)  # (N_img, d)

    def forward(self, input_ids, images, image_id):
        h = self.embed(input_ids)  # (B, T, d)
        h = h.clone()
        for b, pix in enumerate(images):
            feats = self.image_features(pix)
            pos = (input_ids[b] == image_id).nonzero().squeeze(-1)
            if pos.numel() != feats.shape[0]:
                raise ValueError(f"image tokens {pos.numel()} != features {feats.shape[0]}")
            h[b, pos] = feats
        T = input_ids.shape[1]
        h = h + self.pos(torch.arange(T))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        return self.lm_head(self.decoder(h, mask=mask))  # (B, T, V)


def toy_example(rng: torch.Generator, proc: FakeProcessor, size: int = 16):
    c = int(torch.randint(0, len(COLOURS), (1,), generator=rng))
    img = 0.1 * torch.rand(3, size, size, generator=rng)
    img[:, 4:12, 4:12] += RGB[c].view(3, 1, 1)  # the "stamp"
    ex = {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "stamp colour ?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"colour {COLOURS[c]}"}]},
        ],
    }
    return ex, img


def run_smoke() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    # (1) token budget, same numbers as Check 16.1
    assert smart_resize(1024, 1024) == (1036, 1036)
    assert num_image_tokens(1024, 1024) == 1369
    assert num_image_tokens(448, 448) == 256
    assert num_image_tokens(1024, 1024, max_pixels=1003520) == 1225  # library class default budget
    print("image tokens: 1024x1024 -> 1369, 448x448 -> 256 (1225 under the 1,003,520-px class default)")

    # (2) collator: assistant-only labels; image tokens never in the loss
    proc = FakeProcessor()
    ex = {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "extract the total"}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"total": 45500}'}]},
        ],
        "image_sizes": [(448, 448)],
    }
    batch = collate(proc, [ex, ex | {"image_sizes": [(224, 224)]}])
    ids, labels = batch["input_ids"], batch["labels"]
    assert (ids[0] == proc.image_id).sum() == 256 and (ids[1] == proc.image_id).sum() == 64
    assert (labels[ids == proc.image_id] == -100).all(), "image tokens leaked into the loss"
    assert (labels[batch["attention_mask"] == 0] == -100).all(), "padding leaked into the loss"
    n_train = int((labels[0] != -100).sum())
    assert n_train == len(proc.encode_text('{"total": 45500}')) + 1  # answer + <|im_end|>
    print(f"collator: {n_train} supervised tokens out of {int(batch['attention_mask'][0].sum())}")

    # (3) truncation guard
    big = ex | {"image_sizes": [(1024, 1024)]}
    try:
        collate(proc, [big], max_length=1024)
        raise AssertionError("guard did not fire")
    except ValueError as e:
        assert "cuts image tokens" in str(e)
    full = collate(proc, [big], max_length=None)
    assert int((full["input_ids"] == proc.image_id).sum()) == 1369
    print("truncation guard: max_length=1024 rejected, max_length=None keeps 1369 image tokens")

    # (4) toy VLM, frozen vision encoder, train projector + decoder
    rng = torch.Generator().manual_seed(0)
    model = ToyVLM()
    for p in model.vision.parameters():
        p.requires_grad_(False)
    frozen_before = [p.detach().clone() for p in model.vision.parameters()]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    n_img = (16 // 4) ** 2 // 4  # 16x16 image, 4-px patches, 2x2 merge -> 4 tokens

    def make_batch(bs: int = 16):
        exs, imgs = zip(*(toy_example(rng, proc) for _ in range(bs)))
        rows = []
        for e in exs:
            # the fake processor sees a 56x56 image -> 4 image tokens, matching the toy encoder
            ids, asst, _ = proc(e["messages"], [(56, 56)])
            assert ids.count(proc.image_id) == n_img
            rows.append((ids, asst))
        x = torch.tensor([r[0] for r in rows])
        y = x.clone()
        y[torch.tensor([r[1] for r in rows]) == 0] = -100
        return x, y, list(imgs)

    losses = []
    for step in range(200):
        x, y, imgs = make_batch()
        logits = model(x, imgs, proc.image_id)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"toy VLM loss {first:.3f} -> {last:.3f}")
    assert last < 0.5 * first, "loss did not fall"
    # the colour word can only be predicted from the image, so accuracy proves the projector carries it
    with torch.no_grad():
        x, y, imgs = make_batch(64)
        pred = model(x, imgs, proc.image_id)[:, :-1].argmax(-1)
        colour_pos = (y[:, 1:] != -100).float().cumsum(1) == 2  # 2nd supervised token = colour word
        acc = (pred[colour_pos] == y[:, 1:][colour_pos]).float().mean().item()
    print(f"colour accuracy from the image: {acc:.2f}")
    assert acc > 0.9, "image information did not reach the decoder"
    for a, b in zip(frozen_before, model.vision.parameters()):
        assert torch.equal(a, b), "frozen vision encoder moved"
    print(f"smoke OK in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# Full mode: TRL SFTTrainer on CORD-v2 receipts (GPU)
# --------------------------------------------------------------------------

SYSTEM = "Extract the receipt as JSON with keys menu (list of {nm, cnt, price}) and total ({total_price})."


def cord_to_messages(row):
    gt = json.loads(row["ground_truth"])["gt_parse"]
    target = {"menu": gt.get("menu", []), "total": {"total_price": gt.get("total", {}).get("total_price")}}
    return {
        "images": [row["image"]],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract this receipt."}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}]},
        ],
    }


def cord_units(rec):
    """Scoring units for a CORD target: one per menu item name, one for the total."""
    from collections import Counter

    u = Counter()
    for it in rec.get("menu", []) or []:
        u[("menu", " ".join(str(it.get("nm", "")).lower().split()))] += 1
    tp = (rec.get("total") or {}).get("total_price")
    if tp:
        try:
            # the module 16.5 normalisation, adapted to CORD: "," and "." are both
            # thousands separators here ("45,500", "60.000")
            u[("total", round(float(str(tp).replace(",", "").replace(".", "")), 2))] += 1
        except ValueError:
            u[("total", str(tp))] += 1
    return u


def evaluate_cord(model, processor, test, label: str) -> None:  # pragma: no cover - GPU
    """Greedy-decode the test split; print micro and per-field P/R/F1, the FP count on
    fields absent from gold, JSON validity, and mean image tokens per receipt."""
    model.eval()
    per, absent_fp, valid, img_tokens = {}, 0, 0, []
    merge = getattr(processor.image_processor, "merge_size", 2)
    for row in test:
        gold = json.loads(cord_to_messages(row)["messages"][2]["content"][0]["text"])
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": row["image"]},
                                         {"type": "text", "text": "Extract this receipt."}]},
        ]
        inputs = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        img_tokens.append(int(inputs["image_grid_thw"].prod(-1).sum()) // (merge * merge))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            pred = json.loads(text)
            valid += isinstance(pred, dict)
            pred = pred if isinstance(pred, dict) else {}
        except json.JSONDecodeError:
            pred = {}
        G, P = cord_units(gold), cord_units(pred)
        gold_fields = {k[0] for k in G}
        for k in set(G) | set(P):
            t = min(G[k], P[k])
            row_c = per.setdefault(k[0], [0, 0, 0])
            row_c[0] += t; row_c[1] += P[k] - t; row_c[2] += G[k] - t
            if k[0] not in gold_fields:
                absent_fp += P[k]  # a field the receipt does not have: hallucinated

    def prf(tp, fp, fn):
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    tot = [sum(v[i] for v in per.values()) for i in range(3)]
    p, r, f1 = prf(*tot)
    print(f"[{label}] field-level micro P={p:.3f} R={r:.3f} F1={f1:.3f}  "
          f"valid JSON {valid}/{len(test)}  FP on fields absent from gold: {absent_fp}")
    for field, (tp, fp, fn) in sorted(per.items()):
        fp_, fr_, ff_ = prf(tp, fp, fn)
        print(f"[{label}]   {field:6s} TP={tp} FP={fp} FN={fn}  P={fp_:.3f} R={fr_:.3f} F1={ff_:.3f}")
    print(f"[{label}] mean image tokens per receipt: {sum(img_tokens) / len(img_tokens):.0f}")


def run_full(args) -> None:  # pragma: no cover - needs a GPU and downloads
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("naver-clova-ix/cord-v2")
    side = 32 if "Qwen3-VL" in args.model else 28  # px per image token side: 16*2 vs 14*2
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * side * side, max_pixels=args.max_image_tokens * side * side
    )
    bf16 = torch.cuda.is_bf16_supported()
    model_kwargs = {"dtype": torch.bfloat16 if bf16 else torch.float16}
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=model_kwargs["dtype"]
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)

    # baseline: the same eval loop on the untouched base model
    torch.cuda.reset_peak_memory_stats()
    evaluate_cord(model, processor, ds["test"], "base")
    if args.eval_only:
        print(f"peak memory (eval): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return

    train = [cord_to_messages(r) for r in ds["train"]]  # list, not .map(): keeps PIL images as-is
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        # language-model linears only; the vision tower and merger stay frozen
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=None,  # never truncate image tokens
        assistant_only_loss=True,  # needs transformers>=5.18 for vision datasets
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=processor,
                         peft_config=peft_config)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    print(f"peak memory (training): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    trainer.save_model(args.output_dir)
    evaluate_cord(trainer.model, processor, ds["test"], "fine-tuned")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="full mode: score the base model on the CORD test split and stop")
    ap.add_argument("--max-image-tokens", type=int, default=1024)
    ap.add_argument("--output-dir", default="vlm-cord-lora")
    args = ap.parse_args()
    if args.full and torch.cuda.is_available():
        run_full(args)
    else:
        if args.full:
            print("no GPU found: running --smoke instead")
        run_smoke()


if __name__ == "__main__":
    main()
