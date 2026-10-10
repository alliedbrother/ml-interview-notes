"""Prompting baseline for the course's running task: support-email JSON extraction.

Module: 01 — Orientation: What Fine-Tuning Is, and When Not to Do It (Lab 1.7).

What it does
    1. Generates a deterministic synthetic support-email dataset from templates
       (seed 0): 50 golden eval emails and 200 train emails, written as JSONL.
    2. Defines the scorer every later module reuses: JSON parse rate, schema
       validity, per-field exact match, all-fields exact match.
    3. --smoke: checks the scorer on six fixture predictions, runs the generate ->
       score path with a tiny random-initialised torch model (which must score 0),
       and scores a rule-based extractor end to end. CPU, stdlib + torch only,
       no downloads, well under a minute.
    4. --full: zero-shot and 5-shot greedy generation with a real chat model using
       its chat template with thinking disabled; writes baseline_scores.json.

Hardware tier for --full: T0 (free Colab/Kaggle T4 16 GB, fp16) or better.
Pinned versions for --full: transformers>=5.0 (tested against the v5 API:
    `dtype=`, `apply_chat_template(..., enable_thinking=False)`), torch>=2.8.
Default models for --full: Qwen/Qwen3-0.6B (hybrid thinking checkpoint, thinking
    disabled via enable_thinking=False) and Qwen/Qwen3-4B-Instruct-2507
    (non-thinking only).

Run from the repository root:
    python content/courses/finetuning/code/fit_or_not_baseline.py --smoke
    python content/courses/finetuning/code/fit_or_not_baseline.py --full --out ./ft_data/
"""

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from fractions import Fraction

FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
ISSUE_TYPES = ["refund", "shipping", "defect", "account", "other"]
SENTIMENTS = ["positive", "neutral", "negative"]

SYSTEM_PROMPT = (
    "You extract structured data from customer support emails. Reply with one JSON "
    "object and nothing else, with exactly these keys: "
    '"customer_name" (string), "order_id" (string), '
    '"issue_type" (one of "refund", "shipping", "defect", "account", "other"), '
    '"refund_requested" (true or false), '
    '"sentiment" (one of "positive", "neutral", "negative").'
)

# ---------------------------------------------------------------- data generation

FIRST = ["Maya", "Liam", "Priya", "Jonas", "Aiko", "Diego", "Fatima", "Noah",
         "Elena", "Kwame", "Sofia", "Ravi", "Hannah", "Mateo", "Leila", "Owen"]
LAST = ["Patel", "Okafor", "Nguyen", "Schmidt", "Garcia", "Kim", "Rossi",
        "Haddad", "Larsen", "Mensah", "Tanaka", "Silva", "Novak", "Brennan"]
PRODUCTS = ["blender", "desk lamp", "rain jacket", "phone case", "coffee grinder",
            "yoga mat", "backpack", "bluetooth speaker", "kettle", "monitor stand"]

BODY = {
    "refund": [
        "I returned the {p} from order {o} two weeks ago and have not seen the money back.",
        "The {p} (order {o}) is not what I expected and I sent it back last Monday.",
    ],
    "shipping": [
        "My order {o} for a {p} still shows 'label created' after nine days.",
        "Tracking for order {o} says delivered, but there is no {p} at my door.",
    ],
    "defect": [
        "The {p} from order {o} stopped working after three days.",
        "The {p} I got in order {o} arrived with a cracked base.",
    ],
    "account": [
        "I cannot log in to see order {o}; the reset link for my account never arrives.",
        "My account shows someone else's address on order {o} for the {p}.",
    ],
    "other": [
        "Quick question about order {o}: does the {p} come with a manual in Spanish?",
        "I wanted to ask whether the {p} in order {o} can be gift wrapped.",
    ],
}
REFUND_ASK = ["Please refund me.", "I would like my money back.", "Can you issue a refund?"]
NO_REFUND = ["Can you look into it?", "Please let me know what to do.", "Any update helps."]
TONE = {
    "positive": ["Thanks so much, you have always been great.", "Appreciate the help!",
                 "Love your shop otherwise."],
    "neutral": ["Thanks.", "Regards.", "Best."],
    "negative": ["This is really frustrating.", "I am very disappointed with this.",
                 "Not happy at all."],
}
ORDER_FMT = ["ORD-{n}", "order #{n}", "order number {n}"]
SIGNOFF = ["{name}", "{name}\nSent from my phone", "Cheers,\n{name}"]


def make_example(rng):
    issue = rng.choice(ISSUE_TYPES)
    sentiment = rng.choice(SENTIMENTS)
    refund = True if issue == "refund" else rng.random() < 0.25
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    num = rng.randint(10000, 99999)
    order = f"ORD-{num}"                                  # gold is always normalised
    shown = rng.choice(ORDER_FMT).format(n=num)           # the email may not be
    body = rng.choice(BODY[issue]).format(p=rng.choice(PRODUCTS), o=shown)
    body = body.replace("order order", "order")
    ask = rng.choice(REFUND_ASK if refund else NO_REFUND)
    tone = rng.choice(TONE[sentiment])
    sign = rng.choice(SIGNOFF).format(name=name)
    email = f"Hi support,\n\n{body} {ask} {tone}\n\n{sign}"
    gold = {"customer_name": name, "order_id": order, "issue_type": issue,
            "refund_requested": refund, "sentiment": sentiment}
    return {"email": email, "gold": gold}


def make_dataset(seed=0, n_eval=50, n_train=200):
    rng = random.Random(seed)
    rows = [make_example(rng) for _ in range(n_eval + n_train)]
    return rows[:n_eval], rows[n_eval:]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- scorer

FENCE = re.compile(r"^\s*`{3}(?:json)?\s*\n(.*?)\n\s*`{3}\s*$", re.S)


def parse_prediction(text):
    """Strip one surrounding ```json fence, then json.loads. Returns dict or None."""
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def schema_valid(obj):
    if obj is None or any(k not in obj for k in FIELDS):
        return False
    if not isinstance(obj["refund_requested"], bool):
        return False
    if not all(isinstance(obj[k], str) for k in FIELDS if k != "refund_requested"):
        return False
    return obj["issue_type"] in ISSUE_TYPES and obj["sentiment"] in SENTIMENTS


def field_match(obj, gold, key):
    if obj is None or key not in obj:
        return False
    return type(obj[key]) is type(gold[key]) and obj[key] == gold[key]


def score(predictions, golds):
    n = len(golds)
    parsed = [parse_prediction(p) for p in predictions]
    out = {
        "parse_rate": Fraction(sum(o is not None for o in parsed), n),
        "schema_valid": Fraction(sum(schema_valid(o) for o in parsed), n),
        "all_fields_em": Fraction(
            sum(all(field_match(o, g, k) for k in FIELDS) for o, g in zip(parsed, golds)), n),
    }
    for k in FIELDS:
        out[f"em_{k}"] = Fraction(sum(field_match(o, g, k) for o, g in zip(parsed, golds)), n)
    return out


def as_floats(scores):
    return {k: round(float(v), 4) for k, v in scores.items()}


# ---------------------------------------------------------------- smoke mode

def fixture_check():
    gold = {"customer_name": "Maya Patel", "order_id": "ORD-48213", "issue_type": "refund",
            "refund_requested": True, "sentiment": "negative"}
    perfect = json.dumps(gold)
    fence = "`" * 3
    preds = [
        perfect,                                                      # (a) perfect
        perfect.replace('"refund",', '"billing",'),                    # (b) enum violation
        perfect[:-1] + ",}",                                           # (c) trailing comma
        f"{fence}json\n{perfect}\n{fence}",                            # (d) fenced, correct
        json.dumps({k: v for k, v in gold.items() if k != "sentiment"}),  # (e) missing field
        perfect.replace("true", '"true"'),                             # (f) bool as string
    ]
    s = score(preds, [gold] * 6)
    assert s["parse_rate"] == Fraction(5, 6)
    assert s["schema_valid"] == Fraction(2, 6)
    assert s["all_fields_em"] == Fraction(2, 6)
    assert s["em_order_id"] == Fraction(5, 6)
    assert s["em_issue_type"] == Fraction(4, 6)
    return s


def tiny_random_model_outputs(emails, max_new_tokens=24, seed=0):
    """Greedy byte-level generation from a random-init GRU: the 'untrained model' floor."""
    import torch

    torch.manual_seed(seed)

    class TinyByteLM(torch.nn.Module):
        def __init__(self, vocab=256, d=32):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d)
            self.rnn = torch.nn.GRU(d, d, batch_first=True)
            self.head = torch.nn.Linear(d, vocab)

        def forward(self, ids):                       # ids: (B, T)
            h, _ = self.rnn(self.emb(ids))            # (B, T, d)
            return self.head(h)                       # (B, T, V)

    model = TinyByteLM().eval()
    outs = []
    with torch.no_grad():
        for e in emails:
            ids = torch.tensor([list(e.encode("utf-8"))[-64:]])   # (1, T)
            gen = []
            for _ in range(max_new_tokens):
                nxt = model(ids)[:, -1].argmax(-1, keepdim=True)  # (1, 1)
                gen.append(int(nxt))
                ids = torch.cat([ids, nxt], dim=1)
            outs.append(bytes(gen).decode("utf-8", errors="replace"))
    return outs


def rule_based_outputs(emails):
    """A deliberately naive regex extractor: a non-ML baseline that exercises the scorer."""
    outs = []
    for e in emails:
        order = re.search(r"ORD-\d{5}", e)
        name = e.strip().splitlines()[-1].strip()
        low = e.lower()
        issue = ("refund" if "returned" in low or "sent it back" in low else
                 "shipping" if "tracking" in low or "label created" in low else
                 "defect" if "stopped working" in low or "cracked" in low else
                 "account" if "account" in low else "other")
        refund = any(a.lower() in low for a in REFUND_ASK)
        sentiment = ("negative" if "frustrat" in low or "disappointed" in low else
                     "positive" if "appreciate" in low or "thanks so much" in low else "neutral")
        outs.append(json.dumps({"customer_name": name,
                                "order_id": order.group(0) if order else "",
                                "issue_type": issue, "refund_requested": refund,
                                "sentiment": sentiment}))
    return outs


def run_smoke(out_dir):
    eval_rows, train_rows = make_dataset(seed=0)
    assert len(eval_rows) == 50 and len(train_rows) == 200
    assert digest(eval_rows) == digest(make_dataset(seed=0)[0]), "generator not deterministic"
    golds = [r["gold"] for r in eval_rows]
    assert all(schema_valid(g) for g in golds)
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "eval.jsonl"), eval_rows)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    print(f"wrote 50 eval + 200 train rows to {out_dir} (eval sha256 {digest(eval_rows)[:12]})")
    for r in eval_rows[:5]:
        print(json.dumps(r["gold"]))

    fx = fixture_check()
    print("fixture scores:", as_floats(fx))

    rand = score(tiny_random_model_outputs([r["email"] for r in eval_rows]), golds)
    assert rand["parse_rate"] == 0 and rand["all_fields_em"] == 0
    print("random-init model:", as_floats(rand))

    rule = score(rule_based_outputs([r["email"] for r in eval_rows]), golds)
    # Always valid JSON, but surface variation (order-id formats, sign-offs, unlisted
    # tone phrases) defeats the regexes: the dumbest baseline is worth one minute.
    assert rule["parse_rate"] == 1 and rule["schema_valid"] == 1
    assert rule["em_order_id"] == Fraction(17, 50), as_floats(rule)
    assert rule["all_fields_em"] == Fraction(10, 50), as_floats(rule)
    print("rule-based extractor:", as_floats(rule))
    print("SMOKE OK")


# ---------------------------------------------------------------- full mode

def build_messages(email, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": s["email"]})
        msgs.append({"role": "assistant", "content": json.dumps(s["gold"])})
    msgs.append({"role": "user", "content": email})
    return msgs


def run_full(out_dir, models, max_new_tokens, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_rows, train_rows = make_dataset(seed=0)
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "eval.jsonl"), eval_rows)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    golds = [r["gold"] for r in eval_rows]
    shots = train_rows[:5]
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32   # T4 has no bf16
    results = {}
    for name in models:
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to("cuda" if cuda else "cpu").eval()
        for k, shot_list in (("zero_shot", []), ("five_shot", shots)):
            prompts = [tok.apply_chat_template(build_messages(r["email"], shot_list),
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                       for r in eval_rows]
            preds = []
            for i in range(0, len(prompts), batch_size):
                enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens)
                new = gen[:, enc["input_ids"].shape[1]:]                  # (B, new_T)
                preds += tok.batch_decode(new, skip_special_tokens=True)
            s = as_floats(score([p.strip() for p in preds], golds))
            results[f"{name}::{k}"] = s
            print(name, k, s)
        del model
        if cuda:
            torch.cuda.empty_cache()
    with open(os.path.join(out_dir, "baseline_scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(out_dir, "baseline_scores.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU check (default without a GPU)")
    mode.add_argument("--full", action="store_true", help="real zero/5-shot baseline (T0 GPU)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: a temp dir for --smoke, ./ft_data/ for --full)")
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    full = args.full
    if not args.full and not args.smoke:
        try:
            import torch
            full = torch.cuda.is_available()
        except ImportError:
            full = False
    if full:
        run_full(args.out or "./ft_data/", args.models, args.max_new_tokens, args.batch_size)
    elif args.out:
        run_smoke(args.out)
    else:
        with tempfile.TemporaryDirectory() as d:
            run_smoke(d)


if __name__ == "__main__":
    main()
