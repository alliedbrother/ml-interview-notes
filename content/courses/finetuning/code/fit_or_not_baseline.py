"""Running-example generator, scorer and prompting baseline (support-email JSON extraction).

Module: 01 — When to Fine-Tune: What It Is, and When Not to Do It (Lab 1.7).
Contract: every field name, enum, file name and scoring rule below is the course's
running-example contract; later modules copy SYSTEM_PROMPT, the enums and the scorer
verbatim.

What it does
    1. Generates a deterministic synthetic support-email dataset (seed 0 by default):
       train.jsonl (200 rows, train templates) and eval.jsonl (50 rows, built only from
       held-out templates and phrases, 10 per issue type). Rows: {"id", "email", "gold"}.
    2. Defines the scorer every later module reuses: JSON parse rate, schema
       validity, per-field exact match, all-fields exact match.
    3. --smoke: writes the same two files, smaller (20 eval, 40 train, prefixes of the
       full files), checks the scorer on six fixture predictions, runs a tiny
       random-initialised torch model (which must score 0), and scores a rule-based
       extractor tuned on the train templates on both splits. CPU, stdlib + torch,
       no downloads, a few seconds.
    4. --data-only: writes the full-size files and exits (CPU, stdlib only).
    5. --full: zero-shot and 5-shot greedy generation with a real chat model using
       its chat template with thinking disabled; writes baseline_scores.json.

Hardware tier for --full: T0 (free Colab/Kaggle T4 16 GB, fp16) or better.
Pinned versions for --full: transformers>=5.0 (tested against the v5 API:
    `dtype=`, `apply_chat_template(..., enable_thinking=False)`), torch>=2.8.
Default models for --full: Qwen/Qwen3-0.6B (hybrid thinking checkpoint, thinking
    disabled via enable_thinking=False) and Qwen/Qwen3-4B-Instruct-2507
    (non-thinking only).

Run from the repository root:
    python content/courses/finetuning/code/fit_or_not_baseline.py --smoke
    python content/courses/finetuning/code/fit_or_not_baseline.py --data-only --out ft_data/
    python content/courses/finetuning/code/fit_or_not_baseline.py --full --out ft_data/
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

N_EVAL, N_TRAIN = 50, 200
N_EVAL_SMOKE, N_TRAIN_SMOKE = 20, 40

# ---------------------------------------------------------------- data generation
# Every surface component has a train pool and a disjoint eval pool, so the eval set
# tests generalisation to unseen phrasing rather than recall of the train templates.

FIRST = ["Maya", "Liam", "Priya", "Jonas", "Aiko", "Diego", "Fatima", "Noah",
         "Elena", "Kwame", "Sofia", "Ravi", "Hannah", "Mateo", "Leila", "Owen"]
LAST = ["Patel", "Okafor", "Nguyen", "Schmidt", "Garcia", "Kim", "Rossi",
        "Haddad", "Larsen", "Mensah", "Tanaka", "Silva", "Novak", "Brennan"]
PRODUCTS = ["blender", "desk lamp", "rain jacket", "phone case", "coffee grinder",
            "yoga mat", "backpack", "bluetooth speaker", "kettle", "monitor stand"]

POOLS = {
    "train": {
        "greeting": ["Hi support,", "Hi there,"],
        "body": {
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
        },
        "order_fmt": ["ORD-{n}", "order #{n}", "order number {n}"],
        "refund_ask": ["Please refund me.", "I would like my money back.",
                       "Can you issue a refund?"],
        "no_refund": ["Can you look into it?", "Please let me know what to do.",
                      "Any update helps."],
        "tone": {
            "positive": ["Thanks so much, you have always been great.",
                         "Appreciate the help!", "Love your shop otherwise."],
            "neutral": ["Thanks.", "Regards.", "Best."],
            "negative": ["This is really frustrating.", "I am very disappointed with this.",
                         "Not happy at all."],
        },
        "signoff": ["{name}", "Cheers,\n{name}", "{name}\nSent from my phone"],
    },
    "eval": {
        "greeting": ["Hello,", "Dear customer service,"],
        "body": {
            "refund": [
                "The {p} from order {o} went back to your warehouse on the 3rd and the "
                "return shows as received, but no credit yet.",
                "Your returns portal confirms the {p} (order {o}) arrived back with you "
                "a week ago.",
            ],
            "shipping": [
                "Order {o} was due last Tuesday and the {p} is nowhere to be seen.",
                "The courier marked order {o} as lost in transit, so I still have no {p}.",
            ],
            "defect": [
                "The {p} from order {o} makes a grinding noise and will not switch on.",
                "Half the stitching on the {p} (order {o}) came apart the first time I used it.",
            ],
            "account": [
                "Since I changed my password, my profile no longer lists order {o} for the {p}.",
                "The two-factor codes for my login stopped arriving, so I cannot check order {o}.",
            ],
            "other": [
                "Is there a matching cover for the {p} I bought in order {o}?",
                "Could you tell me whether the {p} from order {o} is dishwasher safe?",
            ],
        },
        "order_fmt": ["ORD-{n}", "order #{n}", "order number {n}",
                      "ORD{n}", "ord-{n}", "order no. {n}"],
        "refund_ask": ["I want a full refund.", "Please put the money back on my card."],
        "no_refund": ["Could someone check on this?", "What are my options?"],
        "tone": {
            "positive": ["You folks are wonderful, honestly.", "Big fan of your service."],
            "neutral": ["Thank you.", "Kind regards."],
            "negative": ["Honestly fed up with this.", "This is the last time I order here."],
        },
        "signoff": ["Best wishes,\n{name}", "- {name}", "{name} (customer since 2019)",
                    "Many thanks,\n{name}\nLondon"],
    },
}


def make_example(rng, pool, issue):
    sentiment = rng.choice(SENTIMENTS)
    refund = True if issue == "refund" else rng.random() < 0.25
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    num = f"{rng.randint(0, 99999):05d}"
    shown = rng.choice(pool["order_fmt"]).format(n=num)          # the email may not be
    body = rng.choice(pool["body"][issue]).format(p=rng.choice(PRODUCTS), o=shown)
    body = re.sub(r"\b([Oo])rder (?=order |ORD|ord-)", "", body)   # no "order order 123"
    body = body[0].upper() + body[1:]
    ask = rng.choice(pool["refund_ask"] if refund else pool["no_refund"])
    tone = rng.choice(pool["tone"][sentiment])
    sign = rng.choice(pool["signoff"]).format(name=name)
    email = f"{rng.choice(pool['greeting'])}\n\n{body} {ask} {tone}\n\n{sign}"
    gold = {"customer_name": name, "order_id": f"ORD-{num}", "issue_type": issue,
            "refund_requested": refund, "sentiment": sentiment}  # gold is always canonical
    return {"email": email, "gold": gold}


def make_split(split, n, seed=0):
    """Deterministic; a smaller n gives a prefix of a larger n with the same seed."""
    rng = random.Random(2 * seed + (0 if split == "eval" else 1))
    rows = []
    for i in range(n):
        if split == "eval":                      # stratified: 10 per issue type in 50
            if i % len(ISSUE_TYPES) == 0:
                block = ISSUE_TYPES[:]
                rng.shuffle(block)
            issue = block[i % len(ISSUE_TYPES)]
        else:
            issue = rng.choice(ISSUE_TYPES)
        row = make_example(rng, POOLS[split], issue)
        rows.append({"id": f"{split}-{i:04d}", **row})
    return rows


def make_dataset(seed=0, n_eval=N_EVAL, n_train=N_TRAIN):
    return make_split("eval", n_eval, seed), make_split("train", n_train, seed)


def check_held_out(eval_rows, train_rows):
    """No eval email reuses a train body template or refund/tone phrase; no duplicates."""
    tr, ev = POOLS["train"], POOLS["eval"]
    train_phrases = tr["refund_ask"] + tr["no_refund"] + sum(tr["tone"].values(), [])
    train_bodies = [re.compile(re.escape(b).replace(r"\{p\}", ".+?").replace(r"\{o\}", ".+?"))
                    for bs in tr["body"].values() for b in bs]
    for r in eval_rows:
        assert not any(p in r["email"] for p in train_phrases), r["id"]
        assert not any(t.search(r["email"]) for t in train_bodies), r["id"]
    for k in ("refund_ask", "no_refund", "greeting", "signoff"):
        assert not set(tr[k]) & set(ev[k]), k
    assert not set(sum(tr["body"].values(), [])) & set(sum(ev["body"].values(), []))
    emails = [r["email"] for r in eval_rows + train_rows]
    assert len(set(emails)) == len(emails), "duplicate email"
    assert all(re.match(ORDER_ID_RE, r["gold"]["order_id"]) for r in eval_rows + train_rows)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_split_files(out_dir, eval_rows, train_rows):
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "eval.jsonl"), eval_rows)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    print(f"wrote {len(eval_rows)} eval + {len(train_rows)} train rows to {out_dir} "
          f"(eval sha256 {digest(eval_rows)[:12]})")


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- scorer

FENCE = re.compile(r"^`{3}(?:json)?[ \t]*\n(.*)\n`{3}$", re.S)


def parse_prediction(text):
    """Strip whitespace and one surrounding ```json fence, then json.loads. Dict or None."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def schema_valid(obj):
    if obj is None or set(obj) != set(FIELDS):
        return False
    if not isinstance(obj["refund_requested"], bool):
        return False
    if not all(isinstance(obj[k], str) for k in FIELDS if k != "refund_requested"):
        return False
    return (re.match(ORDER_ID_RE, obj["order_id"]) is not None
            and obj["issue_type"] in ISSUE_TYPES and obj["sentiment"] in SENTIMENTS)


def field_match(obj, gold, key):
    if obj is None or key not in obj:
        return False
    return type(obj[key]) is type(gold[key]) and obj[key] == gold[key]


def score(predictions, golds):
    """All metrics are exact fractions over all len(golds) rows."""
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
    # extra key or non-canonical id: still parses and can match fields, but schema-invalid
    assert not schema_valid(dict(gold, note="x"))
    assert not schema_valid(dict(gold, order_id="#48213"))
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
    """A regex/keyword extractor written by reading the TRAIN templates only."""
    tr = POOLS["train"]
    outs = []
    for e in emails:
        order = re.search(r"(?:ORD-|#|number )(\d{5})\b", e)
        lines = [ln.strip() for ln in e.strip().splitlines() if ln.strip()]
        name = lines[-2] if lines[-1].startswith("Sent from") else lines[-1]
        low = e.lower()
        issue = ("refund" if "returned" in low or "sent it back" in low else
                 "shipping" if "tracking" in low or "label created" in low else
                 "defect" if "stopped working" in low or "cracked" in low else
                 "account" if "account" in low else "other")
        refund = any(a.lower() in low for a in tr["refund_ask"])
        sentiment = ("negative" if any(t.lower() in low for t in tr["tone"]["negative"]) else
                     "positive" if any(t.lower() in low for t in tr["tone"]["positive"]) else
                     "neutral")
        outs.append(json.dumps({"customer_name": name,
                                "order_id": f"ORD-{order.group(1)}" if order else "",
                                "issue_type": issue, "refund_requested": refund,
                                "sentiment": sentiment}))
    return outs


def run_smoke(out_dir, seed=0):
    eval_rows, train_rows = make_dataset(seed)
    assert digest(eval_rows) == digest(make_dataset(seed)[0]), "generator not deterministic"
    check_held_out(eval_rows, train_rows)
    assert all(schema_valid(r["gold"]) for r in eval_rows + train_rows)
    assert sorted(r["gold"]["issue_type"] for r in eval_rows) == sorted(ISSUE_TYPES * 10)
    small_eval, small_train = make_dataset(seed, N_EVAL_SMOKE, N_TRAIN_SMOKE)
    assert small_eval == eval_rows[:N_EVAL_SMOKE] and small_train == train_rows[:N_TRAIN_SMOKE]
    write_split_files(out_dir, small_eval, small_train)
    for r in small_eval[:3]:
        print(json.dumps(r["gold"]))

    fx = fixture_check()
    print("fixture scores:", as_floats(fx))

    golds = [r["gold"] for r in eval_rows]
    rand = score(tiny_random_model_outputs([r["email"] for r in eval_rows]), golds)
    assert rand["parse_rate"] == 0 and rand["all_fields_em"] == 0
    print("random-init model (eval 50):", as_floats(rand))

    # A train-tuned extractor saturates train but not the held-out eval set.
    rule_tr = score(rule_based_outputs([r["email"] for r in train_rows]),
                    [r["gold"] for r in train_rows])
    assert rule_tr["all_fields_em"] == 1, as_floats(rule_tr)
    rule = score(rule_based_outputs([r["email"] for r in eval_rows]), golds)
    assert rule["parse_rate"] == 1
    print("rule-based extractor (train 200):", as_floats(rule_tr))
    print("rule-based extractor (eval 50):", as_floats(rule))
    if seed == 0:                         # the numbers module 01 section 1.7 quotes
        assert rule["em_order_id"] == Fraction(32, 50), as_floats(rule)
        assert rule["em_customer_name"] == Fraction(18, 50), as_floats(rule)
        assert rule["all_fields_em"] == Fraction(1, 50), as_floats(rule)
    print("SMOKE OK")
    return rule


# ---------------------------------------------------------------- full mode

def build_messages(email, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": s["email"]})
        msgs.append({"role": "assistant", "content": json.dumps(s["gold"])})
    msgs.append({"role": "user", "content": email})
    return msgs


def run_full(out_dir, models, max_new_tokens, batch_size, seed=0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_rows, train_rows = make_dataset(seed)
    check_held_out(eval_rows, train_rows)
    write_split_files(out_dir, eval_rows, train_rows)
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
    mode.add_argument("--smoke", action="store_true",
                      help="CPU check; writes 20 eval + 40 train rows (default without a GPU)")
    mode.add_argument("--data-only", action="store_true",
                      help="CPU, stdlib only: write the full 50 eval + 200 train rows and exit")
    mode.add_argument("--full", action="store_true", help="real zero/5-shot baseline (T0 GPU)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: a temp dir for --smoke, ft_data/ otherwise)")
    ap.add_argument("--seed", type=int, default=0, help="dataset seed (default 0)")
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.data_only:
        eval_rows, train_rows = make_dataset(args.seed)
        check_held_out(eval_rows, train_rows)
        write_split_files(args.out or "ft_data/", eval_rows, train_rows)
        return
    full = args.full
    if not args.full and not args.smoke:
        try:
            import torch
            full = torch.cuda.is_available()
        except ImportError:
            full = False
    if full:
        run_full(args.out or "ft_data/", args.models, args.max_new_tokens, args.batch_size,
                 args.seed)
    elif args.out:
        run_smoke(args.out, args.seed)
    else:
        with tempfile.TemporaryDirectory() as d:
            run_smoke(d, args.seed)


if __name__ == "__main__":
    main()
