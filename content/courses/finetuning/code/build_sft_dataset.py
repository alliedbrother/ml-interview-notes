"""Build a clean, deduplicated, decontaminated SFT dataset in every TRL schema.

Module: 04 - Data: Formats, Curation, Synthesis, and Decontamination (labs L4.1, L4.2).

What it does, in order:
  1. load raw support-email records (email text + gold JSON extraction),
  2. normalize and validate the extraction against the course schema,
  3. exact dedup (sha256 of normalized text),
  4. near-duplicate dedup (character 5-gram MinHash, 128 hashes, LSH b=32 x r=4),
  5. decontaminate against the golden eval set (13-gram word overlap),
  6. emit language-modeling, prompt-completion, conversational and preference rows,
  7. validate every emitted row with a schema checker,
  8. analyze templated token lengths and choose max_length at a percentile.

Modes:
  --smoke (default without a GPU): stdlib only, an inline corpus with planted
      exact/near duplicates, a planted contaminated example and a malformed record;
      a whitespace "tokenizer" with a fixed ChatML-like overhead; asserts the
      planted rows are removed. Runs in well under a second on a laptop CPU.
  --full: real inputs and tokenizer. Hardware tier: CPU for --input processing;
      T1 (single 24 GB GPU) for --magpie generation with vLLM.
      Pinned versions used when writing (October 2026): transformers>=5.0,
      datasets>=4.7, vllm>=0.11, huggingface_hub>=1.0. Imports happen inside the
      functions that need them, so --smoke never imports them.

Run (from the repository root):
  python content/courses/finetuning/code/build_sft_dataset.py --smoke
  python content/courses/finetuning/code/build_sft_dataset.py --full \
      --input raw.jsonl --eval golden_eval.jsonl --tokenizer Qwen/Qwen3-0.6B \
      --output-dir out/ [--magpie 2000] [--push-to-hub user/support-email-sft]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import tempfile
from pathlib import Path

ISSUE_TYPES = {"damaged", "late", "missing", "billing", "account", "other"}
SENTIMENTS = {"negative", "neutral", "positive"}
FIELDS = {
    "customer_name": str,
    "order_id": str,
    "issue_type": str,
    "refund_requested": bool,
    "sentiment": str,
}
SYSTEM = ("Extract customer_name, order_id, issue_type, refund_requested and "
          "sentiment from the email. Reply with one JSON object only.")


# ---------------------------------------------------------------- schema --
def validate_extraction(obj) -> list[str]:
    """Return a list of problems; empty means the gold label is valid."""
    if not isinstance(obj, dict):
        return ["not an object"]
    errs = [f"missing {k}" for k in FIELDS if k not in obj]
    errs += [f"extra {k}" for k in obj if k not in FIELDS]
    for k, typ in FIELDS.items():
        if k in obj and not isinstance(obj[k], typ):
            errs.append(f"{k} is not {typ.__name__}")
    if obj.get("issue_type") not in ISSUE_TYPES:
        errs.append("issue_type outside enum")
    if obj.get("sentiment") not in SENTIMENTS:
        errs.append("sentiment outside enum")
    return errs


def _is_messages(x) -> bool:
    return (isinstance(x, list) and len(x) > 0 and all(
        isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"}
        and isinstance(m.get("content"), str) for m in x))


def validate_row(row: dict, kind: str) -> None:
    """Schema checker for the four TRL dataset types this lab emits."""
    if kind == "language_modeling":
        assert set(row) == {"text"} and isinstance(row["text"], str) and row["text"]
    elif kind == "prompt_completion":
        assert set(row) == {"prompt", "completion"}
        assert _is_messages(row["prompt"]) and _is_messages(row["completion"])
        assert row["prompt"][-1]["role"] == "user"
        assert [m["role"] for m in row["completion"]] == ["assistant"]
    elif kind == "conversational":
        assert set(row) == {"messages"} and _is_messages(row["messages"])
        assert row["messages"][-1]["role"] == "assistant"
    elif kind == "preference":
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert _is_messages(row["prompt"]) and _is_messages(row["chosen"])
        assert _is_messages(row["rejected"]) and row["chosen"] != row["rejected"]
    else:
        raise ValueError(kind)


# ------------------------------------------------------------ text utils --
def normalize(text: str) -> str:
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return " ".join(text.lower().translate(table).split())


def sha256(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def char_shingles(text: str, k: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


class MinHasher:
    P = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 0):
        rng = random.Random(seed)
        self.coeffs = [(rng.randrange(1, self.P), rng.randrange(0, self.P))
                       for _ in range(num_perm)]

    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

    def signature(self, shingles: set[str]) -> tuple[int, ...]:
        xs = [self._h(s) for s in shingles]
        return tuple(min((a * x + b) % self.P for x in xs) for a, b in self.coeffs)


def minhash_dedup(texts: list[str], bands: int = 32, rows: int = 4) -> list[int]:
    """Return indices to keep: first member of every LSH-connected near-dup group."""
    mh = MinHasher(num_perm=bands * rows)
    sigs = [mh.signature(char_shingles(t)) for t in texts]
    buckets: dict[tuple, int] = {}
    keep = []
    for i, sig in enumerate(sigs):
        keys = [(b, sig[b * rows:(b + 1) * rows]) for b in range(bands)]
        if any(k in buckets for k in keys):
            continue                                  # near-duplicate of a kept doc
        for k in keys:
            buckets[k] = i
        keep.append(i)
    return keep


def ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    toks = normalize(text).split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contaminated(texts: list[str], eval_texts: list[str], n: int = 13) -> list[int]:
    bank = set().union(*(ngram_set(e, n) for e in eval_texts)) if eval_texts else set()
    return [i for i, t in enumerate(texts) if ngram_set(t, n) & bank]


# --------------------------------------------------------- format export --
def to_formats(rec: dict) -> dict[str, dict]:
    gold = json.dumps(rec["label"], sort_keys=True)
    prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rec["email"]}]
    completion = [{"role": "assistant", "content": gold}]
    # A cheap, schema-violating rejected answer: drop one required field.
    bad = dict(rec["label"])
    bad.pop("refund_requested")
    rejected = [{"role": "assistant", "content": json.dumps(bad, sort_keys=True)}]
    return {
        "language_modeling": {"text": f"Email:\n{rec['email']}\nJSON:\n{gold}"},
        "prompt_completion": {"prompt": prompt, "completion": completion},
        "conversational": {"messages": prompt + completion},
        "preference": {"prompt": prompt, "chosen": completion, "rejected": rejected},
    }


def percentile(xs: list[int], q: float) -> int:
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[idx]


def length_report(lengths: list[tuple[int, int]], max_length: int) -> dict:
    total = [p + c for p, c in lengths]
    lost_completion = sum(min(c, max(0, p + c - max_length)) for p, c in lengths)
    zero_trained = sum(p >= max_length for p, _ in lengths)
    return {
        "n": len(total),
        "p50": percentile(total, 50), "p95": percentile(total, 95),
        "p99": percentile(total, 99), "max": max(total),
        "max_length": max_length,
        "frac_truncated": sum(t > max_length for t in total) / len(total),
        "frac_completion_tokens_lost": lost_completion / max(1, sum(c for _, c in lengths)),
        "examples_with_zero_trained_tokens": zero_trained,
    }


# -------------------------------------------------------------- pipeline --
def run_pipeline(raw: list[dict], eval_set: list[dict], count_tokens) -> dict:
    log = {"raw": len(raw)}
    valid = [r for r in raw if isinstance(r.get("email"), str) and not validate_extraction(r.get("label"))]
    log["after_schema"] = len(valid)

    seen, exact = set(), []
    for r in valid:
        h = sha256(r["email"])
        if h not in seen:
            seen.add(h)
            exact.append(r)
    log["after_exact_dedup"] = len(exact)

    near = [exact[i] for i in minhash_dedup([r["email"] for r in exact])]
    log["after_minhash_dedup"] = len(near)

    bad = set(contaminated([r["email"] for r in near], [e["email"] for e in eval_set], n=13))
    clean = [r for i, r in enumerate(near) if i not in bad]
    log["after_decontamination"] = len(clean)

    rows = {k: [] for k in ("language_modeling", "prompt_completion", "conversational", "preference")}
    lengths = []
    for r in clean:
        for kind, row in to_formats(r).items():
            validate_row(row, kind)
            rows[kind].append(row)
        pc = to_formats(r)["prompt_completion"]
        lengths.append(count_tokens(pc["prompt"], pc["completion"]))
    p99 = percentile([p + c for p, c in lengths], 99)
    max_length = 1 << (p99 - 1).bit_length()          # round p99 up to a power of two
    log["lengths"] = length_report(lengths, max_length)
    return {"log": log, "rows": rows, "clean": clean}


def write_jsonl(rows_by_kind: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for kind, rows in rows_by_kind.items():
        with open(out / f"{kind}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------- smoke --
SMOKE_EMAILS = [
    ("Dana Kim", "10412", "damaged", True, "negative",
     "Hello, order 10412 arrived today and the blender jar is cracked down the side. I want my money back, not a replacement. Dana Kim"),
    ("Luis Ortega", "10449", "late", False, "neutral",
     "Hi, just checking on order 10449. The tracking page says it left the depot nine days ago and nothing since. No rush, but could you look into it? Luis"),
    ("Priya Raman", "10486", "missing", True, "negative",
     "The courier marked parcel 10486 as delivered on Friday, but nothing was on my porch and my neighbours saw no van. Refund please. - Priya Raman"),
    ("Tomasz Nowak", "10523", "billing", True, "negative",
     "My card statement shows two charges of 49.99 for order 10523. I only placed it once. Please reverse the duplicate charge. Tomasz Nowak"),
    ("Aiko Sato", "10560", "account", False, "neutral",
     "I cannot log in since I changed my email address last week; the reset link never arrives. Order 10560 is still pending under my old account. Aiko"),
    ("Femi Ade", "10597", "other", False, "positive",
     "Loved the rain jacket from order 10597, it kept me dry on a long hike! Do you sell the matching trousers in size M? Cheers, Femi Ade"),
    ("Grace Liu", "10634", "damaged", False, "negative",
     "Order 10634: the desk lamp shade came dented and one bulb socket rattles. Can you ship a new shade? I'd rather not return the whole lamp. Grace Liu"),
    ("Omar Haddad", "10671", "late", True, "negative",
     "This is the third time I am writing about order 10671. It was promised for my daughter's birthday, which has now passed. Cancel it and refund me. Omar Haddad"),
    ("Sofia Rossi", "10708", "missing", False, "neutral",
     "Box for order 10708 arrived but the charging cable listed on the packing slip was not inside. Could you send just the cable? Thanks, Sofia Rossi"),
    ("Ben Carter", "10745", "billing", False, "neutral",
     "Quick question about invoice 10745: the VAT line looks higher than the rate on your website. Is that a mistake or am I misreading it? Ben Carter"),
    ("Mei Chen", "10782", "account", False, "negative",
     "Someone changed the shipping address on my profile without my permission and order 10782 is now going to a city I have never lived in. Mei Chen"),
    ("Jonas Berg", "10819", "other", False, "positive",
     "Just wanted to say thanks: your support agent sorted out my exchange for order 10819 in five minutes. Great service. Jonas Berg"),
    ("Ana Souza", "10856", "damaged", True, "negative",
     "The coffee grinder from order 10856 smells of burning plastic after one use and the motor stalls. I will not use it again; please refund. Ana Souza"),
    ("Raj Patel", "10893", "late", False, "neutral",
     "Order 10893 shows 'label created' for a week now. Has it actually shipped? I need it before the 20th if possible. Regards, Raj Patel"),
    ("Chloe Martin", "10930", "billing", True, "negative",
     "I returned the backpack from order 10930 a month ago, you confirmed receipt, and I still have not seen the refund on my card. Chloe Martin"),
    ("Kwame Mensah", "10967", "missing", False, "neutral",
     "Two of the three phone cases in order 10967 arrived. The navy one is missing from the bag. Can you check the warehouse? Kwame Mensah"),
]


def smoke_corpus() -> tuple[list[dict], list[dict]]:
    keys = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
    raw = [{"email": e[-1], "label": dict(zip(keys, e[:-1]))} for e in SMOKE_EMAILS]
    eval_set = [{
        "email": ("Hi, my order 55555 for a standing desk arrived with one leg bent "
                  "and two screws missing from the hardware bag, so I cannot assemble "
                  "it at all. Please refund me. Thanks, Grace Liu"),
        "label": {"customer_name": "Grace Liu", "order_id": "55555", "issue_type": "damaged",
                  "refund_requested": True, "sentiment": "negative"}}]
    planted = [
        dict(raw[3]),                                                         # exact duplicate
        {"email": raw[5]["email"].upper().replace(".", "!"), "label": raw[5]["label"]},  # equal after normalize
        {"email": raw[7]["email"].replace("third", "fourth").replace("Cancel", "Please cancel"),
         "label": raw[7]["label"]},                                           # near duplicate
        {"email": "FWD: " + eval_set[0]["email"].lower(), "label": eval_set[0]["label"]},  # contaminated
        {"email": "Where is my parcel?", "label": {"customer_name": "X", "order_id": 7}},   # malformed
    ]
    return raw + planted, eval_set


def whitespace_count(prompt, completion):
    overhead = 4                                       # role header + end marker per message
    p = sum(len(m["content"].split()) + overhead for m in prompt) + 3   # + generation prompt
    c = sum(len(m["content"].split()) + overhead for m in completion)
    return p, c


def smoke(output_dir: str | None) -> None:
    raw, eval_set = smoke_corpus()
    out = run_pipeline(raw, eval_set, whitespace_count)
    log = out["log"]
    for k, v in log.items():
        print(f"{k:>24}: {v}")
    assert log["raw"] == 21
    assert log["after_schema"] == 20, "malformed record must be dropped"
    assert log["after_exact_dedup"] == 18, "exact + normalized duplicates must be dropped"
    assert log["after_minhash_dedup"] == 17, "near duplicate must be dropped"
    assert log["after_decontamination"] == 16, "contaminated example must be dropped"
    emails = [r["email"] for r in out["clean"]]
    assert not any("55555" in e for e in emails)
    assert all(len(rows) == 16 for rows in out["rows"].values())
    assert log["lengths"]["frac_truncated"] == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(output_dir) if output_dir else Path(tmp)
        write_jsonl(out["rows"], target)
        assert sum(1 for _ in open(target / "preference.jsonl")) == 16
    print("smoke OK: planted duplicates, contamination and malformed row removed; all formats validated")


# ------------------------------------------------------------------ full --
def magpie_generate(model_id: str, n: int, seed: int = 0) -> list[dict]:
    """Magpie: feed only the pre-query template so the model writes the user turn."""
    from vllm import LLM, SamplingParams   # imported here so --smoke never needs vLLM

    llm = LLM(model=model_id, seed=seed, max_model_len=4096)
    pre_query = "<|im_start|>user\n"          # ChatML pre-query template (Qwen3)
    q_params = SamplingParams(n=1, temperature=1.0, top_p=0.99, max_tokens=256,
                              stop=["<|im_end|>"], seed=seed)
    instr = [o.outputs[0].text.strip() for o in llm.generate([pre_query] * n, q_params)]
    instr = [q for q in instr if 20 <= len(q) <= 2000]
    convs = [[{"role": "user", "content": q}] for q in instr]
    a_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=1024, seed=seed)
    answers = llm.chat(convs, a_params)
    return [{"messages": c + [{"role": "assistant", "content": a.outputs[0].text}]}
            for c, a in zip(convs, answers)]


def full(args) -> None:
    from transformers import AutoTokenizer   # noqa: imported only in --full

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.magpie:
        rows = magpie_generate(args.magpie_model, args.magpie)
        texts = [r["messages"][0]["content"] for r in rows]
        keep = minhash_dedup(texts)
        evals = [json.loads(l)["email"] for l in open(args.eval)] if args.eval else []
        bad = set(contaminated(texts, evals, n=13))
        rows = [rows[i] for i in keep if i not in bad]
        with open(out_dir / "magpie.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"magpie: kept {len(rows)} of {args.magpie} after MinHash dedup and decontamination;"
              " judge-filter next (see module 04, section 4.10)")
    if not args.input:
        return
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(prompt, completion):
        p = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        pc = tok.apply_chat_template(prompt + completion, tokenize=True)
        return len(p), len(pc) - len(p)

    raw = [json.loads(l) for l in open(args.input)]
    evals = [json.loads(l) for l in open(args.eval)] if args.eval else []
    out = run_pipeline(raw, evals, count)
    print(json.dumps(out["log"], indent=2))
    write_jsonl(out["rows"], out_dir)
    if args.push_to_hub:
        from datasets import Dataset
        ds = Dataset.from_list(out["rows"]["prompt_completion"])
        split = ds.train_test_split(test_size=0.05, seed=0)
        split.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed to {args.push_to_hub}; now edit its README.md dataset card")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU, inline corpus, asserts (default)")
    mode.add_argument("--full", action="store_true", help="real data/tokenizer; optional vLLM Magpie")
    ap.add_argument("--input", help="raw JSONL with {'email': str, 'label': {...}} rows")
    ap.add_argument("--eval", help="golden eval JSONL (same schema) to decontaminate against")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--magpie", type=int, default=0, help="number of Magpie instructions to generate")
    ap.add_argument("--magpie-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--push-to-hub", default=None)
    args = ap.parse_args()
    if args.full:
        if not args.output_dir:
            ap.error("--full needs --output-dir")
        full(args)
    else:
        smoke(args.output_dir)


if __name__ == "__main__":
    main()
