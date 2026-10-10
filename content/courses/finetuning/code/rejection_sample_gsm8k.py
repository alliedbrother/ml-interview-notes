"""Rejection sampling: build a verified math SFT set from teacher samples on GSM8K.

Module: 04 - Data: Formats, Curation, Synthesis, and Decontamination (lab L4.3).

For every training problem, sample k solutions from a teacher, extract each final
answer, keep only the samples whose answer matches the reference, and report the
keep rate and the length distribution of what survived.

Modes:
  --smoke (default without a GPU): stdlib only. A fake "teacher" returns canned
      solutions with known correctness; asserts the answer extractor on edge cases
      ("#### 1,234", "\\boxed{42}", "-3.50" == "-3.5") and the exact keep rate.
  --full: hardware tier T1 (single 24 GB GPU). vLLM teacher (default
      Qwen/Qwen3-4B-Instruct-2507) on openai/gsm8k, config "main", split "train".
      Versions used when writing (October 2026): vllm>=0.11, datasets>=4.7.
      vllm/datasets are imported inside full() only.

Run (from the repository root):
  python content/courses/finetuning/code/rejection_sample_gsm8k.py --smoke
  python content/courses/finetuning/code/rejection_sample_gsm8k.py --full \
      --teacher Qwen/Qwen3-4B-Instruct-2507 --k 8 --limit 2000 --out gsm8k_rs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from fractions import Fraction

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _clean(num: str) -> str | None:
    num = num.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Fraction(num)
    except (ValueError, ZeroDivisionError):
        return None
    return str(value.numerator) if value.denominator == 1 else str(float(value))


def extract_answer(text: str) -> str | None:
    """Final numeric answer: '#### x' (GSM8K style), then \\boxed{x}, then
    'answer is x', then the last number in the text. Returns a canonical string."""
    m = re.search(r"####\s*(" + _NUM + r")", text)
    if m:
        return _clean(m.group(1))
    boxed = re.findall(r"\\boxed\{\s*(" + _NUM + r")\s*\}", text)
    if boxed:
        return _clean(boxed[-1])
    m = re.search(r"answer is[:\s]*(" + _NUM + r")", text, flags=re.I)
    if m:
        return _clean(m.group(1))
    nums = re.findall(_NUM, text)
    return _clean(nums[-1]) if nums else None


def reference_answer(gsm8k_answer_field: str) -> str:
    ans = extract_answer("#### " + gsm8k_answer_field.split("####")[-1].strip())
    assert ans is not None, gsm8k_answer_field
    return ans


def rejection_sample(problems: list[dict], samples: list[list[str]], max_keep: int = 4) -> tuple[list[dict], dict]:
    kept, solved, total = [], 0, 0
    for prob, outs in zip(problems, samples):
        ref = reference_answer(prob["answer"])
        good, seen = [], set()
        for o in outs:
            total += 1
            if extract_answer(o) == ref and o.strip() not in seen:
                seen.add(o.strip())
                good.append(o)
        solved += bool(good)
        for o in good[:max_keep]:               # cap per problem: easy problems must not dominate
            kept.append({"prompt": [{"role": "user", "content": prob["question"]}],
                         "completion": [{"role": "assistant", "content": o}]})
    correct_samples = sum(
        extract_answer(o) == reference_answer(p["answer"]) for p, outs in zip(problems, samples) for o in outs)
    lengths = [len(r["completion"][0]["content"].split()) for r in kept]
    stats = {
        "problems": len(problems), "samples": total,
        "sample_keep_rate": correct_samples / max(1, total),
        "problems_with_a_kept_solution": solved / max(1, len(problems)),
        "rows_written": len(kept),
        "words_median": statistics.median(lengths) if lengths else 0,
        "words_max": max(lengths) if lengths else 0,
    }
    return kept, stats


# ----------------------------------------------------------------- smoke --
def smoke() -> None:
    cases = {
        "so the total is 1,234 apples.\n#### 1,234": "1234",
        "Therefore the answer is \\boxed{42}.": "42",
        "Step 1 ... \\boxed{7} wait, recheck ... \\boxed{8}": "8",
        "The answer is: -3.50": "-3.5",
        "She pays $18 in total.": "18",
        "It costs 2.0 dollars": "2",
        "no number here": None,
    }
    for text, want in cases.items():
        got = extract_answer(text)
        assert got == want, (text, got, want)

    problems = [
        {"question": "Ali has 3 bags of 4 apples. How many apples?", "answer": "3*4=12\n#### 12"},
        {"question": "A shop sold 1,250 pens on Monday and 750 on Tuesday. How many in total?", "answer": "#### 2,000"},
        {"question": "Hard problem nobody solves.", "answer": "... #### 17"},
    ]
    k = 8
    samples = [
        ["3 times 4 is 12.\n#### 12"] * 3 + ["3+4=7\n#### 7"] * 2
        + ["Each bag has 4, so \\boxed{12}", "The answer is 12", "#### 13"],         # 5 correct (3 identical)
        ["1,250 + 750 = 2,000\n#### 2,000", "\\boxed{2000}", "#### 1,900", "#### 2000.0",
         "The answer is 2500", "#### 2,000", "#### 200", "total 2000"],               # 5 correct
        ["#### 16"] * k,                                                              # 0 correct
    ]
    kept, stats = rejection_sample(problems, samples, max_keep=4)
    print(json.dumps(stats, indent=2))
    assert stats["samples"] == 24
    assert abs(stats["sample_keep_rate"] - 10 / 24) < 1e-12
    assert abs(stats["problems_with_a_kept_solution"] - 2 / 3) < 1e-12
    # problem 1: 5 correct but only 3 unique texts; problem 2: 5 correct and unique, capped at 4
    assert stats["rows_written"] == 3 + 4
    assert all(extract_answer(r["completion"][0]["content"]) in {"12", "2000"} for r in kept)
    print("smoke OK: extractor edge cases, keep rate 10/24, per-problem dedup and cap verified")


# ------------------------------------------------------------------ full --
def full(args) -> None:
    from datasets import load_dataset          # imported here so --smoke never needs them
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    problems = [dict(r) for r in ds]
    llm = LLM(model=args.teacher, seed=args.seed, max_model_len=4096)
    instr = "\n\nSolve step by step. End with a final line of the form '#### <number>'."
    convs = [[{"role": "user", "content": p["question"] + instr}] for p in problems]
    params = SamplingParams(n=args.k, temperature=0.8, top_p=0.95, max_tokens=1024, seed=args.seed)
    outs = llm.chat(convs, params)
    samples = [[c.text for c in o.outputs] for o in outs]
    kept, stats = rejection_sample(problems, samples, max_keep=args.max_keep)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-keep", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gsm8k_rejection_sampled.jsonl")
    args = ap.parse_args()
    full(args) if args.full else smoke()


if __name__ == "__main__":
    main()
