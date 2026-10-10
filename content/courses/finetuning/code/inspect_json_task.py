"""Inspect AI task with a schema-validity scorer for the support-email extraction task.

Purpose
    Score a model's support-email extraction outputs with the course's scoring rules
    (running-example contract, module 01): parse rate, schema validity, per-field exact match
    and all-fields exact match. Schema validity is the scored value (CORRECT/INCORRECT); the
    other metrics ride along in Score.metadata. The scoring functions mirror
    `parse_prediction`, `schema_valid` and `field_match` in fit_or_not_baseline.py.

Module
    LLM Fine-Tuning Deep Dive, module 09 (Evaluating Fine-Tuned Models), lab L9.3.

Hardware tier
    --smoke: CPU, stdlib only. Runs the contract's scorer fixture and, with --data, loads
             module 01's eval.jsonl, validates every gold row and scores the gold answers
             (must be 100%). Does not import inspect-ai.
    --full : T0 (free Colab/Kaggle T4 16 GB) serving Qwen3-0.6B/1.7B through vLLM, or any API
             model Inspect supports. Requires `pip install inspect-ai==0.3.278` (latest release
             on 2026-10-09). Qwen3 hybrid checkpoints are prompted with thinking off via
             `chat_template_kwargs={"enable_thinking": False}`.

Data
    Module 01's frozen eval split: `ft_data/eval.jsonl`, one
    {"id": ..., "email": ..., "gold": {five fields}} object per line. Create it with
    python content/courses/finetuning/code/fit_or_not_baseline.py --data-only

Run commands (from the repository root)
    python content/courses/finetuning/code/inspect_json_task.py --smoke [--data ft_data/eval.jsonl]
    python content/courses/finetuning/code/inspect_json_task.py --full \
        --data ft_data/eval.jsonl --model vllm/Qwen/Qwen3-1.7B
    Then browse per-sample results with:  inspect view
"""

from __future__ import annotations

import argparse
import json
import re

# Running-example contract constants (copied verbatim; scripts stay self-contained).
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


def schema_errors(obj) -> list[str]:
    """Contract rule 3 as a list of reasons (empty list means schema-valid)."""
    if obj is None:
        return ["not a single JSON object"]
    errs = []
    missing = [f for f in FIELDS if f not in obj]
    extra = [k for k in obj if k not in FIELDS]
    if missing:
        errs.append(f"missing {missing}")
    if extra:
        errs.append(f"unexpected {extra}")
    for k in ("customer_name", "order_id", "issue_type", "sentiment"):
        if k in obj and not isinstance(obj[k], str):
            errs.append(f"{k} not a string")
    if "refund_requested" in obj and not isinstance(obj["refund_requested"], bool):
        errs.append("refund_requested not a boolean")
    if isinstance(obj.get("order_id"), str) and not re.match(ORDER_ID_RE, obj["order_id"]):
        errs.append(f"order_id {obj['order_id']!r} does not match {ORDER_ID_RE}")
    if isinstance(obj.get("issue_type"), str) and obj["issue_type"] not in ISSUE_TYPES:
        errs.append(f"issue_type {obj['issue_type']!r} not in enum")
    if isinstance(obj.get("sentiment"), str) and obj["sentiment"] not in SENTIMENTS:
        errs.append(f"sentiment {obj['sentiment']!r} not in enum")
    return errs


def field_match(obj, gold, key) -> bool:
    if obj is None or key not in obj:
        return False
    return type(obj[key]) is type(gold[key]) and obj[key] == gold[key]


def score_text(completion: str, gold: dict) -> dict:
    """Pure scoring logic shared by smoke and full modes."""
    obj = parse_prediction(completion)
    errs = schema_errors(obj)
    fields = {k: field_match(obj, gold, k) for k in FIELDS}
    return {"parsed": obj is not None, "valid": not errs, "errors": errs,
            "fields": fields, "all_fields": all(fields.values())}


def load_rows(path: str) -> list[dict]:
    """Load module 01's eval.jsonl; raise on any row that breaks the contract."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            assert set(row) == {"id", "email", "gold"}, f"bad keys {sorted(row)}"
            assert re.match(r"^(train|eval)-[0-9]{4}$", row["id"]), row["id"]
            assert list(row["gold"]) == FIELDS, f"{row['id']}: gold keys {list(row['gold'])}"
            errs = schema_errors(row["gold"])
            assert not errs, f"{row['id']}: {errs}"
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- smoke
def smoke(data_path: str | None) -> None:
    # The contract's fixture: six predictions against one gold.
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
    s = [score_text(p, gold) for p in preds]
    assert sum(x["parsed"] for x in s) == 5
    assert sum(x["valid"] for x in s) == 2
    assert sum(x["all_fields"] for x in s) == 2
    assert sum(x["fields"]["order_id"] for x in s) == 5
    assert sum(x["fields"]["issue_type"] for x in s) == 4
    # Contract-specific failures the old lab schema let through.
    assert not score_text(perfect.replace("ORD-48213", "48213"), gold)["valid"]
    assert not score_text(perfect.replace('"ORD-48213"', "null"), gold)["valid"]
    assert not score_text("<think>ok</think>\n" + perfect, gold)["parsed"]
    print("contract fixture: parse 5/6, schema 2/6, all-fields 2/6, order_id 5/6, issue_type 4/6")

    if data_path:
        rows = load_rows(data_path)
        res = [score_text(json.dumps(r["gold"]), r["gold"]) for r in rows]
        assert rows and all(x["valid"] and x["all_fields"] for x in res)
        print(f"{data_path}: {len(rows)}/{len(rows)} rows accepted; gold answers score 100%")
    print("smoke OK: scorer logic verified without importing inspect-ai")


# ----------------------------------------------------------------------------- full
def full(data_path: str, model: str, limit: int | None) -> None:
    # Heavy imports live here so --smoke never needs inspect-ai.
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState, generate, system_message

    @scorer(metrics=[accuracy(), stderr()])
    def json_schema_valid():
        async def score(state: TaskState, target: Target) -> Score:
            s = score_text(state.output.completion, json.loads(target.text))
            return Score(
                value=CORRECT if s["valid"] else INCORRECT,
                answer=state.output.completion,
                explanation="; ".join(s["errors"]) or "valid",
                metadata={"parsed": s["parsed"], "all_fields": s["all_fields"], **s["fields"]},
            )
        return score

    samples = [Sample(id=r["id"], input=r["email"], target=json.dumps(r["gold"]))
               for r in load_rows(data_path)]
    task = Task(dataset=samples, solver=[system_message(SYSTEM_PROMPT), generate()],
                scorer=json_schema_valid())
    inspect_eval(task, model=model, limit=limit, temperature=0.0,
                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect AI schema-validity task (module 09)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run scorer fixtures (default)")
    mode.add_argument("--full", action="store_true", help="run the Inspect task against a model")
    ap.add_argument("--data", help="module 01's eval.jsonl ({'id', 'email', 'gold'} rows)")
    ap.add_argument("--model", default="vllm/Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.full:
        if not args.data:
            ap.error("--full needs --data (e.g. ft_data/eval.jsonl)")
        full(args.data, args.model, args.limit)
    else:
        smoke(args.data)


if __name__ == "__main__":
    main()
