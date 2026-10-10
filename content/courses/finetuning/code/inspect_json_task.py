"""Inspect AI task with a JSON-schema-validity scorer for the support-email extraction task.

Purpose
    Score a model's support-email extraction outputs on two things Inspect's built-in scorers
    do not check: (1) is the output a single JSON object that satisfies the course schema, and
    (2) how many fields match the golden label. Validity is the scored value (CORRECT/INCORRECT);
    field accuracy rides along in Score.metadata.

Module
    LLM Fine-Tuning Deep Dive, module 09 (Evaluating Fine-Tuned Models), lab L9.3.

Hardware tier
    --smoke: CPU, stdlib only. Runs the pure scoring logic on inline fixtures; does not import
             inspect-ai.
    --full : T0 (free Colab/Kaggle T4 16 GB) serving Qwen3-0.6B/1.7B through vLLM, or any API
             model Inspect supports. Requires `pip install inspect-ai` (tested against the
             0.3.2xx series; 0.3.278 was the latest release on 2026-10-09).

Schema
    The enums below are this lab's version of the module-01 schema. If your module-01 eval set
    uses different enum values, edit ISSUE_TYPES / SENTIMENTS before running.

Run commands (from the repository root)
    python content/courses/finetuning/code/inspect_json_task.py --smoke
    python content/courses/finetuning/code/inspect_json_task.py --full \
        --data golden_50.jsonl --model vllm/Qwen/Qwen3-1.7B
    (golden_50.jsonl: one {"email": "...", "label": {...}} object per line.)
    Then browse per-sample results with:  inspect view
"""

from __future__ import annotations

import argparse
import json

ISSUE_TYPES = {"late_delivery", "damaged_item", "wrong_item", "billing", "account", "other"}
SENTIMENTS = {"positive", "neutral", "negative"}
FIELDS = ("customer_name", "order_id", "issue_type", "refund_requested", "sentiment")

PROMPT = (
    "Extract the following fields from the support email and reply with ONE JSON object and "
    "nothing else: customer_name (string), order_id (string or null), issue_type (one of "
    + ", ".join(sorted(ISSUE_TYPES))
    + "), refund_requested (true/false), sentiment (one of "
    + ", ".join(sorted(SENTIMENTS))
    + ").\n\nEmail:\n"
)


def extract_json(text: str):
    """Return the parsed object if the reply (after an optional </think> block) is one JSON object."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("`" * 3):  # tolerate a single fenced block, nothing else
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def schema_errors(obj) -> list[str]:
    """Hand-rolled validator for the course schema (no jsonschema dependency)."""
    if obj is None:
        return ["not a single JSON object"]
    errs = []
    missing = [f for f in FIELDS if f not in obj]
    extra = [k for k in obj if k not in FIELDS]
    if missing:
        errs.append(f"missing {missing}")
    if extra:
        errs.append(f"unexpected {extra}")
    if "customer_name" in obj and not isinstance(obj["customer_name"], str):
        errs.append("customer_name not a string")
    if "order_id" in obj and not (obj["order_id"] is None or isinstance(obj["order_id"], str)):
        errs.append("order_id not string/null")
    if "issue_type" in obj and obj["issue_type"] not in ISSUE_TYPES:
        errs.append(f"issue_type {obj['issue_type']!r} not in enum")
    if "refund_requested" in obj and not isinstance(obj["refund_requested"], bool):
        errs.append("refund_requested not a boolean")
    if "sentiment" in obj and obj["sentiment"] not in SENTIMENTS:
        errs.append(f"sentiment {obj['sentiment']!r} not in enum")
    return errs


def score_text(completion: str, label: dict) -> dict:
    """Pure scoring logic shared by smoke and full modes."""
    obj = extract_json(completion)
    errs = schema_errors(obj)
    matched = 0 if obj is None else sum(obj.get(f) == label.get(f) for f in FIELDS)
    return {"valid": not errs, "errors": errs, "field_acc": matched / len(FIELDS)}


# ----------------------------------------------------------------------------- smoke
def smoke() -> None:
    label = {"customer_name": "Dana Ruiz", "order_id": "A-1042", "issue_type": "damaged_item",
             "refund_requested": True, "sentiment": "negative"}
    good = json.dumps(label)
    thinking = "<think>The box arrived crushed...</think>\n" + good
    fenced = "`" * 3 + "json\n" + good + "\n" + "`" * 3
    wrong_enum = good.replace("damaged_item", "broken")
    string_bool = good.replace("true", '"yes"')
    prose = "Sure! Here is the JSON: " + good
    partial = json.dumps({**label, "sentiment": "neutral", "order_id": None})

    cases = [
        (good, True, 1.0), (thinking, True, 1.0), (fenced, True, 1.0),
        (wrong_enum, False, 0.8), (string_bool, False, 0.8), (prose, False, 0.0),
        (partial, True, 0.6),
    ]
    for text, valid, acc in cases:
        s = score_text(text, label)
        assert s["valid"] is valid, (text, s)
        assert abs(s["field_acc"] - acc) < 1e-9, (text, s)
    validity = sum(score_text(t, label)["valid"] for t, _, _ in cases) / len(cases)
    assert abs(validity - 4 / 7) < 1e-9
    print(f"schema validity on fixtures = {validity:.3f} (4/7); field accuracy checks passed")
    print("smoke OK: scorer logic verified without importing inspect-ai")


# ----------------------------------------------------------------------------- full
def full(data_path: str, model: str, limit: int | None) -> None:
    # Heavy imports live here so --smoke never needs inspect-ai.
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState, generate

    @scorer(metrics=[accuracy(), stderr()])
    def json_schema_valid():
        async def score(state: TaskState, target: Target) -> Score:
            label = json.loads(target.text)
            s = score_text(state.output.completion, label)
            return Score(
                value=CORRECT if s["valid"] else INCORRECT,
                answer=state.output.completion,
                explanation="; ".join(s["errors"]) or "valid",
                metadata={"field_acc": s["field_acc"]},
            )
        return score

    samples = []
    with open(data_path) as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            samples.append(Sample(id=i, input=PROMPT + row["email"], target=json.dumps(row["label"])))

    task = Task(dataset=samples, solver=generate(), scorer=json_schema_valid())
    inspect_eval(task, model=model, limit=limit, temperature=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect AI JSON-schema-validity task (module 09)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run scorer fixtures (default)")
    mode.add_argument("--full", action="store_true", help="run the Inspect task against a model")
    ap.add_argument("--data", help="JSONL with {'email', 'label'} rows")
    ap.add_argument("--model", default="vllm/Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.full:
        if not args.data:
            ap.error("--full needs --data")
        full(args.data, args.model, args.limit)
    else:
        smoke()


if __name__ == "__main__":
    main()
