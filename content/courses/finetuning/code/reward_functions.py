"""Verifiable reward functions with TRL's GRPO signature, a hackable verifier, and a hack detector.

Purpose
    Reward functions for module 13 written against TRL's reward-function contract
    (as of TRL v1.15.0, October 2026): each function receives keyword arguments
    `prompts`, `completions`, `completion_ids`, `trainer_state`, `log_extra`, `log_metric`
    and every dataset column except `prompt`, and returns `list[float | None]`
    (`None` = "this function does not apply to this sample"). Completions are strings
    for standard datasets and lists of message dicts for conversational datasets.

    Contents:
      * GSM8K answer extraction (\\boxed{}, '####', last-number fallback) and correctness.
      * Format reward (<think>...</think> then a boxed answer).
      * Running task (support-email JSON extraction): schema validity and field accuracy,
        using module 01's parser, enums and exact-match rule unchanged (the dataset row's
        `gold` column is passed to the reward as a dict).
      * Code rewards: a HACKABLE verifier (reward = "the test process exited 0") and a
        robust one (fresh harness owns the asserts, checks a sentinel the program cannot
        forge), plus a detector that flags completions where the two disagree or where
        the code contains known tampering patterns.

    --smoke : asserts extraction edge cases, None-skipping, JSON scoring (including module
              01's six-prediction scoring fixture), and that the
              detector flags every planted hack while passing honest solutions (CPU, ~2 s;
              runs snippets in a subprocess with a timeout; no network).
    --full  : downloads openai/gsm8k (test split) and checks that the gold answer, rewritten
              as "\\boxed{gold}", scores 1.0 for every row (extraction recall). CPU, needs
              `datasets` and network.

Module      content/courses/finetuning/13-rl-with-verifiable-rewards.md
Hardware    CPU for both modes. The sandbox here is a subprocess with a timeout and is NOT
            a security boundary; real code RL needs container/VM isolation (module 15).
Pinned      Python 3.11+, stdlib only for --smoke; datasets>=3 for --full.

Run (from the repository root)
    python content/courses/finetuning/code/reward_functions.py --smoke
    python content/courses/finetuning/code/reward_functions.py --full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

# Running-example schema, copied verbatim from fit_or_not_baseline.py (module 01).
FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
ISSUE_TYPES = ["refund", "shipping", "defect", "account", "other"]
SENTIMENTS = ["positive", "neutral", "negative"]
ORDER_ID_RE = r"^ORD-[0-9]{5}$"
FENCE = re.compile(r"^`{3}(?:json)?[ \t]*\n(.*)\n`{3}$", re.S)


def _text(completion) -> str:
    """Standard datasets give strings; conversational ones give [{'role': 'assistant', 'content': ...}]."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


# ---------------------------------------------------------------- GSM8K

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    s = re.sub(r"\\(?:text|mathrm)\{[^}]*\}", "", s).strip()  # drop units like \text{ dollars}
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Last \\boxed{...} wins; else the number after the last '####'; else None.

    Deliberately NO 'last number in the text' fallback for the reward: that fallback lets a
    model score by ending every response with a guessed number (an answer-parser exploit).
    """
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        return _to_float(boxed[-1])
    hashes = re.findall(r"####\s*(" + _NUM + ")", text)
    if hashes:
        return _to_float(hashes[-1])
    return None


def gsm8k_gold(answer_field: str) -> float | None:
    """openai/gsm8k stores the gold as '<rationale> #### 72'; a bare number is returned as is."""
    return _to_float(str(answer_field).split("####")[-1])


def correctness_reward(prompts, completions, answer=None, task=None, **kwargs) -> list[float | None]:
    out: list[float | None] = []
    for i, c in enumerate(completions):
        if task is not None and task[i] != "gsm8k":
            out.append(None)  # not my sample: TRL skips this function for it
            continue
        pred = extract_answer(_text(c))
        gold = gsm8k_gold(answer[i]) if answer is not None else None  # raw column or bare number
        out.append(1.0 if pred is not None and gold is not None and math.isclose(pred, gold, abs_tol=1e-6) else 0.0)
    return out


_FORMAT = re.compile(r"^\s*<think>(?P<body>.+?)</think>\s*(?P<tail>.*)$", re.S)


def format_reward(completions, task=None, **kwargs) -> list[float | None]:
    """0.2 for '<think> non-trivial reasoning </think> ... \\boxed{x}' with exactly one think block."""
    out: list[float | None] = []
    for i, c in enumerate(completions):
        if task is not None and task[i] != "gsm8k":
            out.append(None)
            continue
        t = _text(c)
        m = _FORMAT.match(t)
        ok = (m is not None and t.count("<think>") == 1 and len(m.group("body").split()) >= 5
              and "\\boxed{" in m.group("tail"))
        out.append(0.2 if ok else 0.0)
    return out


# ---------------------------------------------------------------- running task: JSON extraction

def parse_prediction(text):
    """Module 01's only normalisation: strip, unwrap one surrounding ```json fence, json.loads."""
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


def schema_valid(obj) -> bool:
    if obj is None or set(obj) != set(FIELDS):
        return False
    if not isinstance(obj["refund_requested"], bool):
        return False
    if not all(isinstance(obj[k], str) for k in FIELDS if k != "refund_requested"):
        return False
    return (re.match(ORDER_ID_RE, obj["order_id"]) is not None
            and obj["issue_type"] in ISSUE_TYPES and obj["sentiment"] in SENTIMENTS)


def field_match(obj, gold, key) -> bool:
    """Exact match with the same JSON type ("true" != true, 1 != true); no case folding."""
    if obj is None or key not in obj:
        return False
    return type(obj[key]) is type(gold[key]) and obj[key] == gold[key]


def schema_reward(completions, task=None, **kwargs) -> list[float | None]:
    """1.0 iff the output is schema-valid under module 01's scorer."""
    out: list[float | None] = []
    for i, c in enumerate(completions):
        if task is not None and task[i] != "email_json":
            out.append(None)
            continue
        out.append(1.0 if schema_valid(parse_prediction(_text(c))) else 0.0)
    return out


def field_accuracy_reward(completions, gold=None, task=None, **kwargs) -> list[float | None]:
    """Fraction of the five fields that exactly match the row's `gold` dict (0 if unparseable)."""
    out: list[float | None] = []
    for i, c in enumerate(completions):
        if task is not None and task[i] != "email_json":
            out.append(None)
            continue
        obj = parse_prediction(_text(c))
        out.append(sum(field_match(obj, gold[i], f) for f in FIELDS) / len(FIELDS))
    return out


# ---------------------------------------------------------------- code rewards

def _run(code: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        return subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                              text=True, timeout=timeout)


def hackable_code_reward(completions, tests, **kwargs) -> list[float]:
    """BAD verifier: concatenates solution + tests and rewards exit code 0.

    Exploitable by sys.exit(0)/os._exit(0) before the tests run, by redefining `assert`-using
    helpers, or by catching AssertionError globally.
    """
    out = []
    for c, t in zip(completions, tests):
        try:
            r = _run(_text(c) + "\n" + t)
            out.append(1.0 if r.returncode == 0 else 0.0)
        except subprocess.TimeoutExpired:
            out.append(0.0)
    return out


def robust_code_reward(completions, tests, **kwargs) -> list[float]:
    """Better verifier: the harness owns control flow and must print an unforgeable sentinel
    AFTER the tests pass. The solution runs inside exec() with SystemExit caught by the harness.
    (Still not a security boundary: use containers for real RL.)"""
    out = []
    for c, t in zip(completions, tests):
        sentinel = os.urandom(8).hex()
        harness = (
            "import sys\n"
            "ns = {}\n"
            f"src = {(_text(c))!r}\n"
            "try:\n"
            "    exec(compile(src, 'solution', 'exec'), ns)\n"
            "except SystemExit:\n"
            "    print('EXITED'); raise SystemExit(1)\n"
            f"tests = {t!r}\n"
            "exec(compile(tests, 'tests', 'exec'), ns)\n"
            f"print({sentinel!r})\n"
        )
        try:
            r = _run(harness)
            out.append(1.0 if r.returncode == 0 and r.stdout.strip().endswith(sentinel) else 0.0)
        except subprocess.TimeoutExpired:
            out.append(0.0)
    return out


TAMPER_PATTERNS = [
    r"\bsys\.exit\s*\(", r"\bos\._exit\s*\(", r"\bexit\s*\(", r"\bquit\s*\(",
    r"except\s+AssertionError", r"except\s*:\s*pass", r"__builtins__", r"open\([^)]*test",
    r"unittest\.skip", r"pytest\.skip", r"def\s+assert",
]


def detect_hacks(completions, tests) -> list[dict]:
    """Flag a completion if the hackable and robust verifiers disagree (reward-quality
    divergence on a held-out verifier) or if it matches a tampering pattern."""
    hack_r = hackable_code_reward(completions, tests)
    good_r = robust_code_reward(completions, tests)
    flags = []
    for c, h, g in zip(completions, hack_r, good_r):
        pats = [p for p in TAMPER_PATTERNS if re.search(p, _text(c))]
        flags.append({"hackable": h, "robust": g, "patterns": pats,
                      "flagged": bool(pats) or (h == 1.0 and g == 0.0)})
    return flags


# ---------------------------------------------------------------- smoke

def smoke():
    # 1. extraction edge cases
    cases = {
        "so the total is \\boxed{1,234}": 1234.0,
        "first \\boxed{3} then corrected \\boxed{4}": 4.0,           # last box wins
        "#### 72": 72.0,
        "The answer is $\\boxed{18}$.": 18.0,
        "\\boxed{-5}": -5.0,
        "\\boxed{2.50}": 2.5,
        "\\boxed{\\text{ dollars}12}": 12.0,
        "I think it is 42": None,                                     # no last-number fallback
        "\\boxed{x+1}": None,
    }
    for text, want in cases.items():
        got = extract_answer(text)
        assert got == want, (text, got, want)
    assert gsm8k_gold("Natalia sold 48/2 = 24 clips.\n#### 72") == 72.0

    # 2. TRL-style batch with mixed tasks: None means "skip"
    completions = [
        [{"role": "assistant", "content": "<think>48 / 2 = 24, and 48 + 24 = 72 clips</think> \\boxed{72}"}],
        [{"role": "assistant", "content": "\\boxed{70}"}],
        [{"role": "assistant", "content": json.dumps({"customer_name": "Ana Ruiz", "order_id": "ORD-00017",
                                                      "issue_type": "refund", "refund_requested": True,
                                                      "sentiment": "negative"})}],
    ]
    task = ["gsm8k", "gsm8k", "email_json"]
    answer = ["Natalia sold 48/2 = 24 clips.\n#### 72", "72", ""]   # raw openai/gsm8k column, or bare
    gold = [None, None, {"customer_name": "Ana Ruiz", "order_id": "ORD-00017", "issue_type": "refund",
                         "refund_requested": True, "sentiment": "neutral"}]
    kw = dict(prompts=[None] * 3, completions=completions, completion_ids=[[0]] * 3, task=task)
    assert correctness_reward(answer=answer, **kw) == [1.0, 0.0, None]
    assert format_reward(**kw) == [0.2, 0.0, None]
    assert schema_reward(**kw) == [None, None, 1.0]
    assert field_accuracy_reward(gold=gold, **kw) == [None, None, 0.8]

    # 2b. module 01's scoring fixture: parse 5/6, schema 2/6, all-fields 2/6, order_id 5/6, issue_type 4/6
    g = {"customer_name": "Maya Patel", "order_id": "ORD-48213", "issue_type": "refund",
         "refund_requested": True, "sentiment": "negative"}
    perfect, fence = json.dumps(g), "`" * 3
    preds = [perfect, perfect.replace('"refund",', '"billing",'), perfect[:-1] + ",}",
             f"{fence}json\n{perfect}\n{fence}",
             json.dumps({k: v for k, v in g.items() if k != "sentiment"}), perfect.replace("true", '"true"')]
    parsed = [parse_prediction(p) for p in preds]
    assert sum(o is not None for o in parsed) == 5
    assert schema_reward(preds) == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    acc = field_accuracy_reward(preds, gold=[g] * 6)
    assert sum(a == 1.0 for a in acc) == 2
    assert sum(field_match(o, g, "order_id") for o in parsed) == 5
    assert sum(field_match(o, g, "issue_type") for o in parsed) == 4

    # 3. planted code hacks vs honest solutions
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    honest = "def add(a, b):\n    return a + b\n"
    wrong = "def add(a, b):\n    return a - b\n"
    hack_exit = "import sys\ndef add(a, b):\n    return 0\nsys.exit(0)\n"
    hack_osexit = "import os\ndef add(a, b):\n    return 0\nos._exit(0)\n"
    comps = [honest, wrong, hack_exit, hack_osexit]
    tt = [tests] * len(comps)
    hack_r = hackable_code_reward(comps, tt)
    good_r = robust_code_reward(comps, tt)
    assert hack_r == [1.0, 0.0, 1.0, 1.0], hack_r     # the hackable verifier pays the hacks
    assert good_r == [1.0, 0.0, 0.0, 0.0], good_r     # the robust one does not
    flags = detect_hacks(comps, tt)
    assert [f["flagged"] for f in flags] == [False, False, True, True], flags
    print("extraction cases:", len(cases), "ok")
    print("hackable rewards:", hack_r, " robust rewards:", good_r)
    print("detector flags:", [f["flagged"] for f in flags])
    print("smoke passed")


def full():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    golds = [gsm8k_gold(a) for a in ds["answer"]]
    comps = [f"<think>reasoning omitted for this check</think> \\boxed{{{g:g}}}" for g in golds]
    r = correctness_reward(prompts=[None] * len(ds), completions=comps, answer=ds["answer"])  # raw column
    assert all(x == 1.0 for x in r), sum(1 for x in r if x != 1.0)
    print(f"gold-as-completion recall: {sum(r)}/{len(r)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    full() if args.full else smoke()


if __name__ == "__main__":
    main()
