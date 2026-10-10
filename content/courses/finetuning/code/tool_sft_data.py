"""Tool-calling SFT data: render, mask, validate, score.

Purpose
    Lab L15.1 of module 15 (Agentic and Tool-Use Fine-Tuning). Renders multi-turn
    tool trajectories with a hand-written Hermes/ChatML renderer (the format Qwen3
    uses), builds assistant-only loss masks, validates predicted calls against JSON
    schemas, and computes AST-match scores the way BFCL-style evaluators do.

Module
    content/courses/finetuning/15-agentic-and-tool-use-fine-tuning.md

Modes
    --smoke  CPU, stdlib only, under a second. Asserts on built-in fixtures.
    --full   Tier T0/T1 (free T4 16 GB with QLoRA, or one 24 GB GPU with LoRA).
             Converts Salesforce/xlam-function-calling-60k (gated: accept the
             terms on the dataset page and run `hf auth login` first) into TRL's
             tool-calling format (`messages` + `tools`), adds no-call examples,
             trains Qwen/Qwen3-4B with LoRA and assistant_only_loss=True, then
             scores a held-out slice with the local AST evaluator.

Pinned versions for --full (as of October 2026)
    trl>=1.15, transformers>=5.2, peft>=0.21, datasets>=4.7, torch>=2.8

Run (from the repository root)
    python content/courses/finetuning/code/tool_sft_data.py --smoke
    python content/courses/finetuning/code/tool_sft_data.py --full --max-steps 300
"""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction

IGNORE = -100

# --------------------------------------------------------------------------
# Rendering: Hermes-style tool calls inside ChatML, as in the Qwen3 template.
# Each message renders to a list of (text, trained) segments. Concatenating
# the texts gives the training string; the flags give the loss mask.
# --------------------------------------------------------------------------


def render_segments(messages, tools):
    segs = []
    i = 0
    if tools:
        sys_text = messages[0]["content"] + "\n\n" if messages and messages[0]["role"] == "system" else ""
        tool_lines = "\n".join(json.dumps(t, sort_keys=True) for t in tools)
        segs.append(("<|im_start|>system\n" + sys_text + "# Tools\n<tools>\n" + tool_lines + "\n</tools><|im_end|>\n", False))
        if messages and messages[0]["role"] == "system":
            i = 1
    while i < len(messages):
        m = messages[i]
        if m["role"] in ("system", "user"):
            segs.append((f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", False))
        elif m["role"] == "assistant":
            segs.append(("<|im_start|>assistant\n", False))  # prompt cue, not generated
            body = m.get("content") or ""
            for k, call in enumerate(m.get("tool_calls") or []):
                fn = call.get("function", call)
                if body or k > 0:
                    body += "\n"
                body += '<tool_call>\n{"name": "' + fn["name"] + '", "arguments": ' + json.dumps(fn["arguments"]) + "}\n</tool_call>"
            segs.append((body + "<|im_end|>", True))  # generated, including the stop token
            segs.append(("\n", False))
        elif m["role"] == "tool":
            # Qwen3 groups consecutive tool results into one user turn.
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            text = ("<|im_start|>user" if first else "") + "\n<tool_response>\n" + m["content"] + "\n</tool_response>"
            if last:
                text += "<|im_end|>\n"
            segs.append((text, False))
        else:
            raise ValueError(f"unknown role {m['role']}")
        i += 1
    return segs


def render(messages, tools):
    return "".join(t for t, _ in render_segments(messages, tools))


def char_mask(messages, tools):
    """Per-character trained flags; a real pipeline maps these to tokens via offsets."""
    out = []
    for text, trained in render_segments(messages, tools):
        out.extend([trained] * len(text))
    return out


SPECIAL = re.compile(r"(<\|im_start\|>|<\|im_end\|>|</?tool_call>|</?tool_response>|</?tools>)")


def toy_tokenize(text):
    """Special tokens are single tokens; everything else splits on whitespace."""
    return [t for piece in SPECIAL.split(text) for t in (piece.split() if not SPECIAL.fullmatch(piece) else [piece])]


def tokens_and_labels(messages, tools):
    """Toy tokenizer applied per segment. Labels are the token or IGNORE."""
    toks, labels = [], []
    for text, trained in render_segments(messages, tools):
        for tok in toy_tokenize(text):
            toks.append(tok)
            labels.append(tok if trained else IGNORE)
    return toks, labels


# --------------------------------------------------------------------------
# Schema validation and AST match (BFCL-style rules, simplified).
# --------------------------------------------------------------------------


def _type_ok(value, json_type):
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":  # int accepted for number; numeric strings and bools rejected
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


def validate_call(call, schemas):
    """Return (valid, reason). schemas: dict name -> JSON-schema 'function' dict."""
    if call["name"] not in schemas:
        return False, "hallucinated_tool"
    params = schemas[call["name"]]["parameters"]
    props, required = params.get("properties", {}), params.get("required", [])
    args = call.get("arguments", {})
    for r in required:
        if r not in args:
            return False, "missing_required"
    for k, v in args.items():
        if k not in props:
            return False, "unknown_argument"
        if not _type_ok(v, props[k].get("type")):
            return False, "type_error"
    return True, "ok"


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    return v


def ast_match(call, gold, schemas):
    ok, _ = validate_call(call, schemas)
    if not ok or call["name"] != gold["name"]:
        return False
    a, g = call.get("arguments", {}), gold.get("arguments", {})
    if set(a) != set(g):
        return False
    return all(_norm(a[k]) == _norm(g[k]) for k in g)


def score(preds, golds, schemas):
    n = len(preds)
    valid = sum(validate_call(p, schemas)[0] for p in preds)
    correct = sum(ast_match(p, g, schemas) for p, g in zip(preds, golds))
    halluc = sum(p["name"] not in schemas for p in preds)
    return {"validity": Fraction(valid, n), "ast_accuracy": Fraction(correct, n), "hallucinated_rate": Fraction(halluc, n)}


# --------------------------------------------------------------------------
# xLAM -> TRL conversion (used by --full; also unit-tested in --smoke).
# --------------------------------------------------------------------------

_XLAM_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
               "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
               "list": "array", "dict": "object"}


def xlam_tool_to_schema(tool):
    props, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        raw = str(spec.get("type", "string"))
        optional = "optional" in raw.lower()
        base = raw.split(",")[0].strip().lower()
        base = "array" if base.startswith("list") else "object" if base.startswith("dict") else _XLAM_TYPES.get(base, "string")
        props[name] = {"type": base, "description": spec.get("description", "")}
        if spec.get("required", not optional) and not optional and "default" not in spec:
            required.append(name)
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def xlam_row_to_trl(row):
    tools = [xlam_tool_to_schema(t) for t in json.loads(row["tools"])]
    answers = json.loads(row["answers"])
    messages = [
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}} for a in answers]},
    ]
    return {"messages": messages, "tools": tools}


# --------------------------------------------------------------------------
# Fixtures and smoke test
# --------------------------------------------------------------------------

SCHEMAS = {
    "get_order_status": {"name": "get_order_status", "description": "Look up an order.",
                         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    "refund": {"name": "refund", "description": "Refund part of an order.",
               "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                              "required": ["order_id", "amount"]}},
}
TOOLS = [{"type": "function", "function": s} for s in SCHEMAS.values()]

TRAJECTORY = [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "Where is order A1?"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_order_status", "arguments": {"order_id": "A1"}}}]},
    {"role": "tool", "name": "get_order_status", "content": "shipped"},
    {"role": "assistant", "content": "Order A1 has shipped."},
]


def smoke():
    toks, labels = tokens_and_labels(TRAJECTORY, TOOLS)
    trained = [t for t, l in zip(toks, labels) if l != IGNORE]
    assert trained[0] == "<tool_call>" and trained[-1] == "<|im_end|>", trained
    assert "shipped" not in trained, "tool output must be masked"
    assert "<tool_response>" not in trained and "Where" not in trained
    assert len(trained) == 13, len(trained)
    # Prefix preservation: rendering a prefix of the conversation is a string prefix.
    for k in range(2, len(TRAJECTORY) + 1):
        assert render(TRAJECTORY, TOOLS).startswith(render(TRAJECTORY[:k], TOOLS))

    gold = {"name": "refund", "arguments": {"order_id": "A1", "amount": 12}}
    preds = [
        {"name": "refund", "arguments": {"order_id": "A1", "amount": 12.0}},
        {"name": "get_order_status", "arguments": {"order_id": "A1"}},
        {"name": "refund", "arguments": {"order_id": "A1"}},
        {"name": "refund", "arguments": {"order_id": "a1 ", "amount": 12}},
        {"name": "refund", "arguments": {"order_id": "A1", "amount": "12.5"}},
        {"name": "cancel_order", "arguments": {"order_id": "A1"}},
    ]
    s = score(preds, [gold] * 6, SCHEMAS)
    assert s == {"validity": Fraction(1, 2), "ast_accuracy": Fraction(1, 3), "hallucinated_rate": Fraction(1, 6)}, s

    row = {"query": "Sum multiples of 3 below 10.",
           "tools": json.dumps([{"name": "sum_multiples", "description": "d",
                                 "parameters": {"limit": {"type": "int", "description": "", "required": True},
                                                "step": {"type": "int, optional", "description": "", "default": 3}}}]),
           "answers": json.dumps([{"name": "sum_multiples", "arguments": {"limit": 10}}])}
    ex = xlam_row_to_trl(row)
    p = ex["tools"][0]["function"]["parameters"]
    assert p["required"] == ["limit"] and p["properties"]["step"]["type"] == "integer"
    assert ex["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"limit": 10}
    print(f"smoke ok: {len(trained)} trained tokens; validity={s['validity']} ast={s['ast_accuracy']} halluc={s['hallucinated_rate']}")


# --------------------------------------------------------------------------
# Full mode (GPU). Imports stay inside so --smoke needs only the stdlib.
# --------------------------------------------------------------------------

NO_CALL_PROMPTS = ["Thanks, that's all!", "What does your refund policy cover in general?", "hi there"]


def full(args):
    import random

    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    raw = raw.shuffle(seed=0)
    held_out, train_rows = raw.select(range(args.eval_n)), raw.select(range(args.eval_n, args.eval_n + args.train_n))
    data = [xlam_row_to_trl(r) for r in train_rows]
    # Irrelevance / no-call data: xLAM has none, so add some (about 10 percent).
    rng = random.Random(0)
    for _ in range(len(data) // 10):
        ex = rng.choice(data)
        data.append({"messages": [{"role": "user", "content": rng.choice(NO_CALL_PROMPTS)},
                                  {"role": "assistant", "content": "I can help with that without calling a tool."}],
                     "tools": ex["tools"]})
    train = Dataset.from_list(data, on_mixed_types="use_json")

    cfg = SFTConfig(output_dir=args.output_dir, assistant_only_loss=True, max_length=2048,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
                    max_steps=args.max_steps, logging_steps=10, bf16=args.bf16, fp16=not args.bf16,
                    model_init_kwargs={"dtype": "bfloat16" if args.bf16 else "float16"})
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=train,
                         peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(args.output_dir)

    # Local AST evaluation on held-out rows (a reduced stand-in for the BFCL harness).
    import torch
    tok, model = trainer.processing_class, trainer.model.eval()
    per = []
    for r in held_out:
        ex = xlam_row_to_trl(r)
        schemas = {t["function"]["name"]: t["function"] for t in ex["tools"]}
        prompt = tok.apply_chat_template(ex["messages"][:1], tools=ex["tools"], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
        try:
            call = json.loads(text.split("<tool_call>")[1].split("</tool_call>")[0])
        except (IndexError, json.JSONDecodeError):
            call = {"name": "<no_call_or_unparseable>", "arguments": {}}
        gold = json.loads(r["answers"])[0]  # first gold call only: a reduced evaluator
        per.append(score([call], [gold], schemas))
    agg = {k: sum(x[k] for x in per) / len(per) for k in per[0]}
    print({k: float(v) for k, v in agg.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--bf16", action="store_true", help="use on Ampere+; leave off on a T4")
    ap.add_argument("--output-dir", default="qwen3-4b-xlam-lora")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
