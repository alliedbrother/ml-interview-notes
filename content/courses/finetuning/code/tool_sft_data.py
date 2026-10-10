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
    --full   Tier T0 (free T4 16 GB: fp16 LoRA, or add --qlora for 4-bit NF4
             QLoRA if you run out of memory) or T1 (one 24 GB GPU, add --bf16).
             Converts Salesforce/xlam-function-calling-60k (gated: accept the
             terms on the dataset page and run `hf auth login` first) into TRL's
             tool-calling format (`messages` + `tools`), adds no-call examples
             (--no-call-frac, default 0.1; 0 disables them), trains Qwen/Qwen3-4B
             with LoRA and assistant_only_loss=True, then scores a held-out slice
             with the local AST evaluator. The held-out set is xLAM rows (a call
             is required) plus no-call prompts (a call is wrong), so it reports
             validity, AST accuracy, hallucinated-tool rate, irrelevance accuracy
             and call/no-call accuracy. --eval-base scores the base model (adapter
             disabled) on the same set before training. It also prints one
             decoded training example with the trained spans in [[...]].

Pinned versions for --full (as of October 2026)
    trl>=1.15, transformers>=5.2, peft>=0.21, datasets>=4.7, torch>=2.8,
    bitsandbytes>=0.45 (only for --qlora)

Run (from the repository root)
    python content/courses/finetuning/code/tool_sft_data.py --smoke
    python content/courses/finetuning/code/tool_sft_data.py --full --max-steps 300 --eval-base
    python content/courses/finetuning/code/tool_sft_data.py --full --max-steps 300 --eval-base --no-call-frac 0
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
    # A missing or unparseable call is invalid but is not a call to an invented tool.
    halluc = sum(p["name"] not in schemas and p["name"] not in ("<no_call>", "<unparseable>") for p in preds)
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


def parse_first_call(text):
    """First Hermes <tool_call> in generated text, None if the model made no call."""
    if "<tool_call>" not in text:
        return None
    try:
        return json.loads(text.split("<tool_call>")[1].split("</tool_call>")[0])
    except (IndexError, json.JSONDecodeError):
        return {"name": "<unparseable>", "arguments": {}}


def decision_accuracy(called, should_call):
    """Fraction of turns where the call/no-call decision was right."""
    return Fraction(sum(c == s for c, s in zip(called, should_call)), len(called))


def add_no_call_rows(data, frac, prompts, rng):
    """Append round(frac * len(data)) no-call rows that reuse a random row's tool list."""
    extra = []
    for _ in range(round(frac * len(data))):
        ex = rng.choice(data)
        extra.append({"messages": [{"role": "user", "content": rng.choice(prompts)},
                                   {"role": "assistant", "content": "No tool is needed for that. How else can I help?"}],
                      "tools": ex["tools"]})
    return data + extra


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

    # Held-out decision scoring and the no-call ablation switch.
    assert parse_first_call('<tool_call>\n{"name": "refund", "arguments": {}}\n</tool_call>')["name"] == "refund"
    assert parse_first_call("You're welcome!") is None
    assert parse_first_call("<tool_call>{bad json")["name"] == "<unparseable>"
    assert decision_accuracy([True, True, False, True], [True, True, False, False]) == Fraction(3, 4)
    import random
    base = [ex] * 20
    assert len(add_no_call_rows(base, 0.1, NO_CALL_PROMPTS, random.Random(0))) == 22
    assert add_no_call_rows(base, 0.0, NO_CALL_PROMPTS, random.Random(0)) == base
    print(f"smoke ok: {len(trained)} trained tokens; validity={s['validity']} ast={s['ast_accuracy']} halluc={s['hallucinated_rate']}")


# --------------------------------------------------------------------------
# Full mode (GPU). Imports stay inside so --smoke needs only the stdlib.
# --------------------------------------------------------------------------

NO_CALL_PROMPTS = ["Thanks, that's all!", "What does your refund policy cover in general?", "hi there",
                   "Can you explain what you are able to do?", "Never mind, I figured it out."]
# Held out from training: scored as irrelevance (the right output is no call).
NO_CALL_EVAL_PROMPTS = ["Good morning!", "Who wrote Pride and Prejudice?", "Thanks a lot, bye.",
                        "What is 2 plus 2?", "Tell me a short joke.", "How are you today?",
                        "What's the capital of France?", "Ok, that answers my question.",
                        "Can you say that again more politely?", "I have no more questions."]


def evaluate(model, tok, held_out, no_call_eval):
    """Score call rows (a call is required) and no-call rows (a call is wrong)."""
    import torch

    called, should_call, per = [], [], []
    for ex, gold in held_out + [(e, None) for e in no_call_eval]:
        schemas = {t["function"]["name"]: t["function"] for t in ex["tools"]}
        prompt = tok.apply_chat_template(ex["messages"][:1], tools=ex["tools"], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        call = parse_first_call(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False))
        called.append(call is not None)
        should_call.append(gold is not None)
        if gold is not None:  # first gold call only: a reduced evaluator
            per.append(score([call or {"name": "<no_call>", "arguments": {}}], [gold], schemas))
    res = {k: float(sum(x[k] for x in per) / len(per)) for k in per[0]}
    n_call = sum(should_call)
    res["irrelevance_accuracy"] = float(Fraction(sum(not c for c in called[n_call:]), len(called) - n_call))
    res["call_no_call_accuracy"] = float(decision_accuracy(called, should_call))
    return res


def full(args):
    import random

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    raw = raw.shuffle(seed=0)
    eval_rows, train_rows = raw.select(range(args.eval_n)), raw.select(range(args.eval_n, args.eval_n + args.train_n))
    data = [xlam_row_to_trl(r) for r in train_rows]
    # Irrelevance / no-call data: xLAM has none. --no-call-frac 0 is the ablation.
    rng = random.Random(0)
    data = add_no_call_rows(data, args.no_call_frac, NO_CALL_PROMPTS, rng)
    train = Dataset.from_list(data, on_mixed_types="use_json")

    held_out = [(xlam_row_to_trl(r), json.loads(r["answers"])[0]) for r in eval_rows]
    no_call_eval = [{"messages": [{"role": "user", "content": p}], "tools": held_out[i % len(held_out)][0]["tools"]}
                    for i, p in enumerate(NO_CALL_EVAL_PROMPTS)]

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    init = {"dtype": dtype}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        init["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    cfg = SFTConfig(output_dir=args.output_dir, assistant_only_loss=True, max_length=2048,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
                    max_steps=args.max_steps, logging_steps=10, bf16=args.bf16, fp16=not args.bf16,
                    model_init_kwargs=init)
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=train,
                         peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    tok = trainer.processing_class

    # One decoded training example with its mask: trained spans in [[...]], masked text plain.
    row = trainer.train_dataset[0]
    shown, cur, cur_m = [], [], None
    mask = row["assistant_masks"] if "assistant_masks" in row else [l != IGNORE for l in row["labels"]]
    for t, m in zip(row["input_ids"], mask):
        if m != cur_m and cur:
            shown.append(f"[[{tok.decode(cur)}]]" if cur_m else tok.decode(cur)); cur = []
        cur.append(t); cur_m = m
    shown.append(f"[[{tok.decode(cur)}]]" if cur_m else tok.decode(cur))
    print("decoded example with mask:\n" + "".join(shown))

    if args.eval_base:  # the base model is the same weights with the LoRA adapter switched off
        trainer.model.eval()
        with trainer.model.disable_adapter():
            print("base:", evaluate(trainer.model, tok, held_out, no_call_eval))
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"trained (no_call_frac={args.no_call_frac}):", evaluate(trainer.model.eval(), tok, held_out, no_call_eval))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--bf16", action="store_true", help="use on Ampere+; leave off on a T4")
    ap.add_argument("--qlora", action="store_true", help="load the base in 4-bit NF4 (needs bitsandbytes)")
    ap.add_argument("--no-call-frac", type=float, default=0.1, help="no-call rows per training row; 0 disables")
    ap.add_argument("--eval-base", action="store_true", help="score the base model before training")
    ap.add_argument("--output-dir", default="qwen3-4b-xlam-lora")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
