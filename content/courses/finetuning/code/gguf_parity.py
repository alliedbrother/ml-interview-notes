"""GGUF export parity: Modelfile template/stop checker and a three-runtime parity eval.

Module: 20 — Production: Serving Adapters, MLOps, and Cost Planning (Lab L20.2).

Smoke mode (CPU, stdlib only, < 1 s): parses Ollama Modelfiles and checks them
against the Hugging Face side of the contract (chat-template turn markers, Qwen3's
empty think block for thinking-off fine-tunes, and the end-of-turn tokens in
generation_config's eos_token_id), and checks the module 01 scorer on its fixture.
With --eval-jsonl, also validates a module 01 eval file and self-scores its gold rows.

Full mode (tier T0 or CPU for 0.6-1.7B models; Transformers v5, llama.cpp built at
or after commit 781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4 with `llama-server` on PATH,
Ollama 0.40.x): runs the checker on your real files, creates the Ollama model,
generates greedily on module 01's eval rows (system prompt + email, thinking off)
with Transformers, llama.cpp and Ollama, and reports parse rate, schema validity,
all-fields exact match (module 01's scoring rules), and disagreement counts.

Run (from the repository root):
    python content/courses/finetuning/code/gguf_parity.py --smoke
    python content/courses/finetuning/code/gguf_parity.py --full \
        --hf-model ./qwen3-1.7b-support-merged --gguf ./qwen3-1.7b-support-Q8_0.gguf \
        --ollama-model qwen3-support --modelfile ./Modelfile --eval-jsonl ./ft_data/eval.jsonl
"""
import argparse
import importlib.util
import json
import os
import re
import time
from fractions import Fraction

# Running-example contract (module 01, fit_or_not_baseline.py); copied verbatim.
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

END_OF_TURN = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>"]
TURN_START = ["<|im_start|>", "<|start_header_id|>", "<start_of_turn>"]
EMPTY_THINK = "<think>\n\n</think>\n\n"


def parse_modelfile(text):
    """Return {'FROM': str, 'TEMPLATE': str|None, 'stop': [..], 'params': {..}}."""
    out = {"FROM": None, "TEMPLATE": None, "stop": [], "params": {}}
    triple = re.compile(r'^(TEMPLATE|SYSTEM)\s+"""(.*?)"""', re.S | re.M)
    for m in triple.finditer(text):
        out[m.group(1)] = m.group(2)
    stripped = triple.sub("", text)
    for line in stripped.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        head = head.upper()
        if head == "FROM":
            out["FROM"] = rest.strip()
        elif head == "PARAMETER":
            key, _, val = rest.strip().partition(" ")
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if key == "stop":
                out["stop"].append(val)
            else:
                out["params"][key] = val
    return out


def check_modelfile(modelfile, hf_chat_template, eos_token_ids, id_to_token, thinking_off=True):
    """List contract problems between an Ollama Modelfile and the HF artifacts.

    thinking_off: the model was trained (and is evaluated) with enable_thinking=False, so
    if the HF template has Qwen3's empty think block, the Ollama template must emit it too.
    """
    problems = []
    tpl = modelfile["TEMPLATE"]
    if tpl is None:
        problems.append("no_explicit_template")
        tpl = ""
    for marker in TURN_START + END_OF_TURN:
        in_hf, in_ollama = marker in hf_chat_template, marker in tpl
        if in_hf and not in_ollama and tpl:
            problems.append("template_missing_marker:" + marker)
        if in_ollama and not in_hf:
            problems.append("template_foreign_marker:" + marker)
    tail = tpl[:-len(EMPTY_THINK)] if tpl.endswith(EMPTY_THINK) else tpl
    if tpl and not re.search(r"assistant\s*\n?\s*$", tail):
        problems.append("template_no_generation_prompt")
    if tpl and thinking_off and "<think>" in hf_chat_template and not tpl.endswith(EMPTY_THINK):
        problems.append("template_missing_empty_think")
    eos_strings = {id_to_token[i] for i in eos_token_ids if i in id_to_token}
    for marker in END_OF_TURN:
        if marker in hf_chat_template and marker not in modelfile["stop"]:
            problems.append("stop_missing:" + marker)
    for tok in sorted(eos_strings):
        if tok not in modelfile["stop"] and tok not in END_OF_TURN:
            problems.append("stop_missing_eos:" + tok)
    if modelfile["params"].get("temperature") not in ("0", "0.0"):
        problems.append("nonzero_temperature_for_parity")
    return problems


# Shape of Qwen3's generation prompt (the real template is longer).
QWEN_HF_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
                    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
                    "{% if enable_thinking is defined and enable_thinking is false %}"
                    "<think>\n\n</think>\n\n{% endif %}{% endif %}")
QWEN_IDS = {151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>"}
GOOD = """FROM ./qwen3-1.7b-support-Q8_0.gguf
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

\"\"\"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
"""


# ------------------------------------------------------- scoring (module 01 contract)
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


def load_eval(path):
    """Read module 01's eval.jsonl: rows of exactly {id, email, gold}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert set(row) == {"id", "email", "gold"}, sorted(row)
            assert list(row["gold"]) == FIELDS and schema_valid(row["gold"]), row["id"]
            rows.append(row)
    return rows


def messages(row):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["email"]}]


def smoke(args):
    good = parse_modelfile(GOOD)
    assert good["FROM"] == "./qwen3-1.7b-support-Q8_0.gguf"
    assert good["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert good["TEMPLATE"].endswith("<|im_start|>assistant\n" + EMPTY_THINK)
    assert check_modelfile(good, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == []

    # The module's earlier template, without the empty think block: a parity bug.
    no_think = parse_modelfile(GOOD.replace(EMPTY_THINK, ""))
    assert check_modelfile(no_think, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "template_missing_empty_think"]

    no_stop = parse_modelfile(GOOD.replace('PARAMETER stop "<|im_end|>"\n', ""))
    assert check_modelfile(no_stop, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "stop_missing:<|im_end|>"]

    llama_tpl = GOOD.replace("<|im_start|>", "<|start_header_id|>").replace("<|im_end|>", "<|eot_id|>")
    probs = check_modelfile(parse_modelfile(llama_tpl), QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS)
    assert "template_foreign_marker:<|start_header_id|>" in probs
    assert "template_missing_marker:<|im_end|>" in probs
    assert "stop_missing:<|im_end|>" in probs

    no_gen = parse_modelfile(GOOD.replace("<|im_start|>assistant\n" + EMPTY_THINK + '"""', '"""'))
    assert check_modelfile(no_gen, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "template_no_generation_prompt", "template_missing_empty_think"]

    sampled = parse_modelfile(GOOD.replace("temperature 0", "temperature 0.7"))
    assert check_modelfile(sampled, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "nonzero_temperature_for_parity"]

    bare = parse_modelfile("FROM ./x.gguf\n")
    assert "no_explicit_template" in check_modelfile(bare, QWEN_HF_TEMPLATE, [151645], QWEN_IDS)

    # Scorer: module 01's six-prediction fixture must give its published counts.
    gold = {"customer_name": "Maya Patel", "order_id": "ORD-48213", "issue_type": "refund",
            "refund_requested": True, "sentiment": "negative"}
    perfect, fence = json.dumps(gold), "`" * 3
    preds = [perfect, perfect.replace('"refund",', '"billing",'), perfect[:-1] + ",}",
             f"{fence}json\n{perfect}\n{fence}",
             json.dumps({k: v for k, v in gold.items() if k != "sentiment"}),
             perfect.replace("true", '"true"')]
    s = score(preds, [gold] * 6)
    assert (s["parse_rate"], s["schema_valid"], s["all_fields_em"]) == (
        Fraction(5, 6), Fraction(2, 6), Fraction(2, 6))
    assert (s["em_order_id"], s["em_issue_type"]) == (Fraction(5, 6), Fraction(4, 6))

    # SYSTEM_PROMPT must equal module 01's export when the generator sits next to this file.
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_or_not_baseline.py")
    if os.path.exists(ref):
        spec = importlib.util.spec_from_file_location("fit_or_not_baseline", ref)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.SYSTEM_PROMPT == SYSTEM_PROMPT and mod.FIELDS == FIELDS

    if args.eval_jsonl:
        rows = load_eval(args.eval_jsonl)
        s = score([json.dumps(r["gold"]) for r in rows], [r["gold"] for r in rows])
        assert all(v == 1 for v in s.values()), s
        print("eval file %s: %d rows conform; gold-as-prediction scores 1.0 on every metric"
              % (args.eval_jsonl, len(rows)))
    print("smoke: Modelfile checker (7 fixtures) and module 01 scorer fixture verified")


# ------------------------------------------------------------------------------ full
def full(args):
    import subprocess
    import urllib.request

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    gen_cfg = json.load(open(os.path.join(args.hf_model, "generation_config.json")))
    eos = gen_cfg["eos_token_id"] if isinstance(gen_cfg["eos_token_id"], list) else [gen_cfg["eos_token_id"]]
    id_to_token = {i: tok.convert_ids_to_tokens(i) for i in eos}
    mf = parse_modelfile(open(args.modelfile).read())
    problems = check_modelfile(mf, tok.chat_template or "", eos, id_to_token)
    print("Modelfile problems:", problems or "none")

    rows = load_eval(args.eval_jsonl)
    golds = [r["gold"] for r in rows]

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype="auto")
    hf_out = []
    for r in rows:
        enc = tok.apply_chat_template(messages(r), add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True, return_dict=True, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        hf_out.append(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))

    def post(url, body):
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    server = subprocess.Popen(["llama-server", "-m", args.gguf, "--port", str(args.llama_port), "--jinja"])
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % args.llama_port, timeout=2)
                break
            except Exception:
                time.sleep(1)
        lc_out = [post("http://127.0.0.1:%d/v1/chat/completions" % args.llama_port,
                       {"messages": messages(r), "temperature": 0, "max_tokens": args.max_new_tokens,
                        "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]["content"]
                  for r in rows]
    finally:
        server.terminate()
        server.wait()

    # The Modelfile TEMPLATE hard-codes the empty think block, so no `think` flag is sent.
    subprocess.run(["ollama", "create", args.ollama_model, "-f", args.modelfile], check=True)
    ol_out = [post("http://127.0.0.1:11434/api/chat",
                   {"model": args.ollama_model, "messages": messages(r), "stream": False,
                    "options": {"temperature": 0, "num_predict": args.max_new_tokens}})["message"]["content"]
              for r in rows]

    for name, outs in (("transformers", hf_out), ("llama.cpp", lc_out), ("ollama", ol_out)):
        s = score(outs, golds)
        differ = sum(a.strip() != b.strip() for a, b in zip(outs, hf_out))
        print("%-12s parse %.2f  schema %.2f  all-fields EM %.2f  differs-from-HF %d/%d" % (
            name, s["parse_rate"], s["schema_valid"], s["all_fields_em"], differ, len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--hf-model")
    p.add_argument("--gguf")
    p.add_argument("--ollama-model", default="ft-parity")
    p.add_argument("--modelfile", default="./Modelfile")
    p.add_argument("--eval-jsonl", default=None,
                   help="module 01 eval.jsonl (full mode default: ./ft_data/eval.jsonl; "
                        "with --smoke, also validates and self-scores this file)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--llama-port", type=int, default=8081)
    args = p.parse_args()
    if args.full:
        if not (args.hf_model and args.gguf):
            p.error("--full needs --hf-model and --gguf")
        args.eval_jsonl = args.eval_jsonl or "./ft_data/eval.jsonl"
        full(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
