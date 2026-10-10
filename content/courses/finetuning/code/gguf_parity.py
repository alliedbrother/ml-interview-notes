"""GGUF export parity: Modelfile template/stop checker and a three-runtime parity eval.

Module: 20 — Production: Serving Adapters, MLOps, and Cost Planning (Lab L20.2).

Smoke mode (CPU, stdlib only, < 1 s): parses Ollama Modelfiles and checks them
against the Hugging Face side of the contract (chat-template turn markers and the
end-of-turn tokens in generation_config's eos_token_id). Asserts on fixtures.

Full mode (tier T0 or CPU for 0.6-1.7B models; Transformers v5, llama.cpp built at
or after commit 781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4 with `llama-server` on PATH,
Ollama 0.40.x): runs the checker on your real files, creates the Ollama model,
generates greedily on module 01's eval prompts with Transformers, llama.cpp and
Ollama, and reports JSON parse rate, all-fields exact match, and disagreement counts.

Run (from the repository root):
    python content/courses/finetuning/code/gguf_parity.py --smoke
    python content/courses/finetuning/code/gguf_parity.py --full \
        --hf-model ./qwen3-1.7b-support-merged --gguf ./qwen3-1.7b-support-Q8_0.gguf \
        --ollama-model qwen3-support --modelfile ./Modelfile --eval-jsonl ./ft_data/eval.jsonl
"""
import argparse
import json
import re
import time

END_OF_TURN = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>"]
TURN_START = ["<|im_start|>", "<|start_header_id|>", "<start_of_turn>"]


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


def check_modelfile(modelfile, hf_chat_template, eos_token_ids, id_to_token):
    """List contract problems between an Ollama Modelfile and the HF artifacts."""
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
    if tpl and not re.search(r"assistant\s*\n?\s*$", tpl.rstrip('"')):
        problems.append("template_no_generation_prompt")
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


QWEN_HF_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
                    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
QWEN_IDS = {151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>"}
GOOD = '''FROM ./qwen3-1.7b-support-Q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
'''


def smoke():
    good = parse_modelfile(GOOD)
    assert good["FROM"] == "./qwen3-1.7b-support-Q8_0.gguf"
    assert good["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert check_modelfile(good, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == []

    no_stop = parse_modelfile(GOOD.replace('PARAMETER stop "<|im_end|>"\n', ""))
    assert check_modelfile(no_stop, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "stop_missing:<|im_end|>"]

    llama_tpl = GOOD.replace("<|im_start|>", "<|start_header_id|>").replace("<|im_end|>", "<|eot_id|>")
    probs = check_modelfile(parse_modelfile(llama_tpl), QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS)
    assert "template_foreign_marker:<|start_header_id|>" in probs
    assert "template_missing_marker:<|im_end|>" in probs
    assert "stop_missing:<|im_end|>" in probs

    no_gen = parse_modelfile(GOOD.replace("<|im_start|>assistant\n\"\"\"", "\"\"\""))
    assert check_modelfile(no_gen, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "template_no_generation_prompt"]

    sampled = parse_modelfile(GOOD.replace("temperature 0", "temperature 0.7"))
    assert check_modelfile(sampled, QWEN_HF_TEMPLATE, [151645, 151643], QWEN_IDS) == [
        "nonzero_temperature_for_parity"]

    bare = parse_modelfile("FROM ./x.gguf\n")
    assert "no_explicit_template" in check_modelfile(bare, QWEN_HF_TEMPLATE, [151645], QWEN_IDS)

    # Scorer used by full mode.
    gold = {"customer_name": "Ana", "order_id": "8812", "issue_type": "refund",
            "refund_requested": True, "sentiment": "negative"}
    assert score_one(json.dumps(gold), gold) == (True, True)
    assert score_one("```json\n" + json.dumps(gold) + "\n```", gold) == (True, True)
    assert score_one('{"customer_name": "Ana",}', gold) == (False, False)
    print("smoke: Modelfile checker and scorer verified on 7 fixtures")


# ------------------------------------------------------------------------------ full
def score_one(text, gold):
    """(parsed, all_fields_exact). Strips one surrounding ```json fence."""
    t = text.strip()
    fence = "`" * 3
    if t.startswith(fence):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith(fence):
            t = t.rstrip()[:-3]
    try:
        pred = json.loads(t)
    except Exception:
        return False, False
    return True, isinstance(pred, dict) and all(pred.get(k) == v for k, v in gold.items())


def load_eval(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = next(r[k] for k in ("prompt", "email", "input", "text") if k in r)
            gold = next(r[k] for k in ("target", "gold", "label", "completion", "output") if k in r)
            rows.append((prompt, json.loads(gold) if isinstance(gold, str) else gold))
    return rows


def full(args):
    import subprocess
    import urllib.request

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    gen_cfg = json.load(open(args.hf_model.rstrip("/") + "/generation_config.json"))
    eos = gen_cfg["eos_token_id"] if isinstance(gen_cfg["eos_token_id"], list) else [gen_cfg["eos_token_id"]]
    id_to_token = {i: tok.convert_ids_to_tokens(i) for i in eos}
    mf = parse_modelfile(open(args.modelfile).read())
    problems = check_modelfile(mf, tok.chat_template or "", eos, id_to_token)
    print("Modelfile problems:", problems or "none")

    rows = load_eval(args.eval_jsonl)
    msgs = lambda p: [{"role": "user", "content": p}]

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype="auto")
    hf_out = []
    for p, _ in rows:
        ids = tok.apply_chat_template(msgs(p), add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        hf_out.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

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
                       {"messages": msgs(p), "temperature": 0, "max_tokens": args.max_new_tokens,
                        "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]["content"]
                  for p, _ in rows]
    finally:
        server.terminate()
        server.wait()

    subprocess.run(["ollama", "create", args.ollama_model, "-f", args.modelfile], check=True)
    ol_out = [post("http://127.0.0.1:11434/api/chat",
                   {"model": args.ollama_model, "messages": msgs(p), "stream": False, "think": False,
                    "options": {"temperature": 0, "num_predict": args.max_new_tokens}})["message"]["content"]
              for p, _ in rows]

    for name, outs in (("transformers", hf_out), ("llama.cpp", lc_out), ("ollama", ol_out)):
        scores = [score_one(o, g) for o, (_, g) in zip(outs, rows)]
        parse = sum(s[0] for s in scores) / len(rows)
        exact = sum(s[1] for s in scores) / len(rows)
        differ = sum(a.strip() != b.strip() for a, b in zip(outs, hf_out))
        print("%-12s parse %.2f  all-fields EM %.2f  differs-from-HF %d/%d" % (name, parse, exact, differ, len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--hf-model")
    p.add_argument("--gguf")
    p.add_argument("--ollama-model", default="ft-parity")
    p.add_argument("--modelfile", default="./Modelfile")
    p.add_argument("--eval-jsonl", default="./ft_data/eval.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--llama-port", type=int, default=8081)
    args = p.parse_args()
    if args.full:
        if not (args.hf_model and args.gguf):
            p.error("--full needs --hf-model and --gguf")
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
