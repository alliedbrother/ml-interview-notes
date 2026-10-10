"""Budget forcing (s1) and thinking-length control.

Module: 14 — Distillation and Reasoning Fine-Tuning (LLM Fine-Tuning Deep Dive).

Smoke mode (CPU, stdlib only, < 1 s):
  * runs the token-stream budget-forcing logic on fake streams and asserts
    truncation at the budget (end tag appended) and "Wait" insertion when the
    model tries to stop thinking before a minimum budget;
  * runs a simulated model whose chance of a correct answer depends on how much
    of its needed thinking it got, and asserts that accuracy is non-decreasing
    in the budget and flattens once the budget covers most traces.

Full mode (tier T0: a free T4 16 GB in fp16 is enough for a 1.5B model):
  vLLM generation with a distilled reasoning model, thinking capped at each
  budget B (stop at </think>, max_tokens=B), the end tag appended, the answer
  generated, boxed answers graded against MATH-500. Writes budget_curve.csv.
  Pinned versions for full mode: vLLM >= 0.22, Transformers 5.x, datasets.

Run from the repository root:
  python content/courses/finetuning/code/budget_forcing.py --smoke
  python content/courses/finetuning/code/budget_forcing.py --full \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --budgets 256 512 1024 2048 --n 100
"""

import argparse
import csv
import random
import re

END, WAIT = 2, 3


def budget_force(stream, max_think, min_think, end_id=END, wait_id=WAIT):
    """Return the kept thinking tokens, always terminated by end_id.

    `stream` stands in for what the model would emit next. An end tag before
    `min_think` tokens is replaced by `wait_id` (s1 "Wait"); reaching
    `max_think` tokens forces the end tag."""
    out = []
    for tok in stream:
        if len(out) >= max_think:
            break
        if tok == end_id:
            if len(out) < min_think:
                out.append(wait_id)
                continue
            out.append(end_id)
            return out
        out.append(tok)
    out = out[:max_think]
    out.append(end_id)
    return out


def check_streams():
    assert budget_force([10, 11, 12, 13, 14, 15], 4, 0) == [10, 11, 12, 13, END]
    assert budget_force([10, 11, END, 12, 13, 14, END], 100, 4) == [10, 11, WAIT, 12, 13, 14, END]
    assert budget_force([10, 11, 12, END], 10, 2) == [10, 11, 12, END]
    # Both rules at once: the early END becomes WAIT, then the cap fires.
    assert budget_force([10, END, 11, 12, 13, 14], 3, 2) == [10, WAIT, 11, END]
    print("token-stream budget forcing ok")


def simulate_curve(budgets, n=2000, seed=0):
    """Toy model: each problem needs L ~ lognormal thinking tokens; with budget B the
    model answers correctly w.p. 0.85 if B >= L, else 0.85 * (B / L) ** 2."""
    rng = random.Random(seed)
    needs = [rng.lognormvariate(6.3, 0.6) for _ in range(n)]      # median ~545 tokens
    draws = [rng.random() for _ in range(n)]                      # shared across budgets
    curve = []
    for b in budgets:
        correct = 0
        for need, u in zip(needs, draws):
            p = 0.85 if b >= need else 0.85 * (b / need) ** 2
            correct += u < p
        curve.append(correct / n)
    return curve


def smoke():
    check_streams()
    budgets = [256, 512, 1024, 2048, 4096]
    curve = simulate_curve(budgets)
    for b, a in zip(budgets, curve):
        print(f"budget {b:5d}: simulated accuracy {a:.3f}")
    assert all(x <= y for x, y in zip(curve, curve[1:])), curve
    gain_early = curve[2] - curve[0]
    gain_late = curve[4] - curve[3]
    assert gain_early > 3 * gain_late, (gain_early, gain_late)   # steep, then flat
    print("smoke OK: truncation, Wait insertion, and a saturating accuracy-vs-budget curve")


BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def grade(text, gold):
    hits = BOXED.findall(text)
    if not hits:
        return False
    norm = lambda s: re.sub(r"\s+", "", s).rstrip(".")
    return norm(hits[-1]) == norm(gold)


def full(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(args.n))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for row in ds:
        p = tok.apply_chat_template(
            [{"role": "user", "content": row["problem"] + "\nPut the final answer in \\boxed{}."}],
            tokenize=False, add_generation_prompt=True)
        if not p.rstrip().endswith("<think>"):
            p += "<think>\n"                       # force thinking (R1-Distill card advice)
        prompts.append(p)
    llm = LLM(model=args.model, dtype="half", max_model_len=max(args.budgets) + 2048)
    rows = []
    for budget in args.budgets:
        think = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=budget, stop=["</think>"])
        texts = [o.outputs[0].text for o in llm.generate(prompts, think)]
        if args.min_think > 0:                     # s1 extension: suppress early stops
            for _ in range(args.max_waits):
                short = [i for i, t in enumerate(texts)
                         if len(tok(t)["input_ids"]) < min(args.min_think, budget)]
                if not short:
                    break
                more = SamplingParams(temperature=0.6, top_p=0.95, stop=["</think>"],
                                      max_tokens=max(64, budget - args.min_think // 2))
                cont = llm.generate([prompts[i] + texts[i] + "\nWait" for i in short], more)
                for i, o in zip(short, cont):
                    texts[i] = texts[i] + "\nWait" + o.outputs[0].text
        answer = SamplingParams(temperature=0.0, max_tokens=512)
        finals = llm.generate([p + t + "\n</think>\n\n" for p, t in zip(prompts, texts)], answer)
        acc = sum(grade(o.outputs[0].text, row["answer"])
                  for o, row in zip(finals, ds)) / len(ds)
        rows.append({"budget": budget, "accuracy": acc})
        print(f"budget {budget}: accuracy {acc:.3f}")
    with open("budget_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["budget", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print("wrote budget_curve.csv")


def main():
    ap = argparse.ArgumentParser(description="s1-style budget forcing")
    ap.add_argument("--smoke", action="store_true", help="CPU logic checks (default)")
    ap.add_argument("--full", action="store_true", help="vLLM experiment on a GPU (T0)")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-think", type=int, default=0)
    ap.add_argument("--max-waits", type=int, default=2)
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
