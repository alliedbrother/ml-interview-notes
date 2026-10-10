"""Generate text with the committed capstone GPT and verify the KV cache (module 35).

Loads code/data/gpt_sherlock.pt through llm.train.load_pretrained, samples a
continuation (temperature / top-k / top-p with a seeded torch.Generator),
optionally checks cached against uncached greedy decoding for 64 steps, times
both decode loops with torch.utils.benchmark, and checks that a left-padded
batch reproduces per-prompt greedy generation.

Run from the repository root:
    python content/courses/pytorch/code/35_generate.py --verify
    python content/courses/pytorch/code/35_generate.py --prompt "Holmes" --temperature 0.8 --top-p 0.95
    python content/courses/pytorch/code/35_generate.py --benchmark --batch-prompts "Holmes|It was|My dear Watson"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm.generate import generate, generate_batch, stream_decode, verify_cache  # noqa: E402
from llm.model import KVCache  # noqa: E402
from llm.train import load_pretrained  # noqa: E402

BENCH_TEXT = "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name."
BENCH_PROMPT_TOKENS, BENCH_NEW_TOKENS = 16, 100


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", default="Sherlock Holmes")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-cache", action="store_true", help="recompute the full prefix every step")
    p.add_argument("--verify", action="store_true", help="64-step cached vs uncached greedy check")
    p.add_argument("--benchmark", action="store_true", help="time cached vs uncached decoding")
    p.add_argument("--batch-prompts", default=None, help='"A|B|C": batched left-padded greedy check')
    p.add_argument("--stream", action="store_true", help="also stream the completion to stderr")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def run_verify(model, tokenizer, prompt, steps=64):
    idx = torch.tensor([tokenizer.encode(prompt)])
    r = verify_cache(model, idx, steps)
    d = r["per_step_max_abs_diff"]
    return {"max_abs_logit_diff": r["max_abs_logit_diff"], "tokens_equal": r["tokens_equal"],
            "steps": steps, "first_divergence": r["first_divergence"],
            "margin_at_divergence": r["margin_at_divergence"], "min_margin": r["min_margin"],
            "max_diff_per_16_steps": [max(d[i:i + 16]) for i in range(0, steps, 16)]}


def run_benchmark(model, tokenizer, number=5):
    from torch.utils import benchmark
    idx = torch.tensor([tokenizer.encode(BENCH_TEXT)[:BENCH_PROMPT_TOKENS]])
    out = {"prompt_tokens": idx.size(1), "new_tokens": BENCH_NEW_TOKENS, "threads": torch.get_num_threads()}
    for name, use_cache in (("cached", True), ("uncached", False)):
        timer = benchmark.Timer(
            stmt="generate(model, idx, n, temperature=0.0, use_cache=use_cache)",
            globals={"generate": generate, "model": model, "idx": idx, "n": BENCH_NEW_TOKENS,
                     "use_cache": use_cache},
            num_threads=torch.get_num_threads(), label="generate 100 tokens", sub_label=name)
        generate(model, idx, 4, temperature=0.0, use_cache=use_cache)   # warm-up
        m = timer.timeit(number)
        out[f"{name}_ms"] = 1e3 * m.median
        out[f"{name}_ms_per_token"] = 1e3 * m.median / BENCH_NEW_TOKENS
    out["speedup"] = out["uncached_ms"] / out["cached_ms"]
    cfg = model.config
    out["kv_cache_bytes_B1"] = KVCache(cfg, 1, cfg.block_size).nbytes
    return out


def run_batch(model, tokenizer, prompts, max_new_tokens=32):
    batched = generate_batch(model, tokenizer, prompts, max_new_tokens, temperature=0.0, return_ids=True)
    single = [generate_batch(model, tokenizer, [p], max_new_tokens, temperature=0.0, return_ids=True)[0]
              for p in prompts]
    return {"prompts": prompts, "consistent": batched == single,
            "rows_equal": [a == b for a, b in zip(batched, single)],
            "completions": [tokenizer.decode(ids) for ids in batched]}


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model, tokenizer, _ = load_pretrained()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("--prompt must not be empty")
    gen = torch.Generator().manual_seed(args.seed)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and args.top_p < 1.0 else None
    out, info = generate(model, torch.tensor([prompt_ids]), args.max_new_tokens,
                         temperature=args.temperature, top_k=top_k, top_p=top_p,
                         eos_id=tokenizer.eot_id, generator=gen, use_cache=not args.no_cache,
                         return_info=True)
    new_ids = out[0, len(prompt_ids):].tolist()
    if args.stream:
        for piece in stream_decode(tokenizer, new_ids):
            print(piece, end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    result = {"prompt": args.prompt, "completion": tokenizer.decode(new_ids),
              "new_tokens": info["new_tokens"], "stop_reason": info["stop_reason"],
              "sampling": {"temperature": args.temperature, "top_k": top_k, "top_p": top_p,
                           "seed": args.seed, "use_cache": not args.no_cache},
              "verify": None, "benchmark": None, "batch_consistent": None}
    if args.verify:
        result["verify"] = run_verify(model, tokenizer, args.prompt)
    if args.benchmark:
        result["benchmark"] = run_benchmark(model, tokenizer)
    if args.batch_prompts:
        batch = run_batch(model, tokenizer, [p for p in args.batch_prompts.split("|") if p])
        result["batch"] = batch
        result["batch_consistent"] = batch["consistent"]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
