"""Evaluate the committed capstone GPT on held-out text (module 34).

Loads code/data/gpt_sherlock.pt through llm.train.load_pretrained (never trains),
then reports for one split: NLL / perplexity / bits per byte at each stride,
uniform / unigram / bigram baselines fitted on the train split, the standard
error and a block-bootstrap interval of the headline number, the "Sherlock" ->
" Holmes" logit margin, and optionally a next-token table and a memorisation probe.

The validation split (story XI) is the default. The test split (story XII) is
for the final report only.

Run from the repository root:
    python content/courses/pytorch/code/34_evaluate.py --prompts
    python content/courses/pytorch/code/34_evaluate.py --strides 128,64,32 --json-out /tmp/eval.json
    python content/courses/pytorch/code/34_evaluate.py --split test     # final report only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm.corpus import load_splits  # noqa: E402
from llm.data import encode_all_splits  # noqa: E402
from llm.evaluate import (block_bootstrap, bigram_nll, evaluate_nll, greedy_continuation,  # noqa: E402
                          greedy_margin, longest_common_substring_in, next_token_table,
                          scored_bytes, standard_error, unigram_nll, uniform_nll)
from llm.train import load_pretrained  # noqa: E402

PROMPTS = ("Sherlock", "Mr. Sherlock", "My dear", "It was a", "said")
MEMO_PROMPT_TOKENS, MEMO_NEW_TOKENS = 24, 30
TEST_BANNER = ("=" * 64 + "\nWARNING: evaluating the TEST split (story XII). Final report only:\n"
               "every decision must already be made on val.\n" + "=" * 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--strides", default="128,64", help="comma-separated, each in [1, 128]")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=200, help="replicates; 0 disables")
    p.add_argument("--limit-tokens", type=int, default=None, help="evaluate only the first N tokens")
    p.add_argument("--prompts", action="store_true", help="next-token table and memorisation probe")
    p.add_argument("--seed", type=int, default=34, help="bootstrap generator seed")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def memorisation_probe(model, tok, source_text: str, reference_texts: dict) -> dict:
    """Greedy-continue the first MEMO_PROMPT_TOKENS tokens of a story's body.

    Reports the longest verbatim substring of the continuation found in each
    reference text, in characters.
    """
    body = source_text.split("\n", 1)[1].lstrip()          # drop the story heading
    ids = tok.encode(body)[:MEMO_PROMPT_TOKENS]
    new = greedy_continuation(model, ids, MEMO_NEW_TOKENS)
    cont = tok.decode(new)
    true_next = tok.decode(tok.encode(body)[MEMO_PROMPT_TOKENS:MEMO_PROMPT_TOKENS + MEMO_NEW_TOKENS])
    return {"prompt": tok.decode(ids), "continuation": cont, "true_continuation": true_next,
            "lcs_chars": {name: len(longest_common_substring_in(cont, text))
                          for name, text in reference_texts.items()}}


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.split == "test":
        print(TEST_BANNER, file=sys.stderr)
    strides = [int(s) for s in args.strides.split(",") if s.strip()]

    t0 = time.perf_counter()
    model, tok, manifest = load_pretrained()
    splits = encode_all_splits(tok)
    tokens = splits[args.split]
    if args.limit_tokens is not None:
        tokens = tokens[:args.limit_tokens]
    T, V = model.config.block_size, model.config.vocab_size
    n_bytes = scored_bytes(tok, tokens)

    results = {}
    for s in strides:
        t = time.perf_counter()
        r = evaluate_nll(model, tokens, T, s, batch_size=args.batch_size, text_bytes=n_bytes)
        r["seconds"] = time.perf_counter() - t
        results[str(s)] = r
    headline = "64" if "64" in results else str(strides[0])
    h = results[headline]

    t = time.perf_counter()
    train = splits["train"]
    baselines = {"uniform": uniform_nll(V), "unigram": unigram_nll(train, tokens, V, alpha=1.0),
                 "bigram": bigram_nll(train, tokens, V, k=0.1)}
    baseline_seconds = time.perf_counter() - t

    se = standard_error(h["window_losses"], h["window_counts"])
    boot = (block_bootstrap(h["window_losses"], h["window_counts"], args.bootstrap, args.seed)
            if args.bootstrap > 0 else None)
    holmes_token, holmes_margin = greedy_margin(model, tok, "Sherlock")

    out = {
        "split": args.split,
        "n_tokens": int(tokens.numel()),
        "scored_bytes": n_bytes,
        "bytes_per_token": n_bytes / (tokens.numel() - 1),
        "headline_stride": int(headline),
        "results": {k: {kk: vv for kk, vv in r.items() if kk not in ("window_losses", "window_counts")}
                    for k, r in results.items()},
        "baselines": baselines,
        "baseline_seconds": baseline_seconds,
        "se": se,
        "normal_ci": [h["nll_per_token"] - 1.96 * se, h["nll_per_token"] + 1.96 * se],
        "bootstrap_ci": None if boot is None else [boot["lo"], boot["hi"]],
        "bootstrap": boot,
        "holmes_token": holmes_token,
        "holmes_margin": holmes_margin,
        "manifest_val_loss": manifest["final_val_loss"],
        "prompts": [],
    }
    if args.prompts:
        out["prompts"] = [{"prompt": p, "top3": next_token_table(model, tok, p, k=3)} for p in PROMPTS]
        texts = load_splits()
        train_text, val_text = "\n".join(texts["train"]), texts["val"][0]
        refs = {"train": train_text, "val": val_text}
        out["memorisation"] = {"train_story_I": memorisation_probe(model, tok, texts["train"][0], refs),
                               "val_story_XI": memorisation_probe(model, tok, val_text, refs)}
    out["wall_seconds"] = time.perf_counter() - t0
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
