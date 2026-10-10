"""Evaluation statistics toolkit for fine-tuned models.

Purpose
    Turn per-item evaluation outcomes into an evaluation report with error bars:
    unbiased pass@k, Wilson intervals, paired bootstrap CIs on A-vs-B deltas,
    exact McNemar tests, and LLM-judge consistency / human-agreement metrics.

Module
    LLM Fine-Tuning Deep Dive, module 09 (Evaluating Fine-Tuned Models), labs L9.1 and L9.2.

Hardware tier
    --smoke: CPU, stdlib + numpy only, a few seconds.
    --full : CPU as well. It only reads files produced elsewhere: lm-evaluation-harness
             `--log_samples` outputs from a T1 GPU run (single 24 GB GPU, vLLM backend)
             and, optionally, a CSV of pairwise judge verdicts plus human labels.

Pinned versions for the full lab (the harness run that produces the inputs)
    lm_eval[vllm] 0.4.13 (CLI subcommands `run`/`ls`/`validate`), vllm >= 0.18 as required
    by that release, Python 3.11+. This script itself needs only numpy.

Run commands (from the repository root)
    python content/courses/finetuning/code/eval_toolkit.py --smoke
    python content/courses/finetuning/code/eval_toolkit.py --full \
        --base-dir results/base --tuned-dir results/sft \
        --tasks gsm8k ifeval mmlu_pro_math --judge-csv judge_labels.csv

Judge CSV format (L9.2): header `item_id,verdict_ab,verdict_ba,human`.
    verdict_ab: winner (A, B or tie) when answer A was shown first.
    verdict_ba: winner (A, B or tie, still named by answer identity) when B was shown first.
    human:      the human label (A, B or tie).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% normal quantile

# Default (metric, filter) per lm-eval task, as defined in the v0.4.13 task YAMLs.
TASK_DEFAULTS = {
    "gsm8k": ("exact_match", "strict-match"),
    "ifeval": ("prompt_level_strict_acc", "none"),
    "mmlu_pro": ("exact_match", "custom-extract"),
}


# ----------------------------------------------------------------------------- metrics
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021) in the numerically stable product form."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for mean(b) - mean(a), resampling item indices jointly."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def mcnemar_exact(a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired binary outcomes.

    Returns (n_a_only, n_b_only, p_value). Only discordant items carry evidence.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))  # B right, A wrong
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    tail = sum(math.comb(m, i) for i in range(0, min(n01, n10) + 1)) / 2**m
    return n01, n10, min(1.0, 2 * tail)


def judge_position_stats(verdict_ab, verdict_ba) -> dict:
    """Consistency and first-position preference for a pairwise judge run in both orders."""
    assert len(verdict_ab) == len(verdict_ba)
    pairs = len(verdict_ab)
    consistent = sum(x == y for x, y in zip(verdict_ab, verdict_ba))
    # In the AB run the first-shown answer is A; in the BA run it is B.
    first_picks = sum(v == "A" for v in verdict_ab) + sum(v == "B" for v in verdict_ba)
    decided = sum(v in ("A", "B") for v in verdict_ab) + sum(v in ("A", "B") for v in verdict_ba)
    return {
        "pairs": pairs,
        "consistency": consistent / pairs,
        "first_position_rate": first_picks / decided if decided else float("nan"),
    }


def combine_orders(verdict_ab, verdict_ba):
    """Final verdict: keep a winner only when both orders agree; otherwise a tie."""
    return [x if x == y else "tie" for x, y in zip(verdict_ab, verdict_ba)]


def cohen_kappa(x, y) -> float:
    labels = sorted(set(x) | set(y))
    n = len(x)
    po = sum(i == j for i, j in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[lab] * cy[lab] for lab in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ----------------------------------------------------------------------------- smoke
def smoke() -> None:
    # pass@k worked example from the module: n=10, c=3.
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(pass_at_k(10, 3, 5) - (1 - math.comb(7, 5) / math.comb(10, 5))) < 1e-12
    assert round(pass_at_k(10, 3, 5), 5) == 0.91667
    assert pass_at_k(10, 0, 5) == 0.0 and pass_at_k(10, 6, 5) == 1.0

    # Wilson worked example: 40/50 correct.
    lo, hi = wilson(40, 50)
    assert (round(lo, 3), round(hi, 3)) == (0.670, 0.888)

    # Paired fixture: 140 both right, 8 A-only, 20 B-only, 32 both wrong (200 items).
    a = np.array([1] * 140 + [1] * 8 + [0] * 20 + [0] * 32)
    b = np.array([1] * 140 + [0] * 8 + [1] * 20 + [0] * 32)
    delta, clo, chi = paired_bootstrap(a, b)
    assert abs(delta - 0.06) < 1e-12 and 0 < clo < delta < chi
    n01, n10, p = mcnemar_exact(a, b)
    assert (n01, n10) == (8, 20) and 0.03 < p < 0.04

    # Judge fixture: 14 consistent pairs, 6 pairs where the judge always picks the first slot.
    ab = ["A"] * 9 + ["B"] * 5 + ["A"] * 6
    ba = ["A"] * 9 + ["B"] * 5 + ["B"] * 6
    stats = judge_position_stats(ab, ba)
    assert stats["consistency"] == 0.70 and stats["first_position_rate"] == 0.65
    human = ["A"] * 9 + ["B"] * 5 + ["A", "B", "tie", "A", "B", "tie"]
    kappa = cohen_kappa(combine_orders(ab, ba), human)
    assert 0.0 < kappa < 1.0

    print(f"pass@5(n=10,c=3)={pass_at_k(10, 3, 5):.5f}  Wilson(40/50)=[{lo:.3f}, {hi:.3f}]")
    print(f"paired delta={delta:+.3f} 95% CI=[{clo:+.3f}, {chi:+.3f}]  McNemar p={p:.4f}")
    print(f"judge consistency={stats['consistency']:.2f} first-position={stats['first_position_rate']:.2f} "
          f"kappa vs human={kappa:.3f}")
    print("smoke OK: all toolkit assertions passed")


# ----------------------------------------------------------------------------- full
def load_samples(run_dir: str, task: str, metric: str, flt: str) -> dict:
    """Map doc_id -> score from lm-eval `samples_<task>_*.jsonl` files under run_dir."""
    paths = sorted(glob.glob(os.path.join(run_dir, "**", f"samples_{task}_*.jsonl"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no samples_{task}_*.jsonl under {run_dir}; did you pass --log_samples?")
    out = {}
    with open(paths[-1]) as fh:  # latest run
        for line in fh:
            rec = json.loads(line)
            if rec.get("filter") != flt:
                continue
            val = rec[metric]
            if isinstance(val, list):  # instance-level metrics such as inst_level_strict_acc
                val = sum(val) / len(val)
            out[rec["doc_id"]] = float(val)
    if not out:
        raise ValueError(f"{task}: no rows with filter={flt!r}; check --filter")
    return out


def full(args) -> None:
    print(f"{'task':<12}{'metric':<26}{'n':>6}{'base':>8}{'tuned':>8}{'delta':>8}   95% paired CI      McNemar p")
    for task in args.tasks:
        key = "mmlu_pro" if task.startswith("mmlu_pro_") else task  # subject subtasks share a template
        metric, flt = TASK_DEFAULTS.get(key, (args.metric, args.filter))
        base = load_samples(args.base_dir, task, metric, flt)
        tuned = load_samples(args.tuned_dir, task, metric, flt)
        ids = sorted(set(base) & set(tuned))
        if len(ids) != len(base) or len(ids) != len(tuned):
            print(f"warning: {task}: {len(base)} base vs {len(tuned)} tuned items; using {len(ids)} shared")
        a = np.array([base[i] for i in ids])
        b = np.array([tuned[i] for i in ids])
        delta, lo, hi = paired_bootstrap(a, b, n_boot=args.n_boot, seed=args.seed)
        binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
        p = mcnemar_exact(a, b)[2] if binary else float("nan")
        print(f"{task:<12}{metric:<26}{len(ids):>6}{a.mean():>8.3f}{b.mean():>8.3f}{delta:>+8.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]   {p:.4f}")
        if binary:
            for name, arr in (("base", a), ("tuned", b)):
                wl, wh = wilson(int(arr.sum()), len(arr))
                print(f"{'':<12}{name + ' Wilson 95%':<26}[{wl:.3f}, {wh:.3f}]")

    if args.judge_csv:
        with open(args.judge_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        ab = [r["verdict_ab"].strip() for r in rows]
        ba = [r["verdict_ba"].strip() for r in rows]
        human = [r["human"].strip() for r in rows]
        stats = judge_position_stats(ab, ba)
        final = combine_orders(ab, ba)
        agree = sum(x == y for x, y in zip(final, human))
        lo, hi = wilson(agree, len(human))
        print(f"\njudge: {stats['pairs']} pairs, consistency={stats['consistency']:.3f}, "
              f"first-position rate={stats['first_position_rate']:.3f}")
        print(f"judge vs human: agreement={agree / len(human):.3f} (Wilson 95% [{lo:.3f}, {hi:.3f}]), "
              f"Cohen kappa={cohen_kappa(final, human):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run fixture checks (default)")
    mode.add_argument("--full", action="store_true", help="read harness outputs and print the report")
    ap.add_argument("--base-dir")
    ap.add_argument("--tuned-dir")
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "ifeval", "mmlu_pro"])
    ap.add_argument("--metric", default="exact_match", help="for tasks not in TASK_DEFAULTS")
    ap.add_argument("--filter", default="none", help="for tasks not in TASK_DEFAULTS")
    ap.add_argument("--judge-csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.full:
        if not (args.base_dir and args.tuned_dir):
            ap.error("--full needs --base-dir and --tuned-dir")
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
