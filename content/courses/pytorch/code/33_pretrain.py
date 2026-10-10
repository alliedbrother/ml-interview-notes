"""Pretrain the capstone GPT on the Sherlock Holmes train split (module 33).

Runs the E.5 recipe from llm/train.py (AdamW 0.9/0.95, wd 0.1 on matrices,
warmup + cosine 3e-3 -> 3e-4, clip 1.0), evaluates on story XI, optionally
saves a full-state checkpoint, resumes from one, checks the exact-resume
contract on a reduced model, and (writer-only) writes the committed artifact
code/data/gpt_sherlock.pt + gpt_sherlock.json.

Run from the repository root:
    python content/courses/pytorch/code/33_pretrain.py --smoke
    python content/courses/pytorch/code/33_pretrain.py --steps 800 --output-dir /tmp/gpt-run
    python content/courses/pytorch/code/33_pretrain.py --steps 800 --save-artifact --json-out /tmp/pretrain.json
"""

import argparse
import datetime
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm.corpus import CORPUS_SHA256, sha256_file  # noqa: E402
from llm.data import TOKENIZER_PATH, encode_all_splits  # noqa: E402
from llm.model import GPTConfig  # noqa: E402
from llm.train import (ARTIFACT_JSON, ARTIFACT_PT, TrainConfig, evaluate_split,  # noqa: E402
                       save_checkpoint, train)

SMOKE_STEPS, SMOKE_EVAL_EVERY = 150, 75
REDUCED_MODEL = dict(n_layer=1, n_head=4, d_model=32, block_size=32, dropout=0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--smoke", action="store_true", help="150 steps, eval every 75, full model size")
    p.add_argument("--batch-size", type=int, default=32, help="micro-batch (sequences)")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=None, help="default steps // 20")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--bf16-autocast", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-dir", default=None, help="keep the final full checkpoint here")
    p.add_argument("--resume", default=None, help="checkpoint written by --output-dir")
    p.add_argument("--check-resume", action="store_true",
                   help="also run the 12-vs-6+6 exact-resume check on a reduced model")
    p.add_argument("--save-artifact", action="store_true",
                   help="writer-only: write code/data/gpt_sherlock.pt and .json")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def config_from_args(args) -> TrainConfig:
    steps = SMOKE_STEPS if args.smoke else args.steps
    eval_every = SMOKE_EVAL_EVERY if args.smoke else args.eval_every
    return TrainConfig(steps=steps, batch_size=args.batch_size, accum=args.accum, lr=args.lr,
                       warmup=args.warmup, eval_every=eval_every, seed=args.seed,
                       bf16_autocast=args.bf16_autocast)


def resume_check(seed: int = 1337, steps: int = 12, workdir=None) -> dict:
    """Train a reduced model `steps` straight vs `steps/2` + checkpoint + fresh objects + rest.

    Dropout is on, so forgetting to restore the global RNG state breaks equality.
    """
    cfg = TrainConfig(steps=steps, batch_size=8, eval_every=0, seed=seed,
                      model=GPTConfig(**REDUCED_MODEL))
    straight = train(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(workdir or tmp) / "half.pt"
        first = train(cfg, max_steps=steps // 2)
        save_checkpoint(path, first["model"], first["optimizer"], first["step"], first["loader"],
                        cfg, first["tokenizer_sha256"])
        resumed = train(cfg, start_checkpoint=path)  # fresh model, optimizer and loader inside
    a, b = straight["model"].state_dict(), resumed["model"].state_dict()
    diff = max((a[k] - b[k]).abs().max().item() for k in a)
    return {"max_abs_diff": diff, "straight_loader": straight["loader"].state_dict(),
            "resumed_loader": resumed["loader"].state_dict(),
            "straight": straight, "resumed": resumed}


def ascii_curve(history, width=40) -> str:
    """Loss vs step as a text bar chart (val loss; train running mean in brackets)."""
    lo, hi = 2.0, max(h["val_loss"] for h in history)
    lines = []
    for h in history:
        n = int(round((h["val_loss"] - lo) / (hi - lo) * width))
        tr = "" if h["train_loss"] is None else f" [train {h['train_loss']:.2f}]"
        lines.append(f"{h['step']:>5} | {'#' * n:<{width}} {h['val_loss']:.3f}{tr}")
    return "\n".join(lines)


def machine_description(num_threads: int) -> str:
    """e.g. "macOS 26.1 arm64, Apple M2 Pro, 4 threads" (CPU name best effort)."""
    system, cpu = platform.system(), platform.processor() or "unknown CPU"
    if system == "Darwin":
        system = f"macOS {platform.mac_ver()[0]}"
        try:
            import subprocess
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    return f"{system} {platform.machine()}, {cpu}, {num_threads} threads"


def write_artifact(model, cfg, result, splits, num_threads) -> dict:
    """fp32 state_dict -> gpt_sherlock.pt (atomic) and the manifest gpt_sherlock.json."""
    tmp = ARTIFACT_PT.with_name(ARTIFACT_PT.name + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ARTIFACT_PT)
    manifest = {
        "format_version": 1,
        "config": cfg.to_dict()["model"],
        "train_config": cfg.to_dict(),
        "params": model.num_params(),
        "steps": result["steps"],
        "seed": cfg.seed,
        "num_threads": num_threads,
        "tokens_seen": result["tokens_seen"],
        "tokenizer_file": TOKENIZER_PATH.name,
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "corpus_sha256": CORPUS_SHA256,
        "checkpoint_file": ARTIFACT_PT.name,
        "checkpoint_sha256": sha256_file(ARTIFACT_PT),
        "checkpoint_bytes": ARTIFACT_PT.stat().st_size,
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
        "eval": "non-overlapping windows (stride = block_size = 128), every target scored once; "
                "train = stories I-X, val = story XI",
        "history": result["history"],
        "tokens_per_second": result["tokens_per_second"],
        "wall_seconds": result["wall_seconds"],
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "machine": machine_description(num_threads),
        "created": datetime.date.today().isoformat(),
    }
    tmp = ARTIFACT_JSON.with_name(ARTIFACT_JSON.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, ARTIFACT_JSON)
    return manifest


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.save_artifact and (args.smoke or args.resume):
        raise SystemExit("--save-artifact needs a full run from scratch")
    num_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_threads)
    cfg = config_from_args(args)

    t0 = time.perf_counter()
    out = train(cfg, start_checkpoint=args.resume)
    wall = time.perf_counter() - t0
    model = out["model"]
    splits = encode_all_splits()
    final_val = evaluate_split(model, splits["val"], cfg.model.block_size)
    result = {
        "steps": out["step"],
        "history": out["history"],
        "final_val_loss": final_val,
        # full-split eval costs ~1 s, so the smoke run (CI) skips it
        "final_train_loss": None if args.smoke else evaluate_split(model, splits["train"],
                                                                   cfg.model.block_size),
        "tokens_seen": out["step"] * cfg.tokens_per_step,
        "tokens_per_second": out["tokens_per_second"],
        "wall_seconds": wall,
        "resume_max_abs_diff": None,
        "params": model.num_params(),
        "config": cfg.to_dict(),
        "lrs": out["lrs"],  # lr used at every step, for checking against lr_at
    }
    if args.check_resume:
        result["resume_max_abs_diff"] = resume_check(args.seed)["max_abs_diff"]

    if args.output_dir:
        path = Path(args.output_dir) / "ckpt.pt"
        save_checkpoint(path, model, out["optimizer"], out["step"], out["loader"], cfg,
                        out["tokenizer_sha256"])
        result["checkpoint"] = str(path)
    if args.save_artifact:
        if final_val >= 3.3:
            raise SystemExit(f"val loss {final_val:.3f} >= 3.3: investigate before committing")
        manifest = write_artifact(model, cfg, result, splits, num_threads)
        result["artifact_sha256"] = manifest["checkpoint_sha256"]
    print(ascii_curve(out["history"]), file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main()
    res.pop("lrs")
    print(json.dumps(res, indent=2))
