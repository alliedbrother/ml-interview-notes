"""Build and audit the capstone token pipeline (module 31).

Run from the repository root:
    python content/courses/pytorch/code/31_token_pipeline.py
    python content/courses/pytorch/code/31_token_pipeline.py --benchmark

Encodes the three story splits with the committed tokenizer, writes them as
uint16 files into --output-dir (default: a temporary directory that is removed),
reads them back, measures how many training windows straddle a story boundary,
checks that sliding-window evaluation scores every target exactly once and that
the training loader resumes exactly. With --benchmark it also times four ways of
producing a training batch (not run in CI).
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from llm.data import (SPLITS, TOKENIZER_PATH, RandomWindowDataset,  # noqa: E402
                      SequentialEvalDataset, boundary_fraction, encode_all_splits,
                      load_uint16, make_train_loader, save_uint16)
from llm.tokenizer import ByteBPE  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--benchmark", action="store_true", help="time batch construction (slow)")
    p.add_argument("--bench-batches", type=int, default=300)
    p.add_argument("--output-dir", default=None, help="keep the uint16 files here")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def eval_coverage(tokens, block_size, stride):
    """Count how often each target index is scored; return window and target counts."""
    ds = SequentialEvalDataset(tokens, block_size, stride)
    hits = torch.zeros(tokens.numel(), dtype=torch.int64)
    positions = torch.arange(tokens.numel())
    for i, b in enumerate(ds.starts):
        _, _, mask = ds[i]
        hits[positions[b + 1:b + 1 + ds.span][mask]] += 1
    once = bool(hits[0] == 0 and (hits[1:] == 1).all())
    return {"stride": stride, "n_windows": len(ds), "n_scored": int(hits.sum()),
            "n_targets": tokens.numel() - 1, "each_target_once": once}


def check_resume(tokens, block_size, batch_size, seed, resume_at=7, upto=12):
    straight = make_train_loader(tokens, block_size, batch_size, seed)
    ref = [next(straight) for _ in range(upto)]
    state = make_train_loader(tokens, block_size, batch_size, seed, start_step=resume_at).state_dict()
    resumed = make_train_loader(tokens, block_size, batch_size, seed=0)
    resumed.load_state_dict(state)
    again = [next(resumed) for _ in range(resume_at, upto)]
    return all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(ref[resume_at:], again))


def _time(fn, n):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def benchmark(tokens, block_size, batch_size, seed, n):
    """Microseconds per (x, y) batch for four constructions of the same thing."""
    g = torch.Generator().manual_seed(seed)
    hi = tokens.numel() - block_size

    def slice_stack():
        ix = torch.randint(0, hi, (batch_size,), generator=g).tolist()
        x = torch.stack([tokens[i:i + block_size] for i in ix])
        y = torch.stack([tokens[i + 1:i + 1 + block_size] for i in ix])
        return x, y

    windows = tokens.unfold(0, block_size + 1, 1)  # zero-copy view, built once

    def unfold_gather():
        w = windows[torch.randint(0, hi, (batch_size,), generator=g)]
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()

    out = {"slice_stack_us": _time(slice_stack, n), "unfold_gather_us": _time(unfold_gather, n)}
    ds = RandomWindowDataset(tokens, block_size, num_samples=n * batch_size, seed=seed)
    for workers in (0, 2):
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers)
        t0 = time.perf_counter()
        it = iter(dl)
        next(it)
        first = time.perf_counter() - t0
        t1 = time.perf_counter()
        k = 0
        for _ in it:
            k += 1
            if k == n - 1:
                break
        steady = (time.perf_counter() - t1) / max(k, 1) * 1e6
        out[f"dataloader_w{workers}_us"] = steady
        out[f"dataloader_w{workers}_first_batch_ms"] = first * 1e3
    return out


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    T, B = args.block_size, args.batch_size
    t0 = time.perf_counter()
    tok = ByteBPE.load(TOKENIZER_PATH)
    splits = encode_all_splits(tok)
    encode_s = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(args.output_dir) if args.output_dir else Path(tmp)
        nbytes = {}
        for s in SPLITS:
            path = out_dir / f"{s}.bin"
            nbytes[s] = save_uint16(splits[s], path)
            if not torch.equal(load_uint16(path), splits[s]):
                raise AssertionError(f"uint16 round trip changed split {s}")

    train = splits["train"]
    x, y = next(make_train_loader(train, T, B, args.seed))
    if not (x.shape == y.shape == (B, T) and x.dtype == y.dtype == torch.int64
            and torch.equal(y[:, :-1], x[:, 1:])):
        raise AssertionError("training batch violates the shift contract")

    frac = boundary_fraction(train, T, tok.eot_id)
    result = {
        "n_tokens": {s: int(splits[s].numel()) for s in SPLITS},
        "n_eot": {s: int((splits[s] == tok.eot_id).sum()) for s in SPLITS},
        "uint16_bytes": nbytes,
        "max_id": int(max(int(t.max()) for t in splits.values())),
        "boundary_fraction": frac,
        "boundary_windows_per_800_steps": round(frac * 800 * B, 1),
        "passes_over_train_800_steps": round(800 * B * T / train.numel(), 2),
        "eval_windows": eval_coverage(splits["val"], T, args.stride),
        "eval_windows_by_stride": [eval_coverage(splits["val"], T, s) for s in (T, T // 2, T // 4)],
        "resume_ok": check_resume(train, T, B, args.seed),
        "encode_seconds": round(encode_s, 3),
        "bench": benchmark(train, T, B, args.seed, args.bench_batches) if args.benchmark else None,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
