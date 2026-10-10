"""Train, kill, resume, and prove the runs are bitwise identical (module 16 lab).

A small MLP classifier with dropout 0.1 trains on a seeded synthetic 4-class
dataset (2,000 points: 1,600 train, 400 validation). The loop uses a shuffled
DataLoader whose order is a pure function of (seed, epoch) via a step-indexed
sampler, AdamW, a warmup-cosine LambdaLR, gradient accumulation, clipping at 1.0,
and evaluation every 20 updates with early-stopping bookkeeping.

Three runs in one process:
  A  uninterrupted, 0 -> steps
  B  0 -> kill_at, atomic checkpoint (tmp file + os.replace), all training objects
     discarded; fresh objects are built, the checkpoint is loaded with
     weights_only=True, and training continues kill_at -> steps
  C  like B, but the RNG states are NOT restored (negative control)
A and B must end with bitwise-identical parameters; C must not.

Run from the repository root:
    python content/courses/pytorch/code/16_train_resume.py [--seed 0] [--steps 120] [--kill-at 50] [--accum 2] [--output-dir DIR] [--json-out PATH]
"""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset

IN_FEATURES = 16
NUM_CLASSES = 4
MICRO_BATCH = 32
EVAL_EVERY = 20
PATIENCE = 3
CKPT_FORMAT = 1


# ---------------------------------------------------------------- data
def make_data(seed):
    """Four Gaussian blobs in 16-d with overlap; fixed train/val split."""
    gen = torch.Generator().manual_seed(1_000 + seed)
    centers = 0.7 * torch.randn(NUM_CLASSES, IN_FEATURES, generator=gen)
    y = torch.arange(2_000) % NUM_CLASSES
    x = centers[y] + torch.randn(2_000, IN_FEATURES, generator=gen)
    perm = torch.randperm(2_000, generator=gen)
    x, y = x[perm], y[perm]
    return TensorDataset(x[:1_600], y[:1_600]), TensorDataset(x[1_600:], y[1_600:])


class StepIndexedSampler(Sampler):
    """Shuffled order that is a pure function of (seed, epoch); can start mid-epoch.

    Resuming at micro-batch b of epoch e means slicing the permutation at
    b * batch_size: no batches are replayed and no generator state is needed.
    """

    def __init__(self, n, seed, batch_size):
        self.n, self.seed, self.batch_size = n, seed, batch_size
        self.epoch, self.start_batch = 0, 0

    def set_position(self, epoch, start_batch):
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed * 10_007 + self.epoch)
        order = torch.randperm(self.n, generator=gen)
        return iter(order[self.start_batch * self.batch_size :].tolist())

    def __len__(self):
        return self.n - self.start_batch * self.batch_size


# ---------------------------------------------------------------- model and objects
def build(seed, steps, train_ds, init_seed=None):
    """Fresh model, optimizer, scheduler, sampler and loader (a 'new process').

    init_seed reseeds the global RNG before init; the data order depends on seed only.
    """
    torch.manual_seed(seed if init_seed is None else init_seed)
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, NUM_CLASSES),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    warmup = max(1, steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    sampler = StepIndexedSampler(len(train_ds), seed, MICRO_BATCH)
    loader = DataLoader(train_ds, batch_size=MICRO_BATCH, sampler=sampler,
                        drop_last=True, num_workers=0)
    return model, opt, sched, sampler, loader


# ---------------------------------------------------------------- RNG capture
def rng_state():
    """Every RNG the run could touch, as weights_only-safe objects."""
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),                       # uint8 tensor
        "python": random.getstate(),                          # tuple of ints
        "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)),
                  int(np_state[2]), int(np_state[3]), float(np_state[4])],
        # torch.cuda.get_rng_state_all() would go here on GPU (not executed here)
    }


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))


# ---------------------------------------------------------------- checkpoint I/O
def atomic_save(obj, path):
    """torch.save to path.tmp, fsync, then os.replace: readers see old or new, never half."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, opt, sched, state, config):
    ckpt = {
        "format": CKPT_FORMAT,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": None,                     # GradScaler.state_dict() when AMP is on (module 17)
        "trainer": dict(state),             # step, micro-batch position, early-stop state
        "rng": rng_state(),
        "config": dict(config),
        "meta": {"torch": str(torch.__version__), "threads": torch.get_num_threads()},
    }
    atomic_save(ckpt, path)
    manifest = {"file": Path(path).name, "bytes": Path(path).stat().st_size,
                "sha256": sha256(path), "step": state["step"]}
    Path(path).with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_checkpoint(path, model, opt, sched, restore_rng=True):
    manifest = json.loads(Path(path).with_suffix(".json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise RuntimeError(f"checksum mismatch for {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        set_rng_state(ckpt["rng"])
    return dict(ckpt["trainer"])


# ---------------------------------------------------------------- train / eval
def evaluate(model, val_ds):
    """Sums, not means; eval mode + inference_mode; restores train mode."""
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for xb, yb in DataLoader(val_ds, batch_size=200, shuffle=False):
            logits = model(xb)
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(-1) == yb).sum().item()
            count += len(yb)
    model.train(was_training)
    return loss_sum / count, correct / count


def train(model, opt, sched, sampler, loader, val_ds, state, stop_at, accum):
    """Run optimizer updates from state['step'] to stop_at; returns per-update losses."""
    per_epoch = len(loader.dataset) // MICRO_BATCH
    losses = []
    model.train()
    while state["step"] < stop_at:
        epoch, start = divmod(state["micro"], per_epoch)
        sampler.set_position(epoch, start)
        it = iter(loader)
        for _ in range(per_epoch - start):
            if state["step"] >= stop_at:
                break
            xb, yb = next(it)
            loss = F.cross_entropy(model(xb), yb) / accum   # equal micro-batch sizes
            loss.backward()
            state["micro"] += 1
            state["window_loss"] += loss.item()
            if state["micro"] % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            losses.append(state["window_loss"])
            state["window_loss"] = 0.0
            if state["step"] % EVAL_EVERY == 0:
                _, acc = evaluate(model, val_ds)
                if acc > state["best_val_acc"]:
                    state["best_val_acc"], state["bad_evals"] = acc, 0
                else:
                    state["bad_evals"] += 1
                state["should_stop"] = state["bad_evals"] >= PATIENCE  # logged, not acted on
    return losses


def fresh_state():
    return {"step": 0, "micro": 0, "window_loss": 0.0,
            "best_val_acc": 0.0, "bad_evals": 0, "should_stop": False}


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


# ---------------------------------------------------------------- experiment
def run(seed=0, steps=120, kill_at=50, accum=2, output_dir=None):
    if not 0 < kill_at < steps:
        raise ValueError("need 0 < kill_at < steps")
    if (len(make_data(seed)[0]) // MICRO_BATCH) % accum:
        raise ValueError("accum must divide micro-batches per epoch in this lab")
    train_ds, val_ds = make_data(seed)
    config = {"seed": seed, "steps": steps, "accum": accum, "micro_batch": MICRO_BATCH}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Run A: uninterrupted.
    model_a, opt, sched, sampler, loader = build(seed, steps, train_ds)
    state_a = fresh_state()
    curve_a = train(model_a, opt, sched, sampler, loader, val_ds, state_a, steps, accum)
    params_a = flat_params(model_a)

    # Run B, first half: identical start, killed after an atomic checkpoint.
    model, opt, sched, sampler, loader = build(seed, steps, train_ds)
    state = fresh_state()
    curve_b = train(model, opt, sched, sampler, loader, val_ds, state, kill_at, accum)
    ckpt_path = out / "last.pt"
    manifest = save_checkpoint(ckpt_path, model, opt, sched, state, config)
    del model, opt, sched, sampler, loader, state           # the "crash"

    def resume(restore_rng):
        # A new process would reseed and rebuild; that consumes RNG differently from A.
        m, o, s, smp, ld = build(seed, steps, train_ds, init_seed=seed + 1)
        st = load_checkpoint(ckpt_path, m, o, s, restore_rng=restore_rng)
        curve = train(m, o, s, smp, ld, val_ds, st, steps, accum)
        return m, st, curve

    model_b, state_b, tail_b = resume(restore_rng=True)
    curve_b = curve_b + tail_b
    model_c, _, _ = resume(restore_rng=False)

    params_b, params_c = flat_params(model_b), flat_params(model_c)
    _, final_acc = evaluate(model_b, val_ds)
    return {
        "bitwise_equal_params": bool(torch.equal(params_a, params_b)),
        "max_param_diff_resume": float((params_a - params_b).abs().max()),
        "max_param_diff_no_rng": float((params_a - params_c).abs().max()),
        "loss_curve_a": curve_a,
        "loss_curve_b": curve_b,
        "best_val_acc": float(state_a["best_val_acc"]),
        "best_val_acc_resumed": float(state_b["best_val_acc"]),
        "final_val_acc": float(final_acc),
        "checkpoint_files": sorted(p.name for p in out.iterdir()),
        "checkpoint_bytes": int(manifest["bytes"]),
        "checkpoint_sha256": manifest["sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--kill-at", type=int, default=50)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.output_dir:
        result = run(args.seed, args.steps, args.kill_at, args.accum, args.output_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(args.seed, args.steps, args.kill_at, args.accum, tmp)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
