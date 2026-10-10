"""Pretraining loop, checkpoints and the committed checkpoint loader (module 33).

The recipe (syllabus E.5): AdamW (betas 0.9/0.95, weight decay 0.1 on matrices
only), linear warmup to 3e-3 then cosine decay to 3e-4, global-norm clipping at
1.0, optional gradient accumulation and CPU bf16 autocast, periodic
non-overlapping evaluation on story XI, and a full-state checkpoint that resumes
exactly.

Frozen interface (modules 34-38 import it):
    TrainConfig, lr_at, train, evaluate_split, save_checkpoint, load_checkpoint,
    ARTIFACT_PT, ARTIFACT_JSON, load_pretrained

Read it next to content/courses/pytorch/33-capstone-pretraining.md.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from llm.corpus import DATA_DIR, sha256_file
from llm.data import TOKENIZER_PATH, SequentialEvalDataset, encode_all_splits, make_train_loader
from llm.model import IGNORE_INDEX, GPT, GPTConfig
from llm.tokenizer import ByteBPE

ARTIFACT_PT = DATA_DIR / "gpt_sherlock.pt"
ARTIFACT_JSON = DATA_DIR / "gpt_sherlock.json"
CHECKPOINT_FORMAT = 1


# ----------------------------------------------------------------- config

@dataclass
class TrainConfig:
    """Every knob of one pretraining run. ``batch_size`` is the micro-batch;
    one optimizer step sees ``batch_size * accum`` sequences of ``block_size`` tokens."""
    steps: int = 800
    batch_size: int = 32
    accum: int = 1
    lr: float = 3e-3
    min_lr_ratio: float = 0.1          # cosine floor = 10 % of peak
    warmup: int | None = None          # None -> steps // 20 (40 for the full run)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0             # 0 disables clipping (the norm is still measured)
    eval_every: int = 200
    eval_batch_size: int = 32
    seed: int = 1337
    bf16_autocast: bool = False
    model: GPTConfig = field(default_factory=GPTConfig)

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = GPTConfig(**self.model)
        if self.warmup is None:
            self.warmup = max(1, self.steps // 20)
        if self.steps < 1 or self.batch_size < 1 or self.accum < 1:
            raise ValueError("steps, batch_size and accum must be >= 1")
        if not 0 < self.warmup < self.steps:
            raise ValueError(f"warmup must lie in (0, steps={self.steps}), got {self.warmup}")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.accum * self.model.block_size

    def to_dict(self) -> dict:
        return asdict(self)  # nested GPTConfig becomes a plain dict

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for 0-indexed optimizer step ``step``.

    Linear warmup reaches the peak at ``step = warmup - 1``; cosine decay from
    the peak at ``step = warmup`` to ``min_lr_ratio * lr`` at ``step = steps``
    (so the last executed step, ``steps - 1``, is a hair above the floor).
    """
    peak, floor = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup:
        return peak * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return floor
    progress = (step - cfg.warmup) / (cfg.steps - cfg.warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------- evaluation

def evaluate_split(model: GPT, tokens: torch.Tensor, block_size: int, batch_size: int = 32) -> float:
    """Mean next-token NLL (nats) over a whole split with non-overlapping windows.

    Every target 1..n-1 is scored exactly once (module 31's SequentialEvalDataset
    with stride = block_size; the last window is end-aligned and masked). Runs in
    eval mode under inference_mode and restores the model's previous mode.
    Module 34 owns the full evaluator (sliding windows, bits per byte, baselines).
    """
    ds = SequentialEvalDataset(tokens, block_size, stride=block_size)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.inference_mode():
            for i in range(0, len(ds), batch_size):
                items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
                x = torch.stack([it[0] for it in items])
                y = torch.stack([it[1] for it in items])
                m = torch.stack([it[2] for it in items])
                y = y.masked_fill(~m, IGNORE_INDEX)
                logits, _ = model(x)
                total += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                                         ignore_index=IGNORE_INDEX, reduction="sum").item()
                count += int(m.sum())
    finally:
        model.train(was_training)
    assert count == ds.n_scored
    return total / count


# ------------------------------------------------------------ checkpoints

def save_checkpoint(path, model, optimizer, step, loader, cfg, tokenizer_sha256) -> Path:
    """Write the full training state atomically (temp file + os.replace).

    Contents are tensors, primitives, lists and dicts only, so the file loads
    with ``torch.load(weights_only=True)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "rng_cpu": torch.get_rng_state(),
        "loader_state": loader.state_dict(),
        "config": cfg.to_dict(),
        "tokenizer_sha256": tokenizer_sha256,
        # str(): torch.__version__ is a TorchVersion object, which weights_only=True rejects
        "torch_version": str(torch.__version__),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # a crash mid-write never leaves a truncated checkpoint at `path`
    return path


def load_checkpoint(path, model, optimizer=None) -> dict:
    """Load a ``save_checkpoint`` file into ``model`` (and ``optimizer``); return the dict.

    The caller restores ``rng_cpu`` and ``loader_state`` (``train`` does both).
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {state.get('format')!r}")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


# ------------------------------------------------------------------ train

def train(cfg: TrainConfig, model: GPT | None = None, start_checkpoint=None,
          max_steps: int | None = None, callback=None) -> dict:
    """Run (or continue) pretraining on the train split; return history and final state.

    start_checkpoint: path written by ``save_checkpoint``; restores model,
        optimizer, step, global CPU RNG and loader position.
    max_steps: stop once this many optimizer steps are done (the schedule still
        uses ``cfg.steps``), e.g. to checkpoint half way.
    callback(step, record, model, optimizer): called after every optimizer step;
        gradients (post-clip) are still attached.
    """
    torch.manual_seed(cfg.seed)  # model init; the loader has its own generators
    splits = encode_all_splits()
    train_tokens, val_tokens = splits["train"], splits["val"]
    tok_sha = sha256_file(TOKENIZER_PATH)
    T = cfg.model.block_size

    if model is None:
        model = GPT(cfg.model)
    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps)
    loader = make_train_loader(train_tokens, T, cfg.batch_size * cfg.accum, cfg.seed)
    step = 0
    if start_checkpoint is not None:
        state = load_checkpoint(start_checkpoint, model, optimizer)
        step = state["step"]
        torch.set_rng_state(state["rng_cpu"])
        loader.load_state_dict(state["loader_state"])
        if state["tokenizer_sha256"] != tok_sha:
            raise ValueError("checkpoint was trained with a different tokenizer")
    end = cfg.steps if max_steps is None else min(max_steps, cfg.steps)

    history, train_losses, lrs = [], [], []

    def eval_point(at_step, grad_norm, lr):
        val = evaluate_split(model, val_tokens, T, cfg.eval_batch_size)
        recent = train_losses[-cfg.eval_every:]
        history.append({"step": at_step,
                         "train_loss": sum(recent) / len(recent) if recent else None,
                         "val_loss": val, "lr": lr, "grad_norm": grad_norm})

    if step == 0 and cfg.eval_every:
        eval_point(0, None, lr_at(0, cfg))

    model.train()
    train_seconds, last = 0.0, {}
    while step < end:
        t0 = time.perf_counter()
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:   # set by hand: no scheduler object to checkpoint
            group["lr"] = lr
        x, y = next(loader)                     # (B * accum, T): one global batch per step
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for xm, ym in zip(x.chunk(cfg.accum), y.chunk(cfg.accum)):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16_autocast):
                _, loss = model(xm, ym)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
            (loss / cfg.accum).backward()      # mean over micro-batches = mean over the global batch
            step_loss += loss.item() / cfg.accum
        max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))  # pre-clip norm
        optimizer.step()
        train_seconds += time.perf_counter() - t0
        step += 1
        train_losses.append(step_loss)
        lrs.append(lr)
        last = {"step": step, "loss": step_loss, "lr": lr, "grad_norm": grad_norm}
        if callback is not None:
            callback(step, last, model, optimizer)
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps):
            eval_point(step, grad_norm, lr)

    steps_run = len(train_losses)
    return {
        "model": model, "optimizer": optimizer, "loader": loader, "step": step,
        "history": history, "train_losses": train_losses, "lrs": lrs, "last": last,
        "tokenizer_sha256": tok_sha,
        "train_seconds": train_seconds,
        "tokens_per_second": steps_run * cfg.tokens_per_step / train_seconds if train_seconds else None,
    }


# ------------------------------------------------- the committed checkpoint

def load_pretrained(device="cpu"):
    """Load the committed checkpoint: returns ``(GPT in eval mode, ByteBPE, manifest)``.

    Verifies the sha256 of ``gpt_sherlock.pt`` and of ``bpe_sherlock_512.json``
    against ``gpt_sherlock.json`` before loading, builds ``GPTConfig(**manifest["config"])``
    and loads the fp32 state_dict with ``weights_only=True`` and ``strict=True``.
    """
    manifest = json.loads(Path(ARTIFACT_JSON).read_text(encoding="utf-8"))
    for path, key in ((ARTIFACT_PT, "checkpoint_sha256"), (TOKENIZER_PATH, "tokenizer_sha256")):
        got = sha256_file(path)
        if got != manifest[key]:
            raise ValueError(f"{Path(path).name}: sha256 {got} != manifest {key} {manifest[key]}")
    model = GPT(GPTConfig(**manifest["config"]))
    state = torch.load(ARTIFACT_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    tokenizer = ByteBPE.load(TOKENIZER_PATH)
    return model.to(device).eval(), tokenizer, manifest
