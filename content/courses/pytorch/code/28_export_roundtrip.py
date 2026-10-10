"""Export a small text classifier with torch.export, reload it, and stress-test its contract.

Steps, all on CPU and all in one process:

1. Build ``TextClassifier``: Embedding (padding id 0) -> masked mean-pool ->
   two-layer MLP head -> logits. A ``torch.cond`` picks the pooled vector when
   the batch has at least one real token and a zero vector otherwise (the
   "empty input" case), so both branches live in the exported graph.
2. Export with ``Dim("B", min=1, max=32)`` and ``Dim("T", min=2, max=64)``,
   preserving the ``head`` call signature so it can be swapped later.
3. ``torch.export.save`` to ``model.pt2`` plus ``manifest.json`` (input spec,
   dims, torch version, byte size, SHA-256 of the file).
4. ``torch.export.load`` and compare against eager over a grid of shapes,
   including the empty-input branch.
5. ``run_decompositions()`` and record the op histogram before and after.
6. Feed contract violations (T out of range, float token ids) and record the
   exception type each raised.
7. Time eager vs exported forward with ``torch.utils.benchmark`` (reported only).
8. ``torch.export.unflatten`` and swap the classifier head.
9. ``draft_export`` a variant that branches on ``.item()`` and count the
   reported issues.

Run from the repository root:

    python content/courses/pytorch/code/28_export_roundtrip.py --output-dir /tmp/pt-export
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim, draft_export, export
from torch.utils import benchmark

V, D_MODEL, D_FF, N_CLASSES = 50, 16, 32, 3
B_MAX, T_MIN, T_MAX = 32, 2, 64
GRID = [(1, 2), (1, 64), (7, 13), (5, 31), (32, 2), (32, 64)]


class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_MODEL, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.ReLU(), nn.Linear(D_FF, N_CLASSES))

    def forward(self, ids):                                    # ids: (B, T) int64, 0 = padding
        mask = (ids != 0).to(self.emb.weight.dtype)            # (B, T)
        summed = (self.emb(ids) * mask.unsqueeze(-1)).sum(1)   # (B, d_model)
        pooled = summed / mask.sum(1, keepdim=True).clamp_min(1.0)
        pooled = torch.cond(
            mask.sum() > 0,
            lambda p: p.clone(),                               # normal batch
            lambda p: torch.zeros_like(p),                     # all padding: logits = head(0)
            (pooled,),
        )
        return self.head(pooled)                               # (B, N_CLASSES)


class ItemBranchClassifier(TextClassifier):
    """Same model, but the empty check is a Python branch on ``.item()``."""

    def forward(self, ids):
        mask = (ids != 0).to(self.emb.weight.dtype)
        n_real = mask.sum().long().item()                      # data-dependent Python int
        pooled = (self.emb(ids) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        if n_real == 0:
            pooled = torch.zeros_like(pooled)
        return self.head(pooled)


def make_ids(gen, b, t, empty=False):
    ids = torch.randint(1, V, (b, t), generator=gen)
    lengths = torch.randint(1, t + 1, (b,), generator=gen)
    ids = ids.masked_fill(torch.arange(t) >= lengths[:, None], 0)  # right padding
    return torch.zeros_like(ids) if empty else ids


def op_histogram(ep):
    counts = collections.Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
    return dict(sorted(counts.items()))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejection(fn):
    try:
        fn()
    except Exception as err:  # the contract violation we expect
        return f"{type(err).__name__}: {str(err).splitlines()[0][:100]}"
    return ""


def run(seed, out_dir):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TextClassifier().eval()

    example = (make_ids(gen, 4, 8),)
    dynamic_shapes = {"ids": {0: Dim("B", min=1, max=B_MAX), 1: Dim("T", min=T_MIN, max=T_MAX)}}
    ep = export(model, example, dynamic_shapes=dynamic_shapes, preserve_module_call_signature=("head",))

    pt2 = out_dir / "model.pt2"
    torch.export.save(ep, pt2)
    manifest = {
        "artifact": pt2.name,
        "bytes": pt2.stat().st_size,
        "sha256": sha256(pt2),
        "torch": torch.__version__,
        "inputs": [{"name": "ids", "dtype": "torch.int64", "shape": ["B", "T"]}],
        "dims": {"B": [1, B_MAX], "T": [T_MIN, T_MAX]},
        "range_constraints": {str(k): str(v) for k, v in ep.range_constraints.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    loaded = torch.export.load(pt2).module()
    max_diff, shapes_checked = 0.0, 0
    with torch.no_grad():
        for b, t, empty in [(b, t, False) for b, t in GRID] + [(3, 5, True)]:
            ids = make_ids(gen, b, t, empty=empty)
            diff = (loaded(ids) - model(ids)).abs().max().item()
            max_diff, shapes_checked = max(max_diff, diff), shapes_checked + 1

    ops_training = op_histogram(ep)
    ops_core = op_histogram(ep.run_decompositions())

    rejections = {
        "T_out_of_range": rejection(lambda: loaded(make_ids(gen, 2, T_MAX + 1))),
        "wrong_dtype": rejection(lambda: loaded(make_ids(gen, 2, 8).float())),
    }

    ids = make_ids(gen, 8, 32)
    with torch.inference_mode():
        timings = {}
        for name, fn in (("eager", model), ("exported", loaded)):
            t = benchmark.Timer("fn(ids)", globals={"fn": fn, "ids": ids}, num_threads=torch.get_num_threads())
            timings[name] = t.timeit(200).median * 1e6

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attempted to insert a get_attr Node")  # torch-internal, benign
        unflat = torch.export.unflatten(ep)
    new_head = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.ReLU(), nn.Linear(D_FF, N_CLASSES)).eval()
    unflat.head = new_head
    model.head, old_head = new_head, model.head
    with torch.no_grad():
        unflatten_ok = bool(torch.allclose(unflat(ids), model(ids), atol=1e-6, rtol=0))
    model.head = old_head

    loggers = [logging.getLogger(n) for n in ("torch.export._draft_export", "torch.fx.experimental.symbolic_shapes")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        variant = ItemBranchClassifier().eval()
        variant.load_state_dict(model.state_dict())
        draft = draft_export(variant, example)
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)

    return {
        "pt2_bytes": manifest["bytes"],
        "sha256": manifest["sha256"],
        "max_diff": max_diff,
        "shapes_checked": shapes_checked,
        "ops_training_ir": ops_training,
        "ops_core_aten": ops_core,
        "rejections": rejections,
        "unflatten_ok": unflatten_ok,
        "draft_export_issues": len(draft._report.failures),
        "eager_us": round(timings["eager"], 2),
        "exported_us": round(timings["exported"], 2),
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run(args.seed, out_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(args.seed, Path(tmp))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
