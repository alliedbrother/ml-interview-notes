"""Assemble the capstone GPT and audit it (module 32).

Builds GPT(GPTConfig()) from llm/model.py, prints its parameter table and a
shape trace, counts FLOPs for one forward+backward with FlopCounterMode against
the 6*N*tokens + attention rule, checks the initial loss against ln V, compares
the SDPA and manual attention paths, and overfits one 4x64 batch.

Run from the repository root:
    python content/courses/pytorch/code/32_gpt_model.py --trace
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm.model import GPT, GPTConfig  # noqa: E402


def init_batch(cfg, source, generator, batch=8):
    """(x, y) of shape (batch, block_size) for the ln V check: real val tokens if possible."""
    if source == "auto":
        try:  # optional: module 31's pipeline; this lab must not depend on it
            from llm.data import load_split_tokens
            from llm.tokenizer import ByteBPE
            tok = ByteBPE.load(HERE / "data" / "bpe_sherlock_512.json")
            val = load_split_tokens("val", tok)
            starts = torch.randint(0, len(val) - cfg.block_size - 1, (batch,), generator=generator)
            x = torch.stack([val[s:s + cfg.block_size] for s in starts.tolist()])
            y = torch.stack([val[s + 1:s + 1 + cfg.block_size] for s in starts.tolist()])
            return x, y, "val"
        except Exception:  # noqa: BLE001 - fall back to random ids
            pass
    ids = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size + 1), generator=generator)
    return ids[:, :-1], ids[:, 1:], "random"


def flop_audit(model, x, y):
    """FlopCounterMode totals for one forward+backward, manual and SDPA attention paths."""
    cfg = model.config
    B, T = x.shape
    tokens = B * T
    out = {}
    for impl in ("manual", "sdpa"):
        model.zero_grad(set_to_none=True)
        with FlopCounterMode(display=False) as counter:
            _, loss = model(x, y, attn_impl=impl)
            loss.backward()
        out[impl] = {str(op): int(n) for op, n in counter.get_flop_counts()["Global"].items()}
        out[impl]["total"] = int(counter.get_total_flops())
    model.zero_grad(set_to_none=True)
    n_nonemb = model.num_params(non_embedding=True)
    flops_6p = 6 * n_nonemb * tokens
    attn_term = 12 * cfg.n_layer * T * cfg.d_model * tokens  # QK^T and AV, fwd+bwd, no causal discount
    return {"tokens": tokens, "flops_measured": out["manual"]["total"],
            "flops_measured_sdpa": out["sdpa"]["total"], "flops_6P": flops_6p,
            "flops_attention_term": attn_term, "flops_formula": flops_6p + attn_term,
            "by_op_manual": out["manual"], "by_op_sdpa": out["sdpa"]}


def overfit(model, x, y, steps, lr=3e-3):
    opt = model.configure_optimizer(weight_decay=0.0, lr=lr, betas=(0.9, 0.95))
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", action="store_true", help="print the parameter table and shape trace")
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--init-batch", choices=("auto", "random"), default="auto",
                        help="auto = val tokens via llm.data when available, else random ids")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    cfg = GPTConfig()
    model = GPT(cfg)
    table = model.param_table()
    trace = model.shape_trace(torch.randint(0, cfg.vocab_size, (2, 16), generator=gen))

    x, y, source = init_batch(cfg, args.init_batch, gen)
    with torch.no_grad():
        logits_sdpa, init_loss = model(x, y, attn_impl="sdpa")
        logits_manual, _ = model(x, attn_impl="manual")
    max_diff = (logits_sdpa - logits_manual).abs().max().item()

    ids = torch.randint(0, cfg.vocab_size, (4, 65), generator=gen)
    bx, by = ids[:, :-1], ids[:, 1:]
    flops = flop_audit(model, bx, by)
    losses = overfit(model, bx, by, args.overfit_steps)

    result = {
        "params": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True),
        "param_table": table,
        "shape_trace": [[name, list(shape)] for name, shape in trace],
        "init_loss": init_loss.item(),
        "init_loss_source": source,
        "ln_vocab": round(math.log(cfg.vocab_size), 3),
        "sdpa_manual_max_diff": max_diff,
        "flops_measured": flops["flops_measured"],
        "flops_measured_sdpa": flops["flops_measured_sdpa"],
        "flops_6P": flops["flops_6P"],
        "flops_attention_term": flops["flops_attention_term"],
        "flops_formula": flops["flops_formula"],
        "flops_by_op_manual": flops["by_op_manual"],
        "flops_by_op_sdpa": flops["by_op_sdpa"],
        "overfit_steps": args.overfit_steps,
        "overfit_first_loss": losses[0] if losses else None,
        "overfit_final_loss": losses[-1] if losses else None,
    }
    if args.trace:
        print(f"{'module':24s} {'numel':>8s}  shapes", file=sys.stderr)
        for row in table:
            note = f"  (tied to {row['tied_to']})" if "tied_to" in row else ""
            print(f"{row['name']:24s} {row['numel']:8,d}  {row['shapes']}{note}", file=sys.stderr)
        print(f"{'total':24s} {result['params']:8,d}\n", file=sys.stderr)
        for name, shape in trace:
            print(f"{name:34s} {tuple(shape)}", file=sys.stderr)
        print(file=sys.stderr)
    if args.json_out is not None:
        with tempfile.NamedTemporaryFile("w", dir=args.json_out.parent, delete=False) as handle:
            json.dump(result, handle, indent=2)
        os.replace(handle.name, args.json_out)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
