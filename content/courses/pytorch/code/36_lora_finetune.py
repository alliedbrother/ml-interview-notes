"""Fine-tune the pretrained capstone GPT on templated Q/A, with LoRA and/or fully (module 36).

Loads the committed checkpoint (llm.train.load_pretrained), measures exact match
on the held-out template C and the val NLL on story XI (forgetting), fine-tunes
with prompt-masked loss, re-measures, saves the LoRA adapter alone, reloads it
into a fresh base model and checks merged vs unmerged logits.

Run from the repository root:
    python content/courses/pytorch/code/36_lora_finetune.py --method both
    python content/courses/pytorch/code/36_lora_finetune.py --method lora --rank 4 --json-out /tmp/lora.json
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm import lora, sft  # noqa: E402
from llm.data import encode_all_splits  # noqa: E402
from llm.train import load_pretrained  # noqa: E402

MAX_LEN = 48
BATCH_SIZE = 16
EVAL_STRIDE = 128


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=("lora", "full", "both"), default="both")
    p.add_argument("--steps", type=int, default=150, help="LoRA steps")
    p.add_argument("--full-steps", type=int, default=None, help="full fine-tune steps (default --steps)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lr", type=float, default=3e-3, help="LoRA learning rate")
    p.add_argument("--full-lr", type=float, default=1e-3, help="full fine-tune learning rate")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-prompt-mask", action="store_true",
                   help="ablation: also train on prompt tokens (labels = every next token)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-dir", default=None, help="keep adapter.pt here (default: temp dir, removed)")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def val_nll(model, val_tokens) -> float:
    """Mean NLL on story XI, non-overlapping 128-token windows (every target once)."""
    try:
        from llm.evaluate import evaluate_nll  # module 34
        return float(evaluate_nll(model, val_tokens, model.config.block_size, EVAL_STRIDE, 32)["nll_per_token"])
    except ImportError:  # LOCAL FALLBACK: module 33's evaluator, identical at stride = block_size
        from llm.train import evaluate_split
        return float(evaluate_split(model, val_tokens, model.config.block_size))


def optimizer_state_bytes(opt) -> int:
    """Bytes of every tensor in optimizer.state (AdamW: exp_avg, exp_avg_sq, step)."""
    return sum(t.numel() * t.element_size() for s in opt.state.values()
               for t in s.values() if torch.is_tensor(t))


def finetune(model, data, steps, lr, batch_size, seed, pad_id):
    """Plain SFT loop: AdamW (no decay), constant lr, clip 1.0, batches drawn by a private Generator."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=True)
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        rows = torch.randint(len(data), (batch_size,), generator=g).tolist()
        x, y = sft.collate([data[i] for i in rows], pad_id)
        _, loss = model(x, targets=y)                     # mean over answer tokens only
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return {"seconds": time.perf_counter() - t0, "first_loss": losses[0], "last_loss": losses[-1],
            "opt_state_bytes": optimizer_state_bytes(opt), "losses": losses}


def measure(model, tok, heldout, train_ex, val_tokens):
    return {"em": sft.answer_exact_match(model, tok, heldout),
            "train_em": sft.answer_exact_match(model, tok, train_ex),
            "val_nll": val_nll(model, val_tokens)}


def base_key(key: str) -> str:
    """Original state_dict key -> its key after wrapping (c_attn.weight -> c_attn.base.weight)."""
    head, _, leaf = key.rpartition(".")
    return f"{head}.base.{leaf}"


def run_lora(args, tok, data, heldout, train_ex, val_tokens, out_dir):
    model, _, _ = load_pretrained()
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    probe, _ = sft.collate(data[:16], tok.eot_id)                   # (16, T) real prompts + answers
    with torch.no_grad():
        ref = model(probe)[0]
    replaced = lora.apply_lora(model, ("c_attn", "c_proj"), r=args.rank, alpha=args.alpha)
    lora.mark_only_lora_trainable(model)
    with torch.no_grad():
        init_diff = (model(probe)[0] - ref).abs().max().item()     # B == 0 -> exactly 0
    trainable = lora.count_trainable(model)
    fit = finetune(model, data, args.steps, args.lr, args.batch_size, args.seed, tok.eot_id)
    res = measure(model, tok, heldout, train_ex, val_tokens)

    sd = model.state_dict()                                         # base weights must be untouched
    base_change = max((sd[base_key(k) if base_key(k) in sd else k] - v).abs().max().item()
                      for k, v in base_sd.items())

    path = Path(out_dir) / "adapter.pt"                             # adapter-only checkpoint
    torch.save(lora.lora_state_dict(model), path)
    fresh, _, _ = load_pretrained()
    lora.apply_lora(fresh, ("c_attn", "c_proj"), r=args.rank, alpha=args.alpha)
    keys = lora.load_lora_state_dict(fresh, torch.load(path, weights_only=True))
    with torch.no_grad():
        unmerged = model(probe)[0]
        reload_diff = (fresh(probe)[0] - unmerged).abs().max().item()
        lora.merge_all(model)
        merged = model(probe)[0]
        lora.unmerge_all(model)
        roundtrip = model(probe)[0]
    return model, dict(res, trainable=trainable, replaced=replaced, init_max_diff=init_diff,
                       base_max_change=base_change, adapter_bytes=path.stat().st_size,
                       adapter_tensors=len(lora.lora_state_dict(model)),
                       reload_max_diff=reload_diff, reload_missing_keys=len(keys.missing_keys),
                       reload_unexpected_keys=len(keys.unexpected_keys),
                       merge_max_diff=(merged - unmerged).abs().max().item(),
                       unmerge_roundtrip_max_diff=(roundtrip - unmerged).abs().max().item(),
                       logit_absmax=unmerged.abs().max().item(),
                       delta_to_base_ratio=max((m.delta_weight().abs().max() / m.base.weight.abs().max()).item()
                                               for _, m in lora.lora_modules(model)),
                       seconds=fit["seconds"], first_loss=fit["first_loss"], last_loss=fit["last_loss"],
                       opt_state_bytes=fit["opt_state_bytes"])


def run_full(args, tok, data, heldout, train_ex, val_tokens):
    model, _, _ = load_pretrained()
    model.requires_grad_(True)
    steps = args.full_steps if args.full_steps is not None else args.steps
    fit = finetune(model, data, steps, args.full_lr, args.batch_size, args.seed, tok.eot_id)
    res = measure(model, tok, heldout, train_ex, val_tokens)
    return dict(res, trainable=lora.count_trainable(model), steps=steps, seconds=fit["seconds"],
                first_loss=fit["first_loss"], last_loss=fit["last_loss"],
                opt_state_bytes=fit["opt_state_bytes"])


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    base, tok, _ = load_pretrained()
    val_tokens = encode_all_splits(tok)["val"]
    train_ex, heldout = sft.build_examples("train"), sft.build_examples("heldout")
    data = [sft.encode_example(tok, p, a, MAX_LEN) for p, a in train_ex]
    if args.no_prompt_mask:  # labels = input shifted by one everywhere (ids[1:])
        data = [(x, torch.cat([x[1:], y[-1:]])) for x, y in data]
    result = {"base": {**measure(base, tok, heldout, train_ex, val_tokens), "params": base.num_params()},
              "lora": None, "full": None, "merge_max_diff": None,
              "generation_backend": sft.generation_backend(), "steps": args.steps,
              "rank": args.rank, "alpha": args.alpha, "prompt_mask": not args.no_prompt_mask, "n_train": len(train_ex), "n_heldout": len(heldout)}
    del base
    tmp = None
    out_dir = args.output_dir
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory()
        out_dir = tmp.name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        if args.method in ("lora", "both"):
            _, res = run_lora(args, tok, data, heldout, train_ex, val_tokens, out_dir)
            result["lora"] = res
            result["merge_max_diff"] = res["merge_max_diff"]
        if args.method in ("full", "both"):
            result["full"] = run_full(args, tok, data, heldout, train_ex, val_tokens)
    finally:
        if tmp is not None:
            tmp.cleanup()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
