"""Export, quantize and serve the committed capstone GPT (module 38).

export:      torch.export with dynamic (B, T), .pt2 save/load, logits and greedy
             generation compared with eager.
quantize:    int8 weight-only (per-output-channel, tied head kept fp32): state_dict
             bytes, val NLL on story XI (stride 128) against the fp32 checkpoint,
             teacher-forced greedy agreement over 64 tokens, per-layer weight error.
serve-check: the pure complete() contract.
all:         the three above (default). --try-ao-dynamic adds the deprecated
             torch.ao.quantization.quantize_dynamic comparison when an engine exists.

Run from the repository root:
    python content/courses/pytorch/code/38_export_quantize.py all --output-dir /tmp/gpt-export
    python content/courses/pytorch/code/38_export_quantize.py quantize --try-ao-dynamic --json-out /tmp/q.json
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

from llm.data import encode_all_splits  # noqa: E402
from llm.export import (UPSTREAM, ExportedLM, complete, export_gpt, forward_ms,  # noqa: E402
                        greedy_agreement, greedy_generate, load_exported, quantize_int8_weight_only,
                        save_exported, state_dict_bytes, try_ao_dynamic, val_nll,
                        weight_reconstruction_errors)
from llm.train import load_pretrained  # noqa: E402

COMMANDS = ("all", "export", "quantize", "serve-check")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", nargs="?", default="all", choices=COMMANDS)
    p.add_argument("--output-dir", default=None, help="keep gpt.pt2 and gpt_int8.pt here")
    p.add_argument("--prompt", default="Sherlock Holmes")
    p.add_argument("--tokens", type=int, default=64, help="tokens for the agreement check")
    p.add_argument("--try-ao-dynamic", action="store_true")
    p.add_argument("--bench", action="store_true", help="also time one T=128 forward per variant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def run_export(model, tokenizer, out_dir: Path, prompt: str, bench: bool) -> dict:
    t0 = time.perf_counter()
    ep = export_gpt(model, max_len=model.config.block_size)
    seconds = time.perf_counter() - t0
    path = save_exported(ep, out_dir / "gpt.pt2")
    t0 = time.perf_counter()
    exported = ExportedLM(load_exported(path), model.config)
    load_seconds = time.perf_counter() - t0
    gen = torch.Generator().manual_seed(38)
    diffs = {}
    for T in (2, 17, 128):
        x = torch.randint(0, model.config.vocab_size, (2, T), generator=gen)
        diffs[T] = (model(x)[0] - exported(x)[0]).abs().max().item()
    try:
        exported.gm(torch.zeros(1, model.config.block_size + 1, dtype=torch.long))
        rejects_129 = False
    except Exception:
        rejects_129 = True
    idx = torch.tensor([tokenizer.encode(prompt)])
    eager_ids = greedy_generate(model, idx, 32)
    exp_ids = greedy_generate(exported, idx, 32)
    sig = ep.graph_signature
    res = {"seconds": round(seconds, 3), "load_seconds": round(load_seconds, 3),
           "nodes": len(ep.graph.nodes),
           "nodes_core_aten": len(ep.run_decompositions().graph.nodes),
           "n_parameter_inputs": len(sig.parameters), "user_inputs": list(sig.user_inputs),
           "range_constraints": {str(k): str(v) for k, v in ep.range_constraints.items()},
           "pt2_bytes": path.stat().st_size, "state_dict_bytes": state_dict_bytes(model),
           "max_logit_diff": max(diffs.values()), "max_logit_diff_by_T": diffs,
           "rejects_T_129": rejects_129,
           "generation_equal": bool(torch.equal(eager_ids, exp_ids)),
           "generated_text": tokenizer.decode(exp_ids[0, idx.size(1):].tolist())}
    if bench:
        x = torch.randint(0, model.config.vocab_size, (1, model.config.block_size), generator=gen)
        res["ms_eager"] = forward_ms(model, x)
        res["ms_exported"] = forward_ms(exported, x)
    return res, exported


@torch.no_grad()
def run_quantize(model, tokenizer, out_dir: Path, prompt: str, steps: int, ao: bool, bench: bool) -> dict:
    val = encode_all_splits(tokenizer)["val"]
    block = model.config.block_size
    qmodel = quantize_int8_weight_only(model)
    torch.save(qmodel.state_dict(), out_dir / "gpt_int8.pt")
    errors = weight_reconstruction_errors(model, qmodel)
    nll_fp32 = val_nll(model, val, block)
    nll_int8 = val_nll(qmodel, val, block)
    idx = torch.tensor([tokenizer.encode(prompt)])
    res = {"fp32_bytes": state_dict_bytes(model), "int8_bytes": state_dict_bytes(qmodel),
           "val_nll_fp32": nll_fp32, "val_nll_int8": nll_int8, "val_nll_delta": nll_int8 - nll_fp32,
           "agreement": greedy_agreement(model, qmodel, idx, steps),
           "free_run_equal": bool(torch.equal(greedy_generate(model, idx, steps),
                                              greedy_generate(qmodel, idx, steps))),
           "weight_rel_err_max": max(r["rel_err"] for r in errors),
           "weight_rel_err_mean": sum(r["rel_err"] for r in errors) / len(errors),
           "n_quantized_linears": len(errors),
           "head_still_tied": qmodel.lm_head.weight is qmodel.wte.weight}
    bench_x = None
    if bench:
        bench_x = torch.randint(0, model.config.vocab_size, (1, block),
                                generator=torch.Generator().manual_seed(1))
        res["ms_int8_dequant"] = forward_ms(qmodel, bench_x)
    ao_res = None
    if ao:
        ao_res = try_ao_dynamic(model, val, block, bench_input=bench_x)
        if ao_res is not None:
            ao_res["val_nll_delta"] = ao_res["val_nll"] - nll_fp32
    return res, ao_res, qmodel


def run_serve_check(model, tokenizer, prompt: str, seed: int) -> dict:
    greedy = complete(model, tokenizer, prompt, max_new_tokens=32, temperature=0.0)
    sampled = complete(model, tokenizer, prompt, max_new_tokens=32, temperature=0.8, seed=seed)
    near_limit = complete(model, tokenizer, prompt, max_new_tokens=500, temperature=0.0)
    return {"text": greedy["text"], "n_tokens": greedy["n_tokens"], "stop_reason": greedy["stop_reason"],
            "model_sha256": greedy["model_sha256"], "sampled_text": sampled["text"],
            "capped": {k: near_limit[k] for k in ("n_tokens", "stop_reason", "prompt_tokens")}}


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    model, tokenizer, manifest = load_pretrained()
    result = {"command": args.command, "upstream": dict(UPSTREAM), "export": None,
              "quantize": None, "ao_dynamic": None, "serve": None,
              "manifest_val_loss": manifest["final_val_loss"], "torch": str(torch.__version__)}
    tmp = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        out_dir = Path(tmp.name)
    try:
        if args.command in ("all", "export"):
            result["export"], _ = run_export(model, tokenizer, out_dir, args.prompt, args.bench)
        if args.command in ("all", "quantize"):
            result["quantize"], result["ao_dynamic"], _ = run_quantize(
                model, tokenizer, out_dir, args.prompt, args.tokens, args.try_ao_dynamic, args.bench)
        if args.command in ("all", "serve-check"):
            result["serve"] = run_serve_check(model, tokenizer, args.prompt, args.seed)
    finally:
        if tmp is not None:
            tmp.cleanup()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
