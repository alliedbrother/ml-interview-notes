#!/usr/bin/env python3
"""Lab 06 — Quantization: what it costs in latency, what it buys in capacity,
and what you pay for both in accuracy.

Three subcommands.

  plan      Memory and capacity arithmetic for a model at several precisions.
            No GPU, no network, no third-party packages. This is the axis you
            can predict, so predict it before you measure anything.

  latency   Batch-1 (and swept-batch) latency against a running server, over
            the OpenAI-compatible endpoint. stdlib urllib only. Writes a JSON
            row with --save so `report` can merge arms.

  report    Merge saved rows into the comparison table. Accuracy cells are
            printed as FILL IN unless you pass a measured value with
            --accuracy LABEL=VALUE. The script CANNOT emit an accuracy number
            it was not given, and that is deliberate: quantization accuracy is
            the one figure nobody may borrow.

    python3 run.py plan --params 8.03e9 --layers 32 --kv-heads 8 --head-dim 128 --vram 24
    python3 run.py plan --config ./config.json --vram 80 --tp 2
    python3 run.py latency --engine vllm --base-url http://127.0.0.1:8000 --label bf16 --save bf16.json
    python3 run.py latency --engine sglang --base-url http://127.0.0.1:30000 --label fp8 --save fp8.json
    python3 run.py report bf16.json fp8.json --accuracy bf16=0.781 --accuracy fp8=0.769

Flags this lab tells you to use, verified present at the SHAs this book pins:

  vLLM    --quantization {fp8, fp8_per_tensor, fp8_per_channel, awq_marlin, gptq_marlin, ...}
              vllm/model_executor/layers/quantization/__init__.py:L12-L46
          --kv-cache-dtype {auto, fp8, fp8_e4m3, fp8_e5m2, ...}
              vllm/engine/arg_utils.py:L1228
          FP8 CUTLASS GEMM is gated on SM89 + CUDA 12.4, or SM90 + CUDA 12.0:
              csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:L145-L159

  SGLang  --quantization {fp8, w8a8_fp8, awq_marlin, gptq_marlin, ...}
              python/sglang/srt/server_args.py:L143-L164
          --kv-cache-dtype {auto, bf16, fp8_e5m2, fp8_e4m3, ...}
              python/sglang/srt/server_args.py:L690-L701

Accuracy is measured with the engines' own harnesses, never with this script:
  vLLM    python tests/evals/gsm8k/gsm8k_eval.py --port 8000
  SGLang  python3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k

NOTHING IN THIS FILE WAS RUN AGAINST A GPU. `plan` output is arithmetic. The
latency path was written against the request schemas cited in the README and has
not been executed against a live engine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GIB = 1024 ** 3

# Bytes per stored element. Weight bytes and KV bytes are separate knobs in both
# engines and this lab insists on treating them separately.
WEIGHT_BYTES = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
    "int4": 0.5, "awq": 0.5, "gptq": 0.5, "nvfp4": 0.5, "mxfp4": 0.5,
}
KV_BYTES = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "nvfp4": 0.5,
}


# ---------------------------------------------------------------- plan


def shape_from_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg = cfg.get("text_config", cfg)
    layers = (cfg.get("num_hidden_layers") or cfg.get("n_layer")
              or cfg.get("num_layers"))
    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", heads)
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim") or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        sys.exit(f"{path} is missing layer count, kv head count, or head dim.")
    if cfg.get("kv_lora_rank"):
        sys.exit("This config is MLA. Its KV footprint does not follow the GQA "
                 "formula; use lab 02's calculator, which handles it.")
    return {"layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
            "params": cfg.get("num_parameters")}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int,
                       elem: float) -> float:
    """One K entry and one V entry per KV head per layer. See FORMULAS."""
    return 2 * layers * kv_heads * head_dim * elem


def plan_row(name: str, params: float, w_elem: float, kv_elem: float,
             shape: dict, vram_gib: float, util: float, tp: int,
             overhead_gib: float, context_len: int) -> dict:
    weights_gib = params * w_elem / GIB / tp
    budget = vram_gib * util - overhead_gib
    kv_gib = budget - weights_gib
    per_token = kv_bytes_per_token(shape["layers"], shape["kv_heads"],
                                   shape["head_dim"], kv_elem) / tp
    tokens = int(kv_gib * GIB / per_token) if kv_gib > 0 and per_token else 0
    return {
        "config": name,
        "weights_gib": weights_gib,
        "kv_budget_gib": kv_gib,
        "kv_bytes_per_token": per_token,
        "tokens": tokens,
        "concurrency": tokens / context_len if context_len else 0.0,
    }


def cmd_plan(args) -> int:
    if args.config:
        shape = shape_from_config(args.config)
        params = args.params or shape.get("params")
    else:
        shape = {"layers": args.layers, "kv_heads": args.kv_heads,
                 "head_dim": args.head_dim}
        params = args.params
    if not params:
        sys.exit("Need --params (parameter count). A config.json rarely carries "
                 "it; read it off the model card.")
    if not (shape.get("layers") and shape.get("kv_heads") and shape.get("head_dim")):
        sys.exit("Need --layers, --kv-heads and --head-dim (or --config).")

    arms = [
        ("bf16 weights, bf16 KV", "bf16", "bf16"),
        ("fp8 weights,  bf16 KV", "fp8", "bf16"),
        ("fp8 weights,  fp8 KV", "fp8", "fp8"),
        ("int4 weights, bf16 KV", "int4", "bf16"),
        ("int4 weights, fp8 KV", "int4", "fp8"),
    ]
    rows = [plan_row(n, params, WEIGHT_BYTES[w], KV_BYTES[k], shape,
                     args.vram, args.util, args.tp, args.overhead_gib,
                     args.context_len)
            for n, w, k in arms]
    base = rows[0]["tokens"] or 1

    print(f"\nModel shape: L={shape['layers']}, h_kv={shape['kv_heads']}, "
          f"d_h={shape['head_dim']}, params={params:,.0f}")
    print(f"Budget: {args.vram:.1f} GiB x util {args.util} "
          f"- {args.overhead_gib:.1f} GiB reserved, tp={args.tp}, "
          f"context_len={args.context_len:,}")

    hdr = (f"{'configuration':<24}{'weights':>10}{'KV budget':>11}"
           f"{'B/token':>10}{'tokens':>12}{'vs bf16':>9}{'concurrency':>13}")
    print("\nDerived — arithmetic, not a measurement")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:<24}{r['weights_gib']:>9.2f}G"
              f"{r['kv_budget_gib']:>10.2f}G{r['kv_bytes_per_token']:>10,.0f}"
              f"{r['tokens']:>12,}{r['tokens'] / base:>8.2f}x"
              f"{r['concurrency']:>12.1f}x")
    print("-" * len(hdr))

    print("\nThings this arithmetic does not model, all of which reduce the real "
          "number:")
    print("  - activation and CUDA-graph memory the engine reserves after "
          "profiling")
    print("    (approximate it with --overhead-gib and re-run)")
    print("  - page-size padding: a partial block still occupies a whole page")
    print("  - anything the driver and the allocator hold that is not yours")
    print("Reconcile against the engine's own line — see lab 02:")
    print("  vLLM  : 'GPU KV cache size: N tokens, Maximum concurrency ... Xx'")
    print("  SGLang: 'max_total_num_tokens=N ... context_len=M'")
    print("\nCapacity is a RESIDUAL. The fractional gain from halving weights "
          "grows with\nhow much of the budget the weights were eating — which is "
          "why the same flag is\nworth 2x on an 8B at 24 GiB and over 5x on a "
          "70B at 2x80 GiB.")
    print("\nEvery number above is arithmetic from the shape you gave. Nothing "
          "was measured.")
    return 0


# ---------------------------------------------------------------- latency


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def resolve_model(args) -> str:
    if args.model:
        return args.model
    if args.engine != "vllm":
        return ""
    try:
        return json.loads(_get(args.base_url.rstrip("/") + "/v1/models",
                               args.timeout))["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not read /v1/models ({e}). Pass --model.")


def one_call(args, model: str, prompt_tokens: list[int]) -> float:
    """Wall-clock seconds for one non-streaming request. Returns e2e latency."""
    base = args.base_url.rstrip("/")
    t0 = time.perf_counter()
    if args.engine == "vllm":
        _post(base + "/v1/completions", {
            "model": model,
            "prompt": prompt_tokens,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        }, args.timeout)
    else:
        _post(base + "/generate", {
            "input_ids": prompt_tokens,
            "sampling_params": {
                "max_new_tokens": args.output_len,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }, args.timeout)
    return time.perf_counter() - t0


def environment(args) -> list[tuple[str, str]]:
    base = args.base_url.rstrip("/")
    rows = [("engine", args.engine), ("base url", base), ("label", args.label)]
    try:
        if args.engine == "vllm":
            rows.append(("vllm version", json.loads(
                _get(base + "/version", args.timeout)).get("version", "?")))
        else:
            info = json.loads(_get(base + "/server_info", args.timeout))
            for k in ("version", "model_path", "quantization", "kv_cache_dtype",
                      "max_total_num_tokens", "context_len"):
                if k in info:
                    rows.append((k, str(info[k])))
    except Exception as e:  # noqa: BLE001
        rows.append(("server info", f"unavailable ({e})"))
    return rows


def cmd_latency(args) -> int:
    model = resolve_model(args)
    rows = environment(args) + [("model", model or "(sglang default)")]
    w = max(len(k) for k, _ in rows)
    print("\nEnvironment — paste this with any result you report")
    print("-" * (w + 28))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")

    # A fixed synthetic prompt. Token ids, not text, so the input length is
    # exactly --input-len on every engine and every tokenizer.
    prompt = [(1000 + (i * 7919) % 19000) for i in range(args.input_len)]

    print(f"\nWarmup: {args.warmup} call(s)")
    try:
        for _ in range(args.warmup):
            one_call(args, model, prompt)

        samples = []
        for i in range(args.iters):
            samples.append(one_call(args, model, prompt))
            print(f"  iter {i + 1}/{args.iters}: {samples[-1] * 1000:.1f} ms")
    except urllib.error.HTTPError as e:
        sys.exit(f"Server returned HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach {args.base_url}: {e.reason}. Is the server up, "
                 f"and is --engine the right one for this port?")

    samples.sort()
    p = lambda q: samples[min(int(q * len(samples)), len(samples) - 1)]  # noqa: E731
    result = {
        "label": args.label,
        "engine": args.engine,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "iters": args.iters,
        "e2e_mean_ms": statistics.fmean(samples) * 1000,
        "e2e_p50_ms": p(0.50) * 1000,
        "e2e_p90_ms": p(0.90) * 1000,
        "tpot_mean_ms": (statistics.fmean(samples) * 1000 / args.output_len
                         if args.output_len else 0.0),
        "output_tok_per_s": (args.output_len / statistics.fmean(samples)
                             if samples else 0.0),
        "note": "e2e includes prefill; TPOT here is e2e/output_len, an upper "
                "bound on true per-token time. For a clean TPOT, stream and "
                "time the gaps, or use the engine's own bench.",
    }

    w = max(len(k) for k in result)
    print("\nMeasured on YOUR hardware — nothing about this was predicted")
    print("-" * (w + 40))
    for k, v in result.items():
        print(f"{k:<{w}}  {v if isinstance(v, str) else f'{v:,.3f}'}")

    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.save}. Merge arms with: "
              f"python3 run.py report {args.save} <other>.json")
    return 0


# ---------------------------------------------------------------- report


def cmd_report(args) -> int:
    acc: dict[str, str] = {}
    for spec in args.accuracy or []:
        if "=" not in spec:
            sys.exit(f"--accuracy wants LABEL=VALUE, got {spec!r}")
        k, v = spec.split("=", 1)
        acc[k.strip()] = v.strip()

    rows = []
    for path in args.results:
        try:
            rows.append(json.loads(Path(path).read_text()))
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not read {path}: {e}")
    if not rows:
        sys.exit("No result files given.")

    base_acc = None
    for r in rows:
        r["accuracy"] = acc.get(r["label"])
        if base_acc is None and r["accuracy"] is not None:
            try:
                base_acc = float(r["accuracy"])
            except ValueError:
                base_acc = None

    hdr = (f"{'arm':<14}{'engine':<9}{'e2e p50 ms':>12}{'TPOT ms':>10}"
           f"{'tok/s':>9}{'vs base':>9}{'accuracy':>11}{'delta':>9}")
    print("\nQuantization comparison")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    base_lat = rows[0]["e2e_p50_ms"]
    for r in rows:
        a = r["accuracy"]
        if a is None:
            a_cell, d_cell = "FILL IN", "FILL IN"
        else:
            a_cell = a
            try:
                d_cell = (f"{float(a) - base_acc:+.4f}"
                          if base_acc is not None else "FILL IN")
            except ValueError:
                d_cell = "FILL IN"
        print(f"{r['label']:<14}{r['engine']:<9}{r['e2e_p50_ms']:>12,.1f}"
              f"{r['tpot_mean_ms']:>10,.2f}{r['output_tok_per_s']:>9,.1f}"
              f"{base_lat / r['e2e_p50_ms']:>8.2f}x{a_cell:>11}{d_cell:>9}")
    print("-" * len(hdr))

    missing = [r["label"] for r in rows if r["accuracy"] is None]
    if missing:
        print(f"\n{len(missing)} arm(s) have no accuracy number: "
              f"{', '.join(missing)}")
        print("This script will not invent one. Measure it, then pass it in:")
        print("  vLLM  : python tests/evals/gsm8k/gsm8k_eval.py --port 8000 \\")
        print("            --num-questions 1319 --num-shots 5 --temperature 0")
        print("  SGLang: python3 -m sglang.test.run_eval --port 30000 \\")
        print("            --eval-name gsm8k --num-examples 1319")
        print("  then  : python3 run.py report *.json --accuracy LABEL=0.781")
    print("\nUse the full 1319-question set for anything you report. A "
          "200-question run\ncarries more sampling error than the effect you "
          "are trying to measure.")
    return 0


# ---------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Price quantization on three axes: latency, capacity, accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{plan,latency,report}")

    pp = sub.add_parser("plan", help="capacity arithmetic; no GPU, no network")
    src = pp.add_mutually_exclusive_group()
    src.add_argument("--config", help="path to a model config.json")
    pp.add_argument("--params", type=float, default=None,
                    help="parameter count, e.g. 8.03e9. Required; config.json "
                         "rarely carries it")
    pp.add_argument("--layers", type=int, default=32,
                    help="number of transformer layers (default: 32)")
    pp.add_argument("--kv-heads", type=int, default=8,
                    help="KV heads after GQA grouping (default: 8)")
    pp.add_argument("--head-dim", type=int, default=128,
                    help="per-head dimension (default: 128)")
    pp.add_argument("--vram", type=float, default=24.0,
                    help="GPU memory in GiB PER CARD (default: 24)")
    pp.add_argument("--util", type=float, default=0.92,
                    help="fraction the engine may use; vLLM's "
                         "--gpu-memory-utilization defaults to 0.92 at this "
                         "SHA (default: 0.92)")
    pp.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size; weights and KV both shard "
                         "(default: 1)")
    pp.add_argument("--overhead-gib", type=float, default=0.0,
                    help="GiB to reserve for activations and CUDA graphs "
                         "before sizing the cache (default: 0)")
    pp.add_argument("--context-len", type=int, default=8192,
                    help="tokens per request, for the concurrency column "
                         "(default: 8192)")
    pp.set_defaults(func=cmd_plan)

    lp = sub.add_parser("latency", help="batch-1 latency against a live server")
    lp.add_argument("--engine", choices=["vllm", "sglang"], required=True,
                    help="picks the request schema and the info endpoint")
    lp.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="server root (default: http://127.0.0.1:8000; "
                         "SGLang's default port is 30000)")
    lp.add_argument("--label", required=True,
                    help="name for this arm, e.g. bf16 / fp8 / fp8-kv / awq. "
                         "Used to join accuracy numbers in `report`")
    lp.add_argument("--model", default=None,
                    help="model id for vLLM; read from /v1/models if omitted")
    lp.add_argument("--input-len", type=int, default=1024,
                    help="prompt length in tokens (default: 1024)")
    lp.add_argument("--output-len", type=int, default=128,
                    help="tokens to generate (default: 128)")
    lp.add_argument("--warmup", type=int, default=3,
                    help="untimed calls before measuring (default: 3)")
    lp.add_argument("--iters", type=int, default=10,
                    help="timed calls (default: 10)")
    lp.add_argument("--timeout", type=float, default=300.0,
                    help="per-call timeout in seconds (default: 300)")
    lp.add_argument("--save", default=None,
                    help="write the result row to this JSON path")
    lp.set_defaults(func=cmd_latency)

    rp = sub.add_parser("report", help="merge saved arms into one table")
    rp.add_argument("results", nargs="+",
                    help="JSON files written by `latency --save`")
    rp.add_argument("--accuracy", action="append", metavar="LABEL=VALUE",
                    help="a MEASURED accuracy for one arm, e.g. bf16=0.781. "
                         "Repeatable. Omitted arms print FILL IN")
    rp.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
