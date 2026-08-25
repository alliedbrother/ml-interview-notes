#!/usr/bin/env python3
"""Lab 02 — KV cache sizing.

Compute the KV cache footprint of a model from its config alone, then reconcile
that arithmetic against what a serving engine actually reports at startup.

The gap between the two is the point of the lab. It is never zero, and the
reasons it is not zero (dtype override, MLA's single latent cache, a padded page
size, memory the engine reserves for activations) are the things worth knowing.

The arithmetic half needs no GPU and no network if you pass --config.

    python3 run.py --config ./llama3-8b-config.json --vram 80
    python3 run.py --model meta-llama/Meta-Llama-3-8B-Instruct --vram 24
    python3 run.py --config ./cfg.json --vram 80 --reported 412768

Reference points for the engine side, at the SHAs this book pins:
  vLLM   logs "GPU KV cache size: N tokens"  vllm/v1/core/kv_cache_utils.py:L1925-L1931
  SGLang logs "max_total_num_tokens=N"       python/sglang/srt/managers/scheduler.py:L1095-L1100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bytes per element. The KV cache dtype is not always the model dtype: both
# engines let you store the cache in fp8 while computing in bf16.
DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float8_e4m3fn": 1, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1,
}
GIB = 1024 ** 3


def load_config(args: argparse.Namespace) -> dict:
    if args.config:
        return json.loads(Path(args.config).read_text())
    try:
        from transformers import AutoConfig
    except ImportError:
        sys.exit("Need either --config <path to config.json> or `pip install transformers`.")
    return AutoConfig.from_pretrained(args.model, trust_remote_code=True).to_dict()


def kv_bytes_per_token(cfg: dict, elem: int) -> tuple[int, dict]:
    """Bytes of KV cache one token occupies across all layers.

    Returns the total and a dict of the shape parameters it was derived from, so
    the caller can print the derivation rather than just the answer.
    """
    text_cfg = cfg.get("text_config", cfg)

    layers = (text_cfg.get("num_hidden_layers")
              or text_cfg.get("n_layer")
              or text_cfg.get("num_layers"))
    if layers is None:
        sys.exit("Could not find a layer count in the config.")

    # MLA (DeepSeek-V2/V3 and kin) caches ONE compressed latent per token per
    # layer, not a separate K and V per head. That is the whole point of it, and
    # it is why the usual 2 * h_kv * d_h formula overstates these models badly.
    if text_cfg.get("kv_lora_rank"):
        lora = text_cfg["kv_lora_rank"]
        rope = text_cfg.get("qk_rope_head_dim", 0)
        per_layer = (lora + rope) * elem
        return per_layer * layers, {
            "attention": "MLA (compressed latent)",
            "layers": layers, "kv_lora_rank": lora,
            "qk_rope_head_dim": rope, "bytes/elem": elem,
            "per-layer bytes/token": per_layer,
        }

    heads = text_cfg.get("num_attention_heads")
    kv_heads = text_cfg.get("num_key_value_heads", heads)
    hidden = text_cfg.get("hidden_size")
    head_dim = text_cfg.get("head_dim") or (hidden // heads if hidden and heads else None)
    if not (kv_heads and head_dim):
        sys.exit("Could not find head count / head dim in the config.")

    # 2 = one K entry and one V entry.
    per_layer = 2 * kv_heads * head_dim * elem
    kind = "MHA" if kv_heads == heads else ("MQA" if kv_heads == 1 else "GQA")
    return per_layer * layers, {
        "attention": f"{kind} ({heads} q heads, {kv_heads} kv heads)",
        "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
        "bytes/elem": elem, "per-layer bytes/token": per_layer,
    }


def table(rows: list[tuple[str, str]], title: str) -> None:
    w = max(len(k) for k, _ in rows)
    print(f"\n{title}\n" + "-" * (w + 24))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute a model's KV cache footprint and reconcile it with an engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="HF model id or local dir (needs transformers)")
    src.add_argument("--config", help="path to a config.json (no network needed)")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_BYTES),
                    help="KV cache dtype, NOT necessarily the model dtype (default: bfloat16)")
    ap.add_argument("--vram", type=float, default=80.0,
                    help="total GPU memory in GiB (default: 80)")
    ap.add_argument("--util", type=float, default=0.92,
                    help="fraction of VRAM the engine may use. Matches vLLM's "
                         "--gpu-memory-utilization, whose default is 0.92 at the "
                         "pinned SHA (vllm/config/cache.py:L80). SGLang's "
                         "--mem-fraction-static is the analogue and differs.")
    ap.add_argument("--weights-gib", type=float, default=None,
                    help="weight footprint in GiB; estimated from param count if omitted")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (default: 1)")
    ap.add_argument("--reported", type=int, default=None,
                    help="token capacity the engine printed at startup, to reconcile against")
    args = ap.parse_args()

    cfg = load_config(args)
    elem = DTYPE_BYTES[args.dtype]
    per_token, shape = kv_bytes_per_token(cfg, elem)

    table([(k, f"{v:,}" if isinstance(v, int) else str(v)) for k, v in shape.items()]
          + [("bytes/token (all layers)", f"{per_token:,}"),
             ("KiB/token", f"{per_token / 1024:.2f}"),
             ("MiB per 1k tokens", f"{per_token * 1000 / GIB * 1024:.2f}")],
          "Derivation — arithmetic from the config, not a measurement")

    # Weights. A rough estimate is fine here; the lab asks you to replace it with
    # the real number the engine prints and see how much the answer moves.
    if args.weights_gib is None:
        n_params = cfg.get("num_parameters")
        if n_params:
            weights = n_params * 2 / GIB
            note = "estimated from num_parameters at 2 bytes/param"
        else:
            weights = 0.0
            note = "UNKNOWN — pass --weights-gib for a real answer"
    else:
        weights, note = args.weights_gib, "supplied"

    budget = args.vram * args.util
    per_gpu_weights = weights / args.tp
    free = budget - per_gpu_weights
    cap = int(free * GIB / (per_token / args.tp)) if per_token else 0

    table([
        ("total VRAM (GiB)", f"{args.vram:.1f}"),
        ("usable at util", f"{budget:.1f}   (util={args.util})"),
        ("weights (GiB, all ranks)", f"{weights:.1f}   [{note}]"),
        ("weights per GPU (GiB)", f"{per_gpu_weights:.1f}   (tp={args.tp})"),
        ("left for KV (GiB)", f"{free:.1f}"),
        ("predicted token capacity", f"{cap:,}"),
    ], "Prediction — derived, still not a measurement")

    if args.reported is not None:
        delta = cap - args.reported
        pct = 100.0 * delta / args.reported if args.reported else float("nan")
        table([
            ("engine reported", f"{args.reported:,} tokens"),
            ("this script predicted", f"{cap:,} tokens"),
            ("gap", f"{delta:+,} tokens  ({pct:+.1f}%)"),
        ], "Reconciliation")
        print("\nA gap is expected. Work down this list before blaming the arithmetic:")
        for line in [
            "  1. Is the KV dtype what you passed? fp8 halves the per-token cost.",
            "  2. MLA model? Check the 'attention' row above says MLA.",
            "  3. The engine reserves memory for activations and CUDA graphs, not just weights.",
            "  4. Page size is padded to a block boundary; partial blocks are still whole pages.",
            "  5. Hybrid/SSM models hold conv+ssm state that is not KV at all.",
            "  6. Under TP the cache is sharded on kv heads — which does not divide evenly for every tp.",
        ]:
            print(line)
    else:
        print("\nRe-run with --reported N to reconcile against a real engine startup line.")
        print("  vLLM  : grep 'GPU KV cache size'")
        print("  SGLang: grep 'max_total_num_tokens'")

    print("\nEvery number above is arithmetic from the config. Nothing here was measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
