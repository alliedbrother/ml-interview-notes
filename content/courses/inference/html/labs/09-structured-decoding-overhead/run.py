#!/usr/bin/env python3
"""Lab 09 — What the grammar mask costs.

Two subcommands.

  schemas   Print the schema complexity tiers this lab uses, with their sizes.
            No GPU, no network, no third-party packages.

  sweep     Drive one running server with and without a JSON schema, at one or
            more concurrency levels, and report the gap. stdlib only: urllib
            for HTTP, concurrent.futures for the load, json for the schema.

Everything goes through the OpenAI-standard `response_format` field, which BOTH
engines accept and BOTH funnel into their own constraint machinery:

    vLLM    vllm/entrypoints/openai/engine/protocol.py:L218-L227
    SGLang  python/sglang/srt/entrypoints/openai/protocol.py:L1044-L1055

That matters. vLLM's own structured-output benchmark attaches the schema through
its private `structured_outputs` extra-body field
(benchmarks/benchmark_serving_structured_output.py:L484-L489), which an SGLang
server ignores — so pointing that benchmark at SGLang measures unconstrained
generation and calls it structured. This script exists to avoid that.

    python3 run.py schemas
    python3 run.py sweep --engine vllm --base-url http://127.0.0.1:8000 \
        --concurrency 1 8 32 128 129 256 --schema none flat --requests 256
    python3 run.py sweep --engine sglang --base-url http://127.0.0.1:30000 \
        --concurrency 64 --schema none flat nested deep --requests 256
    python3 run.py sweep --engine vllm --concurrency 1 --schema nested \
        --requests 64 --unique-schemas        # defeats the compiler cache

Why the arms are what they are, all read out of the pinned source:

  - vLLM launches the forward pass non-blocking and builds the bitmask while the
    GPU is busy (vllm/v1/engine/core.py:L593-L603), so the overhead only appears
    once fill time exceeds forward time. Sweep concurrency to find that point.

  - vLLM's parallel fill needs BOTH max_num_seqs > 128 at startup
    (vllm/v1/structured_output/__init__.py:L61-L69) AND more than 128 constrained
    requests in the step with zero speculative tokens (same file, L250-L256).
    Sweep across 128/129 and watch for the discontinuity.

  - SGLang fills one row at a time in Python on every backend except llguidance,
    which overrides it with a native batched fill
    (python/sglang/srt/constrained/base_grammar_backend.py:L88-L94 versus
    python/sglang/srt/constrained/llguidance_backend.py:L151-L159).

  - Compiled grammars are cached (vllm/v1/structured_output/backend_xgrammar.py:L66-L71,
    sized by VLLM_XGRAMMAR_CACHE_MB). Reusing one schema measures nothing about
    compilation; --unique-schemas is the fix, mirroring the benchmark's own
    json-unique mode.

NOTHING IN THIS FILE WAS RUN AGAINST A GPU. It has no predictions baked in; it
reports what your server does. The schema-size arithmetic in `schemas` is
arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- schemas

FLAT = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
        "active": {"type": "boolean"},
        "score": {"type": "number"},
    },
    "required": ["name", "age", "city", "active", "score"],
}

NESTED = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "profile": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                    "required": ["email", "phone"],
                },
            },
            "required": ["name", "contact"],
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["click", "view", "buy"]},
                    "at": {"type": "integer"},
                },
                "required": ["kind", "at"],
            },
        },
    },
    "required": ["id", "profile", "tags", "events"],
}


def _deep(levels: int) -> dict:
    """A chain of required nested objects, plus an enum and a constrained string
    at the bottom. Automaton size grows with structure, so this is the arm that
    should make compilation visible."""
    node: dict = {
        "type": "object",
        "properties": {
            "leaf": {"type": "string", "minLength": 1, "maxLength": 24},
            "kind": {"type": "string",
                     "enum": ["alpha", "beta", "gamma", "delta", "epsilon"]},
            "n": {"type": "integer"},
        },
        "required": ["leaf", "kind", "n"],
    }
    for i in range(levels):
        node = {
            "type": "object",
            "properties": {
                f"level_{levels - i}": node,
                f"sibling_{levels - i}": {"type": "string"},
            },
            "required": [f"level_{levels - i}", f"sibling_{levels - i}"],
        }
    return node


SCHEMAS = {
    "none": None,
    "flat": FLAT,
    "nested": NESTED,
    "deep": _deep(6),
}


def count_nodes(node) -> int:
    if isinstance(node, dict):
        return 1 + sum(count_nodes(v) for v in node.values())
    if isinstance(node, list):
        return sum(count_nodes(v) for v in node)
    return 0


def uniquify(schema: dict | None) -> dict | None:
    """Append a differently-named optional property so the compiler cache misses.

    Same trick as vLLM's `json-unique` dataset mode, whose own comment reads
    'An unique optional field to avoid cached schemas'
    (benchmarks/benchmark_serving_structured_output.py:L161-L169).
    """
    if schema is None:
        return None
    out = json.loads(json.dumps(schema))
    out.setdefault("properties", {})[f"__optional_field_{uuid.uuid4()}"] = {
        "type": "string",
        "description": "An unique optional field to avoid cached schemas",
    }
    return out


def cmd_schemas(args) -> int:
    hdr = f"{'tier':<8}{'JSON bytes':>12}{'nodes':>8}{'structure':>16}"
    print("\nSchema complexity tiers — arithmetic, no server involved")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, s in SCHEMAS.items():
        if s is None:
            print(f"{'none':<8}{'-':>12}{'-':>8}{'unconstrained':>16}")
            continue
        blob = json.dumps(s)
        print(f"{name:<8}{len(blob):>12,}{count_nodes(s):>8,}"
              f"{('deep chain' if name == 'deep' else 'shallow'):>16}")
    print("-" * len(hdr))
    if args.dump:
        print(json.dumps(SCHEMAS[args.dump], indent=2))
    print("\nBitmask cost is independent of all of this: one row is ceil(V/32) "
          "int32 = V/8 bytes,\nwhatever built the automaton. Schema complexity "
          "moves COMPILE time, not fill size.")
    print("Compile cost is per DISTINCT schema and cached. Use --unique-schemas "
          "in `sweep`\nto measure it; without that you are measuring one "
          "compilation amortised to nothing.")
    return 0


# ---------------------------------------------------------------- transport


PROMPT = ("Produce one example record. Reply with JSON only, no prose, no "
          "code fences.")


def build_payload(args, schema: dict | None) -> dict:
    payload: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "record", "schema": schema},
        }
    return payload


def stream_one(args, schema: dict | None) -> dict:
    """One streamed chat completion. Returns TTFT, e2e, and the text."""
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(build_payload(args, schema)).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    text_parts: list[str] = []
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    completion_tokens = usage["completion_tokens"]
                for ch in obj.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                        text_parts.append(piece)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except urllib.error.URLError as e:
        return {"error": f"unreachable: {e.reason}"}
    e2e = time.perf_counter() - t0
    text = "".join(text_parts)
    return {
        "ttft": ttft if ttft is not None else e2e,
        "e2e": e2e,
        "chunks": chunks,
        "completion_tokens": completion_tokens or chunks,
        "valid_json": _is_json(text),
        "text": text,
    }


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_model(args) -> str:
    if args.model:
        return args.model
    try:
        with urllib.request.urlopen(
                args.base_url.rstrip("/") + "/v1/models", timeout=args.timeout) as r:
            return json.loads(r.read().decode())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not read /v1/models ({e}). Pass --model.")


# ---------------------------------------------------------------- sweep


def run_arm(args, schema_name: str, concurrency: int) -> dict:
    base = SCHEMAS[schema_name]
    schemas = ([uniquify(base) for _ in range(args.requests)]
               if args.unique_schemas else [base] * args.requests)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda s: stream_one(args, s), schemas))
    wall = time.perf_counter() - t0

    errs = [r["error"] for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"schema": schema_name, "concurrency": concurrency,
                "error": errs[0] if errs else "no successful requests"}

    ttfts = sorted(r["ttft"] for r in ok)
    out_toks = sum(r["completion_tokens"] for r in ok)
    valid = sum(1 for r in ok if r["valid_json"])
    return {
        "schema": schema_name,
        "concurrency": concurrency,
        "requests": len(ok),
        "errors": len(errs),
        "wall_s": wall,
        "req_per_s": len(ok) / wall,
        "out_tok_per_s": out_toks / wall,
        "ttft_p50_ms": ttfts[len(ttfts) // 2] * 1000,
        "ttft_p99_ms": ttfts[min(int(0.99 * len(ttfts)), len(ttfts) - 1)] * 1000,
        "e2e_mean_ms": statistics.fmean(r["e2e"] for r in ok) * 1000,
        "valid_json_frac": valid / len(ok),
        "first_error": errs[0] if errs else None,
    }


def cmd_sweep(args) -> int:
    args.model = resolve_model(args)
    print("\nEnvironment — paste this with any result you report")
    print("-" * 62)
    for k, v in [("engine", args.engine), ("base url", args.base_url),
                 ("model", args.model), ("host cores (os.cpu_count)",
                                         str(os.cpu_count())),
                 ("requests per arm", str(args.requests)),
                 ("output_len", str(args.output_len)),
                 ("unique schemas", str(args.unique_schemas))]:
        print(f"{k:<26}  {v}")
    print("\nThe host core count is not a footnote here: vLLM's parallel mask "
          "fill uses\nmin(cores // 2, 8) workers, so this number changes the "
          "answer above batch 128.")

    rows = []
    for conc in args.concurrency:
        for name in args.schema:
            print(f"\n  running: schema={name} concurrency={conc} ...",
                  flush=True)
            rows.append(run_arm(args, name, conc))

    hdr = (f"{'schema':<8}{'conc':>6}{'req/s':>9}{'out tok/s':>11}"
           f"{'ttft p50':>10}{'ttft p99':>10}{'e2e mean':>10}"
           f"{'valid json':>12}{'vs none':>9}")
    print("\nMeasured on YOUR hardware — no number here was predicted")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    baseline = {r["concurrency"]: r for r in rows
                if r.get("schema") == "none" and "error" not in r}
    for r in rows:
        if "error" in r:
            print(f"{r['schema']:<8}{r['concurrency']:>6}  ERROR: {r['error']}")
            continue
        b = baseline.get(r["concurrency"])
        rel = (f"{r['out_tok_per_s'] / b['out_tok_per_s']:.2f}x"
               if b and b["out_tok_per_s"] else "-")
        print(f"{r['schema']:<8}{r['concurrency']:>6}{r['req_per_s']:>9.2f}"
              f"{r['out_tok_per_s']:>11.1f}{r['ttft_p50_ms']:>9.1f}m"
              f"{r['ttft_p99_ms']:>9.1f}m{r['e2e_mean_ms']:>9.1f}m"
              f"{r['valid_json_frac'] * 100:>11.0f}%{rel:>9}")
    print("-" * len(hdr))

    bad = [r for r in rows
           if r.get("schema") not in (None, "none") and "error" not in r
           and r["valid_json_frac"] < 0.99]
    if bad:
        print("\nWARNING: a constrained arm produced output that does not parse "
              "as JSON.")
        print("The constraint may not have been applied at all. Check that the "
              "server was")
        print("started with a grammar backend, and that it accepted "
              "response_format rather")
        print("than ignoring an unknown field.")

    if "none" not in args.schema:
        print("\nNOTE: you did not include the `none` arm, so there is no "
              "baseline to compare\nagainst and the 'vs none' column is empty. "
              "Add --schema none <others>.")

    print("\nReading this table:")
    print("  - At low concurrency, expect schema-on to be close to schema-off: "
          "vLLM builds")
    print("    the mask while the forward pass is already running.")
    print("  - Watch the 128 -> 129 step on vLLM. The parallel fill path needs "
          "max_num_seqs")
    print("    > 128 at startup AND > 128 constrained requests in the step, "
          "with no")
    print("    speculative decoding. Serve with --max-num-seqs 129 or higher to "
          "reach it.")
    print("  - ttft p99 is where compilation shows up. If it moves only with "
          "--unique-schemas,")
    print("    you measured the compiler cache, not the compiler.")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved raw rows to {args.save}")
    return 0


# ---------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Price the grammar mask: throughput with and without a JSON "
                    "schema, across batch size and schema complexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{schemas,sweep}")

    sp = sub.add_parser("schemas", help="print the complexity tiers; no network")
    sp.add_argument("--dump", choices=sorted(k for k in SCHEMAS if k != "none"),
                    help="also print one schema as JSON")
    sp.set_defaults(func=cmd_schemas)

    wp = sub.add_parser("sweep", help="measure a running server")
    wp.add_argument("--engine", choices=["vllm", "sglang"], required=True,
                    help="recorded in the output; both are driven through the "
                         "same OpenAI-standard response_format field")
    wp.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="server root (default: http://127.0.0.1:8000; "
                         "SGLang's default port is 30000)")
    wp.add_argument("--model", default=None,
                    help="model id; read from /v1/models if omitted")
    wp.add_argument("--schema", nargs="+", default=["none", "flat"],
                    choices=sorted(SCHEMAS),
                    help="arms to run, in order. Include `none` or there is no "
                         "baseline (default: none flat)")
    wp.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 128, 129],
                    help="in-flight request counts to sweep. The 128/129 pair "
                         "is deliberate (default: 1 8 32 128 129)")
    wp.add_argument("--requests", type=int, default=256,
                    help="requests per arm (default: 256)")
    wp.add_argument("--output-len", type=int, default=128,
                    help="max tokens per response (default: 128)")
    wp.add_argument("--unique-schemas", action="store_true",
                    help="give every request a differently-named optional field "
                         "so the grammar compiler cache misses. This is how you "
                         "measure compilation instead of measuring the cache")
    wp.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout in seconds (default: 600)")
    wp.add_argument("--save", default=None,
                    help="write the raw rows to this JSON path")
    wp.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    if getattr(args, "concurrency", None) and any(c < 1 for c in args.concurrency):
        sys.exit("--concurrency values must be >= 1.")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
