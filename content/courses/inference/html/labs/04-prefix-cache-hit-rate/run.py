#!/usr/bin/env python3
"""Lab 04 — Prefix cache hit rate: predict it, measure it, then break it.

Two subcommands.

  predict   Pure arithmetic. No GPU, no network, no third-party packages.
            Given a workload shape and a match granularity, print the hit rate
            each engine's counters should report, and what a token-level break
            at a given index would leave behind.

  measure   Drive a shared-prefix workload at a running engine, scraping the
            Prometheus counters around each phase and reporting the delta.
            Uses urllib from the stdlib; no requests, no torch, no transformers.

Prompts are sent as raw token IDs, not text. That is the point: "flip the token
at index 3" has to mean exactly that, and a tokenizer round trip would smear it.

    python3 run.py predict --prefix-len 1000 --suffix-len 200 --requests 64
    python3 run.py predict --prefix-len 1000 --suffix-len 200 --block-size 1
    python3 run.py measure --engine vllm   --base-url http://127.0.0.1:8000
    python3 run.py measure --engine sglang --base-url http://127.0.0.1:30000
    python3 run.py measure --engine vllm --break-mode token --break-at 400
    python3 run.py measure --engine vllm --break-mode salt
    python3 run.py measure --engine vllm --break-mode capacity --flood-requests 400

Metric surfaces this reads, at the SHAs this book pins:

  vLLM    vllm:prefix_cache_queries / vllm:prefix_cache_hits   (Counter, cumulative)
          vllm/v1/metrics/loggers.py:L584-L602
          The logged "Prefix cache hit rate: %.1f%%" is a DIFFERENT number: a
          sliding window over the last 1000 requests, vllm/v1/metrics/stats.py:L106-L111.
          Do not use it to measure a transition.

  SGLang  sglang:prefill_effective_tokens_total{mode=input|device_hit|host_hit|storage_hit}
          (Counter, cumulative) — python/sglang/srt/observability/metrics_collector.py:L892-L902.
          Needs --enable-metrics on the server.
          sglang:cache_hit_rate is a Gauge with multiprocess_mode="mostrecent"
          and reports the last logged batch only; this script ignores it.

NOTHING IN THIS FILE WAS RUN AGAINST A GPU. The predictions are arithmetic. The
measurement path was written against the request schemas and metric names cited
above but has not been executed against a live engine.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- arithmetic


def matchable_tokens(prefix_len: int, prompt_len: int, block: int,
                     break_at: int | None = None) -> int:
    """Tokens an engine can legitimately serve from cache for one warm request.

    Three independent ceilings, and the answer is the smallest:

      1. block * floor(prefix_len / block)  — only full blocks are hashed, so a
         partial trailing block of the shared prefix is unreachable forever
         (vllm/v1/core/kv_cache_utils.py:L745-L752).
      2. block * floor((prompt_len - 1) / block) — the engine holds the last
         token back so there is something to compute logits from
         (vllm/v1/core/kv_cache_manager.py:L256-L262).
      3. block * floor(break_at / block) — if a token at index `break_at` was
         changed, every chained hash from that block onward differs
         (vllm/v1/core/kv_cache_utils.py:L618-L645).
    """
    ceilings = [
        block * (prefix_len // block),
        block * (max(prompt_len - 1, 0) // block),
    ]
    if break_at is not None:
        ceilings.append(block * (max(break_at, 0) // block))
    return max(min(ceilings), 0)


def workload_hit_rate(prefix_len: int, suffix_len: int, requests: int,
                      groups: int, block: int,
                      break_at: int | None = None) -> dict:
    """Aggregate hit rate over a whole shared-prefix workload.

    `groups` distinct prefixes, requests split evenly between them. The first
    request of each group is a cold miss; everything after it is warm.
    """
    prompt_len = prefix_len + suffix_len
    per_warm = matchable_tokens(prefix_len, prompt_len, block, break_at)
    warm = max(requests - groups, 0)
    hits = warm * per_warm
    queries = requests * prompt_len
    return {
        "prompt tokens per request": prompt_len,
        "match granularity (tokens)": block,
        "full blocks in the prefix": prefix_len // block,
        "prefix tokens stranded in a partial block": prefix_len % block,
        "cache-servable tokens per warm request": per_warm,
        "cold requests (one per prefix group)": groups,
        "warm requests": warm,
        "hits (tokens)": hits,
        "queries (tokens)": queries,
        "hit rate": (hits / queries) if queries else 0.0,
    }


# ---------------------------------------------------------------- workload


def token_ids(seed: int, n: int, vocab_lo: int, vocab_hi: int) -> list[int]:
    """A deterministic pseudo-random token id sequence.

    Ids are drawn well inside a conservative vocabulary window so they decode to
    *something* on any tokenizer without tripping special-token handling. The
    text is meaningless; the lab measures cache keys, not output quality.
    """
    rng = random.Random(seed)
    return [rng.randint(vocab_lo, vocab_hi) for _ in range(n)]


def build_requests(args) -> list[tuple[int, list[int]]]:
    """(group_index, token_ids) for every request, prefixes shared within a group."""
    prefixes = [token_ids(args.seed + 1000 * g, args.prefix_len,
                          args.vocab_lo, args.vocab_hi)
                for g in range(args.groups)]
    out = []
    for i in range(args.requests):
        g = i % args.groups
        suffix = token_ids(args.seed + 7_000_000 + i, args.suffix_len,
                           args.vocab_lo, args.vocab_hi)
        out.append((g, prefixes[g] + suffix))
    return out


def apply_break(prompt: list[int], args) -> list[int]:
    """Flip one token inside the shared prefix, leaving length unchanged."""
    if args.break_at >= len(prompt):
        sys.exit(f"--break-at {args.break_at} is past the end of a "
                 f"{len(prompt)}-token prompt.")
    out = list(prompt)
    old = out[args.break_at]
    new = old + 1 if old < args.vocab_hi else args.vocab_lo
    out[args.break_at] = new
    return out


# ---------------------------------------------------------------- transport


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def _get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def scrape(base_url: str, timeout: float) -> dict[str, float]:
    """Every Prometheus sample on /metrics, keyed by the full sample line minus
    its value. Aggregation happens in the caller so label sets stay visible.

    The prefix-match-and-sum style is lifted from vLLM's own integration test,
    tests/v1/kv_connector/nixl_integration/test_mamba_prefix_cache.py:L234-L247,
    and tolerates the `_total` suffix the exposition format appends to counters.
    """
    try:
        text = _get(base_url.rstrip("/") + "/metrics", timeout)
    except urllib.error.HTTPError as e:
        sys.exit(f"/metrics returned HTTP {e.code}. For SGLang, start the server "
                 f"with --enable-metrics.")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach {base_url}/metrics: {e.reason}")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?P<key>\S+)\s+(?P<val>[\d.eE+\-]+)$", line)
        if not m:
            continue
        try:
            out[m.group("key")] = float(m.group("val"))
        except ValueError:
            continue
    return out


def sum_family(samples: dict[str, float], name: str,
               label_must_contain: str | None = None) -> float:
    """Sum every sample whose metric name starts with `name`.

    Two traps, both silent if you get them wrong:

      * The exposition format appends `_total` to a Counter, so the declared
        name `vllm:prefix_cache_hits` appears as `vllm:prefix_cache_hits_total`.
        Prefix matching handles that.
      * The same client also emits a `<name>_created` sample whose VALUE IS A
        UNIX TIMESTAMP. It shares the prefix, and summing it silently turns the
        answer into ~1.7e9. Drop it explicitly.
    """
    total = 0.0
    for key, val in samples.items():
        if not key.startswith(name):
            continue
        metric = key.split("{", 1)[0]
        if metric.endswith("_created"):
            continue
        if label_must_contain and label_must_contain not in key:
            continue
        total += val
    return total


def hit_and_query(engine: str, samples: dict[str, float]) -> tuple[float, float]:
    """(hit tokens, total tokens) from one scrape, per engine."""
    if engine == "vllm":
        return (sum_family(samples, "vllm:prefix_cache_hits"),
                sum_family(samples, "vllm:prefix_cache_queries"))
    fam = "sglang:prefill_effective_tokens_total"
    hits = sum(sum_family(samples, fam, f'mode="{m}"')
               for m in ("device_hit", "host_hit", "storage_hit"))
    total = hits + sum_family(samples, fam, 'mode="input"')
    return hits, total


# ---------------------------------------------------------------- engine calls


def resolve_model(args) -> str:
    if args.model:
        return args.model
    if args.engine != "vllm":
        return ""
    try:
        data = json.loads(_get(args.base_url.rstrip("/") + "/v1/models",
                               args.timeout))
        return data["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - any failure means "tell the user"
        sys.exit(f"Could not read /v1/models to find the model id ({e}). "
                 f"Pass --model explicitly.")


def send(args, model: str, prompt: list[int], salt: str | None) -> None:
    base = args.base_url.rstrip("/")
    if args.engine == "vllm":
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0.0,
        }
        if salt is not None:
            payload["cache_salt"] = salt
        _post(base + "/v1/completions", payload, args.timeout)
    else:
        if salt is not None:
            sys.exit("--break-mode salt is vLLM-only: SGLang has no cache_salt "
                     "field at this SHA. Use --break-mode token instead.")
        _post(base + "/generate", {
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": args.output_len,
                "temperature": 0.0,
            },
        }, args.timeout)


def reset_cache(args) -> str:
    base = args.base_url.rstrip("/")
    try:
        if args.engine == "vllm":
            r = _post(base + "/reset_prefix_cache", {}, args.timeout)
            return f"reset_prefix_cache -> {r}"
        r = _post(base + "/flush_cache", {}, args.timeout)
        return f"flush_cache -> {str(r)[:80]}"
    except urllib.error.HTTPError as e:
        if args.engine == "vllm" and e.code == 404:
            return ("reset_prefix_cache -> HTTP 404. The route only exists when "
                    "the server was started with VLLM_SERVER_DEV_MODE=1.")
        return f"reset -> HTTP {e.code}"
    except urllib.error.URLError as e:
        return f"reset -> unreachable ({e.reason})"


def environment(args) -> list[tuple[str, str]]:
    base = args.base_url.rstrip("/")
    rows = [("engine", args.engine), ("base url", base)]
    try:
        if args.engine == "vllm":
            rows.append(("vllm version",
                         json.loads(_get(base + "/version", args.timeout))
                         .get("version", "?")))
        else:
            info = json.loads(_get(base + "/server_info", args.timeout))
            for k in ("version", "model_path", "page_size", "max_total_num_tokens",
                      "disable_radix_cache"):
                if k in info:
                    rows.append((k, str(info[k])))
    except Exception as e:  # noqa: BLE001
        rows.append(("server info", f"unavailable ({e})"))
    return rows


# ---------------------------------------------------------------- output


def table(rows, title: str) -> None:
    rows = [(k, v if isinstance(v, str) else
             (f"{v:,}" if isinstance(v, int) else f"{v:.4f}"))
            for k, v in rows]
    w = max((len(k) for k, _ in rows), default=0)
    print(f"\n{title}\n" + "-" * (w + 28))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def phase_table(phases: list[dict]) -> None:
    hdr = f"{'phase':<22}{'hit tok':>12}{'total tok':>12}{'hit rate':>11}{'predicted':>11}"
    print("\nMeasured per phase — counter deltas, not the logged sliding window")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for p in phases:
        pred = "-" if p["predicted"] is None else f"{p['predicted'] * 100:.1f}%"
        print(f"{p['name']:<22}{p['hits']:>12,.0f}{p['total']:>12,.0f}"
              f"{p['rate'] * 100:>10.1f}%{pred:>11}")
    print("-" * len(hdr))


# ---------------------------------------------------------------- commands


def cmd_predict(args) -> int:
    base = workload_hit_rate(args.prefix_len, args.suffix_len, args.requests,
                             args.groups, args.block_size)
    table(list(base.items()),
          "Warm shared-prefix workload — derived, arithmetic, not a measurement")

    print("\nBreak-position sweep — hit rate if one token at index i is changed")
    print("(the staircase between block boundaries is the whole point)")
    prompt_len = args.prefix_len + args.suffix_len
    hdr = f"{'break at':>10}{'blocks kept':>14}{'servable tok':>14}{'hit rate':>11}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    positions = sorted({0, 1, args.block_size, args.block_size + 1,
                        args.prefix_len // 2, args.prefix_len - 1,
                        args.prefix_len})
    for i in positions:
        if i > args.prefix_len:
            continue
        w = workload_hit_rate(args.prefix_len, args.suffix_len, args.requests,
                              args.groups, args.block_size, break_at=i)
        print(f"{i:>10,}{i // args.block_size:>14,}"
              f"{w['cache-servable tokens per warm request']:>14,}"
              f"{w['hit rate'] * 100:>10.1f}%")
    print("-" * len(hdr))

    naive = args.prefix_len / prompt_len
    print(f"\nThe naive answer, prefix/(prefix+suffix), is {naive * 100:.1f}%.")
    print(f"The real answer is {base['hit rate'] * 100:.1f}%. The difference is "
          f"the cold request, the\nreserved last token, and "
          f"{args.prefix_len % args.block_size} token(s) stranded in a partial block.")
    print("\nEvery number above is arithmetic. Nothing here was measured.")
    return 0


def cmd_measure(args) -> int:
    model = resolve_model(args)
    table(environment(args) + [("model", model or "(sglang default)")],
          "Environment — paste this with any result you report")

    reqs = build_requests(args)
    phases: list[dict] = []

    def run_phase(name: str, items, salt_for, predicted):
        before = hit_and_query(args.engine, scrape(args.base_url, args.timeout))
        t0 = time.monotonic()
        for idx, prompt in items:
            send(args, model, prompt, salt_for(idx))
        # Counters are updated from the scheduler's stats snapshot, which the
        # logger folds in on its own interval; give it one to land.
        time.sleep(args.settle)
        after = hit_and_query(args.engine, scrape(args.base_url, args.timeout))
        hits, total = after[0] - before[0], after[1] - before[1]
        phases.append({
            "name": name,
            "hits": hits,
            "total": total,
            "rate": (hits / total) if total else 0.0,
            "predicted": predicted,
            "seconds": time.monotonic() - t0,
        })

    if args.reset_first:
        print("\n" + reset_cache(args))
        time.sleep(args.settle)

    cold = [(g, p) for g, p in reqs[:args.groups]]
    warm = [(g, p) for g, p in reqs[args.groups:]]

    run_phase("cold (one/prefix)", cold, lambda g: None, 0.0)
    run_phase("warm", warm, lambda g: None,
              workload_hit_rate(args.prefix_len, args.suffix_len,
                                len(warm) + args.groups, args.groups,
                                args.block_size)["hit rate"]
              if warm else None)

    if args.break_mode == "token":
        broken = [(g, apply_break(p, args)) for g, p in reqs]
        pred = workload_hit_rate(args.prefix_len, args.suffix_len, args.requests,
                                 args.groups, args.block_size,
                                 break_at=args.break_at)["hit rate"]
        run_phase(f"broken @tok {args.break_at}", broken, lambda g: None, pred)
    elif args.break_mode == "salt":
        run_phase("broken (cache_salt)", reqs,
                  lambda g: f"lab04-{args.seed}", 0.0)
    elif args.break_mode == "capacity":
        flood = [(0, token_ids(args.seed + 900_000 + i, args.flood_len,
                               args.vocab_lo, args.vocab_hi))
                 for i in range(args.flood_requests)]
        run_phase("flood (unique)", flood, lambda g: None, 0.0)
        run_phase("re-warm (evicted?)", warm, lambda g: None, None)
        run_phase("re-warm again", warm, lambda g: None, None)

    phase_table(phases)

    print("\nHow to read this:")
    print("  - 'predicted' is arithmetic from --prefix-len/--suffix-len/--block-size.")
    print("  - A gap on the warm phase usually means the prompts overlapped in")
    print("    flight, or --block-size does not match the server's real value.")
    print("  - token/salt breaks stay broken on a second pass; a capacity break")
    print("    recovers. That is the only reliable way to tell them apart, and it")
    print("    is why --break-mode capacity runs the warm phase twice.")
    print("  - The engine's own logged 'Prefix cache hit rate' will disagree with")
    print("    these numbers. It is a 1000-request sliding window. These are")
    print("    counter deltas for exactly this phase.")
    return 0


# ---------------------------------------------------------------- cli


def add_workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prefix-len", type=int, default=1000,
                   help="tokens in the shared prefix (default: 1000)")
    p.add_argument("--suffix-len", type=int, default=200,
                   help="tokens in each request's distinct suffix (default: 200)")
    p.add_argument("--requests", type=int, default=64,
                   help="total requests in the workload (default: 64)")
    p.add_argument("--groups", type=int, default=1,
                   help="distinct shared prefixes; each costs one cold miss "
                        "(default: 1)")
    p.add_argument("--block-size", type=int, default=16,
                   help="match granularity in tokens. vLLM: --block-size, "
                        "default 16. SGLang: --page-size, default 1 (default: 16)")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Predict, measure, and deliberately break a prefix cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{predict,measure}")

    pp = sub.add_parser("predict", help="arithmetic only; no GPU, no network",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workload_args(pp)
    pp.set_defaults(func=cmd_predict)

    mp = sub.add_parser("measure", help="drive a running engine and read its counters",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workload_args(mp)
    mp.add_argument("--engine", choices=["vllm", "sglang"], required=True,
                    help="which engine is listening; picks the API shape and "
                         "the metric names")
    mp.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="server root (default: http://127.0.0.1:8000; "
                         "SGLang's default port is 30000)")
    mp.add_argument("--model", default=None,
                    help="model id for vLLM's /v1/completions; read from "
                         "/v1/models if omitted")
    mp.add_argument("--output-len", type=int, default=8,
                    help="tokens to generate per request; keep it small, this "
                         "lab measures prefill (default: 8)")
    mp.add_argument("--break-mode", choices=["none", "token", "salt", "capacity"],
                    default="none",
                    help="how to break the cache after the warm phase "
                         "(default: none)")
    mp.add_argument("--break-at", type=int, default=0,
                    help="token index to flip for --break-mode token. Sweep it "
                         "across block boundaries to see the staircase "
                         "(default: 0)")
    mp.add_argument("--flood-requests", type=int, default=400,
                    help="unique requests to push through for --break-mode "
                         "capacity (default: 400)")
    mp.add_argument("--flood-len", type=int, default=4096,
                    help="prompt length of each flood request (default: 4096)")
    mp.add_argument("--reset-first", action="store_true",
                    help="POST /reset_prefix_cache (vLLM, needs "
                         "VLLM_SERVER_DEV_MODE=1) or /flush_cache (SGLang) "
                         "before starting")
    mp.add_argument("--settle", type=float, default=2.0,
                    help="seconds to wait after a phase before re-scraping, so "
                         "the stats snapshot lands (default: 2.0)")
    mp.add_argument("--timeout", type=float, default=120.0,
                    help="per-HTTP-call timeout in seconds (default: 120)")
    mp.set_defaults(func=cmd_measure)

    for p in (pp, mp):
        p.add_argument("--seed", type=int, default=0,
                       help="seed for the synthetic token ids (default: 0)")
        p.add_argument("--vocab-lo", type=int, default=1000,
                       help="lowest token id to emit (default: 1000)")
        p.add_argument("--vocab-hi", type=int, default=20000,
                       help="highest token id to emit; keep it inside the "
                            "model's vocab (default: 20000)")

    args = ap.parse_args()
    if args.groups < 1 or args.groups > args.requests:
        sys.exit("--groups must be between 1 and --requests.")
    if args.block_size < 1:
        sys.exit("--block-size must be at least 1.")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
