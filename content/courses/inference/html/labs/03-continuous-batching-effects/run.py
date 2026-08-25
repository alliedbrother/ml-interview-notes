#!/usr/bin/env python3
"""Lab 03 — continuous batching under load.

Continuous batching buys throughput by keeping more rows resident in each decode
step. Rows are not free: every one of them adds KV traffic to the step and KV
bytes to the pool. So throughput climbs with concurrency, then flattens, and
latency climbs the whole way. This script finds where it flattens.

It drives an OpenAI-compatible endpoint (vLLM or SGLang, both speak it) in a
CLOSED loop at a series of concurrency levels, and reports, per level:

    TTFT     time to first token             — queueing plus one prefill
    TPOT     (latency - TTFT) / (tokens - 1) — the steady-state decode cost
    ITL      the raw inter-token gaps        — TPOT's distribution, not its mean
    output tokens/s, requests/s

Metric definitions are lifted from vLLM's own serving benchmark so the numbers
are comparable with `vllm bench serve`:
    TPOT  vllm/benchmarks/serve.py:L607-L613
    TTFT and ITL captured off the SSE stream
          vllm/benchmarks/lib/endpoint_request_func.py:L400-L412

Stdlib only — urllib and threads, no aiohttp, no requests, no torch. It reads
nothing from and writes nothing to either engine checkout.

    # start a server first, e.g.
    #   vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-num-seqs 256
    #   python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct

    python3 run.py --base-url http://localhost:8000 --model meta-llama/Meta-Llama-3-8B-Instruct
    python3 run.py --concurrency 1 2 4 8 16 32 64 128 --requests-per-level 40
    python3 run.py --api chat --port 30000 --json sweep.json
    python3 run.py --help

Predict before you run: at concurrency 1 the step is one weight stream, so
output tok/s is roughly beta / bytes_weights (lab 01 gives you beta). Doubling
concurrency should double throughput until either the weight stream is amortised
or the KV pool fills. Write down which one you expect to bite first.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


# --------------------------------------------------------------------------
# one streamed request
# --------------------------------------------------------------------------

class Result:
    __slots__ = ("ok", "ttft", "latency", "chunks", "tokens", "itl", "error")

    def __init__(self):
        self.ok = False
        self.ttft = 0.0
        self.latency = 0.0
        self.chunks = 0
        self.tokens = 0
        self.itl: list[float] = []
        self.error = ""


def stream_one(args, prompt: str) -> Result:
    """Send one streaming request and time every token arrival.

    Timestamps are taken as each SSE data frame is parsed — the same place
    vLLM's harness takes them. That makes TTFT and ITL client-side numbers that
    include network and framing, which is what a user experiences and what an
    SLO is written against.
    """
    import urllib.error
    import urllib.request

    res = Result()
    if args.api == "chat":
        url = f"{args.base_url}/v1/chat/completions"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": args.ignore_eos,
        }
    else:
        url = f"{args.base_url}/v1/completions"
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": args.ignore_eos,
        }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST")

    start = time.perf_counter()
    last = start
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                # SSE comments (keep-alive pings) start with a colon.
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                choices = data.get("choices") or []
                text = ""
                if choices:
                    c = choices[0]
                    text = c.get("text") or (c.get("delta") or {}).get("content") or ""
                if text:
                    if res.chunks == 0:
                        res.ttft = now - start
                    else:
                        res.itl.append(now - last)
                    res.chunks += 1
                    last = now
                if (usage := data.get("usage")):
                    if (ct := usage.get("completion_tokens")) is not None:
                        res.tokens = ct
        res.latency = time.perf_counter() - start
        res.tokens = res.tokens or res.chunks
        res.ok = res.chunks > 0
        if not res.ok:
            res.error = "no content chunks — is --api right, and is streaming enabled?"
    except urllib.error.HTTPError as exc:
        res.error = f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
    except Exception as exc:                                    # noqa: BLE001
        res.error = f"{exc.__class__.__name__}: {exc}"
    return res


# --------------------------------------------------------------------------
# one concurrency level
# --------------------------------------------------------------------------

def run_level(args, conc: int, prompts: list[str]) -> dict:
    """Closed loop: `conc` workers, each starting a new request the moment its
    previous one finishes. Concurrency is therefore exactly `conc` for the whole
    measured window — no arrival process, no queueing model, just a held load.

    Closed-loop is the right shape for THIS question. It answers "with N users
    each waiting for their answer, what do they see?". It cannot answer "what
    happens at 40 requests/second" — that needs open-loop arrivals, which is
    chapter 10.3's subject and lab 11's job.
    """
    total = args.requests_per_level
    counter = {"i": 0}
    lock = threading.Lock()
    results: list[Result] = []

    def worker() -> None:
        while True:
            with lock:
                i = counter["i"]
                if i >= total:
                    return
                counter["i"] = i + 1
            r = stream_one(args, prompts[i % len(prompts)])
            with lock:
                results.append(r)

    # Warmup at this concurrency: the first request after a level change pays
    # for CUDA-graph capture on a new batch shape and for a cold prefix cache.
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(lambda _: stream_one(args, prompts[0]), range(min(conc, args.warmup))))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(worker) for _ in range(conc)]
        for f in futures:
            f.result()          # surface worker exceptions instead of swallowing them
    wall = time.perf_counter() - t0

    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    row = {"concurrency": conc, "completed": len(good), "failed": len(bad), "wall_s": wall}
    if bad:
        row["first_error"] = bad[0].error
    if not good:
        return row

    ttfts = sorted(r.ttft for r in good)
    tpots = sorted((r.latency - r.ttft) / (r.tokens - 1)
                   for r in good if r.tokens > 1)
    itls = sorted(x for r in good for x in r.itl)
    out_tokens = sum(r.tokens for r in good)

    row.update({
        "ttft_p50": pct(ttfts, 50), "ttft_p99": pct(ttfts, 99),
        "tpot_p50": pct(tpots, 50), "tpot_p99": pct(tpots, 99),
        "itl_p50": pct(itls, 50), "itl_p99": pct(itls, 99),
        "itl_max": itls[-1] if itls else 0.0,
        "output_tok_s": out_tokens / wall if wall else 0.0,
        "req_s": len(good) / wall if wall else 0.0,
        "mean_output_tokens": out_tokens / len(good),
    })
    return row


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def grid(headers: list[str], rows: list[list[str]], title: str) -> None:
    if not rows:
        return
    w = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print(f"\n{title}")
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(r)))


def report(args, rows: list[dict]) -> None:
    good = [r for r in rows if r.get("completed")]
    if not good:
        print("\nEvery request failed. First error:", rows[0].get("first_error", "?"))
        return

    base = good[0]
    grid(["conc", "done", "TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99",
          "ITL p99", "ITL max", "out tok/s", "req/s", "scaling"],
         [[str(r["concurrency"]), str(r["completed"]),
           ms(r["ttft_p50"]), ms(r["ttft_p99"]),
           ms(r["tpot_p50"]), ms(r["tpot_p99"]),
           ms(r["itl_p99"]), ms(r["itl_max"]),
           f"{r['output_tok_s']:.1f}", f"{r['req_s']:.2f}",
           f"{(r['output_tok_s'] / base['output_tok_s']) / (r['concurrency'] / base['concurrency']):.2f}x"]
          for r in good],
         "Measured on your server — closed loop, latencies in ms")

    print("\n'scaling' is throughput per unit concurrency, normalised to the first level.")
    print("1.00 means perfectly linear. It falls as the batch stops amortising the weight")
    print("stream and starts being dominated by KV reads — the crossover derived in 0.4.")

    # The knee: the first level whose MARGINAL throughput gain falls below a
    # fraction of what perfectly linear scaling would have given.
    knee = None
    for prev, cur in zip(good, good[1:]):
        ideal = prev["output_tok_s"] * (cur["concurrency"] / prev["concurrency"] - 1.0)
        gained = cur["output_tok_s"] - prev["output_tok_s"]
        frac = gained / ideal if ideal > 0 else 0.0
        if frac < args.knee_threshold and knee is None:
            knee = (prev, cur, frac)
    if knee:
        prev, cur, frac = knee
        print(f"\nKnee: between concurrency {prev['concurrency']} and {cur['concurrency']} "
              f"the marginal throughput gain is {100 * frac:.0f}% of linear "
              f"(threshold {100 * args.knee_threshold:.0f}%).")
        print(f"  TTFT p99 there goes {ms(prev['ttft_p99'])} ms -> {ms(cur['ttft_p99'])} ms, "
              f"TPOT p50 {ms(prev['tpot_p50'])} ms -> {ms(cur['tpot_p50'])} ms, "
              f"TPOT p99 {ms(prev['tpot_p99'])} ms -> {ms(cur['tpot_p99'])} ms.")
    else:
        print("\nNo knee found in this range — the curve was still climbing at the "
              "top level. Extend --concurrency upward until it bends.")

    print("\nNow name the cause. Read the server log while the top level runs:")
    print("  vLLM  : 'Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%'")
    print("          plus 'Preemptions: N' once it starts evicting")
    print("  SGLang: 'Decode batch, #running-req: N, ... #queue-req: M'")
    print("          plus 'KV cache pool is full. Retract requests.'")
    print("If Running stops rising while Waiting does, the cap is --max-num-seqs")
    print("(vLLM) or --max-running-requests (SGLang). If KV usage is pinned near")
    print("100% and preemptions appear, the cap is the pool. Those are different")
    print("bugs with different fixes.")


def ms(x: float) -> str:
    return f"{1000 * x:.1f}"


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep concurrency against an OpenAI-compatible endpoint and find the knee.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base-url", default=None,
                    help="server root, e.g. http://localhost:8000 "
                         "(default: http://<--host>:<--port>)")
    ap.add_argument("--host", default="localhost", help="server host (default: localhost)")
    ap.add_argument("--port", type=int, default=8000,
                    help="server port (default: 8000; SGLang's launch_server uses 30000)")
    ap.add_argument("--api", choices=["completions", "chat"], default="completions",
                    help="which OpenAI-compatible route to drive (default: completions)")
    ap.add_argument("--model", default=None,
                    help="model name to send; if omitted, the first id from /v1/models")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64], metavar="N",
                    help="concurrency levels to sweep (default: 1 2 4 8 16 32 64)")
    ap.add_argument("--requests-per-level", type=int, default=32,
                    help="completed requests to measure at each level (default: 32). "
                         "Keep it well above the concurrency or the level is all warmup.")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed requests before each level (default: 2)")
    ap.add_argument("--input-len", type=int, default=512,
                    help="approximate prompt length in tokens (default: 512)")
    ap.add_argument("--output-len", type=int, default=128,
                    help="max_tokens per request (default: 128)")
    ap.add_argument("--ignore-eos", action="store_true", default=True,
                    help="pin every response to exactly --output-len tokens (default: on). "
                         "Both engines accept it on the OpenAI routes.")
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                    help="let the model stop naturally; output lengths then vary, and so "
                         "does the batch composition")
    ap.add_argument("--num-prompts", type=int, default=64,
                    help="distinct prompts to cycle through (default: 64). Distinct prompts "
                         "keep the prefix cache from turning this into a cache-hit benchmark.")
    ap.add_argument("--knee-threshold", type=float, default=0.25,
                    help="marginal throughput gain, as a fraction of linear, below which a "
                         "level counts as past the knee (default: 0.25)")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout in seconds (default: 600)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write every measured row to this JSON file")
    ap.add_argument("--seed", type=int, default=0, help="prompt generation seed (default: 0)")
    args = ap.parse_args()

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"
    args.base_url = args.base_url.rstrip("/")

    if args.model is None:
        args.model = discover_model(args)

    prompts = make_prompts(args)
    print(f"server   {args.base_url}  ({args.api})")
    print(f"model    {args.model}")
    print(f"workload {args.input_len}-token prompts, {args.output_len} output tokens, "
          f"ignore_eos={args.ignore_eos}")
    print(f"levels   {args.concurrency}, {args.requests_per_level} requests each")

    rows = []
    for conc in args.concurrency:
        if conc > args.requests_per_level:
            print(f"\n[skip] concurrency {conc} > --requests-per-level "
                  f"{args.requests_per_level}: the level would be pure ramp-up.")
            continue
        print(f"\n[level] concurrency {conc} ...", flush=True)
        row = run_level(args, conc, prompts)
        if row.get("failed"):
            print(f"  {row['failed']} request(s) failed: {row.get('first_error', '')}")
        rows.append(row)

    report(args, rows)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"config": vars(args), "levels": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")

    print("\nThese are measurements of YOUR server. Nothing in this script is a "
          "prediction; compare them against the ones you wrote down before running.")
    return 0


def discover_model(args) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=10) as resp:
            data = json.load(resp)
        return data["data"][0]["id"]
    except Exception as exc:                                    # noqa: BLE001
        sys.exit(f"Could not read {args.base_url}/v1/models ({exc.__class__.__name__}). "
                 f"Is the server up? Otherwise pass --model explicitly.")


def make_prompts(args) -> list[str]:
    """Distinct prompts of roughly --input-len tokens.

    Distinct matters. A single repeated prompt is served almost entirely out of
    the prefix cache after the first request, which turns a batching experiment
    into a cache experiment (that one is lab 04). A pseudo-random integer
    preamble per prompt defeats prefix sharing without needing a dataset.
    """
    import random
    rng = random.Random(args.seed)
    out = []
    for _ in range(args.num_prompts):
        # Roughly one token per whitespace-separated integer for most BPE
        # tokenizers; close enough for a load shape, not a token count you
        # should quote. The server's own logs have the real prompt length.
        words = [str(rng.randrange(10_000, 99_999)) for _ in range(args.input_len)]
        out.append(" ".join(words))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
