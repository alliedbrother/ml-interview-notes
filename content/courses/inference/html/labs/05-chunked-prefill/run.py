#!/usr/bin/env python3
"""Lab 05 — chunked prefill and inter-token-latency jitter.

One long prompt admitted into a batch of decoding requests stalls every one of
them for as long as its prefill takes. Chunked prefill splits that prefill across
iterations so the stall is bounded by one chunk instead of one prompt. This
script measures the stall, with and without it.

The workload is deliberately narrow, because a broad one hides the effect:

    * N short requests looping in a closed loop, generating steadily
    * one very long prompt injected part-way through
    * every token arrival timestamped, so the spike can be located in time

and the headline result is not a mean. It is the WORST inter-token gap the short
requests saw, and where it fell relative to the injection.

    # server A — chunked prefill on (vLLM's default)
    vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 32768 \
        --max-num-batched-tokens 2048 --port 8000

    python3 run.py --port 8000 --label chunked --json chunked.json

    # server B — off, the pathological baseline
    vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 32768 \
        --no-enable-chunked-prefill --max-num-batched-tokens 32768 --port 8001

    python3 run.py --port 8001 --label unchunked --json unchunked.json
    python3 run.py --compare chunked.json unchunked.json
    python3 run.py --help

Measurement caveats that will silently ruin the result if you skip them:

  * ITL is measured client-side off the SSE stream, one timestamp per data frame
    (the idiom is vLLM's own, vllm/benchmarks/lib/endpoint_request_func.py:L400-L412).
    If the server bundles several tokens into one frame, every gap you measure is
    a multi-token gap. SGLang's --stream-interval controls exactly that and
    defaults to 1 (python/sglang/srt/server_args.py:L1480-L1484); leave it there.
  * Run the two servers ONE AT A TIME on the same card. Two engines sharing a GPU
    measure contention, not chunking.
  * The long prompt must be long enough that its unchunked prefill dwarfs a decode
    step. At 8B/bf16 a 32k prompt is roughly two orders of magnitude larger; a 2k
    prompt is not, and you will measure noise.

Stdlib only — urllib and threads. Reads nothing from and writes nothing to either
engine checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


# --------------------------------------------------------------------------
# streamed request with per-token arrival times
# --------------------------------------------------------------------------

def stream_one(args, prompt: str, max_tokens: int, arrivals: list | None = None) -> dict:
    """One streaming request. If `arrivals` is given, append (abs_time, gap)
    for every token after the first, so gaps can be located in wall-clock time
    rather than only summarised."""
    import urllib.error
    import urllib.request

    if args.api == "chat":
        url = f"{args.base_url}/v1/chat/completions"
        body = {"model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0,
                "stream": True, "ignore_eos": args.ignore_eos}
    else:
        url = f"{args.base_url}/v1/completions"
        body = {"model": args.model, "prompt": prompt,
                "max_tokens": max_tokens, "temperature": 0.0,
                "stream": True, "ignore_eos": args.ignore_eos}

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST")

    out = {"ok": False, "ttft": 0.0, "latency": 0.0, "chunks": 0, "itl": [], "error": ""}
    start = time.perf_counter()
    last = start
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                c = choices[0]
                text = c.get("text") or (c.get("delta") or {}).get("content") or ""
                if not text:
                    continue
                now = time.perf_counter()
                if out["chunks"] == 0:
                    out["ttft"] = now - start
                else:
                    gap = now - last
                    out["itl"].append(gap)
                    if arrivals is not None:
                        arrivals.append((now, gap))
                out["chunks"] += 1
                last = now
        out["latency"] = time.perf_counter() - start
        out["ok"] = out["chunks"] > 0
        if not out["ok"]:
            out["error"] = "no content chunks — check --api and that streaming is on"
    except urllib.error.HTTPError as exc:
        out["error"] = f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = f"{exc.__class__.__name__}: {exc}"
    return out


# --------------------------------------------------------------------------
# the mixed workload
# --------------------------------------------------------------------------

def run_workload(args, short_prompts: list[str], long_prompt: str) -> dict:
    """Short requests loop for --duration seconds. At --inject-after seconds,
    --long-count long prompts are submitted. Every short-request token arrival is
    timestamped on one shared list."""
    arrivals: list[tuple[float, float]] = []
    lock = threading.Lock()
    stop_at = [0.0]
    short_results: list[dict] = []
    long_results: list[dict] = []
    idx = {"i": 0}

    def short_worker() -> None:
        local: list[tuple[float, float]] = []
        while time.perf_counter() < stop_at[0]:
            with lock:
                i = idx["i"]; idx["i"] = i + 1
            r = stream_one(args, short_prompts[i % len(short_prompts)],
                           args.short_output_len, local)
            with lock:
                short_results.append(r)
        with lock:
            arrivals.extend(local)

    def long_worker() -> None:
        r = stream_one(args, long_prompt, args.long_output_len)
        with lock:
            long_results.append(r)

    print(f"[warmup] {args.warmup} request(s) ...", flush=True)
    for _ in range(args.warmup):
        stream_one(args, short_prompts[0], args.short_output_len)

    t0 = time.perf_counter()
    stop_at[0] = t0 + args.duration
    inject_at = t0 + args.inject_after

    pool = ThreadPoolExecutor(max_workers=args.short_concurrency + args.long_count)
    futures = [pool.submit(short_worker) for _ in range(args.short_concurrency)]

    print(f"[load] {args.short_concurrency} short streams for {args.duration:.0f}s; "
          f"injecting {args.long_count} x {args.long_input_len}-token prompt(s) at "
          f"t+{args.inject_after:.0f}s", flush=True)

    while time.perf_counter() < inject_at:
        time.sleep(0.01)
    inject_wall = time.perf_counter()
    futures += [pool.submit(long_worker) for _ in range(args.long_count)]

    for f in futures:
        f.result()
    pool.shutdown(wait=True)

    arrivals.sort()
    return {
        "t0": t0, "inject_wall": inject_wall,
        "arrivals": arrivals,
        "short": short_results,
        "long": long_results,
        "wall": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarise(args, run: dict) -> dict:
    arrivals = run["arrivals"]
    if not arrivals:
        return {"label": args.label, "error": "no token arrivals recorded"}

    inject = run["inject_wall"]
    quiet = [g for t, g in arrivals if t < inject]
    # The stall window: from injection until the long request produced its first
    # token. That is exactly the interval during which the long prefill was
    # occupying iterations, chunked or not.
    long_ok = [r for r in run["long"] if r["ok"]]
    long_bad = [r for r in run["long"] if not r["ok"]]
    long_ttft = long_ok[0]["ttft"] if long_ok else 0.0
    # With no long TTFT there is no principled window end, so fall back to the
    # whole post-injection tail and say so rather than inventing a boundary.
    window_end = inject + long_ttft if long_ttft else float("inf")
    during = [g for t, g in arrivals if inject <= t <= window_end]
    after = [g for t, g in arrivals if t > window_end]

    worst_t, worst_g = max(arrivals, key=lambda tg: tg[1])
    all_gaps = [g for _, g in arrivals]

    return {
        "label": args.label,
        "base_url": args.base_url,
        "short_concurrency": args.short_concurrency,
        "long_input_len": args.long_input_len,
        "tokens_observed": len(arrivals),
        "itl_p50": pct(all_gaps, 50), "itl_p90": pct(all_gaps, 90),
        "itl_p99": pct(all_gaps, 99), "itl_max": max(all_gaps),
        "itl_quiet_p50": pct(quiet, 50), "itl_quiet_p99": pct(quiet, 99),
        "itl_during_p50": pct(during, 50), "itl_during_p99": pct(during, 99),
        "itl_during_max": max(during) if during else 0.0,
        "itl_after_p99": pct(after, 99),
        "worst_gap": worst_g,
        "worst_gap_offset": worst_t - inject,
        "spike_ratio": worst_g / pct(quiet, 50) if quiet and pct(quiet, 50) else 0.0,
        "long_ttft": long_ttft,
        "long_failed": len(long_bad),
        "long_error": long_bad[0]["error"] if long_bad else "",
        "short_completed": sum(1 for r in run["short"] if r["ok"]),
        "short_failed": sum(1 for r in run["short"] if not r["ok"]),
        "wall": run["wall"],
        "output_tok_s": len(arrivals) / run["wall"] if run["wall"] else 0.0,
    }


def grid(headers: list[str], rows: list[list[str]], title: str) -> None:
    if not rows:
        return
    w = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print(f"\n{title}")
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(r)))


def ms(x: float) -> str:
    return f"{1000 * x:.1f}"


def report_one(s: dict) -> None:
    if s.get("error"):
        print("\n" + s["error"])
        return
    grid(["window", "p50", "p90", "p99", "max"],
         [["whole run", ms(s["itl_p50"]), ms(s["itl_p90"]), ms(s["itl_p99"]), ms(s["itl_max"])],
          ["before injection", ms(s["itl_quiet_p50"]), "", ms(s["itl_quiet_p99"]), ""],
          ["during long prefill", ms(s["itl_during_p50"]), "", ms(s["itl_during_p99"]),
           ms(s["itl_during_max"])],
          ["after", "", "", ms(s["itl_after_p99"]), ""]],
         f"Inter-token latency, ms — measured, label={s['label']!r}")

    print(f"\nworst gap            {ms(s['worst_gap'])} ms, "
          f"{s['worst_gap_offset']:+.2f} s from injection")
    print(f"spike ratio          {s['spike_ratio']:.1f}x the quiet-window median ITL")
    print(f"long-prompt TTFT     {ms(s['long_ttft'])} ms "
          f"({s['long_input_len']} prompt tokens)")
    if s.get("long_failed"):
        print(f"  WARNING: {s['long_failed']} long request(s) FAILED -- {s['long_error']}")
        print("  Without a long prefill there is nothing to measure. The usual cause is a")
        print("  prompt longer than the server's max-model-len; check the error above.")
    print(f"short requests       {s['short_completed']} ok, {s['short_failed']} failed, "
          f"{s['output_tok_s']:.1f} tok/s aggregate over {s['wall']:.1f}s")
    print("\nThe number that matters is 'worst gap'. Chunked prefill does not remove")
    print("the spike; it bounds it at roughly one chunk of compute plus the decode")
    print("half of the same iteration. If your worst gap is close to the long")
    print("prompt's whole prefill time, chunking is not actually on — check the")
    print("server's startup log for 'Chunked prefill is enabled with")
    print("max_num_batched_tokens=N' (vLLM) or 'chunked_prefill_size=N' (SGLang).")


def report_compare(paths: list[str]) -> int:
    runs = []
    for p in paths:
        with open(p) as fh:
            runs.append(json.load(fh)["summary"])
    grid(["label", "ITL p50", "ITL p99", "worst gap", "spike x", "long TTFT", "tok/s"],
         [[r["label"], ms(r["itl_p50"]), ms(r["itl_p99"]), ms(r["worst_gap"]),
           f"{r['spike_ratio']:.1f}", ms(r["long_ttft"]), f"{r['output_tok_s']:.1f}"]
          for r in runs],
         "Side by side — measured, latencies in ms")
    if len(runs) == 2:
        a, b = runs
        if a["worst_gap"] and b["worst_gap"]:
            print(f"\nworst-gap ratio  {b['worst_gap'] / a['worst_gap']:.1f}x "
                  f"({b['label']} over {a['label']})")
        if a["long_ttft"] and b["long_ttft"]:
            print(f"long TTFT ratio  {b['long_ttft'] / a['long_ttft']:.2f}x")
        print("\nExpect the worst gap to move by orders of magnitude and the long")
        print("prompt's TTFT to barely move: chunking does not change the FLOP count")
        print("of a prefill, only how many iterations it is spread over. If TTFT got")
        print("much WORSE with chunking, your chunk is below the roofline ridge point")
        print("and each chunk is re-streaming the weights for too few tokens — raise it.")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure ITL spikes caused by a long prefill, with and without chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--compare", nargs=2, metavar="JSON",
                    help="skip measurement; print two saved runs side by side")
    ap.add_argument("--base-url", default=None,
                    help="server root (default: http://<--host>:<--port>)")
    ap.add_argument("--host", default="localhost", help="server host (default: localhost)")
    ap.add_argument("--port", type=int, default=8000,
                    help="server port (default: 8000; SGLang's launch_server uses 30000)")
    ap.add_argument("--api", choices=["completions", "chat"], default="completions",
                    help="which OpenAI-compatible route to drive (default: completions)")
    ap.add_argument("--model", default=None,
                    help="model name to send; if omitted, the first id from /v1/models")
    ap.add_argument("--label", default="run",
                    help="name for this run in the output and JSON (e.g. chunked / unchunked)")

    ap.add_argument("--short-concurrency", type=int, default=16,
                    help="short requests kept in flight (default: 16)")
    ap.add_argument("--short-input-len", type=int, default=128,
                    help="approximate prompt length of the short requests (default: 128)")
    ap.add_argument("--short-output-len", type=int, default=256,
                    help="max_tokens for the short requests (default: 256)")
    ap.add_argument("--long-input-len", type=int, default=32768,
                    help="approximate prompt length of the injected long request "
                         "(default: 32768). Must be <= the server's max-model-len.")
    ap.add_argument("--long-output-len", type=int, default=16,
                    help="max_tokens for the long request (default: 16); its decode is "
                         "not what you are measuring")
    ap.add_argument("--long-count", type=int, default=1,
                    help="long prompts to inject at once (default: 1). Try 2 to see "
                         "SGLang serialise them and vLLM interleave them.")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds of short-request load (default: 60)")
    ap.add_argument("--inject-after", type=float, default=20.0,
                    help="seconds into the run at which the long prompt is submitted "
                         "(default: 20). Leave enough quiet time for a clean baseline.")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed requests before the run (default: 2)")
    ap.add_argument("--ignore-eos", action="store_true", default=True,
                    help="pin every response to its max_tokens (default: on)")
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                    help="let responses stop naturally")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request timeout in seconds (default: 900)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write the summary and the raw gap series to this JSON file")
    ap.add_argument("--seed", type=int, default=0, help="prompt generation seed (default: 0)")
    args = ap.parse_args()

    if args.compare:
        return report_compare(args.compare)

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"
    args.base_url = args.base_url.rstrip("/")
    if args.inject_after >= args.duration:
        sys.exit("--inject-after must be less than --duration, or nothing is injected.")
    if args.model is None:
        args.model = discover_model(args)

    import random
    rng = random.Random(args.seed)

    def filler(n: int) -> str:
        # Distinct pseudo-random integers, so the prefix cache cannot serve any
        # of this. A repeated prompt would be a prefix-cache benchmark (lab 04).
        return " ".join(str(rng.randrange(10_000, 99_999)) for _ in range(n))

    short_prompts = [filler(args.short_input_len) for _ in range(32)]
    long_prompt = filler(args.long_input_len)

    print(f"server   {args.base_url}  ({args.api})")
    print(f"model    {args.model}")
    print(f"label    {args.label}")

    run = run_workload(args, short_prompts, long_prompt)
    summary = summarise(args, run)
    report_one(summary)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"config": {k: v for k, v in vars(args).items() if k != "compare"},
                       "summary": summary,
                       "gaps": [{"t_rel_inject": t - run["inject_wall"], "gap": g}
                                for t, g in run["arrivals"]]}, fh, indent=2)
        print(f"\nwrote {args.json}")

    print("\nEverything above was measured on your server. Run the other configuration "
          "and use --compare; a single run tells you nothing about chunking.")
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


if __name__ == "__main__":
    raise SystemExit(main())
