#!/usr/bin/env python3
"""Lab 13 — Batch invariance.

Send the same prompt at several batch sizes with the sampler pinned, diff the
outputs bitwise, then turn the determinism flag on and pay for it.

The engines already ship the right harnesses. USE THEM — this script exists to
drive them, to reproduce the effect over plain HTTP when you cannot run pytest,
and to price the flag. It deliberately does not reimplement their assertions.

    tests/v1/determinism/                     vLLM's correctness suite
    python/sglang/test/test_deterministic.py  SGLang's, as a live-server driver

DO NOT use vllm/benchmarks/benchmark_batch_invariance.py to answer the
correctness question. It inserts a needle prompt into random batches, computes a
`baseline_text`, and then asserts only `needle_output.prompt == needle_prompt`
(benchmarks/benchmark_batch_invariance.py:L148-L152 and L190-L192). It never
compares the generated text against the baseline. It is a performance harness,
and reading its green output as a determinism result is the mistake this lab
exists to prevent.

Modes:

  1. harness  print the exact upstream commands, both engines, both arms.
              No server, no GPU, no network. Start here.
  2. sweep    send one prompt at batch sizes 1..N over HTTP and diff the target
              completion's token ids and per-step logprobs against the B=1
              baseline. Works against either engine, unmodified.
  3. cost     time the same fixed workload with the flag off and on, and report
              the overhead. Two servers, or one restarted.
  4. report   re-read a saved --save file and print the table again.

    python3 run.py --mode harness
    python3 run.py --mode sweep --engine vllm --url http://localhost:8000 --max-batch 32
    python3 run.py --mode sweep --engine sglang --url http://localhost:30000
    python3 run.py --mode cost --url http://localhost:8000 --invariant http://localhost:8001
    python3 run.py --help

Stdlib only — urllib, no requests, no torch, no openai client.

Flags this lab drives, at the SHAs the book pins:
  vLLM   VLLM_BATCH_INVARIANT=1            vllm/envs.py:L629
         VLLM_TEST_MODEL                   tests/v1/determinism/utils.py:L39-L40
  SGLang --enable-deterministic-inference  python/sglang/srt/server_args.py:L3433-L3437

NOTHING HERE WAS RUN AGAINST A GPU. Every threshold below is arithmetic or a
citation; the only measurements are the ones your own run produces.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

HARNESS = """
=== vLLM: the correctness suite ===

  # The conftest turns invariance ON for every test in the directory
  # (tests/v1/determinism/conftest.py:L9-L12), so the "should_fail" test has to
  # turn it back off itself. Run the pair and read them together.

  # 1. Prove the effect exists. PASSES when outputs differ.
  pytest -s tests/v1/determinism/test_batch_invariance.py \\
      -k "without_batch_invariance_should_fail"

  # 2. Prove the flag fixes it. PASSES when BS=1 and BS=N match bitwise.
  VLLM_BATCH_INVARIANT=1 pytest -s tests/v1/determinism/test_batch_invariance.py \\
      -k "bitwise_batch_invariance_bs1_vs_bsN"

  # 3. The same experiment against a real server, over HTTP. This is the one
  #    this script's --mode sweep imitates.
  VLLM_TEST_MODEL=Qwen/Qwen3-1.7B VLLM_TP_SIZE=1 \\
      pytest -s tests/v1/determinism/test_online_batch_invariance.py

  # Op-level, if a model-level test fails and you need to localise it:
  pytest -s tests/v1/determinism/test_matmul_batch_invariant.py
  pytest -s tests/v1/determinism/test_rms_norm_batch_invariant.py
  pytest -s tests/v1/determinism/test_cutlass_batch_invariance.py
  pytest -s tests/v1/determinism/test_nvfp4_batch_invariant.py

=== SGLang: the live-server driver ===

  # Baseline arm — no determinism flag.
  python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
      --attention-backend triton --cuda-graph-max-bs-decode 32

  # In another shell. Sends the same prompt at batch 1..n and prints
  # "Total samples: N, Unique samples: M". M > 1 is the bug, reproduced.
  python3 -m sglang.test.test_deterministic --n-trials 50 --test-mode single

  # Determinism arm — restart with the flag, rerun, expect M == 1.
  python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
      --attention-backend triton --cuda-graph-max-bs-decode 32 \\
      --enable-deterministic-inference

  # The harder modes. `prefix` varies shared-prefix length across 1, 511, 2048
  # and 4097 tokens, which exercises the chunked-prefill alignment constraint;
  # `radix_cache` compares a cached prefill against an uncached one.
  python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 20 --return-logprob
  python3 -m sglang.test.test_deterministic --test-mode radix_cache

  # Server args the in-tree test class uses, verbatim
  # (python/sglang/test/test_deterministic_utils.py:L12-L18):
  #   --trust-remote-code --cuda-graph-max-bs-decode 32 --enable-deterministic-inference

=== What NOT to run for this question ===

  benchmarks/benchmark_batch_invariance.py   # performance only; never diffs text
"""


# ------------------------------------------------------------------- transport


def post(url: str, body: dict, timeout: float) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}:\n{e.read().decode(errors='replace')[:600]}")
    except Exception as e:
        sys.exit(f"{type(e).__name__} talking to {url}: {e}")


def send_batch(a: argparse.Namespace, url: str, n: int) -> list[tuple[list[int], list[float]]]:
    """One request carrying n copies of the same prompt. Returns per-copy
    (token_ids, per-step logprobs).

    Batching inside a single request is what the vLLM online test does
    (tests/v1/determinism/test_online_batch_invariance.py:L88-L103): the engine
    schedules all n prompts together, which is exactly the condition under test.
    """
    if a.engine == "vllm":
        body = {
            "model": a.model,
            "prompt": [a.prompt] * n,
            "max_tokens": a.max_tokens,
            "temperature": a.temperature,
            "top_p": 1.0,
            "seed": a.seed,
            "logprobs": 1,
        }
        resp = post(f"{url.rstrip('/')}/v1/completions", body, a.timeout)
        out = []
        for ch in resp["choices"]:
            lp = ch.get("logprobs") or {}
            ids = lp.get("token_ids")
            if ids is None:  # some builds return only text tokens
                ids = lp.get("tokens", [])
            out.append((list(ids), list(lp.get("token_logprobs") or [])))
        return out

    body = {
        "text": [a.prompt] * n,
        "sampling_params": {
            "temperature": a.temperature,
            "max_new_tokens": a.max_tokens,
            "sampling_seed": a.seed,
        },
        "return_logprob": True,
    }
    resp = post(f"{url.rstrip('/')}/generate", body, a.timeout)
    resp = resp if isinstance(resp, list) else [resp]
    out = []
    for item in resp:
        meta = item.get("meta_info", {})
        # SGLang's output_token_logprobs is a list of [logprob, token_id, text]
        triples = meta.get("output_token_logprobs") or []
        out.append(([int(t[1]) for t in triples], [float(t[0]) for t in triples]))
    return out


# ------------------------------------------------------------------- 2. sweep


def ulp_gap(a_: float, b: float) -> int:
    """Distance in float64 units in the last place. 0 means bitwise identical."""
    ia = struct.unpack("<q", struct.pack("<d", a_))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    if ia < 0:
        ia = (1 << 63) - ia
    if ib < 0:
        ib = (1 << 63) - ib
    return abs(ia - ib)


def mode_sweep(a: argparse.Namespace) -> int:
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else _ladder(a.max_batch)
    print(f"engine   : {a.engine}   {a.url}")
    print(f"prompt   : {a.prompt[:70]!r}")
    print(f"sampling : temperature={a.temperature} seed={a.seed} max_tokens={a.max_tokens}")
    print(f"batches  : {sizes}")
    print("\nEvery batch carries the SAME prompt n times, in ONE request, so the")
    print("engine schedules them together. Copy 0 is the one compared each time.\n")

    base = send_batch(a, a.url, 1)[0]
    if not base[0]:
        sys.exit("The B=1 response carried no token ids. For vLLM pass a server that\n"
                 "returns logprobs; for SGLang check return_logprob is honoured.")
    print(f"baseline: {len(base[0])} tokens, {len(base[1])} logprobs")

    rows, results = [], []
    for n in sizes:
        t0 = time.perf_counter()
        outs = send_batch(a, a.url, n)
        dt = time.perf_counter() - t0
        got = outs[a.compare_index if a.compare_index < len(outs) else 0]
        div = _first_divergence(base[0], got[0])
        gaps = [ulp_gap(x, y) for x, y in zip(base[1], got[1])]
        max_ulp = max(gaps) if gaps else 0
        first_lp = next((i for i, g in enumerate(gaps) if g), None)
        rows.append((
            f"{n:5d}",
            "same" if div is None else f"tok {div}",
            f"{max_ulp:,}" if max_ulp else "bitwise",
            "-" if first_lp is None else str(first_lp),
            f"{dt * 1e3:7.0f} ms",
        ))
        results.append({"batch": n, "first_token_divergence": div,
                        "max_logprob_ulp": max_ulp, "first_logprob_divergence": first_lp,
                        "wall_ms": dt * 1e3, "token_ids": got[0]})
        verdict = ("bitwise identical" if div is None and not max_ulp
                   else "DIFFERENT TOKENS" if div is not None
                   else "same tokens, logprobs moved")
        print(f"  B={n:<5d} {verdict}")

    print("\n batch  tokens vs B=1   max logprob ULP   first lp diff   wall")
    print("-" * 68)
    for r in rows:
        print(f"{r[0]}  {r[1]:<15} {r[2]:>15}   {r[3]:>13}   {r[4]}")

    diverged = [r for r in results if r["first_token_divergence"] is not None]
    lp_moved = [r for r in results if r["max_logprob_ulp"]]
    print()
    if diverged:
        print(f"{len(diverged)}/{len(results)} batch sizes produced DIFFERENT TOKENS.")
        print("Batch invariance does not hold on this server. Expected, with the flag off.")
    elif lp_moved:
        print(f"Tokens matched everywhere, but logprobs moved at "
              f"{len(lp_moved)}/{len(results)} batch sizes.")
        print("The argmax happened to survive the perturbation. It will not always:")
        print("a tie at a high-entropy position flips on a few ULP. Token equality is")
        print("the weaker claim; the logprob column is the one that tells you the")
        print("arithmetic changed. See chapter 10-04.")
    else:
        print("Bitwise identical at every batch size tried.")
        print("Either the flag is on, or you did not vary the batch enough to move a")
        print("reduction tree. Under CUDA graphs the effect is a STEP function: every")
        print("batch inside one captured bucket replays identical launch geometry and")
        print("agrees trivially. Re-run with sizes that straddle a bucket riser — for")
        print("vLLM's default ladder, 16 and 17 — or with graphs off.")

    if a.save:
        Path(a.save).write_text(json.dumps(
            {"engine": a.engine, "url": a.url, "prompt": a.prompt,
             "temperature": a.temperature, "seed": a.seed,
             "max_tokens": a.max_tokens, "results": results}, indent=2))
        print(f"\nsaved -> {a.save}")
    return 0


def _ladder(n: int) -> list[int]:
    """Sizes that straddle vLLM's CUDA-graph bucket risers rather than avoiding them."""
    cands = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 48, 64, 96, 128]
    return [c for c in cands if c <= n]


def _first_divergence(a_: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a_, b)):
        if x != y:
            return i
    return None if len(a_) == len(b) else min(len(a_), len(b))


# -------------------------------------------------------------------- 3. cost


def mode_cost(a: argparse.Namespace) -> int:
    """Same workload, flag off and on. Two servers is the honest way to do it.

    Chapter 10-04 argues the cost is not where intuition puts it: pinned
    attention splits and fixed GEMM tiles are nearly free at small batch because
    those layers are bandwidth-bound anyway; single-channel NCCL and disabled
    prefix caching are what you actually pay. So run this at TP=1 AND at your
    real TP, and on a shared-prefix workload AND a cold one. One number is not
    an answer.
    """
    if not a.invariant:
        sys.exit("--mode cost needs both --url (flag off) and --invariant (flag on).\n"
                 "Start two servers, or restart one and run this twice with --save.")

    def timed(url: str, label: str) -> dict:
        send_batch(a, url, a.batch)  # warm
        lat = []
        for _ in range(a.trials):
            t0 = time.perf_counter()
            send_batch(a, url, a.batch)
            lat.append(time.perf_counter() - t0)
        lat.sort()
        d = {"label": label, "url": url, "trials": a.trials,
             "min_ms": lat[0] * 1e3, "median_ms": lat[len(lat) // 2] * 1e3,
             "max_ms": lat[-1] * 1e3}
        print(f"  {label:<16} min {d['min_ms']:8.1f} ms   median {d['median_ms']:8.1f} ms"
              f"   max {d['max_ms']:8.1f} ms")
        return d

    print(f"workload: batch {a.batch}, {a.max_tokens} new tokens, {a.trials} trials\n")
    off = timed(a.url, "flag off")
    on = timed(a.invariant, "flag on")

    ratio = on["median_ms"] / off["median_ms"]
    print(f"\noverhead (median): {100 * (ratio - 1):+.1f}%   ({ratio:.3f}x)")
    print("\nMeasured, on your hardware, for this workload only. Before quoting it:")
    print("  - was prefix caching on in the baseline? Under determinism vLLM disables")
    print("    it for FLASHINFER and TRITON_MLA, and that alone can dominate.")
    print("  - was TP > 1? The NCCL pinning (tree algo, one channel, NCCL_NTHREADS=1)")
    print("    is inert at world size 1 and expensive above it.")
    print("  - did both arms use the same attention backend? SGLang silently picks a")
    print("    deterministic-capable one when the flag is on.")
    if a.save:
        Path(a.save).write_text(json.dumps({"off": off, "on": on, "ratio": ratio}, indent=2))
        print(f"\nsaved -> {a.save}")
    return 0


# ------------------------------------------------------------------ 4. report


def mode_report(a: argparse.Namespace) -> int:
    if not a.save or not Path(a.save).exists():
        sys.exit("--mode report needs --save pointing at a file a previous run wrote.")
    blob = json.loads(Path(a.save).read_text())
    print(json.dumps(blob, indent=2)[:8000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff one prompt's output across batch sizes, then price the fix.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", required=True, choices=["harness", "sweep", "cost", "report"],
                    help="harness: print the upstream commands. sweep: diff across "
                         "batch sizes. cost: time flag off vs on. report: reprint a save.")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"],
                    help="which API shape to speak (default: vllm)")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="server with the flag OFF (SGLang usually :30000)")
    ap.add_argument("--invariant", default=None,
                    help="server with the flag ON, for --mode cost")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="model name for the vLLM payload; matches the default in "
                         "tests/v1/determinism/utils.py (default: Qwen/Qwen3-1.7B)")
    ap.add_argument("--prompt",
                    default="Tell me about Richard Feynman: ",
                    help="the prompt sent at every batch size; the default is the one "
                         "SGLang's own harness uses")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="tokens to generate (default: 64; more tokens, more chances "
                         "for one flipped argmax to fork the completion)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 is greedy — which is NOT the same as deterministic "
                         "(default: 0.0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed; fixes the draws, not the probabilities they "
                         "are applied to (default: 42)")
    ap.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout, seconds")

    s = ap.add_argument_group("sweep")
    s.add_argument("--max-batch", type=int, default=32,
                   help="largest batch size in the default ladder (default: 32)")
    s.add_argument("--sizes", default=None,
                   help="explicit comma-separated batch sizes, e.g. 1,16,17,24")
    s.add_argument("--compare-index", type=int, default=0,
                   help="which copy in the batch to compare (default: 0)")

    c = ap.add_argument_group("cost")
    c.add_argument("--batch", type=int, default=32,
                   help="batch size for the timing workload (default: 32)")
    c.add_argument("--trials", type=int, default=10, help="timed repetitions (default: 10)")

    ap.add_argument("--save", default=None, help="write results as JSON here")

    a = ap.parse_args()
    if a.mode == "harness":
        print(HARNESS)
        return 0
    return {"sweep": mode_sweep, "cost": mode_cost, "report": mode_report}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
