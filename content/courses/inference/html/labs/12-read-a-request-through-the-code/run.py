#!/usr/bin/env python3
"""Lab 12 — Read a request through the code.

Send one streaming request, then match every line the engine printed about it to
a file and a line number in the pinned source. The deliverable is a hop table:

    observed log line                          ->  file:line  ->  which process

Four modes:

  1. map       print the hop table with no server at all — the reference the
               other modes annotate against
  2. send      POST one streaming request, timestamp every SSE frame, print
               TTFT and the ITL distribution
  3. annotate  read the engine's stdout log and label each line with the
               file:line that emitted it
  4. calltrace read vLLM's VLLM_TRACE_FUNCTION logs and distil them into the
               call sequence for one request — this is ground truth, not a
               lookup table, because the tracer prints file:line itself

    python3 run.py --mode map --engine vllm
    python3 run.py --mode send --engine sglang --url http://localhost:30000
    python3 run.py --mode annotate --engine vllm --log ./server.log
    python3 run.py --mode calltrace --trace-dir /tmp/$USER/vllm
    python3 run.py --help

Stdlib only. `send` uses urllib; nothing here imports torch, requests, or an
engine. Modes 1, 3 and 4 need no server and no GPU.

The instrumentation this lab depends on, at the SHAs the book pins:
  vLLM   --enable-log-requests           vllm/engine/arg_utils.py:L2891-L2899
         VLLM_LOGGING_LEVEL=DEBUG        vllm/envs.py:L846
         VLLM_TRACE_FUNCTION=1           vllm/config/vllm.py:L796-L820
  SGLang --log-requests --log-requests-level 3
                                         python/sglang/srt/server_args.py:L1518-L1530
         --enable-request-time-stats-logging
                                         python/sglang/srt/server_args.py:L1647-L1649
         SGLANG_LOG_FORWARD_ITERS=1      python/sglang/srt/environ.py:L325

NO ENGINE WAS RUN WHILE THIS WAS WRITTEN. The hop table below was built by
reading the source, not by matching against a capture. Report anything that
does not line up.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- the hop map
#
# (hop, process, regex matching the emitted line, file:line, what it means)
#
# Every file:line below was read out of the pinned checkout. The regex matches
# the *rendered* line, so it is written against the format string's literal
# parts only — %s and {} substitutions are wildcards.

VLLM_HOPS: list[tuple[str, str, str, str, str]] = [
    ("1", "api-server", r"Received request .*: params:",
     "vllm/entrypoints/serve/utils/request_logger.py:L63",
     "RequestLogger.log_inputs, INFO. Needs --enable-log-requests."),
    ("1b", "api-server", r"Request .* details: prompt:",
     "vllm/entrypoints/serve/utils/request_logger.py:L53",
     "Same call, DEBUG branch. Needs VLLM_LOGGING_LEVEL=DEBUG too."),
    ("2", "api-server", r"Added request ",
     "vllm/v1/engine/async_llm.py:L439",
     "AsyncLLM._add_request, AFTER both the OutputProcessor registration and "
     "the ZMQ send. The request now exists in two processes."),
    ("3", "engine-core", r"EngineCore waiting for work\.",
     "vllm/v1/engine/core.py:L1429",
     "The busy loop is blocked on an empty input queue. DEBUG only. This line "
     "is the process boundary, visible."),
    ("4", "engine-core", r"EngineCore loop active\.",
     "vllm/v1/engine/core.py:L1441",
     "Something arrived and _process_input_queue drained it. The delta between "
     "hop 3 and hop 4 is your ZMQ + msgpack round trip."),
    ("5", "api-server", r"Avg prompt throughput: .* Avg generation throughput:",
     "vllm/v1/metrics/loggers.py:L310",
     "LoggingStatLogger.log, every VLLM_LOG_STATS_INTERVAL seconds (default 10). "
     "Running/Waiting counts here are the scheduler's, one step behind."),
    ("6", "api-server", r"Generated response .*: output:",
     "vllm/entrypoints/serve/utils/request_logger.py:L98",
     "RequestLogger.log_outputs. Emitted from the frontend, after "
     "detokenisation — so its timestamp is client-side, not engine-side."),
    ("7", "api-server", r"Request .* aborted\.",
     "vllm/v1/engine/async_llm.py:L900",
     "The generator was cancelled: client disconnect, or a stop string found in "
     "the frontend detokeniser."),
    ("8", "api-server", r"Aborted request\(s\) ",
     "vllm/v1/engine/async_llm.py:L761",
     "AsyncLLM.abort: frontend state popped AND an ABORT frame sent to "
     "EngineCore. Both copies of the request die here."),
]

SGLANG_HOPS: list[tuple[str, str, str, str, str]] = [
    ("1", "http-worker", r"Receive OpenAI: obj=",
     "python/sglang/srt/utils/request_logger.py:L156",
     "log_openai_received_request, before adaptation or tokenisation. Gated "
     "on --log-requests AND --log-requests-level >= 2 at "
     "python/sglang/srt/entrypoints/openai/serving_base.py:L88-L90."),
    ("2", "http-worker", r"Receive: obj=",
     "python/sglang/srt/utils/request_logger.py:L110",
     "log_received_request, on the internal GenerateReqInput. Verbosity from "
     "--log-requests-level; level 3 prints the whole prompt."),
    ("3", "scheduler", r"Prefill batch.*#new-seq: ",
     "python/sglang/srt/managers/scheduler_components/metrics_reporter.py:L604",
     "The request is in a running batch. #cached-token tells you what the "
     "radix cache saved. SGLANG_LOG_FORWARD_ITERS=1 adds the forward counter."),
    ("4", "scheduler", r"Decode batch.*#running-req: ",
     "python/sglang/srt/managers/scheduler_components/metrics_reporter.py:L811",
     "One per --decode-log-interval iterations (default 40), NOT one per step."),
    ("5", "scheduler", r"ReqTimeStats\(rid=",
     "python/sglang/srt/managers/schedule_batch.py:L1804",
     "Per-request queue_duration and forward_duration, on finish. Needs "
     "--enable-request-time-stats-logging; gated to attn_tp_rank 0."),
    ("6", "http-worker", r"Finish: obj=.*, out=",
     "python/sglang/srt/utils/request_logger.py:L191",
     "log_finished_request, back in the HTTP worker after the detokeniser "
     "process handed the text over."),
    ("7", "http-worker", r"Streaming backlog: rid=.*coalescing",
     "python/sglang/srt/managers/tokenizer_manager.py:L1619",
     "20+ chunks queued for one request: the HTTP side never ran. Per-token "
     "timing for this request is now fiction."),
    ("8", "http-worker", r"Abort request for rid=.* not found in rid_to_state",
     "python/sglang/srt/managers/tokenizer_manager.py:L3128",
     "The abort lost the race with the finish. Normal in ones; a load balancer "
     "timing clients out in bulk."),
    ("9", "scheduler", r"max_total_num_tokens=",
     "python/sglang/srt/managers/scheduler.py:L1095",
     "Startup, not per-request — but it is the line lab 02 reconciles against, "
     "and it is printed on tp_rank 0 only."),
]

HOPS = {"vllm": VLLM_HOPS, "sglang": SGLANG_HOPS}

LAUNCH_HINT = {
    "vllm": """VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> \\
    --enable-log-requests \\
    --max-log-len 256 \\
  2>&1 | tee server.log

# ground truth, one file per process per thread, VERY slow:
VLLM_TRACE_FUNCTION=1 VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --enforce-eager""",
    "sglang": """SGLANG_LOG_MS=1 SGLANG_LOG_FORWARD_ITERS=1 python3 -m sglang.launch_server \\
    --model-path <model> \\
    --log-level debug \\
    --log-requests --log-requests-level 3 \\
    --enable-request-time-stats-logging \\
    --decode-log-interval 1 \\
  2>&1 | tee server.log

# SGLANG_LOG_MS=1 is not optional for this lab: without it asctime is
# second-resolution and every hop delta rounds to 0 or 1000 ms.
# SGLang's format string carries no file or line at all
# (python/sglang/srt/utils/common.py:L2338-L2343), which is why the table
# above had to be built by hand. To make SGLang print them like vLLM does,
# point SGLANG_LOGGING_CONFIG_PATH at a dictConfig JSON whose format is
#   "[%(asctime)s.%(msecs)03d] [%(pathname)s:%(lineno)d] %(message)s"
# and it takes the custom-config branch at common.py:L2328-L2337 instead.""",
}

# ------------------------------------------------------------------- 1. map


def mode_map(a: argparse.Namespace) -> int:
    hops = HOPS[a.engine]
    print(f"\nHop table — {a.engine} at the SHA this book pins.")
    print("Read the source, not this table, if the two disagree.\n")
    w1 = max(len(h[1]) for h in hops)
    w2 = max(len(h[3]) for h in hops)
    print(f"{'hop':<4} {'process':<{w1}}  {'file:line':<{w2}}  meaning")
    print("-" * (8 + w1 + w2 + 40))
    for hop, proc, _rx, cite, why in hops:
        print(f"{hop:<4} {proc:<{w1}}  {cite:<{w2}}  {why.splitlines()[0]}")
        for extra in why.splitlines()[1:]:
            print(" " * (8 + w1 + w2) + extra)
    print("\nStart the server so these lines actually appear:\n")
    print(LAUNCH_HINT[a.engine])
    return 0


# ------------------------------------------------------------------ 2. send


def mode_send(a: argparse.Namespace) -> int:
    """One streaming request, every frame timestamped.

    Both engines stop their own TTFT clock at `yield`, not at the socket. This
    measures from the client, which is the only side that matters to an SLO.
    """
    import urllib.request

    base = a.url.rstrip("/")
    if a.native and a.engine == "sglang":
        url = f"{base}/generate"
        body = {
            "text": a.prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": a.max_tokens},
            "stream": True,
        }
    else:
        url = f"{base}/v1/chat/completions"
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": a.prompt}],
            "stream": True,
            "max_tokens": a.max_tokens,
            "temperature": 0.0,
        }
    if a.rid:
        body["rid" if a.native else "request_id"] = a.rid

    print(f"POST {url}")
    print(f"body {json.dumps(body)[:200]}")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.perf_counter()
    stamps: list[tuple[float, int, str]] = []
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            hdr_at = time.perf_counter() - t0
            print(f"HTTP {r.status}, headers at {hdr_at * 1e3:.1f} ms")
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    stamps.append((time.perf_counter() - t0, 0, "[DONE]"))
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                text = _delta_text(obj)
                stamps.append((time.perf_counter() - t0, len(text), text))
    except Exception as e:
        sys.exit(f"{type(e).__name__}: {e}\nIs the server up at {base}?")

    content = [s for s in stamps if s[1] > 0]
    if not content:
        sys.exit("No frames carried content. Every frame was bookkeeping — which is "
                 "itself the lesson: measuring TTFT from the FIRST frame is wrong.")

    ttft = content[0][0]
    itls = [b[0] - a_[0] for a_, b in zip(content, content[1:])]
    empties = len(stamps) - len(content) - 1

    table(
        [
            ("frames received", f"{len(stamps)}"),
            ("frames with no content", f"{empties}  (chunked-prefill bookkeeping; "
                                       "counting these as token 1 under-reports TTFT)"),
            ("TTFT, first frame WITH content", f"{ttft * 1e3:8.1f} ms"),
            ("TTFT, first frame at all", f"{stamps[0][0] * 1e3:8.1f} ms"),
            ("content frames", f"{len(content)}"),
            ("mean ITL", f"{1e3 * sum(itls) / len(itls):8.2f} ms" if itls else "n/a"),
            ("p50 ITL", f"{1e3 * _pct(itls, 50):8.2f} ms" if itls else "n/a"),
            ("p99 ITL", f"{1e3 * _pct(itls, 99):8.2f} ms" if itls else "n/a"),
            ("max ITL", f"{1e3 * max(itls):8.2f} ms" if itls else "n/a"),
            ("total", f"{stamps[-1][0] * 1e3:8.1f} ms"),
        ],
        "Measured from the client — the only clock an SLO is written against",
    )

    if itls and max(itls) > 4 * (sum(itls) / len(itls)):
        print("\nA max ITL several times the mean with the rest near zero is the")
        print("coalescing signature: the frontend fell behind and the mailbox merged")
        print("deltas. vLLM merges on put (RequestOutputCollector); SGLang appends and")
        print("merges on get, and warns at 20 queued chunks. Tokens survive; timing")
        print("does not. See chapter 09-03.")

    if a.frames:
        print("\n  #     t(ms)   dt(ms)  chars  text")
        prev = 0.0
        for i, (t, n, txt) in enumerate(stamps):
            print(f"{i:4d} {t * 1e3:9.2f} {(t - prev) * 1e3:8.2f} {n:6d}  {txt[:48]!r}")
            prev = t
    return 0


def _delta_text(obj: dict) -> str:
    if "choices" in obj:  # OpenAI shape
        ch = obj["choices"][0]
        d = ch.get("delta") or {}
        return d.get("content") or ch.get("text") or ""
    return obj.get("text", "")  # SGLang native shape


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


# --------------------------------------------------------------- 3. annotate


def mode_annotate(a: argparse.Namespace) -> int:
    """Label each log line with the file:line that emitted it."""
    path = Path(a.log)
    if not path.exists():
        sys.exit(f"No such log: {path}\n\nProduce one with:\n\n{LAUNCH_HINT[a.engine]}")
    hops = [(h[0], h[1], re.compile(h[2]), h[3], h[4]) for h in HOPS[a.engine]]

    matched: list[tuple[int, str, str, str, str]] = []
    unmatched = 0
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if a.rid and a.rid not in line:
            continue
        for hop, proc, rx, cite, _why in hops:
            if rx.search(line):
                matched.append((n, hop, proc, cite, line.strip()))
                break
        else:
            unmatched += 1

    if not matched:
        print(f"No known hop matched in {path} ({unmatched} lines scanned).")
        print("\nThe usual cause is that the instrumentation was never turned on:\n")
        print(LAUNCH_HINT[a.engine])
        return 1

    w = max(len(m[3]) for m in matched)
    print(f"\n{path.name}: {len(matched)} hop line(s), {unmatched} line(s) with no "
          f"known emitter\n")
    print(f"{'line':>6}  {'hop':<4} {'process':<12} {'file:line':<{w}}  observed")
    print("-" * (30 + w + 60))
    for n, hop, proc, cite, text in matched:
        print(f"{n:6d}  {hop:<4} {proc:<12} {cite:<{w}}  {text[: a.width]}")

    seen = [m[1] for m in matched]
    expected = [h[0] for h in HOPS[a.engine]]
    missing = [h for h in expected if h not in seen]
    if missing:
        print(f"\nHops with no line in this log: {', '.join(missing)}")
        print("Each is either a flag you did not pass or a path this request did not")
        print("take. Decide which before you call the trace complete.")
    if unmatched and a.unmatched:
        print("\nLines with no known emitter (the interesting ones are here — every")
        print("unrecognised line is a hop this lab's table does not yet cover):")
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if not any(rx.search(line) for _h, _p, rx, _c, _w in hops):
                print(f"{n:6d}  {line.strip()[: a.width]}")
    return 0


# --------------------------------------------------------------- 4. calltrace


# vllm/logger.py:L273-L288 writes exactly this shape, one line per call/return.
TRACE_RE = re.compile(
    r"^(?P<ts>[\d\-: .]+) (?P<ev>Call to|Return from) (?P<fn>\S+) in "
    r"(?P<file>\S+):(?P<line>\d+) (?:from|to) (?P<caller>\S+) in (?P<cfile>\S+):(?P<cline>\d+)$"
)

# The functions chapter 09-03 names as hops. Anything else is noise at this
# altitude — a request touches tens of thousands of frames.
DEFAULT_FILTER = [
    "create_chat_completion", "render_chat", "render_chat_async",
    "tokenize_prompts_async", "_create_chat_completion", "generate",
    "add_request", "_add_request", "add_request_async", "_send_input",
    "run_busy_loop", "_process_input_queue", "_handle_client_request",
    "schedule", "execute_model", "sample_tokens", "update_from_output",
    "check_stop", "process_outputs", "make_request_output",
    "chat_completion_stream_generator", "abort", "finish_requests",
    "_free_request", "_free_blocks",
]


def mode_calltrace(a: argparse.Namespace) -> int:
    """Distil VLLM_TRACE_FUNCTION output into the hop sequence.

    This mode needs no lookup table: the tracer prints file:line itself, so what
    it produces IS the answer the lab asks for. The cost is that tracing every
    Python call makes the engine unusably slow — run it with one request and
    --enforce-eager, never under load.
    """
    root = Path(a.trace_dir)
    if not root.exists():
        sys.exit(f"No such directory: {root}\n\nvLLM writes these under\n"
                 "  $TMPDIR/$USER/vllm/vllm-instance-<id>/\n"
                 "(vllm/config/vllm.py:L801-L820). Start the server with "
                 "VLLM_TRACE_FUNCTION=1.")
    files = sorted(root.rglob("VLLM_TRACE_FUNCTION_for_process_*.log"))
    if not files:
        sys.exit(f"No VLLM_TRACE_FUNCTION_* files under {root}.")

    wanted = a.filter.split(",") if a.filter else DEFAULT_FILTER
    print(f"{len(files)} trace file(s) under {root}")
    print(f"filtering to {len(wanted)} function name(s); pass --filter to change, "
          f"--filter '' for everything\n")

    total = 0
    for f in files:
        rows: list[tuple[str, str, str, str]] = []
        with f.open(errors="replace") as fh:
            for line in fh:
                m = TRACE_RE.match(line.rstrip())
                if not m or m.group("ev") != "Call to":
                    continue
                total += 1
                fn = m.group("fn")
                if wanted and fn not in wanted:
                    continue
                rows.append((m.group("ts")[-15:], fn,
                             f"{_rel(m.group('file'))}:{m.group('line')}",
                             m.group("caller")))
        if not rows:
            continue
        # process id and thread id are in the filename; that is how you tell the
        # API server's trace from EngineCore's.
        print(f"=== {f.name}")
        w = max(len(r[1]) for r in rows)
        for ts, fn, cite, caller in rows[: a.limit]:
            print(f"  {ts}  {fn:<{w}}  {cite:<52}  <- {caller}")
        if len(rows) > a.limit:
            print(f"  ... {len(rows) - a.limit} more (raise --limit)")
        print()

    print(f"{total:,} call events scanned.")
    print("One file per process per thread. Compare the API-server file against the")
    print("EngineCore file: the hop that appears in one and not the other IS the")
    print("process boundary, and nothing about it is inferred.")
    return 0


def _rel(p: str) -> str:
    """Absolute site-packages path -> the repo-relative path the book cites.

    SGLang ships a byte-identical copy of this tracer at
    python/sglang/multimodal_gen/runtime/utils/logging_utils.py:L431-L451, so the
    same parser works there if you arrange to call enable_trace_function_call
    with root_dir pointed at srt/. Handle both spellings.
    """
    for marker, keep in (("/python/sglang/", 1), ("/vllm/", 1)):
        i = p.rfind(marker)
        if i >= 0:
            return p[i + keep:]
    return p


# ---------------------------------------------------------------------- shared


def table(rows: list[tuple[str, str]], title: str) -> None:
    if not rows:
        return
    w = max(len(k) for k, _ in rows)
    print(f"\n{title}\n" + "-" * (w + 34))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Watch one request cross every hop, and name the file:line for each.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["map", "send", "annotate", "calltrace"],
                    help="map: print the hop table. send: one streaming request. "
                         "annotate: label a server log. calltrace: distil "
                         "VLLM_TRACE_FUNCTION output.")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"],
                    help="which engine's hop table to use (default: vllm)")

    s = ap.add_argument_group("send")
    s.add_argument("--url", default="http://localhost:8000",
                   help="server base URL (SGLang usually :30000)")
    s.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="model name for the OpenAI payload")
    s.add_argument("--prompt", default="Count to twenty, one number per line.",
                   help="the prompt to send")
    s.add_argument("--max-tokens", type=int, default=32,
                   help="max tokens to generate (default: 32)")
    s.add_argument("--native", action="store_true",
                   help="SGLang only: use /generate instead of /v1/chat/completions, "
                        "which skips the OpenAI adaptation hop")
    s.add_argument("--rid", default=None,
                   help="request id to send AND to grep for in --mode annotate")
    s.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout, seconds")
    s.add_argument("--frames", action="store_true",
                   help="print every SSE frame with its timestamp")

    n = ap.add_argument_group("annotate")
    n.add_argument("--log", help="path to the server's captured stdout+stderr")
    n.add_argument("--width", type=int, default=110, help="truncate observed lines")
    n.add_argument("--unmatched", action="store_true",
                   help="also print lines no hop pattern matched")

    c = ap.add_argument_group("calltrace")
    c.add_argument("--trace-dir", default="/tmp",
                   help="directory holding VLLM_TRACE_FUNCTION_* logs")
    c.add_argument("--filter", default=None,
                   help="comma-separated function names; empty string for all")
    c.add_argument("--limit", type=int, default=60, help="rows per file (default: 60)")

    a = ap.parse_args()
    if a.mode == "annotate" and not a.log:
        ap.error("--mode annotate needs --log")
    return {
        "map": mode_map, "send": mode_send,
        "annotate": mode_annotate, "calltrace": mode_calltrace,
    }[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
