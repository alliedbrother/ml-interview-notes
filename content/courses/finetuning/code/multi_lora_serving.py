"""Multi-LoRA serving: batched adapter math, adapter-cache simulation, vLLM benchmark.

Module: 20 — Production: Serving Adapters, MLOps, and Cost Planning (Lab L20.1).

Smoke mode (CPU, torch only, < 10 s):
  1. A batched multi-adapter linear layer (gather A/B by adapter id, heterogeneous
     ranks zero-padded to the max rank, id -1 = no adapter) checked against a
     per-request loop, plus the padding-waste FLOP count.
  2. A two-level LRU simulator for vLLM's `max_loras` (GPU slots) and
     `max_cpu_loras` (host cache) under Zipf-skewed tenant traffic.

Full mode (tier T1: one 24 GB GPU; vLLM 0.31.x, PEFT 0.21.x, Transformers v5):
  launches `vllm serve` on 127.0.0.1 with three adapters, benchmarks mixed-adapter
  throughput on module 01's eval rows (system prompt + email, thinking off), loads
  a fourth adapter at runtime (local only: runtime loading lets
  any API caller load arbitrary weights), and benchmarks a merged model.

Run (from the repository root):
    python content/courses/finetuning/code/multi_lora_serving.py --smoke
    python content/courses/finetuning/code/multi_lora_serving.py --full \
        --base Qwen/Qwen3-1.7B --adapters a=./adapters/a b=./adapters/b c=./adapters/c \
        --dynamic d=./adapters/d --merged ./merged-a --eval-jsonl ./ft_data/eval.jsonl
"""
import argparse
import collections
import json
import time


# ----------------------------------------------------------------------------- smoke
def batched_lora_check():
    import torch

    torch.manual_seed(0)
    d_out, d_in, alpha = 64, 48, 16.0
    ranks = [2, 4, 8, 16]
    w0 = torch.randn(d_out, d_in)
    a_list = [torch.randn(r, d_in) * 0.1 for r in ranks]
    b_list = [torch.randn(d_out, r) * 0.1 for r in ranks]
    ids = torch.randint(-1, len(ranks), (32,))
    x = torch.randn(len(ids), d_in)                                  # (n, d_in)

    ref = torch.stack([
        w0 @ x[i] + ((alpha / ranks[k]) * (b_list[k] @ (a_list[k] @ x[i])) if k >= 0 else 0)
        for i, k in enumerate(ids.tolist())
    ])

    r_max, slots = max(ranks), len(ranks) + 1
    a_pad = torch.zeros(slots, r_max, d_in)
    b_pad = torch.zeros(slots, d_out, r_max)
    for k, r in enumerate(ranks):
        a_pad[k, :r] = a_list[k]
        b_pad[k, :, :r] = (alpha / r) * b_list[k]
    slot = torch.where(ids < 0, torch.tensor(slots - 1), ids)
    shrink = torch.bmm(a_pad[slot], x.unsqueeze(-1))                # (n, r_max, 1)
    out = x @ w0.T + torch.bmm(b_pad[slot], shrink).squeeze(-1)     # (n, d_out)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)

    adapted = [k for k in ids.tolist() if k >= 0]
    padded = sum(2 * r_max * (d_in + d_out) for _ in adapted)
    useful = sum(2 * ranks[k] * (d_in + d_out) for k in adapted)
    waste = (padded - useful) / padded
    print("batched == loop for %d requests; padding wastes %.1f%% of LoRA FLOPs" % (len(ids), 100 * waste))
    assert 0.0 < waste < 1.0
    return waste


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = collections.OrderedDict()

    def touch(self, key):
        """Return True on hit; insert (evicting LRU) on miss."""
        if key in self.items:
            self.items.move_to_end(key)
            return True
        self.items[key] = None
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return False


def simulate_cache(n_adapters, max_loras, max_cpu_loras, n_steps=2000, batch=16, zipf_s=1.1, seed=0):
    """Each step draws a batch of adapter ids; distinct ids must sit in GPU slots.

    Simplification: a step whose distinct adapters exceed `max_loras` is split into
    sub-batches (extra scheduling rounds), which is the throughput cost of a low
    `max_loras`. Returns GPU-hit, CPU-hit, disk-load rates and rounds per step.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_adapters + 1) ** zipf_s
    weights /= weights.sum()
    gpu, cpu = LRU(max_loras), LRU(max_cpu_loras)
    hits = {"gpu": 0, "cpu": 0, "disk": 0}
    rounds = 0
    for _ in range(n_steps):
        distinct = list(dict.fromkeys(rng.choice(n_adapters, size=batch, p=weights).tolist()))
        rounds += -(-len(distinct) // max_loras)
        for k in distinct:
            if gpu.touch(k):
                hits["gpu"] += 1
            elif cpu.touch(k):
                hits["cpu"] += 1
            else:
                hits["disk"] += 1
    total = sum(hits.values())
    return {k: v / total for k, v in hits.items()}, rounds / n_steps


def smoke(eval_jsonl=None):
    batched_lora_check()
    results = {}
    for max_loras, max_cpu in [(1, 8), (4, 8), (4, 64), (8, 64), (8, 200)]:
        rates, rounds = simulate_cache(200, max_loras, max_cpu)
        results[(max_loras, max_cpu)] = (rates, rounds)
        print("max_loras=%d max_cpu_loras=%3d  gpu-hit %.2f  cpu-hit %.2f  disk %.3f  rounds/step %.2f" % (
            max_loras, max_cpu, rates["gpu"], rates["cpu"], rates["disk"], rounds))
    # A bigger host cache never increases disk loads; more GPU slots never add rounds.
    assert results[(4, 64)][0]["disk"] < results[(4, 8)][0]["disk"]
    assert results[(8, 200)][0]["disk"] <= results[(8, 64)][0]["disk"]
    assert results[(8, 64)][1] < results[(4, 64)][1] < results[(1, 8)][1]
    assert results[(8, 200)][0]["disk"] < 0.15        # 200 adapters all fit in host cache: only cold loads
    if eval_jsonl:
        rows = load_eval(eval_jsonl)
        assert all(messages(r)[1]["content"] == r["email"] for r in rows)
        print("eval file %s: %d rows conform to {id, email, gold}" % (eval_jsonl, len(rows)))
    print("smoke: batched multi-LoRA and cache simulator verified")


# ------------------------------------------------------------------------------ full
FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
SYSTEM_PROMPT = (
    "You extract structured data from customer support emails. Reply with one JSON "
    "object and nothing else, with exactly these keys: "
    '"customer_name" (string), '
    '"order_id" (string: "ORD-" followed by the five-digit order number, e.g. "ORD-04217"), '
    '"issue_type" (one of "refund", "shipping", "defect", "account", "other"), '
    '"refund_requested" (true or false), '
    '"sentiment" (one of "positive", "neutral", "negative").'
)


def load_eval(path):
    """Read module 01's eval.jsonl: rows of exactly {id, email, gold}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert set(row) == {"id", "email", "gold"}, sorted(row)
            assert list(row["gold"]) == FIELDS, row["id"]
            rows.append(row)
    return rows


def messages(row):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["email"]}]


def full(args):
    import os
    import statistics
    import subprocess
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    rows = load_eval(args.eval_jsonl)
    url = "http://127.0.0.1:%d" % args.port

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read() or b"{}")

    def wait_ready():
        for _ in range(600):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=5)
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError("vLLM did not start")

    def bench(models, conc):
        def one(i):
            t = time.time()
            r = post("/v1/chat/completions", {
                "model": models[i % len(models)], "temperature": 0, "max_tokens": 256,
                "messages": messages(rows[i % len(rows)]),
                "chat_template_kwargs": {"enable_thinking": False}})
            return time.time() - t, r["usage"]["completion_tokens"]
        n = max(len(rows), conc * 4)
        t0 = time.time()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(one, range(n)))
        wall = time.time() - t0
        lat = sorted(r[0] for r in res)
        return sum(r[1] for r in res) / wall, lat[int(0.95 * (len(lat) - 1))], statistics.mean(lat)

    def serve(model, extra, env_extra):
        env = dict(os.environ, **env_extra)
        cmd = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(args.port)] + extra
        proc = subprocess.Popen(cmd, env=env)
        wait_ready()
        return proc

    names = [a.split("=", 1)[0] for a in args.adapters]
    lora_flags = ["--enable-lora", "--lora-modules", *args.adapters,
                  "--max-loras", str(len(args.adapters)), "--max-lora-rank", str(args.max_lora_rank),
                  "--max-cpu-loras", "8"]
    # Runtime loading is enabled ONLY for this localhost-bound process.
    proc = serve(args.base, lora_flags, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})
    try:
        for c in args.concurrency:
            tps, p95, mean = bench(names, c)
            print("multi-LoRA conc=%3d  %.0f out tok/s  p95 %.2fs  mean %.2fs" % (c, tps, p95, mean))
        if args.dynamic:
            name, path = args.dynamic.split("=", 1)
            post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
            tps, p95, _ = bench([name], 1)
            print("dynamically loaded %s: %.0f tok/s" % (name, tps))
            post("/v1/unload_lora_adapter", {"lora_name": name})
        base_out = post("/v1/chat/completions", {
            "model": args.base, "temperature": 0, "max_tokens": 128,
            "messages": messages(rows[0]), "chat_template_kwargs": {"enable_thinking": False}})
        print("negative control (base, no adapter):", base_out["choices"][0]["message"]["content"][:200])
    finally:
        proc.terminate()
        proc.wait()
    if args.merged:
        proc = serve(args.merged, [], {})
        try:
            for c in args.concurrency:
                tps, p95, mean = bench([args.merged], c)
                print("merged     conc=%3d  %.0f out tok/s  p95 %.2fs  mean %.2fs" % (c, tps, p95, mean))
        finally:
            proc.terminate()
            proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapters", nargs="+", default=[], help="name=path entries")
    p.add_argument("--dynamic", default=None, help="name=path to load at runtime")
    p.add_argument("--merged", default=None, help="merged model dir for comparison")
    p.add_argument("--eval-jsonl", default=None,
                   help="module 01 eval.jsonl (full mode default: ./ft_data/eval.jsonl; "
                        "with --smoke, also validates this file)")
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.full:
        if len(args.adapters) < 1:
            p.error("--full needs --adapters")
        args.eval_jsonl = args.eval_jsonl or "./ft_data/eval.jsonl"
        full(args)
    else:
        smoke(args.eval_jsonl)


if __name__ == "__main__":
    main()
