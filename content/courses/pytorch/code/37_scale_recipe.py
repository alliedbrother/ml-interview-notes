"""Scale the capstone recipe without changing its math (module 37).

Subcommands (each prints one JSON section; ``all`` runs them in order):

* ``profile``    torch.profiler on 3 recipe steps after 2 warmup steps, plus the
                 unprofiled step time and an fp32 matmul ceiling on this CPU.
* ``flops``      FlopCounterMode (manual attention) vs 6*P_matmul*tokens + 12*L*T*d*tokens.
* ``compile``    torch.compile(backend="aot_eager") loss/grad parity, graph-break
                 audit, and the KV-cache path's recompiles per length.
* ``checkpoint`` per-block activation checkpointing: saved bytes, grads, FLOPs.
* ``ddp``        2 gloo processes, DDP vs single-process full-batch step.
* ``fsdp``       2 gloo processes, FSDP2 (per block + root) vs the same reference.
* ``budget``     GPT-2 124M memory/FLOP/time arithmetic (computed, not run).

Without ``--small`` the model is the committed checkpoint (``load_pretrained``);
nothing is ever saved back. ``checkpoint``, ``ddp`` and ``fsdp`` build fresh
seeded models of the same config (their checks compare two computations of
the same math, so the weights do not matter). ``--small`` uses a randomly initialised
GPTConfig(n_layer=2, d_model=64, n_head=4, block_size=64). ``ddp``/``fsdp``
spawn ``--world-size`` processes with ``torch.multiprocessing.spawn`` (gloo,
``file://`` rendezvous in a TemporaryDirectory). CPU only.

Run from the repository root:
    python content/courses/pytorch/code/37_scale_recipe.py all --small
    python content/courses/pytorch/code/37_scale_recipe.py budget
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm import scale  # noqa: E402
from llm.model import GPT, GPTConfig  # noqa: E402

SMALL = dict(n_layer=2, d_model=64, n_head=4, block_size=64)
COMMANDS = ("profile", "flops", "compile", "checkpoint", "ddp", "fsdp", "budget")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", nargs="?", default="all", choices=COMMANDS + ("all",))
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="global batch (sequences)")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--small", action="store_true", help="2-layer d_model=64 model, random init")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def build_model(args) -> GPT:
    if args.small:
        torch.manual_seed(args.seed)
        return GPT(GPTConfig(**SMALL))
    from llm.train import load_pretrained
    model, _, _ = load_pretrained()
    return model


def run_spawned(worker, args, cfg: GPTConfig) -> dict:
    if args.world_size < 2 or args.batch_size % (2 * args.world_size):
        raise SystemExit("--world-size must be >= 2 and divide --batch-size / 2")
    cfg_dict = scale.make_cfg_dict(cfg, args.batch_size, args.seq_len, args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rank0.json"
        t0 = time.perf_counter()
        mp.spawn(worker, args=(args.world_size, str(Path(tmp) / "rendezvous"), cfg_dict, str(out)),
                 nprocs=args.world_size, join=True)
        result = json.loads(out.read_text())
    result["world_size"] = args.world_size
    result["spawn_seconds"] = round(time.perf_counter() - t0, 3)
    return result


def run(command: str, args, model: GPT, batch) -> dict:
    cfg = model.config
    if command == "profile":
        r = scale.profile_step(copy.deepcopy(model), batch, steps=3, warmup=2)
        r.pop("table")
        r["matmul_ceiling_gflops"] = round(scale.matmul_ceiling_gflops(), 1)
        return r
    if command == "flops":
        r = scale.flop_report(model, batch)
        step = scale.profile_step(copy.deepcopy(model), batch, steps=3, warmup=1)["step_ms_unprofiled"]
        r["step_ms_unprofiled"] = step
        r["achieved_gflops"] = round(r["measured"] / (step / 1e3) / 1e9, 2)
        return r
    if command == "compile":
        r = scale.compile_parity(copy.deepcopy(model), batch, backend="aot_eager")
        decoder = copy.deepcopy(model).eval()
        r["cached_decode"] = scale.cached_recompiles(decoder, steps=5)
        r["cached_decode_static"] = scale.cached_recompiles(decoder, steps=5, automatic_dynamic=False)
        return r
    if command == "checkpoint":
        r = scale.checkpoint_report(cfg, batch, args.seed)
        r["bytes_ratio"] = r["bytes_ckpt"] / r["bytes_plain"]
        r["flops_ratio"] = r["flops_ckpt"] / r["flops_plain"]
        return r
    if command == "ddp":
        return run_spawned(scale.ddp_worker, args, cfg)
    if command == "fsdp":
        r = run_spawned(scale.fsdp_worker, args, cfg)
        r["max_shard_fraction"] = max(r["shard_bytes_per_rank"]) / r["full_bytes"]
        return r
    if command == "budget":
        return scale.gpt2_budget()
    raise ValueError(command)


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = build_model(args)
    model.train()
    batch = scale.make_batch(model.config, args.batch_size, args.seq_len, args.seed)
    commands = COMMANDS if args.command == "all" else (args.command,)
    result = {"config": {**vars(args), "model": dict(vars(model.config))}}
    for command in commands:
        t0 = time.perf_counter()
        result[command] = run(command, args, model, batch)
        result[command]["seconds"] = round(time.perf_counter() - t0, 3)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


def main_with_json_stdout(argv=None):
    """Run ``main`` with file descriptor 1 pointed at stderr; return its result.

    gloo's TCP transport (the Linux default; macOS builds use libuv) prints
    "[Gloo] Rank 0 is connected to 1 peer ranks. ..." with C++ ``std::cout``
    (third_party/gloo at torch v2.8.0, gloo/transport/tcp/context.cc), and
    spawned ranks inherit the parent's fd 1, so ``contextlib.redirect_stdout``
    cannot catch it. Swapping the descriptor sends that chatter to stderr and
    keeps stdout for the one JSON document.
    """
    sys.stdout.flush()
    json_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        return main(argv)
    finally:
        sys.stdout.flush()
        os.dup2(json_fd, 1)
        os.close(json_fd)


if __name__ == "__main__":
    print(json.dumps(main_with_json_stdout(), indent=2))
