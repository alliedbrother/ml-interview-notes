"""Scale the capstone recipe without editing the model (module 37).

Every tool here wraps or drives ``llm.model.GPT`` from the outside:

* ``profile_step`` / ``flop_report``: where one training step's time and FLOPs go,
  and how the counted FLOPs compare with 6 * P_matmul * tokens + 12 * L * T * d * tokens.
* ``compile_parity`` / ``cached_recompiles``: ``torch.compile`` (aot_eager on CPU)
  gives the eager loss and gradients; the KV-cache path's Python-int length
  recompiles per value.
* ``CheckpointedBlock`` / ``enable_activation_checkpointing`` /
  ``saved_activation_bytes``: per-block activation checkpointing
  (``use_reentrant=False``) and the bytes autograd saves with and without it.
* ``ddp_worker`` / ``fsdp_worker`` / ``reference_step``: one recipe step on gloo
  ranks (DDP, FSDP2) checked against a single process on the full batch.
* ``gpt2_budget``: GPT-2 124M memory/FLOP/time arithmetic (computed, never run).

Imports llm.model, llm.data and llm.train only (syllabus E.11). CPU only.
Read it next to content/courses/pytorch/37-capstone-scaling-the-recipe.md.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from torch.utils.flop_counter import FlopCounterMode

from llm.data import encode_all_splits, get_batch
from llm.model import GPT, GPTConfig, KVCache
from llm.train import TrainConfig, lr_at

STEP_LR_INDEX = 0  # the recipe's first optimizer step: lr_at(0) = 3e-3 / 40 = 7.5e-5


# ------------------------------------------------------------ batches and configs

def make_batch(cfg: GPTConfig, batch_size: int, seq_len: int, seed: int):
    """A fixed (x, y) batch of real training text: (B, T) int64 each."""
    if seq_len > cfg.block_size:
        raise ValueError(f"seq_len={seq_len} exceeds block_size={cfg.block_size}")
    tokens = encode_all_splits()["train"]
    return get_batch(tokens, seq_len, batch_size, torch.Generator().manual_seed(seed))


def make_cfg_dict(cfg: GPTConfig, batch_size: int, seq_len: int, seed: int) -> dict:
    """Everything a spawned rank needs, JSON/pickle-friendly (the batch travels as lists)."""
    x, y = make_batch(cfg, batch_size, seq_len, seed)
    return {"model": asdict(cfg), "seed": seed, "x": x.tolist(), "y": y.tolist()}


def _model_and_batch(cfg_dict: dict, seed: int):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**cfg_dict["model"]))
    x = torch.tensor(cfg_dict["x"], dtype=torch.long)
    y = torch.tensor(cfg_dict["y"], dtype=torch.long)
    return model, x, y


def _recipe_optimizer(model) -> torch.optim.AdamW:
    """The module 33 optimizer (two decay groups) at the learning rate of recipe step 0."""
    tc = TrainConfig()
    return model.configure_optimizer(tc.weight_decay, lr_at(STEP_LR_INDEX, tc),
                                     (tc.beta1, tc.beta2), tc.eps)


def _full(t: torch.Tensor) -> torch.Tensor:
    return (t.full_tensor() if hasattr(t, "full_tensor") else t).detach().float().clone()


def _max_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    return max((a[k] - b[k]).abs().max().item() for k in a)


# --------------------------------------------------------------- 37.1 profiling

def _train_step(model, opt, x, y):
    with torch.profiler.record_function("forward"):
        _, loss = model(x, y)
    with torch.profiler.record_function("backward"):
        loss.backward()
    with torch.profiler.record_function("optimizer"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss.detach()


def profile_step(model, batch, steps: int = 3, warmup: int = 2, row_limit: int = 10) -> dict:
    """Profile ``steps`` recipe steps after ``warmup`` steps (CPU activity only).

    Returns the top ops by self CPU time, the share of each labelled phase, the
    printed table, and the median unprofiled step time over ``steps`` more steps.
    Mutates ``model`` (it trains); pass a throwaway copy.
    """
    from torch.profiler import ProfilerActivity, profile, schedule

    x, y = batch
    model.train()
    opt = _recipe_optimizer(model)
    sched = schedule(wait=0, warmup=warmup, active=steps, repeat=1)
    with profile(activities=[ProfilerActivity.CPU], schedule=sched) as prof:
        for _ in range(warmup + steps):
            _train_step(model, opt, x, y)
            prof.step()
    events = prof.key_averages()
    total_self = sum(e.self_cpu_time_total for e in events) or 1.0
    ranked = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)
    top_ops = [{"name": e.key, "calls": e.count,
                "self_cpu_ms_per_step": round(e.self_cpu_time_total / 1e3 / steps, 3),
                "self_pct": round(100.0 * e.self_cpu_time_total / total_self, 1)}
               for e in ranked[:row_limit]]
    phases = {e.key: round(e.cpu_time_total / 1e3 / steps, 3) for e in events
              if e.key in ("forward", "backward", "optimizer")}
    table = events.table(sort_by="self_cpu_time_total", row_limit=row_limit)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        _train_step(model, opt, x, y)
        times.append(time.perf_counter() - t0)
    return {"top_ops": top_ops, "phase_ms_per_step": phases, "table": table,
            "step_ms_unprofiled": round(1e3 * statistics.median(times), 3)}


# -------------------------------------------------------------- 37.2 FLOPs

def matmul_params(cfg: GPTConfig) -> int:
    """Weights that take part in a matmul per token: four Linear weights per block + the tied head.

    Embedding lookups (wte as input, wpe), biases and LayerNorm do no matmul work.
    """
    d, L = cfg.d_model, cfg.n_layer
    per_block = d * 3 * d + d * d + d * 4 * d + 4 * d * d   # c_attn, attn.c_proj, c_fc, mlp.c_proj
    return L * per_block + cfg.vocab_size * d               # + lm_head (= wte.T)


def formula_flops(cfg: GPTConfig, batch_size: int, seq_len: int) -> dict:
    """6 * P_matmul * tokens + 12 * L * T * d * tokens (forward + backward, full T x T scores)."""
    tokens = batch_size * seq_len
    dense = 6 * matmul_params(cfg) * tokens
    attn = 12 * cfg.n_layer * seq_len * cfg.d_model * tokens
    head = 6 * cfg.vocab_size * cfg.d_model * tokens
    return {"tokens": tokens, "dense": dense, "attention": attn, "head_part_of_dense": head,
            "total": dense + attn}


def count_step_flops(model, x, y, attn_impl: str = "manual") -> int:
    """FLOPs FlopCounterMode sees in one forward + backward (no optimizer)."""
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as counter:
        _, loss = model(x, y, attn_impl=attn_impl)
        loss.backward()
    model.zero_grad(set_to_none=True)
    return int(counter.get_total_flops())


def flop_report(model, batch) -> dict:
    """Measured (FlopCounterMode, manual attention) vs formula FLOPs for one step.

    ``attn_impl="manual"``: in 2.8 FlopCounterMode has no formula for the CPU
    flash SDPA op, so the SDPA path would silently undercount attention.
    """
    x, y = batch
    cfg = model.config
    measured = count_step_flops(model, x, y, "manual")
    sdpa = count_step_flops(model, x, y, "sdpa")
    f = formula_flops(cfg, x.size(0), x.size(1))
    return {"measured": measured, "formula": f["total"], "ratio": measured / f["total"],
            "measured_with_sdpa": sdpa, "dense": f["dense"], "attention": f["attention"],
            "head": f["head_part_of_dense"], "tokens": f["tokens"],
            "p_matmul": matmul_params(cfg), "p_total": model.num_params()}


def matmul_ceiling_gflops(n: int = 1024, reps: int = 5) -> float:
    """fp32 GFLOP/s of the best of ``reps`` (n x n) @ (n x n) matmuls on this CPU."""
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2 * n**3 / best / 1e9


# -------------------------------------------------------------- 37.3 compile

def _loss_and_grads(fn, model, x, y):
    model.zero_grad(set_to_none=True)
    _, loss = fn(x, y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return loss.detach(), grads


def compile_parity(model, batch, backend: str = "aot_eager") -> dict:
    """Eager vs ``torch.compile(model, backend=backend)`` loss and gradients, plus a
    graph-break audit of the uncached training forward. Resets Dynamo before and after."""
    x, y = batch
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(model)(x, y)
        torch._dynamo.reset()
        eager_loss, eager_grads = _loss_and_grads(model, model, x, y)
        compiled = torch.compile(model, backend=backend)
        comp_loss, comp_grads = _loss_and_grads(compiled, model, x, y)
        return {"backend": backend,
                "max_loss_diff": (eager_loss - comp_loss).abs().item(),
                "max_grad_diff": _max_diff(eager_grads, comp_grads),
                "graph_breaks": explanation.graph_break_count,
                "graphs": explanation.graph_count}
    finally:
        torch._dynamo.reset()


def cached_recompiles(model, steps: int = 5, automatic_dynamic: bool = True) -> dict:
    """Decode ``steps`` single tokens through a compiled cached forward and count compiles.

    ``cache.length`` is a Python int read inside forward, so Dynamo guards on
    its value (``cache.length == 0``). With automatic dynamic shapes (the
    default) the first guard failure recompiles once with the length as a
    SymInt; with it off, every new length compiles again. Uses a counting
    backend (no Inductor) and stays under the default recompile limit of 8.
    """
    frames = []

    def counting_backend(gm, example_inputs):
        frames.append(sum(isinstance(e, torch.SymInt) for e in example_inputs))
        return gm.forward

    old = torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.reset()
    try:
        torch._dynamo.config.automatic_dynamic_shapes = automatic_dynamic
        cfg = model.config
        cache = KVCache(cfg, batch_size=1, max_len=min(cfg.block_size, steps + 1))
        fn = torch.compile(model, backend=counting_backend)
        idx = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            for _ in range(steps):
                logits, _ = fn(idx, cache=cache)
                idx = logits[:, -1:].argmax(-1)
        return {"decode_steps": steps, "automatic_dynamic": automatic_dynamic,
                "compiles": len(frames), "symint_inputs_per_compile": frames,
                "final_length": cache.length}
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = old
        torch._dynamo.reset()


# ------------------------------------------------- 37.4 activation checkpointing

class CheckpointedBlock(nn.Module):
    """Run a ``Block`` under ``checkpoint(use_reentrant=False)`` while training.

    Only the block's inputs are kept for backward; its interior is recomputed.
    The cached path (inference) and no-grad calls run the block directly.
    State-dict keys stay ``blocks.i.<name>`` (the ``block.`` prefix is hidden),
    so a plain checkpoint loads into a wrapped model and vice versa.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.register_state_dict_post_hook(_strip_block_prefix)
        self.register_load_state_dict_pre_hook(_add_block_prefix)

    def forward(self, x, mask=None, cache=None, attn_impl: str = "sdpa"):
        if cache is not None or not torch.is_grad_enabled():
            return self.block(x, mask=mask, cache=cache, attn_impl=attn_impl)
        return checkpoint(self.block, x, mask=mask, cache=None, attn_impl=attn_impl,
                          use_reentrant=False)


def _strip_block_prefix(module, state_dict, prefix, local_metadata):
    for key in [k for k in state_dict if k.startswith(prefix + "block.")]:
        state_dict[prefix + key[len(prefix) + len("block."):]] = state_dict.pop(key)


def _add_block_prefix(module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
    for key in [k for k in state_dict if k.startswith(prefix) and not k.startswith(prefix + "block.")]:
        state_dict[prefix + "block." + key[len(prefix):]] = state_dict.pop(key)


def enable_activation_checkpointing(model):
    """Wrap every ``model.blocks[i]`` in a ``CheckpointedBlock`` in place; return ``model``."""
    for i, block in enumerate(model.blocks):
        if not isinstance(block, CheckpointedBlock):
            model.blocks[i] = CheckpointedBlock(block)
    return model


def saved_activation_bytes(model, x, y, attn_impl: str = "sdpa") -> int:
    """Bytes of the unique non-parameter storages autograd saves in one forward.

    The pack hook sees what the graph keeps outside checkpointed regions,
    including the block inputs non-reentrant checkpoint saves. Parameters are
    excluded: they live in memory with or without checkpointing.
    """
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen: dict[int, int] = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in param_ptrs:
            seen[s.data_ptr()] = s.nbytes()
        return t

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(x, y, attn_impl=attn_impl)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return sum(seen.values())


def _flops_no_early_stop(model, x, y) -> int:
    # by default non-reentrant recompute stops once the last tensor backward needs exists
    with set_checkpoint_early_stop(False):
        return count_step_flops(model, x, y)


def checkpoint_report(cfg: GPTConfig, batch, seed: int) -> dict:
    """Plain vs checkpointed copies of one seeded model: saved bytes, gradient
    agreement, counted FLOPs (manual attention) with and without recompute early stop."""
    x, y = batch
    torch.manual_seed(seed)
    plain = GPT(cfg)
    ckpt = GPT(cfg)
    ckpt.load_state_dict(plain.state_dict())
    enable_activation_checkpointing(ckpt)
    _, g_plain = _loss_and_grads(plain, plain, x, y)
    _, g_ckpt = _loss_and_grads(ckpt, ckpt, x, y)
    g_ckpt = {n.replace(".block.", "."): g for n, g in g_ckpt.items()}
    return {"bytes_plain": saved_activation_bytes(plain, x, y),
            "bytes_ckpt": saved_activation_bytes(ckpt, x, y),
            "max_grad_diff": _max_diff(g_plain, g_ckpt),
            "flops_plain": count_step_flops(plain, x, y),
            "flops_ckpt": count_step_flops(ckpt, x, y),
            "flops_ckpt_no_early_stop": _flops_no_early_stop(ckpt, x, y),
            "state_dict_keys_equal": list(plain.state_dict()) == list(ckpt.state_dict())}


# ------------------------------------------------------- 37.5-37.6 reference step

def reference_step(cfg_dict: dict, seed: int) -> dict:
    """One recipe step in one process on the full batch.

    Returns tensors (not JSON): ``loss``, ``grads`` (pre-clip, by parameter name),
    ``params`` after clip(1.0) + AdamW at the learning rate of recipe step 0.
    """
    model, x, y = _model_and_batch(cfg_dict, seed)
    opt = _recipe_optimizer(model)
    _, loss = model(x, y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    return {"loss": loss.item(), "grads": grads, "params": params}


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world_size)


def _rank_slice(n: int, rank: int, world_size: int) -> slice:
    if n % world_size:
        raise ValueError(f"global batch {n} is not divisible by world size {world_size}")
    per = n // world_size
    return slice(rank * per, (rank + 1) * per)


def _all_ranks_equal(t: torch.Tensor) -> bool:
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return all(torch.equal(gathered[0], g) for g in gathered)


def _write_rank_result(out_path: str, rank: int, result: dict) -> None:
    dist.barrier()
    if rank == 0:
        Path(out_path).write_text(json.dumps(result))


# --------------------------------------------------------------------- 37.5 DDP

def ddp_worker(rank: int, world_size: int, init_file: str, cfg_dict: dict, out_path: str) -> None:
    """One DDP recipe step on this rank's slice of the global batch.

    Checks (rank 0 writes JSON to ``out_path``): averaged gradients vs the
    single-process full-batch gradients; parameters after one step identical
    across ranks and equal to the reference step; and two ``no_sync``
    micro-batches per rank reproduce the same gradients.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    _init_gloo(rank, world_size, init_file)
    try:
        seed = cfg_dict["seed"]
        ref = reference_step(cfg_dict, seed)
        model, x, y = _model_and_batch(cfg_dict, seed)  # same seed -> same init on every rank
        ddp = DDP(model)                                # broadcasts rank 0's parameters anyway
        opt = _recipe_optimizer(model)
        sl = _rank_slice(x.size(0), rank, world_size)
        _, loss = ddp(x[sl], y[sl])
        loss.backward()                                  # bucketed all-reduce, divided by world
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        ranks_equal = _all_ranks_equal(flat)
        params = {n: p.detach().clone() for n, p in model.named_parameters()}

        # gradient accumulation: two micro-batches per rank, sync only on the last
        model2, _, _ = _model_and_batch(cfg_dict, seed)
        ddp2 = DDP(model2)
        xs, ys = x[sl].chunk(2), y[sl].chunk(2)
        with ddp2.no_sync():
            (ddp2(xs[0], ys[0])[1] / 2).backward()      # local accumulation, no all-reduce
        (ddp2(xs[1], ys[1])[1] / 2).backward()          # all-reduce of the accumulated grads
        accum = {n: p.grad.detach().clone() for n, p in model2.named_parameters()}

        result = {"rank_batch": list(x[sl].shape), "global_batch": list(x.shape),
                  "max_grad_diff_vs_ref": _max_diff(grads, ref["grads"]),
                  "max_param_diff_vs_ref": _max_diff(params, ref["params"]),
                  "max_grad_diff_no_sync_accum": _max_diff(accum, ref["grads"]),
                  "ranks_equal": ranks_equal,
                  "tied_after_wrap": model.lm_head.weight is model.wte.weight}
        _write_rank_result(out_path, rank, result)
    finally:
        dist.destroy_process_group()


# -------------------------------------------------------------------- 37.6 FSDP2

def shard_gpt(model: GPT, mesh, force_sum: bool = True) -> GPT:
    """``fully_shard`` each block, then the root (wte/wpe/ln_f and the tied head).

    ``force_sum``: torch 2.8 gloo ``reduce_scatter_tensor`` ignores ReduceOp.AVG and
    sums, so FSDP2's default path gives world-size-times gradients on gloo.
    Forcing SUM makes FSDP divide by the world size itself (module 27).
    """
    from torch.distributed.fsdp import fully_shard

    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    if force_sum:
        for module in [*model.blocks, model]:
            module.set_force_sum_reduction_for_comms(True)
    return model


def _fsdp_step(cfg_dict, seed, mesh, rank, world_size, force_sum):
    model, x, y = _model_and_batch(cfg_dict, seed)
    shard_gpt(model, mesh, force_sum=force_sum)
    opt = _recipe_optimizer(model)                       # built after sharding: DTensor params
    sl = _rank_slice(x.size(0), rank, world_size)
    _, loss = model(x[sl], y[sl])
    loss.backward()                                      # reduce-scatter: each rank keeps its grad shard
    grads = {n: _full(p.grad) for n, p in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    params = {n: _full(p) for n, p in model.named_parameters()}
    return model, grads, params


def fsdp_worker(rank: int, world_size: int, init_file: str, cfg_dict: dict, out_path: str) -> None:
    """FSDP2 recipe step vs the single-process reference (rank 0 writes JSON).

    Reports the sharded state (DTensor placements, per-rank bytes, tying) and,
    for both the forced-SUM path and the default AVG path, gradient ratios and
    parameter differences against ``reference_step``.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    _init_gloo(rank, world_size, init_file)
    try:
        seed = cfg_dict["seed"]
        ref = reference_step(cfg_dict, seed)
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        plain_bytes = sum(p.numel() * p.element_size() for p in _model_and_batch(cfg_dict, seed)[0].parameters())

        model, grads, params = _fsdp_step(cfg_dict, seed, mesh, rank, world_size, force_sum=True)
        named = dict(model.named_parameters())
        placements = sorted({str(p.placements) for p in named.values()})
        local_bytes = sum(p.to_local().numel() * p.element_size() for p in named.values())
        full_bytes = sum(p.numel() * p.element_size() for p in named.values())  # DTensor.numel is global
        local_all = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(local_all, torch.tensor([local_bytes]))

        _, avg_grads, avg_params = _fsdp_step(cfg_dict, seed, mesh, rank, world_size, force_sum=False)
        head = "lm_head.weight" if "lm_head.weight" in ref["grads"] else "wte.weight"

        def ratio(g):
            return (g[head].norm() / ref["grads"][head].norm()).item()

        result = {
            "all_dtensor": all(isinstance(p, DTensor) for p in named.values()),
            "placements": placements,
            "param_names": len(named),
            "tied_after_shard": model.lm_head.weight is model.wte.weight,
            "full_bytes": full_bytes,
            "plain_model_bytes": plain_bytes,
            "shard_bytes_per_rank": [int(t.item()) for t in local_all],
            "max_grad_diff_vs_ref": _max_diff(grads, ref["grads"]),
            "max_param_diff_vs_ref": _max_diff(params, ref["params"]),
            "grad_ratio_force_sum": ratio(grads),
            "grad_ratio_default_avg": ratio(avg_grads),
            "max_param_diff_default_avg": _max_diff(avg_params, ref["params"]),
        }
        _write_rank_result(out_path, rank, result)
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------- 37.7 GPT-2 budget

GPT2_124M = {"vocab_size": 50304, "block_size": 1024, "n_layer": 12, "n_head": 12, "d_model": 768}


def gpt2_budget(mfu: float = 0.4, tokens: float = 10e9, n_gpus: int = 8,
                peak_tflops: float = 312.0) -> dict:
    """GPT-2 124M arithmetic: parameters, training-state bytes, activations, FLOPs, time.

    Conventions (stated, not measured): V padded to 50,304, tied head; mixed
    precision AdamW = 16 B/param (bf16 weights 2 + bf16 grads 2 + fp32 master 4
    + two fp32 moments 8, ZeRO's 2+2+12); activations per token per layer =
    34*d + 5*H*T bytes (Korthikanti et al., 16-bit, no recompute, attention
    matrix materialised); FLOPs per token = 6*P_matmul + 12*L*T*d.
    """
    c = GPT2_124M
    V, T, L, H, d = c["vocab_size"], c["block_size"], c["n_layer"], c["n_head"], c["d_model"]
    per_block = (2 * d) + (d * 3 * d + 3 * d) + (d * d + d) + (2 * d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    params = V * d + T * d + L * per_block + 2 * d
    p_matmul = L * 12 * d * d + V * d
    bytes_per_param = 16
    act_layer = 34 * d + 5 * H * T                       # bytes per token per layer
    act_flash = 34 * d                                   # attention matrix never stored (flash)
    act_recompute = 2 * d                                # bf16 block input kept per layer
    flops_per_token = 6 * p_matmul + 12 * L * T * d
    total_flops = flops_per_token * tokens
    cluster = n_gpus * peak_tflops * 1e12 * mfu
    seconds = total_flops / cluster
    gib = 2**30
    return {
        "config": dict(c), "params": params, "params_unpadded_vocab": params - (V - 50257) * d,
        "p_matmul": p_matmul,
        "bytes_per_param": bytes_per_param,
        "train_state_bytes": params * bytes_per_param,
        "train_state_gib": params * bytes_per_param / gib,
        "fsdp_bytes_per_param_per_gpu": bytes_per_param / n_gpus,
        "act_bytes_per_token_per_layer": act_layer,
        "act_bytes_per_token_per_layer_flash": act_flash,
        "act_bytes_per_token_per_layer_recompute": act_recompute,
        "act_gib_per_seq_no_recompute": act_layer * L * T / gib,
        "act_gib_per_seq_flash": act_flash * L * T / gib,
        "act_gib_per_seq_recompute": act_recompute * L * T / gib,
        "flops_per_token": flops_per_token,
        "flops_per_token_6p": 6 * params,
        "attention_flops_per_token": 12 * L * T * d,
        "tokens": tokens, "total_flops": total_flops,
        "n_gpus": n_gpus, "peak_tflops": peak_tflops, "mfu": mfu,
        "tokens_per_second": cluster / flops_per_token,
        "hours": seconds / 3600,
    }
