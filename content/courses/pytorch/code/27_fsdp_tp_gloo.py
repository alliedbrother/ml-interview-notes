"""Shard one small model three ways on CPU gloo ranks and check each against one process.

Spawns ``--world-size`` gloo processes (``file://`` rendezvous in a temporary
directory) and runs five experiments on seeded toy models:

1. FSDP2: ``fully_shard`` on each of four MLP-residual blocks (d=32), then on
   the root; AdamW for ``--steps`` steps, each rank on its slice of the batch.
   The gathered parameters (``full_tensor()``) are compared with a
   single-process model trained on the whole batch. A second run uses
   ``MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)``.
   Rank 0 also reports its local parameter numel as a fraction of the total.
   Every FSDP module calls ``set_force_sum_reduction_for_comms(True)``: in
   torch 2.8 gloo's ``reduce_scatter_tensor`` ignores ``ReduceOp.AVG`` and
   sums, so FSDP2's default AVG path yields gradients ``world`` times too
   large. ``gloo_avg_grad_ratio`` measures that default path.
2. Tensor parallelism: ``parallelize_module`` with ``ColwiseParallel`` on the
   up-projection and ``RowwiseParallel`` on the down-projection of an MLP;
   output and weight gradients are compared with the unsharded MLP.
3. DTensor redistribute tour on a 4x2 tensor: Shard(0)->Replicate,
   Partial->Replicate, Shard(0)->Shard(1).
4. Pipeline parallelism: one ``PipelineStage`` per rank, ``ScheduleGPipe`` with
   4 microbatches; last-stage losses are compared with per-microbatch reference
   losses.
5. Distributed checkpointing: ``get_state_dict`` + ``dcp.save`` of the FSDP2
   model and optimizer, ``dcp.load`` + ``set_state_dict`` into a freshly
   sharded model; then, after the process group is gone, the parent process
   loads the same checkpoint into a plain unsharded model (resharding from
   ``world`` shards to one) and also converts it with ``dcp_to_torch_save``.

Scope: correctness of sharding semantics on gloo/CPU. Nothing here measures
memory savings, overlap, NCCL, prefetching or HSDP topologies.

Run from the repository root:

    python content/courses/pytorch/code/27_fsdp_tp_gloo.py --world-size 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 32
N_BLOCKS = 4
BATCH = 8  # global batch; split across ranks for FSDP2
LR = 1e-2


class Block(nn.Module):
    """x + W2(GELU(W1(x))): an MLP residual block, (B, d) -> (B, d)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, 2 * d)
        self.w2 = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(F.gelu(self.w1(x)))


class Net(nn.Module):
    def __init__(self, d: int = D_MODEL, n_blocks: int = N_BLOCKS) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_blocks))
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class MLP(nn.Module):
    def __init__(self, d: int = D_MODEL) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, 4 * d)
        self.w2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


def make_net(seed: int) -> Net:
    torch.manual_seed(seed)
    return Net()


def make_data(seed: int, steps: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed + 1)
    return [
        (torch.randn(BATCH, D_MODEL, generator=g), torch.randn(BATCH, 1, generator=g))
        for _ in range(steps)
    ]


def train_reference(seed: int, steps: int) -> Net:
    model = make_net(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for x, y in make_data(seed, steps):
        F.mse_loss(model(x), y).backward()
        opt.step()
        opt.zero_grad()
    return model


def shard_net(model: Net, mesh, mp_policy=None, force_sum: bool = True) -> Net:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    kwargs = {"mesh": mesh, "mp_policy": mp_policy or MixedPrecisionPolicy()}
    for block in model.blocks:  # bottom-up: one communication group per block
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)  # root group: the head's parameters
    if force_sum:
        # FSDP2 reduce-scatters fp32/bf16 grads with ReduceOp.AVG. Gloo's
        # reduce_scatter_tensor in 2.8 builds AllreduceOptions but never passes
        # them, so AVG silently becomes SUM (grads x world). Forcing SUM makes
        # FSDP divide by the world size itself. The setter is per module.
        for module in [*model.blocks, model]:
            module.set_force_sum_reduction_for_comms(True)
    return model


def gloo_avg_grad_ratio(rank, world, mesh, seed) -> float:
    """Gradient of the default (AVG) path divided by the single-process gradient."""
    ref = make_net(seed)
    x, y = make_data(seed, 1)[0]
    F.mse_loss(ref(x), y).backward()
    model = shard_net(make_net(seed), mesh, force_sum=False)
    per_rank = BATCH // world
    sl = slice(rank * per_rank, (rank + 1) * per_rank)
    F.mse_loss(model(x[sl]), y[sl]).backward()
    g = model.head.weight.grad.full_tensor()
    return (g.norm() / ref.head.weight.grad.norm()).item()


def full_params(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        t = p.full_tensor() if hasattr(p, "full_tensor") else p
        out[name] = t.detach().float().clone()
    return out


def max_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    return max((a[k] - b[k]).abs().max().item() for k in a)


def run_fsdp(rank, world, mesh, seed, steps, mp_policy=None):
    model = shard_net(make_net(seed), mesh, mp_policy)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)  # after fully_shard: DTensor shards
    per_rank = BATCH // world
    for x, y in make_data(seed, steps):
        sl = slice(rank * per_rank, (rank + 1) * per_rank)
        out = model(x[sl]).float()  # bf16 policy returns bf16 activations
        F.mse_loss(out, y[sl]).backward()  # grads reduce-scattered (averaged)
        opt.step()
        opt.zero_grad()
    return model, opt


def experiment_fsdp(rank, world, mesh, seed, steps, ref):
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor, Shard

    model, opt = run_fsdp(rank, world, mesh, seed, steps)
    assert isinstance(model, FSDPModule) and isinstance(model.blocks[0], FSDPModule)
    assert all(isinstance(p, DTensor) for p in model.parameters())
    assert all(p.placements == (Shard(0),) for p in model.parameters())
    local = sum(p.to_local().numel() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())  # DTensor.numel() is global
    fsdp_diff = max_diff(full_params(model), ref)

    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model_bf16, _ = run_fsdp(rank, world, mesh, seed, steps, bf16)
    # sharded parameters stay fp32; only the all-gathered compute copy is bf16
    assert all(p.dtype == torch.float32 for p in model_bf16.parameters())
    bf16_diff = max_diff(full_params(model_bf16), ref)
    return model, opt, {
        "gloo_avg_grad_ratio": gloo_avg_grad_ratio(rank, world, mesh, seed),
        "fsdp_max_diff": fsdp_diff,
        "fsdp_bf16_max_diff": bf16_diff,
        "local_numel_ratio": local / total,
    }


def experiment_tp(rank, world, mesh, seed):
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    torch.manual_seed(seed + 2)
    ref = MLP()
    tp = MLP()
    tp.load_state_dict(ref.state_dict())
    x = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(seed + 3))

    parallelize_module(tp, mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
    # nn.Linear.weight is (out, in): colwise shards out-features = dim 0
    assert tp.w1.weight.to_local().shape == (4 * D_MODEL // world, D_MODEL)
    assert tp.w2.weight.to_local().shape == (D_MODEL, 4 * D_MODEL // world)

    out_ref = ref(x)
    out_tp = tp(x)  # plain tensor in, plain (replicated) tensor out
    out_ref.square().sum().backward()
    out_tp.square().sum().backward()
    grad_diff = max(
        (getattr(tp, m).weight.grad.full_tensor() - getattr(ref, m).weight.grad)
        .abs()
        .max()
        .item()
        for m in ("w1", "w2")
    )
    return {"tp_out_diff": (out_tp - out_ref).abs().max().item(), "tp_grad_diff": grad_diff}


def experiment_redistribute(rank, world, mesh):
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

    t = torch.arange(8.0).reshape(4, 2)
    sharded = distribute_tensor(t, mesh, [Shard(0)])
    ok = {}
    ok["shard0_local_rows"] = torch.equal(sharded.to_local(), t.chunk(world, 0)[rank])
    ok["shard0_to_replicate"] = torch.equal(sharded.redistribute(mesh, [Replicate()]).to_local(), t)
    # Partial("sum"): each rank holds an addend; the logical tensor is their sum
    addend = t * (rank + 1)
    partial = DTensor.from_local(addend, mesh, [Partial()])
    expected = t * sum(range(1, world + 1))
    ok["partial_to_replicate"] = torch.equal(partial.redistribute(mesh, [Replicate()]).to_local(), expected)
    ok["partial_to_shard0"] = torch.equal(
        partial.redistribute(mesh, [Shard(0)]).to_local(), expected.chunk(world, 0)[rank]
    )
    col = sharded.redistribute(mesh, [Shard(1)])
    ok["shard0_to_shard1"] = torch.equal(col.to_local(), t.chunk(world, 1)[rank])
    ok["full_tensor"] = torch.equal(col.full_tensor(), t)
    return {"redistribute_ok": ok}


def experiment_pp(rank, world, seed):
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    torch.manual_seed(seed + 4)
    layers = nn.Sequential(*[nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.Tanh()) for _ in range(2 * world)])
    g = torch.Generator().manual_seed(seed + 5)
    x = torch.randn(8, D_MODEL, generator=g)
    y = torch.randn(8, D_MODEL, generator=g)
    n_micro = 4
    ref_losses = [F.mse_loss(layers(xm), ym) for xm, ym in zip(x.chunk(n_micro), y.chunk(n_micro))]

    per_stage = len(layers) // world
    submod = layers[rank * per_stage : (rank + 1) * per_stage]
    stage = PipelineStage(submod, stage_index=rank, num_stages=world, device=torch.device("cpu"))
    schedule = ScheduleGPipe(stage, n_microbatches=n_micro, loss_fn=F.mse_loss)
    if rank == 0:
        schedule.step(x)
        return {}
    losses: list[torch.Tensor] = []
    if rank == world - 1:
        schedule.step(target=y, losses=losses)
        diff = max((a.detach() - b.detach()).abs().item() for a, b in zip(losses, ref_losses))
        return {"pp_loss_diff": diff, "pp_n_losses": len(losses)}
    schedule.step()
    return {}


def experiment_dcp(rank, world, mesh, seed, model, opt, ckpt_dir):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt_dir)

    fresh = shard_net(make_net(seed + 100), mesh)  # different init on purpose
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=LR)
    f_model_sd, f_optim_sd = get_state_dict(fresh, fresh_opt)
    state = {"model": f_model_sd, "optim": f_optim_sd}
    dcp.load(state, checkpoint_id=ckpt_dir)  # in place, each rank reads its shards
    set_state_dict(fresh, fresh_opt, model_state_dict=state["model"], optim_state_dict=state["optim"])

    diff = max_diff(full_params(fresh), full_params(model))
    for p_old, p_new in zip(model.parameters(), fresh.parameters()):
        for key in ("exp_avg", "exp_avg_sq"):
            a, b = opt.state[p_old][key], fresh_opt.state[p_new][key]
            diff = max(diff, (a.full_tensor() - b.full_tensor()).abs().max().item())
    full = full_params(model)
    if rank == 0:
        torch.save(full, Path(ckpt_dir).parent / "full_params.pt")
    return {"dcp_roundtrip_diff": diff}


def worker(rank: int, world: int, init_file: str, out_dir: str, seed: int, steps: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        result: dict = {}
        ref = full_params(train_reference(seed, steps))
        model, opt, r = experiment_fsdp(rank, world, mesh, seed, steps, ref)
        result.update(r)
        result.update(experiment_tp(rank, world, mesh, seed))
        result.update(experiment_redistribute(rank, world, mesh))
        result.update(experiment_pp(rank, world, seed))
        result.update(experiment_dcp(rank, world, mesh, seed, model, opt, str(Path(out_dir) / "ckpt")))
        dist.barrier()
        (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def reshard_without_process_group(out_dir: Path, seed: int) -> dict:
    """Load the world-sized checkpoint into one unsharded model, no process group."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    saved = torch.load(out_dir / "full_params.pt")
    plain = make_net(seed + 200)
    state = {"model": plain.state_dict()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no process group, loading in single-process mode"
        dcp.load(state, checkpoint_id=str(out_dir / "ckpt"))
    plain.load_state_dict(state["model"])
    diff = max_diff(full_params(plain), saved)

    dcp_to_torch_save(str(out_dir / "ckpt"), str(out_dir / "consolidated.pt"))
    flat = torch.load(out_dir / "consolidated.pt")
    diff = max(diff, max_diff({k: v.float() for k, v in flat["model"].items()}, saved))
    return {"dcp_reshard_diff": diff, "consolidated_top_keys": sorted(flat)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.world_size < 2 or BATCH % args.world_size or (2 * D_MODEL) % args.world_size:
        parser.error("--world-size must be >= 2 and divide 8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        mp.spawn(
            worker,
            args=(args.world_size, str(out_dir / "rendezvous"), tmp, args.seed, args.steps),
            nprocs=args.world_size,
            join=True,
        )
        ranks = [json.loads((out_dir / f"rank{r}.json").read_text()) for r in range(args.world_size)]
        reshard = reshard_without_process_group(out_dir, args.seed)

    def worst(key: str) -> float:
        return max(r[key] for r in ranks if key in r)

    last = ranks[-1]
    result = {
        "world_size": args.world_size,
        "steps": args.steps,
        "gloo_avg_grad_ratio": ranks[0]["gloo_avg_grad_ratio"],
        "fsdp_max_diff": worst("fsdp_max_diff"),
        "fsdp_bf16_max_diff": worst("fsdp_bf16_max_diff"),
        "local_numel_ratio": ranks[0]["local_numel_ratio"],
        "tp_out_diff": worst("tp_out_diff"),
        "tp_grad_diff": worst("tp_grad_diff"),
        "redistribute_ok": {k: all(r["redistribute_ok"][k] for r in ranks) for k in ranks[0]["redistribute_ok"]},
        "pp_loss_diff": last["pp_loss_diff"],
        "pp_n_losses": last["pp_n_losses"],
        "dcp_roundtrip_diff": worst("dcp_roundtrip_diff"),
        "dcp_reshard_diff": reshard["dcp_reshard_diff"],
        "consolidated_top_keys": reshard["consolidated_top_keys"],
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2))
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
