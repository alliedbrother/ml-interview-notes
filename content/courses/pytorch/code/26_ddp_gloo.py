"""Prove that DDP equals large-batch training, with real CPU processes and gloo.

Spawns ``--world-size`` processes (``torch.multiprocessing.spawn``), joins them in
a gloo process group through a ``file://`` rendezvous in a temporary directory,
and on every rank:

1. runs a collectives tour (all_reduce SUM/AVG/MAX, all_gather_into_tensor,
   reduce_scatter_tensor, broadcast, all_to_all_single, isend/irecv ring) and
   checks each result against its closed form;
2. trains a seeded MLP for ``--steps`` steps three ways on DistributedSampler
   shards: naive DDP (broadcast + manual all_reduce), ``DistributedDataParallel``,
   and DDP with ``no_sync`` accumulation over 2 micro-batches;
3. times an all_reduce of a 4 MiB float32 tensor.

Rank 0 also trains a single-process reference on the concatenated global batch,
compares parameters, and writes the JSON result. A negative control trains rank 0's shard alone, with
no communication, to show the comparison can fail.

Run from the repo root:
    python content/courses/pytorch/code/26_ddp_gloo.py --world-size 2
or under torchrun (rank and world size come from the environment):
    torchrun --standalone --nproc_per_node=2 content/courses/pytorch/code/26_ddp_gloo.py --torchrun

Tested with Python 3.11 and PyTorch 2.8.0 on CPU.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

PER_RANK_BATCH = 8
IN_DIM, HIDDEN = 8, 16
LR = 0.1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--torchrun", action="store_true",
                   help="read RANK/WORLD_SIZE from the environment instead of spawning")
    return p.parse_args(argv)


# ---------------------------------------------------------------- collectives
def collectives_tour(rank, world):
    """Each check compares a collective's output with its closed form."""
    ok = {}
    # all_reduce SUM: rank r contributes r+1, so every rank ends with p(p+1)/2.
    x = torch.full((4,), float(rank + 1))
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    ok["all_reduce_sum"] = bool(torch.all(x == world * (world + 1) / 2))
    # all_reduce AVG and MAX.
    x = torch.full((4,), float(rank + 1))
    dist.all_reduce(x, op=dist.ReduceOp.AVG)
    ok["all_reduce_avg"] = bool(torch.allclose(x, torch.full((4,), (world + 1) / 2)))
    x = torch.full((4,), float(rank + 1))
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    ok["all_reduce_max"] = bool(torch.all(x == world))
    # all_gather_into_tensor: output is the rank-ordered concatenation.
    inp = torch.arange(2, dtype=torch.float32) + 10 * rank
    out = torch.empty(2 * world)
    dist.all_gather_into_tensor(out, inp)
    expect = torch.cat([torch.arange(2, dtype=torch.float32) + 10 * r for r in range(world)])
    ok["all_gather_into_tensor"] = bool(torch.equal(out, expect))
    # reduce_scatter_tensor: sum over ranks, then rank r keeps chunk r.
    inp = torch.arange(2 * world, dtype=torch.float32) * (rank + 1)
    out = torch.empty(2)
    dist.reduce_scatter_tensor(out, inp)
    total = torch.arange(2 * world, dtype=torch.float32) * (world * (world + 1) / 2)
    ok["reduce_scatter_tensor"] = bool(torch.equal(out, total[2 * rank: 2 * rank + 2]))
    # broadcast from rank 0 overwrites every other rank's buffer.
    x = torch.full((3,), float(rank) + 7.0)
    dist.broadcast(x, src=0)
    ok["broadcast"] = bool(torch.all(x == 7.0))
    # all_to_all_single: rank r sends element j to rank j (a transpose of a p x p matrix).
    inp = torch.tensor([float(rank * world + j) for j in range(world)])
    out = torch.empty(world)
    dist.all_to_all_single(out, inp)
    ok["all_to_all_single"] = bool(torch.equal(out, torch.tensor([float(j * world + rank) for j in range(world)])))
    # point-to-point ring: send to the right neighbour, receive from the left.
    send = torch.tensor([float(rank)])
    recv = torch.empty(1)
    reqs = [dist.isend(send, (rank + 1) % world), dist.irecv(recv, (rank - 1) % world)]
    for req in reqs:
        req.wait()
    ok["isend_irecv_ring"] = bool(recv.item() == (rank - 1) % world)
    # AND every flag across ranks: MIN over 0/1 is a logical AND.
    flags = torch.tensor([float(v) for v in ok.values()])
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return {k: bool(f) for k, f in zip(ok, flags.tolist())}


# ---------------------------------------------------------------- training
def make_data(seed, n):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g, dtype=torch.float64)
    w = torch.randn(IN_DIM, 1, generator=g, dtype=torch.float64)
    y = torch.sin(x @ w) + 0.1 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    return TensorDataset(x, y)


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1)).double()


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def rank_batches(dataset, world, rank, seed):
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=seed, drop_last=True)
    sampler.set_epoch(0)
    return sampler, list(DataLoader(dataset, batch_size=PER_RANK_BATCH, sampler=sampler))


def train_naive(rank, world, batches, seed):
    # Deliberately different init per rank: the broadcast below must fix it.
    model = make_model(seed + rank)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)
        opt.step()
    return model


def train_ddp(rank, world, batches, seed, accumulate):
    module = make_model(seed + rank)  # DDP's constructor broadcasts rank 0's state
    model = DDP(module)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        if accumulate:
            micro = list(zip(x.chunk(2), y.chunk(2)))
            with model.no_sync():  # grads accumulate locally, no all_reduce
                (F.mse_loss(model(micro[0][0]), micro[0][1]) / 2).backward()
            (F.mse_loss(model(micro[1][0]), micro[1][1]) / 2).backward()  # syncs
        else:
            F.mse_loss(model(x), y).backward()
        opt.step()
    sizes = model._get_ddp_logging_data().get("bucket_sizes", "")
    buckets = len([s for s in str(sizes).split(",") if s.strip()])
    return module, buckets


def train_local(batches, seed):
    """Negative control: rank 0's init, own shard only, no communication."""
    model = make_model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        opt.step()
    return model


def train_reference(dataset, world, steps, seed):
    per_rank = [rank_batches(dataset, world, r, seed)[1] for r in range(world)]
    model = make_model(seed)  # rank 0's init
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for step in range(steps):
        x = torch.cat([per_rank[r][step][0] for r in range(world)])
        y = torch.cat([per_rank[r][step][1] for r in range(world)])
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        opt.step()
    return model


def identical_across_ranks(vec, world):
    out = torch.empty(world * vec.numel(), dtype=vec.dtype)
    dist.all_gather_into_tensor(out, vec.contiguous())
    return all(torch.equal(c, vec) for c in out.chunk(world))


def time_allreduce(nbytes=4 * 1024 * 1024, iters=5):
    t = torch.ones(nbytes // 4, dtype=torch.float32)
    for _ in range(2):
        dist.all_reduce(t)
    times = []
    for _ in range(iters):
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run(rank, world, args):
    torch.set_num_threads(1)  # world processes share the CPU
    steps, seed = args.steps, args.seed
    dataset = make_data(seed, steps * PER_RANK_BATCH * world)
    _, batches = rank_batches(dataset, world, rank, seed)
    assert len(batches) == steps

    collectives_ok = collectives_tour(rank, world)
    naive = train_naive(rank, world, batches, seed)
    ddp, bucket_count = train_ddp(rank, world, batches, seed, accumulate=False)
    nosync, _ = train_ddp(rank, world, batches, seed, accumulate=True)
    same = all(identical_across_ranks(flat(m), world) for m in (naive, ddp, nosync))
    ms = time_allreduce()

    if rank != 0:
        return {}
    ref = flat(train_reference(dataset, world, steps, seed))
    local = flat(train_local(batches, seed))
    return {
        "world_size": world,
        "steps": steps,
        "collectives_ok": collectives_ok,
        "max_diff_naive_vs_ref": float((flat(naive) - ref).abs().max()),
        "max_diff_ddp_vs_ref": float((flat(ddp) - ref).abs().max()),
        "max_diff_nosync_vs_ref": float((flat(nosync) - ref).abs().max()),
        "max_diff_local_only_vs_ref": float((local - ref).abs().max()),
        "params_identical_across_ranks": bool(same),
        "allreduce_4mb_ms": round(ms, 3),
        "bucket_count": bucket_count,
    }


def _spawned(rank, world, args, init_file, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=60))
    try:
        result = run(rank, world, args)
        if rank == 0:
            Path(result_file).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.torchrun:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group("gloo", init_method="env://",
                                timeout=datetime.timedelta(seconds=60))
        try:
            result = run(rank, world, args)
        finally:
            dist.destroy_process_group()
    else:
        world = args.world_size
        with tempfile.TemporaryDirectory() as tmp:
            init_file, result_file = Path(tmp) / "rendezvous", Path(tmp) / "result.json"
            mp.spawn(_spawned, args=(world, args, str(init_file), str(result_file)),
                     nprocs=world, join=True)
            result = json.loads(result_file.read_text())
    if result and args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    out = main()
    if out:
        print(json.dumps(out, indent=2))
