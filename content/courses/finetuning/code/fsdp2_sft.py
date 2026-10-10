"""Distributed SFT lab: DDP correctness on CPU, FSDP2 / ZeRO-3 / QLoRA / MoE-LoRA on GPUs.

Module: 10 -- Scaling Out: Distributed Fine-Tuning, Parallelism, and MoE
(content/courses/finetuning/10-distributed-fine-tuning-and-moe.md), Lab 10.8.

Purpose
    --smoke  Spawns 2 CPU processes with the gloo backend, wraps a toy model in
             DistributedDataParallel, runs one backward pass on a different half
             of a batch on each rank, and asserts that (a) the all-reduced
             gradients are identical on both ranks and (b) they equal the
             single-process full-batch gradient. No FSDP on CPU: FSDP2's
             value is GPU memory, which CPU cannot demonstrate.
    --write-configs DIR
             Writes the Accelerate FSDP2 config, the Accelerate DeepSpeed config
             and the ZeRO-3 JSON used by the GPU runs, so you can read and diff them.
    --full   The real lab. Run it under `accelerate launch` with one of the
             written configs. Variants (--variant):
               full      full fine-tune (L10.1; FSDP2 or ZeRO-3 depending on config)
               lora      bf16 base + all-linear LoRA (dense baseline for L10.4)
               qlora     NF4 base + LoRA, quant storage from --quant-storage
                         (default bfloat16; uint8 reproduces the FSDP failure) (L10.2)
               moe-lora  LoRA on attention + experts via target_parameters (L10.4)
             It logs tokens/s, peak memory per rank and the checkpoint layout.

Hardware tier
    smoke: CPU (any laptop, < 60 s).
    full:  T3 (2-8x A100/H100 80 GB) for an 8B full fine-tune; 2x24 GB or more
           for QLoRA on 32B; T2/T3 for MoE LoRA on Qwen3-30B-A3B.

Pinned versions for --full (as of October 2026)
    torch>=2.8, transformers>=5.19, trl==1.15.0, peft>=0.21.2, accelerate>=1.15,
    bitsandbytes>=0.43.3 (qlora), deepspeed==0.19.7 (ZeRO-3 run only).

Run (from the repository root)
    python content/courses/finetuning/code/fsdp2_sft.py --smoke
    python content/courses/finetuning/code/fsdp2_sft.py --write-configs /tmp/ft10
    accelerate launch --config_file /tmp/ft10/fsdp2.yaml \
        content/courses/finetuning/code/fsdp2_sft.py --full --variant full --model Qwen/Qwen3-8B
    accelerate launch --config_file /tmp/ft10/zero3.yaml \
        content/courses/finetuning/code/fsdp2_sft.py --full --variant full --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

WORLD = 2
BATCH = 8  # global batch; each rank gets BATCH // WORLD rows


def _toy_model() -> nn.Module:
    torch.manual_seed(0)  # identical init on every rank, as DDP requires
    return nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 3))


def _toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, 6, generator=g)
    y = torch.randint(0, 3, (BATCH,), generator=g)
    return x, y


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        model = nn.parallel.DistributedDataParallel(_toy_model())
        x, y = _toy_batch()
        shard = slice(rank * BATCH // WORLD, (rank + 1) * BATCH // WORLD)
        loss = F.cross_entropy(model(x[shard]), y[shard])  # per-rank MEAN loss
        loss.backward()  # DDP all-reduces and AVERAGES gradients across ranks here
        grads = {n: p.grad.detach().clone() for n, p in model.module.named_parameters()}
        torch.save(grads, os.path.join(out_dir, f"rank{rank}.pt"))
        dist.barrier()  # every rank reaches this line: no rank-divergent collectives
    finally:
        dist.destroy_process_group()


def smoke() -> None:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out_dir:
        mp.spawn(_worker, args=(_free_port(), out_dir), nprocs=WORLD, join=True)
        per_rank = [torch.load(os.path.join(out_dir, f"rank{r}.pt")) for r in range(WORLD)]
    ref_model = _toy_model()
    x, y = _toy_batch()
    F.cross_entropy(ref_model(x), y).backward()  # full-batch mean loss, one process
    for name, p in ref_model.named_parameters():
        for r in range(1, WORLD):
            assert torch.equal(per_rank[0][name], per_rank[r][name]), f"ranks disagree on {name}"
        torch.testing.assert_close(per_rank[0][name], p.grad, atol=1e-6, rtol=1e-5)
    print(f"ranks={WORLD} params={len(per_rank[0])} elapsed={time.time() - t0:.1f}s")
    print("smoke OK: DDP gradients identical across ranks and equal to the full-batch gradient")


# ----------------------------------------------------------------------------
# GPU configs (written by --write-configs, consumed by `accelerate launch`).
# Keys verified against the Accelerate v1.15.0 FSDP1-vs-FSDP2 guide and the
# DeepSpeed config-json reference, October 2026.
# ----------------------------------------------------------------------------
FSDP2_YAML = """\
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: {gpus}
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: true        # FULL_SHARD (ZeRO-3-like); false = SHARD_GRAD_OP
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true    # rank 0 loads weights, others get empty tensors
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_offload_params: false
  fsdp_activation_checkpointing: true
"""

ZERO3_ACCELERATE_YAML = """\
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: {gpus}
deepspeed_config:
  deepspeed_config_file: {ds_json}
"""

ZERO3_JSON = {
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
    },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
}


def write_configs(out_dir: str, gpus: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds_json = os.path.join(out_dir, "ds_zero3.json")
    with open(ds_json, "w") as f:
        json.dump(ZERO3_JSON, f, indent=2)
    with open(os.path.join(out_dir, "fsdp2.yaml"), "w") as f:
        f.write(FSDP2_YAML.format(gpus=gpus))
    with open(os.path.join(out_dir, "zero3.yaml"), "w") as f:
        f.write(ZERO3_ACCELERATE_YAML.format(gpus=gpus, ds_json=ds_json))
    print(f"wrote fsdp2.yaml, zero3.yaml, ds_zero3.json to {out_dir}")


def full(args: argparse.Namespace) -> None:  # pragma: no cover - needs GPUs
    # Heavy imports live here so --smoke never needs them.
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model_kwargs: dict = {"dtype": torch.bfloat16}
    peft_config = None
    if args.variant == "lora":
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                 task_type="CAUSAL_LM")
    elif args.variant == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Must equal the model dtype FSDP shards; uint8 shows the failure mode.
            bnb_4bit_quant_storage=getattr(torch, args.quant_storage),
        )
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                 task_type="CAUSAL_LM")
    elif args.variant == "moe-lora":
        cfg = AutoConfig.from_pretrained(args.model)
        n_exp = getattr(cfg, "num_local_experts", None) or cfg.num_experts
        peft_config = LoraConfig(
            r=16, lora_alpha=32, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            rank_pattern={"experts.gate_up_proj": max(1, 16 // n_exp),
                          "experts.down_proj": max(1, 16 // n_exp)},
        )

    class Probe(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if torch.cuda.is_available() and state.global_step % 10 == 0:
                rank = int(os.environ.get("RANK", 0))
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[rank {rank}] step {state.global_step} peak_alloc {peak:.1f} GB", flush=True)

    # Under FSDP, activation checkpointing comes from fsdp_activation_checkpointing in
    # the YAML; enabling Trainer gradient_checkpointing as well is rejected.
    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    ds = load_dataset(args.dataset, split="train").select(range(args.max_examples))
    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not use_fsdp,
        learning_rate=1e-5 if args.variant == "full" else 1e-4,
        bf16=True,
        max_length=args.max_length,
        packing=True,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=args.max_steps,
        report_to="none",
        model_init_kwargs=model_kwargs,
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=ds,
                         peft_config=peft_config, callbacks=[Probe()])
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    trainer.save_model(os.path.join(args.output_dir, "final"))
    if trainer.accelerator.is_main_process:
        steps = args.max_steps
        world = trainer.accelerator.num_processes
        toks = steps * args.batch * args.grad_accum * args.max_length * world
        print(f"variant={args.variant} world={world} wall={elapsed:.0f}s "
              f"~tokens/s(upper bound, packed)={toks / elapsed:.0f}")
        for root, _, files in os.walk(args.output_dir):
            for fn in sorted(files)[:6]:
                print("ckpt:", os.path.join(root, fn))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="2-process gloo DDP check on CPU (default)")
    mode.add_argument("--full", action="store_true", help="GPU run; launch with accelerate launch")
    mode.add_argument("--write-configs", metavar="DIR", help="write FSDP2 / ZeRO-3 configs and exit")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--variant", choices=["full", "lora", "qlora", "moe-lora"], default="full")
    p.add_argument("--quant-storage", choices=["bfloat16", "float16", "uint8"], default="bfloat16",
                   help="bnb_4bit_quant_storage for --variant qlora")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset", default="trl-lib/Capybara")
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output-dir", default="runs/ft10")
    args = p.parse_args()
    if args.write_configs:
        write_configs(args.write_configs, args.gpus)
    elif args.full:
        full(args)
    else:
        if torch.cuda.is_available() and not args.smoke:
            print("GPU found; running the CPU smoke check anyway. Use --full under accelerate launch.",
                  file=sys.stderr)
        smoke()


if __name__ == "__main__":
    main()
