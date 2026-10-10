"""Per-GPU memory planner for distributed fine-tuning (DDP, ZeRO-1/2/3, FSDP, LoRA, QLoRA, MoE).

Module: 10 -- Scaling Out: Distributed Fine-Tuning, Parallelism, and MoE
(content/courses/finetuning/10-distributed-fine-tuning-and-moe.md).

Purpose
    Estimate the model-state memory each GPU must hold (weights, gradients,
    optimizer state) for a model of a given size under each sharding method,
    using the per-GPU formulas of Rajbhandari et al. (2020), "ZeRO: Memory
    Optimizations Toward Training Trillion Parameter Models"
    (https://arxiv.org/abs/1910.02054). It also handles frozen-base adapters
    (LoRA in bf16, QLoRA in NF4) and mixture-of-experts models, whose weight
    memory scales with *total* parameters while compute scales with *active*
    parameters.

    It deliberately ignores activations, temporary all-gather buffers,
    fragmentation and CUDA context (typically 1-3 GB per GPU). Treat its output
    as a floor, not a promise.

Hardware tier
    CPU only. There is no GPU mode: the planner is pure arithmetic. `--full`
    is accepted for symmetry with the other course labs and simply prints the
    full comparison table for the model you pass.

Pinned versions
    Python 3.11+, standard library only (torch/numpy not required).

Run (from the repository root)
    python content/courses/finetuning/code/zero_memory_planner.py --smoke
    python content/courses/finetuning/code/zero_memory_planner.py --full --params 8e9 --gpus 8
    python content/courses/finetuning/code/zero_memory_planner.py --full --preset qwen3-30b-a3b --gpus 4 --lora-rank 16
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GB = 1e9  # decimal gigabytes, as used throughout the course

# Bytes per parameter for mixed-precision AdamW as in the ZeRO paper:
# bf16/fp16 weights (2) + bf16/fp16 gradients (2) + K optimizer bytes, where
# K = 12 covers the fp32 master weights (4), Adam momentum (4) and variance (4).
K_ADAM = 12

# NF4 with double quantization: 4 bits + ~0.127 bits of quantization constants
# per parameter (Dettmers et al. 2023, QLoRA, https://arxiv.org/abs/2305.14314).
NF4_BYTES_PER_PARAM = (4 + 0.127) / 8


def zero_per_gpu_bytes(psi: float, n_d: int, stage: int, k: int = K_ADAM) -> float:
    """Per-GPU model-state bytes for ZeRO stage 0 (DDP), 1, 2 or 3.

    Formulas from the ZeRO paper, Section 5 / Figure 1:
      stage 0 (baseline DDP):   (2 + 2 + K) * psi
      stage 1 (P_os):           2*psi + 2*psi + K*psi / N_d
      stage 2 (P_os+g):         2*psi + (2 + K)*psi / N_d
      stage 3 (P_os+g+p):       (2 + 2 + K)*psi / N_d
    """
    if stage == 0:
        return (2 + 2 + k) * psi
    if stage == 1:
        return 4 * psi + k * psi / n_d
    if stage == 2:
        return 2 * psi + (2 + k) * psi / n_d
    if stage == 3:
        return (2 + 2 + k) * psi / n_d
    raise ValueError(f"unknown ZeRO stage {stage}")


# FSDP sharding modes and the ZeRO stage they shard like (same three things,
# different implementation). HSDP shards like ZeRO-3 inside a replica group.
FSDP_EQUIVALENT = {
    "NO_SHARD (DDP)": 0,
    "SHARD_GRAD_OP / reshard_after_forward=False": 2,
    "FULL_SHARD / reshard_after_forward=True": 3,
}


def hsdp_per_gpu_bytes(psi: float, shard_group: int, k: int = K_ADAM) -> float:
    """HSDP: full sharding inside a group of `shard_group` GPUs, replicated across groups."""
    return zero_per_gpu_bytes(psi, shard_group, 3, k)


def adapter_per_gpu_bytes(base_params: float, trainable: float, n_d: int,
                          base_bytes: float, shard_base: bool) -> float:
    """Frozen base + trainable adapter. The adapter carries the full 2+2+12 bytes,
    the frozen base only its storage bytes (no grads, no optimizer state)."""
    base = base_params * base_bytes / (n_d if shard_base else 1)
    adapter = (2 + 2 + K_ADAM) * trainable / (n_d if shard_base else 1)
    return base + adapter


@dataclass(frozen=True)
class MoEConfig:
    """Enough of a decoder-only MoE config to count parameters exactly."""
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    experts_per_token: int
    moe_intermediate_size: int
    num_layers: int
    vocab_size: int
    tied_embeddings: bool = False
    qk_norm: bool = True

    def _attn(self) -> int:
        d, h, kv, hd = self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim
        n = d * h * hd + 2 * d * kv * hd + h * hd * d  # q, k, v, o projections
        return n + (2 * hd if self.qk_norm else 0)

    def _expert(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size  # gate, up, down

    def _per_layer_shared(self) -> int:
        router = self.hidden_size * self.num_experts
        norms = 2 * self.hidden_size
        return self._attn() + router + norms

    def _embeddings(self) -> int:
        e = self.vocab_size * self.hidden_size
        return e if self.tied_embeddings else 2 * e

    def total_params(self) -> int:
        per_layer = self._per_layer_shared() + self.num_experts * self._expert()
        return self.num_layers * per_layer + self._embeddings() + self.hidden_size

    def active_params(self) -> int:
        per_layer = self._per_layer_shared() + self.experts_per_token * self._expert()
        return self.num_layers * per_layer + self._embeddings() + self.hidden_size

    def expert_lora_params(self, r: int) -> int:
        """Rank-r LoRA on gate, up and down of every expert in every layer:
        L * E * 3 * r * (d + d_ff). Libraries that fuse gate_up_proj lay the
        adapters out differently; treat this as the per-expert mental model."""
        d, f = self.hidden_size, self.moe_intermediate_size
        return self.num_layers * self.num_experts * 3 * r * (d + f)


# Values from Qwen/Qwen3-30B-A3B config.json at commit d47d535f78ec44bd57128f8e8aeba17eeb0285ea
QWEN3_30B_A3B = MoEConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=128,
    num_experts=128, experts_per_token=8, moe_intermediate_size=768,
    num_layers=48, vocab_size=151936, tied_embeddings=False,
)

PRESETS = {"qwen3-30b-a3b": QWEN3_30B_A3B}


def table(psi: float, n_d: int, lora_trainable: float | None = None) -> list[tuple[str, float]]:
    rows = [
        ("DDP (ZeRO-0, NO_SHARD)", zero_per_gpu_bytes(psi, n_d, 0)),
        ("ZeRO-1", zero_per_gpu_bytes(psi, n_d, 1)),
        ("ZeRO-2 / FSDP SHARD_GRAD_OP", zero_per_gpu_bytes(psi, n_d, 2)),
        ("ZeRO-3 / FSDP FULL_SHARD", zero_per_gpu_bytes(psi, n_d, 3)),
    ]
    if n_d >= 16 and n_d % 8 == 0:
        rows.append(("HSDP (shard 8 per node, replicate)", hsdp_per_gpu_bytes(psi, 8)))
    if lora_trainable is not None:
        rows += [
            ("LoRA, bf16 base replicated (DDP)",
             adapter_per_gpu_bytes(psi, lora_trainable, n_d, 2.0, shard_base=False)),
            ("LoRA, bf16 base sharded (FSDP2)",
             adapter_per_gpu_bytes(psi, lora_trainable, n_d, 2.0, shard_base=True)),
            ("QLoRA, NF4 base sharded (FSDP2)",
             adapter_per_gpu_bytes(psi, lora_trainable, n_d, NF4_BYTES_PER_PARAM, shard_base=True)),
        ]
    return rows


def print_table(title: str, rows: list[tuple[str, float]]) -> None:
    print(title)
    width = max(len(name) for name, _ in rows)
    for name, b in rows:
        print(f"  {name:<{width}}  {b / GB:9.2f} GB per GPU (model states only)")


def smoke() -> None:
    # 1. The ZeRO paper's worked example: 7.5B parameters, 64 GPUs, K = 12.
    psi, n_d = 7.5e9, 64
    got = [zero_per_gpu_bytes(psi, n_d, s) / GB for s in range(4)]
    want = [120.0, 31.40625, 16.640625, 1.875]
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-9, (g, w)
    # 2. Each stage strictly reduces memory once there are at least two GPUs.
    for n in (2, 4, 8, 64, 1024):
        vals = [zero_per_gpu_bytes(psi, n, s) for s in range(4)]
        assert all(a > b for a, b in zip(vals, vals[1:])), (n, vals)
    # 3. Stage 3 keeps no redundant copy: per-GPU x N_d equals the baseline.
    assert abs(zero_per_gpu_bytes(psi, n_d, 3) * n_d - zero_per_gpu_bytes(psi, n_d, 0)) < 1.0
    # 4. Qwen3-30B-A3B counts match the model card (30.5B total, 3.3B active).
    cfg = QWEN3_30B_A3B
    total, active = cfg.total_params(), cfg.active_params()
    assert total == 30_532_122_624, total
    assert active == 3_353_032_704, active
    assert abs(total * 2 / GB - 61.064245248) < 1e-9
    # 5. Expert LoRA explodes with E: r=16 on every expert is 0.83B params;
    #    PEFT's suggested per-expert rank r // E (floored at 1) brings it to ~52M.
    assert cfg.expert_lora_params(16) == 830_472_192
    assert cfg.expert_lora_params(max(1, 16 // cfg.num_experts)) == 51_904_512
    print_table("ZeRO paper example: 7.5B params, 64 GPUs", table(psi, n_d))
    print(f"Qwen3-30B-A3B: total {total / 1e9:.2f}B ({total * 2 / GB:.1f} GB bf16), "
          f"active {active / 1e9:.2f}B ({active * 2 / GB:.1f} GB bf16)")
    print("smoke OK: ZeRO formulas, monotonicity, no-redundancy, MoE counts verified")


def full(args: argparse.Namespace) -> None:
    if args.preset:
        cfg = PRESETS[args.preset]
        psi = float(cfg.total_params())
        trainable = None
        if args.lora_rank:
            trainable = float(cfg.expert_lora_params(max(1, args.lora_rank // cfg.num_experts)))
        print(f"{args.preset}: total {psi / 1e9:.2f}B, active {cfg.active_params() / 1e9:.2f}B; "
              "memory follows TOTAL parameters")
    else:
        psi = args.params
        trainable = psi * args.lora_fraction if args.lora_rank else None
    print_table(f"{psi / 1e9:.2f}B params on {args.gpus} GPUs", table(psi, args.gpus, trainable))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="assert the paper's numbers (default)")
    mode.add_argument("--full", action="store_true", help="print a table for --params/--preset")
    p.add_argument("--params", type=float, default=8e9, help="parameter count (Psi)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--gpus", type=int, default=8, help="data-parallel degree N_d")
    p.add_argument("--lora-rank", type=int, default=0, help="add adapter rows if > 0")
    p.add_argument("--lora-fraction", type=float, default=0.005,
                   help="trainable fraction for dense LoRA rows (all-linear r=16 on 8B is ~0.5%%)")
    args = p.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
