"""Long-context memory and communication calculator.

Module: 17 — Long-Context Extension and Long-Sequence Training.

Purpose
    Tabulate, for a decoder config, the sequence-length-dependent memory terms of
    long-context training (logits, checkpointed layer inputs, MLP intermediates
    during recompute, naive attention scores), the inference KV cache, and the
    per-rank sizes under sequence/context parallelism of degree P, plus per-layer
    Ulysses all-to-all and Ring KV traffic (bytes each rank sends). Asserts the
    module's worked numbers for the Qwen3-8B config at T = 131,072: 39.8 GB of bf16
    logits, 19.3 GB of bf16 KV cache, and 0.294 GB (Ulysses) vs 0.470 GB (Ring) per
    layer forward at P = 8, a 5/P ratio: Ring sends less at the usable Ulysses
    degrees 2 and 4, Ulysses less at 8.

Hardware tier: CPU only (stdlib). There is no --full mode: the GPU-side
measurement of this module (L17.2, tier T3) is the TRL/Accelerate run described
in the module's lab section; compare its peak memory with this table.

Pinned versions: Python 3.11+, no third-party packages.

Run (from the repository root):
    python content/courses/finetuning/code/long_context_memory.py --smoke
    python content/courses/finetuning/code/long_context_memory.py --seq 131072 --P 1,2,4,8
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

GB = 1e9  # decimal gigabytes, the course convention


@dataclass(frozen=True)
class Cfg:
    name: str = "Qwen3-8B"
    d: int = 4096          # hidden_size
    d_ff: int = 12288      # intermediate_size
    L: int = 36            # num_hidden_layers
    V: int = 151_936       # vocab_size
    H: int = 32            # num_attention_heads
    H_kv: int = 8          # num_key_value_heads
    d_head: int = 128      # head_dim


def terms(c: Cfg, T: int, P: int = 1, bytes_: int = 2) -> dict:
    """Per-GPU bytes for one sequence of length T split over P sequence ranks."""
    t = T / P
    return {
        "logits T*V": t * c.V * bytes_,
        "ckpt inputs L*T*d": c.L * t * c.d * bytes_,
        "MLP recompute 4*T*d_ff": 4 * t * c.d_ff * bytes_,
        "naive scores H*T^2 (1 layer)": c.H * t * T * bytes_,
        "KV cache 2*L*Hkv*dh*T": 2 * c.L * c.H_kv * c.d_head * t * bytes_,
    }


def ulysses_bytes_per_layer(c: Cfg, T: int, P: int, bytes_: int = 2) -> float:
    """Bytes one rank sends per layer forward in Ulysses' all-to-alls. Each rank holds a
    T/P-token shard of q and out (d wide) and of k and v (H_kv*d_head wide, GQA); an
    all-to-all keeps 1/P of the shard local and sends the other (P-1)/P."""
    width = 2 * c.d + 2 * c.H_kv * c.d_head
    return T / P * width * (P - 1) / P * bytes_


def ring_bytes_per_layer(c: Cfg, T: int, P: int, bytes_: int = 2) -> float:
    """Bytes one rank sends (= receives) per layer forward in Ring attention:
    P-1 hops, each carrying one (k, v) block of T/P tokens."""
    return (P - 1) * 2 * (T / P) * c.H_kv * c.d_head * bytes_


def ulysses_max_degree(c: Cfg, replicate_kv: bool) -> int:
    """Without KV replication each rank needs whole KV heads: P <= H_kv (and divides it).
    ALST replicates KV heads when P > H_kv; then the cap is the query-head count."""
    return c.H if replicate_kv else c.H_kv


def table(c: Cfg, seqs, Ps) -> None:
    print(f"config {c.name}: d={c.d} d_ff={c.d_ff} L={c.L} V={c.V} H={c.H} H_kv={c.H_kv} d_head={c.d_head}; bf16, GB=1e9")
    for P in Ps:
        print(f"\n-- per GPU, P={P} --")
        print(f"{'term':<30}" + "".join(f"{T:>12,}" for T in seqs))
        keys = terms(c, seqs[0], P).keys()
        for k in keys:
            print(f"{k:<30}" + "".join(f"{terms(c, T, P)[k] / GB:>12.2f}" for T in seqs))
        if P > 1:
            print(f"{'Ulysses a2a sent / layer fwd':<30}" + "".join(f"{ulysses_bytes_per_layer(c, T, P) / GB:>12.3f}" for T in seqs))
            print(f"{'Ring KV sent / layer fwd':<30}" + "".join(f"{ring_bytes_per_layer(c, T, P) / GB:>12.3f}" for T in seqs))


def smoke() -> None:
    c = Cfg()
    t128 = terms(c, 131_072)
    assert round(t128["logits T*V"] / GB, 1) == 39.8
    assert round(t128["KV cache 2*L*Hkv*dh*T"] / GB, 1) == 19.3
    assert round(t128["ckpt inputs L*T*d"] / GB, 1) == 38.7
    assert round(t128["MLP recompute 4*T*d_ff"] / GB, 1) == 12.9
    assert round(terms(c, 8192)["logits T*V"] / GB, 2) == 2.49
    assert round(terms(c, 32_768)["logits T*V"] / GB, 2) == 9.96
    # every T-linear term shrinks exactly by P under sequence sharding
    for k, v in terms(c, 131_072, P=8).items():
        if "scores" not in k:
            assert abs(v - t128[k] / 8) < 1e-3
    # Ulysses traffic per rank falls with P; ring traffic per rank does not.
    # Counted the same way (bytes sent), Ulysses/Ring = (d + H_kv*d_head) / (H_kv*d_head*P) = 5/P.
    T = 131_072
    u4, u8 = ulysses_bytes_per_layer(c, T, 4), ulysses_bytes_per_layer(c, T, 8)
    r4, r8 = ring_bytes_per_layer(c, T, 4), ring_bytes_per_layer(c, T, 8)
    assert u8 < u4 and r8 > r4
    assert round(u8 / GB, 3) == 0.294 and round(r8 / GB, 3) == 0.470
    for P in range(2, 33):
        assert math.isclose(ulysses_bytes_per_layer(c, T, P) / ring_bytes_per_layer(c, T, P), 5 / P, rel_tol=1e-12)
    # the formula ties at P = 5, but Ulysses can only use P that divides H and is <= H_kv
    assert math.isclose(ulysses_bytes_per_layer(c, T, 5), ring_bytes_per_layer(c, T, 5))
    usable = [P for P in range(2, c.H_kv + 1) if c.H % P == 0]
    assert usable == [2, 4, 8]
    assert u4 > r4 and ulysses_bytes_per_layer(c, T, 2) > ring_bytes_per_layer(c, T, 2) and u8 < r8
    assert ulysses_max_degree(c, replicate_kv=False) == 8 and ulysses_max_degree(c, replicate_kv=True) == 32
    table(c, [8192, 32_768, 131_072], [1, 8])
    print("\nsmoke OK: 39.8 GB logits, 19.3 GB KV cache at 131,072 tokens; Ulysses/Ring = 5/P; Ring wins at P=2,4, Ulysses at P=8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seq", default="8192,32768,131072")
    ap.add_argument("--P", default="1,2,4,8")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        table(Cfg(), [int(x) for x in args.seq.split(",")], [int(x) for x in args.P.split(",")])


if __name__ == "__main__":
    main()
