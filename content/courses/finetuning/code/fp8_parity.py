"""FP8 numerics on CPU, and FP8-vs-bf16 SFT throughput and loss parity on GPU.

Module: 11 — Quantization Across the Lifecycle: QLoRA, QAT, FP8, and Export
(Lab 11.8, part L11.3).

What it does
    --smoke (CPU, stdlib + torch only, no downloads, a few seconds):
      1. Checks the E4M3 and E5M2 ranges (448 and 57,344) and the scaled
         round-trip relative error bounds for normal values: 2^-4 for E4M3
         (3 mantissa bits) and 2^-3 for E5M2 (2 mantissa bits).
      2. Shows why unscaled casts are unsafe: values above the E4M3 range do
         not survive a plain cast (NaN in recent PyTorch; any saturation would
         be just as wrong).
      3. Delayed (history-based) scaling vs current (just-in-time) scaling when
         an activation spike arrives: the delayed scale overflows the spike,
         the current scale represents it.
      4. Per-tensor vs 1x128 tile scaling on a tensor with one outlier row:
         tile scaling keeps the other rows accurate.
      5. Trains a small MLP twice, once in fp32 and once with simulated FP8
         matmuls (E4M3 inputs/weights forward, E5M2 output gradients backward,
         fp32 master weights and AdamW state), and asserts the final losses
         agree within 10%.
    --full (GPU, T2: one H100 or B200 80 GB): full fine-tunes a 3-8B model
      (default Qwen/Qwen3-4B) for a fixed number of steps twice, bf16 and FP8
      via torchao `convert_to_float8_training` (tensorwise or rowwise recipe),
      both under torch.compile, and prints tokens/s and the loss curves so you
      can judge throughput gain and loss parity.

Hardware tier for --full: T2 (H100/H200 for FP8 tensor cores; B200 also works).
    Ampere (A100) has no FP8 tensor cores; the FP8 run will not be faster there.
Pinned versions for --full: torch>=2.8, torchao>=0.18 (Float8LinearConfig,
    convert_to_float8_training as in v0.18.0), transformers>=5.0, datasets>=3.0.

Run from the repository root:
    python content/courses/finetuning/code/fp8_parity.py --smoke
    python content/courses/finetuning/code/fp8_parity.py --full \
        --model Qwen/Qwen3-4B --steps 200 --seq-len 2048 --batch 4 --recipe tensorwise
"""

import argparse
import time

import torch

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
E4M3_MAX = 448.0
E5M2_MAX = 57344.0


def scaled_roundtrip(x, dtype, fmax, scale=None):
    """Cast x to FP8 with scale s = amax / fmax (or a given s) and back."""
    if scale is None:
        scale = (x.abs().max() / fmax).clamp_min(1e-12)
    return (x / scale).to(dtype).float() * scale, scale


class FP8Matmul(torch.autograd.Function):
    """y = x @ w.T with E4M3 operands forward and E5M2 output gradient backward.

    Mirrors what FP8 training recipes cast: the GEMM inputs. Accumulation and
    outputs stay in fp32 here (bf16/fp32 on real hardware), and the master
    weights the optimizer updates are never stored in FP8.
    """

    @staticmethod
    def forward(ctx, x, w):
        xq, _ = scaled_roundtrip(x, E4M3, E4M3_MAX)
        wq, _ = scaled_roundtrip(w, E4M3, E4M3_MAX)
        ctx.save_for_backward(xq, wq)
        return xq @ wq.T

    @staticmethod
    def backward(ctx, grad_out):
        xq, wq = ctx.saved_tensors
        gq, _ = scaled_roundtrip(grad_out, E5M2, E5M2_MAX)
        return gq @ wq, gq.T @ xq


def smoke():
    t0 = time.time()
    torch.manual_seed(0)

    # 1. Ranges and scaled round-trip bounds.
    assert torch.finfo(E4M3).max == E4M3_MAX
    assert torch.finfo(E5M2).max == E5M2_MAX
    x = torch.randn(4096) * 3.0
    for dtype, fmax, smallest_normal, bound in ((E4M3, E4M3_MAX, 2.0 ** -6, 2.0 ** -4),
                                                (E5M2, E5M2_MAX, 2.0 ** -14, 2.0 ** -3)):
        y, s = scaled_roundtrip(x, dtype, fmax)
        normal = (x / s).abs() >= smallest_normal
        rel = ((y - x).abs() / x.abs())[normal].max().item()
        print(f"[1] {str(dtype):22s} max rel error {rel:.4f} (bound {bound})")
        assert rel <= bound

    # 2. Unscaled cast of out-of-range values destroys them.
    big = torch.tensor([1000.0, -600.0, 500.0])
    plain = big.to(E4M3).float()
    rel_err = (plain - big).abs() / big.abs()
    print(f"[2] plain E4M3 cast of {big.tolist()} -> {plain.tolist()}")
    assert not bool((rel_err < 0.5).any()), "out-of-range values must not survive"

    # 3. Delayed scaling meets a spike.
    history = [torch.randn(1024).abs().max() for _ in range(16)]
    act = torch.randn(1024)
    act[3] = 40.0
    s_delayed = max(history) / E4M3_MAX
    y_delayed, _ = scaled_roundtrip(act, E4M3, E4M3_MAX, scale=s_delayed)
    y_current, _ = scaled_roundtrip(act, E4M3, E4M3_MAX)
    print(f"[3] spike 40.0 -> delayed scaling {y_delayed[3].item()}, "
          f"current scaling {y_current[3].item()}")
    assert not bool((y_delayed[3] - 40.0).abs() < 1.0)
    assert abs(y_current[3].item() - 40.0) <= 40.0 * 2 ** -4

    # 4. Per-tensor vs 1x128 tile scaling with one outlier row.
    a = torch.randn(8, 128) * 0.01
    a[0] *= 1e4
    y_tensor, _ = scaled_roundtrip(a, E4M3, E4M3_MAX)
    y_tile = torch.stack([scaled_roundtrip(row, E4M3, E4M3_MAX)[0] for row in a])
    err_tensor = ((y_tensor - a)[1:] ** 2).mean().item()
    err_tile = ((y_tile - a)[1:] ** 2).mean().item()
    print(f"[4] MSE on the 7 normal rows: per-tensor {err_tensor:.3e}, 1x128 tiles {err_tile:.3e}")
    assert err_tile < err_tensor

    # 5. FP8-simulated training vs fp32 on a small MLP.
    X = torch.randn(512, 32)
    Y = torch.tanh(X @ (torch.randn(16, 32) / 6).T)

    def train(fp8):
        torch.manual_seed(1)
        W1 = (torch.randn(64, 32) / 6).requires_grad_()
        W2 = (torch.randn(16, 64) / 8).requires_grad_()
        opt = torch.optim.AdamW([W1, W2], lr=3e-3)
        mm = FP8Matmul.apply if fp8 else (lambda inp, w: inp @ w.T)
        for _ in range(300):
            loss = ((mm(torch.relu(mm(X, W1)), W2) - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    l32, l8 = train(False), train(True)
    gap = abs(l8 - l32) / l32
    print(f"[5] final loss fp32 {l32:.4f}, simulated FP8 {l8:.4f} (relative gap {100 * gap:.1f}%)")
    assert gap < 0.10

    print(f"smoke ok in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Full mode (GPU, T2). Heavy imports stay inside these functions.
# ---------------------------------------------------------------------------

def packed_batches(tok, seq_len, batch, steps):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    buf, need = [], seq_len * batch
    for row in ds:
        if not row["text"].strip():
            continue
        buf.extend(tok(row["text"]).input_ids)
        while len(buf) >= need:
            yield torch.tensor(buf[:need]).view(batch, seq_len)
            buf = buf[need:]
            steps -= 1
            if steps == 0:
                return


def run(model_id, fp8, recipe, steps, seq_len, batch, lr):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    if fp8:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        def module_filter_fn(mod, fqn):
            # Skip the LM head and any Linear whose dims are not multiples of 16.
            if fqn.endswith("lm_head"):
                return False
            if isinstance(mod, torch.nn.Linear):
                return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
            return True

        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(recipe),
                                   module_filter_fn=module_filter_fn)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    losses, t_start, tokens = [], None, 0
    for step, ids in enumerate(packed_batches(tok, seq_len, batch, steps)):
        ids = ids.cuda()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step == 10:  # exclude compile and warm-up steps from throughput
            torch.cuda.synchronize()
            t_start = time.time()
        elif step > 10:
            tokens += ids.numel()
    torch.cuda.synchronize()
    tps = tokens / (time.time() - t_start)
    del model, opt
    torch.cuda.empty_cache()
    return tps, losses


def full(args):
    if not torch.cuda.is_available():
        raise SystemExit("--full needs a CUDA GPU (tier T2, H100/B200). Use --smoke on CPU.")
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print("warning: no FP8 tensor cores below compute capability 8.9/9.0; expect no speedup")
    tps_bf16, loss_bf16 = run(args.model, False, args.recipe, args.steps, args.seq_len, args.batch, args.lr)
    tps_fp8, loss_fp8 = run(args.model, True, args.recipe, args.steps, args.seq_len, args.batch, args.lr)
    k = max(1, len(loss_bf16) // 10)
    tail_bf16 = sum(loss_bf16[-k:]) / k
    tail_fp8 = sum(loss_fp8[-k:]) / k
    print(f"bf16: {tps_bf16:,.0f} tok/s, mean loss last {k} steps {tail_bf16:.4f}")
    print(f"fp8 ({args.recipe}): {tps_fp8:,.0f} tok/s, mean loss last {k} steps {tail_fp8:.4f}")
    print(f"speedup {tps_fp8 / tps_bf16:.2f}x, loss gap {100 * (tail_fp8 - tail_bf16) / tail_bf16:+.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU numerics run (default)")
    mode.add_argument("--full", action="store_true", help="GPU throughput/parity run (tier T2)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--recipe", default="tensorwise", choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
