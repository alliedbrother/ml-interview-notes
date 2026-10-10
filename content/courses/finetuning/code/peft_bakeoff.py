"""PEFT bake-off: LoRA, DoRA, IA3, prompt tuning and OFT on the same task.

Module: 07 — Beyond LoRA: The PEFT Landscape and Memory-Efficient Full Tuning
(content/courses/finetuning/07-the-wider-peft-landscape.md, section 7.10).

Two modes
---------
--smoke (CPU, default when no GPU is found; stdlib + torch only, < 60 s):
    Pure-torch implementations of LoRA, DoRA, IA3, prompt tuning and a
    Cayley-parameterised block-diagonal OFT on a tiny randomly initialised
    attention block. Every method trains for 100 Adam steps to imitate a
    perturbed "teacher" copy of the block. The script asserts that each method
    reduces the loss, that its trainable-parameter count matches the closed
    form in the module, and that LoRA, DoRA, IA3 and OFT merge into a dense
    weight that reproduces the adapted forward pass. It also runs a toy GaLore
    step (projected Adam state) and the Spectrum SNR scorer.

--full (GPU tier T1 for L7.1: one 24 GB GPU; tier T2 for --galore / --spectrum / --full-ft):
    L7.1: PEFT + TRL SFT of Qwen/Qwen3-1.7B on a 4,000-example slice of
    trl-lib/Capybara. Configs: LoRA and DoRA r=16 all-linear; IA3 on
    k_proj/v_proj/down_proj; 32 virtual tokens with vocabulary init; OFT with
    oft_block_size=32. Each trains with TRL SFTTrainer (processing_class=,
    max_length=1024, model_init_kwargs={"dtype": torch.bfloat16}) for 300 steps
    at a per-method LR (2e-4 LoRA/DoRA, 1e-3 IA3 and prompt, 1e-4 OFT). Prints a
    Markdown table of trainable params, eval loss, s/step, peak memory,
    mergeability and vLLM servability.
    L7.2: on Qwen/Qwen2.5-3B (one 80 GB GPU, gradient checkpointing, bf16
    weights), one of:
      --galore      optim="galore_adamw", optim_target_modules for attention and
                    MLP, rank=128, update_proj_gap=200, scale=0.25;
      --spectrum F  score every attention/MLP matrix with spectrum_snr (the
                    smoke-mode scorer), unfreeze the top fraction F per module
                    type, train with plain AdamW;
      --full-ft     plain AdamW on every parameter (the memory baseline).
    Each prints peak memory, train samples/s and eval loss. The LoRA baseline is
    `--full --methods lora --model Qwen/Qwen2.5-3B`.

Pinned versions for full mode (as of October 2026): transformers 5.19.0,
peft 0.21.2, trl 1.15.x, datasets 4.x, galore-torch (for --galore).

Run commands (from the repository root):
    python content/courses/finetuning/code/peft_bakeoff.py --smoke
    python content/courses/finetuning/code/peft_bakeoff.py --full --methods lora dora ia3 prompt oft
    python content/courses/finetuning/code/peft_bakeoff.py --full --galore
    python content/courses/finetuning/code/peft_bakeoff.py --full --spectrum 0.25
    python content/courses/finetuning/code/peft_bakeoff.py --full --full-ft
    python content/courses/finetuning/code/peft_bakeoff.py --full --methods lora --model Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Smoke mode: a tiny attention + SwiGLU block and five PEFT methods in pure torch
# ---------------------------------------------------------------------------

D, D_KV, D_FF, T, N_HEAD_KV = 32, 16, 64, 8, 1  # width, k/v width, MLP width, seq len
LINEARS = {  # name: (in_features, out_features)
    "q": (D, D), "k": (D, D_KV), "v": (D, D_KV), "o": (D, D),
    "gate": (D, D_FF), "up": (D, D_FF), "down": (D_FF, D),
}


def base_weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(o, i, generator=g) / math.sqrt(i) for n, (i, o) in LINEARS.items()}


def block_forward(x: torch.Tensor, lin) -> torch.Tensor:
    """x: (B, T', D). `lin(name, h)` applies the (possibly adapted) linear `name`."""
    q = lin("q", x)                      # (B, T', D)
    k = lin("k", x)                      # (B, T', D_KV)
    v = lin("v", x)                      # (B, T', D_KV)
    k = k.repeat(1, 1, D // D_KV)        # GQA-style broadcast of the single KV group
    v = v.repeat(1, 1, D // D_KV)
    att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(D), dim=-1)  # (B, T', T')
    h = x + lin("o", att @ v)
    m = F.silu(lin("gate", h)) * lin("up", h)                          # (B, T', D_FF)
    return h + lin("down", m)


def dense_lin(weights):
    return lambda name, h: h @ weights[name].T


class Method(nn.Module):
    """Base class: holds frozen base weights and exposes forward/merge/param count."""

    n_prompt = 0

    def __init__(self, w0: dict[str, torch.Tensor]):
        super().__init__()
        self.w0 = {n: w.clone() for n, w in w0.items()}  # frozen (plain tensors)

    def lin(self, name, h):
        return h @ self.w0[name].T

    def forward(self, x):
        return block_forward(x, self.lin)

    def merged(self) -> dict[str, torch.Tensor] | None:
        return None

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRA(Method):
    def __init__(self, w0, r=4, alpha=8, dora=False):
        super().__init__(w0)
        self.scale, self.dora = alpha / r, dora
        g = torch.Generator().manual_seed(1)
        self.A = nn.ParameterDict({n: nn.Parameter(torch.randn(r, i, generator=g) / math.sqrt(i))
                                   for n, (i, o) in LINEARS.items()})
        self.B = nn.ParameterDict({n: nn.Parameter(torch.zeros(o, r)) for n, (i, o) in LINEARS.items()})
        if dora:  # magnitude initialised to the per-output-row norm of W0 (PEFT convention)
            self.m = nn.ParameterDict({n: nn.Parameter(w.norm(dim=1)) for n, w in w0.items()})

    def adapted_weight(self, name):
        return self.w0[name] + self.scale * self.B[name] @ self.A[name]   # (out, in)

    def lin(self, name, h):
        if not self.dora:
            return h @ self.w0[name].T + self.scale * (h @ self.A[name].T) @ self.B[name].T
        v = self.adapted_weight(name)
        norm = v.norm(dim=1).detach()        # DoRA section 4.3: norm treated as constant
        return (h @ v.T) * (self.m[name] / norm)

    def merged(self):
        out = {}
        for n in LINEARS:
            v = self.adapted_weight(n).detach()
            out[n] = v * (self.m[n].detach() / v.norm(dim=1))[:, None] if self.dora else v
        return out


class IA3(Method):
    """l_k, l_v rescale key/value outputs; l_ff rescales the input of the down projection."""

    def __init__(self, w0):
        super().__init__(w0)
        self.l = nn.ParameterDict({"k": nn.Parameter(torch.ones(D_KV)),
                                   "v": nn.Parameter(torch.ones(D_KV)),
                                   "down": nn.Parameter(torch.ones(D_FF))})

    def lin(self, name, h):
        if name in ("k", "v"):
            return (h @ self.w0[name].T) * self.l[name]
        if name == "down":
            return (h * self.l["down"]) @ self.w0[name].T
        return h @ self.w0[name].T

    def merged(self):
        out = dict(self.w0)
        out["k"] = self.w0["k"] * self.l["k"].detach()[:, None]        # scale output rows
        out["v"] = self.w0["v"] * self.l["v"].detach()[:, None]
        out["down"] = self.w0["down"] * self.l["down"].detach()[None, :]  # scale input columns
        return out


class PromptTuning(Method):
    def __init__(self, w0, n_virtual=4):
        super().__init__(w0)
        self.n_prompt = n_virtual
        g = torch.Generator().manual_seed(2)
        self.prompt = nn.Parameter(0.5 * torch.randn(n_virtual, D, generator=g))

    def forward(self, x):
        p = self.prompt.unsqueeze(0).expand(x.shape[0], -1, -1)   # (B, n, D)
        y = block_forward(torch.cat([p, x], dim=1), self.lin)    # (B, n + T, D)
        return y[:, self.n_prompt:]                              # drop virtual positions


def cayley(s_param: torch.Tensor) -> torch.Tensor:
    """Exact Cayley transform of a batch of skew-symmetric blocks: (nb, b, b) -> (nb, b, b)."""
    s = s_param - s_param.transpose(-1, -2)
    eye = torch.eye(s.shape[-1]).expand_as(s)
    return (eye + s) @ torch.linalg.inv(eye - s)


class OFT(Method):
    """Block-diagonal orthogonal R acting on the input side: W' = W0 @ blockdiag(R_1..R_nb)."""

    def __init__(self, w0, block=8):
        super().__init__(w0)
        self.block = block
        # store only the strictly upper triangle: b(b-1)/2 free parameters per block
        self.iu = torch.triu_indices(block, block, offset=1)
        self.p = nn.ParameterDict({n: nn.Parameter(torch.zeros(i // block, self.iu.shape[1]))
                                   for n, (i, o) in LINEARS.items()})

    def rotation(self, name):
        nb = self.p[name].shape[0]
        s = torch.zeros(nb, self.block, self.block)
        s[:, self.iu[0], self.iu[1]] = self.p[name]
        return torch.block_diag(*cayley(s))           # (in, in), orthogonal

    def lin(self, name, h):
        return h @ (self.w0[name] @ self.rotation(name)).T

    def merged(self):
        return {n: (self.w0[n] @ self.rotation(n)).detach() for n in LINEARS}


def closed_form_counts(r=4, n_virtual=4, block=8) -> dict[str, int]:
    lora = sum(r * (i + o) for i, o in LINEARS.values())
    return {
        "lora": lora,
        "dora": lora + sum(o for _, o in LINEARS.values()),
        "ia3": D_KV + D_KV + D_FF,
        "prompt": n_virtual * D,
        "oft": sum((i // block) * block * (block - 1) // 2 for i, _ in LINEARS.values()),
    }


def run_smoke(steps: int = 100) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    w0 = base_weights(seed=0)
    # Teacher: base + a low-rank change everywhere, plus a rescaling of k, v and the MLP input.
    g = torch.Generator().manual_seed(123)
    teacher = {}
    for n, w in w0.items():
        o, i = w.shape
        teacher[n] = w + 0.3 * (torch.randn(o, 2, generator=g) @ torch.randn(2, i, generator=g)) / math.sqrt(i)
    teacher["k"] = teacher["k"] * 1.3
    teacher["v"] = teacher["v"] * 0.7
    x = torch.randn(64, T, D, generator=g)
    with torch.no_grad():
        y = block_forward(x, dense_lin(teacher))

    methods = {"lora": LoRA(w0), "dora": LoRA(w0, dora=True), "ia3": IA3(w0),
               "prompt": PromptTuning(w0), "oft": OFT(w0)}
    expected = closed_form_counts()
    rows = []
    for name, model in methods.items():
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        with torch.no_grad():
            loss0 = F.mse_loss(model(x), y).item()
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = (time.perf_counter() - t0) / steps
        with torch.no_grad():
            loss1 = F.mse_loss(model(x), y).item()
        n_params = model.n_trainable()
        assert n_params == expected[name], (name, n_params, expected[name])
        # prompt tuning only gets a small foothold on a tiny model (see section 7.2)
        assert loss1 < (0.97 if name == "prompt" else 0.9) * loss0, (name, loss0, loss1)
        merged = model.merged()
        mergeable = merged is not None
        if mergeable:
            with torch.no_grad():
                torch.testing.assert_close(block_forward(x, dense_lin(merged)), model(x),
                                           atol=1e-4, rtol=1e-4)
        rows.append((name, n_params, loss0, loss1, dt * 1e3, mergeable))

    print(f"{'method':8s} {'params':>7s} {'loss@0':>8s} {'loss@100':>9s} {'ms/step':>8s} merged==adapted")
    for name, n, l0, l1, ms, mg in rows:
        print(f"{name:8s} {n:7d} {l0:8.4f} {l1:9.4f} {ms:8.2f} {'yes' if mg else 'n/a (not mergeable)'}")

    galore_toy()
    spectrum_toy()
    print("smoke OK: all five methods reduced the loss; counts and merges verified")


def galore_toy(m=64, n=48, r=8, steps=150, gap=50) -> None:
    """Adam on a rank-r projection of the gradient (GaLore) for least squares W x ~ y."""
    g = torch.Generator().manual_seed(7)
    w_true = torch.randn(m, n, generator=g)
    x = torch.randn(n, 256, generator=g)
    y = w_true @ x
    w = torch.zeros(m, n)
    m1 = torch.zeros(r, n)
    m2 = torch.zeros(r, n)
    b1, b2, lr, eps = 0.9, 0.999, 0.05, 1e-8
    loss0 = ((w @ x - y) ** 2).mean().item()
    p = None
    for t in range(1, steps + 1):
        grad = 2 * (w @ x - y) @ x.T / x.shape[1]          # (m, n)
        if p is None or (t - 1) % gap == 0:
            u, _, _ = torch.linalg.svd(grad, full_matrices=False)
            p = u[:, :r]                                    # (m, r) top-r left singular vectors
        rg = p.T @ grad                                     # (r, n) projected gradient
        m1 = b1 * m1 + (1 - b1) * rg
        m2 = b2 * m2 + (1 - b2) * rg ** 2
        upd = (m1 / (1 - b1 ** t)) / ((m2 / (1 - b2 ** t)).sqrt() + eps)
        w = w - lr * (p @ upd)                              # project back to (m, n)
    loss1 = ((w @ x - y) ** 2).mean().item()
    state = m1.numel() + m2.numel() + p.numel()
    assert state == 2 * r * n + m * r
    assert loss1 < 0.5 * loss0, (loss0, loss1)
    print(f"galore toy: loss {loss0:.2f} -> {loss1:.2f}; optimizer state {state} vs full Adam {2 * m * n}")


def spectrum_snr(w: torch.Tensor) -> float:
    """Spectrum-style SNR: singular values above the Marchenko-Pastur edge are signal.

    sigma is estimated from the interquartile range of the entries (Gaussian IQR = 1.349 sigma),
    as Spectrum does. For unnormalised singular values of an m x n (m <= n) noise matrix the
    Marchenko-Pastur bulk edge is sigma * (sqrt(m) + sqrt(n)) = sigma * sqrt(n) * (1 + sqrt(m/n)).
    The ratio is divided by the largest singular value, following the paper.
    """
    m, n = w.shape
    if m > n:
        w = w.T
        m, n = n, m
    s = torch.linalg.svdvals(w)
    q = torch.quantile(w.flatten(), torch.tensor([0.25, 0.75]))
    sigma = float(q[1] - q[0]) / 1.349
    eps = sigma * (1 + math.sqrt(m / n)) * math.sqrt(n)    # edge of the noise bulk for an m x n noise matrix
    signal, noise = s[s > eps].sum(), s[s <= eps].sum()
    return float(signal / max(float(noise), 1e-12)) / float(s.max())


def spectrum_toy() -> None:
    g = torch.Generator().manual_seed(11)
    noise = [torch.randn(64, 64, generator=g) for _ in range(4)]
    structured = [nz + 3.0 * torch.randn(64, 2, generator=g) @ torch.randn(2, 64, generator=g) for nz in noise[:2]]
    layers = {"noise0": noise[2], "noise1": noise[3], "struct0": structured[0], "struct1": structured[1]}
    snr = {k: spectrum_snr(v) for k, v in layers.items()}
    top = sorted(snr, key=snr.get, reverse=True)[:2]
    assert set(top) == {"struct0", "struct1"}, snr
    print("spectrum toy: SNR", {k: round(v, 4) for k, v in snr.items()}, "-> train", top)


# ---------------------------------------------------------------------------
# Full mode (GPU). Imports live inside these functions so --smoke never needs them.
# ---------------------------------------------------------------------------

VLLM_SERVABLE = {"lora": "yes (rank <= --max-lora-rank)", "dora": "no (vLLM rejects use_dora)",
                 "ia3": "no (merge first)", "prompt": "no (not mergeable)", "oft": "no (merge first)"}


def peft_config_for(method: str):
    from peft import IA3Config, LoraConfig, OFTConfig, PromptTuningConfig, PromptTuningInit

    if method in ("lora", "dora"):
        return LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", use_dora=method == "dora",
                          task_type="CAUSAL_LM")
    if method == "ia3":
        return IA3Config(target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"],
                         task_type="CAUSAL_LM")
    if method == "prompt":
        return PromptTuningConfig(num_virtual_tokens=32, prompt_tuning_init=PromptTuningInit.SAMPLE_VOCAB,
                                  task_type="CAUSAL_LM")
    if method == "oft":
        return OFTConfig(oft_block_size=32, target_modules="all-linear", task_type="CAUSAL_LM")
    raise ValueError(method)


LR = {"lora": 2e-4, "dora": 2e-4, "ia3": 1e-3, "prompt": 1e-3, "oft": 1e-4}


def run_full_l71(methods, model_id="Qwen/Qwen3-1.7B", max_steps=300) -> None:
    import torch as th
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("trl-lib/Capybara", split="train").shuffle(seed=0).select(range(4000))
    ev = load_dataset("trl-lib/Capybara", split="test").select(range(200))
    tok = AutoTokenizer.from_pretrained(model_id)
    results = []
    for method in methods:
        args = SFTConfig(
            output_dir=f"out/peft-bakeoff-{method}", max_steps=max_steps, learning_rate=LR[method],
            per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=1024,
            bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="no",
            assistant_only_loss=False, report_to="none",
            model_init_kwargs={"dtype": th.bfloat16},
        )
        trainer = SFTTrainer(model=model_id, args=args, train_dataset=ds, eval_dataset=ev,
                             processing_class=tok, peft_config=peft_config_for(method))
        n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        th.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        trainer.train()
        step_time = (time.perf_counter() - t0) / max_steps
        peak_gb = th.cuda.max_memory_allocated() / 1e9
        eval_loss = trainer.evaluate()["eval_loss"]
        mergeable = method != "prompt"
        results.append((method, n_train, eval_loss, step_time, peak_gb, mergeable, VLLM_SERVABLE[method]))
        del trainer
        th.cuda.empty_cache()
    print("| method | trainable params | eval loss | s/step | peak GB | mergeable | vLLM unmerged |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r[0]} | {r[1]:,} | {r[2]:.4f} | {r[3]:.2f} | {r[4]:.1f} | {r[5]} | {r[6]} |")


def run_full_l72(galore: bool, spectrum_fraction: float | None, full_ft: bool = False,
                 model_id="Qwen/Qwen2.5-3B", max_steps=300) -> None:
    """L7.2: GaLore, Spectrum-style selective, or plain full-AdamW fine-tuning; compare with
    `--full --methods lora --model Qwen/Qwen2.5-3B` for the LoRA baseline."""
    import torch as th
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("trl-lib/Capybara", split="train").shuffle(seed=0).select(range(4000))
    ev = load_dataset("trl-lib/Capybara", split="test").select(range(200))
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=th.bfloat16)
    common = dict(output_dir="out/l72", max_steps=max_steps, per_device_train_batch_size=4,
                  gradient_accumulation_steps=4, max_length=1024, bf16=True, logging_steps=10,
                  save_strategy="no", report_to="none", gradient_checkpointing=True)
    if galore:
        args = SFTConfig(learning_rate=1e-5, optim="galore_adamw",
                         optim_target_modules=[r".*self_attn.*", r".*mlp.*"],
                         optim_args="rank=128, update_proj_gap=200, scale=0.25", **common)
    elif full_ft:
        args = SFTConfig(learning_rate=1e-5, **common)   # every parameter, plain AdamW
    else:
        frac = spectrum_fraction or 0.25
        scores = {}
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("self_attn" in name or "mlp" in name):
                scores[name] = spectrum_snr(p.detach().float().cpu())
        for p in model.parameters():
            p.requires_grad_(False)
        # top fraction per module type, as Spectrum does
        by_type: dict[str, list[str]] = {}
        for name in scores:
            by_type.setdefault(name.split(".")[-2], []).append(name)
        chosen = set()
        for names in by_type.values():
            names.sort(key=scores.get, reverse=True)
            chosen.update(names[: max(1, int(len(names) * frac))])
        for name, p in model.named_parameters():
            if name in chosen:
                p.requires_grad_(True)
        args = SFTConfig(learning_rate=1e-5, **common)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, eval_dataset=ev, processing_class=tok)
    th.cuda.reset_peak_memory_stats()
    metrics = trainer.train().metrics
    eval_loss = trainer.evaluate()["eval_loss"]
    print(f"peak memory {th.cuda.max_memory_allocated() / 1e9:.1f} GB; "
          f"train samples/s {metrics.get('train_samples_per_second')}; eval loss {eval_loss:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test (default without a GPU)")
    ap.add_argument("--full", action="store_true", help="GPU lab")
    ap.add_argument("--methods", nargs="+", default=["lora", "dora", "ia3", "prompt", "oft"])
    ap.add_argument("--galore", action="store_true", help="L7.2: GaLore full-parameter run")
    ap.add_argument("--spectrum", type=float, default=None, help="L7.2: Spectrum top fraction")
    ap.add_argument("--full-ft", action="store_true", help="L7.2: full-AdamW memory baseline")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--model", default=None, help="override the model id (default per lab step)")
    a = ap.parse_args()
    if a.full and torch.cuda.is_available():
        if a.galore or a.spectrum is not None or a.full_ft:
            run_full_l72(a.galore, a.spectrum, full_ft=a.full_ft, model_id=a.model or "Qwen/Qwen2.5-3B",
                         max_steps=a.max_steps)
        else:
            run_full_l71(a.methods, model_id=a.model or "Qwen/Qwen3-1.7B", max_steps=a.max_steps)
    else:
        if a.full:
            print("no GPU found: falling back to --smoke")
        run_smoke()


if __name__ == "__main__":
    main()
