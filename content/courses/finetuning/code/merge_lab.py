"""Merge two task fine-tunes from a shared base, then recover general ability.

Module: 18 — Model Merging and Adapter Composition (Lab 18.8, parts L18.1 and L18.2).

What it does
    --smoke (CPU, stdlib + torch only, no downloads, well under a minute):
      1. Implements the merge primitives from module 18 on plain state dicts:
         linear averaging (model soup), SLERP, task arithmetic, TIES
         (trim / elect sign / disjoint mean), DARE (drop and rescale),
         DARE-TIES, and the three LoRA factor combinations PEFT exposes
         (`cat`, `linear`, `svd`), using PEFT's weighting convention.
      2. Pretrains a small MLP "base" on a generic task, fine-tunes two copies
         from that shared base on two different tasks (A and B), and evaluates
         every merge on task A, task B and the generic task.
      3. Sweeps the interpolation weight between the base and the task-A
         fine-tune (the L18.2 recovery curve on a toy scale).
      4. Asserts: the TIES merge beats the average of the two single-task
         models on the joint (A+B) eval; LoRA `cat` is exact, `linear` is not,
         `svd` to a lower rank has exactly the Eckart-Young error; a merge of
         two models trained from *different* random inits is worse than a
         merge of two models from the shared base.
    --full (GPU): merges a math LoRA and a code LoRA on the same Qwen3 base with
      PEFT `add_weighted_adapter` (linear, cat, svd, ties, dare_ties), writes
      merged bf16 checkpoints, writes mergekit YAML configs for a TIES merge and
      for the L18.2 instruct-recovery sweep, and prints the lm-evaluation-harness
      commands to score gsm8k, mbpp and ifeval on every candidate. The adapter
      block runs only when --math-adapter and --code-adapter are given; the
      recovery block runs only when --domain and --instruct are given.
      --resize-vocab N resizes the base embeddings before loading adapters that
      were trained with an extended vocabulary (the PEFT guide's TinyLlama
      adapters need 32005). --dtype float16 for T4 (no bf16 before Ampere).

Hardware tier for --full: T0 (T4 16 GB, fp16) for 0.6B/1.7B merges and evals;
    T1 (24 GB, bf16) for the 4B recovery sweep. Merging itself runs on CPU.
Pinned versions for --full: peft>=0.21 (add_weighted_adapter API as in
    v0.21.2), transformers>=5.0 (`dtype=`), mergekit>=0.1.4 (LGPL-3.0),
    lm-eval>=0.4.13.

Run from the repository root:
    python content/courses/finetuning/code/merge_lab.py --smoke
    python content/courses/finetuning/code/merge_lab.py --full \
        --base Qwen/Qwen3-1.7B --math-adapter ./out/math-lora \
        --code-adapter ./out/code-lora --out ./merges/
    python content/courses/finetuning/code/merge_lab.py --full \
        --domain ./out/qwen3-4b-domain --instruct Qwen/Qwen3-4B --out ./merges-4b/
"""

import argparse
import copy
import math
import os
import time

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Merge primitives (pure torch, operate on {name: tensor} state dicts)
# ---------------------------------------------------------------------------

def task_vector(finetuned, base):
    """tau = theta_ft - theta_base, per tensor."""
    return {k: finetuned[k] - base[k] for k in base}


def apply_task_vector(base, tau, lam=1.0):
    """theta = theta_base + lam * tau."""
    return {k: base[k] + lam * tau[k] for k in base}


def linear_merge(state_dicts, weights):
    """Weighted average of full parameter sets (uniform model soup if weights equal)."""
    total = sum(weights)
    return {k: sum(w * sd[k] for w, sd in zip(weights, state_dicts)) / total
            for k in state_dicts[0]}


def slerp(a, b, t, dot_threshold=0.9995):
    """Spherical interpolation of two tensors, angle measured on the flattened tensor.

    Falls back to linear interpolation when the tensors are nearly parallel,
    as mergekit does (its threshold is also 0.9995).
    """
    fa, fb = a.reshape(-1).double(), b.reshape(-1).double()
    dot = (fa / fa.norm().clamp_min(1e-12)) @ (fb / fb.norm().clamp_min(1e-12))
    dot = dot.clamp(-1.0, 1.0)
    if dot.abs() > dot_threshold:
        out = (1 - t) * fa + t * fb
    else:
        omega = torch.acos(dot)
        so = torch.sin(omega)
        out = torch.sin((1 - t) * omega) / so * fa + torch.sin(t * omega) / so * fb
    return out.reshape(a.shape).to(a.dtype)


def slerp_merge(sd_a, sd_b, t):
    return {k: slerp(sd_a[k], sd_b[k], t) for k in sd_a}


def trim_top_k(x, density):
    """Keep the round(density * n) largest-magnitude entries; ties broken by lower index."""
    flat = x.reshape(-1)
    k = int(round(density * flat.numel()))
    order = sorted(range(flat.numel()), key=lambda i: (-abs(float(flat[i])), i)) \
        if flat.numel() <= 64 else None
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if order is not None:
        mask[order[:k]] = True
    else:
        # Large tensors: stable sort on -|x| keeps lower indices first among ties.
        idx = torch.sort(-flat.abs(), stable=True).indices[:k]
        mask[idx] = True
    return (flat * mask).reshape(x.shape)


def ties_merge_tensors(taus, density, weights=None):
    """TIES on a list of same-shaped task-vector tensors: trim, elect sign, disjoint mean."""
    weights = weights or [1.0] * len(taus)
    trimmed = torch.stack([trim_top_k(t, density) for t in taus])        # (n, ...)
    elected = torch.where(trimmed.sum(0) >= 0, 1.0, -1.0).to(trimmed.dtype)
    agree = (torch.sign(trimmed) == elected) & (trimmed != 0)
    w = torch.tensor(weights, dtype=trimmed.dtype).reshape(-1, *[1] * (trimmed.dim() - 1))
    num = (trimmed * w * agree).sum(0)
    cnt = agree.sum(0).clamp_min(1)
    return num / cnt


def dare(tau, p, generator):
    """Drop each entry with probability p, rescale survivors by 1/(1-p) (unbiased)."""
    keep = torch.rand(tau.shape, generator=generator, dtype=torch.float64) >= p
    return tau * keep.to(tau.dtype) / (1.0 - p)


def dare_ties_merge_tensors(taus, p, generator, weights=None):
    """DARE each task vector, then TIES sign election and disjoint mean (no further trim)."""
    dropped = [dare(t, p, generator) for t in taus]
    return ties_merge_tensors(dropped, density=1.0, weights=weights)


def merge_state(base, finetuned_list, method, lam=1.0, density=0.5, p=0.5, seed=0):
    taus = [task_vector(ft, base) for ft in finetuned_list]
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in base:
        ts = [t[k] for t in taus]
        if method == "task_arithmetic":
            m = sum(ts)
        elif method == "ties":
            m = ties_merge_tensors(ts, density)
        elif method == "dare_ties":
            m = dare_ties_merge_tensors(ts, p, gen)
        elif method == "dare_linear":
            m = sum(dare(t, p, gen) for t in ts)
        else:
            raise ValueError(method)
        out[k] = base[k] + lam * m
    return out


def lora_combine(As, Bs, weights, scalings, combination_type, svd_rank=None):
    """Combine LoRA factors the way PEFT v0.21 add_weighted_adapter does.

    A_i: (r_i, in), B_i: (out, r_i). Returns (A_new, B_new); the new adapter has
    lora_alpha == r, so its scaling is 1 and delta W = B_new @ A_new.
    """
    if combination_type == "cat":
        A = torch.cat([w * s * a for a, w, s in zip(As, weights, scalings)], dim=0)
        B = torch.cat(Bs, dim=1)
        return A, B
    if combination_type == "linear":
        if len({a.shape[0] for a in As}) != 1:
            raise ValueError("linear needs equal ranks")
        ws = [w * s for w, s in zip(weights, scalings)]
        A = sum(math.copysign(math.sqrt(abs(w)), w) * a for a, w in zip(As, ws))
        B = sum(math.sqrt(abs(w)) * b for b, w in zip(Bs, ws))
        return A, B
    if combination_type == "svd":
        delta = sum(w * s * (b @ a) for a, b, w, s in zip(As, Bs, weights, scalings))
        r = svd_rank or max(a.shape[0] for a in As)
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        return Vh[:r], U[:, :r] @ torch.diag(S[:r])
    raise ValueError(combination_type)


# ---------------------------------------------------------------------------
# Toy experiment: shared base, two task fine-tunes, generic "general ability"
# ---------------------------------------------------------------------------

D_IN, D_HID, N_CLS = 16, 128, 10
HEADS = {"gen": 0, "A": 1, "B": 2}      # each task reads its own 10-logit slice


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_IN, D_HID)
        self.fc2 = nn.Linear(D_HID, D_HID)
        self.out = nn.Linear(D_HID, 3 * N_CLS)

    def forward(self, x):                       # x: (B, D_IN)
        return self.out(F.gelu(self.fc2(F.gelu(self.fc1(x)))))   # (B, 3 * N_CLS)


def task_logits(model, x, task):
    h = HEADS[task]
    return model(x)[:, h * N_CLS:(h + 1) * N_CLS]                # (B, N_CLS)


def make_tasks(seed=0):
    """Generic task plus two related tasks (half shared structure, half new)."""
    g = torch.Generator().manual_seed(seed)
    t_gen = torch.randn(D_IN, N_CLS, generator=g)
    teachers = {"gen": t_gen}
    for name in ("A", "B"):
        perm = torch.randperm(N_CLS, generator=g)
        teachers[name] = 0.5 * t_gen[:, perm] + 0.5 * torch.randn(D_IN, N_CLS, generator=g)

    def sample(name, n, s):
        x = torch.randn(n, D_IN, generator=torch.Generator().manual_seed(s))
        return x, (x @ teachers[name]).argmax(-1)

    return {name: {"train": sample(name, 4000, 100 + i), "test": sample(name, 2000, 200 + i)}
            for i, name in enumerate(("gen", "A", "B"))}


def train(model, data, task, steps, lr, seed):
    x, y = data[task]["train"]
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (128,), generator=g)
        loss = F.cross_entropy(task_logits(model, x[idx], task), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, sd, data, task):
    model.load_state_dict(sd)
    x, y = data[task]["test"]
    return (task_logits(model, x, task).argmax(-1) == y).float().mean().item()


def scores(model, sd, data):
    a, b, g = (accuracy(model, sd, data, n) for n in ("A", "B", "gen"))
    return {"A": a, "B": b, "gen": g, "joint": (a + b) / 2}


def smoke():
    t0 = time.time()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    data = make_tasks(0)

    # 1. Pretrain the shared base on the generic task.
    torch.manual_seed(1)
    base_model = train(MLP(), data, "gen", steps=800, lr=3e-3, seed=11)
    base = copy.deepcopy(base_model.state_dict())

    # 2. Two fine-tunes from the same base (small LR keeps them in one basin).
    def finetune_from(init_sd, task, seed):
        m = MLP()
        m.load_state_dict(init_sd)
        return copy.deepcopy(train(m, data, task, steps=300, lr=1e-3, seed=seed).state_dict())

    ft_a = finetune_from(base, "A", 21)
    ft_b = finetune_from(base, "B", 22)

    probe = MLP()
    rows = {
        "base": scores(probe, base, data),
        "ft_A": scores(probe, ft_a, data),
        "ft_B": scores(probe, ft_b, data),
        "linear 0.5/0.5": scores(probe, linear_merge([ft_a, ft_b], [0.5, 0.5]), data),
        "slerp t=0.5": scores(probe, slerp_merge(ft_a, ft_b, 0.5), data),
        "task_arith lam=1": scores(probe, merge_state(base, [ft_a, ft_b], "task_arithmetic", lam=1.0), data),
        "ties d=0.5 lam=1": scores(probe, merge_state(base, [ft_a, ft_b], "ties", lam=1.0, density=0.5), data),
        "dare_ties p=0.5": scores(probe, merge_state(base, [ft_a, ft_b], "dare_ties", lam=1.0, p=0.5, seed=3), data),
        "dare_linear p=0.5": scores(probe, merge_state(base, [ft_a, ft_b], "dare_linear", lam=1.0, p=0.5, seed=3), data),
    }

    # 3. Different random inits: same data, no shared base.
    torch.manual_seed(2)
    other_base = copy.deepcopy(train(MLP(), data, "gen", steps=800, lr=3e-3, seed=12).state_dict())
    ft_b_other = finetune_from(other_base, "B", 22)
    rows["linear, different inits"] = scores(probe, linear_merge([ft_a, ft_b_other], [0.5, 0.5]), data)

    print(f"{'model':26s} {'A':>6s} {'B':>6s} {'gen':>6s} {'joint':>6s}")
    for name, s in rows.items():
        print(f"{name:26s} {s['A']:6.3f} {s['B']:6.3f} {s['gen']:6.3f} {s['joint']:6.3f}")

    # 4. L18.2 on a toy scale: interpolate fine-tune A back toward the base.
    print("\nrecovery sweep: theta = w * ft_A + (1 - w) * base")
    sweep = []
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        s = scores(probe, linear_merge([ft_a, base], [w, 1 - w]), data)
        sweep.append((w, s))
        print(f"  w={w:.1f}  A={s['A']:.3f}  gen={s['gen']:.3f}")

    # 5. LoRA factor combination, PEFT convention.
    g = torch.Generator().manual_seed(0)
    A1, A2 = torch.randn(4, 12, generator=g, dtype=torch.float64), torch.randn(4, 12, generator=g, dtype=torch.float64)
    B1, B2 = torch.randn(16, 4, generator=g, dtype=torch.float64), torch.randn(16, 4, generator=g, dtype=torch.float64)
    target = 0.5 * B1 @ A1 + 0.5 * B2 @ A2
    Ac, Bc = lora_combine([A1, A2], [B1, B2], [0.5, 0.5], [1.0, 1.0], "cat")
    Al, Bl = lora_combine([A1, A2], [B1, B2], [0.5, 0.5], [1.0, 1.0], "linear")
    As, Bs = lora_combine([A1, A2], [B1, B2], [0.5, 0.5], [1.0, 1.0], "svd", svd_rank=4)
    cat_err = (Bc @ Ac - target).abs().max().item()
    lin_err = (Bl @ Al - target).abs().max().item()
    svd_err = torch.linalg.matrix_norm(Bs @ As - target, ord=2).item()
    sv = torch.linalg.svdvals(target)
    print(f"\nLoRA cat max err {cat_err:.2e} | linear max err {lin_err:.3f} | "
          f"svd r=4 spectral err {svd_err:.4f} vs sigma_5 {sv[4].item():.4f}")

    # Assertions: the lab's claims.
    avg_single = (rows["ft_A"]["joint"] + rows["ft_B"]["joint"]) / 2
    assert rows["ties d=0.5 lam=1"]["joint"] > avg_single + 0.05, (rows["ties d=0.5 lam=1"], avg_single)
    assert rows["linear 0.5/0.5"]["joint"] > rows["linear, different inits"]["joint"]
    assert sweep[0][1]["gen"] > sweep[-1][1]["gen"]          # fine-tune forgot the generic task
    assert sweep[-1][1]["A"] > sweep[0][1]["A"]              # and learned task A
    assert cat_err < 1e-10 and Ac.shape[0] == 8
    assert lin_err > 1e-3
    assert abs(svd_err - sv[4].item()) < 1e-8
    print(f"\nsmoke OK in {time.time() - t0:.1f}s: TIES joint {rows['ties d=0.5 lam=1']['joint']:.3f} "
          f"> single-task average {avg_single:.3f}")


# ---------------------------------------------------------------------------
# Full mode: PEFT add_weighted_adapter + mergekit configs + lm-eval commands
# ---------------------------------------------------------------------------

TIES_YAML = """merge_method: ties
base_model: {base}
models:
  - model: {math}
    parameters:
      weight: 1.0
      density: 0.5
  - model: {code}
    parameters:
      weight: 1.0
      density: 0.5
parameters:
  normalize: true
  int8_mask: true
dtype: {dtype}
tokenizer:
  source: base
"""

RECOVERY_YAML = """merge_method: linear
models:
  - model: {domain}
    parameters:
      weight: {w}
  - model: {instruct}
    parameters:
      weight: {w_inst}
dtype: {dtype}
tokenizer:
  source: base
"""


def full(args):
    os.makedirs(args.out, exist_ok=True)

    if args.math_adapter and args.code_adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def load_base():
            # Fresh base each time: merge_and_unload folds the active adapter into it.
            base = AutoModelForCausalLM.from_pretrained(args.base, dtype=getattr(torch, args.dtype))
            if args.resize_vocab:
                # Adapters trained after adding tokens carry a larger embedding table.
                base.resize_token_embeddings(args.resize_vocab)
            return base

        tok = AutoTokenizer.from_pretrained(args.math_adapter if args.resize_vocab else args.base)
        recipes = {
            "linear": dict(combination_type="linear"),
            "cat": dict(combination_type="cat"),
            "svd": dict(combination_type="svd", svd_rank=args.svd_rank),
            "ties": dict(combination_type="ties", density=args.density),
            "dare_ties": dict(combination_type="dare_ties", density=args.density),
        }
        for name, kwargs in recipes.items():
            model = PeftModel.from_pretrained(load_base(), args.math_adapter, adapter_name="math")
            model.load_adapter(args.code_adapter, adapter_name="code")
            model.add_weighted_adapter(["math", "code"], [args.w_math, args.w_code],
                                       adapter_name=f"merge_{name}", **kwargs)
            model.set_adapter(f"merge_{name}")
            merged = model.merge_and_unload()
            path = os.path.join(args.out, f"peft-{name}")
            merged.save_pretrained(path)
            tok.save_pretrained(path)
            print("wrote", path)

        # mergekit: TIES over merged full checkpoints of each single-task model.
        for task, adapter in (("math", args.math_adapter), ("code", args.code_adapter)):
            single = PeftModel.from_pretrained(load_base(), adapter).merge_and_unload()
            single.save_pretrained(os.path.join(args.out, f"single-{task}"))
            tok.save_pretrained(os.path.join(args.out, f"single-{task}"))
        with open(os.path.join(args.out, "ties.yml"), "w") as f:
            f.write(TIES_YAML.format(base=args.base, math=os.path.join(args.out, "single-math"),
                                     code=os.path.join(args.out, "single-code"), dtype=args.dtype))
        print(f"mergekit-yaml {args.out}/ties.yml {args.out}/mk-ties --lazy-unpickle")

    # L18.2: interpolate a domain fine-tune with the original instruct model.
    if args.domain and args.instruct:
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            cfg = os.path.join(args.out, f"recover-{w:.1f}.yml")
            with open(cfg, "w") as f:
                f.write(RECOVERY_YAML.format(domain=args.domain, instruct=args.instruct,
                                             w=w, w_inst=round(1 - w, 2), dtype=args.dtype))
            print(f"mergekit-yaml {cfg} {args.out}/recover-{w:.1f} --lazy-unpickle")

    print("\nScore every candidate (base, singles, merges, recovery sweep):")
    print(f"lm_eval --model hf --model_args pretrained=<path>,dtype={args.dtype} "
          "--tasks gsm8k,mbpp,ifeval --confirm_run_unsafe_code --batch_size auto "
          "--output_path <path>/eval")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU toy run (default without a GPU)")
    mode.add_argument("--full", action="store_true", help="GPU run with PEFT, mergekit, lm-eval")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--math-adapter")
    ap.add_argument("--code-adapter")
    ap.add_argument("--domain", help="full domain fine-tune for the L18.2 sweep")
    ap.add_argument("--instruct", help="original instruct model for the L18.2 sweep")
    ap.add_argument("--w-math", type=float, default=1.0)
    ap.add_argument("--w-code", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--svd-rank", type=int, default=None)
    ap.add_argument("--resize-vocab", type=int, default=None,
                    help="resize base embeddings before loading adapters (32005 for the TinyLlama ones)")
    ap.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16",
                    help="float16 on T4 (sm_75 has no bf16)")
    ap.add_argument("--out", default="./merges")
    args = ap.parse_args()
    if args.full:
        adapters = bool(args.math_adapter and args.code_adapter)
        recovery = bool(args.domain and args.instruct)
        if not (adapters or recovery):
            ap.error("--full needs --math-adapter and --code-adapter, "
                     "or --domain and --instruct, or both")
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
