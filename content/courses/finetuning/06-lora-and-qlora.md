# 06 — LoRA and QLoRA in Depth

> **Prerequisites:** modules 03, 05.
> **You will learn:** LoRA's math and every knob, the evidence on when it matches full
> fine-tuning, QLoRA's NF4 machinery, initialisation variants, and the PEFT
> adapter lifecycle from training to merge.

LoRA and QLoRA are how most practitioners fine-tune. The
[notes](../../notes/deep-learning/transfer-learning-and-finetuning.md) give the update
$W' = W_0 + \frac{\alpha}{r}BA$ and count one $4096 \times 4096$ matrix at $r = 16$:
131,072 trainable parameters against 16.8M. This module does the rest: the whole-model
accounting, every knob and how the knobs interact, the 2024–25 evidence on when LoRA
matches full fine-tuning and when it does not, the quantisation machinery under QLoRA,
and the PEFT lifecycle from `get_peft_model` to a merged checkpoint you can ship.

Notation follows the course symbol table. In this module $B$ means the LoRA
up-projection, never batch size; where a batch appears it is written out.

---

## 6.1 LoRA from first principles

### The low-rank update

Full fine-tuning learns a dense update $\Delta W \in \mathbb{R}^{d \times k}$ for every
weight matrix. Hu et al. (2021) [LoRA](https://huggingface.co/papers/2106.09685)
hypothesise that the update needed for adaptation has low intrinsic rank, and constrain
it to be a product of two thin matrices:

$$
W' = W_0 + \gamma_r B A, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r},\; \gamma_r = \frac{\alpha}{r},\; r \ll \min(d, k).
$$

$W_0$ is frozen: no gradient, no optimiser state. Only $A$ and $B$ train. The forward
pass never forms $BA$:

$$
h = W_0 x + \gamma_r B (A x).
$$

$Ax$ is an $r$-dimensional bottleneck; $B$ projects it back up. For a sequence of $T$
tokens the shapes are $x: (T, k)$, $Ax: (T, r)$, $h: (T, d)$.

**The low-rank constraint is an inductive bias, not a theorem**; section 6.3 is about
when it holds.

### Why B starts at zero

PEFT's default initialisation sets $A$ to Kaiming-uniform (the `nn.Linear` default) and
$B$ to zero ([LoraConfig docs, PEFT v0.21](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora)).
With $B = 0$, $\gamma_r BA = 0$, so **at step 0 the adapted model is exactly the
pretrained model.** To see that it still learns, write $g = \partial \mathcal{L} / \partial h$
for one token. From $h = W_0 x + \gamma_r B(Ax)$:

$$
\frac{\partial \mathcal{L}}{\partial B} = \gamma_r\, g\, (Ax)^\top, \qquad
\frac{\partial \mathcal{L}}{\partial A} = \gamma_r\, B^\top g\, x^\top .
$$

At step 0, $B = 0$, so $\partial \mathcal{L}/\partial A = 0$. But $A$ is random, so
$Ax \neq 0$ and $\partial \mathcal{L}/\partial B \neq 0$. **$B$ moves on the first step
and $A$ starts moving on the second**, once $B \neq 0$. The asymmetry is the point: one
factor carries the random features, the other starts at zero and reads them.

The two alternatives both fail:

| Initialisation | Step-0 function | Step-0 gradients | Outcome |
|---|---|---|---|
| $A$ random, $B = 0$ (default) | exactly $W_0$ | $\nabla_B \neq 0$, $\nabla_A = 0$ | clean start, learning begins immediately |
| both random | $W_0 + \gamma_r BA$, a random perturbation | both non-zero | starts from a damaged model; the noise must be unlearned first |
| both zero | exactly $W_0$ | both zero | a saddle point: no gradient ever flows, nothing trains |

PEFT exposes "both random" as `init_lora_weights=False`, for debugging only. The check
below builds a LoRA layer by hand (`nn.Linear(16, 12)`, $r = 4$, $\alpha = 8$) and
asserts the step-0 identity, the two gradients, and that merge and unmerge are exact.

```python finetuning-check
import math
import torch
torch.manual_seed(0)
d_in, d_out, r, alpha = 16, 12, 4, 8
gamma = alpha / r                                    # 2.0
base = torch.nn.Linear(d_in, d_out, dtype=torch.float64)   # f64: merged vs unmerged sum orders differ
A = torch.empty(r, d_in, dtype=torch.float64)
torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))    # PEFT's default for A
A.requires_grad_()
B = torch.zeros(d_out, r, dtype=torch.float64, requires_grad=True)  # zero-init up-projection

def lora_forward(x, B):                              # x: (T, d_in)
    return base(x) + gamma * (x @ A.T) @ B.T         # (T, d_out)

x = torch.randn(5, d_in, dtype=torch.float64)
assert torch.equal(lora_forward(x, B), base(x))      # exact no-op at step 0

target = torch.randn(5, d_out, dtype=torch.float64)
loss = ((lora_forward(x, B) - target) ** 2).sum()
loss.backward()
g = 2 * (base(x) - target)                           # dL/dh, (T, d_out)
assert torch.count_nonzero(A.grad) == 0              # gamma * B^T g x^T with B = 0
assert torch.count_nonzero(B.grad) > 0
torch.testing.assert_close(B.grad, gamma * g.T @ (x @ A.T).detach(), atol=1e-10, rtol=0)

# One SGD step from B = 0 gives delta W = -lr * gamma^2 * g^T x A^T A: it scales as gamma^2.
lr = 1e-3
def first_step_delta(gamma_):
    grad_b = gamma_ * g.T @ (x @ A.T).detach()       # (d_out, r)
    return gamma_ * (-lr * grad_b) @ A.detach()      # (d_out, d_in)
torch.testing.assert_close(first_step_delta(2 * gamma), 4 * first_step_delta(gamma))

# Merge equivalence once B is non-zero, then unmerge.
with torch.no_grad():
    B.copy_(torch.randn(d_out, r))
    W0 = base.weight.clone()
    unmerged = lora_forward(x, B)
    base.weight += gamma * B @ A                     # merge
    torch.testing.assert_close(base(x), unmerged, atol=1e-12, rtol=0)
    base.weight -= gamma * B @ A                     # unmerge
    torch.testing.assert_close(base.weight, W0, atol=1e-12, rtol=0)
print("step-0 identity, step-0 gradients, gamma^2 first step and merge/unmerge verified")
```

This proves the algebra for one fp32 layer. It does not prove that merging is exact in
bf16 or into a quantised base; section 6.7 shows where that breaks.

### Parameter count for a whole model

A LoRA pair on a $d_{\text{in}} \to d_{\text{out}}$ matrix costs
$r(d_{\text{in}} + d_{\text{out}})$ parameters; sum over the targeted matrices. Take
Llama 3 8B ([config.json at revision 8cde5ca](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/config.json),
gated; the same values are in the [Llama 3 paper](https://arxiv.org/abs/2407.21783)'s
architecture table): `hidden_size` 4096, `intermediate_size` 14336, 32 layers, 8 KV
heads of dimension 128 (so K and V output 1024), 8,030,261,248 parameters.

**Table 1. LoRA parameters per layer for Llama 3 8B with every linear layer adapted.**

| Matrix | $d_{\text{in}} \to d_{\text{out}}$ | $d_{\text{in}} + d_{\text{out}}$ | Parameters |
|---|---|---|---|
| `q_proj` | 4096 → 4096 | 8,192 | 8,192 r |
| `k_proj` | 4096 → 1024 | 5,120 | 5,120 r |
| `v_proj` | 4096 → 1024 | 5,120 | 5,120 r |
| `o_proj` | 4096 → 4096 | 8,192 | 8,192 r |
| `gate_proj` | 4096 → 14336 | 18,432 | 18,432 r |
| `up_proj` | 4096 → 14336 | 18,432 | 18,432 r |
| `down_proj` | 14336 → 4096 | 18,432 | 18,432 r |
| **per layer** | | **81,920** | **81,920 r** |
| × 32 layers | | | 2,621,440 r |
| **r = 16, all-linear** | | | **41,943,040** (0.52% of 8.03B) |
| r = 16, attention only (q, k, v, o) | | 26,624 per layer | 13,631,488 |

**The MLP holds 55,296r of the 81,920r LoRA parameters per layer (67.5%, about two
thirds) and 81% of the base linear weights** (176.2M of 218.1M). GQA shrinks attention
in the base, so the MLP's share of the base exceeds its share of the adapter
([transformers module 07](../transformers/07-ffn-and-activations.md) explains the MLP's
size). The check below recomputes the table and both shares, then builds real LoRA
modules for a scaled-down config ($d = 64$, $d_{kv} = 16$, $d_{ff} = 224$, $L = 2$,
$r = 4$) and compares their tensor sizes with the formula.

```python finetuning-check
import math
import torch
torch.manual_seed(0)

def shapes_for(d, d_kv, d_ff):                       # (d_in, d_out) of the 7 linears
    return {"q_proj": (d, d), "k_proj": (d, d_kv), "v_proj": (d, d_kv), "o_proj": (d, d),
            "gate_proj": (d, d_ff), "up_proj": (d, d_ff), "down_proj": (d_ff, d)}

llama = shapes_for(4096, 8 * 128, 14336)             # Llama 3 8B
per_layer = sum(i + o for i, o in llama.values())
attn = sum(i + o for k, (i, o) in llama.items() if k in ("q_proj", "k_proj", "v_proj", "o_proj"))
L, r = 32, 16
assert per_layer == 81_920 and attn == 26_624
assert per_layer * L == 2_621_440
assert per_layer * L * r == 41_943_040
assert attn * L * r == 13_631_488
assert round(100 * per_layer * L * r / 8.03e9, 2) == 0.52
mlp_lora = per_layer - attn                          # MLP share of the LoRA parameters
mlp_base = sum(i * o for k, (i, o) in llama.items()
               if k in ("gate_proj", "up_proj", "down_proj"))
base = sum(i * o for i, o in llama.values())         # base linear weights per layer
assert mlp_lora == 55_296 and mlp_lora / per_layer == 0.675
assert mlp_base == 176_160_768 and base == 218_103_808
assert round(mlp_base / base, 2) == 0.81             # not two thirds: GQA shrinks attention

class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scaling = base, alpha / r
        base.weight.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))   # (r, d_in)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))  # (d_out, r)
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T) @ self.B.T

small = shapes_for(64, 16, 224)
layers = [LoRALinear(torch.nn.Linear(i, o, bias=False), r=4, alpha=8)
          for _ in range(2) for i, o in small.values()]
trainable = sum(p.numel() for m in layers for p in m.parameters() if p.requires_grad)
assert trainable == 4 * 2 * sum(i + o for i, o in small.values()) == 10_240
print("Table 1 and the module-level count verified")
```

This proves the counting and the two MLP shares. It does not prove PEFT targets the
same modules: read `print_trainable_parameters()` and `targeted_module_names`.

AdamW's two fp32 moments for 42M adapter parameters take about 0.34 GB, against
roughly 64 GB for all 8B parameters. Module 03 does the full budget (the
[~16–18 bytes/param rule](./03-training-mechanics-and-memory-math.md)); under LoRA the
frozen weights and activations dominate.

### Merged vs unmerged inference

**Merging** computes $W_0 + \gamma_r BA$ once; the result is architecturally the base
model and costs nothing extra. Unmerged, every adapted layer runs two extra thin matmuls:

$$
\text{FLOPs per token} = \underbrace{2\, d_{\text{in}} d_{\text{out}}}_{\text{base}} + \underbrace{2 r (d_{\text{in}} + d_{\text{out}})}_{\text{adapter}} .
$$

For `q_proj` at $r = 16$: 33.6M base FLOPs plus 262K adapter FLOPs, under 1%. The real
cost is latency from extra kernel launches and small memory-bound matmuls. **Serve
unmerged when one base carries many adapters**: hundreds of customers on one GPU, with
batched kernels such as Punica's SGMV and S-LoRA (module
[20](./20-production-serving-and-cost.md); the inference course covers vLLM's side in
[LoRA and pooling models](/courses/inference/07-architectures/07-05-lora-and-pooling-models.html)).

---

## 6.2 The knobs

### Rank and alpha

**Rank** $r$ sets capacity, the number of independent directions each matrix can move
in (typically 8 to 128). **Alpha** sets $\gamma_r = \alpha/r$. The
[Unsloth guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
recommends $\alpha = r$ or $2r$; Thinking Machines fix $\alpha = 32$ for every rank. As
of October 2026 the PEFT defaults are `r=8`, `lora_alpha=8`
([config.py, v0.21.2](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/config.py#L632-L653)).

### Alpha and the learning rate are coupled

Alpha is not a knob separate from the learning rate. Take one SGD step from $B = 0$
with learning rate $\eta$: $B_1 = -\eta \gamma_r\, g (Ax)^\top$, and the realised update is

$$
\Delta W_1 = \gamma_r B_1 A = -\eta\, \gamma_r^2\, g\, x^\top A^\top A .
$$

**Under SGD the first update scales as $\eta \gamma_r^2$.** Doubling alpha does the same
thing as quadrupling the learning rate. The first check in 6.1 asserted exactly this
factor of 4.

Adam changes the exponent, not the coupling. It normalises each gradient by its
running magnitude, so early updates to $B$ are about $\eta$ whatever $\gamma_r$ is, and
$\Delta W$ scales as $\eta\gamma_r$ (away from Adam's $\epsilon$ floor). **Fix alpha and
sweep the learning rate; sweeping both searches one direction twice.**

The same picture explains why LoRA wants a larger learning rate than full fine-tuning.
Adam moves each full-FT weight by about $\eta$ per step. Under LoRA it moves each entry
of $B$ by about $\eta$, and Kaiming $A$ has entries of size about $1/\sqrt{3k}$, so an
entry of $\gamma_r BA$ moves by roughly $\eta\gamma_r\sqrt{r/(3k)}$: about $0.07\eta$ at
$\alpha = 32$, $r = 16$, $k = 4096$. At the full-FT learning rate LoRA barely moves
$W$, so its optimum sits higher. The factor of about 10 (6.3) is empirical; this
order-of-magnitude argument gives only its direction.

### rsLoRA: alpha over root r

As $r$ grows with $\alpha$ fixed, $\gamma_r$ shrinks. Kalajdzievski (2023)
[rsLoRA](https://arxiv.org/abs/2312.03732) shows that dividing by $r$ "results in
slowed learning and stunted performance" at high rank, and that the scale must go as
$1/\sqrt{r}$ to keep adapter outputs and gradients stable:

$$
\gamma_r^{\text{rs}} = \frac{\alpha}{\sqrt{r}} .
$$

In PEFT this is `use_rslora=True` (default `False`). The scalings agree at $r = 1$; at
$r = 16$ rsLoRA's is four times larger. Thinking Machines (6.3) found the optimal
learning rate under $1/r$ roughly rank-independent because they re-tuned it per
configuration; rsLoRA is for raising rank *without* retuning. In the lab's smoke test,
rank 16 reaches a test MSE of 0.0019 with rsLoRA against 0.0024 with $\alpha/r$ after 20
steps at the same learning rate.

### Target modules

**Target every linear layer, including the MLP**: `target_modules="all-linear"` (all
linears except the output head), which the PEFT docs recommend. Omitting
`target_modules` is a trap: PEFT falls back to a per-architecture default, still
`["q_proj", "v_proj"]` for Llama and Qwen3 ([constants.py,
v0.21.2](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/utils/constants.py#L85-L104)),
a relic of the original paper's GPT-3 experiments. Thinking Machines find that
"attention-only with rank 256 underperforms MLP-only with rank 128", so **at matched
parameter count the MLP is where LoRA's capacity pays**. Table 1 gives the reason: about
80% of each layer's base weights live there.

### Dropout, bias, extra modules, per-layer overrides

- **`lora_dropout`** drops the adapter's input only; the Unsloth guide calls it "not
  that useful" for short runs. Raise it to 0.05–0.1 only on visible overfitting.
- **`bias`** other than `"none"` trains base biases, so disabling the adapter no longer
  recovers the base model.
- **`modules_to_save`** trains and saves a full copy of a module: a new classification
  head (`["score"]`), or `lm_head` where trainers forbid an adapter on it (6.6).
- **`rank_pattern` / `alpha_pattern`** map module names or regexes to per-layer ranks
  and alphas, e.g. `{"^model.layers.0.self_attn.q_proj": 64}`; MoE experts (6.6) use them.

**Table 2. The LoRA knobs (as of October 2026).**

| Knob | Default: PEFT / Unsloth | Typical range | Effect | Evidence |
|---|---|---|---|---|
| `r` | 8 / guide suggests 16 or 32 | 8–128; 1 suffices for RL | capacity per matrix | Thinking Machines; Biderman et al. |
| `lora_alpha` | 8 / `r` or `2r` | $r$, $2r$, or fixed 32 | sets $\gamma_r$; coupled to LR | 6.2 derivation |
| `use_rslora` | `False` / `False` | `True` at $r \ge 64$ | $\gamma_r = \alpha/\sqrt r$ | Kalajdzievski (2023) |
| `target_modules` | arch default (q, v for Llama/Qwen3) / all 7 projections | `"all-linear"` | where capacity goes | Thinking Machines; PEFT docs |
| learning rate | n/a / 2e-4 | ≈ 10× the full-FT LR | dominant knob | Thinking Machines |
| LR schedule, warmup | n/a (Trainer: `linear`, no warmup) | same warmup and linear/cosine decay as full FT | only the peak LR changes | [module 03](./03-training-mechanics-and-memory-math.md) |
| `lora_dropout` | 0.0 / 0 | 0–0.1 | adapter-input regularisation | Unsloth guide |
| `bias` | `"none"` / `"none"` | `"none"` | trains base biases if not none | PEFT docs |
| `modules_to_save` | `None` | heads, `lm_head`, `embed_tokens` | full trainable copy | PEFT docs |
| `rank_pattern`, `alpha_pattern` | `{}` | regex → int | per-layer overrides | PEFT docs |
| `init_lora_weights` | `True` (Kaiming A, zero B) | see Table 4 | starting point | 6.5 |

---

## 6.3 What the evidence says

### LoRA Without Regret (Thinking Machines, 2025)

Schulman and colleagues ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/),
September 2025) compared LoRA with full fine-tuning (FullFT) in SFT and RL post-training,
sweeping the learning rate per configuration, with $\alpha = 32$, $1/r$ scaling and
$B = 0$ init. Their findings:

- **All layers, especially the MLP** ("including MoE"); attention on top of the MLP
  adds nothing.
- **Same sample efficiency as FullFT** while the dataset fits the adapter's capacity.
- **Learning rate ≈ 10× FullFT**, "consistently", and roughly rank-independent under $1/r$.
- **Rank 1 suffices for RL**: an advantage carries "only O(1) bits per episode". This is
  why LoRA RL is cheap ([module 13](./13-rl-with-verifiable-rewards.md)) and why Tinker
  is a LoRA-only training API ([module 08](./08-frameworks-and-tooling.md)).
- **Large batches hurt LoRA more**, and the gap does not depend on rank: shrink the
  batch, do not raise $r$.
- **Compute is about ⅔ of FullFT**: forward ($2N$) and activation gradients ($2N$)
  remain, frozen weights skip weight gradients ($2N$), so $4N/6N$. Module
  [20](./20-production-serving-and-cost.md) turns this into cost.

### LoRA Learns Less and Forgets Less (Biderman et al., 2024)

Biderman et al. (2024) [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
trained Llama-2-7B on code and maths with instruction fine-tuning (about 100K pairs) and
continued pretraining (up to 20B tokens). In standard low-rank configurations "LoRA
substantially underperforms full finetuning" on the target domain, most visibly in CPT,
but better preserves out-of-domain ability, more than weight decay or dropout do. Full
fine-tuning learns perturbations of 10–100× higher rank.

The two papers are consistent. **Biderman et al. are capacity-bound**: billions of
tokens do not fit a rank-16 update. **Thinking Machines are mostly in post-training
regimes**, and name the limit themselves ("for datasets that exceed LoRA capacity, LoRA
underperforms FullFT"). Forgetting less is the same constraint from the other side: an
update that cannot move far cannot destroy much
([module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)).

### Capacity limits: a decision rule

Decide by how much information the data carries relative to adapter capacity:

| Situation | Information to absorb | Choice |
|---|---|---|
| RL with verifiable rewards | ~1 bit per episode | LoRA, any rank (even 1) |
| Format, style, tool-call schema, extraction (the course's support-email task) | small | LoRA r = 8–32, all-linear |
| Large SFT (hundreds of thousands of long examples), new skills | medium to large | LoRA r = 64–256 with rsLoRA, or full FT; compare on eval |
| Continued pretraining on billions of tokens, new language | large | full FT (with replay), or LoRA plus trainable embeddings, accepting a gap |
| You must preserve general ability above all | n/a | LoRA; it forgets less |

**Raise rank when training loss plateaus above what full fine-tuning reaches on the
same data**; that is exhausted capacity. Rank does nothing for a too-large batch or a
bad learning rate. Module [05](./05-supervised-fine-tuning.md) covers continued
pretraining, including separate learning rates for `embed_tokens` and `lm_head`.

**Table 3. What each claim rests on.**

| Claim | Paper | Setting | What it does NOT show |
|---|---|---|---|
| LoRA matches FullFT | Thinking Machines (2025) | SFT and RL post-training, LR swept per config, all-layer LoRA | that LoRA matches on CPT-scale data or with very large batches |
| Optimal LR ≈ 10× FullFT, roughly rank-independent | Thinking Machines (2025) | $\alpha = 32$, $1/r$ scaling, Adam | that the ratio holds under rsLoRA or other inits |
| Rank 1 suffices for RL | Thinking Machines (2025) | policy-gradient RL on reasoning tasks | that rank 1 suffices for SFT |
| LoRA learns less, forgets less | Biderman et al. (2024) | Llama-2-7B, code/maths, IFT and CPT | that LoRA underperforms in small post-training regimes |
| FullFT updates have 10–100× higher rank | Biderman et al. (2024) | spectral analysis of FullFT deltas | that the extra rank is necessary for the task |
| Rank-stable scaling helps high ranks | Kalajdzievski (2023) | fixed LR across ranks | that rsLoRA helps when LR is re-tuned per rank |
| LoRA introduces intruder dimensions | Shuttleworth et al. (2024) | SVD of LoRA vs FullFT weights | that intruder dimensions hurt the target task |

---

## 6.4 QLoRA

Dettmers et al. (2023) [QLoRA](https://arxiv.org/abs/2305.14314) keeps the frozen base
in 4 bits and trains LoRA adapters on top, to "finetune a 65B parameter model on a
single 48GB GPU" while preserving 16-bit task performance. Three ingredients make it
work: NF4 storage, double quantisation and paged optimisers.

### NF4

Pretrained weights are roughly zero-mean normal. A uniform 4-bit grid wastes levels in
the sparse tails and starves the dense centre. **NF4 (4-bit NormalFloat) places its 16
levels at quantiles of $\mathcal{N}(0,1)$, so each level covers equal probability
mass.** The levels are normalised to $[-1, 1]$ and asymmetric (7 negative, an exact 0,
8 positive) so that zero is exact.

Quantisation is block-wise. Each block of 64 weights stores one fp32 **absmax**
constant $c = \max_i |w_i|$; each weight stores the 4-bit index of the level nearest
$w_i / c$. Dequantisation is a table lookup times $c$. Small blocks keep one outlier
from crushing the resolution of all the other weights. The check below carries the 16
values verbatim from bitsandbytes
([`get_4bit_type`, functional.py at 8336490](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L788-L805)).

### Double quantisation

The absmax constants cost $32/64 = 0.5$ bits per weight. **Double quantisation (DQ)
quantises the constants themselves**: blocks of 256 absmaxes become 8-bit floats with
one fp32 second-level constant each (after subtracting the mean absmax,
[L938-L940](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py#L938-L940)).

$$
\underbrace{4 + \tfrac{32}{64}}_{\text{no DQ}} = 4.5 \text{ bits}
\quad\longrightarrow\quad
\underbrace{4 + \tfrac{8}{64} + \tfrac{32}{64 \cdot 256}}_{\text{DQ}} = 4.126953125 \text{ bits},
$$

a saving of 0.373 bits per parameter, matching the paper: 4.5 GB without DQ and 4.127 GB
with it for 8B parameters, against 16 GB in bf16 (embeddings, norms and usually
`lm_head` stay 16-bit, so real footprints are higher). The check below quantises 4,096
normal weights block-wise onto NF4 and onto a uniform 16-level grid, asserts NF4's MSE
is more than 20% lower, then asserts the bit and memory arithmetic.

```python finetuning-check
import torch
torch.manual_seed(0)
NF4 = torch.tensor([                                 # bitsandbytes get_4bit_type("nf4")
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
UNIFORM = torch.linspace(-1.0, 1.0, 16)              # evenly spaced, no exact zero

def block_quant(w, levels, block=64):
    flat = w.reshape(-1, block)                      # (n_blocks, 64)
    absmax = flat.abs().amax(dim=1, keepdim=True)    # one fp32 scale per block
    idx = ((flat / absmax).unsqueeze(-1) - levels).abs().argmin(-1)   # 4-bit codes
    return (levels[idx] * absmax).reshape(w.shape)

w = torch.randn(4096)                                # normal "weights"
mse_nf4 = ((block_quant(w, NF4) - w) ** 2).mean().item()
mse_uni = ((block_quant(w, UNIFORM) - w) ** 2).mean().item()
assert len(NF4) == 16 and int((NF4 == 0).sum()) == 1 and not bool((UNIFORM == 0).any())
assert NF4[0] == -1.0 and NF4[-1] == 1.0
assert int((NF4 < 0).sum()) == 7 and int((NF4 > 0).sum()) == 8
assert mse_nf4 < 0.8 * mse_uni                       # quantile levels win on normal data

bits_plain = 4 + 32 / 64                             # fp32 absmax per 64 weights
bits_dq = 4 + 8 / 64 + 32 / (64 * 256)               # 8-bit absmax + fp32 per 256 absmaxes
assert bits_plain == 4.5 and bits_dq == 4.126953125
assert abs((bits_plain - bits_dq) - 0.373) < 1e-3
gigabytes = lambda n, bits: n * bits / 8 / 1e9
assert gigabytes(8e9, bits_plain) == 4.5
assert abs(gigabytes(8e9, bits_dq) - 4.127) < 1e-3
assert gigabytes(8e9, 16) == 16.0
print("NF4 beats uniform 4-bit on normal weights; 4.5 and 4.127 bits/param verified")
```

This proves NF4's advantage on synthetic normal data with one seed (MSE about 0.0083
against 0.0109). It does not prove that NF4 is optimal for real weight
distributions with heavy-tailed outliers. Module
[11](./11-quantization-qat-and-fp8.md) covers the alternatives.

### Paged optimisers and compute dtype

**Paged optimisers** keep optimiser states in NVIDIA unified memory, so a long-sequence
spike evicts them to CPU RAM instead of crashing (`optim="paged_adamw_8bit"`;
[module 03](./03-training-mechanics-and-memory-math.md)).

"QLoRA trains a 4-bit model" is wrong. **The 4-bit weights are storage only**: each
matmul dequantises the NF4 blocks to `bnb_4bit_compute_dtype` (usually bf16) and
multiplies there. Gradients flow *through* the dequantised base, but only the bf16 or
fp32 adapters get weight gradients; the base is never re-quantised during training.
Teaching fragment (not executed; Transformers v5, bitsandbytes):

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # or "fp4"
    bnb_4bit_use_double_quant=True,       # DQ: 4.127 instead of 4.5 bits/param
    bnb_4bit_compute_dtype=torch.bfloat16,  # T4 has no bf16: use torch.float16
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base", quantization_config=bnb)
```

The same config covers **FP4** (a fixed 1-2-1 float grid that NF4 beat in the QLoRA
paper) and **LLM.int8** (`load_in_8bit=True`: int8 weights, with outlier dimensions above
`llm_int8_threshold=6.0` split into a 16-bit matmul); int8 saves less and runs slower.

**Adapters stay in higher precision** because each update is tiny relative to the
weights; at 4 bits most of it would round to zero. Module
[11](./11-quantization-qat-and-fp8.md) owns quantisation theory, QAT and export; the
inference course covers [formats and kernels](/courses/inference/04-quantization/04-03-formats-and-kernels.html).

---

## 6.5 Initialisation variants

The default starts from a random subspace; these methods pick it more carefully. As of
October 2026 all ship in PEFT under `init_lora_weights` ([LoRA reference, v0.21](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora)).
(SVD and QR recap: [linear algebra note](../../notes/math/linear-algebra.md).)

**Table 4. LoRA initialisations in PEFT v0.21.**

| Method | Idea | Data needed? | Cost | PEFT argument |
|---|---|---|---|---|
| Default | Kaiming-uniform $A$, $B = 0$ | no | none | `init_lora_weights=True` |
| Gaussian | $A \sim \mathcal{N}(0, 1/r^2)$, $B = 0$ | no | none | `"gaussian"` |
| [PiSSA](https://arxiv.org/abs/2404.02948) | $A, B$ from the top-$r$ singular vectors of $W$; the frozen part is the residual $W - \gamma_r BA$; faster convergence, smaller error under quantisation | no | one SVD per matrix (minutes); fast SVD in seconds | `"pissa"`, `"pissa_niter_16"` |
| [OLoRA](https://arxiv.org/abs/2406.01775) | QR decomposition of $W$ gives an orthonormal starting basis; base is translated accordingly | no | one QR per matrix | `"olora"` |
| [EVA](https://arxiv.org/abs/2410.07170) | SVD of each layer's *input activations*; right singular vectors initialise $A$; ranks redistributed across layers by explained variance | yes (a few batches) | forward passes + incremental SVD | `"eva"` + `EvaConfig(rho=2.0)`, then `initialize_lora_eva_weights` |
| [CorDA](https://arxiv.org/abs/2406.05223) | SVD of $W$ multiplied by an activation covariance matrix; knowledge-preserved mode (`"kpm"`) adapts the least-used directions, instruction-previewed mode (`"ipm"`) the most task-relevant | yes | covariance collection + SVD | `"corda"` + `CordaConfig`, `preprocess_corda` |
| [LoftQ](https://arxiv.org/abs/2310.08659) | alternately quantise $W$ and fit a rank-$r$ approximation of the residual so that $Q + \gamma_r BA \approx W$ | no | several quantise/SVD rounds | `"loftq"` + `LoftQConfig` (pass an unquantised model) |
| [LoRA-GA](https://arxiv.org/abs/2407.05000) | SVD of estimated full-FT gradients so the first LoRA step aligns with the first full-FT step | yes (64–128 batches) | gradient estimation | `"lora_ga"` + `LoraGAConfig`, `preprocess_loraga` (v0.19+) |
| [MiCA](https://arxiv.org/abs/2604.01694) | $B$ from the *minor* left singular vectors of $W$ (smallest singular values), $A = 0$, only $A$ trains; aimed at continued pretraining | no | one SVD | `"mica"` (v0.20+) |

PiSSA, OLoRA and MiCA use only the weights. EVA, CorDA and LoRA-GA need a calibration
pass over data. LoftQ targets QLoRA: quantised base plus adapter starts closer to $W$.
The data-driven inits add a call after `get_peft_model`. EVA, for example:

```python
# Teaching fragment (not executed; PEFT 0.21).
from peft import EvaConfig, LoraConfig, get_peft_model, initialize_lora_eva_weights
config = LoraConfig(r=16, target_modules="all-linear",
                    init_lora_weights="eva", eva_config=EvaConfig(rho=2.0))
peft_model = get_peft_model(model, config, low_cpu_mem_usage=True)   # adapters not yet set
initialize_lora_eva_weights(peft_model, dataloader)   # SVD of activations, redistributes ranks
```

### Converting mutated inits back

PiSSA, OLoRA, CorDA and LoRA-GA rewrite the frozen part to
$W_0 - \gamma_r B_{\text{init}} A_{\text{init}}$, so the adapter only works on that
modified base. PEFT converts it at save time:

```python
# Teaching fragment (not executed; PEFT 0.21).
peft_model.save_pretrained("init")                       # before any training step
# ... train ...
peft_model.save_pretrained("final", path_initial_model_for_weight_conversion="init")
```

PEFT's `LoraModel.subtract_mutated_init` computes $\Delta W = BA - B_0A_0$ and stores
it as a standard rank-$2r$ LoRA against the *original* base, which loads, combines and
toggles like any other adapter. Conversion is unsupported when rsLoRA is combined with
`rank_pattern` or `alpha_pattern` ([peft_model.py, v0.21.2](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/peft_model.py#L268-L277)).

---

## 6.6 Special targets: MoE experts, embeddings, LM head

### MoE experts

Many Transformers MoE implementations store all experts as one 3D `nn.Parameter`
$(E, d_{\text{out}}, d_{\text{in}})$, invisible to `target_modules`. PEFT 0.17's
**`target_parameters`** attaches LoRA to a parameter by name, dimension 0 being experts:

```python
# Teaching fragment (not executed; PEFT 0.21, Qwen3-MoE naming).
config = LoraConfig(
    r=32, lora_alpha=32, target_modules="all-linear",
    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    # r / active experts (8 of 128; Thinking Machines). PEFT docs use max(1, r // num_experts).
    rank_pattern={"experts.gate_up_proj": 4, "experts.down_proj": 4},
    alpha_pattern={"experts.gate_up_proj": 4, "experts.down_proj": 4},  # keep gamma_r = 1
)
```

One adapter per expert costs $E \cdot r (d_{\text{in}} + d_{\text{out}})$ per matrix:
128 experts at rank 32 cost as much as one matrix at rank 4,096. The PEFT docs divide
the rank by the total expert count (`max(1, r // num_experts)`, which is 1 for
Qwen3-MoE's 128 experts); Thinking Machines divide it by the number of *active* experts
(8), which the fragment follows. Without `alpha_pattern`, $\gamma_r$ on the experts
would jump to $32/4 = 8$. PEFT computes the LoRA term for every expert even when
few are routed, which slows KV-cached decoding, so it **recommends merging before
serving** ([Targeting `nn.Parameter` directly](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora)).
Module [10](./10-distributed-fine-tuning-and-moe.md) covers expert parallelism, ScatterMoE
kernels and FSDP with QLoRA; [transformers module 12](../transformers/12-mixture-of-experts.md)
covers the router.

### New tokens and tied embeddings

New special tokens add untrained embedding rows
([module 02](./02-tokens-templates-and-the-causal-lm-objective.md)).
`modules_to_save=["embed_tokens"]` trains the whole matrix: 622M parameters for a 152K
vocabulary at width 4096. **`trainable_token_indices=[...]`** (PEFT 0.15+) trains and
saves only the listed rows: 24,576 parameters for 6 new tokens.

Small Qwen3 sizes **tie** `lm_head.weight` to `embed_tokens.weight`
(`tie_word_embeddings=True`; Llama 3 8B does not). **`ensure_weight_tying=True`** (PEFT
0.18+, trainable tokens 0.19+) keeps adapter-side updates to the two tied.

### The LM head in TRL trainers

As of TRL v1.15, the SFT, DPO, KTO, GRPO, RLOO and Distillation trainers multiply
hidden states by `lm_head.weight` directly (the fused LM head), so an adapter on
`lm_head` would get no gradient and they raise `ValueError`. **To train the head, use
`modules_to_save=["lm_head"]`**, which replaces the weight itself
([DPO trainer docs, v1.15.0](https://huggingface.co/docs/trl/v1.15.0/en/dpo_trainer);
[source](https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/dpo_trainer.py)).

---

## 6.7 The adapter lifecycle in PEFT

### Train, save, load

Teaching fragment (not executed; PEFT 0.21, Transformers v5):

```python
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base", dtype=torch.bfloat16)
config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(base, config)          # wraps base in a PeftModel, freezes W0
model.print_trainable_parameters()            # read this every time
# ... train with TRL's SFTTrainer (module 05), or pass peft_config= to it ...
model.save_pretrained("support-extractor")    # adapter_config.json + adapter_model.safetensors

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "support-extractor")   # inference, frozen adapter
```

The saved directory holds only the adapter: `adapter_config.json` (base model, `r`,
`lora_alpha`, targets) and `adapter_model.safetensors`. For the rank-16 Llama 3 8B
adapter in Table 1 that file is about 168 MB in fp32, because `get_peft_model` upcasts
adapter weights to fp32 by default (`autocast_adapter_dtype=True`), or 84 MB if you cast
the adapter to bf16 before saving; the model is 16 GB. **Use PEFT 0.18 or newer with
Transformers v5; older PEFT versions are incompatible.** PEFT 0.21 also made two silent
failures loud: adding an adapter that targets non-existent layers now raises, and so
does deleting an adapter that is currently merged.

### Merging, and the 4-bit trap

PEFT has two merge APIs with different semantics
([merge docs, v0.21](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora)):

- **`merge_and_unload()`** folds the adapter into the wrapped base model and *returns*
  that base model with the LoRA layers removed. Use the return value:
  `model = model.merge_and_unload()`, then `model.save_pretrained(...)` saves a full
  checkpoint; do not keep using the old `PeftModel`.
- **`merge_adapter()` / `unmerge_adapter()`** fold and unfold in place on the
  `PeftModel`. They are reversible, and are useful for fast evaluation of a merged
  model or for generation inside RL loops.

Now the trap. After QLoRA the base in memory is 4-bit, and PEFT will let you merge
into it. For a bitsandbytes 4-bit layer it dequantises the weight, adds $\Delta W$, and
**re-quantises the sum to NF4**, warning that this "may get different generations due to
rounding errors" ([bnb.py, v0.21.2](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/bnb.py#L376-L402)).
The trouble is scale. $\Delta W$ is a few percent of $W$, and NF4's level spacing is
several percent of each block's absmax. Re-quantisation rounds most of the update away
and moves weights that sit near level boundaries by a full step.

**The fix: merge into a high-precision copy, then quantise for deployment.** Load the
base in bf16 (or dequantise the NF4 weights to bf16), apply the adapter,
`merge_and_unload()`, save, then run a real quantiser (GPTQ, AWQ, GGUF;
[module 11](./11-quantization-qat-and-fp8.md)). The dequantised copy reproduces exactly
what training saw; the original bf16 base differs from it by the NF4 error. Evaluate
both if the gap matters.

The check below merges a small adapter delta ($\|\Delta W\| \approx 3\%$ of $\|W\|$)
into a $64 \times 64$ NF4 weight $Q$, against the QLoRA forward pass
$x(Q + \Delta W)^\top$. Path A merges into the dequantised base and stores bf16; path B
re-quantises to NF4. It asserts path B's error is over ten times path A's and about the
size of the adapter's whole effect, and that the surviving update points elsewhere
(cosine to $\Delta W$ below 0.9; it is about 0.14).

```python finetuning-check
import torch
torch.manual_seed(0)
NF4 = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])

def nf4(w, block=64):                                # quantise then dequantise
    flat = w.reshape(-1, block)
    absmax = flat.abs().amax(dim=1, keepdim=True)
    idx = ((flat / absmax).unsqueeze(-1) - NF4).abs().argmin(-1)
    return (NF4[idx] * absmax).reshape(w.shape)

d, r = 64, 8
W = torch.randn(d, d)                                # pretrained weight (out, in)
Q = nf4(W)                                           # frozen NF4 base QLoRA trained against
dW = 3e-2 * (torch.randn(d, r) @ torch.randn(r, d)) / r ** 0.5   # trained LoRA delta
x = torch.randn(32, d)                               # (T, d_in)
ref = x @ (Q + dW).T                                 # what the QLoRA model computes
effect = (x @ dW.T).abs().mean().item()              # size of the adapter's contribution

W_a = (Q + dW).to(torch.bfloat16)                    # path A: merge into dequantised copy, bf16
err_a = (x @ W_a.float().T - ref).abs().mean().item()
W_b = nf4(Q + dW)                                    # path B: merge, then re-quantise to NF4
err_b = (x @ W_b.T - ref).abs().mean().item()
kept = W_b - Q                                       # the update that survived
cos = torch.nn.functional.cosine_similarity(kept.flatten(), dW.flatten(), dim=0).item()

assert 0.02 < (dW.norm() / W.norm()).item() < 0.04   # a small, realistic update
assert err_b > 10 * err_a
assert err_a < 0.1 * effect and err_b > 0.5 * effect
assert cos < 0.9
print("bf16 merge preserves the QLoRA function; NF4 re-quantisation erases the adapter")
```

This proves the mechanism on one synthetic matrix. It does not measure the quality loss
on a real model; lab L6.4 does that with real bitsandbytes NF4. The gap shrinks as
$\Delta W$ grows relative to the level spacing: at 1% relative size, path B's error is
only about 6× path A's, but the surviving update is almost pure noise (cosine 0.05).

### Many adapters in one process

```python
# Teaching fragment (not executed; PEFT 0.21): several adapters in one PeftModel.
model = PeftModel.from_pretrained(base, "adapters/support-en", adapter_name="en")
model.load_adapter("adapters/support-de", adapter_name="de")
model.set_adapter("de")                         # the active adapter for later calls
out = model.generate(**batch, adapter_names=["__base__", "en", "de", "en"])  # one per row
model.delete_adapter("en")                      # raises if "en" is currently merged
```

`adapter_names` routes each batch row through a different adapter (`"__base__"` means
none). This **mixed-adapter batching** is inference-only, fails after `merge_adapter` or
with DoRA, and costs throughput. **Hotswapping** (`hotswap_adapter` in
`peft.utils.hotswap`, after `prepare_model_for_compiled_hotswap` for a compiled model)
replaces weights in place without recompiling
([hotswap docs](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/hotswap)).
Production multi-adapter serving is [module 20](./20-production-serving-and-cost.md)'s;
combining adapters (`add_weighted_adapter`) is [module 18](./18-model-merging-and-composition.md)'s.

### Activated LoRA

A "checker" adapter that judges a base model's 8,000-token answer must, with ordinary
LoRA, prefill the whole context again: its weights differ on every token. **Activated
LoRA (aLoRA)** (Greenewald et al., 2025, [arXiv 2504.12397](https://arxiv.org/abs/2504.12397);
PEFT 0.18+, `LoraConfig(alora_invocation_tokens=[...])`) applies the adapter only from an
invocation sequence onwards, so **the earlier KV cache is the base model's** and is
reused. PEFT reports order-of-magnitude adapter-inference speedups in vLLM. aLoRA cannot
be merged, is causal-LM only, often needs `r=32`, and the invocation string must appear
in training data exactly as at inference.

### Intruder dimensions

Shuttleworth et al. (2024) [LoRA vs Full Fine-tuning: An Illusion of
Equivalence](https://arxiv.org/abs/2410.21228) find that LoRA-trained matrices contain
"intruder dimensions": top singular directions with near-zero cosine similarity to any
pretrained singular vector, which full fine-tuning does not produce. Equal task accuracy
can hide structurally different models. The authors tie forgetting to these directions;
scaling them down recovers pre-training behaviour at little target-task cost. PEFT 0.19's
`reduce_intruder_dimension` (`peft.tuners.lora.intruders`) flags the top `top_k=10`
directions with cosine below `threshold_epsilon=0.5` and writes a new adapter with
`mitigation_lambda=0.75` of them subtracted. **It trades task accuracy for retained
general ability**; measure both ([module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)).

### Unsloth specifics

Unsloth wraps the same ideas in `FastLanguageModel.get_peft_model(...)` with
`lora_dropout=0` and `use_gradient_checkpointing="unsloth"` (its guide credits about 30%
extra memory savings over `True`); [module 08](./08-frameworks-and-tooling.md) compares
frameworks, and [module 07](./07-the-wider-peft-landscape.md) covers DoRA and the rest.

---

## 6.8 Lab: LoRA from scratch, then the ablation grid

Two scripts: [lora_from_scratch.py](./code/lora_from_scratch.py) (L6.1, L6.4) and
[lora_ablation.py](./code/lora_ablation.py) (L6.2, L6.3). Both have a CPU `--smoke`
path that needs only PyTorch. Each docstring holds the full-mode procedure and version
pins (as of October 2026).

```bash
python content/courses/finetuning/code/lora_from_scratch.py --smoke
python content/courses/finetuning/code/lora_ablation.py --smoke
python content/courses/finetuning/code/lora_from_scratch.py --full --model Qwen/Qwen3-0.6B
python content/courses/finetuning/code/lora_ablation.py --full grid --model Qwen/Qwen3-4B-Base
python content/courses/finetuning/code/lora_ablation.py --full compare --model Qwen/Qwen3-1.7B-Base
```

### L6.1: LoRALinear and PEFT parity (CPU smoke, T0 full)

**Goal:** build LoRA yourself and prove it is PEFT's function. `LoRALinear` wraps
`nn.Linear`; `inject_lora` swaps it in by module name. Smoke mode asserts, on a toy GQA
decoder, the step-0 identity, the 10,240-parameter count, frozen weights untouched by
training, and merge/unmerge exactness. Full mode copies PEFT's randomly initialised
`lora_A`/`lora_B` (on `q_proj`/`v_proj`, a parity test only) from `Qwen/Qwen3-0.6B` into
your layers and asserts equal logits within 1e-4. **Record** the
`print_trainable_parameters()` line and the maximum logit difference.

### L6.2: the ablation grid (T1)

**Goal:** see rank, targets and scaling interact on a real model. Train
`Qwen/Qwen3-4B-Base` with TRL's `SFTTrainer` on 4,000 examples of
[`trl-lib/Capybara`](https://huggingface.co/datasets/trl-lib/Capybara): rank {8, 32,
128} × targets {attention-only, all-linear} × {LoRA, rsLoRA, DoRA}, 18 runs of 200
steps at $\alpha = 32$, LR 2e-4. **Record** `lora_grid.json` and `lora_grid.png` (eval
loss against trainable parameters).

**Expect** all-linear to beat attention-only at matched parameters, rsLoRA to pull
ahead at rank 128 (the LR is not retuned per rank), small gaps between ranks on this
little data, and DoRA equal or slightly better but slower per step. Re-sweep the LR for
the best and worst configs before concluding anything.

### L6.3: QLoRA vs LoRA vs full fine-tuning (T0/T1 vs T2)

**Goal:** measure the memory and quality trade. `--full compare` trains
`Qwen/Qwen3-1.7B-Base` as QLoRA (NF4 + DQ, r = 16, all-linear), bf16 LoRA, and full
fine-tuning at a 10× lower LR (2e-5, T2 only). **Record** eval loss, peak VRAM and wall
time. **Expect** QLoRA to use the least memory and be slowest per step (dequantisation on
every matmul), and eval losses within a small margin on this post-training-sized task.

### L6.4: merge into bf16, not into 4-bit (CPU smoke, T0 full)

**Goal:** see the 4-bit trap in numbers. Smoke mode trains a QLoRA-style adapter on
the toy model's NF4-quantised linears and asserts that merging then re-quantising is
over 10× worse than a bf16 merge. Its output:

```text
[5] merge error vs QLoRA reference: bf16 copy 6.04e-03, re-quantised NF4 7.68e-01 (127x)
    mean ||delta||/||W0|| = 0.056; mean cos(delta kept after re-quantisation, delta) = 0.48
```

The cosine is higher than the 6.7 check's 0.14 because the toy adapter is larger
(‖ΔW‖/‖W₀‖ = 0.056 against 0.03); the trap shrinks as the update grows. Full mode
repeats the comparison with real bitsandbytes NF4. **Record** both mean logit errors and
a few generations from each path. **Expect** path A at bf16 rounding level and path B
far above it.

---

## Key takeaways

- **LoRA learns $\Delta W = \gamma_r BA$ with $W_0$ frozen, $B$ zero-initialised**: step 0
  is the pretrained model, yet $\nabla_B = \gamma_r g(Ax)^\top \neq 0$; both zero is a
  saddle. Merged inference costs nothing extra.
- **Count parameters as $r(d_{\text{in}}+d_{\text{out}})$ per matrix**: Llama 3 8B,
  all-linear, $r = 16$ is 41,943,040 parameters (0.52%), two thirds of them in the MLP,
  which holds 81% of the base linear weights.
- **Alpha and learning rate are coupled** ($\eta\gamma_r^2$ under SGD, roughly
  $\eta\gamma_r$ under Adam): fix alpha and sweep the LR. **rsLoRA's $\alpha/\sqrt r$**
  keeps high ranks learning when you do not retune.
- **Use `target_modules="all-linear"`**: the MLP carries most of the benefit, and PEFT's
  fallback for Llama and Qwen3 is still q/v only.
- **LoRA matches full fine-tuning when the data fits its capacity** (all layers, about
  10× the full-FT LR, moderate batches; rank 1 for RL). **In capacity-bound regimes**
  (CPT, very large SFT) it learns less and forgets less.
- **QLoRA stores the frozen base in NF4** (quantile levels, block 64, double-quantised
  scales: 4.127 bits/param) and computes in bf16; only bf16 adapters train.
- **Init variants pick the starting subspace** from weights (PiSSA, OLoRA, MiCA), data
  (EVA, CorDA, LoRA-GA) or quantisation error (LoftQ); convert mutated inits back.
- **Use `trainable_token_indices` for new tokens** and `modules_to_save=["lm_head"]` in
  TRL trainers, which reject adapters on `lm_head`.
- **Use the return value of `merge_and_unload()`; merge into bf16, then quantise.**
  Merging into the 4-bit base re-quantises the sum and erases most of the update.
  aLoRA trades mergeability for KV-cache reuse.

## Self-check

1. Why does LoRA initialise B to zero? Derive the step-0 gradients of A and B, and say
   what happens if both are random or both are zero.
2. Explain the α/r scaling, why it couples alpha with the learning rate, and why rsLoRA
   uses α/√r.
3. What are NF4 and double quantisation, how many bits per parameter does QLoRA's base
   cost, and why must you not merge a LoRA into the 4-bit base before shipping?

---

**Next → [07 — Beyond LoRA: The PEFT Landscape and Memory-Efficient Full Tuning](./07-the-wider-peft-landscape.md)**
