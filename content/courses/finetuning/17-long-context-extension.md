# 17 — Long-Context Extension and Long-Sequence Training

> **Prerequisites:** modules 03, 10.
> **You will learn:** why RoPE models fail beyond their trained window, how position
> interpolation, NTK-aware scaling, YaRN and LongRoPE extend it, and the data,
> memory and context-parallel techniques for long-sequence training.

Long-document RAG, long agent trajectories and 30k-token reasoning traces need more
context than most base models handle. That is three problems. **Positions:** the
model has never seen position 100,000, so you rescale RoPE. **Data:** only genuinely
long examples teach it to use distant tokens. **Systems:** one 128k-token sequence
produces logits, activations and attention traffic that do not fit on one GPU.

Transformers [module 05](../transformers/05-positional-encodings.md) derives RoPE and
names YaRN; Transformers [module 10](../transformers/10-sparse-and-long-context.md)
covers the architectural route (sliding windows, linear attention, SSMs). This module
takes a dense, full-attention RoPE model and asks how you fine-tune it to read further.

---

## 17.1 RoPE refresher and why extrapolation fails

RoPE splits each head's $d_{\text{head}}$-dimensional query and key into
$d_{\text{head}}/2$ pairs. Pair $i$ of a token at position $m$ is rotated by angle
$m\theta_i$, with

$$
\theta_i = b^{-2i/d_{\text{head}}}, \qquad i = 0, \dots, d_{\text{head}}/2 - 1, \qquad \lambda_i = \frac{2\pi}{\theta_i}.
$$

$b$ is the **base** (`rope_theta`) and $\lambda_i$ the **wavelength** of pair $i$, in
tokens per full rotation. Pair 0 has $\theta_0 = 1$ and $\lambda_0 \approx 6.3$. The last pair has wavelength $2\pi b^{(d_{\text{head}}-2)/d_{\text{head}}}$
(54,410 tokens for $b = 10{,}000$, $d_{\text{head}} = 128$). The original Llama used
$b = 10{,}000$; many long-context models raise it to 1,000,000. Qwen3-8B ships
`rope_theta` = 1,000,000, `head_dim` 128 and `max_position_embeddings` 40,960, and
its [model card](https://huggingface.co/Qwen/Qwen3-8B) calls the native window
32,768 tokens with 131,072 "validated" using YaRN.

With $T_0$ the trained context length, $r_i = T_0/\lambda_i$ counts the full rotations
pair $i$ completed inside the training window.

| Base $b$ | $T_0$ | Pairs with $r_i > 32$ (many rotations) | Pairs with $r_i < 1$ (never a full turn) |
|---|---|---|---|
| 10,000 | 4,096 | 21 of 64 | 18 of 64 |
| 1,000,000 | 32,768 | 24 of 64 | 24 of 64 |

**High-frequency pairs have seen every angle many times**, modulo $2\pi$, so they
extrapolate fine. **Low-frequency pairs have not.** A pair with $r_i = 0.08$ only saw
angles in $[0, 0.08 \cdot 2\pi]$. At position $4T_0$ it rotates to four times that,
producing scores the query/key projections were never fitted against; attention
spreads or spikes and perplexity climbs. **Extrapolation fails on the slow
dimensions, not the fast ones.**

The TRL long-context guide shows the effect on Qwen3-4B: per-token loss on a
160k-token sequence stays reasonable to roughly 70k, then climbs until it ends above
the text's unigram entropy, worse than ignoring the context entirely
([TRL v1.15 guide](https://huggingface.co/docs/trl/v1.15.0/en/long_context_training)).

### Where the knobs live

As of October 2026, Transformers v5 configures RoPE through one `rope_parameters`
dictionary that holds `rope_theta` and a `rope_type`: `default`, `linear`, `dynamic`,
`yarn`, `longrope` or `llama3`
([Transformers RoPE utilities](https://huggingface.co/docs/transformers/main/en/internal/rope_utils)).
Older model cards, including Qwen3's, still show the v4-era `rope_scaling` key with
the same inner fields. Teaching fragment (not executed):

```json
{"max_position_embeddings": 131072,
 "rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000, "factor": 4.0,
                     "original_max_position_embeddings": 32768}}
```

Qwen3's config says 40,960 positions while its card uses 32,768 as
`original_max_position_embeddings`. Either works; training with one and serving with
the other breaks things (section 17.8).

## 17.2 Extension methods

Each method changes $\theta_i$ (or the positions) so that a sequence of length $sT_0$
produces angles the model knows; $s$ is the **scale factor** (`factor`).

### Position interpolation (PI)

[Chen et al. (2023)](https://arxiv.org/abs/2306.15595) divide every position by $s$,
equivalently every frequency becomes $\theta_i / s$, so position $sT_0$ produces exactly
the maximum training angle in every pair. They show the interpolated score's upper bound
is about 600 times smaller than the extrapolated one, so PI trains stably, and extend
Llama 7B–65B to 32,768 tokens within 1,000 steps. **The cost is local resolution:** with
$s = 4$, neighbours sit a quarter of a step apart, a spacing never seen in training. The
model must be fine-tuned to recover and still loses some short-context precision. PI is
`rope_type: "linear"`.

### NTK-aware scaling

The fix, first posted by "bloc97" on Reddit and analysed in the YaRN paper, stretches
the slow dimensions and leaves the fast ones alone (the neural-tangent-kernel argument:
networks learn high-frequency detail poorly). **Change the base** so the lowest
frequency is interpolated exactly by $s$:

$$
(b')^{-(d_{\text{head}}-2)/d_{\text{head}}} = \frac{b^{-(d_{\text{head}}-2)/d_{\text{head}}}}{s}
\;\Longrightarrow\;
b' = b\, s^{d_{\text{head}}/(d_{\text{head}}-2)} .
$$

The highest frequency is $\theta_0 = (b')^0 = 1$, untouched. In between, the ratio is
$\theta'_i / \theta_i = s^{-2i/(d_{\text{head}}-2)}$: a geometric blend from 1 (no
change) down to $1/s$ (full interpolation). Worked for $b = 10{,}000$,
$d_{\text{head}} = 128$, $s = 4$: $b' = 10{,}000 \cdot 4^{128/126} = 40{,}889.9$.
Static NTK scaling is just a new `rope_theta`; there is no special type.

**Dynamic NTK** recomputes the base from the current length at every forward pass
(Transformers uses $b' = b\,(sT/T_0 - (s-1))^{d_{\text{head}}/(d_{\text{head}}-2)}$
once $T > T_0$), so sequences inside the original window keep the original base. It
works zero-shot, but a model fine-tuned at one base sees another at inference, and
cached keys were rotated with an older base. Transformers calls it `rope_type: "dynamic"`.

The check below proves the NTK derivation numerically: the base, the exact
interpolation of the slowest pair, the untouched fastest pair, and PI's angle
preservation.

```python finetuning-check
import math
import torch

b, d_head, s, T0 = 10_000.0, 128, 4.0, 4096
i = torch.arange(d_head // 2, dtype=torch.float64)
theta = b ** (-2.0 * i / d_head)                 # (d_head/2,)

# Position interpolation: rotating position m/s with theta == rotating m with theta/s
m = torch.arange(0, 4 * T0, 997, dtype=torch.float64)[:, None]
torch.testing.assert_close((m / s) * theta, m * (theta / s), rtol=1e-15, atol=0)
pi = theta / s
torch.testing.assert_close(pi, theta / 4, rtol=0, atol=0)
# max angle at position s*T0 under PI equals the original max angle at T0, every pair
torch.testing.assert_close((s * T0) * pi, T0 * theta, rtol=1e-15, atol=0)

# NTK-aware base change b' = b * s^(d_head/(d_head-2))
b_ntk = b * s ** (d_head / (d_head - 2))
assert round(b_ntk, 1) == 40889.9
theta_ntk = b_ntk ** (-2.0 * i / d_head)
assert math.isclose(theta_ntk[-1].item(), theta[-1].item() / s, rel_tol=1e-9)
assert theta_ntk[0].item() == theta[0].item() == 1.0
ratio = theta_ntk / theta
torch.testing.assert_close(ratio, s ** (-2.0 * i / (d_head - 2)), rtol=1e-12, atol=0)
assert bool((ratio[1:] < ratio[:-1]).all())      # monotone blend from 1 down to 1/s

# slowest wavelength is 2*pi*b^((d_head-2)/d_head), not 2*pi*b
assert round(2 * math.pi / theta[-1].item()) == 54410

# Pairs that never complete a rotation inside the trained window
r = T0 / (2 * math.pi / theta)
assert int((r < 1).sum()) == 18
r_qwen = 32768 / (2 * math.pi / (1e6 ** (-2.0 * i / d_head)))
assert int((r_qwen < 1).sum()) == 24 and int((r_qwen > 32).sum()) == 24
```

It does not prove that either method produces a model that *uses* long context;
that is an empirical claim about weights, tested in section 17.7.

### YaRN

[Peng et al. (2023)](https://arxiv.org/abs/2309.00071) make the NTK intuition
explicit per dimension, using the rotation count $r_i = T_0/\lambda_i$ from 17.1.
(The paper writes $L$ for the original context; this course reserves $L$ for layers.)
Define a ramp with two thresholds $\alpha$ and $\beta$:

$$
\gamma(r) =
\begin{cases}
0 & r < \alpha \\
1 & r > \beta \\
\dfrac{r - \alpha}{\beta - \alpha} & \text{otherwise}
\end{cases}
\qquad
\theta'_i = (1 - \gamma(r_i))\,\frac{\theta_i}{s} + \gamma(r_i)\,\theta_i .
$$

A pair that rotated many times ($r_i > \beta$) is left alone; one that never completed
a rotation ($r_i < \alpha$) is fully interpolated, like PI. The paper uses $\alpha = 1$,
$\beta = 32$ for Llama (Transformers' `beta_slow` and `beta_fast` defaults). For
$b = 10{,}000$, $d_{\text{head}} = 128$, $T_0 = 4{,}096$ that leaves 21 pairs untouched,
18 fully interpolated and 25 on the ramp.

**Attention temperature.** Interpolation flattens the softmax as context grows, so
YaRN scales logits by $1/t$, applied as $\sqrt{1/t}$ on both queries and keys:

$$
\sqrt{1/t} = 0.1 \ln s + 1, \qquad s = 4 \;\Rightarrow\; \sqrt{1/t} = 1.1386 .
$$

Transformers infers this `attention_factor` from `factor` when unset. YaRN reports
"10x less tokens and 2.5x less training steps than previous methods".

The check below implements the ramp and proves its shape, the temperature value, and
that YaRN always lies between PI and the original frequencies.

```python finetuning-check
import math
import torch

b, d_head, s, T0, alpha, beta = 10_000.0, 128, 4.0, 4096, 1.0, 32.0
i = torch.arange(d_head // 2, dtype=torch.float64)
theta = b ** (-2.0 * i / d_head)
lam = 2 * math.pi / theta                       # wavelength in tokens
r = T0 / lam                                    # rotations inside the trained window
gamma = ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)
theta_yarn = (1 - gamma) * theta / s + gamma * theta

assert gamma[0].item() == 1.0                   # fastest pair: unchanged
assert gamma[-1].item() == 0.0                  # slowest pair: fully interpolated
assert bool((gamma[1:] <= gamma[:-1]).all())    # monotone non-increasing in i
assert (int((gamma == 1).sum()), int(((gamma > 0) & (gamma < 1)).sum()), int((gamma == 0).sum())) == (21, 25, 18)
assert theta_yarn[0].item() == theta[0].item()
assert math.isclose(theta_yarn[-1].item(), theta[-1].item() / s, rel_tol=1e-12)
# YaRN frequencies lie between PI (theta/s) and the original, elementwise
assert bool((theta_yarn >= theta / s - 1e-15).all() and (theta_yarn <= theta + 1e-15).all())
# attention temperature sqrt(1/t) = 0.1 ln s + 1
sqrt_inv_t = 0.1 * math.log(s) + 1
assert round(sqrt_inv_t, 4) == 1.1386
```

It does not prove that $\alpha = 1$, $\beta = 32$ or the $0.1 \ln s$ constant are
optimal for your model; the paper fitted them empirically on Llama.

### LongRoPE and LongRoPE2

[LongRoPE (Ding et al., 2024)](https://arxiv.org/abs/2402.13753) replaces YaRN's
hand-designed ramp with an evolutionary **search** for a per-pair rescale factor,
scored on perplexity, then fine-tunes. Transformers' `rope_type: "longrope"` takes
`short_factor` and `long_factor` lists of length $d_{\text{head}}/2$, for inputs inside
and beyond the original window; Phi-3's long-context checkpoints ship this way.

[LongRoPE2 (Shang et al., 2025)](https://arxiv.org/abs/2502.20082) scores the search by
**needle-driven perplexity** (loss only on a synthetic needle's answer tokens) and
trains with **mixed context windows** (short sequences keep the original RoPE). It
extends Llama 3 8B to 128k with 10B tokens, "80x fewer than Meta's approach". On Phi3-mini-3.8B
(its Table 4a), MMLU drops from 70.78 to 63.22 with YaRN and to 66.43 with NTK, versus
70.04 with LongRoPE2; GSM8K drops from 78.54 to 57.39 with YaRN.

### Table 1: the methods side by side

| Method | What it changes | Fine-tuning needed | Short-context cost | Transformers `rope_type` |
|---|---|---|---|---|
| Position interpolation | all frequencies divided by $s$ | yes (about 1,000 steps in Chen et al.) | high-frequency resolution lost; uniform across dims | `linear` |
| NTK-aware (static) | base $b \to b\,s^{d_{\text{head}}/(d_{\text{head}}-2)}$ | helps; partial zero-shot | small for fast dims, mid dims compressed | none: set `rope_theta` |
| Dynamic NTK | base recomputed from current length | no (inference trick) | none inside $T_0$ | `dynamic` |
| YaRN | per-pair ramp by $r_i$ plus temperature | short fine-tune on long data | static YaRN applies to short inputs too; measurable drops (LongRoPE2 Table 4a) | `yarn` |
| LongRoPE / LongRoPE2 | searched per-pair factors, short/long sets | search plus fine-tune | smallest reported (LongRoPE2 Table 4a) | `longrope` |

### Scaling alone is not long context

Setting `rope_type: "yarn"`, `factor: 4` gives a model that **does not break** at 128k,
not one that **uses** token 3,000 to answer a question at token 120,000. That needs
gradient signal from examples where the answer depends on distant tokens. The
rescaling is also static: Qwen's card warns that static YaRN may hurt shorter texts,
advises adding the block "only when processing long contexts is required", and
suggests `factor` 2.0 if your inputs top out at 65,536 tokens.

## 17.3 Fine-tuning recipes for extension

### LongLoRA and shifted sparse attention

[LongLoRA (Chen et al., 2023)](https://arxiv.org/abs/2309.12307) extends context with
LoRA. **First, plain LoRA fails unless embeddings and norms are trainable:** "LoRA for context extension works well under the premise
of trainable embedding and normalization." Norms are tiny ($O(d)$ per layer);
embeddings are not ($V \cdot d$: 311M parameters for Qwen3-1.7B, about 18% of the
model, and about 5 GB of training state at module 03's 16 bytes per trainable
parameter), so budget for them. They carry the global statistics that shift when the
attention distribution changes.

**Second, shifted sparse attention (S²-Attn)** cuts training cost. Attend only within
groups of $G$ tokens, and in **half the heads** shift the tokens by $G/2$ first
(`roll(-G // 2)`, rolled back after attention) so their groups straddle the others'
boundaries; information diffuses through depth. S²-Attn is a training-time device:
the paper reports it is optional at inference, so the extended model runs with
ordinary full attention. It extends Llama 2 7B from 4k to 100k, or 70B to 32k, on one
8×A100 node with PI. With FlashAttention and sequence parallelism its savings matter
less in 2026; the **trainable embeddings and norms** lesson lasts.

[LongQLoRA (Yang, 2023)](https://arxiv.org/abs/2311.04879) combines PI, QLoRA and
S²-Attn, and extends Llama 2 7B and 13B from 4,096 to 8,192 (and up to 12k) on one
32 GB V100 within 1,000 steps: the small-hardware recipe (QLoRA mechanics in
[module 06](./06-lora-and-qlora.md), memory-efficient full tuning in
[module 07](./07-the-wider-peft-landscape.md)).

### Progressive length schedules

Do not jump from 32k to 128k in one step. Llama 3 went from 8K to 128K in six stages
over about 800B tokens, advancing only when "model performance on short-context
evaluations has recovered completely" and needle retrieval was perfect at the current
length ([Llama 3 report](https://arxiv.org/abs/2407.21783)). For a fine-tune, use two or
three stages (32k → 64k → 128k, a few hundred steps each); the early ones are cheap.

### Data with genuine long-range dependencies

**Concatenating short documents does not teach long context.** Pack twenty 2k-token
articles into one 40k sequence with boundary masking ([module 05](./05-supervised-fine-tuning.md))
and every token's useful context is still under 2k; without masking the model attends
to unrelated text. Either way no gradient rewards reaching 30k tokens back. What does:

| Source | Why it has long-range dependencies |
|---|---|
| Whole books (PG-19, `emozilla/pg19`) | characters, plot and terms introduced early are referenced late |
| Code repositories, files concatenated in dependency order | a call at the end depends on a definition at the start |
| Long legal, financial or scientific documents | cross-references, defined terms, tables cited pages later |
| Synthetic long-context instructions | the question is generated *about* a long document, so answering needs it |

[LongMagpie (2025)](https://arxiv.org/abs/2505.17134) generalises Magpie self-synthesis
(the short-context version is in [module 04](./04-data-curation-and-synthesis.md)):
feed an aligned long-context model a document plus the tokens that open a user turn,
let it generate a grounded query, then collect the response. The authors report strong
HELMET, RULER and LongBench v2 results. Whatever the source, **mix in short data** at
the original length, or short-context quality drifts.

## 17.4 Memory at long context

Weights, gradients and optimizer state are fixed in $T$
([module 03](./03-training-mechanics-and-memory-math.md)); activations are not. From
here on $d$ is the model width (`hidden_size`). With FlashAttention
([transformers module 11](../transformers/11-hardware-aware-attention.md)) no
$T \times T$ matrix is materialized, so **every remaining activation term is linear in
$T$**, and still large. For Qwen3-8B ($d = 4096$, $d_{ff} = 12{,}288$, $L = 36$,
$V = 151{,}936$, 32 query heads, 8 KV heads, $d_{\text{head}} = 128$, from its
[`config.json`](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)),
one sequence, bf16, GB = $10^9$ bytes:

### Table 2: the T-dependent terms

| Term | Formula (bytes) | $T$ = 8,192 | $T$ = 32,768 | $T$ = 131,072 | Mitigation |
|---|---|---|---|---|---|
| Logits | $T \cdot V \cdot 2$ | 2.49 GB | 9.96 GB | 39.8 GB | fused LM-head loss (TRL default) |
| Checkpointed layer inputs | $L \cdot T \cdot d \cdot 2$ | 2.42 GB | 9.66 GB | 38.7 GB | activation offload to CPU; shard $T$ across GPUs |
| MLP intermediates during recompute (4 tensors) | $4 \cdot T \cdot d_{ff} \cdot 2$ | 0.81 GB | 3.22 GB | 12.9 GB | TiledMLP; shard $T$ |
| Naive attention scores, one layer, all heads | $H \cdot T^2 \cdot 2$ | 4.29 GB | 68.7 GB | 1,099.5 GB | FlashAttention / SDPA (never materialized) |
| KV cache (inference only) | $2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot T \cdot 2$ | 1.21 GB | 4.83 GB | 19.3 GB | GQA, paged attention, KV quantization |

### The logits are the first wall

Upcasting the 39.8 GB of logits to fp32 doubles it, and their gradient costs the same
again. The fix is to never hold the full $T \times V$ matrix: compute the LM head and
cross-entropy over chunks of rows and sum, exact because the loss sums over positions.
TRL's fused LM-head loss is on by default for SFT, DPO, KTO, GRPO, RLOO and Distillation
as of v1.15 (the guide uses 256-row chunks; module 03 has the mechanics); on Qwen3-4B it
moves the single-GPU limit from 32k to 160k tokens. The guide's advice to set
`loss_type="nll"` to opt out predates v1.15: in v1.15.0, `nll` *is* the fused path, and
`chunked_nll` is a deprecated alias; no `loss_type` value turns the fused head off
([sft_config.py](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/sft_config.py#L279-L309)).

### Checkpointed activations and offloading

With gradient checkpointing each layer keeps only its $T \times d$ input: 38.7 GB at
128k. Those tensors are not read until the backward pass reaches the layer, so they
can wait in CPU memory. In Transformers v5.17+ this is
`gradient_checkpointing_kwargs={"offload": True}`; the TRL guide reports a Qwen3-4B
peak dropping from 59.9 GB to 48.8 GB and the single-card limit moving from 160k to
256k tokens. The cost is PCIe traffic that must overlap with compute, and at extreme
lengths host RAM becomes the limit: ALST estimates about 915 GiB of CPU memory per
node for Llama-70B offloads at 3M tokens on 32 GPUs (4 nodes).

### TiledMLP

Checkpointing does not help *inside* a layer. Recomputing the SwiGLU MLP builds four
$d_{ff}$-wide tensors: 12.9 GB at 128k for Qwen3-8B, 76 GB at 1M tokens for the TRL
guide's Qwen3-4B. But the MLP is **per token**, so compute it over sequence tiles,
forward and backward, and only one tile's intermediates exist at a time. This is
exact. TiledMLP comes from Snowflake's [Arctic Long Sequence Training (ALST)](https://arxiv.org/abs/2506.13996),
which picks $\lceil T/d \rceil$ tiles (63 tiles for 256,000 tokens at $d = 4096$) and
reports roughly 10× lower memory on a single Llama-8B MLP layer. Attention cannot be
tiled this way; that is what section 17.5 is for.

Neither TRL nor Transformers has a TiledMLP switch as of October 2026. The entry point
is DeepSpeed's autograd function
[`deepspeed.runtime.sequence_parallel.ulysses_sp.TiledMLP`](https://github.com/deepspeedai/DeepSpeed/blob/3631712bd796593bf6e2e70d4f1c352937a44cf8/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L807-L845)
(DeepSpeed 0.18.1), which you patch into the model's MLP class before loading it.
Teaching fragment (not executed; the lab script's `--tiled-mlp` flag does exactly this):

```python
import math
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP
from transformers.models.qwen3 import modeling_qwen3

def mlp_fn(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def tiled_forward(self, x):                      # x: (B, T, d)
    shards = math.ceil(x.shape[-2] / x.shape[-1])  # ceil(T/d) tiles, as in ALST
    params = [self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]
    return TiledMLP.apply(mlp_fn, self, x, shards, params)

modeling_qwen3.Qwen3MLP.forward = tiled_forward  # before from_pretrained
```

Under multi-rank sequence parallelism, all-reduce `shards` to the max across ranks
first, as the DeepSpeed docstring does, or ranks deadlock.

ALST combines TiledMLP, tiled logits and loss, activation offload and Ulysses sequence
parallelism to train Llama 8B at 500K tokens on one H100, 3.7M on one 8×H100 node and
over 15M on four nodes. The lab's calculator (17.9) reproduces Table 2 for any config and $P$.

## 17.5 Sequence and context parallelism

Once one GPU's memory is spent, **split the sequence**: each of $P$ GPUs gets $T/P$
tokens, and everything per-token shrinks by $P$ for free. Only attention needs the
whole sequence; the two methods differ in how they provide it. (Module 10 covers the
other parallelism axes.)

### DeepSpeed Ulysses and ALST

[DeepSpeed-Ulysses (Jacobs et al., 2023)](https://arxiv.org/abs/2309.14509) switches
the sharding dimension around attention:

1. Each rank holds $T/P$ tokens and computes Q, K, V for them: shape $(T/P, H, d_{\text{head}})$.
2. **All-to-all:** re-shard by head. Each rank now holds the **full sequence** for
   $H/P$ heads: $(T, H/P, d_{\text{head}})$.
3. Run ordinary FlashAttention or SDPA locally. Heads are independent, so this is exact.
4. **All-to-all back** to $(T/P, d)$ for the output projection and MLP.

**Communication per layer.** Each all-to-all sends the $(P-1)/P$ of a rank's
$T/P \times d$ shard that belongs to other ranks; the remaining $1/P$ stays local.
Q and the output are $d$ wide; with GQA, K and V are only $H_{kv} d_{\text{head}}$
wide. So each rank sends $2T(d + H_{kv} d_{\text{head}})(P-1)/P^2$ elements per layer
forward, falling as $P$ grows. For Qwen3-8B at $T = 131{,}072$, $P = 8$, bf16:
**0.294 GB per layer forward**.

**The head constraint.** Step 3 needs a whole head's full-length Q, K and V, so $P$ must
divide the query-head count and each rank needs at least one **whole KV head**:
$H_{kv} = 8$ caps $P$ at 8 unless KV heads are replicated. ALST does that ("if kv_heads < SP
then we replicate kv_heads to match SP degree"), which raises the ceiling to the
query-head count (32 for Qwen3-8B) at the cost of extra KV memory and traffic. The TRL
guide states the practical limit as "up to the number of KV heads, 8 here".

### Ring Attention

[Ring Attention (Liu et al., 2023)](https://arxiv.org/abs/2310.01889) keeps the
sequence sharded. Rank $j$ keeps $Q_j$ and passes its KV block around a ring; after
$P - 1$ hops every rank has seen every block and merges the partial results with
**online softmax** (running max $m$, denominator $\ell$, output $o$; on a new block
with scores $S$, set $m' = \max(m, \max S)$, rescale $\ell$ and $o$ by $e^{m - m'}$,
add the block). This is FlashAttention's algebra across GPUs (the inference course's
[online-softmax page](/courses/inference/03-attention-kernels/03-01-naive-attention-and-online-softmax.html)),
and it is exact.

There is **no head constraint**: any $P$ works. The price is traffic that does not
shrink with $P$. Each rank sends (and receives) $(P-1)$ KV blocks of
$2 (T/P) H_{kv} d_{\text{head}}$ elements, about $2 T H_{kv} d_{\text{head}}$ in total:
0.470 GB per layer forward for the same Qwen3-8B example at $P = 8$. Counted the same
way, as bytes each rank sends, the ratio of the two is

$$
\frac{\text{Ulysses}}{\text{Ring}} = \frac{d + H_{kv} d_{\text{head}}}{H_{kv} d_{\text{head}}\,P} = \frac{5}{P}
\quad \text{for Qwen3-8B,}
$$

so with aggressive GQA Ring moves fewer bytes at small $P$: the formula ties at $P = 5$,
but of the degrees Ulysses can actually use on Qwen3-8B (2, 4, 8), Ring sends less at
2 and 4 and Ulysses sends less at 8. Volume is not the whole story: Ulysses is
one collective that is latency-sensitive and wants fast all-to-all (NVLink), while Ring
is $P-1$ sequential point-to-point hops that overlap with compute when they are well tuned.

**Causal load balancing.** With a causal mask, the rank holding the first $T/P$ tokens
has almost nothing to attend to, and the last rank does the most work. The standard
fix (zig-zag) splits the sequence into $2P$ chunks and gives rank $j$ chunks $j$ and
$2P-1-j$, pairing a cheap chunk with an expensive one. That is why TRL asks you to pad
sequences to a multiple of `cp_size * 2` under context parallelism.

**USP** ([Fang and Zhao, 2024](https://arxiv.org/abs/2405.07719)) runs Ulysses within a
node, where all-to-all is fast, and Ring across nodes, where the head cap would bind;
Accelerate's docs note it is not yet integrated there.

The check below proves the claim Ring Attention rests on: causal attention computed
block by block, as if KV blocks arrived from four ranks, with online-softmax
accumulation, equals full causal attention. It then asserts the memory and traffic
arithmetic stated in 17.3, 17.4 and above.

```python finetuning-check
import math
import torch

torch.manual_seed(17)
T, dh, P = 16, 8, 4
q, k, v = torch.randn(T, dh), torch.randn(T, dh), torch.randn(T, dh)
scale = 1.0 / math.sqrt(dh)

# reference: full causal softmax attention
scores = (q @ k.T) * scale
causal = torch.ones(T, T, dtype=torch.bool).tril()
ref = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1) @ v

# blockwise: rank j owns query block j; KV blocks arrive one at a time
blk = T // P
out = torch.empty_like(ref)
for j in range(P):
    qj = q[j * blk:(j + 1) * blk]
    q_pos = torch.arange(j * blk, (j + 1) * blk)[:, None]
    m = torch.full((blk, 1), float("-inf"))      # running row max
    l = torch.zeros(blk, 1)                      # running denominator
    o = torch.zeros(blk, dh)                     # running unnormalised output
    for step in range(P):                        # ring order: own block first
        src = (j - step) % P
        kb, vb = k[src * blk:(src + 1) * blk], v[src * blk:(src + 1) * blk]
        k_pos = torch.arange(src * blk, (src + 1) * blk)[None, :]
        s = (qj @ kb.T) * scale
        s = s.masked_fill(k_pos > q_pos, float("-inf"))
        if torch.isinf(s).all():                 # block entirely in the future: skip
            continue
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p @ vb
        m = m_new
    out[j * blk:(j + 1) * blk] = o / l
torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

# arithmetic stated in prose (Qwen3-8B config, bf16, GB = 1e9)
L_layers, d_model, V, H_kv, d_head, T_long, GB = 36, 4096, 151_936, 8, 128, 131_072, 1e9
assert round(T_long * V * 2 / GB, 1) == 39.8                          # logits
assert round(2 * L_layers * H_kv * d_head * T_long * 2 / GB, 1) == 19.3  # KV cache
# bytes each rank sends per layer forward, counted the same way for both methods
uly = lambda P_: 2 * T_long * (d_model + H_kv * d_head) * (P_ - 1) / P_**2 * 2
ring = lambda P_: (P_ - 1) * 2 * (T_long / P_) * H_kv * d_head * 2
assert round(uly(8) / GB, 3) == 0.294 and round(ring(8) / GB, 3) == 0.470
for P_ in range(2, 33):
    assert math.isclose(uly(P_) / ring(P_), 5 / P_, rel_tol=1e-12)
assert math.isclose(uly(5), ring(5))      # the formula ties at P = 5, not a usable degree
# Ulysses degrees on Qwen3-8B: P divides the 32 query heads and P <= H_kv = 8
usable = [P_ for P_ in range(2, H_kv + 1) if 32 % P_ == 0]
assert usable == [2, 4, 8]
assert uly(2) > ring(2) and uly(4) > ring(4) and uly(8) < ring(8)

# Qwen3-1.7B (tied embeddings): the embedding matrix is ~18% of the model
V17, d17, dff17, L17, H17, Hkv17 = 151_936, 2048, 6144, 28, 16, 8
emb = V17 * d17
per_layer = d17 * (H17 * d_head) * 2 + d17 * (Hkv17 * d_head) * 2 + 3 * d17 * dff17 + 2 * d_head + 2 * d17
total = emb + L17 * per_layer + d17
assert emb == 311_164_928 and round(100 * emb / total) == 18
assert round(emb * 16 / GB, 1) == 5.0                                 # 16 bytes per trainable param
```

It does not prove anything about distributed performance (overlap, latency, load
balance); exactness of the merge and the arithmetic are the only claims.

### Turning it on in TRL and Accelerate

As of October 2026 the two methods are tied to different backends
([TRL guide](https://huggingface.co/docs/trl/v1.15.0/en/long_context_training),
[HF Ulysses blog, March 2026](https://huggingface.co/blog/ulysses-sp)):

| | Context parallelism (Ring) | Ulysses sequence parallelism |
|---|---|---|
| Backend | FSDP2 | DeepSpeed (ZeRO 2/3) |
| Setting | `cp_size` | `sp_size`, `sp_backend="deepspeed"` |
| Splits | tokens | attention heads |
| Scales to | any $P$ | $P$ up to KV heads (more with replication) |
| Attention | SDPA, causal mask only | SDPA, FlashAttention 2/3 |
| Packing | refused | via `position_ids` with FlashAttention |
| Requires | accelerate 1.11 | accelerate 1.12, DeepSpeed 0.18.1 |
| Pad to | multiple of `cp_size * 2` | multiple of `sp_size` |

When your examples are already near the maximum length, packing buys little anyway.
For CP, the Accelerate config sets `fsdp_version: 2` and, for 8 GPUs with one sequence
shared by 4, `parallelism_config_dp_shard_size: 2` and `parallelism_config_cp_size: 4`
(the DeepSpeed variant is in the HF blog). Teaching fragment (not executed), the
matching `SFTConfig`:

```python
from trl import SFTConfig

args = SFTConfig(
    max_length=131072,              # global length, before splitting across ranks
    pad_to_multiple_of=8,           # cp_size * 2 for CP=4; sp_size for Ulysses
    packing=False,                  # required with cp_size
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"offload": True},   # transformers >= 5.17
    model_init_kwargs={"dtype": "bfloat16", "max_position_embeddings": 131072,
                       "rope_parameters": {"rope_type": "yarn", "rope_theta": 1_000_000,
                                           "factor": 4.0,
                                           "original_max_position_embeddings": 32768}},
)
```

The TRL guide measures 34.6 s, 17.8 s and 9.5 s per step for Qwen3-8B at 131k tokens
on 2, 4 and 8 GPUs with CP.

## 17.6 Long-context RL

Long reasoning traces and agent trajectories put RL
([module 13](./13-rl-with-verifiable-rewards.md)) at long context too, with two memory
budgets. **Rollout memory** is the engine's KV cache: Qwen3-8B with $G = 8$ completions
of 32,768 tokens needs $8 \times 32{,}768 \times 147{,}456$ bytes $= 38.7$ GB before
weights. **Training memory** is everything in 17.4, paid for the policy and, when
$\beta > 0$, again for the reference log-probs. RL needs per-token log-probs and
entropies, so fused log-prob kernels matter.

Unsloth reported (February 2026) gpt-oss QLoRA training at 380K context on one 192 GB
B200, and separately that new batching enables "~7x longer context RL" versus setups
already using FA3, kernels and chunked losses
([Unsloth update](https://unslothai.substack.com/p/unsloth-2026-update-faster-moe)).
Treat both as vendor-reported. In TRL, raise `max_completion_length` deliberately
(default 512) and size the rollout engine for the longest trace, not the average:
`vllm_max_model_length` must cover prompt plus completion, and
`vllm_gpu_memory_utilization` (default 0.3 when colocated) sets the KV pool. Keeping the
default `beta=0.0` skips the reference-model pass entirely. The systems of 17.5 apply
to the training side unchanged.

## 17.7 Evaluating long context

### Needle in a haystack tests retrieval, nothing more

The needle test hides one fact ("the passkey is 73914") at a chosen depth in filler text
and asks for it back, sweeping length and depth. It catches broken positions
immediately, but **passing it proves only single-fact lexical retrieval**: the needle
looks nothing like the haystack, so attention finds it by surface mismatch.

[RULER (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654) keeps the synthetic,
length-controllable design and adds harder tasks: multi-key, multi-value and
multi-query needles; **multi-hop tracing** (follow `X1 = 12345`, `X2 = X1`, ...);
**aggregation** (most frequent words); and QA with distractors. Across 17 models,
almost all aced vanilla needles yet dropped substantially as length grew, and only
half of those claiming 32K or more held satisfactory performance at 32K.

### Real documents and position sweeps

Evaluate on long-document QA in your domain and on a public set such as LongBench v2
(`zai-org/LongBench-v2`, multiple choice over realistic long contexts). And sweep
**where** the evidence sits: [Liu et al. (2023)](https://arxiv.org/abs/2307.03172)
found accuracy highest at the start or end of the context and lowest in the middle
("lost in the middle"). A fine-tune that only improves the ends hides behind an average.

### Short-context regression is part of the evaluation

Extension changes the positions short inputs see and shifts the data distribution.
Run the general suite from [module 09](./09-evaluating-fine-tuned-models.md) **before
and after**, with identical decoding settings.

| Test | Measures | Failure it catches |
|---|---|---|
| Needle sweep (length × depth) | retrieval of one planted fact | broken or mismatched RoPE scaling |
| RULER-style tasks | multi-key retrieval, tracing, aggregation | "finds needles, cannot use context" |
| Real long-document QA | the job you actually care about | synthetic-only gains |
| Position sweep | accuracy vs evidence depth | lost in the middle |
| Short-context suite | general capability at normal length | the regression you paid for the extension |

## 17.8 Serving long-context fine-tunes

Serving-side memory is dominated by the KV cache (Table 2's last row): 19.3 GB **per
sequence** for Qwen3-8B at 131,072 tokens, next to about 16.4 GB of bf16 weights. On
one 80 GB GPU that is two or three concurrent full-length requests, which is why GQA
([transformers module 09](../transformers/09-mqa-gqa-mla.md)) and KV quantization
matter so much here; the inference course's
[KV sizing page](/courses/inference/02-memory-and-kv-cache/02-01-kv-cache-sizing.html)
works the formula for other models, and its
[positional-encoding and long-context page](/courses/inference/03-attention-kernels/03-06-positional-encoding-and-long-context.html)
covers the kernel side of RoPE scaling. Paged attention and **prefix caching** (which
pays off when many requests share one long document) are covered in
[module 20](./20-production-serving-and-cost.md).

### RoPE scaling (`rope_parameters`) must match in the serving engine

The quiet failure: you fine-tune with YaRN, `factor: 4`, and the engine serves with
the base config, or a different `original_max_position_embeddings`. Positions beyond
the native window are then unscaled, the attention temperature is missing, and quality
collapses exactly on the long inputs you trained for, while short-prompt smoke tests
pass. Three rules:

1. **Save the scaling in the checkpoint.** After merging, confirm `config.json` has the
   trained `rope_parameters` and `max_position_embeddings` at the extended length.
2. **Pass it explicitly to the engine.** vLLM's `--rope-scaling` flag "is no
   longer supported"; use `--hf-overrides` with `rope_parameters`
   ([vLLM v0.31.0 context-extension docs](https://docs.vllm.ai/en/v0.31.0/features/context_extension/)).
   Qwen's model card still shows the older flag. For llama.cpp, the card gives
   `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`.
3. **Test above the native window after deployment**: a needle at depth 0.5 at 1.5×
   the original length catches the mismatch in a minute.

Teaching fragment (not executed):

```bash
vllm serve ./qwen3-8b-128k-merged \
  --hf-overrides '{"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000}}' \
  --max-model-len 131072
```

---

## 17.9 Lab: extend Qwen3-1.7B with YaRN

### Smoke runs (CPU)

```bash
python content/courses/finetuning/code/rope_scaling_lab.py --smoke
python content/courses/finetuning/code/long_context_memory.py --smoke
```

[`rope_scaling_lab.py`](./code/rope_scaling_lab.py) asserts the 17.2 numbers and runs a
tiny attention layer ($d_{\text{head}} = 16$) to 4× its trained length: unscaled RoPE
drives 4 of 8 pairs to unreachable score values, PI none.
[`long_context_memory.py`](./code/long_context_memory.py) prints Table 2 and the
Ulysses/Ring traffic for any $P$.

### L17.1 (tier T1): YaRN plus a short LoRA fine-tune

- **Goal:** extend `Qwen/Qwen3-1.7B` from its 32,768-token native window by 2–4× and
  measure what the scaling alone buys versus scaling plus fine-tuning.
- **Hardware:** one 24 GB Ampere-or-newer GPU (bf16).
- **Data:** `emozilla/pg19` books for training, PG-19 validation text as needle filler,
  LongBench v2 via lm-evaluation-harness's `longbench2` task (v0.4.13+).
- **Recipe:** YaRN `factor` 4, LoRA on all linear layers, embeddings and norms trainable
  (LongLoRA), 32k sequences (64k with `--offload` or `--tiled-mlp`), gradient
  checkpointing, and TRL's default `loss_type="nll"`, which runs through the fused LM
  head (`chunked_nll` is a deprecated alias). **Budget for the embeddings:** the tied
  matrix is 311M trainable parameters, about 18% of the model and about 5 GB of the
  24 GB card at 16 bytes each, before any activation. The script prints the count.

```bash
python content/courses/finetuning/code/rope_scaling_lab.py --full --factor 4 --train-len 32768 --max-steps 200
python content/courses/finetuning/code/rope_scaling_lab.py --full --factor 4 --train-len 65536 --offload
python content/courses/finetuning/code/rope_scaling_lab.py --full --factor 4 --train-len 65536 --tiled-mlp
```

The third run needs `deepspeed>=0.18.1`; compare its peak with the second. The script
runs a passkey sweep three times (native RoPE up to 32k, YaRN untrained, YaRN after
LoRA) and saves a merged checkpoint carrying the YaRN `rope_parameters` and
`max_position_embeddings`. Then run lm-evaluation-harness
(`"lm_eval[hf,vllm,ifeval,ruler]"`; check the RULER README for its length keys).
LongBench v2 uses the vLLM backend, because the HF backend materializes full-length
logits (17.4):

```bash
lm_eval --model hf --model_args pretrained=Qwen/Qwen3-1.7B,dtype=bfloat16 --tasks ifeval,mmlu_pro --limit 200
lm_eval --model hf --model_args pretrained=/tmp/qwen3-1.7b-yarn/merged,dtype=bfloat16 --tasks ifeval,mmlu_pro --limit 200
lm_eval --model hf --model_args pretrained=/tmp/qwen3-1.7b-yarn/merged,dtype=bfloat16 --tasks ruler --metadata '{"max_seq_lengths":[32768,65536]}'
lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-1.7B,max_model_len=32768 --tasks longbench2
lm_eval --model vllm --model_args pretrained=/tmp/qwen3-1.7b-yarn/merged,max_model_len=65536 --tasks longbench2
```

Most LongBench v2 contexts exceed both windows and the harness truncates them, so the
base model sees less of each document; that difference is what you measure.

**Record:** passkey accuracy by length for the three conditions, LongBench v2 accuracy
for base and merged models, IFEval and MMLU-Pro deltas, peak GPU memory, step time.
**Expect:** native RoPE fails past the window; YaRN alone recovers much of the needle
sweep; the fine-tune adds more on RULER and real QA than on needles; short-context
scores move a little either way. Report the deltas, not one headline number.

### L17.2 (tier T3): 128k SFT of Qwen3-8B with sequence parallelism

- **Goal and hardware:** per-GPU memory and step time against $P$ for 131,072-token SFT
  of `Qwen/Qwen3-8B` (YaRN `rope_parameters` above, PG-19 books truncated to 131,072
  tokens) on one 8×H100 or A100 80 GB node.
- **Steps:** run the 17.5 `SFTConfig` under FSDP2 + `cp_size` with $P \in \{2, 4, 8\}$,
  then DeepSpeed + `sp_size` with $P \in \{4, 8\}$, ten steps each, logging
  `torch.cuda.max_memory_allocated()` per rank and seconds per step. Repeat the $P = 8$
  DeepSpeed run with the 17.4 TiledMLP patch (all-reduce the shard count) and record the
  peak drop. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as the TRL guide did.
  Compare peaks with `long_context_memory.py --seq 131072 --P 2,4,8`.
- **Expect:** activation memory roughly halves as $P$ doubles; CP and SP give the same
  loss to logging precision; step time scales sublinearly as communication grows.

---

## Key takeaways

- **RoPE extrapolation fails on the slow dimensions.** Pairs that never completed a
  rotation inside the trained window (24 of 64 for Qwen3 at 32k) see unseen angles
  beyond it; fast pairs are fine.
- **PI blurs local order; NTK changes the base; YaRN ramps per pair plus temperature.**
  PI divides all frequencies by $s$; NTK sets $b' = b\,s^{d_{\text{head}}/(d_{\text{head}}-2)}$
  so only the slowest pair is fully interpolated; YaRN blends by rotation count and
  scales logits by $(0.1\ln s + 1)^2$.
- **Scaling makes long inputs safe, fine-tuning makes them useful.** Positions in range
  is not the same as attention that retrieves and reasons across 100k tokens.
- **Extension costs short-context quality; measure it.** Static YaRN applies to short
  prompts too, and LongRoPE2 reports measurable drops for YaRN and NTK at 128k.
- **Train embeddings and norms when extending with LoRA, and budget for them.**
  LongLoRA's central finding; the embeddings are 18% of Qwen3-1.7B.
- **Long-context data needs long-range dependencies.** Packed short documents teach
  nothing about distance; books, repositories and LongMagpie-style synthesis do.
- **The logits are the first memory wall.** 39.8 GB of bf16 logits for one 128k sequence
  with a 152k vocabulary; TRL's fused LM-head loss (`loss_type="nll"`) removes it.
- **Everything per-token tiles or shards exactly.** TiledMLP, activation offload and
  sequence sharding fix the rest; only attention needs communication.
- **Ulysses is capped by heads, Ring is not.** Ulysses re-shards by head with two
  all-to-alls and sends $5/P$ of Ring's bytes for Qwen3-8B; Ring passes KV blocks with
  exact online-softmax merging and works for any $P$.
- **Needle tests catch broken positions, not long-context skill; serving must match
  training.** Use RULER, real long-document QA and position sweeps, and pass the same
  `rope_parameters` to vLLM through `--hf-overrides`.

## Self-check

1. How does YaRN differ from linear position interpolation and NTK-aware scaling, and
   why does any of them still need fine-tuning?
2. Why is Ulysses sequence parallelism capped by the number of (KV) heads, and how does
   Ring Attention avoid that cap?
3. Which parallelism and memory techniques would you use for 128k-context SFT of an 8B
   model on 8 GPUs? Estimate the logits memory you must avoid materializing.

---

**Next → [18 — Model Merging and Adapter Composition](./18-model-merging-and-composition.md)**
