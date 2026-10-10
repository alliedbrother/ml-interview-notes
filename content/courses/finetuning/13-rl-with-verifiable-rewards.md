# 13 — RL with Verifiable Rewards: GRPO and Its Successors

> **Prerequisites:** modules 09, 12 (10 for the scale-out sections).
> **You will learn:** policy-gradient foundations for LLMs, GRPO's group-relative advantages, the
> 2025–26 fixes (Dr. GRPO, DAPO, GSPO, CISPO, SAPO), rollout infrastructure and
> importance-sampling correction, and reward design that resists hacking.

Reinforcement learning with verifiable rewards (RLVR) is the post-training technique that defined
2025–26, and open reasoning models are built on it. The idea fits in one sentence: sample answers,
check them with a program, push up the ones that pass. Making it work needs the policy-gradient
foundations, GRPO's group-relative advantage and the biases it hides, the systems layer (vLLM
rollouts, weight sync, importance-sampling correction), and rewards that cannot be gamed.

The [deep-RL note](../../notes/deep-learning/deep-rl.md) derives the policy-gradient theorem and
PPO; the [prompting-and-alignment note](../../notes/nlp/llm-prompting-and-alignment.md) states the
GRPO advantage and places it in the post-training pipeline. This module assumes both and goes to
token-level detail, the 2025 fixes and the TRL knobs. Defaults are quoted from **TRL v1.15.0
(October 2026)**, pinned to commit
[`dae7b6a`](https://github.com/huggingface/trl/tree/dae7b6adc8d2729cfa8d21bc8480656ede769240) and
the [v1.15.0 GRPO docs](https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer).

---

## 13.1 Policy gradients for language models

### REINFORCE at the sequence level

A prompt $x$ produces a completion $y=(y_1,\dots,y_T)$; a verifier returns a scalar $R(x,y)$, say
1 if the boxed answer is right. Treat the whole completion as one action whose log-probability is
a sum over tokens, $\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t})$. The
policy-gradient theorem gives

$$
\nabla_\theta J = \mathbb{E}_{y\sim\pi_\theta}\big[(R(x,y) - b(x))\,\nabla_\theta\log\pi_\theta(y\mid x)\big].
$$

The baseline $b(x)$ does not depend on $y$, so it leaves the expectation unchanged
($\mathbb{E}[\nabla\log\pi]=0$) but cuts variance: without it, every answer to an easy prompt gets
pushed up, and the useful signal is the small *difference* between right and wrong. $R-b$ is the
**advantage** $\hat A$. Every method in this module is a choice of how to estimate $\hat A$ and how
to spread it over tokens.

### RLOO and REINFORCE++

**RLOO** ([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)) samples $k$ completions per
prompt and baselines each against the other $k-1$:

$$
\hat A_i = r_i - \frac{1}{k-1}\sum_{j\neq i} r_j = \frac{k}{k-1}\,(r_i - \bar r).
$$

The leave-one-out mean is independent of $y_i$, so the estimator stays unbiased. TRL's
`RLOOTrainer` is a stable v1 trainer, and it puts the KL **in the reward**: a summed per-token
log-ratio with `beta=0.05` by default
([`rloo_trainer.py`](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/rloo_trainer.py#L1612-L1622)).
**REINFORCE++** ([Hu et al., 2025](https://huggingface.co/papers/2501.03262)) normalizes advantages
over the global batch instead of per prompt, with a token-level KL; TRL approximates it with
`RLOOConfig(normalize_advantages=True)`.

### PPO, and why GRPO drops the critic

PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) reuses rollouts for several steps
and corrects with a per-token ratio $\rho_t=\pi_\theta(y_t\mid\cdot)/\pi_{\text{old}}(y_t\mid\cdot)$,
clipped:

$$
\mathcal{L}^{\text{clip}}_t = \min\big(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon_{\text{high}})\hat A_t\big).
$$

Differentiate it. When $\hat A>0$ and $\rho>1+\epsilon_{\text{high}}$, the clipped branch is the
minimum and is constant in $\theta$: **the gradient is exactly zero**. Likewise for $\hat A<0$ and
$\rho<1-\epsilon$. Everywhere else the token keeps its gradient $\hat A\,\nabla\rho$. Clipping stops
a token moving further in the direction it already moved; it never stops it moving back.

PPO's advantage comes from a learned **value function** $V_\phi$ and GAE
([Schulman et al., 2015](https://arxiv.org/abs/1506.02438)): $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$,
$\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$. For LLMs $V_\phi$ is a second policy-sized
network (module 03's 16 bytes per parameter, twice), and it must learn per-token values from one
**sparse terminal reward**, which early in training is mostly noise. GRPO drops $V_\phi$ and
estimates the baseline from a group of samples. TRL removed `PPOTrainer` in v1.13
([module 12](./12-preference-optimization.md)): learn PPO as the parent, run GRPO or RLOO.

### KL: in the loss or in the reward, and which estimator

| Placement | Form | Who uses it |
|---|---|---|
| In the reward | $r_i \leftarrow r_i - \beta\sum_t \log\frac{\pi(y_t)}{\pi_{\text{ref}}(y_t)}$, then advantages | classic RLHF-PPO, TRL `RLOOTrainer` |
| In the loss | add $\beta\,\widehat{\mathrm{KL}}_t$ per token to the policy loss | DeepSeekMath GRPO, TRL `GRPOTrainer` |

The KL is estimated from samples $y\sim\pi$ (an exact sum over a 150k vocabulary per position is
wasteful). With $r=\pi_{\text{ref}}(y_t)/\pi(y_t)$, Schulman's
[KL-approximation note](http://joschu.net/blog/kl-approx.html) compares $k_1=-\log r$ (unbiased,
but samples go negative and variance is high) with

$$
k_3 = (r-1) - \log r,\qquad \mathbb{E}_\pi[k_3] = \underbrace{\mathbb{E}_\pi[r]-1}_{=\,\sum\pi\frac{\pi_{\text{ref}}}{\pi}-1\,=\,0} + \mathbb{E}_\pi[-\log r] = \mathrm{KL}(\pi\,\|\,\pi_{\text{ref}}).
$$

$k_3$ is unbiased and **never negative** ($\log r\le r-1$). GRPO uses it per token. One subtlety:
differentiating a sample estimate of the KL *value* does not give the reverse-KL *gradient* when
samples came from $\pi_{\text{old}}$. TRL's `use_bias_correction_kl=True` (default since v1.10)
multiplies the per-token KL by $\pi_\theta/\pi_{\text{old}}$, following
[DeepSeek-V3.2](https://huggingface.co/papers/2512.02556), which makes the KL gradient the unbiased
reverse-KL gradient. The ratio is differentiable, so this changes the gradient even on-policy where
its value is 1; it matters only when `beta != 0`
([`grpo_config.py` L967](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L967-L978)).
Background on KL: the [information-theory note](../../notes/math/information-theory.md).

Check 13.1 confirms that $k_1$ and $k_3$ share a mean while $k_3$ is never negative and has far
lower variance when the policies are close.

```python finetuning-check
import torch

torch.manual_seed(0)
pi = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], dtype=torch.float64)
ref = torch.tensor([0.25, 0.25, 0.20, 0.17, 0.13], dtype=torch.float64)
exact = (pi * (pi / ref).log()).sum()
assert abs(exact.item() - 0.009686) < 1e-6 and exact.item() < 0.1

g = torch.Generator().manual_seed(0)
idx = torch.multinomial(pi, 200_000, replacement=True, generator=g)  # samples from pi
r = ref[idx] / pi[idx]                                                # r = pi_ref / pi
k1 = -r.log()
k3 = (r - 1) - r.log()

assert abs(k1.mean().item() - exact.item()) < 2e-4
assert abs(k3.mean().item() - exact.item()) < 0.01
assert abs(k3.mean().item() - exact.item()) < 1e-3        # tighter: k3 is accurate here
assert bool((k3 >= 0).all())                              # k3 never negative
assert bool((k1 < 0).any())                               # k1 has negative samples
assert k3.var().item() < k1.var().item() / 100            # ~1.3e-4 vs ~1.9e-2
print("k1/k3 KL estimators verified")
```

Here the exact KL is 0.00969; both Monte Carlo means land within $10^{-4}$, with $\mathrm{var}(k_1)\approx0.019$
and $\mathrm{var}(k_3)\approx0.00013$. The check does not show $k_3$ is better when the policies are far
apart; it is not, and Schulman's note shows its variance growing fast as the ratio spreads.

---

## 13.2 GRPO

**GRPO** comes from DeepSeekMath ([Shao et al., 2024](https://huggingface.co/papers/2402.03300)).
For each prompt $q$, sample $G$ completions, score each, and use the group as the baseline:

$$
\hat A_{i,t} = \hat A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon},
$$

broadcast to every token of completion $i$. The DeepSeekMath objective, with $\mu$ inner updates per
rollout batch, is

$$
\mathcal{J}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i},\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i}\big) - \beta\,\widehat{\mathrm{KL}}_{i,t}\Big)\Bigg],
\qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\text{old}}(o_{i,t}\mid q,o_{i,<t})}.
$$

Remember the $\frac{1}{|o_i|}$ and the std divisor: 13.3 shows both are biased.

### The TRL knobs

| `GRPOConfig` field | Meaning | Default (TRL v1.15.0) |
|---|---|---|
| `num_generations` | $G$, completions per prompt | 8 |
| `per_device_train_batch_size` | completions (not prompts) per device per micro-step | Trainer default |
| `steps_per_generation` | micro-steps served by one generation batch; above `gradient_accumulation_steps`, later steps are off-policy | `gradient_accumulation_steps` |
| `num_iterations` | $\mu$, passes over the same rollouts; for $\mu>1$, $\rho\neq1$ and clipping matters | 1 |
| `beta` | KL coefficient; `0.0` loads **no reference model** | 0.0 |
| `epsilon`, `epsilon_high` | lower and upper clip ranges | 0.2, `None` (= `epsilon`) |
| `loss_type` / `scale_rewards` | token aggregation / std scaling (13.3) | `"dapo"` / `"group"` |
| `max_completion_length` | generation cap in tokens | 512 (since v1.10) |
| `learning_rate` | AdamW LR | 1e-6 |

Source: [`grpo_config.py`](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L479-L830).
One generation batch is `per_device_train_batch_size × num_processes × steps_per_generation`
completions and must be divisible by `num_generations`.

**GRPO is not "the method without KL".** DeepSeekMath had a $k_3$ term, and TRL's docstring notes
DeepSeek-R1 used `beta=0.001`
([L676](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L676-L683)).
TRL defaults to `0.0` because Open-Reasoner-Zero, Dr. GRPO and DAPO found it unnecessary for
verifiable rewards, and it saves a model copy. It is a choice: turn it on when staying near the
base matters (keeping general chat ability during narrow RL) or when outputs drift.

### Why the group advantage works, and where it does nothing

Take $G=8$ with rewards $[1,0,1,0,0,1,0,0]$: mean $0.375$, population std
$\sqrt{0.375\cdot0.625}=0.4841$, so correct completions get $+1.2910$ and wrong ones $-0.7746$, and
$3\times1.2910=5\times0.7746$. **Group-normalized advantages always sum to zero**: each group moves
mass from its wrong answers to its right ones.

TRL differs in two details
([`grpo_trainer.py` L2686–L2714](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_trainer.py#L2686-L2714)):
its `nanstd` applies **Bessel's correction**
([`utils.py` L1023](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/utils.py#L1023-L1050))
and it adds $\varepsilon=10^{-4}$, giving std $0.5175$ and advantages $+1.2074/-0.7244$, a uniform
$\sqrt{7/8}$ rescale. Harmless, but know it when comparing logs across codebases.

The corollary matters more: **if all $G$ rewards are equal, every advantage is zero and the prompt
contributes no gradient.** All-correct prompts are too easy, all-wrong ones too hard; either way the
generation compute is wasted. TRL logs the fraction as `frac_reward_zero_std`, and 13.10 is largely
about keeping it low.

Check 13.2 asserts the worked values, zero-sum, the zero-variance case, and the RLOO and Dr. GRPO
variants.

```python finetuning-check
import torch

torch.manual_seed(0)
r = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float64)
mean, std_pop = r.mean(), r.std(unbiased=False)
adv = (r - mean) / std_pop
assert round(mean.item(), 4) == 0.375 and round(std_pop.item(), 4) == 0.4841
assert round(adv[0].item(), 4) == 1.2910 and round(adv[1].item(), 4) == -0.7746
assert abs(adv.sum().item()) < 1e-6

# TRL: Bessel-corrected std plus eps = 1e-4
std_trl = r.std(unbiased=True)
adv_trl = (r - mean) / (std_trl + 1e-4)
assert round(std_trl.item(), 4) == 0.5175
assert round(adv_trl[0].item(), 4) == 1.2074 and round(adv_trl[1].item(), 4) == -0.7244

# zero-sum for arbitrary rewards
for _ in range(100):
    x = torch.randn(8, dtype=torch.float64)
    a = (x - x.mean()) / (x.std() + 1e-4)
    assert abs(a.sum().item()) < 1e-6

# zero-variance group: no signal
same = torch.full((8,), 1.0, dtype=torch.float64)
assert torch.equal((same - same.mean()) / (same.std() + 1e-4), torch.zeros(8, dtype=torch.float64))

# RLOO: leave-one-out baseline, also zero-sum, equals k/(k-1) * (r - mean)
k = r.numel()
rloo = r - (r.sum() - r) / (k - 1)
assert abs(rloo.sum().item()) < 1e-6
torch.testing.assert_close(rloo, k / (k - 1) * (r - mean))

# Dr. GRPO (scale_rewards=False): advantages are just r - mean
torch.testing.assert_close(r - mean, torch.tensor([0.625, -0.375, 0.625, -0.375, -0.375, 0.625, -0.375, -0.375], dtype=torch.float64))
print("group advantages verified")
```

The check is arithmetic only. It says nothing about whether $G=8$ can see a 3% pass rate; it cannot,
since an 8-group is all-wrong with probability $0.97^8\approx0.78$.

Prompt-only dataset formats are in [module 04](./04-data-curation-and-synthesis.md). Sampling
temperature defaults to 1.0; do not lower it to "improve quality", because diversity within the
group *is* the signal (sampling parameters: [module 02](./02-tokens-templates-and-the-causal-lm-objective.md)).

---

## 13.3 The biases, and the 2025 fixes

### Dr. GRPO: length and difficulty bias

[Liu et al. (2025)](https://huggingface.co/papers/2503.20783), "Understanding R1-Zero-Like Training",
found two biases in the DeepSeekMath objective. **Length bias from $1/|o_i|$:** a correct 10-token
answer gives each token weight $1/10$, a wrong 100-token answer only $1/100$. Positive advantages
favor short completions, and negative advantages are **diluted in long ones**, so a failing model
lowers its per-token penalty by rambling; the paper shows length growing on incorrect answers.
**Difficulty bias from the std divisor:** a 7/8-correct group has small std, so its advantages are
inflated; nearly-solved and nearly-hopeless prompts outweigh the middle. The fix: normalize by a
constant, `loss_type="dr_grpo"` (sum of token losses over `batch × max_completion_length`), and drop
the std, `scale_rewards=False`.

### DAPO: four changes

DAPO ([Yu et al., 2025](https://huggingface.co/papers/2503.14476)) reached 50 points on AIME 2024
from Qwen2.5-32B base with four changes:

1. **Token-level loss.** Sum token losses over the batch and divide by total tokens, so a token in a
   long answer weighs the same as one in a short answer. TRL's **default**, `loss_type="dapo"`.
2. **Clip-higher.** Decouple the band to $[1-\epsilon,1+\epsilon_{\text{high}}]$, $\epsilon=0.2$,
   $\epsilon_{\text{high}}=0.28$. With symmetric clipping a token at probability 0.01 can only reach
   0.012 per step; the upper bound is what caps exploration, so raise it alone.
3. **Dynamic sampling.** Drop all-equal groups and resample until the batch is full of informative
   ones. The cost follows from 13.10's formula: at $p=0.01$ and $G=8$,
   $0.99^8+0.01^8\approx0.923$ of groups are dropped, so filling a batch takes about 13× the
   generation. TRL's paper index marks this **not supported** in `GRPOTrainer` (v1.15.0); difficulty
   filtering (13.10) gets you part of the way.
4. **Overlong handling.** A truncated completion was cut off, not necessarily wrong. Mask it from
   the loss (`mask_truncated_completions=True`) or add a soft linear penalty near the cap
   (`trl.rewards.get_soft_overlong_punishment`).

### Table 1: aggregation and scaling in TRL

| `loss_type` | Aggregation of per-token loss $\ell_{i,t}$ | Length bias | TRL default? |
|---|---|---|---|
| `grpo` | $\frac{1}{G}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t \ell_{i,t}$ | yes: long wrong answers under-penalized | no |
| `bnpo` | $\sum_{i,t}\ell_{i,t} / \sum_i\lvert o_i\rvert$ over the **local** micro-batch | reduced; varies with local batch size | no |
| `dapo` | same, over active tokens in the **global accumulated** batch | equal per-token weight; long completions carry more total weight | **yes** |
| `dr_grpo` | $\sum_{i,t}\ell_{i,t} / (B\cdot L_{\max})$, $L_{\max}=$ `max_completion_length` | none from normalization | no |

| `scale_rewards` | Advantage | Effect |
|---|---|---|
| `"group"` (default; `True`) | $(r - \bar r_{\text{group}})/(\text{std}_{\text{group}}+\varepsilon)$ | unit variance per prompt; difficulty bias |
| `"batch"` | $(r - \bar r_{\text{group}})/(\text{std}_{\text{batch}}+\varepsilon)$ | "Lite PPO" ([Liu et al., 2025, Tricks or Traps](https://huggingface.co/papers/2508.08221)) |
| `False` (`"none"`) | $r - \bar r_{\text{group}}$ | Dr. GRPO; step size now tracks raw reward scale |

Sources: [`loss_type`](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L796-L830),
[`scale_rewards`](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L784-L795),
[loss code](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_trainer.py#L3078-L3100).

### The per-token weights, worked

A group of two: completion 1 correct, 10 tokens, $\hat A=+1$; completion 2 wrong, 100 tokens,
$\hat A=-1$. At $\rho=1$ the per-token loss is $-\hat A\log\pi$, so each token's gradient is its
aggregation weight times $\hat A$:

| `loss_type` | weight per correct token | weight per wrong token | total, correct vs wrong sequence |
|---|---|---|---|
| `grpo` | $1/(2\cdot10)=1/20$ | $1/(2\cdot100)=1/200$ | 1/2 vs 1/2 |
| `dapo` | $1/110$ | $1/110$ | 10/110 vs 100/110 |
| `dr_grpo` ($L_{\max}=100$) | $1/200$ | $1/200$ | 10/200 vs 100/200 |

Before the $1/G$ average, `grpo` gives $1/10$ vs $1/100$: **each wrong token is penalized ten times
less than each right token is rewarded.** Check 13.3 measures these weights with autograd.

```python finetuning-check
import torch

torch.manual_seed(0)
G, L_max = 2, 100
lens = torch.tensor([10, 100])
adv = torch.tensor([1.0, -1.0])
mask = (torch.arange(L_max).unsqueeze(0) < lens.unsqueeze(1)).float()   # (G, L_max)

def grads(loss_type):
    logp = (-torch.rand(G, L_max)).requires_grad_(True)                 # (G, L_max)
    per_token = -adv.unsqueeze(1) * logp                                 # ratio = 1 case
    if loss_type == "grpo":
        loss = ((per_token * mask).sum(1) / mask.sum(1)).mean()
    elif loss_type == "dapo":
        loss = (per_token * mask).sum() / mask.sum()
    elif loss_type == "dr_grpo":
        loss = (per_token * mask).sum() / (G * L_max)
    loss.backward()
    g = -logp.grad                      # gradient ascent direction on log-probs
    return g[0, 0].item(), g[1, 0].item(), g

g_pos, g_neg, g = grads("grpo")
assert abs(g_pos - 1 / 20) < 1e-7 and abs(g_neg + 1 / 200) < 1e-7
assert abs(G * g_pos - 1 / 10) < 1e-7 and abs(G * g_neg + 1 / 100) < 1e-7   # per sequence: 1/10 vs 1/100
assert abs(g_pos / -g_neg - 10) < 1e-4

g_pos, g_neg, g = grads("dapo")
assert abs(g_pos - 1 / 110) < 1e-7 and abs(g_neg + 1 / 110) < 1e-7

g_pos, g_neg, g = grads("dr_grpo")
assert abs(g_pos - 1 / 200) < 1e-7 and abs(g_neg + 1 / 200) < 1e-7
assert float(g[0, 10:].abs().sum()) == 0.0          # padding gets no gradient
print("aggregation weights verified")
```

The check fixes $\rho=1$ and two sequences; it does not show length *dynamics* over hundreds of
steps. That needs a run (lab L13.2).

### Two misconceptions corrected

**"GRPO's advantages are unbiased."** The leave-one-out baseline is; group-mean centering nearly is
(it shrinks advantages by $(G-1)/G$ because it includes $r_i$). But std scaling and per-sequence
length normalization change *which* prompts and tokens receive gradient. Those are biases in the
update, and Dr. GRPO measured their effect on length.

**"Longer responses mean better reasoning."** Length also grows when `grpo` aggregation dilutes the
penalty on long failures, or when verbosity games a reward. Split length **by correctness** (log it
with `log_metric`) and check that held-out accuracy moves with it.

---

## 13.4 Sequence-level ratios and newer objectives

### GSPO

GRPO's reward is per *sequence*, but its ratio and clipping are per *token*. Token ratios are noisy,
and more so in MoE models, where one step can re-route a token to different experts so its ratio
jumps while the sequence barely changed (routing: [Transformers module 12](../transformers/12-mixture-of-experts.md)).
GSPO ([Zheng et al., 2025](https://huggingface.co/papers/2507.18071), Qwen) uses one length-normalized
ratio per sequence, clipped once:

$$
s_i(\theta) = \left(\frac{\pi_\theta(o_i\mid q)}{\pi_{\text{old}}(o_i\mid q)}\right)^{1/|o_i|}
= \exp\Big(\frac{1}{|o_i|}\sum_t \log\rho_{i,t}\Big),
$$

the geometric mean of the token ratios. Qwen used it to stabilize RL on Qwen3 MoE models. In TRL:
`importance_sampling_level="sequence"` with `loss_type="grpo"` and the paper's tiny clip range
(`epsilon=3e-4`, `epsilon_high=4e-4`), since a geometric mean moves far less than single ratios. It
differs from GRPO only off-policy (`steps_per_generation > gradient_accumulation_steps` or
`num_iterations > 1`). The experimental GSPO-token trainer was removed in v1.14.

### Clipping the weight, not the update: CISPO and SAPO

Hard clipping zeroes a token's gradient once its ratio leaves the band, and in long chain-of-thought
the rare "fork" tokens (*wait*, *however*) are exactly the ones whose ratios move most. **CISPO**
([MiniMax-M1 report, 2025](https://huggingface.co/papers/2506.13585)) clips the importance weight
under a stop-gradient and uses a REINFORCE-style loss:

$$
\ell_{i,t} = -\,\operatorname{sg}\big(\min(\rho_{i,t}, \epsilon_{\max})\big)\,\hat A_i\,\log\pi_\theta(o_{i,t}).
$$

Every token keeps a gradient; only its weight is capped. TRL: `loss_type="cispo"` with `epsilon_high`
as $\epsilon_{\max}$ (ScaleRL uses 5.0). **SAPO** ([Gao et al., 2025](https://huggingface.co/papers/2511.20347),
Qwen) replaces the band with a sigmoid gate $f(\rho)=\sigma(\tau(\rho-1))\cdot4/\tau$, slightly
sharper for negative advantages (`sapo_temperature_neg=1.05`, `sapo_temperature_pos=1.0`), so
gradients fade instead of cutting off.

### Table 2: the 2025–26 objectives at survey level

| Objective | What it changes | Problem addressed | TRL (v1.15.0) |
|---|---|---|---|
| GSPO | sequence-level ratio (geometric mean), clipped per sequence | noisy token ratios, MoE instability | `importance_sampling_level="sequence"` |
| CISPO | clip IS weight under stop-gradient; REINFORCE loss | clipped tokens lose all gradient | `loss_type="cispo"`, `epsilon_high≈5` |
| SAPO | soft sigmoid gate instead of hard clip | brittle hard trust region | `loss_type="sapo"` |
| GDPO | normalize **each reward function** per group, then sum | multi-reward advantage collapse | `multi_objective_aggregation="normalize_then_sum"` |
| Lite PPO | group mean, batch std, token-level loss | robust critic-free scaling | `scale_rewards="batch"`, `loss_type="dapo"` |
| RLOO / REINFORCE++ | leave-one-out baseline / global batch normalization | variance without a critic | `RLOOTrainer` (stable) |

Sources: the [TRL paper index](https://huggingface.co/docs/trl/v1.15.0/en/paper_index), which lists
more (GMPO, LUSPO, VESPO, A\*-PO); GDPO is [2601.05242](https://huggingface.co/papers/2601.05242).
Treat the rows after GSPO as responses to a specific symptom, not defaults; most were shown on one or
two model families. GDPO matters for the running task: summing a schema reward in $\{0,1\}$ with a
field-accuracy reward in $\{0,0.2,\dots,1\}$ when nearly every sample is valid JSON makes the group
std almost all accuracy, so a rare schema failure barely registers. Normalizing each first keeps both.

### ScaleRL: what scales and what only speeds up

[Khatri et al. (2025)](https://huggingface.co/papers/2510.13786) spent over 400,000 GPU-hours
fitting pass rate against compute with a sigmoid whose asymptote $A$ is the ceiling and exponent $B$
the efficiency. Loss type (CISPO and GSPO above DAPO), an FP32 LM head, generation length and batch
size raised $A$; aggregation, normalization, curriculum and the off-policy algorithm mostly changed
$B$. **A choice that wins at 1,000 GPU-hours may only reach the same ceiling sooner.** TRL's
approximation: `loss_type="cispo"`, `epsilon_high=5.0`, `scale_rewards="batch"`,
`num_generations=16`, `cast_lm_head_to_fp32=True`.

---

## 13.5 Entropy dynamics

**Entropy collapse** is the commonest silent failure: per-token entropy falls fast in the first
hundred steps, groups become identical, `frac_reward_zero_std` climbs, and learning stops without
an error. Clip-higher is the first defense. TRL adds:

- **Entropy bonus:** `entropy_coef > 0` subtracts $\alpha\mathcal{H}$ (mean per-token entropy).
- **Adaptive entropy** (Skywork-OR1): `use_adaptive_entropy=True` raises the coefficient while
  entropy is at or below `entropy_target` and lowers it otherwise. The default target (0.2 nats) is
  Skywork-OR1's value; per-token entropy in RL runs is typically ~0.2–1 nats and falls, so check
  your early `entropy` log before choosing a target.
- **High-entropy minority tokens:** [Wang et al. (2025)](https://huggingface.co/papers/2506.01939),
  "Beyond the 80/20 Rule", found about 20% of CoT tokens are high-entropy forks and that RLVR mostly
  adjusts those; restricting the gradient to the top 20% matched full updates on Qwen3-8B and beat
  them on 14B and 32B. TRL: set `top_entropy_quantile=0.2` (default 1.0, i.e. all tokens;
  [L861](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L861-L870)).

Monitor `entropy` (a gradual decline is fine; a cliff in the first 5% of steps is collapse).

---

## 13.6 Rollout infrastructure

**Generation dominates an RL step.** TRL's docs call it "often the main bottleneck"; practitioner
reports for long-CoT runs put it at 70–90% of wall-clock (no single primary source). Decode is
memory-bound, which is why vLLM with
[continuous batching](/courses/inference/01-the-core-loop/01-03-continuous-batching.html) is used. Cost planning: [module 20](./20-production-serving-and-cost.md).

**Colocate** (TRL default when `use_vllm=True`): vLLM runs in the trainer process on the same GPUs,
`vllm_gpu_memory_utilization` (default 0.3) reserves its share, and weights are copied in memory after
each step. `vllm_enable_sleep_mode=True` offloads vLLM's weights and KV cache during the optimizer
step at some wake-up latency. Right for a single GPU (T2 and below).

**Server** (`vllm_mode="server"`): vLLM on separate GPUs, weights pushed over **NCCL** after each
update. As of v1.15.0 the docs use vLLM's own CLI:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs --max-logprobs -1
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch train_grpo.py   # GRPOConfig(use_vllm=True, vllm_mode="server")
```

`trl vllm-serve` (which since v1.11 wraps vLLM's server) is now **deprecated** per the
[vLLM integration page](https://huggingface.co/docs/trl/v1.15.0/en/vllm_integration): it just builds
and prints that `vllm serve` command. TRL supports a moving window of vLLM versions, so pin both.
Internals: [vLLM repo map](/courses/inference/11-vllm-deep-dive/11-01-repo-map-and-v1.html);
multi-node layouts: [module 10](./10-distributed-fine-tuning-and-moe.md).

**Async GRPO** ([`trl.experimental.async_grpo.AsyncGRPOTrainer`](https://huggingface.co/docs/trl/v1.15.0/en/async_grpo_trainer)):
a background worker streams completions from a vLLM server, scores them, computes advantages and
fills a queue that the trainer consumes, so generation and training overlap. Weights sync over NCCL
every `weight_sync_steps`; a sample more than `max_staleness` updates old is discarded. Reward
functions must be picklable (the worker is a CPU process) and distribution requires FSDP2. The price
is **off-policy data by construction**, so 13.7's importance correction becomes essential.

---

## 13.7 Training–inference mismatch

Give vLLM and your trainer the same weights and tokens and they will still disagree on log-probs:
different kernels, precision (bf16 accumulations, FP8 KV caches), and batch-composition dependence
([batch invariance](/courses/inference/10-benchmarking/10-04-batch-invariance.html)). So even with
$\mu=1$ you sample from $\pi^{\text{vllm}}$ and differentiate $\pi^{\text{train}}$:

$$
\nabla J_{\text{biased}} = \mathbb{E}_{y\sim\pi^{\text{vllm}}}\big[\hat A\,\nabla\log\pi^{\text{train}}(y)\big].
$$

**"On-policy" GRPO is silently off-policy.** The gap is reported to grow with long sequences, MoE
and FP8, and to precede collapses ([Yao et al. blog](https://fengyao.notion.site/off-policy-rl);
the Ling Team's [Ring-1T report](https://huggingface.co/papers/2510.18855) introduces IcePop
masking). The fix is importance sampling with $\rho=\pi^{\text{train}}_{\text{old}}/\pi^{\text{vllm}}$
(IS basics: [probability note](../../notes/math/probability.md)), bounded:

- **TIS** (truncated): weight by $\min(\rho,C)$; TRL generalizes to $\operatorname{clip}(\rho,C_{\min},C_{\max})$.
- **MIS** (masked): weight by $\rho\cdot\mathbb{1}[C_{\min}\le\rho\le C_{\max}]$; outliers are dropped, not shrunk.

TRL's exact rule ([`grpo_trainer.py` L2586–L2633](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_trainer.py#L2586-L2633),
[config L920–L957](https://github.com/huggingface/trl/blob/dae7b6adc8d2729cfa8d21bc8480656ede769240/trl/trainer/grpo_config.py#L920-L957)):
`vllm_importance_sampling_correction=True` by default; `vllm_importance_sampling_mode` is one of
`token_truncate`, `token_mask`, `sequence_truncate`, `sequence_mask` (**default**). In sequence modes
the ratio is the **full product** of token ratios, $\exp\sum_t(\log\pi^{\text{train}}_t-\log\pi^{\text{vllm}}_t)$:
the sequence-level IS weight, not GSPO's geometric mean. Defaults: `vllm_importance_sampling_clip_max=3.0`,
`clip_min=None` (no lower bound). The weight multiplies the per-token loss as a constant. Watch
`sampling/sampling_logp_difference/mean` and `sampling/importance_sampling_ratio/{min,mean,max}`; a
rising gap is a precision or infrastructure bug before it is an algorithm problem.

Check 13.4 runs clipping, clip-higher, GSPO and both IS corrections on the same numbers.

```python finetuning-check
import math
import torch

torch.manual_seed(0)
eps_low, eps_high = 0.2, 0.28
ratios = torch.tensor([0.7, 0.95, 1.1, 1.35])

def surrogate_grad(rho, A):
    rho = torch.tensor(rho, requires_grad=True)
    obj = torch.min(rho * A, rho.clamp(1 - eps_low, 1 + eps_high) * A)
    obj.backward()
    return rho.grad.item()

zero = {(r, A) for r in ratios.tolist() for A in (1.0, -1.0) if surrogate_grad(r, A) == 0.0}
assert zero == {(ratios[3].item(), 1.0), (ratios[0].item(), -1.0)}, zero   # (1.35, A>0) and (0.70, A<0)
assert surrogate_grad(0.7, 1.0) == 1.0 and surrogate_grad(1.35, -1.0) == -1.0

# GSPO: sequence ratio = geometric mean of token ratios, clipped once
s = torch.exp(ratios.double().log().mean())                # f64, compared with a float64 product
geo = math.prod(ratios.tolist()) ** (1 / 4)
assert abs(s.item() - geo) < 1e-12 and abs(s.item() - 0.99686) < 1e-4
assert 1 - eps_low <= s.item() <= 1 + eps_high       # sequence not clipped, though token 1.35 would be

# training-inference IS correction on a fixture of log-probs (one 4-token sequence)
lp_vllm = torch.tensor([-2.0, -1.0, -0.5, -3.0])
rho_tok = torch.tensor([0.4, 0.9, 1.5, 2.6])
lp_train = lp_vllm + rho_tok.log()
rho = torch.exp(lp_train - lp_vllm)
torch.testing.assert_close(rho, rho_tok)
tis = torch.clamp(rho, max=2.0)                                       # TIS: min(rho, C)
torch.testing.assert_close(tis, torch.tensor([0.4, 0.9, 1.5, 2.0]))
mis = rho * ((rho >= 0.5) & (rho <= 2.0))                             # MIS band [0.5, 2.0]
torch.testing.assert_close(mis, torch.tensor([0.0, 0.9, 1.5, 0.0]))
# TRL default: sequence_mask, ratio = product of token ratios, C_max = 3.0, no lower bound
rho_seq = torch.exp((lp_train - lp_vllm).sum())
assert abs(rho_seq.item() - 1.404) < 1e-3 and rho_seq.item() <= 3.0   # kept
print("clipping, GSPO and IS corrections verified")
```

The check pins each rule on fixed numbers; it does not tell you which $C$ suits your model. Start from
TRL's defaults and watch the logged ratio statistics.

---

## 13.8 Reward design

A verifier maps (prompt, completion, gold data) to a number. The useful kinds:

- **Math.** Extract the final answer (last `\boxed{...}`, or the number after GSM8K's `####`),
  normalize commas, currency and trailing periods, compare numerically, or **symbolically** for
  expressions: parse both sides into a computer-algebra form (SymPy-style) and test whether the
  difference simplifies to zero, as libraries like `math-verify` do. Never fall back to "the last
  number in the text"; the model learns to end with a guess.
- **Code.** Hidden unit tests in a sandbox with time and memory limits and no network; sandbox and
  environment infrastructure (OpenEnv, tool rollouts) is [module 15](./15-agentic-and-tool-use-fine-tuning.md).
- **Format.** A small, saturating bonus (0.1–0.2) for `<think>…</think>` plus a boxed answer;
  larger and the model optimizes format over correctness.
- **Length.** DAPO's soft overlong punishment, or a per-token cost past a budget.
- **Several functions** via `reward_weights` (summed, or per-function normalized with GDPO).
- **Non-verifiable tasks.** LLM judges or reward models as reward functions; over-optimization, RM
  ensembles, process reward models and iterative/online DPO are in [module 12](./12-preference-optimization.md).

**The TRL contract** (v1.15.0 [docs](https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer)): a
reward function receives keyword arguments `prompts`, `completions`, `completion_ids`, `trainer_state`,
`log_extra`, `log_metric` and **every dataset column except `prompt`**, and returns
`list[float | None]`. `None` means "not applicable": TRL turns it into NaN and ignores it, and a
sample for which every function returns `None` gets advantage 0. `async def` functions are awaited
concurrently. Conversational completions arrive as lists of message dicts.

Teaching fragment (not executed; tested versions, plus field-accuracy, GSM8K and code rewards, in
[`reward_functions.py`](./code/reward_functions.py)): the running task's schema reward. It reuses
module 01's parser and enums unchanged, so 1.0 means exactly "schema-valid" under the eval scorer.

```python
import re
FIELDS = ["customer_name", "order_id", "issue_type", "refund_requested", "sentiment"]
ISSUE_TYPES = ["refund", "shipping", "defect", "account", "other"]
SENTIMENTS = ["positive", "neutral", "negative"]
ORDER_ID_RE = r"^ORD-[0-9]{5}$"
# parse_prediction: strip, unwrap one ```json fence, json.loads, dict or None (module 01, verbatim)
from fit_or_not_baseline import parse_prediction

def schema_reward(completions, task, **kwargs):              # `task` is a dataset column
    out = []
    for c, t in zip(completions, task):
        if t != "email_json":
            out.append(None); continue                       # not my sample: skipped
        o = parse_prediction(c[-1]["content"] if isinstance(c, list) else c)
        ok = (o is not None and set(o) == set(FIELDS) and isinstance(o["refund_requested"], bool)
              and all(isinstance(o[k], str) for k in FIELDS if k != "refund_requested")
              and re.match(ORDER_ID_RE, o["order_id"]) is not None
              and o["issue_type"] in ISSUE_TYPES and o["sentiment"] in SENTIMENTS)
        out.append(1.0 if ok else 0.0)
    return out
```

For GSM8K, `openai/gsm8k`'s `answer` column is the full rationale ending in `#### 72`. Extract the
gold with `answer.split("####")[-1].strip()` in preprocessing (as `grpo_from_scratch.py` does) or in
the reward (as `reward_functions.py` does); comparing against the raw column scores 0 on every row.

---

## 13.9 Reward hacking

**A verifier does not make reward hacking impossible; it moves the hack to whatever the verifier
fails to check.** [Weng (2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
surveys the phenomenon. Under RLVR the recurring patterns are:

| Hack | What the model does | Countermeasure |
|---|---|---|
| Format gaming | `<think>` scaffold with empty reasoning; several boxed answers | require content; score only the last answer; cap format reward |
| Answer-parser exploits | hedges ("12 or 13"), ends with a number, beats a loose regex | strict extraction, no last-number fallback, symbolic equivalence |
| Test tampering | `sys.exit(0)` or `os._exit(0)` before tests, catches `AssertionError`, edits tests | harness owns control flow; unforgeable success sentinel; read-only tests |
| Judge exploitation | flattery, confident tone, length aimed at an LLM judge | rubric judges, length control, ensembles, a held-out judge |
| Length inflation | longer outputs where length correlates with reward | length-split metrics, overlong shaping, `dr_grpo` |

**Monitoring.** Read sample tables (`log_completions=True`). Keep a **held-out verifier** the policy
never trains against: training reward rising while held-out reward stays flat is the signature of
hacking. Monitor the chain of thought for stated intent ("I'll just make the tests pass"), and flag
tampering patterns in code. The lab's `detect_hacks` does this: an "exit code 0" verifier pays
`sys.exit(0)` and `os._exit(0)`, and a harness that owns control flow and checks an unforgeable
sentinel does not. Narrow RL that rewards cheating can generalize into broader misbehavior; the
safety side, including emergent misalignment, is [module 19](./19-safety-forgetting-and-responsible-fine-tuning.md).

---

## 13.10 Data, curriculum and recipes

**Difficulty filtering.** Only mixed-outcome groups carry gradient, so estimate each prompt's
base-model pass rate with $k$ samples (8–16) and **keep $0<\hat p<1$**. With $G=8$ a group is
uninformative with probability $p^8+(1-p)^8$: 0.20 at $p=0.18$ but 0.92 at $p=0.01$. Re-estimate as
the policy improves and retire always-solved prompts (ScaleRL's no-positive-resampling). A
**curriculum** orders prompts easy to hard or keeps batch pass rates mid-range; `trainer_state` in
reward functions also lets you schedule rewards.

**Practical needs.** Unsloth's [RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
advises **at least 500 prompts**, waiting **at least 300 steps** for reward to rise (sometimes
1,000+), and **at least 1.5B parameters** for a model to produce thinking tokens. These are
practitioner guidance, not measured thresholds. Unsloth itself is covered in [module 08](./08-frameworks-and-tooling.md).

**R1-Zero vs R1.** [DeepSeek-R1](https://huggingface.co/papers/2501.12948) reported two recipes.
**R1-Zero** ran GRPO directly on DeepSeek-V3-Base with rule-based accuracy and format rewards:
reasoning length and scores grew and the authors highlighted an "aha moment" of self-reflection, but
outputs mixed languages and read poorly. **R1** added a cold-start SFT on a small set of long-CoT
examples, then reasoning RL (with a language-consistency reward), then rejection-sampling SFT, then
general RL. Tulu 3's open SFT → DPO → RLVR pipeline is the other reference
([module 01](./01-when-to-fine-tune.md)); distilling R1-style traces is [module 14](./14-distillation-and-reasoning.md).

**Be skeptical of the "aha moment".** [Liu et al. (2025)](https://huggingface.co/papers/2503.20783)
found self-reflection already present in base models, DeepSeek-V3-Base included, and some Qwen2.5
bases reasoning well with no template at all. RL raises the frequency of an existing behavior; its
mid-run appearance is not evidence of emergence.

**LoRA is enough for RL.** Thinking Machines' [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
reports that LoRA "fully matches the learning performance of FullFT when running policy gradient
algorithms for reinforcement learning, even with ranks as low as 1", because the advantage provides
"only O(1) bits per episode". Use all-linear targets and about 10× the full-FT learning rate
([module 06](./06-lora-and-qlora.md)).

---

## 13.11 What RL does and does not do

**RL mostly sharpens; whether it also expands is contested.** [Yue et al. (2025)](https://arxiv.org/abs/2504.13837)
measured pass@$k$ up to large $k$ on math, code and visual reasoning: RLVR models win at $k=1$, but
base models catch up and **overtake at large $k$**, and RL-solved problems are largely a subset of
what the base solves with enough samples. Their reading: RLVR improves sampling efficiency more than
it adds reasoning. Check your own model with module 09's unbiased pass@$k$ estimator
([module 09](./09-evaluating-fine-tuned-models.md)).

The rebuttals are real. [ProRL (Liu et al., 2025)](https://arxiv.org/abs/2505.24864) ran prolonged
RL with KL control, reference resets and a broad task mix, and reports RL beating the base across
pass@$k$, including tasks the base fails at any $k$. Short RLVR on narrow math mostly redistributes
probability; longer, broader runs can push past the base's coverage.

**"SFT memorizes, RL generalizes"** ([Chu et al., 2025](https://arxiv.org/abs/2501.17161)): on a
card game (GeneralPoints) and a navigation task (V-IRL), RL generalized to unseen rule and visual
variants while SFT degraded, yet SFT was **still needed first** to stabilize the output format.
The tasks are synthetic and narrow. **RL forgets less:** RL's Razor ([Shenfeld et al., 2025](https://arxiv.org/abs/2509.04259))
argues on-policy RL stays closer (in KL) to the base than SFT at equal task gains. On-policy
distillation exploits the same property ([module 14](./14-distillation-and-reasoning.md)); forgetting
belongs to [module 19](./19-safety-forgetting-and-responsible-fine-tuning.md).

---

## 13.12 Monitoring a GRPO run

### Table 3: what to watch

| Metric (TRL name) | Healthy | Pathological | Action |
|---|---|---|---|
| `reward`, `reward_std` | mean rises after warm-up; std well above 0 | flat at 0, or near max early | check verifier and difficulty; filter data |
| `rewards/<func>/mean` | correctness rises, format saturates early | format at max, correctness flat | shrink format weight; test the parser |
| `frac_reward_zero_std` | low (< 0.5), stable | climbing towards 1 | refresh/filter prompts; raise $G$ or temperature; check entropy |
| `completions/mean_length`, `mean_terminated_length` | moves with accuracy | grows while held-out accuracy is flat | `dapo`/`dr_grpo`, overlong shaping, length by correctness |
| `completions/clipped_ratio` | small (< 0.1) | large: answers truncated | raise `max_completion_length`; `mask_truncated_completions=True`; length penalty |
| `kl` (only if `beta > 0`) | slow growth | spikes | lower LR; check IS stats; raise `beta` |
| `entropy` | gradual decline | early cliff (collapse) or unbounded rise | `epsilon_high`, entropy bonus/adaptive; or lower LR |
| `clip_ratio/region_mean`, `low_mean`, `high_mean` | small fractions | large: too far off-policy | lower LR; fewer `num_iterations`/`steps_per_generation` |
| `sampling/sampling_logp_difference/mean` | small, stable | rising | fix precision/kernels (FP32 LM head); check vLLM version |
| `sampling/importance_sampling_ratio/{min,mean,max}` | mean ≈ 1 | many sequences masked (min 0, mean ≪ 1) | as above; revisit `clip_max` |

Names from TRL's [logged-metrics list](https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer).

### Debugging prompt 1: reward flat at zero for 200 steps

1. **Test the verifier.** Feed it gold answers in the model's output format (`reward_functions.py
   --full` does this for GSM8K). A parser that never matches is the commonest cause, and a template
   or thinking toggle that disagrees with the parser is the second.
2. **Check that the model ever succeeds.** `frac_reward_zero_std` near 1 with `reward` at 0 means
   every group is all-wrong: drop $\hat p=0$ prompts or start from an SFT checkpoint. A high
   `completions/clipped_ratio` means answers never reach the box: raise `max_completion_length`.
3. **Allow for a slow start.** Under ~300 steps on a small model a flat start is normal, provided
   `reward_std` is non-zero.

### Debugging prompt 2: length grows while accuracy plateaus

1. **Check the loss and truncation.** `grpo` dilutes the penalty on long failures: use `dapo` or
   `dr_grpo`. A rising `clipped_ratio` means hitting the cap: mask truncated completions or add the
   soft overlong punishment.
2. **Split length by correctness.** If only wrong answers lengthen, blame aggregation or a
   length-correlated reward, not "deeper reasoning".
3. **Read samples.** Repetition, hedged answers or padded think blocks mean hacking; repetitive loops
   with collapsed entropy call for clip-higher or a small entropy bonus. Confirm on held-out accuracy.

---

## 13.13 Lab: GRPO on GSM8K, and the loss-type ablation

**Goal:** see group advantages, aggregation bias, sequence-level ratios and reward hacking in runs
you control. **Tiers:** CPU smoke; T0–T3 full. **Model:** `Qwen/Qwen3-1.7B` (or 4B). **Data:**
[`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main`; gold after `####`).
Run the CPU versions first, from the repository root:

```bash
python content/courses/finetuning/code/grpo_from_scratch.py --smoke
python content/courses/finetuning/code/reward_functions.py --smoke
```

[`grpo_from_scratch.py`](./code/grpo_from_scratch.py) `--smoke` trains a tabular policy on 25 toy
additions under `grpo`, `dapo`, `dr_grpo` and a $k_3$ KL run, asserting reward above 0.80 and zero
gradient from a zero-variance group. [`reward_functions.py`](./code/reward_functions.py) `--smoke`
asserts extraction edge cases, `None` routing, module 01's scoring fixture, and that the hack
detector flags `sys.exit(0)` and `os._exit(0)` but not honest or merely wrong code.

### L13.1 (T0): Unsloth GRPO on Qwen3

Start from Unsloth's Qwen3 (4B) GRPO notebook (listed in its
[RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)), swap in the GSM8K
correctness and format rewards from `reward_functions.py`, and train with LoRA and colocated vLLM on
a free T4 (fp16 only). Record `reward`, per-function rewards and `completions/mean_length`. Expect
format reward to saturate within ~100 steps and correctness to rise more slowly; if it is flat past
~300 steps, work through debugging prompt 1.

### L13.2 (T1/T2): the loss-type × scale ablation

```bash
for lt in grpo dr_grpo dapo; do for sr in group none; do
  python content/courses/finetuning/code/grpo_from_scratch.py --full \
      --model Qwen/Qwen3-1.7B --loss-type $lt --scale-rewards $sr --max-steps 300 --report-to trackio
done; done
```

TRL `GRPOTrainer` with vLLM colocate, all-linear LoRA, `mask_truncated_completions=True`,
`max_completion_length=1024`. Record mean and terminated length, `clipped_ratio`,
`frac_reward_zero_std`, and accuracy on 200 held-out GSM8K test questions. Expect `grpo` to show the
most length growth on wrong answers and `dr_grpo`/`dapo` flatter length at similar accuracy. Compare
curves, not exact numbers.

### L13.3 (T2): GSPO vs token-level

Run `--is-level sequence`: the script switches to `loss_type="grpo"`, GSPO's
`epsilon=3e-4`/`epsilon_high=4e-4` and off-policy `steps_per_generation=8`. Compare a token-level run
with the same setting on a small MoE: `clip_ratio/*`, reward variance across seeds, collapses.

### L13.4 (T3): server mode, code RL, and a planted hack

Start `vllm serve` on separate GPUs (13.6), train on code with `hackable_code_reward` alone and
count `detect_hacks` flags. Switch to `robust_code_reward` in a per-rollout container
([module 15](./15-agentic-and-tool-use-fine-tuning.md)); the flag rate should fall. The hack is the
divergence between training reward and held-out pass rate.

### L13.5 (T0, analysis): pass@1 vs pass@64

Sample 64 completions per problem from the base and the RL checkpoint on a 100-problem MATH-500
subset and compute pass@1 and pass@64 with module 09's estimator. Expect the RL model to lead at
pass@1 and the gap to narrow, possibly reverse, at pass@64: the shape, not the numbers, of Yue et al.
Vision-language GRPO uses the same trainer ([module 16](./16-multimodal-and-embedding-fine-tuning.md)).

---

## Key takeaways

- **The baseline is what makes RL on LLMs work.** $R-b$ keeps the gradient unbiased and cuts variance; RLOO, GRPO and REINFORCE++ differ mainly in how they build $b$.
- **GRPO replaces PPO's critic with a group of samples**, saving a policy-sized value network and avoiding value learning from sparse terminal rewards.
- **Group advantages sum to zero, and identical rewards give zero gradient.** `frac_reward_zero_std` measures wasted compute.
- **The KL term is optional, not absent.** TRL defaults to `beta=0.0` (no reference model); when used, $k_3$ is unbiased and non-negative, and `use_bias_correction_kl` fixes its gradient.
- **Per-sequence length normalization biases length, and std scaling biases difficulty.** Under `grpo` a wrong 100-token answer is penalized 10× less per token than a right 10-token one is rewarded; TRL defaults to token-level `dapo`, `dr_grpo` removes the normalization bias, and `scale_rewards=False` drops the std.
- **Clip-higher protects exploration**; CISPO and SAPO go further by never fully zeroing a token's gradient.
- **Sequence-level ratios suit sequence-level rewards**: GSPO's geometric-mean ratio is the stable choice for MoE and off-policy updates.
- **Your sampler is not your trainer.** TRL applies masked sequence-level IS by default (`sequence_mask`, $C_{\max}=3$); watch the log-prob gap.
- **Generation is the bottleneck, and async rollouts make IS essential**: colocate for one GPU, server mode with NCCL sync for separate GPUs, async to overlap both at the cost of staleness.
- **Every verifier has a hole.** Strict parsers, harness-owned tests, small format rewards, a held-out verifier, and reading samples.
- **Filter prompts to $0<\hat p<1$ and expect a slow start** (Unsloth: 500+ prompts, 300+ steps).
- **RLVR mostly sharpens what the base can already do**; prolonged, broad RL may expand coverage. Claim only what your pass@$k$ evaluation shows.

## Self-check

1. How does GRPO compute advantages, how does it differ from PPO, and why can it drop the
   critic? What biases did Dr. GRPO identify, and how do DAPO and GSPO address instability?
2. What is the training–inference mismatch when rollouts come from vLLM, and how do truncated
   and masked importance sampling correct it?
3. Your GRPO run's reward is flat at zero for 200 steps; a second run's completion length keeps
   growing while accuracy plateaus. Diagnose both with specific metrics and fixes.

---

**Next → [14 — Distillation and Reasoning Fine-Tuning](./14-distillation-and-reasoning.md)**
