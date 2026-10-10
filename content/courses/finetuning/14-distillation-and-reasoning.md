# 14 — Distillation and Reasoning Fine-Tuning

> **Prerequisites:** modules 05, 13.
> **You will learn:** off-policy, logit and on-policy distillation, forward vs reverse KL, TRL's
> `DistillationTrainer`, cross-tokenizer distillation, and the engineering of
> thinking-format models and their length.

Most small reasoning models are distilled, not RL-trained from scratch: DeepSeek's six
R1-Distill models are SFT on R1 samples, and every small Qwen3 came from a larger Qwen3
teacher. Between SFT on teacher traces and full RL sits **on-policy distillation**, the
compute-efficient middle ground of 2025–26, and TRL promoted `DistillationTrainer` to
stable in 2026. Classic temperature distillation for classifiers is in the notes
([Transfer learning and fine-tuning](../../notes/deep-learning/transfer-learning-and-finetuning.md),
section "Knowledge distillation"); here we go from there to sequences and to `<think>` blocks.

---

## 14.1 Distillation objectives

### Temperature and the $T^2$ factor

Hinton et al. (2015) [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
soften both teacher logits $z_T$ and student logits $z_S$ with a temperature $T$ and match
the resulting distributions:

$$\mathcal{L}_{\text{KD}} = T^2\,\mathrm{KL}\big(\mathrm{softmax}(z_T/T)\,\|\,\mathrm{softmax}(z_S/T)\big)$$

($T$ is temperature in this section, not sequence length.) With $p = \mathrm{softmax}(z_T/T)$
and $q = \mathrm{softmax}(z_S/T)$, the gradient with respect to a student logit is

$$\frac{\partial\,\mathrm{KL}(p\|q)}{\partial z_{S,i}} = \frac{1}{T}\,(q_i - p_i).$$

Two factors of $1/T$ hide in that expression. One is explicit, from the chain rule through
$z_S/T$. The other is inside $q_i - p_i$: for large $T$ both distributions flatten towards
uniform, and $q_i - p_i \approx (z_{S,i} - z_{T,i})/(T\,V)$ after centring the logits. So the
raw gradient shrinks like $1/T^2$. **Multiplying by $T^2$ restores the gradient scale**, which
means you can change $T$ without retuning the learning rate or the weight against the
hard-label loss. Check 14.2 measures this.

For LLM distillation $T=1$ is the common choice (TRL's `DistillationConfig.temperature`
defaults to `1.0`); $T > 1$ helps when the teacher is sharply peaked and the runners-up
carry the information.

### Forward KL versus reverse KL

The two directions of KL ask different questions:

$$\mathrm{KL}(p_T\|q_S) = \sum_y p_T(y)\log\frac{p_T(y)}{q_S(y)} \qquad \mathrm{KL}(q_S\|p_T) = \sum_y q_S(y)\log\frac{q_S(y)}{p_T(y)}$$

**Forward KL is mass-covering (mean-seeking).** The expectation is under the teacher, so
any $y$ the teacher likes but the student nearly excludes makes $\log(p_T/q_S)$ explode.
A student too small for the teacher's modes spreads mass *between* them, onto outputs the
teacher thinks are bad.

**Reverse KL is mode-seeking.** The expectation is under the student. Missing a teacher
mode costs nothing; putting mass where the teacher has none explodes $\log(q_S/p_T)$. The
cheapest fix is to collapse onto one mode and cover it well, which for generation is often
what you want: a sample the teacher would write, not an average of two answers.

Check 14.1 makes this concrete. The teacher is a bimodal distribution on a grid of 41
points from $-4$ to $4$, two narrow bumps ($\sigma = 0.4$) at $-2$ and $+2$. The student is
a single discretised Gaussian with fixed $\sigma = 0.6$ and a free mean $\mu$, scanned
over $[-3, 3]$. Forward KL is minimised at $\mu = 0$, exactly between the modes, with value
4.99 nats. Reverse KL is minimised at $\mu = \pm 2$, on a mode, with value 0.91 nats
($\log 2 \approx 0.69$ for ignoring half the teacher's mass, plus 0.22 nats of width mismatch).

```python finetuning-check
import torch

x = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)

def bump(mu, s):
    return torch.exp(-0.5 * ((x - mu) / s) ** 2)

def normalise(t):
    return t / t.sum()

def kl(a, b):
    return float((a * (a.log() - b.log())).sum())

teacher = normalise(bump(-2.0, 0.4) + bump(2.0, 0.4))        # bimodal, (41,)
mus = torch.linspace(-3.0, 3.0, 121, dtype=torch.float64)
students = [normalise(bump(float(m), 0.6)) for m in mus]      # one Gaussian each

forward = torch.tensor([kl(teacher, q) for q in students])    # KL(p_T || q_S)
reverse = torch.tensor([kl(q, teacher) for q in students])    # KL(q_S || p_T)

mu_fwd = float(mus[forward.argmin()])
mu_rev = float(mus[reverse.argmin()])
assert abs(mu_fwd) < 0.25, mu_fwd                  # mass-covering: between the modes
assert abs(abs(mu_rev) - 2.0) < 0.25, mu_rev       # mode-seeking: on a mode
assert abs(float(forward.min()) - 4.99) < 0.01
assert abs(float(reverse.min()) - 0.91) < 0.01
assert bool((forward >= 0).all()) and bool((reverse >= 0).all())
assert abs(kl(teacher, teacher)) < 1e-12           # KL(p || p) = 0

```

The check proves the geometry on a one-dimensional toy; it does not prove which direction
gives a better *language model*, which depends on student capacity and on how you sample.

### Generalised JSD: one knob between the two

Agarwal et al. (2024) [GKD](https://huggingface.co/papers/2306.13649) interpolate with the
generalised Jensen–Shannon divergence. With $p = p_T$, $q = q_S$ and the mixture
$m = \beta p + (1-\beta)q$:

$$\mathrm{JSD}_\beta(p\|q) = \beta\,\mathrm{KL}(p\|m) + (1-\beta)\,\mathrm{KL}(q\|m)$$

This is TRL's convention too: `beta=0.0` is forward KL, `beta=1.0` is reverse KL, `0.5` is
the symmetric JSD ([DistillationTrainer docs, v1.15.0](https://huggingface.co/docs/trl/v1.15.0/en/distillation_trainer)).
Read the endpoints carefully. The formula itself goes to zero at $\beta \in \{0, 1\}$
(the mixture collapses onto one distribution), and near $\beta = 0$ it behaves like
$\beta\,\mathrm{KL}(p\|q)$, so implementations define the endpoints as the pure divergences.
At $\beta = 0.5$ the divergence is bounded:

$$0 \le \mathrm{JSD}_{0.5}(p\|q) \le \log 2,$$

with equality on the right exactly when $p$ and $q$ have disjoint supports. The bound
is the practical attraction: unlike either KL, JSD cannot blow up when the student puts
mass on a token the teacher considers impossible. Check 14.3 asserts both bounds.

### Sequence-level KD is forward KL by Monte Carlo

Kim and Rush (2016) [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
train the student on the teacher's *outputs* rather than its distributions: sample (or beam
search) $y \sim p_T(\cdot \mid x)$ and run plain SFT. Take the expectation of the SFT loss:

$$\mathbb{E}_{y \sim p_T}\big[-\log q_S(y \mid x)\big] = H\big(p_T(\cdot\mid x)\big) + \mathrm{KL}\big(p_T(\cdot\mid x)\,\|\,q_S(\cdot\mid x)\big)$$

The entropy term does not depend on the student, so **SFT on teacher samples is a Monte
Carlo estimate of sequence-level forward KL**, one sample at a time. That is what every
"distill R1 into a 7B" recipe does. It inherits forward KL's mass-covering behaviour and
adds a problem of its own, exposure bias (14.2).

Table 1 in 14.3 compares the three families: off-policy sequence KD, logit KD on fixed
data, and on-policy KD.


---

## 14.2 Off-policy reasoning distillation

### What the landmark recipes did

| Recipe | Student base | Data | Training | Headline result (as reported) |
|---|---|---|---|---|
| [DeepSeek-R1-Distill](https://huggingface.co/papers/2501.12948) (2025) | Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B, Llama-3.3-70B-Instruct | ~800k samples curated with R1 ([model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)) | SFT only | six dense models, 1.5B to 70B |
| [s1](https://arxiv.org/abs/2501.19393) (Muennighoff et al., 2025) | Qwen2.5-32B-Instruct | s1K: 1,000 questions filtered from 59,029 by quality, difficulty and diversity; traces from Gemini Flash Thinking | SFT, 26 minutes on 16 H100s | AIME24 50% → 57% with budget forcing |
| [LIMO](https://arxiv.org/abs/2502.03387) (Ye et al., 2025) | Qwen2.5-32B-Instruct | 800 curated problems with traces (v3 of the paper) | SFT | 63.3% AIME24, 95.6% MATH500, "1% of the training data" of prior approaches |
| [OpenThoughts](https://arxiv.org/abs/2506.04178) (Guha et al., 2025) | 7B | OpenThoughts3, 1.2M examples, QwQ-32B as teacher, 1,000+ controlled pipeline experiments | SFT | 53% AIME 2025, 51% LiveCodeBench, 54% GPQA Diamond |

The R1 recipe that produced the teacher (R1-Zero pure RL, cold-start SFT, RL) belongs to
[module 13](./13-rl-with-verifiable-rewards.md).

**What these results establish.** A strong model plus a few curated long chain-of-thought
traces switches on long-form reasoning: s1 and LIMO reach competitive AIME scores with
800 to 1,000 examples. Curation matters: in s1's ablations the full 59K pool gave no
substantial gain over the selected 1K, and single-criterion selection did much worse.

**What they do not establish.** Both small-data results start from a 32B *instruct* model
and measure a handful of competition benchmarks. They do not show that 1,000 examples
teach reasoning to a 1.5B base model, that gains transfer outside math, or that small data
beats large data; OpenThoughts shows scaling to 1.2M examples keeps paying at 7B. **Small
curated sets elicit capability that exists; large sets build it.** Which checkpoint to
start from (base, instruct, reasoning) is the [module 01](./01-when-to-fine-tune.md) decision.

### Exposure bias and compounding errors

Off-policy distillation trains on teacher prefixes: at step $t$ the student conditions on
a $y_{<t}$ the teacher wrote. At inference it conditions on its *own* prefix, and the two
prefix distributions match only if the student already matches the teacher.

Make it quantitative. Suppose the student leaves the teacher's distribution with
probability $\epsilon$ per token and never recovers. A $K$-token trace stays on-distribution
with probability $(1-\epsilon)^K$; with $\epsilon = 10^{-3}$ and $K = 4{,}000$ that is
$e^{-4} \approx 1.8\%$. Real models recover from many slips, so this is a worst case, but
the shape is right: **per-token errors compound with trace length.** The student never saw
a prefix containing its own mistakes, so it never learned to recover from them. This is
the "distribution mismatch" in GKD's abstract and the main argument for 14.3.

### Building the dataset

The off-policy pipeline is rejection sampling with a stronger model, which
[module 04](./04-data-curation-and-synthesis.md) covers in general:

1. Pick prompts with verifiable answers (boxed math answers, code with tests).
2. Sample $k$ traces per prompt from the teacher with vLLM in thinking mode at its
   recommended settings (Qwen3 thinking mode: temperature 0.6, top-p 0.95, top-k 20; the
   card warns against greedy decoding).
3. **Filter by correctness**: keep traces whose final answer verifies.
4. **Filter by form and length**: drop traces with no closing `</think>`, loops, and
   anything beyond your `max_length`. The kept length histogram becomes the student's
   prior on how long to think.
5. **Decontaminate** prompts against every benchmark you report
   ([module 04](./04-data-curation-and-synthesis.md)). A teacher that memorised a
   benchmark can also leak answers into its traces.

One trap in L14.1: `SFTConfig`'s default `max_length=1024` truncates long traces, cutting
off the answer and teaching the student to stop mid-thought. Set it to cover the histogram.

---

## 14.3 On-policy distillation

### The method

Thinking Machines (Lu et al., Oct 2025) [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
popularised the simplest version. For each prompt:

1. Sample a completion $y \sim q_S$ from the **student**.
2. Run the teacher on the same tokens (one forward pass, no generation) to get
   $\log p_T(y_t \mid y_{<t})$ at every position.
3. Set a per-token advantage equal to the negative per-token reverse KL estimate:

$$A_t = -\big(\log q_S(y_t \mid y_{<t}) - \log p_T(y_t \mid y_{<t})\big) = \log p_T(y_t \mid y_{<t}) - \log q_S(y_t \mid y_{<t})$$

4. Take a policy-gradient step with these advantages (the post feeds them to the same
   importance-sampled loss used for RL, with a discount factor of zero, so each token is
   credited only with its own term).

Why this optimises reverse KL: summing over the sampled path,

$$-\sum_t A_t = \sum_t \log\frac{q_S(y_t\mid y_{<t})}{p_T(y_t\mid y_{<t})} = \log\frac{q_S(y\mid x)}{p_T(y\mid x)},$$

and the expectation of that log-ratio under $y \sim q_S$ is exactly
$\mathrm{KL}(q_S\|p_T)$ at the sequence level. **Each sampled path is a single-sample
estimate of the negative reverse KL, decomposed token by token.** The score-function
identity $\mathbb{E}_{q}[\nabla \log q] = 0$ turns
$\mathbb{E}_q[A\,\nabla \log q_S]$ into $-\nabla\,\mathrm{KL}(q_S\|p_T)$, so the
REINFORCE step descends the reverse KL.

Token by token: where the student is **overconfident** relative to the teacher
($\log q_S > \log p_T$), $A_t < 0$ and the token is pushed down; where the teacher likes
it more, $A_t > 0$. Every token gets its own signal, whereas GRPO spreads one terminal
reward over a 4,000-token trace ([module 13](./13-rl-with-verifiable-rewards.md)).

Check 14.2 does the arithmetic on a six-token path. Student log-probs
$[-0.1, -2.0, -0.3, -1.5, -0.05, -0.7]$, teacher log-probs
$[-0.2, -0.5, -2.5, -1.4, -0.05, -3.0]$. The advantages are
$[-0.1, 1.5, -2.2, 0.1, 0.0, -2.3]$, they sum to $-3.0$, so this path is a 3.0-nat
single-sample estimate of the reverse KL. Tokens 0, 2 and 5 are where the student is
overconfident, and they are exactly the negative entries. The second half measures the
$T^2$ claim from 14.1 on five fixed logits: the plain-KL gradient norm is 0.770 at $T=1$
and 0.0425 at $T=4$ (18 times smaller), while the $T^2$-scaled gradient at $T=4$ is 0.680,
within 12% of the $T=1$ value.

```python finetuning-check
import torch

# --- Per-token advantages on one sampled path (6 tokens) ---
logq = torch.tensor([-0.1, -2.0, -0.3, -1.5, -0.05, -0.7], dtype=torch.float64)  # student
logp = torch.tensor([-0.2, -0.5, -2.5, -1.4, -0.05, -3.0], dtype=torch.float64)  # teacher

adv = logp - logq                                   # A_t = -(log q_S - log p_T)
torch.testing.assert_close(
    adv, torch.tensor([-0.1, 1.5, -2.2, 0.1, 0.0, -2.3], dtype=torch.float64),
    atol=1e-12, rtol=0)
seq_log_ratio = float(logq.sum() - logp.sum())      # log q_S(y) - log p_T(y)
assert abs(-float(adv.sum()) - seq_log_ratio) < 1e-12
assert abs(seq_log_ratio - 3.0) < 1e-12             # one-sample reverse-KL estimate
overconfident = logq > logp
assert overconfident.tolist() == [True, False, True, False, False, True]
assert bool((adv[overconfident] < 0).all())
assert bool((adv[~overconfident] >= 0).all())

# --- Temperature KD: T^2 keeps the gradient scale ---
z_t = torch.tensor([3.0, 1.0, 0.2, -1.0, -2.0], dtype=torch.float64)
z_s0 = torch.tensor([1.0, 2.0, 0.0, 0.5, -1.0], dtype=torch.float64)

def grad_norm(T, scale):
    z_s = z_s0.clone().requires_grad_(True)
    p = torch.softmax(z_t / T, dim=0)
    log_q = torch.log_softmax(z_s / T, dim=0)
    loss = scale * (p * (p.log() - log_q)).sum()
    loss.backward()
    # analytic gradient of KL: (q - p) / T
    torch.testing.assert_close(z_s.grad, scale * (log_q.exp() - p).detach() / T)
    return float(z_s.grad.norm())

g1 = grad_norm(1.0, 1.0)
g4_plain = grad_norm(4.0, 1.0)
g4_scaled = grad_norm(4.0, 16.0)
assert abs(g1 - 0.770) < 1e-3 and abs(g4_plain - 0.0425) < 1e-3 and abs(g4_scaled - 0.680) < 1e-3
assert 0.2 <= g4_scaled / g1 <= 5.0                 # same order of magnitude
assert g1 / g4_plain >= 10.0                        # plain KL at T=4 is >= 10x smaller
```

The check proves the advantage algebra and the gradient scaling; it does not show that a
policy-gradient update on these advantages converges, which the lab's toy loop
(`kd_losses.py --smoke`) demonstrates on an 8-token categorical student.

### GKD and MiniLLM: the predecessors

**GKD** (Agarwal et al.) adds `lmbda`, the fraction of batches built from student samples
(`0.0` is logit KD on dataset sequences, `1.0` fully on-policy), to the JSD `beta`; per the
[TRL GKD docs](https://huggingface.co/docs/trl/v1.15.0/en/gkd_trainer), high `lmbda` works
better and the best `beta` is task-dependent. GKD needs the teacher's full next-token
distribution; the Thinking Machines recipe needs only the log-prob of the sampled token, so
it fits an RL codebase unchanged. **MiniLLM** (Gu et al., 2023, [paper](https://huggingface.co/papers/2306.08543))
first minimised sequence-level reverse KL with a policy gradient; TRL removed its trainer in
[v1.15](https://github.com/huggingface/trl/releases/tag/v1.15.0). Use `DistillationTrainer` with `beta=1.0`.

### Compute economics

On-policy distillation pays for student sampling (like RL) and a teacher forward pass (like
logit KD), but it gets a dense signal and needs no reward function. Two published
comparisons:

| Source | Setup | Result |
|---|---|---|
| [Qwen3 report](https://arxiv.org/abs/2505.09388), Table 21 | Qwen3-8B student, starting from an off-policy distilled checkpoint (AIME'24 55.0) | RL: 17,920 GPU-hours, AIME'24 67.6, AIME'25 55.5. On-policy distillation: 1,800 GPU-hours, AIME'24 74.4, AIME'25 65.5 |
| [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) | Qwen3-8B-Base student, SFT-initialised on 400K OpenThoughts-3 prompts (60% AIME'24); target 70% | OPD reaches ~70% in ~150 steps. Versus SFT extrapolated to 2M prompts: ~9× fewer FLOPs if the SFT data already exists (~18× in GPU-hours), ~30× counting the teacher's cost of generating it |

Read the conditions. The Qwen3 comparison is one student and one teacher family (it also
reports OPD improving pass@64 where RL did not; see the "does RL expand reasoning?" debate
in [module 13](./13-rl-with-verifiable-rewards.md)). The Thinking Machines SFT figure at
2M prompts is extrapolated, and the teacher actually used was Qwen3-8B with FLOPs counted
as if 32B. Neither shows OPD beating RL when **no better teacher exists**: OPD needs a
teacher, RL needs a verifier.

**Table 1. SFT on teacher traces vs on-policy distillation vs RL**

| | Off-policy SFT (sequence KD) | On-policy distillation | RL with verifiable rewards |
|---|---|---|---|
| Who samples | teacher, once, offline | student, every step | student, every step |
| Signal density | dense (every target token) | dense (every sampled token) | sparse (one reward per sequence) |
| Sampling cost | teacher generation, amortised and reusable | student generation each step | student generation each step, usually $G$ samples per prompt |
| Teacher cost | full generation | one forward pass over student tokens | none (verifier or reward model instead) |
| Exposure bias | yes | no | no |
| Needs | a better teacher, a filter | a better teacher with the same tokenizer (or GOLD) | a verifier or reward model |
| Typical use | cold start, bootstrap thinking format | after SFT: close the gap to the teacher cheaply | push past every available teacher |

The practical order follows: **SFT first** to put the student near the teacher (a student
that cannot reason samples garbage for the teacher to score), **OPD** to close the gap,
**RL** only to exceed the teacher. Qwen3's small models follow the first two steps.

---

## 14.4 TRL's distillation trainers

### `DistillationTrainer` (stable since v1.10)

`DistillationTrainer` and `DistillationConfig` graduated from `trl.experimental` in
[v1.10.0](https://github.com/huggingface/trl/releases/tag/v1.10.0) (August 2026), which
also added VLM support; tool calling arrived in
[v1.11.0](https://github.com/huggingface/trl/releases/tag/v1.11.0). As of v1.15 it is
**always on-policy**: the dataset is prompt-only, the student generates every completion,
and the loss is the generalised JSD over the teacher's full next-token distribution,
computed in chunks so the vocabulary-by-length logits tensor is never materialised.
The knobs that matter, as of the
[v1.15.0 docs](https://huggingface.co/docs/trl/v1.15.0/en/distillation_trainer):

| Parameter | Default | What it does |
|---|---|---|
| `teacher_model` (trainer arg) / `teacher_model_name_or_path` | none | teacher; **must share the student's vocabulary** |
| `teacher_model_init_kwargs` | none | e.g. `{"dtype": "bfloat16"}` for the teacher |
| `beta` | `1.0` | JSD interpolation: `0.0` forward KL, `1.0` reverse KL, `0.5` JSD. Not a KL penalty against a reference model |
| `temperature` | `1.0` | used for sampling **and** for the loss |
| `max_completion_length` | `512` | far too short for thinking traces; set it explicitly |
| `learning_rate` | `1e-6` | lower than the `TrainingArguments` default of `5e-5` |
| `use_vllm`, `vllm_mode` | `False`, `"colocate"` | vLLM generation for the student; `"server"` for a separate GPU |
| `chat_template_kwargs` | none | e.g. `{"enable_thinking": True}` for Qwen3 students |
| `max_tool_calling_iterations` | none (no cap) | agent distillation; tool-result tokens are masked out of the loss |
| `gradient_checkpointing`, `bf16` | `True`, `True` unless `fp16` | memory defaults |

Mind the `beta` collision: in GRPO it weights a KL penalty to a reference model (default
0.0; the k3 estimator and bias correction are in [module 13](./13-rl-with-verifiable-rewards.md)).
Here it selects the divergence, so `beta=1.0` is not a "huge KL penalty".

Teaching fragment (not executed; TRL v1.15, Transformers v5):

```python
from datasets import load_dataset
from trl import DistillationConfig, DistillationTrainer

# Prompt-only conversational data: {"prompt": [{"role": "user", "content": "..."}]}
train = load_dataset("json", data_files="math_prompts.jsonl", split="train")

args = DistillationConfig(
    output_dir="qwen3-1.7b-opd",
    beta=1.0,                                   # reverse KL (Thinking Machines / MOPD style)
    temperature=1.0,
    max_completion_length=4096,                 # default 512 truncates thinking
    chat_template_kwargs={"enable_thinking": True},
    teacher_model_init_kwargs={"dtype": "bfloat16"},
    model_init_kwargs={"dtype": "bfloat16"},    # TRL trainers otherwise load float32
    learning_rate=1e-6,
    use_vllm=True,                              # colocated vLLM samples the student
)
trainer = DistillationTrainer(model="Qwen/Qwen3-1.7B", teacher_model="Qwen/Qwen3-8B",
                              args=args, train_dataset=train)   # same Qwen3 vocabulary
trainer.train()
```

The same run from the shell is `trl distillation --model_name_or_path ... --teacher_model_name_or_path ...`.
With PEFT, adapters on `lm_head` are rejected because the loss reads `lm_head.weight`
directly; use `modules_to_save=["lm_head"]` if you must train the head.

### `AsyncDistillationTrainer` and multi-teacher OPD

`trl.experimental.async_distillation.AsyncDistillationTrainer` (v1.11) is built like
AsyncGRPO: a background worker generates student rollouts on a vLLM server, and the teacher
is **never loaded locally**, only reached over HTTP
([docs](https://huggingface.co/docs/trl/v1.15.0/en/async_distillation_trainer)). Its `beta`
defaults to `0.0` (forward KL), and the teacher returns only a top-k slice
(`teacher_top_k`, default 8), so for `beta != 0` the divergence covers just the teacher's
top-1 token and the sampled token.

Multi-teacher on-policy distillation (MOPD, [paper](https://huggingface.co/papers/2606.30406))
is the reason to use it. Pass several entries in `teacher_server_urls`, e.g.
`{"math": "http://localhost:8002", "code": "http://localhost:8003"}`, and give every row a
`teacher_id` column. Each sample goes to exactly one teacher, never averaged; a missing or
unknown `teacher_id` raises. In the MOPD paper this is the last of three stages (general
SFT, one RL-trained expert per domain, then fusion with per-token reverse KL, so set
`beta=1.0`). Every teacher must share the student's tokenizer, since token ids travel
unconverted; a mismatch can train against the wrong tokens silently.

### The experimental zoo

As of TRL v1.15, `GOLDTrainer` (cross-tokenizer, 14.5), `SDFTTrainer`, `SDPOTrainer`,
`SSDTrainer`, `IWOPDTrainer` (14.6) and `GKDTrainer` live in `trl.experimental` and may
change without notice ([paper index](https://huggingface.co/docs/trl/v1.15.0/en/paper_index)).

**Default choice in October 2026:** `DistillationTrainer` for same-tokenizer distillation,
`AsyncDistillationTrainer` when the teacher is large or there are several,
`GOLDTrainer` only when the tokenizers differ. Use `GKDTrainer` only if you need
`lmbda < 1` mixing that the stable trainer does not offer.

---

## 14.5 Cross-tokenizer distillation

**Logit distillation needs a shared vocabulary.** A KL compares two distributions over the
same events: if teacher token 1,234 is `" Face"` and the student's is `"ing"`, or the
tokenizers split `"Hugging Face"` differently, positions and events no longer line up.
Sequence-level KD passes **text**, so off-policy SFT works across any pair of families.

GOLD (Hugging Face H4,
[Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/general-on-policy-logit-distillation))
builds on Universal Logit Distillation (ULD). As documented in the
[GOLD trainer docs](https://huggingface.co/docs/trl/v1.15.0/en/gold_trainer), it does two
things:

1. **Align spans by text.** Decode both token sequences incrementally and group tokens
   that cover the same visible text. When one student token corresponds to teacher tokens
   $[t_0, \dots, t_k]$, merge the teacher side by the chain rule: take the full
   distribution at $t_0$ and multiply it by the scalar conditional probabilities of the
   actual continuation tokens $t_1, \dots, t_k$. In the docs' example,
   $P(\text{"HF"}) = 0.6$ and $P(\text{"is"} \mid \text{"HF"}) = 0.9$ give a merged
   $P(\text{"HF is"}) = 0.54$. The merged distribution is unnormalised, which is fine for
   the next step.
2. **Compare distributions without a shared index.** ULD sorts each distribution's
   probabilities and takes an L1 distance between the sorted vectors, an optimal-transport
   style comparison that ignores token identity. With `uld_use_hybrid_loss=True`, tokens
   that exist in both vocabularies are compared directly (JSD) and only the unmatched rest
   falls back to sorted ULD.

The relevant `GOLDConfig` flags: `use_uld_loss=True`, `teacher_tokenizer_name_or_path`
(required for cross-tokenizer runs), `uld_use_hybrid_loss`, and the inherited GKD knobs
`lmbda` (default 0.5) and `beta` (default 0.5). **Use it when the teacher you want is from
a different family** (a Qwen teacher for a Llama, Gemma or LFM student). The sorted term
discards token identity, so it transfers confidence more than preferences; if an adequate
same-family teacher exists, prefer it and the exact loss.

---

## 14.6 Self-distillation and continual learning

Domain fine-tuning damages what the model already did well. In the Thinking Machines OPD
post, mid-training Qwen3-8B on internal documents (100% document mix) raised internal-QA
from 18% to 43% but dropped IF-eval from 85% to 45%. A 70/30 document/chat mix gave 36%
QA and 79% IF-eval. A short on-policy distillation phase with the **original Qwen3-8B as
teacher** on Tulu 3 prompts then brought IF-eval to 83% with QA at 41%: "recovers nearly
full performance on IF-eval without losing any knowledge". The teacher is the model's
earlier self, so this is self-distillation: the student only has to return to behaviour it
used to have, and on-policy samples show exactly where it drifted.

Shenfeld et al. (2026) [Self-Distillation Enables Continual Learning](https://huggingface.co/papers/2601.19897)
make the teacher the same model with extra context. SDFT puts a demonstration in the
teacher's prompt, samples from the student without it, and distils the teacher's
next-token predictions back. The paper reports that SDFT "consistently outperforms SFT,
achieving higher new-task accuracy while substantially reducing catastrophic forgetting",
and that sequential SDFT accumulates several skills without regression. TRL's
`SDFTTrainer` takes the extra information as a `privileged_context` column; the teacher
can be the initial model (`"base"`), the current model (`"live"`) or an EMA (`"ema"`).

**IW-OPD** ([position bias of OPD](https://huggingface.co/papers/2606.22600)) down-weights
later tokens once the student's prefix drifts from the teacher. **SDPO** uses the self-teacher
inside RLVR, conditioned on feedback text. **SSD** (`SSDTrainer`) is plain SFT on the model's
own unverified samples at a chosen temperature and truncation, used for code generation.

**Context distillation** (prompt baking; Snell et al. 2022,
[Learning by Distilling Context](https://arxiv.org/abs/2209.15189)) is the oldest form of the
same trick. The teacher is the model *with* a long system prompt or few-shot context, the
student is the same model *without* it, and training (SFT on the teacher's outputs, or token
KL as in 14.1) bakes the prompt's behaviour into the weights. It is the standard way to drop
a long system prompt at serve time; score it against the
[module 01](./01-when-to-fine-tune.md) prompting baseline it replaces.

Why on-policy methods forget less, the alternatives (replay, lower learning rate, LoRA)
and how to measure forgetting are in [module 19](./19-safety-forgetting-and-responsible-fine-tuning.md);
merging back towards the original weights is the training-free alternative
([module 18](./18-model-merging-and-composition.md)).

---

## 14.7 Engineering thinking models

### Formats: where the reasoning lives

Two conventions coexist. **Inline**: the assistant content holds `<think>...</think>` and
then the answer, with `<think>`/`</think>` as single special tokens (Qwen3 ids 151667 and
151668). **Structured**: the message carries a separate `reasoning_content` field and the
chat template decides how to render it. That is the chat-template input field;
[vLLM v0.31](https://docs.vllm.ai/en/v0.31.0/features/reasoning_outputs/) returns it as
`reasoning`, so rename when you turn served outputs into training rows.

The [Qwen3-8B chat template](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json)
shows every engineering decision in about 30 lines of Jinja:

- It accepts both conventions: it uses `message.reasoning_content` if present, else splits
  the content on `</think>`.
- It renders the `<think>` block **only for assistant turns after the last real user
  query**. Reasoning from earlier turns is dropped from the history. The model card says
  the same: history "should only include the final output part".
- With `enable_thinking=False`, the generation prompt ends with an empty
  `<think>\n\n</think>\n\n`, so the model starts directly with the answer.

Dropping old reasoning saves context but breaks **prefix preservation**: turn 1 renders
differently once turn 2 exists
([module 02](./02-tokens-templates-and-the-causal-lm-objective.md) covers loss masks and caching). Agent loops make it worse, since the model must still see its
reasoning after a tool result returns; TRL swaps in patched prefix-preserving templates for
Qwen3 and DeepSeek-V3 when tools are enabled. Reasoning before tool calls is in
[module 15](./15-agentic-and-tool-use-fine-tuning.md).

### Hybrid thinking: Qwen3's thinking mode fusion

The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) builds both modes into one
model in four stages: Long-CoT Cold Start, Reasoning RL, **Thinking Mode Fusion**, and
General RL. In the fusion stage the SFT data mixes thinking and non-thinking samples, with
`/think` and `/no_think` flags in the user query or system message. Non-thinking samples
keep an **empty thinking block**, so the output format is identical in both modes. In
multi-turn data the flags are inserted at random and the model follows the last one.

Per the [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B), `enable_thinking` in
`apply_chat_template` is a hard switch (default `True`); `/think` and `/no_think` are soft
switches that work only while it is `True`. When you fine-tune a hybrid model, **train
both modes or lose one**: pure thinking traces teach that every answer starts with long
reasoning, and the `enable_thinking=False` path degrades. Keep non-thinking examples
rendered with the empty block, and evaluate both modes.

### Thinking budgets and budget forcing

Muennighoff et al.'s **budget forcing** ([s1](https://arxiv.org/abs/2501.19393)) controls
thinking length at decode time with no retraining:

- **Cap.** When thinking reaches the budget $B$, stop, append the end-of-thinking
  delimiter (optionally followed by "Final Answer:"), and let the model answer.
- **Extend.** When the model emits the end delimiter before a minimum budget, suppress it,
  append "Wait", and continue. The model often re-checks and fixes its reasoning.

On AIME24 this is the 50% → 57% gain in the s1 row of 14.2's table. Qwen3 ships the cap half as
a product feature. When thinking hits the user's budget, generation halts and the
instruction "Considering the limited time by the user, I have to give the solution based
on the thinking directly now." is inserted; the report says this ability was not trained
explicitly but emerged from thinking mode fusion. Lilian Weng's
[Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) surveys the wider
test-time compute literature.

Check 14.3 implements both halves on token lists (`END=2` is `</think>`, `WAIT=3`). A
stream that never stops, `[10, 11, 12, 13, 14, 15]`, with budget 4 becomes
`[10, 11, 12, 13, END]`. The stream `[10, 11, END, 12, 13, 14, END]` with minimum budget 4
ends too early, so the first `END` becomes `WAIT`: `[10, 11, WAIT, 12, 13, 14, END]`. The
check then asserts the JSD bounds from 14.1 on 100 seeded random pairs, and
$\mathrm{JSD}_{0.5} = \log 2$ for disjoint supports.

```python finetuning-check
import math
import torch

END, WAIT = 2, 3

def budget_force(stream, max_think, min_think, end_id=END, wait_id=WAIT):
    """Apply s1-style budget forcing to a stream of thinking-token ids.

    `stream` stands in for what the model would emit next. Returns the thinking
    tokens actually kept, always terminated by end_id."""
    out = []
    for tok in stream:
        if len(out) >= max_think:            # cap: force the end of thinking
            break
        if tok == end_id:
            if len(out) < min_think:         # extend: replace the early end with "Wait"
                out.append(wait_id)
                continue
            out.append(end_id)
            return out
        out.append(tok)
    out = out[:max_think]
    out.append(end_id)
    return out

# Fixture A: the model never stops thinking; budget 4 truncates and appends END.
assert budget_force([10, 11, 12, 13, 14, 15], max_think=4, min_think=0) == [10, 11, 12, 13, END]
# Fixture B: an early END becomes WAIT, the later END is accepted.
assert budget_force([10, 11, END, 12, 13, 14, END], max_think=100, min_think=4) == \
    [10, 11, WAIT, 12, 13, 14, END]
# No forcing needed: the stream ends inside the window and passes through unchanged.
assert budget_force([10, 11, 12, END], max_think=10, min_think=2) == [10, 11, 12, END]

# --- Generalised JSD bounds (beta = 0.5) ---
def jsd(p, q, beta=0.5):
    m = beta * p + (1 - beta) * q
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * (a[mask].log() - b[mask].log())).sum())
    return beta * kl(p, m) + (1 - beta) * kl(q, m)

g = torch.Generator().manual_seed(0)
for _ in range(100):
    p = torch.softmax(3 * torch.randn(16, generator=g, dtype=torch.float64), 0)
    q = torch.softmax(3 * torch.randn(16, generator=g, dtype=torch.float64), 0)
    v = jsd(p, q)
    assert -1e-12 <= v <= math.log(2) + 1e-12

p = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64)
q = torch.tensor([0.0, 0.0, 0.3, 0.7], dtype=torch.float64)
assert abs(jsd(p, q) - math.log(2)) < 1e-6          # disjoint supports hit the bound
assert abs(jsd(p, p)) < 1e-12
```

The check proves the forcing logic and the bounds; it does not show that "Wait" improves
accuracy for your model, which is what L14.3 measures.

### Overthinking and length control

Reasoning models spend thousands of tokens on questions that need fifty. Every token costs
latency and money at serve time ([module 20](./20-production-serving-and-cost.md)), so
length is a first-class target. The levers, from training time to decode time:

| Lever | Where | Mechanism | Risk |
|---|---|---|---|
| Length penalty / overlong shaping | RL reward ([module 13](./13-rl-with-verifiable-rewards.md)) | penalise tokens beyond a target; mask or penalise truncated completions | reward hacking: skip reasoning on hard problems |
| Short-CoT distillation | SFT data | keep only the shortest correct teacher trace per prompt | removes the self-verification behaviour on hard items |
| Mixed-mode training | SFT data | non-thinking examples for easy prompts (Qwen3 fusion) | the model must learn when to think |
| Budget forcing / thinking budget | decode | cap at $B$ tokens, then force the answer | accuracy falls when $B$ is below what the problem needs |
| Early exit | decode | stop thinking when a trial answer is already confident | needs a calibrated confidence signal |

On early exit, [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)
reports, training-free across 11 reasoning models and 10 benchmarks, average CoT length
reductions of 19.1% to 80.1% with accuracy changes of +0.3% to +5.0%.

The curve to draw for any reasoning model is **accuracy against thinking budget** (L14.3):
a steep rise while the cap cuts most traces mid-thought, then a plateau once most finish
naturally. "Wait" far beyond the plateau mostly buys repetition. Ship the knee, and put the
curve in the model card.

---

## 14.8 Distilling to tiny models and evaluating distilled reasoners

### Student capacity

Distillation cannot add capacity. Forward KL (and SFT on traces) makes a too-small student
smear mass into hedged traces; reverse KL collapses it onto behaviour it can execute.

- **Pick the student by deployment budget first** (memory, latency, tokens per second),
  then distil the best teacher into it. Qwen3 distilled dense students from 0.6B to 14B
  plus a 30B-A3B MoE from Qwen3-32B and Qwen3-235B-A22B teachers.
- **Long traces hurt small students most.** Compounding error (14.2) grows with trace
  length and with a small model's per-token error rate, so prefer shorter-CoT data at 1B.
- **A bigger teacher is not always better.** Thinking Machines found Qwen3-8B slightly
  better than Qwen3-32B as teacher for an 8B student.
- **Use multiple teachers for multiple skills.** MOPD (14.4) routes each prompt to a
  domain expert instead of one generalist teacher.

### Evaluating distilled reasoners

The protocol from [module 09](./09-evaluating-fine-tuned-models.md), specialised:

- **Sample, then average.** AIME has 30 problems a year, so one greedy run has a standard
  error of several points. Report pass@1 averaged over $k$ samples per problem ([LIMO v3](https://arxiv.org/abs/2502.03387v3)
  used 4 samples at temperature 0.6 for AIME24, AMC23 and CHMath). Never decode Qwen3 thinking mode
  greedily; the card warns of repetition.
- **Report the settings**: thinking on/off, thinking budget, max new tokens (Qwen3
  recommends 32,768, or 38,912 for competition math), temperature, top-p. A score without
  its budget is comparable to nothing.
- **Spread the benchmarks**: AIME 2024/2025, MATH-500, GPQA Diamond, LiveCodeBench, one
  held-out domain task, plus instruction-following and safety evals
  ([module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)), which reasoning
  SFT often degrades.
- **Assume contamination.** Teacher traces can encode memorised benchmark answers, and
  public reasoning sets overlap benchmarks ([module 04](./04-data-curation-and-synthesis.md)).
  Prefer benchmarks released after the teacher's cutoff, such as a new AIME year.

Related material: "SFT memorizes, RL generalizes" in [module 13](./13-rl-with-verifiable-rewards.md);
distilled students as speculative-decoding drafts in [module 20](./20-production-serving-and-cost.md),
[transformers 14.3](../transformers/14-inference-optimizations.md#143-speculative-decoding) and the
[inference course](/courses/inference/06-decoding-algorithms/06-02-speculative-decoding.html);
Qwen3 and DeepSeek distillation in the [case studies](../transformers/15-modern-architecture-case-studies.md);
KL in the [information theory note](../../notes/math/information-theory.md) and temperature in
[text generation and decoding](../../notes/nlp/text-generation-and-decoding.md).

---

## 14.9 Lab: off-policy SFT vs on-policy distillation

**Goal.** Distil math reasoning into a 1.7B Qwen3 student two ways, compare
steps-to-accuracy, then map the student's accuracy-versus-thinking-budget curve.

**Scripts.** [`kd_losses.py`](./code/kd_losses.py) (L14.1, L14.2) and
[`budget_forcing.py`](./code/budget_forcing.py) (L14.3). Both run a CPU smoke mode first:

```bash
python content/courses/finetuning/code/kd_losses.py --smoke
python content/courses/finetuning/code/budget_forcing.py --smoke
```

The `kd_losses.py` smoke run trains a toy 8-token student both ways and asserts its reverse
KL falls at least 50×.

### L14.1 Off-policy: generate, filter, SFT (T1)

- **Teacher:** `Qwen/Qwen3-8B` in thinking mode via vLLM; `Qwen/Qwen3-32B` is the T2
  option (bf16), as is an API teacher whose licence allows training on outputs.
- **Prompts:** 1,000 problems from `open-r1/OpenR1-Math-220k` (`problem`, `answer`).
- **Student:** `Qwen/Qwen3-1.7B-Base`; `--lora` (all linear layers) fits T1, full FT wants T2.

```bash
python content/courses/finetuning/code/kd_losses.py --full generate \
    --teacher Qwen/Qwen3-8B --n-prompts 1000 --samples-per-prompt 4
python content/courses/finetuning/code/kd_losses.py --full sft \
    --student Qwen/Qwen3-1.7B-Base --max-length 8192 --lora
```

Record the filter's yield and the kept-trace length histogram. Checkpoints are saved
every 50 steps; score each on 100 `HuggingFaceH4/MATH-500` problems with
`budget_forcing.py --full --model <checkpoint> --budgets 4096` (merge LoRA adapters first).

### L14.2 On-policy distillation with `DistillationTrainer` (T2)

- **Teacher:** `Qwen/Qwen3-8B`. **Student:** `Qwen/Qwen3-1.7B`, or better, your L14.1
  checkpoint (OPD works best from an SFT initialisation).
- Same prompts, prompt-only; `beta=1.0`, `max_completion_length=4096`, vLLM colocated.

```bash
python content/courses/finetuning/code/kd_losses.py --full opd \
    --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-1.7B --max-completion-length 4096
```

Record the logged `jsd`, `completions/mean_length` and `completions/clipped_ratio`, and
the same checkpoint eval. Plot accuracy against steps and GPU-hours for L14.1 and L14.2 on
one set of axes. **Expected outcome:** OPD reaches the SFT run's final accuracy in fewer
steps at a higher cost per step; do not expect the published 9–30× at this scale. A high
clipped ratio means truncated thinking: raise `max_completion_length` (the RL analogue,
`mask_truncated_completions`, is in [module 13](./13-rl-with-verifiable-rewards.md)).

### L14.3 Budget forcing at inference (T0)

On a free T4 (fp16), take a distilled model such as
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (or your L14.2 student) and vary the maximum
thinking budget over {256, 512, 1024, 2048} tokens on 100 MATH-500 problems:

```bash
python content/courses/finetuning/code/budget_forcing.py --full \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --budgets 256 512 1024 2048 --n 100
```

The script caps thinking with vLLM (`stop=["</think>"]`, `max_tokens=B`), appends the end
tag, generates and grades the boxed answer, and writes `budget_curve.csv`; `--min-think 512`
tests the "Wait" extension. **Expected outcome:** a steep rise at small budgets that
flattens; report the knee.

---

## Key takeaways

- **SFT on teacher samples is Monte Carlo sequence-level forward KL**: its expected loss is
  teacher entropy plus $\mathrm{KL}(p_T\|q_S)$, so it inherits mass-covering behaviour.
- **Forward KL covers, reverse KL seeks a mode**: Check 14.1 puts one optimum between the
  teacher's modes and the other on a mode.
- **$T^2$ keeps KD gradients at constant scale**: the raw gradient shrinks about as
  $1/T^2$ (0.770 at $T=1$, 0.0425 at $T=4$).
- **Generalised JSD interpolates the KLs and is bounded by $\log 2$ at $\beta=0.5$**; in
  TRL `beta=0.0` is forward and `beta=1.0` reverse KL, not a KL penalty.
- **Off-policy distillation suffers exposure bias**: the student never trains on its own
  mistakes, and errors compound with trace length.
- **On-policy distillation is RL with a dense reward**: $A_t = \log p_T - \log q_S$ per
  token, whose path sum estimates $-\mathrm{KL}(q_S\|p_T)$.
- **OPD is cheap when a better teacher exists**: Qwen3 reports 1,800 vs 17,920 GPU-hours
  for RL with higher AIME; Thinking Machines report 9–30× fewer FLOPs than extrapolated SFT.
- **`DistillationTrainer` is the stable default**: always on-policy, `beta=1.0`, shared
  vocabulary required, and `max_completion_length=512` must be raised for reasoning.
- **Async for big or many teachers, GOLD for different tokenizers**:
  `AsyncDistillationTrainer` routes rows by `teacher_id` (MOPD); MiniLLM left TRL in v1.15.
- **Self-distillation repairs forgetting**: in Thinking Machines' example, mid-training on
  documents cut IF-eval from 85% to 79% even with a 70/30 chat mix; OPD with the original
  model as teacher brought it back to 83% with knowledge intact.
- **Thinking templates are training data**: Qwen3 strips old reasoning and uses an empty
  think block for non-thinking mode, so train both modes and watch prefix preservation.
- **Length is a target**: budget forcing caps with the end tag and extends with "Wait";
  report accuracy against budget, pass@1 over $k$ samples, and contamination status.

## Self-check

1. Explain on-policy distillation and why it can be more compute-efficient than SFT on
   teacher traces or RL from a verifier.
2. What is the difference between forward and reverse KL in distillation, and which one is
   mode-seeking? Which does SFT on teacher samples approximate?
3. Describe a full post-training pipeline for a small open reasoning model: data, SFT,
   RL or on-policy distillation, evaluation, and safety checks.

---

**Next → [15 — Agentic and Tool-Use Fine-Tuning](./15-agentic-and-tool-use-fine-tuning.md)**
