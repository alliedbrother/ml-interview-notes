---
order: 9
description: SFT, reward modelling, RLHF, DPO and GRPO explained; prompting technique that actually works; chain-of-thought and self-consistency; tool use and agents; and the failure modes to design around.
meta: NLP · LLMs
---

# LLM Prompting and Alignment

A base language model completes text. Turning it into something that answers
questions, follows instructions, uses tools, and declines harmful requests takes
a post-training pipeline — and getting good results out of the finished model
takes a set of prompting techniques that are more empirical than most people
admit. This page covers both halves.

## The post-training pipeline

```mermaid
flowchart TD
    PT["pretrained base model<br/>completes text;<br/>does not follow instructions"] --> SFT["SFT<br/>supervised fine-tuning on<br/>instruction-response pairs"]
    SFT --> A{"how do you have<br/>signal about quality?"}
    A -->|"human preference pairs"| RM["reward model<br/>Bradley-Terry on<br/>chosen vs rejected"]
    A -->|"preference pairs, directly"| DPO["DPO<br/>closed-form; no reward model,<br/>no sampling loop"]
    A -->|"a checkable answer"| GRPO["GRPO<br/>verifiable reward,<br/>group-relative advantage"]
    RM --> PPO["PPO<br/>maximise reward MINUS<br/>a KL penalty to the SFT policy"]
    PPO --> AL["aligned model"]
    DPO --> AL
    GRPO --> AL
```

### Supervised fine-tuning

Train on (instruction, response) pairs with the standard next-token loss,
**computing the loss on the response tokens only**. Including the prompt teaches
the model to generate user turns.

| Detail | Guidance |
|---|---|
| Data quality over quantity | 1,000 excellent examples beat 100,000 mediocre ones — the LIMA result |
| Diversity | cover the task distribution you actually expect |
| Format consistency | use the model's chat template; do not hand-write control tokens |
| Learning rate | 1e-5 to 2e-5 full, 1e-4 to 3e-4 for LoRA |
| Epochs | 2–3; more memorises |
| Packing | concatenate short examples to fill the context — can double throughput |
| Mask the prompt | loss on the response only |

**LIMA's finding — that 1,000 carefully curated examples produce a strong
assistant — is the most actionable result in this area.** The base model already
has the capability; SFT mostly teaches format and style. Curating a small,
excellent dataset beats scraping a large mediocre one.

### Reward modelling

Train a model to score responses, using pairwise human preferences and the
Bradley–Terry likelihood:

$$L = -\log\sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr)$$

Pairwise comparison is used rather than absolute rating because humans are far
more consistent at "which of these two is better" than at "rate this 1–10".

### RLHF with PPO

$$\max_\pi\;\mathbb{E}_{y\sim\pi}\bigl[r_\phi(x,y)\bigr] - \beta\,D_{\mathrm{KL}}\bigl(\pi\,\Vert\,\pi_{\text{SFT}}\bigr)$$

**The KL penalty is not optional.** Reward models are imperfect proxies, and an
unconstrained optimiser finds their failure modes — repetitive phrasings,
excessive hedging, characteristic filler that scores well and reads badly. This
is Goodhart's law with a learned metric, and $\beta$ is the dial between
alignment and drift.

PPO's practical difficulties: four models in memory (policy, reference, reward,
value), a sampling loop inside training, and notorious hyperparameter
sensitivity.

### DPO

Direct preference optimisation observes that the KL-constrained objective has a
**closed-form optimal policy**, which can be inverted to express the implicit
reward in terms of the policy itself. Substituting into the Bradley–Terry
likelihood gives a supervised loss over preference pairs:

$$L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

**No reward model, no sampling, no RL machinery** — optimising the same objective
with ordinary supervised learning. That is why it became the default.

Practical notes: learning rates are very small ($5\times10^{-7}$ to
$1\times10^{-6}$); $\beta$ around 0.1; and over-training collapses output
diversity noticeably. DPO is also **off-policy** — it never samples from the
current policy — which is its main quality limitation relative to online methods.

### GRPO and verifiable rewards

For tasks with a **checkable** answer — mathematics, code that must pass tests,
formal proofs — replace the learned reward model with a verifier. This removes
reward hacking almost entirely, because the verifier cannot be fooled by
plausible-sounding text.

Group relative policy optimisation samples $G$ completions per prompt and
standardises the rewards within the group:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

The group mean **is** the baseline, so no value network is needed — halving
memory and simplifying the implementation. This is the method behind the recent
generation of reasoning models, and the broader lesson generalises: **RL works
far better when the reward is verifiable than when it is learned.**

### The method table

| Method | Needs | Trade-off |
|---|---|---|
| SFT | demonstrations | simple; capped by demonstration quality |
| RLHF (PPO) | reward model + sampling | strongest control; complex, unstable |
| **DPO** | preference pairs | simple and stable; off-policy |
| IPO / KTO / ORPO | variants | KTO needs only binary good/bad labels |
| **GRPO** | a verifiable reward | excellent where verification exists |
| Constitutional AI / RLAIF | a principle set, AI feedback | scales past human labelling |
| Best-of-$n$ | reward model + inference compute | no training; pay at inference |

## Prompting

### What reliably works

| Technique | Why |
|---|---|
| **Be specific about the output format** | ambiguity is filled with the training distribution's default |
| **Give examples** (few-shot) | demonstrates format and edge cases better than description |
| **Assign a role or context** | narrows the distribution usefully |
| **Chain of thought** | "think step by step" — allocates compute to intermediate reasoning |
| **Reasoning before the answer** | in structured output, put the reasoning field *first* |
| **Decompose** | several focused calls beat one overloaded prompt |
| **Delimit sections clearly** | XML tags or markdown headers; reduces instruction/content confusion |
| **Say what to do, not what to avoid** | negative instructions are followed less reliably |
| **Provide an escape hatch** | "if the answer is not in the context, say so" — measurably reduces fabrication |
| Prefill the response | starting the assistant turn constrains the format |

### Chain of thought

$$\text{prompt} \to \text{reasoning steps} \to \text{answer}$$

The mechanism is best understood as **compute allocation**: a transformer does a
fixed amount of computation per token, so a problem needing more computation than
one forward pass provides must be spread across tokens. Chain of thought gives
the model somewhere to do the work.

Consequences that follow from that framing:

- It helps most on multi-step problems and barely at all on single-step recall.
- The reasoning must come **before** the answer. Reasoning generated after the
  answer cannot influence it.
- The written reasoning is **not necessarily faithful** — models produce
  post-hoc justifications for answers reached otherwise. Do not treat it as an
  explanation.
- Reasoning-trained models (o-series, R1-style) internalise this and perform
  extended reasoning by default.

### Self-consistency

Sample $n$ chains of thought at temperature ~0.7 and take the **majority final
answer**. It routinely adds 10–20 points on mathematical reasoning benchmarks,
because errors are diverse while correct reasoning converges. It is the simplest
and most reliable inference-time compute technique.

### What is over-sold

| Claim | Reality |
|---|---|
| "Prompt engineering is a durable skill" | tricks that worked on GPT-3 are unnecessary now; the transferable skills are specificity, examples, and decomposition |
| Emotional appeals ("this is very important") | inconsistent, model-dependent, unreliable |
| Elaborate persona prompts | mild effect; specificity about the *task* matters more |
| Prompt "magic words" | brittle, and break across model versions |
| One prompt to do everything | decomposition beats overloading |
| Politeness affecting quality | no reliable effect |

**Build an evaluation set before optimising prompts.** Twenty to a hundred
examples from your real distribution, scored automatically where possible. Prompt
changes without measurement are superstition, and the field is full of it.

### Prompts are code

| Practice | Why |
|---|---|
| Version control | prompts change behaviour as much as code |
| An evaluation suite in CI | catch regressions from prompt or model changes |
| Templating, not string concatenation | injection safety and maintainability |
| Track model version with prompt version | a prompt tuned on one model may fail on another |
| A/B test prompt changes | intuition about prompts is unreliable |
| Log inputs and outputs | you cannot debug what you did not record |

## Tool use and agents

Give the model function definitions; it emits structured calls; your code
executes them and returns results.

```mermaid
flowchart LR
    U["user request"] --> M["model"]
    M -->|"tool call<br/>structured JSON"| E["your executor"]
    E -->|"result"| M
    M -->|"another call, or done"| M
    M --> A["final answer"]
    E -.->|"errors, timeouts,<br/>permission denials<br/>all go back as results"| M
```

| Pattern | Description |
|---|---|
| Function calling | the model emits a call matching a schema |
| **ReAct** | interleaved reasoning and acting |
| Plan-and-execute | plan first, then run the steps |
| Reflexion | critique and retry after failures |
| Multi-agent | specialised agents with a coordinator |
| Code as action | generate and execute code rather than calling fixed tools |

**What makes tool use work in practice:**

- **Clear, minimal schemas.** Fewer parameters, unambiguous names, examples in
  the description.
- **Constrained decoding** for the call itself, so parsing never fails.
- **Errors returned as results**, not as exceptions. The model can recover from
  "file not found"; it cannot recover from a crash.
- **Idempotency and confirmation** for anything destructive.
- **A step limit.** Agents loop.
- **Observability.** Log every call, result, and decision.

**Agents fail differently from models.** Compounding errors across steps (95%
per-step reliability over 20 steps is 36% end-to-end), infinite loops,
overconfident tool selection, and unbounded cost. Design for a bounded step
budget, checkpointing, and human confirmation at high-consequence actions.

## Failure modes to design around

| Failure | Nature | Mitigation |
|---|---|---|
| **Hallucination** | a direct consequence of the training objective — probable text is not true text | RAG with citations, abstention training, verification, restricted claims |
| Sycophancy | preference tuning rewards agreement | ask for critique explicitly; avoid leading questions |
| **Prompt injection** | untrusted input contains instructions | never trust retrieved or user content as instructions; separate channels; least privilege on tools |
| Jailbreaks | adversarial framing bypasses refusals | layered defences, output filtering, monitoring |
| Position bias | "lost in the middle" — long-context recall dips in the middle | put key material at the start and end |
| Verbosity bias | reward models prefer longer answers | length-controlled evaluation |
| Format brittleness | small prompt changes cause large output changes | constrained decoding, evaluation suites |
| Knowledge cutoff | no information after training | retrieval, tools |
| Non-determinism | batching and kernels vary results even at temperature 0 | do not assume reproducibility |
| Overconfidence | fluent text regardless of certainty | calibration prompts, self-consistency as a confidence signal |

**Prompt injection is the security issue that has no clean solution.** If a model
processes untrusted text — a retrieved web page, a user-uploaded document, an
email — that text can contain instructions the model may follow. There is no
reliable way to make a language model distinguish "instructions from my
principal" from "text I was asked to read", because both arrive as tokens in the
same context.

The workable defences are architectural: give the model the **least privilege**
needed, require human confirmation for consequential actions, never let retrieved
content trigger tool calls without review, and treat every model output derived
from untrusted input as untrusted itself.

## Evaluating an aligned model

| Method | Note |
|---|---|
| Public benchmarks (MMLU, GSM8K, HumanEval) | comparable, and **contaminated** — assume they are in training data |
| **Your own held-out set** | the only number that predicts your production quality |
| LLM-as-judge | scalable; biased toward length, position, and its own style |
| Pairwise human comparison | reliable, expensive |
| Arena-style Elo | good aggregate signal; not task-specific |
| Behavioural test suites | invariance, minimum functionality, directional expectation |
| Red teaming | adversarial probing for harmful behaviour |

**LLM-as-judge biases are well documented and correctable**: position bias (swap
the order and average), verbosity bias (control for length), self-preference
(judges favour their own family), and score compression (use pairwise comparison
rather than absolute scoring). A judge with a detailed rubric and randomised
positions is usable; a naive "rate this 1–10" judge is not.

## Self-check

1. Why must SFT mask the prompt tokens from the loss?
2. What does DPO replace from the RLHF pipeline, and what objective is it
   equivalent to?
3. Why does GRPO not need a value network?
4. Explain chain of thought as compute allocation, and derive two practical rules
   from that framing.
5. Why is chain-of-thought reasoning not a reliable explanation?
6. Why is prompt injection structurally unsolvable at the model level, and what
   are the architectural defences?
7. Name three LLM-as-judge biases and the correction for each.

## Where to go next

- [RAG & Retrieval](./rag-and-retrieval.md) — the standard answer to
  hallucination and stale knowledge.
- [Text Generation & Decoding](./text-generation-and-decoding.md) — sampling and
  constrained output.
- [NLP Evaluation](./nlp-evaluation.md) — measuring generation quality.
