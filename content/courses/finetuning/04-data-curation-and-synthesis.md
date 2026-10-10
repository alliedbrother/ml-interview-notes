# 04 — Data Curation and Synthesis: Formats, Dedup, Decontamination, and Synthesis

> **Prerequisites:** module 02.
> **You will learn:** how to produce a clean, deduplicated, decontaminated dataset in the right
> schema for SFT, preference, RL and tool training, including synthetic
> generation and filtering pipelines.

**Data quality dominates hyperparameters.** A learning rate off by 2x costs a point or two; a
training set full of near-duplicates, broken labels or leaked benchmark items costs the whole
experiment, often unnoticed. Most post-training data in 2026 is model-generated, so generation
and filtering pipelines are core skills.

---

## 4.1 Dataset schemas: the TRL taxonomy

TRL classifies a dataset along two axes
([TRL dataset formats, v1.15.0](https://huggingface.co/docs/trl/v1.15.0/en/dataset_formats)).
The **format** is *standard* (plain strings) or *conversational* (lists of
`{"role", "content"}` messages). The **type** is the task, defined by which columns exist.
The trainer reads the columns to decide what to tokenize, what to mask and what to compare.
Get the columns wrong and you get either an error or, worse, silent training on the wrong tokens.

**Table 1. Dataset types, their columns, and their consumers (as of TRL v1.15, October 2026).**

| Type | Columns | Example row (abridged) | Consumed by |
|---|---|---|---|
| Language modeling | `text` (standard) or `messages` (conversational) | `{"text": "Email: ... JSON: {...}"}` or `{"messages": [system, user, assistant]}` | `SFTTrainer` (loss on every token; with `messages` and `assistant_only_loss=True`, assistant turns only) |
| Prompt-completion | `prompt`, `completion` | `{"prompt": [{"role":"user","content":"Order ORD-00551..."}], "completion": [{"role":"assistant","content":"{...}"}]}` | `SFTTrainer` (loss on completion only by default) |
| Preference, explicit prompt | `prompt`, `chosen`, `rejected` | `{"prompt": [...], "chosen": [{"role":"assistant","content":"{...}"}], "rejected": [...]}` | `DPOTrainer` (explicit recommended), `KTOTrainer` |
| Preference, implicit prompt | `chosen`, `rejected` (each a full conversation) | `{"chosen": [user, assistant_good], "rejected": [user, assistant_bad]}` | `RewardTrainer` (implicit recommended) |
| Unpaired preference | `prompt`, `completion`, `label` (bool) | `{"prompt": [...], "completion": [...], "label": false}` | `KTOTrainer` |
| Prompt-only | `prompt` (+ any extra columns) | `{"prompt": [{"role":"user","content":"Solve ..."}], "answer": "42"}` | `GRPOTrainer`, `RLOOTrainer` |
| Tool calling | `messages` with assistant `tool_calls` and `tool` role turns, plus a `tools` column of JSON schemas | `{"messages": [...], "tools": [{"type":"function","function":{...}}]}` | `SFTTrainer`; owned by [module 15](./15-agentic-and-tool-use-fine-tuning.md) |
| Vision | `image` or `images` + content blocks `[{"type":"image"},{"type":"text",...}]` | `{"images": [img], "messages": [...]}` | `SFTTrainer` (VLM path) and other trainers with VLM support; owned by [module 16](./16-multimodal-and-embedding-fine-tuning.md) |

"Conversational" is a format, not a type: a `messages` column is conversational
language-modeling data. The mapping is TRL's own table (its `experimental` trainers are left out).
One rendering detail matters: a prompt-only row ends with the assistant header (a generation
prompt), a language-modeling row ends with end-of-sequence.

### Converting between formats

Convert **before** applying the chat template, while the data is still a list of messages (TRL's
recommendation):

| From | To | Rule |
|---|---|---|
| Conversational `messages` | Prompt-completion | `prompt` = every message before the last assistant turn; `completion` = that last assistant turn |
| Prompt-completion | Language modeling | concatenate (and accept that the prompt becomes trained tokens) |
| Implicit preference | Explicit preference | split the longest common prefix off as `prompt` (TRL: `extract_prompt`) |
| Explicit preference | Implicit preference | prepend `prompt` to both `chosen` and `rejected` |
| Preference | Unpaired preference | emit `(prompt, chosen, True)` and `(prompt, rejected, False)` (TRL: `unpair_preference_dataset`) |
| Unpaired preference | Prompt-completion | keep rows with `label=True`, drop `label` |

TRL warns about unpairing: a `rejected` response is only worse *than its partner*, so check
(for example with a reward model) before labeling it `False`.

**The type controls masking.** For prompt-completion data, `SFTConfig.completion_only_loss`
defaults to `None`, which TRL resolves to "completion only"; for language-modeling data it
resolves to the full sequence ([SFTConfig reference](https://huggingface.co/docs/trl/v1.15.0/en/sft_trainer)).
A `messages` row trains on every token unless you set `assistant_only_loss=True` with a template
that marks assistant spans with `{% generation %}`. So converting the support-email set from
`messages` to prompt-completion changes the objective from "model the email and the JSON" to
"model the JSON given the email" ([module 05](./05-supervised-fine-tuning.md) measures the
difference). Check 4.4 in section 4.8 asserts these conversions.

### Prompt-only datasets for RL

GRPO and RLOO generate their own completions, but the **reward function** needs whatever it takes
to score one. TRL passes every column except `prompt` to each reward function as a keyword
argument, and `GRPOConfig.remove_unused_columns` defaults to `False` so they survive
([GRPO trainer docs, v1.15.0](https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer)):

```json
{"prompt": [{"role": "system", "content": "You extract structured data from customer support emails. ..."},
            {"role": "user", "content": "Order ORD-00551 never came. - Dana Kim"}],
 "gold": {"customer_name": "Dana Kim", "order_id": "ORD-00551", "issue_type": "shipping",
          "refund_requested": false, "sentiment": "negative"}}
```

`def reward(completions, gold, **kwargs)` then scores field-level agreement. A missing column
surfaces only at the first training step, so validate prompt-only sets with the SFT schema
checker. [Module 13](./13-rl-with-verifiable-rewards.md) designs the rewards.

---

## 4.2 The datasets library for fine-tuners

The [Hugging Face note](../../notes/libraries/huggingface.md) covers the basics: `load_dataset`
returns an Arrow-backed, memory-mapped `Dataset`, and `map`/`filter` cache results under a
**fingerprint** (a hash of the previous fingerprint plus the function and its arguments), so an
unchanged pipeline re-runs as a cache hit.

Teaching fragment: not executed (needs `datasets` and network access).

```python
import json
from datasets import load_dataset

# SYSTEM_PROMPT: module 01's extraction instruction; valid(): the schema checker from section 4.4
ds = load_dataset("json", data_files={"train": "ft_data/train.jsonl"}, split="train")

def to_messages(batch):                       # batched=True: dict of lists in, dict of lists out
    return {"messages": [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": e},
         {"role": "assistant", "content": json.dumps(g)}]      # module 01 key order
        for e, g in zip(batch["email"], batch["gold"])]}

ds = ds.map(to_messages, batched=True, num_proc=8, remove_columns=ds.column_names)
ds = ds.filter(lambda b: [valid(m) for m in b["messages"]], batched=True, num_proc=8)
splits = ds.train_test_split(test_size=0.05, seed=0)      # seeded: reproducible split
splits.push_to_hub("you/support-email-sft", private=True)
```

| Feature | What it does | Trap |
|---|---|---|
| `batched=True` | hands your function 1,000 rows at a time by default | none; use it for tokenizers and anything vectorized |
| `num_proc` | forks worker processes | a non-picklable closure or a changing global breaks fingerprint caching: silent recompute or a stale cache hit |
| `streaming=True` ([guide](https://huggingface.co/docs/datasets/main/en/stream)) | returns an `IterableDataset` read lazily | no `len()`; shuffles only within a buffer (default 1,000); `take`/`skip` lock shard order, so shuffle first |
| `IterableDataset` in TRL | trains without materializing the set | **TRL trainers require `max_steps`**: epochs are undefined ([SFTTrainer reference](https://huggingface.co/docs/trl/v1.15.0/en/sft_trainer)) |
| `interleave_datasets([a, b], probabilities=[0.8, 0.2], seed=0)` | draws from sources in fixed ratios | how section 4.8's replay mix is built without materializing it |
| `push_to_hub` | writes Parquet shards; `README.md` is the **dataset card** ([guide](https://huggingface.co/docs/datasets/main/en/dataset_card)) | a card without provenance (generator models, terms, filters, thresholds, dedup and decontamination settings) cannot be audited; release content is [module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)'s |

---

## 4.3 Where data comes from

| Source | Strength | Main risk |
|---|---|---|
| Human-written (experts, annotators) | correct, on-distribution, sets the bar | slow, expensive, inconsistent between annotators |
| Product logs | exactly your input distribution | consent, PII, label noise, feedback bias |
| Public instruction datasets | free, large, varied | license mix, contamination, stale styles |
| Distillation from a stronger model | cheap, scalable, high fluency | terms of service, teacher errors, homogeneity |
| Programmatic generation (templates, solvers, code) | perfect labels, unlimited | low diversity, unrealistic inputs |

Most production sets are blends: for the running task, a few hundred scrubbed real emails with
human-checked labels, padded with synthetic emails whose labels are correct by construction.

### Quality over quantity

LIMA ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206); [module 01](./01-when-to-fine-tune.md)) used
1,000 curated examples; for reasoning, LIMO ([Ye et al., 2025, v3](https://arxiv.org/abs/2502.03387v3))
used 800 (817 in the paper's v1) and s1 ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393))
used 1,000, selected for difficulty, diversity and quality ([module 14](./14-distillation-and-reasoning.md)).
They **establish** that with a strong base model, selection beats volume. They **do not establish**
that a tiny set covers a strict schema with many edge cases or messy real inputs. Your extractor
needs every `issue_type` and the ugly cases (forwarded threads, two orders, no order ID), which
usually means hundreds to a few thousand examples.

### Licensing, PII and terms of service

A mix inherits the most restrictive **dataset license** you rely on; record each in the card
([module 01](./01-when-to-fine-tune.md)). Several API **terms of service** restrict using outputs to
build competing models; open-weight teachers (Apache-2.0 Qwen3) avoid most of that. Scrub **PII**
with regex, then NER, then spot checks ([module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)
owns memorization); `customer_name` is a target field, so substitute consistent synthetic names.

### Preference data from production logs

Thumbs-up/down logs are not pairs. Each event is one response to one prompt with a binary label:
the **unpaired preference** type, whose natural consumer is KTO
([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306); [module 12](./12-preference-optimization.md)).
DPO needs two responses **to the same prompt**. You have them only when the product showed two
candidates and the user picked one, when a regenerate-then-thumbs-up gives a weak pair, or when
you build pairs offline: sample several fresh responses per logged prompt, rank them with a judge
or reward model, and take best vs worst (the log supplies the prompts, the judge the preference).

Design around **selection bias** (few users click, and those who do skew negative), **label
noise** (a thumbs-down may be about latency or a refusal), **prompt imbalance** (cap events per
user and per prompt cluster) and **off-policy staleness** (last quarter's responses hurt DPO more
than KTO). Deduplicate prompts before splitting so near-identical prompts do not straddle the split.

---

## 4.4 Cleaning and filtering

Run filters cheapest first, so each expensive stage sees fewer rows. The full pipeline is:
normalize schema → filter (this table) → deduplicate (section 4.5) → decontaminate (4.6) → mix
and templated-length check (4.8) → publish with a card (4.2).

| # | Filter | What it does | Note |
|---|---|---|---|
| 1 | Language ID | fastText-style classifier; drop or route other languages | cheapest |
| 2 | Length | drop empty, truncated and over-budget responses, and few-character prompts | budget from section 4.8 |
| 3 | Format and schema | parse every structured target; for the running task, exactly the five fields, `order_id` matching `ORD-` plus five digits, `issue_type` in `refund`, `shipping`, `defect`, `account`, `other`, `refund_requested` a boolean ([module 01](./01-when-to-fine-tune.md) fixes the schema) | usually removes more bad rows than all the others combined |
| 4 | Heuristics | repetition, boilerplate ("As an AI language model"), leftover template tokens, unbalanced fences, unwarranted refusals | Gopher and C4 rule sets; datatrove implements both |
| 5 | Quality classifier | LLM-label a sample, train a small classifier, score everything | FineWeb-Edu did this for pretraining text ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557)); scales to millions |
| 6 | LLM judge | score (prompt, response) against a written rubric on 1–5 or 1–10; keep rows above a threshold | pick the threshold from agreement with 100 hand-labeled rows; judges prefer length and their own style ([module 09](./09-evaluating-fine-tuned-models.md)) |
| 7 | Reward model | scalar per response | reliable for ranking candidates of one prompt, not as an absolute threshold across prompts |

### Rejection sampling as a data engine

The most important filter generates its candidates: sample $k$ responses per prompt and keep
those that pass a verifier (unit tests, exact answer, schema check) or a reward threshold. Llama 3
ran this loop every post-training round before DPO
([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783); [module 01](./01-when-to-fine-tune.md)).
If a teacher solves a prompt with probability $p$, at least one of $k$ samples passes with
probability $1-(1-p)^k$; at $p=0.2$, $k=8$ that is $0.832$. **So $k$ buys coverage of hard
prompts**, while easy prompts pass every sample: cap kept samples per prompt (the lab uses 4) and
deduplicate identical solutions.

Synthetic reasoning data is this loop with a long chain of thought: keep traces with a correct
answer, drop traces that hit the token limit, and consider dropping extreme-length outliers
([module 14](./14-distillation-and-reasoning.md)).

---

## 4.5 Deduplication

Duplicates do three kinds of damage. They overweight some examples, which wastes compute and
pushes the model to memorize them. They leak between train and validation splits, which inflates
your metrics. And in synthetic data they signal **mode collapse**: the generator keeps saying the
same thing. Repeat that over generations, training each model on the previous model's outputs,
and the tails of the distribution disappear ([Shumailov et al., 2024](https://www.nature.com/articles/s41586-024-07566-y));
keep a fixed share of human or original data in every round. Deduplicate before you split.

### Exact

Normalize (lowercase, strip punctuation, collapse whitespace, optionally NFKC; see
[text preprocessing](../../notes/nlp/text-preprocessing.md)), hash with `hashlib.sha256`, and keep
the first row per hash. It is $O(n)$ and misses anything with one changed word.

### Near-duplicate: MinHash-LSH

**Problem.** You want all pairs of documents with Jaccard similarity
$J(A,B) = |A\cap B|/|A\cup B|$ above some $s^*$, where $A$ and $B$ are shingle sets (character
5-grams or word $n$-grams). All-pairs comparison is $O(n^2)$: at $10^6$ documents that is
$5\times10^{11}$ comparisons.

**MinHash ([Broder, 1997](https://doi.org/10.1109/SEQUEN.1997.666900)).** Apply a random hash $h$
to every shingle and keep the minimum. A random hash acts as a random permutation of the shingle
universe: it orders the shingles uniformly at random, and the minimum is the first shingle in that
order. So the minimum element of $A\cup B$ is equally likely to be any of its $|A\cup B|$
elements, and $\min h(A) = \min h(B)$ exactly when that element lies in $A\cap B$. Hence

$$
P\big[\min h(A) = \min h(B)\big] = \frac{|A\cap B|}{|A\cup B|} = J(A,B).
$$

With $k$ independent hash functions, the estimate $\hat J = \frac1k\sum_{i=1}^k \mathbb{1}[\min h_i(A) = \min h_i(B)]$
is a binomial proportion, so it is unbiased with standard error $\sqrt{J(1-J)/k}$
([probability note](../../notes/math/probability.md)). At $J=0.8$ and $k=128$ that is $0.035$.
Each document becomes a $k$-integer signature, regardless of its length.

**LSH banding.** Signatures are compact, but comparing all pairs of them is still $O(n^2)$. Split
each signature into $b$ bands of $r$ rows ($k = br$) and hash each band into a bucket. Two
documents become a **candidate pair** if they collide in at least one band. A band matches with
probability $s^r$, so it fails with $1-s^r$; all $b$ bands fail with $(1-s^r)^b$. Therefore

$$
P[\text{candidate} \mid J=s] = 1-(1-s^r)^b .
$$

This is an S-curve whose steepest point sits near $s^* \approx (1/b)^{1/r}$
(Leskovec, Rajaraman and Ullman, *Mining of Massive Datasets*, ch. 3). Raising $r$ sharpens the
curve and moves it right; raising $b$ moves it left.

| $b \times r$ ($k$) | threshold $(1/b)^{1/r}$ | $P$ at $s=0.3$ | $P$ at $s=0.5$ | $P$ at $s=0.8$ |
|---|---|---|---|---|
| $20\times5$ (100) | 0.549 | 0.047 | 0.470 | 0.9996 |
| $32\times4$ (128) | 0.420 | 0.229 | 0.873 | 1.000 |
| $14\times8$ (112), datatrove default | 0.719 | <0.001 | 0.053 | 0.924 |

In the first row, 80%-similar pairs are caught with probability 0.9996 and 30%-similar pairs
become false candidates 4.7% of the time, cheap to discard by checking $\hat J$. The datatrove
default is deliberately stricter: it targets web-scale near-copies, not paraphrases.

Check 4.1 builds 128 MinHash functions from `blake2b` plus seeded $(ax+b) \bmod p$. For a support
email A and a copy B with two words changed, the true Jaccard is 0.824 and the estimate 0.797;
an unrelated text C stays under 0.2; $32\times4$ banding makes (A, B) a candidate and not (A, C);
and every value in the table holds.

```python finetuning-check
import hashlib
import random

def shingles(text, k=5):
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

P = (1 << 61) - 1                      # Mersenne prime modulus
rng = random.Random(0)
K = 128
COEFFS = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(K)]

def base_hash(s):
    # stable 64-bit hash; never Python's salted hash()
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

def signature(sh):
    xs = [base_hash(s) for s in sh]
    return [min((a * x + b) % P for x in xs) for a, b in COEFFS]

def estimate(sa, sb):
    return sum(x == y for x, y in zip(sa, sb)) / len(sa)

def lsh_candidate(sa, sb, b, r):
    assert b * r == len(sa)
    return any(sa[i * r:(i + 1) * r] == sb[i * r:(i + 1) * r] for i in range(b))

A = ("Hi team, my order 48213 arrived with a cracked screen and the box was "
     "soaked. I would like a full refund to my original card, not store credit. "
     "Please confirm once the return label is ready. Thanks, Priya Raman")
B = A.replace("cracked", "broken").replace("soaked", "damp")      # 2 words changed
C = ("Quarterly revenue grew on strong cloud demand while hardware margins "
     "narrowed; the board approved a new buyback and raised the dividend.")

shA, shB, shC = shingles(A), shingles(B), shingles(C)
sigA, sigB, sigC = signature(shA), signature(shB), signature(shC)

j_ab, j_ac = jaccard(shA, shB), jaccard(shA, shC)
assert j_ab > 0.7 and round(j_ab, 3) == 0.824
assert abs(estimate(sigA, sigB) - j_ab) < 0.1
assert round(estimate(sigA, sigB), 3) == 0.797
assert estimate(sigA, sigC) < 0.2 and j_ac < 0.2

# LSH with b=32 bands x r=4 rows (k=128): near-dup collides, unrelated does not
assert lsh_candidate(sigA, sigB, b=32, r=4)
assert not lsh_candidate(sigA, sigC, b=32, r=4)

# Closed-form S-curve values quoted in the table
def p_candidate(s, b, r):
    return 1 - (1 - s ** r) ** b

assert round((1 / 20) ** (1 / 5), 3) == 0.549
assert round(p_candidate(0.8, 20, 5), 4) == 0.9996
assert round(p_candidate(0.3, 20, 5), 3) == 0.047
assert round(p_candidate(0.5, 20, 5), 3) == 0.470
assert round((1 / 32) ** (1 / 4), 3) == 0.420
assert round(p_candidate(0.3, 32, 4), 3) == 0.229
assert round(p_candidate(0.5, 32, 4), 3) == 0.873
assert p_candidate(0.3, 14, 8) < 0.001
assert round((1 / 14) ** (1 / 8), 3) == 0.719
assert round(p_candidate(0.8, 14, 8), 3) == 0.924
assert round(p_candidate(0.5, 14, 8), 3) == 0.053
print("MinHash estimate, LSH banding and S-curve values verified.")
```

The check shows that the estimator and the banding work on one pair. It does not show
that any threshold is right for your data. Pick shingle size and threshold by inspecting
the pairs that land just above and just below the cut.

### Semantic dedup with embeddings

MinHash sees characters, so two paraphrases ("my parcel never came" and "the package was
not delivered") look unrelated to it. Semantic dedup embeds each example with a sentence
encoder (see [text representation](../../notes/nlp/text-representation.md)) and treats pairs
above a cosine threshold (often 0.9–0.95) as duplicates. SemDeDup
([Abbas et al., 2023](https://arxiv.org/abs/2303.09540)) scales it by comparing pairs only within
k-means clusters. Two emails about different orders with the same complaint can sit at 0.95 and
both be useful, so **use semantic dedup on instructions and prompts**, where paraphrase
duplication is the failure mode.

Check 4.2 runs the threshold rule on fixed toy 4-dimensional vectors that stand in for encoder
outputs (no model). The paraphrase pair "my parcel never came" / "the package was not delivered"
has cosine 0.999 and is the only pair at or above 0.95, while its character 5-gram Jaccard is 0.0,
so MinHash cannot see it.

```python finetuning-check
import torch

E = {
    "my parcel never came":          [0.90, 0.40, 0.10, 0.05],
    "the package was not delivered": [0.88, 0.43, 0.12, 0.04],
    "the blender jar is cracked":    [0.10, 0.20, 0.95, 0.10],
    "please reset my password":      [0.05, 0.10, 0.15, 0.97],
}
X = torch.nn.functional.normalize(torch.tensor(list(E.values())), dim=1)   # (4, 4)
S = X @ X.T                                                                 # cosine matrix
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4) if S[i, j] >= 0.95]
assert pairs == [(0, 1)] and round(S[0, 1].item(), 3) == 0.999

def shingles(t, k=5):
    return {t[i:i + k] for i in range(len(t) - k + 1)}
a, b = shingles("my parcel never came"), shingles("the package was not delivered")
assert len(a & b) / len(a | b) == 0.0          # MinHash sees no overlap at all
print("cosine-threshold semantic dedup verified.")
```

This shows the threshold rule, not that a real encoder places your paraphrases above 0.95. Inspect
pairs near the threshold on your own data.

---

## 4.6 Decontamination

If training contains your eval items, the score measures memorization and **every reported gain
becomes uninterpretable**. A web-trained teacher has seen GSM8K, MMLU and HumanEval and can
reproduce their items almost verbatim when asked for "a grade-school math word problem".

**n-gram overlap.** Build the set of word $n$-grams from every eval item, after the same
normalization you apply to training text. Then flag any training example that shares at least
one $n$-gram with the set. GPT-3 used 13-grams
([Brown et al., 2020](https://arxiv.org/abs/2005.14165), appendix C), which became the
classic choice. A long $n$ gives few false positives, because 13 consecutive matching words are
rarely coincidental, but it misses light paraphrase. 8-grams flag more (more aggressive, more
false positives); below about 6, common
phrases ("the customer name, the order id") start triggering. Normalization matters as much as
$n$: without lowercasing and punctuation stripping, a casing change defeats the match.

Check 4.3 shows both effects on three eval items and five training items. Item 0 holds a 22-word
span of an eval item with changed casing and punctuation; item 1 shares only a 6-word phrase;
items 2–4 are clean. At $n=13$ only item 0 is flagged, at $n=6$ items 0 and 1, and without
normalization the 13-gram match misses item 0.

```python finetuning-check
import string

def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return text.split()                      # split() also collapses whitespace

def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

EVAL = [
    "A customer writes that order 7731 arrived two weeks late and the ceramic "
    "mug inside was shattered into pieces, and she asks whether she can get her "
    "money back or a replacement shipped by express courier.",
    "Extract the customer name, the order id and the issue type from the "
    "following email, and return a single JSON object with no extra text.",
    "My subscription renewed even though I cancelled it last month through the "
    "settings page, please reverse the charge and confirm by email today.",
]
TRAIN = [
    # 0: verbatim 22-word span of EVAL[0], different casing and punctuation
    "Ticket #88: ORDER 7731 Arrived Two Weeks Late; and the Ceramic Mug inside "
    "was SHATTERED into pieces -- and she asks whether she can get... a refund?",
    # 1: shares only the 6-word span "the customer name the order id" with EVAL[1]
    "Please list the customer name, the order id, plus a short summary of "
    "the complaint for our weekly support review meeting.",
    # 2-4: clean
    "The parcel never arrived and tracking has shown no movement since Monday.",
    "I love the new blender, it is quiet and the jug is easy to clean.",
    "Can you change the delivery address on my account to my office downtown?",
]

def flagged(train, evals, n):
    bank = set().union(*(ngrams(normalize(e), n) for e in evals))
    return [i for i, t in enumerate(train) if ngrams(normalize(t), n) & bank]

assert flagged(TRAIN, EVAL, 13) == [0]
assert flagged(TRAIN, EVAL, 6) == [0, 1]
assert not set(flagged(TRAIN, EVAL, 6)) & {2, 3, 4}
# without normalization, 13-gram matching misses the contaminated item
def raw_grams(t, n):
    w = t.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
raw_bank = set().union(*(raw_grams(e, 13) for e in EVAL))
assert [i for i, t in enumerate(TRAIN) if raw_grams(t, 13) & raw_bank] == []
print("13-gram and 6-gram decontamination verified.")
```

This shows that normalization and $n$ control what gets flagged. It does not catch a
paraphrased or translated eval item. For that you need the next tool.

**Embedding similarity** (the Check 4.2 rule, eval items against training prompts, flagged pairs
reviewed by hand) catches rephrased benchmark questions with different numbers, the main route
for teacher-generated math. Run both: n-grams for verbatim leaks, embeddings for paraphrase.

Decontaminate against **every** eval set you report, including module 01's `ft_data/eval.jsonl`,
after generation and before mixing; check teacher traces for benchmark answers too
([module 14](./14-distillation-and-reasoning.md)). [Module 09](./09-evaluating-fine-tuned-models.md)
covers the evaluation side: canaries, perturbed benchmarks, public-versus-private score gaps.

---

## 4.7 Synthetic data generation

Every method below gets data that is varied, correct and hard enough to teach something out of a
model whose default outputs are homogeneous. Table 2 lists each one's diversity source, failure
mode and filter.

- **Self-Instruct** ([Wang et al., 2022](https://arxiv.org/abs/2212.10560)) grows a pool from 175
  human seed tasks, admitting a new instruction only if its ROUGE-L to every pooled one is below 0.7.
- **Evol-Instruct** (WizardLM; [Xu et al., 2023](https://arxiv.org/abs/2304.12244)) rewrites
  instructions to be harder (in-depth: constraints, reasoning steps) or rarer (in-breadth), then
  eliminates evolutions that copy the original or cannot be answered.
- **Persona-driven generation** (Persona Hub; [Ge et al., 2024](https://arxiv.org/abs/2406.20094),
  about a billion personas) conditions each generation on a persona ("a retired teacher disputing a
  double charge"). For the running task, personas supply the varied styles and anger levels that
  templated emails lack.
- **Instruction back-translation** ([Li et al., 2023](https://arxiv.org/abs/2308.06259)) has a model
  write the instruction that a high-quality human text answers, then self-curates the pairs.
- **Multi-turn simulation** pairs a user simulator (persona, goal, stopping rule) with the
  assistant. It is the only cheap source of follow-ups such as "actually it was ORD-00552".

**Magpie** ([Xu et al., 2024](https://arxiv.org/abs/2406.08464)) needs no seeds. An aligned chat model
has learned what user turns look like. Give it **only the pre-query template**, the chat template up
to and including the user header (for Qwen3's ChatML, `<|im_start|>user\n`), and it autocompletes a
plausible query; send that query back through the normal template to get the answer. The authors
generated about 4M pairs from Llama-3-Instruct and selected 300K; Llama-3-8B-Base fine-tuned on
them was comparable to the official instruct model on some alignment benchmarks. Its failures are
**topic collapse** (the generator's favorite topics, not your users'), **easy, safety-filtered
queries**, and **self-agreement** (no one catches the answer's errors). Filter with a judge, a topic
classifier, MinHash and semantic dedup, and decontamination; steer the domain with a system prompt
placed before the pre-query template.

### Synthetic preference pairs

**UltraFeedback** ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)) sampled four responses
per prompt from different models and had GPT-4 rate them; the binarized version takes the best as
chosen and a random lower one as rejected. The now more common on-policy variant samples from
**the model you are about to train** and ranks with a judge or reward model. Pitfalls: **length
bias** (judges prefer longer answers, so DPO learns verbosity; measure the chosen-minus-rejected
length gap), **near-identical pairs** (noise gradients; drop small judge-score gaps), and **chosen
from a much stronger model** (teaches "write like the other model" and sits far from your policy,
weakening DPO's implicit reward). [Module 12](./12-preference-optimization.md) covers the losses.

**Tool-use data** needs format, execution and semantic checks on every call (APIGen,
[Liu et al., 2024](https://arxiv.org/abs/2406.18518)); [module 15](./15-agentic-and-tool-use-fine-tuning.md)
owns it, along with the tool-calling schema and its `Json()` storage.

**Table 2. Synthetic generation methods.**

| Method | Seed needed | Diversity source | Main failure mode | Filter to apply |
|---|---|---|---|---|
| Self-Instruct | ~100–200 human tasks | seed pool + ROUGE-L novelty filter | generic, easy tasks | novelty filter, judge for difficulty |
| Evol-Instruct | existing instructions | evolution operators | convoluted, unanswerable prompts | elimination step, judge for answerability |
| Magpie | none (pre-query template) | the model's own prior | topic collapse, low difficulty | topic classifier + rebalancing, judge, semantic dedup |
| Persona-driven | persona pool | personas | persona stereotypes, shallow variation | dedup, judge, persona coverage audit |
| Back-translation | high-quality responses | the response corpus | instructions too tailored to the answer | self-curation score, judge |
| Multi-turn simulation | personas + goals | user simulator | over-cooperative users, short dialogues | turn-count and goal-completion checks |
| Rejection sampling | prompts + verifier | sampling temperature, $k$ | easy prompts dominate | per-prompt cap, verifier, length filter |
| UltraFeedback-style pairs | prompts | several models or samples | length bias, near-identical pairs | length-gap audit, score-gap threshold |
| APIGen-style tool data | API pool | APIs × query templates | calls that parse but do the wrong thing | format, execution and semantic checks |

---

## 4.8 Mixing, curriculum and length

**Domain ratios.** Set the mixture of task, general, reasoning and safety data deliberately and
record it. Upsample rare slices (hard examples, rare `issue_type`s, multi-order emails) rather
than letting the generator decide ([imbalanced data note](../../notes/ml/imbalanced-data-and-pitfalls.md)).

**Replay general data.** Narrow fine-tuning erodes general abilities, a problem called
**catastrophic forgetting** ([module 19](./19-safety-forgetting-and-responsible-fine-tuning.md)).
The cheapest mitigation is to mix in general instruction data, ideally from the same distribution
the base model was post-trained on. A common practitioner **heuristic** for SFT is 5–20% replay.
Continued pretraining on raw text usually replays more, 10–30%
([module 05](./05-supervised-fine-tuning.md)); that is a different setting. Both are starting
points to tune against your forgetting evals, not established constants. Use more replay for
large, narrow task data and less for small sets trained for few steps.

**Curriculum.** Easy-to-hard ordering has mixed evidence; most recipes shuffle. The robust version
is **difficulty filtering**: drop examples the base model already solves and those no teacher
sample solves, and train on the band between (as s1 and LIMO did).

### Tokenized-length analysis

Measure lengths in tokens **after** the target model's chat template (role markers add tens of
tokens per turn; JSON costs more tokens than its characters suggest). With templated prompt
length $p$, completion length $c$ and candidate `max_length` $M$:

$$
\text{frac truncated} = \frac{\#\{p+c > M\}}{n}, \qquad
\text{completion tokens lost} = \frac{\sum_i \min\big(c_i, \max(0, p_i + c_i - M)\big)}{\sum_i c_i}.
$$

`SFTConfig.max_length` defaults to 1024 and `truncation_mode` defaults to `"keep_start"`, which
keeps the first $M$ tokens; `"keep_end"` is deprecated and slated for removal in TRL v2.0
([SFTConfig reference](https://huggingface.co/docs/trl/v1.15.0/en/sft_trainer), as of
October 2026). Under keep_start, **truncation eats the end of the completion first**: a JSON
object cut before its closing brace teaches the model to stop mid-object. If the prompt alone fills
$M$, the example has **zero trained tokens**; it costs a forward pass, adds no loss and can distort
per-batch normalization, so filter it out.

Choose $M$ at p95–p99 of the templated length. Then drop the tail, pack
([module 05](./05-supervised-fine-tuning.md); fewer truncations help, per
[Ding et al., 2024](https://huggingface.co/papers/2404.10830)), or shorten prompts; far longer
sequences are [module 17](./17-long-context-extension.md)'s.

Worked example: ten examples with $(p, c)$ from $(40, 60)$ to $(300, 1500)$ at $M=1024$. Two (20%)
are truncated; 972 of 6,110 tokens (15.9%) are cut, and 18.6% of **completion** tokens are lost,
because keep_start cuts from the end. Check 4.4 asserts these numbers, the format
conversions from section 4.1, and the zero-trained-token drop rule on a 12-token prompt with an
8-token completion.

```python finetuning-check
SYS = {"role": "system", "content": "Extract the fields as JSON."}
U1 = {"role": "user", "content": "Order ORD-00551 never came. - Dana Kim"}
A1 = {"role": "assistant", "content": '{"customer_name": "Dana Kim", "order_id": "ORD-00551"}'}
U2 = {"role": "user", "content": "Add refund_requested too."}
A2 = {"role": "assistant", "content": '{"customer_name": "Dana Kim", "order_id": "ORD-00551", "refund_requested": false}'}
record = {"messages": [SYS, U1, A1, U2, A2]}

def to_prompt_completion(rec):
    msgs = rec["messages"]
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    return {"prompt": msgs[:last], "completion": msgs[last:last + 1]}

def to_implicit_preference(rec, rejected_text):
    pc = to_prompt_completion(rec)
    bad = {"role": "assistant", "content": rejected_text}
    return {"chosen": pc["prompt"] + pc["completion"], "rejected": pc["prompt"] + [bad]}

pc = to_prompt_completion(record)
assert pc == {"prompt": [SYS, U1, A1, U2], "completion": [A2]}
pref = to_implicit_preference(record, '{"customer_name": "Dana Kim"}')
assert pref == {
    "chosen": [SYS, U1, A1, U2, A2],
    "rejected": [SYS, U1, A1, U2, {"role": "assistant", "content": '{"customer_name": "Dana Kim"}'}],
}
# explicit-prompt form is recovered by splitting off the shared prefix
assert pref["chosen"][:4] == pref["rejected"][:4] == pc["prompt"]

# keep_start truncation of a tokenized prompt-completion example
def truncate_keep_start(ids, mask, max_length):
    return ids[:max_length], mask[:max_length]

prompt_ids = list(range(100, 112))           # 12 prompt tokens
completion_ids = list(range(200, 208))       # 8 completion tokens
ids = prompt_ids + completion_ids
mask = [0] * 12 + [1] * 8                    # 1 = token contributes to the loss

ids16, mask16 = truncate_keep_start(ids, mask, 16)
assert ids16 == ids[:16] and len(ids16) == 16
assert sum(mask16) == 4                      # only 4 of 8 completion tokens survive
ids12, mask12 = truncate_keep_start(ids, mask, 12)
assert sum(mask12) == 0                      # nothing left to train on

def keep(mask_after_truncation, min_trained_tokens=1):
    return sum(mask_after_truncation) >= min_trained_tokens

assert keep(mask16) and not keep(mask12)

# dataset-level truncation statistics for a choice of max_length
lengths = [(40, 60), (50, 90), (30, 150), (80, 200), (60, 300),
           (70, 420), (90, 610), (40, 800), (120, 1100), (300, 1500)]   # (prompt, completion)
M = 1024
total = [p + c for p, c in lengths]
frac_truncated = sum(t > M for t in total) / len(total)
lost_tokens = sum(max(0, t - M) for t in total)
lost_completion = sum(min(c, max(0, p + c - M)) for p, c in lengths)
assert frac_truncated == 0.2
assert (lost_tokens, sum(total)) == (972, 6110)
assert round(lost_tokens / sum(total), 3) == 0.159
assert round(lost_completion / sum(c for _, c in lengths), 3) == 0.186
print("Schema conversion, keep_start truncation and drop rule verified.")
```

This proves the bookkeeping, not TRL's tokenizer-level behavior: measure real lengths with
`apply_chat_template` on the actual tokenizer, as the lab does.

---

## 4.9 Tooling map

**Table 3. Data tooling (as of October 2026).**

| Tool | What it is | Use it when |
|---|---|---|
| `datasets` | Arrow-backed loading, `map`/`filter`, streaming, Hub I/O | always; it is the interchange format every trainer reads |
| [distilabel](https://distilabel.argilla.io/) | pipelines of LLM steps: generation, judges (UltraFeedback-style), embeddings; its [`MinHashDedup`](https://distilabel.argilla.io/dev/components-gallery/steps/minhashdedup/) step wraps `datasketch` (`num_perm` 128, `threshold` 0.9, words or n-grams) and writes a `keep_row_after_minhash_filtering` column | building a synthetic or judged dataset as a reproducible graph, dedup right after generation |
| [datatrove](https://github.com/huggingface/datatrove/tree/1977fbb0f3c163d43cace334b073dda17fa3ef26) | large-scale Gopher/C4/FineWeb/fastText filters and exact, sentence and MinHash dedup (four stages; default 5-gram word shingles, 14 buckets × 8 hashes per [`MinhashConfig`](https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/dedup/minhash.py#L41-L53)) on local, Slurm or Ray executors | millions of documents, continued-pretraining corpora, CPU clusters |
| [Unsloth Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe) | visual graph workflow in Unsloth Studio that turns PDFs, CSVs and Hub datasets into synthetic datasets (built on NVIDIA NeMo Data Designer) | quick document-to-dataset generation feeding an Unsloth fine-tune |
| Your own scripts (this module's lab) | schema checks, dedup, decontamination, length report | small and medium sets where you want every step visible and tested |

[Module 16](./16-multimodal-and-embedding-fine-tuning.md) covers hard-negative mining for embedding models.

---

## 4.10 Lab: a clean, decontaminated SFT set

**Goal.** Validated SFT and preference sets from support emails, then Magpie and rejection-sampled math.

| Part | Tier | Model | Data | Script |
|---|---|---|---|---|
| L4.1 formats, dedup, decontam, length | CPU | none (smoke) / `Qwen/Qwen3-0.6B` tokenizer (full) | inline corpus or module 01's `ft_data/train.jsonl` | [build_sft_dataset.py](./code/build_sft_dataset.py) |
| L4.2 Magpie generation | T1 or API | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (non-thinking, Apache-2.0) | generated | [build_sft_dataset.py](./code/build_sft_dataset.py) `--magpie` |
| L4.3 rejection-sampled math | T1 | same teacher | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), config `main`, split `train` | [rejection_sample_gsm8k.py](./code/rejection_sample_gsm8k.py) |

```bash
python content/courses/finetuning/code/build_sft_dataset.py --smoke
python content/courses/finetuning/code/rejection_sample_gsm8k.py --smoke
python content/courses/finetuning/code/build_sft_dataset.py --smoke \
    --input ft_data/train.jsonl --eval ft_data/eval.jsonl          # CPU, whitespace token counts

python content/courses/finetuning/code/build_sft_dataset.py --full \
    --input ft_data/train.jsonl --eval ft_data/eval.jsonl --tokenizer Qwen/Qwen3-0.6B --output-dir out/
python content/courses/finetuning/code/build_sft_dataset.py --full --magpie 2000 \
    --eval ft_data/eval.jsonl --output-dir out/
python content/courses/finetuning/code/rejection_sample_gsm8k.py --full --k 8 --limit 2000 \
    --out gsm8k_rs.jsonl
```

Create `ft_data/` first with module 01's `fit_or_not_baseline.py --data-only`. Smoke runs take
about a second on CPU (stdlib only); `--full` imports `transformers`, `datasets` and `vllm` lazily.

### L4.1 Formats, dedup and decontamination (CPU)

The smoke corpus has 16 hand-written support emails with gold extractions in module 01's
schema, plus five planted problems: an exact duplicate, a casing-and-punctuation duplicate, a
near-duplicate with two words changed, a forwarded copy of a golden eval email, and a record whose
`gold` violates the schema.

The script runs the schema → exact dedup → MinHash → decontaminate → length stages of the
section 4.4 pipeline and asserts the row counts: 21 raw, 20 after schema, 18 after exact dedup, 17
after MinHash, 16 after decontamination. It then emits language-modeling, prompt-completion,
conversational and preference rows (the assistant target is `json.dumps(gold)` in module 01's key
order), validates each, and writes the length report. The preference rows' rejected side is the
gold JSON minus `refund_requested`: a format exercise, not real preference data.

On module 01's 200 training rows, all 200 pass the schema check, exact dedup and decontamination
remove none (eval uses disjoint templates), and MinHash at $32\times4$ keeps 57. **Templated
synthetic data is near-duplicate data**, and the dedup stage is how you find out. Module 05 still
trains on all 200; here you measure how little independent variety they contain. Record the rows
removed per stage, the p50/p95/p99 templated lengths, the chosen `max_length` and the completion
tokens lost.

### L4.2 Magpie generation, judge filter, publish (T1 or API)

`--magpie N` calls vLLM's `LLM.generate` on the bare pre-query template `<|im_start|>user\n`
(stop at `<|im_end|>`), keeps queries of 20–2,000 characters, answers them with `LLM.chat`, applies
MinHash and 13-gram decontamination, and writes `magpie.jsonl`. The engine is seeded once
(`LLM(seed=...)`): **never put a fixed per-request seed on identical Magpie prompts**, or every
request samples the same query and MinHash collapses 2,000 rows to about one. Then judge-filter
with a larger model (`Qwen/Qwen3-32B` on vLLM, or any API judge your terms allow).

Teaching fragment: not executed (needs a GPU or an API).

```python
RUBRIC = ("Rate the user query and answer from 1-10 for: clear and answerable query, "
          "correct and complete answer, appropriate difficulty. Reply with only the integer.")

def judge_score(llm, conv):                        # llm: a vLLM LLM serving the judge
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": json.dumps(conv["messages"])}]
    out = llm.chat([msgs], SamplingParams(temperature=0.0, max_tokens=4))[0].outputs[0].text
    return int(out.strip()) if out.strip().isdigit() else 0

kept = [c for c in rows if judge_score(judge, c) >= 7]   # threshold chosen on 100 hand-labeled rows
```

Calibrate the threshold on 100 rows you label yourself, cluster the query embeddings to see the
topic collapse, and rebalance. `--push-to-hub user/repo` pushes the unjudged `magpie.jsonl` as
config `magpie`; push the judged rows with `Dataset.from_list(kept).push_to_hub("user/repo",
config_name="magpie_judged", private=True)`, then write the card (section 4.2). Expect dedup and the
judge to remove a large share, and a topic histogram unlike real support traffic.

### L4.3 A verified math SFT set by rejection sampling (T1)

`rejection_sample_gsm8k.py --full` samples $k=8$ solutions per GSM8K training problem at
temperature 0.8. It extracts each sample's final answer and keeps samples that match the
reference. It also deduplicates identical solutions and caps kept rows at 4 per problem. The
extractor is the piece most likely to be wrong, so the smoke mode tests it on `#### 1,234`, the
last of several `\boxed{}`, `-3.50` == `-3.5`, a leading dollar sign, and text with no number.

On the fake teacher, the smoke run asserts a keep rate of 10/24, 2/3 problem coverage, and 7 rows
after dedup and capping.

Record the **sample keep rate**, **problem coverage** (problems with at least one kept solution),
the kept-versus-all word-length distribution (rambling samples run out of tokens before the answer),
and the zero-kept problems: your hard set for a stronger teacher, a larger $k$, or RL with a
verifier ([module 13](./13-rl-with-verifiable-rewards.md)).

Expect high coverage from a modern 4B instruct teacher, a lower per-sample keep rate, and shorter
kept solutions. Decontaminate against the GSM8K **test** split before reporting any math score.

## Key takeaways

- **The dataset type decides what the trainer trains on.** Prompt-completion data gets
  completion-only loss by default; `messages` data needs `assistant_only_loss=True` and a
  `{% generation %}` template; prompt-only RL rows must carry the verifier's columns (`gold`).
- **Thumbs-up/down logs are unpaired preference data.** They fit KTO; DPO needs same-prompt pairs.
- **Filter cheapest first, and validate the schema early.** For structured tasks it removes more
  bad rows than any model-based filter.
- **Rejection sampling turns a verifier into a data generator.** Sampling $k$ times raises
  coverage to $1-(1-p)^k$; cap rows per prompt so easy prompts do not dominate.
- **MinHash estimates Jaccard without bias**, standard error $\sqrt{J(1-J)/k}$; LSH banding sets a threshold near $(1/b)^{1/r}$.
- **Use exact, MinHash and semantic dedup, then decontaminate with normalized n-grams and
  embeddings against every eval you report.** Copies, light edits and paraphrases each need
  their own tool; a casing change defeats an unnormalized 13-gram match.
- **Every synthetic method gets diversity from somewhere specific and fails in a known way.**
  Magpie collapses toward the model's favorite topics; judged pairs inherit length bias.
- **Replay of general data is the cheapest guard against forgetting.** 5–20% for SFT (10–30%
  for continued pretraining) is a heuristic starting point, not a constant.
- **Measure lengths in templated tokens and pick `max_length` at p95–p99.** keep_start
  truncation cuts completions first, and an example truncated to zero trained tokens must be dropped.
- **A dataset without a card is not reproducible.** Record sources, licenses, generators,
  filters, thresholds, and dedup and decontamination settings.

## Self-check

1. Describe the Magpie data-generation trick. What are its failure modes, and how would you
   filter its output?
2. How would you detect and prevent benchmark contamination in a synthetic training set
   generated by a strong teacher model?
3. You have six months of production thumbs-up/down logs. How do you turn them into
   preference-training data, and which training method fits the data you actually have?

---

**Next → [05 — Supervised Fine-Tuning and Continued Pretraining](./05-supervised-fine-tuning.md)**
