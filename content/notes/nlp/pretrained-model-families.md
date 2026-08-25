---
order: 8
description: BERT and the encoder family, GPT and the decoder family, T5 and encoder-decoders, multilingual and domain models, and how to choose a checkpoint for a task.
meta: NLP · models
---

# Pretrained Model Families

There are three architectural families, a handful of pretraining objectives, and
a long list of checkpoints. Knowing which family fits which task — and why the
field converged on decoder-only for generation — saves more time than any amount
of prompt engineering.

## The three families

```mermaid
flowchart TD
    T["Transformer, 2017<br/>encoder-decoder for translation"] --> E["ENCODER-ONLY<br/>bidirectional attention<br/>masked language modelling"]
    T --> D["DECODER-ONLY<br/>causal attention<br/>next-token prediction"]
    T --> ED["ENCODER-DECODER<br/>bidirectional encoder,<br/>causal decoder, cross-attention"]
    E --> E1["BERT, RoBERTa, DeBERTa,<br/>ELECTRA, ModernBERT"]
    D --> D1["GPT, Llama, Mistral,<br/>Qwen, Gemma, DeepSeek"]
    ED --> ED1["T5, BART, mT5,<br/>Whisper, NLLB"]
    E1 --> EU["classification, NER,<br/>retrieval embeddings,<br/>reranking"]
    D1 --> DU["generation, chat, agents,<br/>and in practice everything"]
    ED1 --> EDU["translation, summarisation,<br/>speech recognition"]
```

## Encoder-only

### BERT

Two pretraining objectives, one of which turned out to matter:

- **Masked language modelling** — mask 15% of tokens and predict them from
  bidirectional context. The 80/10/10 split (replace with `[MASK]` / a random
  token / unchanged) exists because `[MASK]` never appears at fine-tuning time,
  so training on it exclusively creates a train–inference mismatch.
- **Next sentence prediction** — do these two segments follow each other? RoBERTa
  showed this **hurt**, and it was dropped from essentially every successor.

Bidirectional context is BERT's defining property and the reason it dominated
classification and tagging: predicting a masked word uses both sides, which is
strictly more information than a causal model has.

### The successors

| Model | Change |
|---|---|
| **RoBERTa** | drop NSP, more data, longer training, dynamic masking, larger batches — the same architecture trained properly |
| ALBERT | factorised embeddings, cross-layer parameter sharing — fewer parameters, not faster |
| **DeBERTa-v3** | disentangled content/position attention, ELECTRA-style pretraining — consistently the strongest base-size encoder |
| **ELECTRA** | replaced-token detection: a generator corrupts tokens, a discriminator finds them. Trains on **100%** of positions rather than 15%, so it is far more sample-efficient |
| DistilBERT | distilled: 40% smaller, ~97% of the quality, 60% faster |
| **ModernBERT** | 2024-era: 8k context, RoPE, FlashAttention, modern data mixture, much faster |
| Longformer / BigBird | sparse attention for long documents |

**ELECTRA's insight is worth internalising**: MLM computes a loss on only 15% of
positions, wasting 85% of each forward pass. Replaced-token detection gives every
position a binary training signal, which is why ELECTRA-style pretraining reaches
BERT quality with a fraction of the compute.

### Why encoders still matter

Encoders are widely written off and should not be. For a task with a few thousand
labelled examples, a fine-tuned 100M-parameter encoder **beats a prompted 70B
model** at a thousandth of the cost and a fraction of the latency. They also
remain the correct architecture for retrieval and reranking, where bidirectional
attention over a fixed input is exactly what is needed.

Use an encoder for: classification, NER and token tagging, extractive QA,
sentence embeddings, and cross-encoder reranking.

## Decoder-only

Trained on next-token prediction with causal attention. The family that won.

| Generation | Models | Notable |
|---|---|---|
| GPT-1/2 | 117M–1.5B | showed unsupervised pretraining transfers |
| GPT-3 | 175B | in-context learning at scale |
| Chinchilla | 70B | compute-optimal scaling — 20 tokens per parameter |
| **Llama 1–3** | 7B–405B | open weights, RoPE, RMSNorm, SwiGLU, GQA; trained far past Chinchilla-optimal |
| Mistral / Mixtral | 7B, 8×7B | sliding-window attention; sparse mixture-of-experts |
| Qwen, Gemma, Phi | various | strong open models; Phi demonstrates curated/synthetic data |
| DeepSeek | various | multi-head latent attention, MoE, and RL-trained reasoning |
| Frontier proprietary | GPT, Claude, Gemini | undisclosed architectures |

### The modern decoder recipe

The architectural choices that converged across essentially every 2023+ model:

| Component | Choice | Reason |
|---|---|---|
| Normalisation | **pre-RMSNorm** | stable training without careful warmup; cheaper than LayerNorm |
| Position | **RoPE** | relative position, extends to longer contexts via frequency scaling |
| FFN activation | **SwiGLU** | consistently better than GELU at matched parameters |
| Attention | **GQA** | 4–8× smaller KV cache at negligible quality cost |
| Bias terms | usually removed | no measurable benefit, fewer parameters |
| Vocabulary | 32k–256k | larger vocabularies help multilingual coverage |
| Context | 8k–1M | long-context training and RoPE scaling |
| Optimiser | AdamW, $\beta_2 = 0.95$ | shorter second-moment window for stability |

**Mixture of experts** is the other major structural choice: replace the FFN with
$N$ experts and route each token to $k$ of them (typically $k = 1$ or 2). Total
parameters grow while per-token compute stays fixed, so an 8×7B MoE has ~47B
parameters and roughly 13B active per token. The costs are memory (all experts
must be resident), load-balancing complexity, and harder fine-tuning.

### Why decoder-only won

1. **Universal objective.** Next-token prediction applies to any text with no
   annotation.
2. **Full training signal.** Every position contributes a loss in one forward
   pass; MLM uses 15%.
3. **Simplicity.** No cross-attention, one attention pattern, one stack.
4. **Scaling behaviour.** Clean, predictable power laws.
5. **In-context learning.** Emerges from the objective, and turns one model into
   many task-specific ones without training.

## Encoder–decoder

### T5

Cast **every** task as text-to-text. Translation, classification, regression,
summarisation — all become "input text → output text", with a task prefix.
Pretrained with **span corruption**: mask contiguous spans and generate them,
which is closer to the generation task than BERT's single-token masking.

The unified framing was influential well beyond T5, and the "instruction as
prefix" idea is a direct ancestor of instruction tuning.

### The family

| Model | Note |
|---|---|
| T5, Flan-T5 | Flan-T5 adds large-scale instruction tuning; still competitive for its size |
| BART | denoising autoencoder (token masking, deletion, sentence permutation); strong for summarisation |
| mT5, mBART | multilingual |
| **Whisper** | encoder–decoder for speech-to-text; the standard open ASR model |
| NLLB-200 | 200-language translation |
| Pegasus | gap-sentence pretraining, designed for summarisation |

Encoder–decoders remain the right choice where input and output are genuinely
different objects and the input must be encoded once and attended to many times —
translation, speech recognition, and summarisation of a fixed source.

## Multilingual models

| Model | Coverage |
|---|---|
| mBERT | 104 languages |
| **XLM-R** | 100 languages; strong cross-lingual transfer |
| mDeBERTa | 100 languages; stronger than XLM-R |
| mT5 / umT5 | 101 languages, text-to-text |
| BLOOM | 46 languages, open training data |
| NLLB-200 | 200 languages, translation-focused |
| Aya | 101 languages, instruction-tuned |

**The curse of multilinguality**: at fixed capacity, adding languages helps
low-resource ones and hurts high-resource ones. Mitigations are more parameters,
language-specific adapters, MoE routing, and temperature-sampling the language
mix so low-resource languages are over-sampled relative to their corpus share.

**Cross-lingual zero-shot transfer** is the practical payoff: fine-tune XLM-R on
English NER and it works reasonably on 50 other languages without any labelled
data in them. This is the single most useful property of multilingual encoders.

## Domain-specific models

| Domain | Models |
|---|---|
| Biomedical | BioBERT, PubMedBERT, BioGPT, Med-PaLM |
| Clinical | ClinicalBERT, GatorTron |
| Scientific | SciBERT, SPECTER (paper embeddings) |
| Legal | LegalBERT, CaseLawBERT |
| Finance | FinBERT, BloombergGPT |
| Code | CodeBERT, StarCoder, CodeLlama, Qwen-Coder |
| Chemistry / proteins | ChemBERTa, ESM-2, ProtBERT |
| Tabular / time series | TabPFN, Chronos, TimesFM |

**Domain pretraining wins when the vocabulary and distribution genuinely differ.**
PubMedBERT trained from scratch on biomedical text beats BERT continued-pretrained
on it, because a domain-specific tokenizer vocabulary matters — general
tokenizers shatter medical terms into many meaningless pieces.

Check the tokenizer's **fertility** (tokens per word) on your domain text. If
your key terms cost 6 tokens each, a domain model or vocabulary extension will
pay for itself.

## Choosing a checkpoint

| Need | Pick |
|---|---|
| Text classification, 1k+ labels | DeBERTa-v3-base or ModernBERT |
| NER / token tagging | DeBERTa-v3 or a domain encoder |
| Sentence embeddings | BGE, E5, GTE — check MTEB for your task type |
| Reranking | a cross-encoder (bge-reranker, monoT5) |
| Generation, chat, agents | a decoder-only instruct model sized to your latency budget |
| Translation | NLLB, or an LLM for high-resource pairs |
| Speech to text | Whisper |
| Code | StarCoder2, CodeLlama, Qwen-Coder |
| Multilingual classification | XLM-R or mDeBERTa |
| Very long documents | ModernBERT, Longformer, or a long-context LLM |
| Tight latency or CPU-only | DistilBERT, MiniLM, or a distilled small LLM |

### Practical selection criteria

| Criterion | Why |
|---|---|
| **Licence** | Apache-2.0, Llama community licence, research-only — check before building on it |
| Size vs latency budget | a 7B model needs a GPU; a 100M encoder runs on CPU |
| Context length | do your documents fit? |
| Tokenizer fertility on your domain | 3× more tokens is 3× the cost |
| **Base vs instruct** | base models complete text, instruct models follow instructions |
| Training data recency | knowledge cutoff |
| Community support | quantised versions, serving support, fine-tuning recipes |
| Benchmark relevance | evaluate on **your** data, not a leaderboard average |

**Base and instruct models behave completely differently** and confusing them is
a common early mistake. A base model given "What is the capital of France?" may
continue with more questions, because that is what its training distribution
contains.

**Benchmark contamination is pervasive.** Assume public test sets are in the
training data of any model trained on the web. A private evaluation set drawn
from your own distribution is worth more than any leaderboard position.

## Sizes and what they can do

| Size | Runs on | Capability |
|---|---|---|
| < 500M encoder | CPU | classification, NER, embeddings — excellent when fine-tuned |
| 1–3B | consumer GPU, quantised CPU | simple instruction following, classification, summarisation |
| 7–8B | one consumer GPU | the practical open-model workhorse; good general capability |
| 13–34B | one datacentre GPU | noticeably better reasoning |
| 70B+ | multi-GPU | strong general capability |
| Frontier | API | best reasoning, long context, tool use |

**Task-specific fine-tuning of a small model frequently beats a large model
prompted zero-shot**, at a fraction of the cost. The large model's advantage is
generality, not per-task quality — which is why the LLM-labels-then-distil
pipeline is so effective.

## Self-check

1. Why did RoBERTa drop next sentence prediction?
2. What does ELECTRA's objective change, and why is it more sample-efficient?
3. Give five architectural choices shared by essentially every modern
   decoder-only LLM, and the reason for each.
4. Why did decoder-only win over encoder–decoder for general use?
5. When is a 100M encoder the right answer over a 70B LLM?
6. What is the curse of multilinguality and what mitigates it?
7. How would you decide whether a domain-specific model is worth using?

## Where to go next

- [Language Models](./language-models.md) — objectives, perplexity, scaling.
- [LLM Prompting & Alignment](./llm-prompting-and-alignment.md) — turning a base
  model into an assistant.
- [Transformers Deep Dive](/courses/transformers/) — the architecture in detail.
