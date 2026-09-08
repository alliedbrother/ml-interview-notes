---
order: 6
description: Statistical MT, seq2seq and the bottleneck, attention and the transformer, multilingual and low-resource translation, BLEU and its successors, and the practical realities of deploying translation.
meta: NLP · tasks
---

# Machine Translation

Machine translation drove more architectural innovation than any other NLP task.
Attention was invented for it. The transformer paper is a translation paper.
BLEU and many influential attention/back-translation applications grew from MT;
beam search and subword/compression methods also have histories outside translation.
Its architectural arc is central to understanding modern NLP.

## The arc

```mermaid
flowchart TD
    RULE["rule-based MT<br/>hand-written grammars<br/>and dictionaries"] -->|"unmaintainable,<br/>brittle"| SMT["statistical MT<br/>learn alignments and phrase<br/>tables from parallel text"]
    SMT -->|"pipeline of separately<br/>trained components"| NMT["neural seq2seq<br/>one model, end to end"]
    NMT -->|"one fixed-size vector<br/>for the whole sentence"| ATT["seq2seq + attention<br/>a different context<br/>per output token"]
    ATT -->|"if attention does the work,<br/>is recurrence needed?"| TRF["Transformer<br/>attention only,<br/>fully parallel"]
    TRF --> MULTI["multilingual and<br/>massively multilingual models"]
    TRF --> LLM["general LLMs<br/>translation as one capability"]
```

## Statistical machine translation

The noisy-channel formulation: to translate source $f$ into target $e$,

$$\hat{e} = \arg\max_e P(e\mid f) = \arg\max_e P(f\mid e)\,P(e)$$

Two separately trained components: a **translation model** $P(f\mid e)$ learned
from parallel text, and a **language model** $P(e)$ learned from monolingual
target text. The language model is what makes the output fluent, and having it as
a separate component was a genuine strength — target-language monolingual data is
far more abundant than parallel data.

The IBM Models 1–5 introduced **word alignment** as a latent variable trained
with EM. Phrase-based SMT extended this to multi-word units, which handled
idioms and local reordering far better.

SMT's weaknesses were structural: a pipeline of separately-tuned components
(alignment, phrase extraction, reordering model, language model, tuning), an
enormous phrase table, and no ability to generalise beyond observed phrases.

## Neural machine translation

### Seq2seq and the bottleneck

An encoder RNN compresses the source into a vector; a decoder RNN generates the
target from it. Elegant, end-to-end, and limited by one thing: **everything the
decoder knows about the source passes through a single fixed-size vector.**
Longer sentences exposed this bottleneck in early experiments; twenty tokens is
not a universal architectural failure boundary.

### Attention

Instead of one context vector, compute a **different** weighted combination of
encoder states at each output step:

$$e_{tj} = a(\mathbf{s}_{t-1},\mathbf{h}_j), \qquad \alpha_{tj} = \mathrm{softmax}_j(e_{tj}), \qquad \mathbf{c}_t = \sum_j \alpha_{tj}\mathbf{h}_j$$

Bahdanau's additive scoring ($a = \mathbf{v}^\top\tanh(W[\mathbf{s};\mathbf{h}])$)
came first; Luong's multiplicative variant
($a = \mathbf{s}^\top W\mathbf{h}$) is cheaper and became standard.

The effect on long sentences was dramatic, and the attention weights turned out
to align roughly with word correspondences — giving a free, inspectable
soft-alignment matrix that SMT had needed a separate model to produce.

### Transformer

Then the obvious question: if attention does the work, is the recurrence needed?
"Attention Is All You Need" answered no, and the encoder–decoder transformer it
introduced remains the reference architecture for dedicated MT systems.

The decoder has **two** attention mechanisms: masked self-attention over the
target generated so far, and **cross-attention** over the encoder output. That
separation provides target-context and source-conditioned computation. Both are
jointly trained inside $P(e\mid f)$; they are not separate factors $P(e)$ and
$P(f\mid e)$ from the noisy-channel model.

## Data

Parallel corpora are the constraint.

| Source | Note |
|---|---|
| Europarl, UN corpus | high quality, formal register, limited domains |
| OPUS | an aggregation of many corpora |
| ParaCrawl, CCMatrix | web-mined; large and noisy |
| **Back-translation** | translate target-language monolingual text into the source, use the synthetic pair |
| Multilingual pivoting | translate via a high-resource language |
| LLM-generated pairs | increasingly viable, needs verification |

**Back-translation is the single most effective data technique in MT.** Take
abundant monolingual target text, translate it into the source with a
reverse-direction model, and train on the (synthetic source, real target) pair.
The target side is real and fluent, which is the side that matters for output
quality, and the noisy source side acts as a regulariser. It routinely adds
several BLEU points and is standard practice.

**Corpus filtering matters more than corpus size.** Web-mined parallel data is
full of misalignments, boilerplate, machine-translated text, and wrong-language
pairs. Filtering with a cross-lingual similarity model (LASER, LaBSE) before
training reliably improves quality despite reducing data volume.

## Multilingual translation

One model, many language pairs. Add a target-language tag to the input and share
all parameters.

| Effect | Direction |
|---|---|
| **Transfer to low-resource pairs** | strongly positive — related languages share representation |
| **Zero-shot translation** | possible between pairs never seen together in training |
| Parameter efficiency | one model instead of $N^2$ |
| **Capacity dilution** | negative for high-resource pairs — the "curse of multilinguality" |
| Vocabulary sharing | efficient for related scripts, wasteful across scripts |

The tension is real: adding languages helps the low-resource ones and hurts the
high-resource ones at fixed capacity. Mitigations are language-specific adapters,
mixture-of-experts routing, temperature-based sampling of the language mix, and
simply making the model bigger.

Massively multilingual models — mBART, M2M-100, NLLB-200 — cover 100–200
languages in one model, and NLLB explicitly targeted low-resource languages that
commercial systems ignore.

## LLMs as translators

General-purpose LLMs are now competitive with or better than dedicated MT systems
for high-resource pairs, and worse for low-resource ones.

| LLM advantage | Dedicated MT advantage |
|---|---|
| Document-level context and coherence | much lower cost per token |
| Follows style and terminology instructions | lower latency |
| Handles idioms and cultural adaptation better | better on low-resource pairs |
| Can explain choices, offer alternatives | deterministic and auditable |
| No per-pair model needed | smaller footprint |
| Better at register and formality control | — |

The most interesting LLM advantage is **document-level translation**: pronoun
resolution, consistent terminology, and register agreement across sentences all
require context that sentence-level MT systems structurally do not have.

## Evaluation

### BLEU

$$\mathrm{BLEU} = \mathrm{BP}\cdot\exp\left(\sum_{n=1}^{4}w_n\log p_n\right), \qquad \mathrm{BP} = \min\left(1, e^{1-r/c}\right)$$

Modified $n$-gram precision for $n = 1..4$, geometrically averaged, with a
**brevity penalty** because precision alone rewards short output.

The "modified" part matters: each $n$-gram's count is clipped at its maximum
count in the reference, so repeating "the the the the" cannot inflate unigram
precision.

**BLEU's problems**, and they are serious:

- No credit for synonyms or paraphrase — a perfect translation using different
  words scores poorly.
- No notion of meaning or grammaticality.
- Correlates weakly with human judgement at the sentence level.
- **Not comparable across papers** unless tokenisation, casing, and the number of
  references match. This is why **sacreBLEU** exists: it standardises
  tokenisation and emits a signature describing the configuration.
- Poor at distinguishing strong systems from each other, which is exactly the
  regime modern MT operates in.

Use sacreBLEU, always, and report the signature.

### The alternatives

| Metric | Type | Note |
|---|---|---|
| **chrF / chrF++** | character F-score; chrF++ adds word n-grams | character matching reduces tokenization sensitivity; chrF++ word segmentation still matters |
| TER | edit distance to the reference | interpretable as post-editing effort |
| METEOR | unigram matching with stems and synonyms | better sentence-level correlation |
| **BERTScore** | contextual embedding similarity | credits paraphrase |
| **COMET** | a trained neural metric using source, hypothesis, and reference | the current standard; much better human correlation |
| **COMET-QE / CometKiwi** | **reference-free** quality estimation | can score production output with no reference |
| BLEURT | trained regression metric | similar family to COMET |
| Human evaluation | direct assessment, MQM error annotation | the ground truth |

COMET-family metrics can support system comparisons, but specify checkpoint,
language/domain coverage and human validation. Reference-free quality estimation scores live
production translations, route low-confidence outputs to human review, and detect
degradation without maintaining a reference set.

**MQM** (multidimensional quality metrics) is the human protocol worth knowing:
annotators mark specific errors by category and severity rather than giving a
holistic score, which is far more reliable and more actionable.

## Decoding

| Strategy | Note |
|---|---|
| Greedy | fast, noticeably worse |
| **Beam search** | the standard; beam 4–5 |
| Length normalisation | essential — otherwise beam search prefers short output |
| Coverage penalty | discourages repeating or dropping source content |
| Minimum Bayes risk | pick the candidate most similar to other candidates under a metric |
| Sampling | for diversity; generally worse for translation |

**The beam search curse** describes experiments where increasing beam width
worsened translation metrics despite finding higher-probability sequences. Five
is not a universal threshold. The model's probability distribution and
translation quality diverge — larger beams find degenerate high-probability
outputs, typically too short or overly generic. It is a clean example of a model
whose objective and whose goal are not the same function.

**MBR decoding** attacks this directly: instead of maximising probability,
sample many candidates and pick the one with the highest average similarity to
the others under a chosen utility metric. Gains depend on candidate diversity,
the utility's validity and budget; it does not consistently win every comparison.

## Practical deployment

| Concern | Handling |
|---|---|
| **Terminology control** | constrained decoding, terminology injection, or a glossary in the prompt |
| Formatting and tags | protect HTML/XML/placeholders from being translated or reordered |
| Numbers, dates, units | locale-aware formatting; MT models get these wrong |
| Named entities | should usually pass through untranslated; entity-aware handling |
| Domain adaptation | fine-tune on in-domain parallel data; even a few thousand pairs helps a lot |
| Quality estimation | route low-confidence segments to human post-editing |
| Translation memory | reuse exact and fuzzy matches from previous translations |
| Latency | distil to a smaller model; quantise; batch |
| Gender and bias | "the doctor" defaults to masculine in many target languages; provide alternatives or context |

**Terminology control is the most common enterprise requirement** and the one
generic systems handle worst. A pharmaceutical company needs a drug name rendered
exactly one way, every time. Constrained decoding that forces specified target
strings is the reliable solution; prompt-based glossaries with an LLM work but
are not guaranteed.

**Gender bias in translation is well documented and structural.** Translating
from a gender-neutral language into a gendered one forces a choice, and the model
makes it from training-data statistics — nurses become feminine, engineers
masculine. Serious systems detect the ambiguity and offer both, which is what
Google Translate does for short queries.

## Related generation tasks

The same encoder–decoder machinery, different data:

| Task | Note |
|---|---|
| **Summarisation** | extractive (select sentences) or abstractive (generate); faithfulness is the hard problem |
| Paraphrasing | often trained via round-trip translation |
| Grammatical error correction | monolingual "translation" from erroneous to correct |
| Style transfer | formality, simplification, register |
| Data-to-text | tables or knowledge graphs to prose |
| Simplification | for accessibility or reading level |

**Summarisation's central difficulty is faithfulness**, not fluency. Abstractive
models produce readable summaries containing facts absent from the source. The
countermeasures — entailment-based faithfulness metrics, question-answering
consistency checks, and explicit grounding — matter more than any ROUGE
improvement, because a fluent unfaithful summary is worse than a clumsy faithful
one.

## Self-check

### Worked alignment and tiny encoder-decoder contract

In IBM Model 1, suppose a source word has candidate target alignments to `house`
and `home` with lexical scores .6 and .3, ignoring NULL for this small example.
The posterior responsibilities are $2/3$ and $1/3$. Add those fractional counts
across sentence pairs, then normalize lexical counts for each conditioning word
in the M-step. The alignment is latent: choosing only the maximum would be hard
assignment, not this EM update.

Attention with scores $(0,\log2)$ gives weights $(1/3,2/3)$; encoder states
$(1,0)$ and $(0,3)$ then produce context $(1/3,2)$. This is conditional feature
aggregation, not a standalone noisy-channel translation factor.

```python runnable
import os
os.environ["USE_TF"] = "0"  # select the PyTorch backend before importing Transformers
import torch
from transformers import T5Config, T5ForConditionalGeneration
torch.manual_seed(3)
torch.set_num_threads(1)
config = T5Config(vocab_size=16, d_model=16, d_ff=24, num_layers=1,
    num_decoder_layers=1, num_heads=2, d_kv=8, dropout_rate=0.,
    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
model = T5ForConditionalGeneration(config)
source = torch.tensor([[3, 4, 1, 0], [5, 6, 7, 1]])
labels = torch.tensor([[8, 9, 1, -100], [10, 11, 12, 1]])
decoder_input = model.prepare_decoder_input_ids_from_labels(labels)
assert decoder_input.tolist() == [[0, 8, 9, 1], [0, 10, 11, 12]]
optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
output = model(input_ids=source, attention_mask=source.ne(0), labels=labels)
assert output.logits.shape == (2, 4, 16) and torch.isfinite(output.loss)
output.loss.backward()
assert model.shared.weight.grad is not None
optimizer.step()
expected = torch.softmax(torch.tensor([0., torch.log(torch.tensor(2.)).item()]), 0)
assert torch.allclose(expected, torch.tensor([1/3, 2/3]))
print("shifted targets, padding mask, seq2seq update and attention calculation passed")
```

The random tiny T5 tests architecture and loss plumbing, not translation quality.
A real checkpoint experiment must pin tokenizer/model revisions, language tags,
license, hardware, and train/development/test domains. Keep genuine source/target
pairs separate from back-translated data, filter and deduplicate before mixing,
and compare on an untouched real-domain test set.

**How is corpus BLEU aggregated?** Sum clipped matching counts and candidate counts
for each n-gram order over the corpus, then take precision ratios and the geometric
mean with corpus candidate/reference lengths. Do not average sentence BLEUs.
Zero matching counts require a stated smoothing/effective-order policy; an empty
candidate needs defined behavior. Use [sacreBLEU](https://github.com/mjpost/sacrebleu)
and record its signature for actual comparisons. A fluent output changing `15 mg`
to `50 mg` can retain most n-grams while failing a critical numeric check. Add
placeholder, entity, number and terminology assertions and human error categories
instead of relying on one metric.

1. What was the seq2seq bottleneck, and how did attention remove it?
2. Why does the transformer decoder need two attention mechanisms?
3. Explain back-translation and why the synthetic data goes on the source side.
4. Give three reasons BLEU is a poor metric, and name the current alternative.
5. What is the beam search curse, and what does it reveal about the objective?
6. What is the curse of multilinguality, and what mitigates it?
7. Why is terminology control hard, and what is the reliable solution?

## Where to go next

- [Text Generation & Decoding](./text-generation-and-decoding.md) — decoding
  strategies in depth.
- [NLP Evaluation](./nlp-evaluation.md) — BLEU, ROUGE, BERTScore, COMET, and
  LLM judges.
- [Language Models](./language-models.md) — the models doing the translating now.
