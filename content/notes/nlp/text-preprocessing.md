---
order: 1
description: Unicode normalization, the tokenization problem, BPE/WordPiece/SentencePiece derived with a worked example, stemming and lemmatization, and what modern pipelines should and should not do.
meta: NLP · foundations
---

# Text Preprocessing and Tokenization

Text is the messiest common data type. It has no fixed length, no numeric
representation, ambiguous boundaries, hundreds of writing systems, and multiple
byte sequences that render identically. Everything downstream depends on how you
turn it into integers, and tokenization decisions leak into model behaviour in
ways that surprise people years later.

## The pipeline, and how much of it survives

```mermaid
flowchart TD
    R["raw text"] --> U["Unicode normalisation<br/>NFC or NFKC"]
    U --> C["cleaning:<br/>strip markup, control chars,<br/>fix mojibake, dedupe"]
    C --> D{"classical or<br/>neural pipeline?"}
    D -->|"classical:<br/>TF-IDF, naive Bayes,<br/>linear models"| CL["lowercase, remove stopwords,<br/>strip punctuation,<br/>stem or lemmatise,<br/>then whitespace tokenise"]
    D -->|"neural:<br/>transformers"| NE["subword tokenise<br/>with the MODEL'S tokenizer<br/>and almost nothing else"]
    CL --> V["vocabulary and vectorisation"]
    NE --> V
```

The single most important thing to know: **for a pretrained transformer, most of
the classical preprocessing is wrong.** Lowercasing destroys the casing signal
the model was pretrained with. Removing stopwords destroys syntax. Stemming
produces strings absent from the tokenizer's vocabulary. Use the model's own
tokenizer and leave the text alone.

Classical preprocessing is still correct for classical models — TF-IDF plus a
linear classifier remains a strong, fast baseline — so the distinction is
between pipelines, not between old and new.

## Unicode, and why it bites

| Issue | Example | Handling |
|---|---|---|
| Multiple encodings of one glyph | "é" as U+00E9, or "e" + U+0301 | NFC normalisation |
| Compatibility variants | "ﬁ" ligature, full-width "Ａ" | NFKC (lossier) |
| Invisible characters | zero-width joiner, soft hyphen, BOM | strip explicitly |
| Homoglyphs | Cyrillic "а" vs Latin "a" | confusable detection; a real spam-evasion vector |
| Mojibake | "â€™" from UTF-8 read as Latin-1 | `ftfy`, or fix the ingestion |
| Emoji and skin-tone modifiers | multi-codepoint grapheme clusters | do not split graphemes |
| Right-to-left and bidi controls | Arabic, Hebrew, bidi override attacks | strip bidi control characters |

```python
import unicodedata, ftfy, re

def clean(text):
    text = ftfy.fix_text(text)                      # repair mojibake
    text = unicodedata.normalize("NFKC", text)      # canonical + compatibility
    text = "".join(ch for ch in text
                   if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return re.sub(r"[ \t]+", " ", text).strip()
```

**NFC or NFKC?** NFC composes canonical equivalents and is lossless for meaning.
NFKC additionally folds compatibility variants — it turns "½" into "1⁄2" and
full-width characters into ASCII, which normalises away real distinctions.
Use NFC by default; NFKC when you want aggressive normalisation and have checked
that the folds are acceptable for your language.

## Why tokenization is hard

**Whitespace splitting fails immediately.** "don't" → is that one token or two?
"New York" is one concept and two words. Chinese and Japanese have no spaces at
all. German compounds ("Donaudampfschiffahrtsgesellschaft") are single words
that encode a whole phrase. URLs, code, chemical formulae, and emoji all break
the assumption differently.

**Word-level vocabularies fail at scale.** A fixed vocabulary of 50k words has
three problems: an enormous embedding matrix, no way to represent any word
outside it (every unseen word becomes `[UNK]`, destroying information), and no
relationship between "run", "running", and "runner".

**Character-level solves coverage and fails at length.** No out-of-vocabulary
words ever, a tiny vocabulary — but sequences are 4–5× longer, attention is
quadratic in length, and the model must learn word structure from scratch.

**Subword tokenization is the compromise**: frequent words stay whole, rare
words decompose into meaningful pieces. `unhappiness` → `un` + `happi` + `ness`.

## Byte-pair encoding

Originally a compression algorithm, adapted to tokenization. Start from
characters (or bytes) and repeatedly merge the most frequent adjacent pair.

### Worked example

Corpus with word frequencies: `low` ×5, `lower` ×2, `newest` ×6, `widest` ×3.
Represent each word as characters with an end-of-word marker:

```
l o w </w>        x5
l o w e r </w>    x2
n e w e s t </w>  x6
w i d e s t </w>  x3
```

Count adjacent pairs across the corpus:

| Pair | Count |
|---|---|
| `e s` | 6 + 3 = **9** |
| `s t` | 6 + 3 = 9 |
| `l o` | 5 + 2 = 7 |
| `o w` | 5 + 2 = 7 |
| `t </w>` | 6 + 3 = 9 |

Merge `e s` → `es` (ties broken by first occurrence). Recount, and the next
merges follow:

| Step | Merge | Result |
|---|---|---|
| 1 | `e s` → `es` | `n e w es t </w>`, `w i d es t </w>` |
| 2 | `es t` → `est` | `n e w est </w>`, `w i d est </w>` |
| 3 | `est </w>` → `est</w>` | `n e w est</w>`, `w i d est</w>` |
| 4 | `l o` → `lo` | `lo w </w>`, `lo w e r </w>` |
| 5 | `lo w` → `low` | `low </w>`, `low e r </w>` |

The algorithm has **discovered the suffix `est`** and the stem `low` without any
linguistic input — purely from co-occurrence statistics. Continue until the
vocabulary reaches the target size. The learned merge list, applied in order, is
the tokenizer.

### Byte-level BPE

GPT-2's variant operates on **bytes** rather than Unicode characters. The base
vocabulary is exactly 256, and every possible byte sequence is representable, so
there is **no `[UNK]` token, ever** — any input in any script, plus binary
garbage, tokenises successfully. This is why modern LLMs handle arbitrary
Unicode gracefully.

The cost: non-Latin scripts are less efficient, because a character that is 3–4
UTF-8 bytes may become several tokens. Thai, Burmese, and many Indic languages
consume 3–5× more tokens per character than English, which means proportionally
higher API cost and less effective context — a real and under-discussed equity
issue in LLM access.

## The tokenizer families

| Algorithm | Merge criterion | Used by |
|---|---|---|
| **BPE** | most frequent adjacent pair | GPT-2/3/4, Llama, RoBERTa |
| **WordPiece** | pair maximising $\frac{P(xy)}{P(x)P(y)}$ — likelihood gain, not raw frequency | BERT, DistilBERT, ELECTRA |
| **Unigram LM** | start large, iteratively **remove** tokens whose deletion costs least likelihood | ALBERT, T5, XLNet, many multilingual models |
| **SentencePiece** | a framework wrapping BPE or Unigram | most multilingual models |

**WordPiece's criterion** differs from BPE's in a meaningful way: it merges the
pair whose combination most increases the corpus likelihood, which prefers pairs
that occur together more than chance predicts. In practice the vocabularies are
similar, and WordPiece marks continuations with `##` (`playing` → `play`,
`##ing`).

**Unigram is subtractive**, and its useful property is that it defines a
*probabilistic* segmentation — a word can be tokenised several ways with
different probabilities, which enables **subword regularisation**: sample a
different segmentation each epoch as data augmentation. This measurably helps
low-resource translation.

**SentencePiece** treats input as a raw stream including spaces, encoding them as
`▁`. That makes it fully reversible — detokenisation is exact string
concatenation — and language-agnostic, since it needs no pre-tokenizer and works
for Chinese and Japanese without modification.

## Practical tokenization facts

| Rule of thumb | Value |
|---|---|
| English tokens per word | ~1.3 |
| English characters per token | ~4 |
| Tokens per 1,000 English words | ~1,300 |
| Code tokens per line | ~10–15 |
| Non-Latin scripts | 2–5× more tokens per character |

**The arithmetic tokenization problem** is a good illustration of tokenizer
consequences. If `1234` tokenises as `12`+`34` and `5678` as `567`+`8`, digit
positions do not align, and the model must learn arithmetic over inconsistent
groupings. Llama 3 and several other models tokenise each digit separately for
exactly this reason, and it measurably improves arithmetic.

**Trailing whitespace is a real bug source.** A prompt ending in a space
tokenises differently from one that does not, because most tokenizers attach a
leading space to the following word (`" the"` is a different token from `"the"`).
The result is a prompt slightly off the model's training distribution, which
degrades output for no visible reason.

**Special tokens** must be handled deliberately: `[CLS]`, `[SEP]`, `[MASK]`,
BOS/EOS, padding, and chat control tokens. Use `apply_chat_template` rather than
hand-writing them — every instruct model has its own scheme, and getting it wrong
degrades quality substantially while looking fine.

```python
tok = AutoTokenizer.from_pretrained(model_name)     # ALWAYS from the same checkpoint

enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
tok.padding_side = "left"        # required for batched generation with a causal LM
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## Classical preprocessing

Still correct when you are building a TF-IDF or bag-of-words pipeline.

### Stopword removal

Remove high-frequency function words ("the", "is", "of").

| Helps | Hurts |
|---|---|
| Topic modelling | sentiment ("not good" → "good") |
| Keyword extraction | negation-sensitive tasks |
| Search indexing | phrase matching ("to be or not to be") |
| Reducing dimensionality | anything where syntax matters |

TF-IDF already down-weights common terms, so explicit stopword removal is often
redundant. **Never remove stopwords for a transformer.**

### Stemming vs lemmatization

| | Stemming | Lemmatization |
|---|---|---|
| Method | rule-based suffix chopping | dictionary + morphological analysis |
| Output | may not be a real word (`studies` → `studi`) | always a valid lemma (`studies` → `study`) |
| Needs POS | no | yes, for accuracy (`meeting` as noun vs verb) |
| Speed | very fast | slower |
| Algorithms | Porter, Snowball, Lancaster | WordNet, spaCy, Stanza |

Stemming is aggressive and cheap; lemmatization is accurate and slower. For
search, stemming is usually enough. For anything where the output is shown to a
human, lemmatize.

### Other classical steps

| Step | Note |
|---|---|
| Lowercasing | loses proper nouns and acronyms ("US" vs "us"); fine for topic tasks |
| Punctuation removal | loses sentence boundaries and emphasis |
| Number normalisation | replace with a `<NUM>` token, or spell out |
| Contraction expansion | "don't" → "do not" |
| Spelling correction | risky — it can destroy names and domain terms |
| Sentence splitting | harder than it looks: "Dr. Smith went to Washington." |
| Language identification | `fasttext` or `langdetect`; essential for multilingual corpora |
| Deduplication | **the highest-value step for pretraining corpora** — MinHash/LSH near-duplicate removal |

**Deduplication deserves emphasis.** Duplicated documents in a pretraining corpus
cause memorisation, inflate evaluation scores through test-set contamination, and
waste compute. Exact-match dedup catches little; MinHash-LSH near-duplicate
detection at document and paragraph level is standard practice for every serious
corpus.

## Handling the awkward cases

| Case | Approach |
|---|---|
| Very long documents | chunk with overlap; or a long-context model; or hierarchical encoding |
| Code | a code-aware tokenizer; preserve indentation and newlines |
| Tables in text | linearise with explicit separators, or a table-aware model |
| Mixed languages | a multilingual tokenizer; do not split by language |
| Noisy user text | character n-grams are robust to typos; do not over-correct |
| Domain jargon | check the tokenizer's fertility on your terms; consider vocabulary extension |
| PII | detect and redact **before** training; it will otherwise be memorised |

**Chunking for retrieval** is worth doing carefully. Fixed-size chunks split
sentences and lose context; the usual recipe is to respect structural boundaries
(paragraphs, sections, headings), use 200–500 tokens with 10–20% overlap, and
prepend the document title and section heading to each chunk so the embedding
carries context the chunk text alone does not.

## Common mistakes

| Mistake | Consequence |
|---|---|
| Lowercasing before a cased pretrained model | throws away a signal the model uses |
| Removing stopwords before a transformer | destroys syntax |
| Using a different tokenizer than the checkpoint | garbage embeddings, silently |
| Fitting a vectorizer on train+test | vocabulary leakage |
| Ignoring `max_length` truncation | silently dropping the end of every long document |
| Right padding for causal generation | the model generates from a pad position |
| Assuming 1 token = 1 word | context and cost estimates off by ~30% |
| Not stripping trailing whitespace from prompts | off-distribution tokenization |
| Skipping deduplication in a pretraining corpus | memorisation and contaminated evaluation |
| Normalising away emoji or casing in sentiment tasks | both carry signal |

## Self-check

1. Run three BPE merge steps on the corpus `low`×5, `lowest`×2, `newer`×6.
2. Why does byte-level BPE never produce an `[UNK]` token, and what does it cost?
3. How does WordPiece's merge criterion differ from BPE's, and what does it
   prefer?
4. Why does a trailing space in a prompt change model output?
5. Give three preprocessing steps that are correct for TF-IDF and wrong for BERT.
6. Why do some models tokenise digits individually?
7. What is the highest-value preprocessing step for a pretraining corpus, and
   why?

## Where to go next

- [Text Representation](./text-representation.md) — turning tokens into vectors.
- [Language Models](./language-models.md) — what the tokens are fed into.
- [RAG & Retrieval](./rag-and-retrieval.md) — where chunking decisions matter
  most.
