---
order: 8
description: The Hugging Face stack — transformers, tokenizers, datasets, accelerate, PEFT/LoRA, TRL, and how to fine-tune, quantize, and serve a pretrained model without reinventing the loop.
meta: Libraries · LLMs
---

# The Hugging Face Ecosystem

Nobody trains a language model from scratch to solve a business problem. The
default workflow is: find a pretrained checkpoint, adapt it, evaluate it, serve
it. Hugging Face is the toolchain that makes each of those four steps a few lines
rather than a few weeks, and it has become the de facto standard interface for
open models.

## The libraries and what each does

| Library | Job |
|---|---|
| `transformers` | model architectures + pretrained weights + tokenizers + `Trainer` |
| `tokenizers` | fast Rust BPE/WordPiece/Unigram implementations |
| `datasets` | memory-mapped, streamable dataset loading and processing |
| `accelerate` | device placement and distributed training without rewriting the loop |
| `peft` | LoRA, QLoRA, prefix tuning, IA³ — parameter-efficient fine-tuning |
| `trl` | SFT, reward modelling, PPO, DPO, GRPO for preference tuning |
| `evaluate` | metric implementations with a common interface |
| `safetensors` | a weight format that is fast to load and cannot execute code |
| `optimum` | export and acceleration — ONNX Runtime, OpenVINO, TensorRT, Neuron |
| `diffusers` | diffusion pipelines for image/audio/video generation |
| `huggingface_hub` | download, upload, versioning, and model cards |
| `text-generation-inference` | production LLM serving |

```mermaid
flowchart LR
    HUB["Hugging Face Hub<br/>weights, datasets, cards"] --> TOK["tokenizers<br/>text to token ids"]
    HUB --> MOD["transformers<br/>architecture + weights"]
    DS["datasets<br/>memory-mapped Arrow"] --> COL["collator<br/>pad and batch"]
    TOK --> COL
    COL --> TR["Trainer / accelerate<br/>the training loop"]
    MOD --> PEFT["peft<br/>freeze base, train adapters"]
    PEFT --> TR
    TR --> EVAL["evaluate<br/>metrics"]
    TR --> OUT["fine-tuned checkpoint"]
    OUT --> SERVE["optimum / TGI / vLLM<br/>serving"]
```

## Loading a model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "microsoft/deberta-v3-base"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(
    name, num_labels=3, id2label=id2label, label2id=label2id)
```

The `Auto*` classes read the checkpoint's `config.json` and instantiate the right
architecture, which is why the same three lines work for BERT, DeBERTa, RoBERTa,
or a model released next month.

**Pick the head that matches the task**, because it determines the output shape
and the loss:

| `AutoModelFor…` | Task | Output |
|---|---|---|
| `SequenceClassification` | sentiment, intent, NLI | logits over labels |
| `TokenClassification` | NER, POS | logits per token |
| `QuestionAnswering` | extractive QA | start/end logits |
| `CausalLM` | GPT-style generation | logits over the vocabulary |
| `MaskedLM` | BERT-style pretraining | logits at masked positions |
| `Seq2SeqLM` | translation, summarisation (T5, BART) | decoder logits |
| `MultipleChoice` | multiple-choice benchmarks | one logit per option |

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",             # shard across available GPUs, offload if needed
    attn_implementation="flash_attention_2",
)
```

`device_map="auto"` uses `accelerate` to place layers across GPUs, CPU, and disk
by available memory. Convenient for experimentation; for production, place
deliberately or use a serving engine.

**Prefer `safetensors`.** PyTorch `.bin` checkpoints are pickles and executing an
untrusted one runs arbitrary code. `safetensors` is a flat, zero-copy,
memory-mappable format that cannot execute anything, and it loads faster.

## Tokenizers

```python
enc = tok(
    texts, padding=True, truncation=True, max_length=512,
    return_tensors="pt", return_attention_mask=True,
)
enc.keys()      # input_ids, attention_mask, (token_type_ids for BERT-likes)
```

| Argument | Meaning |
|---|---|
| `padding` | `True`/`"longest"` pads to the longest in batch; `"max_length"` pads to `max_length` |
| `truncation` | cut sequences longer than `max_length` |
| `return_offsets_mapping` | character spans per token — required for NER alignment |
| `add_special_tokens` | `[CLS]`/`[SEP]`/BOS/EOS; on by default |
| `return_tensors` | `"pt"`, `"tf"`, `"np"`, or Python lists |

**Pad to the longest in the batch, not to `max_length`.** Padding everything to
512 when the median length is 40 wastes most of your compute on padding tokens.
With a `DataCollatorWithPadding` and length-grouped batching, throughput often
doubles.

**Left vs right padding matters for generation.** Decoder-only models generate
from the last position, so right padding puts pad tokens where the model should
be reading. Set `tok.padding_side = "left"` for batched generation with a causal
LM. This is a genuine silent-garbage bug.

**Alignment for token classification.** Subword tokenization splits words, so
word-level labels must be expanded and the continuation subwords masked:

```python
enc = tok(words, is_split_into_words=True, truncation=True)
labels = []
for i, word_ids in enumerate([enc.word_ids(i) for i in range(len(words))]):
    prev, seq = None, []
    for wid in word_ids:
        if wid is None:           seq.append(-100)      # special token
        elif wid != prev:         seq.append(tags[i][wid])
        else:                     seq.append(-100)      # continuation subword
        prev = wid
    labels.append(seq)
```

`-100` is the ignore index that PyTorch's cross-entropy skips — the standard
convention throughout the library.

**Chat templates** are the correct way to format instruction data. Every
instruct model has its own control tokens, and getting them wrong degrades
quality badly:

```python
messages = [{"role": "system", "content": "You are terse."},
            {"role": "user",   "content": "Explain LoRA in one sentence."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

Never hand-write `<|im_start|>` strings; let the template do it.

## Datasets

```python
from datasets import load_dataset

ds = load_dataset("imdb")                       # DatasetDict with train/test
ds = load_dataset("json", data_files={"train": "train.jsonl"})
stream = load_dataset("c4", "en", split="train", streaming=True)   # no download
```

`datasets` stores data as **Apache Arrow on disk, memory-mapped**, so a 500 GB
corpus uses almost no RAM and multiple processes share the same pages.

```python
def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=512)

ds = ds.map(preprocess, batched=True, batch_size=1000,
            num_proc=8, remove_columns=["text"])
ds = ds.filter(lambda ex: len(ex["input_ids"]) > 10)
ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

| Practice | Why |
|---|---|
| `batched=True` | calls the fast tokenizer once per 1,000 examples instead of per example — often 20× faster |
| `num_proc=` | parallel processing across cores |
| `remove_columns=` | drop raw text so the collator does not choke on strings |
| caching | `map` results are cached on disk by a hash of the function; change the function and it recomputes |
| `streaming=True` | iterate a dataset far larger than disk |

Deduplication before training matters more than most people expect: duplicated
documents in a pretraining corpus cause memorisation and inflate evaluation
scores. MinHash-LSH deduplication is standard practice.

## Fine-tuning with `Trainer`

```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

args = TrainingArguments(
    output_dir="out", num_train_epochs=3,
    per_device_train_batch_size=16, gradient_accumulation_steps=4,
    learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True, gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=200, save_steps=200,
    load_best_model_at_end=True, metric_for_best_model="f1",
    logging_steps=50, report_to="wandb", seed=42,
    group_by_length=True,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.push_to_hub()
```

Sensible starting points for full fine-tuning of an encoder: learning rate
$2\times10^{-5}$ to $5\times10^{-5}$, 2–4 epochs, warmup 6%, weight decay 0.01.
Rates that work for training from scratch ($10^{-3}$) will destroy pretrained
weights.

`group_by_length=True` batches similar-length sequences together, cutting padding
waste substantially.

For full control, `accelerate` wraps a hand-written loop instead:

```python
from accelerate import Accelerator
acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
model, opt, loader, sched = acc.prepare(model, opt, loader, sched)

for batch in loader:
    with acc.accumulate(model):
        loss = model(**batch).loss
        acc.backward(loss)
        opt.step(); sched.step(); opt.zero_grad()
```

The same script then runs on CPU, one GPU, multi-GPU DDP, DeepSpeed, or FSDP
depending on `accelerate config`, with no code changes.

## PEFT and LoRA

Full fine-tuning of a 7B model in bf16 with AdamW needs roughly 18 bytes per
parameter — about 126 GB before activations. LoRA makes it fit on one consumer
GPU.

**The idea**: freeze $W$ and learn a low-rank update.

$$W' = W + \frac{\alpha}{r}BA, \qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k)$$

$A$ is initialised randomly and $B$ to zero, so training starts exactly at the
pretrained model. Only $A$ and $B$ receive gradients — typically 0.1–1% of the
parameters — so optimiser state shrinks by the same factor. At inference the
product can be merged back into $W$, giving **zero added latency**.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()   # trainable: 0.24% || all params: 8.03B
```

| Parameter | Guidance |
|---|---|
| `r` | 8–16 for style/format adaptation; 32–64 for new knowledge or hard tasks |
| `lora_alpha` | commonly $2r$; the effective scale is $\alpha/r$ |
| `target_modules` | attention projections at minimum; **including the MLP projections consistently helps** |
| `lora_dropout` | 0.05–0.1 on small datasets |

**QLoRA** goes further: quantise the frozen base to 4-bit NF4 and train LoRA
adapters on top of it, with paged optimisers to survive memory spikes. A 70B
model becomes trainable on a single 48 GB GPU.

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, cfg)
```

Other PEFT methods worth knowing: **DoRA** (decomposes into magnitude and
direction, usually a small win over LoRA at the same rank), **IA³** (learned
rescaling vectors, even fewer parameters), **prefix/prompt tuning** (learned
virtual tokens, weakest but tiniest), and **adapters** (bottleneck layers,
add inference latency).

**Adapters are composable.** One base model can serve dozens of task-specific
LoRAs, swapped per request — the multi-LoRA serving pattern that vLLM and TGI
support natively, and a strong argument for LoRA over full fine-tuning in
multi-tenant products.

## Preference tuning with TRL

```python
from trl import SFTTrainer, DPOTrainer, SFTConfig, DPOConfig

sft = SFTTrainer(model="base", train_dataset=chat_ds, peft_config=cfg,
                 args=SFTConfig(max_length=2048, packing=True))
sft.train()

dpo = DPOTrainer(model=sft_model, ref_model=None,      # ref_model=None uses the frozen adapter base
                 train_dataset=pref_ds, args=DPOConfig(beta=0.1, learning_rate=5e-7))
dpo.train()
```

The standard alignment pipeline:

| Stage | Data | Objective |
|---|---|---|
| **Pretraining** | raw text | next-token prediction |
| **SFT** | (prompt, good response) pairs | supervised next-token on the response only |
| **Reward modelling** | (prompt, chosen, rejected) | Bradley–Terry ranking loss |
| **RLHF (PPO)** | prompts + reward model | maximise reward with a KL penalty toward the SFT model |
| **DPO** | (prompt, chosen, rejected) | closed-form equivalent of the above; **no reward model, no sampling** |
| **GRPO** | prompts + a verifiable reward | group-relative advantages; used for reasoning training |

DPO is the pragmatic default: it optimises the same KL-regularised objective as
RLHF but derives a closed-form loss over preference pairs, removing the reward
model and the sampling loop. Note the very small learning rate ($5\times10^{-7}$)
— preference tuning moves the model far more per step than SFT does, and
over-training it collapses diversity.

`packing=True` in SFT concatenates short examples into full-length sequences,
which can double throughput on chat data where most turns are short.

**Mask the prompt in SFT.** You want loss on the response tokens only; computing
it over the prompt teaches the model to generate user turns.

## Generation

```python
out = model.generate(
    **enc, max_new_tokens=256,
    do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
    repetition_penalty=1.05, no_repeat_ngram_size=0,
    eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

| Strategy | Setting | Use for |
|---|---|---|
| Greedy | `do_sample=False` | deterministic, factual, extraction |
| Beam search | `num_beams=4` | translation, summarisation; degenerate for open-ended text |
| Temperature sampling | `temperature` | the main creativity dial |
| Top-k | `top_k=50` | truncate to the k most likely |
| Top-p (nucleus) | `top_p=0.9` | truncate to the smallest set covering p mass — adapts to the distribution |
| Min-p | `min_p=0.05` | threshold relative to the top token; robust at high temperature |
| Contrastive search | `penalty_alpha=0.6, top_k=4` | fluent and non-repetitive without sampling |

Slicing off the prompt (`out[0][input_len:]`) is necessary because `generate`
returns prompt plus continuation for causal models.

For anything serving-shaped, use **vLLM, SGLang, or TGI** rather than
`generate` in a loop: continuous batching and paged KV cache give an order of
magnitude more throughput.

## Evaluation

```python
import evaluate
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metric.compute(predictions=logits.argmax(-1), references=labels,
                          average="macro")
```

For generative models, the harnesses that matter are `lm-evaluation-harness`
(standard academic benchmarks, comparable numbers), `lighteval`, and for
instruction quality, LLM-as-judge protocols. Be aware that benchmark
contamination is pervasive — a model trained on web data has very likely seen
the test sets — so a held-out, private evaluation on your own data is worth more
than any leaderboard position.

## Deployment

| Path | Use for |
|---|---|
| `optimum` → ONNX Runtime | CPU inference, cross-platform, encoder models |
| `optimum` → TensorRT / OpenVINO | maximum GPU / Intel CPU throughput |
| `text-generation-inference` | Hugging Face's production LLM server |
| vLLM / SGLang | highest-throughput open LLM serving |
| `pipeline()` | prototypes only — no batching, no caching |
| GGUF + `llama.cpp` | CPU and consumer-GPU local inference |
| Inference Endpoints | managed hosting |

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
m = ORTModelForSequenceClassification.from_pretrained(name, export=True)
```

For encoder models on CPU, ONNX Runtime with int8 dynamic quantisation commonly
gives 2–4× lower latency at negligible accuracy cost — the single best-value
optimisation in the whole list.

## Practical cautions

| Issue | Detail |
|---|---|
| Licences differ per model | Apache-2.0, Llama community licence, research-only — check before shipping |
| Pickled weights execute code | prefer `safetensors`; do not load untrusted `.bin` |
| Model cards can be aspirational | evaluate on your own data before believing benchmark claims |
| Silent tokenizer mismatch | always load the tokenizer from the same checkpoint as the weights |
| `pipeline()` is not production | no batching, no continuous batching, no KV-cache management |
| Downloads are cached in `~/.cache/huggingface` | set `HF_HOME` on shared machines; it grows to hundreds of GB |
| Version churn | pin `transformers`; APIs and defaults move quickly |
| Benchmark contamination | assume public test sets are in the training data |
| Gated repos | need `huggingface-cli login` and accepted terms |

## Self-check

1. Why must padding be left-side for batched generation with a causal LM?
2. What does `-100` mean in a labels tensor, and where does it come from?
3. Write the LoRA update and say why $B$ is initialised to zero.
4. Why does DPO not need a reward model? What objective is it equivalent to?
5. Your fine-tuned model outputs garbage after a few hundred steps at
   `lr=1e-3`. Diagnose it.
6. What does `batched=True` change in `datasets.map`, and roughly how much?
7. Give two reasons to prefer `safetensors` over a `.bin` checkpoint.

## Where to go next

- [PyTorch](./pytorch.md) — the tensors and autograd underneath.
- [MLOps & Serving](./mlops-and-serving.md) — getting the result into production.
- [Transformers Deep Dive](/courses/transformers/) — what the architecture is
  actually doing.
- [The Inference Engineering Book](/courses/inference/) — how serving engines
  make generation fast.
