# 10 — Scaling Out: Distributed Fine-Tuning, Parallelism, and MoE

> **Prerequisites:** modules 03, 06.
> **You will learn:** what DDP, ZeRO, FSDP2, tensor, context, pipeline and expert parallelism
> each shard, how to configure them through Accelerate, how to debug multi-GPU
> runs, and what changes when you fine-tune a mixture-of-experts model.

A full fine-tune past about 8B parameters, an RL run with large rollouts, or a 128k-context
SFT job does not fit on one GPU. You then need to know what each parallelism scheme splits,
how to switch it on through Accelerate or DeepSpeed, and what its failures look like. The
transformers course's
[memory section](../transformers/13-training-considerations.md#134-the-memory-budget)
lists these techniques in one line each. This module derives them.

Two facts from [module 03](./03-training-mechanics-and-memory-math.md) carry everything
below: mixed-precision AdamW costs about 16 bytes per parameter (128 GB of model state for
8B), and activations scale with $B \times T \times d \times L$. Every technique here spreads
one of those terms across GPUs and pays in communication.

---

## 10.1 Data parallelism

### DDP: replicate everything, average the gradients

Distributed Data Parallel (DDP) is the baseline. Every rank (one process per GPU) holds a
**full replica** of the model and optimizer state, runs forward and backward on its slice of
the batch, then **all-reduces** gradients so every replica takes the same averaged step.
With a mean loss over a batch split into $N$ equal shards, the full-batch gradient is

$$
\nabla \frac{1}{NB}\sum_{i=1}^{NB} \ell_i = \frac{1}{N}\sum_{r=1}^{N} \nabla \underbrace{\frac{1}{B}\sum_{i \in \text{shard } r} \ell_i}_{\text{rank } r\text{'s mean loss}}
$$

so the average of per-rank mean-loss gradients **is** the full-batch gradient. The identity
breaks when shards hold different numbers of loss tokens (padding, masking, packing): a
per-rank mean over-weights short shards. Normalize by the global loss-token count instead,
the same bug class as the accumulation error in
[module 03](./03-training-mechanics-and-memory-math.md); the HF Trainer switch is
`average_tokens_across_devices`.

### What the all-reduce costs

A **ring all-reduce** is a reduce-scatter (each rank owns the sum of one $1/N$ chunk) then
an all-gather. Each phase moves $\frac{N-1}{N}\Psi$ elements per rank, so per step:

$$
V_{\text{all-reduce}} = 2\,\frac{N-1}{N}\,\Psi \text{ elements.}
$$

**The volume per rank is almost independent of $N$.** For an 8B model with bf16 gradients
on 8 GPUs that is $2 \times \tfrac{7}{8} \times 16$ GB $= 28$ GB per rank per step. On
NVLink this is a fraction of a second; across a PCIe-only box it is not (section 10.5).

DDP hides most of this cost by launching each 25 MB gradient **bucket**'s all-reduce as soon
as backward fills it. The [PyTorch note](../../notes/libraries/pytorch.md) covers the DDP
API; the [optimization note](../../notes/deep-learning/optimization-and-training.md) covers
batch size and learning rate.

### `nn.DataParallel` is gone; effective batch

`nn.DataParallel` is the older single-process API: one GIL-bound Python process drives all
GPUs and scatters and gathers through GPU 0. [TRL v1.15.0](https://github.com/huggingface/trl/releases/tag/v1.15.0)
now **refuses** it outright, so launch with `accelerate launch` or `torchrun` to get DDP
(or FSDP2/ZeRO, configured below). Effective batch is micro-batch × devices ×
accumulation steps ([module 08](./08-frameworks-and-tooling.md)), so moving from 1 GPU to 8
means dividing `gradient_accumulation_steps` by 8 or retuning the learning rate.

---

## 10.2 Sharding the state: ZeRO

DDP stores the same 16 bytes per parameter on every rank. Rajbhandari et al. (2020),
[ZeRO](https://huggingface.co/papers/1910.02054), observed that each rank needs optimizer
state only for the parameters it updates, and full weights only for the current layer.

### The three stages

Let $\Psi$ be the parameter count, $N_d$ the data-parallel degree, and $K$ the optimizer
bytes per parameter ($K = 12$ for mixed-precision Adam: fp32 master weights, momentum,
variance). With 2-byte weights and 2-byte gradients:

| Stage | Shards | Per-GPU model-state memory |
|---|---|---|
| Baseline (DDP) | nothing | $(2 + 2 + K)\,\Psi$ |
| ZeRO-1, $P_{os}$ | optimizer states | $4\Psi + K\Psi / N_d$ |
| ZeRO-2, $P_{os+g}$ | + gradients | $2\Psi + (2 + K)\Psi / N_d$ |
| ZeRO-3, $P_{os+g+p}$ | + parameters | $(2 + 2 + K)\Psi / N_d$ |

Under ZeRO-3 no rank holds a full layer at rest. Before a layer's forward the ranks
**all-gather** its shards, compute, and free the copy; backward re-gathers, computes
gradients, and **reduce-scatters** them so each rank keeps the slice it owns and runs Adam
on its $1/N_d$ of the parameters.

The paper's worked example is a 7.5B model on 64 GPUs. Plugging in:

| | Baseline | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| GB per GPU | 120.0 | 31.4 | 16.6 | 1.9 |

The next check reproduces these four numbers exactly (120.0, 31.40625, 16.640625 and 1.875
GB), and confirms that stage 3 stores no redundant byte: per-GPU memory times $N_d$ equals
the baseline.

```python finetuning-check
# ZeRO per-GPU model-state memory (Rajbhandari et al. 2020), K = 12.
def zero_bytes(psi, n_d, stage, k=12):
    if stage == 0:
        return (2 + 2 + k) * psi
    if stage == 1:
        return 4 * psi + k * psi / n_d
    if stage == 2:
        return 2 * psi + (2 + k) * psi / n_d
    if stage == 3:
        return (2 + 2 + k) * psi / n_d
    raise ValueError(stage)

GB = 1e9
psi, n_d = 7.5e9, 64
got = [zero_bytes(psi, n_d, s) / GB for s in range(4)]
for g, w in zip(got, [120.0, 31.40625, 16.640625, 1.875]):
    assert abs(g - w) < 1e-9, (g, w)

# Every stage strictly lowers memory for any data-parallel degree >= 2.
for n in (2, 3, 4, 8, 16, 64, 512):
    for p in (1e8, 7.5e9, 7e10):
        vals = [zero_bytes(p, n, s) for s in range(4)]
        assert all(a > b for a, b in zip(vals, vals[1:])), (n, p, vals)

# Stage 3 keeps exactly one copy of everything across the group.
assert abs(zero_bytes(psi, n_d, 3) * n_d - zero_bytes(psi, n_d, 0)) < 1.0
# With N_d = 1 every stage collapses to the baseline: sharding needs partners.
assert len({zero_bytes(psi, 1, s) for s in range(4)}) == 1
print("ZeRO memory formulas verified")
```

The check covers model states only, not activations, gather buffers or fragmentation,
which often add several GB per GPU.

### What stage 3 costs in communication

ZeRO-1 and ZeRO-2 move **the same volume as DDP**: they reorder the all-reduce into
reduce-scatter gradients, step locally, all-gather parameters. ZeRO-3 gathers parameters
once for forward and again for backward: three passes of $\frac{N-1}{N}\Psi$ instead of two,
so **ZeRO-3 costs about 1.5× DDP's communication**. That trade is good on NVLink and
questionable over PCIe.

### Offload and the ZeRO family

| Variant | Paper | What moves | When it helps |
|---|---|---|---|
| ZeRO-Offload | [Ren et al. 2021](https://arxiv.org/abs/2101.06840) | optimizer state and the Adam step to CPU (with ZeRO-2) | one or few GPUs, lots of host RAM |
| ZeRO-Infinity | [Rajbhandari et al. 2021](https://arxiv.org/abs/2104.07857) | parameters, gradients, optimizer state to CPU **or NVMe** (ZeRO-3) | models larger than aggregate GPU memory |
| ZeRO++ | [Wang et al. 2023](https://arxiv.org/abs/2306.10209) | quantized weight all-gather, hierarchical (intra-node) weight partition, quantized gradient reduce | multi-node runs on slow inter-node links |

### DeepSpeed configuration

Keys checked against the [DeepSpeed config reference](https://www.deepspeed.ai/docs/config-json/)
and the [Transformers v5.19 DeepSpeed guide](https://huggingface.co/docs/transformers/v5.19.0/en/deepspeed)
(October 2026). `"auto"` copies the value from your `TrainingArguments`/`SFTConfig`.

```json
{
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "none" }
  },
  "gradient_clipping": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto"
}
```

`overlap_comm` overlaps reduction with backward; `contiguous_gradients` avoids
fragmentation; `offload_param` is valid only at stage 3. One trap from the Transformers
guide: **under ZeRO-3, build the `TrainingArguments` before loading the model**, or every
rank materializes the full model first.

### Saving a ZeRO-3 checkpoint

Under ZeRO-3 a rank's `state_dict()` holds placeholders, so a DeepSpeed checkpoint is a
directory of per-rank shards that only DeepSpeed can resume. Two ways out:

1. **Gather at save time.** `stage3_gather_16bit_weights_on_model_save: true` makes
   `save_model()` write a normal HF checkpoint, but rank 0 must hold the whole model: the
   classic **OOM at gather** for 70B-class models.
2. **Consolidate offline.**
   [`zero_to_fp32.py`](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/utils/zero_to_fp32.py#L8-L16)
   is copied into every checkpoint directory and consolidates the shards offline:

```bash
cd runs/ft10/checkpoint-500
python zero_to_fp32.py . merged/ --safe_serialization   # fp32 safetensors, sharded at 5GB
```

To resume on a different number of GPUs, convert the shards with DeepSpeed's
[Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing/)
([`ds_to_universal.py`](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/checkpoint/ds_to_universal.py)),
then resume with `"checkpoint": {"load_universal": true}` in the DeepSpeed JSON:

```bash
python -m deepspeed.checkpoint.ds_to_universal \
  --input_folder runs/ft10/checkpoint-500/global_step500 \
  --output_folder runs/ft10/checkpoint-500/global_step500_universal
```

---

## 10.3 FSDP: from FSDP1 to FSDP2

PyTorch's Fully Sharded Data Parallel ([Zhao et al. 2023](https://arxiv.org/abs/2304.11277))
implements ZeRO-3 natively in PyTorch.

**FSDP and ZeRO-3 are not rival algorithms.** They shard the same three things with the
same collectives and differ in implementation, configuration, checkpoint format, and
ecosystem (offload engines and ZeRO++ on one side, DTensor composition with TP and
`torch.compile` on the other). The
[Accelerate FSDP guide](https://huggingface.co/docs/accelerate/v1.15.0/en/usage_guides/fsdp)
spells out the mapping:

| ZeRO | FSDP1 `sharding_strategy` | FSDP2 / Accelerate | Shards |
|---|---|---|---|
| Stage 0 (DDP) | `NO_SHARD` | plain DDP | nothing |
| Stage 2 | `SHARD_GRAD_OP` | `fsdp_reshard_after_forward: false` | optimizer state, gradients (params kept gathered between forward and backward) |
| Stage 3 | `FULL_SHARD` | `fsdp_reshard_after_forward: true` | optimizer state, gradients, parameters |
| no exact ZeRO stage (closest: ZeRO++ hpZ, which replicates only weights per node) | `HYBRID_SHARD` | HSDP: 2-D mesh, `dp_shard_size` + `dp_replicate_size` | full sharding within a group, replication across groups |

### FSDP1: the `FlatParameter`

FSDP1 wraps a module (say, one decoder layer) and **flattens its parameters into one 1-D
`FlatParameter`** chunked across ranks. A flat buffer has one dtype and one
`requires_grad`, so a unit that mixes frozen bf16 base weights with trainable LoRA
matrices, or bf16 weights with fp8 or packed 4-bit ones, needs workarounds: either
`use_orig_params=True` (mixed `requires_grad` allowed, at a memory cost) or PEFT's recipe of
`use_orig_params=False` with a wrap policy that puts LoRA layers in their own units, plus
uniform storage dtypes.

### FSDP2: per-parameter `DTensor` sharding

FSDP2 (`torch.distributed.fsdp.fully_shard`) drops the flat buffer: each parameter becomes a
**`DTensor`** chunked "on dim-0", which "relaxes constraints around frozen parameters"
([PyTorch 2.8 docs](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)). The
[Accelerate FSDP1-vs-FSDP2 guide](https://huggingface.co/docs/accelerate/v1.15.0/en/concept_guides/fsdp1_vs_fsdp2)
lists what follows:

- **Partial freezing just works**: `requires_grad` lives on each parameter, so LoRA needs no
  special wrap policy, and mixed dtypes (bf16 next to fp8 or quantized storage) coexist.
- **Sharded checkpoints are cheap**: each rank saves its own shard, with async saves.
- **Deterministic memory**: no `recordStream`, so fewer surprise spikes.

### Configuring FSDP2 in Accelerate

As of October 2026, Accelerate still defaults to `fsdp_version: 1` so old configs do not
break; set 2 explicitly. Teaching fragment: an Accelerate config, not executed.

```yaml
distributed_type: FSDP
mixed_precision: bf16
num_processes: 8
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: true       # replaces fsdp_sharding_strategy (true ~ FULL_SHARD)
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true   # rank 0 reads the checkpoint, others start empty
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_offload_params: false
  fsdp_activation_checkpointing: true
```

Three FSDP1 flags are **gone**: `fsdp_use_orig_params` (FSDP2 always uses the original
parameters), `fsdp_sync_module_states` (redundant), and `fsdp_backward_prefetch` (FSDP2
always prefetches like `BACKWARD_PRE`). Convert an inherited FSDP1 YAML rather than editing
by hand:

```bash
accelerate to-fsdp2 --config_file fsdp1.yaml --output_file fsdp2.yaml
```

### FULL versus SHARDED state dicts

| | `FULL_STATE_DICT` | `SHARDED_STATE_DICT` |
|---|---|---|
| What is written | one full model, gathered to rank 0 | one shard per rank plus metadata (`.distcp` files) |
| Save cost | all-gather plus a memory spike on rank 0; OOM risk for big models | no extra communication; fast |
| Ship to inference | directly loadable with `from_pretrained` | needs `accelerate merge-weights ckpt/pytorch_model_fsdp_0 out/` first |

Train with `SHARDED_STATE_DICT` and merge once at the end, offline, on a high-RAM host.
`SHARDED_STATE_DICT` is written by `torch.distributed.checkpoint` (DCP), which **reshards at
load time**: the [DCP docs](https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html)
cover "saving in one cluster topology and loading into another", so a 16-GPU run resumes on 8.

### HSDP: shard inside, replicate across

Full sharding across 8 nodes sends every layer's all-gather over the slow inter-node
network. **Hybrid sharded data parallel (HSDP)** shards within a group (usually one NVLink
node) and replicates across groups (a 2-D device mesh in FSDP2), so only the replica
gradient all-reduce crosses nodes. The cost is memory: the ZeRO-3 formula with $N_d$ replaced
by the group size. For 8B on 16 GPUs, full sharding needs 8 GB of model state per GPU and
HSDP with 8-GPU groups 16 GB; `zero_memory_planner.py` (section 10.8) prints the table.

### FSDP with LoRA and QLoRA

**LoRA under FSDP2 needs nothing special.** Frozen base weights are sharded `DTensor`s
without gradients or optimizer state, so memory per GPU is roughly
$2\Psi_{\text{base}}/N_d$ plus 16 bytes per trainable parameter over $N_d$. The
[PEFT FSDP guide](https://huggingface.co/docs/peft/main/en/accelerate/fsdp) still shows the
FSDP1 recipe above (October 2026) and notes that under FSDP2 `set_adapter()` and
`disable_adapter()` must reshard so `requires_grad` changes reach the shards.

**FSDP + QLoRA** adds one hard constraint. FSDP shards floating-point tensors, and
bitsandbytes normally packs 4-bit weights into `uint8`. Answer.AI's
[FSDP-QLoRA work](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) (March 2024, with
Tim Dettmers and Hugging Face) fixed this by letting the packed storage use a float dtype.
The [PEFT FSDP guide](https://huggingface.co/docs/peft/main/en/accelerate/fsdp) sets it as
`bnb_4bit_quant_storage=torch.bfloat16`, which packs four 4-bit values into each bf16 element. The
**storage dtype must be the same on every rank and match the dtype of the rest of the
model** (`dtype=torch.bfloat16` in `from_pretrained`), or FSDP cannot build uniform shards.
With CPU offload and gradient checkpointing they fine-tuned a **70B model on two 24 GB
GPUs**; the PEFT guide reports 19.6 GB per GPU (about 107 GB of CPU RAM) for Llama-2-70B,
rising to 35.6 GB without offload.

The lab's `--variant qlora` path sets both; [module 11](./11-quantization-qat-and-fp8.md)
owns NF4 itself.

### Simulating the collectives

The check below runs the collectives as plain tensor algebra in one process: four "ranks"
of two examples each average to the full-batch gradient, and reduce-scatter of four padded
rank gradients equals the shard of their sum.

```python finetuning-check
import torch
import torch.nn.functional as F

torch.manual_seed(10)
world, per_rank = 4, 2
x = torch.randn(world * per_rank, 5, dtype=torch.float64)   # f64: per-rank vs full-batch sum order
y = torch.randint(0, 3, (world * per_rank,))
w = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)

# Full-batch mean-loss gradient in one process.
F.cross_entropy(x @ w.T, y).backward()
full_grad = w.grad.clone()

# DDP: each rank's mean-loss gradient, then all-reduce(sum) / world.
rank_grads = []
for r in range(world):
    w.grad = None
    sl = slice(r * per_rank, (r + 1) * per_rank)
    F.cross_entropy(x[sl] @ w.T, y[sl]).backward()
    rank_grads.append(w.grad.clone())
ddp_grad = torch.stack(rank_grads).sum(0) / world
torch.testing.assert_close(ddp_grad, full_grad, atol=1e-12, rtol=0)

# Reduce-scatter: rank r keeps shard r of the SUM of all ranks' (padded) gradients.
pad = (-10) % world                                 # 10 elements padded to 12
g = [torch.randn(10) for _ in range(world)]
padded = [torch.cat([gi, gi.new_zeros(pad)]) for gi in g]
summed = torch.stack(padded).sum(0)
for r in range(world):
    mine = torch.stack([pg.chunk(world)[r] for pg in padded]).sum(0)
    torch.testing.assert_close(mine, summed.chunk(world)[r])
print("DDP averaging and reduce-scatter verified")
```

It does not exercise real process groups, bucketing, overlap, or bf16 reduction order;
the lab's `--smoke` mode repeats the DDP equality with two real `gloo` processes.

---

## 10.4 Model parallelism: tensor, sequence/context, pipeline, expert

Data parallelism still runs **every layer's full math on every GPU**. When one layer's
activations or one sequence no longer fits, you split the model itself.

### Tensor parallelism

Megatron-LM ([Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)) splits each MLP into a
**column-parallel** first linear and a **row-parallel** second linear: with TP degree $t$
each GPU holds $1/t$ of the columns of $W_{\text{up}}$ and the matching rows of
$W_{\text{down}}$, and one **all-reduce** sums the partial outputs. Attention splits by heads.
A layer costs two all-reduces of a $(B, T, d)$ activation in forward and two in backward
([inference course diagram](/courses/inference/05-parallelism/05-01-tensor-parallelism.html)),
so **TP belongs inside a node, on NVLink**; the
[Accelerate N-D guide](https://huggingface.co/blog/accelerate-nd-parallel) calls it "not
recommended for PCIe linked GPUs". TP also cuts activation memory, which sharded DP cannot.

Transformers moved TP to a **DTensor-native backend in
[v5.16](https://github.com/huggingface/transformers/releases/tag/v5.16.0)** (breaking for the
legacy TP API). As of the
[v5.19 docs](https://huggingface.co/docs/transformers/v5.19.0/en/tensor_parallelism), you
pass a `DistributedConfig` (models must define `base_model_tp_plan`); `tp_plan="auto"` now
goes inside it, and passing `tp_plan` directly to `from_pretrained()` is deprecated.
Teaching fragment, not executed:

```python
import torch
from transformers import AutoModelForCausalLM, DistributedConfig
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4, tp_plan="auto"),
)
# launch: torchrun --nproc-per-node 4 train_tp.py
```

LoRA works on a TP base model: PEFT shards the adapter to match each target's TP layout. Per
the [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
(October 2026) this needs **PEFT ≥ 0.21.1 with Transformers ≥ 5.17**, and `save_pretrained`
gathers the adapter so the checkpoint is TP-degree independent.

### Sequence and context parallelism

For 64k to 1M-token sequences, shard the **sequence** dimension: Ulysses
([Jacobs et al. 2023](https://arxiv.org/abs/2309.14509)) uses all-to-alls around attention,
Ring Attention ([Liu et al. 2023](https://arxiv.org/abs/2310.01889)) passes KV blocks around a
ring. [Module 17](./17-long-context-extension.md) owns both.

### Pipeline parallelism

Pipeline parallelism (GPipe, [Huang et al. 2019](https://arxiv.org/abs/1811.06965)) puts
contiguous layer blocks ("stages") on different GPUs and streams micro-batches through them,
communicating only point-to-point at stage boundaries. With $p$ stages and $m$ micro-batches
the idle **bubble** is $\frac{p-1}{m+p-1}$: $3/19 \approx 15.8\%$ for $p = 4$, $m = 16$. It is
rare in HF fine-tuning stacks because it needs the model rewritten as stages (inference
course: [pipeline chapter](/courses/inference/05-parallelism/05-02-pipeline-parallelism.html)).

### Expert parallelism

**Expert parallelism (EP)** places MoE experts on different ranks; each MoE layer does an
**all-to-all dispatch** of tokens to their experts' ranks, local compute, and an
**all-to-all combine**, so traffic scales with tokens times $k$, not parameter count.
Transformers exposes it as `DistributedConfig(ep_size=..., fsdp_size=...)`
([EP guide](https://huggingface.co/docs/transformers/v5.19.0/en/expert_parallelism)); the
inference course covers
[serving with EP](/courses/inference/05-parallelism/05-03-data-and-expert-parallelism.html).

### Comparison

| Parallelism | What is split | Communication per step | When to use | HF entry point |
|---|---|---|---|---|
| DDP | data (batch) | all-reduce grads, $2\frac{N-1}{N}\Psi$ | model + optimizer fit on one GPU | `accelerate launch` (multi-GPU) |
| ZeRO-1/2 | data; optimizer state (+ grads) | same as DDP | weights fit, optimizer does not | DeepSpeed config `stage: 1/2` |
| ZeRO-3 / FSDP2 | data; params, grads, optimizer | ~1.5× DDP (2 all-gathers + reduce-scatter) | full fine-tune beyond one GPU | FSDP2 YAML or `stage: 3` |
| HSDP | as FSDP inside group, replicate across | intra-group gathers, inter-group all-reduce | multi-node | `ParallelismConfig(dp_shard_size, dp_replicate_size)` |
| Tensor | weight matrices (heads, MLP columns/rows) | 2 activation all-reduces per layer, fwd and bwd | layer activations or weights too big; inside NVLink | `DistributedConfig(tp_size)`, `ParallelismConfig(tp_size)` |
| Context / sequence | sequence dimension | all-to-all (Ulysses) or ring KV passes | 64k+ token sequences | `ParallelismConfig(cp_size)`; module 17 |
| Pipeline | layers into stages | point-to-point activations; bubble $\frac{p-1}{m+p-1}$ | very large models, slow links | Megatron-style frameworks |
| Expert | MoE experts | 2 all-to-alls per MoE layer | large MoE models | `DistributedConfig(ep_size)`; Axolotl EP |

### N-D parallelism

Accelerate's `ParallelismConfig`
([N-D parallel blog](https://huggingface.co/blog/accelerate-nd-parallel), August 2025)
composes `dp_replicate_size`, `dp_shard_size` (FSDP), `tp_size` and `cp_size`; 1 disables an
axis and the product must equal the world size. Teaching fragment, not executed:

```python
from accelerate import Accelerator, ParallelismConfig
pc = ParallelismConfig(dp_replicate_size=2, dp_shard_size=4, tp_size=2, cp_size=1)  # 16 GPUs
accelerator = Accelerator(parallelism_config=pc)
```

[Axolotl](./08-frameworks-and-tooling.md) exposes the same axes in YAML plus
`expert_parallel_size`. The rule of thumb: **TP inside a node (NVLink), FSDP across the
node's remaining GPUs, replicate across nodes**, CP only when sequence length forces it. In
the fragment's two nodes × four shards × TP pairs, activation all-reduces stay inside NVLink
pairs, parameter all-gathers inside a node, and only the replica gradient all-reduce crosses
the network.

---

## 10.5 Communication versus compute

An 8B model in bf16 is 16 GB of weights. On 8 GPUs, ZeRO-3/FSDP moves three passes of
$\frac{7}{8} \times 16$ GB per rank per micro-step (two all-gathers, one reduce-scatter):
**42 GB per rank**.

| Link | Bandwidth | Time for 42 GB (ideal) |
|---|---|---|
| NVLink, H100 SXM ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/)) | 900 GB/s total, ~450 GB/s each direction | 0.09 s |
| PCIe Gen5 x16 (same page) | 128 GB/s bidirectional, ~64 GB/s each direction | 0.66 s |
| InfiniBand NDR (ConnectX-7) | 400 Gb/s per port = 50 GB/s | 0.84 s |

Now the compute: about $6N$ FLOPs per token. NVIDIA's 1,979 bf16 TFLOPS for H100 SXM is
*with sparsity*, so about 989 dense; at 40% utilization, about 400 TFLOPS. One 2,048-token
sequence is $6 \times 8\times10^9 \times 2048 \approx 9.8\times10^{13}$ FLOPs, or **0.25 s**.

On NVLink the traffic (0.09 s) hides behind compute even at one sequence per micro-step. On
PCIe it is more than double the compute: **the run is communication-bound**, and overlap
cannot hide traffic longer than the work it overlaps with.

What helps is **more tokens per micro-step** (bigger micro-batches, packing): parameter
traffic is fixed per micro-step while compute grows with tokens. Gradient accumulation does
**not** amortize ZeRO-3's all-gathers, since every micro-batch re-gathers every layer. On PCIe,
prefer ZeRO-2 if the weights fit, or LoRA. The inference course's
[collectives chapter](/courses/inference/05-parallelism/05-04-collectives-and-topology.html)
covers ring versus tree algorithms and topology in more depth.

---

## 10.6 Fine-tuning MoE models

### Memory follows total parameters

A common misconception: "an MoE with 3B active parameters fine-tunes like a 3B model." It
computes like one but does not **store** like one: any token can go to any expert, so every
expert must be resident somewhere every step. **Weight memory scales with total parameters;
FLOPs scale with active parameters.**

Take [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/d47d535f78ec44bd57128f8e8aeba17eeb0285ea/config.json)
(model card: 30.5B total, 3.3B active; the config values are in the check below). Counting
gives 30,532,122,624 total and 3,353,032,704 active parameters. In bf16:

| | Parameters | bf16 weights |
|---|---|---|
| Qwen3-30B-A3B, total | 30.53B | 61.1 GB |
| Qwen3-30B-A3B, active per token | 3.35B | 6.7 GB |
| A 3.3B dense model | 3.3B | 6.6 GB |

The MoE needs **nine times** the weight memory of a dense model with the same per-token
compute. A full fine-tune is about 490 GB of model state, a multi-node FSDP2 or EP job. Bf16
LoRA needs about 61 GB of frozen weights, matching Unsloth's reported 63 GB
([Unsloth MoE docs](https://unsloth.ai/docs/basics/faster-moe)); a 24 GB card means QLoRA or
sharding.

### Expert LoRA and its parameter cost

Transformers MoE experts are often one 3-D `nn.Parameter`, which `target_modules` cannot
reach; PEFT's `target_parameters` can
([PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora);
[module 06](./06-lora-and-qlora.md)). With a rank-$r$ adapter on each expert's gate, up and
down projections the cost multiplies by $E$:

$$
\text{expert LoRA params} = L \times E \times 3 \times r \times (d + d_{ff}).
$$

For Qwen3-30B-A3B at $r = 16$ that is $48 \times 128 \times 3 \times 16 \times 2816 =$
830,472,192 trainable parameters, over ten times all-linear LoRA on an 8B dense model. PEFT
suggests a per-expert rank of `r // num_experts` (floored at 1) via `rank_pattern`: 51,904,512
here. Fused `gate_up_proj` layouts shift exact counts. PEFT materializes every expert's LoRA
delta, so **merge before serving**.

The check below recomputes the Qwen3-30B-A3B totals and bf16 sizes, confirms the expert-LoRA
formula against real tensors on a toy MoE ($E = 4$, $d = 8$, $d_{ff} = 16$, $r = 2$), and
asserts the pipeline bubble of 3/19.

```python finetuning-check
import torch
from fractions import Fraction

# Qwen3-30B-A3B config (commit d47d535): parameter counting.
d, heads, kv, hd = 2048, 32, 4, 128
E, k, d_ff, L, V = 128, 8, 768, 48, 151936
attn = d * heads * hd + 2 * d * kv * hd + heads * hd * d + 2 * hd  # q,k,v,o + q/k norms
expert = 3 * d * d_ff                                               # gate, up, down
shared = attn + d * E + 2 * d                                       # + router + 2 RMSNorms
emb = 2 * V * d                                                     # untied in/out embeddings
total = L * (shared + E * expert) + emb + d
active = L * (shared + k * expert) + emb + d
assert total == 30_532_122_624 and active == 3_353_032_704
assert round(total / 1e9, 1) == 30.5 and round(active / 1e9, 1) == 3.4
assert round(total * 2 / 1e9, 1) == 61.1      # bf16 GB, all experts resident
assert round(active * 2 / 1e9, 1) == 6.7
assert round(3.3e9 * 2 / 1e9, 1) == 6.6       # dense 3.3B model
assert total / active > 9

# Expert-LoRA count formula vs real tensors on a toy MoE layer.
def expert_lora_formula(L, E, r, d, d_ff):
    return L * E * 3 * r * (d + d_ff)

E2, d2, f2, r2, L2 = 4, 8, 16, 2, 1
adapters = []
for _ in range(E2):
    for (din, dout) in [(d2, f2), (d2, f2), (f2, d2)]:   # gate, up, down
        adapters.append(torch.zeros(r2, din))           # lora_A: (r, in)
        adapters.append(torch.zeros(dout, r2))          # lora_B: (out, r)
assert sum(t.numel() for t in adapters) == expert_lora_formula(L2, E2, r2, d2, f2) == 576
assert expert_lora_formula(L, E, 16, d, d_ff) == 830_472_192
assert expert_lora_formula(L, E, max(1, 16 // E), d, d_ff) == 51_904_512

# Pipeline bubble (p - 1) / (m + p - 1) for p = 4 stages, m = 16 micro-batches.
p, m = 4, 16
bubble = Fraction(p - 1, m + p - 1)
assert bubble == Fraction(3, 19) and abs(float(bubble) - 0.158) < 1e-3
print("MoE memory, expert-LoRA counts and pipeline bubble verified")
```

It proves the counting; it does not measure activations, routing buffers, or the transient
memory of materializing expert LoRA deltas, where real MoE runs spend surprise gigabytes.

### Router stability

Narrow-task fine-tuning can collapse routing onto a few experts, creating **load
imbalance**: under EP the overloaded rank sets everyone's step time. Defaults that work:

- **Keep the auxiliary load-balancing loss on** (`router_aux_loss_coef`, 0.001 for
  Qwen3-30B-A3B). Transformers computes it only when `output_router_logits=True`, which the
  config leaves `False`. The loss is derived in
  [module 02](./02-tokens-templates-and-the-causal-lm-objective.md) and the
  [transformers MoE module](../transformers/12-mixture-of-experts.md) (serving side:
  [inference MoE chapter](/courses/inference/07-architectures/07-01-mixture-of-experts.html)).
  TRL v1.15 fixed SFT so it no longer changes with world size or accumulation.
- **Consider freezing the router** for small-data SFT or LoRA, so adapters learn within the
  pretrained routing.
- **Watch per-expert token counts.** If a few experts take most tokens, raise the aux
  coefficient or freeze the router.
- For RL on MoE models, sequence-level importance ratios (GSPO) are more stable than
  token-level ones; [module 13](./13-rl-with-verifiable-rewards.md) covers why.

### Kernels, quantization and EP in practice

Grouped-GEMM kernels replace a Python loop over experts: Axolotl's **ScatterMoE** (Triton,
LoRA fused in) and SonicMoE (Hopper+)
([Axolotl custom integrations](https://docs.axolotl.ai/docs/custom_integrations.html)); Unsloth
reports about 12× faster MoE training than Transformers v4
([vendor-reported](https://unsloth.ai/docs/basics/faster-moe)). Axolotl's `expert_parallel_size`
requires FSDP2 and rejects EP with TP; it reports Qwen3-30B-A3B LoRA at 32k tokens on 2×H100
peaking at 40.7 GiB with EP=2 against 48.5 GiB for FSDP2 alone. Transformers'
`DistributedConfig(fsdp_size=8, ep_size=4)` cannot yet resume models sharded at load time.

---

## 10.7 Launchers and debugging

### Launchers

| Launcher | Command shape | Use when |
|---|---|---|
| Accelerate | `accelerate launch --config_file fsdp2.yaml train.py` | HF Trainer/TRL; FSDP2 or DeepSpeed chosen in YAML |
| torchrun | `torchrun --nproc_per_node 8 train.py` | plain PyTorch, Transformers TP/EP examples |
| DeepSpeed | `deepspeed --num_gpus 8 train.py --deepspeed ds.json` | DeepSpeed-native scripts |
| Slurm | `srun accelerate launch ...` or `srun torchrun --nnodes 2 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 ...` | HPC clusters |

Multi-node needs the same code and config on every node, the rank-0 address and port
(`MASTER_ADDR`/`MASTER_PORT`, or Accelerate's `main_process_ip` plus a per-node
`machine_rank`), and a rendezvous before the first collective.

### Debugging

Most distributed failures are **hangs**. A collective blocks until every rank calls it; if
one rank takes a different path, the others wait for the timeout (10 to 30 minutes) and die
with "Watchdog caught collective operation timeout". Typical culprits:

- A GRPO **reward function** that logs, gathers or raises on some ranks only.
- Eval, generation or `save` under `if accelerator.is_main_process:` when the call needs a
  collective. Every rank must enter `save_model`.
- Uneven batch counts per rank, or a `break` decided from a rank-local metric.

The tools, roughly in order:

```bash
export NCCL_DEBUG=INFO               # NCCL init, transport (NVLink/PCIe/IB/socket), errors
export NCCL_DEBUG_SUBSYS=INIT,NET    # narrow the INFO output by subsystem
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # checks collective shapes/order across ranks
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # kill the job on NCCL errors rather than hang
export NCCL_P2P_DISABLE=1            # isolate broken GPU-to-GPU P2P (slow, diagnosis only)
export NCCL_IB_DISABLE=1             # force sockets to test whether InfiniBand is at fault
py-spy dump --pid <pid>              # where is each rank stuck right now?
```

Axolotl's [NCCL page](https://docs.axolotl.ai/docs/nccl.html) adds `NCCL_P2P_LEVEL=NVL` and
NVIDIA's `nccl-tests` (`all_reduce_perf`) to measure real bandwidth.

| Symptom | Likely cause | Fix |
|---|---|---|
| All ranks at 100% util, low power, then watchdog timeout | rank-divergent collective (rank-0-only eval, reward, or save) | make every rank call the same collectives; `TORCH_DISTRIBUTED_DEBUG=DETAIL`; `py-spy dump` on each rank |
| Throughput barely improves from 1 to 8 GPUs | PCIe-only topology, ZeRO-3 comm-bound | check `NCCL_DEBUG=INFO` transport; ZeRO-2 or LoRA; larger micro-batch; TP only on NVLink |
| Multi-node job never starts | wrong `MASTER_ADDR`/port, firewall, wrong NIC | `NCCL_SOCKET_IFNAME`, open the port, `NCCL_DEBUG=INFO` |
| Random IB errors or very slow cross-node steps | IB misconfig or ACS on PCIe switches | `NCCL_IB_DISABLE=1` to confirm, then fix fabric; disable ACS per NVIDIA guidance |

Past roughly 70B dense, Megatron-style frameworks are the usual answer
([module 08](./08-frameworks-and-tooling.md)); RL rollout topology is in
[module 13](./13-rl-with-verifiable-rewards.md), cost in
[module 20](./20-production-serving-and-cost.md).

---

## 10.8 Lab: FSDP2 vs ZeRO-3, and QLoRA across GPUs

**Goal.** Run the same SFT job under FSDP2 and ZeRO-3, then shard QLoRA and MoE LoRA runs;
measure memory, throughput and checkpoint handling.
**Tier.** CPU for smoke; T3 (2 to 8× A100/H100, or 2×24 GB for QLoRA); T2/T3 for MoE.
**Models.** `Qwen/Qwen3-8B` (full), `Qwen/Qwen3-32B` (QLoRA), `Qwen/Qwen3-30B-A3B` (MoE),
`Qwen/Qwen3-4B` (dense LoRA baseline).
**Dataset.** `trl-lib/Capybara` (multi-turn chat).
**Scripts.** [`fsdp2_sft.py`](./code/fsdp2_sft.py) and
[`zero_memory_planner.py`](./code/zero_memory_planner.py).

### Step 0: smoke checks on CPU

```bash
python content/courses/finetuning/code/fsdp2_sft.py --smoke
python content/courses/finetuning/code/zero_memory_planner.py --smoke
```

The first asserts two-process `gloo` DDP gradients equal the full-batch gradient; the second
asserts the ZeRO and Qwen3-30B-A3B numbers. Then predict before you measure:

```bash
python content/courses/finetuning/code/zero_memory_planner.py --full --params 8.2e9 --gpus 4
python content/courses/finetuning/code/zero_memory_planner.py --full --preset qwen3-30b-a3b --gpus 4 --lora-rank 16
```

### Step 1 (L10.1): FSDP2 versus ZeRO-3 on an 8B full fine-tune

```bash
python content/courses/finetuning/code/fsdp2_sft.py --write-configs runs/ft10/cfg --gpus 4
accelerate launch --config_file runs/ft10/cfg/fsdp2.yaml \
  content/courses/finetuning/code/fsdp2_sft.py --full --variant full --model Qwen/Qwen3-8B
accelerate launch --config_file runs/ft10/cfg/zero3.yaml \
  content/courses/finetuning/code/fsdp2_sft.py --full --variant full --model Qwen/Qwen3-8B \
  --output-dir runs/ft10-zero3
```

Record tokens/s, peak memory per rank (logged every 10 steps), and the checkpoint layout
(`.distcp` shards versus `global_step*/` plus `zero_to_fp32.py`). Convert both to
safetensors and confirm they load with `from_pretrained`.

### Step 2 (L10.2): FSDP2 + QLoRA across GPUs

```bash
accelerate launch --config_file runs/ft10/cfg/fsdp2.yaml \
  content/courses/finetuning/code/fsdp2_sft.py --full --variant qlora --model Qwen/Qwen3-32B
```

The script sets `bnb_4bit_quant_storage=torch.bfloat16`. Re-run with `--quant-storage uint8`
to see FSDP fail to shard the packed weights, and once with `fsdp_offload_params: true` in
`fsdp2.yaml` to see GPU memory fall and step time rise.

**Step 3 (L10.3, optional)** is the context-parallel run in [module 17](./17-long-context-extension.md).

### Step 4 (L10.4): LoRA on an MoE

```bash
accelerate launch --config_file runs/ft10/cfg/fsdp2.yaml \
  content/courses/finetuning/code/fsdp2_sft.py --full --variant moe-lora --model Qwen/Qwen3-30B-A3B
```

This adapts attention with `target_modules` and the experts with `target_parameters` at
rank `r // num_experts`. Then run the dense baseline at similar active parameters
(`--variant lora` is bf16 all-linear LoRA, r = 16) and compare peak memory:

```bash
accelerate launch --config_file runs/ft10/cfg/fsdp2.yaml \
  content/courses/finetuning/code/fsdp2_sft.py --full --variant lora --model Qwen/Qwen3-4B \
  --output-dir runs/ft10-dense-lora
```

Log the aux loss and per-expert token counts if your stack exposes them.

**Expected outcome (qualitative).** FSDP2 and ZeRO-3 land at similar memory and throughput
on NVLink, ZeRO-3's checkpoint needing an extra conversion; on PCIe both slow at micro-batch 1
and recover with longer packed sequences. QLoRA fits 32B within 24 GB per GPU across two.
The MoE LoRA run needs several times the memory of the 4B dense run.

---

## Key takeaways

- **DDP replicates and all-reduces.** Per-rank mean-loss gradients average to the full-batch
  gradient only with equal token counts; TRL v1.15 refuses `nn.DataParallel`.
- **ZeRO stages shard optimizer state, then gradients, then parameters.** For 7.5B on 64
  GPUs: 120, 31.4, 16.6 and 1.9 GB per GPU; stage 3 costs about 1.5× DDP's traffic.
- **FSDP2 is ZeRO-3 with per-parameter `DTensor` sharding.** Frozen and trainable, bf16 and
  quantized parameters coexist without FSDP1's flat-buffer workarounds.
- **Use `fsdp_version: 2` with `fsdp_reshard_after_forward`.** `use_orig_params`,
  `sync_module_states` and `backward_prefetch` are gone; `accelerate to-fsdp2` converts.
- **Train with sharded checkpoints; consolidate offline.** A rank-0 full gather is the usual end-of-run OOM.
- **FSDP + QLoRA needs a float quant storage dtype matching the model dtype** (70B on 2×24 GB).
- **Put TP inside NVLink, FSDP across a node, replication across nodes.** `ParallelismConfig`
  composes them; HSDP keeps parameter all-gathers off the network.
- **Interconnect decides whether ZeRO-3 is free.** For 8B on 8 GPUs, ~42 GB per micro-step
  takes 0.09 s on NVLink and 0.66 s on PCIe; more tokens per micro-step amortize it.
- **MoE memory follows total parameters.** Qwen3-30B-A3B needs 61 GB of bf16 weights for
  3.3B active, and expert LoRA multiplies adapter size by $E$ unless you divide the rank.
- **Distributed hangs are almost always rank-divergent code.** Every rank must make the same collectives.

## Self-check

1. Compare FSDP2 and DeepSpeed ZeRO-3. What is per-parameter sharding, and why does it make
   LoRA and QLoRA easier than FSDP1's flat parameters?
2. Using the ZeRO formulas, how much memory per GPU does each stage need for a 7.5B model
   on 64 GPUs, and what does stage 3 cost in communication?
3. You are fine-tuning a 30B-total / 3B-active MoE. Why does it not fit where a 3B dense
   model does, and which parallelism and adapter choices would you make?

---

**Next → [11 — Quantization Across the Lifecycle: QLoRA, QAT, FP8, and Export](./11-quantization-qat-and-fp8.md)**
