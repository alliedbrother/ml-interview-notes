---
order: 4
description: PyTorch from tensors and autograd through nn.Module, data loading, the training loop, mixed precision, distributed training, torch.compile, and inference export.
meta: Libraries · deep learning
---

# PyTorch: Tensors, Autograd, and a Training Loop You Wrote Yourself

PyTorch won because it made the computation graph an ordinary Python execution
trace. There is no session, no placeholder, no compile step before you can print
a tensor. The cost of that design — you write the training loop — turns out to be
the thing that makes it teachable: nothing is hidden.

## Tensors

A tensor is NumPy's `ndarray` with three additions: it can live on a GPU, it can
record operations for autograd, and it carries a `requires_grad` flag.

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
x.shape, x.dtype, x.device          # torch.Size([2,2]) torch.float32 cpu
x.to("cuda")                        # or x.cuda()
x.float(), x.half(), x.bfloat16()   # dtype casts
```

**PyTorch defaults to `float32`**, unlike NumPy's `float64`. That is the right
default for deep learning and a common source of dtype mismatch when converting.

### Creation and the NumPy bridge

```python
torch.zeros(3, 4);  torch.ones(3, 4);  torch.empty(3, 4)
torch.arange(10);   torch.linspace(0, 1, 11)
torch.randn(3, 4);  torch.rand(3, 4);  torch.randint(0, 10, (3,))
torch.zeros_like(x); torch.full((2,2), 7.0)
torch.eye(4)

t = torch.from_numpy(a)        # SHARES memory with the NumPy array
a2 = t.numpy()                 # also shares (CPU only)
t2 = torch.as_tensor(a)        # shares if possible, else copies
t3 = torch.tensor(a)           # ALWAYS copies (and warns if given a tensor)
```

### Shape manipulation, and the view/reshape distinction

| Operation | Returns | Note |
|---|---|---|
| `view(shape)` | view | requires contiguous memory; errors otherwise |
| `reshape(shape)` | view or copy | works always; copies when it must |
| `permute(dims)` / `transpose(a,b)` | view | makes the tensor non-contiguous |
| `contiguous()` | copy if needed | what `view` complains about |
| `squeeze` / `unsqueeze(dim)` | view | drop or add a size-1 axis |
| `expand(shape)` | view, stride 0 | broadcast without copying |
| `repeat(shape)` | copy | actually duplicates data |
| `flatten(start, end)` | view or copy | flatten a range of dims |
| `einops.rearrange` | as needed | far more readable for 4-D+ |

```python
x = torch.randn(2, 3, 4)
x.permute(2, 0, 1).view(-1)          # RuntimeError: view size is not compatible
x.permute(2, 0, 1).contiguous().view(-1)   # fine
x.permute(2, 0, 1).reshape(-1)             # fine, copies internally
```

The distinction is exactly NumPy's strides story: `permute` rewrites strides,
`view` requires the strides to describe a contiguous block.

`expand` vs `repeat` matters for memory: `expand` sets a stride to zero and costs
nothing; `repeat` materialises. Use `expand` for broadcasting a mask over a
batch, never `repeat`.

## Autograd

Every tensor with `requires_grad=True` records the operations applied to it into
a dynamic graph. Calling `.backward()` on a scalar walks that graph in reverse
and accumulates gradients into `.grad`.

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x            # y.grad_fn = <MulBackward0>
L = y ** 2           # L.grad_fn = <PowBackward0>
L.backward()
w.grad               # dL/dw = 2*(w*x)*x = 2*6*3 = 36
```

```mermaid
flowchart TD
    W["w<br/>requires_grad=True<br/>leaf"] --> MUL["mul"]
    X["x<br/>no grad"] --> MUL
    MUL --> Y["y = w*x<br/>grad_fn MulBackward"]
    Y --> POW["pow"]
    POW --> L["L = y^2<br/>scalar"]
    L -.->|"backward: seed dL/dL = 1"| POW
    POW -.->|"dL/dy = 2y"| Y
    Y -.->|"dL/dw = dL/dy * x"| W
    W -.->|"accumulate into w.grad"| ACC["w.grad += 36"]
```

### The rules people get wrong

**Gradients accumulate.** `.grad` is added to, not replaced. This is deliberate —
it lets you sum gradients from several backward passes, which is exactly what
gradient accumulation and multi-task losses need. It also means forgetting
`zero_grad()` silently trains on the sum of all previous batches.

```python
optimizer.zero_grad(set_to_none=True)   # set_to_none frees memory and is faster
```

**The graph is freed after `backward()`.** Calling it twice raises unless you
pass `retain_graph=True`. If you need two backward passes over one forward, that
flag is the answer — but the more common cause of that error is accidentally
keeping a graph across iterations.

**Detach breaks the graph.**

```python
z = y.detach()          # same data, no grad history
with torch.no_grad():   # nothing inside records history
    val_loss = criterion(model(xb), yb)
```

`torch.no_grad()` for inference and validation is not just tidiness: it avoids
building the graph, which saves substantial memory. `torch.inference_mode()` is
stricter and slightly faster still, and is the right choice for serving.

**Accumulating a Python float, not a tensor.** `total_loss += loss` keeps every
graph alive and leaks memory until you OOM. Use `total_loss += loss.item()` or
`loss.detach()`.

**In-place operations can break autograd.** `x += 1` on a tensor needed for the
backward pass raises "a variable needed for gradient computation has been
modified". Underscore-suffixed methods (`add_`, `relu_`, `clamp_`) are in-place.
Use them only where you know the value is not needed.

### Custom autograd

```python
class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g            # pretend round has derivative 1
```

This is the straight-through estimator that makes quantisation-aware training
and discrete latents work. Verify any custom Function with
`torch.autograd.gradcheck` in `float64`.

## nn.Module

A `Module` owns parameters and submodules, and knows how to move, save, and
switch modes for all of them recursively.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()                       # forgetting this breaks everything
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
        )
        self.register_buffer("running_count", torch.zeros(1))   # state, not a parameter

    def forward(self, x):
        return self.net(x)
```

| Concept | Meaning |
|---|---|
| `nn.Parameter` | a tensor registered as learnable; appears in `.parameters()` |
| `register_buffer` | persistent state that is **not** learned (BN statistics, masks, positional tables) |
| `model.train()` / `model.eval()` | switches Dropout and BatchNorm behaviour |
| `state_dict()` | an `OrderedDict` of tensors — parameters and buffers |
| `.to(device)` | moves parameters and buffers in place |
| `.apply(fn)` | recursively applies `fn` to every submodule (used for init) |

**`model.eval()` is not `torch.no_grad()`.** The first changes layer behaviour
(dropout off, BatchNorm uses running statistics); the second stops graph
building. Validation needs both.

**A plain Python list of modules is invisible.** Use `nn.ModuleList` or
`nn.ModuleDict`, or the parameters will not move to the GPU, will not be
saved, and will not be optimised.

## Data loading

```python
from torch.utils.data import Dataset, DataLoader

class TabularDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(
    TabularDS(X, y), batch_size=256, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True,
    prefetch_factor=4, drop_last=True,
)
```

| Argument | Why |
|---|---|
| `num_workers` | parallel loading in subprocesses; 4–8 per GPU is typical |
| `pin_memory` | page-locked host memory enables async `.to(device, non_blocking=True)` |
| `persistent_workers` | avoids re-spawning workers every epoch |
| `prefetch_factor` | batches queued per worker |
| `drop_last` | avoids a ragged final batch — matters for BatchNorm and for fixed shapes |
| `collate_fn` | custom batching: padding variable-length sequences |
| `sampler` | `WeightedRandomSampler` for imbalance, `DistributedSampler` for DDP |

**Use `IterableDataset` for streaming** data that does not fit on disk or arrives
from a queue — but then you must shard by worker yourself, or every worker
yields the same data.

**The RNG-in-workers bug**: with `fork`, every worker inherits the same NumPy
random state and produces identical augmentations. Use a `worker_init_fn` that
reseeds from `torch.initial_seed()`, or use `torch.rand` (which PyTorch seeds
per worker correctly).

## The training loop, annotated

```python
model = MLP(d_in, 512, n_classes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
scaler = torch.amp.GradScaler("cuda")          # fp16 only; unnecessary for bf16
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(xb), yb) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(opt)                                   # before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

    model.eval()
    total, correct = 0, 0
    with torch.inference_mode():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            correct += (logits.argmax(-1).cpu() == yb).sum().item()
            total += len(yb)
    print(f"epoch {epoch}  val acc {correct/total:.4f}")
```

Points worth stating explicitly:

- **`CrossEntropyLoss` takes logits, not probabilities.** It applies
  `log_softmax` internally. Passing softmax output is a real and common bug that
  degrades training silently. The same applies to
  `BCEWithLogitsLoss` vs `BCELoss` — always prefer the `WithLogits` version for
  numerical stability.
- **Unscale before clipping.** Clipping scaled gradients clips the wrong
  magnitude.
- **Divide the loss by `accum_steps`** so the accumulated gradient matches a real
  large batch.
- **`.item()` synchronises** the GPU. Calling it every step in a tight loop
  serialises host and device; accumulate on-device and sync once per epoch when
  it matters.

## Checkpointing

```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scheduler": sched.state_dict(),
    "scaler": scaler.state_dict(),
    "rng": torch.get_rng_state(),
    "config": cfg,
}, "ckpt.pt")

ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
```

Save the `state_dict`, not the module object — pickling the module ties the file
to your source layout. Save the optimiser too: resuming without Adam's moments
is a different training run. `weights_only=True` is the safe load path and is now
the default in recent versions.

## Mixed precision

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x)
    loss = criterion(out, y)
```

`autocast` runs matmul-heavy ops in low precision and keeps reductions, softmax,
and normalisation in fp32. **bf16 needs no `GradScaler`**; fp16 does, because its
5-bit exponent underflows on small gradients. On Ampere and later, use bf16 and
delete the scaler.

Never wrap `loss.backward()` inside `autocast` — the backward automatically uses
the dtypes recorded in the forward.

## Distributed training

| Strategy | Splits | Use when |
|---|---|---|
| `DataParallel` | batch, single process | deprecated — don't |
| `DistributedDataParallel` | batch, one process per GPU | the standard for data parallelism |
| FSDP / ZeRO-3 | parameters, gradients, optimiser state | model does not fit on one GPU |
| Tensor parallel | individual matrices across GPUs | very large layers; needs fast interconnect |
| Pipeline parallel | layers across GPUs | very deep models; has bubble overhead |

```python
# torchrun --nproc_per_node=8 train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
model = DDP(model.to(rank), device_ids=[rank])

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=per_gpu_bs)

for epoch in range(epochs):
    sampler.set_epoch(epoch)     # without this, every epoch has the same order
    ...
```

DDP overlaps the gradient all-reduce with the backward pass, which is why it
scales far better than `DataParallel`'s scatter/gather. Note that the effective
batch is `per_gpu_bs × world_size`, so scale the learning rate accordingly, and
that logging should be guarded on `rank == 0`.

## torch.compile

```python
model = torch.compile(model)     # mode="max-autotune" for the aggressive path
```

`torch.compile` traces the model with TorchDynamo, lowers to an intermediate
representation, and generates fused Triton kernels via TorchInductor. Typical
speedups are 1.3–2× on training and more on inference-heavy small ops, mostly by
eliminating kernel-launch overhead and memory round-trips through fusion.

What breaks it: data-dependent control flow, `.item()` inside the model, printing
tensors, and shapes that change every step (each new shape triggers a
recompilation). Diagnose with `TORCH_LOGS="graph_breaks,recompiles"`.

## Memory

Rough training memory per parameter under AdamW with bf16 autocast:
2 (bf16 weights) + 4 (fp32 master) + 4 + 4 (Adam moments) + 4 (fp32 gradients)
≈ **18 bytes**, plus activations, which scale with batch size × sequence length ×
depth and are frequently the dominant term.

| Technique | Saves | Costs |
|---|---|---|
| Gradient accumulation | activation memory | wall-clock (more steps per update) |
| Gradient checkpointing | most activations | ~30% more compute (recomputes forward) |
| Mixed precision | ~half of activations and weights | needs care with fp16 |
| 8-bit optimiser (`bitsandbytes`) | ~6 bytes/param | tiny quality impact |
| FSDP / ZeRO | shards states across GPUs | communication |
| LoRA / PEFT | optimiser state for frozen weights | limited expressivity |
| Smaller batch + accumulation | activations | throughput |

```python
from torch.utils.checkpoint import checkpoint
h = checkpoint(self.expensive_block, x, use_reentrant=False)
```

Debug OOM with `torch.cuda.memory_summary()` and
`torch.cuda.max_memory_allocated()`. A steadily growing allocation across epochs
almost always means a retained graph — look for a tensor accumulated without
`.detach()`.

## Inference and export

```python
model.eval()
with torch.inference_mode():
    logits = model(x)
```

| Path | Use for |
|---|---|
| `torch.compile` | fastest Python-native serving |
| `torch.export` | ahead-of-time graph capture, the modern replacement for TorchScript |
| TorchScript (`trace`/`script`) | legacy; still deployed widely |
| ONNX (`torch.onnx.export`) | cross-runtime portability, ONNX Runtime / TensorRT |
| `torchao` quantisation | int8/int4 weights for memory-bound decoding |
| vLLM / SGLang / TensorRT-LLM | serving LLMs; do not hand-roll this |

`torch.jit.trace` records one execution and therefore **bakes in any control
flow** — a model with `if` branches traces incorrectly. `torch.jit.script`
compiles the source instead but supports only a subset of Python.

## Debugging checklist

| Symptom | First things to check |
|---|---|
| Loss does not decrease | LR far too low/high; forgot `zero_grad`; forgot `optimizer.step()`; passing probabilities to `CrossEntropyLoss` |
| Loss is `NaN` | LR too high; `log(0)`; fp16 overflow; a bad batch — add `clip_grad_norm_` and check inputs |
| Train loss ≪ val loss | overfitting, or `model.eval()` never called |
| Val loss lower than train | dropout active in train only — usually fine |
| Memory grows each epoch | accumulating tensors with graphs attached |
| Model does not move to GPU | modules in a plain list; use `nn.ModuleList` |
| Results not reproducible | unseeded RNG, non-deterministic kernels, `num_workers` ordering |
| GPU utilisation low | dataloader-bound — raise `num_workers`, `pin_memory`, prefetch |
| Slow with `torch.compile` | graph breaks and recompilations from dynamic shapes |

**Overfit one batch first.** Take a single batch, train on it repeatedly, and
confirm the loss goes to ~0. If it cannot memorise 32 examples, the bug is in the
model or the loss, not in the data or the schedule. This is the single most
effective debugging step in deep learning.

## The ecosystem

| Library | Purpose |
|---|---|
| `torchvision` / `torchaudio` / `torchtext` | datasets, transforms, pretrained models |
| `einops` | readable tensor rearrangement — `rearrange(x, 'b h n d -> b n (h d)')` |
| PyTorch Lightning / `accelerate` | remove training-loop boilerplate, handle distributed |
| Hugging Face `transformers` | pretrained models and trainers |
| `torchmetrics` | distributed-correct metric computation |
| `bitsandbytes` | 8-bit optimisers, 4-bit quantised inference |
| `torchao` | native quantisation and sparsity |
| `timm` | vision backbones |
| `captum` | attribution and interpretability |

## Self-check

1. Why does PyTorch accumulate gradients rather than overwrite them, and what
   does that enable?
2. Give the difference between `model.eval()` and `torch.no_grad()`, and say what
   validation needs.
3. When does `view` fail where `reshape` succeeds?
4. Why is bf16 preferable to fp16 for training, and what can you delete when you
   switch?
5. Your GPU memory grows every epoch. Name the most likely cause and the fix.
6. What does `expand` do that `repeat` does not, and why does it matter?
7. Your model will not learn. Describe the "overfit one batch" test and what each
   outcome tells you.

## Where to go next

- [TensorFlow & Keras](./tensorflow.md) — the other framework, and what it does
  differently.
- [Hugging Face ecosystem](./huggingface.md) — pretrained models on top of this.
- [Transformers Deep Dive](/courses/transformers/) — the architecture these
  tensors are usually arranged into.
