---
order: 4
description: SGD through AdamW, learning-rate schedules and warmup, batch size and scaling laws, gradient accumulation and clipping, mixed precision, distributed training, and a debugging playbook.
meta: Deep Learning · training
---

# Optimization and Training in Practice

The mathematics of gradient descent is a page. Making a large network actually
converge is a craft with a specific set of decisions, each of which has a reason.
This page is that set of decisions.

## The optimisers

### SGD with momentum

$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\,v_{t+1}$$

Momentum accumulates a velocity: consistent gradient directions compound while
oscillating components cancel. With $\beta = 0.9$, a persistent gradient reaches
an effective step of $\eta/(1-\beta) = 10\eta$, and the **effective averaging
window is $1/(1-\beta)$ steps** — the number to reason with when tuning $\beta$.

For suitably tuned methods on appropriate strongly convex quadratics, momentum/acceleration can improve condition-number dependence from order $\kappa$ to order $\sqrt\kappa$, with accuracy-dependent factors. Those assumptions do not hold globally for arbitrary neural losses; momentum can also overshoot or amplify instability.

### Adam and AdamW

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**Bias correction** exists because $m_0 = v_0 = 0$, so early estimates are
biased toward zero — with $\beta_2 = 0.999$, $v_1$ underestimates by 1000×.
Dividing by $1-\beta^t$ removes exactly that bias, and the correction decays to 1
as training proceeds.

**Read the update correctly**: $\hat{m}/\sqrt{\hat{v}} \approx \pm1$ regardless
of gradient scale. Adam is closer to *sign descent with a smoothed sign* than to
scaled gradient descent. That explains both its robustness to bad loss scaling
and why its learning rates ($3\times10^{-4}$) are so much smaller than SGD's
(0.1).

**AdamW decouples weight decay**, and the distinction is not cosmetic. L2
regularisation added to the gradient gets divided by $\sqrt{\hat{v}}$ along with
everything else, so parameters with large gradients receive *less*
regularisation — the opposite of the intent. AdamW applies the decay directly:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right)$$

**Excluding biases and normalization parameters is a common recipe, not a theorem.** Gains and biases can overfit too; compare parameter-group policies under validation. Identify normalization parameters by module ownership when multidimensional gains are possible.

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)

opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
)
```

The dimensionality heuristic is convenient but does not catch every normalization parameter: LayerNorm over a multidimensional normalized shape has a multidimensional gain. It can also exclude unrelated one-dimensional parameters. The complete experiment below groups parameters by module type.

### The full family

| Optimiser | State/param | Best for | Watch out for |
|---|---|---|---|
| SGD | 0 | rarely alone | very LR-sensitive |
| SGD + momentum | 1 | CNNs, vision, long schedules | needs a good schedule |
| Nesterov | 1 | same | marginal gain in practice |
| RMSProp | one main squared-gradient state | some RNN/RL recipes | optional momentum and centering add state |
| Adam | 2 | general default | can generalise worse than SGD on vision |
| **AdamW** | 2 | **transformers, LLMs** | tune $\lambda$ separately |
| LAMB | 2 | large-batch adaptive training | layerwise trust ratio on an Adam-like update |
| LARS + momentum | 1 | large-batch SGD recipes | layerwise scaling; zero persistent moment state without momentum |
| **Lion** | 1 | memory-constrained training | needs ~10× smaller LR, higher decay |
| Adafactor | factored second moments $O(n+m)$ for a matrix | large matrices | approximation and scale/learning-rate conventions |
| 8-bit Adam | quantized states plus scale metadata | memory-limited fine-tuning | supported tensors and numerical quality must be checked |
| Shampoo / SOAP | matrix | frontier pretraining | expensive, complex |
| Sophia | 2 | LLM pretraining | diagonal Hessian estimate |

**Memory is the reason this table has so many rows.** AdamW stores two extra
tensors per parameter. A 7B model in fp32 needs 28 GB for weights and 56 GB for
optimiser state — the optimiser is twice the model. 8-bit Adam, Adafactor's
factored second moment, Lion's single state, and ZeRO sharding are all attacks on
that number.

### SGD or Adam?

| | SGD + momentum | AdamW |
|---|---|---|
| Vision CNNs, long schedules | often better final accuracy | faster to converge |
| Transformers / language | possible, often needs a different recipe | widely used adaptive baseline |
| Tuning burden | high (LR is critical) | lower |
| Memory | 1 state | 2 states |

Gradient-scale heterogeneity across embeddings, attention and normalization helps motivate adaptive methods. It does not prove SGD cannot train a transformer. AdamW is a useful baseline; compare SGD plus momentum for convolutional vision when a well-tuned long schedule is affordable.

## Learning rate: the one that matters most

### Finding it

The **LR range test**: start absurdly low, increase exponentially over a few
hundred steps, plot loss against LR. Choose roughly an order of magnitude below
where the loss starts rising.

```python
lrs, losses = [], []
for i, batch in enumerate(loader):
    lr = 1e-7 * (10 ** (i / 50))          # 10x every 50 steps
    for g in opt.param_groups: g["lr"] = lr
    loss = step(batch)
    lrs.append(lr); losses.append(loss)
    if loss > 4 * min(losses): break
```

Typical starting points: $3\times10^{-4}$ for AdamW on transformers,
$1\times10^{-3}$ for AdamW on small MLPs, $2\times10^{-5}$ to $5\times10^{-5}$
for fine-tuning a pretrained encoder, $0.1$ for SGD+momentum on a CNN.

### Schedules

| Schedule | Where used |
|---|---|
| **Warmup + cosine decay** | transformers, essentially universal |
| **Warmup + linear decay to zero** | LLM pretraining; very competitive with cosine |
| Step decay | classic ResNet recipes |
| One-cycle | fast convergence, `fastai` |
| Inverse square root | the original Transformer paper |
| Cosine with warm restarts | escaping poor basins; snapshot ensembling |
| ReduceLROnPlateau | when total steps are unknown |
| WSD (warmup–stable–decay) | continual pretraining; the stable phase allows checkpoint branching |

```python
def lr_lambda(step, warmup, total, min_ratio=0.1):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, 2000, total_steps))
```

**Warmup is often useful for transformers, but is not universally mandatory.**
Early optimizer moments reflect few observations, while feature scales and
attention patterns can change rapidly. Limiting initial movement can help some
parameterizations avoid instability. Initial logits need not be uniform and
LayerNorm does not maintain BatchNorm-style running statistics. Warmup duration
depends on residual scaling, normalization placement, batch size, initialization
and optimizer settings. Treat percentages or fixed step counts as candidate
recipes, not independent laws.

**Decay to near zero.** The end-of-training low learning rate does real work — it
is where the model settles into a minimum rather than bouncing around it.
Truncating a schedule early loses a surprising amount of final quality.

## Batch size

| Effect | Direction |
|---|---|
| Gradient noise | $\propto 1/\sqrt{B}$ — larger is smoother |
| Hardware utilisation | rises steeply, then saturates |
| Steps per epoch | falls, so fewer updates for the same data |
| Generalisation | very large batches can hurt without compensation |
| Memory | linear in $B$ |

**Scaling rules** when you change $B$:

- **Linear**: $\eta \propto B$, with warmup. Works well for SGD up to
  $B\approx8$k on ImageNet-scale problems.
- **Square root**: $\eta \propto \sqrt{B}$. Better motivated for Adam-family
  optimisers, whose update is already normalised.

The **critical batch size** is the point beyond which more batch buys little:
gradient noise is already small relative to the curvature, so you are spending
compute for nothing. It grows during training as the gradient becomes smaller
and noisier relative to itself, which is why some large runs ramp the batch size
up over time.

**Gradient accumulation** adds microbatch gradients before one optimizer update. If microbatches have sizes $n_j$, accumulate each loss sum divided by $N=\sum_j n_j$. Dividing each microbatch mean by a fixed count is equivalent only when the relevant denominators match. For masked language objectives, count valid tokens rather than padded slots. The final incomplete accumulation group must be flushed with its actual denominator.

Accumulation matches a full-batch gradient only when per-example computations are independent and parameters remain fixed during the group. BatchNorm uses different statistics, dropout draws different masks, and floating-point summation changes ordering. With SGD, an unnormalized sum changes update scale; Adam's normalized moments make the effect more complicated than multiplying its learning rate.

## Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If $\|g\| > $ `max_norm`, rescale the whole gradient vector to that norm,
preserving direction. Standard for RNNs and transformers, and nearly free
insurance against a single bad batch destroying a run.

Clip by **global** norm, not per-parameter — clipping each tensor separately
distorts the update direction. And under mixed precision with fp16, **unscale
before clipping** or you clip the wrong magnitude:

```python
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt); scaler.update()
```

Log the pre-clip gradient norm. A rising norm can warn of instability, but no fixed warning horizon is guaranteed; inspect batches and loss scale too.

## Mixed precision

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = criterion(model(x), y)
loss.backward()
```

`autocast` runs matmuls in low precision on tensor cores while keeping
reductions, softmax, and normalisation statistics in fp32.

**bf16 versus fp16**: both are 16 bits. fp16 has ten stored fraction bits and five exponent bits. Its largest finite value is 65,504, smallest positive normal about $6.10\times10^{-5}$, and smallest positive subnormal about $5.96\times10^{-8}$; hardware may flush subnormals. Loss scaling protects small gradients, not arbitrary unstable forward arithmetic. BF16 has eight exponent bits and seven stored fraction bits, giving a much wider finite range but lower precision. BF16 usually does not need loss scaling, yet it can still overflow or produce invalid values.

A common AMP recipe keeps parameters and optimizer state in FP32 while casting selected operations. Other supported recipes use master copies, quantized optimizer states or specialized low-precision training. FP32 accumulation is often prudent, but 'must always' would contradict those deliberate alternatives. Check the actual operation/device autocast policy rather than assuming every reduction has the same dtype.

## Distributed training

| Strategy | Shards | Use when |
|---|---|---|
| DDP | the batch | the model fits on one GPU |
| FSDP / ZeRO-3 | parameters, gradients, optimiser state | it does not |
| Tensor parallel | individual weight matrices | very large layers, fast interconnect |
| Pipeline parallel | layers across devices | very deep models; has bubble overhead |
| Sequence/context parallel | the sequence dimension | very long context |
| Expert parallel | MoE experts | mixture-of-experts models |

```python
# torchrun --nproc_per_node=8 train.py
dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
model = DDP(model.to(rank), device_ids=[rank], gradient_as_bucket_view=True)

sampler = DistributedSampler(dataset, shuffle=True)
for epoch in range(epochs):
    sampler.set_epoch(epoch)        # without this, every epoch has the same order
```

DDP overlaps the gradient all-reduce with the backward pass, which is why it
scales far better than the older `DataParallel`. Remember that the effective
batch is `per_gpu_batch × world_size` — scale the learning rate accordingly — and
guard singleton logging/checkpointing with global `dist.get_rank() == 0`. A local rank of zero exists on every node and would create multiple writers.

**ZeRO stages** are worth knowing by number: stage 1 shards optimiser state,
stage 2 adds gradients, stage 3 adds parameters. Stage 3 (equivalently FSDP)
shards persistent model state, while parameters are materialized for computation at the configured wrapping/gather granularity. The exact peak depends on prefetching, resharding and wrapping; it is not a promise that full-model materialization can never occur.

## A training loop that works

A complete CPU loop with tail-safe accumulation appears below. GPU AMP and distributed snippets on this page are intentionally partial adaptations, not independently executable programs. Keep their additional synchronization and skipped-update semantics separate from the single-process reference.

### Name the clocks before writing the loop

An **example** is one observation, a **token** may be one supervised position,
a **microbatch** is one forward/backward pass, an **update** changes parameters,
and an **epoch** visits the chosen training set once. Accumulation makes several
microbatches one update, except for a smaller final group. Schedules usually
advance per successful update, not per backward call.

Suppose five microbatches contain $8,8,8,8,3$ examples and accumulation is two.
The update groups contain $16,16,3$ examples. The final group divides its loss
sum by three and performs its own update. An unweighted mean of five microbatch
means also misweights the final observations; aggregate sums and counts for
epoch metrics. With token objectives, count valid supervised tokens rather than
padded positions. Equal sequence weighting would be a different objective.

### Executable reference and equivalence check

This experiment verifies unequal-microbatch reduction, then trains with a partial
final group, global clipping, module-aware decay, update-based warmup/cosine and
validation-selected weights held in memory. It downloads and writes nothing.

```python runnable
import copy
import math
import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
torch.manual_seed(51)
np.random.seed(51)
base = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 1)).double()
micro = copy.deepcopy(base)
x = torch.randn(11, 3, dtype=torch.float64)
y = torch.randn(11, 1, dtype=torch.float64)
nn.functional.mse_loss(base(x), y).backward()
for indices in torch.arange(11).split(4):
    (nn.functional.mse_loss(micro(x[indices]), y[indices], reduction="sum") / 11).backward()
for whole_parameter, micro_parameter in zip(base.parameters(), micro.parameters()):
    torch.testing.assert_close(whole_parameter.grad, micro_parameter.grad, atol=1e-12, rtol=1e-12)

X, labels = make_moons(n_samples=602, noise=0.16, random_state=51)
X_train, X_hold, y_train, y_hold = train_test_split(
    X, labels, test_size=202, stratify=labels, random_state=52
)
X_val, X_test, y_val, y_test = train_test_split(
    X_hold, y_hold, test_size=101, stratify=y_hold, random_state=53
)
scaler = StandardScaler().fit(X_train)
def convert(a, b):
    return (torch.tensor(scaler.transform(a), dtype=torch.float32),
            torch.tensor(b, dtype=torch.long))
xt, yt = convert(X_train, y_train)
xv, yv = convert(X_val, y_val)
xs, ys = convert(X_test, y_test)
torch.manual_seed(54)
model = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 2))
normalization_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                       nn.BatchNorm2d, nn.BatchNorm3d)
no_decay_ids = set()
for module in model.modules():
    for name, parameter in module.named_parameters(recurse=False):
        if name == "bias" or isinstance(module, normalization_types):
            no_decay_ids.add(id(parameter))
decay, no_decay = [], []
for parameter in model.parameters():
    if parameter.requires_grad:
        (no_decay if id(parameter) in no_decay_ids else decay).append(parameter)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}], lr=0.02
)
epochs, micro_size, accumulation = 35, 37, 3
updates_per_epoch = math.ceil(math.ceil(len(xt) / micro_size) / accumulation)
total_updates = epochs * updates_per_epoch
warmup_updates = 8
generator = torch.Generator().manual_seed(55)
updates = 0
best = float("inf")
best_state = copy.deepcopy(model.state_dict())
with torch.inference_mode():
    initial_loss = nn.functional.cross_entropy(model(xt), yt).item()
for epoch in range(epochs):
    model.train()
    batches = list(torch.randperm(len(xt), generator=generator).split(micro_size))
    seen = 0
    for offset in range(0, len(batches), accumulation):
        group = batches[offset:offset + accumulation]
        denominator = sum(len(indices) for indices in group)
        optimizer.zero_grad(set_to_none=True)
        for indices in group:
            loss_sum = nn.functional.cross_entropy(model(xt[indices]), yt[indices], reduction="sum")
            assert torch.isfinite(loss_sum)
            (loss_sum / denominator).backward()
            seen += len(indices)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(gradient_norm)
        if updates < warmup_updates:
            ratio = (updates + 1) / warmup_updates
        else:
            progress = (updates - warmup_updates) / max(1, total_updates - warmup_updates - 1)
            ratio = 0.1 + 0.9 * (1 + math.cos(math.pi * min(progress, 1))) / 2
        for group_parameters in optimizer.param_groups:
            group_parameters["lr"] = 0.02 * ratio
        optimizer.step()
        updates += 1
    assert seen == len(xt)
    model.eval()
    with torch.inference_mode():
        validation = nn.functional.cross_entropy(model(xv), yv).item()
    if validation < best:
        best = validation
        best_state = copy.deepcopy(model.state_dict())
assert updates == total_updates
model.load_state_dict(best_state)
model.eval()
with torch.inference_mode():
    final_loss = nn.functional.cross_entropy(model(xt), yt).item()
    accuracy = (model(xs).argmax(1) == ys).float().mean().item()
assert final_loss < initial_loss * 0.6
assert accuracy > 0.88
print("Updates:", updates, "initial/final training loss:", initial_loss, final_loss)
print("Selected validation loss:", best, "test accuracy:", accuracy)
print("Unequal-microbatch gradient equivalence and complete-tail checks passed.")
```

The equivalence check uses one scalar output. For $D$ equally weighted regression
outputs, default MSE mean divides by $BD$; match that extra dimension when
accumulating sums. Reduction definitions belong beside the implementation.

### AMP and distributed extensions

With FP16 scaling, keep one scale throughout an accumulation group, unscale once
after its final backward, then clip and step. Updating scale between microbatches
mixes differently scaled gradients. If `GradScaler` skips a nonfinite update,
do not advance a successful-update schedule as though weights changed. The
[official AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
describe this order.

DDP normally averages rank gradients. With $R$ ranks and unequal valid-token
counts $N_r$, averaging local means gives an average of rank means, not the
global token mean. If $N=\sum_rN_r$, backpropagating each local loss sum scaled
by $R/N$ compensates for the later rank average. All ranks must agree on update
boundaries and handle empty contributions consistently.

Use `no_sync()` for nonfinal microbatches when appropriate, with the forward
inside the context too. The final backward synchronizes accumulated gradients.
Clipping each rank independently before averaging does not generally equal
clipping the final global gradient. Sharded gradients require a distributed-aware
global norm rather than the norm of one shard.

### Resuming is more than loading weights

A resumable state includes model parameters/buffers, optimizer moments,
scheduler state, successful-update count, scaler state if used, random generator
states and data-order progress. Mid-epoch replay additionally needs an iterator
position or equivalent deterministic schedule. Copy an in-memory best state;
references to live tensors may change under subsequent training.

Distinguish best-validation from latest-resumable checkpoints. Combining old best
weights with current optimizer moments is an intentional restart at best, not
an exact resume. Preserve preprocessing, label mappings and configuration too.
Some serialization formats can execute code during loading; do not load untrusted
training artifacts as arbitrary Python objects. The reference experiment avoids
file serialization entirely.

## Debugging playbook

### Start here, always

**Overfit one batch.** Take 32 examples and train until the loss is ~0. Disable strong regularization, check sufficient capacity and remove contradictory examples first. Failure then narrows the diagnosis but can still involve data contracts or optimization; it is not automatic proof of an implementation bug.

**Check the initial loss.** Uniform logits give $\log K$ (2.303 for ten classes), while random nonuniform logits or intentional class-prior biases can give different losses. A regression on standardised targets should
start near the target variance. Use those references to inspect scale and targets, not as unconditional assertions.

### The symptom table

| Symptom | Causes, in order of likelihood | Diagnostics |
|---|---|---|
| Loss is `NaN` immediately | LR far too high; `log(0)`; bad data; fp16 overflow | `set_detect_anomaly(True)`; check for `NaN` in inputs |
| Loss `NaN` after N steps | LR too high for a sharp region; a specific bad batch | clip gradients; log the batch index; lower LR |
| Loss does not move | LR ~0; `zero_grad` missing; `step` missing; frozen parameters | print parameter deltas; check `requires_grad` |
| Loss decreases then plateaus high | underfitting; too much regularisation; too little capacity | overfit-one-batch; raise capacity |
| Train loss falls, val rises | overfitting | more data, augmentation, dropout, early stop |
| Val loss below train loss | dropout is active only in training — usually fine | check `model.eval()` in validation |
| Loss oscillates | LR too high; batch too small | lower LR; raise batch or accumulation |
| Sudden spike then recovery | one outlier batch | clip; inspect that batch |
| Sudden spike, never recovers | divergence | restart from checkpoint with lower LR |
| GPU utilisation low | dataloader-bound | more workers, `pin_memory`, prefetch, profile |
| Memory grows every epoch | a graph retained across iterations | `.detach()`/`.item()` on accumulators |
| Different results every run | unseeded RNG, non-deterministic kernels | seed everything; `use_deterministic_algorithms` |

### What to log, every run

| Metric | Reveals |
|---|---|
| Train and validation loss | the basic picture |
| Learning rate | that the schedule fired as intended |
| **Gradient norm (pre-clip)** | scale changes and potential instability |
| **Actual update-to-weight ratio** $\|\theta_{t+1}-\theta_t\|/(\|\theta_t\|+\epsilon)$ | compare actual optimizer movement across layers and time |
| Per-unit activation frequency across representative batches | persistent inactive units; aggregate zero fraction alone is not a dying-ReLU diagnosis |
| Weight and activation histograms | saturation, collapse |
| Throughput (examples/sec, tokens/sec) | regressions in the input pipeline |
| GPU memory and utilisation | headroom and bottleneck |

## Efficiency levers

| Lever | Typical gain | Cost |
|---|---|---|
| Mixed precision (bf16) | device- and workload-dependent throughput and activation-memory gains | finite range, reduced precision and operation-specific dtype behavior still require validation |
| `torch.compile` | 1.3–2× | recompiles on shape changes |
| FlashAttention | avoids materialized attention matrices | supported shapes, masks, hardware and precision conventions |
| Fused optimiser (`fused=True`) | lower launch/intermediate overhead in supported cases | device/dtype and implementation constraints |
| `channels_last` for CNNs | potentially faster supported convolutions | layout conversions and backend dependence |
| Gradient checkpointing | fits larger models | recomputation and state/RNG handling |
| Larger batch to saturate the GPU | throughput | memory |
| Dataloader tuning | can be the entire bottleneck | none |
| 8-bit optimiser | smaller moment storage plus quantization metadata | validate numerical behavior and supported tensors |
| Sequence packing (LLMs) | up to 2× on short sequences | implementation complexity |

Profile before optimizing. Low utilization can reflect input stalls, tiny kernels, synchronization, communication, insufficient parallelism or memory limits. A timeline distinguishes them; a single utilization percentage does not identify the bottleneck.

## Worked optimizer reasoning

### One step is not the whole trajectory

For a constant scalar gradient $g=2$, Adam initialized at zero with
$\beta_1=0.9,\beta_2=0.999$ has $m_1=0.2$, $v_1=0.004$.
Bias correction gives $\hat m_1=2$, $\hat v_1=4$, so the normalized direction
is approximately one. With learning rate $0.01$, the first gradient step is
about $0.01$, not $0.02$. SGD without momentum would take $0.02$.

If that gradient changes sign, the first moment and second moment respond on
different time scales. The resulting direction need not be exactly a sign and
can temporarily oppose the current gradient. Adam's bias correction removes
the zero-initialization attenuation of exponential averages under a stationary
expectation; it does not make a rapidly changing estimate a perfect moment of
the current distribution.

With AdamW and parameter $\theta=3$, learning rate $0.01$ and decay $0.1$,
the decoupled shrinkage contribution is $0.003$ for that step. Across a changing
schedule, decay-only evolution is

$$\theta_T=\theta_0\prod_{t=0}^{T-1}(1-\eta_t\lambda).$$

Thus decay strength interacts with update count and the learning-rate schedule
even though it is decoupled from adaptive gradient normalization. Holding
`weight_decay` fixed while doubling updates changes cumulative shrinkage.
Adding $\lambda\theta$ inside Adam instead modifies both moment histories;
it is a different algorithm, not merely another spelling of AdamW.

### Curvature and batch noise

For the quadratic $L(\theta)=\frac12\theta^\top H\theta$ with symmetric
positive-definite $H$, plain gradient descent updates each eigen-coordinate by
$1-\eta\lambda_i$. Stability requires $0<\eta<2/\lambda_{max}$.
If eigenvalues are $1$ and $100$, a step stable in the steep direction can move
slowly in the flat one. This explains the motivation for momentum and
preconditioning without treating a deep network as a fixed quadratic globally.

For independent per-example gradients with covariance $\Sigma$, the covariance
of a batch mean is $\Sigma/B$. Its standard deviation therefore scales as
$1/\sqrt B$, while correlations, sampling without replacement and nonstationary
data modify the simple model. A large batch reduces noise per update but may
give fewer updates for a fixed data budget. Compare methods under both example
count and wall-clock budget rather than silently switching the denominator.

Linear and square-root learning-rate scaling are approximations tied to different
limits and optimizer parameterizations. Once the batch exceeds a useful noise
scale, increasing it can improve hardware throughput without improving
sample efficiency. That can still be valuable when latency or training time,
rather than total FLOPs, is the binding constraint.

## Self-check

1. **What does Adam bias correction fix?** Zero-initialized exponential averages
   carry factors $1-\beta^t$ under stationary expectations. At the first step
   with $\beta_2=0.999$, the raw second moment has a factor $0.001$. Correction
   removes that attenuation, not nonstationarity or finite-sample noise.
2. **Why AdamW rather than Adam plus L2?** L2 enters the gradient and moment
   histories; AdamW applies a separate parameter shrinkage. Bias/norm exclusions
   are common validation-tested policies, not proof those parameters cannot overfit.
3. **Why consider warmup?** It limits early movement while moments and feature
   scales settle, and can stabilize some residual/normalization parameterizations.
   Pre-norm or different initialization may reduce the need; no architecture name
   makes a fixed warmup duration mandatory.
4. **What if batch size quadruples?** Linear scaling suggests four times the LR;
   square-root scaling suggests twice. Neither is universal. Keep objective and
   update budget explicit, test stability and retune the schedule if needed.
5. **Why unscale before clipping?** A scale factor multiplies every gradient
   norm. Clipping the scaled vector at the ordinary threshold imposes a much
   smaller true threshold. Unscale once after accumulation, then clip.
6. **What does update ratio $0.1$ mean?** The parameter moved by ten percent of
   its norm in one update. Inspect whether that is intended, particularly for
   near-zero initial parameters; it is a warning context, not a universal failure.
7. **What does fitting one batch establish?** It checks that the chosen model,
   objective and optimizer can fit a small consistent dataset. It does not prove
   generalization or absence of leakage, and failure can reflect capacity or labels.
8. **How do $8,8,3$ samples accumulate into one update?** Sum losses and divide
   by nineteen. Averaging three microbatch means equally overweights the final
   three observations. For masked tokens, substitute valid-token counts.
9. **Which rank writes one checkpoint?** Global rank zero, not local rank zero.
   The latter occurs once per node. Distributed checkpoint APIs may intentionally
   involve every rank, following a different coordinated protocol.
10. **What must an exact resume restore?** At least model, optimizer, schedule,
    update counter and relevant RNG/data progress, plus scaler state when used.
    Loading weights alone restores a predictor, not the original training process.

## Where to go next

- [Regularization & Normalization](./regularization-and-normalization.md) — the
  layers that make these schedules work.
- [Backpropagation & Autodiff](./backpropagation-and-autodiff.md) — where the
  gradients come from.
- [Activations & Initialization](./activations-and-initialization.md) — the
  starting conditions.

Primary reading: [Adam](https://arxiv.org/abs/1412.6980),
[decoupled weight decay](https://arxiv.org/abs/1711.05101),
[LARS, Algorithm 1](https://arxiv.org/html/1708.03888v3#S4),
[AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
and [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The mathematical background is developed in [optimization](../math/optimization.md).
