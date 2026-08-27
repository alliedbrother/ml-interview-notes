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

Theoretically, momentum improves the convergence rate from $O(\kappa)$ to
$O(\sqrt{\kappa})$ iterations for a condition number $\kappa$. For an
ill-conditioned loss surface — which every deep network has — that is the
difference between converging and not.

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

**Exclude biases and normalisation parameters from weight decay.** Decaying a
LayerNorm gain toward zero is meaningless and measurably harmful.

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

The `p.ndim <= 1` test is the standard idiom: it catches every bias and every
normalisation gain in one condition.

### The full family

| Optimiser | State/param | Best for | Watch out for |
|---|---|---|---|
| SGD | 0 | rarely alone | very LR-sensitive |
| SGD + momentum | 1 | CNNs, vision, long schedules | needs a good schedule |
| Nesterov | 1 | same | marginal gain in practice |
| RMSProp | 1 | RNNs, RL | no momentum |
| Adam | 2 | general default | can generalise worse than SGD on vision |
| **AdamW** | 2 | **transformers, LLMs** | tune $\lambda$ separately |
| LAMB / LARS | 2 | batch sizes in the tens of thousands | layerwise trust ratio |
| **Lion** | 1 | memory-constrained training | needs ~10× smaller LR, higher decay |
| Adafactor | $O(n+m)$ | huge embedding matrices | slight quality cost |
| 8-bit Adam | ~0.5 | memory-limited fine-tuning | negligible quality cost |
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
| Transformers / language | **does not train well** | required |
| Tuning burden | high (LR is critical) | lower |
| Memory | 1 state | 2 states |

The reason SGD fails on transformers appears to be the extreme heterogeneity of
gradient scales across embeddings, attention projections, and LayerNorm
parameters — exactly what per-parameter adaptive scaling exists to handle. Use
AdamW for anything transformer-shaped; consider SGD+momentum for convolutional
vision with a long cosine schedule.

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
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, 2000, total_steps))
```

**Warmup is not optional for transformers.** At initialisation, attention logits
are near-uniform, the residual stream carries no useful signal, and Adam's
second-moment estimate is based on a handful of samples. A full-size step then
destabilises normalisation statistics and can land the model in a basin it never
escapes. Use 1–3% of total steps, or 2,000–10,000 steps.

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

**Gradient accumulation** simulates a large batch on small hardware:

```python
for i, batch in enumerate(loader):
    loss = model(batch).loss / accum_steps      # scale so the sum matches a real batch
    loss.backward()                             # gradients accumulate
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        opt.zero_grad(set_to_none=True)
```

The division by `accum_steps` is essential and easy to forget — without it your
effective learning rate is `accum_steps` times too large. Note that accumulation
is not *identical* to a real large batch when BatchNorm is present, since BN
statistics are computed per micro-batch.

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

Log the pre-clip gradient norm. A steadily rising norm predicts divergence
several hundred steps before it happens.

## Mixed precision

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = criterion(model(x), y)
loss.backward()
```

`autocast` runs matmuls in low precision on tensor cores while keeping
reductions, softmax, and normalisation statistics in fp32.

**bf16 versus fp16**: both are 16 bits. fp16 has 10 mantissa bits and a 5-bit
exponent, so it overflows above 65,504 and underflows below $6\times10^{-5}$ —
and gradients routinely live below that, which is why fp16 needs loss scaling.
bf16 keeps fp32's 8-bit exponent, so it never overflows where fp32 would not, at
the cost of 3 mantissa bits. **bf16 needs no `GradScaler`**, which is why it is
the default on Ampere and later and on TPUs.

What must stay fp32 regardless: master weights, optimiser state, loss
accumulation, and normalisation statistics. The standard mixed-precision recipe
keeps all four.

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
guard logging and checkpointing on `rank == 0`.

**ZeRO stages** are worth knowing by number: stage 1 shards optimiser state,
stage 2 adds gradients, stage 3 adds parameters. Stage 3 (equivalently FSDP)
means no single GPU ever holds the whole model, at the cost of gathering
parameters per layer during the forward pass.

## A training loop that works

```python
model = build_model().to(device)
opt = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine)
best, patience = float("inf"), 0

for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        batch = to_device(batch, device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / accum
        loss.backward()

        if (i + 1) % accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                log({"loss": loss.item() * accum, "lr": sched.get_last_lr()[0],
                     "grad_norm": gn.item()})

    model.eval()
    val = evaluate(model, val_loader)          # under torch.inference_mode()
    if val < best:
        best, patience = val, 0
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch}, "best.pt")
    else:
        patience += 1
        if patience >= early_stop:
            break
```

## Debugging playbook

### Start here, always

**Overfit one batch.** Take 32 examples and train until the loss is ~0. If the
model cannot memorise 32 examples, the bug is in the model, the loss, or the
gradient path — not in the data, the schedule, or the optimiser. This single test
localises most bugs in minutes.

**Check the initial loss.** A $K$-class classifier should start at
$\log K$ (2.303 for 10 classes). A regression on standardised targets should
start near the target variance. If not, something is wrong before step 1.

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
| **Gradient norm (pre-clip)** | divergence, several hundred steps early |
| **Update-to-weight ratio** $\lVert\eta g\rVert/\lVert w\rVert$ | should be $\approx10^{-3}$; orders off means the LR is wrong |
| Fraction of zero activations per layer | dying ReLUs |
| Weight and activation histograms | saturation, collapse |
| Throughput (examples/sec, tokens/sec) | regressions in the input pipeline |
| GPU memory and utilisation | headroom and bottleneck |

## Efficiency levers

| Lever | Typical gain | Cost |
|---|---|---|
| Mixed precision (bf16) | 2–3× | care with fp16 only |
| `torch.compile` | 1.3–2× | recompiles on shape changes |
| FlashAttention | large memory saving, some speed | none — strictly better |
| Fused optimiser (`fused=True`) | 5–10% | none |
| `channels_last` for CNNs | 20–40% on tensor cores | none |
| Gradient checkpointing | fits larger models | ~30% more compute |
| Larger batch to saturate the GPU | throughput | memory |
| Dataloader tuning | can be the entire bottleneck | none |
| 8-bit optimiser | ~6 bytes/param saved | negligible quality |
| Sequence packing (LLMs) | up to 2× on short sequences | implementation complexity |

Profile before optimising. If GPU utilisation is 40%, no kernel optimisation will
help — the input pipeline is the problem, and `torch.profiler` will show the gaps
between steps.

## Self-check

1. Explain Adam's bias correction: what goes wrong without it, and for how long?
2. Why is AdamW different from Adam plus L2, and which parameters are excluded
   from decay?
3. Why do transformers need warmup? Give three independent reasons.
4. You quadruple the batch size. What do you do to the learning rate, and why do
   two answers exist?
5. Why must you unscale before clipping under fp16?
6. What does an update-to-weight ratio of $10^{-1}$ tell you?
7. Describe the overfit-one-batch test and what each outcome rules out.

## Where to go next

- [Regularization & Normalization](./regularization-and-normalization.md) — the
  layers that make these schedules work.
- [Backpropagation & Autodiff](./backpropagation-and-autodiff.md) — where the
  gradients come from.
- [Activations & Initialization](./activations-and-initialization.md) — the
  starting conditions.
