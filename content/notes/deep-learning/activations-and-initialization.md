---
order: 3
description: Every activation function and what it fixes, the dying ReLU problem, Xavier and He initialization derived from variance analysis, and why initialization decides whether a deep network trains at all.
meta: Deep Learning · foundations
---

# Activations and Initialization

Two choices that look like details and are not. The activation function decides
whether gradients survive depth; the initialisation decides whether the forward
signal survives depth. Get either wrong and a network that is architecturally
perfect will not train.

## Why a non-linearity is required

Without one, a stack of affine maps collapses:

$$W_3(W_2(W_1\mathbf{x})) = (W_3W_2W_1)\mathbf{x} = W'\mathbf{x}$$

A hundred layers becomes one. Depth buys nothing. The activation is what makes
composition expressive.

## The activation functions

### Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

Squashes to $(0,1)$, so it reads as a probability. Historically dominant, now
used only in output layers and gates.

**Three fatal problems for hidden layers:**

1. **$\sigma' \le 0.25$ everywhere.** Ten stacked sigmoids shrink the gradient by
   at least $4^{-10}\approx10^{-6}$. Vanishing gradients, guaranteed by
   arithmetic.
2. **Saturation.** For $|x|>5$ the derivative is essentially zero. A saturated
   unit stops learning entirely.
3. **Not zero-centred.** Outputs are always positive, so all gradients with
   respect to a layer's weights share a sign, forcing a zig-zag optimisation
   path.

Still correct where it belongs: a binary classification output, and the gates
inside an LSTM or GRU, where "a number in $(0,1)$ that multiplies something" is
exactly what is wanted.

### Tanh

$$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}} = 2\sigma(2x)-1, \qquad \tanh'(x) = 1-\tanh^2(x)$$

Zero-centred, range $(-1,1)$, maximum derivative 1. Strictly better than sigmoid
for hidden layers, and still saturating. Used in LSTM cell candidates and in
small recurrent networks.

### ReLU

$$\mathrm{ReLU}(x) = \max(0,x), \qquad \mathrm{ReLU}'(x) = \mathbb{1}[x>0]$$

The change that made deep networks trainable.

| Advantage | Detail |
|---|---|
| No saturation for $x>0$ | derivative is exactly 1; gradients pass through undiminished |
| Trivially cheap | a comparison, not an exponential |
| Sparse activations | ~50% of units output zero at initialisation |
| Empirically converges much faster than tanh | the original AlexNet result |

| Problem | Detail |
|---|---|
| **Dying ReLU** | a unit whose pre-activation is negative for every input has zero gradient forever, and can never recover |
| Not zero-centred | outputs are non-negative |
| Non-differentiable at 0 | frameworks define $\mathrm{ReLU}'(0)=0$ by convention |
| Unbounded above | can produce very large activations |

**The dying ReLU deserves a precise account.** A large gradient step can push a
unit's bias so negative that $\mathbf{w}^\top\mathbf{x}+b < 0$ for the entire
data distribution. Then the output is 0, the gradient is 0, and no update will
ever change it. The unit is permanently dead. In badly configured networks —
usually too high a learning rate — 40% or more of units can die. Diagnose it by
logging the fraction of zero activations per layer.

### The ReLU family

| Function | Formula | Fixes |
|---|---|---|
| **Leaky ReLU** | $\max(\alpha x, x)$, $\alpha=0.01$ | non-zero gradient when negative — no dying units |
| **PReLU** | same, $\alpha$ learned | lets the network choose the slope |
| **ELU** | $x$ if $x>0$, else $\alpha(e^x-1)$ | smooth, negative saturation, closer to zero-centred |
| **SELU** | scaled ELU with specific constants | self-normalising: activations converge to zero mean, unit variance |
| **GELU** | $x\,\Phi(x)$ | smooth, non-monotonic; the transformer default |
| **SiLU / Swish** | $x\,\sigma(x)$ | smooth, non-monotonic; very similar to GELU |
| **Mish** | $x\tanh(\mathrm{softplus}(x))$ | smoother still; more expensive |
| **SwiGLU** | $\mathrm{Swish}(xW)\odot(xV)$ | gated; used in Llama, PaLM, most modern LLMs |
| **Softplus** | $\log(1+e^x)$ | smooth ReLU; rarely worth the cost |
| **Maxout** | $\max_k(\mathbf{w}_k^\top\mathbf{x}+b_k)$ | learns the activation; multiplies parameters |

**GELU** is the default in transformers:

$$\mathrm{GELU}(x) = x\,\Phi(x) \approx 0.5x\left(1+\tanh\left[\sqrt{2/\pi}\left(x+0.044715x^3\right)\right]\right)$$

Interpret it as a **stochastic regulariser made deterministic**: it multiplies
the input by the probability that a standard normal is below it, so it is a
smooth, input-dependent gate rather than a hard threshold. The small negative dip
around $x \approx -0.75$ is not a defect — it gives the function a non-monotonic
region that appears to help expressivity.

**SwiGLU** is the current frontier-model choice. It splits the FFN's up-
projection into two halves and gates one by the other:

$$\mathrm{FFN}(x) = \bigl(\mathrm{Swish}(xW_1)\odot xW_3\bigr)W_2$$

Because it uses three matrices instead of two, implementations shrink the hidden
dimension to $\frac{2}{3}\cdot 4d$ to keep the parameter count matched. The
gating consistently buys a small but real quality gain at equal parameters.

### Output activations

| Task | Activation | Loss |
|---|---|---|
| Binary | sigmoid (fused into the loss) | `BCEWithLogitsLoss` |
| Multiclass exclusive | softmax (fused) | `CrossEntropyLoss` |
| Multilabel | sigmoid per label (fused) | `BCEWithLogitsLoss` |
| Regression | **none** | MSE / Huber |
| Positive regression | softplus or exp | MSE on the log, or Poisson NLL |
| Bounded regression | sigmoid or tanh, scaled | MSE |

**Leave the final layer linear and let the loss apply the non-linearity.** The
fused kernels are numerically stable and faster, and this is the single most
common source of "my model trains but badly".

### Choosing

| Situation | Use |
|---|---|
| Default for a new MLP or CNN | ReLU |
| Transformers | GELU, or SwiGLU in the FFN |
| Many dead units observed | LeakyReLU or GELU |
| Very deep network without normalisation | SELU (with LeCun init and AlphaDropout) |
| Recurrent gates | sigmoid (gates) and tanh (candidates) |
| Need a smooth, differentiable everywhere | GELU, SiLU, ELU |
| Extreme efficiency (edge, quantised) | ReLU or ReLU6 — quantises cleanly |

The honest summary: **ReLU and GELU cover almost everything.** The differences
between the modern smooth activations are small (fractions of a percent), and
architecture, data, and schedule matter far more. Do not spend a week on this
choice.

## Initialization

### Why not zeros

If every weight is zero, every neuron in a layer computes the same thing,
receives the same gradient, and updates identically. The layer collapses to a
single unit and never recovers. This is the **symmetry breaking** problem, and
it is why weights must be random.

**Biases can be zero** — the weights already break symmetry.

### Why not "small random"

The naive fix, $\mathcal{N}(0, 0.01^2)$, works for shallow networks and fails for
deep ones. Track the variance of activations through layers:

- Weights too small → activations shrink geometrically → by layer 20 the signal
  is numerically zero → gradients vanish.
- Weights too large → activations grow geometrically → saturation or overflow →
  gradients explode.

The goal is to **keep the variance of activations roughly constant through
depth**, in both directions.

### The variance derivation

For $z = \sum_{j=1}^{n_{in}} w_j x_j$ with independent, zero-mean $w$ and $x$:

$$\mathrm{Var}(z) = n_{in}\,\mathrm{Var}(w)\,\mathrm{Var}(x)$$

To preserve variance ($\mathrm{Var}(z) = \mathrm{Var}(x)$) we need

$$\mathrm{Var}(w) = \frac{1}{n_{in}}$$

That is the forward-pass condition. The backward pass, by the same argument
applied to $\bar{\mathbf{x}} = \bar{\mathbf{z}}W^\top$, wants
$\mathrm{Var}(w) = 1/n_{out}$.

**Xavier/Glorot initialisation** splits the difference:

$$\mathrm{Var}(w) = \frac{2}{n_{in}+n_{out}}$$

This assumes the activation is roughly linear near zero and symmetric — true for
tanh, false for ReLU.

**He/Kaiming initialisation** corrects for ReLU. ReLU zeroes half the inputs, so
it halves the variance:

$$\mathrm{Var}(w) = \frac{2}{n_{in}}$$

The factor of 2 compensates exactly for that halving. Using Xavier with ReLU in a
30-layer network causes activations to decay by $2^{-15}$ — an entirely
predictable failure.

### The table

| Scheme | Variance | Use with |
|---|---|---|
| **He / Kaiming normal** | $2/n_{in}$ | ReLU, LeakyReLU, GELU, SiLU |
| **Xavier / Glorot** | $2/(n_{in}+n_{out})$ | tanh, sigmoid, linear |
| **LeCun** | $1/n_{in}$ | SELU (required for self-normalisation) |
| **Orthogonal** | orthonormal columns | RNNs, very deep networks |
| **Truncated normal, std 0.02** | fixed | transformers (GPT/BERT convention) |
| **Zeros** | — | biases, and the last layer of a residual block |
| **Identity/near-identity** | — | recurrent state matrices |

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
```

### Initialisation tricks that matter in practice

**Zero-init the last layer of each residual block.** If the block's final
convolution or its normalisation gain starts at zero, the block initially
computes the identity: $\mathbf{h} = \mathbf{h} + 0$. The network begins as a
shallow one and deepens as training progresses. This measurably stabilises very
deep ResNets and transformers, and it is nearly free.

**Scale residual-branch outputs by $1/\sqrt{2L}$** (GPT-2's convention) so that
the variance added by $L$ residual branches does not accumulate through depth.

**Initialise forget-gate biases to 1** in an LSTM. It starts the cell in a
"remember by default" state, which substantially improves learning of long
dependencies.

**Set the output-layer bias to the base rate.** For a classifier on a 1%-positive
problem, initialising the output bias to $\log(0.01/0.99) \approx -4.6$ means the
model starts by predicting the correct prior instead of spending its first
hundred steps discovering it. This is a genuinely useful trick on imbalanced
problems.

**Embedding layers** use $\mathcal{N}(0, 0.02^2)$ by convention in transformers,
not He — an embedding lookup is not a matmul over a fan-in, so the fan-based
reasoning does not apply.

## Initialization and normalization interact

Batch and layer normalisation renormalise activations at every layer, which makes
networks far less sensitive to initialisation. That is a real part of why they
are used.

But **initialisation still matters** even with normalisation: a badly scaled
start produces large early gradients, which interacts badly with adaptive
optimisers whose second-moment estimates are based on those first few steps. This
is one reason learning-rate warmup exists.

For transformers specifically, the interaction is well studied:

| Setup | Behaviour |
|---|---|
| Post-LN (original Transformer) | needs warmup; gradients at the last layer are much larger at init |
| **Pre-LN** (modern default) | far more stable; trains without warmup, though warmup still helps |
| RMSNorm | cheaper than LayerNorm, no re-centring; used in Llama-family models |
| DeepNorm / scaled residuals | enables 1000-layer transformers |

## Diagnosing initialisation problems

Print activation and gradient statistics per layer on the very first batch.

```python
acts, grads = {}, {}
for name, mod in model.named_modules():
    if isinstance(mod, (nn.Linear, nn.Conv2d)):
        mod.register_forward_hook(
            lambda m, i, o, n=name: acts.__setitem__(n, (o.mean().item(), o.std().item())))
```

| Observation | Diagnosis | Fix |
|---|---|---|
| Activation std shrinks toward zero with depth | init too small, or Xavier with ReLU | He init |
| Activation std grows with depth | init too large | He init, or scale residual branches |
| >30% of ReLU outputs are exactly zero after a few steps | dying ReLU | LeakyReLU/GELU, lower LR |
| Tanh/sigmoid activations pinned at ±1 | saturation | rescale init, add normalisation |
| Loss is exactly $\log K$ and does not move | output layer saturated or LR ~0 | check the output bias and the LR |
| Gradient norms differ by $10^6$ across layers | init or architecture problem | per-layer norms, add normalisation |

**A well-initialised classifier starts at loss $\approx \log K$** — the entropy
of a uniform prediction over $K$ classes. For 10 classes that is 2.303. If your
initial loss is far from that, something is wrong before training has begun, and
that is a two-second check worth doing every time.

## Self-check

1. Why can weights not be initialised to zero, but biases can?
2. Derive $\mathrm{Var}(w) = 1/n_{in}$ from the variance of a weighted sum.
3. Why does He initialisation have a factor of 2 that Xavier does not?
4. Explain the dying ReLU precisely: what makes it permanent?
5. Your 10-class classifier starts at loss 7.0. What does that tell you?
6. Why is zero-initialising the last layer of a residual block helpful?
7. What is the practical reason to set the output bias to the log-odds of the
   base rate?

## Where to go next

- [Backpropagation & Autodiff](./backpropagation-and-autodiff.md) — the gradient
  flow these choices protect.
- [Regularization & Normalization](./regularization-and-normalization.md) — the
  layers that make initialisation less fragile.
- [Optimization & Training](./optimization-and-training.md) — learning rates,
  schedules, and warmup.
