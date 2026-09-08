---
order: 3
description: Every activation function and what it fixes, the dying ReLU problem, Xavier and He initialization derived from variance analysis, and why initialization decides whether a deep network trains at all.
meta: Deep Learning · foundations
---

# Activations and Initialization

Activations and initialization jointly determine the scale, geometry and local derivatives of a network at the start of training. Neither acts alone: width, residual paths, normalization and optimizer settings also affect signal transport. This chapter derives the useful approximations, tests them on finite networks and explains when they fail.

Prerequisites are [probability and moments](../math/probability.md), [matrix norms](../math/linear-algebra.md) and [backpropagation](./backpropagation-and-autodiff.md). Distinguish a tensor's mean, centered variance and uncentered second moment throughout; they are not interchangeable.

## Why a non-linearity is required

Without one, a stack of affine maps collapses:

$$W_3(W_2(W_1\mathbf{x})) = (W_3W_2W_1)\mathbf{x} = W'\mathbf{x}$$

With biases included the result is affine. Depth adds no nonlinear expressivity, although a linear factorization can alter rank constraints and optimization. The activation is what makes
composition expressive.

## The activation functions

### Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

Squashes to $(0,1)$, so it reads as a probability. Historically common in hidden layers, it remains especially useful in output parameterizations and gates.

**Three considerations for hidden layers:**

1. **$\sigma' \le 0.25$ everywhere.** The activation-only derivative product is at most $4^{-10}$ over ten layers; the complete Jacobian also includes weights. This encourages contraction but does not prove every sigmoid network vanishes.
2. **Saturation.** For $|x|>5$ the derivative is essentially zero. A saturated unit can learn very slowly, and finite precision may round an already small derivative to zero.
3. **Not zero-centred.** For one example and one output unit, positive inputs make incoming-weight gradient signs follow the same upstream scalar. Batch summation, different units and later layers invalidate a blanket same-sign claim; centering can still improve conditioning.

Still correct where it belongs: a binary classification output, and the gates
inside an LSTM or GRU, where "a number in $(0,1)$ that multiplies something" is
exactly what is wanted.

### Tanh

$$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}} = 2\sigma(2x)-1, \qquad \tanh'(x) = 1-\tanh^2(x)$$

Zero-centred, range $(-1,1)$, maximum derivative 1. Often easier to center than sigmoid for hidden layers, but still saturating and not universally better. Used in LSTM cell candidates and in
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
| **Dying ReLU** | a unit inactive on all observed inputs gets no task gradient through this activation |
| Not zero-centred | outputs are non-negative |
| Non-differentiable at 0 | frameworks define $\mathrm{ReLU}'(0)=0$ by convention |
| Unbounded above | can produce very large activations |

**The dying ReLU deserves a precise account.** A large gradient step can push a
unit's bias so negative that $\mathbf{w}^\top\mathbf{x}+b < 0$ for the entire
data distribution. Then its output and local task gradient are zero. With fixed inputs, plain gradient descent and no other update sources, it cannot recover through that path. Upstream representation changes, momentum, regularization or new examples may reactivate it. Diagnose persistent per-unit inactivity across representative examples, not the overall fraction of zeros.

### The ReLU family

| Function | Formula | Fixes |
|---|---|---|
| **Leaky ReLU** | $\max(\alpha x, x)$, $\alpha=0.01$ | nonzero negative-side derivative when alpha is positive |
| **PReLU** | same, $\alpha$ learned | lets the network choose the slope |
| **ELU** | $x$ if $x>0$, else $\alpha(e^x-1)$ | negative saturation; differentiable at zero only when alpha=1 |
| **SELU** | scaled ELU with specific constants | self-normalizing under architectural and initialization assumptions |
| **GELU** | $x\,\Phi(x)$ | smooth, non-monotonic; the transformer default |
| **SiLU / Swish** | $x\,\sigma(x)$ | smooth, non-monotonic; very similar to GELU |
| **Mish** | $x\tanh(\mathrm{softplus}(x))$ | smoother still; more expensive |
| **SwiGLU** | $\mathrm{Swish}(xW)\odot(xV)$ | gated; used in Llama, PaLM, most modern LLMs |
| **Softplus** | $\log(1+e^x)$ | smooth positive map, useful for positive parameterizations |
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
gating improved quality in the cited matched experiments; it is not a universal
guarantee across datasets, training budgets or implementations.

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
| Need a smooth, differentiable everywhere | GELU, SiLU, or ELU with alpha=1 |
| Extreme efficiency (edge, quantised) | ReLU or ReLU6 — quantises cleanly |

Use an activation compatible with the architecture and training recipe, then change it in a controlled comparison. A pretrained model's activation is part of its function: replacing it without retraining is not an innocuous inference optimization.

## Initialization

### Why not zeros

If every weight is zero, every neuron in a layer computes the same thing,
receives the same gradient, and updates identically. The layer collapses to a
single unit and never recovers. This is the **symmetry breaking** problem, and
it is why hidden units need symmetry breaking. Randomness is convenient, not logically necessary; a deterministic asymmetric construction can also work.

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

**He/Kaiming initialization** tracks the second moment through ReLU. If $z$ has a symmetric distribution, $E[\operatorname{ReLU}(z)^2]=E[z^2]/2$. It does not halve centered variance. Independent zero-mean weights make the next preactivation variance depend on this uncentered input moment, leading to

$\operatorname{Var}(w)=\frac{2}{n_{in}}.$

For equal-width layers, the idealized second-moment recurrence under Xavier instead multiplies by about $1/2$ per ReLU layer; RMS magnitude multiplies by about $1/\sqrt2$. Finite width, correlations, biases and normalization can change that prediction.

### The table

| Scheme | Variance | Use with |
|---|---|---|
| **He / Kaiming normal** | $2/n_{in}$ for ReLU | use the slope-adjusted gain for LeakyReLU; smooth gates need their own analysis |
| **Xavier / Glorot** | $2/(n_{in}+n_{out})$ | tanh, sigmoid, linear |
| **LeCun** | $1/n_{in}$ | SELU (required for self-normalisation) |
| **Orthogonal** | orthonormal rows or columns depending on shape, times gain | recurrent/deep networks; nonlinear masks still change singular values |
| **Truncated normal, std 0.02** | fixed | transformers (GPT/BERT convention) |
| **Zeros** | — | biases, and the last layer of a residual block |
| **Identity/near-identity** | — | recurrent state matrices |

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

### Initialisation tricks that matter in practice

**Zero-init the last layer of each residual block.** If the block's final
convolution or its normalisation gain starts at zero, the block initially
computes the identity: $\mathbf{h} = \mathbf{h} + 0$. The block begins at identity when the skip is identity. The final zero factor initially blocks gradients to some earlier branch parameters, so the exact placement matters. This measurably stabilises very
deep ResNets and transformers, and it is nearly free.

**Scale selected residual output-projection initialization by $1/\sqrt{2L}$** in a GPT-2-style two-branch-per-block recipe. This controls approximate accumulated branch variance; it is not a universal instruction to rescale every activation.

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
| Post-LN (original Transformer) | used warmup in the original recipe; large final-layer gradients at initialization can motivate it, depending on parameterization |
| **Pre-LN** | often easier to optimize; warmup requirements still depend on scale and recipe |
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
| Many units are inactive for every representative example over time | potentially dying ReLU | inspect per-unit maxima and upstream changes; test LR or activation |
| Tanh near -1/+1, sigmoid near 0/1 | saturation | inspect preactivations, input scale and initialization |
| Loss remains near $\log K$ | near-uniform predictions or averaging effects | inspect logits, labels, gradients and actual updates |
| Gradient norms differ by $10^6$ across layers | init or architecture problem | per-layer norms, add normalisation |

Uniform logits give loss $\log K$; for ten classes this is about 2.303. This is a useful reference, not a mandatory initial loss. Random nonuniform logits can produce larger expected cross-entropy; class-prior biases intentionally produce a different baseline. Inspect the predicted distribution before diagnosing a bug.

## Moments, gains, and finite-width effects

### A Gaussian ReLU calculation

Let $z\sim\mathcal N(0,q)$ and $h=\max(0,z)$. Symmetry gives

$$E[h^2]=q/2,\qquad E[h]=\sqrt{q/(2\pi)},\qquad
\operatorname{Var}(h)=q\left(\frac12-\frac1{2\pi}\right).$$

For $q=1$, the mean is about $0.3989$, second moment $0.5$, and centered
variance $0.3408$. ReLU outputs are not zero-centered. Saying that it halves
variance loses the mean-square contribution and obscures the actual argument
behind Kaiming initialization.

Now let $z_i=\sum_{j=1}^n w_{ji}h_j$, with independent zero-mean weights of
variance $\sigma_w^2$, independent of the inputs. Averaging over these random
weights makes cross terms vanish:

$$E[z_i^2]=n\sigma_w^2E[h_j^2].$$

The previous activations do not need zero mean for this step. Choosing
$\sigma_w^2=2/n$ compensates for the ReLU second-moment factor. In a single
finite sampled network, unit means and correlations fluctuate; this ensemble
calculation is an approximation to typical propagation, not an exact identity
for every realized layer.

For LeakyReLU with negative slope $a$, symmetry gives

$$E[\phi(z)^2]=\frac{1+a^2}{2}E[z^2],\qquad
\sigma_w^2=\frac{2}{(1+a^2)n}.$$

At $a=0.2$ and fan-in $100$, variance is $2/104\approx0.01923$, standard
deviation about $0.13868$. Using variance as the normal distribution's standard
deviation would shrink scale drastically. Check whether an API expects `std`,
variance or a uniform bound before translating a formula.

### Forward and backward objectives need not coincide

For a linear layer, fan-in preservation asks for variance $1/n_{in}$ while
backward preservation asks for $1/n_{out}$. Rectangular layers cannot satisfy
both exactly unless the widths agree. Xavier's $2/(n_{in}+n_{out})$ is a
compromise, not an equality satisfying both equations. Gains account for the
chosen nonlinearity under a specified approximation.

A uniform variable on $[-a,a]$ has variance $a^2/3$. Consequently Xavier uniform
uses $a=\sqrt{6/(n_{in}+n_{out})}$ before an optional gain; ReLU fan-in uniform
uses $a=\sqrt{6/n_{in}}$. Normal and uniform variants can match second moments
while differing in tails, which matters for extreme activations and quantization.

For GELU or SiLU, $E[\phi(\sqrt q Z)^2]$ and
$E[\phi'(\sqrt q Z)^2]$ depend on the operating variance $q$. A ReLU gain is a
reasonable recipe in some networks, not an exact preservation theorem for all
smooth activations. Gated FFNs multiply two projections, introducing additional
moments and correlations; simply assigning one scalar gain to the gate cannot
fully characterize their dynamics.

### Fan axes and parameter conventions

PyTorch linear weights are `(out_features,in_features)`. Applying a fan-based
initializer to a custom matrix intended for `x @ W` requires attention to that
transpose convention. A grouped convolution stores
`(C_out,C_in/groups,kH,kW)`, so a filter's fan-in is
$(C_{in}/g)k_Hk_W$, not $C_{in}k_Hk_W$. Verify the library's fan-out convention
when requesting backward preservation for grouped operators.

Rectangular orthogonal initialization can make columns orthonormal when there
are enough rows, or rows orthonormal in the opposite case. It cannot make both
$W^\top W$ and $WW^\top$ identities for a nonsquare matrix. Even a square
orthogonal weight followed by a ReLU mask has rank loss whenever coordinates are
inactive. Scalar variance preservation is therefore weaker than preserving all
singular values of the input-output Jacobian, sometimes called dynamical isometry.

### Symmetry-breaking exceptions

Initializing every hidden unit identically creates an invariant symmetry: equal
units receive equal updates on the same data. Random asymmetric weights break
it conveniently; deterministic distinct directions also can. Zero biases do
not reintroduce the symmetry when weight rows already differ.

Zeroing only a final residual projection is different from zeroing every layer.
For $F(x)=W_2\phi(W_1x)$ with $W_2=0$, the block output initially equals its
skip input. $W_2$ can receive a nonzero gradient because $\phi(W_1x)$ is not
zero, while $W_1$ initially receives zero gradient through that product. After
$W_2$ changes, earlier parameters can learn. If both matrices and hidden features
are zero, the same escape path may be blocked.

Class-prior bias initialization is another deliberate asymmetry. A Bernoulli
prior $\pi$ corresponds to bias $\log(\pi/(1-\pi))$ when initial feature logits
are near zero. For $\pi=0.01$, bias is approximately $-4.5951$ and expected
cross-entropy of the prior predictor is its entropy, about $0.0560$ nats, not
$\log2$. Estimate that prior only from training labels and reconsider it if
sampling or class weighting changes the effective training distribution.

## Experiment: measure signal rather than trusting a label

The experiment measures Gaussian moments, compares depth propagation for three
weight scales, records input-gradient norms, and diagnoses persistent inactivity
separately from ordinary ReLU sparsity. It uses fixed shapes and finite samples:
the numerical tolerances test the intended phenomenon without claiming exact
ensemble identities for every random seed.

```python runnable
import math
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(41)
z = torch.randn(200000, dtype=torch.float64)
h = z.relu()
expected_mean = 1 / math.sqrt(2 * math.pi)
expected_variance = 0.5 - 1 / (2 * math.pi)
assert abs(h.mean().item() - expected_mean) < 0.006
assert abs(h.square().mean().item() - 0.5) < 0.009
assert abs(h.var(unbiased=False).item() - expected_variance) < 0.009
print("ReLU mean, variance, second moment:", h.mean().item(),
      h.var(unbiased=False).item(), h.square().mean().item())

def propagation(weight_variance_factor):
    torch.manual_seed(42)
    width, depth = 128, 18
    layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth)])
    for layer in layers:
        nn.init.normal_(layer.weight, std=math.sqrt(weight_variance_factor / width))
    x = torch.randn(96, width, requires_grad=True)
    value = x
    records = []
    for index, layer in enumerate(layers):
        value = layer(value).relu()
        records.append((index + 1, value.mean().item(),
                        value.var(unbiased=False).item(),
                        value.square().mean().item(),
                        (value == 0).float().mean().item()))
    value.sum().backward()
    return records, x.grad.norm().item()

small, small_gradient = propagation(0.2)
xavier, xavier_gradient = propagation(1.0)
he, he_gradient = propagation(2.0)
for name, records, gradient in (("small", small, small_gradient),
                                 ("Xavier", xavier, xavier_gradient),
                                 ("He", he, he_gradient)):
    print(name, "last layer (depth,mean,var,second moment,zero fraction)",
          records[-1], "input gradient norm", gradient)
assert small[-1][3] < xavier[-1][3] * 1e-8
assert he[-1][3] > xavier[-1][3] * 1000
assert all(math.isfinite(row[3]) for row in he)

torch.manual_seed(43)
features = torch.randn(2000, 8)
healthy = nn.Linear(8, 16)
nn.init.zeros_(healthy.bias)
with torch.no_grad():
    activations = healthy(features).relu()
    ordinary_zero_fraction = (activations == 0).float().mean().item()
    inactive_units = (activations.amax(dim=0) == 0).sum().item()
    unhealthy = nn.Linear(8, 16)
    unhealthy.weight.copy_(healthy.weight)
    unhealthy.bias.fill_(-100)
    dead_count = (unhealthy(features).relu().amax(dim=0) == 0).sum().item()
assert 0.4 < ordinary_zero_fraction < 0.6
assert inactive_units == 0 and dead_count == 16
print("Healthy zero fraction:", ordinary_zero_fraction,
      "persistently inactive:", inactive_units, "constructed inactive:", dead_count)
```

The constructed inactive layer is inactive on this finite input sample, not a
proof about every point of an unbounded Gaussian distribution. In production,
aggregate per-unit maxima or activation frequencies across representative batches
and over time. Fifty percent zero entries is compatible with every neuron being
useful on some observations.

For visual inspection, plot the measured second moment against depth on a log
axis, and plot activation value and derivative against preactivation on common
axes. The curves distinguish saturation from scale drift: sigmoid derivatives
approach zero at either extreme, ReLU has a zero negative branch, and SiLU/GELU
have smooth nonmonotonic negative regions. Do not infer derivative behavior from
an activation curve alone when a tiny slope matters numerically.

## Initialization in a complete training decision

Start by measuring inputs: large feature-unit differences can undo a carefully
chosen weight scale. Standardize continuous tabular features using training-only
statistics, use preprocessing matching a pretrained image model, and inspect
embedding magnitudes rather than applying a dense fan-in formula to lookup tables.

Then measure preactivation mean/std, activation second moment, saturation or
inactivity rates and gradient norms on the first few batches. Hooks should be
removed when diagnostics finish; detach scalar statistics rather than storing
entire graph-connected tensors. Distinguish a healthy but narrow bottleneck from
unintended collapse by checking the representation covariance rank.

Normalization can reduce sensitivity to input scale but cannot restore
information already destroyed by a saturated or rank-deficient transformation.
Its epsilon breaks exact scale invariance near zero, and learned gains, optimizer
moments and residual placement still affect gradients. Pre-norm often makes
transformers easier to optimize, but whether warmup can be omitted must be
validated for the particular size and initialization, not inferred from the name.

SELU's self-normalizing analysis assumes a compatible feedforward architecture,
initialization and moment regime. Ordinary dropout changes those moments; use
AlphaDropout when following that recipe. Arbitrary residual branches,
normalization layers or correlated inputs can move the network outside the
analysis. For LSTMs, a positive forget-gate bias initially favors retention,
but library implementations may split biases across input and recurrent terms;
the effective summed bias is what controls the gate.

Finally compare alternatives with the same data order and reasonable optimizer
tuning. A larger activation variance can look beneficial merely because it
changes effective update scale. Record the initial function, not only the
initializer's name, and retain the architecture's intended residual scaling when
loading a pretrained checkpoint.

### Smoothness, clipping, and bounded outputs

Smooth activations are useful when the objective itself differentiates model
outputs, as in derivative matching or some physics-informed losses. ReLU's
second derivative is zero almost everywhere and undefined at its kink; a model
can represent a piecewise-linear function accurately yet be a poor choice for
matching a smooth second derivative. Conversely, a smooth activation does not
automatically make its numerical derivatives well-conditioned at saturation.

ReLU6 clips at six as well as zero. That bounded range can simplify activation
quantization, but values beyond six have zero local derivative and the clipping
threshold becomes part of the learned function. Hard-sigmoid or hard-swish
approximations also trade smoothness for implementation properties. Replacing a
trained smooth gate with a hard approximation changes outputs; evaluate or
retrain rather than assuming mathematical equivalence.

For a positive scale, `softplus(raw) + epsilon` can prevent an invalid negative
parameter while avoiding the extreme growth of `exp(raw)`. But it introduces a
minimum scale and a particular derivative profile. Choose that minimum from the
modeled quantity's units and numerical requirements, not a universal epsilon
copied between unrelated tasks.

## Self-check

1. **Why can biases be zero?** Distinct weight vectors already break hidden-unit
   symmetry. Identical weights and biases preserve equal-unit dynamics, while
   deterministic asymmetric weights can break the symmetry without randomness.
2. **Derive fan-in scaling.** With independent zero-mean weights,
   $E[z_i^2]=n\sigma_w^2E[x_j^2]$. Preserving a linear signal asks for
   $\sigma_w^2=1/n$; a ReLU before the next map contributes a factor one-half
   to the second moment under symmetry.
3. **Does ReLU halve variance?** No. For standard Gaussian input, second moment
   is $0.5$, mean is $1/\sqrt{2\pi}$, and variance is
   $0.5-1/(2\pi)\approx0.3408$. The He derivation tracks the first of these.
4. **What makes a ReLU locally dead?** All relevant observed preactivations are
   negative, so task gradients through that unit are zero. It remains dead under
   fixed inputs and updates solely from that gradient, but momentum or changing
   upstream features can reactivate it.
5. **A ten-class classifier starts at loss seven. Is it broken?** It is much worse
   than uniform predictions, so inspect logit scale, label indices and class-prior
   assumptions. Random overconfident logits can cause it without a software bug.
6. **Why zero a residual's final projection?** The residual starts at zero while
   its nonzero hidden features allow the final projection to learn. Earlier
   branch layers can begin receiving gradients once that projection moves.
7. **Why use prior log-odds?** With nearly zero feature contribution, sigmoid of
   the bias equals the training prior. For one percent positives, the bias is
   approximately $-4.595$, avoiding an initially fifty-percent positive predictor.
8. **Compute LeakyReLU scale.** For slope $0.2$ and fan-in $100$, variance is
   $2/(1.04\cdot100)$ and standard deviation about $0.13868$. A normal sampler
   expects the standard deviation, not that variance.
9. **Are fifty-percent zeros a failure?** Not by themselves. A healthy neuron can
   activate on half the examples. Count neurons inactive across the dataset and
   inspect the pattern over training, not just the aggregate zero fraction.
10. **Does an orthogonal matrix guarantee gradient preservation?** Only for the
    appropriate linear map and subspace before nonlinearities. ReLU masks,
    rectangular bottlenecks and residual sums change the complete Jacobian.

## Where to go next

- [Backpropagation & Autodiff](./backpropagation-and-autodiff.md) — the gradient
  flow these choices protect.
- [Regularization & Normalization](./regularization-and-normalization.md) — the
  layers that make initialisation less fragile.
- [Optimization & Training](./optimization-and-training.md) — learning rates,
  schedules, and warmup.

Primary references: [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html),
[He et al., rectifier initialization](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html),
[self-normalizing networks](https://arxiv.org/abs/1706.02515),
[GLU variants](https://arxiv.org/abs/2002.05202), and
[PyTorch initialization conventions](https://docs.pytorch.org/docs/stable/nn.init.html).
