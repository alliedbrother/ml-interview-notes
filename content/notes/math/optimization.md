---
order: 5
description: Convexity, gradient descent and its variants, momentum, Adam, learning-rate schedules, constrained and second-order methods — why each exists and when each fails.
meta: Math for ML · core
---

# Optimization: Actually Finding the Minimum

Training is optimisation. You have a loss surface defined over hundreds of
millions of dimensions, no ability to see it, and a budget of a few hundred
thousand steps. Every optimiser is a different answer to the same question: given
only local information, where should I step next?

This page builds that answer from convex analysis, through the whole family of
first-order methods, to the practical recipes that actually train models.

```mermaid
flowchart TD
    P["optimisation problem<br/>min f of x subject to constraints"] --> Q{"convex?"}
    Q -->|"yes"| CV["global optimum guaranteed<br/>LP / QP / SOCP solvers,<br/>closed forms exist"]
    Q -->|"no"| NC["local methods only"]
    NC --> D{"gradient available?"}
    D -->|"no"| ZO["derivative-free:<br/>random search, CMA-ES,<br/>Bayesian optimisation"]
    D -->|"yes"| FO{"data size?"}
    FO -->|"small, full batch"| BATCH["gradient descent,<br/>L-BFGS, Newton"]
    FO -->|"large"| SGD["stochastic methods:<br/>SGD, momentum, Adam"]
```

## The problem, stated

$$\min_{\theta \in \mathbb{R}^d} f(\theta) \quad \text{subject to} \quad g_i(\theta) \le 0, \; h_j(\theta) = 0$$

In supervised learning $f$ is the empirical risk

$$f(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(f_\theta(x_i), y_i) + \Omega(\theta)$$

and this is worth pausing on. The thing you actually want to minimise is the
**true risk** $\mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell]$, which you cannot
evaluate. Empirical risk minimisation substitutes the sample average. **Every
generalisation problem in ML is the gap between those two objectives**, and no
amount of optimiser cleverness closes it — a better optimiser can make it worse
by fitting the sample more exactly.

## Convexity, and why it is the dividing line

$f$ is convex if for all $x, y$ and $\lambda \in [0,1]$:

$$f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$$

Equivalent characterisations for differentiable $f$:

- **First order**: $f(y) \ge f(x) + \nabla f(x)^\top (y - x)$ — the tangent plane
  lies below the function everywhere.
- **Second order**: $\nabla^2 f \succeq 0$ — the Hessian is positive
  semi-definite everywhere.

**The payoff**: for convex $f$, every local minimum is global, and
$\nabla f(x^\star)=0$ is sufficient for optimality. For non-convex $f$, neither
holds.

| Model | Convex in its parameters? |
|---|---|
| Linear/ridge/lasso regression | yes |
| Logistic regression | yes |
| Linear SVM (hinge loss) | yes, non-smooth |
| PCA | non-convex but solvable exactly via SVD |
| $k$-means | non-convex; Lloyd's algorithm is a local method |
| Matrix factorisation | non-convex, benign landscape |
| Any neural network with a hidden layer | no |

Two operations preserve convexity and are worth knowing because they let you
*prove* a new objective is convex: non-negative weighted sums, and composition
with an affine map. $\|Ax-b\|^2 + \lambda\|x\|_1$ is convex because both terms
are, and $\|\cdot\|$ composed with the affine $x \mapsto Ax-b$ stays convex.

### Strong convexity and smoothness

Two constants govern how fast first-order methods converge.

- **$L$-smooth**: $\|\nabla f(x)-\nabla f(y)\| \le L\|x-y\|$. Gradients do not
  change too fast; equivalently $\nabla^2 f \preceq LI$.
- **$\mu$-strongly convex**: $f(y) \ge f(x)+\nabla f(x)^\top(y-x)+\frac{\mu}{2}\|y-x\|^2$;
  equivalently $\nabla^2 f \succeq \mu I$.

The **condition number** $\kappa = L/\mu$ decides everything:

| Assumption | Gradient descent rate | Steps to $\epsilon$ |
|---|---|---|
| Convex, $L$-smooth | $O(1/k)$ | $O(1/\epsilon)$ |
| Strongly convex, $L$-smooth | linear, $\left(\frac{\kappa-1}{\kappa+1}\right)^{k}$ | $O(\kappa\log\frac1\epsilon)$ |
| + Nesterov momentum | accelerated | $O(\sqrt{\kappa}\log\frac1\epsilon)$ |

Momentum turns $\kappa$ into $\sqrt{\kappa}$. For $\kappa = 10^4$ that is 100×
fewer iterations. This is not a heuristic — it is a proven optimal rate for
first-order methods, and it is why every serious optimiser has a momentum term.

## Gradient descent

$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$$

### The learning rate is a stability question

For a quadratic $f(\theta) = \frac12\theta^\top H\theta$, the update is
$\theta_{t+1} = (I - \eta H)\theta_t$. Along the eigenvector with eigenvalue
$\lambda_i$ the iterate is multiplied by $(1 - \eta\lambda_i)$ each step. This
converges iff

$$|1-\eta\lambda_i| < 1 \quad\text{for all } i \quad\Longleftrightarrow\quad 0 < \eta < \frac{2}{\lambda_{\max}}$$

So **the largest stable learning rate is set by the sharpest direction, while
progress is limited by the flattest.** That single sentence explains ill
conditioning, why loss explodes past a threshold, and why normalisation layers
(which shrink $\lambda_{\max}$) let you use bigger learning rates.

| $\eta$ relative to $2/\lambda_{\max}$ | Behaviour |
|---|---|
| far below | converges, slowly, monotonically |
| near $1/\lambda_{\max}$ | fastest for that direction |
| between $1/\lambda_{\max}$ and $2/\lambda_{\max}$ | converges while oscillating |
| above $2/\lambda_{\max}$ | diverges — loss goes to `inf` or `NaN` |

### Batch, stochastic, and mini-batch

| Variant | Gradient per step | Cost/step | Noise | Notes |
|---|---|---|---|---|
| Batch GD | all $N$ examples | $O(N)$ | none | smooth, exact, unusable at scale |
| SGD | 1 example | $O(1)$ | high | noisy, escapes saddles, poor hardware use |
| Mini-batch | $B$ examples | $O(B)$ | $\propto 1/\sqrt{B}$ | the actual default |

Mini-batch wins for two independent reasons. Statistically, the gradient
estimate's standard error falls as $1/\sqrt{B}$ — so going from $B=32$ to
$B=512$ (16× the compute) only halves the noise, a poor trade past a point.
Computationally, GPUs are matrix-multiply engines: $B=1$ leaves them idle, and
throughput rises steeply up to a saturation point.

**Gradient noise is not purely a cost.** It helps escape saddle points and sharp
minima, and there is good evidence that the flat minima SGD prefers generalise
better. This is why simply cranking the batch size up often *hurts* test
accuracy unless you compensate.

**Scaling rules for large batches** (both are empirical, both work):

- **Linear scaling**: multiply $\eta$ by $k$ when multiplying $B$ by $k$, with a
  warmup of a few epochs. Works up to $B \approx 8$k for ImageNet-scale work.
- **Square-root scaling**: $\eta \propto \sqrt{B}$, better motivated for
  Adam-family optimisers where the update is normalised.

Warmup exists because at initialisation the gradient direction is nearly random
and the curvature estimate in adaptive methods is based on almost no data; a
large step then is actively destructive.

## Momentum

Plain gradient descent in a ravine bounces across the steep walls and creeps
along the floor. Momentum accumulates a velocity that cancels the oscillation and
compounds the consistent direction.

$$v_{t+1} = \beta v_t + \nabla f(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

With $\beta = 0.9$, a persistent gradient reaches an effective step of
$\eta/(1-\beta) = 10\eta$ — a 10× speedup in consistent directions, while
alternating-sign components cancel. The **effective averaging window is
$1/(1-\beta)$ steps**, which is the number to reason with when tuning $\beta$.

**Nesterov accelerated gradient** evaluates the gradient at the *look-ahead*
point:

$$v_{t+1} = \beta v_t + \nabla f(\theta_t - \eta\beta v_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

Because it sees where momentum is about to put it, it corrects earlier when
overshooting. Theoretically it gives the accelerated $O(\sqrt\kappa)$ rate; in
deep learning the practical gain over heavy-ball momentum is modest but real.

```mermaid
flowchart LR
    subgraph GD["plain gradient descent"]
        A1["step 1"] -->|"across the ravine"| A2["step 2"]
        A2 -->|"back across"| A3["step 3"]
        A3 -->|"zig-zag, little progress"| A4["step 4"]
    end
    subgraph MOM["with momentum"]
        B1["step 1"] -->|"cross-ravine terms cancel"| B2["step 2"]
        B2 -->|"along-ravine terms accumulate"| B3["step 3"]
        B3 -->|"fast progress along the floor"| B4["step 4"]
    end
```

## Adaptive methods

The insight: different parameters deserve different learning rates. A rare
feature's weight sees a gradient once in ten thousand steps and should take a
big step when it does.

### AdaGrad

$$G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t$$

(All operations elementwise.) Parameters with large accumulated gradients get
smaller steps. Excellent for sparse features — it was built for convex problems
with sparse data. **Fatal flaw**: $G_t$ only grows, so the effective learning
rate decays to zero monotonically and training stalls.

### RMSProp

Replace the sum with an exponential moving average, so old gradients are
forgotten:

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t}+\epsilon}g_t$$

Now the effective rate can rise again when gradients shrink. This fixed
AdaGrad's stall.

### Adam

RMSProp plus momentum, plus bias correction.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad\text{(first moment, direction)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \qquad\text{(second moment, scale)}$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**Why bias correction?** $m_0 = 0$, so $m_1 = (1-\beta_1)g_1 = 0.1 g_1$ — a
10× underestimate. Dividing by $1-\beta_1^t$ removes exactly that
initialisation bias, and the correction decays to 1 as $t$ grows. Without it,
the first few hundred steps take absurdly small steps with $\beta_2 = 0.999$.

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. For
transformers, $\beta_2 = 0.95$ and $\epsilon = 10^{-8}$ is common — a shorter
second-moment window reacts faster to the loss spikes large language models
suffer.

**Reading Adam correctly**: the update magnitude is roughly $\eta$ regardless of
gradient scale, because $\hat m/\sqrt{\hat v} \approx \pm 1$. Adam is closer to
*sign* descent with a smoothed sign than to scaled gradient descent. That is why
Adam is robust to bad loss scaling and why its learning rates are so much
smaller than SGD's (3e-4 vs 0.1).

### AdamW — and why plain Adam + weight decay is wrong

L2 regularisation adds $\lambda\theta$ to the gradient. Inside Adam that term
gets divided by $\sqrt{\hat v}$ along with everything else, so parameters with
large gradients get *less* regularisation — the opposite of the intent. AdamW
**decouples** it:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)$$

The decay is applied directly to the parameters, not routed through the adaptive
scaling. This is the default for essentially every modern transformer. **Exclude
biases and normalisation parameters from weight decay** — decaying a LayerNorm
gain toward zero is meaningless and measurably harmful.

### The full family, compared

| Optimiser | State per param | Key idea | Best for | Watch out for |
|---|---|---|---|---|
| SGD | 0 | plain gradient | convex, tiny models | slow, LR-sensitive |
| SGD + momentum | 1 | velocity | CNNs, vision | needs LR schedule |
| Nesterov | 1 | look-ahead gradient | same | marginal gains |
| AdaGrad | 1 | accumulate $g^2$ | sparse convex | learning rate dies |
| RMSProp | 1 | EMA of $g^2$ | RNNs, RL | no momentum |
| Adam | 2 | EMA of $g$ and $g^2$ | default everywhere | can generalise worse than SGD |
| AdamW | 2 | decoupled decay | transformers, LLMs | tune $\lambda$ separately |
| LAMB / LARS | 2 | layerwise trust ratio | batch sizes in the tens of thousands | complex |
| Lion | 1 | sign of momentum | memory-constrained training | needs smaller LR, higher decay |
| Adafactor | $O(n+m)$ | factored second moment | huge embedding matrices | slightly worse quality |
| Shampoo / SOAP | matrix | full-matrix preconditioning | frontier-scale pretraining | expensive, complex |

**Memory matters at scale.** Adam stores two extra tensors per parameter. In
fp32, a 7B model needs 28 GB for weights and 56 GB for optimiser state — the
optimiser is twice the model. Adafactor factors the second moment into row and
column statistics; 8-bit optimisers quantise it; ZeRO shards it across devices.

### The SGD-vs-Adam generalisation question

Adam converges faster in training loss; SGD with momentum often reaches better
test accuracy on vision tasks. The going explanations are that Adam's
per-parameter normalisation finds sharper minima, and that its implicit
regularisation differs from SGD's. In language modelling Adam is simply
necessary — SGD does not train transformers well at all, probably because of the
extreme heterogeneity in gradient scale across layers, embeddings, and
LayerNorm parameters. Use AdamW for transformers, and consider SGD+momentum for
convolutional vision models with long schedules.

## Learning-rate schedules

The single highest-leverage hyperparameter, and the one most worth scheduling.

| Schedule | Formula | Where it is used |
|---|---|---|
| Step decay | $\eta \times \gamma$ every $k$ epochs | classic ResNet recipes |
| Exponential | $\eta_0 e^{-kt}$ | simple, smooth |
| Cosine | $\eta_t = \eta_{\min}+\tfrac12(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ | the modern default |
| Linear decay to zero | $\eta_0(1 - t/T)$ | LLM pretraining, very competitive |
| Warmup + cosine | linear rise for $w$ steps, then cosine | transformers, almost universally |
| Inverse sqrt | $\eta \propto 1/\sqrt{t}$ | original Transformer paper |
| One-cycle | rise then fall, with inverse momentum schedule | fast convergence, `fastai` |
| Cyclical / warm restarts | sawtooth with restarts | escaping poor basins; ensembling snapshots |
| ReduceLROnPlateau | drop when validation stalls | when you cannot pre-plan $T$ |

**Warmup is not optional for transformers.** At step 0 the attention logits are
near-uniform, the residual stream has no useful signal, and Adam's second-moment
estimate is based on a handful of samples. A large step then destabilises
LayerNorm statistics and can put the model in a bad basin it never leaves. 2,000
to 10,000 warmup steps is typical, or roughly 1% of training.

**Decay to (nearly) zero.** The end-of-training low learning rate does real work
— it is where the model settles into a minimum rather than bouncing around it.
Schedules truncated early lose a surprising amount of final quality.

## Constrained optimisation

### Lagrange multipliers

To minimise $f(x)$ subject to $h(x) = 0$, form

$$\mathcal{L}(x,\lambda) = f(x) + \lambda h(x)$$

and set $\nabla_x\mathcal{L} = 0$, $h(x) = 0$. Geometrically: at the optimum the
gradient of the objective is parallel to the gradient of the constraint, because
any component along the constraint surface could still be exploited.

### KKT conditions

With inequality constraints $g_i(x)\le 0$, the necessary conditions at an
optimum are:

1. **Stationarity**: $\nabla f + \sum_i \mu_i \nabla g_i + \sum_j \lambda_j \nabla h_j = 0$
2. **Primal feasibility**: $g_i \le 0$, $h_j = 0$
3. **Dual feasibility**: $\mu_i \ge 0$
4. **Complementary slackness**: $\mu_i g_i = 0$

Condition 4 is the interesting one: either a constraint is active ($g_i = 0$) or
its multiplier is zero. **This is exactly what makes SVM support vectors
sparse** — only points on the margin have non-zero multipliers, and every other
training point could be deleted without changing the solution.

### Duality

Every convex problem has a dual. Weak duality ($d^\star \le p^\star$) always
holds; strong duality ($d^\star = p^\star$) holds under Slater's condition. The
dual is why SVMs can use kernels: the dual formulation depends on the data only
through inner products $x_i^\top x_j$, which can be swapped for $K(x_i,x_j)$
without ever computing the feature map.

### Projected gradient and proximal methods

For simple constraint sets, take a gradient step and project back:

$$\theta_{t+1} = \Pi_{\mathcal{C}}\bigl(\theta_t - \eta\nabla f(\theta_t)\bigr)$$

For non-smooth regularisers, use the proximal operator. For L1 this gives
**soft-thresholding**, the closed form behind ISTA/FISTA and behind why lasso
produces exact zeros:

$$\mathrm{prox}_{\eta\lambda\|\cdot\|_1}(v)_i = \mathrm{sign}(v_i)\max(|v_i|-\eta\lambda,\,0)$$

## Second-order and beyond

| Method | Update | Cost | Reality |
|---|---|---|---|
| Newton | $-H^{-1}\nabla f$ | $O(d^3)$ | exact for quadratics; impossible for $d>10^4$ |
| Gauss–Newton | $-(J^\top J)^{-1}J^\top r$ | $O(d^3)$ | least squares; PSD by construction |
| Levenberg–Marquardt | $-(J^\top J+\lambda I)^{-1}J^\top r$ | $O(d^3)$ | interpolates Newton and GD |
| BFGS | rank-2 update to $H^{-1}$ | $O(d^2)$ memory | small/medium problems |
| L-BFGS | last $m$ pairs only | $O(md)$ | the workhorse for classical ML; poor with minibatch noise |
| Conjugate gradient | $H$-orthogonal directions | $O(d)$/iter | large sparse linear systems |
| K-FAC | Kronecker-factored Fisher | practical | genuine speedups on some networks |
| Hessian-free | CG on Hessian-vector products | practical | needs no explicit $H$ |

The trick that makes several of these possible is that a **Hessian-vector
product costs one extra backward pass**, no explicit Hessian required:

$$Hv = \nabla_\theta\bigl(\nabla_\theta f \cdot v\bigr)$$

L-BFGS deserves a specific warning: it assumes a deterministic objective. With
minibatch noise its curvature pairs are garbage. Use it for full-batch problems
(logistic regression, CRFs, physics-informed nets with small data), not for SGD
training.

## Gradient-free optimisation

When gradients do not exist — hyperparameters, discrete architectures, black-box
simulators, non-differentiable metrics like BLEU or revenue.

| Method | Idea | Sample efficiency |
|---|---|---|
| Grid search | try everything | terrible in $>3$ dims |
| Random search | sample uniformly | better than grid; strictly dominates it when few dims matter |
| Bayesian optimisation (GP/TPE) | fit a surrogate, optimise an acquisition function | best for expensive evaluations |
| Hyperband / ASHA | aggressive early stopping of bad runs | best when a partial run predicts the final one |
| BOHB | Bayesian + Hyperband | strong practical default |
| Evolutionary / CMA-ES | population, mutation, selection | robust, parallel, many evaluations |
| Simulated annealing | accept worse moves with decaying probability | combinatorial problems |

**Random beats grid** for a specific and often-missed reason: if only 2 of your
6 hyperparameters matter, a grid of $4^6$ points tries only 4 distinct values of
each important one, while 4,096 random points try 4,096 distinct values of each.

## Practical failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| Loss `NaN` in the first few steps | LR too high; no warmup; fp16 overflow | lower LR 10×, add warmup, check loss scaling |
| Loss flat from step 0 | LR too low; dead activations; wrong loss reduction | LR range test; check activation statistics |
| Loss decreases then explodes | LR too high for the late-training sharp region | decay schedule, gradient clipping |
| Train loss falls, val loss rises | overfitting | more data, augmentation, regularisation, early stop |
| Loss oscillates without trend | batch too small; LR at the stability edge | raise batch or lower LR |
| Sudden spike then recovery | bad batch, or an outlier example | clip gradients; inspect the batch |
| Progress stalls mid-training | plateau/saddle; schedule decayed too early | warm restart; check the schedule |
| Works at batch 32, fails at 512 | LR not rescaled | linear or sqrt scaling, longer warmup |

**Gradient clipping** by global norm is nearly free insurance:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

It rescales the whole gradient vector when $\|g\| > 1$, preserving direction and
capping magnitude. Standard for RNNs and transformers.

**The learning-rate range test** is the fastest way to pick $\eta$: start
absurdly low, increase exponentially over a few hundred steps, plot loss vs LR.
Choose roughly an order of magnitude below where the loss starts rising.

## A default recipe that works

```python
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},   # biases, norms
    ],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(  # or a warmup+cosine lambda
    optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.03,
)

for step, batch in enumerate(loader):
    loss = model(batch).loss / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

Every line is a decision made above: decoupled weight decay excluding norms and
biases, $\beta_2 = 0.95$ for transformer stability, ~3% warmup, cosine decay,
global-norm clipping, and gradient accumulation to reach a large effective batch
without the memory.

## Self-check

1. Derive the largest stable learning rate for gradient descent on a quadratic
   with Hessian eigenvalues $\{100, 1\}$, and say how many steps it takes to make
   progress along the flat direction.
2. Explain Adam's bias correction: what goes wrong without it, and for how long?
3. Why is Adam + L2 different from AdamW, and which parameters should be excluded
   from weight decay?
4. Your transformer diverges at step 300 with LR 1e-3. Give four independent
   fixes, ranked.
5. What does complementary slackness say, and what does it imply about SVMs?
6. You quadruple the batch size. What do you do to the learning rate, and why
   two different answers exist.
7. Why does random search beat grid search, in one sentence about effective
   dimensionality?

## Where to go next

- [Calculus](./calculus.md) — where the gradients come from.
- [Linear Algebra](./linear-algebra.md) — eigenvalues, conditioning, and the
  geometry of the loss surface.
- [Numerical Computing](./numerical-methods.md) — floating point, stability, and
  the arithmetic that makes all of this actually run.
