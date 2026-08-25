---
order: 3
description: Probability from sample spaces to Bayes, distributions, expectation, the CLT, and the concentration and information-theoretic tools that justify ML losses.
meta: Math for ML · core
---

# Probability: Reasoning Under Uncertainty

Machine learning is applied probability with a compute budget. A classifier does
not output a label, it outputs $p(y \mid x)$. A loss function is almost always a
negative log-likelihood. Regularisation is a prior. Dropout is a Bernoulli mask.
Diffusion models are a Markov chain run backwards. If you understand probability
properly, half of ML stops being a list of tricks and becomes one idea applied
repeatedly.

## Two ways to read a probability

Before any formula, settle what a probability *means*, because the two schools
lead to different algorithms.

| | Frequentist | Bayesian |
|---|---|---|
| A probability is | a long-run frequency | a degree of belief |
| Parameters are | fixed but unknown | random variables with distributions |
| Data is | random | fixed once observed |
| You estimate with | MLE, confidence intervals | posteriors, credible intervals |
| ML examples | ERM, cross-validation, bootstrap | MAP, Bayesian NNs, Gaussian processes, Thompson sampling |

Neither is "correct". They answer different questions. "What is the probability
this specific coin is fair?" is meaningless to a strict frequentist (the coin
either is or is not) and perfectly natural to a Bayesian. Most practical ML is
frequentist in its training loop and Bayesian in its regularisers.

## The foundations

### Sample space, events, axioms

A **sample space** $\Omega$ is the set of all outcomes. An **event** is a subset
of $\Omega$. A probability measure $P$ satisfies three axioms (Kolmogorov):

1. $P(A) \ge 0$ for every event $A$.
2. $P(\Omega) = 1$.
3. For pairwise disjoint $A_1, A_2, \dots$: $P(\bigcup_i A_i) = \sum_i P(A_i)$.

Everything else is a theorem. The inclusion–exclusion rule, for instance:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

You subtract the intersection because axiom 3 only applies to disjoint sets and
you would otherwise count the overlap twice.

### Conditional probability

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0$$

Read it geometrically: conditioning **restricts the sample space to $B$ and
renormalises**. That is the whole idea, and it is why $P(\cdot \mid B)$ is itself
a valid probability measure.

The **chain rule** follows by rearranging and iterating:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

That factorisation is not a piece of trivia — it is *literally the definition of
an autoregressive language model*. GPT computes each conditional on the right
and multiplies. Nothing more.

### Independence

$A$ and $B$ are independent iff $P(A \cap B) = P(A)P(B)$, equivalently
$P(A \mid B) = P(A)$. Knowing $B$ tells you nothing about $A$.

**Conditional independence** is subtler and more useful:
$A \perp B \mid C$ iff $P(A, B \mid C) = P(A\mid C)P(B\mid C)$. Naive Bayes
assumes features are conditionally independent given the label — false in
practice, useful anyway. Graphical models are entirely a language for stating
which conditional independencies hold.

Independence does **not** follow from conditional independence, nor the reverse.
Ice cream sales and drownings are dependent, but conditionally independent given
temperature. Two independent coin flips become *dependent* once you condition on
their sum — this is "explaining away", and it is why adding a collider to a
causal graph creates spurious correlation.

## Bayes' theorem

Rearranging the definition of conditional probability two ways gives:

$$\underbrace{P(H \mid E)}_{\text{posterior}} = \frac{\overbrace{P(E \mid H)}^{\text{likelihood}} \; \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}$$

with $P(E) = \sum_h P(E \mid h)P(h)$ by the law of total probability.

```mermaid
flowchart LR
    PR["prior P of H<br/>what you believed<br/>before the data"] --> B["Bayes rule"]
    LK["likelihood P of E given H<br/>how well H explains<br/>the observed data"] --> B
    B --> PO["posterior P of H given E<br/>updated belief"]
    PO -.->|"becomes the prior<br/>for the next observation"| PR
```

### The medical-test example, done properly

A disease affects 1 in 1000 people. A test has 99% sensitivity
($P(+ \mid D) = 0.99$) and 99% specificity ($P(- \mid \neg D) = 0.99$). You test
positive. What is $P(D \mid +)$?

Work in natural frequencies over 100,000 people:

| | Has disease (100) | No disease (99,900) | Total |
|---|---|---|---|
| Test positive | 99 | 999 | 1,098 |
| Test negative | 1 | 98,901 | 98,902 |

$$P(D \mid +) = \frac{99}{1098} \approx 9\%$$

A 99%-accurate test on a positive result leaves you 91% likely to be healthy.
The reason is **base rates**: false positives are drawn from a pool 999 times
larger than true positives.

This is not a puzzle, it is your fraud detector, your anomaly detector, and your
rare-disease classifier. It is why precision collapses on imbalanced data even
at high recall, and why "99% accurate" is a meaningless claim without the base
rate. Take the same test to a population where 30% are sick and
$P(D\mid+)$ jumps to 98%.

### MLE, MAP, and full Bayes

Three ways to turn Bayes into an algorithm:

$$\theta_{\text{MLE}} = \arg\max_\theta \; p(\mathcal{D}\mid\theta) \qquad \theta_{\text{MAP}} = \arg\max_\theta \; p(\mathcal{D}\mid\theta)\,p(\theta) \qquad p(\theta \mid \mathcal{D}) \text{ in full}$$

MLE ignores the prior. MAP includes it but returns a single point. Full Bayes
keeps the whole posterior and integrates over it.

**The crucial connection.** Take the log of the MAP objective:

$$\log p(\mathcal{D}\mid\theta) + \log p(\theta)$$

With a Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 I)$, $\log p(\theta) =
-\|\theta\|^2/(2\tau^2) + c$. Negate to get a loss:

$$\underbrace{-\log p(\mathcal{D}\mid\theta)}_{\text{your usual loss}} + \underbrace{\lambda \|\theta\|_2^2}_{\text{weight decay}}$$

**L2 regularisation is a Gaussian prior.** L1 is a Laplace prior. Early stopping
approximates a prior too. Every regulariser you use is a statement of belief
about parameters before seeing data, and $\lambda = 1/(2\tau^2)$ — a stronger
prior is a narrower one.

## Random variables and distributions

A **random variable** is a function $X : \Omega \to \mathbb{R}$. It is neither
random nor a variable; it is a deterministic map from outcomes to numbers, and
the randomness lives in $\Omega$.

### PMF, PDF, CDF

| Object | Discrete | Continuous |
|---|---|---|
| Mass/density | $p(x) = P(X=x)$ | $f(x)$, with $P(X=x)=0$ |
| Normalisation | $\sum_x p(x) = 1$ | $\int f(x)\,dx = 1$ |
| CDF | $F(x)=\sum_{t\le x}p(t)$ | $F(x)=\int_{-\infty}^x f(t)\,dt$ |
| Probability of an interval | sum | $F(b)-F(a)$ |

A density can exceed 1 — $\mathrm{Uniform}(0, 0.5)$ has $f = 2$ everywhere on its
support. Densities are not probabilities; only their integrals are.

### The distributions you must know cold

| Distribution | Support | Parameters | Mean | Variance | Where it shows up in ML |
|---|---|---|---|---|---|
| Bernoulli | $\{0,1\}$ | $p$ | $p$ | $p(1-p)$ | binary labels, dropout masks |
| Binomial | $\{0..n\}$ | $n, p$ | $np$ | $np(1-p)$ | counts of successes, A/B tests |
| Categorical | $\{1..K\}$ | $\mathbf{p}$ | — | — | softmax output, next-token distribution |
| Multinomial | counts | $n,\mathbf{p}$ | $np_k$ | — | bag-of-words counts |
| Poisson | $\{0,1,2,...\}$ | $\lambda$ | $\lambda$ | $\lambda$ | arrival rates, count regression |
| Geometric | $\{1,2,...\}$ | $p$ | $1/p$ | $(1-p)/p^2$ | trials until success |
| Uniform | $[a,b]$ | $a,b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | initialisation, sampling |
| Gaussian | $\mathbb{R}$ | $\mu,\sigma^2$ | $\mu$ | $\sigma^2$ | noise, weight init, VAE latents |
| Exponential | $[0,\infty)$ | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | waiting times, survival models |
| Beta | $[0,1]$ | $\alpha,\beta$ | $\frac{\alpha}{\alpha+\beta}$ | — | prior on a probability, Thompson sampling |
| Dirichlet | simplex | $\boldsymbol{\alpha}$ | — | — | prior over categorical, LDA topics |
| Gumbel | $\mathbb{R}$ | $\mu,\beta$ | — | — | Gumbel-max / Gumbel-softmax sampling |
| Laplace | $\mathbb{R}$ | $\mu,b$ | $\mu$ | $2b^2$ | L1 prior, differential privacy noise |

**Conjugacy** is why Beta and Dirichlet appear: a Beta prior with a Binomial
likelihood gives a Beta posterior, so the update is arithmetic on the
parameters rather than an integral. Beta$(\alpha,\beta)$ + $s$ successes and $f$
failures $\to$ Beta$(\alpha+s, \beta+f)$. That is one line of code and it is the
whole of a Thompson-sampling bandit.

### The Gaussian, and why it is everywhere

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Four reasons it dominates:

1. **The CLT** makes sums of many small effects Gaussian regardless of their own
   distribution.
2. **Maximum entropy**: among all distributions on $\mathbb{R}$ with a given mean
   and variance, the Gaussian has the highest entropy — it assumes the least
   beyond those two facts.
3. **Closure**: sums of Gaussians are Gaussian; linear maps of Gaussians are
   Gaussian; marginals and conditionals of a multivariate Gaussian are Gaussian.
   Nothing else is this well behaved.
4. **Analytic convenience**: negative log-likelihood is exactly squared error.
   $-\log f(x) = \frac{(x-\mu)^2}{2\sigma^2} + c$. **MSE loss is a Gaussian
   likelihood assumption**, and if your residuals are heavy-tailed, MSE is the
   wrong loss for a *probabilistic* reason, not an aesthetic one.

The multivariate form:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

The quadratic form $(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$
is the squared **Mahalanobis distance** — Euclidean distance after whitening by
the covariance. Level sets are ellipsoids whose axes are $\Sigma$'s
eigenvectors, scaled by $\sqrt{\lambda_i}$. That is the same eigendecomposition
PCA uses, which is why PCA and Gaussian modelling keep meeting.

## Expectation, variance, and their algebra

$$\mathbb{E}[X] = \sum_x x\,p(x) \quad\text{or}\quad \int x f(x)\,dx, \qquad \mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

The properties that actually get used:

| Property | Holds when | Note |
|---|---|---|
| $\mathbb{E}[aX+b] = a\mathbb{E}[X]+b$ | always | linearity |
| $\mathbb{E}[X+Y] = \mathbb{E}[X]+\mathbb{E}[Y]$ | **always**, even if dependent | the most under-used fact in probability |
| $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ | only if independent | |
| $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$ | always | shifts do not change spread |
| $\mathrm{Var}(X+Y) = \mathrm{Var}X + \mathrm{Var}Y$ | only if uncorrelated | otherwise add $2\mathrm{Cov}(X,Y)$ |

Linearity of expectation without independence is what makes variance-reduction
arguments work. Averaging $n$ i.i.d. estimates keeps the mean and divides the
variance by $n$:

$$\mathrm{Var}\!\left(\tfrac{1}{n}\sum X_i\right) = \frac{\sigma^2}{n}$$

**That single line is the mathematical case for ensembles, for bagging, for
larger minibatches, and for averaging multiple sampled generations.** Bagging
reduces variance without touching bias precisely because of it — and the
$\rho\sigma^2 + \frac{1-\rho}{n}\sigma^2$ correction for correlated estimators
is why random forests bother to decorrelate trees with feature subsampling.

### Law of total expectation and variance

$$\mathbb{E}[X] = \mathbb{E}\bigl[\mathbb{E}[X\mid Y]\bigr]$$

$$\mathrm{Var}(X) = \underbrace{\mathbb{E}[\mathrm{Var}(X\mid Y)]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}(\mathbb{E}[X\mid Y])}_{\text{epistemic}}$$

The variance decomposition is the formal version of the distinction between
**aleatoric uncertainty** (irreducible noise in the data) and **epistemic
uncertainty** (uncertainty about the model, reducible with more data). Deep
ensembles estimate the second term; a predicted variance head estimates the
first.

### Covariance and correlation

$$\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)], \qquad \rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1,1]$$

Correlation measures **linear** dependence only. $Y = X^2$ with $X$ symmetric
about zero has $\rho = 0$ and total dependence. Zero correlation does not imply
independence — except for jointly Gaussian variables, where it does.

## Concentration: why finite samples work at all

### Law of large numbers

The sample mean converges to the true mean as $n \to \infty$. Weak LLN gives
convergence in probability; strong LLN gives almost-sure convergence. This is the
licence to estimate expectations by averaging, i.e. to use minibatches.

### Central limit theorem

For i.i.d. $X_i$ with finite mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$$

Three things people get wrong:

- The CLT is about the **sampling distribution of the mean**, not about the data.
  Your features do not become Gaussian.
- Convergence rate is $1/\sqrt{n}$. To halve your error bar you need **four
  times** the data. This is why evaluation sets need to be large and why small
  benchmark deltas are usually noise.
- It needs finite variance. Cauchy-distributed data breaks it entirely.

### Tail bounds

| Bound | Statement | Assumption |
|---|---|---|
| Markov | $P(X \ge a) \le \mathbb{E}[X]/a$ | $X \ge 0$ |
| Chebyshev | $P(\lvert X-\mu\rvert \ge k\sigma) \le 1/k^2$ | finite variance |
| Hoeffding | $P(\lvert\bar{X}-\mu\rvert\ge t)\le 2e^{-2nt^2/(b-a)^2}$ | bounded in $[a,b]$ |

Hoeffding is the one to remember: it gives **exponential** concentration, and it
is the engine behind generalisation bounds and behind honest error bars on
accuracy. For accuracy in $[0,1]$ on $n=1000$ test points, a 95% interval is
roughly $\pm 1.36/\sqrt{n} \approx \pm 4.3$ points. Reporting a 1-point
improvement on a 1000-example test set is reporting noise.

## From probability to loss functions

This section is the payoff. Nearly every ML loss is a negative log-likelihood.

Assume the data is i.i.d. from $p_\theta$. The likelihood is
$\prod_i p_\theta(y_i \mid x_i)$; maximising it is minimising

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)$$

Products underflow and sums do not, and the log is monotone, so the log is free.

| Assumed $p(y\mid x)$ | Negative log-likelihood | Known as |
|---|---|---|
| $\mathcal{N}(f_\theta(x), \sigma^2)$ | $\frac{1}{2\sigma^2}(y - f_\theta(x))^2 + c$ | MSE / L2 loss |
| $\mathrm{Laplace}(f_\theta(x), b)$ | $\frac{1}{b}\lvert y - f_\theta(x)\rvert + c$ | MAE / L1 loss |
| $\mathrm{Bernoulli}(\sigma(f_\theta(x)))$ | $-y\log \hat p - (1-y)\log(1-\hat p)$ | binary cross-entropy |
| $\mathrm{Categorical}(\mathrm{softmax}(f_\theta(x)))$ | $-\sum_k y_k \log \hat p_k$ | cross-entropy |
| $\mathrm{Poisson}(e^{f_\theta(x)})$ | $e^{f} - y f + c$ | Poisson regression loss |

Choosing a loss *is* choosing a noise model. If you use MSE you have asserted
Gaussian, homoscedastic residuals. If your target is a count, Poisson NLL will
beat MSE not because it is fancier but because it is *true*.

## Information theory, the part you need

### Entropy

$$H(X) = -\sum_x p(x)\log p(x) = \mathbb{E}[-\log p(X)]$$

$-\log p(x)$ is the **surprisal** of an outcome; entropy is average surprisal, in
bits (log base 2) or nats (natural log). A fair coin has 1 bit. A biased coin
has less. A deterministic outcome has zero.

### Cross-entropy and KL divergence

$$H(p, q) = -\sum_x p(x)\log q(x), \qquad D_{\mathrm{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)$$

KL is the **expected extra nats** you pay for coding data from $p$ with a code
optimised for $q$. It is $\ge 0$, zero iff $p = q$, and **not symmetric** — so
it is a divergence, not a distance.

Because $H(p)$ is a constant of the data, **minimising cross-entropy is exactly
minimising $D_{\mathrm{KL}}(p_{\text{data}} \| p_{\text{model}})$**. Training a
classifier is fitting a distribution, not learning a decision rule; the decision
rule is a downstream `argmax`.

The asymmetry has real consequences:

| Direction | Called | Behaviour | Used by |
|---|---|---|---|
| $D_{\mathrm{KL}}(p \,\Vert\, q)$ | forward, "mean-seeking" | $q$ must cover every mode of $p$ or pay $\infty$ | MLE, cross-entropy training |
| $D_{\mathrm{KL}}(q \,\Vert\, p)$ | reverse, "mode-seeking" | $q$ can ignore modes; collapses onto one | variational inference, VAE ELBO, RLHF KL penalty |

Mode collapse in variational methods is not a bug in the optimiser; it is what
reverse KL asks for.

### Mutual information

$$I(X;Y) = D_{\mathrm{KL}}\bigl(p(x,y)\,\|\,p(x)p(y)\bigr) = H(X) - H(X\mid Y)$$

How many nats knowing $Y$ saves you about $X$. Zero iff independent. It captures
*any* dependence, not just linear — unlike correlation. It underpins
information-gain splits in decision trees, InfoNCE in contrastive learning, and
the information bottleneck view of representation learning.

### Perplexity

$$\mathrm{PPL} = \exp\left(-\frac{1}{T}\sum_{t}\log p(x_t \mid x_{<t})\right) = e^{H}$$

The exponentiated average cross-entropy. Interpret it as the **effective
vocabulary size** the model is choosing among at each step: perplexity 20 means
the model is as uncertain as if picking uniformly among 20 tokens. It is
tokeniser-dependent, so cross-model comparisons are only valid on identical
tokenisation.

## Sampling: how randomness gets generated

| Method | Idea | Used for |
|---|---|---|
| Inverse CDF | $X = F^{-1}(U)$, $U\sim\mathrm{Unif}(0,1)$ | exponential, any invertible CDF |
| Box–Muller | two uniforms to two Gaussians | Gaussian RNG |
| Rejection sampling | propose from $q$, accept with prob $\propto p/Mq$ | low dimensions only |
| Importance sampling | reweight by $p(x)/q(x)$ | off-policy RL, rare-event estimation |
| MCMC (Metropolis–Hastings, Gibbs, HMC) | build a chain whose stationary distribution is $p$ | Bayesian posteriors |
| Reparameterisation | $z = \mu + \sigma\epsilon$ | VAEs, anything needing gradients through a sample |
| Gumbel-max | $\arg\max_k(\log p_k + g_k)$, $g_k\sim$ Gumbel | exact categorical sampling; softened into Gumbel-softmax for gradients |

The Gumbel-max trick is worth internalising: adding i.i.d. Gumbel noise to
log-probabilities and taking the argmax draws *exactly* from the categorical
distribution. Replacing the argmax with a temperature-softmax gives a
differentiable relaxation, which is how discrete latents get trained.

Temperature sampling in an LLM is the same object: dividing logits by $T$ before
softmax interpolates between greedy ($T\to0$) and uniform ($T\to\infty$).

## Curse of dimensionality, probabilistically

Intuitions built in 2D fail badly in 500D.

- The volume of the unit ball concentrates in a thin shell near its surface. Draw
  points uniformly in a $d$-ball and almost all of them are near the boundary.
- Pairwise distances concentrate: $\frac{\max d - \min d}{\min d} \to 0$. Nearest
  neighbours stop being meaningfully nearer.
- Two random high-dimensional vectors are nearly orthogonal with high
  probability — which is why random projections preserve structure
  (Johnson–Lindenstrauss) and why random initialisation gives near-orthogonal
  features for free.
- Sample requirements for density estimation grow exponentially in $d$.

The escape hatch is the **manifold hypothesis**: real data occupies a
low-dimensional manifold inside the ambient space. Images of faces are a tiny
subset of all $256^{H\times W\times 3}$ pixel arrays. Representation learning is
the business of finding coordinates on that manifold.

## Common traps

| Trap | Reality |
|---|---|
| "The test is 99% accurate so I'm 99% likely sick" | base-rate neglect; compute $P(D\mid+)$ |
| "$\rho = 0$ so they're independent" | only true for jointly Gaussian |
| "The CLT makes my data Gaussian" | it makes the *sample mean's* distribution Gaussian |
| "More features can only help" | curse of dimensionality; variance grows |
| "$P(A\mid B) = P(B\mid A)$" | prosecutor's fallacy; they differ by the base-rate ratio |
| "This coin came up heads 5 times, tails is due" | gambler's fallacy; i.i.d. has no memory |
| "The model is 95% confident so it's right 95% of the time" | only if calibrated; modern nets are overconfident |
| "Averaging predictions always helps" | only when errors are decorrelated |

## Self-check

1. A model outputs $p = 0.9$ on 100 examples and is right 70 times. What is
   wrong, what is the term for it, and name one fix.
2. Derive the L2-regularised objective from a Gaussian prior over weights, and
   say what $\lambda$ corresponds to.
3. Why is minimising cross-entropy the same as minimising a KL divergence, and
   which of the two arguments is the data distribution?
4. You have 500 test examples and observe 84% accuracy. Give a rough 95%
   interval and say whether a rival model at 86% is meaningfully better.
5. Explain why reverse KL causes mode collapse and forward KL causes
   over-dispersed models.
6. Two events are independent. You condition on a third event that both cause.
   Are they still independent? Name the phenomenon.
7. Write down the decomposition of predictive variance into aleatoric and
   epistemic parts, and say which one more data reduces.

## Where to go next

- [Statistics & Inference](./statistics.md) — estimators, tests, and the
  bootstrap, built on this foundation.
- [Calculus](./calculus.md) — differentiating the likelihoods defined here.
- [Optimization Techniques](./optimization.md) — minimising them.
