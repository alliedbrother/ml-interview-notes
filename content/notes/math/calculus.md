---
order: 2
description: Derivatives, gradients, Jacobians, Hessians, the chain rule, matrix calculus, and automatic differentiation — derived from scratch and connected to backpropagation.
meta: Math for ML · core
---

# Calculus: How Models Learn to Move

Linear algebra tells you what a model *is* — a stack of matrices acting on
vectors. Calculus tells you how to *change* it. Every training run you have ever
launched is the same loop: measure how wrong the model is, ask calculus which
direction reduces that wrongness, take a step, repeat. This page derives that
machinery end to end, from the definition of a derivative up to the matrix
calculus you need to hand-derive a backward pass.

## The one question calculus answers

> If I nudge this input a little, how much does the output move, and in which
> direction?

That is it. Everything below is a more precise, higher-dimensional, or more
computationally efficient way of asking that question.

In machine learning the "input" is a parameter $\theta$ (a weight, a bias, an
embedding entry), the "output" is a scalar loss $L$, and the answer is
$\partial L / \partial \theta$. A model with 70 billion parameters asks that
question 70 billion times per step, which is why the *efficiency* of answering it
— reverse-mode automatic differentiation — matters as much as the mathematics.

```mermaid
flowchart LR
    P["parameters<br/>theta"] -->|"forward pass"| Y["prediction<br/>y_hat"]
    Y -->|"compare to target"| L["scalar loss<br/>L"]
    L -->|"reverse pass:<br/>dL/dtheta"| G["gradient"]
    G -->|"theta - lr * grad"| P
```

## Derivatives from first principles

### The definition

The derivative of $f$ at $x$ is the limit of the slope of a secant line as the
secant collapses to a tangent:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Read it as: *the rate at which $f$ changes per unit change in $x$, measured
infinitesimally close to $x$.*

Let's actually do one. Take $f(x) = x^2$:

$$\frac{(x+h)^2 - x^2}{h} = \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h$$

Let $h \to 0$ and the $h$ term vanishes: $f'(x) = 2x$. No rule memorised, just
algebra and a limit.

### What a derivative buys you: local linear approximation

The single most useful way to think about $f'(x)$ is that it gives you the best
straight-line approximation to $f$ near $x$:

$$f(x + \Delta) \approx f(x) + f'(x)\,\Delta$$

This is the first-order Taylor expansion, and it is the *entire justification for
gradient descent*. If you only trust the model of the loss surface near your
current point, you should only take a small step — which is exactly what a
learning rate is.

| $\Delta$ | true $f(2+\Delta)$ for $f=x^2$ | linear estimate $4 + 4\Delta$ | error |
|---|---|---|---|
| $0.1$ | $4.41$ | $4.40$ | $0.01$ |
| $0.5$ | $6.25$ | $6.00$ | $0.25$ |
| $1.0$ | $9.00$ | $8.00$ | $1.00$ |
| $2.0$ | $16.00$ | $12.00$ | $4.00$ |

The error grows like $\Delta^2$. Small steps are cheap in error; big steps are
not. That table is the learning-rate trade-off in miniature.

### Continuity, differentiability, and why ReLU is fine

A function is **continuous** at $x$ if small input changes give small output
changes. It is **differentiable** at $x$ if it also has a well-defined tangent
there. Differentiable implies continuous; the converse is false.

$\mathrm{ReLU}(x) = \max(0, x)$ is continuous everywhere but not differentiable
at $x = 0$ — the slope jumps from $0$ to $1$. Frameworks paper over this with a
**subgradient**: PyTorch and TensorFlow both define $\mathrm{ReLU}'(0) = 0$.
This is a convention, not an ordinary derivative. Floating-point values are discrete: exact zeros occur through initialization, cancellation and nonlinearities, so there is no floating-point measure-zero justification. Check the selected derivative when implementing a nonsmooth operator; see [PyTorch's rules](https://docs.pytorch.org/docs/stable/notes/autograd.html).

### The rules, and why they are true

| Rule | Statement | One-line reason |
|---|---|---|
| Constant | $\frac{d}{dx} c = 0$ | a flat line has no slope |
| Power | $\frac{d}{dx} x^n = n x^{n-1}$ | binomial expansion of $(x+h)^n$; all but one term dies |
| Sum | $(f+g)' = f' + g'$ | limits are linear |
| Product | $(fg)' = f'g + fg'$ | expand $f(x+h)g(x+h)$, drop the $O(h^2)$ cross term |
| Quotient | $(f/g)' = \frac{f'g - fg'}{g^2}$ | product rule applied to $f \cdot g^{-1}$ |
| Chain | $(f \circ g)'(x) = f'(g(x)) \, g'(x)$ | rates multiply through a composition |
| Exponential | $\frac{d}{dx} e^x = e^x$ | $e$ is *defined* as the base where this holds |
| Log | $\frac{d}{dx} \ln x = 1/x$ | inverse-function rule applied to $e^x$ |

The chain rule is the load-bearing one. Every other rule is convenience; the
chain rule *is* backpropagation.

### Chain rule, intuitively

If a car travels twice as fast as a bicycle, and the bicycle travels three times
as fast as a walker, the car travels six times as fast as the walker. Rates
compose by multiplication. Formally, with $z = f(y)$ and $y = g(x)$:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

The notation is suggestive — it looks like the $dy$ terms cancel — and while
that is not a proof, it is a reliable mnemonic.

**Worked example.** The logistic sigmoid $\sigma(x) = \dfrac{1}{1 + e^{-x}}$.

Write it as $\sigma = u^{-1}$ where $u = 1 + e^{-x}$.

$$\frac{d\sigma}{dx} = \frac{d\sigma}{du}\cdot\frac{du}{dx} = \left(-u^{-2}\right)\cdot\left(-e^{-x}\right) = \frac{e^{-x}}{(1+e^{-x})^2}$$

Now the classic rearrangement. Note $\dfrac{e^{-x}}{1+e^{-x}} = 1 - \sigma(x)$, so

$$\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

This is why sigmoid was popular: the derivative costs nothing extra once you
have the forward value. The bound $\sigma'\le0.25$ explains one source of vanishing gradients: ten activation Jacobians have norm at most $4^{-10}$. Full layer Jacobians also include weight matrices, giving the bound $4^{-10}\prod_l\|W_l\|_2$; weights can offset or amplify the effect. Sigmoid remains useful for output probabilities and gates.

## Going multivariate

A neural network is not a function of one number. It is a function of millions.
Three objects generalise the derivative.

### Partial derivatives

$\partial f / \partial x_i$ is the ordinary derivative of $f$ with respect to
$x_i$, holding every other variable fixed.

For $f(x, y) = x^2 y + 3y$:

$$\frac{\partial f}{\partial x} = 2xy, \qquad \frac{\partial f}{\partial y} = x^2 + 3$$

When differentiating with respect to $x$, the symbol $y$ is a constant. That is
the whole trick.

### The gradient

Stack the partials into a vector:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \partial f/\partial x_1 \\ \partial f/\partial x_2 \\ \vdots \\ \partial f/\partial x_n \end{bmatrix}$$

Two facts make the gradient the centre of optimisation:

1. **$\nabla f$ points in the direction of steepest ascent.** So $-\nabla f$ is
   steepest descent — the direction gradient descent walks.
2. **$\|\nabla f\|$ is the rate of change in that direction.** A flat region has
   a small gradient; a cliff has a huge one. Gradient clipping exists because
   cliffs exist.

Why is (1) true? The **directional derivative** of $f$ along a unit vector
$\mathbf{u}$ is

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \, \|\mathbf{u}\| \cos\theta = \|\nabla f\| \cos\theta$$

That is maximised when $\cos\theta = 1$, i.e. when $\mathbf{u}$ points along
$\nabla f$. The dot product from linear algebra does the work; steepest ascent is
a corollary, not an axiom.

```mermaid
flowchart TD
    S["point x on the loss surface"] --> G["compute grad f at x"]
    G --> D["-grad f is the direction<br/>of fastest decrease"]
    D --> STEP["x_new = x - lr * grad f"]
    STEP --> C{"grad norm small?"}
    C -->|"no"| G
    C -->|"yes"| STOP["stationary point:<br/>min, max, or saddle"]
```

### The Jacobian

When the output is also a vector, $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$,
every output has a gradient. Stack them as rows:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{bmatrix} \in \mathbb{R}^{m \times n}$$

The gradient is the special case $m = 1$ (transposed). The Jacobian of a linear
map $\mathbf{f}(\mathbf{x}) = W\mathbf{x}$ is just $W$ — linear functions are
their own derivative, which is precisely why linear algebra and calculus fit
together so cleanly in a neural network.

Crucially, **ordinary scalar-loss backprop avoids building the full Jacobian**. A layer mapping 4096
activations to 4096 activations has a $4096 \times 4096$ Jacobian — 16.7M
entries per layer per example. Autodiff computes *vector–Jacobian products*
$\mathbf{v}^\top J$ instead, which cost the same as one forward pass. Hold that
thought for the autodiff section. Explicit Jacobian APIs also exist when the full matrix is needed.

### The Hessian

Second derivatives of a scalar function, arranged in a matrix:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j}, \qquad H \in \mathbb{R}^{n \times n}$$

By Schwarz's theorem $H$ is symmetric for any function with continuous second
partials. ReLU and hinge losses are nonsmooth exceptions at their kinks. The Hessian describes **curvature**
— how the gradient itself changes as you move.

| Hessian at a stationary point | Eigenvalues | Meaning |
|---|---|---|
| Positive definite | all $> 0$ | local minimum — bowl |
| Negative definite | all $< 0$ | local maximum — dome |
| Indefinite | mixed signs | **saddle point** |
| Semidefinite and singular | some $= 0$, no mixed signs | second-order test inconclusive; mixed signs still prove a saddle even with zeros |

Saddles and plateaus can slow high-dimensional optimization, but a random-matrix heuristic does not prove that stationary points of an arbitrary network are almost surely saddles. Their distribution depends on the objective, data and parameterization.

For an SPD Hessian, the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$
predicts how badly gradient descent will zig-zag. A ravine that is 1000 times
steeper across than along has $\kappa = 1000$, and plain gradient descent needs
roughly $\kappa$ iterations to make progress along the ravine floor. Momentum,
Adam, and batch normalisation are all, in different ways, attacks on a bad
condition number.

## Taylor series: the bridge to optimisation

Expand $f$ around $\mathbf{x}_0$ to second order:

$$f(\mathbf{x}_0 + \Delta) \approx f(\mathbf{x}_0) + \nabla f^\top \Delta + \tfrac{1}{2}\Delta^\top H \Delta$$

Every optimiser is a decision about how much of this expansion to use.

| Method | Uses | Step | Cost per step |
|---|---|---|---|
| Gradient descent | first order only | $-\eta \nabla f$ | $O(n)$ |
| Newton's method | full second order | $-H^{-1}\nabla f$ | $O(n^3)$ solve |
| Quasi-Newton (L-BFGS) | approximate $H^{-1}$ | $-B \nabla f$ | $O(nm)$ |
| Adam / RMSProp | diagonal gradient-moment scaling | $-\eta \, \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ | $O(n)$ |

Newton has local quadratic convergence near a nondegenerate solution under suitable Hessian regularity, and one step solves an unconstrained SPD quadratic in exact arithmetic. Use a solve, not an explicit inverse. Adam's second moment is a gradient-scale statistic, not generally a Hessian estimate.

**Derive Newton's step yourself.** Minimise the quadratic model over $\Delta$:
set its gradient to zero.

$$\nabla_\Delta \left[ f + \nabla f^\top \Delta + \tfrac12 \Delta^\top H \Delta \right] = \nabla f + H\Delta = 0 \;\Longrightarrow\; \Delta = -H^{-1}\nabla f$$

## Convexity, checked with calculus

A function is **convex** if the line between any two points on its graph lies on
or above the graph:

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}), \quad \lambda \in [0,1]$$

For twice differentiable $f$ on an open convex domain, $f$ is convex iff its Hessian is PSD everywhere. In one dimension the domain must be an interval and the criterion is $f''\ge0$.

Why it matters: **for a convex function every local minimum is global.** Linear
regression, logistic regression, and SVMs with convex losses have this
guarantee. Typical jointly trained neural-network objectives are nonconvex. Convexity of the scalar loss or activation does not establish convexity in all parameters. Permutation symmetry explains nonunique representations, but symmetry alone is not a proof of nonconvexity.

| Loss | Convex in parameters? | Consequence |
|---|---|---|
| MSE for linear regression | yes | a least-squares minimizer exists; algorithm and step conditions still matter |
| Log-loss for logistic regression | yes | finite attainment and uniqueness need additional conditions, including identifiability; nonseparation alone is insufficient |
| Hinge loss for linear SVM | yes (not smooth) | subgradient methods |
| Typical jointly trained 2-layer MLP objective | generally no | initialisation and schedule matter |

## Matrix calculus: deriving a backward pass by hand

This is the section that separates people who *use* autograd from people who can
*debug* it.

### Layout convention — pick one and never waver

Two conventions exist for $\partial y / \partial \mathbf{x}$ when $y$ is scalar:
numerator layout (a row vector) and denominator layout (a column vector). ML
practice, and this page, uses the convention that **the gradient of a scalar
with respect to any tensor has the same shape as that tensor.** If $W$ is
$m \times n$, then $\partial L / \partial W$ is $m \times n$. This makes
$W \leftarrow W - \eta \, \partial L/\partial W$ type-check, which is the only
thing you actually need.

**Shape-checking is your debugger.** If a hand-derived gradient does not have
the same shape as its parameter, the derivation is wrong. In practice you can
often recover the right expression from shapes alone.

### The identities worth memorising

| $f$ | $\partial f / \partial \mathbf{x}$ |
|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$; $=2A\mathbf{x}$ if $A$ symmetric |
| $\lVert\mathbf{x}\rVert_2^2 = \mathbf{x}^\top\mathbf{x}$ | $2\mathbf{x}$ |
| $\lVert A\mathbf{x} - \mathbf{b}\rVert_2^2$ | $2A^\top(A\mathbf{x} - \mathbf{b})$ |
| $\mathrm{tr}(A^\top B)$ w.r.t. $A$ | $B$ |
| $\log \det X$ w.r.t. real $X$ with positive determinant | $X^{-\top}$ |

### Worked derivation 1 — linear regression normal equations

Loss: $L(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|_2^2$ with
$X \in \mathbb{R}^{N \times d}$.

Expand:

$$L = (X\mathbf{w} - \mathbf{y})^\top(X\mathbf{w}-\mathbf{y}) = \mathbf{w}^\top X^\top X \mathbf{w} - 2\mathbf{y}^\top X \mathbf{w} + \mathbf{y}^\top \mathbf{y}$$

Differentiate term by term using the table ($X^\top X$ is symmetric):

$$\nabla_{\mathbf{w}} L = 2X^\top X \mathbf{w} - 2X^\top \mathbf{y}$$

Set to zero. If $X$ has full column rank:

$\boxed{\;\mathbf{w}^\star = (X^\top X)^{-1} X^\top \mathbf{y}\;}$$

Four lines of matrix calculus produce the normal equations. Add L2
regularisation $\lambda\|\mathbf{w}\|^2$ and the gradient gains $2\lambda
\mathbf{w}$, giving ridge regression
$\mathbf{w}^\star = (X^\top X + \lambda I)^{-1}X^\top \mathbf{y}$ — and now the
matrix is invertible for $\lambda>0$ even when $X^\top X$ is singular. At zero penalty use a rank-aware [least-squares solver](./linear-algebra.md#18-pseudoinverses-minimum-norm-and-ridge-regression). That is the entire
mathematical content of "regularisation stabilises the solution".

### Worked derivation 2 — a linear layer's backward pass

Forward: $Y = XW + \mathbf{b}$, where $X \in \mathbb{R}^{B \times d_{in}}$,
$W \in \mathbb{R}^{d_{in} \times d_{out}}$, $Y \in \mathbb{R}^{B \times d_{out}}$.

Given the incoming gradient $G = \partial L/\partial Y \in \mathbb{R}^{B \times d_{out}}$:

$$\frac{\partial L}{\partial W} = X^\top G \in \mathbb{R}^{d_{in}\times d_{out}}, \qquad \frac{\partial L}{\partial X} = G W^\top \in \mathbb{R}^{B \times d_{in}}, \qquad \frac{\partial L}{\partial \mathbf{b}} = \sum_{i=1}^{B} G_{i,:}$$

The chain rule establishes these expressions; shapes are a necessary check, not a proof. $X^\top G$ is the only way to get
$(d_{in}, d_{out})$ from a $(B, d_{in})$ and a $(B, d_{out})$. The bias
gradient sums over the batch because the bias was *broadcast* over the batch in
the forward pass — and the rule is general: **the backward of a broadcast is a
sum over the broadcast axis.**

```python
import numpy as np

class Linear:
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) * (2.0 / d_in) ** 0.5
        self.b = np.zeros(d_out)

    def forward(self, X):
        self.X = X                       # cached for the backward pass
        return X @ self.W + self.b

    def backward(self, G):
        self.dW = self.X.T @ G           # (d_in, d_out)
        self.db = G.sum(axis=0)          # broadcast -> sum
        return G @ self.W.T              # (B, d_in), passed to the layer below
```

Note what `forward` had to keep: `X`. That cached activation is why training
memory scales with batch size and depth, and why gradient checkpointing —
recomputing `X` instead of storing it — trades compute for memory.

### Worked derivation 3 — softmax + cross-entropy

Softmax over logits $\mathbf{z} \in \mathbb{R}^K$:

$$p_i = \frac{e^{z_i}}{\sum_{k} e^{z_k}}$$

Its Jacobian, derived with the quotient rule (case $i = j$ and $i \ne j$
separately):

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

Cross-entropy against a one-hot target $\mathbf{y}$:
$L = -\sum_k y_k \log p_k$, so $\partial L/\partial p_k = -y_k/p_k$. Chain them:

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j} = -\sum_i \frac{y_i}{p_i} p_i(\delta_{ij}-p_j) = -y_j + p_j \sum_i y_i$$

Since $\sum_i y_i = 1$ for a one-hot target:

$$\boxed{\;\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}\;}$$

Predicted minus actual. All that algebra collapses to a subtraction. This is
why every framework fuses the two operations into one kernel
(`cross_entropy_with_logits`): fusing skips the $K \times K$ Jacobian entirely,
and it is also numerically safer, because you never materialise
$e^{z_i}$ for a large $z_i$.

The same clean form appears for sigmoid + binary cross-entropy and for linear +
MSE. That is not a coincidence: all three are canonical link functions for
exponential-family likelihoods, and the identity $\partial L/\partial \mathbf{z}
= \mathbf{p} - \mathbf{y}$ is a general property of that pairing.

## Automatic differentiation

Autodiff is neither symbolic differentiation (which explodes in expression size)
nor numerical differentiation (which is inaccurate and slow). It applies the
chain rule numerically over the computation graph.

### Forward mode vs reverse mode

Consider $f: \mathbb{R}^n \to \mathbb{R}^m$ built from elementary operations.

- **Forward mode** propagates derivatives *with* the computation. One pass gives
  you one column of the Jacobian — the derivative of *all outputs* with respect
  to *one input*. Cost: $O(n)$ passes for the full Jacobian.
- **Reverse mode** runs the graph forward, then walks it backwards accumulating
  *adjoints* $\bar{v} = \partial L/\partial v$. One pass gives you one row — the
  derivative of *one output* with respect to *all inputs*. Cost: $O(m)$ passes.

Neural network training has $n \approx 10^9$ parameters and $m = 1$ scalar loss.
Reverse mode wins by nine orders of magnitude. That asymmetry is the reason
deep learning is computationally possible at all.

| | Forward mode | Reverse mode |
|---|---|---|
| Best when | few inputs, many outputs | many inputs, few outputs |
| Memory | propagates a tangent with each live primal; no reverse tape required | stores or recomputes needed forward values |
| Passes for full Jacobian | $n$ | $m$ |
| Used for | Jacobian-vector products, some ODE/sensitivity work | all of deep learning |

```mermaid
flowchart TD
    subgraph FWD["forward pass — build the tape"]
        X["x"] --> A["a = x * w"]
        A --> B["h = relu of a"]
        B --> C["y = h * v"]
        C --> L["L = loss of y"]
    end
    L -->|"bar_L = 1"| RL["seed the adjoint"]
    RL -->|"bar_y = dL/dy"| RC["node y"]
    RC -->|"bar_h = bar_y * v<br/>bar_v = bar_y * h"| RB["node h"]
    RB -->|"bar_a = bar_h * 1[a>0]"| RA["node a"]
    RA -->|"bar_w = bar_a * x"| RW["gradient for w"]
```

### The adjoint rule in one line

For every node $v$ with children $c$ that consume it:

$$\bar{v} = \sum_{c \,:\, v \to c} \bar{c} \, \frac{\partial c}{\partial v}$$

The sum matters. If a tensor is used twice (a residual connection, a shared
embedding matrix, weight tying between input and output embeddings), gradients
from every consumer **add**. Getting this wrong — overwriting instead of
accumulating — is the classic hand-rolled-autograd bug, and it is why PyTorch's
`.grad` accumulates and you must call `optimizer.zero_grad()`.

### Vector–Jacobian products

Reverse mode never forms $J$. It computes $\mathbf{v}^\top J$ for the incoming
adjoint $\mathbf{v}$. For the linear layer above, $\mathbf{v}^\top J$ with
respect to $X$ *is* $GW^\top$ — a matmul, not a $10^7$-entry matrix. Every
`backward()` you have ever written is a VJP rule.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)

y = (x * w).relu()
L = y ** 2

L.backward()                 # reverse-mode sweep, seeded with dL/dL = 1
assert x.grad.item() == 36.0
assert w.grad.item() == 24.0
print(x.grad, w.grad)        # tensor([36.]) tensor([24.])
```

The assertions agree with the direct calculation. $L = (xw)^2$, so
$\partial L/\partial x = 2xw \cdot w = 2 \cdot 6 \cdot 3 = 36$ and
$\partial L/\partial w = 2xw \cdot x = 2 \cdot 6 \cdot 2 = 24$. Always verify a
gradient you did not derive.

### Gradient checking

When you write a custom kernel, verify it against a finite difference. Use the
**central** difference, whose error is $O(h^2)$ rather than the forward
difference's $O(h)$:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$$

```python
def grad_check(f, x, analytic_grad, h=1e-5):
    """Central-difference diagnostic; choose tolerance for scale and dtype."""
    numeric = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        try:
            x[i] = old + h
            f_plus = f(x)
            x[i] = old - h
            f_minus = f(x)
        finally:
            x[i] = old
        numeric[i] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    denom = np.maximum(np.abs(numeric) + np.abs(analytic_grad), 1e-8)
    return np.max(np.abs(numeric - analytic_grad) / denom)
```

Two practical warnings. Use `float64` — `float32` round-off swamps the signal at
$h = 10^{-5}$. And do not gradient-check through ReLU at a kink or through
dropout with a live RNG; freeze the mask first.

## Integration, briefly but honestly

Derivatives dominate training; integrals dominate probabilistic modelling.

| Where integrals show up | Form |
|---|---|
| Expectation of a loss | $\mathbb{E}_{x\sim p}[f(x)] = \int f(x)p(x)\,dx$ |
| Normalising a density | $\int p(x)\,dx = 1$ |
| Marginalising a latent | $p(x) = \int p(x, z)\,dz$ |
| Evidence lower bound (VAE) | $\log p(x) \ge \mathbb{E}_{q}[\log p(x \mid z)] - \mathrm{KL}(q \,\Vert\, p)$ |
| Continuous normalising flows | $\log p(x_T) = \log p(x_0) - \int \mathrm{tr}(J)\,dt$ |

Two techniques carry most of the weight in ML.

**Change of variables.** If $z = g(x)$ is invertible, then

$$p_Z(z) = p_X(x)\left|\det \frac{\partial x}{\partial z}\right|$$

That determinant of a Jacobian is exactly why normalising flows are designed
around architectures with cheap determinants (triangular Jacobians, coupling
layers).

**The reparameterisation trick.** You cannot backpropagate through a sample. But
if $z \sim \mathcal{N}(\mu, \sigma^2)$ is rewritten as $z = \mu + \sigma
\epsilon$ with $\epsilon \sim \mathcal{N}(0,1)$, the randomness moves into a
constant input and $\partial z/\partial \mu = 1$, $\partial z/\partial \sigma =
\epsilon$ flow normally. This single substitution is what makes VAEs trainable
by ordinary autodiff.

## Worked extensions: differentiability, integration and stochastic gradients

### Partial derivatives need not form a local linear approximation

Set $f(x,y)=xy/\sqrt{x^2+y^2}$ away from zero and $f(0,0)=0$.
Both coordinate partials at zero vanish. Along $(t,t)$, however,
$f(t,t)=|t|/\sqrt2$, and $|f(t,t)|/\|(t,t)\|=1/2$ does not approach zero.
The candidate zero Jacobian is therefore not a total derivative. Total
differentiability requires $f(x+h)=f(x)+Jh+o(\|h\|)$ uniformly over directions.
Continuous first partials in a neighborhood are a sufficient condition.

"Steepest" also depends on geometry. Under $\|u\|_2=1$ it is the normalized
Euclidean gradient. Under $u^\top Mu=1$ for SPD $M$, maximizing $g^\top u$
instead gives $u\propto M^{-1}g$. The derivative is unchanged; the meaning
of a unit step changed.

### Three integration techniques, actually evaluated

The fundamental theorem says that for continuous $f$,
$F(x)=\int_a^x f(t)\,dt$ satisfies $F'=f$, and
$\int_a^b F'(x)\,dx=F(b)-F(a)$ when the requisite regularity holds.
Thus $\int_0^1 x^2\,dx=[x^3/3]_0^1=1/3$.

Substitution is the chain rule backwards:
$\int_0^1 2x e^{x^2}\,dx=\int_0^1 e^u\,du=e-1$, with $u=x^2$.
Change the integration limits as well as the integrand.
Integration by parts is the product rule backwards:
$\int u\,dv=uv-\int v\,du$. For exponential rate $\lambda>0$,

$$
\mathbb E[X]=\int_0^\infty x\lambda e^{-\lambda x}\,dx
=[-xe^{-\lambda x}]_0^\infty+\int_0^\infty e^{-\lambda x}\,dx
=1/\lambda.
$$

For a joint density $f(x,y)=2$ on $0<y<x<1$, marginalization must respect
the triangular support: $f_X(x)=\int_0^x2\,dy=2x$ and
$\mathbb E[X]=\int_0^1 2x^2\,dx=2/3$. Integrating $y$ from zero to one
for every $x$ would count points outside the support.

### When a derivative can pass through an expectation

A sufficient local condition is an almost-everywhere parameter derivative
dominated by an integrable envelope on a neighborhood, with a fixed integration
domain. Differentiation under the integral then follows from dominated
convergence. Boundary-moving distributions need boundary terms or a careful
change of variables; formal symbol manipulation is not enough.

For $Z=g_\theta(\epsilon)$ with parameter-independent noise,

$$
\nabla_\theta\mathbb E[h_\theta(Z)]
=\mathbb E[\partial_\theta h_\theta(g_\theta(\epsilon))
J_{g_\theta}^\top\nabla_z h_\theta(g_\theta(\epsilon))].
$$

Alternatively, differentiating a density on fixed support gives the
score-function identity

$$
\nabla_\theta\mathbb E_{p_\theta}[h_\theta(Z)]
=\mathbb E[\partial_\theta h_\theta(Z)
h_\theta(Z)\nabla_\theta\log p_\theta(Z)].
$$

For $Z\sim N(\mu,1)$ and $h(Z)=Z^2$, both yield derivative $2\mu$:
pathwise uses $\mathbb E[2Z]$, while the score form uses
$\mathbb E[Z^2(Z-\mu)]$. Subtracting a parameter-independent baseline from
$h$ in the score term preserves expectation because the score has zero mean
under those regularity conditions. It can reduce variance. For uniform
$Z\sim U(0,\theta)$, omitting the moving endpoint gives an incorrect result;
using $Z=\theta U$ correctly gives $\partial_\theta\mathbb E[Z]=1/2$.

### A branched graph with JVP and VJP checks

Let $u=xw$, $L=u^2+u$. At $(x,w)=(2,3)$, $u=6$ and the two branches
contribute $\bar u=12+1=13$, hence $(\bar x,\bar w)=(39,26)$.
For the vector output $F=(xw,x+w)$, its Jacobian there is
$J=\begin{bmatrix}3&2\\1&1\end{bmatrix}$.
With $v=(1,-1)$ and $a=(2,3)$, $Jv=(1,0)$ and $J^\top a=(9,7)$.

```python runnable
import numpy as np
import torch

torch.manual_seed(31)
torch.set_num_threads(1)
z = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
def fun(t):
    return torch.stack((t[0]*t[1], t.sum()))
v = torch.tensor([1., -1.], dtype=torch.float64)
a = torch.tensor([2., 3.], dtype=torch.float64)
J = torch.autograd.functional.jacobian(fun, z)
_, jvp = torch.autograd.functional.jvp(fun, z, v)
_, vjp = torch.autograd.functional.vjp(fun, z, a)
torch.testing.assert_close(jvp, J @ v)
torch.testing.assert_close(vjp, J.T @ a)
u = z[0]*z[1]
grad = torch.autograd.grad(u*u+u, z)[0]
torch.testing.assert_close(grad, torch.tensor([39., 26.], dtype=z.dtype))

# Smooth derivative: truncation dominates large h, roundoff eventually dominates.
x = .7
steps = 10.**np.arange(-1, -15, -1)
errors = [abs((np.sin(x+h)-np.sin(x-h))/(2*h)-np.cos(x)) for h in steps]
assert min(errors) < 1e-9
assert errors[0] > min(errors) and errors[-1] > min(errors)
zero = torch.tensor(0., requires_grad=True)
zero.relu().backward()
assert zero.grad.item() == 0.
assert (max(0., 1e-6)-max(0., -1e-6))/(2e-6) == .5
print("JVP:", jvp.tolist(), "VJP:", vjp.tolist(), "best error:", min(errors))
```

The central difference at the ReLU kink is $1/2$, while PyTorch selects zero.
Neither is evidence that the smooth derivative checker is broken: no ordinary
derivative exists at that point.

## Where calculus quietly fails you

| Symptom | Calculus cause | Standard fix |
|---|---|---|
| Vanishing gradients | products of derivatives $< 1$ over depth | ReLU-family activations, residual connections, normalisation |
| Exploding gradients | products of derivatives $> 1$; sharp cliffs | gradient clipping by global norm |
| Dead ReLUs | $f' = 0$ for all $x < 0$; unit never recovers | LeakyReLU/GELU, better init, lower LR |
| Slow zig-zag progress | ill-conditioned Hessian | normalisation, momentum, Adam |
| `NaN` after a few steps | $\log 0$ or $e^{\text{large}}$ in the loss | log-sum-exp trick, fused loss kernels, clamp inputs to $\log$ |
| Plateau in the middle of training | saddle point, small gradient in every direction | momentum carries you across; noise from SGD helps |

The **log-sum-exp trick** is worth stating outright because it appears in every
softmax implementation:

$$\log \sum_k e^{z_k} = z_{\max} + \log \sum_k e^{z_k - z_{\max}}$$

Mathematically an identity; numerically the difference between a working model
and `inf`. Every exponent is now $\le 0$, so nothing overflows.

## Interview-grade questions and their answers

**Why do we minimise the loss instead of maximising accuracy directly?**
Accuracy is piecewise constant — its gradient is zero almost everywhere and
undefined at the jumps. Cross-entropy is a differentiable surrogate that is
useful for learning probabilities, but lowering it does not monotonically improve finite-dataset accuracy. This is the single most common "why" in
ML and the answer is purely calculus.

**What is the gradient of $\|\mathbf{w}\|_1$?** $\mathrm{sign}(\mathbf{w})$,
with the subgradient interval $[-1,1]$ at zero. Exact zeros arise naturally from the L1 optimality condition and proximal soft-thresholding; ordinary finite gradient steps need not land on zero. L2 shrinks smoothly and does not generally induce sparse optima.

**Why divide attention scores by $\sqrt{d_k}$?** For independent
zero-mean unit-variance components, $\mathbf{q}\cdot\mathbf{k}$ has variance
$d_k$. Large-magnitude logits push softmax into a saturated regime where its
Jacobian $p_i(\delta_{ij}-p_j)$ is near zero — vanishing gradients again.
Dividing by $\sqrt{d_k}$ restores unit variance.

**Can you have a zero gradient at a point that is not a minimum?** Yes: maxima,
saddle points, and flat plateaus. Check the Hessian's eigenvalues to tell them
apart when the second-order test is decisive; a zero Hessian is inconclusive.

**Why is the gradient of the loss w.r.t. a shared weight a sum?** Because the
adjoint rule sums over every consumer of a node. Weight tying, convolution
(one kernel applied at every position), and recurrence (one $W_{hh}$ applied at
every timestep) all produce summed gradients — and that summation over $T$
timesteps includes repeated state-Jacobian products; their amplification is the central explosion mechanism, not summation alone.

## Self-check

1. Derive $\sigma'(x) = \sigma(x)(1-\sigma(x))$ without looking, then explain in
   one sentence why it causes vanishing gradients.
2. A layer computes $Y = XW$ with $X \in \mathbb{R}^{32 \times 512}$ and
   $W \in \mathbb{R}^{512 \times 128}$. Write $\partial L/\partial W$ and
   $\partial L/\partial X$ from shapes alone.
3. Why does reverse-mode autodiff need to store the forward activations, and what
   is the standard technique to avoid storing all of them?
4. Show that $\partial L/\partial \mathbf{z} = \mathbf{p} - \mathbf{y}$ for
   softmax + cross-entropy.
5. Your loss goes to `NaN` at step 40 with a large learning rate. Name three
   distinct calculus-level explanations and the diagnostic for each.
6. Given $H$ with eigenvalues $\{100, 1, 0.01\}$ at a stationary point, classify
   the point and estimate how badly plain gradient descent will behave.

## Worked self-check answers

1. Differentiating $(1+e^{-x})^{-1}$ gives
   $e^{-x}/(1+e^{-x})^2=\sigma(1-\sigma)$. Its maximum is $1/4$.
   The full depth bound also contains every weight operator norm.
2. With $G:(32,128)$, $X^\top G:(512,128)$ and
   $GW^\top:(32,512)$. Shapes reject mistakes, while the chain rule proves
   these particular products.
3. Local backward rules need operands such as $X$ in $X^\top G$.
   Checkpointing recomputes selected forward values, trading arithmetic for memory.
4. Sum $(-y_i/p_i)p_i(\delta_{ij}-p_j)$ over $i$ and use
   $\sum_i y_i=1$ to obtain $p_j-y_j$.
5. Inspect the first nonfinite tensor: large exponentials suggest unstable
   logits; invalid logarithm/division suggests a domain error; rapidly growing
   Jacobian products suggest unstable updates. Lower LR is a diagnostic, not
   proof that no implementation bug exists.
6. All three eigenvalues are positive, so under the local smoothness assumptions
   this is a strict local minimum. The local quadratic has $\kappa=10^4$ and
   stability requires $0<\eta<.02$. At $\eta=.01$, the flat coordinate
   contracts by $.9999$ per step, needing roughly 10,000 steps for an
   $e^{-1}$ reduction in its magnitude.

## Where to go next

- [Linear Algebra](./linear-algebra.md) — the objects calculus differentiates.
- [Optimization Techniques](./optimization.md) — what to do with the gradient
  once you have it.
- [Probability](./probability.md) — where the losses you differentiate come from.
