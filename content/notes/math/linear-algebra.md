---
order: 1
description: >
  A comprehensive linear algebra reference for machine learning: vector spaces,
  linear maps, geometry, matrix factorizations, least squares, SVD, PCA,
  numerical stability, and worked applications.
meta: Math for ML · foundations to advanced · interactive
scripts:
  - https://cdn.jsdelivr.net/npm/d3@7
  - /assets/pages/linear-algebra.js
---

# Linear Algebra for Machine Learning

A dataset is a matrix, but linear algebra is not merely a notation for storing
it. It tells us which predictions a model can express, which parameters the data
can identify, how a transformation changes distances, and when a mathematically
correct calculation will fail on a computer. The same ideas connect a linear
regression fit to principal component analysis, the geometry of an embedding to
the stability of a neural network, and a low-rank approximation to an efficient
model update.

This chapter develops those connections from first principles. The scope is
finite-dimensional linear algebra for ML: real vector spaces, a short extension
to complex numbers, matrix geometry, exact and approximate systems,
factorizations, numerical algorithms, and their use in learning systems. It is
not a substitute for a separate course on infinite-dimensional functional
analysis or abstract tensor algebra.

The running convention is **one observation per row of a data matrix**, while
an individual mathematical vector is a column. Keeping that distinction explicit
prevents a surprising fraction of implementation errors.

```html
<details class="chapter-outline">
<summary>Chapter contents</summary>
<ol>
<li><a href="#1-a-map-of-the-subject">A map of the subject</a></li>
<li><a href="#2-scalars-arrays-and-shapes">Scalars, arrays, and shapes</a></li>
<li><a href="#3-vector-spaces-span-and-coordinates">Vector spaces, span, and coordinates</a></li>
<li><a href="#4-matrix-multiplication-as-a-linear-map">Matrix multiplication as a linear map</a></li>
<li><a href="#5-norms-inner-products-and-similarity">Norms, inner products, and similarity</a></li>
<li><a href="#6-rank-linear-systems-and-the-four-subspaces">Rank, linear systems, and the four subspaces</a></li>
<li><a href="#7-orthogonality-projections-and-hyperplanes">Orthogonality, projections, and hyperplanes</a></li>
<li><a href="#8-structured-matrices-and-efficient-operations">Structured matrices and efficient operations</a></li>
<li><a href="#9-determinants-trace-and-inverses">Determinants, trace, and inverses</a></li>
<li><a href="#10-eigenvalues-and-invariant-directions">Eigenvalues and invariant directions</a></li>
<li><a href="#11-quadratic-forms-curvature-and-positive-definiteness">Quadratic forms, curvature, and positive definiteness</a></li>
<li><a href="#12-singular-value-decomposition">Singular value decomposition</a></li>
<li><a href="#13-low-rank-approximation-and-spectral-energy">Low-rank approximation and spectral energy</a></li>
<li><a href="#14-covariance-and-mahalanobis-geometry">Covariance and Mahalanobis geometry</a></li>
<li><a href="#15-principal-component-analysis-from-objective-to-implementation">Principal component analysis, from objective to implementation</a></li>
<li><a href="#16-from-elimination-to-reliable-solvers">From elimination to reliable solvers</a></li>
<li><a href="#17-qr-factorization-and-least-squares">QR factorization and least squares</a></li>
<li><a href="#18-pseudoinverses-minimum-norm-and-ridge-regression">Pseudoinverses, minimum norm, and ridge regression</a></li>
<li><a href="#19-conditioning-and-finite-precision-reasoning">Conditioning and finite-precision reasoning</a></li>
<li><a href="#20-computing-with-positive-definite-and-large-matrices">Computing with positive-definite and large matrices</a></li>
<li><a href="#21-matrix-calculus-and-neural-network-shapes">Matrix calculus and neural-network shapes</a></li>
<li><a href="#22-attention-low-rank-adaptation-kernels-and-graphs">Attention, low-rank adaptation, kernels, and graphs</a></li>
<li><a href="#23-foundations-self-checks-with-worked-answers">Foundations: self-checks with worked answers</a></li>
<li><a href="#24-spectral-methods-self-checks-with-worked-answers">Spectral methods: self-checks with worked answers</a></li>
<li><a href="#25-numerical-and-ml-self-checks-with-worked-answers">Numerical and ML self-checks with worked answers</a></li>
<li><a href="#26-a-compact-decision-checklist">A compact decision checklist</a></li>
<li><a href="#27-further-study-and-connections">Further study and connections</a></li>
</ol>
</details>
```

## 1. A map of the subject

Three questions organize the chapter:

1. **What can the model represent?** Span, basis, rank, and nullspaces describe
   the set of possible outputs and the information a transformation destroys.
2. **Which approximation is best?** Inner products, projections, least squares,
   SVD, and PCA turn vague notions of similarity and compression into precise
   optimization problems.
3. **Can we compute it reliably?** Factorizations, conditioning, regularization,
   and floating-point error determine whether the answer survives finite
   precision.

For a first pass, follow the sections in order through PCA and least squares.
Then return to structured matrices, numerical conditioning, and the application
sections. The worked problems at the end require calculations and explanations,
not just recognition of terminology.

| Object or operation | Mathematical question | ML consequence |
|---|---|---|
| Column space of $X$ | Which vectors can $Xw$ produce? | Which training predictions are representable? |
| Nullspace of $X$ | Which parameter changes produce no output change? | Are fitted coefficients identifiable? |
| Inner product and norm | Which directions align, and how large are they? | Similarity, distance, penalties, and margins |
| Projection | Which representable vector is closest to a target? | Least-squares fitting and residuals |
| Eigenvalues of a symmetric matrix | How does a quadratic form vary by direction? | Curvature, covariance, and convergence |
| Singular values | How much does a map stretch independent directions? | Rank, compression, sensitivity, and regularization |
| Matrix factorization | How can a problem be reduced to simpler operations? | Stable, efficient solvers |

## 2. Scalars, arrays, and shapes

### Mathematical objects and their representations

A scalar is a single number. A vector $x\in\mathbb R^d$ has $d$ coordinates. A
matrix $A\in\mathbb R^{m\times d}$ represents a linear map from $d$ input
coordinates to $m$ output coordinates. These are distinct roles: a matrix can
also store observations without our immediately treating it as a transformation.

We write

$$
x=\begin{bmatrix}x_1\\\vdots\\x_d\end{bmatrix},\qquad
A=\begin{bmatrix}
a_{11}&\cdots&a_{1d}\\
\vdots&&\vdots\\
a_{m1}&\cdots&a_{md}
\end{bmatrix}.
$$

The transpose swaps indices: $(A^\top)_{ij}=A_{ji}$, so
$A^\top\in\mathbb R^{d\times m}$. Transposition is not inversion: it exchanges
the roles of rows and columns, not necessarily the action of a map and its undo
operation.

For $n$ observations and $d$ features, the data matrix is

$$
X=\begin{bmatrix}x_1^\top\\x_2^\top\\\vdots\\x_n^\top\end{bmatrix}
\in\mathbb R^{n\times d}.
$$

Here $x_i$ is the feature vector for observation $i$, not the $i$th feature across
all observations. A linear prediction vector is $Xw\in\mathbb R^n$ when
$w\in\mathbb R^d$. An intercept adds $b\mathbf1_n$, not a new multiplication
dimension.

```html
<figure class="widget">
  <div class="widget__k">Figure 01 · Matrix operations</div>
  <h4 class="widget__t">A dataset is a rectangular array</h4>
  <p class="widget__hint">Four observations and three measured features form \(X \in \mathbb{R}^{4\times3}\). Scalar multiplication changes every entry. Transposition swaps the observation and feature axes; it does not create new measurements.</p>
  <div class="widget__bar">
    <div class="widget__group">
      <label class="widget__lbl" for="scalar-input">Scalar</label>
      <input class="widget__num" type="number" step="0.5" id="scalar-input" value="2">
      <button class="widget__btn widget__btn--accent" type="button" id="scalar-multiply-btn">Multiply</button>
    </div>
    <span class="widget__sep" aria-hidden="true"></span>
    <button class="widget__btn" type="button" id="transpose-btn">Transpose</button>
    <span class="widget__sep" aria-hidden="true"></span>
    <button class="widget__btn" type="button" id="reset-matrix-btn">Reset</button>
  </div>
  <div class="widget__stage widget__stage--matrix" id="interactive-matrix"></div>
  <figcaption class="widget__cap">Rows are observations; columns are features. Applying a scalar twice compounds the scaling. Transposing twice returns the current matrix.</figcaption>
</figure>
```

### Tensor axes have meaning

In ML software, a tensor usually means a multidimensional array. An image batch
might have shape $(B,C,H,W)$; token activations might have shape $(B,T,d)$.
An axis has a semantic role as well as a length. Accidentally swapping two
equal-length axes can silently change a computation while preserving its shape.

A reshape changes how the same entries are grouped. A transpose or axis
permutation changes which index refers to which entry. They are not generally
interchangeable. Flattening an image preserves its pixel values but removes the
explicit two-dimensional neighborhood structure from the representation.

For a batched matrix product, write the contracted index before writing code:

$$
Y_{bti}=\sum_{j=1}^{d}H_{btj}W_{ji},
\qquad H:(B,T,d),\quad W:(d,k),\quad Y:(B,T,k).
$$

This is a linear operation on the last feature axis of each token, with the same
$W$ used across batches and positions.

### Shape-safe NumPy

NumPy's one-dimensional arrays do not distinguish row vectors from column
vectors. For `v.shape == (3,)`, `v.T.shape` is still `(3,)`. Use `v[:, None]`
for an explicit column and `v[None, :]` for an explicit row.

```python
import numpy as np

X = np.array([[1., 2., 3.], [4., 5., 6.]])  # (2, 3)
w = np.array([2., -1., 0.5])                # (3,)
y = X @ w                                 # (2,): [1.5, 6.0]
y_column = X @ w[:, None]                  # (2, 1)
assert np.allclose(y_column[:, 0], y)

# Multiplication and elementwise multiplication are different operations.
feature_scaled = X * w                     # (2, 3)
assert np.allclose(feature_scaled.sum(axis=1), y)

# Explicitly align the target axis before subtracting.
target = np.array([1., 5.])
residual = y_column - target[:, None]      # (2, 1)
assert residual.shape == (2, 1)
```

Subtracting `(n,)` from `(n,1)` can produce `(n,n)` through broadcasting, rather
than the intended $n$ residuals. Compatible shapes are not proof that the
operation is semantically correct. NumPy compares trailing axes, accepting equal
sizes or a size of one; its official [broadcasting documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)
gives the precise rules.

## 3. Vector spaces, span, and coordinates

### Linear combinations and subspaces

A linear combination of $v_1,\ldots,v_k$ is
$\alpha_1v_1+\cdots+\alpha_kv_k$. Their **span** is the set of every vector
obtainable by choosing the scalar coefficients. Two nonparallel vectors in
$\mathbb R^3$ span a plane through the origin, not all of three-dimensional
space.

A vector space is a set on which addition and scalar multiplication obey the
usual algebraic rules. A subset is a **linear subspace** if it contains zero and
is closed under addition and scalar multiplication. Equivalently, every linear
combination of its elements stays in the subset.

For example,

$$
S=\{(x,y,z):x+y+z=0\}
$$

is a subspace. If $u$ and $v$ each have coordinate sum zero, so does
$\alpha u+\beta v$. The set $x+y+z=1$ is not a subspace because it does not
contain zero. It is an **affine** plane: a translated subspace.

Polynomials of degree at most two form a vector space as well. Relative to the
basis $(1,t,t^2)$, the polynomial $3-2t+t^2$ has coordinates $(3,-2,1)$.
The entries in a vector are coordinates of an object, not necessarily physical
positions. Feature maps exploit exactly this freedom.

### Independence, basis, and dimension

Vectors are linearly independent if

$$
\alpha_1v_1+\cdots+\alpha_kv_k=0
\quad\Longrightarrow\quad
\alpha_1=\cdots=\alpha_k=0.
$$

Dependence means at least one vector can be expressed using the others. The
zero vector cannot belong to an independent set. More than $d$ vectors in
$\mathbb R^d$ must be dependent, although fewer than $d$ can also be dependent.

A **basis** is an independent spanning set. Independence gives uniqueness of
coordinates; spanning gives existence. The number of basis vectors is the
dimension of the space. A matrix can have thousands of columns and still have a
column space of dimension two if most columns repeat information.

Consider

$$
v_1=\begin{bmatrix}1\\1\\0\end{bmatrix},\qquad
v_2=\begin{bmatrix}0\\1\\1\end{bmatrix},\qquad
v_3=\begin{bmatrix}1\\2\\1\end{bmatrix}.
$$

Since $v_3=v_1+v_2$, the three vectors are dependent. The first two are
independent: their first and third coordinates force both coefficients to zero
in a zero combination. They form a basis for the span, which has dimension two.
The vector $(2,5,3)^\top$ has coordinates $(2,3)^\top$ in that basis.

The first two columns of a matrix are not automatically a basis. One must choose
independent columns. Row reduction identifies pivot column indices; those
indices select basis columns from the **original** matrix, because row
operations generally change the column space itself.

### Changing coordinates is not changing the vector

Let $B$ be an invertible square matrix whose columns are a basis. The coordinate
vector $c$ and ordinary coordinates $x$ satisfy $x=Bc$, hence $c=B^{-1}x$.

For

$$
B=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
c=\begin{bmatrix}2\\3\end{bmatrix},
\quad x=Bc=\begin{bmatrix}5\\3\end{bmatrix}.
$$

The same geometric vector is described by $(2,3)$ in the new basis and $(5,3)$
in the standard basis. A non-orthonormal basis changes the coordinate expression
for length:

$$
\|x\|_2^2=c^\top B^\top Bc,
$$

not generally $c^\top c$. Here the true squared length is $34$, while the sum
of squared new coordinates is $13$. This distinction matters when comparing
feature representations after a non-orthogonal transformation.

## 4. Matrix multiplication as a linear map

### Columns describe where the basis vectors go

A map $T$ is linear when
$T(\alpha x+\beta y)=\alpha T(x)+\beta T(y)$. A finite-dimensional linear map
is completely determined by what it does to basis vectors. If
$A=[a_1\ \cdots\ a_d]$, then

$$
Ax=x_1a_1+\cdots+x_da_d.
$$

Thus the columns of $A$ are the images of the standard input basis. From the
row viewpoint, $(Ax)_i$ is the dot product between the $i$th row and $x$.
These are two descriptions of the same computation, each useful for a different
question: columns explain representability, rows explain measurements.

Take

$$
A=\begin{bmatrix}2&1\\0&1\end{bmatrix},\qquad
x=\begin{bmatrix}3\\-1\end{bmatrix}.
$$

Then $Ax=(5,-1)^\top$. By columns, this is
$3(2,0)^\top-(1,1)^\top$. By rows, the first output is $2(3)+1(-1)$,
and the second is $0(3)+1(-1)$.

An affine map $Ax+b$ is linear only when $b=0$. For example, translating every
point by $(1,0)$ sends zero to a nonzero point and therefore fails linearity.
Many layers called "linear" in software are actually affine layers.

### Composition, order, and the transpose

If $B:\mathbb R^k\to\mathbb R^d$ and
$A:\mathbb R^d\to\mathbb R^m$, then $AB$ represents applying $B$ first and
$A$ second. Its entries are

$$
(AB)_{ij}=\sum_{\ell=1}^{d}A_{i\ell}B_{\ell j}.
$$

Matrix multiplication is associative but usually not commutative. With

$$
A=\begin{bmatrix}2&0\\0&1\end{bmatrix},\qquad
B=\begin{bmatrix}1&1\\0&1\end{bmatrix},
$$

we obtain $AB=\begin{bmatrix}2&2\\0&1\end{bmatrix}$ but
$BA=\begin{bmatrix}2&1\\0&1\end{bmatrix}$. Scaling horizontally before a
shear does not have the same effect as scaling afterward.

Three identities are worth deriving rather than memorizing blindly:

$$
(AB)^\top=B^\top A^\top,\qquad
x^\top Ay=(A^\top x)^\top y,\qquad
(AB)^{-1}=B^{-1}A^{-1}
$$

where the last expression requires square invertible factors. Transposition
reverses order because an output-input pairing can be moved backward through
the map. The same reversal is central to backpropagation.

### Outer products, rank-one maps, and elementwise products

For $u\in\mathbb R^m$ and $v\in\mathbb R^d$, the outer product
$uv^\top\in\mathbb R^{m\times d}$ acts as

$$
(uv^\top)x=u(v^\top x).
$$

It first extracts one scalar measurement, then emits a multiple of $u$. Its
image is a line when both vectors are nonzero, so it has rank one. In contrast,
the inner product $u^\top v$ is a scalar and requires matching dimensions.

With $u=(1,2)^\top$ and $v=(3,-1)^\top$,
$uv^\top=\begin{bmatrix}3&-1\\6&-2\end{bmatrix}$. Every output lies along
$(1,2)^\top$, regardless of the input.

The Hadamard product $(A\odot B)_{ij}=A_{ij}B_{ij}$ is elementwise. It is used
for masks, gates, and activation derivatives; it is not composition of linear
maps. A dropout mask acts elementwise even though a dense layer uses a matrix
product.

An ordinary product also has an outer-product decomposition:

$$
AB=\sum_{j=1}^{d}A_{:j}B_{j:}.
$$

Each intermediate dimension contributes one rank-one map. This immediately
explains why a narrow intermediate layer limits rank.

## 5. Norms, inner products, and similarity

### A norm measures size

A norm is nonnegative, zero only at zero, homogeneous under scalar
multiplication, and satisfies the triangle inequality. Common vector norms are

$$
\|x\|_1=\sum_i|x_i|,\qquad
\|x\|_2=\sqrt{\sum_i x_i^2},\qquad
\|x\|_\infty=\max_i|x_i|.
$$

For $x=(3,-4)$, these are $7$, $5$, and $4$. The unit balls are a diamond,
circle, and axis-aligned square in two dimensions. A penalty's geometry changes
which solutions an optimization problem favors. The corners of an $\ell_1$
ball help explain sparse solutions, but do not imply that every problem with
an $\ell_1$ penalty produces a useful sparse model.

The frequently used $\|x\|_0$ counts nonzero coordinates. It is not a norm:
scaling a nonzero vector by two does not double that count.

In $\mathbb R^d$,

$$
\|x\|_\infty\leq\|x\|_2\leq\|x\|_1,
\qquad
\|x\|_1\leq\sqrt d\,\|x\|_2.
$$

These inequalities show why dimensions matter when comparing thresholds across
norms. A small per-coordinate error need not imply a small total error in a very
high-dimensional vector.

### The dot product includes both angle and magnitude

The Euclidean inner product is $x^\top y=\sum_i x_i y_i$. Expanding
$\|x-y\|_2^2$ gives

$$
\|x-y\|_2^2=\|x\|_2^2+\|y\|_2^2-2x^\top y.
$$

Combining this with the geometric cosine rule yields

$$
x^\top y=\|x\|_2\|y\|_2\cos\theta.
$$

The Cauchy-Schwarz inequality
$|x^\top y|\leq\|x\|_2\|y\|_2$ follows, for example, by requiring
$\|x-ty\|_2^2\geq0$ for every real $t$. Equality holds when the two vectors
are linearly dependent, including the zero-vector cases.

For $a=(3,2)^\top$ and $b=(1,4)^\top$, the dot product is $11$, but the
cosine similarity is $11/\sqrt{221}\approx0.740$. Multiplying $b$ by ten
multiplies the dot product by ten without changing the angle. A raw dot product
is therefore not a magnitude-independent similarity score.

```html
<figure class="widget">
  <div class="widget__k">Figure 02 · Dot product</div>
  <h4 class="widget__t">Magnitude, angle, and alignment</h4>
  <p class="widget__hint">The dot product combines lengths and alignment. Cosine similarity removes the lengths, but is undefined when either vector is zero. The amber wedge marks the smaller angle, using equal scales on both axes.</p>
  <div class="widget__eq">\[a \cdot b = \|a\|\,\|b\|\cos(\theta)\]</div>
  <div class="widget__split">
    <div class="widget__stage" id="dot-product-viz"></div>
    <div class="widget__side">
      <div class="widget__vec">
        <span class="widget__dot widget__dot--a" aria-hidden="true"></span>
        <span>a = [</span>
        <input class="widget__num" type="number" step="0.1" id="vec-a-x" value="3" aria-label="vector a, x component">
        <span>,</span>
        <input class="widget__num" type="number" step="0.1" id="vec-a-y" value="2" aria-label="vector a, y component">
        <span>]</span>
      </div>
      <div class="widget__vec">
        <span class="widget__dot widget__dot--b" aria-hidden="true"></span>
        <span>b = [</span>
        <input class="widget__num" type="number" step="0.1" id="vec-b-x" value="1" aria-label="vector b, x component">
        <span>,</span>
        <input class="widget__num" type="number" step="0.1" id="vec-b-y" value="4" aria-label="vector b, y component">
        <span>]</span>
      </div>
      <div class="widget__readout">
        <div class="widget__val">a · b = <span id="dot-product-val">11.00</span></div>
        <div class="widget__meta">Angle <strong id="angle-val">42.3°</strong></div>
      </div>
      <p class="widget__note" id="dot-product-explanation" aria-live="polite"></p>
    </div>
  </div>
</figure>
```

Cosine similarity divides by both norms. It is undefined if either vector is
zero. Libraries sometimes add an epsilon or return a convention-dependent value;
that numerical convention does not create a geometric angle for the zero
vector.

For unit vectors $u$ and $v$, squared Euclidean distance is
$2-2u^\top v$. Thus maximizing cosine similarity and minimizing Euclidean
distance give the same ranking **after normalization**. Without normalization
they can disagree. Whether normalization is desirable depends on whether vector
magnitude contains information in the learned representation.

Orthogonality means zero inner product. It does not, by itself, establish
statistical independence or a lack of semantic relationship. Those are claims
about a distribution or a representation, not consequences of an angle alone.

### Cross products are a special three-dimensional operation

For $u,v\in\mathbb R^3$, the cross product is

$$
u\times v=\begin{bmatrix}
u_2v_3-u_3v_2\\u_3v_1-u_1v_3\\u_1v_2-u_2v_1
\end{bmatrix}.
$$

It is perpendicular to both inputs, with length
$\|u\|\|v\||\sin\theta|$, the area of their spanned parallelogram. The
right-hand rule chooses its orientation; reversing the inputs reverses its
sign. For $u=(1,0,0)^\top$ and $v=(0,2,0)^\top$, the result is
$(0,0,2)^\top$. It is useful for surface normals and rotations in geometric
vision, but it is not the high-dimensional similarity operation used for word
embeddings. The inner and outer products generalize directly to arbitrary
dimensions; the familiar vector-valued cross product does not generalize in the
same way.

### Other inner products and complex vectors

An inner product can be defined by $\langle x,y\rangle_M=x^\top My$ when
$M$ is symmetric positive definite. The corresponding norm weights directions
according to $M$. If $M$ is merely positive semidefinite, nonzero vectors can
have zero length, giving a seminorm instead. This is the geometric foundation
of weighted least squares and Mahalanobis distance, developed later.

For complex vectors, use the conjugate transpose:
$\langle x,y\rangle=x^*y$ and $\|x\|_2^2=x^*x$. Using an ordinary transpose
would give $(i)^2=-1$ as a supposed squared length. Complex arithmetic appears
in Fourier methods and in the eigenvalues of real nonsymmetric matrices, even
when the original data is real.

### Matrix norms measure different things

The Frobenius norm is the Euclidean norm of all entries:

$$
\|A\|_F^2=\sum_{ij}|a_{ij}|^2=\operatorname{tr}(A^* A).
$$

Here $A^*$ is conjugate transpose; for real matrices it equals $A^\top$. The remaining matrix formulas return to the real setting unless stated otherwise.

The induced operator norm instead asks for the largest output relative to the
input:

$$
\|A\|_p=\sup_{x\ne0}\frac{\|Ax\|_p}{\|x\|_p}.
$$

For $p=2$ this is the largest singular value, not the sum of entry magnitudes.
For $p=1$ it is the maximum absolute column sum; for $p=\infty$ it is the
maximum absolute row sum. Induced norms obey
$\|AB\|_p\leq\|A\|_p\|B\|_p$, so they bound amplification through
composed layers.

For $A=\operatorname{diag}(3,4)$, $\|A\|_2=4$ while $\|A\|_F=5$.
The first describes worst-direction stretching; the second accumulates energy
across directions. The nuclear norm, the sum of singular values, serves yet
another role as a convex surrogate for rank in some optimization problems.

## 6. Rank, linear systems, and the four subspaces

### Representability and uniqueness are different questions

Solving $Ax=b$, with $A\in\mathbb R^{m\times d}$, asks whether $b$ is a
linear combination of the columns of $A$. Define the column space
$\mathcal C(A)=\{Ax:x\in\mathbb R^d\}$ and the nullspace
$\mathcal N(A)=\{x:Ax=0\}$. The **rank** $r$ is the dimension of the column
space and also of the row space.

There is an exact solution precisely when $b\in\mathcal C(A)$. If $x_0$ is
one solution, every solution has the form

$$
x=x_0+z,\qquad z\in\mathcal N(A).
$$

To see completeness, subtract the equations for any two solutions:
$A(x-x_0)=b-b=0$. There is a unique solution, when one exists, precisely when
the nullspace contains only zero.

The rank-nullity identity is

$$
\operatorname{rank}(A)+\dim\mathcal N(A)=d.
$$

The right side is the **number of input coordinates**, not the number of
equations. A map cannot create independent output directions that were not
present in its input: $r\leq\min(m,d)$.

### A complete worked system

Let

$$
A=\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix},\qquad
b=\begin{bmatrix}4\\8\end{bmatrix}.
$$

The second equation repeats the first. The rank is one; three input parameters
are constrained in only one independent way. The solution family is

$$
x=\begin{bmatrix}4\\0\\0\end{bmatrix}
+s\begin{bmatrix}-2\\1\\0\end{bmatrix}
+t\begin{bmatrix}-3\\0\\1\end{bmatrix}.
$$

The last two vectors form a nullspace basis. Changing $b$ to $(4,9)^\top$
makes the equations inconsistent: the second output must always be twice the
first. More optimization iterations cannot overcome that representational
restriction.

For a feature matrix, dependent columns imply non-identifiable coefficients.
Duplicating a feature makes its two weights individually ambiguous, even if the
sum of those weights and the fitted predictions are uniquely determined.
Regularization can select one coefficient vector, but it does not retroactively
add information to the data.

### Row reduction with pivots, free variables and inconsistency

Consider the augmented system

$$
\left[\begin{array}{ccc|c}0&1&2&3\\1&2&3&4\\2&4&6&8\end{array}\right].
$$

Swap $R_1,R_2$, subtract $2R_1$ from $R_3$, then subtract $2R_2$ from
$R_1$. These reversible operations give

$$
\left[\begin{array}{ccc|c}1&0&-1&-2\\0&1&2&3\\0&0&0&0\end{array}\right].
$$

Columns one and two are pivots; $x_3=t$ is free, giving
$x=(-2+t,3-2t,t)^\top$. A nullspace basis is $(1,-2,1)^\top$.
Use the original first two columns $(0,1,2)^\top,(1,2,4)^\top$ for the
column space, and the nonzero reduced rows $(1,0,-1),(0,1,2)$ for the row
space. The left nullspace is spanned by $(0,-2,1)^\top$, which records
the dependency between equations two and three. Replacing the final target
8 with 9 produces the last row $[0\ 0\ 0\mid1]$: no solution exists.
Thus elimination identifies constraints and their compatibility, not only a
candidate coefficient vector.

### The four fundamental subspaces

| Subspace | Ambient space | Dimension | Interpretation |
|---|---|---|---|
| Column space $\mathcal C(A)$ | $\mathbb R^m$ | $r$ | Representable outputs |
| Left nullspace $\mathcal N(A^\top)$ | $\mathbb R^m$ | $m-r$ | Output directions orthogonal to every possible output |
| Row space $\mathcal C(A^\top)$ | $\mathbb R^d$ | $r$ | Input directions the map can distinguish |
| Nullspace $\mathcal N(A)$ | $\mathbb R^d$ | $d-r$ | Input directions sent to zero |

The orthogonal decompositions are

$$
\mathbb R^d=\mathcal C(A^\top)\oplus\mathcal N(A),\qquad
\mathbb R^m=\mathcal C(A)\oplus\mathcal N(A^\top).
$$

For the worked matrix, the row space is the line spanned by $(1,2,3)^\top$;
its orthogonal complement is the two-dimensional nullspace. The column space
is the line spanned by $(1,2)^\top$; its orthogonal complement is spanned by
$(-2,1)^\top$.

The orthogonality is not a coincidence: if $Az=0$, then every row of $A$ has
zero inner product with $z$. Dimension counting turns that inclusion into an
equality of orthogonal complements. MIT's [four-subspace lecture](https://web.mit.edu/18.06/www/Spring16/lecture16.pdf)
is a useful independent reference for this structure.

### Full rank and invertibility

Full column rank ($r=d$) means the map is injective: two different inputs cannot
have the same output. Full row rank ($r=m$) means it is surjective onto
$\mathbb R^m$: every target output is attainable. A square matrix is invertible
exactly when it has both properties.

For a square $d\times d$ matrix, the following are equivalent: rank $d$,
trivial nullspace, independent columns, a nonzero determinant, zero not being an
eigenvalue, and existence of an inverse. For rectangular matrices, these square
matrix equivalences must not be applied indiscriminately.

Finally,

$$
\operatorname{rank}(AB)\leq\min(\operatorname{rank}(A),\operatorname{rank}(B)).
$$

A linear bottleneck of width $k$ cannot represent a matrix of rank larger than
$k$. Stacking linear layers without nonlinearities still produces one linear
map, even if the factorized parameterization changes optimization behavior.

## 7. Orthogonality, projections, and hyperplanes

### Orthonormal coordinates simplify geometry

Vectors $q_1,\ldots,q_k$ are orthonormal when $q_i^\top q_j$ equals one for
$i=j$ and zero otherwise. For $Q=[q_1\ \cdots\ q_k]$, this means
$Q^\top Q=I_k$.

If $Q$ is square, it is an orthogonal matrix: $Q^{-1}=Q^\top$ and
$QQ^\top=I$. It preserves inner products and lengths because
$(Qx)^\top(Qy)=x^\top y$. Orthogonal transformations include reflections as
well as rotations. A tall $Q$ with orthonormal columns is an isometric embedding,
but $QQ^\top$ is then a projection, not the identity on the entire output space.

Given independent $v_1,v_2$, Gram-Schmidt starts with

$$
q_1=\frac{v_1}{\|v_1\|_2},\qquad
u_2=v_2-q_1(q_1^\top v_2),\qquad
q_2=\frac{u_2}{\|u_2\|_2}.
$$

For $v_1=(1,1)^\top$ and $v_2=(1,0)^\top$, the result is
$q_1=(1,1)^\top/\sqrt2$ and $q_2=(1,-1)^\top/\sqrt2$. If a residual becomes
zero in exact arithmetic, the new vector is dependent on those already
processed. Near-zero residuals in floating point require more care; the QR
section explains why naive Gram-Schmidt is not the default numerical algorithm.

### Deriving projection onto a line

To approximate $x$ by a point $\alpha u$ on the line through a nonzero $u$,
minimize

$$
f(\alpha)=\|x-\alpha u\|_2^2
=x^\top x-2\alpha u^\top x+\alpha^2u^\top u.
$$

Setting the derivative to zero gives

$$
\widehat x=\frac{u^\top x}{u^\top u}u,
\qquad
P_u=\frac{uu^\top}{u^\top u}.
$$

The residual $r=x-\widehat x$ is orthogonal to $u$. With $u=(1,2)^\top$ and
$x=(3,1)^\top$, the coefficient is $5/5=1$, so $\widehat x=(1,2)^\top$ and
$r=(2,-1)^\top$. The squared lengths satisfy $10=5+5$, a direct instance of
the Pythagorean theorem.

### Projection onto a subspace

For an orthonormal basis matrix $Q$, projection is $P=QQ^\top$. Its defining
properties are symmetry and idempotence:

$$
P^\top=P,\qquad P^2=P.
$$

Idempotence says a vector already projected onto the subspace does not change
on projection again. Symmetry distinguishes an orthogonal projection from a
general oblique projection. For example,
$\begin{bmatrix}1&1\\0&0\end{bmatrix}$ is idempotent but not symmetric: its
residual is not generally perpendicular to its output line.

If $B$ has independent but non-orthonormal columns, then

$$
P=B(B^\top B)^{-1}B^\top.
$$

This formula follows by making the residual orthogonal to every column of $B$.
It describes the geometry; computing a QR factorization and applying
$Q(Q^\top x)$ is usually preferable to explicitly forming either an inverse
or the potentially huge projector. For dependent columns, a pseudoinverse or a
rank-revealing factorization is needed.

Every orthogonal projection has eigenvalues only zero or one: from
$Pv=\lambda v$ and $P^2=P$, we obtain $\lambda^2=\lambda$. Directions in the
subspace survive; orthogonal directions disappear.

### Hyperplanes, distances, and margins

A hyperplane is

$$
H=\{x:w^\top x+b=0\},\qquad w\ne0.
$$

The normal vector is $w$, not a vector running along the boundary. Its signed
distance from a point $x$ is

$$
\frac{w^\top x+b}{\|w\|_2},
$$

and the closest point on the plane is

$$
x_H=x-\frac{w^\top x+b}{\|w\|_2^2}w.
$$

Substitution verifies that $w^\top x_H+b=0$, and the displacement is parallel
to the normal, so it is the shortest route. With $w=(3,4)^\top$, $b=-10$, and
$x=(2,2)^\top$, the score is $4$, the signed distance is $4/5$, and the nearest
boundary point is $(1.52,1.36)^\top$.

Multiplying both $w$ and $b$ by a positive scalar leaves the boundary and class
predictions unchanged, but scales the raw score. A classifier score is not
automatically a distance or a probability. With a negative scalar the boundary
stays the same but the positive and negative class labels swap.

```html
<figure class="widget">
  <div class="widget__k">Figure 03 · Linear classifier</div>
  <h4 class="widget__t">A normal vector defines a decision boundary</h4>
  <p class="widget__hint">Circles have label \(-1\); triangles have label \(+1\). This classifier predicts \(+1\) when \(w^T x+b\geq0\). Its unit normal is \(w=[\cos\theta,\sin\theta]^T\). The centered intercept \(c\) gives \(w^T(x-[5,5]^T)+c=0\), so \(b=c-5w_1-5w_2\). The angle changes orientation; the intercept translates the boundary.</p>
  <div class="widget__stage" id="interactive-classifier"></div>
  <div class="widget__legend">
    <span><i class="widget__key widget__key--line" aria-hidden="true"></i>Decision boundary \(\vec{w} \cdot \vec{x} + b = 0\)</span>
  </div>
</figure>
```

For a hard-margin SVM, choosing the normalization
$y_i(w^\top x_i+b)\geq1$ makes the separation between the supporting planes
$2/\|w\|_2$. Minimizing $\tfrac12\|w\|_2^2$ then maximizes that margin,
provided the data is linearly separable. Soft-margin formulations introduce
slack rather than claiming every dataset admits a perfect separator. See
[SVMs and kernels](../ml/svm-and-kernels.md) for the statistical learning problem;
the geometric distance formula above remains the same.

## 8. Structured matrices and efficient operations

### Diagonal, triangular, permutation, and orthogonal matrices

A diagonal matrix scales coordinates independently. Multiplying by
$D=\operatorname{diag}(d_1,\ldots,d_n)$ needs only elementwise scaling, not a
general dense matrix multiplication. Its inverse exists when every $d_i$ is
nonzero and has diagonal entries $1/d_i$.

A permutation matrix reorders coordinates. Its inverse is its transpose. A
triangular system can be solved one coordinate at a time by forward or backward
substitution; this is why factorizations that produce triangular factors are so
useful. A block-diagonal matrix represents independent transformations on
separate groups of coordinates.

An orthogonal basis change expresses a square transformation as $Q^\top A Q$.
For a general invertible basis $B$, the corresponding representation is
$B^{-1}AB$. Such **similarity transformations** preserve eigenvalues and trace.
They are not the same as congruence transformations $B^\top A B$, which
describe how a quadratic form changes coordinates and preserve definiteness
when $B$ is invertible.

### Block elimination and the Schur complement

Block matrices are ordinary matrices partitioned to expose structure. Consider

$$
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}f\\g\end{bmatrix}.
$$

If $A$ is invertible, the first row gives $x=A^{-1}(f-By)$. Substitution into
the second gives

$$
(D-CA^{-1}B)y=g-CA^{-1}f.
$$

The matrix $S=D-CA^{-1}B$ is the Schur complement of $A$. The formulas use
inverse notation to express elimination; an implementation solves systems with
$A$ rather than explicitly forming $A^{-1}$.

For the scalar-block example
$\begin{bmatrix}2&1\\1&3\end{bmatrix}(x,y)^\top=(1,2)^\top$,
the reduced equation is $(3-1/2)y=2-1/2$, giving $y=0.6$ and $x=0.2$.
The same logic scales to grouped parameters, constrained optimization systems,
and elimination of latent variables.

```html
<h3 id="low-rank-updates-sherman-morrison-and-woodbury">Low-rank updates: Sherman-Morrison and Woodbury</h3>
```

Suppose a fitted system changes from $A$ to $M=A+UCV^T$, where $A$ is
$n\times n$, $U,V$ are $n\times r$, and $C$ is $r\times r$. When $r$ is small,
the change is restricted to a low-dimensional subspace even though $M$ need not
have low rank. Updating observations, covariance models and a ridge system can
create this structure. A LoRA weight update also has a low-rank form, but a
forward neural-network layer normally multiplies by its weights; it does not
therefore need a Woodbury inverse.

Derive a solve instead of memorizing an inverse formula. To solve $Mx=b$, first
solve $Ay=b$ and $AZ=U$. The original equation becomes

$$
x=y-ZCV^Tx.
$$

Introduce $q=CV^Tx$. Substitution gives the small system

$$
(I_r+CV^TZ)q=CV^Ty,\qquad x=y-Zq.
$$

This form requires invertible $A$ and $I_r+CV^TA^{-1}U$, but **does not require
invertible $C$**. The matrix determinant identity

$$
\det(A+UCV^T)=\det(A)\det(I_r+CV^TA^{-1}U)
$$

explains the equivalence of the two nonsingularity conditions when $A$ is
invertible. If $C$ is invertible, rearranging the same algebra yields the familiar
Woodbury identity with $(C^{-1}+V^TA^{-1}U)^{-1}$. That version should not be
applied blindly when the update contains zero directions.

For $r=1$, write the update as $A+uv^T$. The correction reduces to

$$
x=y-z\frac{v^Ty}{1+v^Tz},\qquad Ay=b,\quad Az=u.
$$

The denominator is a mathematical condition, not a detail to handle by adding an
arbitrary epsilon. For $A=I_2$, $u=(1,0)^T$, and $v=(-1,0)^T$, the updated
matrix is singular even though the original matrix was perfectly conditioned.
Replacing $-1$ by $-1+\epsilon$ produces a nearly singular system for small
positive $\epsilon$.

The [downloadable spectral/update lab](/assets/examples/spectral_updates.py)
uses library solves, combines $b$ and $U$ into one multiple-right-hand-side solve,
and tests nonsymmetric updates, singular $C$, multiple right-hand sides and
singular updated systems. It also reconstructs the dense updated matrix for a
residual check; this is a small correctness experiment, not a memory benchmark.
Run it in the [pinned NumPy/SciPy example environment](/assets/examples/requirements.txt):

```bash
python spectral_updates.py
```

With an existing dense factorization of $A$, the extra solves cost roughly
$O(n^2r)$, the reduced products $O(nr^2)$, and the small factorization $O(r^3)$,
before accounting for additional right-hand sides. If you refactor $A$ for every
single update, that original $O(n^3)$ cost remains. Sparse structure, batching and
factor reuse determine actual speed; low rank alone is not a timing result.

Repeated updates may accumulate numerical error. Cancellation in $y-Zq$, a
fragile base solve and a poorly scaled reduced system can all matter. Check the
residual against the **updated** system, and periodically compare with a fresh
factorization. See [the numerical update diagnostics](./numerical-methods.md#structured-solves-still-need-error-checks)
for why even the small system's condition number is not a sufficient certificate.

### Kronecker products and vectorization

The Kronecker product $A\otimes B$ replaces every entry $a_{ij}$ by a block
$a_{ij}B$. If $A$ is $m\times n$ and $B$ is $p\times q$, its shape is
$mp\times nq$. It differs from both ordinary and elementwise multiplication.

With **column-major** vectorization, which stacks a matrix's columns,

$$
\operatorname{vec}(AXB)=(B^\top\otimes A)\operatorname{vec}(X).
$$

Each entry on either side sums the same products $A_{ij}X_{jk}B_{k\ell}$;
only the indexing changes. The ordering convention is essential: NumPy's
default flattening is row-major, so use `order="F"` for this identity.

```python
import numpy as np

A = np.array([[1., 2.], [0., 1.]])
X = np.array([[1., 3.], [2., 4.]])
B = np.array([[2., 0.], [1., 1.]])
left = (A @ X @ B).reshape(-1, order="F")
right = np.kron(B.T, A) @ X.reshape(-1, order="F")
assert np.allclose(left, right)
```

The identity explains separable operators and structured curvature
approximations, but materializing the Kronecker matrix can be wasteful. Applying
$X\mapsto AXB$ directly often uses far less memory.

### Sparse and matrix-free computation

A sparse matrix has relatively few stored nonzero entries. Matrix-vector
multiplication can then cost proportional to the number of nonzeros, rather
than the product of its dimensions. Sparsity is a storage and computational
property; it does not imply low rank. The identity matrix is extremely sparse
and full rank, whereas an outer product of dense vectors is dense and rank one.

Some algorithms need only the functions $v\mapsto Av$ and $u\mapsto A^\top u$,
not an explicitly stored matrix. For a very large least-squares problem,
$v\mapsto X^\top(Xv)$ can be evaluated without constructing $X^\top X$.
This avoids storing a dense $d\times d$ Gram matrix, although it does not erase
the conditioning issues associated with that operator. Later sections discuss
which factorization or iterative solver fits which problem.


## 9. Determinants, trace, and inverses

A square matrix has several useful numerical summaries, but they answer different questions. The determinant measures signed volume change, the trace sums diagonal action, and the inverse reverses a transformation when reversal is possible. None of these, alone, tells you everything about a matrix.

### The determinant is a volume multiplier

For a two-dimensional transformation,

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},\qquad \det A=ad-bc.
$$

Its columns are the transformed coordinate directions. The parallelogram they span has area $|ad-bc|$. In $d$ dimensions, the same idea gives a volume multiplier $|\det A|$. A negative determinant reverses orientation; a zero determinant collapses at least one dimension.

Consider

$$
A=\begin{bmatrix}2&1\\0&3\end{bmatrix}.
$$

The unit square becomes a sheared parallelogram of area $6$. The shear changes its shape, but the determinant is still the product of the diagonal entries. By contrast, $B=\begin{bmatrix}2&4\\1&2\end{bmatrix}$ has determinant zero: its second column is twice its first, so every output lies on one line.

Three identities connect this geometric story to computation:

$$
\det(AB)=\det A\det B,\qquad
\det(A^T)=\det A,\qquad
\det(A^{-1})=\frac1{\det A}.
$$

The last identity assumes an inverse exists. For a square matrix, nonzero determinant, full rank, trivial null space, and invertibility are equivalent in exact arithmetic.

**A determinant is not a conditioning test.** The matrix $\operatorname{diag}(10^8,10^{-8})$ has determinant $1$, yet stretches one direction enormously and nearly erases another. Its spectral condition number is $10^{16}$. Conversely, $0.1I_{100}$ has determinant $10^{-100}$ but condition number $1$. Absolute volume change is not the same as relative sensitivity.

Large products of eigenvalues can overflow or underflow. When a model needs a log determinant, compute it through a stable factorization or a sign-and-log-determinant routine, not by computing a determinant and then taking its logarithm. For an SPD matrix $C=LL^T$, for example,

$$
\log\det C=2\sum_i\log L_{ii}.
$$

### Trace collects diagonal contributions

The trace is $\operatorname{tr}(A)=\sum_i A_{ii}$. For compatible rectangular factors,

$$
\operatorname{tr}(AB)=\operatorname{tr}(BA).
$$

Indeed, both sides sum the same products $A_{ij}B_{ji}$, only in a different order. This permits cyclic rotations of factors inside a trace, but not arbitrary reordering. In general, $\operatorname{tr}(ABC)$ need not equal $\operatorname{tr}(ACB)$.

Trace turns several matrix expressions into familiar scalar quantities:

$$
\|A\|_F^2=\operatorname{tr}(A^TA),\qquad
x^TAx=\operatorname{tr}(Axx^T).
$$

For a covariance matrix, trace is total feature variance. Both trace and determinant are unchanged by a change of basis $S^{-1}AS$. They equal the sum and product of the eigenvalues, respectively, counting algebraic multiplicity, even if the matrix is not diagonalizable. Individual diagonal entries are basis-dependent; their sum is not.

### An inverse is an operation, not a default algorithm

If $A$ is invertible, $A^{-1}A=AA^{-1}=I$. The formula $x=A^{-1}b$ explains the solution of $Ax=b$, but code should normally solve the system directly. Explicit inversion computes more than one right-hand side needs and adds an avoidable numerical step.

For $A=\begin{bmatrix}2&1\\1&1\end{bmatrix}$,

$$
A^{-1}=\begin{bmatrix}1&-1\\-1&2\end{bmatrix}.
$$

Applying the inverse to $b=(5,3)^T$ yields $(2,1)^T$, which you can verify by multiplying by $A$. The distinction matters in machine learning: inverse notation appears in regression, Gaussian densities, and second-order optimization, but the implementation usually uses a factorization and triangular solves. A pseudoinverse extends useful solution behavior to rectangular or singular matrices; it does not make a noninvertible map genuinely reversible.

## 10. Eigenvalues and invariant directions

An eigenvector is a nonzero direction that a square transformation does not rotate away from its own line:

$$
Av=\lambda v,\qquad v\ne0.
$$

The scalar $\lambda$ describes expansion, contraction, reversal, or collapse along that direction. The zero vector is excluded because $A0=\lambda0$ holds for every $\lambda$ and reveals nothing.

### A complete two-dimensional calculation

Let

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

An eigenvector must lie in the null space of $A-\lambda I$, so this matrix must be singular:

$$
\det(A-\lambda I)=(2-\lambda)^2-1
=(\lambda-3)(\lambda-1)=0.
$$

For $\lambda=3$, the equation $-v_1+v_2=0$ gives direction $(1,1)^T$. For $\lambda=1$, the equation $v_1+v_2=0$ gives $(1,-1)^T$. Normalize both:

$$
q_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\qquad
q_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

Any input can be written $x=c_1q_1+c_2q_2$. The matrix acts independently on these coordinates:

$$
Ax=3c_1q_1+c_2q_2.
$$

For $x=(2,0)^T$, both coefficients equal $\sqrt2$, so $Ax=(4,2)^T$. In the ordinary axes, both coordinates change together; in the eigenvector basis, the transformation is just two scalar multiplications.

### Diagonalization has a condition

If a $d\times d$ matrix has $d$ linearly independent eigenvectors, collect them as columns of an invertible matrix $S$. Then

$$
AS=S\Lambda,\qquad A=S\Lambda S^{-1}.
$$

Consequently $A^k=S\Lambda^kS^{-1}$. This explains why eigenvalues matter for repeated transformations and iterative algorithms. When a diagonalizable matrix has all eigenvalue magnitudes below $1$, its powers approach zero. The eigenvector coordinates determine the rates.

But repeated eigenvalues do not guarantee enough eigenvectors. Consider

$$
J=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$

Its only eigenvalue is $1$, with algebraic multiplicity two. Solving $(J-I)v=0$ forces $v_2=0$, leaving a one-dimensional eigenspace. There is no eigenvector basis, so $J$ is defective, not diagonalizable. In fact,

$$
J^k=\begin{bmatrix}1&k\\0&1\end{bmatrix}.
$$

Although both eigenvalues equal $1$, repeated application creates growing shear. Eigenvalues alone do not bound the finite-step amplification of a general matrix. Even when all eigenvalues are strictly inside the unit circle, a non-normal matrix can exhibit transient growth before its powers decay.

### Real matrices can have complex eigenvalues

The $90$-degree rotation

$$
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
$$

has characteristic equation $\lambda^2+1=0$. Its eigenvalues are $i$ and $-i$. There is no real invariant line: every nonzero real vector rotates to a perpendicular direction. Complex eigenvectors encode this rotational behavior without implying that the original real-valued transformation is invalid.

Eigenvalues concern square maps from a space back to itself. Rectangular matrices instead call for singular values, which describe how input directions map into a potentially different output space.

### The symmetric spectral theorem

Real symmetric matrices are exceptionally well behaved. If $A=A^T$, there is an orthonormal basis of real eigenvectors:

$$
A=Q\Lambda Q^T,\qquad Q^TQ=QQ^T=I.
$$

Repeated eigenvalues are allowed. They make the choice of basis within an eigenspace nonunique, not nonexistent. Because $Q^{-1}=Q^T$, the coordinate change preserves Euclidean distances. Covariance matrices, Gram matrices, and real Hessians with continuous second derivatives belong to this important symmetric family.

For a nonzero vector, define the Rayleigh quotient

$$
\mathcal R_A(x)=\frac{x^TAx}{x^Tx}.
$$

Writing $x=Qc$ gives

$$
\mathcal R_A(x)=\frac{\sum_i\lambda_i c_i^2}{\sum_i c_i^2}.
$$

It is a weighted average of eigenvalues, so it lies between the smallest and largest. The largest eigenvalue is the maximum of $x^TAx$ over unit vectors, achieved by a leading eigenvector. This is the bridge from an algebraic equation to the variance-maximization formulation of PCA.

```html
<h3 id="generalized-eigenvectors-use-a-different-metric">Generalized eigenvectors use a different metric</h3>
```

In a generalized symmetric eigenproblem,

$$
Ax=\lambda Bx,\qquad A=A^T,\quad B=B^T\succ0,
$$

the natural normalization is $x^TBx=1$, not necessarily $x^Tx=1$. The associated
Rayleigh quotient is

$$
\mathcal R_{A,B}(x)=\frac{x^TAx}{x^TBx}.
$$

Maximizing $x^TAx$ subject to $x^TBx=1$ gives $Ax=\lambda Bx$ through a Lagrange
multiplier. The constraint measures length in the $B$ metric. Examples include
variance relative to within-class scatter and eigenproblems defined by a mass
matrix. If the denominator matrix is singular, the SPD formulation no longer
applies without an explicitly chosen restriction or regularization.

Use a Cholesky factorization $B=LL^T$ and coordinates $z=L^Tx$. Then

$$
(L^{-1}AL^{-T})z=\lambda z,\qquad x=L^{-T}z.
$$

The transformed matrix is symmetric, and the ordinary orthonormal eigenvectors
$Z$ recover generalized eigenvectors $X=L^{-T}Z$ satisfying

$$
AX=BX\Lambda,\qquad X^TBX=I.
$$

Triangular solves implement the inverse symbols. Directly forming $B^{-1}A$
usually creates a nonsymmetric matrix in Euclidean coordinates and discards the
structure a symmetric-definite solver could exploit. Do not pass that product
to an ordinary symmetric eigensolver and assume it will detect the mistake.

The lab compares Cholesky whitening with
[`scipy.linalg.eigh(A, B, type=1)`](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.linalg.eigh.html)
on $A=\begin{bmatrix}3&1\\1&2\end{bmatrix}$ and
$B=\begin{bmatrix}2&0.4\\0.4&1\end{bmatrix}$. It checks the original eigen-equation
and $B$-orthogonality, and deliberately rejects a nonsymmetric $A$ or a non-SPD
$B$. SciPy's routine reads a selected triangle rather than automatically
certifying symmetry, so the wrapper checks its real-matrix contract first.
The symmetry check is relative to the matrix's own scale; it averages only
accepted roundoff-level asymmetry before both methods. An ill-conditioned $B$
can amplify whitening errors, so the lab reports its condition number and the
methods' eigenvalue discrepancy instead of demanding a universal agreement
tolerance. A small residual alone does not establish accurate eigenvectors.

Eigenvector signs are arbitrary. Repeated eigenvalues also allow rotation within
an eigenspace. Compare residuals, metric orthogonality and the relevant invariant
subspace, not raw column equality between two correct solvers. Adding a ridge
term to $B$ can make the problem solvable, but changes its metric and eigenproblem;
report that change instead of calling it a harmless implementation detail.

```html
<h3 id="schur-form-survives-when-diagonalization-fails">Schur form survives when diagonalization fails</h3>
```

Every real square matrix has a real Schur factorization

$$
A=QTQ^T,\qquad Q^TQ=I,
$$

where $T$ is upper quasi-triangular: real eigenvalues occupy diagonal entries and
complex-conjugate pairs occupy real $2\times2$ diagonal blocks. Over the complex
numbers, a unitary Schur factorization has an upper triangular $T$. Unlike an
eigenvector diagonalization, Schur form does not require a complete eigenvector
basis. Its off-diagonal entries retain the coupling that a diagonal picture can
hide. Do not confuse Schur **decomposition** with the Schur **complement** from
block elimination; they solve different problems.

[`scipy.linalg.schur`](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.linalg.schur.html)
returns the Schur form before the orthogonal/unitary factor. The lab checks
reconstruction and orthogonality for both a defective matrix and a real rotation
with complex eigenvalues. A defective Jordan block is not a failure of Schur
factorization; it is a counterexample to universal diagonalization.

For a continuous linear system $\dot x=Ax$ and a discrete one $x_{k+1}=Ax_k$,

$$
x(t)=e^{tA}x(0),\qquad x_k=A^kx_0,
$$

$$
e^{tA}=Qe^{tT}Q^T,\qquad A^k=QT^kQ^T.
$$

The [library matrix exponential](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.linalg.expm.html)
is a matrix function, not an elementwise exponential. The lab compares direct
propagation with propagation in Schur coordinates; it does not implement a
custom matrix-exponential algorithm.

Continuous asymptotic stability requires every eigenvalue to have negative real
part; discrete asymptotic stability requires every eigenvalue to have magnitude
strictly below one. These are different tests. The matrix

$$
A_c=\begin{bmatrix}-1&8\\0&-2\end{bmatrix}
$$

is continuous-time stable, yet $\|e^{tA_c}\|_2$ initially exceeds one because it
is non-normal. For this example the off-diagonal exponential entry is
$8(e^{-t}-e^{-2t})$, which initially grows while the diagonal entries decay.

Likewise,

$$
A_d=\begin{bmatrix}0.8&3\\0&0.8\end{bmatrix}
$$

is discrete-time stable and defective. For $k\ge1$ its off-diagonal entry is
$3k\,0.8^{k-1}$, so finite-time amplification can be large even though powers
eventually vanish. The script prints operator-norm curves at specified sample
times/steps; a finite grid is not a certified global maximum.

The operator norm describes the largest amplification over initial directions,
not the trajectory of every input. This distinction matters when interpreting
recurrent-state sensitivity: eigenvalues alone cannot rule out transient growth,
and a small stable linear fixture does not establish the behavior of an entire
nonlinear, input-dependent recurrent network.

## 11. Quadratic forms, curvature, and positive definiteness

A quadratic form assigns a scalar to a vector:

$$
q(x)=x^TAx.
$$

Only the symmetric part of $A$ contributes, because $x^T(A-A^T)x=0$. Therefore

$$
x^TAx=x^T\left(\frac{A+A^T}{2}\right)x.
$$

When discussing positive definiteness here, the matrix is assumed real and symmetric. It is positive semidefinite (PSD) if $x^TAx\ge0$ for every $x$, and positive definite (SPD) if the inequality is strict for every nonzero $x$. Indefinite matrices have both positive and negative quadratic directions.

### Eigenvalues reveal the shape

Using $A=Q\Lambda Q^T$ and $z=Q^Tx$,

$$
x^TAx=\sum_i\lambda_i z_i^2.
$$

Thus PSD means all eigenvalues are nonnegative, while SPD means all are strictly positive. For SPD matrices, constant positive values of the quadratic form describe ellipsoids. A large eigenvalue means a short ellipsoid axis: less movement in that direction reaches the same quadratic value. Zero eigenvalues give flat directions; mixed signs give saddle geometry.

Take $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$. Its quadratic form is $2x_1^2+2x_1x_2+2x_2^2$. Rotating to the eigenvector coordinates from the previous section turns this into $3z_1^2+z_2^2$. The apparently coupled cross term is simply unequal curvature viewed in a rotated coordinate system.

Positive diagonal entries are not sufficient: $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ has eigenvalues $3$ and $-1$. A positive determinant is not sufficient either: $-I_2$ has determinant $1$ and is negative definite. For a symmetric $2\times2$ matrix $\begin{bmatrix}a&b\\b&c\end{bmatrix}$, SPD is equivalent to $a>0$ and $ac-b^2>0$.

### Why Gram and covariance matrices are PSD

For any real matrix $X$ and any vector $v$,

$$
v^TX^TXv=\|Xv\|_2^2\ge0.
$$

So $X^TX$ is always PSD. It is SPD exactly when $X$ has independent columns. Sample covariance is a scaled Gram matrix of centered data, so it is also PSD. However, PSD does not imply invertibility. Duplicated features, constant features, or more features than independent centered observations can make covariance singular.

A covariance matrix and a Hessian are not interchangeable. Covariance measures variation in a dataset or distribution. A Hessian measures local second-order change in an objective. Covariance is always PSD; a general smooth objective can have an indefinite Hessian.

For a twice continuously differentiable function near a stationary point $x_*$,

$$
f(x_*+h)=f(x_*)+\tfrac12h^TH(x_*)h+o(\|h\|^2).
$$

An SPD Hessian is sufficient for a strict local minimum. A PSD Hessian alone is inconclusive: $f(x)=x^4$ has a strict minimum at zero, while $g(x)=-x^4$ has a strict maximum there, yet both have zero second derivative at zero. A twice differentiable function on an open convex domain is convex when its Hessian is PSD everywhere on that domain, not merely at one point.

## 12. Singular value decomposition

The SVD is the general-purpose geometric factorization of a matrix. Unlike an eigendecomposition, it exists for every real rectangular matrix, including rank-deficient ones. It uses one orthonormal coordinate system in the input space and another in the output space, with nonnegative scaling between them. The relationship to $A^TA$ and the four fundamental subspaces is also developed in [MIT's SVD lesson](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/singular-value-decomposition/).

### Full, thin, and compact shapes

Let $A\in\mathbb R^{m\times d}$, $p=\min(m,d)$, and $r=\operatorname{rank}(A)$. The full decomposition is

$$
A=U\Sigma V^T,
$$

with $U\in\mathbb R^{m\times m}$, $\Sigma\in\mathbb R^{m\times d}$, and $V\in\mathbb R^{d\times d}$. Both $U$ and $V$ are orthogonal. The rectangular diagonal matrix $\Sigma$ has singular values $\sigma_1\ge\cdots\ge\sigma_p\ge0$ on its main diagonal.

The thin, or reduced, SVD uses $U_p\in\mathbb R^{m\times p}$, $\Sigma_p\in\mathbb R^{p\times p}$, and $V_p\in\mathbb R^{d\times p}$. It retains zero singular values if rank is below $p$. The compact SVD keeps only the $r$ positive singular values and corresponding vectors:

$$
A=U_r\Sigma_rV_r^T=\sum_{i=1}^r\sigma_i u_iv_i^T.
$$

Thin and compact are not synonymous for a rank-deficient matrix. Also, $U_r^TU_r=I_r$, but $U_rU_r^T$ is generally a projector rather than $I_m$. Dimension checks prevent many mistaken simplifications.

Each outer product $u_iv_i^T$ is a rank-one map: measure the input's coordinate along $v_i$, then send that scalar along $u_i$. The full matrix adds these independent channels with strengths $\sigma_i$.

```html
<figure class="widget">
  <div class="widget__k">Figure 04 · Singular value decomposition</div>
  <h4 class="widget__t">The dimensions of a compact SVD</h4>
  <p class="widget__hint">For a nonzero real matrix of rank \(r\), \(A=U_r\Sigma_rV_r^T\). The columns of \(U_r\) and \(V_r\) are orthonormal; they need not form square matrices. In the full SVD, the square orthogonal factors are rotations or reflections, while the rectangular diagonal factor scales and may discard directions.</p>
  <div class="widget__stage widget__stage--svd" id="svd-visualization"></div>
  <figcaption class="widget__cap">The blocks show factor dimensions, not numerical entries. Keeping the first k singular triplets gives a best rank-at-most-k approximation in the spectral and Frobenius norms.</figcaption>
</figure>
```

### Input directions, output directions, and stretch

For each positive singular value,

$$
Av_i=\sigma_i u_i,\qquad A^Tu_i=\sigma_i v_i.
$$

Consequently $A^TAv_i=\sigma_i^2v_i$ and $AA^Tu_i=\sigma_i^2u_i$. Right singular vectors are input directions; left singular vectors are output directions. A unit ball maps into an ellipsoid whose nonzero semiaxes have lengths $\sigma_i$. Any input component in the null space disappears.

An eigenvalue can be negative or complex; a singular value cannot. Eigenvectors compare a direction with its own image in the same space; singular vectors pair directions in two spaces. For a real symmetric matrix, singular values are the absolute eigenvalues, but their ordering and associated signs still need care. For a general matrix, eigenvalue magnitudes and singular values need not coincide.

Although $A^TA$ explains the theory, explicitly forming it is often a poor numerical way to obtain the SVD: for full-column-rank $A$, its spectral condition number becomes $\kappa_2(A)^2$. Production SVD routines avoid this unnecessary loss of information.

### A fully worked SVD

Consider the nonsymmetric matrix

$$
A=\begin{bmatrix}3&0\\4&5\end{bmatrix},\qquad
A^TA=\begin{bmatrix}25&20\\20&25\end{bmatrix}.
$$

The latter has eigenvalues $45$ and $5$, with orthonormal eigenvectors

$$
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\qquad
v_2=\frac1{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}.
$$

Therefore $\sigma_1=3\sqrt5$ and $\sigma_2=\sqrt5$. Compute the left vectors by applying $A$ and dividing by the matching singular value:

$$
u_1=\frac{Av_1}{3\sqrt5}
=\frac1{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix},\qquad
u_2=\frac{Av_2}{\sqrt5}
=\frac1{\sqrt{10}}\begin{bmatrix}-3\\1\end{bmatrix}.
$$

Both have unit length and their dot product is zero. Thus

$$
U=\frac1{\sqrt{10}}\begin{bmatrix}1&-3\\3&1\end{bmatrix},\quad
\Sigma=\begin{bmatrix}3\sqrt5&0\\0&\sqrt5\end{bmatrix},\quad
V=\frac1{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}.
$$

The two rank-one terms are

$$
\sigma_1u_1v_1^T=
\begin{bmatrix}1.5&1.5\\4.5&4.5\end{bmatrix},\qquad
\sigma_2u_2v_2^T=
\begin{bmatrix}1.5&-1.5\\-0.5&0.5\end{bmatrix}.
$$

Adding them recovers $A$. As independent checks, $\sigma_1^2+\sigma_2^2=50=\|A\|_F^2$ and $\sigma_1\sigma_2=15=|\det A|$. The eigenvalues of $A$ itself are $3$ and $5$, visibly different from its singular values.

## 13. Low-rank approximation and spectral energy

Suppose a matrix must be stored, transmitted, or applied using only $k$ independent channels. Truncating its SVD gives

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^T.
$$

This is not merely a convenient approximation. For $0\le k<r$, the Eckart-Young theorem says it minimizes error among matrices of rank at most $k$ under both the spectral norm and the Frobenius norm:

$$
\min_{\operatorname{rank}(B)\le k}\|A-B\|_2=\sigma_{k+1},
$$

$$
\min_{\operatorname{rank}(B)\le k}\|A-B\|_F^2
=\sum_{i=k+1}^r\sigma_i^2.
$$

The spectral error measures the worst remaining distortion of a unit input. The squared Frobenius error measures the total squared residual across matrix entries. They are different objectives, even though truncated SVD solves both.

### Why the discarded tail determines the error

Orthogonal coordinate changes preserve these norms. In singular-vector coordinates, truncation simply replaces selected diagonal entries of $\Sigma$ by zero. The largest remaining entry gives spectral error; the sum of squared remaining entries gives squared Frobenius error.

Why can no rank-$k$ matrix do better in spectral norm? The space spanned by $v_1,\ldots,v_{k+1}$ has dimension $k+1$. Any rank-$k$ map $B$ must annihilate a nonzero vector in that space. For a unit such vector $z$, $(A-B)z=Az$, whose norm is at least $\sigma_{k+1}$. Truncation reaches this lower bound.

For Frobenius error, think of choosing a $k$-dimensional output subspace. Projecting onto a chosen subspace is the best approximation whose columns lie there. The retained squared energy is maximized by choosing the leading $k$ eigenvectors of $AA^T$, so the unretained energy is exactly the squared singular-value tail. This connects low-rank approximation to the variance-maximization argument used in PCA.

In the worked SVD above, keeping just the first term retains $45/50=90\%$ of squared Frobenius energy. The residual has squared Frobenius norm $5$ and spectral norm $\sqrt5$. Storing $U_k$, singular values, and $V_k$ uses approximately $k(m+d+1)$ numbers instead of $md$, which helps only when $k$ is sufficiently small.

### What an optimum does not promise

The theorem optimizes reconstruction under particular norms, not downstream classification accuracy or human meaning. A low-energy direction can contain a rare but decisive signal. A leading direction can be dominated by nuisance scale or background variation. If $\sigma_k=\sigma_{k+1}$, different subspaces within the tied singular-value group can be equally good. Even without exact ties, a small spectral gap can make individual vectors unstable under perturbations, while the larger subspace remains relatively stable.

Similarly, low-rank compression does not imply denoising unless the signal and noise structure justify discarding small components. The approximation error is mathematically determined; whether that error is desirable is a modeling decision.

## 14. Covariance and Mahalanobis geometry

Let $X\in\mathbb R^{n\times d}$ contain observations as rows. Define the training mean $\mu=\frac1n\sum_i x_i$ and centered data $X_c=X-\mathbf1\mu^T$. For $n>1$, the sample covariance is

$$
C=\frac1{n-1}X_c^TX_c.
$$

The diagonal entry $C_{jj}$ is feature $j$'s sample variance. The off-diagonal entry $C_{jk}$ measures signed co-variation. Its magnitude depends on both feature units; correlation divides by the two standard deviations when they are nonzero.

For a direction $w$, the projected observations are $X_cw$ and their sample variance is

$$
\operatorname{Var}(X_cw)=\frac1{n-1}\|X_cw\|^2=w^TCw.
$$

Covariance therefore describes a whole family of one-dimensional variances, not merely a table of pairwise relationships. Its eigenvectors identify uncorrelated principal directions, while its eigenvalues give their variances. Uncorrelated coordinates need not be independent. Independence follows from diagonal covariance for a jointly Gaussian vector, not for arbitrary distributions.

### Distance relative to expected variation

If $C$ is SPD, Mahalanobis distance is

$$
d_C(x,y)=\sqrt{(x-y)^TC^{-1}(x-y)}.
$$

With $C=Q\Lambda Q^T$ and $z=Q^T(x-y)$, its square becomes $\sum_j z_j^2/\lambda_j$. A deviation along a high-variance direction counts less; the same deviation along a tightly concentrated direction counts more.

For $C=\operatorname{diag}(100,1)$, deviations $(10,0)^T$ and $(0,1)^T$ both have Mahalanobis distance $1$. Their Euclidean lengths differ by a factor of ten, but each is one standard deviation along its respective axis. Off-diagonal covariance rotates these axes rather than merely rescaling the original features.

Do not form $C^{-1}$ just to evaluate this distance. Solve $Cz=x-y$ and compute $(x-y)^Tz$, or use a Cholesky factor $C=LL^T$ and solve $Lu=x-y$; then the squared distance is $\|u\|^2$.

If covariance is singular, replacing the inverse with a pseudoinverse ignores deviations in its null space. The resulting expression is a pseudometric on the ambient space: distinct points can have zero distance. It is meaningful on the supported subspace but is not an automatic solution for off-subspace anomalies. Adding $\alpha I$, with $\alpha>0$, gives an SPD regularized geometry, but changes the statistical model and must respect feature scales.

## 15. Principal component analysis, from objective to implementation

PCA finds an orthogonal coordinate system that captures as much centered variance as possible in a chosen number of directions. It is unsupervised: labels do not appear in the objective. It also has a complementary interpretation as the best linear reconstruction of centered data under squared error.

```html
<figure class="widget">
  <div class="widget__k">Figure 05 · Eigenvectors</div>
  <h4 class="widget__t">Principal directions of a centered sample</h4>
  <p class="widget__hint">The covariance matrix is computed from these 150 centered points using the \(n-1\) denominator. Its eigenvectors define the two principal directions. Each ray has length \(2\sqrt{\lambda_i}\): two sample standard deviations, not two variances. The PC1 variance fraction quantifies what a one-dimensional projection retains.</p>
  <div class="widget__stage" id="pca-visualization"></div>
  <div class="widget__legend">
    <span><i class="widget__key widget__key--pc1" aria-hidden="true"></i>PC1 · most variance</span>
    <span><i class="widget__key widget__key--pc2" aria-hidden="true"></i>PC2 · remaining orthogonal variance</span>
  </div>
</figure>
```

### Deriving the first component

For a unit direction $v$, the projected training scores are $X_cv$. Their variance is $v^TCv$, so the first principal direction solves

$$
\max_{v^Tv=1}v^TCv.
$$

The Lagrangian $v^TCv-\lambda(v^Tv-1)$ has stationarity condition $Cv=\lambda v$. The Rayleigh quotient proves that the maximizing solution is a leading eigenvector of $C$. Later directions solve the same problem subject to orthogonality to earlier ones.

Let $X_c=U_r\Sigma_rV_r^T$. Then

$$
C=V_r\frac{\Sigma_r^2}{n-1}V_r^T.
$$

The right singular vectors are principal directions in feature space. The corresponding covariance eigenvalues are $\lambda_j=\sigma_j^2/(n-1)$, not $\sigma_j$ or $\sigma_j^2$ without a denominator. With $k$ retained directions, scores and reconstruction are

$$
Z=X_cV_k=U_k\Sigma_k,\qquad
\widehat X=ZV_k^T+\mathbf1\mu^T.
$$

Check the shapes: $V_k$ is $d\times k$, $Z$ is $n\times k$, and $\widehat X$ is $n\times d$. Each observation is replaced by its projection onto the affine subspace through the training mean.

### A numeric PCA example

Take four two-feature observations:

$$
X=\begin{bmatrix}13&21\\11&23\\9&17\\7&19\end{bmatrix},\qquad
\mu=\begin{bmatrix}10\\20\end{bmatrix},\qquad
X_c=\begin{bmatrix}3&1\\1&3\\-1&-3\\-3&-1\end{bmatrix}.
$$

The covariance is

$$
C=\frac13\begin{bmatrix}20&12\\12&20\end{bmatrix}.
$$

Its eigenvalues are $32/3$ and $8/3$, with directions $v_1=(1,1)^T/\sqrt2$ and $v_2=(1,-1)^T/\sqrt2$. The singular values of $X_c$ are $\sqrt{32}$ and $\sqrt8$.

The first-component scores are $(2\sqrt2,2\sqrt2,-2\sqrt2,-2\sqrt2)^T$. Projecting back and restoring the mean yields

$$
\widehat X_1=
\begin{bmatrix}12&22\\12&22\\8&18\\8&18\end{bmatrix}.
$$

Every row now lies on the line through $(10,20)$ in direction $(1,1)$. The discarded squared reconstruction error is $8$, matching the second squared singular value. One component explains $32/(32+8)=80\%$ of centered variance. Two components reconstruct these data exactly.

Notice what was lost: the first two observations receive identical one-dimensional representations, as do the final two. A large explained-variance ratio does not guarantee preservation of every distinction important to a downstream task.

### Explained variance, centering, and scaling

Provided total variance is positive, the explained-variance ratio of component $j$ is

$$
\frac{\lambda_j}{\sum_i\lambda_i}
=\frac{\sigma_j^2}{\sum_i\sigma_i^2}.
$$

The denominator includes all components of the centered data, not just those retained. If every observation is identical, centered variance is zero and these ratios are undefined. Because centering imposes $\mathbf1^TX_c=0$, its rank is at most $\min(n-1,d)$, a stricter bound than the uncentered matrix's possible rank.

Without centering, leading singular vectors optimize squared distance relative to the origin, not variance around the mean. The identity

$$
X^TX=X_c^TX_c+n\mu\mu^T
$$

shows the extra mean-related term explicitly. Uncentered truncated SVD can be intentional, especially for sparse representations, but it is a different objective.

Scaling is another separate decision. A feature measured in millimeters can dominate the same feature measured in meters because PCA responds to variance magnitudes. Standardizing columns before PCA instead emphasizes variance relative to each feature's scale. Neither choice is universally correct: standardization can prevent arbitrary units from dominating, but can also amplify a low-variance noise feature.

Fit means, scales, and principal directions on training data only. Apply those learned values unchanged to validation and test data. In cross-validation, fit them separately inside every training fold. Using the full dataset to choose a basis leaks information about held-out geometry even though labels are not used.

### A shape-explicit implementation

```python
import numpy as np

X_train = np.array([[13., 21.], [11., 23.],
                    [9., 17.], [7., 19.]])
mean = X_train.mean(axis=0)
Xc = X_train - mean
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
k = 1
directions = Vt[:k].T                  # (features, components)
scores = Xc @ directions              # (samples, components)
reconstruction = scores @ directions.T + mean
variance = s**2 / (len(X_train) - 1)
ratio = variance / variance.sum()
np.testing.assert_allclose(
    np.sum((X_train - reconstruction)**2), np.sum(s[k:]**2)
)

# Future observations use the training mean and directions.
X_new = np.array([[12., 21.]])
new_scores = (X_new - mean) @ directions
```

NumPy returns singular values as a one-dimensional array and returns $V^T$, not $V$, for real inputs. Its reduced shapes are documented in the [NumPy SVD reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html). In scikit-learn, `PCA` centers input but does not automatically standardize each feature; `components_` stores principal directions as rows, and `explained_variance_` uses the $n-1$ convention. See the [PCA API reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for estimator and solver details.

Flipping a singular vector's sign, together with its corresponding score signs, leaves reconstruction unchanged. Tests should compare reconstructed matrices, projectors, or subspaces rather than requiring an arbitrary sign convention. Repeated singular values permit larger rotations within the tied subspace.

### Whitening and its cost

PCA decorrelates retained training coordinates but does not make their variances equal. Whitening also rescales them:

$$
Z_{\mathrm{white}}=X_cV_k\Lambda_k^{-1/2},\qquad
\frac1{n-1}Z_{\mathrm{white}}^TZ_{\mathrm{white}}=I_k.
$$

This expression uses sample covariance with denominator $n-1$ and requires retained eigenvalues to be positive. The alternative population-style denominator $n$ changes the scaling. State the convention when comparing formulas or implementations.

For the numeric example, the first scores $\pm2\sqrt2$ become $\pm\sqrt3/2$ after division by $\sqrt{32/3}$. Their squared sum is $3$, giving sample variance $3/(4-1)=1$.

Full-rank whitening is invertible if the mean, directions, and scaling are retained. Truncated whitening is not: discarded coordinates cannot be recovered. ZCA whitening rotates the whitened coordinates back into the original feature axes, using $X_cV\Lambda^{-1/2}V^T$ in the full-rank case. It produces identity covariance too, but retains the original coordinate orientation as closely as this symmetric transformation allows.

Small eigenvalues are dangerous because inverse square roots amplify them. If a direction has variance $10^{-8}$, exact whitening multiplies its coordinates by $10^4$, potentially magnifying estimation error or measurement noise. Truncation or replacing $\lambda_j$ by $\lambda_j+\varepsilon$ can help, but regularized whitening no longer yields exactly unit variance: the resulting variance is $\lambda_j/(\lambda_j+\varepsilon)$. Finally, whitened training covariance does not guarantee whitened future covariance under distribution shift, and decorrelation still does not imply statistical independence.


## 16. From elimination to reliable solvers

A matrix equation is a mathematical object; solving it on a computer is an algorithmic choice. The formula $x=A^{-1}b$ describes a solution when $A$ is invertible, but it does not instruct us to construct the inverse. A factorization usually computes the desired answer with less work and less unnecessary numerical error.

### Gaussian elimination as a sequence of equivalent systems

Consider

$$
\begin{bmatrix}2&1\\4&3\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=\begin{bmatrix}5\\11\end{bmatrix}.
$$

Subtract twice the first equation from the second. The transformed equations are $2x_1+x_2=5$ and $x_2=1$, so back substitution gives $x_1=2$. The row operation preserves the solution set because it is reversible. Subtracting a multiple of an equation does not introduce an independent constraint or discard one.

Elimination records the multiplier $2$ in a lower-triangular factor:

$$
A=LU,
\qquad
L=\begin{bmatrix}1&0\\2&1\end{bmatrix},
\qquad
U=\begin{bmatrix}2&1\\0&1\end{bmatrix}.
$$

Instead of solving $Ax=b$ directly, first solve $Ly=b$ by forward substitution, obtaining $y=(5,1)^T$. Then solve $Ux=y$ by back substitution. For a triangular system, each new unknown depends only on unknowns already computed.

For a dense $d\times d$ matrix, factorization costs order $d^3$ operations, whereas a solve with existing triangular factors costs order $d^2$ per right-hand side. This distinction matters when the same operator is used repeatedly. Factoring a covariance matrix once and solving for many vectors is fundamentally different from refactoring it inside every iteration.

### Why pivoting is not optional bookkeeping

An invertible matrix can have a zero leading entry. For example, $\begin{bmatrix}0&1\\1&1\end{bmatrix}$ is invertible, but elimination cannot divide by its first entry. Swap the rows first. A very small pivot is also dangerous: dividing by it produces a large multiplier, and subsequent subtraction can magnify rounding error.

For

$$
A=\begin{bmatrix}\varepsilon&1\\1&1\end{bmatrix},
\qquad 0<\varepsilon\ll1,
$$

unpivoted elimination uses multiplier $1/\varepsilon$. Swapping the rows changes that multiplier to $\varepsilon$. The underlying linear system has not become easier or harder; the sequence of arithmetic operations has become more sensible.

Partial pivoting chooses the largest-magnitude available entry in the current column. With the convention used here, the factorization is

$$PA=LU,$$

where $P$ permutes rows. The solve becomes $Ly=Pb$, then $Ux=y$. Other references write $A=PLU$ with a differently defined permutation, so check conventions before comparing factorization outputs. Pivoting is highly effective in practice, but pathological matrices can still cause large growth in intermediate entries. It is not a theorem that every pivoted LU solve has small forward error.

```python
import numpy as np

A = np.array([[2., 1.], [4., 3.]])
b = np.array([5., 11.])
x = np.linalg.solve(A, b)
assert np.allclose(x, [2., 1.])
assert np.allclose(A @ x, b)

# Each column is a separate right-hand side.
B = np.column_stack([b, [1., 0.]])
solutions = np.linalg.solve(A, B)
assert np.allclose(A @ solutions, B)
```

`numpy.linalg.solve` expects a square, full-rank coefficient matrix and uses a LAPACK linear-system solver. Rectangular or rank-deficient fitting problems belong in a least-squares routine. See the [NumPy solve reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html).

### Structure changes the best algorithm

If $A$ is symmetric positive definite, Cholesky writes $A=LL^T$ with positive diagonal entries in $L$. This exploits symmetry, uses less work than general LU, and reduces the solve to two triangular systems. It is inappropriate for a merely symmetric indefinite matrix. An indefinite symmetric system may instead use a pivoted $LDL^T$ factorization.

Sparse matrices introduce another issue: a factorization can create nonzero entries where the input had zeros, called **fill-in**. Reordering variables can reduce fill-in dramatically. For sufficiently large problems, an iterative method that only computes products $Av$ may be preferable to storing factors. Conjugate gradients requires a symmetric positive-definite operator; GMRES addresses more general square systems; LSQR targets least squares. Their stopping criteria, preconditioning, and assumptions are part of the solution, not implementation details to ignore.

## 17. QR factorization and least squares

### Orthogonal coordinates simplify the problem

Let $X\in\mathbb R^{n\times d}$ have independent columns, with $n\ge d$. Its reduced QR factorization is

$$X=QR,\qquad Q^TQ=I_d,$$

where $Q\in\mathbb R^{n\times d}$ and $R\in\mathbb R^{d\times d}$ is invertible and upper triangular. The columns of $Q$ provide an orthonormal coordinate system for the same column space as $X$. The matrix $R$ expresses the original columns in that coordinate system.

For two independent columns $a_1,a_2$, Gram-Schmidt constructs

$$
q_1=\frac{a_1}{\|a_1\|},\qquad
r_{12}=q_1^Ta_2,\qquad
v_2=a_2-r_{12}q_1,\qquad
q_2=\frac{v_2}{\|v_2\|}.
$$

The subtraction removes the component already explained by $q_1$. If $v_2=0$, the second column supplied no new direction. Classical Gram-Schmidt projects against the original column when forming each coefficient. Modified Gram-Schmidt updates the residual after each projection. They are equivalent in exact arithmetic, but modified Gram-Schmidt generally retains orthogonality better in finite precision. Reorthogonalization can be necessary for nearly dependent inputs.

### Householder reflections

A Householder matrix has the form

$$H=I-2\frac{vv^T}{v^Tv},\qquad v\ne0.$$

It is symmetric and orthogonal: $H^T=H$ and $H^TH=I$. Geometrically it reflects a vector across the hyperplane perpendicular to $v$. Computationally, choose $v$ so that $Hx$ has only its first coordinate nonzero. Applying successive reflections to trailing portions of a matrix creates an upper-triangular $R$ without repeatedly subtracting projections onto a growing collection of imperfectly orthogonal vectors.

A numerically sensible construction takes $\alpha=-\operatorname{sign}(x_1)\|x\|_2$, treating a zero first component with sign $+1$, and sets $v=x-\alpha e_1$. If the entire vector is zero, skip the reflection rather than divide by $v^Tv=0$. The sign avoids subtracting two nearly equal leading quantities. We need not explicitly form the dense $H$: applying it to a matrix uses $HM=M-2v(v^TM)/(v^Tv)$.

NumPy exposes reduced and complete QR and uses LAPACK Householder-based routines. Reduced QR normally avoids storing the unused orthogonal complement. See the [NumPy QR reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html).

### Least squares is a projection, not an approximate inverse

With observations in rows of $X$, fit $w\in\mathbb R^d$ to targets $y\in\mathbb R^n$ by minimizing

$$f(w)=\frac12\|Xw-y\|_2^2.$$

Differentiating gives $\nabla f=X^T(Xw-y)$. At a minimizer,

$$X^TXw=X^Ty.$$

These are the normal equations. They say the residual $r=y-Xw$ is orthogonal to every column of $X$. No direction available inside the model's prediction space can further reduce its distance to $y$.

Using $X=QR$, decompose the target into $QQ^Ty$ and $(I-QQ^T)y$. These components are perpendicular, so

$$
\|Xw-y\|_2^2
=\|Rw-Q^Ty\|_2^2+\|(I-QQ^T)y\|_2^2.
$$

The second term does not depend on $w$. Therefore the minimizer solves the triangular system $Rw=Q^Ty$. This derivation explains both the algorithm and the irreducible training residual.

### Worked example: fitting a line

Fit $\hat y=a+bt$ to points $(0,1),(1,2),(2,2)$. Then

$$
X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad
y=\begin{bmatrix}1\\2\\2\end{bmatrix},\quad
X^TX=\begin{bmatrix}3&3\\3&5\end{bmatrix},\quad
X^Ty=\begin{bmatrix}5\\6\end{bmatrix}.
$$

Subtracting the first normal equation from the second gives $2b=1$. Hence $b=1/2$ and $a=7/6$. Predictions are $(7/6,5/3,13/6)^T$ and residuals are $(-1/6,1/3,-1/6)^T$. Their sum is zero because the design contains an intercept. Their dot product with the time column is also zero: $1/3-2/6=0$. The residual sum of squares is $1/6$.

```python
import numpy as np

t = np.array([0., 1., 2.])
X = np.column_stack([np.ones_like(t), t])
y = np.array([1., 2., 2.])

Q, R = np.linalg.qr(X, mode="reduced")
w_qr = np.linalg.solve(R, Q.T @ y)
w, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
assert np.allclose(w, [7 / 6, 1 / 2])
assert np.allclose(w, w_qr)
assert np.allclose(X.T @ (y - X @ w), 0.)
assert np.isclose(np.sum((y - X @ w) ** 2), 1 / 6)
```

### Why not always solve the normal equations?

If $X$ has full column rank, its singular values satisfy $\sigma_1\ge\cdots\ge\sigma_d>0$. The eigenvalues of $X^TX$ are $\sigma_i^2$. Consequently,

$$\kappa_2(X^TX)=\frac{\sigma_1^2}{\sigma_d^2}=\kappa_2(X)^2.$$

Forming $X^TX$ can therefore turn a moderately sensitive problem into a severely sensitive one. It also rounds the dot products before the solve begins. The normal equations are excellent for derivation, and sometimes acceptable for well-conditioned, performance-sensitive problems, but QR or SVD is the more robust general starting point. Do not describe the normal equations as incorrect; distinguish algebraic equivalence from floating-point behavior.

Weighted least squares replaces the objective with $\sum_i \omega_i(x_i^Tw-y_i)^2$ for nonnegative weights. For positive weights, solve ordinary least squares after multiplying row $i$ of both $X$ and $y$ by $\sqrt{\omega_i}$. This changes the geometry: an error on a highly weighted observation contributes more to the distance being minimized.

### A weighted projection and generalized least squares

Fit a constant to $y=(0,2)^\top$ with $X=(1,1)^\top$ and
$W=\operatorname{diag}(1,3)$. Differentiating
$(y-Xa)^\top W(y-Xa)$ gives $X^\top W(y-Xa)=0$, hence $a=6/4=1.5$.
The residual $r=(-1.5,.5)^\top$ is not Euclidean-orthogonal to $X$:
$X^\top r=-1$. It is **weighted-orthogonal**, since $X^\top Wr=0$.
The weighted residual sum is $2.25+3(.25)=3$.

When measurement noise has known SPD covariance $\Sigma$, generalized least
squares uses $W=\Sigma^{-1}$, including off-diagonal correlations. With
$\Sigma=LL^\top$, whiten by solving $L\widetilde X=X$ and
$L\widetilde y=y$, then use QR/SVD least squares. A diagonal covariance with
variances $(1,1/3)$ gives the weights above. Relative confidence weights and
frequency replication weights can have the same fitting algebra but different
uncertainty interpretations.

## 18. Pseudoinverses, minimum norm, and ridge regression

### What survives when a matrix has no inverse?

Write the compact rank-$r$ SVD as

$$X=U_r\Sigma_rV_r^T.$$

The Moore-Penrose pseudoinverse is

$$X^+=V_r\Sigma_r^{-1}U_r^T.$$

It first extracts the target's coordinates along the attainable output directions $u_i$, divides each by its nonzero stretch $\sigma_i$, and maps back along $v_i$. It does not invert zero singular values. The matrices $XX^+=U_rU_r^T$ and $X^+X=V_rV_r^T$ are orthogonal projectors onto the column and row spaces, respectively, rather than necessarily being identity matrices.

For any target, $w_*=X^+y$ minimizes the residual norm. Every other least-squares minimizer has the form

$$w=w_*+z,\qquad z\in\ker X.$$

Since $w_*$ lies in the row space and the row space is perpendicular to $\ker X$,

$$\|w_*+z\|_2^2=\|w_*\|_2^2+\|z\|_2^2.$$

Thus the pseudoinverse selects the unique smallest-norm parameter vector among all best fits. A nonunique parameter vector can still produce a unique fitted prediction $XX^+y$.

### Worked example: redundant features

Take

$$X=\begin{bmatrix}1&1\\2&2\end{bmatrix},\qquad y=\begin{bmatrix}1\\2\end{bmatrix}.$$

Every $w$ with $w_1+w_2=1$ fits exactly. Write $w=(1/2+t,1/2-t)^T$. Its squared norm is $1/2+2t^2$, minimized at $t=0$. Therefore $X^+y=(1/2,1/2)^T$. The training data cannot tell which of the identical features deserves the effect. Minimum norm chooses one convention; it does not recover missing statistical information.

For the underdetermined equation $[1\ \ 2]w=3$, the minimum-norm solution must lie along $(1,2)^T$. Set $w=c(1,2)^T$ to obtain $5c=3$, giving $(3/5,6/5)^T$. Any added vector proportional to $(-2,1)^T$ leaves the prediction unchanged and increases the norm.

### Numerical rank is a scale-dependent decision

In floating point, a singular value need not equal zero to be unusable. A tolerance treats sufficiently small singular values as absent. This is equivalent to replacing the original matrix by a nearby lower-rank approximation before solving. The cutoff depends on numerical precision, problem scale, and sometimes measurement noise. A threshold appropriate for exact synthetic data need not be appropriate for noisy sensor measurements.

`lstsq` returns a minimum-norm best fit along with rank and singular-value diagnostics. Its residual output may be empty, including for rank-deficient or underdetermined systems; calculate `y - X @ w` explicitly when you need a residual vector. See [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html). `pinv` uses an SVD cutoff to decide which singular values to invert; see [NumPy pseudoinverse](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html).

### Experiment: rank tolerance changes the question

For $X=\operatorname{diag}(1,10^{-6},10^{-12})$ and
$y=(1,10^{-6},10^{-6})^\top$, a tiny cutoff fits all three coordinates with
$w=(1,1,10^6)^\top$. Cutting the last direction leaves a residual $10^{-6}$
but removes the enormous coefficient. A still larger cutoff removes the
second direction too. Low residual, small coefficient norm and estimated rank
are distinct diagnostics; the right compromise requires a noise/model argument.

```python runnable
import numpy as np

X = np.diag([1., 1e-6, 1e-12])
y = np.array([1., 1e-6, 1e-6])
ranks, norms, residuals = [], [], []
for cutoff in [1e-14, 1e-9, 1e-3]:
    w, _, rank, _ = np.linalg.lstsq(X, y, rcond=cutoff)
    ranks.append(rank)
    norms.append(np.linalg.norm(w))
    residuals.append(np.linalg.norm(X @ w - y))
    print(cutoff, rank, norms[-1], residuals[-1])
assert ranks == [3, 2, 1]
assert norms[0] > 1e5 and norms[1] < 2
assert residuals[0] < 1e-15
np.testing.assert_allclose(residuals[1:], [1e-6, np.sqrt(2)*1e-6])

# Weighted projection through whitening, without an explicit inverse.
design = np.ones((2, 1))
target = np.array([0., 2.])
L = np.linalg.cholesky(np.diag([1., 1/3]))
w = np.linalg.lstsq(np.linalg.solve(L, design),
                    np.linalg.solve(L, target), rcond=None)[0]
np.testing.assert_allclose(w, [1.5])
np.testing.assert_allclose(design.T @ np.diag([1., 3.]) @
                           (target - design @ w), [0.], atol=1e-14)
```

### Ridge is controlled shrinkage in singular directions

Ridge regression minimizes

$$\frac12\|Xw-y\|_2^2+\frac\lambda2\|w\|_2^2,\qquad\lambda>0.$$

Its first-order condition is $(X^TX+\lambda I)w=X^Ty$. This matrix is positive definite even if $X$ is rank deficient, because

$$v^T(X^TX+\lambda I)v=\|Xv\|_2^2+\lambda\|v\|_2^2>0$$

for every nonzero $v$. Substituting the SVD yields

$$
w_\lambda=\sum_{i=1}^r
\frac{\sigma_i}{\sigma_i^2+\lambda}(u_i^Ty)v_i.
$$

Unregularized least squares multiplies a target coordinate by $1/\sigma_i$. Ridge uses $\sigma_i/(\sigma_i^2+\lambda)$ instead, suppressing directions whose weak data signal would otherwise be strongly amplified. In prediction space, the corresponding factor is $\sigma_i^2/(\sigma_i^2+\lambda)$, between zero and one.

For $X=\operatorname{diag}(4,1)$, $y=(4,1)^T$, and $\lambda=1$, ordinary least squares gives $(1,1)^T$, while ridge gives $(16/17,1/2)^T$. Shrinkage is much stronger in the less well-supported direction. It is not generally one common scaling applied to every coefficient.

An augmented least-squares formulation avoids explicitly forming $X^TX$:

$$
\min_w\left\|
\begin{bmatrix}X\\\sqrt\lambda I\end{bmatrix}w-
\begin{bmatrix}y\\0\end{bmatrix}
\right\|_2^2.
$$

```python
import numpy as np

X = np.diag([4., 1.])
y = np.array([4., 1.])
lam = 1.0
U, s, Vt = np.linalg.svd(X, full_matrices=False)
w_svd = Vt.T @ ((s / (s * s + lam)) * (U.T @ y))
X_aug = np.vstack([X, np.sqrt(lam) * np.eye(X.shape[1])])
y_aug = np.concatenate([y, np.zeros(X.shape[1])])
w_aug = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
assert np.allclose(w_svd, [16 / 17, 1 / 2])
assert np.allclose(w_svd, w_aug)
```

Feature scaling matters because the penalty is measured in parameter coordinates. Changing a feature's units changes the meaning of $\|w\|^2$. Standardization is often appropriate, with statistics fitted on training data only. An intercept is commonly left unpenalized; replace the penalty identity with a diagonal matrix having zero in the intercept position. Also state the loss normalization: adding $\lambda\|w\|^2/2$ to an average loss produces $X^TX+n\lambda I$, not $X^TX+\lambda I$.

## 19. Conditioning and finite-precision reasoning

### Sensitivity belongs to the problem

For invertible $A$, the spectral condition number is

$$\kappa_2(A)=\|A\|_2\|A^{-1}\|_2=\frac{\sigma_{\max}}{\sigma_{\min}}.$$

With $A$ fixed, a perturbation $\delta b$ produces $\delta x=A^{-1}\delta b$. Combining norm inequalities gives

$$
\frac{\|\delta x\|_2}{\|x\|_2}
\le \kappa_2(A)\frac{\|\delta b\|_2}{\|b\|_2},
$$

for nonzero $b$. This is a worst-case bound, not a promise that every perturbation is amplified maximally. Direction matters: perturbations along a small-singular-value output direction are the dangerous ones.

Take $A=\operatorname{diag}(1,10^{-8})$, $b=(1,10^{-8})^T$. Then $x=(1,1)^T$. Changing only the second target entry by $10^{-8}$ changes the solution to $(1,2)^T$. The target perturbation is tiny relative to the norm of $b$, but the parameter change is substantial. No algorithm can infer that the changed second target should secretly have been the original one.

If the matrix itself is also perturbed, subtracting the two systems gives
$(A+\delta A)\delta x=\delta b-\delta A\,x$. Writing
$\epsilon_A=\|\delta A\|_2/\|A\|_2$ and
$\epsilon_b=\|\delta b\|_2/\|b\|_2$, one obtains the bound

$$
\frac{\|\delta x\|_2}{\|x\|_2}
\leq\frac{\kappa_2(A)}{1-\kappa_2(A)\epsilon_A}
(\epsilon_A+\epsilon_b),
\qquad \kappa_2(A)\epsilon_A<1.
$$

The denominator comes from bounding the inverse of the perturbed operator.
It emphasizes that a perturbation small relative to the largest matrix scale
can still be large relative to the smallest singular direction. Once the stated
condition fails, this bound gives no guarantee; it does not assert that the
system must have become singular.

### Forward error, backward error, and residuals

**Forward error** compares a computed answer $\hat x$ with the desired exact answer $x$. **Backward error** asks how much the input would need to change for $\hat x$ to be exact. A backward-stable algorithm returns the exact answer to a nearby problem, up to a suitably small perturbation. If the problem is ill-conditioned, a nearby problem can still have a distant solution.

The residual is $r=b-A\hat x$. A useful normwise scaled residual is

$$\eta=\frac{\|r\|}{\|A\|\|\hat x\|+\|b\|}.$$

It measures consistency with the supplied equations relative to their scale. A small residual alone does not establish small forward error. With $A=\operatorname{diag}(1,10^{-8})$ and $b=(1,10^{-8})^T$, the poor approximation $\hat x=(1,0)^T$ has residual norm only $10^{-8}$ despite an error of one in its second component.

### Rounding, cancellation, and diagnostics

Floating-point arithmetic stores a finite approximation to most real numbers. For ordinary normalized operations away from overflow and underflow, relative rounding error is on the order of the format's unit roundoff. Subtracting nearly equal quantities can expose the error already present in their leading digits: the small result may have few trustworthy relative digits left.

A rough warning indicator is $\kappa(A)u$, where $u$ is unit roundoff. If this approaches one, worst-case relative accuracy may be poor even for a stable algorithm. This is not an exact count of reliable digits, and least-squares sensitivity also depends on the residual and perturbation model. Low precision can be entirely suitable for many neural-network operations while being unsuitable for a sensitive linear solve.

```python
import numpy as np

A = np.diag([1., 1e-8])
b = np.array([1., 1e-8])
x = np.linalg.solve(A, b)
x_bad = np.array([1., 0.])
residual = b - A @ x_bad
scaled_residual = np.linalg.norm(residual) / (
    np.linalg.norm(A, 2) * np.linalg.norm(x_bad) + np.linalg.norm(b)
)
print("condition:", np.linalg.cond(A))
print("scaled residual:", scaled_residual)
print("relative forward error:", np.linalg.norm(x_bad - x) / np.linalg.norm(x))
assert scaled_residual < 1e-7
assert np.linalg.norm(x_bad - x) > 0.9
```

Do not use the determinant as a conditioning diagnostic. The matrix $10^{-20}I$ has a tiny determinant but condition number one. Conversely, $\operatorname{diag}(10^{10},10^{-10})$ has determinant one and condition number $10^{20}$. Absolute volume scaling and relative directional sensitivity answer different questions.

Useful responses to numerical difficulty include checking units, scaling variables, inspecting singular values, regularizing when scientifically justified, increasing precision, and using iterative refinement. None excuses ignoring a modeling identifiability problem: more bits cannot create information absent from the observations.

### Iterative refinement preserves the original target system

Given an approximate solution $\hat x$, compute $r=b-A\hat x$ in higher
precision, approximately solve $A\delta=r$ with the existing lower-precision
factors, and update $\hat x\leftarrow\hat x+\delta$ in higher precision.
The correction targets the original system, unlike changing its diagonal.
It works when the correction solve is sufficiently accurate to contract the
error; high conditioning, factorization growth or low-precision overflow can
prevent that. A useful experiment compares both residual and forward error.

```python runnable
import numpy as np
from scipy.linalg import lu_factor, lu_solve

rng = np.random.default_rng(29)
Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
A = (Q*np.geomspace(1., 1e-4, 8)) @ Q.T
truth = rng.normal(size=8)
b = A @ truth
factors = lu_factor(A.astype(np.float32))
x = lu_solve(factors, b.astype(np.float32)).astype(np.float64)
before = np.linalg.norm(x-truth)
for _ in range(4):
    residual = b-A@x  # evaluated in float64, not from rounded A
    correction = lu_solve(factors, residual.astype(np.float32))
    x += correction.astype(np.float64)
after = np.linalg.norm(x-truth)
assert after < before*1e-3
assert np.linalg.norm(b-A@x) < 1e-12
print("forward error before / after refinement:", before, after)
```

This checks one benign CPU example, not all mixed-precision solvers. Increasing
precision cannot identify a coefficient direction absent from the data.

### A practical solver decision guide

| Problem | Starting method | Essential check |
| --- | --- | --- |
| Square, general, nonsingular dense system | Pivoted LU through a solve routine | Residual and conditioning |
| Symmetric positive-definite system | Cholesky | Positive definiteness, not symmetry alone |
| Overdetermined full-column-rank fit | QR or a least-squares routine | Residual and feature scaling |
| Rank-deficient or underdetermined fit | SVD-based minimum-norm least squares | Rank threshold and identifiability |
| Regularized least squares | Augmented QR/SVD or an appropriate positive-definite solve | Penalty units and loss normalization |
| Huge sparse positive-definite system | Preconditioned conjugate gradients | Convergence and preconditioner quality |
| Huge sparse rectangular fit | An iterative least-squares method such as LSQR | Stopping tolerance and regularization |

Use a solver to apply an inverse; form an inverse only when its entries are genuinely required. Preserve sparse structure, reuse factorizations, and inspect diagnostics in the same units as the application.



## 20. Computing with positive-definite and large matrices

### Cholesky, derived entry by entry

For a real SPD matrix, the Cholesky factorization $A=LL^\top$ has a unique
lower-triangular $L$ with positive diagonal. Matching entries gives the
recurrences

$$
L_{jj}=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^2},\qquad
L_{ij}=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}},\quad i>j.
$$

Positive definiteness ensures positive pivots in exact arithmetic. Consider

$$
A=\begin{bmatrix}4&2\\2&3\end{bmatrix}.
$$

We obtain $L_{11}=2$, $L_{21}=1$, and $L_{22}=\sqrt2$, hence
$L=\begin{bmatrix}2&0\\1&\sqrt2\end{bmatrix}$. To solve $Ax=(6,5)^\top$,
first solve $Ly=b$: $y_1=3$, $y_2=\sqrt2$. Then solve $L^\top x=y$, giving
$x=(1,1)^\top$.

The same factor supports several operations. The log determinant is
$2(\log2+\log\sqrt2)=\log8$. A vector $z$ with identity covariance becomes
a vector with covariance $A$ after applying $L$, because
$\operatorname{Cov}(Lz)=LI L^\top=A$. Conversely, applying $L^{-1}$ whitens
that covariance. Triangular factors are thus useful for both inference and
simulation.

```python
import numpy as np

A = np.array([[4., 2.], [2., 3.]])
b = np.array([6., 5.])
L = np.linalg.cholesky(A)
x = np.linalg.solve(L.T, np.linalg.solve(L, b))
assert np.allclose(L @ L.T, A)
assert np.allclose(x, [1., 1.])
assert np.allclose(2 * np.log(np.diag(L)).sum(), np.log(8.))
```

Cholesky failure can indicate a genuinely indefinite matrix, an incorrect
symmetry assumption, or a nearly singular SPD matrix whose small positive
directions were lost in rounding. Adding a diagonal "jitter" is a model change,
not a universal repair. Validate symmetry and scale first. Some routines consume
only one triangle rather than checking both halves; see the
[NumPy Cholesky contract](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html).

### Conjugate gradients and preconditioning

For SPD $A$, solving $Ax=b$ is equivalent to minimizing
$f(x)=\tfrac12x^\top Ax-b^\top x$. Gradient descent repeatedly follows the
negative gradient $r=b-Ax$, but on a long narrow quadratic valley it can zigzag
and converge slowly. Conjugate gradients (CG) chooses search directions that are
orthogonal in the $A$-inner product: $p_i^\top Ap_j=0$ for $i\ne j$.

Starting from $x_0$, let $r_0=b-Ax_0$ and $p_0=r_0$. For a nonzero residual,
the unpreconditioned iteration is

$$
\alpha_k=\frac{r_k^\top r_k}{p_k^\top Ap_k},\quad
x_{k+1}=x_k+\alpha_kp_k,\quad
r_{k+1}=r_k-\alpha_kAp_k,
$$

$$
\beta_k=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k},\qquad
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

Each iteration needs a matrix-vector product and a few vector operations. In
exact arithmetic, CG reaches the solution in at most the dimension of $A$
iterations; practical finite-precision stopping is based on tolerances, not on
that guarantee. One classical error bound, with
$\|e\|_A=\sqrt{e^\top Ae}$, is

$$
\|x_k-x_*\|_A\leq
2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k
\|x_0-x_*\|_A,
\qquad \kappa=\kappa_2(A).
$$

This is a worst-case bound; eigenvalue clustering can make convergence much
faster. It nevertheless explains why conditioning matters even when no explicit
inverse or dense factorization is computed.

A preconditioner is a cheaply solvable approximation to the operator. For an
SPD $M=CC^\top$, introduce $x=C^{-\top}z$ and solve

$$
(C^{-1}AC^{-\top})z=C^{-1}b.
$$

The transformed matrix remains SPD. The goal is to make its spectrum better
clustered while keeping applications of the preconditioner inexpensive. A
diagonal preconditioner corrects coordinate scales but may do little for strong
off-diagonal coupling. Although algorithms often write $M^{-1}A$, this product
is not necessarily symmetric in the ordinary Euclidean inner product; using a
proper preconditioned CG implementation handles the geometry correctly.

In SciPy's API, the argument named `M` represents the action of an approximate
inverse, rather than the matrix $M$ in the factorization convention above.
Check this distinction before passing a preconditioner.

SciPy's [CG interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
accepts sparse matrices and linear operators. It expects a Hermitian
positive-definite system and exposes a convergence status. Inspect that status
and compute a true residual; silently consuming an unconverged iterate is not a
successful solve. For a rectangular least-squares problem, use an appropriate
method such as LSQR rather than pretending the original operator is SPD.

### Power iteration and dominant directions

For a symmetric matrix with a unique largest-magnitude eigenvalue,
power iteration repeatedly computes

$$
z_{k+1}=Av_k,\qquad v_{k+1}=z_{k+1}/\|z_{k+1}\|_2.
$$

Expanding the initial vector in an orthonormal eigenbasis shows why this works:
after $k$ products the coefficient along $q_i$ is multiplied by $\lambda_i^k$.
Relative to the dominant coefficient, the next-largest contribution shrinks at
rate $|\lambda_2/\lambda_1|^k$. Initialization must have a nonzero component in
the dominant direction. A zero product cannot be normalized, and tied leading
magnitudes can prevent convergence to a unique direction.

For $A=\operatorname{diag}(5,2)$ and initial vector proportional to $(1,1)$,
the ratio of the second coordinate to the first after $k$ steps is $(2/5)^k$.
The Rayleigh quotient $v_k^\top Av_k$ estimates the eigenvalue. Check the
eigenpair residual $\|Av_k-(v_k^\top Av_k)v_k\|$, not just how little the
reported eigenvalue changes.

For covariance matrices, power iteration can target a leading principal
direction without computing every eigenvector. Apply
$v\mapsto X_c^\top(X_cv)/(n-1)$ to avoid constructing covariance explicitly.
A small spectral gap makes individual directions slow to isolate. Block
iterations or Krylov-subspace methods are more suitable when several leading
directions are needed. General nonsymmetric eigenproblems require additional
care: non-orthogonal eigenvectors and complex pairs can change the behavior.

### Randomized low-rank approximation

When only the top $k$ singular directions of a huge matrix are needed, computing
a full SVD may be unnecessary. A randomized range finder sketches the column
space:

1. Draw a random matrix $\Omega\in\mathbb R^{d\times\ell}$, with
   $\ell=\min(k+p,m,d)$ and a small oversampling allowance $p$, for
   a target rank $k\leq\min(m,d)$.
2. Form $Y=A\Omega$ and obtain an orthonormal basis $Q$ for its columns.
3. Compute an SVD of the smaller matrix $B=Q^\top A$.
4. Lift the left vectors back with $U\approx Q\widetilde U$ and truncate.

Random combinations tend to sample directions carrying substantial energy,
but the result is approximate and probabilistic. With a slowly decaying
spectrum, extra power steps can sharpen separation. These steps require
reorthogonalization to prevent floating-point arithmetic from losing weaker
directions. Oversampling and iteration counts trade computation for accuracy;
validate reconstruction error rather than assuming a chosen rank guarantees
an acceptable result. This is an algorithm for approximating a spectral
subspace, not a change to the definition of SVD or PCA. The original
[randomized decomposition survey](https://arxiv.org/abs/0909.4061) develops this
range-finding framework and its error guarantees.

### Operation counts guide implementation choices

For dense matrices and classical algorithms, leading-order costs are roughly:

| Operation | Arithmetic scale | Important qualification |
|---|---|---|
| $A_{m\times d}x$ | $O(md)$ | Often limited by memory movement |
| $A_{m\times d}B_{d\times k}$ | $O(mdk)$ | Blocking and hardware strongly affect runtime |
| LU or Cholesky of a $d\times d$ matrix | $O(d^3)$ | Cholesky has a smaller leading constant |
| Solve with triangular factors | $O(d^2)$ per target | Reuse factors for repeated targets |
| Reduced QR of $n\times d$, $n\ge d$ | $O(nd^2)$ | Full orthogonal factors need additional storage |
| Dense reduced SVD | $O(md\min(m,d))$ | Leading-order scale, not a timing prediction |
| Sparse matrix-vector product | $O(\operatorname{nnz}(A))$ | Assumes a suitable sparse representation |
| Apply $U_k\Sigma_kV_k^\top$ to a vector | $O(k(m+d))$ | Avoid reconstructing the full matrix |

Memory layout, batching, communication, sparsity patterns, and precision can
matter as much as asymptotic counts. A faster kernel for the wrong mathematical
operation remains the wrong algorithm.


## 21. Matrix calculus and neural-network shapes

### Trace differentials prevent transpose mistakes

For matrices of equal shape, the Frobenius inner product is $\langle A,B\rangle_F=\operatorname{tr}(A^TB)$. Define the gradient by

$$dL=\langle\nabla_WL,dW\rangle_F.$$

The coefficient of $dW$ must be rearranged into this form before reading off the gradient. This is safer than treating matrix derivatives like scalar fractions. Cyclically rotating a trace is allowed when dimensions conform; arbitrarily swapping factors is not.

For $L(w)=\tfrac12\|Xw-y\|^2$, write $r=Xw-y$ and $dr=X\,dw$. Then $dL=r^TX\,dw$, so $\nabla_wL=X^Tr$. The Hessian is $X^TX$, which is positive semidefinite because $v^TX^TXv=\|Xv\|^2$. Full column rank makes it positive definite and the least-squares minimizer unique.

### Batched affine layers

Use rows for observations:

$$
X\in\mathbb R^{n\times d},\quad W\in\mathbb R^{d\times h},\quad
b\in\mathbb R^h,\quad Z=XW+\mathbf1b^T\in\mathbb R^{n\times h}.
$$

An individual observation is the column vector $x_i$, so its output column is $z_i=W^Tx_i+b$. There is no contradiction: stacking transposed examples as rows transposes the individual-example expression.

Given upstream derivative $G=\nabla_ZL\in\mathbb R^{n\times h}$,

$$dZ=dX\,W+X\,dW+\mathbf1\,db^T.$$

Insert this into $dL=\operatorname{tr}(G^TdZ)$ and collect each differential:

$$
\nabla_XL=GW^T,\qquad
\nabla_WL=X^TG,\qquad
\nabla_bL=G^T\mathbf1.
$$

The bias gradient sums over the batch because the same bias affects every observation. If an average loss is used, its $1/n$ factor must already be included in $G$ or applied consistently afterward, but not twice. For an elementwise activation $H=\phi(Z)$, the derivative is $\nabla_ZL=\nabla_HL\odot\phi'(Z)$.

```python
import numpy as np

rng = np.random.default_rng(7)
X = rng.normal(size=(4, 3))
W = rng.normal(size=(3, 2))
b = rng.normal(size=2)
Y = rng.normal(size=(4, 2))
n = X.shape[0]

def loss(weights):
    residual = X @ weights + b - Y
    return np.sum(residual ** 2) / (2 * n)

G = (X @ W + b - Y) / n
grad_W = X.T @ G
grad_b = G.sum(axis=0)
grad_X = G @ W.T
direction = rng.normal(size=W.shape)
eps = 1e-6
finite_difference = (loss(W + eps * direction) - loss(W - eps * direction)) / (2 * eps)
analytic = np.sum(grad_W * direction)
assert np.allclose(finite_difference, analytic, rtol=1e-6, atol=1e-7)
assert grad_X.shape == X.shape
assert grad_b.shape == b.shape
```

A directional derivative checks all entries jointly without constructing a full Jacobian. Use several random directions for stronger coverage; this is a numerical test, not a proof. Nondifferentiable points and an excessively tiny finite-difference step require separate care.

## 22. Attention, low-rank adaptation, kernels, and graphs

### Attention as two different matrix products

For one sequence, let $X\in\mathbb R^{t\times d}$ and define

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,$$

with $Q,K\in\mathbb R^{t\times d_k}$ and $V\in\mathbb R^{t\times d_v}$. The product $QK^T\in\mathbb R^{t\times t}$ compares every query with every key. Row-wise softmax produces weights, and multiplication by $V$ forms a weighted value combination for each query:

$$O=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.$$

Here $M$ is an optional mask with zero allowed entries and negative infinity disallowed entries. Every row must retain at least one allowed entry. The original Transformer formulation uses scaled dot-product attention; see [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Why the square-root scale? Under the simplifying assumption that all query and key components in this calculation are mutually independent, centered, and have unit variance (or, more generally, corresponding products have variance one and distinct products are uncorrelated), each product has variance one and their sum has variance $d_k$. Dividing by $\sqrt{d_k}$ brings the logit's variance back to one. Learned activations need not satisfy those assumptions exactly; this calculation explains the scale rather than proving all attention statistics are normalized.

For a row $s$, subtracting $\max_j s_j$ before exponentiation leaves softmax unchanged and avoids overflow. Attention weights are nonnegative and sum to one, so each output row lies in the convex hull of its permitted value rows. This property applies to that weighted sum, not necessarily after the output projection or a residual addition. Also, $QK^T$ is not generally symmetric or positive semidefinite: queries and keys use different projections.

### Low rank describes a restriction on directions

With the row-batch convention, write a dense layer as $XW$, $W\in\mathbb R^{d\times h}$. A rank-constrained update can be parameterized as

$$\Delta W=AB,\qquad A\in\mathbb R^{d\times r},\quad B\in\mathbb R^{r\times h}.$$

Then $\operatorname{rank}(\Delta W)\le r$ and the update uses $r(d+h)$ parameters rather than $dh$. For $d=h=4096$ and $r=8$, that is $65{,}536$ instead of $16{,}777{,}216$ parameters. Its application can be computed as $(XA)B$, without first materializing $\Delta W$.

LoRA freezes the base weights and trains low-rank update factors, with a scaling convention often written $\alpha/r$. Matrix-factor order changes with the author's weight orientation; the dimensions are the reliable guide. See the original [LoRA paper](https://arxiv.org/abs/2106.09685).

A low-rank update does not imply that $W+\Delta W$ is low rank: the frozen $W$ may already have full rank. Nor does the factorization identify unique latent coordinates. For any invertible $R\in\mathbb R^{r\times r}$,

$$AB=(AR)(R^{-1}B).$$

This change of basis alters the factors but not their product. It explains why comparing factor entries across independent runs can be misleading.

### Matrix completion is not truncated SVD with zeros

Suppose a user-item matrix is only partially observed on indices $\Omega$. A factor model might minimize

$$\sum_{(i,j)\in\Omega}(M_{ij}-u_i^Tv_j)^2+\lambda(\|U\|_F^2+\|V\|_F^2).$$

Unobserved entries are absent from the data term. Filling them with zero and computing a truncated SVD solves a different problem: it treats every missing interaction as an observed zero. Rank alone also cannot guarantee recoverability. If an entire row is never observed, its latent vector may remain unconstrained by the observations. Sampling coverage, structural assumptions, and noise determine what can be inferred.

### Gram matrices and the kernel viewpoint

For rows $x_i^T$ of $X$, the Gram matrix is $K=XX^T$ with entries $K_{ij}=x_i^Tx_j$. It is symmetric positive semidefinite because

$$a^TKa=a^TXX^Ta=\|X^Ta\|^2\ge0.$$

More generally, if $k(x,z)=\langle\phi(x),\phi(z)\rangle$ for a feature map, every finite kernel matrix is PSD. A similarity function being intuitive or symmetric is not enough. For example, $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ has eigenvalues $3$ and $-1$, so it cannot be a Gram matrix in a real Euclidean feature space.

Ridge has a useful dual expression. Since

$$ (X^TX+\lambda I)X^T=X^T(XX^T+\lambda I),$$

we obtain $w=X^T\alpha$ where $(XX^T+\lambda I)\alpha=y$. A new prediction becomes $x_*^Tw=\sum_i\alpha_i x_*^Tx_i$. Replacing inner products with a valid kernel gives kernel ridge regression. The primal solve has dimension $d$; the dual solve has dimension $n$, so relative dataset and feature sizes matter. A kernel avoids explicitly constructing a feature vector but may still require an expensive dense $n\times n$ matrix.

### Graph Laplacians measure disagreement

Let $W$ be a symmetric adjacency matrix with nonnegative weights and zero diagonal. Define $D_{ii}=\sum_jW_{ij}$ and the unnormalized Laplacian $L=D-W$. For a real signal $f$ on the vertices,

$$f^TLf=\frac12\sum_{i,j}W_{ij}(f_i-f_j)^2\ge0.$$

The identity follows by expanding the square and using symmetry. It makes $L$ PSD and interprets its quadratic form as total variation across connected neighbors. Constant signals have zero energy because $L\mathbf1=0$. More generally, the nullspace consists of signals constant on each connected component, so its dimension equals the number of components.

For a three-node unit-weight path,

$$L=\begin{bmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{bmatrix}.$$

The signal $(1,0,-1)^T$ has energy $(1-0)^2+(0-(-1))^2=2$. Adding the same constant to every entry does not change that energy. Penalizing $f^TLf$ favors smooth predictions over a graph; small-eigenvalue eigenvectors describe slowly varying patterns. The normalized Laplacian changes the geometry by incorporating degrees, so it should not be substituted without considering what notion of smoothness the application needs.



## 23. Foundations: self-checks with worked answers

### 1. Find every solution, then select the minimum-norm one

**Question.** For
$A=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}$ and
$b=(2,3)^\top$, find the rank, nullspace, and all exact solutions. Which
solution has the smallest Euclidean norm?

**Answer.** The first two columns are independent, so the rank is two and
the nullity is one. The equations $x_1+x_3=2$ and $x_2+x_3=3$ give
$x=(2-t,3-t,t)^\top$. The nullspace is spanned by $(-1,-1,1)^\top$.
The squared norm is $13-10t+3t^2$, minimized when $-10+6t=0$, hence
$t=5/3$. The minimum-norm solution is $(1/3,4/3,5/3)^\top$. Its dot product
with the null vector is zero, confirming that it belongs to the row space.
Notice that two equations can determine a unique minimum-norm choice without
determining a unique exact solution.

### 2. Diagnose an impossible target geometrically

**Question.** Can $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$ map any vector
to $b=(1,3)^\top$? What is the closest attainable output?

**Answer.** Every output is a multiple of $u=(1,2)^\top$, but the target's
second coordinate is not twice its first. The closest output is its orthogonal
projection onto that line:
$\widehat b=u(u^\top b)/(u^\top u)=(7/5,14/5)^\top$.
The residual is $(-2/5,1/5)^\top$ and is perpendicular to $u$ because
$-2/5+2/5=0$. Its squared norm is $1/5$. This irreducible residual is a
restriction of the column space, not a failure of the optimizer.

### 3. Do two similarity rankings agree?

**Question.** For query $q=(1,0)^\top$, compare candidates
$a=(2,0)^\top$ and $b=(100,100)^\top$ using raw dot product and cosine.

**Answer.** The dot products are $2$ and $100$, so raw dot product ranks $b$
higher. The cosine scores are $1$ and $1/\sqrt2$, so cosine ranks $a$ higher.
Normalizing the candidates removes the norm advantage of $b$. Neither ranking
is intrinsically correct for every learned embedding: the question is whether
magnitude should influence retrieval in that model.

### 4. Construct a projector without an inverse

**Question.** Let $q_1=(1,1,0)^\top/\sqrt2$ and $q_2=(0,0,1)^\top$.
Project $x=(3,1,4)^\top$ onto their span and verify the defining properties.

**Answer.** The vectors are orthonormal, so

$$
P=q_1q_1^\top+q_2q_2^\top
=\begin{bmatrix}1/2&1/2&0\\1/2&1/2&0\\0&0&1\end{bmatrix}.
$$

The projection is $(2,2,4)^\top$ and the residual is $(1,-1,0)^\top$.
Both basis vectors have zero dot product with the residual. Direct multiplication
gives $P^2=P$, and inspection gives $P^\top=P$. Squared lengths satisfy
$26=24+2$. The third coordinate is retained exactly; only the difference between
the first two coordinates is discarded.

### 5. Separate a classifier's score from its distance

**Question.** A boundary is $2x_1-x_2-3=0$. For $x=(4,1)^\top$, find its
score, geometric distance, and nearest boundary point. What changes if every
coefficient is multiplied by ten?

**Answer.** The score is $4$, while the distance is $4/\sqrt5$. Subtracting
$\frac45(2,-1)^\top$ gives nearest point $(12/5,9/5)^\top$; substitution
returns zero in the boundary equation. Rescaling gives score $40$ and normal
length $10\sqrt5$, leaving the distance unchanged. Thus the raw score's scale
cannot be interpreted without the normal's scale.

### 6. Find the silently incorrect residual array

**Question.** Predictions have shape `(64, 1)` and targets have shape `(64,)`.
What shape does their NumPy difference have, and why is that a problem?

**Answer.** Broadcasting aligns trailing axes. It treats the shapes as
`(64, 1)` and `(1, 64)`, giving `(64, 64)`. Entry $(i,j)$ becomes the difference
between prediction $i$ and target $j$, rather than comparing matched examples.
Use either two one-dimensional arrays or two explicit columns. A scalar mean
computed afterward can hide this mistake: the final loss has a plausible shape
even though the comparisons were wrong.

## 24. Spectral methods: self-checks with worked answers

### 1. Read an ellipse from its matrix

**Question.** Find the principal axes of
$x^\top\begin{bmatrix}3&1\\1&3\end{bmatrix}x=1$.

**Answer.** The normalized eigenvectors are $(1,1)^\top/\sqrt2$ and
$(1,-1)^\top/\sqrt2$, with eigenvalues four and two. In those coordinates,
the equation is $4z_1^2+2z_2^2=1$. The semiaxis lengths are $1/2$ and
$1/\sqrt2$, not four and two. A larger quadratic coefficient creates a shorter
axis at a fixed function value. For a covariance ellipse written with the
inverse covariance instead, the corresponding lengths scale with square roots
of covariance eigenvalues.

### 2. Why can a matrix with unit eigenvalues stretch a vector?

**Question.** Let $A=\begin{bmatrix}1&2\\0&1\end{bmatrix}$. Both eigenvalues
are one. Must it preserve Euclidean length?

**Answer.** No: $A(0,1)^\top=(2,1)^\top$ has length $\sqrt5$.
Moreover, $A^\top A=\begin{bmatrix}1&2\\2&5\end{bmatrix}$ has eigenvalues
$3+2\sqrt2$ and $3-2\sqrt2$. Its singular values are therefore
$\sqrt2+1$ and $\sqrt2-1$. The largest singular value, not the largest
eigenvalue magnitude, gives worst-direction stretching. The matrix is a shear,
not an orthogonal transformation.

### 3. Compute both optimal rank-two errors

**Question.** A matrix has singular values $5,3,1,0$. What are the best
rank-at-most-two spectral error, Frobenius error, and retained energy fraction?

**Answer.** The spectral error is the first discarded singular value, one.
The Frobenius error is $\sqrt{1^2+0^2}=1$; its square is also one in this
example, but that is not a general identity between error and squared error.
Retained squared energy is $(25+9)/(25+9+1)=34/35$. A high fraction of retained
energy is a reconstruction statement; it says nothing by itself about whether
the discarded direction contained a class-separating signal.

### 4. Compare centered PCA with uncentered SVD

**Question.** Two observations are $(10,1)$ and $(10,-1)$. Which direction is
selected by one-component centered PCA, and which by uncentered SVD?

**Answer.** The mean is $(10,0)$, so centered data contain only variation in the
second coordinate. PCA selects the vertical direction and its sample variance
is $2/(2-1)=2$. The uncentered Gram matrix is
$\operatorname{diag}(200,2)$, so uncentered SVD selects the horizontal direction
dominated by the mean. Centering changes the objective from energy relative to
zero to variation around the mean.

### 5. Can whitening recover a discarded feature?

**Question.** A rank-two centered dataset in three-dimensional feature space has
covariance eigenvalues $9,1,0$. Can ordinary inverse-square-root whitening
produce three coordinates with identity covariance?

**Answer.** No. There is no finite inverse square root of the zero eigenvalue,
and the data contain no variation in that direction. Retain the two nonzero
components and scale by $1/3$ and $1$ to obtain a two-dimensional identity
covariance. Replacing zero by a positive regularization value makes an operator
well-defined, but it does not create variance in a direction that was constant
in the training data. Information loss and coordinate rescaling are different
operations.

### 6. Why might two correct PCA implementations return different vectors?

**Question.** The top two covariance eigenvalues are equal. Must two correct
implementations return the same first eigenvector, up to sign?

**Answer.** No. Any orthonormal basis of that two-dimensional eigenspace is
valid. The two outputs can differ by an arbitrary two-dimensional rotation, not
just a sign flip. Retaining the whole tied eigenspace gives the same projector
and reconstruction. Retaining only one direction inside the tie gives a
nonunique but equally optimal rank-one PCA solution. Compare the mathematical
object that is actually identifiable: a subspace or reconstructed matrix, not
an arbitrary basis convention.


## 25. Numerical and ML self-checks with worked answers

### 1. Can a zero pivot prove singularity?

**Question.** Elimination encounters the top-left zero in $A=\begin{bmatrix}0&2\\1&3\end{bmatrix}$. Is the system unsolvable?

**Answer.** No. Its determinant is $-2$, so it is invertible. Swapping rows makes the first pivot one and leaves the upper-triangular matrix $\begin{bmatrix}1&3\\0&2\end{bmatrix}$. A zero pivot without considering available row swaps only describes the current elimination order. It does not characterize the original matrix's rank.

### 2. What does a residual orthogonality check establish?

**Question.** For a candidate $w$, suppose $X^T(y-Xw)=0$. Does that prove the fit is exact?

**Answer.** It proves $w$ is a least-squares minimizer because the convex quadratic has zero gradient. It does not prove $y-Xw=0$. In the line-fitting example above, the residual is $(-1/6,1/3,-1/6)^T$, which is nonzero but perpendicular to both columns. If $X$ has independent columns the minimizing parameter is unique; otherwise only the fitted prediction must be unique.

### 3. How much conditioning is lost by normal equations?

**Question.** A full-column-rank matrix has singular values $10$, $1$, and $10^{-4}$. Compare its condition number with that of its normal-equation matrix.

**Answer.** $\kappa_2(X)=10/10^{-4}=10^5$. The eigenvalues of $X^TX$ are $100$, $1$, and $10^{-8}$, giving condition number $10^{10}$. The squaring comes from the singular values, not from the number of rows. QR avoids forming this squared-spectrum matrix, though it cannot remove the original problem's sensitivity.

### 4. Find a minimum-norm interpolator

**Question.** Among all solutions of $2w_1+w_2=5$, which has minimum Euclidean norm?

**Answer.** The row space is the line spanned by $(2,1)^T$. The minimum-norm solution lies on that line, so write $w=c(2,1)^T$. Substitution gives $5c=5$, hence $w=(2,1)^T$. Every other solution adds $t(-1,2)^T$, an orthogonal null vector. Its squared norm becomes $5+5t^2$, proving minimality at zero.

### 5. Does ridge uniformly scale a solution?

**Question.** For $X=\operatorname{diag}(3,1)$, $y=(3,1)^T$, and $\lambda=2$, compute ridge coefficients and explain the different shrinkage.

**Answer.** The normal-equation matrix is $\operatorname{diag}(11,3)$ and $X^Ty=(9,1)^T$, so $w=(9/11,1/3)^T$. Ordinary least squares gives $(1,1)^T$. The strong singular direction retains $9/11$ of its coefficient, while the weak one retains only $1/3$. Ridge is isotropic in its parameter penalty, but data support is anisotropic.

### 6. Can a tiny residual accompany a large error?

**Question.** Let $A=\operatorname{diag}(1,10^{-12})$, $b=(1,10^{-12})^T$, and $\hat x=(1,0)^T$. Evaluate the residual and forward error.

**Answer.** The exact solution is $(1,1)^T$. The residual is $(0,10^{-12})^T$, whose norm is tiny. The forward error is $(0,-1)^T$, whose norm is one. Multiplication by $A$ almost erases errors along the second coordinate, so residual size cannot detect that error without considering conditioning.

### 7. Why is determinant magnitude the wrong warning signal?

**Question.** Compare $A=10^{-6}I_3$ and $B=\operatorname{diag}(10^6,1,10^{-6})$.

**Answer.** $\det A=10^{-18}$ but all its singular values are equal, so $\kappa_2(A)=1$. Meanwhile $\det B=1$ but $\kappa_2(B)=10^{12}$. Uniform scaling changes determinant magnitude without creating directional imbalance. A determinant can indicate exact singularity algebraically, but its floating-point magnitude is not a useful general test of numerical invertibility.

### 8. Derive the shared-bias gradient

**Question.** An affine layer has three examples and upstream gradient $G=\begin{bmatrix}1&2\\-1&4\\3&-2\end{bmatrix}$. What is $\nabla_bL$?

**Answer.** Every row contains the same bias, so the derivative contributions add: $\nabla_bL=G^T\mathbf1=(3,4)^T$. Averaging these values is correct only if the loss definition requires a batch average that has not already been included in $G$. Blindly averaging again changes the optimization scale by another factor of three.

### 9. Is an attention score matrix always a Gram matrix?

**Question.** With $Q,K\in\mathbb R^{t\times d_k}$, is $QK^T$ necessarily PSD?

**Answer.** No. If $Q\ne K$, symmetry is not guaranteed. Even a symmetric result can be negative semidefinite, as when $K=-Q$, yielding $-QQ^T$. Only $QQ^T$ itself is automatically a Gram matrix. Row-wise softmax also does not generally preserve symmetry, so attention weights should not be treated as a covariance matrix.

### 10. Does a rank-two update make a model rank two?

**Question.** If $W$ is an invertible $100\times100$ matrix and $\Delta W=AB$ has rank at most two, must $W+\Delta W$ have rank at most two?

**Answer.** No. The update is restricted, not the entire weight matrix. For example, choosing $A=0$ leaves $W$ unchanged and full rank. More generally, the rank inequality gives $\operatorname{rank}(W+\Delta W)\ge\operatorname{rank}(W)-\operatorname{rank}(\Delta W)\ge98$. Some updates can make the matrix singular, but they cannot force arbitrary collapse to rank two in this example.

### 11. Choose a primal or dual ridge solve

**Question.** There are $n=200$ examples and $d=20{,}000$ explicit features. Which dense ridge system is smaller, and what caveat remains?

**Answer.** The primal matrix is $20{,}000\times20{,}000$; the dual matrix $XX^T+\lambda I$ is $200\times200$. Solving for $\alpha$ and obtaining $w=X^T\alpha$ is much smaller in factorization size. Building the Gram matrix still costs work and may affect numerical behavior. Positive regularization makes the dual matrix invertible, but selecting its strength and checking scaling remain modeling decisions.

### 12. Read graph connectivity from zero energy

**Question.** A graph has two disconnected components. Can a nonconstant vector have $f^TLf=0$?

**Answer.** Yes. Assign one constant to every vertex in the first component and a different constant to every vertex in the second. Every existing edge still connects equal values, so every squared difference in the energy is zero. There are two independent component-indicator vectors in the nullspace. This connects an algebraic property, the multiplicity of eigenvalue zero, with the graph's connectivity.


## 26. A compact decision checklist

Before implementing a linear-algebra expression, answer five questions:

1. **Shapes:** What does each axis mean, and which indices are contracted?
2. **Geometry:** Which subspace contains the output, and which input directions
   can be lost?
3. **Assumptions:** Is the matrix square, full rank, symmetric, positive definite,
   centered, or sparse? Which claims actually require those properties?
4. **Numerics:** Will forming a Gram matrix square the condition number? Can a
   factorization, a solve, or an operator application avoid a large intermediate?
5. **Validation:** Can a residual, reconstruction identity, gradient check, or
   known small example expose an error?

The most useful habits are to separate exact algebra from numerical algorithms,
parameter identifiability from prediction quality, and reconstruction quality
from task relevance. These distinctions recur in almost every ML system.

## 27. Further study and connections

- [MIT 18.06 lecture notes](https://www-math.mit.edu/~gs/LectureNotes/) organize
  the core theory around subspaces, orthogonality, least squares, and
  factorizations.
- [Stanford EE263 lectures](https://see.stanford.edu/Course/EE263)
  connect linear dynamical systems, estimation, least squares, and matrix
  structure.
- [Calculus and matrix calculus](./calculus.md) develops derivatives and the
  chain rule beyond the affine-layer derivation here.
- [Numerical computing](./numerical-methods.md) examines floating point,
  cancellation, mixed precision, and reproducibility in greater detail.
- [Optimization](./optimization.md) builds on curvature, conditioning, and
  quadratic models to explain practical training algorithms.
- [Linear models](../ml/linear-models.md) develops the modeling and statistical
  interpretation of regression; [unsupervised learning](../ml/unsupervised-learning.md)
  places PCA beside other dimensionality-reduction and clustering methods.
- [Attention and transformers](../deep-learning/attention-and-transformers.md)
  extends the attention shape calculation into the complete architecture.
