---
order: 1
description: Why linear algebra is the bedrock of machine learning — vectors, dot products, hyperplanes, eigenvectors, and SVD, with interactive visualizations.
meta: Math for ML · interactive
scripts:
  - https://cdn.jsdelivr.net/npm/d3@7
  - /assets/pages/linear-algebra.js
---

# Linear Algebra: The Secret Sauce of Machine Learning

Admit it, the words "Linear Algebra" probably give you flashbacks to a class you
thought you'd never need. You figured it was just for mathematicians and people
who enjoy pain. Well, surprise! It turns out this stuff is the bedrock of AI,
from your Netflix queue to self-driving cars. Without it, ML would just be... L.

## So, Why Should You Care?

Think of Machine Learning as building with LEGOs. You can follow the
instructions and build a cool spaceship, but if you want to design your own
Death Star, you need to understand the physics of how the bricks connect. Linear
Algebra is that physics. It's the language we use to describe and manipulate the
massive datasets that fuel modern AI.

For example:

- **Netflix Recommendations:** Ever wonder how Netflix knows you'll love a weird
  documentary about competitive cheese rolling? It uses a powerful Linear Algebra
  technique called Singular Value Decomposition (SVD) to analyze a giant matrix of
  user ratings. Your taste is literally a vector in a high-dimensional space!
- **Image Recognition:** When your phone recognizes your face, it's converting
  your picture into a matrix of pixel values and performing a series of matrix
  multiplications. You're not a face; you're a math problem. A very beautiful math
  problem, of course.
- **Natural Language Processing (NLP):** How does Google Translate work? By
  representing words and sentences as vectors in a process called "word
  embedding." The relationships between words are captured by the distances and
  angles between these vectors.

Ready to peek behind the curtain? Let's dive in. Don't worry, we'll keep the math
interesting. Mostly.

## The Building Blocks: Scalars, Vectors, and Matrices

Let's start with the absolute basics. These are the nouns of our new language.

| | What it is | Example |
|---|---|---|
| **Scalar** | A single, lonely number. Like the temperature, your age, or the number of times you've promised to learn Linear Algebra. | $7$ |
| **Vector** | An ordered list of numbers. Think of it as a point in space or a set of features. A fish could be represented as `[weight, length, color]`. | `[1.2, 15.5, 0.8]` |
| **Matrix** | A grid of numbers, or a list of vectors. This is where the magic happens. A whole dataset of fish is a matrix, where each row is one fish. | a grid of the above |

```html
<figure class="widget">
  <div class="widget__k">Figure 01 · Matrix operations</div>
  <h4 class="widget__t">Scale it, flip it, put it back</h4>
  <p class="widget__hint">Four fish, three features each — one row per fish. Hover any cell to light up the row and the column it belongs to. Then multiply the whole grid by a scalar, or transpose it and watch \(4 \times 3\) become \(3 \times 4\).</p>
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
  <figcaption class="widget__cap">Each row is one fish: [ weight, length, colour ]</figcaption>
</figure>
```

## Vector Operations: The Dot Product, a.k.a. "The Similarity Meter"

The dot product is one of the most fundamental operations in all of ML. On the
surface, it's a simple calculation, but its true power lies in its geometric
meaning: it tells us how "similar" two vectors are.

$$a \cdot b = \|a\| \|b\| \cos(\theta)$$

```html
<figure class="widget">
  <div class="widget__k">Figure 02 · Dot product</div>
  <h4 class="widget__t">The similarity meter</h4>
  <p class="widget__hint">Drag either arrow tip, or type the numbers directly. The amber wedge is the angle \(\theta\) between the two vectors — the whole reason the dot product measures similarity.</p>
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

That $\cos(\theta)$ is the key. It tells us about the angle ($\theta$) between
the two vectors.

- If the vectors point in the same direction, $\cos(\theta) = 1$, and the dot
  product is at its max. They are very similar!
- If the vectors are perpendicular, $\cos(\theta) = 0$, and the dot product is
  zero. They have nothing in common, like a cat and a vacuum cleaner.
- If they point in opposite directions, $\cos(\theta) = -1$, and the dot product
  is negative. They are "anti-similar."

**ML Connection:** This is used everywhere! Neural networks are essentially a
series of dot products. In NLP, the dot product between word vectors measures
semantic similarity. "King" and "Queen" will have a high dot product; "King" and
"Cabbage" will not.

## Lines and Planes: The Great Dividers

Many machine learning models, at their core, are just trying to find a line (in
2D), a plane (in 3D), or a "hyperplane" (in more dimensions) to separate data
points into different classes. This is the essence of a **linear classifier**.

The equation of a plane is $w_1x_1 + w_2x_2 + w_3x_3 + b = 0$. This can be
written beautifully using the dot product: $\vec{w} \cdot \vec{x} + b = 0$. The
vector $\vec{w}$ is the "normal vector" — it's perpendicular to the plane and
defines its orientation.

A model like a **Support Vector Machine (SVM)** is obsessed with finding the
perfect hyperplane that has the maximum possible margin (distance) between the
different classes. It's like a bouncer at a club creating the biggest possible
"no-man's-land" between two rival groups to prevent a fight.

```html
<figure class="widget">
  <div class="widget__k">Figure 03 · Linear classifier</div>
  <h4 class="widget__t">Interactive vibe check</h4>
  <p class="widget__hint">At a trendy coffee shop, can you separate the <strong>Artisanal Hipsters (🥸)</strong> from the <strong>Silicon Valley Techies (🚀)</strong>? <strong>Your mission:</strong> drag the dot to swing the line and find the optimal "Vibe" separator. Hover a customer for their coordinates.</p>
  <div class="widget__stage" id="interactive-classifier"></div>
  <div class="widget__legend">
    <span><i class="widget__key widget__key--line" aria-hidden="true"></i>Decision boundary \(\vec{w} \cdot \vec{x} + b = 0\)</span>
  </div>
</figure>
```

## Eigen-stuff: What's So Special?

"Eigen" is German for "own" or "characteristic." Eigenvectors and eigenvalues are
the characteristic properties of a matrix. When you apply a matrix (a
transformation like squishing, stretching, or rotating) to one of its
eigenvectors, something amazing happens: the vector's direction doesn't change!
It only gets scaled (stretched or shrunk).

$$A\vec{v} = \lambda\vec{v}$$

Here, $A$ is the matrix, $\vec{v}$ is the mighty eigenvector, and $\lambda$
(lambda) is the eigenvalue — the factor by which the eigenvector is scaled.

```html
<figure class="widget">
  <div class="widget__k">Figure 04 · Eigenvectors</div>
  <h4 class="widget__t">The directions the data actually cares about</h4>
  <p class="widget__hint">A tilted cloud of 150 points with its two principal components drawn on top. Each arrow is an eigenvector of the covariance matrix; its length is proportional to the variance — the eigenvalue \(\lambda\) — captured along it. Keep the long one, drop the short one, and you have gone from 2D to 1D while losing almost nothing.</p>
  <div class="widget__stage" id="pca-visualization"></div>
  <div class="widget__legend">
    <span><i class="widget__key widget__key--pc1" aria-hidden="true"></i>PC1 · most variance</span>
    <span><i class="widget__key widget__key--pc2" aria-hidden="true"></i>PC2 · the leftovers</span>
  </div>
</figure>
```

**ML Connection: Principal Component Analysis (PCA).** This is a powerhouse
technique for dimensionality reduction. Imagine your data has 100 features (100
dimensions). It's impossible to visualize and slow to process. PCA uses
eigenvectors to find the "principal components" — the directions in your data
where the variance is highest. By keeping only the top few principal components
(the eigenvectors with the largest eigenvalues), you can squish your data down to
2 or 3 dimensions while losing as little information as possible. It's like
creating a really good summary of a book.

## SVD: The Swiss Army Knife of Linear Algebra

Singular Value Decomposition (SVD) is the final boss of many linear algebra
courses, but it's incredibly useful. It states that any matrix $A$ can be
factored into three other matrices:

$$A = U \Sigma V^T$$

- $U$: A matrix of "left singular vectors" (describes the axes of the output
  space).
- $\Sigma$ (Sigma): A diagonal matrix of "singular values." These are similar to
  eigenvalues and tell you the "importance" of each dimension.
- $V^T$: The transpose of a matrix of "right singular vectors" (describes the
  axes of the input space).

```html
<figure class="widget">
  <div class="widget__k">Figure 05 · Singular value decomposition</div>
  <h4 class="widget__t">One matrix, three jobs</h4>
  <p class="widget__hint">Any matrix \(A\) splits into a rotation, a stretch, and another rotation. The shaded squares down the diagonal of \(\Sigma\) are the singular values, drawn fading because they always arrive in descending order.</p>
  <div class="widget__stage widget__stage--svd" id="svd-visualization"></div>
  <figcaption class="widget__cap">Keep the first few σ and throw the rest away — that is a low-rank approximation</figcaption>
</figure>
```

**ML Connection: Recommendation Engines and More.** Remember Netflix? Let $A$ be
a huge matrix where rows are users and columns are movies. Most entries are empty
because no one can watch every movie. SVD is magical because it can "fill in the
blanks." By taking the SVD of this matrix and then recreating it using only the
most important singular values (a process called low-rank approximation), we get
a "denoised" version of the original matrix. The new values in the once-empty
cells are our predictions for what rating a user would give a movie! This same
principle is used for image compression and topic modeling in NLP.

## The Grand Finale

So there you have it. Linear Algebra isn't just a bunch of abstract rules; it's
the powerful, elegant engine driving the most exciting technology of our time.
From simple lines to complex decompositions, these concepts give us the tools to
find patterns, make predictions, and teach machines to learn.

The next time your phone unlocks with your face or a chatbot understands your
question, give a little nod to the humble vector and the mighty matrix. They're
the unsung heroes of the AI revolution. Now go forth and multiply (matrices, that
is).
