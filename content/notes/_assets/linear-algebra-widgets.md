Markup for the five interactive linear-algebra figures.

Paste each fenced block straight into the article body. The build treats an
html-tagged fence as raw passthrough, so the markup lands in the page untouched
and KaTeX picks up the \(...\) / \[...\] delimiters afterwards.

Requires, on the page: d3 v7, then /assets/linear-algebra.js. Every block is
independent — drop any one of them and the rest still work.


1. Interactive matrix — #interactive-matrix

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


2. Dot product — #dot-product-viz

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


3. Linear classifier — #interactive-classifier

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


4. Eigenvectors / PCA — #pca-visualization

```html
<figure class="widget">
  <div class="widget__k">Figure 04 · Eigenvectors</div>
  <h4 class="widget__t">Principal directions of a centered sample</h4>
  <p class="widget__hint">The covariance matrix is computed from these 150 centered points using the \(n-1\) denominator. Its eigenvectors define the two principal directions. Each ray has length \(2\sqrt{\lambda_i}\): two sample standard deviations, not two variances. The PC1 variance fraction quantifies what a one-dimensional projection retains.</p>
  <div class="widget__stage" id="pca-visualization"></div>
  <div class="widget__legend">
    <span><i class="widget__key widget__key--pc1" aria-hidden="true"></i>PC1 · most variance</span>
    <span><i class="widget__key widget__key--pc2" aria-hidden="true"></i>PC2 · remaining orthogonal variance</span>
  </div>
</figure>
```


5. SVD — #svd-visualization

```html
<figure class="widget">
  <div class="widget__k">Figure 05 · Singular value decomposition</div>
  <h4 class="widget__t">The dimensions of a compact SVD</h4>
  <p class="widget__hint">For a nonzero real matrix of rank \(r\), \(A=U_r\Sigma_rV_r^T\). The columns of \(U_r\) and \(V_r\) are orthonormal; they need not form square matrices. In the full SVD, the square orthogonal factors are rotations or reflections, while the rectangular diagonal factor scales and may discard directions.</p>
  <div class="widget__stage widget__stage--svd" id="svd-visualization"></div>
  <figcaption class="widget__cap">The blocks show factor dimensions, not numerical entries. Keeping the first k singular triplets gives a best rank-at-most-k approximation in the spectral and Frobenius norms.</figcaption>
</figure>
```
