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


2. Dot product — #dot-product-viz

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


3. Linear classifier — #interactive-classifier

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


4. Eigenvectors / PCA — #pca-visualization

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


5. SVD — #svd-visualization

```html
<figure class="widget">
  <div class="widget__k">Figure 05 · Singular value decomposition</div>
  <h4 class="widget__t">One matrix, three jobs</h4>
  <p class="widget__hint">Any matrix \(A\) splits into a rotation, a stretch, and another rotation. The shaded squares down the diagonal of \(\Sigma\) are the singular values, drawn fading because they always arrive in descending order.</p>
  <div class="widget__stage widget__stage--svd" id="svd-visualization"></div>
  <figcaption class="widget__cap">Keep the first few σ and throw the rest away — that is a low-rank approximation</figcaption>
</figure>
```
