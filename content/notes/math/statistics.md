---
order: 4
description: Estimators and their bias-variance, confidence intervals, hypothesis testing, A/B tests, the bootstrap, causal inference, and how to read an experiment without fooling yourself.
meta: Math for ML · core
---

# Statistics: Turning Data Into Claims

Probability starts from a known distribution and asks what data looks like.
Statistics runs the arrow backwards: given data, what can you say about the
process that produced it? Every model comparison you make, every "our model is
better" claim, every dashboard metric with a confidence band, is an act of
statistical inference — usually an implicit one, which is exactly why it goes
wrong so often.

```mermaid
flowchart LR
    W["population / true process<br/>parameter theta"] -->|"sampling"| D["observed data<br/>x_1 ... x_n"]
    D -->|"estimator"| T["estimate theta_hat"]
    T -->|"uncertainty quantification"| C["interval or test<br/>a claim about theta"]
    C -.->|"is this claim<br/>justified?"| W
```

## Descriptive statistics, and when each summary lies

### Centre

| Statistic | Definition | Breaks when |
|---|---|---|
| Mean | $\bar{x} = \frac1n\sum x_i$ | outliers or heavy tails — one billionaire moves the average income |
| Median | 50th percentile | you need differentiability or algebraic convenience |
| Mode | most frequent value | continuous data, multimodality |
| Trimmed mean | mean of the middle $1-2\alpha$ | you need every observation to count |

Mean latency measures average delay but hides tail experience. Report relevant p50, p95 and p99 alongside it. The same logic applies to
per-example loss — a mean loss of 0.4 can hide a subpopulation at 3.0.

### Spread

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Why $n-1$?** Because $\bar{x}$ was estimated from the same data, the deviations
$x_i - \bar{x}$ are constrained to sum to zero — only $n-1$ of them are free.
Dividing by $n$ gives a systematically small (biased) estimate; dividing by
$n-1$ makes $\mathbb{E}[s^2] = \sigma^2$. This is **Bessel's correction**, and
it is the simplest concrete example of "degrees of freedom".

Also worth knowing: the **interquartile range** $\mathrm{IQR} = Q_3 - Q_1$ and
the outlier fence $[Q_1 - 1.5\,\mathrm{IQR},\; Q_3 + 1.5\,\mathrm{IQR}]$, which
defines usual boxplot fences; whiskers reach extreme observations inside them, and the **coefficient of variation**
$\sigma/\mu$ for comparing spread across different scales.

### Shape

- **Skewness** — third standardised moment. Positive means a long right tail
  (income, latency, word frequency).
- **Kurtosis** is the fourth standardized moment, when finite. Positive excess means a larger standardized fourth moment than a Gaussian, not an ordering of every tail probability.

**Anscombe's quartet** and the Datasaurus dozen make the point that must be
made once and remembered forever: four datasets can share mean, variance,
correlation, and regression line while looking completely different. **Plot the
data.**

## Estimators and their properties

An **estimator** $\hat\theta$ is a function of the sample. It is itself a random
variable — run the experiment again and you get a different number. All of
inference is reasoning about that sampling distribution.

| Property | Definition | Interpretation |
|---|---|---|
| Bias | $\mathbb{E}[\hat\theta] - \theta$ | systematically off |
| Variance | $\mathrm{Var}(\hat\theta)$ | jumpy across samples |
| MSE | $\mathbb{E}[(\hat\theta-\theta)^2] = \mathrm{Bias}^2 + \mathrm{Var}$ | total error |
| Consistency | $\hat\theta \xrightarrow{p} \theta$ as $n\to\infty$ | converges eventually |
| Efficiency | attains the Cramér–Rao lower bound | lowest possible variance |

### The bias–variance decomposition, derived

For a fixed input $x$ with true value $y = f(x) + \varepsilon$,
$\mathbb E[\varepsilon\mid x]=0$, $\mathrm{Var}(\varepsilon\mid x)=\sigma^2$, independent test noise, and a model $\hat f$ trained on a random
dataset:

$$\mathbb{E}\bigl[(y - \hat f(x))^2\bigr] = \underbrace{\bigl(\mathbb{E}[\hat f(x)] - f(x)\bigr)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat f(x))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

The derivation is: add and subtract $\mathbb{E}[\hat f(x)]$ inside the square,
expand, and observe the cross term has expectation zero.

This is the single most important formula in applied ML, and it is a statistics
result, not a deep-learning one. A biased estimator can beat an unbiased one on
MSE — which is the entire justification for ridge regression, for shrinkage, for
early stopping, and for using a smaller model when data is scarce.

**A modern caveat worth stating.** The classical U-shaped test-error curve is not
the whole story for overparameterised models. **Double descent** — test error
rises to a peak at the interpolation threshold (roughly, parameters ≈ samples)
and then *falls again* as capacity grows further — is real and reproducible in
both deep networks and simple linear models. The bias–variance decomposition is
still true as algebra; the assumption that variance must grow monotonically with
capacity is what fails, because implicit regularisation from the optimiser
selects low-norm interpolants.

### Maximum likelihood estimation

$$\hat\theta_{\mathrm{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

**Worked example — the Gaussian.** With
$\log p = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2$:

$$\frac{\partial}{\partial\mu} = \frac{1}{\sigma^2}\sum(x_i-\mu) = 0 \;\Rightarrow\; \hat\mu = \bar{x}$$

$$\frac{\partial}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i-\hat\mu)^2 = 0 \;\Rightarrow\; \hat\sigma^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$$

Note the MLE variance divides by $n$ and is therefore **biased** — MLE is not
guaranteed unbiased. Consistency, asymptotic normality and asymptotic efficiency require identifiability and regularity, such as suitable interior parameters, smooth likelihood and information. Boundary and support-dependent models can violate the usual limits.

The **Fisher information** $I(\theta) = -\mathbb{E}[\partial^2 \log p/\partial\theta^2]$
uses the log density of one observation and measures expected local curvature. Under differentiability/interchange and unbiased-estimator regularity conditions, the scalar Cramér–Rao bound says
$\mathrm{Var}(\hat\theta) \ge 1/(nI(\theta))$: no unbiased estimator can do
better. Fisher information is also the metric used by natural gradient descent
and K-FAC, so it is not purely theoretical.

## Confidence intervals

A 95% CI is **not** "95% probability the parameter is in this interval" — under
the frequentist reading the parameter is fixed and the interval is random. It is
"a procedure that, run repeatedly, produces intervals covering the true value
95% of the time." The Bayesian object that *does* mean what people want is the
**credible interval**.

For iid Gaussian observations with known $\sigma$, the exact mean interval is:

$$\bar{x} \pm z_{1-\alpha/2}\frac{\sigma}{\sqrt{n}}, \qquad z_{0.975} \approx 1.96$$

Here $z_p$ denotes the lower-tail $p$ quantile. With iid Gaussian observations and unknown variance, replace $\sigma$ by $s$ and use $t_{n-1,1-\alpha/2}$ for an exact interval. The CLT permits approximations under further conditions; thirty observations are not a universal safeguard against heavy tails or dependence.

**For accuracy on a test set** (a proportion), the Wald interval
$\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$ is the usual one, and it is bad near
0 or 1. Prefer **Wilson** or **Clopper–Pearson** for extreme rates. A quick
worst-case rule: half-width $\le 0.98/\sqrt{n}$, so

| Test set size | Worst-case 95% half-width |
|---|---|
| 100 | ±9.8 pts |
| 1,000 | ±3.1 pts |
| 10,000 | ±1.0 pt |
| 100,000 | ±0.31 pts |

Print that table on the wall next to any leaderboard.

## Hypothesis testing

### The machinery

1. State $H_0$ (no effect) and $H_1$.
2. Choose a significance level $\alpha$, conventionally 0.05, **before** looking.
3. Compute a test statistic and its distribution under $H_0$.
4. The **p-value** is $P(\text{statistic at least this extreme} \mid H_0)$.
5. Reject $H_0$ if $p < \alpha$.

**What a p-value is not**: the probability $H_0$ is true; the probability the
result was chance; the size of the effect; evidence a result will replicate. It
is one conditional probability, conditioning on the null being true.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | Type I error, prob. $\alpha$ | correct (power $= 1-\beta$) |
| Fail to reject | correct | Type II error, prob. $\beta$ |

Lowering $\alpha$ trades Type I for Type II errors. More data, improved design or variance reduction can improve both errors; a larger effect is not the only alternative.

### Which test, when

| Situation | Test |
|---|---|
| One mean vs a constant, $\sigma$ unknown | one-sample $t$-test |
| Two independent group means | two-sample (Welch) $t$-test |
| Same subjects measured twice | paired $t$-test |
| Two proportions (conversion, click rate) | two-proportion $z$-test / chi-square |
| Categorical association | chi-square test of independence |
| 3+ group means | ANOVA, then post-hoc with correction |
| Non-normal, small $n$, ordinal | Mann–Whitney U, Wilcoxon signed-rank |
| Two ML models on the same test set | **paired** test — McNemar for classification |
| Distribution equality | Kolmogorov–Smirnov |

**McNemar's test deserves its own line** because it is the right tool for the
most common ML question and almost nobody uses it. Comparing two classifiers on
the *same* test set, build the disagreement table: $b$ = examples A got right and
B got wrong, $c$ = the reverse. Then

$$\chi^2 = \frac{\max(0,\lvert b-c\rvert-1)^2}{b+c},\qquad b+c>0.$$

For small discordance use the exact conditional binomial test: under equal marginal correctness, $b\mid(b+c)\sim\mathrm{Binomial}(b+c,1/2)$. If $b+c=0$, the observed correctness difference is zero and there is no evidence ($p=1$). McNemar tests marginal accuracy, not every metric.

Examples both models get right or both get wrong carry no information about which
is better, and an unpaired test wastes exactly that structure — which is why
paired comparisons preserve relevant covariance; ignoring it may lose power, depending on that covariance.

### Statistical power and sample size

Power $= 1-\beta$ is the probability of detecting a real effect. Conventionally
you target 0.8. For a two-sample comparison of means with effect size
$d = \Delta/\sigma$:

$$n \text{ per group} \approx \frac{2(z_{1-\alpha/2}+z_{1-\beta})^2}{d^2} \approx \frac{16}{d^2} \text{ for } \alpha=0.05,\ \text{power}=0.8$$

To detect a 0.1-standard-deviation effect you need about 1,600 per group. Run
this calculation *before* the experiment. An underpowered experiment that finds
nothing tells you nothing, and — worse — an underpowered experiment that *does*
find something has an inflated effect size (the winner's curse).

### Multiple comparisons

Test 20 independent hypotheses at $\alpha = 0.05$ with all nulls true and the probability of
at least one false positive is $1 - 0.95^{20} \approx 64\%$. This is why
hyperparameter sweeps produce "significant" improvements that vanish on a fresh
test set.

| Correction | Controls | Note |
|---|---|---|
| Bonferroni: use $\alpha/m$ | family-wise error rate | simple, very conservative |
| Holm–Bonferroni | FWER | uniformly more powerful than Bonferroni |
| Benjamini–Hochberg | false discovery rate | usual guarantee needs independence or specified positive dependence; arbitrary dependence requires another justification or conservative correction |

**p-hacking** is what happens without this discipline: trying variants, peeking
early, slicing subgroups, and reporting whatever crossed 0.05. Pre-register the
metric and the stopping rule, or use a sequential testing procedure designed for
peeking (always-valid p-values, mSPRT).

## A/B testing, end to end

The applied form of everything above. A checklist that survives contact with
production:

1. **One primary metric**, defined before launch. Guardrail metrics (latency,
   error rate, revenue) are monitored, not optimised.
2. **Randomisation unit** = the unit of independence. Randomise by user, not by
   request, or the same user lands in both arms and your independence assumption
   dies.
3. **Power analysis first.** Compute the minimum detectable effect for the
   traffic and duration you can afford. If the MDE is larger than any plausible
   effect, do not run the test.
4. **Cover the intended calendar population.** A prespecified Wednesday endpoint does not intrinsically bias randomization, but a short window may not represent the deployment period.
5. **A/A test** the pipeline. One rejection can occur by chance under a valid test. Investigate repeated miscalibration, sample-ratio mismatch and assignment/logging checks before diagnosing a bug.
6. **No peeking.** Continuous monitoring with a fixed-horizon test inflates false
   positives dramatically. Use sequential methods if you must look.
7. **Check for interference.** Marketplace and social products violate SUTVA —
   the treatment of one user affects the control group. Cluster or switchback
   designs help.
8. **Novelty and primacy effects.** Early lifts often decay. Look at the trend,
   not just the total.
9. **Simpson's paradox check.** Segment the result; a positive aggregate can hide
   a negative effect in every segment when segment sizes shift.

### Simpson's paradox, concretely

| Group | Model A | Model B |
|---|---|---|
| Easy examples | 93% (900 of 970) | **95%** (190 of 200) |
| Hard examples | 60% (18 of 30) | **65%** (520 of 800) |
| **Overall** | **91.8%** (918 of 1000) | 71.0% (710 of 1000) |

B is better on *both* segments and much worse overall, because B was evaluated
mostly on hard examples. Aggregate comparisons across non-identical
distributions are meaningless. This is the same failure mode as comparing models
on different test splits, and it is why fixed benchmarks exist.

## Resampling: the bootstrap and permutation tests

When you cannot write down a sampling distribution, simulate one.

### Bootstrap

Resample $n$ points **with replacement** from your data, recompute the
statistic, repeat $B \approx 10{,}000$ times. The spread of those values
estimates the sampling distribution.

```python
import numpy as np

def bootstrap_ci(data, statistic, B=10_000, alpha=0.05, seed=0):
    data = np.asarray(data)
    if data.ndim == 0 or len(data) == 0 or not 0 < alpha < 1 or B < 1:
        raise ValueError("Require nonempty observations, B >= 1 and 0 < alpha < 1")
    rng = np.random.default_rng(seed)
    n = len(data)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)          # with replacement
        stats[b] = statistic(data[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi

# Minimal percentile interval for suitably iid rows and a well-defined statistic.
```

Bootstrap intervals can approximate uncertainty for regular quantiles and nonlinear metrics when resampling matches the sampling design and the statistic remains well-defined. It fails for extreme-order statistics (the maximum),
for very small $n$, and under strong dependence (use a block bootstrap for time
series).

For comparing two models on the same test set, **bootstrap the paired
difference**, not each score separately. Resample example indices once and
compute $\Delta = \mathrm{score}_A - \mathrm{score}_B$ on the same resample. If
the 95% interval for $\Delta$ excludes zero, this is evidence against zero under the resampling assumptions, not certainty.

### Permutation tests

Under $H_0$ the group labels are exchangeable. Shuffle them thousands of times,
recompute the statistic, and see where the observed value falls in that null
distribution. Validity requires the relevant exchangeability null. Complete enumeration can be finite-sample exact; random permutations add Monte Carlo uncertainty. Match paired swaps, group permutation or independent shuffles to the design; see [SciPy's definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

### Jackknife

Leave one out, recompute, repeat $n$ times. Cheaper, older, works for smooth
statistics, and its influence values are useful for spotting the single example
that is dominating your metric.

## Regression as inference

Fitting $y = X\beta + \varepsilon$ gives more than predictions. Under the
classical assumptions, including full-column-rank design, zero conditional-mean errors, independence, homoscedasticity and Gaussian errors for exact small-sample t inference
— the coefficient estimates have standard errors, so you can test them.

$$\mathrm{Var}(\hat\beta) = \sigma^2 (X^\top X)^{-1}, \qquad t_j = \frac{\hat\beta_j}{\mathrm{SE}(\hat\beta_j)}$$

| Diagnostic | What it catches |
|---|---|
| Residuals vs fitted | non-linearity, heteroscedasticity (funnel shape) |
| Q-Q plot of residuals | non-normal errors, heavy tails |
| Variance inflation factor $> 5$–$10$ | multicollinearity; coefficients unstable |
| Cook's distance | single points dominating the fit |
| Durbin–Watson | autocorrelated residuals (time series) |
| $R^2$ vs adjusted $R^2$ | training $R^2$ cannot decrease in nested OLS with an intercept; adjusted $R^2$ can increase or decrease |

**Multicollinearity** is the one that bites in practice. If two features are
nearly collinear, $X^\top X$ is near-singular, its inverse blows up, and
coefficient estimates become enormous with opposite signs and huge standard
errors. Predictions can still be fine — it is *interpretation* that breaks. This
is a precise reason not to read feature importances off a linear model with
correlated inputs.

## Causal inference, briefly

Correlation supports prediction; causation supports intervention. The difference
matters the moment anyone acts on your model.

| Confound | Example | Fix |
|---|---|---|
| Common cause | ice cream and drowning (both caused by heat) | control for the confounder |
| Selection bias | survey only of surviving customers | model the selection process |
| Survivorship bias | reinforce the bullet holes on returning planes | ask what is missing from the sample |
| Collider bias | conditioning on a common effect | do **not** control for colliders |
| Reverse causation | "hospitals cause death" | temporal ordering, design |

Tools with distinct identification assumptions, not a universal strength ranking:

1. **Randomised controlled trial** — randomisation destroys confounding by
   construction. The gold standard; an A/B test is one.
2. **Difference-in-differences** — compare the change over time in treated vs
   untreated groups; needs the parallel-trends assumption.
3. **Instrumental variables** — find a variable affecting treatment but not the
   outcome except through treatment.
4. **Regression discontinuity** — exploit a sharp cutoff in treatment assignment.
5. **Propensity score matching / weighting** — match on estimated probability of
   treatment. Only handles *observed* confounders.

The **backdoor criterion** on a causal DAG tells you which variables to
condition on. The important negative result: adding more controls is not safer.
Conditioning on a collider *creates* bias where none existed.

## Worked inference and identification

### Deriving and evaluating an interval

For iid $X_i\sim N(\mu,\sigma^2)$ with known $\sigma$,
$Z=(\bar X-\mu)/(\sigma/\sqrt n)\sim N(0,1)$. Start with
$P(-1.96\le Z\le1.96)\approx.95$ and rearrange inequalities to obtain
the random interval around $\bar X$. The probability describes repeated samples,
not uncertainty assigned to a fixed parameter after observing this interval.
With $\bar x=12$, $\sigma=3$, $n=36$, the endpoints are $11.02,12.98$.

For paired differences $d=(1,2,0,1,1)$, the sample mean is one and
$s_d^2=.5$. Assuming iid Gaussian differences, the standard error is
$\sqrt{.5/5}=.3162$ and the $t_4$ statistic is $\sqrt{10}$.
The two-sided p-value is approximately $.0341$; the 95% interval is
$1\pm2.776(.3162)=[.122,1.878]$. The assumptions concern the differences,
not two independently sampled arms.

For independent unequal-variance groups, Welch uses
$SE^2=s_A^2/n_A+s_B^2/n_B$ and degrees of freedom

$$
\nu=\frac{(s_A^2/n_A+s_B^2/n_B)^2}
{(s_A^2/n_A)^2/(n_A-1)+(s_B^2/n_B)^2/(n_B-1)}.
$$

It is an approximation, not the exact paired procedure. Allocating observations
unequally changes the standard error: at fixed total cost and equal per-unit
cost, variance-optimal allocation for means satisfies $n_A/n_B=\sigma_A/\sigma_B$.
Equal allocation is optimal when the variances match.

### Paired accuracy and power need disagreements

Let $D_i=1$ when A alone is correct, $-1$ when B alone is correct, and zero
otherwise. Then the effect is $\delta=E[D]$ and
$\operatorname{Var}(D)=q-\delta^2$, where $q=P(D\ne0)$.
The standard error is approximately $\sqrt{(q-\delta^2)/n}$.
For a small effect, a planning approximation is
$n\approx(z_{.975}+z_{.8})^2q/\delta^2$.
At $\delta=.012$, $q=.10$ gives about 5,450 observations while $q=.30$
gives about 16,350. Marginal accuracies do not determine $q$.
Specify a practically useful effect and consider exact/simulation-based power
when discordance is rare.

### Resampling the actual sampling unit

Percentile intervals take quantiles of bootstrap estimates. BCa adjusts for
bias and acceleration, while studentized intervals bootstrap a standardized
statistic and require a usable standard-error estimate within resamples.
None repairs leakage or unidentified sampling. Resample users for clustered
data, temporal blocks for dependent sequences, and the same example indices
for both models' nonlinear metrics.

```python runnable
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

rng = np.random.default_rng(51)
d = np.array([1., 2., 0., 1., 1.])
result = stats.ttest_1samp(d, 0.)
se = stats.sem(d)
interval = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=se)
assert .033 < result.pvalue < .035
assert interval[0] > 0
assert stats.binomtest(10, 10, .5).pvalue == 2/1024

# Known-sigma coverage: simulation checks implementation, not a theorem.
samples = rng.normal(2., 3., size=(5000, 36))
means = samples.mean(1)
coverage = np.mean(np.abs(means-2.) <= stats.norm.ppf(.975)*3/6)
assert .93 < coverage < .97

y = rng.binomial(1, .2, 500)
a = np.where(rng.random(500) < .12, 1-y, y)
b = np.where(rng.random(500) < .18, 1-y, y)
differences = []
for _ in range(1500):
    idx = rng.integers(0, len(y), len(y))
    # zero_division=0 explicitly defines a degenerate resample's F1.
    differences.append(f1_score(y[idx], a[idx], zero_division=0) -
                       f1_score(y[idx], b[idx], zero_division=0))
lo, hi = np.quantile(differences, [.025, .975])
assert np.isfinite([lo, hi]).all() and lo <= hi
print("paired t:", result.pvalue, interval)
print("normal interval coverage:", coverage, "paired F1 interval:", (lo, hi))
```

### A causal estimand and numerical adjustment

Define potential outcomes $Y(1),Y(0)$, treatment $A$, and baseline covariates
$Z$. ATE is $E[Y(1)-Y(0)]$; ATT conditions this difference on $A=1$.
Consistency connects observed $Y$ to $Y(A)$; exchangeability requires
$(Y(1),Y(0))\perp A\mid Z$; positivity requires both treatment choices
at covariate values relevant to the estimand.
For the DAG $Z\to A$, $Z\to Y$, $A\to Y$, adjusting for the pre-treatment
common cause gives

$$E[Y(a)]=\sum_z E[Y\mid A=a,Z=z]P(Z=z).$$

Suppose half the target population is low-risk and half high-risk.
Untreated outcome rates are $.1,.6$; treated rates are $.05,.4$.
The adjusted ATE is $(.05-.1)/2+(.4-.6)/2=-.125$.
If 80% of treated observations are high-risk but only 20% of controls are,
the raw treated rate is $.33$ and control rate $.20$: the naive difference
$.13$ reverses the effect. Equalizing covariate weights explains the reversal;
the causal conclusion still depends on the stated assumptions and does not
follow from reweighting alone. Avoid adjusting for mediators or colliders merely
because they predict outcomes.

### Missing data is an assumption about selection

Let $R=1$ mean an outcome is observed. MCAR means missingness is independent
of all relevant values; MAR permits dependence on observed variables but not
remaining missing values conditional on them; MNAR retains such dependence.
Under appropriate MAR/positivity and modeling assumptions, multiple imputation
or inverse-observation weighting may recover an estimand. Complete-case analysis
is not generally valid under MAR. MNAR requires explicit sensitivity assumptions,
such as how unobserved outcomes differ from observed outcomes at the same
covariates; the observed data alone generally cannot identify that difference.
Fit imputers within training folds and propagate uncertainty when doing inference.

## Statistics for evaluating ML systems — a practical checklist

- Report a confidence interval, not a point estimate, on every headline metric.
- Use a **paired** test when models share a test set. McNemar for accuracy,
  paired bootstrap for anything else.
- Fix seeds and report variance **across seeds** — for small models, seed
  variance often exceeds the improvement being claimed.
- Never tune on the test set. If you looked at it $k$ times, the error inflation depends on selection and dependence, not a universal $k$ multiplier.
- Check for distribution shift between train, validation, and test with a
  two-sample test on features or by training a classifier to distinguish the
  splits (an AUC well above 0.5 means they differ).
- Slice metrics by segment. An aggregate number can hide a subgroup regression,
  which is both a quality problem and a fairness problem.
- Prefer a nested CV or a held-out test set that is touched exactly once for the
  final number.

## Self-check

1. Explain why $s^2$ divides by $n-1$ without using the word "unbiased", then
   with it.
2. You get $p = 0.03$. Write down three statements this does *not* license.
3. Model A: 84.0% on 500 test examples. Model B: 85.2%. Which test do you run,
   and roughly how many examples would you need for a 1.2-point difference to be
   detectable?
4. Derive the bias–variance decomposition, then explain what double descent
   contradicts and what it does not.
5. Your A/B test is significant after 2 days. Give three reasons not to ship yet.
6. When is the bootstrap invalid? Name two cases and the alternatives.
7. A feature has a large, significant coefficient in a linear model. Give two
   reasons this may not mean the feature matters.

## Worked self-check answers

1. Deviations around the fitted sample mean sum to zero, leaving $n-1$ free
   directions. For iid finite-variance observations,
   $E[\sum(X_i-\bar X)^2]=(n-1)\sigma^2$, so division by $n-1$ is unbiased.
2. $p=.03$ does not mean the null has probability .03, that the effect is large,
   or that it will replicate. It is a tail probability under the null and design.
3. Use the paired correctness table and an exact McNemar test if discordances
   are few. Sample size is not identifiable from .840 and .852 alone:
   the preceding $q$-dependent calculation shows why.
4. Add and subtract the mean fitted predictor, expand the square, and use
   zero-mean independent test noise to cancel cross terms. Double descent
   contradicts a universal monotone variance-capacity story, not that algebra.
5. A fixed-horizon test may have been peeked at; two days may miss calendar or
   delayed-outcome effects; guardrails and practical effect size can still fail.
6. An iid bootstrap fails for clustered rows and can fail for a sample maximum.
   Use cluster/block designs for dependence; extreme-value modeling or justified
   subsampling may be needed for extrema.
7. Confounding can create a noncausal association, and multicollinearity makes
   coefficients conditional and unstable. Statistical significance is neither
   causal identification nor practical importance.

## Where to go next

- [Probability](./probability.md) — the distributions these estimators assume.
- [Machine Learning notes](../ml.md) — metrics and validation protocols built
  on this.
- [Optimization Techniques](./optimization.md) — fitting the estimators.
