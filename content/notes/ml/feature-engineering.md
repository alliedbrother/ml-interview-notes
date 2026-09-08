---
order: 9
description: Numeric transforms, categorical encoding, dates and cyclical features, text and interaction features, aggregations and time-safe windows, feature selection, and the leakage rules that govern all of it.
meta: Machine Learning · practice
---

# Feature Engineering: Where the Accuracy Actually Comes From

Feature engineering expresses information in a form a particular model can use.
Its value depends on the task, sample size, and representation; there is no
universal ranking of engineered trees and learned neural representations. A useful
feature captures a stable relationship, is available at decision time, and can be
reproduced by the serving system.

The one rule that governs everything below: **a feature must be computable, with
that value, at the moment the prediction is made.** Every leakage disaster is a
violation of availability, independence, or evaluation isolation. Even features
available at prediction time can leak evaluation labels if their fitted
transformation used the holdout. Keep both the timeline and the fitting boundary
explicit.

## Numeric transforms

### Scaling

| Transform | Formula | Use when |
|---|---|---|
| Standardisation | $(x-\mu)/\sigma$ | roughly symmetric; the default |
| Min–max | $(x-\min)/(\max-\min)$ | bounded inputs, neural nets, images |
| Robust | $(x-\text{median})/\text{IQR}$ | heavy outliers |
| Max-abs | $x/\max\lvert x\rvert$ | **sparse data** — preserves zeros |
| Unit norm (per row) | $x/\lVert x\rVert$ | text vectors, cosine similarity |

Axis-aligned exact trees generally do not require scaling. Distance methods,
regularized linear models, PCA, and neural networks are sensitive to units, so
scaling is often useful, not a universal requirement. For example, physically
meaningful coordinate units or deliberately chosen distance weights may already
be correct. Scaling count noise to unit variance can amplify an unhelpful feature.

For $z_j=(x_j-\mu_j)/s_j$ and a linear score $\beta^Tz+b$, the original-coordinate
coefficient is $\beta_j/s_j$. An L2 penalty on $\beta$ becomes
$\sum_j s_j^2 w_j^2$ in original coordinates. Standardization therefore changes
the regularizer, not just the speed of optimization. A coefficient of two on a
standardized feature measures a one-training-standard-deviation change, not a
one-unit change in the original measurement.

Mean centering usually destroys sparsity: a zero count becomes a nonzero negative
mean. For a million-row vocabulary matrix this can be catastrophic. Use sparse
preserving transforms such as `StandardScaler(with_mean=False)` or row
normalization when their geometry is appropriate. Row normalization intentionally
discards magnitude; document whether document length or transaction volume should
be supplied separately.

### Distribution shaping

| Transform | For | Note |
|---|---|---|
| $\log(1+x)$ | right-skewed positive values | `log1p` handles zeros |
| Square root | mild skew, counts | gentler than log |
| Box–Cox | positive values | learns the exponent; requires $x>0$ |
| **Yeo–Johnson** | any real values | Box–Cox generalised to zero and negatives |
| Quantile transform | anything | maps to uniform or normal; non-linear and very effective |
| Rank transform | ordered values | reduces magnitude sensitivity; ties, sampling, and distribution shift still matter |

Skewness alone does not violate linear regression assumptions: a linear
conditional mean can be correct with skewed predictors. Transform because the
relationship, noise scale, leverage, or distance geometry calls for it. An exact
tree's candidate training partitions are invariant to a strictly increasing
transform, but histogram approximation, rounding, ties, and midpoint thresholds
between observations can change details and out-of-sample predictions.

For a multiplicative relationship $y=ax^b$, logging positive values yields
$\log y=\log a+b\log x$. For an additive relationship $y=a+bx$, logging $x$
instead changes the model class without justification. If the target is logged,
$\exp(\widehat{\mathbb E[\log Y\mid X]})$ is generally not
$\mathbb E[Y\mid X]$; retransformation requires assumptions or a residual-based
correction. Quantile maps also saturate beyond observed training extremes, so
they can erase extrapolation information that matters in forecasting.

### Binning and outliers

Binning (`KBinsDiscretizer`) gives linear models non-linearity and can help with
noisy measurements. Prefer quantile bins over uniform ones for skewed data.
Trees do their own binning, so this is mostly a linear-model tool.

For outliers, distinguish data errors from rare valid events. Clipping at fitted
quantiles is one option, not a default law: it can erase exactly the fraud or
equipment-failure signal the task needs. Compare robust losses, transforms,
clipping with an exceedance indicator, and leaving values intact. Learn thresholds
on training folds and monitor how often production values exceed them.

### Missing values

```python
SimpleImputer(strategy="median", add_indicator=True)
```

Consider a missingness indicator when the collection process is informative.
It can distinguish a genuine median value from an imputed median, but may also
encode a temporary operational artifact that disappears after a form redesign.
Evaluate it rather than assuming it always helps. By default an imputer's
indicator covers columns missing during fitting; a newly missing production
column needs an explicit schema and missingness policy.

| Mechanism | Meaning | Consequence |
|---|---|---|
| MCAR | missingness independent of observed and unobserved data | complete cases may be representative; arbitrary imputation is not unbiased |
| MAR | missingness conditionally independent of missing values given observed data | appropriate models can support inference under further assumptions |
| MNAR | dependence remains on unobserved values | needs assumptions or sensitivity analysis; an indicator does not identify missing values |

| Method | Note |
|---|---|
| Mean/median | fast; median for skew |
| Most frequent | categoricals |
| Constant sentinel | a distinct "missing" category for trees |
| $k$-NN imputer | uses similar rows; slow, and leaks if fit on all data |
| Iterative (MICE) | models each feature from the others; strong, expensive |
| Native handling | LightGBM/XGBoost/CatBoost/HistGB learn a default direction — **often the best option** |

Some boosters learn missing-value routing at each split. Compare that with
imputation plus indicators; there is no blanket strict-expressivity ordering over
all model configurations. Single mean imputation shrinks marginal variance even
under MCAR: replacing half of a zero-mean variable by zero roughly halves its
second moment. Predictive imputation and valid uncertainty about missing data are
different goals. Multiple imputation is about propagating uncertainty, not merely
calling one iterative imputer once.

## Categorical encoding

| Encoding | Cardinality | Notes |
|---|---|---|
| **One-hot** | low to moderately high | sparse width can be practical well beyond 15 levels |
| Ordinal / label | genuinely ordered values | arbitrary codes impose ordering even on numerical tree splits |
| **Target / mean encoding** | high | powerful, leaks unless cross-fitted |
| Frequency / count | high | often a surprisingly strong single feature |
| Binary / base-N | high | compact, but the bit positions are arbitrary |
| **Hashing** | very high, unbounded | fixed width, no fit, collisions |
| Native categorical | medium–high | estimator-specific categorical split or encoding algorithms; not arbitrary numerical codes |
| Learned embeddings | very high | needs a neural model; transfers across tasks |
| WoE (weight of evidence) | any | standard in credit scoring; log-odds per level |

### Target encoding, done correctly

Replace each level with a smoothed mean of the target:

$$\text{enc}(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}$$

$m$ is the smoothing strength: rare levels are pulled toward the global mean,
common levels keep their own.

**Without cross-fitting this leaks catastrophically.** A level appearing once
gets its own target as its encoding, and the model learns to read the label off
the feature. The fixes, in order of preference:

1. `sklearn.preprocessing.TargetEncoder.fit_transform` cross-fits training rows;
   `fit(...).transform(...)` does not produce the same out-of-fold representation.
2. CatBoost's ordered target statistics — uses only preceding rows in a random
   permutation.
3. Manual out-of-fold encoding, encoding each fold using only the other folds.

Add noise or increase $m$ for very high cardinality. And never compute the
encoding before the train/test split.

Suppose the overall positive rate is $0.2$, category $c$ appears twice with one
positive label, and $m=8$. Its smoothed value is $(1+8\cdot0.2)/(2+8)=0.26$.
For a singleton positive category with $m=0$, in-sample encoding is exactly one:
the feature reveals that observation's label. Smoothing reduces this dependence
but does not remove it. Cross-fitting encodes a training row without its own fold's
labels; after training, held-out rows use statistics fitted on the full training
set. Never recompute those statistics using held-out outcomes.

Group and time constraints apply to the encoder's inner folds too. A random
permutation is not automatically a valid chronological simulation, and default
random cross-fitting can leak related entities across an intended group boundary.
Use explicitly appropriate folds or past-only statistics when required. The
[TargetEncoder documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
explains the distinction between training cross-fitting and subsequent transform.

### Handling unseen categories

A category unseen at training time will appear in production. Decide in advance:

```python
OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
```

This folds rare levels into an `infrequent` bucket at fit time, which both
controls dimensionality. An unseen value maps there only if an infrequent bucket
was actually created during fitting; otherwise its behavior is like `ignore`,
an all-zero encoding. Missing, unseen, and infrequent are different states and
should be monitored separately. With a dropped reference category, an all-zero
unknown may be indistinguishable from that reference in the model representation.

### High-cardinality strategies

For an ID-like column with 100k levels:

- **Frequency encoding** — how common is this ID? Often more useful than
  identity.
- **Target encoding** with strong smoothing.
- **Aggregate features** — statistics of the target or other features for that
  ID, computed out-of-fold or over a prior time window.
- **Embeddings** — if you have a neural model and enough data.
- **Hashing** — when the space is unbounded (URLs, user agents).
- **Drop it** when opaque unique identity has no reusable signal. Structured IDs
  may contain legitimate location or product information, but sequential IDs can
  also reveal collection time or a label-generation artifact.

With ordinal codes `red=0, green=1, blue=2`, a numerical tree threshold can group
red alone or blue alone, but cannot group red and blue against green in one split.
Relabeling the same categories changes the available partitions. Native categorical
support must actually be enabled according to the estimator's API.

### Runnable lab: a deployable mixed-column pipeline

All fitted preprocessing belongs to the estimator passed to cross-validation.
This lab includes missing numeric and categorical values plus a holdout-only
category. It checks feature-name alignment and deterministic repeated prediction;
it does not require an external dataset or serialize files.

```python runnable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(12)
n = 400
age = rng.normal(40, 11, n)
income = rng.lognormal(3.8, 0.4, n)
region = rng.choice(["north", "south", "west"], n).astype(object)
score = 0.10 * (age - 40) + 0.025 * (income - 45) + (region == "north")
y = (score + rng.normal(0, 0.5, n) > 0).astype(int)
age[rng.random(n) < 0.08] = np.nan
region[rng.random(n) < 0.05] = np.nan
X = pd.DataFrame({"age": age, "income": income, "region": region})
train, test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=4)
test = test.copy()
test.loc[test.index[:3], "region"] = "unseen"
numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
preprocess = ColumnTransformer([
    ("numeric", numeric, ["age", "income"]),
    ("categorical", categorical, ["region"])])
model = Pipeline([("features", preprocess),
                  ("classifier", LogisticRegression(max_iter=500, random_state=4))])
cv_auc = cross_val_score(model, train, y_train, cv=3, scoring="roc_auc", n_jobs=1)
model.fit(train, y_train)
probability = model.predict_proba(test)[:, 1]
features = model.named_steps["features"]
names = features.get_feature_names_out()
assert len(names) == features.transform(test).shape[1]
assert np.isfinite(probability).all()
assert np.allclose(probability, model.predict_proba(test.copy())[:, 1])
assert roc_auc_score(y_test, probability) > 0.80
encoder = features.named_transformers_["categorical"].named_steps["encode"]
assert "unseen" not in encoder.categories_[0]
print("CV AUC", cv_auc, "holdout AUC", roc_auc_score(y_test, probability))
print("feature columns", names.tolist())
```

## Dates and times

A timestamp can encode trend, elapsed duration, seasonality, and collection
artifacts. Choose a representation based on the prediction horizon. The following
fragment assumes parsed datetimes and rows sorted within user before `diff`:

```python
d = df["ts"].dt
df["hour"], df["dow"], df["dom"] = d.hour, d.dayofweek, d.day
df["month"], df["quarter"], df["woy"] = d.month, d.quarter, d.isocalendar().week
df["is_weekend"] = d.dayofweek >= 5
df["is_month_end"] = d.is_month_end
df["days_since_signup"] = (df["ts"] - df["signup"]).dt.days
df["hours_since_last_event"] = df.groupby("user")["ts"].diff().dt.total_seconds() / 3600
```

**Cyclical encoding** so that hour 23 and hour 0 are adjacent:

$$x_{\sin} = \sin\left(\frac{2\pi x}{P}\right), \qquad x_{\cos} = \cos\left(\frac{2\pi x}{P}\right)$$

Without this, a linear model believes 23:00 and 00:00 are maximally distant.
Trees can recover the wrap with enough splits, but the encoding still helps.

Also worth engineering: holiday flags (country-specific), business-day counts,
time since and time until known events, and local time rather than UTC when human
behaviour is the signal.

The sine/cosine pair lies on a unit circle. The squared distance between adjacent
hours is $2-2\cos(2\pi/24)$, including the midnight transition; raw integer hours
instead give squared distance $23^2$ between 23 and zero. Both coordinates are
needed because a single sine identifies multiple hours. A linear model using the
pair fits one sinusoidal pattern; it cannot represent arbitrary hourly effects
without higher harmonics or categorical hour indicators.

Store a timezone-aware event instant and derive local calendar fields using the
appropriate location. Daylight-saving changes can produce repeated or nonexistent
local times. A seven-day duration and seven local calendar days are not always
the same window. "Time until renewal" is valid only if the renewal date was known
then, not if it is a retrospectively finalized date.

## Aggregation features

The single richest source of signal in transactional data: summarise a group's
history into features for the current row.

```python
g = df.groupby("user_id")
df["user_txn_count"]   = g["amount"].transform("count")
df["user_amount_mean"] = g["amount"].transform("mean")
df["user_amount_std"]  = g["amount"].transform("std")
df["amount_vs_user_mean"] = df["amount"] / (df["user_amount_mean"] + 1e-9)
df["amount_zscore_in_user"] = (df["amount"] - df["user_amount_mean"]) / (df["user_amount_std"] + 1e-9)
```

That last pair — a value relative to its group's typical value — is often far
more predictive than either the raw value or the group mean alone. "£500 is
normal for this user" and "£500 is 8 standard deviations above normal for this
user" are entirely different facts.

For an online decision, future transactions cannot contribute. Including the
current transaction is not inherently leakage if its amount is already known,
but it changes a historical-baseline feature into a current-inclusive summary.
State which meaning is intended. A standard deviation estimated from one prior
observation is undefined, not zero evidence of perfect stability.

### Time-safe aggregation

For expanding row-based summaries, sorting and shifting can exclude the current
row. For duration windows, keep values attached to their original timestamps and
use a datetime index with `closed="left"`. Shifting values first moves them onto
the next event's timestamp, which gives the wrong window on irregular data.
`rolling("7D")` on an ordinary integer index is also invalid.

### Runnable lab: irregular-time windows and a future-invariance test

The window below is $[t-7\text{ days},t)$ and the fixture has distinct timestamps
within each user. The January 3 amount belongs in January 10's window, but the
January 1 amount does not. Adding a later event must not alter any existing feature.

```python runnable
import numpy as np
import pandas as pd

events = pd.DataFrame({
    "user_id": ["a", "a", "a", "a", "b", "b"],
    "ts": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-10",
                          "2025-01-11", "2025-01-02", "2025-01-06"], utc=True),
    "amount": [10.0, 20.0, 30.0, 40.0, 5.0, 15.0]})

def past_features(frame):
    result = pd.DataFrame(index=frame.index, columns=["mean7", "past_mean"], dtype=float)
    for _, group in frame.groupby("user_id", sort=False):
        ordered = group.sort_values("ts")
        assert ordered["ts"].is_unique, "Define a simultaneous-event policy first"
        series = ordered.set_index("ts")["amount"]
        result.loc[ordered.index, "mean7"] = series.rolling(
            "7D", closed="left", min_periods=1).mean().to_numpy()
        result.loc[ordered.index, "past_mean"] = series.shift(1).expanding().mean().to_numpy()
    return result

before = past_features(events)
assert np.allclose(before["mean7"], [np.nan, 10, 20, 30, np.nan, 5], equal_nan=True)
later = pd.DataFrame({"user_id": ["a"],
    "ts": pd.to_datetime(["2025-01-20"], utc=True), "amount": [999.0]})
extended = pd.concat([events, later], ignore_index=True)
after = past_features(extended).iloc[:len(events)]
assert np.allclose(before, after, equal_nan=True)

snapshots = pd.DataFrame({"user_id": ["a", "a", "b"],
    "available_at": pd.to_datetime(["2024-12-31", "2025-01-10", "2025-01-01"], utc=True),
    "known_count": [2, 9, 1]})
joined = pd.merge_asof(events.sort_values("ts"), snapshots.sort_values("available_at"),
    left_on="ts", right_on="available_at", by="user_id", direction="backward",
    allow_exact_matches=False)
assert (joined["available_at"] < joined["ts"]).all()
assert joined.loc[(joined.user_id == "a") &
                  (joined.ts == pd.Timestamp("2025-01-10", tz="UTC")), "known_count"].item() == 2
print(events.join(before).to_string(index=False))
```

For tied timestamps, define whether simultaneous events are mutually visible.
If none are visible, aggregate by timestamp first and compute strict-past sums
and counts, or join precomputed snapshots strictly before the event instant. Row
order is not a defensible causal ordering unless a real sequence number establishes
it. The [pandas rolling reference](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html)
specifies duration windows and endpoint inclusion.

For joining separate feature tables, the basic primitive is:

```python
out = pd.merge_asof(events.sort_values("ts"), features.sort_values("ts"),
                    on="ts", by="user_id", direction="backward")
```

This takes the latest matching row at or before the join timestamp, but correctness
depends on what that timestamp means. Join on availability time when a measurement
arrives after its event time. Use `allow_exact_matches=False` for strict-past
semantics and `tolerance` when stale snapshots should count as unavailable.
Both tables must be sorted by the join key; grouping by user does not remove that
requirement. Late-arriving corrections need historical versions so an offline
replay does not accidentally see a revised value that serving never saw.

## Interactions

Trees find interactions automatically. Linear models do not, and explicit
interactions are often what closes the gap.

```python
df["price_per_sqft"] = df["price"] / df["sqft"]
df["debt_to_income"] = df["debt"] / df["income"]
df["clicks_per_impression"] = df["clicks"] / (df["impressions"] + 1)
```

Ratios often encode useful domain structure. A linear model on $a,b$ can represent
$a-b$ but not $a/b$ over a general domain. Trees and neural networks can approximate
ratios, though explicit construction may reduce sample requirements. Define zero,
negative, and missing denominator behavior. Adding $10^{-9}$ prevents a division
exception but can create a meaningless billion-scale value; it is not a semantic
policy.

`PolynomialFeatures(degree=2, interaction_only=True)` generates all pairwise
products. There are $\binom{d}{2}$ new pairwise products, but the default output
also retains $d$ original columns and a bias, totaling $1+d+\binom{d}{2}$.
For 50 inputs that is 1,276 columns. Use a small deliberate subset.

A click-through rate from one click in one impression is less reliable than the
same rate from 1,000 clicks in 1,000 impressions. Keep the denominator, or use
prior smoothing $(c+\alpha)/(n+\alpha+\beta)$. The denominator-plus-one fragment
above is a particular shrinkage choice, not an unbiased estimator. Fit any
data-derived prior on training information only. Interactions also require
compatible units: amount divided by elapsed days is a rate, while amount divided
by an arbitrary categorical code has no stable interpretation.

## Text features

| Method | Produces | Note |
|---|---|---|
| Bag of words | sparse counts | order-free |
| **TF-IDF** | sparse weighted counts | down-weights common terms; still a strong baseline |
| Character n-grams | sparse | robust to typos and morphology; good for names and codes |
| Hashing vectoriser | fixed-width sparse | no vocabulary fit; streaming-friendly |
| Word embeddings averaged | dense, ~300-d | cheap, loses order |
| Sentence transformers | dense, ~384–1024-d | strong general-purpose semantic features |
| Fine-tuned encoder | task-specific | potentially strong; needs suitable data, evaluation, and compute |
| Hand-crafted | length, punctuation, caps ratio, URL count, readability | cheap and surprisingly effective for spam and quality tasks |

TF-IDF plus a linear model is still a competitive baseline for topical
classification and trains in seconds. Reach for embeddings when meaning matters
more than vocabulary — paraphrase detection, semantic search, small labelled
sets.

## Feature selection

| Family | Method | Cost | Note |
|---|---|---|---|
| **Filter** | variance threshold, correlation with target, mutual information, chi-square | cheap | model-agnostic; ignores interactions |
| **Wrapper** | forward/backward selection, RFE, RFECV | expensive | model-specific, leakage-prone if done outside CV |
| **Embedded** | L1, tree importances, `SelectFromModel` | free with training | the usual practical choice |
| **Permutation** | shuffle and measure degradation on held-out data | moderate | honest; misleads under correlation |

**Do you need it at all?** Modern regularised models and boosters handle many
irrelevant features well. Select when you need faster inference, lower data
collection cost, interpretability, or when $d \gg n$. Do not select just because
you have a lot of columns.

**Selection must happen inside the cross-validation fold.** Choosing features by
their correlation with the target on the full dataset is leakage, and it can
produce impressive scores on pure noise — the classic demonstration is selecting
the 10 best of 10,000 random features on 100 samples and obtaining excellent
cross-validated accuracy.

Correlated features make importance dependent on the question and procedure. A
tree may assign nearly all split importance to one of two substitutes, whereas
individual permutation importance can be small because the other remains. Grouped
permutation asks about their joint predictive information. None of these values
automatically identifies causal effects.

Selection is part of model fitting even when described as "exploration." If
5,000 independent noise columns are screened against labels, the largest observed
correlation is an extreme selected fluctuation. Cross-validating only the already
selected columns reuses that fluctuation in every fold. Put `SelectKBest`, RFE,
or the entire feature-building search inside the training procedure evaluated by
the outer split. If the research team repeatedly inspects the same outer results,
that outer set eventually becomes development data too.

An operational selection objective may include acquisition latency, sensor cost,
missingness, or privacy exposure, not just column count. Removing 99 cheap columns
while retaining one expensive remote lookup may not improve serving latency.
Measure performance after ablation, including worst-group and shifted-cohort
behavior, rather than treating a feature-importance ranking as a deletion order.

## Domain patterns worth stealing

| Domain | Features that usually work |
|---|---|
| **Fraud** | velocity (count in the last hour/day/week), deviation from the user's own history, device/IP/card entity linkage, time-of-day anomaly, first-time-seen flags |
| **Churn** | recency, frequency, monetary value (RFM), trend of activity (this month vs last), support-ticket counts, plan changes, engagement decay |
| **Credit** | debt-to-income, utilisation, delinquency history, credit-history length, inquiry counts, WoE-encoded categoricals |
| **E-commerce** | session depth, cart value, time on page, price relative to category median, brand affinity, return history |
| **Recommendation** | user–item interaction counts, item popularity, time since last interaction, category affinity, collaborative-filtering embeddings |
| **Time series** | lags (1, 7, 28), rolling means and stds, differences, seasonal decomposition, calendar and holiday flags, exogenous regressors |
| **Sensors / IoT** | rolling statistics, FFT band energies, peak counts, rate of change, cross-sensor correlations |

RFM is a useful customer-history baseline, but no fixed fraction of achievable
performance applies across tasks. Define an observation cutoff, lookback window,
and future outcome horizon separately. For churn, a "days since last activity"
feature measured after the churn-label window is a reformulation of the answer.

## Automated feature engineering

| Tool | Approach |
|---|---|
| Featuretools | deep feature synthesis across related tables |
| tsfresh | hundreds of time-series features with significance filtering |
| AutoFeat / OpenFE | automated non-linear feature construction |
| Deep learning | learns representations directly — the right answer for images, text, audio |

These are useful for generating candidates, but they produce many features with
no domain meaning, they are computationally heavy, and they make leakage easier
to introduce and harder to spot. Treat them as a source of hypotheses, not as a
replacement for understanding the data.

## Leakage rules

1. **Timeline test.** For every feature, ask: at prediction time, would this
   value exist, and would it have *this* value? If not, it leaks.
2. **Fit inside the fold.** Scalers, imputers, encoders, PCA, feature selection,
   resampling — everything with a `fit` belongs in the Pipeline.
3. **Aggregations use the past only.** `shift`, expanding/rolling windows,
   `merge_asof`.
4. **Target encoding must be cross-fitted.**
5. **Define deduplication and grouping before assigning splits**, using a fixed
   label-independent policy where possible. Keep related records together; then
   fit data-dependent transforms and tune cleaning thresholds on training folds.
6. **Audit any suspiciously strong feature.** Overwhelming importance from one
   column is a leakage alarm, not a triumph.
7. **Check the feature's availability latency.** A feature computed by a nightly
   batch job is not available for a real-time decision at 09:00.

## Self-check

### Solved cases

**A mean-imputed variable was MCAR. Is its variance unbiased?** No. With true mean
zero and independent observation probability $q$, zero-imputation gives
$\mathbb E[X_{imputed}^2]=q\mathbb E[X^2]$. Missingness independence does not make
every downstream statistic valid. Predictive usefulness is a separate empirical
question.

**A category is absent from a cross-fitting training fold. What value should its
held-out row receive?** A fallback based only on that training fold, commonly its
global target mean. Its own label must not resolve the absence. After final fitting,
production unknowns use the full-training fallback, not a rolling average that
silently incorporates unavailable outcomes.

**Why does shifting before a seven-day rolling window fail?** An event on January
1 shifted to the January 10 row appears to occur nine days later. Window membership
then follows the wrong timestamp. Keep values on their real instants and exclude
the current instant with the window endpoint policy.

**What should happen when income is zero in debt-to-income?** There is no universal
numeric answer. A documented policy might separate zero income with an indicator,
compute a ratio only for positive income, and use a bounded representation. The
training and serving implementations must agree; silently replacing zero by a
tiny epsilon is not an economic interpretation.

**What tests belong beside a feature?** Check schema and units, missing/unseen
cases, exact window boundaries, simultaneous events, and invariance to appended
future rows. For learned transforms, verify training-only state. For serving,
replay historical requests and compare feature values before comparing model
scores. A correct mathematical formula with the wrong join key still fails.

1. Under what assumptions can `log1p` help a linear model while preserving an exact tree's training partitions?
2. Explain exactly how target encoding leaks and give two correct
   implementations.
3. Write the cyclical encoding for hour-of-day and say what it fixes.
4. Your fraud model has one feature with 90% of the importance. What do you do
   before shipping?
5. When is a missingness indicator informative, and when can it fail under deployment shift?
6. Give the pandas primitive for "the most recent value known at or before this
   row's timestamp" and say what it prevents.
7. You select the 20 best of 5,000 features by correlation, then cross-validate,
   and get 0.95 AUC on random data. Explain.

## Where to go next

- [Model Evaluation](./model-evaluation.md) — protocols that catch the leakage
  described here.
- [Imbalanced Data & Pitfalls](./imbalanced-data-and-pitfalls.md) — more ways to
  get a good number and a bad model.
- [Trees & Ensembles](./trees-and-ensembles.md) — the models that consume these
  features.
- [scikit-learn](../libraries/scikit-learn.md) — composite estimators, feature names, and persistence contracts.
- [pandas](../libraries/pandas.md) — datetime indexing, grouping, and join semantics.
- [MLOps & Serving](../libraries/mlops-and-serving.md) — production feature parity and monitoring.
