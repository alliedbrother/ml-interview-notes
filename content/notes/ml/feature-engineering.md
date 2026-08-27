---
order: 9
description: Numeric transforms, categorical encoding, dates and cyclical features, text and interaction features, aggregations and time-safe windows, feature selection, and the leakage rules that govern all of it.
meta: Machine Learning · practice
---

# Feature Engineering: Where the Accuracy Actually Comes From

On tabular problems, feature engineering routinely moves a metric more than model
choice does. A boosted tree on good features beats a neural network on raw
columns almost every time, because features are where you inject knowledge the
model has no way to discover from a few thousand rows.

The one rule that governs everything below: **a feature must be computable, with
that value, at the moment the prediction is made.** Every leakage disaster is a
violation of it.

## Numeric transforms

### Scaling

| Transform | Formula | Use when |
|---|---|---|
| Standardisation | $(x-\mu)/\sigma$ | roughly symmetric; the default |
| Min–max | $(x-\min)/(\max-\min)$ | bounded inputs, neural nets, images |
| Robust | $(x-\text{median})/\text{IQR}$ | heavy outliers |
| Max-abs | $x/\max\lvert x\rvert$ | **sparse data** — preserves zeros |
| Unit norm (per row) | $x/\lVert x\rVert$ | text vectors, cosine similarity |

Trees do not need any of these. Distance-based methods, regularised linear
models, PCA, and neural networks all do.

### Distribution shaping

| Transform | For | Note |
|---|---|---|
| $\log(1+x)$ | right-skewed positive values | `log1p` handles zeros |
| Square root | mild skew, counts | gentler than log |
| Box–Cox | positive values | learns the exponent; requires $x>0$ |
| **Yeo–Johnson** | any real values | Box–Cox generalised to zero and negatives |
| Quantile transform | anything | maps to uniform or normal; non-linear and very effective |
| Rank transform | anything | fully robust, discards magnitude |

Skewed features hurt linear models (a few extreme points dominate the fit) and
distance methods. They do not hurt trees, which only use ordering — so a
monotone transform is a no-op for a tree and can be a large win for a linear
model.

### Binning and outliers

Binning (`KBinsDiscretizer`) gives linear models non-linearity and can help with
noisy measurements. Prefer quantile bins over uniform ones for skewed data.
Trees do their own binning, so this is mostly a linear-model tool.

For outliers: **winsorise** (clip at the 1st and 99th percentiles) rather than
delete, unless you know a value is an error. Deleting rows changes the
distribution you are modelling; and remember to fit the clipping thresholds on
training data only.

### Missing values

```python
SimpleImputer(strategy="median", add_indicator=True)
```

**Always add the indicator.** Missingness is frequently predictive — a blank
income field may correlate strongly with the target — and imputing without an
indicator destroys that signal.

| Mechanism | Meaning | Consequence |
|---|---|---|
| MCAR | missing completely at random | any imputation is unbiased |
| MAR | missing depends on observed features | model-based imputation works |
| MNAR | missing depends on the unobserved value | imputation biases; the indicator is essential |

| Method | Note |
|---|---|
| Mean/median | fast; median for skew |
| Most frequent | categoricals |
| Constant sentinel | a distinct "missing" category for trees |
| $k$-NN imputer | uses similar rows; slow, and leaks if fit on all data |
| Iterative (MICE) | models each feature from the others; strong, expensive |
| Native handling | LightGBM/XGBoost/CatBoost/HistGB learn a default direction — **often the best option** |

Modern boosters handle `NaN` natively by learning which side of each split
missing values should go. That is strictly more expressive than imputing a
constant, and it is a good reason not to impute at all for tree models.

## Categorical encoding

| Encoding | Cardinality | Notes |
|---|---|---|
| **One-hot** | low (< ~15) | standard for linear models; explodes dimension |
| Ordinal / label | any | imposes false ordering; fine for trees |
| **Target / mean encoding** | high | powerful, leaks unless cross-fitted |
| Frequency / count | high | often a surprisingly strong single feature |
| Binary / base-N | high | compact, but the bit positions are arbitrary |
| **Hashing** | very high, unbounded | fixed width, no fit, collisions |
| Native categorical | medium–high | LightGBM/CatBoost partition levels optimally |
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

1. `sklearn.preprocessing.TargetEncoder` — cross-fits internally.
2. CatBoost's ordered target statistics — uses only preceding rows in a random
   permutation.
3. Manual out-of-fold encoding, encoding each fold using only the other folds.

Add noise or increase $m$ for very high cardinality. And never compute the
encoding before the train/test split.

### Handling unseen categories

A category unseen at training time will appear in production. Decide in advance:

```python
OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
```

This folds rare levels into an `infrequent` bucket at fit time, which both
controls dimensionality and gives unseen levels somewhere sensible to land.

### High-cardinality strategies

For an ID-like column with 100k levels:

- **Frequency encoding** — how common is this ID? Often more useful than
  identity.
- **Target encoding** with strong smoothing.
- **Aggregate features** — statistics of the target or other features for that
  ID, computed out-of-fold or over a prior time window.
- **Embeddings** — if you have a neural model and enough data.
- **Hashing** — when the space is unbounded (URLs, user agents).
- **Drop it** — an ID that appears once per row carries no generalisable signal
  and is a leakage risk.

## Dates and times

A raw timestamp is nearly useless; the information is in its decomposition.

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

**But all of these leak if computed over the whole history.** The row's own
transaction contributed to the mean it is being compared against, and future
transactions did too.

### Time-safe aggregation

```python
# expanding statistics using only the past, per user
g = df.sort_values("ts").groupby("user_id")["amount"]
df["past_mean"]  = g.transform(lambda s: s.shift(1).expanding().mean())
df["past_max"]   = g.transform(lambda s: s.shift(1).expanding().max())
df["roll7_mean"] = g.transform(lambda s: s.shift(1).rolling("7D").mean())
```

The `shift(1)` excludes the current row; the sort ensures "past" means past. For
joining features from a separate table at the right point in time, use an as-of
join:

```python
out = pd.merge_asof(events.sort_values("ts"), features.sort_values("ts"),
                    on="ts", by="user_id", direction="backward")
```

This takes, for each event, the most recent feature row **at or before** its
timestamp. It is the correct primitive for point-in-time correctness, and it is
what feature stores implement.

## Interactions

Trees find interactions automatically. Linear models do not, and explicit
interactions are often what closes the gap.

```python
df["price_per_sqft"] = df["price"] / df["sqft"]
df["debt_to_income"] = df["debt"] / df["income"]
df["clicks_per_impression"] = df["clicks"] / (df["impressions"] + 1)
```

**Ratios are the most valuable form.** A model can learn $a - b$ from $a$ and
$b$ easily; it cannot learn $a/b$ from them at all without the right functional
form. Domain ratios (utilisation, rates, densities, per-capita figures) usually
carry the actual meaning.

`PolynomialFeatures(degree=2, interaction_only=True)` generates all pairwise
products, but $d$ features become $\binom{d}{2}$ — 50 features become 1,225. Use
it only on a small, deliberately chosen subset.

## Text features

| Method | Produces | Note |
|---|---|---|
| Bag of words | sparse counts | order-free |
| **TF-IDF** | sparse weighted counts | down-weights common terms; still a strong baseline |
| Character n-grams | sparse | robust to typos and morphology; good for names and codes |
| Hashing vectoriser | fixed-width sparse | no vocabulary fit; streaming-friendly |
| Word embeddings averaged | dense, ~300-d | cheap, loses order |
| Sentence transformers | dense, ~384–1024-d | strong general-purpose semantic features |
| Fine-tuned encoder | task-specific | best, most expensive |
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

**Correlated features confuse every importance measure.** Two features carrying
the same information split their importance, so both look weak. Cluster features
by correlation and evaluate clusters rather than individual columns when this
matters.

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

**RFM is a remarkably strong baseline** wherever there is a customer and a
transaction log — recency, frequency, and monetary value alone often reach 80% of
achievable performance on churn and value prediction.

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
5. **Split before you do anything else**, including deduplication decisions and
   outlier removal.
6. **Audit any suspiciously strong feature.** Overwhelming importance from one
   column is a leakage alarm, not a triumph.
7. **Check the feature's availability latency.** A feature computed by a nightly
   batch job is not available for a real-time decision at 09:00.

## Self-check

1. Why does `log1p` help a linear model but do nothing for a decision tree?
2. Explain exactly how target encoding leaks and give two correct
   implementations.
3. Write the cyclical encoding for hour-of-day and say what it fixes.
4. Your fraud model has one feature with 90% of the importance. What do you do
   before shipping?
5. Why is `add_indicator=True` on an imputer usually the right choice?
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
