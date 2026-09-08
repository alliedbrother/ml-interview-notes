"""Offline grouped/temporal classification evaluation. See --help.

Download artifact_capstone.py beside this file. The default corpus is synthetic;
its scores test plumbing, not generalization. Reports contain labels and IDs.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import classification_report, log_loss

import artifact_capstone
from artifact_capstone import canonical_json, fixture, make_pipeline, sha256


PARTS = ("train", "calibration", "policy", "test")
REQUIRED = {"id", "group", "text", "label", "timestamp"}
DEFAULT_CUTOFFS = ("2026-01-07T00:00:00Z", "2026-01-09T00:00:00Z",
                   "2026-01-11T00:00:00Z")


def instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps require an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def controlled_fixture():
    rows = fixture()
    for row in rows:
        day = int(row["group"].rsplit("-", 1)[1])
        row["timestamp"] = (datetime(2026, 1, 1, tzinfo=timezone.utc)
                            + timedelta(days=day)).isoformat()
        # Deliberately ambiguous fixture cases, not measured real-world errors.
        if day >= 6 and row["id"].endswith("-1"):
            row["text"] = "please help with this request " + row["marker"]
    return rows


def validate_rows(rows):
    if not rows:
        raise ValueError("Dataset cannot be empty")
    ids, text_groups = set(), {}
    for row in rows:
        if not REQUIRED <= row.keys() or any(not isinstance(row[k], str) or not row[k].strip()
                                             for k in REQUIRED):
            raise ValueError("Every row requires nonblank id, group, text, label, timestamp")
        if row["id"] in ids:
            raise ValueError("Duplicate row ID")
        ids.add(row["id"])
        instant(row["timestamp"])
        normalized = " ".join(row["text"].casefold().split())
        if normalized in text_groups and text_groups[normalized] != row["group"]:
            raise ValueError("Duplicate normalized text spans groups; merge or audit groups")
        text_groups[normalized] = row["group"]


def load_csv(path, *, return_digest=False):
    data = Path(path).read_bytes()
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED <= set(reader.fieldnames):
            raise ValueError("CSV header requires id,group,text,label,timestamp")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV header contains duplicate columns")
        rows = list(reader)
    validate_rows(rows)
    return (rows, sha256(data)) if return_digest else rows


def temporal_split(rows, cutoffs=DEFAULT_CUTOFFS):
    """Half-open time windows; purge an entire group if it crosses a cutoff."""
    validate_rows(rows)
    boundaries = [instant(value) for value in cutoffs]
    if len(boundaries) != 3 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Require three strictly increasing cutoffs")
    group_windows = {}
    for row in rows:
        window = sum(instant(row["timestamp"]) >= boundary for boundary in boundaries)
        group_windows.setdefault(row["group"], set()).add(window)
    purged = sorted(group for group, windows in group_windows.items() if len(windows) > 1)
    splits = {part: [] for part in PARTS}
    for row in rows:
        windows = group_windows[row["group"]]
        if len(windows) == 1:
            splits[PARTS[next(iter(windows))]].append(row)
    if any(not subset for subset in splits.values()):
        raise ValueError("Each split needs retained rows after boundary-group purging")
    classes = {row["label"] for row in splits["train"]}
    if len(classes) < 2:
        raise ValueError("Training requires at least two classes")
    if any(row["label"] not in classes for part in PARTS[1:] for row in splits[part]):
        raise ValueError("Held-out label is absent from training; define an unknown-label policy")
    return splits, purged


def probabilities(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2
            or not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Require a nonempty finite row-normalized probability matrix")
    return values


def label_indices(labels, classes, n):
    classes = list(classes)
    if len(classes) != len(set(classes)) or len(labels) != n:
        raise ValueError("Unique classes and one label per prediction required")
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        return np.array([lookup[label] for label in labels])
    except KeyError as error:
        raise ValueError("Unknown evaluation label") from error


def temperature_scale(values, temperature):
    values = probabilities(values)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.log(np.clip(values, 1e-15, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(values, labels, classes):
    values = probabilities(values)
    if len(classes) != values.shape[1]:
        raise ValueError("Class-order width mismatch")
    targets = label_indices(labels, classes, len(values))

    def objective(log_temperature):
        calibrated = temperature_scale(values, float(np.exp(log_temperature)))
        return float(-np.log(np.clip(calibrated[np.arange(len(targets)), targets], 1e-15, 1)).mean())

    result = minimize_scalar(objective, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization did not converge")
    # Identity is an explicit candidate, so numerical optimization cannot worsen fit loss.
    candidates = [0.0, float(result.x), float(np.log(0.1)), float(np.log(10.0))]
    selected = min(candidates, key=lambda value: (objective(value), abs(value)))
    return float(np.exp(selected))


def policy_metrics(correct, accepted, *, error_cost=1.0, review_cost=0.2):
    correct, accepted = np.asarray(correct, bool), np.asarray(accepted, bool)
    if correct.ndim != 1 or not len(correct) or accepted.shape != correct.shape:
        raise ValueError("Require equal nonempty correctness/acceptance vectors")
    if any(not np.isfinite(cost) or cost < 0 for cost in (error_cost, review_cost)):
        raise ValueError("Costs must be finite and nonnegative")
    errors = int((accepted & ~correct).sum())
    count = int(accepted.sum())
    return {"count": len(correct), "accepted": count, "errors_accepted": errors,
            "coverage": count / len(correct), "risk": errors / count if count else None,
            "cost": (error_cost * errors + review_cost * (len(correct) - count)) / len(correct)}


def select_policy(values, labels, classes, *, error_cost=1.0, review_cost=0.2,
                  min_coverage=0.0):
    values = probabilities(values)
    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError("Minimum coverage must be between zero and one")
    if len(classes) != values.shape[1]:
        raise ValueError("Class-order width mismatch")
    targets = label_indices(labels, classes, len(values))
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == targets
    # A threshold just above one represents reject-all, including exact confidence 1.
    thresholds = sorted({0.0, float(np.nextafter(1.0, 2.0)), *confidence.tolist()})
    curve = [{"threshold": threshold,
              **policy_metrics(correct, confidence >= threshold,
                               error_cost=error_cost, review_cost=review_cost)}
             for threshold in thresholds]
    feasible = [row for row in curve if row["coverage"] >= min_coverage]
    selected = min(feasible, key=lambda row: (row["cost"], -row["coverage"], row["threshold"]))
    return selected, curve


def calibration_metrics(values, labels, classes, bins=10):
    values = probabilities(values)
    if len(classes) != values.shape[1] or not isinstance(bins, int) or bins < 1:
        raise ValueError("Invalid class-order width or bin count")
    targets = label_indices(labels, classes, len(values))
    confidence, predictions = values.max(axis=1), values.argmax(axis=1)
    membership = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = membership == index
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": int(mask.sum()),
                            "confidence": float(confidence[mask].mean()) if mask.any() else None,
                            "accuracy": float((predictions[mask] == targets[mask]).mean()) if mask.any() else None})
    ece = sum(row["count"] / len(values) * abs(row["accuracy"] - row["confidence"])
              for row in reliability if row["count"])
    one_hot = np.eye(values.shape[1])[targets]
    return {"log_loss": float(log_loss(targets, values, labels=np.arange(len(classes)))),
            "multiclass_brier_sum": float(((values - one_hot) ** 2).sum(axis=1).mean()),
            "top_label_ece": ece, "reliability": reliability}


def group_bootstrap(correct, accepted, groups, *, repetitions=1000, seed=41,
                    error_cost=1.0, review_cost=0.2):
    correct, accepted, groups = np.asarray(correct, bool), np.asarray(accepted, bool), np.asarray(groups)
    policy_metrics(correct, accepted, error_cost=error_cost, review_cost=review_cost)
    if groups.shape != correct.shape or not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("Require one group per row and at least two bootstrap repetitions")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Group uncertainty requires at least two independent groups")
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in ("coverage", "risk", "cost")}
    for _ in range(repetitions):
        chosen = np.concatenate([indices[index] for index in rng.integers(len(unique), size=len(unique))])
        stats = policy_metrics(correct[chosen], accepted[chosen], error_cost=error_cost, review_cost=review_cost)
        for metric in samples:
            if stats[metric] is not None:
                samples[metric].append(stats[metric])
    return {"method": "group percentile bootstrap; pooled-row estimand; fixed fitted policy",
            "groups": len(unique), "repetitions": repetitions, "seed": seed,
            "intervals": {metric: {"low": float(np.quantile(values, .025)) if values else None,
                                    "high": float(np.quantile(values, .975)) if values else None,
                                    "valid_replicates": len(values)}
                          for metric, values in samples.items()}}


def evaluate(rows, *, cutoffs=DEFAULT_CUTOFFS, error_cost=1.0, review_cost=0.2,
             min_coverage=0.0, repetitions=1000, provenance="controlled synthetic fixture"):
    splits, purged = temporal_split(rows, cutoffs)
    texts = lambda part: [row["text"] for row in splits[part]]
    labels = lambda part: [row["label"] for row in splits[part]]
    model = make_pipeline(2.0).fit(texts("train"), labels("train"))
    classes = model.classes_.tolist()
    temperature = fit_temperature(model.predict_proba(texts("calibration")), labels("calibration"), classes)
    selection_values = temperature_scale(model.predict_proba(texts("policy")), temperature)
    selected, curve = select_policy(selection_values, labels("policy"), classes,
                                    error_cost=error_cost, review_cost=review_cost,
                                    min_coverage=min_coverage)
    raw = model.predict_proba(texts("test"))
    calibrated = temperature_scale(raw, temperature)
    targets = label_indices(labels("test"), classes, len(raw))
    correct = calibrated.argmax(axis=1) == targets
    accepted = calibrated.max(axis=1) >= selected["threshold"]
    test = policy_metrics(correct, accepted, error_cost=error_cost, review_cost=review_cost)
    test["raw_calibration"] = calibration_metrics(raw, labels("test"), classes)
    test["scaled_calibration"] = calibration_metrics(calibrated, labels("test"), classes)
    test["closed_set_classification"] = classification_report(
        labels("test"), model.classes_[calibrated.argmax(axis=1)], labels=classes,
        output_dict=True, zero_division=0)
    test["bootstrap"] = group_bootstrap(correct, accepted, [row["group"] for row in splits["test"]],
                                         repetitions=repetitions, error_cost=error_cost, review_cost=review_cost)
    return {"format_version": 1, "provenance": provenance,
            "dataset_sha256": sha256(canonical_json(rows)), "source_sha256": sha256(Path(__file__).read_bytes()),
            "pipeline_source_sha256": sha256(Path(artifact_capstone.__file__).read_bytes()),
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
            "classes": classes, "cutoffs_utc": [instant(value).isoformat() for value in cutoffs],
            "purged_groups": purged,
            "splits": {part: [{key: row[key] for key in ("id", "group", "timestamp", "label")} for row in subset]
                       for part, subset in splits.items()},
            "fixed_C": 2.0, "temperature": temperature, "temperature_bounds": [0.1, 10.0],
            "costs": {"error": error_cost, "review": review_cost}, "min_coverage": min_coverage,
            "selected_on": "policy", "selected": selected, "policy_curve": curve, "test": test,
            "test_predictions": [{"id": row["id"], "probabilities": values.tolist(), "accepted": bool(keep)}
                                 for row, values, keep in zip(splits["test"], calibrated, accepted)],
            "limitations": ["Synthetic default is a mechanics fixture, not a quality benchmark",
                            "Group bootstrap assumes independent groups; temporal dependence is not modeled",
                            "Intervals condition on fitted model and selected policy; no retraining uncertainty",
                            "Review has fixed cost and no modeled reviewer error or capacity constraint",
                            "Closed-set confidence is not an out-of-distribution detector"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--provenance", help="Corpus source, license/permission, immutable revision and annotation version")
    parser.add_argument("--cutoffs", nargs=3, metavar=("TRAIN_END", "CALIBRATION_END", "POLICY_END"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--error-cost", type=float, default=1.0)
    parser.add_argument("--review-cost", type=float, default=0.2)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.csv and (not args.provenance or not args.cutoffs):
        parser.error("--csv requires --provenance and three explicit --cutoffs")
    if args.provenance and not args.csv:
        parser.error("--provenance applies only to --csv; fixture provenance is fixed")
    if args.csv:
        rows, csv_digest = load_csv(args.csv, return_digest=True)
    else:
        rows = controlled_fixture()
    report = evaluate(rows, cutoffs=args.cutoffs or DEFAULT_CUTOFFS,
                      error_cost=args.error_cost, review_cost=args.review_cost,
                      min_coverage=args.min_coverage, repetitions=args.bootstrap,
                      provenance=args.provenance or "controlled synthetic fixture")
    if args.csv:
        report["csv_sha256"] = csv_digest
    with args.output.open("xb") as stream:
        stream.write(canonical_json(report))
    print(json.dumps({"report": str(args.output), "test": {key: report["test"][key]
                      for key in ("count", "coverage", "risk", "cost")}}, indent=2))


if __name__ == "__main__":
    main()
