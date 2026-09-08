"""Checksum-pinned real-corpus classification study, without temporal claims.

Data: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI repository.
https://doi.org/10.24432/C5CC84 ; current UCI license: CC BY 4.0.
https://creativecommons.org/licenses/by/4.0/ ; supplied as-is, without warranty.
No raw messages are redistributed by this module or its JSON report.
Download artifact_capstone.py and evaluation_protocol.py beside this file.
"""

import argparse
from collections import Counter, defaultdict
import io
import importlib.metadata
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

import artifact_capstone
import evaluation_protocol as protocol


URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
DATA_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
MAX_BYTES = 1_000_000
SEED = 41
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"


def verified_archive(data):
    """Only read named in-memory members; never extract arbitrary ZIP paths."""
    if len(data) > MAX_BYTES or artifact_capstone.sha256(data) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch; do not silently accept a new corpus revision")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo("SMSSpamCollection")
        if info.file_size > MAX_BYTES:
            raise ValueError("Dataset member exceeds size limit")
        corpus = archive.read(info)
    if artifact_capstone.sha256(corpus) != DATA_SHA256:
        raise ValueError("Dataset member SHA-256 mismatch")
    return corpus


def fetch_archive(destination):
    destination = Path(destination)
    if destination.exists():
        data = destination.read_bytes()
        verified_archive(data)
        return data
    request = urllib.request.Request(URL, headers={"User-Agent": "MLInterviewNotes-corpus-study/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    verified_archive(data)
    with destination.open("xb") as stream:
        stream.write(data)
    return data


def parse_corpus(data):
    """First tab separates label from raw message; normalized text defines groups."""
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        fields = line.split("\t", 1)
        if len(fields) != 2 or fields[0] not in {"ham", "spam"} or not fields[1].strip():
            raise ValueError(f"Malformed labeled message at row {line_number}")
        label, text = fields
        normalized = " ".join(text.casefold().split())
        rows.append({"id": line_number, "label": label, "text": text,
                     "group": artifact_capstone.sha256(normalized.encode("utf-8"))})
    if not rows:
        raise ValueError("Corpus is empty")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    conflicts = [group for group, members in groups.items() if len({row["label"] for row in members}) > 1]
    if conflicts:
        raise ValueError("Conflicting labels within normalized duplicate groups; audit before splitting")
    return rows


def grouped_split(rows, seed=SEED):
    """60/10/10/20 by normalized group, stratified by group label, not chronology."""
    group_labels = {}
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate row ID")
        ids.add(row["id"])
        if row["group"] in group_labels and group_labels[row["group"]] != row["label"]:
            raise ValueError("Conflicting labels within group")
        group_labels[row["group"]] = row["label"]
    groups = sorted(group_labels)
    if set(group_labels.values()) != {"ham", "spam"} or min(Counter(group_labels.values()).values()) < 10:
        raise ValueError("Require at least ten distinct groups per ham/spam class")

    def split(selected, fraction):
        return train_test_split(selected, test_size=fraction, random_state=seed,
                                stratify=[group_labels[group] for group in selected])

    train, remainder = split(groups, .4)
    development, test = split(remainder, .5)
    calibration, policy = split(development, .5)
    membership = {part: set(selected) for part, selected in zip(protocol.PARTS, (train, calibration, policy, test))}
    return {part: sorted([row for row in rows if row["group"] in selected], key=lambda row: row["id"])
            for part, selected in membership.items()}


def duplicate_audit(rows):
    sizes = Counter(row["group"] for row in rows)
    return {"rows": len(rows), "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "groups": len(sizes), "duplicate_groups": sum(size > 1 for size in sizes.values()),
            "rows_in_duplicate_groups": sum(size for size in sizes.values() if size > 1),
            "extra_duplicate_rows": len(rows) - len(sizes), "largest_group": max(sizes.values()),
            "normalization": "Unicode casefold plus whitespace collapse; full SHA-256 group key",
            "conflicting_group_labels": 0,
            "limitations": "No sender/conversation IDs, timestamps, or near-duplicate/campaign clustering"}


def error_metadata(rows, values, model, accepted):
    """No message excerpts: public corpus IDs still make records re-identifiable."""
    classes = model.classes_.tolist()
    analyzer = model.named_steps["tfidf"].build_analyzer()
    vocabulary = model.named_steps["tfidf"].vocabulary_
    errors = []
    for row, probability, keep in zip(rows, values, accepted):
        predicted = classes[int(probability.argmax())]
        if predicted == row["label"]:
            continue
        features = analyzer(row["text"])
        errors.append({"id": row["id"], "truth": row["label"], "prediction": predicted,
                       "confidence": float(probability.max()), "accepted": bool(keep),
                       "characters": len(row["text"]), "has_digits": any(c.isdigit() for c in row["text"]),
                       "oov_ngram_fraction": sum(feature not in vocabulary for feature in features) / len(features)
                       if features else None})
    return sorted(errors, key=lambda row: (-row["confidence"], row["id"]))


def study(rows, *, repetitions=1000, seed=SEED):
    splits = grouped_split(rows, seed)
    texts = lambda part: [row["text"] for row in splits[part]]
    labels = lambda part: [row["label"] for row in splits[part]]
    model = artifact_capstone.make_pipeline(2.0).fit(texts("train"), labels("train"))
    classes = model.classes_.tolist()
    temperature = protocol.fit_temperature(model.predict_proba(texts("calibration")), labels("calibration"), classes)
    policy_values = protocol.temperature_scale(model.predict_proba(texts("policy")), temperature)
    selected, curve = protocol.select_policy(policy_values, labels("policy"), classes, error_cost=1, review_cost=.2)
    raw = model.predict_proba(texts("test"))
    values = protocol.temperature_scale(raw, temperature)
    targets = protocol.label_indices(labels("test"), classes, len(raw))
    predicted = values.argmax(axis=1)
    correct = predicted == targets
    accepted = values.max(axis=1) >= selected["threshold"]
    majority = Counter(labels("train")).most_common(1)[0][0]
    counts = Counter(labels("train"))
    prior = np.tile([counts[label] / len(splits["train"]) for label in classes], (len(values), 1))
    test = {"accuracy": float(correct.mean()),
            "classification": classification_report(labels("test"), model.classes_[predicted],
                                                       labels=classes, output_dict=True, zero_division=0),
            "confusion_matrix_true_rows_predicted_columns": confusion_matrix(targets, predicted, labels=[0, 1]).tolist(),
            "spam_average_precision": float(average_precision_score(targets == classes.index("spam"), raw[:, classes.index("spam")])),
            "spam_roc_auc": float(roc_auc_score(targets == classes.index("spam"), raw[:, classes.index("spam")])),
            "raw_calibration": protocol.calibration_metrics(raw, labels("test"), classes),
            "scaled_calibration": protocol.calibration_metrics(values, labels("test"), classes),
            "majority_baseline": {"label": majority, "accuracy": float(np.mean(np.array(labels("test")) == majority)),
                                  "calibration": protocol.calibration_metrics(prior, labels("test"), classes)},
            "policy": protocol.policy_metrics(correct, accepted, error_cost=1, review_cost=.2),
            "bootstrap": protocol.group_bootstrap(correct, accepted, [row["group"] for row in splits["test"]],
                                                    repetitions=repetitions, seed=seed, error_cost=1, review_cost=.2)}
    test["errors_without_message_text"] = error_metadata(splits["test"], values, model, accepted)
    split_manifest = {part: {"rows": len(subset), "groups": len({row["group"] for row in subset}),
                             "class_counts": dict(sorted(Counter(row["label"] for row in subset).items())),
                             "row_ids": [row["id"] for row in subset],
                             "membership_sha256": artifact_capstone.sha256(artifact_capstone.canonical_json(
                                 [{"id": row["id"], "group": row["group"]} for row in subset]))}
                      for part, subset in splits.items()}
    return {"format_version": 1, "study": "SMS duplicate-group holdout, fixed TF-IDF/logistic baseline",
            "dataset_attribution": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI. DOI:10.24432/C5CC84",
            "dataset_source": SOURCE, "license": LICENSE, "license_verified": "2026-09-08",
            "archive_url": URL, "expected_archive_sha256": ARCHIVE_SHA256, "expected_dataset_sha256": DATA_SHA256,
            "corpus_identity_verified": False,
            "data_identity_note": "Only the CLI run path verifies the expected archive/member hashes before parsing",
            "parsed_rows_sha256": artifact_capstone.sha256(artifact_capstone.canonical_json(rows)),
            "sources_sha256": {Path(module.__file__).name: artifact_capstone.sha256(Path(module.__file__).read_bytes())
                                for module in (artifact_capstone, protocol)},
            "study_source_sha256": artifact_capstone.sha256(Path(__file__).read_bytes()),
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn")}},
            "seed": seed, "audit": duplicate_audit(rows), "splits": split_manifest,
            "split_policy": "60/10/10/20 by group; stratified by group label; random, not chronological",
            "model": {"C": float(model.named_steps["classifier"].C),
                      "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                      "vocabulary_sha256": artifact_capstone.sha256(artifact_capstone.canonical_json(
                          sorted(model.named_steps["tfidf"].vocabulary_.items()))),
                      "classes": classes, "ngram_range": list(model.named_steps["tfidf"].ngram_range),
                      "sublinear_tf": model.named_steps["tfidf"].sublinear_tf},
            "temperature": temperature, "selected_on": "policy", "selected": selected,
            "policy_curve": curve, "test": test,
            "limitations": ["Public historic corpus; not current spam or future-time performance",
                            "Normalized duplicate groups are not verified independent senders or campaigns",
                            "All duplicate rows retained; pooled-row results weight repeated messages",
                            "Bootstrap conditions on fitted policy and assumes independent groups",
                            "Single prespecified split/model; no test-set model or threshold selection",
                            "Uniform error/review costs are illustrative; no reviewer errors or capacity model",
                            "Error IDs enable corpus lookup; absence of raw text is not anonymization"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Explicitly download and checksum the official UCI archive")
    fetch.add_argument("--archive", type=Path, required=True)
    run = subparsers.add_parser("run", help="Offline evaluation of an already downloaded, pinned archive")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "fetch":
        data = fetch_archive(args.archive)
        print(json.dumps({"archive": str(args.archive), "bytes": len(data), "sha256": ARCHIVE_SHA256}))
        return
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation report")
    rows = parse_corpus(verified_archive(args.archive.read_bytes()))
    report = study(rows, repetitions=args.bootstrap)
    report.update(corpus_identity_verified=True, archive_sha256=ARCHIVE_SHA256, dataset_sha256=DATA_SHA256)
    with args.output.open("xb") as stream:
        stream.write(artifact_capstone.canonical_json(report))
    print(json.dumps({"report": str(args.output), "accuracy": report["test"]["accuracy"],
                      "test_policy": report["test"]["policy"]}, indent=2))


if __name__ == "__main__":
    main()
