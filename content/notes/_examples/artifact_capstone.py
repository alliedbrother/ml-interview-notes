"""Offline text-classifier artifact and HTTP-contract capstone. Python 3.11.

Only load artifacts from a producer you trust. A digest is integrity, not trust.
Run `python artifact_capstone.py demo --output ./ticket-artifact`.
"""

import argparse
from contextlib import asynccontextmanager
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import random
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline


LABELS = ["account", "billing", "delivery"]
SCHEMA = {"version": 1, "input": "texts", "min_items": 1, "max_items": 32,
          "max_characters": 4000, "extra_fields": "forbid", "blank": "reject"}
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "fastapi", "pydantic")


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def runtime_versions():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in PACKAGES}}


def fixture():
    """Paired paraphrases share an incident group; IDs never become features."""
    phrases = {
        "account": ("reset password login account access", "account login password help"),
        "billing": ("invoice charge payment refund billing", "billing payment invoice help"),
        "delivery": ("parcel tracking shipment delivery address", "delivery parcel shipment help"),
    }
    rows = []
    for label, variants in phrases.items():
        for incident in range(12):
            group = f"{label}-{incident:02d}"
            # Unique markers make train-only vocabulary fitting directly testable.
            marker = f"marker{LABELS.index(label) * 12 + incident:03d}"
            for variant, phrase in enumerate(variants):
                rows.append({"id": f"{group}-{variant}", "group": group, "label": label,
                             "text": f"{phrase} {marker}", "marker": marker})
    return rows


def split_fixture(rows):
    rng = random.Random(41)
    groups = {part: set() for part in ("train", "development", "test")}
    for label in LABELS:
        candidates = sorted({r["group"] for r in rows if r["label"] == label})
        if len(candidates) != 12:
            raise ValueError("This fixed teaching split expects 12 groups per label")
        rng.shuffle(candidates)
        for part, selected in zip(groups, (candidates[:8], candidates[8:10], candidates[10:])):
            groups[part].update(selected)
    return {part: [r for r in rows if r["group"] in selected]
            for part, selected in groups.items()}


def make_pipeline(c):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ("classifier", LogisticRegression(C=c, solver="lbfgs", max_iter=500,
                                                       random_state=41))])


def train_artifact(output):
    """Select C using development log loss, evaluate test once, persist the pipeline."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = fixture()
    splits = split_fixture(rows)
    text = lambda part: [r["text"] for r in splits[part]]
    labels = lambda part: [r["label"] for r in splits[part]]
    candidates = []
    for c in (0.5, 2.0, 8.0):
        model = make_pipeline(c).fit(text("train"), labels("train"))
        loss = log_loss(labels("development"), model.predict_proba(text("development")),
                        labels=model.classes_)
        candidates.append((float(loss), c, model))
    _, selected_c, model = min(candidates, key=lambda item: (item[0], item[1]))
    # Keep the selected train-only fit: its vocabulary supports a leakage invariant.
    test_probabilities = model.predict_proba(text("test"))
    predictions = model.classes_[test_probabilities.argmax(axis=1)]
    metrics = {"fixture_only": True, "selection_metric": "development_log_loss",
               "candidates": [{"C": c, "development_log_loss": loss}
                              for loss, c, _ in candidates],
               "selected_C": selected_c,
               "test_log_loss": float(log_loss(labels("test"), test_probabilities,
                                                labels=model.classes_)),
               "test_report": classification_report(labels("test"), predictions,
                                                     labels=LABELS, output_dict=True,
                                                     zero_division=0),
               "confusion_matrix": confusion_matrix(labels("test"), predictions,
                                                      labels=LABELS).tolist()}
    split_manifest = {part: {"ids": [r["id"] for r in subset],
                             "groups": sorted({r["group"] for r in subset})}
                      for part, subset in splits.items()}
    prediction_fixture = {"texts": text("test"), "labels": labels("test"),
                          "classes": model.classes_.tolist(),
                          "probabilities": test_probabilities.tolist(),
                          "absolute_tolerance": 1e-12}
    payload = io.BytesIO()
    joblib.dump(model, payload, compress=3)
    model_bytes = payload.getvalue()
    files = {"model.joblib": model_bytes, "dataset.json": canonical_json(rows),
             "splits.json": canonical_json(split_manifest),
             "metrics.json": canonical_json(metrics),
             "predictions.json": canonical_json(prediction_fixture)}
    for name, data in files.items():
        (output / name).write_bytes(data)
    manifest = {"format_version": 1, "task": "synthetic-ticket-classification",
                "artifact_version": sha256(model_bytes), "model_file": "model.joblib",
                "classes": LABELS, "schema": SCHEMA, "runtime": runtime_versions(),
                "files": {name: sha256(data) for name, data in files.items()},
                "source_sha256": sha256(Path(__file__).read_bytes()), "seed": 41,
                "split_policy": "group-disjoint, stratified per label; 8/2/2 groups",
                "selection": "C chosen on development loss; no train+dev refit",
                "limitations": "Synthetic plumbing fixture, not a quality benchmark"}
    manifest_bytes = canonical_json(manifest)
    (output / "manifest.json").write_bytes(manifest_bytes)
    return {"manifest_sha256": sha256(manifest_bytes), "metrics": metrics,
            "artifact_version": manifest["artifact_version"]}


def load_artifact(directory, *, trusted=False, expected_manifest_sha256=None):
    """Validate a trusted producer's manifest and bytes before pickle execution.

    The expected manifest digest must come from a trusted release channel, not
    from an adjacent untrusted checksum file. This is NOT a pickle sandbox.
    """
    if not trusted:
        raise ValueError("Explicit trusted=True required: joblib can execute code")
    if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64:
        raise ValueError("A trusted expected manifest SHA-256 is required")
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if sha256(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("Manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("format_version") != 1 or manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported artifact or input schema version")
    if manifest.get("runtime") != runtime_versions():
        raise ValueError("Exact runtime version mismatch; rebuild in this environment")
    expected_files = {"model.joblib", "dataset.json", "splits.json", "metrics.json", "predictions.json"}
    if set(manifest.get("files", {})) != expected_files or manifest.get("model_file") != "model.joblib":
        raise ValueError("Unexpected artifact file contract")
    # Hash and deserialize the same in-memory bytes, avoiding a hash/load reopen race.
    files = {name: (directory / name).read_bytes() for name in expected_files}
    if any(sha256(data) != manifest["files"][name] for name, data in files.items()):
        raise ValueError("Artifact file digest mismatch")
    if manifest.get("artifact_version") != sha256(files["model.joblib"]):
        raise ValueError("Artifact identity mismatch")
    model = joblib.load(io.BytesIO(files["model.joblib"]))
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["tfidf", "classifier"]:
        raise ValueError("Expected the full TF-IDF/classifier pipeline")
    if manifest.get("classes") != LABELS or model.classes_.tolist() != LABELS:
        raise ValueError("Class-order contract mismatch")
    warm = model.predict_proba(["invoice payment"])
    if warm.shape != (1, len(LABELS)) or not np.isfinite(warm).all() or not np.allclose(warm.sum(1), 1):
        raise ValueError("Warm-up failed probability contract")
    return model, manifest


Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4000)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    texts: list[Text] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def reject_blank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("Texts cannot be whitespace only")
        return values


def create_app(directory, *, trusted=False, expected_manifest_sha256=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        model, manifest = load_artifact(directory, trusted=trusted,
                                       expected_manifest_sha256=expected_manifest_sha256)
        app.state.model, app.state.manifest = model, manifest
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.model = None

    app = FastAPI(title="Ticket classifier teaching service", lifespan=lifespan)
    app.state.ready = False

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready")
        return {"ok": True, "artifact_version": app.state.manifest["artifact_version"]}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready")
        probabilities = app.state.model.predict_proba(request.texts)
        return {"schema_version": 1, "artifact_version": app.state.manifest["artifact_version"],
                "classes": LABELS,
                "predictions": [{"label": LABELS[int(row.argmax())], "probabilities": row.tolist()}
                                for row in probabilities]}

    return app


def run_demo(output):
    result = train_artifact(output)
    app = create_app(output, trusted=True, expected_manifest_sha256=result["manifest_sha256"])
    saved = json.loads((Path(output) / "predictions.json").read_text())
    with TestClient(app) as client:
        response = client.post("/predict", json={"texts": saved["texts"]})
        response.raise_for_status()
        body = response.json()
        np.testing.assert_allclose([r["probabilities"] for r in body["predictions"]],
                                   saved["probabilities"], rtol=0, atol=saved["absolute_tolerance"])
        if client.post("/predict", json={"texts": [" "]}).status_code != 422:
            raise AssertionError("Invalid input was accepted")
    return {**result, "roundtrip": "passed", "http_status": response.status_code,
            "test_examples": len(saved["texts"]), "network_server_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "demo"):
        training = commands.add_parser(command)
        training.add_argument("--output", type=Path, required=True,
                              help="New artifact directory; existing paths are never overwritten")
    serving = commands.add_parser("serve", help="Single-process loopback HTTP server")
    serving.add_argument("--artifact", type=Path, required=True)
    serving.add_argument("--manifest-sha256", required=True,
                         help="Expected manifest digest obtained from a trusted release channel")
    serving.add_argument("--trust-artifact", action="store_true",
                         help="Explicitly trust the producer of this executable joblib artifact")
    serving.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.command == "serve":
        if not args.trust_artifact:
            parser.error("serve requires --trust-artifact; a digest alone is not trust")
        if not 1 <= args.port <= 65535:
            parser.error("port must be between 1 and 65535")
        import uvicorn

        app = create_app(args.artifact, trusted=True,
                         expected_manifest_sha256=args.manifest_sha256)
        uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
        return
    result = run_demo(args.output) if args.command == "demo" else train_artifact(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
