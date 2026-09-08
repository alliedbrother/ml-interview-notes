"""Exercise the downloadable classifier's artifact and service failure contracts."""

import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import httpx
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/notes/_examples/artifact_capstone.py"
spec = importlib.util.spec_from_file_location("artifact_capstone", SOURCE)
capstone = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capstone
spec.loader.exec_module(capstone)


class ArtifactCapstoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="artifact-capstone-")
        cls.original = Path(cls.workspace.name) / "original"
        cls.result = capstone.train_artifact(cls.original)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="artifact-contract-")
        self.artifact = Path(self.scratch.name) / "artifact"
        shutil.copytree(self.original, self.artifact)
        self.digest = self.result["manifest_sha256"]

    def tearDown(self):
        self.scratch.cleanup()

    def load(self):
        return capstone.load_artifact(self.artifact, trusted=True,
                                      expected_manifest_sha256=self.digest)

    def edit_manifest(self, mutation):
        path = self.artifact / "manifest.json"
        manifest = json.loads(path.read_text())
        mutation(manifest)
        data = capstone.canonical_json(manifest)
        path.write_bytes(data)
        self.digest = capstone.sha256(data)

    def assert_rejected_before_deserialization(self, pattern):
        with patch.object(capstone.joblib, "load", side_effect=AssertionError("unsafe load")) as loader:
            with self.assertRaisesRegex(ValueError, pattern):
                self.load()
            loader.assert_not_called()

    def test_group_disjoint_split_and_train_only_vocabulary(self):
        model, _ = self.load()
        splits = capstone.split_fixture(capstone.fixture())
        groups = {part: {r["group"] for r in rows} for part, rows in splits.items()}
        self.assertEqual([len(splits[p]) for p in splits], [48, 12, 12])
        for a, b in (("train", "development"), ("train", "test"), ("development", "test")):
            self.assertFalse(groups[a] & groups[b])
        vocabulary = model.named_steps["tfidf"].vocabulary_
        self.assertTrue(all(r["marker"] in vocabulary for r in splits["train"]))
        for part in ("development", "test"):
            self.assertTrue(all(r["marker"] not in vocabulary for r in splits[part]))
        for part in splits:
            self.assertEqual({r["label"] for r in splits[part]}, set(capstone.LABELS))

    def test_selection_and_test_report_are_separate(self):
        metrics = self.result["metrics"]
        best = min(metrics["candidates"], key=lambda row: (row["development_log_loss"], row["C"]))
        self.assertEqual(metrics["selected_C"], best["C"])
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 12)
        self.assertTrue(metrics["fixture_only"])
        self.assertGreaterEqual(metrics["test_report"]["macro avg"]["f1-score"], 0.9)

    def test_roundtrip_and_class_order(self):
        model, manifest = self.load()
        saved = json.loads((self.artifact / "predictions.json").read_text())
        np.testing.assert_allclose(model.predict_proba(saved["texts"]), saved["probabilities"],
                                   rtol=0, atol=1e-12)
        self.assertEqual(manifest["classes"], model.classes_.tolist())
        self.assertEqual(saved["classes"], manifest["classes"])

    def test_fresh_process_can_load_and_predict(self):
        code = """import sys, json
sys.path.insert(0, sys.argv[1])
import artifact_capstone as c
model, manifest = c.load_artifact(sys.argv[2], trusted=True, expected_manifest_sha256=sys.argv[3])
saved = json.loads((c.Path(sys.argv[2]) / 'predictions.json').read_text())
c.np.testing.assert_allclose(model.predict_proba(saved['texts']), saved['probabilities'], rtol=0, atol=1e-12)
print('fresh-process parity passed')
"""
        result = subprocess.run([sys.executable, "-c", code, str(SOURCE.parent),
                                 str(self.artifact), self.digest], capture_output=True,
                                text=True, timeout=30, cwd=self.scratch.name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_trust_and_external_digest_required(self):
        with patch.object(capstone.joblib, "load") as loader:
            with self.assertRaisesRegex(ValueError, "trusted=True"):
                capstone.load_artifact(self.artifact)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                capstone.load_artifact(self.artifact, trusted=True)
            loader.assert_not_called()

    def test_tampered_manifest_rejected(self):
        with (self.artifact / "manifest.json").open("ab") as stream:
            stream.write(b" ")
        self.assert_rejected_before_deserialization("Manifest digest mismatch")

    def test_tampered_model_rejected(self):
        (self.artifact / "model.joblib").write_bytes(b"not a model")
        self.assert_rejected_before_deserialization("file digest mismatch")

    def test_tampered_prediction_fixture_rejected(self):
        (self.artifact / "predictions.json").write_text("{}")
        self.assert_rejected_before_deserialization("file digest mismatch")

    def test_schema_and_runtime_version_rejected(self):
        original = (self.artifact / "manifest.json").read_bytes()
        for mutate, pattern in (
            (lambda m: m["schema"].update(version=2), "schema version"),
            (lambda m: m["runtime"].update({"scikit-learn": "0.0"}), "runtime version"),
            (lambda m: m.update(format_version=2), "schema version"),
        ):
            with self.subTest(pattern=pattern):
                (self.artifact / "manifest.json").write_bytes(original)
                self.edit_manifest(mutate)
                self.assert_rejected_before_deserialization(pattern)

    def test_path_and_identity_contract_rejected(self):
        original = (self.artifact / "manifest.json").read_bytes()
        for mutate, pattern in (
            (lambda m: m.update(model_file="../other.joblib"), "file contract"),
            (lambda m: m["files"].update({"../outside": "0" * 64}), "file contract"),
            (lambda m: m.update(artifact_version="0" * 64), "identity mismatch"),
        ):
            with self.subTest(pattern=pattern):
                (self.artifact / "manifest.json").write_bytes(original)
                self.edit_manifest(mutate)
                self.assert_rejected_before_deserialization(pattern)

    def test_class_order_mismatch_rejected(self):
        self.edit_manifest(lambda m: m.update(classes=list(reversed(capstone.LABELS))))
        with self.assertRaisesRegex(ValueError, "Class-order"):
            self.load()

    def test_http_parity_order_and_lifecycle(self):
        app = capstone.create_app(self.artifact, trusted=True, expected_manifest_sha256=self.digest)
        before_startup = TestClient(app)
        self.assertEqual(before_startup.get("/health").status_code, 200)
        self.assertEqual(before_startup.get("/ready").status_code, 503)
        self.assertEqual(before_startup.post("/predict", json={"texts": ["invoice"]}).status_code, 503)
        before_startup.close()
        texts = ["invoice refund", "reset password", "tracking parcel"]
        model, _ = self.load()
        with TestClient(app) as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post("/predict", json={"texts": texts})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["artifact_version"], self.result["artifact_version"])
            self.assertEqual(body["classes"], capstone.LABELS)
            self.assertEqual([p["label"] for p in body["predictions"]], model.predict(texts).tolist())
            np.testing.assert_allclose([p["probabilities"] for p in body["predictions"]],
                                       model.predict_proba(texts), rtol=0, atol=1e-12)
        self.assertFalse(app.state.ready)
        self.assertIsNone(app.state.model)

    def test_http_rejects_invalid_input(self):
        app = capstone.create_app(self.artifact, trusted=True, expected_manifest_sha256=self.digest)
        with TestClient(app) as client:
            invalid = ({}, {"texts": []}, {"texts": "invoice"}, {"texts": [1]},
                       {"texts": [None]}, {"texts": [""]}, {"texts": ["\t \n"]},
                       {"texts": ["x" * 4001]}, {"texts": ["x"] * 33},
                       {"texts": ["invoice"], "features": [1]}, {"texts": [True]})
            for payload in invalid:
                with self.subTest(payload=str(payload)[:80]):
                    self.assertEqual(client.post("/predict", json=payload).status_code, 422)
            self.assertEqual(client.post("/predict", json={"texts": ["x"] * 32}).status_code, 200)
            self.assertEqual(client.post("/predict", json={"texts": ["x" * 4000]}).status_code, 200)

    def test_corruption_fails_startup(self):
        (self.artifact / "model.joblib").unlink()
        app = capstone.create_app(self.artifact, trusted=True, expected_manifest_sha256=self.digest)
        with self.assertRaises(FileNotFoundError):
            with TestClient(app):
                self.fail("Corrupt artifact should not start")
        self.assertFalse(app.state.ready)

    def test_repeat_training_and_existing_directory_protection(self):
        second = Path(self.scratch.name) / "second"
        result = capstone.train_artifact(second)
        self.assertEqual(result["metrics"], self.result["metrics"])
        a = json.loads((self.artifact / "predictions.json").read_text())
        b = json.loads((second / "predictions.json").read_text())
        np.testing.assert_allclose(a["probabilities"], b["probabilities"], rtol=0, atol=1e-12)
        with self.assertRaises(FileExistsError):
            capstone.train_artifact(self.artifact)

    def test_cli_demo_and_unknown_vocabulary(self):
        result = subprocess.run([sys.executable, str(SOURCE), "demo", "--output",
                                 str(Path(self.scratch.name) / "cli")], capture_output=True,
                                text=True, timeout=30, cwd=self.scratch.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["roundtrip"], "passed")
        self.assertFalse(body["network_server_started"])
        model, _ = self.load()
        unknown = ["zzzzunseenword"]
        self.assertEqual(model.named_steps["tfidf"].transform(unknown).nnz, 0)
        self.assertTrue(np.isfinite(model.predict_proba(unknown)).all())

    def test_published_downloads_match_sources(self):
        if not (ROOT / "_site").is_dir():
            self.skipTest("Run site/build.py first to check published downloads")
        for name in ("artifact_capstone.py", "requirements-artifact-capstone.txt"):
            published = ROOT / "_site/assets/examples" / name
            self.assertTrue(published.is_file(), f"Missing published capstone asset: {name}")
            self.assertEqual(published.read_bytes(), (SOURCE.parent / name).read_bytes())

    def test_serve_cli_requires_trust_and_binds_only_loopback(self):
        arguments = [str(SOURCE), "serve", "--artifact", str(self.artifact),
                     "--manifest-sha256", self.digest]
        with patch.object(sys, "argv", arguments), patch("uvicorn.run") as run:
            with self.assertRaises(SystemExit) as error:
                capstone.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        with patch.object(sys, "argv", arguments + ["--trust-artifact", "--port", "8123"]), \
                patch("uvicorn.run") as run:
            capstone.main()
            self.assertEqual(run.call_args.kwargs,
                             {"host": "127.0.0.1", "port": 8123, "workers": 1, "access_log": False})
            with TestClient(run.call_args.args[0]) as client:
                self.assertEqual(client.get("/ready").status_code, 200)

    def test_real_loopback_http_prediction_parity(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen([
            sys.executable, str(SOURCE), "serve", "--artifact", str(self.artifact),
            "--manifest-sha256", self.digest, "--trust-artifact", "--port", str(port),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1., trust_env=False) as client:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), "Local server exited before readiness")
                    try:
                        if client.get("/ready").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(.05)
                else:
                    self.fail("Local server did not become ready")
                response = client.post("/predict", json={"texts": ["invoice payment", "reset password"]})
                self.assertEqual(response.status_code, 200)
                model, _ = self.load()
                np.testing.assert_allclose(
                    [row["probabilities"] for row in response.json()["predictions"]],
                    model.predict_proba(["invoice payment", "reset password"]), rtol=0, atol=1e-12)
                self.assertEqual(client.post("/predict", json={"texts": [" "]}).status_code, 422)
        finally:
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate()
        self.assertEqual(process.returncode, 0, stderr)


if __name__ == "__main__":
    unittest.main()
