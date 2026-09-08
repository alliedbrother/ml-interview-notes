"""Leakage, calibration, policy and uncertainty contracts for the offline lab."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "content/notes/_examples"))
import evaluation_protocol as protocol


class EvaluationProtocolTests(unittest.TestCase):
    def test_group_time_disjoint_and_cutoff_equality(self):
        splits, purged = protocol.temporal_split(protocol.controlled_fixture())
        self.assertEqual([len(splits[part]) for part in protocol.PARTS], [36, 12, 12, 12])
        self.assertEqual(purged, [])
        seen = set()
        for part in protocol.PARTS:
            groups = {row["group"] for row in splits[part]}
            self.assertFalse(groups & seen)
            seen |= groups
        self.assertTrue(all(row["group"].endswith(("06", "07")) for row in splits["calibration"]))
        for a, b in zip(protocol.PARTS, protocol.PARTS[1:]):
            self.assertLess(max(protocol.instant(row["timestamp"]) for row in splits[a]),
                            min(protocol.instant(row["timestamp"]) for row in splits[b]))

    def test_boundary_spanning_group_is_purged_entirely(self):
        rows = protocol.controlled_fixture()
        rows[1]["timestamp"] = "2026-01-07T00:00:00Z"
        splits, purged = protocol.temporal_split(rows)
        self.assertEqual(purged, ["account-00"])
        self.assertFalse(any(row["group"] == "account-00" for subset in splits.values() for row in subset))

    def test_rejects_naive_time_duplicate_ids_and_cross_group_duplicates(self):
        for mutation, message in (
            (lambda rows: rows[0].update(timestamp="2026-01-01"), "UTC offset"),
            (lambda rows: rows[1].update(id=rows[0]["id"]), "Duplicate row ID"),
            (lambda rows: rows[2].update(text=rows[0]["text"].upper()), "spans groups"),
        ):
            rows = protocol.controlled_fixture()
            mutation(rows)
            with self.assertRaisesRegex(ValueError, message):
                protocol.temporal_split(rows)

    def test_rejects_bad_windows_and_unseen_label(self):
        rows = protocol.controlled_fixture()
        with self.assertRaisesRegex(ValueError, "increasing"):
            protocol.temporal_split(rows, list(reversed(protocol.DEFAULT_CUTOFFS)))
        with self.assertRaisesRegex(ValueError, "retained rows"):
            protocol.temporal_split(rows, ["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"])
        rows[-1]["label"] = "unknown"
        with self.assertRaisesRegex(ValueError, "absent from training"):
            protocol.temporal_split(rows)

    def test_train_only_vocabulary_and_untouched_test_selection(self):
        rows = protocol.controlled_fixture()
        captured = []
        make = protocol.make_pipeline

        def capture(c):
            pipeline = make(c)
            captured.append(pipeline)
            return pipeline

        with patch.object(protocol, "make_pipeline", side_effect=capture):
            original = protocol.evaluate(rows, repetitions=30)
        splits, _ = protocol.temporal_split(rows)
        vocabulary = captured[0].named_steps["tfidf"].vocabulary_
        self.assertTrue(all(row["marker"] in vocabulary for row in splits["train"]))
        self.assertTrue(all(row["marker"] not in vocabulary for part in protocol.PARTS[1:] for row in splits[part]))
        changed = copy.deepcopy(rows)
        test_ids = {row["id"] for row in splits["test"]}
        for row in changed:
            if row["id"] in test_ids:
                row["label"] = "billing"
                row["text"] = "changed held out content " + row["id"]
        mutated = protocol.evaluate(changed, repetitions=30)
        for key in ("temperature", "selected", "policy_curve"):
            self.assertEqual(original[key], mutated[key])

    def test_temperature_fits_calibration_not_policy_labels(self):
        rows = protocol.controlled_fixture()
        original = protocol.evaluate(rows, repetitions=20)
        splits, _ = protocol.temporal_split(rows)
        policy_ids = {row["id"] for row in splits["policy"]}
        for row in rows:
            if row["id"] in policy_ids:
                row["label"] = "billing"
        changed = protocol.evaluate(rows, repetitions=20)
        self.assertEqual(original["temperature"], changed["temperature"])

    def test_temperature_preserves_argmax_and_can_soften_overconfidence(self):
        values = np.array([[.99, .01]] * 10)
        labels = ["a"] * 7 + ["b"] * 3
        temperature = protocol.fit_temperature(values, labels, ["a", "b"])
        self.assertGreater(temperature, 1)
        scaled = protocol.temperature_scale(values, temperature)
        np.testing.assert_array_equal(values.argmax(1), scaled.argmax(1))
        self.assertLess(protocol.calibration_metrics(scaled, labels, ["a", "b"])["log_loss"],
                        protocol.calibration_metrics(values, labels, ["a", "b"])["log_loss"])
        np.testing.assert_allclose(protocol.temperature_scale(values, 1), values)

    def test_probability_contract_and_label_order_validation(self):
        for values in ([], [[.8, .8]], [[-1, 2]], [[float("nan"), .5]], [[.5]]):
            with self.assertRaises(ValueError):
                protocol.temperature_scale(values, 1)
        for temperature in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                protocol.temperature_scale([[.5, .5]], temperature)
        with self.assertRaisesRegex(ValueError, "width"):
            protocol.select_policy([[.5, .5]], ["a"], ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            protocol.select_policy([[.5, .5]], ["z"], ["a", "b"])

    def test_worked_policy_cost_coverage_and_reject_all(self):
        values = [[.95, .05], [.80, .20], [.60, .40], [.55, .45]]
        selected, curve = protocol.select_policy(values, ["a", "a", "b", "b"], ["a", "b"])
        self.assertAlmostEqual(selected["threshold"], .8)
        self.assertEqual(selected["coverage"], .5)
        self.assertEqual(selected["risk"], 0)
        self.assertAlmostEqual(selected["cost"], .1)
        self.assertIsNone(curve[-1]["risk"])
        constrained, _ = protocol.select_policy(values, ["a", "a", "b", "b"], ["a", "b"], min_coverage=1)
        self.assertEqual(constrained["coverage"], 1)
        reject, _ = protocol.select_policy([[1, 0]], ["b"], ["a", "b"])
        self.assertEqual(reject["accepted"], 0)

    def test_ties_are_accepted_together_and_cost_ties_prefer_coverage(self):
        selected, curve = protocol.select_policy([[.5, .5], [.5, .5]], ["a", "b"], ["a", "b"], review_cost=.5)
        self.assertEqual(selected["coverage"], 1)
        self.assertTrue(all(row["coverage"] in (0, 1) for row in curve))
        for kwargs in ({"min_coverage": 1.1}, {"error_cost": -1}, {"review_cost": float("nan")}):
            with self.assertRaises(ValueError):
                protocol.select_policy([[.5, .5]], ["a"], ["a", "b"], **kwargs)

    def test_reliability_bins_include_one_and_zero_empty_bins(self):
        metrics = protocol.calibration_metrics([[1, 0], [.5, .5]], ["a", "b"], ["a", "b"], bins=2)
        self.assertEqual(metrics["reliability"][1]["count"], 2)
        self.assertIsNone(metrics["reliability"][0]["accuracy"])
        self.assertAlmostEqual(metrics["multiclass_brier_sum"], .25)
        self.assertAlmostEqual(metrics["top_label_ece"], .25)

    def test_calibration_metrics_preserve_declared_class_order(self):
        values = np.array([[.8, .1, .1], [.1, .6, .3], [.2, .1, .7]])
        expected = protocol.calibration_metrics(values, ["a", "b", "c"], ["a", "b", "c"])
        permuted = protocol.calibration_metrics(values[:, [2, 0, 1]], ["a", "b", "c"], ["c", "a", "b"])
        for metric in ("log_loss", "multiclass_brier_sum", "top_label_ece"):
            self.assertAlmostEqual(expected[metric], permuted[metric])

    def test_group_bootstrap_keeps_clusters_and_pooled_row_estimand(self):
        result = protocol.group_bootstrap([True, True, True, False], [True] * 4,
                                          ["a", "a", "a", "b"], repetitions=100, seed=2)
        self.assertEqual(result["intervals"]["risk"]["valid_replicates"], 100)
        self.assertEqual(result["intervals"]["risk"]["low"], 0)
        self.assertEqual(result["intervals"]["risk"]["high"], 1)
        self.assertEqual(result, protocol.group_bootstrap([True, True, True, False], [True] * 4,
                                                         ["a", "a", "a", "b"], repetitions=100, seed=2))
        empty = protocol.group_bootstrap([True, False], [False, False], ["a", "b"], repetitions=20)
        self.assertEqual(empty["intervals"]["risk"], {"low": None, "high": None, "valid_replicates": 0})
        with self.assertRaisesRegex(ValueError, "two independent groups"):
            protocol.group_bootstrap([True], [True], ["a"])

    def test_csv_and_fresh_process_report(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path, output = Path(directory) / "fixture.csv", Path(directory) / "report.json"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(protocol.REQUIRED), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(protocol.controlled_fixture())
            self.assertEqual(len(protocol.load_csv(csv_path)), 72)
            original_bytes = csv_path.read_bytes()
            with patch.object(Path, "read_bytes", return_value=original_bytes) as snapshot:
                loaded, digest = protocol.load_csv(csv_path, return_digest=True)
            self.assertEqual(len(loaded), 72)
            self.assertEqual(digest, protocol.sha256(original_bytes))
            snapshot.assert_called_once_with()
            command = [sys.executable, str(Path(protocol.__file__)), "--csv", str(csv_path),
                       "--provenance", "controlled fixture CSV, not a real corpus", "--cutoffs",
                       *protocol.DEFAULT_CUTOFFS, "--bootstrap", "20", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["selected_on"], "policy")
            self.assertEqual(result["csv_sha256"], protocol.sha256(csv_path.read_bytes()))
            self.assertEqual(result["test"]["count"], 12)
            self.assertEqual(len(result["test_predictions"]), 12)
            self.assertNotIn("NaN", output.read_text())
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)


if __name__ == "__main__":
    unittest.main()
