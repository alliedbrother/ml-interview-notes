"""Offline tests for the checksum-pinned SMS study; no network or raw corpus needed."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "content/notes/_examples"))
import sms_spam_study as sms


def fixture_bytes():
    lines = []
    for label, phrase in (("ham", "meeting lunch friends"), ("spam", "offer sale prize")):
        for group in range(20):
            text = f"{phrase} marker{label}{group}"
            lines.extend((f"{label}\t{text}", f"{label}\t{text.upper()}"))
    return ("\n".join(lines) + "\n").encode()


def fixture_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("SMSSpamCollection", fixture_bytes())
        archive.writestr("../must-not-extract.txt", "not extracted")
    return output.getvalue()


class RealCorpusTests(unittest.TestCase):
    def test_normalized_duplicate_groups_and_audit(self):
        rows = sms.parse_corpus(fixture_bytes())
        audit = sms.duplicate_audit(rows)
        self.assertEqual(audit["rows"], 80)
        self.assertEqual(audit["groups"], 40)
        self.assertEqual(audit["duplicate_groups"], 40)
        self.assertEqual(audit["extra_duplicate_rows"], 40)
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertNotIn("timestamp", rows[0])

    def test_parser_rejects_bad_records_and_label_conflicts(self):
        for data in (b"", b"missing-tab", b"unknown\tmessage", b"ham\t ", b"ham\tone\nspam\t ONE\n"):
            with self.assertRaises(ValueError):
                sms.parse_corpus(data)
        rows = sms.parse_corpus(b"ham\tfirst\tsecond\nspam\tother\n")
        self.assertEqual(rows[0]["text"], "first\tsecond")

    def test_four_way_split_group_integrity_and_stratification(self):
        rows = sms.parse_corpus(fixture_bytes())
        splits = sms.grouped_split(rows)
        seen = set()
        for part in sms.protocol.PARTS:
            groups = {row["group"] for row in splits[part]}
            self.assertFalse(groups & seen)
            self.assertEqual({row["label"] for row in splits[part]}, {"ham", "spam"})
            self.assertTrue(all(sum(r["group"] == group for r in splits[part]) == 2 for group in groups))
            seen |= groups
        self.assertEqual([len(splits[part]) for part in sms.protocol.PARTS], [48, 8, 8, 16])
        self.assertEqual(splits, sms.grouped_split(list(reversed(rows))))
        self.assertNotEqual(splits, sms.grouped_split(rows, seed=42))

    def test_rejects_underpowered_classes_or_duplicate_ids(self):
        rows = sms.parse_corpus(fixture_bytes())
        with self.assertRaisesRegex(ValueError, "ten distinct"):
            sms.grouped_split(rows[:8])
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate row ID"):
            sms.grouped_split(rows)

    def test_archive_checksums_before_parsing(self):
        data = fixture_archive()
        with self.assertRaisesRegex(ValueError, "Archive SHA"):
            sms.verified_archive(data)
        with patch.object(sms, "ARCHIVE_SHA256", sms.artifact_capstone.sha256(data)):
            with self.assertRaisesRegex(ValueError, "member SHA"):
                sms.verified_archive(data)
            with patch.object(sms, "DATA_SHA256", sms.artifact_capstone.sha256(fixture_bytes())):
                self.assertEqual(sms.verified_archive(data), fixture_bytes())

    def test_download_is_explicit_verified_and_cached(self):
        data = fixture_archive()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with patch.object(sms, "ARCHIVE_SHA256", sms.artifact_capstone.sha256(data)), \
                    patch.object(sms, "DATA_SHA256", sms.artifact_capstone.sha256(fixture_bytes())), \
                    patch.object(sms.urllib.request, "urlopen", return_value=io.BytesIO(data)) as download:
                self.assertEqual(sms.fetch_archive(path), data)
                self.assertEqual(sms.fetch_archive(path), data)
                download.assert_called_once()
                self.assertEqual(download.call_args.args[0].full_url, sms.URL)
                self.assertEqual(download.call_args.kwargs["timeout"], 30)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            path.write_bytes(b"corrupt cache")
            with patch.object(sms.urllib.request, "urlopen") as download:
                with self.assertRaisesRegex(ValueError, "Archive SHA"):
                    sms.fetch_archive(path)
                download.assert_not_called()

    def test_corrupt_download_never_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with patch.object(sms.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong corpus")):
                with self.assertRaises(ValueError):
                    sms.fetch_archive(path)
            self.assertFalse(path.exists())

    def test_train_only_vocabulary_and_frozen_test_no_selection_leakage(self):
        rows = sms.parse_corpus(fixture_bytes())
        splits = sms.grouped_split(rows)
        captured = []
        make = sms.artifact_capstone.make_pipeline

        def capture(c):
            model = make(c)
            captured.append(model)
            return model

        with patch.object(sms.artifact_capstone, "make_pipeline", side_effect=capture):
            original = sms.study(rows, repetitions=20)
        vocabulary = captured[0].named_steps["tfidf"].vocabulary_
        for part, subset in splits.items():
            for row in subset:
                marker = row["text"].lower().split()[-1]
                self.assertEqual(marker in vocabulary, part == "train")
        frozen = copy.deepcopy(splits)
        for row in frozen["test"]:
            row["text"] = "changed held-out message"
            row["label"] = "spam" if row["label"] == "ham" else "ham"
        with patch.object(sms, "grouped_split", return_value=frozen):
            changed = sms.study(rows, repetitions=20)
        for key in ("temperature", "selected", "policy_curve", "model"):
            self.assertEqual(original[key], changed[key])
        self.assertFalse(original["corpus_identity_verified"])
        self.assertNotIn("archive_sha256", original)

    def test_error_metadata_contains_no_message_text(self):
        rows = sms.parse_corpus(fixture_bytes())
        splits = sms.grouped_split(rows)
        train = splits["train"]
        model = sms.artifact_capstone.make_pipeline(2).fit([r["text"] for r in train], [r["label"] for r in train])
        test = splits["test"]
        values = np.array([[.1, .9] if row["label"] == "ham" else [.9, .1] for row in test])
        errors = sms.error_metadata(test, values, model, np.ones(len(test), dtype=bool))
        self.assertEqual(len(errors), len(test))
        expected = {"id", "truth", "prediction", "confidence", "accepted", "characters", "has_digits", "oov_ngram_fraction"}
        self.assertTrue(all(set(row) == expected for row in errors))
        self.assertTrue(all(row["oov_ngram_fraction"] > 0 for row in errors))

    def test_cli_report_records_verified_identity_and_refuses_overwrite(self):
        data = fixture_archive()
        with tempfile.TemporaryDirectory() as directory:
            archive, output = Path(directory) / "fixture.zip", Path(directory) / "report.json"
            archive.write_bytes(data)
            with patch.object(sms, "ARCHIVE_SHA256", sms.artifact_capstone.sha256(data)), \
                    patch.object(sms, "DATA_SHA256", sms.artifact_capstone.sha256(fixture_bytes())), \
                    patch.object(sys, "argv", ["sms_spam_study.py", "run", "--archive", str(archive),
                                              "--output", str(output), "--bootstrap", "20"]), \
                    patch("builtins.print"):
                sms.main()
                report = json.loads(output.read_text())
                self.assertTrue(report["corpus_identity_verified"])
                self.assertEqual(report["archive_sha256"], sms.artifact_capstone.sha256(data))
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    sms.main()

    def test_saved_real_measurement_identity_membership_and_arithmetic(self):
        report = json.loads((ROOT / "content/notes/_examples/sms_spam_results.json").read_text())
        self.assertTrue(report["corpus_identity_verified"])
        self.assertEqual(report["archive_sha256"], sms.ARCHIVE_SHA256)
        self.assertEqual(report["dataset_sha256"], sms.DATA_SHA256)
        self.assertEqual(report["audit"]["rows"], 5574)
        self.assertEqual(report["audit"]["groups"], 5159)
        ids = [row_id for part in report["splits"].values() for row_id in part["row_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(range(1, 5575)))
        matrix = np.array(report["test"]["confusion_matrix_true_rows_predicted_columns"])
        self.assertEqual(int(matrix.sum()), report["splits"]["test"]["rows"])
        self.assertAlmostEqual(float(matrix.trace() / matrix.sum()), report["test"]["accuracy"])
        policy = report["test"]["policy"]
        errors = report["test"]["errors_without_message_text"]
        self.assertEqual(sum(error["accepted"] for error in errors), policy["errors_accepted"])
        self.assertAlmostEqual(policy["errors_accepted"] / policy["accepted"], policy["risk"])

    def test_saved_report_source_hashes_match_executed_implementation(self):
        report = json.loads((ROOT / "content/notes/_examples/sms_spam_results.json").read_text())
        self.assertEqual(report["study_source_sha256"], sms.artifact_capstone.sha256(Path(sms.__file__).read_bytes()))
        for module in (sms.artifact_capstone, sms.protocol):
            path = Path(module.__file__)
            self.assertEqual(report["sources_sha256"][path.name], sms.artifact_capstone.sha256(path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
