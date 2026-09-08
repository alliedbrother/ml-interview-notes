"""Offline integrity, evidence-binding, schedule and lesson-table regressions."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/transformers/code"
SOURCE = CODE / "architecture_provenance.py"
LESSON = ROOT / "content/courses/transformers/15-modern-architecture-case-studies.md"
spec = importlib.util.spec_from_file_location("architecture_provenance", SOURCE)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TransformerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="architecture-evidence-")
        self.evidence = Path(self.temporary.name) / "provenance"
        shutil.copytree(CODE / "provenance", self.evidence)
        self.ledger = json.loads((self.evidence / "ledger.json").read_text())

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        (self.evidence / "ledger.json").write_text(json.dumps(self.ledger))

    def config(self, identity):
        entry = next(row for row in self.ledger["configs"] if row["id"] == identity)
        return json.loads((self.evidence / entry["snapshot"]).read_text())

    def test_complete_offline_inventory(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("Offline check attempted network")):
            report = checker.check_ledger(self.evidence)
        self.assertEqual(report["rows"], 20)
        self.assertEqual(report["cells"], 120)
        self.assertEqual(report["public_json_snapshots"], 20)
        self.assertEqual(report["unavailable_configs"], 2)
        self.assertEqual(report["field_assertions"], 370)
        self.assertEqual(report["cell_states"]["unverified"], 11)
        self.assertEqual(report["cell_states"]["partial"], 2)
        self.assertEqual(report["reference_records"], 42)

    def test_tampered_snapshot_rejected(self):
        path = self.evidence / "configs/deepseek-v3.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            checker.check_ledger(self.evidence)

    def test_stale_field_assertion_rejected(self):
        source = next(row for row in self.ledger["configs"] if row["id"] == "gpt-oss120b")
        fact = next(fact for fact in source["facts"] if fact["pointer"] == "/num_local_experts")
        fact["expected"] = 32
        self.save()
        with self.assertRaisesRegex(ValueError, "Field mismatch"):
            checker.check_ledger(self.evidence)

    def test_json_values_compare_types_not_only_python_equality(self):
        source = next(row for row in self.ledger["configs"] if row["id"] == "glm45")
        fact = next(fact for fact in source["facts"] if fact["pointer"] == "/attention_bias")
        fact["expected"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "Field mismatch"):
            checker.check_ledger(self.evidence)

    def test_config_cells_require_actual_checked_field_bindings(self):
        original = copy.deepcopy(self.ledger)
        for alteration in ("delete", "unknown-pointer", "unknown-source"):
            with self.subTest(alteration=alteration):
                self.ledger = copy.deepcopy(original)
                cell = self.ledger["rows"][0]["cells"]["attention"]
                if alteration == "delete":
                    cell.pop("field_evidence")
                elif alteration == "unknown-pointer":
                    cell["field_evidence"][0]["pointers"] = ["/not_a_field"]
                else:
                    cell["field_evidence"][0]["source_id"] = "unknown"
                self.save()
                with self.assertRaisesRegex(ValueError, "field evidence|field pointer"):
                    checker.check_ledger(self.evidence)

    def test_gated_metadata_cannot_be_promoted_to_config_evidence(self):
        blocked = [row for row in self.ledger["configs"] if row["status"] == "unavailable"]
        self.assertEqual({row["id"] for row in blocked}, {"llama4-maverick", "gemma3"})
        self.assertTrue(all("401" in row["error"] for row in blocked))
        row = next(row for row in self.ledger["rows"] if row["id"] == "llama4-maverick")
        row["cells"]["attention"]["status"] = "config-backed"
        self.save()
        with self.assertRaisesRegex(ValueError, "available pinned bytes"):
            checker.check_ledger(self.evidence)

    def test_mutable_revision_and_path_escape_rejected(self):
        original = copy.deepcopy(self.ledger)
        for field, value, message in (("revision", "main", "immutable commit"),
                                      ("snapshot", "../outside.json", "escapes evidence")):
            self.ledger = copy.deepcopy(original)
            self.ledger["configs"][0][field] = value
            self.save()
            with self.assertRaisesRegex(ValueError, message):
                checker.check_ledger(self.evidence)

    def test_pointer_escaping_lists_and_missing_values(self):
        data = {"a/b": {"~key": [0, None, False]}}
        self.assertIs(checker.json_pointer(data, "/a~1b/~0key/2"), False)
        self.assertIsNone(checker.json_pointer(data, "/a~1b/~0key/1"))
        for path in ("a", "/a~2b", "/a~1b/~0key/02"):
            with self.assertRaises(ValueError):
                checker.json_pointer(data, path)
        with self.assertRaises(KeyError):
            checker.json_pointer(data, "/missing")

    def test_real_schema_and_variant_regressions(self):
        gpt = self.config("gpt-oss120b")
        self.assertEqual((gpt["num_local_experts"], gpt["num_experts_per_tok"]), (128, 4))
        self.assertEqual(gpt["layer_types"].count("sliding_attention"), 18)
        ds = self.config("deepseek-v3")
        self.assertEqual(ds["num_attention_heads"], ds["num_key_value_heads"])
        self.assertEqual(ds["kv_lora_rank"], 512)
        mistral = self.config("mistral-large3")
        self.assertEqual(mistral["moe"]["num_experts_per_tok"], 4)
        self.assertEqual(mistral["moe"]["num_shared_experts"], 1)
        self.assertEqual(self.config("glm45")["partial_rotary_factor"], .5)
        dense, moe = [self.config(name)["text_config"] for name in ("gemma4-dense", "gemma4-moe")]
        self.assertFalse(dense["enable_moe_block"])
        self.assertTrue(moe["enable_moe_block"])
        self.assertEqual((moe["num_experts"], moe["top_k_experts"]), (128, 8))
        self.assertEqual(dense["rope_parameters"]["full_attention"]["partial_rotary_factor"], .25)

    def test_exact_layer_schedules_not_rounded_motifs(self):
        kimi = self.config("kimi-linear")
        layers = kimi["linear_attn_config"]
        self.assertEqual(len(layers["kda_layers"]), 20)
        self.assertEqual(len(layers["full_attn_layers"]), 7)
        self.assertEqual(sorted(layers["kda_layers"] + layers["full_attn_layers"]), list(range(1, 28)))
        self.assertEqual(kimi["num_shared_experts"], 1)
        mimo = self.config("mimo-v2-flash")
        self.assertEqual(mimo["hybrid_layer_pattern"].count(1), 39)
        self.assertEqual(mimo["hybrid_layer_pattern"].count(0), 9)
        nemo = self.config("nemotron3-nano")
        pattern = nemo["hybrid_override_pattern"]
        self.assertEqual((pattern.count("M"), pattern.count("E"), pattern.count("*")), (23, 23, 6))
        self.assertIn("rope_theta", nemo)
        row = next(row for row in self.ledger["rows"] if row["id"] == "nemotron3-nano")
        self.assertEqual(row["cells"]["position"]["status"], "implementation-reviewed")

    def test_all_table_cells_match_the_published_evidence_matrix(self):
        chapter = LESSON.read_text()
        matrix = chapter.split("### Per-cell coverage\n", 1)[1].split("### What checking", 1)[0]
        actual = {}
        for line in matrix.splitlines():
            if line.startswith("|"):
                columns = [part.strip() for part in line.split("|")[1:-1]]
                if columns[0] != "Model" and not columns[0].startswith("---"):
                    actual[columns[0]] = columns[1:]
        short = {"config-backed": "C", "implementation-reviewed": "I", "vendor-reported": "V",
                 "partial": "P", "unverified": "U", "not-applicable": "N/A"}
        order = ("size", "attention", "moe", "shared_expert", "norm", "position")
        expected = {row["label"]: [short[row["cells"][name]["status"]] for name in order]
                    for row in self.ledger["rows"]}
        self.assertEqual(actual, expected)
        table = chapter.split("### Full table\n", 1)[1].split("### What is universal", 1)[0]
        self.assertEqual(len([line for line in table.splitlines() if line.startswith("| **")]), 20)
        self.assertNotIn("32–128 exp", table)
        self.assertIn("**128 experts, 4 active**", table)
        self.assertIn("**NoPE in inspected attention paths**", table)
        self.assertIn("**20:7 actual layers**", table)
        self.assertIn("39 local / 9 global", table)
        self.assertIn("370 exact JSON-pointer assertions", chapter)

    def test_cli_and_native_alternative_sources(self):
        result = subprocess.run([sys.executable, str(SOURCE)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["network_requests"], 0)
        refs = {ref["id"]: ref for ref in self.ledger["references"]}
        self.assertIn("google-gemma-config", refs)
        self.assertIn("meta-llama4-card", refs)
        self.assertEqual(refs["google-gemma-config"]["locators"][0]["symbol"], "get_config_for_27b_v3")
        grok = next(row for row in self.ledger["rows"] if row["id"] == "grok25")
        self.assertTrue(all(cell["status"] == "unverified" for cell in grok["cells"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
