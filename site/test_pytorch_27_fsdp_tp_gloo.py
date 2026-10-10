"""Contracts for the module 27 lab: FSDP2, TP, DTensor, PP and DCP on 2 gloo ranks.

The lab spawns processes, so it runs in a subprocess (syllabus D.4): spawn
cannot re-import a digit-named module from inside the unittest process.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
SCRIPT = CODE / "27_fsdp_tp_gloo.py"
_STATE = {}


def last_json_document(text):
    """Return the JSON object that ends ``text``.

    The lab sends native gloo chatter to stderr, but a stray line on stdout (a
    future torch log, a debug print) must not turn a numerical contract into a
    JSONDecodeError, so parse from the last line that opens a top-level object.
    """
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"^\{", text, re.M))):
        try:
            obj, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not text[end:].strip():
            return obj
    raise AssertionError(f"stdout does not end with a JSON object: {text[-500:]!r}")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class ShardedTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--world-size", "2", "--steps", "3",
                 "--seed", "0", "--json-out", str(path)],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            cls.proc = proc
            cls.result = json.loads(path.read_text()) if path.exists() else None

    def test_subprocess_succeeds_and_writes_json(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        self.assertIsNotNone(self.result)
        self.assertEqual(self.result["world_size"], 2)
        self.assertEqual(last_json_document(self.proc.stdout), self.result)

    def test_fsdp2_matches_single_process(self):
        r = self.result
        self.assertLess(r["fsdp_max_diff"], 1e-5)
        self.assertLess(r["fsdp_bf16_max_diff"], 5e-2)
        self.assertGreater(r["fsdp_bf16_max_diff"], r["fsdp_max_diff"])

    def test_each_rank_holds_about_half_the_parameters(self):
        self.assertGreaterEqual(self.result["local_numel_ratio"], 0.45)
        self.assertLessEqual(self.result["local_numel_ratio"], 0.55)

    def test_gloo_avg_ratio_is_documented_value(self):
        # 2.0 on torch 2.8 (gloo reduce_scatter ignores AVG); 1.0 once fixed upstream.
        ratio = self.result["gloo_avg_grad_ratio"]
        self.assertTrue(abs(ratio - 2.0) < 1e-4 or abs(ratio - 1.0) < 1e-4, ratio)

    def test_tensor_parallel_matches_reference(self):
        self.assertLess(self.result["tp_out_diff"], 1e-5)
        self.assertLess(self.result["tp_grad_diff"], 1e-5)

    def test_redistribute_tour(self):
        ok = self.result["redistribute_ok"]
        self.assertGreaterEqual(len(ok), 5)
        self.assertTrue(all(ok.values()), ok)

    def test_pipeline_losses_match_reference(self):
        self.assertEqual(self.result["pp_n_losses"], 4)
        self.assertLess(self.result["pp_loss_diff"], 1e-5)

    def test_dcp_roundtrip_and_reshard(self):
        self.assertLess(self.result["dcp_roundtrip_diff"], 1e-7)
        self.assertLess(self.result["dcp_reshard_diff"], 1e-7)
        self.assertEqual(self.result["consolidated_top_keys"], ["model", "optim"])


if __name__ == "__main__":
    unittest.main()
