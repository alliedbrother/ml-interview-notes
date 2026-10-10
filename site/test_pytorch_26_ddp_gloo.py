"""Contracts for the module 26 lab: two gloo CPU processes prove DDP equals large-batch training.

The lab spawns processes, so it runs in a subprocess (never mp.spawn from the
unittest process: spawn cannot re-import digit-named modules).
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
SCRIPT = CODE / "26_ddp_gloo.py"
_STATE = {}

EXPECTED_COLLECTIVES = {
    "all_reduce_sum", "all_reduce_avg", "all_reduce_max", "all_gather_into_tensor",
    "reduce_scatter_tensor", "broadcast", "all_to_all_single", "isend_irecv_ring",
}


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


class DDPGlooContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            cls.proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--world-size", "2", "--steps", "5",
                 "--seed", "0", "--json-out", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            cls.result = json.loads(out.read_text()) if out.is_file() else None

    def test_process_exits_cleanly_and_writes_json(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        self.assertIsNotNone(self.result)
        self.assertEqual(self.result["world_size"], 2)
        self.assertEqual(json.loads(self.proc.stdout), self.result)

    def test_every_collective_matches_its_closed_form(self):
        ok = self.result["collectives_ok"]
        self.assertEqual(set(ok), EXPECTED_COLLECTIVES)
        self.assertTrue(all(ok.values()), ok)

    def test_three_ddp_variants_equal_the_full_batch_reference(self):
        for key in ("max_diff_naive_vs_ref", "max_diff_ddp_vs_ref", "max_diff_nosync_vs_ref"):
            self.assertLess(self.result[key], 1e-6, key)

    def test_negative_control_shows_the_comparison_has_teeth(self):
        self.assertGreater(self.result["max_diff_local_only_vs_ref"], 1e-3)

    def test_replicas_stay_bit_identical(self):
        self.assertIs(self.result["params_identical_across_ranks"], True)

    def test_ddp_built_buckets_and_timing_is_reported(self):
        self.assertGreaterEqual(self.result["bucket_count"], 1)
        self.assertGreater(self.result["allreduce_4mb_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
