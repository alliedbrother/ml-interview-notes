"""Contracts for the module 18 lab: five seeded bugs and the detectors that catch them."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
_STATE = {}


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("18_bug_hunt")

DESIGNATED = {
    "no_zero_grad": ("overfit_one_batch", "grad_accumulation"),
    "softmax_before_ce": ("init_loss", "logit_grad_scale"),
    "eval_dropout": ("eval_determinism",),
    "broadcast_loss": ("shape_contract",),
    "detached_layer": ("unused_params",),
}


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()
    _STATE["grad"] = torch.is_grad_enabled()


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch.set_grad_enabled(_STATE["grad"])


class BugHuntContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            cls.result = lab.main(["--seed", "0", "--bug", "all", "--json-out", str(path)])
            cls.from_disk = json.loads(path.read_text())
        cls.matrix = cls.result["matrix"]

    def test_matrix_is_json_serialisable_and_complete(self):
        self.assertEqual(self.result, self.from_disk)
        self.assertEqual(set(self.matrix), {"none", *lab.BUGS})
        for row in self.matrix.values():
            self.assertEqual(set(row), set(lab.DETECTORS))
            self.assertTrue(all(isinstance(v, bool) for v in row.values()))

    def test_clean_recipe_trips_no_detector_and_overfits(self):
        self.assertFalse(any(self.matrix["none"].values()), self.matrix["none"])
        self.assertLess(self.result["clean_overfit_loss"], 0.05)

    def test_each_bug_trips_a_designated_detector(self):
        for bug, detectors in DESIGNATED.items():
            with self.subTest(bug=bug):
                fired = [d for d in detectors if self.matrix[bug][d]]
                self.assertTrue(fired, f"{bug}: none of {detectors} fired; row={self.matrix[bug]}")

    def test_detector_evidence_is_specific(self):
        details = self.result["details"]
        self.assertEqual(details["detached_layer"]["unused_params"]["unused"],
                         ["fc1.weight", "fc1.bias"])
        self.assertEqual(details["broadcast_loss"]["shape_contract"]["per_example_shape"],
                         [lab.B, lab.B])
        self.assertAlmostEqual(details["no_zero_grad"]["grad_accumulation"]["ratio"], 2.0, places=3)
        self.assertLess(details["softmax_before_ce"]["logit_grad_scale"]["ratio"], 0.25)

    def test_init_loss_is_near_ln_c_for_clean_model(self):
        init = self.result["details"]["none"]["init_loss"]
        self.assertLess(abs(init["loss"] - init["expected"]), 0.1)

    def test_find_unused_params_helper(self):
        model = lab.MLP("detached_layer")
        model(torch.randn(2, lab.D_IN)).sum().backward()
        self.assertEqual(lab.find_unused_params(model), ["fc1.weight", "fc1.bias"])

    def test_single_bug_cli_and_unknown_bug(self):
        single = lab.main(["--bug", "eval_dropout"])
        self.assertEqual(list(single["matrix"]), ["eval_dropout"])
        self.assertIsNone(single["clean_overfit_loss"])
        with self.assertRaises(ValueError):
            lab.Recipe("typo")


if __name__ == "__main__":
    unittest.main()
