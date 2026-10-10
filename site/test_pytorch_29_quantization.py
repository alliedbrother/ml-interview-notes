"""Contracts for the module 29 lab: weight-only, static PTQ, QAT and 2:4 sparsity."""

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


lab = load("29_quantization")


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()
    _STATE["engine"] = torch.backends.quantized.engine


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])
    torch.backends.quantized.engine = _STATE["engine"]


class QuantizationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            cls.result = lab.main(["--seed", "0", "--json-out", str(path)])
            cls.from_disk = json.loads(path.read_text())

    def test_result_is_json_round_trippable(self):
        self.assertEqual(self.result, self.from_disk)

    def test_task_is_learnable(self):
        self.assertGreater(self.result["fp32"]["acc"], 0.85)

    def test_byte_accounting(self):
        fp32 = self.result["fp32"]["bytes"]
        self.assertLess(self.result["int8_tensor"]["bytes"], 0.3 * fp32)
        self.assertLess(self.result["int8_channel"]["bytes"], 0.3 * fp32)
        self.assertGreater(self.result["int8_channel"]["bytes"], 0.25 * fp32)  # codes alone are fp32/4
        self.assertLess(self.result["int4_group32"]["bytes"], 0.2 * fp32)
        self.assertLess(self.result["int4_group32"]["bytes"], self.result["int8_channel"]["bytes"])

    def test_per_channel_error_not_worse_than_per_tensor(self):
        self.assertLessEqual(self.result["int8_channel"]["logit_mse"],
                             self.result["int8_tensor"]["logit_mse"])

    def test_accuracy_drops_are_bounded(self):
        fp32 = self.result["fp32"]["acc"]
        self.assertLessEqual(fp32 - self.result["int8_channel"]["acc"], 0.02)
        self.assertLessEqual(fp32 - self.result["qat_int8"]["acc"], 0.03)
        self.assertGreaterEqual(self.result["qat_int8"]["acc"],
                                self.result["ptq_static_int8"]["acc"] - 0.01)

    def test_2_4_pattern_is_exact(self):
        sparse = self.result["sparse_2_4"]
        self.assertEqual(sparse["zero_fraction"], 0.5)
        self.assertEqual(sparse["group_zero_counts"], [2])  # every group of 4 has 2 zeros

    def test_ao_dynamic_skipped_by_default(self):
        self.assertIn("skipped", self.result["ao_dynamic"])


class Int4Packing(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        g = torch.Generator().manual_seed(1)
        q = torch.randint(-8, 8, (6, 32), generator=g, dtype=torch.int8)
        packed = lab.pack_int4(q.reshape(-1))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), q.numel() // 2)
        self.assertTrue(torch.equal(lab.unpack_int4(packed, q.shape), q))


class AoDynamic(unittest.TestCase):
    def test_ao_dynamic_runs_or_skips_and_restores_engine(self):
        before = torch.backends.quantized.engine
        result = lab.main(["--epochs", "2", "--finetune-epochs", "1", "--with-ao-dynamic"])
        self.assertEqual(torch.backends.quantized.engine, before)
        ao = result["ao_dynamic"]
        if "skipped" in ao:
            self.assertIsInstance(ao["skipped"], str)
        else:
            self.assertIn(ao["engine"], torch.backends.quantized.supported_engines)
            self.assertLess(ao["bytes"], 0.3 * result["fp32"]["bytes"])
            self.assertLessEqual(result["fp32"]["acc"] - ao["acc"], 0.05)


if __name__ == "__main__":
    unittest.main()
