"""CPU regression oracles for inference chapters 11-13, not runtime certification."""

from pathlib import Path
import unittest

from bs4 import BeautifulSoup
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "content/courses/inference/html"


def published_code(relative_path, section_id):
    soup = BeautifulSoup((LESSONS / relative_path).read_text(), "html.parser")
    return soup.find(id=section_id).find("pre").get_text()


class EngineDeepDiveCorrections(unittest.TestCase):
    def test_published_indexed_update(self):
        source = published_code(
            "11-vllm-deep-dive/11-05-vllm-extension-points.html",
            "indexed-logits-check",
        )
        exec(compile(source, "published indexed logits example", "exec"), {})

    def test_published_ragged_layout(self):
        source = published_code(
            "11-vllm-deep-dive/11-04-model-runner-and-backends.html",
            "ragged-layout-oracle",
        )
        exec(compile(source, "published ragged layout example", "exec"), {})

    def test_parameter_units_and_streamed_subset(self):
        resident = 8.03e9 * 2
        self.assertAlmostEqual(resident / 1e9, 16.06)
        self.assertAlmostEqual(resident / 2**30, 14.957040548324585)
        self.assertAlmostEqual(resident / 3.35e12 * 1000, 4.794029850746269)
        self.assertAlmostEqual(15e9 / 3.35e12 * 1000, 4.477611940298507)

    def test_logprob_payload_counts(self):
        width = 5 + 1
        for index_dtype, expected in [(np.int32, 52), (np.int64, 80)]:
            arrays = [
                np.zeros((1, width), dtype=np.float32),
                np.zeros((1, width), dtype=index_dtype),
                np.zeros((1,), dtype=index_dtype),
            ]
            self.assertEqual(sum(a.nbytes for a in arrays), expected)
            self.assertTrue(all(a.nbytes < 256 for a in arrays))
            self.assertEqual(64 * sum(a.nbytes for a in arrays), 64 * expected)
        # Encoding one batched tensor is a different threshold decision.
        self.assertGreater(np.zeros((64, width), dtype=np.float32).nbytes, 256)

    def test_prefill_first_output_and_computed_cache(self):
        prompt, cached, outputs = 2144, 2048, 96
        extend = prompt - cached
        decode_forwards = outputs - 1
        self.assertEqual(extend, 96)
        self.assertEqual(decode_forwards, 95)
        computed = cached + extend + decode_forwards
        self.assertEqual(computed, 2239)
        self.assertEqual(computed - cached, 191)
        self.assertEqual(computed, prompt + outputs - 1)
        self.assertEqual(257 * 32768 * 4 / 2**20, 32.125)
        for page_size in (1, 16):
            table_bytes = 257 * 32768 * 4
            self.assertEqual(table_bytes, 33685504)
            self.assertEqual((4104 // page_size) * page_size, 4104 if page_size == 1 else 4096)

    def test_amdahl_and_preemption_assumptions(self):
        speedup = lambda removable: 1 / (1 - removable)
        self.assertAlmostEqual(speedup(0.2), 1.25)
        self.assertAlmostEqual(speedup(0.4), 5 / 3)
        for x in np.linspace(0, 0.2, 30):
            self.assertLessEqual(speedup(x), 1.25)
        recompute = lambda f: 372 * (1 - f)
        self.assertAlmostEqual(recompute(0) / recompute(0.99), 100)
        self.assertAlmostEqual(recompute(0.5) / recompute(0.5), 1)
        # A graph coverage limit and scheduler admission limit are separate gates.
        admitted, graph_max = 32, 24
        self.assertLessEqual(admitted, 256)
        self.assertFalse(admitted <= graph_max)

    def test_logical_consensus_not_physical_identity(self):
        logical = [("a", 5), ("b", 7)]
        local_maps = [{"a": 3, "b": 8}, {"a": 9, "b": 2}]
        memories = [{3: 11, 8: 17}, {9: 11, 2: 17}]
        outputs = [
            [memory[mapping[request]] for request, _ in logical]
            for mapping, memory in zip(local_maps, memories)
        ]
        self.assertNotEqual(local_maps[0], local_maps[1])
        self.assertEqual(outputs[0], outputs[1])
        # Equal shapes do not imply equal logical row order or correct reduction.
        corrupted = np.array(outputs[0]) + np.array(outputs[1][::-1])
        correct = 2 * np.array(outputs[0])
        self.assertEqual(corrupted.shape, correct.shape)
        self.assertFalse(np.array_equal(corrupted, correct))

    def test_common_mode_checker_needs_independent_oracle(self):
        node = {"resident": True, "locked": True, "resident_child": False}
        bad_predicate = lambda n: n["resident"] and not n["resident_child"]
        maintained = {"node"} if bad_predicate(node) else set()
        recomputed = {"node"} if bad_predicate(node) else set()
        self.assertEqual(maintained, recomputed)
        independent = (
            {"node"}
            if node["resident"] and not node["locked"] and not node["resident_child"]
            else set()
        )
        self.assertNotEqual(maintained, independent)
        allocated = [7, 8]
        freed = [7, 7]
        self.assertEqual(len(allocated), len(freed))
        self.assertNotEqual(sorted(allocated), sorted(freed))

    def test_margin_required_for_exact_greedy_comparison(self):
        logits = np.array([1.0, 1.00001, 0.0])
        perturbed = logits + np.array([0.00002, -0.00002, 0.0])
        self.assertNotEqual(logits.argmax(), perturbed.argmax())
        error = np.max(np.abs(logits - perturbed))
        margin = np.sort(logits)[-1] - np.sort(logits)[-2]
        self.assertLess(margin, 2 * error)
        stable = np.array([2.0, 1.0, 0.0])
        self.assertGreater(1.0, 2 * error)
        self.assertEqual(stable.argmax(), (stable + perturbed - logits).argmax())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
