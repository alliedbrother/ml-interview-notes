"""CPU reference checks for inference arithmetic; no engine, GPU or network.

Run with Python 3.11+. If downloaded alone, core checks still run; adjacent lab
parser checks require the lab directories. Schema tests additionally require
09-structured-decoding-overhead/requirements.txt. No result is a GPU measurement.
"""

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import json
import contextlib
import io
from unittest.mock import patch


def square_gemm_crossing(width, element_bytes, ridge):
    if min(width, element_bytes, ridge) <= 0:
        raise ValueError("Positive dimensions, element bytes and ridge required")
    denominator = 2 * (width - element_bytes * ridge)
    return element_bytes * ridge * width / denominator if denominator > 0 else math.inf


def kv_saving(traffic_ratio, compression=2):
    if traffic_ratio < 0 or compression < 1:
        raise ValueError("Nonnegative traffic ratio and compression >= 1 required")
    return traffic_ratio * (1 - 1 / compression) / (1 + traffic_ratio)


def expected_yield(acceptance, drafts):
    if not 0 <= acceptance <= 1 or not isinstance(drafts, int) or drafts < 0:
        raise ValueError("Acceptance in [0,1] and nonnegative integer draft count required")
    return sum(acceptance ** i for i in range(drafts + 1))


def kv_cell(layers, kv_heads, key_width, value_width, element_bytes, tp=1):
    values = (layers, kv_heads, key_width, value_width, element_bytes, tp)
    if any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Positive integer shape and byte counts required")
    if max(kv_heads, tp) % min(kv_heads, tp):
        raise ValueError("This reference supports evenly sharded or replicated heads")
    return layers * max(kv_heads // tp, 1) * (key_width + value_width) * element_bytes


def token_capacity(total_bytes, utilization, weights, overhead, cell, block=16, reserved=1):
    if not 0 < utilization <= 1 or min(total_bytes, cell, block) <= 0 or min(weights, overhead, reserved) < 0:
        raise ValueError("Invalid budget")
    pool = total_bytes * utilization - weights - overhead
    if pool < 0:
        raise ValueError("Weights and overhead exceed the budget")
    return max(math.floor(pool / (cell * block)) - reserved, 0) * block


def rejection_mixture(target, draft):
    if len(target) != len(draft) or not target:
        raise ValueError("Equal nonempty vocabularies required")
    for distribution in (target, draft):
        if any(x < 0 or not math.isfinite(x) for x in distribution) or not math.isclose(sum(distribution), 1):
            raise ValueError("Normalized finite probabilities required")
    accepted = [min(p, q) for p, q in zip(target, draft)]
    residual = [max(p - q, 0) for p, q in zip(target, draft)]
    # Conditional rejection output is the normalized residual, not the target.
    mass = sum(residual)
    conditional = [r / mass for r in residual] if mass else None
    mixed = [a + r for a, r in zip(accepted, residual)]
    return conditional, mixed


def request_metrics(arrival, token_times, completed=True):
    if not completed or not token_times:
        return None
    if arrival > token_times[0] or any(b < a for a, b in zip(token_times, token_times[1:])):
        raise ValueError("Monotonic timestamps required")
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    return dict(ttft=token_times[0] - arrival, e2e=token_times[-1] - arrival,
                tpot=sum(gaps) / len(gaps) if gaps else None, gaps=gaps)


def load_lab(directory):
    path = Path(__file__).parent / directory / "run.py"
    if not path.exists():
        raise unittest.SkipTest("Adjacent lab source is not present in this standalone download")
    name = "inference_lab_" + directory.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ArithmeticChecks(unittest.TestCase):
    def test_gemm_dtypes(self):
        for b in (1, 2, 4):
            t = square_gemm_crossing(4096, b, 295)
            self.assertAlmostEqual(2 * t * 4096 / (b * (2 * t + 4096)), 295)

    def test_no_finite_crossing(self):
        self.assertTrue(math.isinf(square_gemm_crossing(128, 2, 295)))
        self.assertTrue(math.isinf(square_gemm_crossing(590, 2, 295)))

    def test_kv_below_crossover(self):
        self.assertAlmostEqual(kv_saving(.5), 1 / 6)
        self.assertAlmostEqual(kv_saving(1), .25)
        self.assertEqual(kv_saving(0), 0)
        self.assertLess(kv_saving(.018), .01)

    def test_speculation_boundaries(self):
        self.assertEqual(expected_yield(1, 3), 4)
        self.assertEqual(expected_yield(0, 3), 1)
        self.assertEqual(expected_yield(.7, 0), 1)
        self.assertAlmostEqual(expected_yield(.7, 3), 2.533)
        self.assertAlmostEqual(295 * expected_yield(.7, 3) / 4, 186.80875)

    def test_speculation_invalid(self):
        for alpha, count in ((-.1, 2), (1.1, 2), (.5, -1), (.5, 1.5)):
            with self.assertRaises(ValueError):
                expected_yield(alpha, count)

    def test_rejection_conditioning(self):
        conditional, output = rejection_mixture([.8, .2], [.2, .8])
        self.assertEqual(conditional, [1, 0])
        for got, want in zip(output, [.8, .2]):
            self.assertAlmostEqual(got, want)

    def test_rejection_identical_and_disjoint(self):
        self.assertEqual(rejection_mixture([.5, .5], [.5, .5]), (None, [.5, .5]))
        self.assertEqual(rejection_mixture([1, 0], [0, 1]), ([1, 0], [1, 0]))

    def test_cache_shapes(self):
        self.assertEqual(kv_cell(32, 8, 128, 128, 2), 131072)
        self.assertEqual(kv_cell(32, 8, 128, 128, 2, 2), 65536)
        self.assertEqual(kv_cell(32, 8, 128, 128, 2, 8), kv_cell(32, 8, 128, 128, 2, 16))
        self.assertEqual(kv_cell(61, 128, 192, 128, 2), 4997120)

    def test_capacity_both_weights_and_cells_shrink(self):
        gib = 1024 ** 3
        c1 = token_capacity(79.65 * gib, .92, 14.96 * gib, 6 * gib, 131072)
        c2 = token_capacity(79.65 * gib, .92, 7.48 * gib, 6 * gib, 65536)
        self.assertGreater(c2, 2 * c1)
        self.assertEqual(c1 % 16, 0)
        with self.assertRaises(ValueError):
            token_capacity(10, .9, 10, 1, 1)

    def test_static_utilization(self):
        useful = 1 / sum(1 / i for i in range(1, 33))
        self.assertAlmostEqual(useful, .24639674, places=6)
        self.assertAlmostEqual(1 - useful, .75360326, places=6)

    def test_prefix_cold_and_rounding(self):
        warm = 1000 / 1200
        cold = 1000 / (64 * 1200)
        rounding = 63 * 8 / (64 * 1200)
        actual = 63 * 992 / (64 * 1200)
        self.assertAlmostEqual(warm - actual, cold + rounding)

    def test_timestamp_weighting(self):
        a = request_metrics(0, [1, 2])
        b = request_metrics(0, [1, 4, 7, 10])
        request_mean = (a['tpot'] + b['tpot']) / 2
        pooled = sum(a['gaps'] + b['gaps']) / 4
        self.assertEqual(request_mean, 2)
        self.assertEqual(pooled, 2.5)
        self.assertIsNone(request_metrics(0, [1])['tpot'])
        self.assertIsNone(request_metrics(0, [], False))


class LabRegressionChecks(unittest.TestCase):
    def test_schema_report_unconstrained_baseline(self):
        lab = load_lab("09-structured-decoding-overhead")
        args = SimpleNamespace(model='synthetic-model', engine='vllm', base_url='http://unused',
                               requests=1, output_len=3, unique_schemas=False,
                               concurrency=[1], schema=['none', 'simple'], save=None)
        common = dict(concurrency=1, req_per_s=2., out_tok_per_s=6., ttft_p50_ms=1.,
                      ttft_p99_ms=1., e2e_mean_ms=2., valid_json_frac=1.)
        rows = [common | dict(schema='none', valid_schema_frac=None),
                common | dict(schema='simple', valid_schema_frac=.5)]
        output = io.StringIO()
        with patch.object(lab, 'run_arm', side_effect=rows), contextlib.redirect_stdout(output):
            self.assertEqual(lab.cmd_sweep(args), 0)
        self.assertIn('n/a', output.getvalue())
        self.assertIn('50%', output.getvalue())
        self.assertIn('failed its actual JSON Schema', output.getvalue())

    def test_prefix_module(self):
        lab = load_lab("04-prefix-cache-hit-rate")
        self.assertAlmostEqual(lab.workload_hit_rate(1000, 200, 64, 1, 16)['hit rate'], .81375)
        self.assertEqual(lab.matchable_tokens(1000, 1200, 16, break_at=3), 0)
        self.assertEqual(lab.matchable_tokens(16, 16, 16), 0)

    def test_tp_measurement_definitions(self):
        lab = load_lab("07-tp-scaling")
        args = SimpleNamespace(output_len=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            path.write_text(json.dumps({'avg_latency': 8}))
            self.assertEqual(lab.read_vllm(path, args)[0], 2)
            path.write_text(json.dumps({'median_decode_latency': .4}))
            self.assertEqual(lab.read_sglang(path, args)[0], .4)

    def test_trace_union(self):
        lab = load_lab("10-profile-a-decode-step")
        intervals = [(5, 10), (0, 6), (12, 14)]
        self.assertEqual(lab.union_us(intervals), 12)
        self.assertEqual(intervals[0], (5, 10))
        self.assertEqual(lab.union_us([]), 0)
        with self.assertRaises(ValueError):
            lab.union_us([(5, 1)])

    def test_float_order(self):
        lab = load_lab("13-batch-invariance")
        self.assertEqual(lab.ulp_gap(-0., 0.), 0)
        self.assertEqual(lab.ulp_gap(1., math.nextafter(1., math.inf)), 1)
        self.assertEqual(lab.ulp_gap(-1., math.nextafter(-1., 0)), 1)
        tiny = math.ulp(0.)
        self.assertEqual(lab.ulp_gap(-tiny, tiny), 2)
        self.assertEqual(lab._first_divergence([1, 2], [1, 2, 3]), 2)
        with self.assertRaises(ValueError):
            lab.ulp_gap(math.nan, 0.)

    def test_schema_not_just_json(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest('Install Lab 09 requirements for schema checks')
        lab = load_lab("09-structured-decoding-overhead")
        schema = {'type': 'object', 'required': ['n'], 'properties': {'n': {'type': 'integer'}}, 'additionalProperties': False}
        self.assertTrue(lab.valid_schema('{"n": 3}', schema))
        self.assertFalse(lab.valid_schema('{"n": "three"}', schema))
        self.assertFalse(lab.valid_schema('{}', schema))
        self.assertFalse(lab.valid_schema('{', schema))


if __name__ == '__main__':
    unittest.main(verbosity=2)
