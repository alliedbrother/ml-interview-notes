"""CPU regression checks for the audited inference foundations arithmetic."""

from pathlib import Path
import math
import unittest

from bs4 import BeautifulSoup
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "content/courses/inference/html"


def online_attention(logits, values, block_size):
    maximum, total = -np.inf, 0.0
    numerator = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(logits), block_size):
        x = logits[start:start + block_size]
        v = values[start:start + block_size]
        valid = np.isfinite(x)
        if not valid.any():
            continue
        new_maximum = max(maximum, x[valid].max())
        scale = np.exp(maximum - new_maximum) if total else 0.0
        weights = np.exp(x - new_maximum)
        total = scale * total + weights.sum()
        numerator = scale * numerator + weights @ v
        maximum = new_maximum
    return numerator / total if total else numerator


def rope(x, position, rotary_dim, neox):
    out = x.copy()
    angle = position * 10000.0 ** (-np.arange(0, rotary_dim, 2) / rotary_dim)
    if neox:
        a, b = x[:rotary_dim // 2], x[rotary_dim // 2:rotary_dim]
        out[:rotary_dim // 2] = a * np.cos(angle) - b * np.sin(angle)
        out[rotary_dim // 2:rotary_dim] = a * np.sin(angle) + b * np.cos(angle)
    else:
        a, b = x[:rotary_dim:2], x[1:rotary_dim:2]
        out[:rotary_dim:2] = a * np.cos(angle) - b * np.sin(angle)
        out[1:rotary_dim:2] = a * np.sin(angle) + b * np.cos(angle)
    return out


class InferenceFoundationsCorrections(unittest.TestCase):
    def test_published_online_recurrence(self):
        page = ROOT / "03-attention-kernels/03-01-naive-attention-and-online-softmax.html"
        soup = BeautifulSoup(page.read_text(), "html.parser")
        code = next(c.get_text() for c in soup.select("main pre code")
                    if "N, D, BC = 8192" in c.get_text())
        self.assertIn("m = m_new", code)
        self.assertIn("assert_allclose", code)
        exec(compile(code, str(page), "exec"), {})

    def test_online_masks_order_and_late_maxima(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=63)
        x[48] = 40
        x[:7] = -np.inf
        v = rng.normal(size=(63, 9))
        v[::2] *= -100
        weights = np.exp(x - np.max(x))
        expected = weights @ v / weights.sum()
        for width in (1, 7, 16, 64):
            np.testing.assert_allclose(online_attention(x, v, width), expected,
                                       rtol=1e-12, atol=1e-12)
            order = rng.permutation(len(x))
            np.testing.assert_allclose(online_attention(x[order], v[order], width),
                                       expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(online_attention(np.full(63, -np.inf), v, 7),
                                      np.zeros(9))
        np.testing.assert_allclose(online_attention(np.zeros(63), v, 7), v.mean(axis=0))

    def test_rope_relative_identity_both_layouts_partial(self):
        rng = np.random.default_rng(9)
        q, k = rng.normal(size=(2, 12))
        for neox in (False, True):
            for r in (8, 12):
                a = rope(q, 5, r, neox)
                np.testing.assert_allclose(np.linalg.norm(a), np.linalg.norm(q))
                self.assertAlmostEqual(a @ rope(k, 3, r, neox),
                                       rope(q, 105, r, neox) @ rope(k, 103, r, neox))
                np.testing.assert_array_equal(a[r:], q[r:])
                self.assertNotAlmostEqual(a @ rope(k, 3, r, neox),
                                          a @ rope(k, 103, r, neox))
        mu = 1 + .1 * np.log(8)
        self.assertAlmostEqual(1 / mu**2, .6853403266295983)

    def test_roofline_general_dtype_and_kernel_composition(self):
        for width in (1, 2, 4):
            d, ridge = 4096, 295
            tokens = width * ridge * d / (2 * (d - width * ridge))
            intensity = 2 * tokens * d / (width * (2 * tokens + d))
            self.assertAlmostEqual(intensity, ridge)
        per_kernel = max(3, 1) + max(1, 4)
        aggregate = max(3 + 1, 1 + 4)
        self.assertEqual(per_kernel, 7)
        self.assertGreater(per_kernel, aggregate)

    def test_queue_goodput_and_percentile_mixture(self):
        mean_service, arrival = .1, 9.5
        rate = 1 / mean_service - arrival
        self.assertAlmostEqual(-math.log(.01) / rate, 9.210340371976182)
        self.assertAlmostEqual(arrival * (1 - math.exp(-rate * .9)), 3.44253256)
        a = np.r_[np.full(990, 800), np.full(10, 10000)]
        b = np.r_[np.full(990, 1200), np.full(10, 20000)]
        quantile = lambda z: np.quantile(z, .99, method="inverted_cdf")
        self.assertLessEqual(quantile(np.r_[a, b]), max(quantile(a), quantile(b)))
        gaps = [np.array([10.0]), np.full(99, 100.0)]
        self.assertEqual(np.mean([x.mean() for x in gaps]), 55)
        self.assertAlmostEqual(np.concatenate(gaps).mean(), 99.1)

    def test_chunk_alignment_and_decode_count(self):
        self.assertEqual(math.ceil(32768 / (2048 - 32)), 17)
        self.assertEqual(32768 - 16 * 2016, 512)
        self.assertEqual(math.ceil(8192 / (2048 - 64)), 5)
        ceiling = 500e12 * (.025 - .0095) / 1.606e10
        aligned = math.floor(ceiling / 16) * 16
        self.assertEqual(aligned, 480)
        self.assertLessEqual(aligned, ceiling)
        self.assertGreater(512, ceiling)
        self.assertEqual(1 + 511, 512)
        np.testing.assert_array_equal(np.r_[0, np.cumsum([2, 5, 3])], [0, 2, 7, 10])

    def test_cache_capacity_and_inclusive_residency(self):
        self.assertAlmostEqual(1 - 612 / 32768, .9813232421875)
        self.assertEqual(16 * 128 * 1024, 2 * 1024**2)
        pool1, pool2 = .92 * 79.65 - 14.96 - 6, .92 * 79.65 - 7.48 - 6
        self.assertGreater((pool2 / 64) / (pool1 / 128), 2)
        self.assertAlmostEqual(2 * 445632 / 2000, 445.632)
        self.assertNotAlmostEqual(2 * 445632 / 2000, 668.448)
        self.assertEqual(2992 + 12 * math.ceil(165 / 16) * 16, 5104)

    def test_quantization_savings_and_rotation_counterexample(self):
        for ratio, expected in ((.01, .004950495), (.5, 1 / 6), (1, .25)):
            self.assertAlmostEqual(ratio / (2 * (1 + ratio)), expected)
        h = np.ones((1, 1))
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        h /= np.sqrt(128)
        x = h[0]
        self.assertAlmostEqual(np.max(np.abs(h @ x)), 1)
        self.assertGreater(np.max(np.abs(h @ x)), 3.5 / np.sqrt(128))
        np.testing.assert_allclose(np.linalg.norm(h @ x), np.linalg.norm(x))

    def test_float_rounding_boundaries_and_zero_scale(self):
        minimum = np.nextafter(np.float16(0), np.float16(1), dtype=np.float16)
        self.assertEqual(np.float16(float(minimum) / 2), 0)
        self.assertEqual(np.float16(float(minimum) * .75), minimum)
        self.assertAlmostEqual(np.log(float(minimum) / 2), -17.328679513998633)
        x = np.zeros(16)
        scale = max(np.max(np.abs(x)) / 127, 1e-12)
        np.testing.assert_array_equal(np.round(x / scale), x)
        self.assertFalse(np.isfinite(np.array([np.nan, np.inf])).all())

    def test_mla_partial_head_accounting(self):
        partial = 2 * 128 * 16 * (512 + 2) * 4
        kv = 576 * 4096 * 2
        self.assertEqual(partial, 8421376)
        self.assertAlmostEqual(partial / kv, 1.7847222222222223)
        self.assertAlmostEqual(partial / (.1 * 576 * 2), 73102.22222222222)


if __name__ == "__main__":
    unittest.main()
