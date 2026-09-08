"""Executable references for inference Parts 04-05, using only Python 3.11+.

Run: python quant_parallel_checks.py
These check mathematical contracts, not GPU kernels or serving performance.
"""

import itertools
import math
import random
import unittest


def matmul(a, b):
    if not a or not b or any(len(row) != len(b) for row in a):
        raise ValueError("Incompatible nonempty matrices")
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def reconstruction_error(w, q, x):
    residual = matmul([[a - b for a, b in zip(w, q)]], x)[0]
    return sum(value * value for value in residual)


def quantized_bytes(parameters, group=128, bits=4, scale_bytes=2, zero_bits=4):
    if parameters < 0 or group <= 0 or bits <= 0:
        raise ValueError("Invalid representation")
    groups = math.ceil(parameters / group)
    return math.ceil(parameters * bits / 8) + groups * scale_bytes + math.ceil(groups * zero_bits / 8)


def pack_nibbles(values):
    if len(values) != 8 or any(not isinstance(x, int) or not 0 <= x < 16 for x in values):
        raise ValueError("Exactly eight unsigned four-bit values required")
    return sum(value << (4 * i) for i, value in enumerate(values))


def unpack_nibbles(word):
    return [(word >> (4 * i)) & 15 for i in range(8)]


def expected_remote(experts, ranks, top_k):
    if experts < 1 or ranks < 1 or experts % ranks or not 1 <= top_k <= experts:
        raise ValueError("Even placement and valid distinct top-k required")
    available = experts - experts // ranks
    miss = math.comb(available, top_k) / math.comb(experts, top_k) if available >= top_k else 0
    return (ranks - 1) * (1 - miss)


def paired_correctness(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("At least two matched examples required")
    differences = [int(y) - int(x) for x, y in zip(a, b)]
    n = len(differences)
    delta = sum(differences) / n
    discordance = sum(d != 0 for d in differences) / n
    se = math.sqrt((discordance - delta * delta) / n)
    return dict(delta=delta, discordance=discordance, se=se,
                normal_ci=(delta - 1.96 * se, delta + 1.96 * se),
                improved=differences.count(1), regressed=differences.count(-1))


class QuantizationChecks(unittest.TestCase):
    def test_byte_ledger(self):
        # Per-tensor group rounding matters; headers and unquantized tensors remain.
        layers = [129, 127]
        separately = sum(quantized_bytes(n) for n in layers)
        pooled = quantized_bytes(sum(layers))
        self.assertGreater(separately, pooled)
        unquantized_head = 64 * 2
        logical = separately + unquantized_head
        allocated = sum(math.ceil(quantized_bytes(n) / 256) * 256 for n in layers) + 256
        self.assertGreater(allocated, logical)
        print(f"Byte ledger: packed+metadata={separately}, with head={logical}, aligned={allocated}")

    def test_own_ridge_is_not_pairwise_crossing(self):
        bf16_flops, bandwidth = 989.4e12, 3.35e12
        own_w4 = bf16_flops / (4 * bandwidth)
        pair_w4_w8 = bf16_flops / (2 * bandwidth)
        own_w8 = bf16_flops / bandwidth
        def time(batch, bytes_per_weight, flops):
            return max(bytes_per_weight / bandwidth, 2 * batch / flops)
        self.assertLess(time(100, .5, bf16_flops), time(100, 1, 2 * bf16_flops))
        self.assertAlmostEqual(time(pair_w4_w8, .5, bf16_flops), time(pair_w4_w8, 1, 2 * bf16_flops))
        print(f"Ideal thresholds: W4 ridge={own_w4:.2f}, W4/W8={pair_w4_w8:.2f}, W8 ridge={own_w8:.2f}")

    def test_gptq_coordinate_correction(self):
        x = [[1., 1., 0.], [1., 0., 1.]]
        w = [.7, -.2]
        h = [[4., 2.], [2., 4.]]  # 2 X X^T, output-by-input W convention.
        inverse = [[1 / 3, -1 / 6], [-1 / 6, 1 / 3]]
        q0 = round(w[0])
        error = w[0] - q0
        corrected = w[1] - error * inverse[0][1] / inverse[0][0]
        naive_loss = reconstruction_error(w, [q0, w[1]], x)
        corrected_loss = reconstruction_error(w, [q0, corrected], x)
        self.assertLess(corrected_loss, naive_loss)
        self.assertAlmostEqual(corrected_loss, error ** 2 / (2 * inverse[0][0]))
        # Upper Cholesky R with R^T R = H^-1 gives the same first update.
        r00 = math.sqrt(inverse[0][0])
        r01 = inverse[0][1] / r00
        self.assertAlmostEqual(w[1] - error / r00 * r01, corrected)
        choices = list(itertools.product((-1, 0, 1), repeat=2))
        optimum = min(choices, key=lambda q: reconstruction_error(w, q, x))
        self.assertEqual(optimum, (1, 0))
        damped = [[h[i][j] + (.04 if i == j else 0) for j in range(2)] for i in range(2)]
        self.assertGreater(damped[0][0] * damped[1][1] - damped[0][1] ** 2, 0)

    def test_awq_smoothquant_identity_and_group_tradeoff(self):
        x, w, s = [[2., 8.]], [[.2], [1.]], [1., 4.]
        actual = matmul([[x[0][j] / s[j] for j in range(2)]], [[w[j][0] * s[j]] for j in range(2)])
        self.assertAlmostEqual(actual[0][0], matmul(x, w)[0][0])
        def shared_quant(values):
            scale = max(abs(v) for v in values) / 7
            return [round(v / scale) * scale for v in values]
        original = [.2, 1.]
        base = shared_quant(original)
        scaled = shared_quant([original[i] * s[i] for i in range(2)])
        restored = [scaled[i] / s[i] for i in range(2)]
        self.assertGreater(abs(restored[0] - original[0]), abs(base[0] - original[0]))

    def test_all_nibbles_all_positions(self):
        for position in range(8):
            for value in range(16):
                values = [0] * 8
                values[position] = value
                self.assertEqual(unpack_nibbles(pack_nibbles(values)), values)
        for zero in (0, 7, 15):
            for scale in (1e-6, .5, 1000):
                values = list(range(8))
                restored = [(v - zero) * scale for v in unpack_nibbles(pack_nibbles(values))]
                self.assertEqual(restored, [(v - zero) * scale for v in values])

    def test_paired_not_independent_accuracy(self):
        # B improves on 33 examples and regresses on 7: 26/1319 improvement.
        a = [False] * 33 + [True] * 7 + [True] * 1279
        b = [True] * 33 + [False] * 7 + [True] * 1279
        result = paired_correctness(a, b)
        self.assertGreater(result['normal_ci'][0], 0)
        self.assertEqual((result['improved'], result['regressed']), (33, 7))
        print(f"Paired accuracy: {result}")


class ParallelismChecks(unittest.TestCase):
    def test_column_row_reference_and_bias(self):
        x = [[1., 2.], [-1., 3.]]
        up = [[1., 2., 3., 4.], [2., 1., -1., 0.]]
        down = [[1., 2.], [2., 1.], [-1., 2.], [3., 0.]]
        bias = [.5, -.5]
        reference = matmul(matmul(x, up), down)
        partials = [matmul(matmul(x, [row[start:start + 2] for row in up]), down[start:start + 2])
                    for start in (0, 2)]
        reduced = [[sum(p[i][j] for p in partials) + bias[j] for j in range(2)] for i in range(2)]
        self.assertEqual(reduced, [[v + bias[j] for j, v in enumerate(row)] for row in reference])
        # Bias belongs after reduction or on exactly one contribution, not every rank.
        wrong = reduced[0][0] + bias[0]
        self.assertNotEqual(wrong, reduced[0][0])

    def test_pipeline_fixed_batch_baseline(self):
        stages, microbatches, stage_time = 4, 2, 3.
        finished = [[0.] * stages for _ in range(microbatches)]
        for batch in range(microbatches):
            for stage in range(stages):
                previous_stage = finished[batch][stage - 1] if stage else 0
                previous_batch = finished[batch - 1][stage] if batch else 0
                finished[batch][stage] = max(previous_stage, previous_batch) + stage_time
        self.assertEqual(finished[-1][-1] / (stages * stage_time), 5 / 4)

    def test_exact_distinct_remote_routing(self):
        experts, ranks, top_k = 8, 4, 2
        destinations = [{e // (experts // ranks) for e in chosen} - {0}
                        for chosen in itertools.combinations(range(experts), top_k)]
        self.assertAlmostEqual(sum(map(len, destinations)) / len(destinations), expected_remote(experts, ranks, top_k))
        self.assertAlmostEqual(expected_remote(256, 8, 8), 4.632820500334669)
        rng = random.Random(23)
        observed = sum(len({e // 32 for e in rng.sample(range(256), 8)} - {0}) for _ in range(20000)) / 20000
        self.assertLess(abs(observed - expected_remote(256, 8, 8)), .04)
        self.assertEqual(256 * (1 - (1 - 8 / 256) ** 1), 8)

    def test_uneven_idle_scatter_and_weighted_combine(self):
        ranks = [[('a', 2.), ('b', 3.)], [], [('c', 5.)]]
        flat = [item for rank in ranks for item in rank]
        restored, offset = [], 0
        for rank in ranks:
            restored.append(flat[offset:offset + len(rank)])
            offset += len(rank)
        self.assertEqual(restored, ranks)
        # Two experts on one destination require one transfer but two contributions.
        token, choices, coefficients = 3., [0, 1], [.25, .75]
        expert_outputs = [(e + 1) * token for e in choices]
        self.assertEqual(sum(w * v for w, v in zip(coefficients, expert_outputs)), 5.25)
        self.assertEqual(len({e // 2 for e in choices}), 1)

    def test_same_shape_semantic_divergence(self):
        plan_a, plan_b = ['request-a', 'request-b'], ['request-b', 'request-a']
        values_a, values_b = [1., 10.], [100., 2.]
        self.assertEqual(len(values_a), len(values_b))
        wrong = [a + b for a, b in zip(values_a, values_b)]
        mapped = dict(zip(plan_b, values_b))
        correct = [value + mapped[key] for key, value in zip(plan_a, values_a)]
        self.assertNotEqual(wrong, correct)
        self.assertNotEqual(tuple(plan_a), tuple(plan_b))

    def test_traffic_and_capacity_conventions(self):
        ranks, size = 8, 128 * 1024
        sent = 2 * (ranks - 1) / ranks * size
        self.assertAlmostEqual(sent / 450e9, (2 * sent) / 900e9)
        self.assertEqual(2 * size / ranks, 32768)
        pool = 40 * 2 ** 30
        self.assertEqual(pool / (40 * 1024), 8 * pool / (320 * 1024))
        traffic_ratio = ((7 / 8) * 32768) / (2 * (7 / 8) * 4103)
        self.assertAlmostEqual(traffic_ratio, 3.993, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
