"""Numerical contracts for the standalone spectral/update lab."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/notes/_examples/spectral_updates.py"
spec = importlib.util.spec_from_file_location("spectral_updates", SOURCE)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class SpectralUpdateTests(unittest.TestCase):
    def test_seeded_nonsymmetric_updates_multiple_rhs(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            a = rng.normal(size=(6, 6)) + 8 * np.eye(6)
            u, v = rng.normal(size=(2, 6, 2))
            c = .1 * rng.normal(size=(2, 2))
            rhs = rng.normal(size=(6, 3))
            answer, report = lab.low_rank_solve(a, u, c, v, rhs)
            np.testing.assert_allclose(answer, np.linalg.solve(a + u @ c @ v.T, rhs), atol=1e-12)
            self.assertLess(report["scaled_residual"], 1e-14)

    def test_vector_rhs_and_singular_c_are_supported(self):
        a = np.diag([2., 3., 4.])
        u = np.array([[1., 0.], [1., 1.], [0., 1.]])
        rhs = np.array([1., 2., 3.])
        for c in (np.zeros((2, 2)), np.diag([.5, 0.])):
            answer, _ = lab.low_rank_solve(a, u, c, u, rhs)
            self.assertEqual(answer.shape, (3,))
            np.testing.assert_allclose(answer, np.linalg.solve(a + u @ c @ u.T, rhs))

    def test_singular_base_and_updated_system_fail(self):
        u = np.array([[1.], [0.]])
        for a, c in ((np.zeros((2, 2)), np.ones((1, 1))), (np.eye(2), -np.ones((1, 1)))):
            with self.assertRaises(np.linalg.LinAlgError):
                lab.low_rank_solve(a, u, c, u, np.ones(2))

    def test_scalar_core_condition_does_not_certify_updated_system(self):
        u = np.array([[1.], [0.]])
        _, report = lab.low_rank_solve(np.eye(2), u, np.array([[-1 + 1e-8]]), u, np.ones(2))
        self.assertEqual(report["core_condition"], 1.)
        self.assertGreater(report["updated_condition"], 1e7)
        self.assertLess(report["core_smallest_singular_value"], 2e-8)

    def test_bad_shapes_nonfinite_and_complex_rejected(self):
        for a in (np.ones((2, 3)), np.array([[np.nan]]), np.eye(2).astype(complex)):
            with self.assertRaises(ValueError):
                lab.low_rank_solve(a, np.ones((2, 1)), np.ones((1, 1)), np.ones((2, 1)), np.ones(2))
        with self.assertRaises(ValueError):
            lab.low_rank_solve(np.eye(2), np.ones((2, 2)), np.eye(1), np.ones((2, 1)), np.ones(2))

    def test_generalized_residual_and_metric_orthogonality(self):
        a, b = np.array([[3., 1.], [1., 2.]]), np.array([[2., .4], [.4, 1.]])
        values, vectors, report = lab.generalized_eigen(a, b)
        np.testing.assert_allclose(a @ vectors, (b @ vectors) * values, atol=1e-12)
        np.testing.assert_allclose(vectors.T @ b @ vectors, np.eye(2), atol=1e-12)
        self.assertFalse(np.allclose(vectors.T @ vectors, np.eye(2)))
        q = report["whitened_vectors"]
        np.testing.assert_allclose(q.T @ b @ q, np.eye(2), atol=1e-12)

    def test_repeated_eigenvalues_compare_subspaces_not_vector_signs(self):
        b = np.array([[2., .3], [.3, 1.]])
        values, vectors, _ = lab.generalized_eigen(4 * b, b)
        np.testing.assert_allclose(values, [4., 4.])
        np.testing.assert_allclose(vectors @ vectors.T @ b, np.eye(2), atol=1e-12)

    def test_ill_conditioned_metric_reports_instead_of_asserting_agreement(self):
        rng = np.random.default_rng(2)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        b = q @ np.diag([1e-10, 1., 2.]) @ q.T
        a = rng.normal(size=(3, 3))
        values, vectors, report = lab.generalized_eigen((a + a.T) / 2, b)
        self.assertTrue(np.isfinite(values).all() and np.isfinite(vectors).all())
        self.assertGreater(report["metric_condition"], 1e9)
        self.assertTrue(np.isfinite(report["eigenvalue_method_max_absolute_difference"]))

    def test_generalized_requires_symmetry_and_spd_metric(self):
        with self.assertRaises(ValueError):
            lab.generalized_eigen(np.array([[1., 2.], [0., 1.]]), np.eye(2))
        for b in (np.diag([1., 0.]), np.diag([1., -1.])):
            with self.assertRaises(np.linalg.LinAlgError):
                lab.generalized_eigen(np.eye(2), b)

    def test_symmetry_check_is_scale_relative_and_averages_roundoff(self):
        for scale in (1e-200, 1., 1e200):
            with self.assertRaises(ValueError):
                lab.symmetric(scale * np.array([[0., 1.], [0., 0.]]), "A")
        noisy = np.array([[2., 1. + 1e-13], [1., 2.]])
        accepted = lab.symmetric(noisy, "A")
        np.testing.assert_array_equal(accepted, accepted.T)
        np.testing.assert_allclose(accepted, noisy, rtol=1e-12)

    def test_schur_handles_defective_and_complex_pair_matrices(self):
        for a in (np.array([[.8, 3.], [0., .8]]), np.array([[0., -1.], [1., 0.]])):
            report = lab.dynamics(a, [0., .5, 1.], [0, 1, 4])
            self.assertLess(report["schur_residual"], 1e-12)
            self.assertLess(report["orthogonality_error"], 1e-12)

    def test_stable_dynamics_can_amplify_before_decaying(self):
        continuous = lab.dynamics(np.array([[-1., 8.], [0., -2.]]), [0., .5, 8.], [0])
        self.assertGreater(continuous["continuous"][1]["operator_norm"], 1.5)
        self.assertLess(continuous["continuous"][-1]["operator_norm"], .01)
        discrete = lab.dynamics(np.array([[.8, 3.], [0., .8]]), [0.], [0, 4, 100])
        self.assertGreater(discrete["discrete"][1]["operator_norm"], 5.)
        self.assertLess(discrete["discrete"][-1]["operator_norm"], .001)

    def test_dynamics_time_contract(self):
        for times, steps in (([-1.], [0]), ([0.], [-1]), ([0.], [True]), ([np.nan], [0]),
                             (np.array([1. + 2j]), [0])):
            with self.assertRaises(ValueError):
                lab.dynamics(np.eye(2), times, steps)

    def test_cli_runs_without_network(self):
        result = subprocess.run([sys.executable, str(SOURCE)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertLess(report["low_rank_update"]["scaled_residual"], 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
