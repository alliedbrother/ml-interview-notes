"""Independent analytic, autodiff, finite-difference and solve regression checks."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/notes/_examples/implicit_hvp.py"
spec = importlib.util.spec_from_file_location("implicit_hvp", SOURCE)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
torch.set_num_threads(1)


class ImplicitHVPTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.x, self.v = lab.curvature_fixture()
        self.ta, self.tb, self.tx, self.tv = [torch.tensor(value, dtype=lab.DTYPE)
                                            for value in (self.a, self.b, self.x, self.v)]
        self.loss = lambda point: lab.smooth_loss(point, self.ta, self.tb)

    def test_gradient_and_full_hessian_analytic_reference(self):
        np.testing.assert_allclose(lab.grad(self.loss)(self.tx).numpy(),
                                   lab.analytic_gradient(self.x, self.a, self.b), rtol=1e-12)
        full = lab.hessian(self.loss)(self.tx).numpy()
        self.assertEqual(full.shape, (3, 3))
        np.testing.assert_allclose(full, full.T, atol=1e-12)
        np.testing.assert_allclose(full, lab.analytic_hessian(self.x, self.a), rtol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(full)[0], 0.)

    def test_two_hvp_modes_without_full_hessian(self):
        expected = lab.analytic_hessian(self.x, self.a) @ self.v
        with patch.object(lab, "hessian", side_effect=AssertionError("Full Hessian forbidden")):
            for method in (lab.hvp_forward_reverse, lab.hvp_reverse_reverse):
                actual = method(self.loss, self.tx, self.tv)
                self.assertEqual(actual.shape, self.tx.shape)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)

    def test_hvp_is_linear_in_fixed_direction(self):
        other = torch.tensor([.2, .4, -.3], dtype=lab.DTYPE)
        a = lab.hvp_forward_reverse(self.loss, self.tx, 2 * self.tv - 3 * other)
        b = 2 * lab.hvp_forward_reverse(self.loss, self.tx, self.tv)
        b -= 3 * lab.hvp_forward_reverse(self.loss, self.tx, other)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_direction_is_frozen_even_if_it_has_a_graph(self):
        x = self.tx.clone().requires_grad_()
        direction = x.square()
        expected = lab.analytic_hessian(self.x, self.a) @ self.x**2
        for method in (lab.hvp_forward_reverse, lab.hvp_reverse_reverse):
            np.testing.assert_allclose(method(self.loss, x, direction).detach().numpy(), expected,
                                       rtol=1e-12, atol=1e-12)

    def test_zero_hessian_for_constant_and_linear_objectives(self):
        for loss in (lambda x: (x * 0).sum() + 3, lambda x: (x * self.tv).sum()):
            for method in (lab.hvp_forward_reverse, lab.hvp_reverse_reverse):
                torch.testing.assert_close(method(loss, self.tx, self.tv), torch.zeros_like(self.tx))

    def test_validation_rejects_shapes_dtypes_and_nonfinite(self):
        pairs = ((self.tx[:, None], self.tv), (self.tx, self.tv[:2]),
                 (self.tx.float(), self.tv.float()), (self.tx[:0], self.tv[:0]),
                 (self.tx, torch.tensor([float("nan"), 0., 1.], dtype=lab.DTYPE)))
        for x, v in pairs:
            with self.subTest(shape=tuple(x.shape)), self.assertRaises(ValueError):
                lab.hvp_forward_reverse(self.loss, x, v)

    def test_finite_difference_step_sweep_and_input_immutability(self):
        original = self.x.copy()
        expected = lab.analytic_hessian(self.x, self.a) @ self.v
        steps = [1e-1, 1e-3, 1e-5, 1e-10]
        errors = [np.linalg.norm(lab.finite_difference_hvp(self.x, self.v, self.a, self.b, h) - expected)
                  for h in steps]
        self.assertLess(min(errors), 1e-8)
        self.assertGreater(errors[0], 100 * min(errors))
        self.assertGreater(errors[-1], 10 * min(errors))
        np.testing.assert_array_equal(original, self.x)
        for step in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                lab.finite_difference_hvp(self.x, self.v, self.a, self.b, step)

    def test_matrix_free_cg_matches_dense_damped_solve(self):
        answer, report = lab.solve_hvp_system(self.loss, self.tx, self.tv, damping=.2)
        expected = np.linalg.solve(lab.analytic_hessian(self.x, self.a) + .2 * np.eye(3), self.v)
        np.testing.assert_allclose(answer, expected, rtol=1e-10, atol=1e-12)
        self.assertLess(report["relative_residual"], 1e-10)
        self.assertLessEqual(report["iterations"], 3)

    def test_cg_nonconvergence_zero_rhs_and_invalid_controls(self):
        with self.assertRaisesRegex(RuntimeError, "residual contract"):
            lab.solve_hvp_system(self.loss, self.tx, self.tv, maxiter=1)
        answer, report = lab.solve_hvp_system(self.loss, self.tx, torch.zeros_like(self.tv))
        np.testing.assert_array_equal(answer, np.zeros(3))
        self.assertEqual(report["relative_residual"], 0.)
        for kwargs in ({"damping": -1.}, {"rtol": 0.}, {"maxiter": 0}):
            with self.assertRaises(ValueError):
                lab.solve_hvp_system(self.loss, self.tx, self.tv, **kwargs)

    def test_indefinite_curvature_is_not_automatically_spd(self):
        loss = lambda point: point[0]**2 - .5 * point[1]**2
        point = torch.zeros(2, dtype=lab.DTYPE)
        direction = torch.tensor([0., 1.], dtype=lab.DTYPE)
        hv = lab.hvp_forward_reverse(loss, point, direction)
        self.assertLess(float(direction @ hv), 0.)
        answer, _ = lab.solve_hvp_system(loss, point, direction, damping=2.)
        np.testing.assert_allclose(answer, [0., 1.], atol=1e-12)

    def test_implicit_derivative_matches_solve_autograd_and_finite_difference(self):
        data = lab.ridge_fixture()
        tensors = [torch.tensor(value, dtype=lab.DTYPE) for value in data]
        for lam in (.01, .3, 2.):
            for penalty in (0., .1):
                with self.subTest(regularization=lam, direct_penalty=penalty):
                    result = lab.ridge_hypergradient(*data, lam, direct_penalty=penalty)
                    scalar = torch.tensor(lam, dtype=lab.DTYPE, requires_grad=True)
                    objective = lab.differentiable_ridge_objective(*tensors, scalar, direct_penalty=penalty)
                    auto = torch.autograd.grad(objective, scalar)[0].item()
                    h = min(1e-5, lam * 1e-3)
                    plus = lab.ridge_hypergradient(*data, lam + h, direct_penalty=penalty)["objective"]
                    minus = lab.ridge_hypergradient(*data, lam - h, direct_penalty=penalty)["objective"]
                    self.assertAlmostEqual(result["hypergradient"], auto, places=10)
                    np.testing.assert_allclose(result["hypergradient"], (plus - minus) / (2*h),
                                               rtol=2e-6, atol=1e-8)
                    self.assertLess(result["stationarity_residual"], 1e-12)
                    self.assertLess(result["adjoint_residual"], 1e-12)

    def test_forward_sensitivity_and_adjoint_agree(self):
        data = lab.ridge_fixture()
        result = lab.ridge_hypergradient(*data, .3, direct_penalty=.1)
        chain = result["outer_gradient"] @ result["sensitivity"] + result["direct_term"]
        self.assertAlmostEqual(chain, result["hypergradient"], places=12)
        np.testing.assert_allclose(result["hessian"] @ result["sensitivity"], -result["weights"],
                                   rtol=1e-12, atol=1e-12)

    def test_unrolled_derivative_differs_then_converges(self):
        data = lab.ridge_fixture()
        tensors = [torch.tensor(value, dtype=lab.DTYPE) for value in data]
        reference = lab.ridge_hypergradient(*data, .3)
        eta = 1. / np.linalg.eigvalsh(reference["hessian"])[-1]
        derivatives = []
        for steps in (1, 300):
            scalar = torch.tensor(.3, dtype=lab.DTYPE, requires_grad=True)
            outer = lab.unrolled_ridge_objective(*tensors, scalar, steps=steps, learning_rate=eta)
            derivatives.append(torch.autograd.grad(outer, scalar)[0].item())
        self.assertGreater(abs(derivatives[0] - reference["hypergradient"]), 1e-3)
        self.assertAlmostEqual(derivatives[1], reference["hypergradient"], places=9)

    def test_regularization_improves_this_spd_condition_number(self):
        data = lab.ridge_fixture()
        weak = lab.ridge_hypergradient(*data, 1e-6)
        stronger = lab.ridge_hypergradient(*data, .3)
        self.assertLess(stronger["condition_number"], weak["condition_number"])
        self.assertGreater(np.linalg.norm(weak["weights"] - stronger["weights"]), .01)

    def test_ridge_validation_and_singular_boundary(self):
        data = lab.ridge_fixture()
        for lam in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                lab.ridge_hypergradient(*data, lam)
        x, y, z, target = data
        for values in ((x, y[:-1], z, target), (x, y, z[:, :-1], target),
                       (x * float("nan"), y, z, target)):
            with self.assertRaises(ValueError):
                lab.ridge_hypergradient(*values, .3)

    def test_cli_is_offline_and_reports_actual_numeric_checks(self):
        result = subprocess.run([sys.executable, str(SOURCE)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(len(report["finite_difference_sweep"]), 10)
        self.assertAlmostEqual(report["ridge"]["implicit"], report["ridge"]["through_solve"], places=10)

    def test_expectation_chain_rules_keep_both_additive_terms(self):
        chapter = (ROOT / "content/notes/math/calculus.md").read_text()
        self.assertIn("\n+J_{g_\\theta}^\\top", chapter)
        self.assertIn("\n+h_\\theta(Z)\\nabla_\\theta\\log", chapter)
        # E[theta * Z], Z ~ N(theta, 1), has two theta contributions.
        theta = torch.tensor(.7, dtype=lab.DTYPE, requires_grad=True)
        mean_outer = theta * theta
        self.assertAlmostEqual(torch.autograd.grad(mean_outer, theta)[0].item(), 1.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
