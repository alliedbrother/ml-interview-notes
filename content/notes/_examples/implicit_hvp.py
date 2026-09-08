"""Float64 CPU checks for curvature products and ridge hypergradients.

Python 3.11, NumPy 1.26.4, SciPy 1.11.4, PyTorch 2.8.0.
Run: python implicit_hvp.py
No downloads, GPU, training service, or output files are required.
"""

import json
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import torch
from torch.func import grad, hessian, jvp


DTYPE = torch.float64


def validate_vectors(x, v):
    if x.ndim != 1 or v.shape != x.shape or x.numel() == 0:
        raise ValueError("x and v must be nonempty vectors with identical shapes")
    if x.dtype != DTYPE or v.dtype != DTYPE or x.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("This diagnostic lab requires CPU float64 tensors")
    if not torch.isfinite(x).all() or not torch.isfinite(v).all():
        raise ValueError("Inputs must be finite")


def hvp_forward_reverse(loss, x, v):
    """JVP of the reverse-mode gradient; v is held fixed at its supplied value."""
    validate_vectors(x, v)
    return jvp(grad(loss), (x,), (v.detach(),))[1]


def hvp_reverse_reverse(loss, x, v):
    """Gradient of grad(loss).dot(v), with no differentiated dependence in v."""
    validate_vectors(x, v)
    fixed_v = v.detach()
    return grad(lambda point: torch.dot(grad(loss)(point), fixed_v))(x)


def curvature_fixture():
    a = np.array([[1., 2., -.5], [0., -1., 2.], [2., .25, 1.], [-.5, 1., .5]])
    b = np.array([.2, -.3, .8, .1])
    x = np.array([.4, -.7, 1.1])
    v = np.array([.3, -.5, .8])
    return a, b, x, v


def smooth_loss(point, a, b):
    residual = a @ point - b
    return .5 * residual.square().sum() + .1 * point.pow(4).sum() + .2 * point.sin().sum()


def analytic_gradient(point, a, b):
    return a.T @ (a @ point - b) + .4 * point**3 + .2 * np.cos(point)


def analytic_hessian(point, a):
    return a.T @ a + np.diag(1.2 * point**2 - .2 * np.sin(point))


def finite_difference_hvp(point, direction, a, b, step):
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Finite-difference step must be finite and positive")
    return (analytic_gradient(point + step * direction, a, b)
            - analytic_gradient(point - step * direction, a, b)) / (2 * step)


def solve_hvp_system(loss, point, rhs, *, damping=0., rtol=1e-10, maxiter=100):
    """SciPy CG with an HVP LinearOperator. Caller must establish SPD.

    Finite residual/CG convergence are checked; they do not establish SPD.
    The linear solve is a numerical operation, not itself a differentiable layer.
    """
    validate_vectors(point, rhs)
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("Damping must be finite and nonnegative")
    if not math.isfinite(rtol) or rtol <= 0 or maxiter < 1:
        raise ValueError("Positive tolerance and iteration budget required")
    fixed_point = point.detach()
    iterations = 0

    def matvec(vector):
        tangent = torch.tensor(np.asarray(vector), dtype=DTYPE)
        product = hvp_forward_reverse(loss, fixed_point, tangent).detach().numpy()
        return product + damping * vector

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    operator = LinearOperator((point.numel(), point.numel()), matvec=matvec, dtype=np.float64)
    target = rhs.detach().numpy()
    answer, info = cg(operator, target, tol=rtol, atol=0., maxiter=maxiter,
                      callback=count_iteration)
    residual = float(np.linalg.norm(matvec(answer) - target))
    denominator = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    relative_residual = residual / denominator
    if info != 0 or not np.isfinite(answer).all() or relative_residual > max(10 * rtol, 1e-12):
        raise RuntimeError(f"CG did not meet residual contract: info={info}, residual={relative_residual}")
    return answer, {"iterations": iterations, "relative_residual": relative_residual,
                    "damping": damping}


def ridge_fixture():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(12, 4))
    x[:, 3] = .98 * x[:, 0] + .02 * x[:, 3]
    z = rng.normal(size=(7, 4))
    truth = np.array([1., -.7, .3, .2])
    y = x @ truth + .1 * rng.normal(size=12)
    target = z @ truth + .1 * rng.normal(size=7)
    return x, y, z, target


def ridge_hypergradient(x, y, z, target, regularization, *, direct_penalty=0.):
    """Differentiate validation MSE/2 + direct_penalty * lambda**2/2.

    Training: ||Xw-y||^2/(2n) + lambda ||w||^2/2, no intercept.
    Positive lambda guarantees a unique stationary solution for finite X.
    """
    x, y, z, target = [np.asarray(item, dtype=float) for item in (x, y, z, target)]
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or not x.size or not z.size:
        raise ValueError("Nonempty train/validation matrices must have the same feature count")
    if y.shape != (x.shape[0],) or target.shape != (z.shape[0],):
        raise ValueError("Target vectors must match sample counts")
    if not all(np.isfinite(item).all() for item in (x, y, z, target)):
        raise ValueError("Data must be finite")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive finite regularization required for this ridge lab")
    if not math.isfinite(direct_penalty) or direct_penalty < 0:
        raise ValueError("Direct penalty must be finite and nonnegative")
    a = x.T @ x / len(x) + regularization * np.eye(x.shape[1])
    b = x.T @ y / len(x)
    weights = np.linalg.solve(a, b)
    residual = z @ weights - target
    outer_gradient = z.T @ residual / len(z)
    adjoint = np.linalg.solve(a.T, outer_gradient)
    sensitivity = np.linalg.solve(a, -weights)
    direct = direct_penalty * regularization
    hypergradient = direct - float(adjoint @ weights)
    objective = float(residual @ residual / (2 * len(z)) + .5 * direct_penalty * regularization**2)
    return {"weights": weights, "sensitivity": sensitivity, "adjoint": adjoint,
            "hessian": a, "outer_gradient": outer_gradient, "hypergradient": hypergradient,
            "objective": objective, "stationarity_residual": float(np.linalg.norm(a @ weights - b)),
            "adjoint_residual": float(np.linalg.norm(a.T @ adjoint - outer_gradient)),
            "condition_number": float(np.linalg.cond(a)), "direct_term": direct}


def differentiable_ridge_objective(x, y, z, target, regularization, *, direct_penalty=0.):
    a = x.T @ x / x.shape[0] + regularization * torch.eye(x.shape[1], dtype=x.dtype)
    weights = torch.linalg.solve(a, x.T @ y / x.shape[0])
    residual = z @ weights - target
    return residual.square().sum() / (2 * z.shape[0]) + .5 * direct_penalty * regularization.square()


def unrolled_ridge_objective(x, y, z, target, regularization, *, steps, learning_rate):
    """Differentiate a fixed count of GD updates from a constant zero initial state."""
    if steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Nonnegative step count and positive finite learning rate required")
    weights = torch.zeros(x.shape[1], dtype=x.dtype)
    for _ in range(steps):
        gradient = x.T @ (x @ weights - y) / x.shape[0] + regularization * weights
        weights = weights - learning_rate * gradient
    residual = z @ weights - target
    return residual.square().sum() / (2 * z.shape[0])


def run_checks():
    torch.set_num_threads(1)
    a, b, point, direction = curvature_fixture()
    ta, tb, tx, tv = [torch.tensor(item, dtype=DTYPE) for item in (a, b, point, direction)]
    loss = lambda value: smooth_loss(value, ta, tb)
    expected = analytic_hessian(point, a) @ direction
    forward = hvp_forward_reverse(loss, tx, tv)
    reverse = hvp_reverse_reverse(loss, tx, tv)
    full = hessian(loss)(tx)
    for actual in (forward.detach().numpy(), reverse.detach().numpy(), (full @ tv).detach().numpy()):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    sweep = [{"step": step, "absolute_error": float(np.linalg.norm(
        finite_difference_hvp(point, direction, a, b, step) - expected))}
        for step in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)]
    solution, cg_report = solve_hvp_system(loss, tx, tv, damping=.2)
    np.testing.assert_allclose(solution, np.linalg.solve(analytic_hessian(point, a) + .2 * np.eye(3), direction),
                               rtol=1e-10, atol=1e-12)
    x, y, z, target = ridge_fixture()
    lam, penalty = .3, .1
    result = ridge_hypergradient(x, y, z, target, lam, direct_penalty=penalty)
    inputs = [torch.tensor(item, dtype=DTYPE) for item in (x, y, z, target)]
    scalar = torch.tensor(lam, dtype=DTYPE, requires_grad=True)
    objective = differentiable_ridge_objective(*inputs, scalar, direct_penalty=penalty)
    auto = float(torch.autograd.grad(objective, scalar)[0])
    step = 1e-5
    plus = ridge_hypergradient(x, y, z, target, lam + step, direct_penalty=penalty)["objective"]
    minus = ridge_hypergradient(x, y, z, target, lam - step, direct_penalty=penalty)["objective"]
    finite = (plus - minus) / (2 * step)
    np.testing.assert_allclose(result["hypergradient"], auto, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result["hypergradient"], finite, rtol=1e-7, atol=1e-9)
    return {"dtype": "float64", "device": "cpu", "hvp": forward.tolist(),
            "finite_difference_sweep": sweep, "hvp_cg": cg_report,
            "ridge": {"implicit": result["hypergradient"], "through_solve": auto,
                      "finite_difference": finite, "stationarity_residual": result["stationarity_residual"],
                      "adjoint_residual": result["adjoint_residual"],
                      "condition_number": result["condition_number"], "direct_term": result["direct_term"]}}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
