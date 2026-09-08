"""Float64 CPU lab: low-rank updates, generalized eigenproblems and dynamics.

Python 3.11, NumPy 1.26.4, SciPy 1.11.4. No network or model downloads.
Run: python spectral_updates.py
"""

import json

import numpy as np
from scipy.linalg import cholesky, eigh, expm, schur, solve_triangular


def matrix(value, name, *, square=False):
    original = np.asarray(value)
    if np.iscomplexobj(original):
        raise ValueError(f"{name}: this lab accepts real matrices only")
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name}: require a nonempty finite matrix")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(f"{name}: require a square matrix")
    return value


def symmetric(value, name):
    value = matrix(value, name, square=True)
    scale = np.max(np.abs(value))
    normalized = value / scale if scale else value
    if not np.allclose(normalized, normalized.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name}: require a symmetric matrix")
    # Both solvers use the same symmetric interpretation of roundoff-level noise.
    return value * .5 + value.T * .5


def scaled_residual(a, x, b):
    numerator = np.linalg.norm(a @ x - b)
    scale = np.linalg.norm(a) * np.linalg.norm(x) + np.linalg.norm(b)
    return float(numerator / scale) if scale else float(numerator)


def low_rank_solve(a, u, c, v, rhs):
    """Solve (A + U C V.T) X = B without requiring C to be invertible.

    Dense A and the updated matrix are retained for teaching diagnostics. A
    production repeated-update solver would reuse an existing factorization.
    """
    a = matrix(a, "A", square=True)
    u, v, c = matrix(u, "U"), matrix(v, "V"), matrix(c, "C", square=True)
    n, rank = u.shape
    if a.shape != (n, n) or v.shape != (n, rank) or c.shape != (rank, rank):
        raise ValueError("A, U, C, V have incompatible shapes")
    raw_rhs = np.asarray(rhs)
    if np.iscomplexobj(raw_rhs):
        raise ValueError("This lab accepts real right-hand sides only")
    vector = raw_rhs.ndim == 1
    b = matrix(raw_rhs[:, None] if vector else raw_rhs, "B")
    if b.shape[0] != n:
        raise ValueError("Right-hand side row count must match A")
    solved = np.linalg.solve(a, np.concatenate([b, u], axis=1))
    y, z = solved[:, :b.shape[1]], solved[:, b.shape[1]:]
    core = np.eye(rank) + c @ (v.T @ z)
    correction = z @ np.linalg.solve(core, c @ (v.T @ y))
    answer = y - correction
    updated = a + u @ c @ v.T
    diagnostics = {"scaled_residual": scaled_residual(updated, answer, b),
                   "core_condition": float(np.linalg.cond(core)),
                   "core_smallest_singular_value": float(np.linalg.svd(core, compute_uv=False)[-1]),
                   "updated_condition": float(np.linalg.cond(updated))}
    return (answer[:, 0] if vector else answer), diagnostics


def generalized_eigen(a, b):
    """Symmetric A, SPD B: compare LAPACK against Cholesky whitening."""
    a, b = symmetric(a, "A"), symmetric(b, "B")
    if a.shape != b.shape:
        raise ValueError("A and B must have the same shape")
    lower = cholesky(b, lower=True)
    left = solve_triangular(lower, a, lower=True)
    whitened = solve_triangular(lower, left.T, lower=True).T
    whitened = (whitened + whitened.T) / 2
    manual_values, q = eigh(whitened)
    manual_vectors = solve_triangular(lower.T, q, lower=False)
    values, vectors = eigh(a, b, type=1)
    residual = a @ vectors - (b @ vectors) * values
    scale = np.linalg.norm(a) * np.linalg.norm(vectors) + np.linalg.norm(b @ vectors * values)
    return values, vectors, {"scaled_residual": float(np.linalg.norm(residual) / max(scale, 1e-300)),
                            "metric_condition": float(np.linalg.cond(b)),
                            "eigenvalue_method_max_absolute_difference": float(np.max(np.abs(values - manual_values))),
                            "metric_orthogonality_error": float(np.linalg.norm(vectors.T @ b @ vectors - np.eye(len(values)))),
                            "whitened_vectors": manual_vectors}


def dynamics(a, times, steps):
    """Compare continuous exp(tA) and discrete A**k in Schur coordinates."""
    a = matrix(a, "A", square=True)
    if np.iscomplexobj(np.asarray(times)):
        raise ValueError("Times must be real")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Times must be a nonempty finite nonnegative vector")
    if not steps or any(type(k) is not int or k < 0 for k in steps):
        raise ValueError("Steps must be nonempty nonnegative integers")
    triangular, q = schur(a, output="real")
    continuous, discrete = [], []
    for time in times:
        propagated = q @ expm(time * triangular) @ q.T
        reference = expm(time * a)
        np.testing.assert_allclose(propagated, reference, rtol=1e-11, atol=1e-12)
        continuous.append({"time": float(time), "operator_norm": float(np.linalg.norm(reference, 2))})
    for k in steps:
        propagated = q @ np.linalg.matrix_power(triangular, k) @ q.T
        reference = np.linalg.matrix_power(a, k)
        np.testing.assert_allclose(propagated, reference, rtol=1e-11, atol=1e-12)
        discrete.append({"step": k, "operator_norm": float(np.linalg.norm(reference, 2))})
    return {"schur_residual": scaled_residual(np.eye(len(a)), q @ triangular @ q.T, a),
            "orthogonality_error": float(np.linalg.norm(q.T @ q - np.eye(len(a)))),
            "continuous": continuous, "discrete": discrete}


def demo():
    rng = np.random.default_rng(314)
    base = rng.normal(size=(8, 8))
    a = base.T @ base + np.eye(8)
    u = rng.normal(size=(8, 2))
    c = np.diag([.4, 0.])
    rhs = rng.normal(size=(8, 3))
    answer, update = low_rank_solve(a, u, c, u, rhs)
    np.testing.assert_allclose(answer, np.linalg.solve(a + u @ c @ u.T, rhs), rtol=1e-11, atol=1e-12)
    near = 1e-8
    _, fragile = low_rank_solve(np.eye(2), np.array([[1.], [0.]]),
                               np.array([[-1. + near]]), np.array([[1.], [0.]]), np.ones(2))
    values, vectors, eig_report = generalized_eigen(np.array([[3., 1.], [1., 2.]]),
                                                  np.array([[2., .4], [.4, 1.]]))
    eig_report.pop("whitened_vectors")
    eig_report["eigenvalues"] = values.tolist()
    stable_continuous = dynamics(np.array([[-1., 8.], [0., -2.]]), [0., .25, .5, 1., 8.], [0, 1])
    stable_discrete = dynamics(np.array([[.8, 3.], [0., .8]]), [0.], [0, 1, 2, 4, 20, 100])
    return {"low_rank_update": update, "near_singular_rank_one_update": fragile,
            "generalized_eigen": eig_report, "continuous_stable_generator": stable_continuous,
            "discrete_stable_step_matrix": stable_discrete,
            "scope": "Tiny CPU correctness fixtures, not performance benchmarks"}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
