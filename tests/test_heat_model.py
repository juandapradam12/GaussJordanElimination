"""Tests for the steady-state heat-conduction assembly."""

from __future__ import annotations

import numpy as np

from gauss_jordan import solve
from models import assemble_heat_rod


def test_tridiagonal_structure():
    problem = assemble_heat_rod(n_interior=5, source=2.0)
    A = problem.A
    # Off-tridiagonal entries must vanish.
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(i - j) > 1:
                assert A[i, j] == 0.0


def test_constant_source_matches_exact_parabola():
    problem = assemble_heat_rod(
        length=1.0,
        conductivity=1.0,
        n_interior=50,
        left_temperature=10.0,
        right_temperature=30.0,
        source=4.0,
    )
    exact = problem.exact_quadratic_solution()
    assert exact is not None

    numerical = solve(problem.A, problem.b)
    # Second-order finite differences: O(h^2) accuracy.
    max_err = float(np.max(np.abs(numerical.x - exact)))
    assert max_err < 5e-3
    assert numerical.relative_residual < 1e-10


def test_homogeneous_no_source_is_linear_profile():
    problem = assemble_heat_rod(
        n_interior=20,
        left_temperature=0.0,
        right_temperature=100.0,
        source=0.0,
    )
    result = solve(problem.A, problem.b)
    expected = np.linspace(
        problem.left_temperature,
        problem.right_temperature,
        problem.n_interior + 2,
    )[1:-1]
    np.testing.assert_allclose(result.x, expected, rtol=1e-10, atol=1e-10)


def test_sine_source_converges_second_order():
    from models import exact_sine_solution, sine_source

    errors = []
    hs = []
    for n in (20, 40, 80):
        problem = assemble_heat_rod(
            n_interior=n,
            source=sine_source(amplitude=5.0, length=1.0),
            left_temperature=0.0,
            right_temperature=0.0,
        )
        result = solve(problem.A, problem.b)
        exact = exact_sine_solution(
            problem.x_interior,
            amplitude=5.0,
            length=1.0,
            conductivity=1.0,
        )
        errors.append(float(np.max(np.abs(result.x - exact))))
        hs.append(problem.h)

    # Observed order ≈ log2(e_h / e_{h/2})
    order = np.log(errors[0] / errors[1]) / np.log(hs[0] / hs[1])
    assert order > 1.8
