"""Unit tests for from-scratch Gauss–Jordan elimination."""

from __future__ import annotations

import numpy as np
import pytest

from gauss_jordan import (
    EliminationResult,
    determinant,
    gauss_jordan_eliminate,
    invert,
    matrix_rank,
    reduced_row_echelon_form,
    solve,
    solve_with_steps,
)


def test_identity_system():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    result = solve(A, b)
    np.testing.assert_allclose(result.x, b)
    assert result.residual_norm < 1e-12
    assert result.determinant == pytest.approx(1.0)
    assert result.rank == 3
    assert result.n_flops > 0


def test_matches_numpy_solve():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    A = A @ A.T + 0.5 * np.eye(8)
    b = rng.normal(size=8)

    ours = solve(A, b)
    reference = np.linalg.solve(A, b)
    np.testing.assert_allclose(ours.x, reference, rtol=1e-9, atol=1e-9)
    assert ours.relative_residual < 1e-10


def test_multi_rhs_matches_numpy():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    A = A @ A.T + 0.3 * np.eye(6)
    B = rng.normal(size=(6, 4))

    ours = solve(A, B)
    reference = np.linalg.solve(A, B)
    np.testing.assert_allclose(ours.x, reference, rtol=1e-9, atol=1e-9)
    assert ours.x.shape == (6, 4)
    assert ours.n_rhs == 4
    assert ours.relative_residual < 1e-10


def test_singular_matrix_raises_on_solve():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        solve(A, b)


def test_rank_deficient_rref_reports_rank():
    A = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 0.0, 1.0],
        ]
    )
    result = gauss_jordan_eliminate(A, allow_rank_deficient=True)
    assert isinstance(result, EliminationResult)
    assert result.rank == 2
    assert matrix_rank(A) == 2
    assert determinant(A) == pytest.approx(0.0)
    # Pivot rows of the RREF should be canonical.
    rref = result.rref
    assert abs(rref[0, result.pivot_columns[0]] - 1.0) < 1e-12
    assert abs(rref[1, result.pivot_columns[1]] - 1.0) < 1e-12


def test_partial_pivoting_handles_zero_diagonal():
    A = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = np.array([5.0, 8.0])
    result = solve(A, b)
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-12)


def test_no_pivoting_fails_on_zero_diagonal():
    A = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = np.array([5.0, 8.0])
    with pytest.raises(np.linalg.LinAlgError):
        solve(A, b, pivoting=False)


def test_rref_of_nonsingular_matrix_is_identity():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    rref = reduced_row_echelon_form(A)
    np.testing.assert_allclose(rref, np.eye(2), atol=1e-12)


def test_rref_left_block_is_identity_after_solve():
    A = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    result = solve(A, b)
    np.testing.assert_allclose(result.rref[:, :-1], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(result.x, [2.0, 3.0, -1.0], atol=1e-12)


def test_eliminate_always_returns_result_object():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    history = gauss_jordan_eliminate(A, b, record_steps=True)
    assert isinstance(history, EliminationResult)
    assert len(history.steps) == 2
    assert history.growth_factor >= 1.0
    bare = gauss_jordan_eliminate(A, b, record_steps=False)
    assert bare.steps == []


def test_solve_with_steps_agrees_with_solve():
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    b = np.array([1.0, 5.0])
    direct = solve(A, b)
    with_steps, history = solve_with_steps(A, b)
    np.testing.assert_allclose(direct.x, with_steps.x)
    np.testing.assert_allclose(history.solution, direct.x)


def test_determinant_matches_numpy():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    assert determinant(A) == pytest.approx(np.linalg.det(A), rel=1e-10, abs=1e-10)


def test_invert_matches_numpy_and_recovers_identity():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4))
    A = A @ A.T + 0.1 * np.eye(4)
    inv_A, det_A = invert(A)
    np.testing.assert_allclose(inv_A, np.linalg.inv(A), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(A @ inv_A, np.eye(4), atol=1e-9)
    assert det_A == pytest.approx(np.linalg.det(A), rel=1e-9, abs=1e-9)


def test_growth_factor_reduced_by_pivoting():
    eps = 1e-12
    A = np.array([[eps, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, -1.0]])
    b = A @ np.array([1.0, -2.0, 3.0])
    with_pivot = solve(A, b, pivoting=True)
    without = solve(A, b, pivoting=False)
    assert with_pivot.growth_factor < without.growth_factor


def test_flop_count_scales_cubically():
    rng = np.random.default_rng(4)
    flops = []
    for n in (5, 10, 20):
        A = rng.normal(size=(n, n))
        A = A @ A.T + np.eye(n)
        flops.append(solve(A, rng.normal(size=n)).n_flops)
    # Rough cubic: flops(20)/flops(10) should be near 8.
    ratio = flops[2] / flops[1]
    assert 6.0 < ratio < 10.0


def test_input_validation():
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.eye(2), np.ones(3))
