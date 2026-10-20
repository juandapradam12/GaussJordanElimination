"""Unit tests for from-scratch Gauss–Jordan elimination."""

from __future__ import annotations

import numpy as np
import pytest

from gauss_jordan import (
    EliminationResult,
    determinant,
    gauss_jordan_eliminate,
    invert,
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


def test_matches_numpy_solve():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    A = A @ A.T + 0.5 * np.eye(8)
    b = rng.normal(size=8)

    ours = solve(A, b)
    reference = np.linalg.solve(A, b)
    np.testing.assert_allclose(ours.x, reference, rtol=1e-9, atol=1e-9)
    assert ours.relative_residual < 1e-10


def test_singular_matrix_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        gauss_jordan_eliminate(A, b)


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
    bare = gauss_jordan_eliminate(A, b, record_steps=False)
    assert isinstance(bare, EliminationResult)
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


def test_input_validation():
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.eye(2), np.ones(3))
