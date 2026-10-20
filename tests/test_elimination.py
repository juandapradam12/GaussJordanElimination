"""Unit tests for Gauss–Jordan elimination and the solve API."""

from __future__ import annotations

import numpy as np
import pytest

from gauss_jordan import gauss_jordan_eliminate, reduced_row_echelon_form, solve


def test_identity_system():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    result = solve(A, b)
    np.testing.assert_allclose(result.x, b)
    assert result.residual_norm < 1e-12


def test_matches_numpy_solve():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    # Improve conditioning slightly for a fair comparison.
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
    # Without pivoting the first pivot would be zero.
    A = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = np.array([5.0, 8.0])
    result = solve(A, b)
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-12)


def test_rref_of_nonsingular_matrix_is_identity():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    rref = reduced_row_echelon_form(A)
    np.testing.assert_allclose(rref, np.eye(2), atol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError):
        gauss_jordan_eliminate(np.eye(2), np.ones(3))
