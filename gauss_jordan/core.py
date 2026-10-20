"""Core Gauss–Jordan elimination with partial pivoting.

Transforms an augmented matrix ``[A | b]`` into reduced row echelon form
(RREF) so that the solution of ``A x = b`` can be read from the last column
when ``A`` is nonsingular.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.floating]


def _partial_pivot(augmented: ArrayF, column: int, n_rows: int) -> None:
    """Swap row ``column`` with the row of largest |pivot| below it (in-place)."""
    pivot_row = column + int(np.argmax(np.abs(augmented[column:n_rows, column])))
    if pivot_row != column:
        augmented[[column, pivot_row]] = augmented[[pivot_row, column]]


def gauss_jordan_eliminate(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
) -> ArrayF:
    """Return the RREF of the augmented system ``[A | b]``.

    Parameters
    ----------
    A:
        Square coefficient matrix of shape ``(n, n)``.
    b:
        Right-hand side of shape ``(n,)`` or ``(n, 1)``.
    pivot_tol:
        Absolute tolerance below which a pivot is treated as singular.

    Returns
    -------
    ndarray
        Augmented matrix of shape ``(n, n + 1)`` in reduced row echelon form.
        For a nonsingular ``A``, the last column equals the unique solution.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")

    n = A.shape[0]
    augmented = np.hstack([A.copy(), b.reshape(n, 1)])

    for col in range(n):
        _partial_pivot(augmented, col, n)
        pivot = augmented[col, col]

        if abs(pivot) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Matrix is singular (or nearly singular) at column {col}: "
                f"|pivot| = {abs(pivot):.3e} < {pivot_tol:.3e}."
            )

        augmented[col, :] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row, col]
            if factor != 0.0:
                augmented[row, :] -= factor * augmented[col, :]

    return augmented


def reduced_row_echelon_form(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
) -> ArrayF:
    """Compute the RREF of a single matrix ``A`` (no right-hand side).

    Useful for inspecting the algebraic structure of ``A`` itself.
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    work = A.copy()
    row = 0

    for col in range(m):
        if row >= n:
            break

        pivot_row = row + int(np.argmax(np.abs(work[row:, col])))
        if abs(work[pivot_row, col]) < pivot_tol:
            continue

        work[[row, pivot_row]] = work[[pivot_row, row]]
        work[row, :] /= work[row, col]

        for r in range(n):
            if r == row:
                continue
            factor = work[r, col]
            if factor != 0.0:
                work[r, :] -= factor * work[row, :]

        row += 1

    return work
