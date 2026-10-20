"""From-scratch Gauss–Jordan elimination with partial pivoting.

Given a nonsingular square system ``A x = b``, the algorithm transforms the
augmented matrix ``[A | b]`` into reduced row echelon form ``[I | x]``, from
which the solution is read in the last column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]


@dataclass
class EliminationStep:
    """Snapshot of the augmented matrix after one pivot column is processed."""

    column: int
    pivot_row: int
    pivot_value: float
    swapped: bool
    matrix: ArrayF


@dataclass
class EliminationResult:
    """Final RREF together with the sequence of intermediate states."""

    rref: ArrayF
    steps: list[EliminationStep] = field(default_factory=list)

    @property
    def solution(self) -> ArrayF:
        return self.rref[:, -1].copy()


def _partial_pivot(augmented: ArrayF, column: int, n_rows: int) -> tuple[int, bool]:
    """Swap row ``column`` with the largest-magnitude pivot candidate below it."""
    pivot_row = column + int(np.argmax(np.abs(augmented[column:n_rows, column])))
    swapped = pivot_row != column
    if swapped:
        augmented[[column, pivot_row]] = augmented[[pivot_row, column]]
    return pivot_row, swapped


def gauss_jordan_eliminate(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    record_steps: bool = False,
) -> ArrayF | EliminationResult:
    """Reduce ``[A | b]`` to RREF by Gauss–Jordan elimination.

    Parameters
    ----------
    A:
        Square coefficient matrix of shape ``(n, n)``.
    b:
        Right-hand side of shape ``(n,)`` or ``(n, 1)``.
    pivot_tol:
        Absolute tolerance below which a pivot is treated as singular.
    record_steps:
        If ``True``, return an :class:`EliminationResult` that also stores
        a snapshot after each pivot column. Otherwise return only the RREF
        array (shape ``(n, n + 1)``).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")

    n = A.shape[0]
    augmented = np.hstack([A.copy(), b.reshape(n, 1)])
    steps: list[EliminationStep] = []

    for col in range(n):
        pivot_row, swapped = _partial_pivot(augmented, col, n)
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

        if record_steps:
            steps.append(
                EliminationStep(
                    column=col,
                    pivot_row=pivot_row,
                    pivot_value=float(pivot),
                    swapped=swapped,
                    matrix=augmented.copy(),
                )
            )

    if record_steps:
        return EliminationResult(rref=augmented, steps=steps)
    return augmented


def reduced_row_echelon_form(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
) -> ArrayF:
    """Compute the RREF of a single matrix ``A`` (no right-hand side)."""
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
