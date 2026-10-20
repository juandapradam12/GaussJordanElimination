"""From-scratch Gauss–Jordan elimination with partial pivoting.

Given a nonsingular square system ``A x = b``, the algorithm transforms the
augmented matrix ``[A | b]`` into reduced row echelon form ``[I | x]``, from
which the solution is read in the last column.

The same reduction applied to ``[A | I]`` yields ``[I | A^{-1}]``. During
elimination the product of the (pre-normalization) pivots, corrected by the
number of row swaps, recovers ``det(A)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]


@dataclass
class EliminationStep:
    """Snapshot of the working matrix after one pivot column is processed."""

    column: int
    pivot_row: int
    pivot_value: float
    swapped: bool
    matrix: ArrayF


@dataclass
class EliminationResult:
    """RREF of an augmented matrix together with elimination metadata."""

    rref: ArrayF
    determinant: float
    n_swaps: int
    steps: list[EliminationStep] = field(default_factory=list)

    @property
    def solution(self) -> ArrayF:
        """Last column of ``[A | b]`` after reduction (square single-RHS case)."""
        return self.rref[:, -1].copy()

    @property
    def left_block(self) -> ArrayF:
        n = self.rref.shape[0]
        return self.rref[:, :n].copy()

    @property
    def right_block(self) -> ArrayF:
        n = self.rref.shape[0]
        return self.rref[:, n:].copy()


def _select_pivot_row(
    work: ArrayF,
    column: int,
    start_row: int,
    n_rows: int,
    *,
    pivoting: bool,
) -> tuple[int, bool]:
    """Return the pivot row index and whether a swap is required."""
    if pivoting:
        pivot_row = start_row + int(np.argmax(np.abs(work[start_row:n_rows, column])))
    else:
        pivot_row = start_row
    return pivot_row, pivot_row != start_row


def _reduce_augmented(
    augmented: ArrayF,
    *,
    pivot_tol: float,
    pivoting: bool,
    record_steps: bool,
) -> EliminationResult:
    """In-place Gauss–Jordan reduction of a general augmented matrix."""
    work = augmented.copy()
    n = work.shape[0]
    steps: list[EliminationStep] = []
    det_sign = 1
    det_mag = 1.0
    n_swaps = 0

    for col in range(n):
        pivot_row, needs_swap = _select_pivot_row(
            work, col, col, n, pivoting=pivoting
        )
        if needs_swap:
            work[[col, pivot_row]] = work[[pivot_row, col]]
            det_sign *= -1
            n_swaps += 1

        pivot = work[col, col]
        if abs(pivot) < pivot_tol:
            raise np.linalg.LinAlgError(
                f"Matrix is singular (or nearly singular) at column {col}: "
                f"|pivot| = {abs(pivot):.3e} < {pivot_tol:.3e}."
            )

        det_mag *= float(pivot)
        work[col, :] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = work[row, col]
            if factor != 0.0:
                work[row, :] -= factor * work[col, :]

        if record_steps:
            steps.append(
                EliminationStep(
                    column=col,
                    pivot_row=pivot_row,
                    pivot_value=float(pivot),
                    swapped=needs_swap,
                    matrix=work.copy(),
                )
            )

    return EliminationResult(
        rref=work,
        determinant=det_sign * det_mag,
        n_swaps=n_swaps,
        steps=steps,
    )


def gauss_jordan_eliminate(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
    record_steps: bool = False,
) -> EliminationResult:
    """Reduce ``[A | b]`` to RREF by Gauss–Jordan elimination.

    Parameters
    ----------
    A:
        Square coefficient matrix of shape ``(n, n)``.
    b:
        Right-hand side of shape ``(n,)``, ``(n, 1)``, or ``(n, m)``.
    pivot_tol:
        Absolute tolerance below which a pivot is treated as singular.
    pivoting:
        If ``True`` (default), use partial pivoting. If ``False``, use the
        current diagonal entry and fail when it is too small.
    record_steps:
        If ``True``, store a snapshot after each pivot column.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")

    if b.ndim == 1:
        if b.shape[0] != A.shape[0]:
            raise ValueError("b must have length equal to A.shape[0].")
        rhs = b.reshape(-1, 1)
    elif b.ndim == 2:
        if b.shape[0] != A.shape[0]:
            raise ValueError("b must have as many rows as A.")
        rhs = b
    else:
        raise ValueError("b must be 1-D or 2-D.")

    augmented = np.hstack([A, rhs])
    return _reduce_augmented(
        augmented,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        record_steps=record_steps,
    )


def invert(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> tuple[ArrayF, float]:
    """Compute ``A^{-1}`` by reducing ``[A | I]`` to ``[I | A^{-1}]``.

    Returns
    -------
    inv_A, det_A
        The inverse matrix and the determinant accumulated from the pivots.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")

    n = A.shape[0]
    result = gauss_jordan_eliminate(
        A,
        np.eye(n),
        pivot_tol=pivot_tol,
        pivoting=pivoting,
    )
    return result.right_block, result.determinant


def determinant(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> float:
    """Return ``det(A)`` from the pivots of a Gauss–Jordan reduction of ``[A | 0]``."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    result = gauss_jordan_eliminate(
        A,
        np.zeros(n),
        pivot_tol=pivot_tol,
        pivoting=pivoting,
    )
    return result.determinant


def reduced_row_echelon_form(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> ArrayF:
    """Compute the RREF of a single matrix ``A`` (no right-hand side)."""
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    work = A.copy()
    row = 0

    for col in range(m):
        if row >= n:
            break

        pivot_row, needs_swap = _select_pivot_row(
            work, col, row, n, pivoting=pivoting
        )
        if abs(work[pivot_row, col]) < pivot_tol:
            continue

        if needs_swap:
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
