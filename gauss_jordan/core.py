"""From-scratch Gauss–Jordan elimination with partial pivoting.

Given a nonsingular square system ``A x = b``, the algorithm transforms the
augmented matrix ``[A | b]`` into reduced row echelon form ``[I | x]``.
The same reduction applied to ``[A | I]`` yields ``[I | A^{-1}]``.

When a column has no admissible pivot the implementation can either raise
(full-rank solve path) or skip the column and continue, returning the
numerical rank together with the RREF (rank-deficient path).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]


@dataclass
class EliminationStep:
    """Snapshot after one successful pivot is processed."""

    column: int
    pivot_row: int
    pivot_value: float
    swapped: bool
    matrix: ArrayF
    max_abs_entry: float


@dataclass
class EliminationResult:
    """RREF of an augmented matrix together with elimination metadata."""

    rref: ArrayF
    n_rows: int
    n_cols_a: int
    rank: int
    pivot_columns: list[int]
    determinant: float
    n_swaps: int
    n_flops: int
    growth_factor: float
    steps: list[EliminationStep] = field(default_factory=list)

    @property
    def is_full_rank(self) -> bool:
        return self.rank == self.n_cols_a == self.n_rows

    @property
    def solution(self) -> ArrayF:
        """Right-hand block after reduction (vector if a single RHS)."""
        rhs = self.right_block
        if rhs.shape[1] == 1:
            return rhs[:, 0].copy()
        return rhs.copy()

    @property
    def left_block(self) -> ArrayF:
        return self.rref[:, : self.n_cols_a].copy()

    @property
    def right_block(self) -> ArrayF:
        return self.rref[:, self.n_cols_a :].copy()


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
    n_cols_a: int,
    pivot_tol: float,
    pivoting: bool,
    record_steps: bool,
    allow_rank_deficient: bool,
) -> EliminationResult:
    """Gauss–Jordan reduction of ``[A | …]`` with optional rank deficiency."""
    work = np.array(augmented, dtype=float, copy=True)
    n_rows, n_cols = work.shape
    if n_cols_a > n_cols:
        raise ValueError("n_cols_a cannot exceed the augmented width.")

    steps: list[EliminationStep] = []
    pivot_columns: list[int] = []
    det_sign = 1
    det_mag = 1.0
    n_swaps = 0
    n_flops = 0

    a0 = work[:, :n_cols_a]
    max_a0 = float(np.max(np.abs(a0))) if a0.size else 0.0
    max_during = max_a0

    row = 0
    for col in range(n_cols_a):
        if row >= n_rows:
            break

        pivot_row, needs_swap = _select_pivot_row(
            work, col, row, n_rows, pivoting=pivoting
        )
        pivot = work[pivot_row, col]

        if abs(pivot) < pivot_tol:
            if allow_rank_deficient:
                continue
            raise np.linalg.LinAlgError(
                f"Matrix is singular (or nearly singular) at column {col}: "
                f"|pivot| = {abs(pivot):.3e} < {pivot_tol:.3e}."
            )

        if needs_swap:
            work[[row, pivot_row]] = work[[pivot_row, row]]
            det_sign *= -1
            n_swaps += 1
            # A row swap touches  n_cols entries (bookkeeping, not arithmetic flops).

        pivot = work[row, col]
        det_mag *= float(pivot)

        # Normalize pivot row: n_cols divisions.
        work[row, :] /= pivot
        n_flops += n_cols

        for r in range(n_rows):
            if r == row:
                continue
            factor = work[r, col]
            if factor != 0.0:
                # n_cols mults + n_cols adds.
                work[r, :] -= factor * work[row, :]
                n_flops += 2 * n_cols

        pivot_columns.append(col)
        block_max = float(np.max(np.abs(work[:, :n_cols_a])))
        max_during = max(max_during, block_max)

        if record_steps:
            steps.append(
                EliminationStep(
                    column=col,
                    pivot_row=pivot_row,
                    pivot_value=float(pivot),
                    swapped=needs_swap,
                    matrix=work.copy(),
                    max_abs_entry=block_max,
                )
            )

        row += 1

    rank = len(pivot_columns)
    if rank < n_cols_a:
        determinant = 0.0
    else:
        determinant = det_sign * det_mag

    growth_factor = max_during / max_a0 if max_a0 > 0.0 else 0.0

    return EliminationResult(
        rref=work,
        n_rows=n_rows,
        n_cols_a=n_cols_a,
        rank=rank,
        pivot_columns=pivot_columns,
        determinant=determinant,
        n_swaps=n_swaps,
        n_flops=n_flops,
        growth_factor=growth_factor,
        steps=steps,
    )


def gauss_jordan_eliminate(
    A: ArrayF,
    b: ArrayF | None = None,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
    record_steps: bool = False,
    allow_rank_deficient: bool = False,
) -> EliminationResult:
    """Reduce ``[A | b]`` (or ``A`` alone) to RREF.

    Parameters
    ----------
    A:
        Coefficient matrix of shape ``(n, n)`` for solves, or ``(n, m)`` when
        only RREF is requested with ``b is None``.
    b:
        Right-hand side of shape ``(n,)``, ``(n, 1)``, or ``(n, k)``. Omit to
        reduce ``A`` alone.
    allow_rank_deficient:
        If ``False`` (default), raise when a required pivot is missing.
        If ``True``, skip zero columns and return the numerical rank.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D array.")

    n_rows, n_cols_a = A.shape

    if b is None:
        augmented = A.copy()
    else:
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            if b.shape[0] != n_rows:
                raise ValueError("b must have length equal to A.shape[0].")
            rhs = b.reshape(-1, 1)
        elif b.ndim == 2:
            if b.shape[0] != n_rows:
                raise ValueError("b must have as many rows as A.")
            rhs = b
        else:
            raise ValueError("b must be 1-D or 2-D.")
        augmented = np.hstack([A, rhs])

    return _reduce_augmented(
        augmented,
        n_cols_a=n_cols_a,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        record_steps=record_steps,
        allow_rank_deficient=allow_rank_deficient,
    )


def invert(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> tuple[ArrayF, float]:
    """Compute ``A^{-1}`` by reducing ``[A | I]`` to ``[I | A^{-1}]``."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")

    n = A.shape[0]
    result = gauss_jordan_eliminate(
        A,
        np.eye(n),
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        allow_rank_deficient=False,
    )
    return result.right_block, result.determinant


def determinant(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> float:
    """Return ``det(A)`` from the pivots of a Gauss–Jordan reduction."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")
    result = gauss_jordan_eliminate(
        A,
        np.zeros(A.shape[0]),
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        allow_rank_deficient=True,
    )
    return result.determinant


def matrix_rank(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> int:
    """Numerical rank of ``A`` from the number of accepted RREF pivots."""
    return gauss_jordan_eliminate(
        A,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        allow_rank_deficient=True,
    ).rank


def reduced_row_echelon_form(
    A: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> ArrayF:
    """Compute the RREF of ``A``, allowing rank deficiency."""
    return gauss_jordan_eliminate(
        A,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        allow_rank_deficient=True,
    ).rref
