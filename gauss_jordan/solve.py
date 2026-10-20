"""High-level interface for solving ``A x = b`` (and ``A X = B``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gauss_jordan.core import EliminationResult, gauss_jordan_eliminate

ArrayF = NDArray[np.floating]


@dataclass(frozen=True)
class LinearSolveResult:
    """Solution of a linear system together with residual diagnostics."""

    x: ArrayF
    residual: ArrayF
    residual_norm: float
    condition_number: float
    relative_residual: float
    rref: ArrayF
    determinant: float
    n_swaps: int
    rank: int
    n_flops: int
    growth_factor: float

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_rhs(self) -> int:
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])


def _pack_result(A: ArrayF, b: ArrayF, history: EliminationResult) -> LinearSolveResult:
    x = history.solution
    residual = A @ x - b

    if residual.ndim == 1:
        residual_norm = float(np.linalg.norm(residual, ord=2))
        b_norm = float(np.linalg.norm(b, ord=2))
    else:
        residual_norm = float(np.linalg.norm(residual, ord="fro"))
        b_norm = float(np.linalg.norm(b, ord="fro"))

    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm

    return LinearSolveResult(
        x=x,
        residual=residual,
        residual_norm=residual_norm,
        condition_number=float(np.linalg.cond(A, p=2)),
        relative_residual=relative_residual,
        rref=history.rref,
        determinant=history.determinant,
        n_swaps=history.n_swaps,
        rank=history.rank,
        n_flops=history.n_flops,
        growth_factor=history.growth_factor,
    )


def solve(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> LinearSolveResult:
    """Solve ``A x = b`` or ``A X = B`` with residual / conditioning diagnostics.

    Parameters
    ----------
    b:
        Shape ``(n,)`` for a single right-hand side, or ``(n, k)`` for ``k``
        right-hand sides solved in one elimination pass.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")
    if b.ndim not in (1, 2) or b.shape[0] != A.shape[0]:
        raise ValueError("b must have shape (n,) or (n, k) matching A.")

    history = gauss_jordan_eliminate(
        A,
        b,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        record_steps=False,
        allow_rank_deficient=False,
    )
    return _pack_result(A, b, history)


def solve_with_steps(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
    pivoting: bool = True,
) -> tuple[LinearSolveResult, EliminationResult]:
    """Solve ``A x = b`` (or ``A X = B``) and return the elimination history."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")
    if b.ndim not in (1, 2) or b.shape[0] != A.shape[0]:
        raise ValueError("b must have shape (n,) or (n, k) matching A.")

    history = gauss_jordan_eliminate(
        A,
        b,
        pivot_tol=pivot_tol,
        pivoting=pivoting,
        record_steps=True,
        allow_rank_deficient=False,
    )
    return _pack_result(A, b, history), history
