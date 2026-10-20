"""High-level interface for solving ``A x = b`` via Gauss–Jordan elimination."""

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

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def solve(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
) -> LinearSolveResult:
    """Solve ``A x = b`` and report residual / conditioning diagnostics.

    The relative residual is ``||A x - b||_2 / ||b||_2`` (or ``||A x - b||_2``
    when ``b`` is identically zero).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    rref = gauss_jordan_eliminate(A, b, pivot_tol=pivot_tol)
    assert isinstance(rref, np.ndarray)
    x = rref[:, -1].copy()

    residual = A @ x - b
    residual_norm = float(np.linalg.norm(residual, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
    condition_number = float(np.linalg.cond(A, p=2))

    return LinearSolveResult(
        x=x,
        residual=residual,
        residual_norm=residual_norm,
        condition_number=condition_number,
        relative_residual=relative_residual,
        rref=rref,
    )


def solve_with_steps(
    A: ArrayF,
    b: ArrayF,
    *,
    pivot_tol: float = 1e-12,
) -> tuple[LinearSolveResult, EliminationResult]:
    """Solve ``A x = b`` and also return the recorded elimination history."""
    history = gauss_jordan_eliminate(A, b, pivot_tol=pivot_tol, record_steps=True)
    assert isinstance(history, EliminationResult)

    residual = A @ history.solution - b
    residual_norm = float(np.linalg.norm(residual, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm

    result = LinearSolveResult(
        x=history.solution,
        residual=residual,
        residual_norm=residual_norm,
        condition_number=float(np.linalg.cond(A, p=2)),
        relative_residual=relative_residual,
        rref=history.rref,
    )
    return result, history
