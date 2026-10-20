"""From-scratch Gauss–Jordan elimination for dense linear systems."""

from gauss_jordan.core import (
    EliminationResult,
    EliminationStep,
    determinant,
    gauss_jordan_eliminate,
    invert,
    reduced_row_echelon_form,
)
from gauss_jordan.solve import LinearSolveResult, solve, solve_with_steps

__all__ = [
    "EliminationResult",
    "EliminationStep",
    "determinant",
    "gauss_jordan_eliminate",
    "invert",
    "reduced_row_echelon_form",
    "solve",
    "solve_with_steps",
    "LinearSolveResult",
]

__version__ = "1.1.0"
