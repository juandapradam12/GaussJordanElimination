"""Gauss–Jordan elimination for dense linear systems."""

from gauss_jordan.core import gauss_jordan_eliminate, reduced_row_echelon_form
from gauss_jordan.solve import LinearSolveResult, solve

__all__ = [
    "gauss_jordan_eliminate",
    "reduced_row_echelon_form",
    "solve",
    "LinearSolveResult",
]

__version__ = "1.0.0"
