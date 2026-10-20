#!/usr/bin/env python3
"""Run the heat-rod case study and print a concise numerical summary."""

from __future__ import annotations

import numpy as np

from gauss_jordan import solve
from models import assemble_heat_rod


def main() -> None:
    problem = assemble_heat_rod(
        length=1.0,
        conductivity=1.0,
        n_interior=40,
        left_temperature=20.0,
        right_temperature=20.0,
        source=80.0,
    )
    result = solve(problem.A, problem.b)
    exact = problem.exact_quadratic_solution()
    assert exact is not None

    max_err = float(np.max(np.abs(result.x - exact)))
    print("Steady-state heat conduction — Gauss–Jordan solve")
    print(f"  interior nodes n     : {problem.n_interior}")
    print(f"  mesh size h          : {problem.h:.6f}")
    print(f"  condition κ₂(A)      : {result.condition_number:.6e}")
    print(f"  relative residual    : {result.relative_residual:.6e}")
    print(f"  max error vs exact   : {max_err:.6e}")
    print(f"  T range (interior)   : [{result.x.min():.4f}, {result.x.max():.4f}]")


if __name__ == "__main__":
    main()
