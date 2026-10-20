#!/usr/bin/env python3
"""Run a compact Gauss–Jordan solve and print diagnostics."""

from __future__ import annotations

import numpy as np

from gauss_jordan import solve


def main() -> None:
    A = np.array(
        [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]
    )
    b = np.array([8.0, -11.0, -3.0])

    result = solve(A, b)
    reference = np.linalg.solve(A, b)

    print("Gauss–Jordan elimination (from scratch)")
    print(f"  solution x           : {np.array2string(result.x, precision=6)}")
    print(f"  NumPy reference      : {np.array2string(reference, precision=6)}")
    print(f"  max |x - x_ref|      : {np.max(np.abs(result.x - reference)):.3e}")
    print(f"  relative residual    : {result.relative_residual:.3e}")
    print(f"  condition κ₂(A)      : {result.condition_number:.3e}")


if __name__ == "__main__":
    main()
