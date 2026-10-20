#!/usr/bin/env python3
"""Run a compact Gauss–Jordan solve and print diagnostics."""

from __future__ import annotations

import numpy as np

from gauss_jordan import invert, solve


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
    inv_A, det_A = invert(A)

    print("Gauss–Jordan elimination (from scratch)")
    print(f"  solution x           : {np.array2string(result.x, precision=6)}")
    print(f"  NumPy reference      : {np.array2string(reference, precision=6)}")
    print(f"  max |x - x_ref|      : {np.max(np.abs(result.x - reference)):.3e}")
    print(f"  det(A)               : {result.determinant:.6g}  (invert path: {det_A:.6g})")
    print(f"  relative residual    : {result.relative_residual:.3e}")
    print(f"  condition κ₂(A)      : {result.condition_number:.3e}")
    print(f"  ‖A A⁻¹ - I‖_∞        : {np.max(np.abs(A @ inv_A - np.eye(3))):.3e}")


if __name__ == "__main__":
    main()
