# Architecture

```text
gauss_jordan/
  core.py      # RREF kernel, invert(), determinant(), step history
  solve.py     # solve() / solve_with_steps() + residual diagnostics

scripts/
  generate_figures.py
  run_case_study.py

tests/
figures/
docs/
```

## Design choices

- **Single reduction kernel.** `gauss_jordan_eliminate` always returns an
  `EliminationResult` (RREF, determinant, swap count, optional step log).
  Solving $Ax=b$ and inverting $A$ are thin wrappers around the same path.
- **Pivoting flag.** `pivoting=True` (default) uses partial pivoting;
  `pivoting=False` keeps the diagonal entry for controlled comparisons.
- **Diagnostics at the boundary.** `solve()` reports residual norms and
  $\kappa_2(A)$ alongside $x$.
- **Reproducible figures.** Every plot in the README comes from
  `scripts/generate_figures.py`.
