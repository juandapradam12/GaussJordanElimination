# Architecture

```text
gauss_jordan/
  core.py      # rank-aware RREF kernel, invert, determinant, matrix_rank
  solve.py     # solve() / solve_with_steps() for one or many RHS

scripts/
  generate_figures.py
  run_case_study.py

tests/
figures/
docs/
```

## Design choices

- **Single reduction kernel.** `gauss_jordan_eliminate` always returns an
  `EliminationResult` carrying RREF, rank, pivot columns, determinant, swap
  count, flop tally, and growth factor.
- **Rank policy.** `allow_rank_deficient=False` (solve / invert) raises on a
  missing pivot. `True` (RREF / rank / det) skips zero columns and continues.
- **Multi-RHS.** A 2-D `b` with shape `(n, k)` is reduced in one pass; `solve`
  returns `x` with shape `(n, k)`.
- **Pivoting flag.** `pivoting=True` by default; `False` keeps the diagonal
  entry for controlled comparisons.
- **Reproducible figures.** Every plot in the README comes from
  `scripts/generate_figures.py`.
