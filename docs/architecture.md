# Architecture

The package separates the **elimination kernel** from the **solve API** that
adds residual diagnostics.

```text
gauss_jordan/
  core.py      # RREF / Gauss–Jordan with partial pivoting (+ optional step log)
  solve.py     # solve(), solve_with_steps(), LinearSolveResult

scripts/
  generate_figures.py   # reproducible README figures
  run_case_study.py     # short numerical summary

tests/                  # correctness, pivoting, history recording
figures/                # plots referenced by the README
docs/                   # mathematical notes
```

## Design choices

- **From scratch.** Elimination is implemented explicitly with NumPy arrays;
  there is no call to `numpy.linalg.solve` inside the solver path.
- **Optional history.** `record_steps=True` (or `solve_with_steps`) stores a
  snapshot after each pivot column, which drives the elimination-step figures
  without changing the default return type.
- **Diagnostics at the boundary.** `solve()` returns residual norms and
  $\kappa_2(A)$ alongside $x$, so verification is part of the public API.
- **Reproducible figures.** Every plot in the README is produced by
  `scripts/generate_figures.py` from the same code path used in tests.
