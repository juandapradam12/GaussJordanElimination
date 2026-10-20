# Architecture

The repository separates the **numerical linear algebra** from the **physical
model** that produces the system $A\mathbf{x}=\mathbf{b}$.

```text
gauss_jordan/          # solver library (problem-agnostic)
  core.py              # RREF / Gauss–Jordan with partial pivoting
  solve.py             # high-level solve + residual diagnostics

models/                # physics → linear algebra
  heat_conduction.py   # 1-D steady heat rod (finite differences)

scripts/
  generate_figures.py  # reproducible README figures

tests/                 # solver and model checks
figures/               # generated plots referenced by the README
docs/                  # mathematical notes
```

## Design choices

- **Single responsibility.** `gauss_jordan` never knows about temperatures or
  rods; `models` never implements elimination.
- **Explicit assemblies.** Boundary conditions enter only through the load
  vector $\mathbf{b}$, keeping $A$ a pure discrete Laplacian.
- **Diagnostics at the boundary.** `solve()` returns residual norms and
  $\kappa_2(A)$ alongside $\mathbf{x}$, so verification is part of the API
  rather than an afterthought.
- **Reproducible figures.** Every plot in the README is produced by
  `scripts/generate_figures.py` from the same code path used in tests.
