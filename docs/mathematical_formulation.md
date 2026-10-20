# Mathematical formulation

## Continuous problem

Consider a thin conducting rod of length $L$ with thermal conductivity $\kappa > 0$.
In the steady state, the temperature $T(x)$ satisfies the one-dimensional heat
equation with a volumetric source $q(x)$ and Dirichlet ends:

$$
-\kappa\, T''(x) = q(x), \qquad x \in (0, L),
$$

$$
T(0) = T_L, \qquad T(L) = T_R.
$$

When $q(x) \equiv q_0$ is constant, the exact solution is the parabola

$$
T(x) = T_L + (T_R - T_L)\frac{x}{L} + \frac{q_0}{2\kappa}\, x(L - x).
$$

## Finite-difference discretization

Introduce $n$ interior nodes with spacing $h = L/(n+1)$:

$$
x_i = i h, \qquad i = 1,\ldots,n.
$$

The second derivative is replaced by the central second difference

$$
T''(x_i) \approx \frac{T_{i-1} - 2T_i + T_{i+1}}{h^2},
$$

where $T_0 = T_L$ and $T_{n+1} = T_R$ are known. Substituting into the
differential equation produces the linear system

$$
A\, \mathbf{T} = \mathbf{b},
$$

with the symmetric positive-definite tridiagonal matrix

$$
A = \frac{\kappa}{h^2}
\begin{pmatrix}
2 & -1 & & \\
-1 & 2 & -1 & \\
& \ddots & \ddots & \ddots \\
& & -1 & 2
\end{pmatrix},
$$

and right-hand side

$$
b_i = q(x_i)
+ \delta_{i1}\,\frac{\kappa T_L}{h^2}
+ \delta_{in}\,\frac{\kappa T_R}{h^2}.
$$

The boundary contributions appear only in the first and last entries of
$\mathbf{b}$: they “lift” the known Dirichlet data into the load vector.

## Gauss–Jordan elimination

Given a nonsingular $n \times n$ matrix $A$ and a vector $\mathbf{b}$, form the
augmented matrix $[A\mid \mathbf{b}]$. Gauss–Jordan elimination with partial
pivoting performs, for each column $k = 1,\ldots,n$:

1. **Pivot selection.** Swap row $k$ with the row $p \ge k$ that maximizes
   $|a_{pk}|$.
2. **Normalization.** Divide row $k$ by the pivot so that $a_{kk} = 1$.
3. **Elimination.** Subtract suitable multiples of row $k$ from every other
   row, clearing column $k$ above and below the pivot.

The result is the reduced row echelon form

$$
[I \mid \mathbf{x}],
$$

so the last column is the unique solution $\mathbf{x} = A^{-1}\mathbf{b}$.

Partial pivoting improves numerical stability relative to naïve elimination
by avoiding unnecessarily small pivots.

## Diagnostics

After computing $\mathbf{x}$, the algebraic residual

$$
\mathbf{r} = A\mathbf{x} - \mathbf{b}, \qquad
\frac{\|\mathbf{r}\|_2}{\|\mathbf{b}\|_2}
$$

measures how faithfully the discrete equations are satisfied. Separately, the
pointwise error against the exact parabola (when $q$ is constant) measures the
$O(h^2)$ truncation error of the finite-difference scheme. The matrix
2-norm condition number $\kappa_2(A)$ grows like $O(n^2)$ for this Laplacian
operator, which is the expected price of refining the mesh.
