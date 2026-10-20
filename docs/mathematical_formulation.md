# Mathematical formulation

## The linear problem

Given a square coefficient matrix $A \in \mathbb{R}^{n \times n}$ and a
right-hand side $b \in \mathbb{R}^{n}$, the task is to find
$x \in \mathbb{R}^{n}$ such that

$$
A x = b.
$$

If $A$ is nonsingular, the solution exists and is unique:
$x = A^{-1} b$. For several right-hand sides collected as columns of
$B \in \mathbb{R}^{n \times k}$, the same statement reads $A X = B$.

## Augmented matrix

Gauss–Jordan elimination works on the augmented matrix $[A \mid b]$ (or
$[A \mid B]$). Elementary row operations do not change the solution set.

## Algorithm

Maintain a pivot-row index $r$, starting at $0$. For each column
$c = 0,\ldots,n-1$:

1. **Partial pivoting.** Among rows $i = r,\ldots,n-1$, select the entry of
   largest magnitude in column $c$.
2. If that entry is below a tolerance, **skip** the column (rank-deficient
   path) or **raise** (full-rank solve path).
3. Otherwise swap it into row $r$, **normalize** so the pivot is $1$, and
   **eliminate** above and below. Then set $r \leftarrow r+1$.

The number of accepted pivots is the numerical **rank**. When
$\mathrm{rank}(A)=n$, the left block is $I$ and the right block is the
solution (or the inverse, if the right block started as $I$).

## Determinant from pivots

Let $p_k$ be the $k$-th accepted pivot **before** normalization and $s$ the
number of row swaps. If $\mathrm{rank}(A)=n$,

$$
\det(A) = (-1)^{s}\prod_{k} p_k;
$$

otherwise $\det(A)=0$.

## Growth factor

Let $a_{ij}^{(0)}$ be the entries of $A$ and $a_{ij}^{(\ell)}$ the entries of
the $A$-block after step $\ell$. The growth factor

$$
\rho = \frac{\max_{\ell,i,j}|a_{ij}^{(\ell)}|}{\max_{i,j}|a_{ij}^{(0)}|}
$$

measures intermediate swell. Large $\rho$ warns that round-off may be
amplified even when pivots are nonzero.

## Residual versus forward error

After computing $\hat x$,

$$
r = A\hat x - b, \qquad
\frac{\|r\|_2}{\|b\|_2}
$$

measures consistency with the discrete equations. The forward error is
controlled in the worst case by $\kappa_2(A)\,\|r\|/\|b\|$. On Hilbert
matrices $H_n$, a tiny residual can coexist with a large forward error.

## Complexity

Normalizing a pivot row of width $w$ costs $w$ divisions; clearing one other
row costs $2w$ flops (multiply–add). Summing over all pivots yields
$\Theta(n^3)$ arithmetic for dense full-rank $n \times n$ systems. The
implementation tallies these operations exactly as it runs.
