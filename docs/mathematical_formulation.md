# Mathematical formulation

## The linear problem

Given a square coefficient matrix $A \in \mathbb{R}^{n \times n}$ and a
right-hand side $b \in \mathbb{R}^{n}$, the task is to find
$x \in \mathbb{R}^{n}$ such that

$$
A x = b.
$$

If $A$ is nonsingular, the solution exists and is unique:
$x = A^{-1} b$.

## Augmented matrix

Gauss–Jordan elimination works on the augmented matrix

$$
[A \mid b]
=
\begin{pmatrix}
a_{11} & \cdots & a_{1n} & b_1 \\
\vdots & \ddots & \vdots & \vdots \\
a_{n1} & \cdots & a_{nn} & b_n
\end{pmatrix}.
$$

Elementary row operations (swap, scale, add a multiple of one row to another)
do not change the solution set of the underlying system.

## Algorithm

For each column $k = 1,\ldots,n$:

1. **Partial pivoting.** Among rows $i = k,\ldots,n$, select the entry of
   largest magnitude in column $k$ and swap that row with row $k$.
2. **Normalization.** Divide row $k$ by the pivot so that the diagonal entry
   becomes $1$.
3. **Elimination.** For every other row $i \neq k$, subtract
   $a_{ik}$ times row $k$, clearing column $k$ above and below the pivot.

When $A$ is nonsingular, the result is the reduced row echelon form

$$
[I \mid x],
$$

so the last column is exactly the solution vector.

## Determinant from pivots

Let $p_k$ be the pivot used in column $k$ **before** normalization, and let
$s$ be the number of row swaps. Then

$$
\det(A) = (-1)^{s}\prod_{k=1}^{n} p_k.
$$

## Inverse from $[A \mid I]$

The same reduction applied to the double-width augmentation

$$
[A \mid I]
$$

produces

$$
[I \mid A^{-1}],
$$

so the right block is the inverse. Equivalently, one solves $A X = I$ for all
$n$ canonical right-hand sides at once.

## Partial pivoting and stability

Without pivoting, a zero or tiny diagonal entry aborts or destabilizes the
process even when $A$ is nonsingular. Choosing the largest available pivot at
each step avoids unnecessary division by small numbers and reduces amplified
round-off in the computed $x$.

## Residual versus forward error

After computing $\hat x$, the residual

$$
r = A\hat x - b, \qquad
\frac{\|r\|_2}{\|b\|_2}
$$

measures consistency with the discrete equations. The forward error
$\|\hat x - x\|_2 / \|x\|_2$ is controlled in the worst case by
$\kappa_2(A)\,\|r\|/\|b\|$. On severely ill-conditioned matrices (e.g. Hilbert
$H_n$), a tiny residual can coexist with a large forward error.

## Complexity

A dense $n \times n$ Gauss–Jordan pass performs $\Theta(n^3)$ arithmetic
operations — the same leading-order cost class as Gaussian elimination with
back-substitution, with a larger constant because both upper and lower
entries are cleared.
