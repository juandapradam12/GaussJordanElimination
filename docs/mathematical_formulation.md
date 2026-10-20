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

## Partial pivoting

Without pivoting, a zero (or tiny) diagonal entry aborts or destabilizes the
process even when $A$ is nonsingular. Choosing the largest available pivot at
each step avoids unnecessary division by small numbers and improves numerical
stability on dense systems.

## Diagnostics

After computing $x$, the residual

$$
r = A x - b, \qquad
\frac{\|r\|_2}{\|b\|_2}
$$

measures how faithfully the discrete equations are satisfied. The matrix
2-norm condition number $\kappa_2(A)$ indicates how sensitive $x$ is to
perturbations in $A$ or $b$; large $\kappa_2$ warns that even a tiny residual
need not imply a tiny forward error.
