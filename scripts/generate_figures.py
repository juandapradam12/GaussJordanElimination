#!/usr/bin/env python3
"""Generate workflow figures for the Gauss–Jordan case study.

Covers exploratory inspection of ``A x = b``, the elimination process,
the role of pivoting, and verification of the recovered solution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from gauss_jordan import gauss_jordan_eliminate, solve, solve_with_steps

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"

PALETTE = {
    "bg": "#f7f4ef",
    "ink": "#1c2430",
    "accent": "#c45c26",
    "steel": "#2f5d8c",
    "muted": "#6b7280",
    "grid": "#d8d2c8",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "font.family": "DejaVu Serif",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": PALETTE["bg"],
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _demo_system() -> tuple[np.ndarray, np.ndarray]:
    """A small nonsingular system used across several figures."""
    A = np.array(
        [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]
    )
    b = np.array([8.0, -11.0, -3.0])
    return A, b


def figure_01_problem_statement() -> None:
    A, b = _demo_system()
    augmented = np.hstack([A, b.reshape(-1, 1)])

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4))

    im0 = axes[0].imshow(A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    axes[0].set_title(r"Coefficient matrix $A$")
    for (i, j), val in np.ndenumerate(A):
        axes[0].text(j, i, f"{val:g}", ha="center", va="center", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(np.arange(1, len(b) + 1), b, color=PALETTE["steel"], width=0.55)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xlabel(r"equation index")
    axes[1].set_ylabel(r"$b_i$")
    axes[1].set_title(r"Right-hand side $b$")
    axes[1].axhline(0, color=PALETTE["ink"], lw=0.8)
    axes[1].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    im2 = axes[2].imshow(augmented, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    axes[2].axvline(2.5, color=PALETTE["ink"], lw=1.2, ls="--")
    axes[2].set_title(r"Augmented matrix $[A\,|\,b]$")
    for (i, j), val in np.ndenumerate(augmented):
        axes[2].text(j, i, f"{val:g}", ha="center", va="center", fontsize=9)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(r"The linear problem: solve $A x = b$", y=1.05, fontsize=14)
    fig.tight_layout()
    _save(fig, "01_problem_statement.png")


def figure_02_exploratory_conditioning() -> None:
    rng = np.random.default_rng(7)
    sizes = [5, 10, 20, 40, 80]
    conds_hilbert = []
    conds_random = []
    residuals = []

    for n in sizes:
        # Hilbert matrices are classically ill-conditioned.
        H = np.fromfunction(lambda i, j: 1.0 / (i + j + 1.0), (n, n), dtype=float)
        conds_hilbert.append(np.linalg.cond(H))

        A = rng.normal(size=(n, n))
        A = A @ A.T + n * 0.05 * np.eye(n)
        conds_random.append(np.linalg.cond(A))
        b = rng.normal(size=n)
        residuals.append(solve(A, b).relative_residual)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    axes[0].semilogy(sizes, conds_hilbert, "o-", color=PALETTE["accent"], lw=2, label="Hilbert")
    axes[0].semilogy(sizes, conds_random, "s-", color=PALETTE["steel"], lw=2, label="SPD random")
    axes[0].set_xlabel(r"dimension $n$")
    axes[0].set_ylabel(r"$\kappa_2(A)$")
    axes[0].set_title("Exploratory conditioning")
    axes[0].legend(frameon=False)
    axes[0].grid(True, color=PALETTE["grid"], lw=0.7)

    axes[1].semilogy(sizes, residuals, "o-", color=PALETTE["steel"], lw=2)
    axes[1].set_xlabel(r"dimension $n$")
    axes[1].set_ylabel(r"$\|Ax-b\|_2/\|b\|_2$")
    axes[1].set_title("Residual on well-conditioned SPD draws")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.7)

    fig.suptitle("Before solving: how hard is the system?", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "02_exploratory_conditioning.png")


def figure_03_elimination_steps() -> None:
    A, b = _demo_system()
    augmented0 = np.hstack([A, b.reshape(-1, 1)])
    _, history = solve_with_steps(A, b)

    panels = [augmented0] + [step.matrix for step in history.steps]
    titles = [r"$[A\,|\,b]$"] + [rf"after column ${s.column + 1}$" for s in history.steps]

    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 3.2))
    vmax = max(float(np.max(np.abs(p))) for p in panels)

    for ax, mat, title in zip(axes, panels, titles):
        ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.axvline(2.5, color=PALETTE["ink"], lw=1.0, ls="--")
        ax.set_title(title, fontsize=11)
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))

    fig.suptitle("Gauss–Jordan reduction of the augmented matrix", y=1.08, fontsize=14)
    fig.tight_layout()
    _save(fig, "03_elimination_steps.png")


def figure_04_pivoting() -> None:
    # Without pivoting the first pivot is zero — the system is still nonsingular.
    A = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, -1.0],
        ]
    )
    b = np.array([1.0, 2.0, 1.0])

    with_pivot = solve(A, b)
    reference = np.linalg.solve(A, b)

    # Naïve first step without a swap would divide by A[0,0] = 0.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    axes[0].imshow(A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    for (i, j), val in np.ndenumerate(A):
        color = PALETTE["accent"] if (i, j) == (0, 0) else PALETTE["ink"]
        weight = "bold" if (i, j) == (0, 0) else "normal"
        axes[0].text(j, i, f"{val:g}", ha="center", va="center", color=color, fontweight=weight)
    axes[0].set_title(r"$A$ with a zero diagonal entry")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])

    idx = np.arange(len(reference))
    width = 0.35
    axes[1].bar(idx - width / 2, reference, width, color=PALETTE["muted"], label="NumPy")
    axes[1].bar(idx + width / 2, with_pivot.x, width, color=PALETTE["accent"], label="Gauss–Jordan + pivot")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([rf"$x_{i+1}$" for i in idx])
    axes[1].set_title("Partial pivoting recovers the solution")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    fig.suptitle("Why partial pivoting matters", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "04_pivoting.png")


def figure_05_solution_and_residual() -> None:
    A, b = _demo_system()
    result = solve(A, b)
    reference = np.linalg.solve(A, b)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    idx = np.arange(len(result.x))
    width = 0.35
    axes[0].bar(idx - width / 2, reference, width, color=PALETTE["muted"], label="NumPy reference")
    axes[0].bar(idx + width / 2, result.x, width, color=PALETTE["accent"], label="Gauss–Jordan")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([rf"$x_{i+1}$" for i in idx])
    axes[0].set_title("Recovered solution")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    axes[1].stem(idx + 1, result.residual, linefmt=PALETTE["steel"], markerfmt="o", basefmt=" ")
    axes[1].axhline(0.0, color=PALETTE["ink"], lw=0.8)
    axes[1].set_xlabel("equation")
    axes[1].set_ylabel(r"$(Ax - b)_i$")
    axes[1].set_title(rf"Residual ($\|r\|_2 = {result.residual_norm:.1e}$)")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.7)

    fig.suptitle("Result: solution vector and algebraic residual", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "05_solution_residual.png")


def figure_06_rref_identity() -> None:
    A, b = _demo_system()
    result = solve(A, b)
    left = result.rref[:, :-1]
    right = result.rref[:, -1]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    im0 = axes[0].imshow(left, cmap="Blues", vmin=0.0, vmax=1.0)
    for (i, j), val in np.ndenumerate(left):
        axes[0].text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=12)
    axes[0].set_title(r"Left block of RREF $\approx I$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(np.arange(1, len(right) + 1), right, color=PALETTE["accent"], width=0.5)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xlabel(r"component")
    axes[1].set_ylabel(r"$x_i$")
    axes[1].set_title(r"Last column $= x$")
    axes[1].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    fig.suptitle(r"Reduced row echelon form $[I\,|\,x]$", y=1.04, fontsize=14)
    fig.tight_layout()
    _save(fig, "06_rref_identity.png")


def figure_07_workflow_summary() -> None:
    A, b = _demo_system()
    result, history = solve_with_steps(A, b)
    augmented0 = np.hstack([A, b.reshape(-1, 1)])

    fig = plt.figure(figsize=(10.2, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(augmented0, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    ax0.axvline(2.5, color=PALETTE["ink"], lw=1.0, ls="--")
    ax0.set_title(r"1. Form $[A\,|\,b]$")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    mid = history.steps[0].matrix
    ax1.imshow(mid, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    ax1.axvline(2.5, color=PALETTE["ink"], lw=1.0, ls="--")
    ax1.set_title("2. Pivot, normalize, eliminate")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(result.rref[:, :-1], cmap="Blues", vmin=0, vmax=1)
    ax2.set_title(r"3. Reach $[I\,|\,x]$")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(np.arange(1, len(result.x) + 1), result.x, color=PALETTE["accent"], width=0.5)
    ax3.set_title("4. Read the solution")
    ax3.set_xlabel(r"component")
    ax3.set_ylabel(r"$x_i$")
    ax3.grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    fig.suptitle(
        r"Workflow: augmented system $\rightarrow$ RREF $\rightarrow$ solution",
        fontsize=14,
        y=0.98,
    )
    _save(fig, "07_workflow_summary.png")


def main() -> None:
    _style()
    figure_01_problem_statement()
    figure_02_exploratory_conditioning()
    figure_03_elimination_steps()
    figure_04_pivoting()
    figure_05_solution_and_residual()
    figure_06_rref_identity()
    figure_07_workflow_summary()


if __name__ == "__main__":
    main()
