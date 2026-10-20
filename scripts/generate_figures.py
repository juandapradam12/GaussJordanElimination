#!/usr/bin/env python3
"""Generate workflow figures for the Gauss–Jordan case study."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from gauss_jordan import invert, solve, solve_with_steps

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
    A = np.array(
        [
            [2.0, 1.0, -1.0],
            [-3.0, -1.0, 2.0],
            [-2.0, 1.0, 2.0],
        ]
    )
    b = np.array([8.0, -11.0, -3.0])
    return A, b


def _hilbert(n: int) -> np.ndarray:
    return np.fromfunction(lambda i, j: 1.0 / (i + j + 1.0), (n, n), dtype=float)


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
        conds_hilbert.append(np.linalg.cond(_hilbert(n)))
        A = rng.normal(size=(n, n))
        A = A @ A.T + n * 0.05 * np.eye(n)
        conds_random.append(np.linalg.cond(A))
        residuals.append(solve(A, rng.normal(size=n)).relative_residual)

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


def figure_04_pivoting_accuracy() -> None:
    """Compare no-pivot vs partial pivot on a system with a tiny first pivot."""
    # Small enough that naïve division loses digits; large enough to pass pivot_tol.
    eps = 1e-12
    A = np.array(
        [
            [eps, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, -1.0],
        ]
    )
    # Construct b from a known true solution so forward error is measurable.
    x_true = np.array([1.0, -2.0, 3.0])
    b = A @ x_true

    with_pivot = solve(A, b, pivoting=True)
    without = solve(A, b, pivoting=False)

    err_pivot = float(np.linalg.norm(with_pivot.x - x_true, ord=np.inf))
    err_none = float(np.linalg.norm(without.x - x_true, ord=np.inf))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    axes[0].imshow(A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    for (i, j), val in np.ndenumerate(A):
        label = f"{val:.0e}" if abs(val) < 1e-3 else f"{val:g}"
        color = PALETTE["accent"] if (i, j) == (0, 0) else PALETTE["ink"]
        weight = "bold" if (i, j) == (0, 0) else "normal"
        axes[0].text(j, i, label, ha="center", va="center", color=color, fontweight=weight, fontsize=9)
    axes[0].set_title(r"$A$ with a tiny leading pivot")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])

    labels = ["no pivoting", "partial pivoting"]
    # Floor zeros so a log axis remains readable when a residual underflows.
    floor = 1e-18
    errors = [max(err_none, floor), max(err_pivot, floor)]
    residuals = [
        max(without.relative_residual, floor),
        max(with_pivot.relative_residual, floor),
    ]
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, errors, width, color=PALETTE["accent"], label=r"forward $\|x-\hat x\|_\infty$")
    axes[1].bar(x + width / 2, residuals, width, color=PALETTE["steel"], label=r"rel. residual")
    axes[1].set_yscale("log")
    axes[1].set_ylim(floor, None)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Stability: pivoting vs naïve diagonal")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    fig.suptitle("Partial pivoting as a stability device", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "04_pivoting_accuracy.png")


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


def figure_06_rref_and_inverse() -> None:
    A, b = _demo_system()
    result = solve(A, b)
    inv_A, det_A = invert(A)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))

    axes[0].imshow(result.rref[:, :-1], cmap="Blues", vmin=0.0, vmax=1.0)
    for (i, j), val in np.ndenumerate(result.rref[:, :-1]):
        axes[0].text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=11)
    axes[0].set_title(r"$[A\,|\,b] \to$ left block $I$")

    axes[1].bar(np.arange(1, 4), result.x, color=PALETTE["accent"], width=0.5)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_title(rf"solution $x$  (det $A = {det_A:.3g}$)")
    axes[1].grid(True, axis="y", color=PALETTE["grid"], lw=0.7)

    im = axes[2].imshow(inv_A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    for (i, j), val in np.ndenumerate(inv_A):
        axes[2].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    axes[2].set_title(r"$[A\,|\,I] \to A^{-1}$")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(r"RREF byproducts: solution, determinant, inverse", y=1.05, fontsize=14)
    fig.tight_layout()
    _save(fig, "06_rref_byproducts.png")


def figure_07_hilbert_forward_error() -> None:
    """Small residual does not imply small forward error when κ is large."""
    sizes = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    rel_residuals = []
    forward_errors = []
    conds = []

    for n in sizes:
        H = _hilbert(int(n))
        x_true = np.ones(int(n))
        b = H @ x_true
        result = solve(H, b)
        rel_residuals.append(result.relative_residual)
        forward_errors.append(float(np.linalg.norm(result.x - x_true, ord=2) / np.linalg.norm(x_true)))
        conds.append(result.condition_number)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    axes[0].semilogy(sizes, rel_residuals, "s-", color=PALETTE["steel"], lw=2, label="rel. residual")
    axes[0].semilogy(sizes, forward_errors, "o-", color=PALETTE["accent"], lw=2, label="rel. forward error")
    axes[0].set_xlabel(r"Hilbert order $n$")
    axes[0].set_ylabel("magnitude")
    axes[0].set_title(r"Residual vs forward error on $H_n$")
    axes[0].legend(frameon=False)
    axes[0].grid(True, color=PALETTE["grid"], lw=0.7)

    axes[1].semilogy(sizes, conds, "o-", color=PALETTE["muted"], lw=2)
    axes[1].set_xlabel(r"Hilbert order $n$")
    axes[1].set_ylabel(r"$\kappa_2(H_n)$")
    axes[1].set_title("Condition number growth")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.7)

    fig.suptitle(r"Ill-conditioning: $\|Ax-b\|$ small $\nRightarrow$ $x$ accurate", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "07_hilbert_forward_error.png")


def figure_08_runtime_complexity() -> None:
    rng = np.random.default_rng(0)
    sizes = np.array([20, 40, 60, 80, 100, 120])
    times = []

    for n in sizes:
        A = rng.normal(size=(int(n), int(n)))
        A = A @ A.T + 0.5 * np.eye(int(n))
        b = rng.normal(size=int(n))
        # Warm-up + median of a few runs
        solve(A, b)
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            solve(A, b)
            samples.append(time.perf_counter() - t0)
        times.append(float(np.median(samples)))

    times = np.asarray(times)
    # Fit c * n^3 through the last points in log space for a guide line
    guide = times[-1] * (sizes / sizes[-1]) ** 3

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(sizes, times, "o-", color=PALETTE["accent"], lw=2, label="Gauss–Jordan wall time")
    ax.loglog(sizes, guide, "--", color=PALETTE["muted"], lw=1.6, label=r"$O(n^3)$ guide")
    ax.set_xlabel(r"dimension $n$")
    ax.set_ylabel("seconds")
    ax.set_title("Dense Gauss–Jordan cost scales as $O(n^3)$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", color=PALETTE["grid"], lw=0.7)
    fig.tight_layout()
    _save(fig, "08_runtime_complexity.png")


def figure_09_workflow_summary() -> None:
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
    ax1.imshow(history.steps[0].matrix, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
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
    _save(fig, "09_workflow_summary.png")


def main() -> None:
    _style()
    figure_01_problem_statement()
    figure_02_exploratory_conditioning()
    figure_03_elimination_steps()
    figure_04_pivoting_accuracy()
    figure_05_solution_and_residual()
    figure_06_rref_and_inverse()
    figure_07_hilbert_forward_error()
    figure_08_runtime_complexity()
    figure_09_workflow_summary()


if __name__ == "__main__":
    main()
