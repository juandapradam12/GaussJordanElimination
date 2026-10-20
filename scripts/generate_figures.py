#!/usr/bin/env python3
"""Generate workflow figures for the README case study.

Produces plots covering exploratory matrix analysis through the final
physical temperature field and solver diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from gauss_jordan import solve
from models import assemble_heat_rod, exact_sine_solution, sine_source

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"

# Visual language: cool steel / warm heat — not the usual AI purple/cream defaults.
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


def _save(fig: plt.Figure, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def figure_01_problem_schematic() -> None:
    """Schematic of the rod, boundary data, and heat source."""
    problem = assemble_heat_rod(
        n_interior=12,
        left_temperature=20.0,
        right_temperature=80.0,
        source=12.0,
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot([0, problem.length], [0, 0], color=PALETTE["steel"], lw=8, solid_capstyle="round")
    ax.scatter(
        [0, problem.length],
        [0, 0],
        s=160,
        c=[PALETTE["steel"], PALETTE["accent"]],
        zorder=3,
        edgecolors=PALETTE["ink"],
        linewidths=0.6,
    )
    ax.scatter(
        problem.x_interior,
        np.zeros_like(problem.x_interior),
        s=36,
        c=PALETTE["ink"],
        zorder=3,
        alpha=0.75,
    )

    # Heat-source arrows along the rod.
    for xi in problem.x_interior[::2]:
        ax.annotate(
            "",
            xy=(xi, 0.08),
            xytext=(xi, 0.32),
            arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.4),
        )

    ax.text(0, -0.22, r"$T(0)=T_L$", ha="center", fontsize=11)
    ax.text(problem.length, -0.22, r"$T(L)=T_R$", ha="center", fontsize=11)
    ax.text(problem.length / 2, 0.42, r"volumetric source $q(x)$", ha="center", color=PALETTE["accent"])
    ax.text(problem.length / 2, -0.42, r"unknown interior nodes $T_i$", ha="center", color=PALETTE["muted"])

    ax.set_xlim(-0.08, problem.length + 0.08)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel(r"position $x$ along the rod")
    ax.set_title("Physical setup: Dirichlet ends + distributed heat source")
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "01_problem_schematic.png")


def figure_02_matrix_structure() -> None:
    problem = assemble_heat_rod(n_interior=24, source=8.0)
    A = problem.A

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    im0 = axes[0].imshow(A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    axes[0].set_title("Coefficient matrix $A$ (scaled Laplacian)")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Spy / sparsity pattern emphasis
    mask = np.abs(A) > 0
    axes[1].imshow(mask.astype(float), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Nonzero pattern (tridiagonal band)")
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("row")

    fig.suptitle("Exploratory view of the discrete operator", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "02_matrix_structure.png")


def figure_03_spectrum_and_conditioning() -> None:
    sizes = [10, 20, 40, 80, 160]
    conds = []
    spectra = {}

    for n in sizes:
        problem = assemble_heat_rod(n_interior=n, source=1.0)
        conds.append(np.linalg.cond(problem.A))
        if n in (20, 80):
            spectra[n] = np.linalg.eigvalsh(problem.A)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    axes[0].plot(sizes, conds, "o-", color=PALETTE["steel"], lw=2, markersize=7)
    axes[0].set_xlabel(r"interior nodes $n$")
    axes[0].set_ylabel(r"$\kappa_2(A)$")
    axes[0].set_title("Condition number vs. mesh resolution")
    axes[0].grid(True, color=PALETTE["grid"], lw=0.8)

    for n, vals in spectra.items():
        axes[1].plot(
            np.arange(1, len(vals) + 1),
            vals,
            "o-",
            label=rf"$n={n}$",
            markersize=4,
            lw=1.5,
        )
    axes[1].set_xlabel("eigenvalue index")
    axes[1].set_ylabel(r"$\lambda_i(A)$")
    axes[1].set_title("Eigenvalue spectrum of $A$")
    axes[1].legend(frameon=False)
    axes[1].grid(True, color=PALETTE["grid"], lw=0.8)

    fig.suptitle("Exploratory spectral diagnostics", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "03_spectrum_conditioning.png")


def figure_04_rhs_and_load() -> None:
    def localized_source(x: np.ndarray) -> np.ndarray:
        # Smooth bump near the center — physically a localized heater.
        return 25.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.08**2))

    # Homogeneous Dirichlet ends so b mirrors q without large BC spikes.
    problem = assemble_heat_rod(
        n_interior=60,
        left_temperature=0.0,
        right_temperature=0.0,
        source=localized_source,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].fill_between(problem.x_interior, problem.source(problem.x_interior), color=PALETTE["accent"], alpha=0.35)
    axes[0].plot(problem.x_interior, problem.source(problem.x_interior), color=PALETTE["accent"], lw=2)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$q(x)$")
    axes[0].set_title("Heat source profile")
    axes[0].grid(True, color=PALETTE["grid"], lw=0.8)

    axes[1].stem(problem.x_interior, problem.b, linefmt=PALETTE["steel"], markerfmt="o", basefmt=" ")
    axes[1].set_xlabel(r"$x_i$")
    axes[1].set_ylabel(r"$b_i$")
    axes[1].set_title("Assembled right-hand side (source + BC lift)")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.8)

    fig.suptitle("From physics to the load vector $b$", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "04_source_and_rhs.png")


def figure_05_solution_profile() -> None:
    # Equal Dirichlet ends + strong source → a clear parabolic arch.
    problem = assemble_heat_rod(
        length=1.0,
        conductivity=1.0,
        n_interior=60,
        left_temperature=20.0,
        right_temperature=20.0,
        source=80.0,
    )
    result = solve(problem.A, problem.b)
    exact = problem.exact_quadratic_solution()
    assert exact is not None

    x_full = np.concatenate([[0.0], problem.x_interior, [problem.length]])
    T_num = np.concatenate([[problem.left_temperature], result.x, [problem.right_temperature]])
    T_ex = np.concatenate([[problem.left_temperature], exact, [problem.right_temperature]])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    axes[0].plot(x_full, T_ex, color=PALETTE["muted"], lw=2.5, label="exact parabola")
    axes[0].plot(x_full, T_num, "o-", color=PALETTE["accent"], markersize=3.5, lw=1.4, label="Gauss–Jordan")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$T(x)$")
    axes[0].set_title("Steady temperature field")
    axes[0].legend(frameon=False)
    axes[0].grid(True, color=PALETTE["grid"], lw=0.8)

    err = result.x - exact
    axes[1].plot(problem.x_interior, err, color=PALETTE["steel"], lw=2)
    axes[1].axhline(0.0, color=PALETTE["ink"], lw=0.8)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$T_{\mathrm{num}} - T_{\mathrm{exact}}$")
    axes[1].set_title("Pointwise discretization error")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.8)

    fig.suptitle("Physical result: interior temperatures", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "05_temperature_solution.png")


def figure_06_convergence_and_residuals() -> None:
    # Non-quadratic source so the FD truncation error is visible and O(h^2).
    amplitude = 8.0
    length = 1.0
    kappa = 1.0
    t_left, t_right = 0.0, 0.0
    sizes = np.array([10, 20, 40, 80, 160])
    max_errors = []
    rel_residuals = []

    for n in sizes:
        problem = assemble_heat_rod(
            length=length,
            conductivity=kappa,
            n_interior=int(n),
            left_temperature=t_left,
            right_temperature=t_right,
            source=sine_source(amplitude=amplitude, length=length),
        )
        result = solve(problem.A, problem.b)
        exact = exact_sine_solution(
            problem.x_interior,
            amplitude=amplitude,
            length=length,
            conductivity=kappa,
            left_temperature=t_left,
            right_temperature=t_right,
        )
        max_errors.append(float(np.max(np.abs(result.x - exact))))
        rel_residuals.append(result.relative_residual)

    h = length / (sizes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))

    axes[0].loglog(h, max_errors, "o-", color=PALETTE["accent"], lw=2, label="max error")
    ref = max_errors[0] * (h / h[0]) ** 2
    axes[0].loglog(h, ref, "--", color=PALETTE["muted"], lw=1.5, label=r"$O(h^2)$ guide")
    axes[0].set_xlabel(r"mesh size $h$")
    axes[0].set_ylabel(r"$\|T_{\mathrm{num}}-T_{\mathrm{exact}}\|_\infty$")
    axes[0].set_title(r"Spatial convergence ($q \propto \sin$)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, which="both", color=PALETTE["grid"], lw=0.7)

    axes[1].semilogy(sizes, rel_residuals, "s-", color=PALETTE["steel"], lw=2, label="relative residual")
    axes[1].set_xlabel(r"interior nodes $n$")
    axes[1].set_ylabel(r"$\|Ax-b\|_2 / \|b\|_2$")
    axes[1].set_title("Algebraic residual after elimination")
    axes[1].grid(True, color=PALETTE["grid"], lw=0.8)
    axes[1].legend(frameon=False)

    fig.suptitle("Verification: discretization order and solver fidelity", y=1.03, fontsize=14)
    fig.tight_layout()
    _save(fig, "06_convergence_residuals.png")


def figure_07_workflow_summary() -> None:
    """Compact panel summarizing the full pipeline narrative."""
    problem = assemble_heat_rod(
        n_interior=40,
        left_temperature=20.0,
        right_temperature=20.0,
        source=80.0,
    )
    result = solve(problem.A, problem.b)
    exact = problem.exact_quadratic_solution()
    assert exact is not None

    fig = plt.figure(figsize=(10.2, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(problem.A, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0))
    ax0.set_title("1. Assemble $A$")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    # Show the physical source; Dirichlet lift enters b but would dominate this scale.
    ax1.plot(
        problem.x_interior,
        problem.source(problem.x_interior),
        color=PALETTE["steel"],
        lw=2,
    )
    ax1.set_title(r"2. Prescribe source $q(x)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$q$")
    ax1.grid(True, color=PALETTE["grid"], lw=0.7)

    ax2 = fig.add_subplot(gs[1, 0])
    evals = np.linalg.eigvalsh(problem.A)
    ax2.plot(evals, color=PALETTE["accent"], lw=2)
    ax2.set_title(r"3. Inspect $\lambda(A)$ / conditioning")
    ax2.set_xlabel("index")
    ax2.grid(True, color=PALETTE["grid"], lw=0.7)

    ax3 = fig.add_subplot(gs[1, 1])
    x_full = np.concatenate([[0.0], problem.x_interior, [1.0]])
    T = np.concatenate(
        [[problem.left_temperature], result.x, [problem.right_temperature]]
    )
    ax3.plot(x_full, T, color=PALETTE["accent"], lw=2.4)
    ax3.set_title("4. Recover $T(x)$")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$T$")
    ax3.grid(True, color=PALETTE["grid"], lw=0.7)

    fig.suptitle(
        "Workflow: physics → linear algebra → verified temperature field",
        fontsize=14,
        y=0.98,
    )
    _save(fig, "07_workflow_summary.png")


def main() -> None:
    _style()
    figure_01_problem_schematic()
    figure_02_matrix_structure()
    figure_03_spectrum_and_conditioning()
    figure_04_rhs_and_load()
    figure_05_solution_profile()
    figure_06_convergence_and_residuals()
    figure_07_workflow_summary()


if __name__ == "__main__":
    main()
