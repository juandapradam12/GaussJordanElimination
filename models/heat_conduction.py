"""Steady-state heat conduction on a one-dimensional rod.

The continuous problem on ``[0, L]`` is

.. math::

    -\\kappa\\, T''(x) = q(x), \\qquad T(0) = T_L, \\qquad T(L) = T_R,

where ``\\kappa`` is the thermal conductivity and ``q`` is a volumetric heat
source. A uniform finite-difference grid with spacing ``h = L / (n + 1)``
yields the discrete Laplacian system ``A T = b`` for the unknown interior
temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]
SourceFn = Callable[[ArrayF], ArrayF]


@dataclass(frozen=True)
class HeatRodProblem:
    """Discrete steady-state heat conduction problem on a rod."""

    length: float
    conductivity: float
    n_interior: int
    left_temperature: float
    right_temperature: float
    source: SourceFn
    A: ArrayF
    b: ArrayF
    x_interior: ArrayF
    h: float

    def exact_quadratic_solution(self) -> ArrayF | None:
        """Closed-form interior temperatures when ``q`` is constant.

        For constant source ``q0`` the exact solution is the parabola

        .. math::

            T(x) = T_L + (T_R - T_L)\\frac{x}{L}
                   + \\frac{q_0}{2\\kappa} x (L - x).

        The central second-difference stencil is exact on polynomials of
        degree at most two, so the discrete solution matches this parabola
        up to round-off.
        """
        samples = self.source(self.x_interior)
        if samples.size == 0 or not np.allclose(samples, samples[0]):
            return None

        q0 = float(samples[0])
        L = self.length
        kappa = self.conductivity
        x = self.x_interior
        linear = self.left_temperature + (self.right_temperature - self.left_temperature) * (x / L)
        parabolic = (q0 / (2.0 * kappa)) * x * (L - x)
        return linear + parabolic


def sine_source(amplitude: float = 1.0, length: float = 1.0) -> SourceFn:
    """Return ``q(x) = amplitude * sin(π x / L)``."""

    def q(x: ArrayF, _amp: float = amplitude, _L: float = length) -> ArrayF:
        return _amp * np.sin(np.pi * x / _L)

    return q


def exact_sine_solution(
    x: ArrayF,
    *,
    amplitude: float,
    length: float,
    conductivity: float,
    left_temperature: float = 0.0,
    right_temperature: float = 0.0,
) -> ArrayF:
    """Exact solution for ``q(x) = A sin(π x / L)`` with Dirichlet ends.

    .. math::

        T(x) = T_L + (T_R - T_L)\\frac{x}{L}
               + \\frac{A L^2}{\\kappa \\pi^2}\\sin\\!\\left(\\frac{\\pi x}{L}\\right).
    """
    linear = left_temperature + (right_temperature - left_temperature) * (x / length)
    particular = (amplitude * length**2) / (conductivity * np.pi**2) * np.sin(np.pi * x / length)
    return linear + particular


def assemble_heat_rod(
    *,
    length: float = 1.0,
    conductivity: float = 1.0,
    n_interior: int = 40,
    left_temperature: float = 0.0,
    right_temperature: float = 0.0,
    source: SourceFn | float = 1.0,
) -> HeatRodProblem:
    """Assemble the tridiagonal finite-difference system for the heat rod.

    Parameters
    ----------
    length:
        Rod length ``L > 0``.
    conductivity:
        Thermal conductivity ``\\kappa > 0``.
    n_interior:
        Number of unknown interior nodes.
    left_temperature, right_temperature:
        Dirichlet data at ``x = 0`` and ``x = L``.
    source:
        Either a constant heat source or a callable ``q(x)``.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    if conductivity <= 0:
        raise ValueError("conductivity must be positive.")
    if n_interior < 1:
        raise ValueError("n_interior must be at least 1.")

    h = length / (n_interior + 1)
    x_interior = np.linspace(h, length - h, n_interior)

    if callable(source):
        source_fn: SourceFn = source
    else:
        q0 = float(source)

        def source_fn(x: ArrayF, _q0: float = q0) -> ArrayF:
            return np.full_like(x, _q0, dtype=float)

    q = source_fn(x_interior)

    # Second-difference stencil: (-T_{i-1} + 2 T_i - T_{i+1}) / h^2 = q_i / kappa
    diag = 2.0 * np.ones(n_interior)
    off = -1.0 * np.ones(n_interior - 1)
    A = (conductivity / h**2) * (
        np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
    )

    b = q.copy()
    b[0] += conductivity * left_temperature / h**2
    b[-1] += conductivity * right_temperature / h**2

    return HeatRodProblem(
        length=length,
        conductivity=conductivity,
        n_interior=n_interior,
        left_temperature=left_temperature,
        right_temperature=right_temperature,
        source=source_fn,
        A=A,
        b=b,
        x_interior=x_interior,
        h=h,
    )
