"""Physical models that assemble linear systems ``A x = b``."""

from models.heat_conduction import (
    HeatRodProblem,
    assemble_heat_rod,
    exact_sine_solution,
    sine_source,
)

__all__ = [
    "HeatRodProblem",
    "assemble_heat_rod",
    "exact_sine_solution",
    "sine_source",
]
