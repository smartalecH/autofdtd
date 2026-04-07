"""Mode-solver IR models and lowering for Phase 1 execution IR."""

from __future__ import annotations

from typing import Literal

from autofdtd.core.models import TaggedModel
from autofdtd.modes.models import (
    BoundaryCondition,
    ModeSolution,
    ModeSolverConfig,
    ModeSolverCrossSection,
    ModeSpec,
)


class ModeSpecIR(TaggedModel):
    """Typed IR for mode-specification controls."""

    type: Literal["ModeSpecIR"] = "ModeSpecIR"
    component_type: Literal["ModeSpec"] = "ModeSpec"
    num_modes: int
    target_neff: float | None = None
    precision: Literal["single", "double"] = "single"
    polynomial_degree: int = 2


class BoundaryConditionIR(TaggedModel):
    """Typed IR for mode-solver cross-sectional boundary conditions."""

    type: Literal["BoundaryConditionIR"] = "BoundaryConditionIR"
    component_type: Literal["BoundaryCondition"] = "BoundaryCondition"
    condition: Literal["dirichlet", "neumann", "bloch"]
    bloch_phase: float


class ModeSolverCrossSectionIR(TaggedModel):
    """Typed IR for the cross-sectional plane specification."""

    type: Literal["ModeSolverCrossSectionIR"] = "ModeSolverCrossSectionIR"
    component_type: Literal["ModeSolverCrossSection"] = "ModeSolverCrossSection"
    normal_axis: Literal[0, 1, 2]
    position: float
    boundaries: tuple[BoundaryConditionIR, BoundaryConditionIR, BoundaryConditionIR, BoundaryConditionIR]


class ModeSolverConfigIR(TaggedModel):
    """Typed IR for the mode-solver configuration bundle."""

    type: Literal["ModeSolverConfigIR"] = "ModeSolverConfigIR"
    component_type: Literal["ModeSolverConfig"] = "ModeSolverConfig"
    wavelength: float
    cross_section: ModeSolverCrossSectionIR
    mode_spec: ModeSpecIR
    min_cells: int
    max_cells: int
    tolerance: float


class ModeSolutionIR(TaggedModel):
    """Typed IR for a solved mode field profile."""

    type: Literal["ModeSolutionIR"] = "ModeSolutionIR"
    component_type: Literal["ModeSolution"] = "ModeSolution"
    neff: float
    wavelength: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    Ex: tuple[tuple[float, float], ...]
    Ey: tuple[tuple[float, float], ...]
    Ez: tuple[tuple[float, float], ...]
    Hx: tuple[tuple[float, float], ...]
    Hy: tuple[tuple[float, float], ...]
    Hz: tuple[tuple[float, float], ...]
    power: float


def boundary_condition_to_ir(bc: BoundaryCondition) -> BoundaryConditionIR:
    return BoundaryConditionIR(
        condition=bc.condition,
        bloch_phase=bc.bloch_phase,
    )


def mode_spec_to_ir(spec: ModeSpec) -> ModeSpecIR:
    return ModeSpecIR(
        num_modes=spec.num_modes,
        target_neff=spec.target_neff,
        precision=spec.precision,
        polynomial_degree=spec.polynomial_degree,
    )


def cross_section_to_ir(cs: ModeSolverCrossSection) -> ModeSolverCrossSectionIR:
    return ModeSolverCrossSectionIR(
        normal_axis=cs.normal_axis,
        position=cs.position,
        boundaries=tuple(boundary_condition_to_ir(b) for b in cs.boundaries),
    )


def mode_solver_config_to_ir(config: ModeSolverConfig) -> ModeSolverConfigIR:
    return ModeSolverConfigIR(
        wavelength=config.wavelength,
        cross_section=cross_section_to_ir(config.cross_section),
        mode_spec=mode_spec_to_ir(config.mode_spec),
        min_cells=config.min_cells,
        max_cells=config.max_cells,
        tolerance=config.tolerance,
    )


def mode_solution_to_ir(solution: ModeSolution) -> ModeSolutionIR:
    return ModeSolutionIR(
        neff=solution.neff,
        wavelength=solution.wavelength,
        x=solution.x,
        y=solution.y,
        Ex=solution.Ex,
        Ey=solution.Ey,
        Ez=solution.Ez,
        Hx=solution.Hx,
        Hy=solution.Hy,
        Hz=solution.Hz,
        power=solution.power,
    )


__all__ = [
    "ModeSpecIR",
    "BoundaryConditionIR",
    "ModeSolverCrossSectionIR",
    "ModeSolverConfigIR",
    "ModeSolutionIR",
    "boundary_condition_to_ir",
    "mode_spec_to_ir",
    "cross_section_to_ir",
    "mode_solver_config_to_ir",
    "mode_solution_to_ir",
]