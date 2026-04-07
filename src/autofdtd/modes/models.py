"""Mode-solver public models for Phase 1."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel


def _positive_float(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value")
    return numeric


def _nonnegative_float(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")
    return numeric


class ModeSpec(TaggedModel):
    """Mode-specification controls for mode-solver and mode-source workflows."""

    type: Literal["ModeSpec"] = "ModeSpec"
    num_modes: int = 1
    target_neff: float | None = None
    precision: Literal["single", "double"] = "single"
    polynomial_degree: int = 2

    @field_validator("num_modes")
    @classmethod
    def _validate_num_modes(cls, value: int) -> int:
        numeric = int(value)
        if numeric <= 0:
            raise ValueError("num_modes must be a positive integer")
        return numeric

    @field_validator("target_neff")
    @classmethod
    def _validate_target_neff(cls, value: float | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("target_neff must be finite")
        return numeric

    @field_validator("polynomial_degree")
    @classmethod
    def _validate_polynomial_degree(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 0 or numeric > 4:
            raise ValueError("polynomial_degree must be between 0 and 4 for Phase 1")
        return numeric


class BoundaryCondition(TaggedModel):
    """One-dimensional boundary condition for mode-solver cross-section."""

    type: Literal["BoundaryCondition"] = "BoundaryCondition"
    # 0 = Dirichlet (E=0 at boundary), 1 = Neumann (dE/dn=0), 2 = Bloch periodic
    condition: Literal["dirichlet", "neumann", "bloch"] = "dirichlet"
    bloch_phase: float = 0.0

    @field_validator("bloch_phase")
    @classmethod
    def _validate_bloch_phase(cls, value: float) -> float:
        return float(value)


class ModeSolverCrossSection(TaggedModel):
    """Cross-sectional plane for mode-solver eigenvalue problem."""

    type: Literal["ModeSolverCrossSection"] = "ModeSolverCrossSection"
    # Axis normal to the cross-section (0=x, 1=y, 2=z)
    normal_axis: Literal[0, 1, 2] = 2
    # Position along the normal axis
    position: float = 0.0
    # Boundary conditions for the four sides: [north, south, east, west]
    boundaries: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition, BoundaryCondition] = (
        BoundaryCondition(condition="dirichlet"),
        BoundaryCondition(condition="dirichlet"),
        BoundaryCondition(condition="dirichlet"),
        BoundaryCondition(condition="dirichlet"),
    )

    @field_validator("position")
    @classmethod
    def _validate_position(cls, value: float) -> float:
        return float(value)


class ModeSolverConfig(TaggedModel):
    """Configuration for the vector finite-difference mode solver."""

    type: Literal["ModeSolverConfig"] = "ModeSolverConfig"
    # Wavelength in vacuum (SI meters)
    wavelength: float
    # Cross-sectional plane specification
    cross_section: ModeSolverCrossSection = Field(default_factory=ModeSolverCrossSection)
    # Mode specification
    mode_spec: ModeSpec = Field(default_factory=ModeSpec)
    # Minimum number of cells in each in-plane direction
    min_cells: int = 20
    # Maximum number of cells in each in-plane direction
    max_cells: int = 200
    # Tolerance for eigensolver convergence
    tolerance: float = 1e-8

    @field_validator("wavelength")
    @classmethod
    def _validate_wavelength(cls, value: float) -> float:
        return _positive_float(value, field_name="wavelength")

    @field_validator("min_cells")
    @classmethod
    def _validate_min_cells(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 3:
            raise ValueError("min_cells must be at least 3 for a finite-difference grid")
        return numeric

    @field_validator("max_cells")
    @classmethod
    def _validate_max_cells(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 3:
            raise ValueError("max_cells must be at least 3")
        return numeric

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        return _positive_float(value, field_name="tolerance")


class ModeSolution(TaggedModel):
    """Solved electromagnetic mode with field components and effective index."""

    type: Literal["ModeSolution"] = "ModeSolution"
    # Effective index neff = beta / k0
    neff: float
    # Wavelength this mode was solved for
    wavelength: float
    # Coordinate arrays for the cross-sectional grid
    x: tuple[float, ...]
    y: tuple[float, ...]
    # Electric field components on Yee-like staggered grid (cell centers)
    Ex: tuple[tuple[float, float], ...]  # (real, imag) pairs
    Ey: tuple[tuple[float, float], ...]
    Ez: tuple[tuple[float, float], ...]
    # Magnetic field components on Yee-like staggered grid
    Hx: tuple[tuple[float, float], ...]
    Hy: tuple[tuple[float, float], ...]
    Hz: tuple[tuple[float, float], ...]
    # Power normalization factor
    power: float = 1.0

    @property
    def num_cells(self) -> int:
        return len(self.x) * len(self.y)

    @property
    def neff_real(self) -> float:
        return float(self.neff.real) if isinstance(self.neff, complex) else float(self.neff)

    @property
    def k0(self) -> float:
        """Free-space wave number."""
        return 2.0 * math.pi / self.wavelength


__all__ = [
    "ModeSpec",
    "ModeSolverConfig",
    "ModeSolverCrossSection",
    "BoundaryCondition",
    "ModeSolution",
]