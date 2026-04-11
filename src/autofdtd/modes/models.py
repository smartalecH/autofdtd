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
    # Bend parameters for bent waveguide mode solving
    bend_radius: float | None = Field(
        default=None,
        description="Bend radius for bent waveguide mode solving (m). "
        "If set, the mode solver accounts for curvature in the eigenvalue problem.",
    )
    bend_axis: Literal[0, 1, 2] = Field(
        default=2,
        description="Axis of the bend plane for bent mode solving (0=x, 1=y, 2=z). "
        "For bend_axis=2, the bend is in the x-y plane (curving around z). "
        "For bend_axis=1, the bend is in the x-z plane (curving around y). "
        "For bend_axis=0, the bend is in the y-z plane (curving around x).",
    )
    # PML (Perfectly Matched Layer) parameters for absorbing radiation modes
    num_pml_layers: int = Field(
        default=0,
        description="Number of PML layers at each boundary. "
        "PML absorbs radiation modes and allows finding leaky modes. "
        "Typical values are 10-30 layers.",
    )
    pml_sigma_max: float = Field(
        default=2.0,
        description="Maximum PML conductivity (sigma) in units of sigma_max formula. "
        "Controls absorption strength. Higher values absorb more but may cause reflections.",
    )
    pml_kappa_min: float = Field(
        default=1.0,
        description="Minimum PML kappa (real part of s-factor) at inner boundary. "
        "Kappa stretches coordinates to increase absorption.",
    )
    pml_kappa_max: float = Field(
        default=3.0,
        description="Maximum PML kappa (real part of s-factor) at outer boundary. "
        "The kappa profile goes from kappa_min at inner edge to kappa_max at outer edge.",
    )
    pml_order: int = Field(
        default=3,
        description="Polynomial order of PML absorption profile. "
        "Higher orders absorb more strongly near outer boundary but may cause reflections.",
    )

    @field_validator("position")
    @classmethod
    def _validate_position(cls, value: float) -> float:
        return float(value)

    @field_validator("bend_radius", mode="before")
    @classmethod
    def _validate_bend_radius(cls, value: float | None) -> float | None:
        if value is None:
            return None
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("bend_radius must be positive")
        return radius

    @field_validator("num_pml_layers")
    @classmethod
    def _validate_num_pml_layers(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 0:
            raise ValueError("num_pml_layers must be non-negative")
        return numeric

    @field_validator("pml_sigma_max")
    @classmethod
    def _validate_pml_sigma_max(cls, value: float) -> float:
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("pml_sigma_max must be positive")
        return numeric

    @field_validator("pml_kappa_min")
    @classmethod
    def _validate_pml_kappa_min(cls, value: float) -> float:
        numeric = float(value)
        if numeric < 1.0:
            raise ValueError("pml_kappa_min must be >= 1.0")
        return numeric

    @field_validator("pml_kappa_max")
    @classmethod
    def _validate_pml_kappa_max(cls, value: float) -> float:
        numeric = float(value)
        if numeric < 1.0:
            raise ValueError("pml_kappa_max must be >= 1.0")
        return numeric

    @field_validator("pml_order")
    @classmethod
    def _validate_pml_order(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 1 or numeric > 4:
            raise ValueError("pml_order must be between 1 and 4")
        return numeric


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
    # Group index (n_g = n_eff - f * (dn_eff/df))
    n_group: float | None = None
    # TE fraction = integral(|Ex|^2 + |Ey|^2) / integral(|E|^2)
    te_fraction: float | None = None
    # TM fraction = 1 - TE_fraction
    tm_fraction: float | None = None
    # Effective mode area = (integral |E|^2)^2 / integral |E|^4
    effective_area: float | None = None

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