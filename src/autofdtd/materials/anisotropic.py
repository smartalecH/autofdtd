"""Phase 1 anisotropic material models and explicit defer-policy surfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.materials.dispersive import Debye, Drude, Lorentz, PoleResidue, Sellmeier
from autofdtd.materials.isotropic import Medium, PECMedium, PMCMedium, _normalize_name

AxisLike = int | str
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
DiagonalMediumComponent = Medium | PECMedium | PMCMedium | PoleResidue | Sellmeier | Lorentz | Drude | Debye


def _medium_component_from_value(value: object) -> DiagonalMediumComponent:
    if isinstance(value, (Medium, PECMedium, PMCMedium, PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or supported medium component, got {type(value)!r}")
    medium_type = str(value.get("type", ""))
    if medium_type == "Medium":
        return Medium.model_validate(value)
    if medium_type == "PECMedium":
        return PECMedium.model_validate(value)
    if medium_type == "PMCMedium":
        return PMCMedium.model_validate(value)
    if medium_type == "PoleResidue":
        return PoleResidue.model_validate(value)
    if medium_type == "Sellmeier":
        return Sellmeier.model_validate(value)
    if medium_type == "Lorentz":
        return Lorentz.model_validate(value)
    if medium_type == "Drude":
        return Drude.model_validate(value)
    if medium_type == "Debye":
        return Debye.model_validate(value)
    raise TypeError(f"unsupported diagonal medium component type {medium_type!r}")


def _normalize_axis(value: AxisLike, *, field_name: str) -> int:
    if value in (0, 1, 2):
        return int(value)
    axis_map = {"x": 0, "y": 1, "z": 2}
    if isinstance(value, str) and value.lower() in axis_map:
        return axis_map[value.lower()]
    raise ValueError(f"{field_name} must be one of 0, 1, 2, 'x', 'y', or 'z'")


def _positive_finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value")
    return normalized


def _matrix3x3(value: object, *, field_name: str) -> Matrix3x3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a 3x3 real-valued matrix")
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three rows")
    rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"{field_name}[{row_index}] must be a length-3 row")
        if len(row) != 3:
            raise ValueError(f"{field_name}[{row_index}] must contain exactly three entries")
        normalized_row = tuple(float(entry) for entry in row)
        if any(not math.isfinite(entry) for entry in normalized_row):
            raise ValueError(f"{field_name} entries must be finite")
        rows.append(normalized_row)
    return (rows[0], rows[1], rows[2])


def _scale_surface_component(component: DiagonalMediumComponent, *, thickness: float) -> DiagonalMediumComponent:
    scale = 1.0 / _positive_finite(thickness, field_name="thickness")
    if isinstance(component, PECMedium):
        return component
    if isinstance(component, PMCMedium):
        raise ValueError("Medium2D volumetric conversion does not support PMCMedium components in Phase 1")
    if isinstance(component, Medium):
        pole_residue = PoleResidue.from_medium(component)
    elif isinstance(component, Sellmeier | Lorentz | Drude | Debye):
        pole_residue = component.to_pole_residue()
    else:
        pole_residue = component
    scaled = PoleResidue(
        name=component.name,
        eps_inf=1.0 + scale * (pole_residue.eps_inf - 1.0),
        poles=tuple(
            (pole, (residue[0] * scale, residue[1] * scale))
            for pole, residue in pole_residue.poles
        ),
    )
    try:
        return scaled.to_medium()
    except ValueError:
        return scaled


class AnisotropicMedium(TaggedModel):
    """Phase 1 diagonal anisotropic medium with per-axis scalar component media."""

    type: Literal["AnisotropicMedium"] = "AnisotropicMedium"
    name: str | None = None
    xx: DiagonalMediumComponent = Medium()
    yy: DiagonalMediumComponent = Medium()
    zz: DiagonalMediumComponent = Medium()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("xx", "yy", "zz", mode="before")
    @classmethod
    def _validate_component(cls, value: object) -> DiagonalMediumComponent:
        return _medium_component_from_value(value)

    @property
    def components(self) -> dict[str, DiagonalMediumComponent]:
        return {"xx": self.xx, "yy": self.yy, "zz": self.zz}

    @property
    def component_types(self) -> tuple[str, str, str]:
        return (self.xx.type, self.yy.type, self.zz.type)

    def principal_medium(self, axis: AxisLike) -> DiagonalMediumComponent:
        normalized = _normalize_axis(axis, field_name="axis")
        return (self.xx, self.yy, self.zz)[normalized]

    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        omega = 2.0 * math.pi * _positive_finite(frequency, field_name="frequency")
        components: list[complex] = []
        for medium in (self.xx, self.yy, self.zz):
            if isinstance(medium, Medium):
                components.append(complex(medium.permittivity, -medium.conductivity / omega))
            elif isinstance(medium, PECMedium):
                components.append(complex(0.0, 0.0))
            elif isinstance(medium, PMCMedium):
                components.append(complex(1.0, 0.0))
            else:
                components.append(medium.eps_model(frequency))
        return (components[0], components[1], components[2])

    @property
    def supports_phase1_runtime(self) -> bool:
        return True


class FullyAnisotropicMedium(TaggedModel):
    """Deferred Phase 1 surface for full tensor media."""

    type: Literal["FullyAnisotropicMedium"] = "FullyAnisotropicMedium"
    name: str | None = None
    permittivity: Matrix3x3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    conductivity: Matrix3x3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("permittivity", "conductivity", mode="before")
    @classmethod
    def _validate_matrix(cls, value: object, info) -> Matrix3x3:
        return _matrix3x3(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _validate_permittivity(self) -> FullyAnisotropicMedium:
        diagonal = tuple(self.permittivity[i][i] for i in range(3))
        if any(value <= 0.0 for value in diagonal):
            raise ValueError("permittivity diagonal entries must be positive")
        return self

    @property
    def phase1_policy(self) -> str:
        return (
            "FullyAnisotropicMedium is parsed and serialized, but direct Phase 1 simulation support "
            "is deferred; use AnisotropicMedium for diagonal tensor media."
        )


class Medium2D(TaggedModel):
    """Deferred Phase 1 2D-material surface with explicit conversion helpers."""

    type: Literal["Medium2D"] = "Medium2D"
    name: str | None = None
    ss: DiagonalMediumComponent
    tt: DiagonalMediumComponent

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("ss", "tt", mode="before")
    @classmethod
    def _validate_component(cls, value: object) -> DiagonalMediumComponent:
        component = _medium_component_from_value(value)
        if isinstance(component, PMCMedium):
            raise ValueError("Medium2D does not support PMCMedium components in Phase 1")
        return component

    @model_validator(mode="after")
    def _validate_matching_pec_policy(self) -> Medium2D:
        if isinstance(self.ss, PECMedium) != isinstance(self.tt, PECMedium):
            raise ValueError("Medium2D requires ss and tt to be either both PECMedium or both non-PEC")
        return self

    @property
    def components(self) -> dict[str, DiagonalMediumComponent]:
        return {"ss": self.ss, "tt": self.tt}

    @property
    def phase1_policy(self) -> str:
        return (
            "Medium2D is not a directly supported simulation medium in Phase 1; convert it to a "
            "volumetric AnisotropicMedium with to_anisotropic_medium()."
        )

    def to_anisotropic_medium(self, *, axis: AxisLike, thickness: float) -> AnisotropicMedium:
        normal_axis = _normalize_axis(axis, field_name="axis")
        scaled_ss = _scale_surface_component(self.ss, thickness=thickness)
        scaled_tt = _scale_surface_component(self.tt, thickness=thickness)
        components: list[DiagonalMediumComponent] = [Medium(), Medium(), Medium()]
        in_plane_axes = [index for index in range(3) if index != normal_axis]
        components[in_plane_axes[0]] = scaled_ss
        components[in_plane_axes[1]] = scaled_tt
        return AnisotropicMedium(name=self.name, xx=components[0], yy=components[1], zz=components[2])

    def to_pole_residue(self, *, thickness: float) -> PoleResidue:
        scaled_ss = _scale_surface_component(self.ss, thickness=2.0 * thickness)
        scaled_tt = _scale_surface_component(self.tt, thickness=2.0 * thickness)
        if isinstance(scaled_ss, PECMedium) or isinstance(scaled_tt, PECMedium):
            raise ValueError("Medium2D.to_pole_residue does not support PECMedium components in Phase 1")

        def as_pole_residue(component: DiagonalMediumComponent) -> PoleResidue:
            if isinstance(component, Medium):
                return PoleResidue.from_medium(component)
            if isinstance(component, Sellmeier | Lorentz | Drude | Debye):
                return component.to_pole_residue()
            if isinstance(component, PoleResidue):
                return component
            raise TypeError(f"cannot reduce {component.type!r} to PoleResidue")

        pole_ss = as_pole_residue(scaled_ss)
        pole_tt = as_pole_residue(scaled_tt)
        return PoleResidue(
            name=self.name,
            eps_inf=0.5 * (pole_ss.eps_inf + pole_tt.eps_inf),
            poles=tuple(pole_ss.poles) + tuple(pole_tt.poles),
        )


__all__ = [
    "AnisotropicMedium",
    "AxisLike",
    "DiagonalMediumComponent",
    "FullyAnisotropicMedium",
    "Matrix3x3",
    "Medium2D",
]
