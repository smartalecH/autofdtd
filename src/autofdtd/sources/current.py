"""Current-source models for Phase 1 source injection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel
from autofdtd.sources.time import (
    BroadbandPulse,
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    SourceTimeModel,
    source_time_model_from_value,
)

Vec3 = tuple[float, float, float]
Polarization = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
AmplitudeDefinition = Literal["density", "total"]
PlacementKind = Literal["point", "line", "sheet", "volume"]


def _normalize_name(name: str | None) -> str | None:
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        raise ValueError("names must not be empty or whitespace-only")
    return stripped


def _normalize_vec3(value: Sequence[object], *, field_name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three components")
    normalized = (float(value[0]), float(value[1]), float(value[2]))
    if any(not math.isfinite(component) for component in normalized):
        raise ValueError(f"{field_name} components must be finite")
    return normalized


def _non_negative_size(value: Sequence[object]) -> Vec3:
    size = _normalize_vec3(value, field_name="size")
    if any(component < 0.0 for component in size):
        raise ValueError("size components must be non-negative")
    return size


class UniformCurrentSource(TaggedModel):
    """Uniform electric or magnetic current source over a rectangular support."""

    type: Literal["UniformCurrentSource"] = "UniformCurrentSource"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    polarization: Polarization
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False
    current_amplitude_definition: AmplitudeDefinition = "density"

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Sequence[object]) -> Vec3:
        return _non_negative_size(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("source_time", mode="before")
    @classmethod
    def _validate_source_time(cls, value: object) -> SourceTimeModel:
        return source_time_model_from_value(value)

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        half = tuple(extent / 2.0 for extent in self.size)
        lower = tuple(c - h for c, h in zip(self.center, half, strict=True))
        upper = tuple(c + h for c, h in zip(self.center, half, strict=True))
        return lower, upper

    @property
    def field_kind(self) -> Literal["electric", "magnetic"]:
        return "electric" if self.polarization[0] == "E" else "magnetic"

    @property
    def component_axis(self) -> int:
        return "xyz".index(self.polarization[1].lower())

    @property
    def zero_size_axes(self) -> tuple[int, ...]:
        return tuple(index for index, extent in enumerate(self.size) if math.isclose(extent, 0.0))

    @property
    def placement_kind(self) -> PlacementKind:
        dimensionality = sum(0 if math.isclose(extent, 0.0) else 1 for extent in self.size)
        return ("point", "line", "sheet", "volume")[dimensionality]


class PointDipole(TaggedModel):
    """Infinitesimal point dipole source with strictly zero size.

    This is a localized source that corresponds to an infinitesimal antenna
    with a fixed current density. Unlike UniformCurrentSource with size=(0,0,0),
    PointDipole strictly enforces zero size and uses density-based amplitude
    interpretation (no cross-sectional normalization).
    """

    type: Literal["PointDipole"] = "PointDipole"
    center: Vec3 = (0.0, 0.0, 0.0)
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    polarization: Polarization
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("source_time", mode="before")
    @classmethod
    def _validate_source_time(cls, value: object) -> SourceTimeModel:
        return source_time_model_from_value(value)

    @property
    def size(self) -> Vec3:
        """PointDipole always has zero size."""
        return (0.0, 0.0, 0.0)

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        """Point dipole has degenerate support bounds matching the center."""
        return (self.center, self.center)

    @property
    def field_kind(self) -> Literal["electric", "magnetic"]:
        """Electric or magnetic source based on polarization."""
        return "electric" if self.polarization[0] == "E" else "magnetic"

    @property
    def component_axis(self) -> int:
        """Component axis index (0=x, 1=y, 2=z)."""
        return "xyz".index(self.polarization[1].lower())

    @property
    def zero_size_axes(self) -> tuple[int, ...]:
        """All axes are zero-size for a point dipole."""
        return (0, 1, 2)

    @property
    def placement_kind(self) -> PlacementKind:
        """Always a point placement."""
        return "point"


class CustomCurrentSource(TaggedModel):
    """Custom current source with explicit field data arrays.

    Phase 1 implements a simplified CustomCurrentSource that accepts explicit
    field data arrays instead of Tidy3D's FieldDataset. The source injects
    the provided E and H field components directly as J and M current
    distributions. Coordinates are relative to the source center.
    """

    type: Literal["CustomCurrentSource"] = "CustomCurrentSource"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False
    # Field data as simple arrays; keys are field components
    # E and H field components at each grid point, relative to center
    e_fields: dict[str, tuple[float, ...]] | None = None
    h_fields: dict[str, tuple[float, ...]] | None = None
    # Grid coordinates for field data, relative to center
    coordinates: dict[str, tuple[float, ...]] | None = None

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Sequence[object]) -> Vec3:
        return _non_negative_size(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("source_time", mode="before")
    @classmethod
    def _validate_source_time(cls, value: object) -> SourceTimeModel:
        return source_time_model_from_value(value)

    @property
    def field_kind(self) -> Literal["electric", "magnetic"]:
        """CustomCurrentSource can have both electric and magnetic components."""
        return "electric"

    @property
    def has_electric(self) -> bool:
        """Whether any electric field components are provided."""
        return bool(self.e_fields)

    @property
    def has_magnetic(self) -> bool:
        """Whether any magnetic field components are provided."""
        return bool(self.h_fields)

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        half = tuple(extent / 2.0 for extent in self.size)
        lower = tuple(c - h for c, h in zip(self.center, half, strict=True))
        upper = tuple(c + h for c, h in zip(self.center, half, strict=True))
        return lower, upper

    def get_electric_component(self, polarization: str) -> tuple[float, ...] | None:
        """Get electric field component by polarization name."""
        if self.e_fields is None:
            return None
        return self.e_fields.get(polarization)

    def get_magnetic_component(self, polarization: str) -> tuple[float, ...] | None:
        """Get magnetic field component by polarization name."""
        if self.h_fields is None:
            return None
        return self.h_fields.get(polarization)


class CustomFieldSource(TaggedModel):
    """Custom field source using the equivalence principle on a planar surface.

    Phase 1 implements a simplified CustomFieldSource that accepts explicit
    field data arrays for tangential E and H components on a plane.
    The source uses the equivalence principle: J = n × H and M = -n × E,
    where n is the source normal direction determined by the injection axis
    (the non-zero size dimension).

    Unlike CustomCurrentSource which injects current densities directly,
    CustomFieldSource derives the equivalent currents from the tangential
    field components at the source plane.

    The source is directional: ``direction="+"`` injects forward-propagating
    waves in the +axis direction, ``direction="-"`` injects backward.

    Coordinates for field data are relative to the source center.
    """

    type: Literal["CustomFieldSource"] = "CustomFieldSource"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    direction: Literal["+", "-"] = "+"
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False
    # Field data as simple arrays; keys are tangential field components
    # For a planar source in xy-plane (z-size=0), tangential components are
    # Ex, Ey, Hx, Hy. For yz-plane (x-size=0): Ey, Ez, Hy, Hz. etc.
    e_fields: dict[str, tuple[float, ...]] | None = None
    h_fields: dict[str, tuple[float, ...]] | None = None
    # Grid coordinates for field data, relative to center
    coordinates: dict[str, tuple[float, ...]] | None = None

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Sequence[object]) -> Vec3:
        size = _normalize_vec3(value, field_name="size")
        # Must have exactly one zero dimension for planar source
        zero_count = sum(1 for s in size if math.isclose(s, 0.0))
        if zero_count != 1:
            raise ValueError(
                "CustomFieldSource requires exactly one zero-size dimension "
                f"(planar source), got size={size}"
            )
        return size

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("source_time", mode="before")
    @classmethod
    def _validate_source_time(cls, value: object) -> SourceTimeModel:
        return source_time_model_from_value(value)

    @property
    def injection_axis(self) -> int:
        """Axis of source normal (the zero-size dimension, injection direction)."""
        for i, extent in enumerate(self.size):
            if math.isclose(extent, 0.0):
                return i
        raise ValueError("CustomFieldSource must have exactly one zero-size dimension")

    @property
    def _tangential_axes(self) -> tuple[int, int]:
        """The two tangential axes for this planar source."""
        axis = self.injection_axis
        return tuple(a for a in range(3) if a != axis)

    @property
    def _tangential_components(self) -> tuple[str, ...]:
        """The tangential field component names for this planar source."""
        axes = self._tangential_axes
        components = []
        for ax in axes:
            components.extend([f"E{'xyz'[ax]}", f"H{'xyz'[ax]}"])
        return tuple(components)

    @property
    def field_kind(self) -> Literal["electric", "magnetic"]:
        """CustomFieldSource can have both electric and magnetic components."""
        return "electric"

    @property
    def has_electric(self) -> bool:
        """Whether any electric field components are provided."""
        return bool(self.e_fields)

    @property
    def has_magnetic(self) -> bool:
        """Whether any magnetic field components are provided."""
        return bool(self.h_fields)

    @property
    def has_tangential_fields(self) -> bool:
        """Whether at least one tangential field component is provided."""
        if self.e_fields:
            for comp in self.e_fields:
                if comp in self._tangential_components:
                    return True
        if self.h_fields:
            for comp in self.h_fields:
                if comp in self._tangential_components:
                    return True
        return False

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        half = tuple(extent / 2.0 for extent in self.size)
        lower = tuple(c - h for c, h in zip(self.center, half, strict=True))
        upper = tuple(c + h for c, h in zip(self.center, half, strict=True))
        return lower, upper

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    def get_electric_component(self, polarization: str) -> tuple[float, ...] | None:
        """Get electric field component by polarization name."""
        if self.e_fields is None:
            return None
        return self.e_fields.get(polarization)

    def get_magnetic_component(self, polarization: str) -> tuple[float, ...] | None:
        """Get magnetic field component by polarization name."""
        if self.h_fields is None:
            return None
        return self.h_fields.get(polarization)


CurrentSourceModel = (
    UniformCurrentSource | PointDipole | CustomCurrentSource | CustomFieldSource
)


def current_source_model_from_value(value: object) -> CurrentSourceModel:
    """Coerce a mapping payload into a supported Phase 1 current-source model."""

    if isinstance(value, (UniformCurrentSource, PointDipole, CustomCurrentSource, CustomFieldSource)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or current-source model, got {type(value)!r}")

    source_type = str(value.get("type", ""))
    if source_type == "UniformCurrentSource":
        return UniformCurrentSource.model_validate(value)
    if source_type == "PointDipole":
        return PointDipole.model_validate(value)
    if source_type == "CustomCurrentSource":
        return CustomCurrentSource.model_validate(value)
    if source_type == "CustomFieldSource":
        return CustomFieldSource.model_validate(value)
    raise TypeError(f"unsupported Phase 1 current-source type {source_type!r}")
