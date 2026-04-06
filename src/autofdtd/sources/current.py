"""Current-source models for Phase 1 source injection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

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


CurrentSourceModel = UniformCurrentSource


def current_source_model_from_value(value: object) -> CurrentSourceModel:
    """Coerce a mapping payload into a supported Phase 1 current-source model."""

    if isinstance(value, UniformCurrentSource):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or current-source model, got {type(value)!r}")

    source_type = str(value.get("type", ""))
    if source_type == "UniformCurrentSource":
        return UniformCurrentSource.model_validate(value)
    raise TypeError(f"unsupported Phase 1 current-source type {source_type!r}")
