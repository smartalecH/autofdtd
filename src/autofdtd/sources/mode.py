"""Mode-source models for Phase 1 source injection backed by the mode solver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel
from autofdtd.modes.models import ModeSpec as _ModeSpec
from autofdtd.sources.time import (
    GaussianPulse,
    ContinuousWave,
    BroadbandPulse,
    CustomSourceTime,
    SourceTimeModel,
    source_time_model_from_value,
)


def _normalize_name(name: str | None) -> str | None:
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        raise ValueError("names must not be empty or whitespace-only")
    return stripped


Vec3 = tuple[float, float, float]


class ModeSource(TaggedModel):
    """Mode source that injects a guided mode field profile.

    Phase 1 ModeSource is backed by the mode solver. It uses the mode
    specification to look up or compute a mode profile, then injects
    the corresponding field distribution as a planar source using the
    equivalence principle.

    The source is directional: ``direction="+"`` injects forward-propagating
    waves in the +axis direction, ``direction="-"`` injects backward.

    Example
    -------
    >>> from autofdtd import GaussianPulse, ModeSource, ModeSpec
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> mode_source = ModeSource(
    ...     size=(10, 2, 0),
    ...     source_time=pulse,
    ...     mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
    ...     mode_index=1,
    ...     direction="+",
    ... )
    """

    type: Literal["ModeSource"] = "ModeSource"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    mode_spec: "_ModeSpec" = Field(default_factory=_ModeSpec)
    mode_index: int = 0
    direction: Literal["+", "-"] = "+"
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        if len(value) != 3:
            raise ValueError("center must contain exactly three components")
        normalized = (float(value[0]), float(value[1]), float(value[2]))
        if any(not math.isfinite(component) for component in normalized):
            raise ValueError("center components must be finite")
        return normalized

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Sequence[object]) -> Vec3:
        if len(value) != 3:
            raise ValueError("size must contain exactly three components")
        normalized = tuple(float(v) for v in value)
        if any(not math.isfinite(component) for component in normalized):
            raise ValueError("size components must be finite")
        # Must have exactly one zero dimension (planar source for mode injection)
        zero_count = sum(1 for s in normalized if math.isclose(s, 0.0))
        if zero_count != 1:
            raise ValueError(
                "ModeSource requires exactly one zero-size dimension "
                f"(planar source), got size={normalized}"
            )
        return normalized

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("source_time", mode="before")
    @classmethod
    def _validate_source_time(cls, value: object) -> SourceTimeModel:
        return source_time_model_from_value(value)

    @field_validator("mode_index")
    @classmethod
    def _validate_mode_index(cls, value: int) -> int:
        numeric = int(value)
        if numeric < 0:
            raise ValueError("mode_index must be non-negative")
        return numeric

    @property
    def injection_axis(self) -> int:
        """Axis of source normal (the zero-size dimension, injection direction)."""
        for i, extent in enumerate(self.size):
            if math.isclose(extent, 0.0):
                return i
        raise ValueError("ModeSource must have exactly one zero-size dimension")

    @property
    def _tangential_axes(self) -> tuple[int, int]:
        """The two tangential axes for this planar source."""
        axis = self.injection_axis
        return tuple(a for a in range(3) if a != axis)

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        half = tuple(extent / 2.0 for extent in self.size)
        lower = tuple(c - h for c, h in zip(self.center, half, strict=True))
        upper = tuple(c + h for c, h in zip(self.center, half, strict=True))
        return lower, upper


__all__ = ["ModeSource"]
