"""Plane wave and angular specification models for Phase 1 source injection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel
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


class AbstractAngularSpec(TaggedModel):
    """Abstract base for defining angular variability specifications for plane waves."""

    type: str = "AbstractAngularSpec"


class FixedInPlaneKSpec(AbstractAngularSpec):
    """Plane wave is injected such that its in-plane wavevector is constant.

    That is, the injected field satisfies Bloch boundary conditions and its
    propagation direction is frequency dependent.
    """

    type: Literal["FixedInPlaneKSpec"] = "FixedInPlaneKSpec"


class FixedAngleSpec(AbstractAngularSpec):
    """Plane wave is injected such that its propagation direction is frequency independent.

    When using this option, boundary conditions in tangential directions must be
    set to periodic.
    """

    type: Literal["FixedAngleSpec"] = "FixedAngleSpec"


AngularSpec = FixedInPlaneKSpec | FixedAngleSpec


def angular_spec_model_from_value(value: object) -> AngularSpec:
    """Coerce a mapping payload into a supported Phase 1 angular spec model."""

    if isinstance(value, (FixedInPlaneKSpec, FixedAngleSpec)):
        return value
    if isinstance(value, dict):
        spec_type = str(value.get("type", ""))
        if spec_type == "FixedInPlaneKSpec":
            return FixedInPlaneKSpec.model_validate(value)
        if spec_type == "FixedAngleSpec":
            return FixedAngleSpec.model_validate(value)
    raise TypeError(f"unsupported Phase 1 angular spec type: {type(value)!r}")


class PlaneWave(TaggedModel):
    """Uniform plane wave source on a planar surface.

    A PlaneWave injects a spatially uniform electromagnetic wave with a defined
    propagation direction (via ``angle_theta`` and ``angle_phi``) and polarization
    (via ``pol_angle``). The source is directional: ``direction="+"`` injects
    forward-propagating waves in the +axis direction, ``direction="-"`` injects
    backward.

    The source has exactly one zero-size dimension defining the injection plane.

    Phase 1 supports two angular specifications:
    - ``FixedInPlaneKSpec``: constant in-plane k-vector (frequency-dependent angle)
    - ``FixedAngleSpec``: fixed propagation angle (frequency-independent)

    Example
    -------
    >>> from autofdtd import GaussianPulse, PlaneWave
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw = PlaneWave(
    ...     size=(0, 0, 10),
    ...     source_time=pulse,
    ...     pol_angle=0.0,
    ...     direction="+",
    ... )
    """

    type: Literal["PlaneWave"] = "PlaneWave"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    direction: Literal["+", "-"] = "+"
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    angular_spec: AngularSpec = Field(default_factory=FixedInPlaneKSpec)
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
        # Must have exactly one zero dimension (planar source)
        zero_count = sum(1 for s in normalized if math.isclose(s, 0.0))
        if zero_count != 1:
            raise ValueError(
                "PlaneWave requires exactly one zero-size dimension "
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

    @field_validator("angle_theta", "angle_phi", "pol_angle")
    @classmethod
    def _validate_angle(cls, value: float) -> float:
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("angle components must be finite")
        return normalized

    @property
    def injection_axis(self) -> int:
        """Axis of source normal (the zero-size dimension, injection direction)."""
        for i, extent in enumerate(self.size):
            if math.isclose(extent, 0.0):
                return i
        raise ValueError("PlaneWave must have exactly one zero-size dimension")

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

    @property
    def _dir_vector(self) -> Vec3:
        """Source direction normal vector in cartesian coordinates."""

        # Propagation vector assuming injection along z
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * math.cos(self.angle_phi) * math.sin(self.angle_theta)
        dy = radius * math.sin(self.angle_phi) * math.sin(self.angle_theta)
        dz = radius * math.cos(self.angle_theta)

        # Move to original injection axis
        return self._unpop_axis(dz, (dx, dy))

    @property
    def _pol_vector(self) -> Vec3:
        """Source polarization normal vector in cartesian coordinates."""

        # Polarization vector assuming propagation along z
        pol_x = math.cos(self.pol_angle)
        pol_y = math.sin(self.pol_angle)
        pol_z = 0.0

        # Rotate polarization back to original propagation axes
        # First rotate around y-axis by angle_theta
        cos_t = math.cos(self.angle_theta)
        sin_t = math.sin(self.angle_theta)
        pol_x_rot = pol_x * cos_t + pol_z * sin_t
        pol_y_rot = pol_y
        pol_z_rot = -pol_x * sin_t + pol_z * cos_t

        # Then rotate around z-axis by angle_phi
        cos_p = math.cos(self.angle_phi)
        sin_p = math.sin(self.angle_phi)
        pol_x_final = pol_x_rot * cos_p - pol_y_rot * sin_p
        pol_y_final = pol_x_rot * sin_p + pol_y_rot * cos_p
        pol_z_final = pol_z_rot

        # Move to original injection axis
        return self._unpop_axis(pol_z_final, (pol_x_final, pol_y_final))

    def _unpop_axis(self, z_component: float, xy_pair: tuple[float, float]) -> Vec3:
        """Restore components to the original injection axis ordering."""
        axis = self.injection_axis
        result = [0.0, 0.0, 0.0]
        if axis == 0:
            result[0] = z_component
            result[1] = xy_pair[0]
            result[2] = xy_pair[1]
        elif axis == 1:
            result[0] = xy_pair[0]
            result[1] = z_component
            result[2] = xy_pair[1]
        else:  # axis == 2
            result[0] = xy_pair[0]
            result[1] = xy_pair[1]
            result[2] = z_component
        return tuple(result)

    @property
    def is_fixed_angle(self) -> bool:
        """Whether the plane wave uses fixed-angle specification (frequency-independent)."""
        return isinstance(self.angular_spec, FixedAngleSpec) and self.angle_theta != 0.0


__all__ = [
    "AbstractAngularSpec",
    "FixedInPlaneKSpec",
    "FixedAngleSpec",
    "AngularSpec",
    "angular_spec_model_from_value",
    "PlaneWave",
]