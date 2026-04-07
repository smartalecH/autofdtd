"""Total-field scattered-field (TFSF) source models for Phase 1 source injection."""

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
Direction = Literal["+", "-"]


class TFSF(TaggedModel):
    """Total-field scattered-field (TFSF) source that injects a plane wave in a finite region.

    The TFSF source separates the simulation domain into total-field (inside the source
    region) and scattered-field (outside the source region) regions. This is useful for
    computing scattering and absorption cross-sections without needing additional
    normalization.

    Key characteristics:
    - The source injects 1 W of power per um^2 of source area along the injection_axis
    - The normalization for the incident field is |E_0|^2 = 2/(c*eps0) for any source size
    - For angled incidence, the same power is injected along the injection_axis, not
      the propagation direction
    - The source region is a 3D volume with finite extents in all three dimensions

    The TFSF boundary interaction works as follows:
    - Fields inside the TFSF box contain the total field (incident + scattered)
    - Fields outside the TFSF box contain only the scattered field
    - The incident field is cancelled at the TFSF boundary edges

    Phase 1 supports TFSF with:
    - Normal incidence (angle_theta=0)
    - Oblique incidence with angle_theta and angle_phi
    - Chebyshev broadband method (num_freqs >= 1)

    Example
    -------
    >>> from autofdtd import GaussianPulse, TFSF
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> tfsf = TFSF(
    ...     center=(0, 0, 0),
    ...     size=(2e-6, 2e-6, 2e-6),
    ...     source_time=pulse,
    ...     injection_axis=2,  # z-axis
    ...     direction="+",
    ...     pol_angle=0.0,
    ... )
    """

    type: Literal["TFSF"] = "TFSF"
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    source_time: GaussianPulse | ContinuousWave | BroadbandPulse | CustomSourceTime
    injection_axis: int = Field(
        title="Injection Axis",
        description="Specifies the injection axis. The plane of incidence is defined via this "
        "injection_axis and the direction. The propagation axis is defined with respect "
        "to the injection_axis by angle_theta and angle_phi.",
    )
    direction: Direction = Field(
        default="+",
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the injection "
        "axis.",
    )
    angle_theta: float = Field(
        default=0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
    )
    angle_phi: float = Field(
        default=0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
    )
    pol_angle: float = Field(
        default=0.0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the injection axis and the propagation axis (rad). "
        "pol_angle=0 (default) specifies P polarization, "
        "while pol_angle=pi/2 specifies S polarization.",
    )
    num_freqs: int = Field(
        default=1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. A Chebyshev interpolation is used, thus only a small number of points is "
        "typically sufficient to obtain converged results.",
        ge=1,
        le=20,
    )
    name: str | None = None
    interpolate: bool = True
    confine_to_bounds: bool = False

    @field_validator("injection_axis")
    @classmethod
    def _validate_injection_axis(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("injection_axis must be 0 (x), 1 (y), or 2 (z)")
        return value

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
        # TFSF is a volume source - all dimensions should be non-zero
        if any(math.isclose(s, 0.0) for s in normalized):
            raise ValueError(
                "TFSF requires a 3D volume (all size components must be non-zero), "
                f"got size={normalized}"
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
    def injection_plane_center(self) -> Vec3:
        """Center of the injection plane.

        The injection plane is located at one face of the TFSF box, offset by half
        the source size along the injection axis in the direction opposite to
        propagation (for direction="+") or in the propagation direction (for direction="-").

        This is where the incident plane wave is injected.
        """
        sign = 1 if self.direction == "-" else -1
        center = list(self.center)
        center[self.injection_axis] += sign * self.size[self.injection_axis] / 2.0
        return tuple(center)

    @property
    def tfsf_bounds(self) -> tuple[Vec3, Vec3]:
        """The lower and upper bounds of the TFSF source volume."""
        half = tuple(s / 2.0 for s in self.size)
        lower = tuple(c - h for c, h in zip(self.center, half, strict=True))
        upper = tuple(c + h for c, h in zip(self.center, half, strict=True))
        return lower, upper

    @property
    def placement_kind(self) -> Literal["volume"]:
        """Always a volume (3D) placement for TFSF."""
        return "volume"

    @property
    def support_bounds(self) -> tuple[Vec3, Vec3]:
        """TFSF support bounds matching the source volume."""
        return self.tfsf_bounds

    @property
    def _tangential_axes(self) -> tuple[int, int]:
        """The two tangential axes for this source's injection direction."""
        axis = self.injection_axis
        return tuple(a for a in range(3) if a != axis)

    @property
    def _dir_vector(self) -> Vec3:
        """Source direction normal vector in cartesian coordinates.

        Returns a unit vector pointing in the propagation direction.
        """
        # Propagation vector assuming injection along z
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * math.cos(self.angle_phi) * math.sin(self.angle_theta)
        dy = radius * math.sin(self.angle_phi) * math.sin(self.angle_theta)
        dz = radius * math.cos(self.angle_theta)

        # Move to original injection axis
        return self._unpop_axis(dz, (dx, dy))

    @property
    def _pol_vector(self) -> Vec3:
        """Source polarization normal vector in cartesian coordinates.

        Returns a unit vector pointing in the electric field polarization direction.
        """
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

    @property
    def _reference_wavelength(self) -> float | None:
        """Reference wavelength from source_time if available."""
        if isinstance(self.source_time, GaussianPulse):
            freq0 = getattr(self.source_time, "freq0", None)
            if freq0 and freq0 > 0:
                return 2.998e8 / freq0
        elif isinstance(self.source_time, ContinuousWave):
            freq0 = getattr(self.source_time, "freq0", None)
            if freq0 and freq0 > 0:
                return 2.998e8 / freq0
        return None

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


__all__ = [
    "TFSF",
]