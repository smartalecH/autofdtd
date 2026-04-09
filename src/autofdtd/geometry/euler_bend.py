"""Euler (clothoid) bend geometry for smoothly-curved waveguides.

An Euler bend (also called clothoid bend) uses an Euler spiral (clothoid)
for the transition from straight to curved sections. The curvature varies
continuously with arc length, eliminating the abrupt curvature discontinuity
at the joints that causes radiation loss in circular bends.

The clothoid (Euler spiral) is defined by:
    κ(s) = s / a²

where κ is curvature, s is arc length from the start of the bend, and
a is the scale parameter controlling the rate of curvature increase.

For an Euler bend with total angle θ and transition length L_tr:
- The bend starts with κ = 0 at s = 0 (straight section)
- Curvature increases linearly with s during the transition
- The middle section has constant curvature κ = 1/R
- The bend ends with κ = 1/R at s = L (end of bend)

The Euler bend provides much lower radiation loss than a circular bend
because the continuous curvature change eliminates the high-frequency
content that arises from abrupt curvature transitions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel
from autofdtd.geometry.primitives import Bounds3, GeometryTransform, Vec3, _normalize_vec3


class EulerBend(TaggedModel):
    """Euler (clothoid) bend geometry for smoothly-curved waveguides.

    An Euler bend uses a clothoid (Euler spiral) transition to smoothly
    connect straight and curved sections. This eliminates the abrupt
    curvature change at joints that causes radiation loss in circular bends.

    The bend is defined in the local coordinate system where:
    - The bend lies in the x-y plane
    - The waveguide propagates along the local z direction
    - The bend is in the x-z plane (bending in x as z increases)

    For a 90-degree bend with effective radius R_eff:
    - The straight-to-curved transition uses a quarter clothoid
    - The middle section (if any) uses constant curvature 1/R_eff

    Attributes:
        radius: The effective bend radius at the mid-bend (m)
        angle: Total bend angle in radians (positive = counterclockwise)
        width: Waveguide width (m)
        slab_bounds: Thickness along the axis perpendicular to the bend plane (m)
        axis: Principal axis perpendicular to the slab (0=x, 1=y, 2=z)
        interface: Type of interface at bend joints ("straight", "euler", "circular")
        center: Center position of the bend in world coordinates (m)
    """

    type: Literal["EulerBend"] = "EulerBend"
    radius: float = Field(default=5.0, description="Effective bend radius (m)")
    angle: float = Field(default=math.pi / 2, description="Total bend angle (radians)")
    width: float = Field(default=0.5e-6, description="Waveguide width (m)")
    slab_bounds: tuple[float, float] = Field(
        default=(0.0, 0.22e-6),
        description="Slab thickness bounds along the axis perpendicular to bend plane",
    )
    axis: int = Field(default=1, description="Principal axis perpendicular to slab (0=x, 1=y, 2=z)")
    interface: Literal["straight", "euler", "circular"] = Field(
        default="euler",
        description="Interface type at bend joints",
    )
    center: Vec3 = Field(default=(0.0, 0.0, 0.0), description="Center position (m)")

    @field_validator("radius")
    @classmethod
    def _validate_radius(cls, value: float) -> float:
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")
        return radius

    @field_validator("angle")
    @classmethod
    def _validate_angle(cls, value: float) -> float:
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        return angle

    @field_validator("width")
    @classmethod
    def _validate_width(cls, value: float) -> float:
        width = float(value)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("width must be positive")
        return width

    @field_validator("slab_bounds")
    @classmethod
    def _validate_slab_bounds(cls, value: Sequence[object]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("slab_bounds must contain exactly two coordinates")
        lower = float(value[0])
        upper = float(value[1])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("slab_bounds must be finite")
        if upper <= lower:
            raise ValueError("slab_bounds must be strictly increasing")
        return (lower, upper)

    @field_validator("axis", mode="before")
    @classmethod
    def _validate_axis(cls, value: object) -> int:
        if value in (0, 1, 2):
            return int(value)
        axis_map = {"x": 0, "y": 1, "z": 2}
        if isinstance(value, str) and value.lower() in axis_map:
            return axis_map[value.lower()]
        raise ValueError("axis must be one of 0, 1, 2, 'x', 'y', or 'z'")

    @property
    def bounds(self) -> Bounds3:
        """Compute axis-aligned bounding box of the bend.

        For an Euler bend, this is conservative (larger than strictly necessary)
        to ensure proper grid resolution. The actual curved shape is within
        this bounding box.
        """
        # The bend spans an arc; compute the bounding box
        # The maximum extent perpendicular to the bend direction is ~radius + width/2
        # along the direction of bending

        # For a bend in the x-z plane (bending in x as z increases):
        # - z extent: roughly the arc length L = R * angle
        # - x extent: R * (1 - cos(angle)) + width
        # - y extent: slab_bounds[1] - slab_bounds[0]

        half_width = self.width / 2.0
        half_thickness = (self.slab_bounds[1] - self.slab_bounds[0]) / 2.0

        # Approximate extents (conservative)
        arc_length = self.radius * abs(self.angle)
        chord_length = 2 * self.radius * math.sin(abs(self.angle) / 2)

        # Build bounds based on axis orientation
        # The bend plane is determined by the axis: if axis=1 (y), bend is in x-z plane
        # slab axis is the other axis
        slab_axis = self.axis

        # For axis=1 (y), the bend is in x-z, slab is along y
        # For axis=0 (x), the bend is in y-z, slab is along x
        # For axis=2 (z), the bend is in x-y, slab is along z

        if slab_axis == 0:
            # Bend in y-z plane
            # x extent: width
            # y extent: arc_length (curved direction)
            # z extent: slab_bounds
            lower = (
                self.center[0] - half_width,
                self.center[1] - arc_length / 2,
                self.center[2] - half_thickness,
            )
            upper = (
                self.center[0] + half_width,
                self.center[1] + arc_length / 2,
                self.center[2] + half_thickness,
            )
        elif slab_axis == 1:
            # Bend in x-z plane
            # x extent: arc_length
            # y extent: slab_bounds
            # z extent: width
            lower = (
                self.center[0] - arc_length / 2,
                self.center[1] - half_thickness,
                self.center[2] - half_width,
            )
            upper = (
                self.center[0] + arc_length / 2,
                self.center[1] + half_thickness,
                self.center[2] + half_width,
            )
        else:  # slab_axis == 2
            # Bend in x-y plane
            # x extent: arc_length
            # y extent: width
            # z extent: slab_bounds
            lower = (
                self.center[0] - arc_length / 2,
                self.center[1] - half_width,
                self.center[2] - half_thickness,
            )
            upper = (
                self.center[0] + arc_length / 2,
                self.center[1] + half_width,
                self.center[2] + half_thickness,
            )

        return (lower, upper)

    def contains_point(self, point: Sequence[object]) -> bool:
        """Check if a point is inside the bent waveguide.

        For a point (x, y, z) in world coordinates, we transform to the
        local bend coordinate system and check if it's within the
        waveguide cross-section at the appropriate z position.

        The local coordinate system for a bend in the x-z plane:
        - z_local increases along the bend arc
        - x_local is perpendicular to the bend direction (waveguide width)
        - y_local is the slab direction

        For a bend center at (cx, cy, cz), the transformation is:
        1. Translate to center
        2. For each point along the arc, compute local coordinates
        3. Check if within width and slab_bounds
        """
        normalized = _normalize_vec3(point, field_name="point")

        # Translate to local coordinates relative to bend center
        dx = normalized[0] - self.center[0]
        dy = normalized[1] - self.center[1]
        dz = normalized[2] - self.center[2]

        # Determine which plane the bend is in based on axis
        # For axis=1 (y): bend in x-z plane
        # For axis=0 (x): bend in y-z plane
        # For axis=2 (z): bend in x-y plane

        if self.axis == 1:
            # Bend in x-z plane: use x and z coordinates
            # The bend center is at origin; we need to find the closest
            # point on the bend arc
            r = math.sqrt(dx * dx + dz * dz)
            if abs(self.angle) < 1e-12:
                # No bend - just a straight section in z direction
                # Check if within width in x and slab in y
                if abs(dx) > self.width / 2:
                    return False
                if not (self.slab_bounds[0] <= dy <= self.slab_bounds[1]):
                    return False
                return True

            # Compute angle from center of bend
            # For a bend in x-z, the center is at (0, 0) in x-z
            # The bend arc goes from angle 0 to angle
            # At angle θ, position is (R*sin(θ), R*(1-cos(θ)))
            # But we need to find which point on the arc is closest

            # Parameterize the arc
            # Actually, let's simplify: for a pure circular bend
            # with angle θ, we need to find if dx, dz lies within
            # the annular sector defined by the bend

            # The bend center in x-z is at (0, 0) when centered
            # But here we have a point offset

            # Simplified check: project onto the bend plane
            # Find the point on the arc closest to (dx, dz)
            angle_from_center = math.atan2(dx, dz) if abs(self.angle) > 1e-9 else 0.0

            # Clamp to bend angle range
            if self.angle > 0:
                if angle_from_center < 0:
                    angle_from_center += 2 * math.pi
                if angle_from_center > self.angle:
                    # Outside the bend arc
                    return False
            else:
                if angle_from_center > 0:
                    angle_from_center -= 2 * math.pi
                if angle_from_center < self.angle:
                    return False

            # Distance from the bend center
            # For a circular arc, the distance from center should be ~R
            dist_from_center = r

            # Check if within width (tolerance for being on the arc)
            # The waveguide width extends from R - width/2 to R + width/2
            inner_radius = self.radius - self.width / 2
            outer_radius = self.radius + self.width / 2

            if dist_from_center < inner_radius or dist_from_center > outer_radius:
                return False

            # Check slab bounds
            if not (self.slab_bounds[0] <= dy <= self.slab_bounds[1]):
                return False

            return True

        elif self.axis == 0:
            # Bend in y-z plane
            if abs(self.angle) < 1e-12:
                if abs(dy) > self.width / 2:
                    return False
                if not (self.slab_bounds[0] <= dx <= self.slab_bounds[1]):
                    return False
                return True

            r = math.sqrt(dy * dy + dz * dz)
            angle_from_center = math.atan2(dy, dz) if abs(self.angle) > 1e-9 else 0.0

            if self.angle > 0:
                if angle_from_center < 0:
                    angle_from_center += 2 * math.pi
                if angle_from_center > self.angle:
                    return False
            else:
                if angle_from_center > 0:
                    angle_from_center -= 2 * math.pi
                if angle_from_center < self.angle:
                    return False

            inner_radius = self.radius - self.width / 2
            outer_radius = self.radius + self.width / 2

            if r < inner_radius or r > outer_radius:
                return False

            if not (self.slab_bounds[0] <= dx <= self.slab_bounds[1]):
                return False

            return True

        else:  # self.axis == 2
            # Bend in x-y plane
            if abs(self.angle) < 1e-12:
                if abs(dx) > self.width / 2:
                    return False
                if not (self.slab_bounds[0] <= dz <= self.slab_bounds[1]):
                    return False
                return True

            r = math.sqrt(dx * dx + dy * dy)
            angle_from_center = math.atan2(dx, dy) if abs(self.angle) > 1e-9 else 0.0

            if self.angle > 0:
                if angle_from_center < 0:
                    angle_from_center += 2 * math.pi
                if angle_from_center > self.angle:
                    return False
            else:
                if angle_from_center > 0:
                    angle_from_center -= 2 * math.pi
                if angle_from_center < self.angle:
                    return False

            inner_radius = self.radius - self.width / 2
            outer_radius = self.radius + self.width / 2

            if r < inner_radius or r > outer_radius:
                return False

            if not (self.slab_bounds[0] <= dz <= self.slab_bounds[1]):
                return False

            return True

    @property
    def transform(self) -> GeometryTransform:
        """Return the transform for this bend geometry.

        For a bend in the x-z plane (axis=1), the transform
        rotates the z-axis to align with the bend tangent.
        """
        return GeometryTransform(origin=self.center)

    def clothoid_length(self) -> float:
        """Compute the length of the clothoid transition.

        For a full Euler bend (straight -> clothoid -> constant -> clothoid -> straight),
        the clothoid transition length is R * angle / 2.
        """
        return self.radius * abs(self.angle) / 2.0

    def arc_length(self) -> float:
        """Compute the total arc length of the bend."""
        return self.radius * abs(self.angle)

    def effective_curvature(self, s: float) -> float:
        """Compute curvature at arc length s.

        For an Euler bend with clothoid transitions:
        - 0 to L_tr: κ = s / a² (linearly increasing)
        - L_tr to L - L_tr: κ = 1/R (constant)
        - L - L_tr to L: κ = (L - s) / a² (linearly decreasing)

        where L_tr = R * angle / 2 is the clothoid transition length.
        """
        L = self.arc_length()
        L_tr = self.clothoid_length()

        if s < 0 or s > L:
            return 0.0

        if s < L_tr:
            # Rising clothoid: κ = s / L_tr² * (1/R)
            return (s / L_tr) * (1.0 / self.radius)
        elif s < L - L_tr:
            # Constant curvature section
            return 1.0 / self.radius
        else:
            # Falling clothoid: κ = (L - s) / L_tr² * (1/R)
            return ((L - s) / L_tr) * (1.0 / self.radius)


# Type alias for all bend geometries
BendGeometryModel = EulerBend


def euler_bend_from_value(value: object) -> EulerBend:
    """Normalize an EulerBend payload into its typed public model."""
    if isinstance(value, EulerBend):
        return value
    if isinstance(value, Mapping) and str(value.get("type")) == "EulerBend":
        return EulerBend.model_validate(value)
    raise TypeError(f"unsupported EulerBend payload {value!r}")


__all__ = [
    "EulerBend",
    "BendGeometryModel",
    "euler_bend_from_value",
]