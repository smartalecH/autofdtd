"""Primitive analytic geometry models for Phase 1 scene construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel

Vec3 = tuple[float, float, float]
Bounds3 = tuple[Vec3, Vec3]
Axes3 = tuple[Vec3, Vec3, Vec3]

_IDENTITY_AXES: Axes3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_CYLINDER_AXES: dict[int, Axes3] = {
    0: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    1: ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    2: _IDENTITY_AXES,
}


def _normalize_vec3(value: Sequence[object], *, field_name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three components")
    normalized = (float(value[0]), float(value[1]), float(value[2]))
    if any(not math.isfinite(component) for component in normalized):
        raise ValueError(f"{field_name} components must be finite")
    return normalized


def _component_type_name(component: object) -> str | None:
    if isinstance(component, TaggedModel):
        value = component.type
        return str(value) if value is not None else None
    if isinstance(component, Mapping):
        value = component.get("type")
        return str(value) if value is not None else None
    return None


def _dot(lhs: Vec3, rhs: Vec3) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _add(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _sub(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _scale(vector: Vec3, factor: float) -> Vec3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


class GeometryTransform(TaggedModel):
    """Translation-plus-orientation transform for primitive scene compilation."""

    type: Literal["GeometryTransform"] = "GeometryTransform"
    origin: Vec3 = (0.0, 0.0, 0.0)
    axes: Axes3 = Field(default_factory=lambda: _IDENTITY_AXES)

    @field_validator("origin")
    @classmethod
    def _validate_origin(cls, value: Vec3) -> Vec3:
        return _normalize_vec3(value, field_name="origin")

    @field_validator("axes")
    @classmethod
    def _validate_axes(cls, value: Sequence[Sequence[object]]) -> Axes3:
        if len(value) != 3:
            raise ValueError("axes must contain exactly three basis vectors")
        axes = tuple(_normalize_vec3(axis, field_name="axes") for axis in value)
        if any(abs(_dot(axis, axis) - 1.0) > 1e-9 for axis in axes):
            raise ValueError("axes must be unit vectors")
        if abs(_dot(axes[0], axes[1])) > 1e-9:
            raise ValueError("axes[0] and axes[1] must be orthogonal")
        if abs(_dot(axes[0], axes[2])) > 1e-9:
            raise ValueError("axes[0] and axes[2] must be orthogonal")
        if abs(_dot(axes[1], axes[2])) > 1e-9:
            raise ValueError("axes[1] and axes[2] must be orthogonal")
        return axes

    def local_to_world(self, point: Sequence[object]) -> Vec3:
        """Map a point from canonical primitive coordinates into world space."""
        local = _normalize_vec3(point, field_name="point")
        world = self.origin
        for scale, axis in zip(local, self.axes, strict=True):
            world = _add(world, _scale(axis, scale))
        return world

    def world_to_local(self, point: Sequence[object]) -> Vec3:
        """Map a world-space point into canonical primitive coordinates."""
        world = _normalize_vec3(point, field_name="point")
        offset = _sub(world, self.origin)
        return tuple(_dot(offset, axis) for axis in self.axes)

    @property
    def matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return a homogeneous rigid-transform matrix."""
        return (
            (self.axes[0][0], self.axes[1][0], self.axes[2][0], self.origin[0]),
            (self.axes[0][1], self.axes[1][1], self.axes[2][1], self.origin[1]),
            (self.axes[0][2], self.axes[1][2], self.axes[2][2], self.origin[2]),
            (0.0, 0.0, 0.0, 1.0),
        )

    def apply_to_bounds(self, bounds: Bounds3) -> Bounds3:
        """Transform an axis-aligned bounds box and return a world-space AABB."""
        lower, upper = bounds
        corners = tuple(
            self.local_to_world(
                (
                    lower[0] if x_index == 0 else upper[0],
                    lower[1] if y_index == 0 else upper[1],
                    lower[2] if z_index == 0 else upper[2],
                )
            )
            for x_index in (0, 1)
            for y_index in (0, 1)
            for z_index in (0, 1)
        )
        return (
            tuple(min(corner[axis] for corner in corners) for axis in range(3)),
            tuple(max(corner[axis] for corner in corners) for axis in range(3)),
        )

    def compose(self, inner: GeometryTransform) -> GeometryTransform:
        """Return the transform produced by applying `inner` then `self`."""
        axes = tuple(
            _sub(self.local_to_world(axis), self.origin) for axis in inner.axes
        )
        origin = self.local_to_world(inner.origin)
        return GeometryTransform(origin=origin, axes=axes)  # type: ignore[arg-type]

    @staticmethod
    def identity() -> GeometryTransform:
        return GeometryTransform()

    @staticmethod
    def translation(offset: Sequence[object]) -> GeometryTransform:
        return GeometryTransform(origin=_normalize_vec3(offset, field_name="offset"))

    @staticmethod
    def rotation(
        *,
        axis: Sequence[object],
        angle: float,
        origin: Sequence[object] = (0.0, 0.0, 0.0),
    ) -> GeometryTransform:
        rotation_axis = _normalize_vec3(axis, field_name="axis")
        if abs(_dot(rotation_axis, rotation_axis) - 1.0) > 1e-9:
            length = math.sqrt(_dot(rotation_axis, rotation_axis))
            rotation_axis = _scale(rotation_axis, 1.0 / length)
        ux, uy, uz = rotation_axis
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        one_minus_cosine = 1.0 - cosine
        axes: Axes3 = (
            (
                cosine + ux * ux * one_minus_cosine,
                uy * ux * one_minus_cosine + uz * sine,
                uz * ux * one_minus_cosine - uy * sine,
            ),
            (
                ux * uy * one_minus_cosine - uz * sine,
                cosine + uy * uy * one_minus_cosine,
                uz * uy * one_minus_cosine + ux * sine,
            ),
            (
                ux * uz * one_minus_cosine + uy * sine,
                uy * uz * one_minus_cosine - ux * sine,
                cosine + uz * uz * one_minus_cosine,
            ),
        )
        pivot = _normalize_vec3(origin, field_name="origin")
        rotated_pivot = GeometryTransform(
            origin=(0.0, 0.0, 0.0),
            axes=axes,
        ).local_to_world(pivot)
        translation = _sub(pivot, rotated_pivot)
        return GeometryTransform(origin=translation, axes=axes)

    @staticmethod
    def reflection(
        *,
        normal: Sequence[object],
        origin: Sequence[object] = (0.0, 0.0, 0.0),
    ) -> GeometryTransform:
        unit_normal = _normalize_vec3(normal, field_name="normal")
        if abs(_dot(unit_normal, unit_normal) - 1.0) > 1e-9:
            length = math.sqrt(_dot(unit_normal, unit_normal))
            unit_normal = _scale(unit_normal, 1.0 / length)
        nx, ny, nz = unit_normal
        axes: Axes3 = (
            (1.0 - 2.0 * nx * nx, -2.0 * ny * nx, -2.0 * nz * nx),
            (-2.0 * nx * ny, 1.0 - 2.0 * ny * ny, -2.0 * nz * ny),
            (-2.0 * nx * nz, -2.0 * ny * nz, 1.0 - 2.0 * nz * nz),
        )
        plane_origin = _normalize_vec3(origin, field_name="origin")
        distance = 2.0 * _dot(unit_normal, plane_origin)
        return GeometryTransform(origin=_scale(unit_normal, distance), axes=axes)


class PrimitiveGeometry(TaggedModel):
    """Common contract for Phase 1 analytic primitive geometries."""

    center: Vec3 = (0.0, 0.0, 0.0)

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Vec3) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @property
    def bounds(self) -> Bounds3:
        raise NotImplementedError

    @property
    def transform(self) -> GeometryTransform:
        return GeometryTransform(origin=self.center, axes=_IDENTITY_AXES)

    def contains_point(self, point: Sequence[object]) -> bool:
        raise NotImplementedError

    def translate(self, offset: Sequence[object]) -> PrimitiveGeometry:
        delta = _normalize_vec3(offset, field_name="offset")
        return self.copy_update(center=_add(self.center, delta))


class Box(PrimitiveGeometry):
    """Axis-aligned rectangular prism."""

    type: Literal["Box"] = "Box"
    size: Vec3

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Vec3) -> Vec3:
        size = _normalize_vec3(value, field_name="size")
        if any(component < 0.0 for component in size):
            raise ValueError("size components must be non-negative")
        if all(component == 0.0 for component in size):
            raise ValueError("size must include at least one positive extent")
        return size

    @classmethod
    def from_bounds(cls, *, rmin: Sequence[object], rmax: Sequence[object]) -> Box:
        lower = _normalize_vec3(rmin, field_name="rmin")
        upper = _normalize_vec3(rmax, field_name="rmax")
        if any(high < low for low, high in zip(lower, upper, strict=True)):
            raise ValueError("rmax must be greater than or equal to rmin on every axis")
        center = tuple((low + high) / 2.0 for low, high in zip(lower, upper, strict=True))
        size = tuple(high - low for low, high in zip(lower, upper, strict=True))
        return cls(center=center, size=size)

    @property
    def bounds(self) -> Bounds3:
        half = tuple(component / 2.0 for component in self.size)
        return (_sub(self.center, half), _add(self.center, half))

    def contains_point(self, point: Sequence[object]) -> bool:
        local = self.transform.world_to_local(point)
        half = tuple(component / 2.0 for component in self.size)
        return all(abs(coord) <= extent for coord, extent in zip(local, half, strict=True))


class Sphere(PrimitiveGeometry):
    """Spherical primitive."""

    type: Literal["Sphere"] = "Sphere"
    radius: float

    @field_validator("radius")
    @classmethod
    def _validate_radius(cls, value: float) -> float:
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")
        return radius

    @property
    def bounds(self) -> Bounds3:
        radius_vec = (self.radius, self.radius, self.radius)
        return (_sub(self.center, radius_vec), _add(self.center, radius_vec))

    def contains_point(self, point: Sequence[object]) -> bool:
        local = self.transform.world_to_local(point)
        return _dot(local, local) <= self.radius * self.radius


class Cylinder(PrimitiveGeometry):
    """Axis-aligned cylinder around one of the principal axes."""

    type: Literal["Cylinder"] = "Cylinder"
    radius: float
    length: float
    axis: int = 2

    @field_validator("radius")
    @classmethod
    def _validate_radius(cls, value: float) -> float:
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")
        return radius

    @field_validator("length")
    @classmethod
    def _validate_length(cls, value: float) -> float:
        length = float(value)
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("length must be non-negative")
        return length

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
        extents = [self.radius, self.radius, self.radius]
        extents[self.axis] = self.length / 2.0
        extent_vec = (extents[0], extents[1], extents[2])
        return (_sub(self.center, extent_vec), _add(self.center, extent_vec))

    @property
    def transform(self) -> GeometryTransform:
        return GeometryTransform(origin=self.center, axes=_CYLINDER_AXES[self.axis])

    def contains_point(self, point: Sequence[object]) -> bool:
        local = self.transform.world_to_local(point)
        radial = local[0] * local[0] + local[1] * local[1]
        return radial <= self.radius * self.radius and abs(local[2]) <= self.length / 2.0


PrimitiveGeometryModel = Box | Sphere | Cylinder


def geometry_model_from_value(value: object) -> PrimitiveGeometryModel:
    """Normalize a primitive geometry payload into its typed public model."""
    if isinstance(value, (Box, Sphere, Cylinder)):
        return value
    if isinstance(value, Mapping):
        geometry_type = _component_type_name(value)
        if geometry_type == "Box":
            return Box.model_validate(value)
        if geometry_type == "Sphere":
            return Sphere.model_validate(value)
        if geometry_type == "Cylinder":
            return Cylinder.model_validate(value)
    raise TypeError(f"unsupported primitive geometry payload {value!r}")
