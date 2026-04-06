"""Polygon-extrusion geometry for Phase 1 planar photonics cases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.geometry.primitives import Bounds3, GeometryTransform, Vec3, _normalize_vec3

Vec2 = tuple[float, float]
_POLYSLAB_AXES: dict[int, tuple[Vec3, Vec3, Vec3]] = {
    0: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    1: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    2: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}
_PLANE_AXES: dict[int, tuple[int, int]] = {
    0: (1, 2),
    1: (0, 2),
    2: (0, 1),
}
_ZERO_TOL = 1e-12


def _normalize_vec2(value: Sequence[object], *, field_name: str) -> Vec2:
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two components")
    normalized = (float(value[0]), float(value[1]))
    if any(not math.isfinite(component) for component in normalized):
        raise ValueError(f"{field_name} components must be finite")
    return normalized


def _normalize_vertices(value: Sequence[Sequence[object]]) -> tuple[Vec2, ...]:
    vertices = tuple(
        _normalize_vec2(vertex, field_name="vertices") for vertex in value
    )
    if len(vertices) < 3:
        raise ValueError("vertices must contain at least three points")
    if vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ValueError("vertices must contain at least three distinct points")
    for index, vertex in enumerate(vertices):
        if vertex == vertices[index - 1]:
            raise ValueError("vertices must not repeat consecutive points")
    return vertices


def _signed_area(vertices: Sequence[Vec2]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _orient2d(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: Vec2, start: Vec2, end: Vec2) -> bool:
    cross = _orient2d(start, end, point)
    if abs(cross) > 1e-12:
        return False
    return (
        min(start[0], end[0]) - 1e-12 <= point[0] <= max(start[0], end[0]) + 1e-12
        and min(start[1], end[1]) - 1e-12 <= point[1] <= max(start[1], end[1]) + 1e-12
    )


def _segments_intersect(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool:
    o1 = _orient2d(a0, a1, b0)
    o2 = _orient2d(a0, a1, b1)
    o3 = _orient2d(b0, b1, a0)
    o4 = _orient2d(b0, b1, a1)
    if (
        abs(o1) <= 1e-12
        and _point_on_segment(b0, a0, a1)
        or abs(o2) <= 1e-12
        and _point_on_segment(b1, a0, a1)
        or abs(o3) <= 1e-12
        and _point_on_segment(a0, b0, b1)
        or abs(o4) <= 1e-12
        and _point_on_segment(a1, b0, b1)
    ):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _validate_simple_polygon(vertices: Sequence[Vec2]) -> None:
    edge_count = len(vertices)
    for index in range(edge_count):
        a0 = vertices[index]
        a1 = vertices[(index + 1) % edge_count]
        for other in range(index + 1, edge_count):
            if other in {index, (index - 1) % edge_count, (index + 1) % edge_count}:
                continue
            if index == 0 and other == edge_count - 1:
                continue
            b0 = vertices[other]
            b1 = vertices[(other + 1) % edge_count]
            if _segments_intersect(a0, a1, b0, b1):
                raise ValueError("vertices must define a simple non-self-intersecting polygon")
    area = _signed_area(vertices)
    if abs(area) <= 1e-12:
        raise ValueError("vertices must span a polygon with non-zero area")


def point_in_polygon(point: Vec2, vertices: Sequence[Vec2]) -> bool:
    """Return whether a 2D point lies inside or on the boundary of a simple polygon."""
    inside = False
    for start, end in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        if _point_on_segment(point, start, end):
            return True
        crosses = (start[1] > point[1]) != (end[1] > point[1])
        if not crosses:
            continue
        x_cross = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
        if point[0] < x_cross:
            inside = not inside
    return inside


class PolySlab(TaggedModel):
    """Axis-aligned extrusion of a simple polygon.

    Phase 1 scope intentionally covers the most common planar photonics case:
    a constant cross-section polygon extruded between `slab_bounds` on one
    principal axis. `sidewall_angle` and `dilation` are accepted only when they
    are effectively zero so unsupported taper semantics fail explicitly.
    """

    type: Literal["PolySlab"] = "PolySlab"
    vertices: tuple[Vec2, ...]
    slab_bounds: tuple[float, float]
    axis: int = 2
    sidewall_angle: float = 0.0
    dilation: float = 0.0
    reference_plane: Literal["bottom", "middle", "top"] = "middle"
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("vertices")
    @classmethod
    def _validate_vertices(cls, value: tuple[Vec2, ...]) -> tuple[Vec2, ...]:
        vertices = _normalize_vertices(value)
        _validate_simple_polygon(vertices)
        return vertices

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

    @field_validator("sidewall_angle", "dilation")
    @classmethod
    def _validate_phase1_zero_only(cls, value: float, info: object) -> float:
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{info.field_name} must be finite")
        if abs(normalized) > _ZERO_TOL:
            raise ValueError(
                f"{info.field_name}={normalized} is not supported in Phase 1; "
                "only constant-cross-section PolySlab extrusions are implemented"
            )
        return 0.0

    @model_validator(mode="after")
    def _validate_reference_plane(self) -> PolySlab:
        return self

    @property
    def polygon_bounds(self) -> tuple[Vec2, Vec2]:
        xs = [vertex[0] for vertex in self.vertices]
        ys = [vertex[1] for vertex in self.vertices]
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    @property
    def center(self) -> Vec3:
        (plane_min_0, plane_min_1), (plane_max_0, plane_max_1) = self.polygon_bounds
        center = [0.0, 0.0, 0.0]
        axis0, axis1 = _PLANE_AXES[self.axis]
        center[axis0] = (plane_min_0 + plane_max_0) / 2.0
        center[axis1] = (plane_min_1 + plane_max_1) / 2.0
        center[self.axis] = (self.slab_bounds[0] + self.slab_bounds[1]) / 2.0
        return (center[0], center[1], center[2])

    @property
    def bounds(self) -> Bounds3:
        lower = [0.0, 0.0, 0.0]
        upper = [0.0, 0.0, 0.0]
        (plane_min_0, plane_min_1), (plane_max_0, plane_max_1) = self.polygon_bounds
        axis0, axis1 = _PLANE_AXES[self.axis]
        lower[axis0] = plane_min_0
        lower[axis1] = plane_min_1
        lower[self.axis] = self.slab_bounds[0]
        upper[axis0] = plane_max_0
        upper[axis1] = plane_max_1
        upper[self.axis] = self.slab_bounds[1]
        return ((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))

    @property
    def transform(self) -> GeometryTransform:
        return GeometryTransform(origin=self.center, axes=_POLYSLAB_AXES[self.axis])

    def contains_point(self, point: Sequence[object]) -> bool:
        normalized = _normalize_vec3(point, field_name="point")
        if not (self.slab_bounds[0] <= normalized[self.axis] <= self.slab_bounds[1]):
            return False
        axis0, axis1 = _PLANE_AXES[self.axis]
        return point_in_polygon((normalized[axis0], normalized[axis1]), self.vertices)

    def translate(self, offset: Sequence[object]) -> PolySlab:
        delta = _normalize_vec3(offset, field_name="offset")
        axis0, axis1 = _PLANE_AXES[self.axis]
        shifted_vertices = tuple(
            (vertex[0] + delta[axis0], vertex[1] + delta[axis1]) for vertex in self.vertices
        )
        shifted_bounds = (
            self.slab_bounds[0] + delta[self.axis],
            self.slab_bounds[1] + delta[self.axis],
        )
        return self.copy_update(vertices=shifted_vertices, slab_bounds=shifted_bounds)


GeometryModel = PolySlab


def polyslab_model_from_value(value: object) -> PolySlab:
    """Normalize a PolySlab payload into its typed public model."""
    if isinstance(value, PolySlab):
        return value
    if isinstance(value, Mapping) and str(value.get("type")) == "PolySlab":
        return PolySlab.model_validate(value)
    raise TypeError(f"unsupported PolySlab payload {value!r}")
