"""Composite geometry wrappers for grouped, transformed, clipped, and repeated layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.geometry.polyslab import PolySlab, polyslab_model_from_value
from autofdtd.geometry.primitives import (
    Bounds3,
    Box,
    Cylinder,
    GeometryTransform,
    Sphere,
    Vec3,
    _normalize_vec3,
)

_SUPPORTED_CLIP_OPERATIONS = frozenset({"union", "intersection", "difference"})
_WRAPPER_TYPES = frozenset({"GeometryGroup", "Transformed", "ClipOperation", "GeometryArray"})


def _component_type_name(component: object) -> str | None:
    if isinstance(component, TaggedModel):
        value = component.type
        return str(value) if value is not None else None
    if isinstance(component, Mapping):
        value = component.get("type")
        return str(value) if value is not None else None
    return None


def _bounds_union(items: Sequence[Bounds3]) -> Bounds3:
    return (
        tuple(min(bounds[0][axis] for bounds in items) for axis in range(3)),
        tuple(max(bounds[1][axis] for bounds in items) for axis in range(3)),
    )


def _bounds_intersection(a: Bounds3, b: Bounds3) -> Bounds3:
    lower = tuple(max(a[0][axis], b[0][axis]) for axis in range(3))
    upper = tuple(min(a[1][axis], b[1][axis]) for axis in range(3))
    if any(lower[axis] > upper[axis] for axis in range(3)):
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return (lower, upper)


class GeometryGroup(TaggedModel):
    """Treat multiple geometries as one structure-shaped region."""

    type: Literal["GeometryGroup"] = "GeometryGroup"
    geometries: tuple[object, ...]

    @field_validator("geometries")
    @classmethod
    def _validate_geometries(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        geometries = tuple(geometry_model_from_value(item) for item in value)
        if not geometries:
            raise ValueError("geometries must contain at least one geometry")
        return geometries

    @property
    def bounds(self) -> Bounds3:
        return _bounds_union(tuple(geometry.bounds for geometry in self.geometries))

    def contains_point(self, point: Sequence[object]) -> bool:
        normalized = _normalize_vec3(point, field_name="point")
        return any(_geometry_contains_point(geometry, normalized) for geometry in self.geometries)

    def translate(self, offset: Sequence[object]) -> GeometryGroup:
        delta = _normalize_vec3(offset, field_name="offset")
        return self.copy_update(
            geometries=tuple(_geometry_translate(geometry, delta) for geometry in self.geometries)
        )


class Transformed(TaggedModel):
    """Apply a rigid transform to another geometry."""

    type: Literal["Transformed"] = "Transformed"
    geometry: object
    transform: GeometryTransform = Field(default_factory=GeometryTransform)

    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, value: object) -> object:
        return geometry_model_from_value(value)

    @model_validator(mode="after")
    def _flatten_nested_transforms(self) -> Transformed:
        if not isinstance(self.geometry, Transformed):
            return self
        nested = self.geometry
        return self.copy_update(
            geometry=nested.geometry,
            transform=self.transform.compose(nested.transform),
        )

    @property
    def bounds(self) -> Bounds3:
        return self.transform.apply_to_bounds(self.geometry.bounds)

    def contains_point(self, point: Sequence[object]) -> bool:
        local_point = self.transform.world_to_local(point)
        return _geometry_contains_point(self.geometry, local_point)

    def translate(self, offset: Sequence[object]) -> Transformed:
        delta = _normalize_vec3(offset, field_name="offset")
        return self.copy_update(
            transform=GeometryTransform.translation(delta).compose(self.transform)
        )


class ClipOperation(TaggedModel):
    """Boolean combination of two geometries using a narrow Phase 1 subset."""

    type: Literal["ClipOperation"] = "ClipOperation"
    operation: Literal["union", "intersection", "difference"]
    geometry_a: object
    geometry_b: object

    @field_validator("geometry_a", "geometry_b")
    @classmethod
    def _validate_geometry_operand(cls, value: object) -> object:
        return geometry_model_from_value(value)

    @field_validator("operation", mode="before")
    @classmethod
    def _validate_operation(cls, value: object) -> str:
        operation = str(value)
        if operation not in _SUPPORTED_CLIP_OPERATIONS:
            raise ValueError(
                "Phase 1 ClipOperation supports only 'union', 'intersection', and 'difference'"
            )
        return operation

    @property
    def bounds(self) -> Bounds3:
        bounds_a = self.geometry_a.bounds
        bounds_b = self.geometry_b.bounds
        if self.operation == "difference":
            return bounds_a
        if self.operation == "intersection":
            return _bounds_intersection(bounds_a, bounds_b)
        return _bounds_union((bounds_a, bounds_b))

    def contains_point(self, point: Sequence[object]) -> bool:
        normalized = _normalize_vec3(point, field_name="point")
        inside_a = _geometry_contains_point(self.geometry_a, normalized)
        inside_b = _geometry_contains_point(self.geometry_b, normalized)
        if self.operation == "union":
            return inside_a or inside_b
        if self.operation == "intersection":
            return inside_a and inside_b
        return inside_a and not inside_b

    def translate(self, offset: Sequence[object]) -> ClipOperation:
        delta = _normalize_vec3(offset, field_name="offset")
        return self.copy_update(
            geometry_a=_geometry_translate(self.geometry_a, delta),
            geometry_b=_geometry_translate(self.geometry_b, delta),
        )


class GeometryArray(TaggedModel):
    """Repeat a base geometry with per-instance rigid transforms and offsets."""

    type: Literal["GeometryArray"] = "GeometryArray"
    geometry: object
    offsets: tuple[Vec3, ...] | None = None
    transforms: tuple[GeometryTransform, ...] | None = None

    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, value: object) -> object:
        return geometry_model_from_value(value)

    @field_validator("offsets")
    @classmethod
    def _validate_offsets(
        cls,
        value: tuple[Sequence[object], ...] | None,
    ) -> tuple[Vec3, ...] | None:
        if value is None:
            return None
        offsets = tuple(_normalize_vec3(item, field_name="offsets") for item in value)
        if not offsets:
            raise ValueError("offsets must contain at least one offset when provided")
        return offsets

    @field_validator("transforms")
    @classmethod
    def _validate_transforms(
        cls,
        value: tuple[GeometryTransform | Mapping[str, object], ...] | None,
    ) -> tuple[GeometryTransform, ...] | None:
        if value is None:
            return None
        transforms = tuple(
            item if isinstance(item, GeometryTransform) else GeometryTransform.model_validate(item)
            for item in value
        )
        if not transforms:
            raise ValueError("transforms must contain at least one transform when provided")
        for transform in transforms:
            if any(abs(component) > 1e-12 for component in transform.origin):
                raise ValueError(
                    "GeometryArray transforms must be linear-only; use offsets for translation"
                )
        return transforms

    @model_validator(mode="after")
    def _validate_instance_lengths(self) -> GeometryArray:
        if self.offsets is None and self.transforms is None:
            return self
        if self.offsets is not None and self.transforms is not None:
            if len(self.offsets) != len(self.transforms):
                raise ValueError("offsets and transforms must have the same number of instances")
            return self
        return self

    @property
    def instance_count(self) -> int:
        if self.offsets is not None:
            return len(self.offsets)
        if self.transforms is not None:
            return len(self.transforms)
        return 1

    @property
    def instances(self) -> tuple[GeometryWrapperModel, ...]:
        instances: list[GeometryWrapperModel] = []
        offsets = self.offsets or tuple((0.0, 0.0, 0.0) for _ in range(self.instance_count))
        transforms = self.transforms or tuple(
            GeometryTransform.identity() for _ in range(self.instance_count)
        )
        for transform, offset in zip(transforms, offsets, strict=True):
            instance: GeometryWrapperModel = self.geometry
            if transform != GeometryTransform.identity():
                instance = Transformed(geometry=instance, transform=transform)
            if any(abs(component) > 1e-12 for component in offset):
                instance = _geometry_translate(instance, offset)
            instances.append(instance)
        return tuple(instances)

    @property
    def bounds(self) -> Bounds3:
        return _bounds_union(tuple(instance.bounds for instance in self.instances))

    def contains_point(self, point: Sequence[object]) -> bool:
        normalized = _normalize_vec3(point, field_name="point")
        return any(_geometry_contains_point(instance, normalized) for instance in self.instances)

    def translate(self, offset: Sequence[object]) -> GeometryArray:
        delta = _normalize_vec3(offset, field_name="offset")
        if self.offsets is None and self.transforms is None:
            return self.copy_update(geometry=_geometry_translate(self.geometry, delta))
        updated_offsets = tuple(
            tuple(
                offset_component + delta_component
                for offset_component, delta_component in zip(offset, delta, strict=True)
            )
            for offset in (
                self.offsets or tuple((0.0, 0.0, 0.0) for _ in range(self.instance_count))
            )
        )
        return self.copy_update(offsets=updated_offsets)

    def as_geometry_group(self) -> GeometryGroup:
        return GeometryGroup(geometries=self.instances)


LeafGeometryModel = Box | Sphere | Cylinder | PolySlab
CompositeGeometryModel = GeometryGroup | Transformed | ClipOperation | GeometryArray
GeometryWrapperModel = LeafGeometryModel | CompositeGeometryModel


def _geometry_contains_point(geometry: GeometryWrapperModel, point: Vec3) -> bool:
    return geometry.contains_point(point)


def _geometry_translate(geometry: GeometryWrapperModel, offset: Vec3) -> GeometryWrapperModel:
    return geometry.translate(offset)


def composite_geometry_model_from_value(value: object) -> CompositeGeometryModel:
    """Normalize a composite wrapper payload into its typed public model."""
    if isinstance(value, (GeometryGroup, Transformed, ClipOperation, GeometryArray)):
        return value
    if isinstance(value, Mapping):
        geometry_type = _component_type_name(value)
        if geometry_type == "GeometryGroup":
            return GeometryGroup.model_validate(value)
        if geometry_type == "Transformed":
            return Transformed.model_validate(value)
        if geometry_type == "ClipOperation":
            return ClipOperation.model_validate(value)
        if geometry_type == "GeometryArray":
            return GeometryArray.model_validate(value)
    raise TypeError(f"unsupported composite geometry payload {value!r}")


def geometry_model_from_value(value: object) -> GeometryWrapperModel:
    """Normalize a supported geometry payload into its typed public model."""
    if isinstance(value, (Box, Sphere, Cylinder)):
        return value
    if isinstance(value, PolySlab):
        return value
    if isinstance(value, (GeometryGroup, Transformed, ClipOperation, GeometryArray)):
        return value
    if isinstance(value, Mapping):
        geometry_type = _component_type_name(value)
        if geometry_type == "PolySlab":
            return polyslab_model_from_value(value)
        if geometry_type in _WRAPPER_TYPES:
            return composite_geometry_model_from_value(value)
        if geometry_type == "Box":
            return Box.model_validate(value)
        if geometry_type == "Sphere":
            return Sphere.model_validate(value)
        if geometry_type == "Cylinder":
            return Cylinder.model_validate(value)
    raise TypeError(f"unsupported geometry payload {value!r}")
