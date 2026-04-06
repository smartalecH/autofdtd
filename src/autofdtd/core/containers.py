"""Tagged immutable container models for Phase 1 scene and simulation setup."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.core.validation import (
    coerce_zero_dim_boundaries,
    component_type_name,
    geometry_bounds,
    geometry_contains_point,
    normalize_component,
    normalize_name,
    normalize_vec3,
    validate_simulation_bounds,
)
from autofdtd.grid import ResolvedGrid, resolve_grid_spec
from autofdtd.version import __version__

Vec3 = tuple[float, float, float]
TNamed = TypeVar("TNamed")


class StructurePriorityMode(StrEnum):
    """Policy for resolving implicit structure priorities."""

    EQUAL = "equal"
    CONDUCTOR = "conductor"


TaggedContainerModel = TaggedModel


def _validate_name(name: str | None) -> str | None:
    return normalize_name(name)


def _default_priority_for_medium(mode: StructurePriorityMode, medium: object) -> int:
    if mode is not StructurePriorityMode.CONDUCTOR:
        return 0

    medium_type = component_type_name(medium)
    if medium_type == "PECMedium":
        return 100
    if medium_type == "LossyMetalMedium":
        return 90
    return 0


def _extract_name(value: object) -> str | None:
    if isinstance(value, BaseModel):
        candidate = getattr(value, "name", None)
        return str(candidate) if candidate is not None else None
    if isinstance(value, Mapping):
        candidate = value.get("name")
        return str(candidate) if candidate is not None else None
    return None


def _assert_unique_names(values: tuple[TNamed, ...], field_name: str) -> tuple[TNamed, ...]:
    names = [name for item in values if (name := _extract_name(item))]
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} names must be unique; got {names!r}")
    return values


class Structure(TaggedModel):
    """Geometry plus medium shell with explicit overlap precedence semantics."""

    type: str = Field(default="Structure")
    geometry: object
    medium: object
    name: str | None = None
    priority: int | None = None
    background_medium: object | None = None

    @field_validator("name")
    @classmethod
    def _validate_name_field(cls, value: str | None) -> str | None:
        return _validate_name(value)

    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, value: object) -> object:
        return normalize_component(value, context="geometry")

    @field_validator("medium")
    @classmethod
    def _validate_medium(cls, value: object) -> object:
        return normalize_component(value, context="medium")

    @field_validator("background_medium")
    @classmethod
    def _validate_background_medium(cls, value: object | None) -> object | None:
        if value is None:
            return None
        return normalize_component(value, context="background_medium")

    def resolved_priority(
        self, priority_mode: StructurePriorityMode = StructurePriorityMode.EQUAL
    ) -> int:
        """Resolve the effective priority under the scene or simulation policy."""
        if self.priority is not None:
            return self.priority
        return _default_priority_for_medium(priority_mode, self.medium)

    def bounds(self) -> tuple[Vec3, Vec3] | None:
        """Return geometry bounds when the geometry family supports them."""
        return geometry_bounds(self.geometry)

    def contains_point(self, point: Vec3) -> bool:
        """Check whether a world-space point lies inside the structure geometry."""
        return geometry_contains_point(self.geometry, point)


class StructurePrecedence(TaggedModel):
    """Explicit precedence metadata for ordered structure materialization."""

    type: Literal["StructurePrecedence"] = "StructurePrecedence"
    structure: Structure
    source_index: int
    resolved_priority: int
    precedence_rank: int


class Scene(TaggedModel):
    """Ordered structure container with named access helpers."""

    type: str = Field(default="Scene")
    medium: object = Field(default_factory=dict)
    structures: tuple[Structure, ...] = ()
    structure_priority_mode: StructurePriorityMode = StructurePriorityMode.EQUAL

    @field_validator("medium")
    @classmethod
    def _validate_scene_medium(cls, value: object) -> object:
        return normalize_component(value, context="scene.medium")

    @field_validator("structures")
    @classmethod
    def _validate_structure_names(cls, value: tuple[Structure, ...]) -> tuple[Structure, ...]:
        return _assert_unique_names(value, "structures")

    def structure_names(self) -> tuple[str, ...]:
        return tuple(structure.name for structure in self.structures if structure.name is not None)

    def structures_by_name(self) -> OrderedDict[str, Structure]:
        return OrderedDict(
            (structure.name, structure)
            for structure in self.structures
            if structure.name is not None
        )

    def get_structure(self, name: str) -> Structure:
        try:
            return self.structures_by_name()[name]
        except KeyError as exc:
            raise KeyError(f"unknown structure {name!r}") from exc

    def structure_precedence(self) -> tuple[StructurePrecedence, ...]:
        """Return explicit precedence metadata in materialization order."""
        indexed = list(enumerate(self.structures))
        indexed.sort(
            key=lambda item: (
                item[1].resolved_priority(self.structure_priority_mode),
                item[0],
            )
        )
        return tuple(
            StructurePrecedence(
                structure=structure,
                source_index=source_index,
                resolved_priority=structure.resolved_priority(self.structure_priority_mode),
                precedence_rank=precedence_rank,
            )
            for precedence_rank, (source_index, structure) in enumerate(indexed)
        )

    def structures_in_resolution_order(self) -> tuple[Structure, ...]:
        """Return structures in the order they should be applied during materialization."""
        return tuple(entry.structure for entry in self.structure_precedence())

    def resolve_structure_stack_at_point(self, point: Vec3) -> tuple[StructurePrecedence, ...]:
        """Return overlapping structures at a point in precedence order."""
        normalized = normalize_vec3(point, field_name="point")
        return tuple(
            entry
            for entry in self.structure_precedence()
            if entry.structure.contains_point(normalized)
        )

    def resolve_structure_at_point(self, point: Vec3) -> Structure | None:
        """Resolve the winning structure at a world-space point under precedence rules."""
        stack = self.resolve_structure_stack_at_point(point)
        return stack[-1].structure if stack else None


class Simulation(Scene):
    """Top-level Phase 1 simulation shell with tagged container fields."""

    type: str = Field(default="Simulation")
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3
    run_time: float
    sources: tuple[object, ...] = ()
    monitors: tuple[object, ...] = ()
    boundary_spec: object | None = None
    grid_spec: object | None = None
    symmetry: tuple[int, int, int] = (0, 0, 0)
    shutoff: float | None = None
    courant: float = 0.99
    subpixel: object | None = None
    normalize_index: int | None = None
    relax_courant: bool = False
    low_freq_smoothing: object | None = None
    lumped_elements: tuple[object, ...] = ()
    version: str = __version__

    def resolved_grid(self) -> ResolvedGrid | None:
        """Resolve the simulation grid_spec against the simulation domain."""
        return resolve_grid_spec(self.grid_spec, center=self.center, size=self.size)

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Vec3) -> Vec3:
        return normalize_vec3(value, field_name="center")

    @field_validator("sources")
    @classmethod
    def _normalize_sources(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(normalize_component(item, context="source") for item in value)

    @field_validator("monitors")
    @classmethod
    def _normalize_monitors(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(normalize_component(item, context="monitor") for item in value)

    @field_validator("sources")
    @classmethod
    def _validate_source_names(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return _assert_unique_names(value, "sources")

    @field_validator("monitors")
    @classmethod
    def _validate_monitor_names(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return _assert_unique_names(value, "monitors")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Vec3) -> Vec3:
        value = normalize_vec3(value, field_name="size")
        if any(component < 0.0 for component in value):
            raise ValueError("size components must be non-negative")
        return value

    @field_validator("run_time")
    @classmethod
    def _validate_run_time(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("run_time must be positive")
        return value

    @field_validator("shutoff")
    @classmethod
    def _validate_shutoff(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError("shutoff must be non-negative")
        return value

    @field_validator("courant")
    @classmethod
    def _validate_courant(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("courant must be in the interval (0, 1]")
        return value

    @field_validator("boundary_spec")
    @classmethod
    def _validate_boundary_spec(cls, value: object | None) -> object | None:
        if value is None:
            return None
        return normalize_component(value, context="boundary_spec")

    @field_validator("grid_spec")
    @classmethod
    def _validate_grid_spec(cls, value: object | None) -> object | None:
        if value is None:
            return None
        return normalize_component(value, context="grid_spec")

    @field_validator("subpixel")
    @classmethod
    def _validate_subpixel(cls, value: object | None) -> object | None:
        if value is None:
            return None
        return normalize_component(value, context="subpixel")

    @field_validator("normalize_index")
    @classmethod
    def _validate_normalize_index(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("normalize_index must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_symmetry(self) -> Simulation:
        if len(self.symmetry) != 3:
            raise ValueError("symmetry must contain exactly three axes")
        for axis in self.symmetry:
            if axis not in (-1, 0, 1):
                raise ValueError("symmetry values must be -1, 0, or 1")
        if self.relax_courant:
            raise ValueError(
                "relax_courant is deferred in Phase 1; keep it disabled until grid planning lands"
            )
        if self.low_freq_smoothing is not None:
            raise ValueError(
                "low_freq_smoothing is deferred in Phase 1; omit it from the simulation shell"
            )
        if self.lumped_elements:
            raise ValueError(
                "lumped_elements are rejected clearly in Phase 1 and must not be supplied"
            )
        if self.normalize_index is not None:
            if self.normalize_index >= len(self.sources):
                raise ValueError(
                    f"normalize_index {self.normalize_index} is out of bounds for "
                    f"{len(self.sources)} sources"
                )
            source = self.sources[self.normalize_index]
            amplitude = getattr(getattr(source, "source_time", None), "amplitude", None)
            if amplitude == 0:
                raise ValueError("normalize_index cannot reference a zero-amplitude source")
        if self.boundary_spec is not None:
            object.__setattr__(
                self,
                "boundary_spec",
                coerce_zero_dim_boundaries(self.boundary_spec, size=self.size),
            )
        if self.grid_spec is not None:
            self.resolved_grid()
        validate_simulation_bounds(center=self.center, size=self.size, structures=self.structures)
        return self
