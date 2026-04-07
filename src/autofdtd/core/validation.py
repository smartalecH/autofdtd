"""Checklist-aware normalization and validation helpers for Phase 1 models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import SupportsFloat, cast

from pydantic import BaseModel

from autofdtd.boundaries import (
    ABCBoundary,
    Absorber,
    BlochBoundary,
    BroadbandModeABCFitterParam,
    BroadbandModeABCSpec,
    Boundary,
    BoundarySpec,
    ModeABCBoundary,
    PECBoundary,
    PML,
    PMCBoundary,
    Periodic,
    StablePML,
    boundary_spec_model_from_value,
)
from autofdtd.geometry import (
    ClipOperation,
    GeometryArray,
    GeometryGroup,
    PolySlab,
    Transformed,
    geometry_model_from_value,
)
from autofdtd.geometry.polyslab import _normalize_vertices
from autofdtd.geometry.primitives import Box, Cylinder, Sphere
from autofdtd.grid import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    SubpixelSpec,
    UniformGrid,
    grid_model_from_value,
    subpixel_model_from_value,
)
from autofdtd.materials import (
    AnisotropicMedium,
    Debye,
    Drude,
    Lorentz,
    LossyMetalMedium,
    Medium,
    PECMedium,
    PMCMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    PoleResidue,
    Sellmeier,
    medium_model_from_value,
)
from autofdtd.planning import FeatureStatus, feature_entry
from autofdtd.monitors import monitor_model_from_value
from autofdtd.sources import UniformCurrentSource, current_source_model_from_value

Vec3 = tuple[float, float, float]
_IMPLEMENTED_BOUNDS_GEOMETRY = frozenset(
    {
        "Box",
        "Sphere",
        "Cylinder",
        "PolySlab",
        "GeometryGroup",
        "Transformed",
        "ClipOperation",
        "GeometryArray",
    }
)
_ABSORBING_BOUNDARY_TYPES = frozenset({"PML", "StablePML", "Absorber"})
_PARTIAL_GEOMETRY_TYPES = frozenset({"ClipOperation"})
_PHASE1_BOUNDARY_TYPES = frozenset(
    {
        "BoundarySpec",
        "Boundary",
        "Periodic",
        "BlochBoundary",
        "PECBoundary",
        "PMCBoundary",
        "ABCBoundary",
        "ModeABCBoundary",
        "BroadbandModeABCSpec",
        "BroadbandModeABCFitterParam",
        "PML",
        "PMLParams",
        "StablePML",
        "Absorber",
        "AbsorberParams",
    }
)
_PHASE1_GRID_TYPES = frozenset(
    {
        "GridSpec",
        "UniformGrid",
        "CustomGrid",
        "CustomGridBoundaries",
        "AutoGrid",
        "GridRefinement",
        "LayerRefinementSpec",
    }
)
_PHASE1_SUBPIXEL_TYPES = frozenset({"SubpixelSpec", "Staircasing", "PolarizedAveraging"})
_IMPLEMENTED_MEDIUM_TYPES = frozenset(
    {
        "Medium",
        "PECMedium",
        "PMCMedium",
        "LossyMetalMedium",
        "PoleResidue",
        "Sellmeier",
        "Lorentz",
        "Drude",
        "Debye",
        "AnisotropicMedium",
        "PerturbationMedium",
        "PerturbationPoleResidue",
    }
)
_PLANNED_GRID_TYPES = frozenset(
    {
        "QuasiUniformGrid",
    }
)
_PHASE1_MONITOR_TYPES = frozenset({
    "Monitor",
    "FieldMonitor",
    "FieldTimeMonitor",
    "AuxFieldTimeMonitor",
    "FluxMonitor",
    "FluxTimeMonitor",
    "ModeMonitor",
    "ModeSolverMonitor",
    "MediumMonitor",
    "PermittivityMonitor",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceMonitor",
    "DiffractionMonitor",
    "DirectivityMonitor",
    "GaussianOverlapMonitor",
    "AstigmaticGaussianOverlapMonitor",
    "SurfaceFieldMonitor",
    "SurfaceFieldTimeMonitor",
})


class AutoFDTDValidationWarning(UserWarning):
    """Warning category for non-fatal normalization and coercion behavior."""


class UnsupportedFeatureError(ValueError):
    """Raised when a feature is explicitly deferred or rejected in Phase 1."""


def normalize_name(name: str | None) -> str | None:
    """Normalize optional names by trimming whitespace and rejecting empties."""

    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        raise ValueError("names must not be empty or whitespace-only")
    return stripped


def normalize_vec3(value: Sequence[object], *, field_name: str) -> Vec3:
    """Coerce a 3-vector to a finite float tuple."""

    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three components")
    normalized = (
        float(cast(SupportsFloat, value[0])),
        float(cast(SupportsFloat, value[1])),
        float(cast(SupportsFloat, value[2])),
    )
    if any(not math.isfinite(component) for component in normalized):
        raise ValueError(f"{field_name} components must be finite")
    return normalized


def component_type_name(component: object) -> str | None:
    """Extract the tagged type field from a component-like payload."""

    if isinstance(component, BaseModel) and hasattr(component, "type"):
        value = component.type
        return str(value) if value is not None else None
    if isinstance(component, Mapping):
        value = component.get("type")
        return str(value) if value is not None else None
    return None


def normalize_component(component: object, *, context: str) -> object:
    """Normalize a tagged component payload and reject unsupported feature tags."""

    normalized = _normalize_name_field(component)
    _validate_supported_types(normalized, context=context)
    if context == "geometry":
        return _normalize_geometry(normalized)
    if context == "grid_spec":
        return _normalize_grid(normalized)
    if context == "subpixel":
        return _normalize_subpixel(normalized)
    if context == "boundary_spec":
        return _normalize_boundary_spec(normalized)
    if context in {"medium", "background_medium", "scene.medium"}:
        return _normalize_medium(normalized)
    if context == "source":
        return _normalize_source(normalized)
    if context == "monitor":
        return _normalize_monitor(normalized)
    return normalized


def validate_simulation_bounds(
    *,
    center: Vec3,
    size: Vec3,
    structures: Sequence[object],
) -> None:
    """Validate known structure bounds against the simulation domain."""

    sim_min = tuple(origin - extent / 2.0 for origin, extent in zip(center, size, strict=True))
    sim_max = tuple(origin + extent / 2.0 for origin, extent in zip(center, size, strict=True))

    for structure in structures:
        geometry = getattr(structure, "geometry", None)
        bounds = geometry_bounds(geometry)
        if bounds is None:
            continue
        geom_min, geom_max = bounds
        structure_name = getattr(structure, "name", None) or "<unnamed>"
        if any(
            geom_max[axis] < sim_min[axis] or geom_min[axis] > sim_max[axis] for axis in range(3)
        ):
            raise ValueError(
                f"structure {structure_name!r} lies fully outside the simulation bounds"
            )
        if any(low < allowed for low, allowed in zip(geom_min, sim_min, strict=True)) or any(
            high > allowed for high, allowed in zip(geom_max, sim_max, strict=True)
        ):
            warnings.warn(
                f"structure {structure_name!r} extends beyond the simulation bounds and will "
                "require clipping during scene compilation",
                AutoFDTDValidationWarning,
                stacklevel=3,
            )


def coerce_zero_dim_boundaries(boundary_spec: object, *, size: Vec3) -> object:
    """Coerce absorbing boundaries to periodic boundaries on zero-sized axes."""

    if isinstance(boundary_spec, BoundarySpec):
        updated = {}
        changed = False
        for axis_index, axis_name in enumerate("xyz"):
            axis_boundary = boundary_spec[axis_name]
            if size[axis_index] != 0.0:
                updated[axis_name] = axis_boundary
                continue
            coerced = _coerce_axis_boundary(axis_boundary, axis_name=axis_name)
            updated[axis_name] = coerced
            changed = changed or coerced is not axis_boundary
        return boundary_spec.copy_update(**updated) if changed else boundary_spec

    if not isinstance(boundary_spec, Mapping):
        return boundary_spec

    updated = dict(boundary_spec)
    changed = False

    for axis_index, axis_name in enumerate("xyz"):
        if size[axis_index] != 0.0:
            continue
        axis_boundary = updated.get(axis_name)
        if axis_boundary is None:
            continue
        coerced = _coerce_axis_boundary(axis_boundary, axis_name=axis_name)
        if coerced is not axis_boundary:
            updated[axis_name] = coerced
            changed = True

    return updated if changed else boundary_spec


def geometry_bounds(geometry: object) -> tuple[Vec3, Vec3] | None:
    """Return axis-aligned bounds for the known primitive geometries."""

    if isinstance(
        geometry,
        (
            Box,
            Sphere,
            Cylinder,
            PolySlab,
            GeometryGroup,
            Transformed,
            ClipOperation,
            GeometryArray,
        ),
    ):
        return geometry.bounds

    if not isinstance(geometry, Mapping):
        return None
    geometry_type = component_type_name(geometry)
    if geometry_type not in _IMPLEMENTED_BOUNDS_GEOMETRY:
        return None
    required_fields = {
        "Box": {"size"},
        "Sphere": {"radius"},
        "Cylinder": {"radius", "length", "axis"},
        "PolySlab": {"vertices", "slab_bounds"},
        "GeometryGroup": {"geometries"},
        "Transformed": {"geometry", "transform"},
        "ClipOperation": {"operation", "geometry_a", "geometry_b"},
        "GeometryArray": {"geometry"},
    }[geometry_type]
    if not required_fields <= geometry.keys():
        return None
    return geometry_model_from_value(geometry).bounds


def geometry_contains_point(geometry: object, point: Sequence[object]) -> bool:
    """Return whether the point lies inside a known primitive geometry."""

    normalized_point = normalize_vec3(point, field_name="point")
    if isinstance(
        geometry,
        (
            Box,
            Sphere,
            Cylinder,
            PolySlab,
            GeometryGroup,
            Transformed,
            ClipOperation,
            GeometryArray,
        ),
    ):
        return geometry.contains_point(normalized_point)
    if (
        isinstance(geometry, Mapping)
        and component_type_name(geometry) in _IMPLEMENTED_BOUNDS_GEOMETRY
    ):
        return geometry_model_from_value(geometry).contains_point(normalized_point)
    raise TypeError(f"geometry containment is not available for {type(geometry)!r}")


def _normalize_name_field(component: object) -> object:
    if isinstance(component, BaseModel) and hasattr(component, "name"):
        name = normalize_name(component.name)
        if name != component.name:
            return component.model_copy(update={"name": name})
        return component
    if isinstance(component, Mapping) and "name" in component:
        normalized_name = normalize_name(component.get("name"))
        if normalized_name != component.get("name"):
            updated = dict(component)
            updated["name"] = normalized_name
            return updated
    return component


def _validate_supported_types(component: object, *, context: str) -> None:
    for feature_name in _iter_type_tags(component):
        if context == "geometry" and feature_name in {"ClipOperation", "GeometryTransform"}:
            continue
        entry = _feature_entry_for_type(feature_name, context=context)
        if entry is None:
            raise UnsupportedFeatureError(
                f"{context} uses unknown feature type {feature_name!r}; add it to the Phase 1 "
                "feature matrix before accepting it in the public API"
            )
        if entry.status is FeatureStatus.IMPLEMENT:
            continue
        status_text = {
            FeatureStatus.DEFER: "deferred",
            FeatureStatus.REJECT_CLEARLY: "rejected clearly",
        }[entry.status]
        raise UnsupportedFeatureError(
            f"{context} uses {status_text} feature {feature_name!r}. {entry.notes}"
        )


def _feature_entry_for_type(feature_name: str, *, context: str):
    entry = feature_entry(feature_name)
    if entry is not None:
        return entry
    if context == "boundary_spec" and feature_name == "PMLParams":
        return feature_entry("PML")
    if context == "boundary_spec" and feature_name == "AbsorberParams":
        return feature_entry("Absorber")
    if context == "boundary_spec" and feature_name in {
        "BroadbandModeABCSpec",
        "BroadbandModeABCFitterParam",
    }:
        return feature_entry("ModeABCBoundary")
    if context in {"medium", "background_medium", "scene.medium"}:
        if feature_name.startswith("Custom"):
            return feature_entry("CustomMedia")
    return None


def _iter_type_tags(value: object) -> Iterable[str]:
    if isinstance(value, BaseModel):
        yield from _iter_type_tags(value.model_dump(mode="python", exclude_none=True))
        return
    if isinstance(value, Mapping):
        tagged_type = value.get("type")
        if tagged_type is not None:
            yield str(tagged_type)
        for nested in value.values():
            yield from _iter_type_tags(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_type_tags(item)


def _normalize_geometry(component: object) -> object:
    if not isinstance(component, Mapping):
        return component

    geometry_type = component_type_name(component)
    if geometry_type not in _IMPLEMENTED_BOUNDS_GEOMETRY:
        return component

    updated = dict(component)
    if geometry_type in {"GeometryGroup", "Transformed", "ClipOperation", "GeometryArray"}:
        if (
            geometry_type in _PARTIAL_GEOMETRY_TYPES
            and not _supports_phase1_clip_operation(updated)
        ):
            operation = updated.get("operation")
            raise ValueError(
                f"geometry.operation={operation!r} is not supported in Phase 1; only "
                "'union', 'intersection', and 'difference' ClipOperation wrappers are implemented"
            )
        return geometry_model_from_value(updated).model_dump(mode="python", exclude_none=True)
    if "center" in updated:
        updated["center"] = normalize_vec3(updated["center"], field_name="geometry.center")
    if geometry_type == "Box":
        if "size" not in updated:
            return updated
        size = normalize_vec3(updated["size"], field_name="geometry.size")
        if any(component < 0.0 for component in size):
            raise ValueError("geometry.size components must be non-negative")
        if all(component == 0.0 for component in size):
            raise ValueError("geometry.size must include at least one positive extent")
        updated["size"] = size
        return updated
    if geometry_type == "Sphere":
        if "radius" not in updated:
            return updated
        updated["radius"] = _positive_float(updated.get("radius"), field_name="geometry.radius")
        return updated
    if geometry_type == "PolySlab":
        if "vertices" in updated:
            updated["vertices"] = _normalize_vertices(updated["vertices"])
        if "slab_bounds" in updated:
            updated["slab_bounds"] = _normalize_slab_bounds(updated["slab_bounds"])
        if "axis" in updated:
            updated["axis"] = _normalize_axis(updated.get("axis"))
        if "sidewall_angle" in updated:
            updated["sidewall_angle"] = _phase1_zero_only(
                updated.get("sidewall_angle"),
                field_name="geometry.sidewall_angle",
            )
        if "dilation" in updated:
            updated["dilation"] = _phase1_zero_only(
                updated.get("dilation"),
                field_name="geometry.dilation",
            )
        if {"vertices", "slab_bounds"} <= updated.keys():
            return geometry_model_from_value(updated).model_dump(mode="python", exclude_none=True)
        return updated

    if not {"radius", "length", "axis"} <= updated.keys():
        return updated
    updated["radius"] = _positive_float(updated.get("radius"), field_name="geometry.radius")
    updated["length"] = _non_negative_float(updated.get("length"), field_name="geometry.length")
    updated["axis"] = _normalize_axis(updated.get("axis"))
    return updated


def _normalize_grid(component: object) -> object:
    if isinstance(
        component,
        (
            GridSpec,
            UniformGrid,
            CustomGrid,
            CustomGridBoundaries,
            AutoGrid,
            GridRefinement,
            LayerRefinementSpec,
        ),
    ):
        return component
    if not isinstance(component, Mapping):
        return component

    supported_tags = _PHASE1_GRID_TYPES | _PLANNED_GRID_TYPES
    grid_types = {tag for tag in _iter_type_tags(component) if tag in supported_tags}
    unsupported = sorted(grid_types & _PLANNED_GRID_TYPES)
    if unsupported:
        feature_list = ", ".join(repr(name) for name in unsupported)
        raise UnsupportedFeatureError(
            f"grid_spec includes planned-but-not-yet-implemented grid feature(s) {feature_list}; "
            "Phase 1 currently supports GridSpec with UniformGrid, CustomGrid, "
            "CustomGridBoundaries, AutoGrid, and layer refinement metadata"
        )

    grid_type = component_type_name(component)
    if grid_type != "GridSpec":
        raise ValueError("simulation.grid_spec must use a top-level 'GridSpec' container")
    return grid_model_from_value(component).model_dump(mode="python", exclude_none=True)


def _normalize_subpixel(component: object) -> object:
    if isinstance(component, (bool, SubpixelSpec)):
        return subpixel_model_from_value(component)
    if not isinstance(component, Mapping):
        return component
    component_type = component_type_name(component)
    if component_type not in _PHASE1_SUBPIXEL_TYPES:
        raise ValueError("simulation.subpixel must be a bool or a SubpixelSpec payload in Phase 1")
    return subpixel_model_from_value(component)


def _normalize_boundary_spec(component: object) -> object:
    if isinstance(component, BoundarySpec):
        return component
    if not isinstance(component, Mapping):
        return component
    boundary_type = component_type_name(component)
    if boundary_type != "BoundarySpec":
        raise ValueError("simulation.boundary_spec must use a top-level 'BoundarySpec' container")
    boundary_tags = set(_iter_type_tags(component))
    if boundary_tags and boundary_tags <= _PHASE1_BOUNDARY_TYPES:
        return boundary_spec_model_from_value(component).model_dump(
            mode="python", exclude_none=True
        )
    return component


def _normalize_medium(component: object) -> object:
    if isinstance(
        component,
        (
            Medium,
            PECMedium,
            PMCMedium,
            LossyMetalMedium,
            PoleResidue,
            Sellmeier,
            Lorentz,
            Drude,
            Debye,
            PerturbationMedium,
            PerturbationPoleResidue,
            AnisotropicMedium,
        ),
    ):
        return component
    if not isinstance(component, Mapping):
        return component
    medium_type = component_type_name(component)
    if medium_type not in _IMPLEMENTED_MEDIUM_TYPES:
        return component
    return medium_model_from_value(component).model_dump(mode="python", exclude_none=True)


def _normalize_source(component: object) -> object:
    if isinstance(component, UniformCurrentSource):
        return component
    if not isinstance(component, Mapping):
        return component
    source_type = component_type_name(component)
    if source_type != "UniformCurrentSource":
        return component
    return current_source_model_from_value(component).model_dump(mode="python", exclude_none=True)


def _normalize_monitor(component: object) -> object:
    """Normalize a monitor component using the monitor model factory."""
    if isinstance(component, BaseModel):
        return component
    if not isinstance(component, Mapping):
        return component
    monitor_type = component_type_name(component)
    # If it's a recognized monitor type, validate via the model factory
    if monitor_type is not None and monitor_type != "Monitor":
        return monitor_model_from_value(component).model_dump(mode="python", exclude_none=True)
    return component


def _supports_phase1_clip_operation(component: object) -> bool:
    if isinstance(component, ClipOperation):
        return component.operation in {"union", "intersection", "difference"}
    if isinstance(component, Mapping) and component_type_name(component) == "ClipOperation":
        operation = component.get("operation")
        return operation in {"union", "intersection", "difference"}
    return False


def _normalize_axis(value: object) -> int:
    if value in (0, 1, 2):
        return int(value)
    axis_map = {"x": 0, "y": 1, "z": 2}
    if isinstance(value, str) and value.lower() in axis_map:
        return axis_map[value.lower()]
    raise ValueError("geometry.axis must be one of 0, 1, 2, 'x', 'y', or 'z'")


def _positive_float(value: object, *, field_name: str) -> float:
    normalized = float(cast(SupportsFloat, value))
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value")
    return normalized


def _non_negative_float(value: object, *, field_name: str) -> float:
    normalized = float(cast(SupportsFloat, value))
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite value")
    return normalized


def _phase1_zero_only(value: object, *, field_name: str) -> float:
    normalized = float(cast(SupportsFloat, value))
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if abs(normalized) > 1e-12:
        raise ValueError(
            f"{field_name}={normalized} is not supported in Phase 1; "
            "only constant-cross-section PolySlab extrusions are implemented"
        )
    return 0.0


def _normalize_slab_bounds(value: Sequence[object]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("geometry.slab_bounds must contain exactly two coordinates")
    lower = float(cast(SupportsFloat, value[0]))
    upper = float(cast(SupportsFloat, value[1]))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("geometry.slab_bounds must be finite")
    if upper <= lower:
        raise ValueError("geometry.slab_bounds must be strictly increasing")
    return (lower, upper)


def _coerce_axis_boundary(boundary: object, *, axis_name: str) -> object:
    if isinstance(boundary, Boundary):
        updated = {}
        changed = False
        for side in ("minus", "plus"):
            edge = boundary[side]
            coerced = _coerce_axis_boundary(edge, axis_name=axis_name)
            updated[side] = coerced
            changed = changed or coerced is not edge
        return boundary.copy_update(**updated) if changed else boundary
    if isinstance(boundary, (Periodic, BlochBoundary, PECBoundary, PMCBoundary)):
        return boundary
    if isinstance(boundary, (PML, StablePML, Absorber)):
        warnings.warn(
            f"zero-sized axis {axis_name!r} cannot use absorbing boundaries; coercing to Periodic",
            AutoFDTDValidationWarning,
            stacklevel=3,
        )
        return Periodic()
    if not isinstance(boundary, Mapping):
        return boundary

    boundary_type = component_type_name(boundary)
    if boundary_type == "BlochBoundary":
        raise ValueError(f"zero-sized axis {axis_name!r} cannot use BlochBoundary")
    if boundary_type in _ABSORBING_BOUNDARY_TYPES:
        warnings.warn(
            f"zero-sized axis {axis_name!r} cannot use absorbing boundaries; coercing to Periodic",
            AutoFDTDValidationWarning,
            stacklevel=3,
        )
        return {"type": "Periodic"}
    if boundary_type != "Boundary":
        return boundary

    updated = dict(boundary)
    changed = False
    for side in ("minus", "plus"):
        edge = updated.get(side)
        if isinstance(edge, Mapping) and component_type_name(edge) in _ABSORBING_BOUNDARY_TYPES:
            warnings.warn(
                f"zero-sized axis {axis_name!r} cannot use absorbing boundaries; coercing "
                f"{axis_name}.{side} to Periodic",
                AutoFDTDValidationWarning,
                stacklevel=3,
            )
            updated[side] = {"type": "Periodic"}
            changed = True
    return updated if changed else boundary
