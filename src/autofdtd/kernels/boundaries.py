"""Boundary halo and PML update helpers for the foundational Phase 1 surface."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from autofdtd.compiler.boundaries import (
    BoundaryMode,
    CompiledABCCoefficients,
    CompiledBoundaryAxis,
    CompiledBoundaryEdge,
    CompiledBoundarySpec,
    CompiledPMLCoefficients,
    CompiledSymmetryAxis,
)
from autofdtd.core.models import AutoFDTDModel

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


class PMLFaceState(AutoFDTDModel):
    """Persistent per-face memory for the staged PML update."""

    type: str = "PMLFaceState"
    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    memory_shape: tuple[int, ...]
    memory: np.ndarray

    model_config = AutoFDTDModel.model_config | {"arbitrary_types_allowed": True}


class PMLBoundaryState(AutoFDTDModel):
    """Persistent state bundle for all active PML faces."""

    type: str = "PMLBoundaryState"
    electric: dict[str, PMLFaceState]
    magnetic: dict[str, PMLFaceState]

    model_config = AutoFDTDModel.model_config | {"arbitrary_types_allowed": True}


class ABCFaceState(AutoFDTDModel):
    """Persistent per-face state for first-order ABC ghost updates."""

    type: str = "ABCFaceState"
    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    value_shape: tuple[int, ...]
    previous_boundary: np.ndarray
    previous_adjacent: np.ndarray

    model_config = AutoFDTDModel.model_config | {"arbitrary_types_allowed": True}


class ABCBoundaryState(AutoFDTDModel):
    """Persistent state bundle for all active ABC faces."""

    type: str = "ABCBoundaryState"
    electric: dict[str, ABCFaceState]
    magnetic: dict[str, ABCFaceState]

    model_config = AutoFDTDModel.model_config | {"arbitrary_types_allowed": True}


def _ghost_and_source_slices(
    axis: int, side: str, *, ndim: int
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    ghost = [slice(None)] * ndim
    source = [slice(None)] * ndim
    if side == "minus":
        ghost[axis] = 0
        source[axis] = -2
    else:
        ghost[axis] = -1
        source[axis] = 1
    return tuple(ghost), tuple(source)


def _interior_reflection_slices(
    axis: int, side: str, *, ndim: int
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    ghost = [slice(None)] * ndim
    source = [slice(None)] * ndim
    if side == "minus":
        ghost[axis] = 0
        source[axis] = 1
    else:
        ghost[axis] = -1
        source[axis] = -2
    return tuple(ghost), tuple(source)


def _apply_edge(
    field: np.ndarray,
    axis_index: int,
    side: str,
    mode: BoundaryMode,
    signs: tuple[int, int, int],
    phase_factor: complex,
) -> None:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("boundary halo updates expect arrays shaped (nx, ny, nz, 3)")
    if min(field.shape[:3]) < 3:
        raise ValueError(
            "boundary halo updates require one ghost cell and one interior cell per side"
        )
    if mode is BoundaryMode.PERIODIC:
        ghost, source = _ghost_and_source_slices(axis_index, side, ndim=field.ndim)
        field[ghost] = field[source]
        return
    if mode is BoundaryMode.BLOCH:
        ghost, source = _ghost_and_source_slices(axis_index, side, ndim=field.ndim)
        field[ghost] = field[source] * field.dtype.type(phase_factor)
        return
    if mode in {BoundaryMode.ABC, BoundaryMode.PML, BoundaryMode.STABLE_PML}:
        return
    ghost, source = _interior_reflection_slices(axis_index, side, ndim=field.ndim)
    field[ghost] = field[source] * np.asarray(signs, dtype=field.dtype)


def _pml_region_slices(
    axis_index: int,
    side: Literal["minus", "plus"],
    num_layers: int,
    *,
    ndim: int,
) -> tuple[object, ...]:
    region = [slice(None)] * ndim
    if side == "minus":
        region[axis_index] = slice(1, 1 + num_layers)
    else:
        region[axis_index] = slice(-1 - num_layers, -1)
    return tuple(region)


def _validate_pml_capacity(field: np.ndarray, axis_index: int, num_layers: int) -> None:
    interior = field.shape[axis_index] - 2
    if num_layers > interior:
        raise ValueError(
            f"PML with {num_layers} layers does not fit along axis index {axis_index}; "
            f"only {interior} interior cells are available"
        )


def _apply_pml_region(
    field: np.ndarray,
    face_state: PMLFaceState,
    coefficients: CompiledPMLCoefficients,
    *,
    axis_index: int,
    side: Literal["minus", "plus"],
) -> None:
    region = _pml_region_slices(axis_index, side, coefficients.num_layers, ndim=field.ndim)
    region_values = field[region]
    line_shape = [1, 1, 1, 1]
    line_shape[axis_index] = coefficients.num_layers
    attenuation = np.asarray(coefficients.attenuation, dtype=field.real.dtype).reshape(line_shape)
    memory_decay = np.asarray(coefficients.memory_decay, dtype=field.real.dtype).reshape(line_shape)
    memory_drive = np.asarray(coefficients.memory_drive, dtype=field.real.dtype).reshape(line_shape)
    stretch = np.asarray(coefficients.stretch, dtype=field.real.dtype).reshape(line_shape)
    face_state.memory[...] = face_state.memory * memory_decay + region_values * memory_drive
    field[region] = region_values * attenuation * stretch


def _abc_state_slices(
    axis_index: int, side: Literal["minus", "plus"], *, ndim: int
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    ghost = [slice(None)] * ndim
    adjacent = [slice(None)] * ndim
    if side == "minus":
        ghost[axis_index] = 0
        adjacent[axis_index] = 1
    else:
        ghost[axis_index] = -1
        adjacent[axis_index] = -2
    return tuple(ghost), tuple(adjacent)


def _apply_abc_region(
    field: np.ndarray,
    face_state: ABCFaceState,
    coefficients: CompiledABCCoefficients,
    *,
    axis_index: int,
    side: Literal["minus", "plus"],
) -> None:
    ghost, adjacent = _abc_state_slices(axis_index, side, ndim=field.ndim)
    current_adjacent = np.array(field[adjacent], copy=True)
    next_boundary = (
        face_state.previous_adjacent
        + coefficients.reflection_coefficient * (current_adjacent - face_state.previous_boundary)
    ) * coefficients.attenuation
    field[ghost] = next_boundary
    face_state.previous_boundary[...] = next_boundary
    face_state.previous_adjacent[...] = current_adjacent


def boundary_edge_transform(
    values: np.ndarray,
    edge: CompiledBoundaryEdge,
    *,
    field_family: str,
) -> np.ndarray:
    """Apply the runtime transform associated with one boundary edge."""

    if edge.mode is BoundaryMode.PERIODIC:
        return np.array(values, copy=True)
    if edge.mode is BoundaryMode.BLOCH:
        return np.array(values, copy=True) * values.dtype.type(edge.phase_factor)
    if edge.mode in {BoundaryMode.ABC, BoundaryMode.PML, BoundaryMode.STABLE_PML}:
        return np.array(values, copy=True)
    signs_name = "electric_signs" if field_family == "electric" else "magnetic_signs"
    return np.array(values, copy=True) * np.asarray(getattr(edge, signs_name), dtype=values.dtype)


def symmetry_transform(
    values: np.ndarray,
    symmetry_axis: CompiledSymmetryAxis,
    *,
    field_family: str,
) -> np.ndarray:
    """Apply mirror-symmetry signs to vector values imported from a reflected chunk."""

    signs_name = "electric_signs" if field_family == "electric" else "magnetic_signs"
    return np.array(values, copy=True) * np.asarray(getattr(symmetry_axis, signs_name), dtype=values.dtype)


def apply_axis_boundary_ghosts(
    field: np.ndarray,
    axis_boundary: CompiledBoundaryAxis,
    *,
    field_family: str,
) -> np.ndarray:
    """Apply one axis boundary pair to a vector field with one ghost cell per face."""

    signs_name = "electric_signs" if field_family == "electric" else "magnetic_signs"
    updated = np.array(field, copy=True)
    axis_index = "xyz".index(axis_boundary.axis)
    _apply_edge(
        updated,
        axis_index,
        "minus",
        axis_boundary.minus.mode,
        getattr(axis_boundary.minus, signs_name),
        axis_boundary.minus.phase_factor,
    )
    _apply_edge(
        updated,
        axis_index,
        "plus",
        axis_boundary.plus.mode,
        getattr(axis_boundary.plus, signs_name),
        axis_boundary.plus.phase_factor,
    )
    return updated


def apply_boundary_ghosts(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    boundary_spec: CompiledBoundarySpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply compiled boundary ghost updates to electric and magnetic vector fields."""

    electric = np.array(electric_field, copy=True)
    magnetic = np.array(magnetic_field, copy=True)
    for axis_boundary in boundary_spec.axes():
        electric = apply_axis_boundary_ghosts(electric, axis_boundary, field_family="electric")
        magnetic = apply_axis_boundary_ghosts(magnetic, axis_boundary, field_family="magnetic")
    return electric, magnetic


def apply_pml_layers(
    field: np.ndarray,
    boundary_spec: CompiledBoundarySpec,
    state: PMLBoundaryState,
    *,
    field_family: Literal["electric", "magnetic"],
) -> np.ndarray:
    """Apply the staged PML damping update to one vector field family."""

    updated = np.array(field, copy=True)
    face_states = state.electric if field_family == "electric" else state.magnetic
    for axis_boundary in boundary_spec.axes():
        axis_index = "xyz".index(axis_boundary.axis)
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            if edge.mode not in {
                BoundaryMode.PML,
                BoundaryMode.STABLE_PML,
                BoundaryMode.ABSORBER,
            } or edge.pml is None:
                continue
            face_key = f"{axis_boundary.axis}.{side}"
            face_state = face_states[face_key]
            _validate_pml_capacity(updated, axis_index, edge.pml.num_layers)
            _apply_pml_region(
                updated,
                face_state,
                edge.pml,
                axis_index=axis_index,
                side=side,
            )
    return updated


def apply_abc_layers(
    field: np.ndarray,
    boundary_spec: CompiledBoundarySpec,
    state: ABCBoundaryState,
    *,
    field_family: Literal["electric", "magnetic"],
) -> np.ndarray:
    """Apply the staged first-order ABC update to one vector field family."""

    updated = np.array(field, copy=True)
    face_states = state.electric if field_family == "electric" else state.magnetic
    for axis_boundary in boundary_spec.axes():
        axis_index = "xyz".index(axis_boundary.axis)
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            if edge.mode is not BoundaryMode.ABC or edge.abc is None:
                continue
            face_key = f"{axis_boundary.axis}.{side}"
            _apply_abc_region(
                updated,
                face_states[face_key],
                edge.abc,
                axis_index=axis_index,
                side=side,
            )
    return updated


def apply_boundary_stages(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    boundary_spec: CompiledBoundarySpec,
    state: PMLBoundaryState | None = None,
    *,
    abc_state: ABCBoundaryState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply all compiled boundary stages in declared order."""

    electric = np.array(electric_field, copy=True)
    magnetic = np.array(magnetic_field, copy=True)
    for stage in boundary_spec.stage_order:
        if stage == "electric_boundary":
            electric = apply_boundary_ghosts(electric, magnetic, boundary_spec)[0]
        elif stage == "magnetic_boundary":
            magnetic = apply_boundary_ghosts(electric, magnetic, boundary_spec)[1]
        elif stage == "electric_abc":
            if abc_state is None:
                raise ValueError("ABC boundary stages require allocated ABCBoundaryState")
            electric = apply_abc_layers(electric, boundary_spec, abc_state, field_family="electric")
        elif stage == "magnetic_abc":
            if abc_state is None:
                raise ValueError("ABC boundary stages require allocated ABCBoundaryState")
            magnetic = apply_abc_layers(magnetic, boundary_spec, abc_state, field_family="magnetic")
        elif stage == "electric_pml":
            if state is None:
                raise ValueError("PML boundary stages require allocated PMLBoundaryState")
            electric = apply_pml_layers(electric, boundary_spec, state, field_family="electric")
        elif stage == "magnetic_pml":
            if state is None:
                raise ValueError("PML boundary stages require allocated PMLBoundaryState")
            magnetic = apply_pml_layers(magnetic, boundary_spec, state, field_family="magnetic")
    return electric, magnetic


def boundary_kernel_metadata() -> dict[str, Any]:
    """Expose the current boundary backend and staged update choices."""

    return {
        "backend": "warp" if WARP_AVAILABLE else "numpy",
        "warp_available": WARP_AVAILABLE,
        "module_contents_stable": True,
        "stages": (
            "electric_boundary",
            "magnetic_boundary",
            "electric_abc",
            "magnetic_abc",
            "electric_pml",
            "magnetic_pml",
        ),
        "supported_modes": (
            "periodic",
            "bloch",
            "pec",
            "pmc",
            "abc",
            "pml",
            "stable_pml",
            "absorber",
        ),
        "supports_complex_fields": True,
        "supports_symmetry_transforms": True,
        "supports_abc": True,
        "supports_pml_layers": True,
        "supports_stable_pml": True,
        "supports_absorber": True,
        "halo_depth": 1,
    }
