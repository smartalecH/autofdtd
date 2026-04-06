"""Runtime-facing boundary orchestration for the foundational Phase 1 surface."""

from __future__ import annotations

from typing import Literal

import numpy as np

from autofdtd.compiler.boundaries import (
    BoundaryMode,
    CompiledBoundaryEdge,
    CompiledBoundarySpec,
    CompiledSymmetryAxis,
    compile_boundary_spec,
)
from autofdtd.kernels.boundaries import (
    ABCBoundaryState,
    ABCFaceState,
    PMLBoundaryState,
    PMLFaceState,
    apply_boundary_stages,
    boundary_edge_transform,
    symmetry_transform,
)


def allocate_abc_boundary_state(
    compiled_boundary_spec: CompiledBoundarySpec,
    field_shape: tuple[int, int, int, int],
    *,
    dtype: np.dtype[np.floating] | np.dtype[np.complexfloating] = np.float64,
) -> ABCBoundaryState:
    """Allocate persistent first-order ABC state for all active faces."""

    if len(field_shape) != 4 or field_shape[-1] != 3:
        raise ValueError("ABC state allocation expects field shapes (nx, ny, nz, 3)")
    electric: dict[str, ABCFaceState] = {}
    magnetic: dict[str, ABCFaceState] = {}

    for axis_boundary in compiled_boundary_spec.axes():
        axis_index = "xyz".index(axis_boundary.axis)
        value_shape = tuple(
            size for index, size in enumerate(field_shape) if index != axis_index
        )
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            if edge.mode is not BoundaryMode.ABC or edge.abc is None:
                continue
            face_key = f"{axis_boundary.axis}.{side}"
            electric[face_key] = ABCFaceState(
                axis=axis_boundary.axis,
                side=side,
                value_shape=value_shape,
                previous_boundary=np.zeros(value_shape, dtype=dtype),
                previous_adjacent=np.zeros(value_shape, dtype=dtype),
            )
            magnetic[face_key] = ABCFaceState(
                axis=axis_boundary.axis,
                side=side,
                value_shape=value_shape,
                previous_boundary=np.zeros(value_shape, dtype=dtype),
                previous_adjacent=np.zeros(value_shape, dtype=dtype),
            )
    return ABCBoundaryState(electric=electric, magnetic=magnetic)


def allocate_pml_boundary_state(
    compiled_boundary_spec: CompiledBoundarySpec,
    field_shape: tuple[int, int, int, int],
    *,
    dtype: np.dtype[np.floating] | np.dtype[np.complexfloating] = np.float64,
) -> PMLBoundaryState:
    """Allocate persistent absorbing-boundary memory arrays for all active faces."""

    if len(field_shape) != 4 or field_shape[-1] != 3:
        raise ValueError("PML state allocation expects field shapes (nx, ny, nz, 3)")
    electric: dict[str, PMLFaceState] = {}
    magnetic: dict[str, PMLFaceState] = {}

    for axis_boundary in compiled_boundary_spec.axes():
        axis_index = "xyz".index(axis_boundary.axis)
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            if edge.mode not in {
                BoundaryMode.PML,
                BoundaryMode.STABLE_PML,
                BoundaryMode.ABSORBER,
            } or edge.pml is None:
                continue
            memory_shape = list(field_shape)
            memory_shape[axis_index] = edge.pml.num_layers
            face_state = PMLFaceState(
                axis=axis_boundary.axis,
                side=side,
                memory_shape=tuple(int(value) for value in memory_shape),
                memory=np.zeros(memory_shape, dtype=dtype),
            )
            face_key = f"{axis_boundary.axis}.{side}"
            electric[face_key] = face_state
            magnetic[face_key] = PMLFaceState(
                axis=axis_boundary.axis,
                side=side,
                memory_shape=face_state.memory_shape,
                memory=np.zeros(memory_shape, dtype=dtype),
            )
    return PMLBoundaryState(electric=electric, magnetic=magnetic)


class BoundaryRuntime:
    """Small runtime wrapper around compiled Phase 1 boundary metadata."""

    def __init__(
        self,
        boundary_spec: object | None = None,
        *,
        symmetry: tuple[int, int, int] = (0, 0, 0),
        dt: float = 1.0,
        grid_spacing: float | tuple[float, float, float] = 1.0,
        field_shape: tuple[int, int, int, int] | None = None,
        field_dtype: np.dtype[np.floating] | np.dtype[np.complexfloating] = np.float64,
    ) -> None:
        self.compiled = compile_boundary_spec(
            boundary_spec,
            symmetry=symmetry,
            dt=dt,
            grid_spacing=grid_spacing,
        )
        self.pml_state = (
            allocate_pml_boundary_state(self.compiled, field_shape, dtype=field_dtype)
            if field_shape is not None and self.compiled.pml_faces
            else None
        )
        self.abc_state = (
            allocate_abc_boundary_state(self.compiled, field_shape, dtype=field_dtype)
            if field_shape is not None and self.compiled.abc_faces
            else None
        )

    def allocate_state(
        self,
        field_shape: tuple[int, int, int, int],
        *,
        dtype: np.dtype[np.floating] | np.dtype[np.complexfloating] = np.float64,
    ) -> PMLBoundaryState | ABCBoundaryState | tuple[PMLBoundaryState | None, ABCBoundaryState | None] | None:
        """Allocate or refresh persistent absorbing-boundary state for a field layout."""

        if self.compiled.pml_faces:
            self.pml_state = allocate_pml_boundary_state(self.compiled, field_shape, dtype=dtype)
        else:
            self.pml_state = None
        if self.compiled.abc_faces:
            self.abc_state = allocate_abc_boundary_state(self.compiled, field_shape, dtype=dtype)
        else:
            self.abc_state = None
        if self.pml_state is not None and self.abc_state is not None:
            return self.pml_state, self.abc_state
        return self.pml_state if self.pml_state is not None else self.abc_state

    def apply(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ghost-cell and PML stage updates for both field families."""

        if (self.compiled.pml_faces and self.pml_state is None) or (
            self.compiled.abc_faces and self.abc_state is None
        ):
            self.allocate_state(electric_field.shape, dtype=electric_field.dtype)
        return apply_boundary_stages(
            electric_field,
            magnetic_field,
            self.compiled,
            self.pml_state,
            abc_state=self.abc_state,
        )

    def transform_boundary_values(
        self,
        values: np.ndarray,
        edge: CompiledBoundaryEdge,
        *,
        field_family: Literal["electric", "magnetic"],
    ) -> np.ndarray:
        """Transform halo values imported from a neighboring chunk or wrapped face."""

        return boundary_edge_transform(values, edge, field_family=field_family)

    def transform_symmetry_values(
        self,
        values: np.ndarray,
        symmetry_axis: CompiledSymmetryAxis,
        *,
        field_family: Literal["electric", "magnetic"],
    ) -> np.ndarray:
        """Transform vector values imported through a mirror-symmetry reduction."""

        return symmetry_transform(values, symmetry_axis, field_family=field_family)


def build_boundary_runtime(
    boundary_spec: object | None = None,
    *,
    symmetry: tuple[int, int, int] = (0, 0, 0),
    dt: float = 1.0,
    grid_spacing: float | tuple[float, float, float] = 1.0,
    field_shape: tuple[int, int, int, int] | None = None,
    field_dtype: np.dtype[np.floating] | np.dtype[np.complexfloating] = np.float64,
) -> BoundaryRuntime:
    """Build the default Phase 1 boundary runtime wrapper."""

    return BoundaryRuntime(
        boundary_spec,
        symmetry=symmetry,
        dt=dt,
        grid_spacing=grid_spacing,
        field_shape=field_shape,
        field_dtype=field_dtype,
    )


def apply_compiled_boundaries(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_boundary_spec: CompiledBoundarySpec,
    *,
    state: PMLBoundaryState | None = None,
    abc_state: ABCBoundaryState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply boundary ghosts and PML stages using a precompiled specification."""

    return apply_boundary_stages(
        electric_field,
        magnetic_field,
        compiled_boundary_spec,
        state,
        abc_state=abc_state,
    )
