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


# ---------------------------------------------------------------------------
# Chunk-aware halo exchange helpers
# ---------------------------------------------------------------------------


def _halo_slice(
    axis: int,
    side: str,
    depth: int,
    total_size: int,
) -> tuple[slice, slice]:
    """Return source and destination slices for halo exchange.

    Returns (dest_slice, src_slice) where:
    - dest_slice selects the ghost cells to fill
    - src_slice selects the interior cells to copy from
    """
    if side == "minus":
        dest = slice(0, depth)
        src = slice(depth, 2 * depth)
    else:
        dest = slice(total_size - depth, total_size)
        src = slice(total_size - 2 * depth, total_size - depth)
    return dest, src


def pack_halo(
    field: np.ndarray,
    axis: Literal["x", "y", "z"],
    side: Literal["minus", "plus"],
    depth: int,
    chunk_size: int,
) -> np.ndarray:
    """Pack halo cells from one face of a chunk's field array.

    Parameters
    ----------
    field : np.ndarray
        The field array with shape (nx, ny, nz, 3) or (nx, ny, nz).
    axis : "x", "y", "z"
        The axis perpendicular to the face.
    side : "minus" or "plus"
        Which side of the chunk to pack from.
    depth : int
        Number of ghost cells to pack.
    chunk_size : int
        Total size along the axis.

    Returns
    -------
    np.ndarray
        The packed halo data with shape appropriate for the transverse face.
    """
    axis_index = "xyz".index(axis)
    if field.ndim == 4:
        # Vector field: (nx, ny, nz, 3)
        # Extract the face at depth cells in
        if side == "minus":
            slc: tuple[object, ...] = (slice(0, depth),)
        else:
            slc = (slice(chunk_size - depth, chunk_size),)

        # Build full slice for all dimensions
        full_slc = list(slc)
        for _ in range(field.ndim - len(slc)):
            full_slc.append(slice(None))

        return field[tuple(full_slc)]
    elif field.ndim == 3:
        # Scalar field: (nx, ny, nz)
        if side == "minus":
            slc: tuple[object, ...] = (slice(0, depth),)
        else:
            slc = (slice(chunk_size - depth, chunk_size),)

        full_slc = list(slc)
        for _ in range(field.ndim - len(slc)):
            full_slc.append(slice(None))

        return field[tuple(full_slc)]
    else:
        raise ValueError(f"Expected 3D or 4D field, got {field.ndim}D")


def unpack_halo(
    packed: np.ndarray,
    field: np.ndarray,
    axis: Literal["x", "y", "z"],
    side: Literal["minus", "plus"],
    depth: int,
    chunk_size: int,
) -> np.ndarray:
    """Unpack halo cells into one face of a chunk's field array.

    Parameters
    ----------
    packed : np.ndarray
        The halo data to unpack.
    field : np.ndarray
        The field array to write into with shape (nx, ny, nz, 3) or (nx, ny, nz).
    axis : "x", "y", "z"
        The axis perpendicular to the face.
    side : "minus" or "plus"
        Which side of the chunk to unpack into.
    depth : int
        Number of ghost cells to unpack.
    chunk_size : int
        Total size along the axis.

    Returns
    -------
    np.ndarray
        The field array with halo cells updated.
    """
    axis_index = "xyz".index(axis)
    if side == "minus":
        dest_slice: tuple[object, ...] = (slice(0, depth),)
    else:
        dest_slice = (slice(chunk_size - depth, chunk_size),)

    full_dest = list(dest_slice)
    for _ in range(field.ndim - len(dest_slice)):
        full_dest.append(slice(None))

    result = field.copy()
    result[tuple(full_dest)] = packed
    return result


def apply_signs_to_halo(
    halo_data: np.ndarray,
    signs: tuple[int, int, int],
    field_family: Literal["electric", "magnetic"],
) -> np.ndarray:
    """Apply reflection signs to halo data.

    Parameters
    ----------
    halo_data : np.ndarray
        The halo data to transform.
    signs : (s0, s1, s2)
        Sign for each vector component.
    field_family : "electric" or "magnetic"
        Which field family this is for (affects how signs are applied to scalar vs vector).

    Returns
    -------
    np.ndarray
        The transformed halo data.
    """
    if halo_data.ndim == 4:
        # Vector field with shape (..., 3)
        result = halo_data.copy()
        for comp, sign in enumerate(signs):
            result[..., comp] *= sign
        return result
    else:
        return halo_data.copy()


def apply_phase_to_halo(
    halo_data: np.ndarray,
    phase_factor: complex,
) -> np.ndarray:
    """Apply Bloch phase factor to halo data.

    Parameters
    ----------
    halo_data : np.ndarray
        The halo data to transform.
    phase_factor : complex
        The phase factor e^{i k · d}.

    Returns
    -------
    np.ndarray
        The phase-rotated halo data.
    """
    if np.iscomplexobj(halo_data):
        return halo_data * phase_factor
    # Real data with complex phase - need to promote to complex
    return halo_data.astype(np.complex128) * phase_factor


class ChunkHaloExchange:
    """Manages halo exchange operations for a chunk layout.

    This class provides the pack, unpack, and orchestration helpers
    for stage-specific halo exchange in a chunked simulation.

    Parameters
    ----------
    chunk_layout : ChunkLayout
        The chunk decomposition plan.
    field_shape : tuple[int, int, int, int]
        Shape of each chunk's field array (nx, ny, nz, 3).
    dtype : np.dtype
        Field array dtype.
    """

    def __init__(
        self,
        chunk_layout: "ChunkLayout",
        field_shape: tuple[int, int, int, int],
        dtype: np.dtype = np.float64,
    ) -> None:
        self.chunk_layout = chunk_layout
        self.field_shape = field_shape
        self.dtype = dtype

    def exchange_kind_for_face(
        self,
        chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> "ExchangeKind":
        """Return the exchange kind for a chunk face."""
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            from autofdtd.runtime.chunk import ExchangeKind

            return ExchangeKind.INTERIOR
        halo = chunk.face_halo(axis, side)
        if halo is None:
            from autofdtd.runtime.chunk import ExchangeKind

            return ExchangeKind.INTERIOR
        return halo.exchange_kind

    def pack_face_halo(
        self,
        chunk_index: tuple[int, int, int],
        field: np.ndarray,
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> np.ndarray | None:
        """Pack halo data from a chunk face.

        Returns None if the face is an interior exchange (no packing needed).
        """
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return None
        halo = chunk.face_halo(axis, side)
        if halo is None:
            return None

        exchange_kind = halo.exchange_kind
        from autofdtd.runtime.chunk import ExchangeKind

        if exchange_kind in {ExchangeKind.INTERIOR, ExchangeKind.PML, ExchangeKind.STABLE_PML, ExchangeKind.ABSORBER}:
            # Interior faces don't need halo packing; PML/ABC are applied in-place
            return None

        axis_size = self.field_shape["xyz".index(axis)]
        packed = pack_halo(field, axis, side, halo.depth, axis_size)

        # Apply phase factor for Bloch
        if exchange_kind == ExchangeKind.BLOCH:
            packed = apply_phase_to_halo(packed, halo.phase_factor)

        # Apply signs for PEC/PMC/symmetry
        if exchange_kind in {ExchangeKind.PEC, ExchangeKind.PMC, ExchangeKind.SYMMETRY_COPY}:
            if side == "minus":
                signs = halo.electric_signs
            else:
                signs = halo.electric_signs
            packed = apply_signs_to_halo(packed, signs, "electric")

        return packed

    def unpack_face_halo(
        self,
        chunk_index: tuple[int, int, int],
        field: np.ndarray,
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
        packed: np.ndarray | None,
    ) -> np.ndarray:
        """Unpack halo data into a chunk face.

        No-ops if the face is interior or if packed is None.
        """
        if packed is None:
            return field

        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return field
        halo = chunk.face_halo(axis, side)
        if halo is None:
            return field

        axis_size = self.field_shape["xyz".index(axis)]
        return unpack_halo(packed, field, axis, side, halo.depth, axis_size)

    def interior_copy_slice(
        self,
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
        depth: int,
    ) -> tuple[slice, slice]:
        """Return slices for interior face-to-face copy (no neighbor needed).

        For interior chunks sharing a face, we copy from one chunk's interior
        face to another chunk's interior face.
        """
        return _halo_slice("xyz".index(axis), side, depth, self.field_shape["xyz".index(axis)])


def build_chunk_halo_exchange(
    chunk_layout: "ChunkLayout",
    field_shape: tuple[int, int, int, int],
    dtype: np.dtype = np.float64,
) -> ChunkHaloExchange:
    """Build a chunk-aware halo exchange manager."""
    return ChunkHaloExchange(chunk_layout, field_shape, dtype)

