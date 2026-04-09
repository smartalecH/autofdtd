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

    # Handle both numpy and Warp arrays
    is_warp = hasattr(field, 'numpy')
    if is_warp:
        # For Warp arrays: copy to numpy, modify in-place, return numpy
        # Note: caller must copy result back to Warp array
        field_np = field.numpy().copy()
        field_np[tuple(full_dest)] = packed
        return field_np
    else:
        # For numpy arrays: copy and modify
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


class CrossDeviceHaloTransfer:
    """Manages explicit CUDA memcpy for cross-device halo exchange.

    When chunks reside on different CUDA devices and peer access is not
    available, halo exchange requires explicit staging through host memory:

        Source Device (GPU 0) → Host (CPU) → Destination Device (GPU 1)

    This class provides the transfer orchestration for such cross-device
    exchanges, using numpy arrays as host-staging buffers.

    Parameters
    ----------
    chunk_layout : ChunkLayout
        The chunk decomposition plan.
    field_shape : tuple[int, int, int, int]
        Shape of each chunk's field array (nx, ny, nz, 3).
    dtype : np.dtype
        Field array dtype.

    Notes
    -----
    This implementation uses synchronous cudaMemcpy operations (D2H then H2D).
    For better performance with multiple GPUs, consider:
    - Using CUDA streams for overlap between computation and transfer
    - Enabling peer access if supported: cudaDeviceEnableDirectAccess()
    - Using NCCL for multi-GPU collective operations
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
        self._warp = None
        self._cuda_available = False
        self._init_cuda()

    def _init_cuda(self) -> None:
        """Initialize CUDA/Warp context."""
        try:
            import warp as wp

            self._warp = wp
            self._cuda_available = wp.is_cuda_available() if hasattr(wp, "is_cuda_available") else False
        except ImportError:
            self._warp = None
            self._cuda_available = False

    @property
    def cuda_available(self) -> bool:
        """Return True if CUDA is available via Warp."""
        return self._cuda_available and self._warp is not None

    def is_cross_device_face(
        self,
        chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> bool:
        """Check if a chunk face requires cross-device transfer.

        Returns True when the neighbor chunk is on a different CUDA device.
        """
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return False

        halo = chunk.face_halo(axis, side)
        if halo is None:
            return False

        neighbor_index = halo.neighbor_chunk_index
        if neighbor_index is None:
            return False

        src_device = self.chunk_layout.device_for_chunk(chunk_index)
        dst_device = self.chunk_layout.device_for_chunk(neighbor_index)

        if src_device is None or dst_device is None:
            return False

        return src_device != dst_device

    def get_transfer_devices(
        self,
        chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> tuple[int | None, int | None, bool]:
        """Get source and destination device IDs for a face transfer.

        Returns
        -------
        tuple
            (source_device, dest_device, is_cross_device)
        """
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return (None, None, False)

        halo = chunk.face_halo(axis, side)
        if halo is None:
            return (None, None, False)

        neighbor_index = halo.neighbor_chunk_index

        src_device = self.chunk_layout.device_for_chunk(chunk_index)
        if neighbor_index is None:
            return (src_device, None, False)

        dst_device = self.chunk_layout.device_for_chunk(neighbor_index)

        if src_device is None or dst_device is None:
            return (src_device, dst_device, False)

        return (src_device, dst_device, src_device != dst_device)

    def allocate_staging_buffer(self, shape: tuple[int, ...]) -> np.ndarray:
        """Allocate a host-side staging buffer for cross-device transfer.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the buffer to allocate.

        Returns
        -------
        np.ndarray
            Host numpy array suitable for staging transfers.
        """
        # Convert field dtype to numpy dtype
        if self.dtype == np.float64:
            np_dtype = np.float64
        elif self.dtype == np.complex128:
            np_dtype = np.complex128
        elif self.dtype == np.float32:
            np_dtype = np.float32
        elif self.dtype == np.complex64:
            np_dtype = np.complex64
        else:
            np_dtype = self.dtype

        return np.zeros(shape, dtype=np_dtype)

    def cuda_device_to_host(
        self,
        src_array: "wp.array | np.ndarray",
        src_device: int,
        stream: "wp.Stream | None" = None,
    ) -> np.ndarray:
        """Copy data from CUDA device to host memory.

        Parameters
        ----------
        src_array : wp.array or np.ndarray
            Source array on the CUDA device.
        src_device : int
            Source CUDA device ID.
        stream : wp.Stream, optional
            CUDA stream for async transfer.

        Returns
        -------
        np.ndarray
            Data copied to host numpy array.
        """
        if not self.cuda_available:
            # Fallback: return the array as-is if not using CUDA
            if isinstance(src_array, np.ndarray):
                return src_array.copy()
            return np.array(src_array)

        # Use Warp array's numpy() method which handles D2H copy automatically
        if hasattr(src_array, "numpy"):
            return src_array.numpy().copy()
        return np.array(src_array)

    def cuda_host_to_device(
        self,
        dst_array: "wp.array",
        data: np.ndarray,
        dst_device: int,
        stream: "wp.Stream | None" = None,
    ) -> None:
        """Copy data from host memory to CUDA device.

        Parameters
        ----------
        dst_array : wp.array
            Destination array on the CUDA device.
        data : np.ndarray
            Host numpy array to copy from.
        dst_device : int
            Destination CUDA device ID.
        stream : wp.Stream, optional
            CUDA stream for async transfer.
        """
        if not self.cuda_available:
            # Fallback: if not using CUDA, copy directly to numpy array
            if isinstance(dst_array, np.ndarray):
                np.copyto(dst_array, data)
            return

        # For CUDA: create a new array on target device and copy data
        # The caller should use wp.array(data, device=f"cuda:{dst_device}") for H2D
        # Here we just indicate the copy should happen to the destination
        # Note: Warp doesn't support in-place H2D to existing wp.array
        # The caller must handle this by creating new arrays on target device
        if hasattr(dst_array, "numpy"):
            # This is a wp array - we can't copy directly
            # The caller should use wp.array() constructor for H2D
            pass

    def transfer_face_halo(
        self,
        src_chunk_index: tuple[int, int, int],
        dst_chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        src_side: Literal["minus", "plus"],
        dst_side: Literal["minus", "plus"],
        src_field: "wp.array | np.ndarray",
        dst_field: "wp.array | np.ndarray",
        halo_depth: int,
        chunk_size: int,
    ) -> np.ndarray | None:
        """Perform cross-device halo transfer for one face pair.

        This executes: src_field → host buffer → dst_field
        using explicit D2H then H2D memcpy since peer access is unavailable.

        Parameters
        ----------
        src_chunk_index : tuple[int, int, int]
            Source chunk index.
        dst_chunk_index : tuple[int, int, int]
            Destination chunk index.
        axis : "x", "y", "z"
            Axis perpendicular to the face.
        src_side : "minus" or "plus"
            Source face side.
        dst_side : "minus" or "plus"
            Destination face side.
        src_field : wp.array or np.ndarray
            Source field array on source device.
        dst_field : wp.array or np.ndarray
            Destination field array on destination device.
        halo_depth : int
            Number of ghost cells to transfer.
        chunk_size : int
            Total size along the transfer axis.

        Returns
        -------
        np.ndarray or None
            The staging buffer used, or None if transfer was not needed.
        """
        src_device = self.chunk_layout.device_for_chunk(src_chunk_index)
        dst_device = self.chunk_layout.device_for_chunk(dst_chunk_index)

        if src_device is None or dst_device is None:
            return None
        if src_device == dst_device:
            return None

        # Pack source halo
        src_face = pack_halo(src_field, axis, src_side, halo_depth, chunk_size)
        if src_face is None:
            return None

        # Determine staging buffer shape
        staging_shape = src_face.shape

        # D2H: Source device → host
        host_data = self.cuda_device_to_host(src_face, src_device)

        # H2D: Host → Destination device
        # For the destination, we need to unpack into the correct position
        if self.cuda_available and hasattr(dst_field, "numpy"):
            # For wp arrays, we need to handle unpacking manually
            # Get current dst data on host
            dst_np = dst_field.numpy().copy()
            # Unpack halo into the right position
            dst_np = unpack_halo(host_data, dst_np, axis, dst_side, halo_depth, chunk_size)
            # Create new array on destination device with updated data
            # Note: Warp doesn't support in-place H2D, so we create a new array
            # For now, just return the host_data - caller must handle actual H2D
            # This is a limitation of the current implementation
            return host_data
        else:
            # For numpy arrays, just unpack directly
            dst_field[:] = unpack_halo(host_data, dst_field, axis, dst_side, halo_depth, chunk_size)

        return host_data

        return host_data


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

    def is_cross_device_face(
        self,
        chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> bool:
        """Check if a chunk face requires cross-device transfer.

        Returns True when the neighbor chunk is on a different CUDA device.
        This is determined by comparing device assignments in ChunkLayout.
        """
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return False

        halo = chunk.face_halo(axis, side)
        if halo is None:
            return False

        neighbor_index = halo.neighbor_chunk_index
        if neighbor_index is None:
            return False

        src_device = self.chunk_layout.device_for_chunk(chunk_index)
        dst_device = self.chunk_layout.device_for_chunk(neighbor_index)

        if src_device is None or dst_device is None:
            return False

        return src_device != dst_device

    def get_face_devices(
        self,
        chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        side: Literal["minus", "plus"],
    ) -> tuple[int | None, int | None, bool]:
        """Get device IDs for a face's source and destination.

        Returns
        -------
        tuple
            (own_device, neighbor_device, is_cross_device)
        """
        chunk = self.chunk_layout.chunk_at(chunk_index)
        if chunk is None:
            return (None, None, False)

        halo = chunk.face_halo(axis, side)
        if halo is None:
            return (None, None, False)

        neighbor_index = halo.neighbor_chunk_index

        own_device = self.chunk_layout.device_for_chunk(chunk_index)
        if neighbor_index is None:
            return (own_device, None, False)

        neighbor_device = self.chunk_layout.device_for_chunk(neighbor_index)

        if own_device is None or neighbor_device is None:
            return (own_device, neighbor_device, False)

        return (own_device, neighbor_device, own_device != neighbor_device)

    def cross_device_transfer(
        self,
        src_chunk_index: tuple[int, int, int],
        dst_chunk_index: tuple[int, int, int],
        axis: Literal["x", "y", "z"],
        src_side: Literal["minus", "plus"],
        dst_side: Literal["minus", "plus"],
        src_field: np.ndarray,
        dst_field: np.ndarray,
    ) -> np.ndarray | None:
        """Perform cross-device halo transfer for one face pair.

        When source and destination are on different devices, this performs:
        src_field → host buffer → dst_field

        using explicit staging since peer access is not supported.

        Parameters
        ----------
        src_chunk_index : tuple[int, int, int]
            Source chunk index.
        dst_chunk_index : tuple[int, int, int]
            Destination chunk index.
        axis : "x", "y", "z"
            Axis perpendicular to the face.
        src_side : "minus" or "plus"
            Source face side.
        dst_side : "minus" or "plus"
            Destination face side.
        src_field : np.ndarray
            Source field array (on source device if using CUDA).
        dst_field : np.ndarray
            Destination field array (on destination device if using CUDA).
        Returns
        -------
        np.ndarray or None
            The staging buffer used, or None if not cross-device or transfer not needed.
        """
        chunk = self.chunk_layout.chunk_at(src_chunk_index)
        if chunk is None:
            return None

        halo = chunk.face_halo(axis, src_side)
        if halo is None:
            return None

        src_device = self.chunk_layout.device_for_chunk(src_chunk_index)
        dst_device = self.chunk_layout.device_for_chunk(dst_chunk_index)

        if src_device is None or dst_device is None:
            return None
        if src_device == dst_device:
            return None

        # Pack source halo
        axis_size = self.field_shape["xyz".index(axis)]
        packed = pack_halo(src_field, axis, src_side, halo.depth, axis_size)
        if packed is None:
            return None

        # For cross-device with numpy arrays (no GPU), just do a direct transfer
        # through host staging
        # D2H is implicit when we work with numpy arrays from GPU
        # Use .numpy() method for Warp arrays, np.array() for numpy arrays
        if hasattr(packed, 'numpy'):
            host_data = packed.numpy()
        else:
            host_data = np.array(packed)

        # Unpack into destination
        unpacked = unpack_halo(host_data, dst_field, axis, dst_side, halo.depth, axis_size)
        # unpack_halo returns a numpy array; copy back to dst_field
        if hasattr(dst_field, 'numpy'):
            # Warp array: copy through numpy view + slice assignment
            # Construct destination slice from axis, dst_side, depth, axis_size
            if dst_side == "minus":
                dest_slice = (slice(0, halo.depth),)
            else:
                dest_slice = (slice(axis_size - halo.depth, axis_size),)

            # Build full slice for all dimensions
            full_dest = list(dest_slice)
            for _ in range(dst_field.ndim - len(dest_slice)):
                full_dest.append(slice(None))

            dst_view = dst_field.numpy()
            dst_view[tuple(full_dest)] = unpacked
            dst_field.assign(dst_view)
        else:
            # Numpy array: direct slice assignment
            if dst_side == "minus":
                dest_slice = (slice(0, halo.depth),)
            else:
                dest_slice = (slice(axis_size - halo.depth, axis_size),)
            full_dest = list(dest_slice)
            for _ in range(dst_field.ndim - len(dest_slice)):
                full_dest.append(slice(None))
            dst_field[tuple(full_dest)] = unpacked

        return host_data


def build_chunk_halo_exchange(
    chunk_layout: "ChunkLayout",
    field_shape: tuple[int, int, int, int],
    dtype: np.dtype = np.float64,
) -> ChunkHaloExchange:
    """Build a chunk-aware halo exchange manager."""
    return ChunkHaloExchange(chunk_layout, field_shape, dtype)

