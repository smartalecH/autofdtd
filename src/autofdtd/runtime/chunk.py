"""Chunk data model and multi-chunk decomposition runtime.

This module provides the chunk-level planning metadata and device/rank
placement contracts defined in the Phase 1 chunk architecture.

Design Principles
----------------
1. **Chunk is the unit of parallelism.** Rank or device is placement,
   not computation.
2. **Halo depth is per-face.** Different boundary families require
   different halo widths and exchange semantics.
3. **Chunk metadata is planning-only.** Actual field buffers and PML
   state are allocated at runtime using ChunkSpec.local_grid_shape.
4. **Bulk-synchronous by default.** Phase 1 uses bulk-synchronous
   stepping; chunk metadata preserves the hooks for later systolic
   or overlapped execution.

References
---------
- Chunk contract: ``../../phase1/architecture/chunk-contract.md``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from autofdtd.core.models import AutoFDTDModel


class ExchangeKind(StrEnum):
    """Kinds of halo exchange for one face."""

    PERIODIC = "periodic"
    BLOCH = "bloch"
    PEC = "pec"
    PMC = "pmc"
    PML = "pml"
    STABLE_PML = "stable_pml"
    ABSORBER = "absorber"
    ABC = "abc"
    SYMMETRY_COPY = "symmetry_copy"
    INTERIOR = "interior"


@dataclass(frozen=True)
class FaceHalo:
    """Halo metadata for one face of one chunk.

    Attributes
    ----------
    axis : str
        "x", "y", or "z".
    side : str
        "minus" or "plus".
    depth : int
        Number of ghost cells on this face.
    neighbor_chunk_index : tuple[int, int, int] | None
        None = domain boundary, otherwise index of neighbor chunk.
    exchange_kind : ExchangeKind
        The type of exchange to perform.
    phase_factor : complex
        For Bloch: e^{i k · d}. Real fields use 1.0+0j.
    electric_signs : tuple[int, int, int]
        Sign transform for electric field reflection.
    magnetic_signs : tuple[int, int, int]
        Sign transform for magnetic field reflection.
    """

    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    depth: int = 1
    neighbor_chunk_index: tuple[int, int, int] | None = None
    exchange_kind: ExchangeKind = ExchangeKind.INTERIOR
    phase_factor: complex = 1.0 + 0.0j
    electric_signs: tuple[int, int, int] = (1, 1, 1)
    magnetic_signs: tuple[int, int, int] = (1, 1, 1)

    @property
    def face_key(self) -> str:
        """Canonical string key for this face, e.g. 'x.minus'."""
        return f"{self.axis}.{self.side}"


@dataclass(frozen=True)
class ChunkSpec:
    """Planning metadata for one chunk — produced during scene compilation.

    Attributes
    ----------
    chunk_index : tuple[int, int, int]
        Logical position in the chunk grid.
    global_bounds : tuple[tuple[int, int, int], tuple[int, int, int]]
        Full-domain index bounds of this chunk as ((i0, j0, k0), (i1, j1, k1)).
    interior_bounds : tuple[tuple[int, int, int], tuple[int, int, int]]
        Interior (non-halo) cell index range.
    owned_bounds : tuple[tuple[int, int, int], tuple[int, int, int]]
        Cells this chunk updates (including boundary).
    local_grid_shape : tuple[int, int, int]
        Full local array shape including ghost cells.
    face_halos : dict[str, FaceHalo]
        Per-face halo metadata keyed by face key (e.g. "x.minus").
    is_reduced : bool
        True if chunk is in a symmetry-reduced region.
    symmetry_multiplicity : int
        How many full-domain points this chunk represents.
    source_indices : tuple[int, ...]
        Indices into compiled_sources owned by this chunk.
    monitor_indices : tuple[int, ...]
        Indices into compiled_monitors owned by this chunk.
    """

    chunk_index: tuple[int, int, int]
    global_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    interior_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    owned_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    local_grid_shape: tuple[int, int, int]
    face_halos: dict[str, FaceHalo] = field(default_factory=dict)
    is_reduced: bool = False
    symmetry_multiplicity: int = 1
    source_indices: tuple[int, ...] = ()
    monitor_indices: tuple[int, ...] = ()

    @property
    def num_cells(self) -> int:
        """Total number of cells in the full local array."""
        nx, ny, nz = self.local_grid_shape
        return nx * ny * nz

    @property
    def interior_num_cells(self) -> int:
        """Number of interior (non-ghost) cells."""
        (i0, j0, k0), (i1, j1, k1) = self.interior_bounds
        return (i1 - i0) * (j1 - j0) * (k1 - k0)

    def face_halo(self, axis: str, side: str) -> FaceHalo | None:
        """Get halo metadata for a specific face."""
        return self.face_halos.get(f"{axis}.{side}")


@dataclass(frozen=True)
class ExchangeDescriptor:
    """Descriptor for one halo exchange operation between two chunks."""

    source_chunk_index: tuple[int, int, int]
    dest_chunk_index: tuple[int, int, int]
    axis: Literal["x", "y", "z"]
    source_side: Literal["minus", "plus"]
    dest_side: Literal["minus", "plus"]
    exchange_kind: ExchangeKind
    phase_factor: complex = 1.0 + 0.0j


@dataclass(frozen=True)
class ChunkLayout:
    """Global chunk decomposition plan — immutable after compilation.

    Attributes
    ----------
    num_chunks : tuple[int, int, int]
        Number of chunks per axis (nx_chunks, ny_chunks, nz_chunks).
    total_chunks : int
        Product of num_chunks.
    chunks : tuple[ChunkSpec, ...]
        One ChunkSpec per logical chunk.
    periodic_axes : tuple[str, ...]
        Axes with periodic or Bloch boundary.
    bloch_axes : tuple[str, ...]
        Axes with Bloch boundary.
    exchange_plan : dict[str, ExchangeDescriptor]
        Exchange operations keyed by ("chunk_a", "chunk_b", "face").
    device_assignment : tuple[int, ...]
        chunk_index → device_id mapping (for CUDA). Length = total_chunks.
    rank_assignment : tuple[int, ...]
        chunk_index → MPI rank mapping. Length = total_chunks.
    """

    num_chunks: tuple[int, int, int]
    total_chunks: int
    chunks: tuple[ChunkSpec, ...]
    periodic_axes: tuple[str, ...] = ()
    bloch_axes: tuple[str, ...] = ()
    exchange_plan: dict[str, ExchangeDescriptor] = field(default_factory=dict)
    device_assignment: tuple[int, ...] = ()
    rank_assignment: tuple[int, ...] = ()

    def chunk_at(self, chunk_index: tuple[int, int, int]) -> ChunkSpec | None:
        """Return the ChunkSpec for a given logical chunk index."""
        idx = self._flatten_index(chunk_index)
        if idx is not None and 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return None

    def device_for_chunk(self, chunk_index: tuple[int, int, int]) -> int | None:
        """Return the CUDA device ID for a chunk, or None if not assigned."""
        idx = self._flatten_index(chunk_index)
        if idx is not None and idx < len(self.device_assignment):
            return self.device_assignment[idx]
        return None

    def rank_for_chunk(self, chunk_index: tuple[int, int, int]) -> int | None:
        """Return the MPI rank for a chunk, or None if not assigned."""
        idx = self._flatten_index(chunk_index)
        if idx is not None and idx < len(self.rank_assignment):
            return self.rank_assignment[idx]
        return None

    def _flatten_index(self, chunk_index: tuple[int, int, int]) -> int | None:
        """Convert 3D chunk index to flat array index."""
        i, j, k = chunk_index
        nx, ny, nz = self.num_chunks
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return None
        return i * ny * nz + j * nz + k


# ---------------------------------------------------------------------------
# Chunk-aware boundary compilation
# ---------------------------------------------------------------------------


def build_chunk_layout(
    num_chunks: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
    cell_sizes: tuple[float, float, float],
    boundary_spec: "CompiledBoundarySpec | None" = None,
    symmetry: tuple[int, int, int] = (0, 0, 0),
) -> ChunkLayout:
    """Build a ChunkLayout from grid shape and boundary spec.

    For single-chunk layouts (num_chunks == (1, 1, 1)), this produces a monolithic
    layout covering the full domain. For multi-chunk layouts, this creates a
    decomposition where chunk boundaries align with PML layer edges — edge chunks
    absorb their adjacent PML layers into owned_bounds, while interior boundaries
    between chunks use halo_depth=1.

    Parameters
    ----------
    num_chunks : tuple[int, int, int]
        Desired chunks per axis (nx_chunks, ny_chunks, nz_chunks).
    grid_shape : tuple[int, int, int]
        Full grid dimensions (nx, ny, nz).
    cell_sizes : tuple[float, float, float]
        Cell sizes (dx, dy, dz).
    boundary_spec : CompiledBoundarySpec, optional
        Compiled boundary specification.
    symmetry : tuple[int, int, int], optional
        Symmetry flags per axis.

    Returns
    -------
    ChunkLayout
        Chunk layout with multi-chunk decomposition.
    """
    from autofdtd.compiler.boundaries import compile_boundary_spec

    if boundary_spec is None:
        boundary_spec = compile_boundary_spec(None, symmetry=symmetry)

    nx, ny, nz = grid_shape
    ncx, ncy, ncz = num_chunks
    total_chunks = ncx * ncy * ncz

    # Determine PML depths per axis from boundary spec
    pml_depths = _get_pml_depths(boundary_spec)

    # For multi-chunk, find the best axis to split
    # Prefer: PML axis with enough interior cells, then non-periodic axis, then largest
    split_axis_info = _find_best_split_axis(
        boundary_spec, grid_shape, num_chunks, pml_depths
    )

    # Single-chunk fallback or multi-chunk path
    if total_chunks == 1:
        return _build_monolithic_chunk_layout(
            num_chunks, grid_shape, boundary_spec, pml_depths
        )

    return _build_multi_chunk_layout(
        num_chunks, grid_shape, boundary_spec, pml_depths, split_axis_info
    )


def _get_pml_depths(
    boundary_spec: "CompiledBoundarySpec",
) -> dict[str, tuple[int, int]]:
    """Extract PML depths for each axis from compiled boundary spec.

    Returns
    -------
    dict mapping axis name to (minus_pml_depth, plus_pml_depth)
    """
    depths: dict[str, tuple[int, int]] = {}
    for axis_name, axis_boundary in zip("xyz", boundary_spec.axes(), strict=True):
        minus_depth = 0
        plus_depth = 0

        minus_edge = axis_boundary.minus
        plus_edge = axis_boundary.plus

        if minus_edge.mode in {
            "pml",
            "stable_pml",
            "absorber",
        } and minus_edge.pml is not None:
            minus_depth = minus_edge.pml.num_layers

        if plus_edge.mode in {
            "pml",
            "stable_pml",
            "absorber",
        } and plus_edge.pml is not None:
            plus_depth = plus_edge.pml.num_layers

        depths[axis_name] = (minus_depth, plus_depth)

    return depths


def _find_best_split_axis(
    boundary_spec: "CompiledBoundarySpec",
    grid_shape: tuple[int, int, int],
    num_chunks: tuple[int, int, int],
    pml_depths: dict[str, tuple[int, int]],
) -> tuple[str, int, int, int] | None:
    """Find the best axis to split for multi-chunk decomposition.

    Returns
    -------
    (axis_name, axis_index, pml_minus, pml_plus) or None if no split possible
    """
    nx, ny, nz = grid_shape

    candidates: list[tuple[str, int, int, int, int]] = []

    axis_info = [
        ("x", 0, nx, num_chunks[0]),
        ("y", 1, ny, num_chunks[1]),
        ("z", 2, nz, num_chunks[2]),
    ]

    for axis_name, axis_idx, axis_size, n_chunks_axis in axis_info:
        if n_chunks_axis <= 1:
            continue

        pml_minus, pml_plus = pml_depths[axis_name]
        interior_size = axis_size - pml_minus - pml_plus

        # Check axis boundary modes
        axis_boundary = boundary_spec.axes()[axis_idx]
        minus_mode = axis_boundary.minus.mode
        plus_mode = axis_boundary.plus.mode

        # Compute cells per chunk (interior only, for PML-aligned split)
        cells_per_chunk = interior_size // n_chunks_axis

        priority = 0
        # Prefer axis with PML on at least one side
        if pml_minus > 0 or pml_plus > 0:
            priority = 2
        # Prefer non-periodic/bloch axes
        elif minus_mode not in {"periodic", "bloch"} and plus_mode not in {"periodic", "bloch"}:
            priority = 1

        candidates.append((axis_name, axis_idx, pml_minus, pml_plus, priority))

    if not candidates:
        return None

    # Sort by priority (higher is better), then by axis index (lower is better)
    candidates.sort(key=lambda x: (x[4], -x[1]))
    best = candidates[-1]  # Highest priority (last after ascending sort)

    return (best[0], best[1], best[2], best[3])


def _build_monolithic_chunk_layout(
    num_chunks: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
    boundary_spec: "CompiledBoundarySpec",
    pml_depths: dict[str, tuple[int, int]],
) -> ChunkLayout:
    """Build a single-chunk layout (monolithic domain)."""
    nx, ny, nz = grid_shape

    # Interior excludes only the outermost ghost cells (index 0 and nx-1)
    interior_lo = (1, 1, 1)
    interior_hi = (nx - 1, ny - 1, nz - 1)
    owned_lo = (1, 1, 1)
    owned_hi = (nx - 1, ny - 1, nz - 1)

    face_halos: dict[str, FaceHalo] = {}

    for axis_name, axis_boundary in zip("xyz", boundary_spec.axes(), strict=True):
        axis_index = "xyz".index(axis_name)
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            face_key = f"{axis_name}.{side}"

            exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_index)
            depth = _halo_depth_for_edge(edge, axis_index)

            face_halos[face_key] = FaceHalo(
                axis=axis_name,
                side=side,
                depth=depth,
                neighbor_chunk_index=None,
                exchange_kind=exchange_kind,
                phase_factor=edge.phase_factor,
                electric_signs=edge.electric_signs,
                magnetic_signs=edge.magnetic_signs,
            )

    chunk_spec = ChunkSpec(
        chunk_index=(0, 0, 0),
        global_bounds=((0, 0, 0), (nx, ny, nz)),
        interior_bounds=(interior_lo, interior_hi),
        owned_bounds=(owned_lo, owned_hi),
        local_grid_shape=(nx, ny, nz),
        face_halos=face_halos,
        is_reduced=False,
        symmetry_multiplicity=1,
        source_indices=(),
        monitor_indices=(),
    )

    total = num_chunks[0] * num_chunks[1] * num_chunks[2]
    device_assignment = tuple(0 for _ in range(total))
    rank_assignment = tuple(0 for _ in range(total))

    return ChunkLayout(
        num_chunks=num_chunks,
        total_chunks=total,
        chunks=(chunk_spec,),
        periodic_axes=boundary_spec.periodic_axes,
        bloch_axes=boundary_spec.bloch_axes,
        exchange_plan={},
        device_assignment=device_assignment,
        rank_assignment=rank_assignment,
    )


def _build_multi_chunk_layout(
    num_chunks: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
    boundary_spec: "CompiledBoundarySpec",
    pml_depths: dict[str, tuple[int, int]],
    split_axis_info: tuple[str, int, int, int] | None,
) -> ChunkLayout:
    """Build a multi-chunk layout with PML-aligned boundaries."""
    nx, ny, nz = grid_shape
    ncx, ncy, ncz = num_chunks
    total_chunks = ncx * ncy * ncz

    # Determine split axis
    if split_axis_info is None:
        # Fallback: split along x if possible
        if ncx > 1:
            split_axis_name = "x"
            split_axis_idx = 0
            pml_minus, pml_plus = pml_depths["x"]
        elif ncy > 1:
            split_axis_name = "y"
            split_axis_idx = 1
            pml_minus, pml_plus = pml_depths["y"]
        else:
            split_axis_name = "z"
            split_axis_idx = 2
            pml_minus, pml_plus = pml_depths["z"]
    else:
        split_axis_name, split_axis_idx, pml_minus, pml_plus = split_axis_info

    axis_size = grid_shape[split_axis_idx]
    n_split = num_chunks[split_axis_idx]

    # Compute interior range (excluding PML layers)
    interior_start = pml_minus
    interior_end = axis_size - pml_plus
    interior_size = interior_end - interior_start

    # For PML-aligned split: divide interior cells among chunks
    # Each edge chunk absorbs its adjacent PML layer into owned_bounds
    cells_per_chunk = interior_size // n_split

    # Build all chunks
    chunks: list[ChunkSpec] = []

    # Iterate through all chunk indices
    for k in range(ncz):
        for j in range(ncy):
            for i in range(ncx):
                chunk_index = (i, j, k)

                # Compute bounds for each axis
                axis_bounds = {}
                axis_chunk_positions = [i, j, k]

                for ax_idx, ax_size, ax_nc in [(0, nx, ncx), (1, ny, ncy), (2, nz, ncz)]:
                    chunk_pos = axis_chunk_positions[ax_idx]

                    if ax_nc == 1:
                        # Single chunk on this axis: full grid
                        global_lo_ax = 0
                        global_hi_ax = ax_size
                    else:
                        # Divide grid among chunks on this axis
                        cells_per = ax_size // ax_nc
                        global_lo_ax = chunk_pos * cells_per
                        global_hi_ax = (chunk_pos + 1) * cells_per

                    axis_bounds[ax_idx] = (global_lo_ax, global_hi_ax)

                # For split axis, incorporate PML absorption into edge chunks
                chunk_pos = axis_chunk_positions[split_axis_idx]

                if chunk_pos == 0:
                    # First chunk along split axis: absorbs minus PML
                    split_owned_lo = 0
                    split_owned_hi = pml_minus + cells_per_chunk
                    split_interior_lo = pml_minus
                    split_interior_hi = split_owned_hi
                elif chunk_pos == n_split - 1:
                    # Last chunk along split axis: absorbs plus PML
                    split_owned_lo = interior_start + chunk_pos * cells_per_chunk
                    split_owned_hi = axis_size
                    split_interior_lo = split_owned_lo
                    split_interior_hi = interior_end
                else:
                    # Interior chunk: no PML absorption
                    split_owned_lo = interior_start + chunk_pos * cells_per_chunk
                    split_owned_hi = split_owned_lo + cells_per_chunk
                    split_interior_lo = split_owned_lo
                    split_interior_hi = split_owned_hi

                # Set split axis bounds
                axis_bounds[split_axis_idx] = (split_owned_lo, split_owned_hi)

                # Build global bounds
                global_lo: tuple[int, int, int] = (
                    axis_bounds[0][0],
                    axis_bounds[1][0],
                    axis_bounds[2][0],
                )
                global_hi: list[int, int, int] = [
                    axis_bounds[0][1],
                    axis_bounds[1][1],
                    axis_bounds[2][1],
                ]

                # Interior bounds: exclude ghost cells at DOMAIN boundaries only
                # At interior faces between chunks, ghost cells are owned by neighbor
                interior_lo: tuple[int, int, int] = [1, 1, 1]
                interior_hi_list: list[int, int, int] = [nx - 1, ny - 1, nz - 1]

                for ax_idx, ax_size, ax_nc in [(0, nx, ncx), (1, ny, ncy), (2, nz, ncz)]:
                    if ax_idx == split_axis_idx:
                        # Split axis: handled separately
                        interior_lo[ax_idx] = split_interior_lo
                        interior_hi_list[ax_idx] = split_interior_hi
                    else:
                        # Non-split axis: exclude ghost cells only at domain boundaries
                        # Interior face ghost cells are owned by neighbor, not excluded
                        if global_lo[ax_idx] == 0:
                            interior_lo[ax_idx] = 1
                        if global_hi[ax_idx] == ax_size:
                            # At domain boundary: exclude ghost cell at ax_size-1
                            interior_hi_list[ax_idx] = ax_size - 1
                        else:
                            # At interior face: ghost cell is owned by neighbor, include up to global_hi
                            interior_hi_list[ax_idx] = global_hi[ax_idx]

                interior_hi = tuple(interior_hi_list)

                # Owned bounds: for split axis, edge chunks include PML in owned
                # For non-split axes, exclude ghost cells at DOMAIN boundaries
                owned_lo = list(global_lo)
                owned_hi = list(global_hi)
                for ax_idx, ax_size, ax_nc in [(0, nx, ncx), (1, ny, ncy), (2, nz, ncz)]:
                    if global_lo[ax_idx] == 0:
                        owned_lo[ax_idx] = 1
                    if global_hi[ax_idx] == ax_size:
                        owned_hi[ax_idx] = ax_size - 1
                owned_lo = tuple(owned_lo)
                owned_hi = tuple(owned_hi)

                # Build face halos
                face_halos = _build_chunk_face_halos(
                    chunk_index=chunk_index,
                    global_lo=global_lo,
                    global_hi=global_hi,
                    interior_lo=interior_lo,
                    interior_hi=interior_hi,
                    boundary_spec=boundary_spec,
                    pml_depths=pml_depths,
                    split_axis_idx=split_axis_idx,
                    split_axis_name=split_axis_name,
                    n_split=n_split,
                    num_chunks=num_chunks,
                )

                local_shape = tuple(
                    global_hi[ax] - global_lo[ax] for ax in range(3)
                )

                chunk_spec = ChunkSpec(
                    chunk_index=chunk_index,
                    global_bounds=(tuple(global_lo), tuple(global_hi)),
                    interior_bounds=(tuple(interior_lo), tuple(interior_hi)),
                    owned_bounds=(owned_lo, owned_hi),
                    local_grid_shape=local_shape,
                    face_halos=face_halos,
                    is_reduced=False,
                    symmetry_multiplicity=1,
                    source_indices=(),
                    monitor_indices=(),
                )
                chunks.append(chunk_spec)

    # Build exchange plan
    exchange_plan = _build_exchange_plan(chunks, split_axis_idx, n_split, num_chunks)

    # Device assignment: round-robin chunks to available GPUs
    # For 2 GPUs, alternate chunks between device 0 and device 1
    num_gpus = 2  # Assume 2 GPUs for Phase 1
    device_assignment = tuple(
        chunks[i].chunk_index[split_axis_idx] % num_gpus
        if num_chunks[split_axis_idx] > 1
        else 0
        for i in range(total_chunks)
    )
    rank_assignment = tuple(0 for _ in range(total_chunks))

    return ChunkLayout(
        num_chunks=num_chunks,
        total_chunks=total_chunks,
        chunks=tuple(chunks),
        periodic_axes=boundary_spec.periodic_axes,
        bloch_axes=boundary_spec.bloch_axes,
        exchange_plan=exchange_plan,
        device_assignment=device_assignment,
        rank_assignment=rank_assignment,
    )


def _build_chunk_face_halos(
    chunk_index: tuple[int, int, int],
    global_lo: list[int, int, int],
    global_hi: list[int, int, int],
    interior_lo: tuple[int, int, int],
    interior_hi: tuple[int, int, int],
    boundary_spec: "CompiledBoundarySpec",
    pml_depths: dict[str, tuple[int, int]],
    split_axis_idx: int,
    split_axis_name: str,
    n_split: int,
    num_chunks: tuple[int, int, int],
) -> dict[str, FaceHalo]:
    """Build face halos for one chunk."""
    face_halos: dict[str, FaceHalo] = {}
    i, j, k = chunk_index

    for axis_name, axis_idx in [("x", 0), ("y", 1), ("z", 2)]:
        axis_boundary = boundary_spec.axes()[axis_idx]
        pml_minus, pml_plus = pml_depths[axis_name]

        for side, global_edge, interior_edge, is_split_axis in [
            ("minus", global_lo[axis_idx], interior_lo[axis_idx], axis_idx == split_axis_idx),
            ("plus", global_hi[axis_idx] - 1, interior_hi[axis_idx] - 1, axis_idx == split_axis_idx),
        ]:
            face_key = f"{axis_name}.{side}"
            edge = getattr(axis_boundary, side)

            # Determine chunk position along split axis
            chunk_pos = chunk_index[split_axis_idx]
            is_first = chunk_pos == 0
            is_last = chunk_pos == n_split - 1

            if is_split_axis:
                if side == "minus":
                    # Left edge of chunk along split axis
                    if is_first:
                        # Domain boundary (first chunk absorbs PML on this side)
                        exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_idx)
                        depth = _halo_depth_for_edge(edge, axis_idx)
                        neighbor = None
                    else:
                        # Interior face to previous chunk
                        exchange_kind = ExchangeKind.INTERIOR
                        depth = 1
                        neighbor_idx = list(chunk_index)
                        neighbor_idx[split_axis_idx] = chunk_pos - 1
                        neighbor = tuple(neighbor_idx)
                else:  # side == "plus"
                    # Right edge of chunk along split axis
                    if is_last:
                        # Domain boundary (last chunk absorbs PML on this side)
                        exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_idx)
                        depth = _halo_depth_for_edge(edge, axis_idx)
                        neighbor = None
                    else:
                        # Interior face to next chunk
                        exchange_kind = ExchangeKind.INTERIOR
                        depth = 1
                        neighbor_idx = list(chunk_index)
                        neighbor_idx[split_axis_idx] = chunk_pos + 1
                        neighbor = tuple(neighbor_idx)
            else:
                # Non-split axis
                n_axis_chunks = num_chunks[axis_idx]
                if n_axis_chunks == 1:
                    # Single chunk on this axis — both faces are domain boundaries
                    exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_idx)
                    depth = _halo_depth_for_edge(edge, axis_idx)
                    neighbor = None
                else:
                    # Compute first/last along this axis, not the split axis
                    axis_chunk_pos = chunk_index[axis_idx]
                    is_first = axis_chunk_pos == 0
                    is_last = axis_chunk_pos == n_axis_chunks - 1

                    if side == "minus":
                        if is_first:
                            # Domain boundary at start of axis
                            exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_idx)
                            depth = _halo_depth_for_edge(edge, axis_idx)
                            neighbor = None
                        else:
                            # Interior face to previous chunk
                            exchange_kind = ExchangeKind.INTERIOR
                            depth = 1
                            neighbor_idx = list(chunk_index)
                            neighbor_idx[axis_idx] = axis_chunk_pos - 1
                            neighbor = tuple(neighbor_idx)
                    else:  # side == "plus"
                        if is_last:
                            # Domain boundary at end of axis
                            exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_idx)
                            depth = _halo_depth_for_edge(edge, axis_idx)
                            neighbor = None
                        else:
                            # Interior face to next chunk
                            exchange_kind = ExchangeKind.INTERIOR
                            depth = 1
                            neighbor_idx = list(chunk_index)
                            neighbor_idx[axis_idx] = axis_chunk_pos + 1
                            neighbor = tuple(neighbor_idx)

            face_halos[face_key] = FaceHalo(
                axis=axis_name,
                side=side,
                depth=depth,
                neighbor_chunk_index=neighbor,
                exchange_kind=exchange_kind,
                phase_factor=edge.phase_factor,
                electric_signs=edge.electric_signs,
                magnetic_signs=edge.magnetic_signs,
            )

    return face_halos


def _build_exchange_plan(
    chunks: list[ChunkSpec],
    split_axis_idx: int,
    n_split: int,
    num_chunks: tuple[int, int, int],
) -> dict[str, ExchangeDescriptor]:
    """Build exchange plan for interior chunk boundaries."""
    exchange_plan: dict[str, ExchangeDescriptor] = {}

    for chunk in chunks:
        i, j, k = chunk.chunk_index
        chunk_pos = chunk.chunk_index[split_axis_idx]

        # Only interior faces need exchange descriptors
        for axis_name, axis_idx in [("x", 0), ("y", 1), ("z", 2)]:
            for side, neighbor_pos, src_side, dst_side in [
                ("minus", chunk_pos - 1, "plus", "minus"),
                ("plus", chunk_pos + 1, "minus", "plus"),
            ]:
                if axis_idx != split_axis_idx:
                    continue
                if neighbor_pos < 0 or neighbor_pos >= n_split:
                    continue

                # Find neighbor chunk
                neighbor_index = list(chunk.chunk_index)
                neighbor_index[split_axis_idx] = neighbor_pos
                neighbor_tuple = tuple(neighbor_index)

                # Get exchange kind from chunk's face halo
                face_key = f"{axis_name}.{side}"
                halo = chunk.face_halos.get(face_key)
                if halo is None:
                    continue

                # Create exchange descriptor
                desc_key = (
                    f"({chunk.chunk_index},{neighbor_tuple},{face_key})"
                )
                exchange_plan[desc_key] = ExchangeDescriptor(
                    source_chunk_index=neighbor_tuple,
                    dest_chunk_index=chunk.chunk_index,
                    axis=axis_name,
                    source_side=src_side,
                    dest_side=dst_side,
                    exchange_kind=halo.exchange_kind,
                    phase_factor=halo.phase_factor,
                )

    return exchange_plan


def _exchange_kind_from_boundary_edge(
    edge: "CompiledBoundaryEdge", axis_index: int
) -> ExchangeKind:
    """Determine the exchange kind for a boundary edge."""
    from autofdtd.compiler.boundaries import BoundaryMode

    mode = edge.mode
    if mode is BoundaryMode.PERIODIC:
        return ExchangeKind.PERIODIC
    if mode is BoundaryMode.BLOCH:
        return ExchangeKind.BLOCH
    if mode is BoundaryMode.PEC:
        return ExchangeKind.PEC
    if mode is BoundaryMode.PMC:
        return ExchangeKind.PMC
    if mode is BoundaryMode.ABC:
        return ExchangeKind.ABC
    if mode is BoundaryMode.PML:
        return ExchangeKind.PML
    if mode is BoundaryMode.STABLE_PML:
        return ExchangeKind.STABLE_PML
    if mode is BoundaryMode.ABSORBER:
        return ExchangeKind.ABSORBER
    return ExchangeKind.INTERIOR


def _halo_depth_for_edge(edge: "CompiledBoundaryEdge", axis_index: int) -> int:
    """Determine halo depth for a boundary edge."""
    from autofdtd.compiler.boundaries import BoundaryMode

    mode = edge.mode
    if mode in {
        BoundaryMode.PML,
        BoundaryMode.STABLE_PML,
        BoundaryMode.ABSORBER,
    }:
        if edge.pml is not None:
            return edge.pml.num_layers
    # All other boundaries use depth 1
    return 1


# ---------------------------------------------------------------------------
# Chunk-local CFL computation
# ---------------------------------------------------------------------------


def chunk_min_timestep(
    chunk_spec: ChunkSpec,
    cell_sizes: tuple[float, float, float],
    min_permittivity: float = 1.0,
    min_permeability: float = 1.0,
) -> float:
    """Compute the maximum allowed timestep for one chunk (CFL condition).

    Uses the 3D CFL condition for Yee grid with arbitrary cell sizes::

        dt <= min(dx, dy, dz) / (c * sqrt(3))

    where c = 1/sqrt(eps*mu) for the minimum constitutive parameters.

    Parameters
    ----------
    chunk_spec : ChunkSpec
        Chunk specification with local_grid_shape.
    cell_sizes : tuple[float, float, float]
        Physical cell sizes (dx, dy, dz).
    min_permittivity : float, optional
        Minimum relative permittivity in the chunk.
    min_permeability : float, optional
        Minimum relative permeability in the chunk.

    Returns
    -------
    float
        Maximum stable timestep for this chunk.
    """
    dx, dy, dz = cell_sizes
    c0 = 299_792_458.0
    eps = max(min_permittivity, 1e-10)
    mu = max(min_permeability, 1e-10)
    vmax = c0 / (eps**0.5 * mu**0.5)
    min_spacing = min(dx, dy, dz)
    cfl_factor = 0.99  # Safety margin
    return cfl_factor * min_spacing / (vmax * (3**0.5))


def global_min_timestep(
    chunks: tuple[ChunkSpec, ...],
    cell_sizes: tuple[float, float, float],
    chunk_min_permittivity: tuple[float, ...] | None = None,
    chunk_min_permeability: tuple[float, ...] | None = None,
) -> float:
    """Compute the global maximum allowed timestep across all chunks.

    Uses the minimum of all chunk-local CFL timesteps.

    Parameters
    ----------
    chunks : tuple[ChunkSpec, ...]
        All chunks in the layout.
    cell_sizes : tuple[float, float, float]
        Physical cell sizes.
    chunk_min_permittivity : tuple[float, ...], optional
        Per-chunk minimum permittivity.
    chunk_min_permeability : tuple[float, ...], optional
        Per-chunk minimum permeability.

    Returns
    -------
    float
        Global maximum stable timestep.
    """
    if not chunks:
        return float("inf")

    min_eps = chunk_min_permittivity or (1.0,) * len(chunks)
    min_mu = chunk_min_permeability or (1.0,) * len(chunks)

    dt_min = float("inf")
    for chunk, eps, mu in zip(chunks, min_eps, min_mu, strict=False):
        dt_chunk = chunk_min_timestep(chunk, cell_sizes, eps, mu)
        if dt_chunk < dt_min:
            dt_min = dt_chunk
    return dt_min


# ---------------------------------------------------------------------------
# Chunk boundary metadata helpers
# ---------------------------------------------------------------------------


def chunk_boundary_face_keys(
    chunk_spec: ChunkSpec,
) -> list[str]:
    """Return all boundary face keys for a chunk (excluding interior faces)."""
    boundary_keys = []
    for face_key, halo in chunk_spec.face_halos.items():
        if halo.neighbor_chunk_index is None:
            boundary_keys.append(face_key)
    return boundary_keys


def chunk_neighbor_info(
    chunk_spec: ChunkSpec,
    axis: str,
    side: str,
) -> tuple[ExchangeKind, complex, tuple[int, int, int], tuple[int, int, int]]:
    """Get neighbor exchange info for a chunk face.

    Returns
    -------
    tuple
        (exchange_kind, phase_factor, electric_signs, magnetic_signs)
    """
    halo = chunk_spec.face_halo(axis, side)
    if halo is None:
        return (
            ExchangeKind.INTERIOR,
            1.0 + 0.0j,
            (1, 1, 1),
            (1, 1, 1),
        )
    return (
        halo.exchange_kind,
        halo.phase_factor,
        halo.electric_signs,
        halo.magnetic_signs,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def chunk_layout_summary(layout: ChunkLayout) -> dict[str, Any]:
    """Return a diagnostic summary of a chunk layout."""
    return {
        "num_chunks": layout.num_chunks,
        "total_chunks": layout.total_chunks,
        "periodic_axes": layout.periodic_axes,
        "bloch_axes": layout.bloch_axes,
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "local_grid_shape": c.local_grid_shape,
                "num_cells": c.num_cells,
                "interior_num_cells": c.interior_num_cells,
                "is_reduced": c.is_reduced,
                "symmetry_multiplicity": c.symmetry_multiplicity,
                "boundary_faces": chunk_boundary_face_keys(c),
            }
            for c in layout.chunks
        ],
        "device_assignment": layout.device_assignment,
        "rank_assignment": layout.rank_assignment,
    }
