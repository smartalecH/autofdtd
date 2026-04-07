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

    For Phase 1, this produces a monolithic single-chunk layout that
    covers the full domain. Future tasks (task-054+) extend to
    multi-chunk decomposition.

    Parameters
    ----------
    num_chunks : tuple[int, int, int]
        Desired chunks per axis. Phase 1 uses (1, 1, 1).
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
        Chunk layout with one monolithic chunk.
    """
    from autofdtd.compiler.boundaries import compile_boundary_spec

    if boundary_spec is None:
        boundary_spec = compile_boundary_spec(None, symmetry=symmetry)

    nx, ny, nz = grid_shape
    # Halo depth of 1 for all faces in Phase 1 monolithic layout
    interior_lo = (1, 1, 1)
    interior_hi = (nx - 1, ny - 1, nz - 1)
    owned_lo = (1, 1, 1)
    owned_hi = (nx - 1, ny - 1, nz - 1)

    face_halos: dict[str, FaceHalo] = {}

    # Build face halos from compiled boundary spec
    for axis_name, axis_boundary in zip("xyz", boundary_spec.axes(), strict=True):
        axis_index = "xyz".index(axis_name)
        for side in ("minus", "plus"):
            edge = getattr(axis_boundary, side)
            face_key = f"{axis_name}.{side}"

            # Determine exchange kind
            exchange_kind = _exchange_kind_from_boundary_edge(edge, axis_index)
            depth = _halo_depth_for_edge(edge, axis_index)

            # Determine neighbor (None for domain boundary in monolithic layout)
            neighbor = None

            # Get signs
            electric_signs = edge.electric_signs
            magnetic_signs = edge.magnetic_signs
            phase_factor = edge.phase_factor

            face_halos[face_key] = FaceHalo(
                axis=axis_name,
                side=side,
                depth=depth,
                neighbor_chunk_index=neighbor,
                exchange_kind=exchange_kind,
                phase_factor=phase_factor,
                electric_signs=electric_signs,
                magnetic_signs=magnetic_signs,
            )

    # Single chunk covering the entire domain
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

    # Device assignment: all chunks on device 0 for Phase 1
    device_assignment = tuple(0 for _ in range(num_chunks[0] * num_chunks[1] * num_chunks[2]))
    rank_assignment = tuple(0 for _ in range(num_chunks[0] * num_chunks[1] * num_chunks[2]))

    return ChunkLayout(
        num_chunks=num_chunks,
        total_chunks=num_chunks[0] * num_chunks[1] * num_chunks[2],
        chunks=(chunk_spec,),
        periodic_axes=boundary_spec.periodic_axes,
        bloch_axes=boundary_spec.bloch_axes,
        exchange_plan={},
        device_assignment=device_assignment,
        rank_assignment=rank_assignment,
    )


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
