# Phase 1 Chunk Contract and Multi-Node Decomposition Architecture

## Status

**Draft — for review and task handoff**

## 1. Background and Design Motivation

Phase 1 (`task-047`) establishes the architectural contract for chunk-based decomposition before solver kernels are wired to a runtime executor. The goal is to define what a chunk owns, how chunks expose halo and monitor metadata, how chunk placement maps onto devices or nodes, and how this avoids blocking later systolic or overlapped execution phases.

### Reference Foundations

- **Meep chunk architecture** (`../meep/doc/docs/Chunks_and_Symmetry.md`): chunk is the stable decomposition unit, not MPI rank. A rank may own multiple chunks. Chunk boundaries align with PML layers and symmetry planes, not with MPI ranks. Communication plans are stage-specific and component-family-specific.
- **Systolic FDTD scheduling** (`papers/systolic_fdtd.pdf`): the important architectural seam is the split between chunk-local interior work, lower-dimensional boundary exchange, and a pluggable execution policy. Per-face, per-stage halo semantics and interior-versus-boundary planning must be explicit now so future systolic or overlapped execution can be added without redesign.
- **Phase 1 codebase state**: `CompiledSimulation` in `compiler/pipeline.py` treats the entire grid as one monolithic entity. `halo_depth=1` in `CompiledBoundarySpec` is a global constant. No chunk index, rank, ownership mask, or per-chunk field decomposition exists yet.

### Design Principles

1. **Chunk is the unit of parallelism.** Rank or device is placement, not computation.
2. **Logical and stored topology diverge.** Symmetry reduction and periodic/Bloch remapping are logical-address transforms over a full-domain view, not data-layout changes.
3. **Scene compilation is separate from stepping.** Geometry and material initialization produce coefficient fields; those fields are then chunked for execution.
4. **Bulk-synchronous by default, systolic-ready by design.** Phase 1 is bulk-synchronous, but chunk metadata and runtime orchestration preserve per-face, per-stage halo semantics so overlap can be layered in later without rearchitecting the chunk contract.
5. **Halo depth is per-face, not global.** Different boundary families (PML, ABC, Bloch, periodic) require different halo widths and different exchange semantics.

---

## 2. Chunk Data Model

### 2.1 ChunkSpec — Planning Metadata

`ChunkSpec` is a frozen planning metadata object produced during scene compilation (not a runtime data structure). It describes the intended decomposition before any field buffers are allocated.

```python
@dataclass(frozen=True)
class ChunkSpec:
    """Planning metadata for one chunk — produced during compilation."""

    chunk_index: tuple[int, int, int]          # logical position in chunk grid
    global_bounds: tuple[Vec3, Vec3]            # full-domain bounds of this chunk (lo, hi)
    interior_bounds: tuple[Vec3, Vec3]          # interior (non-halo) cell index range
    owned_bounds: tuple[Vec3, Vec3]              # cells this chunk updates (including boundary)
    local_grid_shape: tuple[int, int, int]       # full local array shape incl. ghost cells

    # Per-face halo metadata
    face_halos: dict[str, FaceHalo]              # key: "x.minus", "x.plus", "y.minus", ...

    # Symmetry metadata
    is_reduced: bool                             # True if chunk is in symmetry-reduced region
    symmetry_multiplicity: int                   # how many full-domain points this chunk represents

    # Ownership flags for sources and monitors
    source_indices: tuple[int, ...]             # indices into compiled_sources owned here
    monitor_indices: tuple[int, ...]              # indices into compiled_monitors owned here
```

**Key invariants:**
- `interior_bounds` defines the region this chunk updates without reading halo data.
- `owned_bounds` includes the one-cell-wide boundary ring that this chunk writes and other chunks read.
- `local_grid_shape` is the shape of the full local array including ghost cells — this is what field buffers are allocated to.
- For a chunk at chunk_index `(i, j, k)`, the global cell index range of its interior is `[interior_bounds.lo, interior_bounds.hi)` in grid coordinates.

### 2.2 FaceHalo — Per-Face Halo Descriptor

```python
@dataclass(frozen=True)
class FaceHalo:
    """Halo metadata for one face of one chunk."""

    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    depth: int                                  # number of ghost cells on this face
    neighbor_chunk_index: tuple[int, int, int] | None  # None = domain boundary
    exchange_kind: Literal[
        "periodic", "bloch", "pec", "pmc",
        "pml", "stable_pml", "absorber", "abc",
        "symmetry_copy", "interior"
    ]
    phase_factor: complex = 1.0 + 0.0j          # for Bloch: e^{i k · d}
    electric_signs: tuple[int, int, int] = (1, 1, 1)   # for PEC/PMC/symmetry reflection
    magnetic_signs: tuple[int, int, int] = (1, 1, 1)   # for PEC/PMC/symmetry reflection
    pml_profile: PMLFaceProfile | None = None   # for PML faces: coefficient profiles
    abc_coefficients: ABCFaceCoefficients | None = None  # for ABC faces
```

**Exchange kinds:**
- `periodic`: copy from neighbor's `plus` face to my `minus` ghost (or vice versa)
- `bloch`: same as periodic but multiply by `phase_factor` (complex conjugate on minus side for H, no conjugate on E per Maxwell's convention)
- `pec`: set ghost E to `-E_interior` reflected; H ghost from `+H_interior`
- `pmc`: opposite parity of PEC
- `pml` / `stable_pml` / `absorber`: staged damping update applied in PML stage (see Section 5)
- `abc`: first-order Engquist-Majda absorption via one-step recurrence
- `symmetry_copy`: fill ghost from interior using electric/magnetic sign transforms
- `interior`: no exchange — this face is adjacent to another local chunk (same process/device)

### 2.3 ChunkLayout — Global Decomposition Plan

```python
@dataclass(frozen=True)
class ChunkLayout:
    """Global chunk decomposition plan — immutable after compilation."""

    num_chunks: tuple[int, int, int]           # chunks per axis
    total_chunks: int                          # product of above
    chunks: tuple[ChunkSpec, ...]               # one per chunk
    periodic_axes: tuple[str, ...]             # axes with periodic/Bloch boundary
    bloch_axes: tuple[str, ...]                 # axes with Bloch boundary
    exchange_plan: dict[str, ExchangeDescriptor]  # keyed by ("chunk_a", "chunk_b", "face")

    # For device/rank mapping
    device_assignment: tuple[int, ...]         # chunk_index → device_id (for CUDA)
    rank_assignment: tuple[int, ...]            # chunk_index → MPI rank
```

**Key property:** `ChunkLayout.total_chunks` is the number of *logical* chunks, not necessarily the number of MPI ranks or CUDA devices. One rank/device may own multiple chunks.

---

## 3. What a Chunk Owns

A chunk is the unit of **field storage and field update**. Each chunk owns:

### 3.1 Field Arrays

For a chunk with `local_grid_shape = (Nx, Ny, Nz)`:

| Array | Shape | Staggering |
|---|---|---|
| `electric_field` | `(Nx, Ny, Nz, 3)` | Yee electric cell centers |
| `magnetic_field` | `(Nx, Ny, Nz, 3)` | Yee magnetic cell centers |
| `eps_xx`, `eps_yy`, `eps_zz` | `(Nx, Ny, Nz)` | permittivity on E-grid |
| `mu_xx`, `mu_yy`, `mu_zz` | `(Nx, Ny, Nz)` | permeability on H-grid |
| `sigma_e` (optional) | `(Nx, Ny, Nz)` | electric conductivity |
| `sigma_h` (optional) | `(Nx, Ny, Nz)` | magnetic conductivity |

**Note:** These are *local* arrays including ghost cells. The interior region `[interior_bounds.lo, interior_bounds.hi)` contains valid owned field values. Ghost cells are filled by the boundary exchange stage before each update.

### 3.2 Dispersive Auxiliary state

For PoleResidue media, per pole per component:

```
pol_x[p, i, j, k], pol_y[p, i, j, k], pol_z[p, i, j, k]  # polarization currents
```

Shape: `(num_poles, Nx, Ny, Nz)` per component axis. These are allocated only for chunks that contain dispersive material.

### 3.3 Anisotropic Auxiliary State

For diagonal AnisotropicMedia, the six `eps_xx` etc. arrays above are sufficient (no extra auxiliary state). If Phase 2 introduces fully anisotropic media, additional off-diagonal arrays would be needed.

### 3.4 PML/ABC Boundary state

Per face with PML/ABC:

```
PML memory: dict[face_key, np.ndarray]   # shape from PMLFaceState.memory_shape
ABC previous: dict[face_key, np.ndarray]  # shape from ABCFaceState.value_shape
```

These are allocated per chunk per absorbing face, not global.

### 3.5 Source State

Sources are not stored per chunk persistently. Instead, each chunk's `ChunkSpec.source_indices` tells which sources from `CompiledSources` inject into this chunk's domain. At runtime, source injection kernels use the compiled source metadata (placements, amplitude weights) to add source contributions directly into the chunk's `electric_field` or `magnetic_field` arrays.

### 3.6 Monitor State

Similarly, monitor recording state is per-chunk. For field monitors:
- `FieldMonitorState` per monitor that overlaps this chunk
- DFT accumulation buffers for frequency-domain monitors

For flux monitors:
- `FluxMonitorState` per monitor with surfaces intersecting this chunk

Chunk-local monitor data is gathered by a monitor collection stage after each recording interval.

---

## 4. Halo and Monitor Metadata

### 4.1 Halo Exchange Protocol

Halo exchange is **stage-specific** and **component-family-specific**, following the pattern established in the boundary compiler but scoped per chunk pair:

```
Timestep stage order per chunk:
  1. [boundary_exchange]    — fill ghost cells from neighbors
  2. [electric_update]       — update E in interior + owned-boundary cells
  3. [source_injection]      — add source contributions to E
  4. [boundary_exchange]     — re-exchange E ghosts (needed for H update)
  5. [magnetic_update]        — update H in interior + owned-boundary cells
  6. [monitor_collection]     — record E/H at monitor points (interval filter)
  7. [pml_stage]             — apply PML/ABC damping to E and H ghosts
  8. [convergence_check]     — (on interval) compute integrated E², check shutoff
```

**Key property:** The `boundary_exchange` stage operates on ghost cells only. Interior cells are never communicated during the timestep — only at the start of the step for halo fill and after the step for PML memory update.

### 4.2 Halo Depth

Halo depth is **per-face**, not a global constant. Typical depths:

| Boundary Type | Halo Depth | Notes |
|---|---|---|
| Periodic / Bloch | 1 cell | standard Yee ghost cell |
| PEC / PMC | 1 cell | ghost set by reflection transform |
| PML | variable (num_layers) | full PML depth, staggered |
| ABC (first-order) | 1 cell | single-step recurrence |
| Symmetry | 1 cell | copy from interior with sign transform |
| Interior (chunk neighbor) | 1 cell | local memory copy |

**The current `halo_depth=1` in `CompiledBoundarySpec` must be replaced** with per-face `FaceHalo.depth` from the `ChunkLayout`.

### 4.3 Monitor Overlap with Chunks

A monitor may span multiple chunks. Monitor metadata in `ChunkSpec.monitor_indices` tells which monitors overlap this chunk. Monitor data is accumulated per-chunk and gathered by a global collection pass.

For a `FieldMonitor` with interval/start semantics:
- Each chunk computes which of its interior cells fall within the monitor's spatial region
- Recording is done by the chunk that owns the cell
- Gathered data is assembled in rank-order for `SimulationData`

For a `FluxMonitor`:
- The flux surface may be split across multiple chunks
- Each chunk computes its portion of the Poynting flux integral
- Partial sums are reduced across chunks

---

## 5. PML / ABC State Decomposition

PML and ABC state must be decomposed per-chunk per-face. The current monolithic `PMLBoundaryState` in the codebase must be replaced.

### 5.1 PML Face State (per chunk, per face)

```python
@dataclass(frozen=True)
class ChunkPMLFaceState:
    """PML state for one face of one chunk."""

    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    memory_shape: tuple[int, ...]         # transverse shape of the PML layer
    sigma_profile: np.ndarray               # compiled sigma (sigma_min..sigma_max polynomial)
    kappa_profile: np.ndarray               # compiled kappa
    alpha_profile: np.ndarray               # compiled alpha
    memory: np.ndarray                      # persistent PML memory (attenuation term)
```

### 5.2 ABC Face State (per chunk, per face)

```python
@dataclass(frozen=True)
class ChunkABCFaceState:
    """First-order ABC state for one face of one chunk."""

    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    value_shape: tuple[int, ...]            # transverse shape of the ABC surface
    previous_boundary: np.ndarray           # E or H at previous timestep on boundary
    previous_adjacent: np.ndarray           # E or H at previous timestep one cell inside
```

---

## 6. Device and Node Placement

### 6.1 CUDA GPU Placement

The `ChunkLayout.device_assignment` field maps each logical `chunk_index` to a CUDA device ID. For Phase 1 single-GPU or dual-GPU smoke tests:

- All chunks map to device 0 (single-GPU fallback)
- Or chunks 0..N//2 on device 0, chunks N//2..N on device 1

**This mapping is a planning artifact, not the execution runtime.** The actual field arrays are allocated on the target device via CUDA managed memory or explicit device arrays.

### 6.2 Multi-Node / MPI Placement

`ChunkLayout.rank_assignment` maps each logical chunk to an MPI rank. For Phase 1, this is always all chunks to rank 0 (no MPI decomposition). Later phases extend to per-rank ownership with MPI halo exchange.

**Separation of concerns:**
- `ChunkSpec` is pure planning metadata — no field data, no communication
- `ChunkLayout` adds device/rank placement hints
- Actual field arrays and PML state are allocated at runtime using `ChunkSpec.local_grid_shape`
- Halo exchange uses `ChunkLayout.exchange_plan` to know which chunk pairs need to communicate

### 6.3 Warp Kernel Conventions

Each chunk's field arrays are Warp-compatible `wp.array` objects allocated on the target device. Warp kernel launches use chunk-local index ranges:

```python
# Pseudo-code for chunk-local kernel launch
chunk = chunks[chunk_id]
wp.launch(
    electric_update_kernel,
    dim=chunk.local_grid_shape,
    inputs=[chunk.electric_field, chunk.magnetic_field, chunk.eps, chunk.mu],
    device=chunk.device_id,
)
```

Halo exchange is implemented as separate Warp kernels that read from neighbor chunks' field arrays and write to ghost cells.

---

## 7. Symmetry and Logical vs. Stored Topology

### 7.1 Symmetry Reduction

When `Simulation.symmetry` is non-zero, only the symmetry-reduced portion of the domain is stored. `ChunkSpec.is_reduced` and `ChunkSpec.symmetry_multiplicity` track which chunks are in the reduced region.

`ChunkSpec.face_halos["x.minus"].electric_signs` and `.magnetic_signs` carry the pre-computed reflection signs per face from the `SymmetryAxisIR` data. These are the same signs used for PEC/PMC boundaries — the symmetry transform is a logical address transform applied during the halo exchange.

### 7.2 Periodic and Bloch Remapping

For periodic axes, the `ChunkLayout.exchange_plan` connects the `plus` face of the last chunk to the `minus` face of the first chunk (and vice versa). For Bloch, the exchange includes the `phase_factor` from `BlochBoundaryIR`.

**Complex field storage:** `BoundarySpecIR.requires_complex_fields=True` when any Bloch axis is present. This flag forces complex dtype for all field arrays even when `bloch_vec == 0` for that axis.

---

## 8. Gaps and Deferred Design Points

The following are intentionally **deferred** beyond Phase 1:

| Item | Reason for Deferral | Future Task |
|---|---|---|
| MPI halo exchange kernels | No MPI runner in Phase 1 | task-050+ |
| Dynamic load balancing | Static chunk layout sufficient for Phase 1 | Meep-style chunk_balancer |
| Chunk-local CFL computation | Global CFL sufficient for Phase 1; per-chunk min_step available in ChunkSpec for future | task-051 |
| Overlapped / systolic execution | Bulk-synchronous is the Phase 1 contract; ChunkLayout preserves the hooks | task-052+ |
| Non-uniform PML profile per chunk | PML coefficients are compiled globally; chunk-local profiles are a refinement | task-053 |
| Chunk-local subpixel materialization | Phase 1 subpixel uses AnisotropicMedium path; per-chunk averaging is a refinement | task-054 |

---

## 9. Handoff Points for Later Tasks

### task-048 — Warp Timestepping Kernel
- Consume `ChunkLayout` and `ChunkSpec` to allocate field arrays per chunk
- Implement `electric_update_kernel` and `magnetic_update_kernel` with chunk-local index ranges
- Halo exchange becomes a separate kernel stage

### task-049 — Runtime Executor
- Consume `CompiledSimulation` + `ChunkLayout` to build per-chunk field state
- Implement `run_chunked_simulation()` that loops over chunks, applies stage order, handles convergence

### task-050 — MPI Halo Exchange
- Extend `ChunkLayout.rank_assignment` to multiple ranks
- Implement `mpi_halo_exchange()` using `exchange_plan`
- Collective operations for flux/mode monitor gathering

### task-051 — CFL and Chunk Balancing
- Use `ChunkSpec.local_grid_shape` and per-axis `cell_sizes` to compute per-chunk `min_step`
- If min_step varies across chunks, use the minimum for the global timestep

### task-053 — Monitor Accumulation Kernels
- Use `ChunkSpec.monitor_indices` to route recording to the owning chunk
- Implement gather/reduce for multi-chunk monitors (flux surfaces, mode overlaps)

---

## 10. Summary of Required Changes

To realize this contract in code, the following changes are needed:

1. **New module `autofdtd/runtime/chunk.py`** with `ChunkSpec`, `FaceHalo`, `ChunkLayout`, `ChunkPMLFaceState`, `ChunkABCFaceState`.

2. **`autofdtd/compiler/pipeline.py`** extended to produce `ChunkLayout` after `CompiledSimulation`.

3. **`autofdtd/compiler/boundaries.py`** — replace `halo_depth: int = 1` with per-face `FaceHalo` descriptors attached to each `ChunkSpec`.

4. **`autofdtd/ir/models.py`** — add `ChunkSpecIR`, `FaceHaloIR`, `ChunkLayoutIR` as versioned IR transport types for chunk decomposition.

5. **`autofdtd/kernels/boundaries.py`** — refactor `_ghost_and_source_slices` into chunk-aware exchange kernels that use `FaceHalo.exchange_kind` and `FaceHalo.phase_factor`.

6. **`autofdtd/runtime/controls.py`** — extend `RuntimeController` to work with `ChunkLayout` and per-chunk step tracking.

7. **`autofdtd/diagnostics/runtime_logging.py`** — extend `RuntimeExecutionLog` to include per-chunk throughput metrics (cells updated per second per chunk).

---

## References

- Meep: `../meep/doc/docs/Chunks_and_Symmetry.md`
- Meep Parallel: `../meep/doc/docs/Parallel_Meep.md`
- Systolic FDTD: `papers/systolic_fdtd.pdf`
- GPU Benchmarking: `papers/gpu_benchmarking.pdf`
- Meep paper: `papers/meep_paper.pdf`
- Current `CompiledSimulation`: `src/autofdtd/compiler/pipeline.py`
- Current `CompiledBoundarySpec`: `src/autofdtd/compiler/boundaries.py`
- Current `RuntimeController`: `src/autofdtd/runtime/controls.py`
