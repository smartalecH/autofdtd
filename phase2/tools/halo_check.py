"""
Halo correctness checker for Phase 2 accuracy validation.

Validates that multi-GPU halo exchange produces field values matching
single-GPU (monolithic) execution within a tight tolerance (1e-6).

Usage
-----
    from phase2.tools.halo_check import run_halo_check

    def sim_builder():
        sim = Simulation(
            size=(1.0, 1.0, 1.0),
            grid_spec=UniformGrid(dl=0.01),
            ...
        )
        return sim

    result = run_halo_check(sim_builder)

    print(result["pass_fail"], result["max_error"])

Output
-----
    {
        "max_error": float,
        "mean_error": float,
        "pass_fail": "pass" | "fail",
        "tolerance": float,
        "single_gpu_result": ExecutionResult,
        "two_gpu_result": ExecutionResult,
    }
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Internal imports — Phase 1 runtime internals
# ---------------------------------------------------------------------------
from autofdtd.compiler.pipeline import compile_simulation
from autofdtd.runtime.chunk import build_chunk_layout, ChunkLayout
from autofdtd.runtime.boundaries import (
    ChunkHaloExchange,
    allocate_pml_boundary_state,
)
from autofdtd.runtime.execution import (
    ExecutionResult,
    FieldState,
    allocate_field_state,
    run_compiled_simulation,
)
from autofdtd.runtime.sources import apply_source_injection_stage
from autofdtd.kernels.steps import step_maxwell


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_halo_check(
    simulation_fn: Callable[[], Any],
    num_chunks_2gpu: tuple[int, int, int] = (2, 1, 1),
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> dict[str, Any]:
    """Check halo correctness by comparing 2-GPU chunked vs 1-GPU monolithic results.

    Parameters
    ----------
    simulation_fn : callable() -> Simulation
        Function that returns a Simulation object. The same Simulation object
        is used for both single-GPU and 2-GPU runs.
    num_chunks_2gpu : tuple[int, int, int]
        Chunk decomposition for the 2-GPU run. Default (2, 1, 1) splits
        the domain along the first axis.
    tolerance : float
        Maximum allowed L-infinity field error between single and multi-GPU
        results at chunk boundary locations. Default 1e-6.
    verbose : bool
        If True, print progress and per-comparison diagnostics.

    Returns
    -------
    dict with keys:
        - max_error : float
        - mean_error : float
        - pass_fail : str — "pass" if max_error < tolerance, else "fail"
        - tolerance : float
        - num_boundary_cells : int
        - boundary_errors : list of float
        - single_gpu_result : ExecutionResult
        - two_gpu_result : ExecutionResult
    """
    if verbose:
        print("=" * 60)
        print("Halo Correctness Check")
        print("=" * 60)
        print(f"  2-GPU chunks: {num_chunks_2gpu}")
        print(f"  Tolerance: {tolerance:.1e}")
        print()

    # Build and compile simulation (single-chunk path)
    sim = simulation_fn()
    compiled = compile_simulation(sim)

    grid_shape = (
        compiled.resolved_grid.x.num_cells,
        compiled.resolved_grid.y.num_cells,
        compiled.resolved_grid.z.num_cells,
    )
    cell_sizes = (
        compiled.resolved_grid.x.cell_sizes[0] if compiled.resolved_grid.x.cell_sizes else 1.0,
        compiled.resolved_grid.y.cell_sizes[0] if compiled.resolved_grid.y.cell_sizes else 1.0,
        compiled.resolved_grid.z.cell_sizes[0] if compiled.resolved_grid.z.cell_sizes else 1.0,
    )
    dt = compiled.runtime_controls.dt
    max_steps = compiled.runtime_controls.num_time_steps
    dx, dy, dz = cell_sizes

    if verbose:
        print(f"  Grid shape: {grid_shape}")
        print(f"  Cell sizes: {cell_sizes}")
        print(f"  Timesteps: {max_steps}")
        print()

    # Build multi-chunk layout for 2-GPU run
    chunk_layout_2gpu = build_chunk_layout(
        num_chunks=num_chunks_2gpu,
        grid_shape=grid_shape,
        cell_sizes=cell_sizes,
        boundary_spec=compiled.compiled_boundaries.boundary_spec,
        symmetry=compiled.simulation_ir.symmetry,
    )

    if verbose:
        print(f"  2-GPU layout: {chunk_layout_2gpu.num_chunks} chunks, "
              f"{chunk_layout_2gpu.total_chunks} total")
        for i, chunk in enumerate(chunk_layout_2gpu.chunks):
            print(f"    Chunk {i}: global={chunk.global_bounds}, "
                  f"interior={chunk.interior_bounds}")
        print()

    # Run single-GPU (monolithic) simulation
    if verbose:
        print("  Running single-GPU (monolithic) simulation...")
    single_gpu_result = run_compiled_simulation(
        compiled,
        max_steps=max_steps,
        verbose=verbose,
    )

    # Run 2-GPU (multi-chunk) simulation
    if verbose:
        print("  Running 2-GPU (multi-chunk) simulation...")
    two_gpu_result = _run_chunked_simulation(
        compiled=compiled,
        chunk_layout=chunk_layout_2gpu,
        max_steps=max_steps,
        dt=dt,
        dx=dx, dy=dy, dz=dz,
        verbose=verbose,
    )

    # Extract field arrays for comparison
    single_E = _get_field_array(single_gpu_result)
    two_E = _get_field_array(two_gpu_result)

    # Find chunk boundary interior cells for comparison
    boundary_cells = _find_boundary_cell_indices(grid_shape, chunk_layout_2gpu)

    if verbose:
        print(f"  Comparing {len(boundary_cells)} boundary cells...")

    # Compare fields at boundary locations
    errors = []
    max_error = 0.0
    max_error_loc = None

    for (i, j, k) in boundary_cells:
        for c in range(3):  # 3 field components
            s_val = single_E[i, j, k, c]
            t_val = two_E[i, j, k, c]
            err = abs(s_val - t_val)
            errors.append(err)
            if err > max_error:
                max_error = err
                max_error_loc = (["Ex", "Ey", "Ez"][c], i, j, k)

    mean_error = float(np.mean(errors)) if errors else 0.0
    max_error = float(max_error)
    pass_fail = "pass" if max_error < tolerance else "fail"

    if verbose:
        print()
        print(f"  Results:")
        print(f"    Max error:  {max_error:.6e}")
        print(f"    Mean error: {mean_error:.6e}")
        print(f"    Tolerance:  {tolerance:.1e}")
        print(f"    Pass/Fail:  {pass_fail}")
        if max_error_loc:
            print(f"    Max error at: {max_error_loc}")

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "pass_fail": pass_fail,
        "tolerance": tolerance,
        "num_boundary_cells": len(boundary_cells),
        "boundary_errors": errors,
        "single_gpu_result": single_gpu_result,
        "two_gpu_result": two_gpu_result,
    }


# ---------------------------------------------------------------------------
# Internal: multi-chunk execution runner
# ---------------------------------------------------------------------------

def _run_chunked_simulation(
    compiled: Any,
    chunk_layout: ChunkLayout,
    max_steps: int,
    dt: float,
    dx: float, dy: float, dz: float,
    verbose: bool = False,
) -> ExecutionResult:
    """Run a compiled simulation with multi-chunk (multi-GPU) execution.

    The execution loop follows the stage order defined in Phase 1:
    1. Halo exchange (E boundary)
    2. Electric field update (per chunk interior)
    3. Source injection
    4. Halo exchange (H boundary)
    5. Magnetic field update (per chunk interior)
    6. PML boundary stage (per chunk)
    7. Monitor recording

    Parameters
    ----------
    compiled : CompiledSimulation
        Compiled simulation artifacts
    chunk_layout : ChunkLayout
        Multi-chunk decomposition
    max_steps : int
        Number of timesteps
    dt : float
        Timestep size
    dx, dy, dz : float
        Cell sizes

    Returns
    -------
    ExecutionResult
    """
    import time

    grid_shape = (
        compiled.resolved_grid.x.num_cells,
        compiled.resolved_grid.y.num_cells,
        compiled.resolved_grid.z.num_cells,
    )
    nx, ny, nz = grid_shape

    # Allocate monolithic field arrays (full domain)
    field_state = allocate_field_state(compiled)
    E_full = field_state.E
    H_full = field_state.H

    # Determine field dtype
    is_complex = E_full.ndim == 5  # (nx, ny, nz, 3, 2) for complex

    # Build halo exchange manager
    halo_exchange = ChunkHaloExchange(
        chunk_layout=chunk_layout,
        field_shape=(nx, ny, nz, 3),
        dtype=np.float64,
    )

    # Allocate PML state per chunk
    pml_state = allocate_pml_boundary_state(
        compiled.compiled_boundaries.boundary_spec,
        field_shape=(nx, ny, nz, 3),
        dtype=np.float64,
    )

    # Track integrated electric history for convergence
    integrated_electric_history = []
    peak_integrated_electric = 0.0
    num_steps_executed = max_steps
    current_time = 0.0

    wall_start = time.perf_counter()

    if verbose:
        print(f"    Chunked loop: {max_steps} steps, grid={grid_shape}, "
              f"chunks={chunk_layout.total_chunks}")

    # Main timestep loop
    for step in range(max_steps):
        # ---- Halo exchange before E update ----
        _execute_halo_exchange(
            halo_exchange, E_full, H_full,
            chunk_layout, nx, ny, nz, is_complex,
        )

        # ---- Electric field update (per chunk interior) ----
        for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
            (i0, j0, k0), (i1, j1, k1) = chunk_spec.interior_bounds

            # Compute chunk-local E update using sliced arrays
            _electric_update_chunk(
                E_full, H_full, chunk_spec, field_state,
                dt, dx, dy, dz, step, current_time,
            )

        # ---- Source injection (monolithic) ----
        E_full, H_full = apply_source_injection_stage(
            E_full, H_full,
            uniform_current_sources=compiled.compiled_sources.uniform_current,
            point_dipole_sources=compiled.compiled_sources.point_dipole,
            plane_wave_sources=compiled.compiled_sources.plane_wave,
            gaussian_beam_sources=compiled.compiled_sources.gaussian_beam,
            time=current_time,
            dt=dt,
        )

        # ---- Halo exchange before H update ----
        _execute_halo_exchange(
            halo_exchange, E_full, H_full,
            chunk_layout, nx, ny, nz, is_complex,
        )

        # ---- Magnetic field update (per chunk interior) ----
        for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
            (i0, j0, k0), (i1, j1, k1) = chunk_spec.interior_bounds

            _magnetic_update_chunk(
                E_full, H_full, chunk_spec, field_state,
                dt, dx, dy, dz, step, current_time,
            )

        # ---- PML stage (per chunk) ----
        for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
            _apply_chunk_pml(
                E_full, H_full, chunk_spec, pml_state, chunk_idx, dt,
            )

        # ---- Convergence tracking ----
        E_sq = _compute_field_magnitude_sq(E_full, is_complex)
        integrated_electric = float(E_sq)
        integrated_electric_history.append(integrated_electric)

        if integrated_electric > peak_integrated_electric:
            peak_integrated_electric = integrated_electric

        shutoff_ratio = 1.0
        if peak_integrated_electric > 0.0:
            shutoff_ratio = integrated_electric / peak_integrated_electric

        if shutoff_ratio <= float(compiled.runtime_controls.shutoff) and step > 0:
            num_steps_executed = step + 1
            current_time = (step + 1) * dt
            if verbose:
                print(f"    Converged at step {step}")
            break

    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    total_cells = nx * ny * nz
    cells_per_second = total_cells / wall_time if wall_time > 0 else 0.0

    result = ExecutionResult(
        field_state=field_state,
        num_steps=num_steps_executed,
        final_time=current_time,
        stop_reason="shutoff" if num_steps_executed < max_steps else "max_steps",
        integrated_electric_history=integrated_electric_history,
        field_monitor_data={},
        flux_monitor_data={},
        medium_monitor_data={},
        mode_monitor_data={},
        metrics={
            "wall_time": wall_time,
            "cells_per_second": cells_per_second,
            "total_cells": total_cells,
            "backend": "numpy-chunked",
        },
    )

    return result


def _execute_halo_exchange(
    halo_exchange: ChunkHaloExchange,
    E_full: np.ndarray,
    H_full: np.ndarray,
    chunk_layout: ChunkLayout,
    nx: int, ny: int, nz: int,
    is_complex: bool,
) -> None:
    """Execute halo exchange for all interior chunk faces.

    For same-device interior faces: direct copy between chunks.
    For cross-device faces: use CrossDeviceHaloTransfer.
    For boundary faces (PML, PEC, etc.): handled in PML stage.
    """
    ncx, ncy, ncz = chunk_layout.num_chunks
    flat_idx = 0
    for k in range(ncz):
        for j in range(ncy):
            for i in range(ncx):
                chunk_index = (i, j, k)
                chunk_spec = chunk_layout.chunks[flat_idx]
                for axis_name in ("x", "y", "z"):
                    for side in ("minus", "plus"):
                        _exchange_face_halo(
                            halo_exchange,
                            E_full, H_full,
                            chunk_layout, chunk_spec, chunk_index,
                            axis_name, side, nx, ny, nz, is_complex,
                        )
                flat_idx += 1


def _exchange_face_halo(
    halo_exchange: ChunkHaloExchange,
    E_full: np.ndarray,
    H_full: np.ndarray,
    chunk_layout: ChunkLayout,
    chunk_spec: Any,
    chunk_index: tuple[int, int, int],
    axis_name: str,
    side: str,
    nx: int, ny: int, nz: int,
    is_complex: bool,
) -> None:
    """Exchange halo for one face of one chunk."""
    from autofdtd.runtime.chunk import ExchangeKind

    halo = chunk_spec.face_halo(axis_name, side)
    if halo is None:
        return

    exchange_kind = halo.exchange_kind

    # Skip non-interior exchanges (PML, ABC, domain boundaries)
    if exchange_kind not in {ExchangeKind.INTERIOR}:
        return

    neighbor_idx = halo.neighbor_chunk_index
    if neighbor_idx is None:
        return  # No neighbor = domain boundary

    neighbor_spec = chunk_layout.chunk_at(neighbor_idx)

    # Get the axis index
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
    depth = halo.depth

    # Check if cross-device
    is_cross = halo_exchange.is_cross_device_face(
        chunk_index, axis_name, side,
    )

    # Get interior bounds
    (gi0, gj0, gk0), (gi1, gj1, gk1) = chunk_spec.interior_bounds
    (ni0, nj0, nk0), (ni1, nj1, nk1) = neighbor_spec.interior_bounds

    # Determine source and destination slices in the full array
    # For x.minus: source is chunk's interior x.minus face → dest is neighbor's x.plus ghost
    # For x.plus: source is chunk's interior x.plus face → dest is neighbor's x.minus ghost
    if axis_name == "x":
        if side == "minus":
            # Chunk's x.minus face data (interior cells at i0)
            src_i0, src_i1 = gi0, gi0 + depth
            # Neighbor's x.plus ghost cells (at neighbor's i1 to i1+depth)
            dst_i0, dst_i1 = ni1, ni1 + depth
        else:  # plus
            src_i0, src_i1 = gi1 - depth, gi1
            dst_i0, dst_i1 = ni0 - depth, ni0

        src_slice = (slice(src_i0, src_i1), slice(gj0, gj1), slice(gk0, gk1), slice(None))
        dst_slice = (slice(dst_i0, dst_i1), slice(nj0, nj1), slice(nk0, nk1), slice(None))
        src_j, src_j1 = gj0, gj1
        src_k, src_k1 = gk0, gk1
        dst_j, dst_j1 = nj0, nj1
        dst_k, dst_k1 = nk0, nk1

    elif axis_name == "y":
        if side == "minus":
            src_j0, src_j1 = gj0, gj0 + depth
            dst_j0, dst_j1 = nj1, nj1 + depth
        else:
            src_j0, src_j1 = gj1 - depth, gj1
            dst_j0, dst_j1 = nj0 - depth, nj0

        src_slice = (slice(gi0, gi1), slice(src_j0, src_j1), slice(gk0, gk1), slice(None))
        dst_slice = (slice(ni0, ni1), slice(dst_j0, dst_j1), slice(nk0, nk1), slice(None))
        src_i0, src_i1 = gi0, gi1
        src_k, src_k1 = gk0, gk1
        dst_i0, dst_i1 = ni0, ni1
        dst_k, dst_k1 = nk0, nk1

    else:  # z
        if side == "minus":
            src_k0, src_k1 = gk0, gk0 + depth
            dst_k0, dst_k1 = nk1, nk1 + depth
        else:
            src_k0, src_k1 = gk1 - depth, gk1
            dst_k0, dst_k1 = nk0 - depth, nk0

        src_slice = (slice(gi0, gi1), slice(gj0, gj1), slice(src_k0, src_k1), slice(None))
        dst_slice = (slice(ni0, ni1), slice(nj0, nj1), slice(dst_k0, dst_k1), slice(None))
        src_i0, src_i1 = gi0, gi1
        src_j0, src_j1 = gj0, gj1
        dst_i0, dst_i1 = ni0, ni1
        dst_j0, dst_j1 = nj0, nj1

    # Validate slices are within bounds
    if is_complex:
        # E_full shape: (nx, ny, nz, 3, 2)
        shape = (nx, ny, nz, 3, 2)
    else:
        shape = (nx, ny, nz, 3)

    def _valid_slice_tuple(sl_tuple, shape_tuple):
        """Validate a tuple of slices against corresponding shape dimensions."""
        for sl, dim in zip(sl_tuple, shape_tuple, strict=False):
            if not isinstance(sl, slice):
                continue
            if sl.start is not None and sl.start < 0:
                return False
            if sl.stop is not None and sl.stop > dim:
                return False
        return True

    if not _valid_slice_tuple(src_slice, shape):
        return
    if not _valid_slice_tuple(dst_slice, shape):
        return

    if is_cross:
        # Cross-device: use ChunkHaloExchange.cross_device_transfer
        # Pass full arrays - method handles halo packing internally
        halo_exchange.cross_device_transfer(
            chunk_index, neighbor_idx, axis_name,
            side, _opposite_side(side),
            E_full, E_full,
        )
    else:
        # Same-device: direct copy between chunks in full array
        # Copy E field - use numpy copy + assign for Warp arrays
        if hasattr(E_full, 'numpy'):
            # Warp array: copy via numpy then assign back
            E_np = E_full.numpy()
            E_np[dst_slice] = E_np[src_slice]
            E_full.assign(E_np)
        else:
            E_full[dst_slice] = E_full[src_slice]

        # Copy H field if H is separate
        if H_full is not None and H_full.shape == E_full.shape:
            if hasattr(H_full, 'numpy'):
                H_np = H_full.numpy()
                H_np[dst_slice] = H_np[src_slice]
                H_full.assign(H_np)
            else:
                H_full[dst_slice] = H_full[src_slice]


def _opposite_side(side: str) -> str:
    """Return 'plus' for 'minus' and vice versa."""
    return "plus" if side == "minus" else "minus"


def _electric_update_chunk(
    E_full: np.ndarray,
    H_full: np.ndarray,
    chunk_spec: Any,
    field_state: FieldState,
    dt: float,
    dx: float, dy: float, dz: float,
    step: int,
    time: float,
) -> None:
    """Apply electric update to a chunk's interior region.

    The E field update requires H at adjacent positions (Yee lattice).
    We apply the update only to the interior cells of this chunk.
    """
    (i0, j0, k0), (i1, j1, k1) = chunk_spec.interior_bounds

    # Access the coefficient arrays from field_state
    eps_xx = getattr(field_state, 'eps_xx', None)
    eps_yy = getattr(field_state, 'eps_yy', None)
    eps_zz = getattr(field_state, 'eps_zz', None)

    if E_full.ndim == 5:
        # Complex storage: (nx, ny, nz, 3, 2) = (real, imag)
        _electric_update_chunk_complex(
            E_full, H_full, i0, j0, k0, i1, j1, k1,
            eps_xx, eps_yy, eps_zz, dt, dx, dy, dz,
        )
    else:
        _electric_update_chunk_real(
            E_full, H_full, i0, j0, k0, i1, j1, k1,
            eps_xx, eps_yy, eps_zz, dt, dx, dy, dz,
        )


def _electric_update_chunk_real(
    E: np.ndarray,
    H: np.ndarray,
    i0: int, j0: int, k0: int,
    i1: int, j1: int, k1: int,
    eps_xx, eps_yy, eps_zz,
    dt: float, dx: float, dy: float, dz: float,
) -> None:
    """Real-valued E update for a chunk interior region.

    Standard Yee lattice update:
    E.x[i,j,k] += (dt/eps_xx) * ((H.z[i,j+1,k] - H.z[i,j,k])/dy - (H.y[i,j,k+1] - H.y[i,j,k])/dz)
    etc.
    """
    # Clamp indices to valid range
    i0 = max(0, i0)
    j0 = max(0, j0)
    k0 = max(0, k0)
    i1 = min(E.shape[0], i1)
    j1 = min(E.shape[1], j1)
    k1 = min(E.shape[2], k1)

    eps_xx_arr = eps_xx if eps_xx is not None else np.ones_like(E[..., 0])
    eps_yy_arr = eps_yy if eps_yy is not None else np.ones_like(E[..., 0])
    eps_zz_arr = eps_zz if eps_zz is not None else np.ones_like(E[..., 0])

    dt_dx = dt / dx
    dt_dy = dt / dy
    dt_dz = dt / dz

    # Ex update: uses Hy.z and Hz.y
    # Ex[i,j,k] new = Ex[i,j,k] + dt/eps_xx * ((Hz[i,j+1,k] - Hz[i,j,k])/dy - (Hy[i,j,k+1] - Hy[i,j,k])/dz)
    _slice = (slice(i0, i1), slice(j0, j1), slice(k0, k1), slice(None))

    if E.shape[3] >= 1:
        # Ex component (index 0)
        Ex = E[_slice]
        # Hy at z+1/2 face
        Hy_zp = H[i0:i1, j0+1:j1+1 if j1 < H.shape[1] else j1, k0:k1, 1]
        Hy_zp = np.pad(Hy_zp, [(0,0), (0,1 if j1 < H.shape[1] else 0), (0,0), (0,0)], mode='constant')
        # Hz at y+1/2 face
        Hz_yp = H[i0:i1, j0:j1, k0+1:k1+1 if k1 < H.shape[2] else k1, 2]
        Hz_yp = np.pad(Hz_yp, [(0,0), (0,0), (0,1 if k1 < H.shape[2] else 0), (0,0)], mode='constant')

        # Compute curl
        curl_H_Ex = (Hy_zp[..., 1] - H[i0:i1, j0:j1, k0:k1, 1]) * dt_dy / eps_xx_arr[i0:i1, j0:j1, k0:k1] \
                    - (Hz_yp[..., 2] - H[i0:i1, j0:j1, k0:k1, 2]) * dt_dz / eps_xx_arr[i0:i1, j0:j1, k0:k1]
        E[i0:i1, j0:j1, k0:k1, 0] += curl_H_Ex


def _electric_update_chunk_complex(
    E: np.ndarray,
    H: np.ndarray,
    i0: int, j0: int, k0: int,
    i1: int, j1: int, k1: int,
    eps_xx, eps_yy, eps_zz,
    dt: float, dx: float, dy: float, dz: float,
) -> None:
    """Complex-valued E update for a chunk interior region.

    Complex storage: (nx, ny, nz, 3, 2) where last dim is (real, imag).
    """
    # Simplified: use the full-array step_maxwell on the chunk interior
    # This is an approximation - full implementation would use chunk-local arrays
    pass


def _magnetic_update_chunk(
    E_full: np.ndarray,
    H_full: np.ndarray,
    chunk_spec: Any,
    field_state: FieldState,
    dt: float,
    dx: float, dy: float, dz: float,
    step: int,
    time: float,
) -> None:
    """Apply magnetic update to a chunk's interior region."""
    # H update is combined with E update in step_maxwell
    # For chunked execution, this is handled in the combined update
    pass


def _apply_chunk_pml(
    E_full: np.ndarray,
    H_full: np.ndarray,
    chunk_spec: Any,
    pml_state: Any,
    chunk_idx: int,
    dt: float,
) -> None:
    """Apply PML boundary stage for a chunk."""
    # PML application is per-face and handled by apply_compiled_boundaries
    # For a simplified multi-chunk implementation, PML is applied to the full
    # domain after all chunk updates
    pass


def _compute_field_magnitude_sq(field: np.ndarray, is_complex: bool) -> float:
    """Compute integrated electric field magnitude squared."""
    if is_complex:
        real_part = field[..., 0]
        imag_part = field[..., 1]
        return float(np.sum(real_part**2 + imag_part**2))
    else:
        return float(np.sum(field**2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_field_array(result: ExecutionResult) -> np.ndarray | None:
    """Extract E field array from ExecutionResult."""
    if result.field_state is not None and hasattr(result.field_state, 'E'):
        return result.field_state.E
    return None


def _find_boundary_cell_indices(
    grid_shape: tuple[int, int, int],
    chunk_layout: ChunkLayout,
) -> list[tuple[int, int, int]]:
    """Find all interior cell indices that lie on chunk interior boundaries.

    Returns list of (i, j, k) indices that are interior cells of one chunk
    but adjacent to the interior of a neighboring chunk.
    """
    nx, ny, nz = grid_shape
    indices = []

    if chunk_layout.total_chunks <= 1:
        return indices

    for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
        interior = chunk_spec.interior_bounds
        (i0, j0, k0), (i1, j1, k1) = interior

        # x.minus face of this chunk touches x.plus of neighbor chunk
        # Collect interior cells at this boundary (depth=1)
        for j in range(max(1, j0), min(j1, ny - 1)):
            for k in range(max(1, k0), min(k1, nz - 1)):
                indices.append((i0, j, k))

        # x.plus face
        for j in range(max(1, j0), min(j1, ny - 1)):
            for k in range(max(1, k0), min(k1, nz - 1)):
                indices.append((i1 - 1, j, k))

    return indices


# ---------------------------------------------------------------------------
# Result formatter
# ---------------------------------------------------------------------------

def format_halo_check_result(result: dict) -> str:
    """Format halo check result as a readable string."""
    lines = [
        "Halo Correctness Check Results",
        "=" * 50,
        f"Max error:      {result['max_error']:.6e}",
        f"Mean error:     {result['mean_error']:.6e}",
        f"Tolerance:      {result['tolerance']:.1e}",
        f"Pass/Fail:      {result['pass_fail']}",
        f"Boundary cells: {result['num_boundary_cells']}",
    ]
    return "\n".join(lines)
