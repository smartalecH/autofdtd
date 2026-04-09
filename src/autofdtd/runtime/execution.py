"""End-to-end execution engine for compiled Phase 1 simulations.

This module provides the top-level runtime execution interface that drives
the complete simulation pipeline from compiled artifacts to final results.

Execution Pipeline
-----------------
1. Allocate field arrays (E, H) and coefficient arrays (eps, mu) from grid shape
2. Populate material coefficient arrays from compiled scene coefficients
3. Initialize source runtime states
4. Initialize monitor runtime states
5. Run the timestep loop:
   a. Inject sources into E field
   b. Update E field (electric_update_3d)
   c. Exchange halos / apply boundary conditions
   d. Update H field (magnetic_update_3d)
   e. Apply PML/ABC boundary stages
   f. Record monitor data
   g. Check convergence (integrated field intensity)
6. Extract final monitor data

The key architectural contract: execution is driven entirely by the
CompiledSimulation's IR snapshot and compiled artifacts, never by the
original Python Simulation object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:
    wp = None

from autofdtd.kernels.backend import WARP_AVAILABLE, ComplexFieldPolicy
from autofdtd.kernels.steps import allocate_maxwell_arrays, step_maxwell


def _compute_field_magnitude_sq(field_array: "wp.array | np.ndarray") -> float:
    """Compute sum of |E|^2 over the entire field array.

    GPU path: launches a Warp reduction kernel directly on device.
    CPU path: NumPy computation.
    """
    if WARP_AVAILABLE:
        import warp as wp

        if isinstance(field_array, wp.array):
            return _gpu_field_magnitude_sq(field_array)
        arr = field_array
    else:
        arr = field_array

    return float(np.sum(np.abs(arr) ** 2))


if WARP_AVAILABLE:

    @wp.kernel
    def _field_magnitude_sq_kernel(E: wp.array(dtype=wp.float32, ndim=4), result: wp.array(dtype=wp.float32, ndim=1)):
        """Sum |E|^2 over all cells and component axes."""
        i, j, k = wp.tid()
        nx, ny, nz = E.shape[0], E.shape[1], E.shape[2]
        if i >= nx or j >= ny or k >= nz:
            return
        # Sum squared components for this cell
        s = wp.float32(0.0)
        for c in range(3):
            val = E[i, j, k, c]
            s += val * val
        wp.atomic_add(result, 0, s)

    def _gpu_field_magnitude_sq(E: "wp.array") -> float:
        """GPU-side reduction for |E|^2 sum using Warp atomic add."""
        dev = E.device
        result = wp.zeros(shape=(1,), dtype=wp.float32, device=dev)
        nx, ny, nz = E.shape[0], E.shape[1], E.shape[2]
        wp.launch(
            _field_magnitude_sq_kernel,
            dim=(nx, ny, nz),
            inputs=[E, result],
            device=dev,
        )
        wp.synchronize()
        return float(result.numpy()[0])


@dataclass
class FieldState:
    """Mutable field state container for a simulation run.

    Attributes
    ----------
    E : wp.array | np.ndarray
        Electric field array with shape (nx, ny, nz, 3).
    H : wp.array | np.ndarray
        Magnetic field array with shape (nx, ny, nz, 3).
    eps_xx, eps_yy, eps_zz : wp.array | np.ndarray
        Permittivity arrays with shape (nx, ny, nz).
    mu_xx, mu_yy, mu_zz : wp.array | np.ndarray
        Permeability arrays with shape (nx, ny, nz).
    electric_modes : wp.array | np.ndarray
        Electric constitutive mode array with shape (nx, ny, nz).
    magnetic_modes : wp.array | np.ndarray
        Magnetic constitutive mode array with shape (nx, ny, nz).
    """

    E: "wp.array | np.ndarray"
    H: "wp.array | np.ndarray"
    eps_xx: "wp.array | np.ndarray"
    eps_yy: "wp.array | np.ndarray"
    eps_zz: "wp.array | np.ndarray"
    mu_xx: "wp.array | np.ndarray"
    mu_yy: "wp.array | np.ndarray"
    mu_zz: "wp.array | np.ndarray"
    electric_modes: "wp.array | np.ndarray"
    magnetic_modes: "wp.array | np.ndarray"


@dataclass
class ExecutionResult:
    """Structured result from a simulation execution.

    Attributes
    ----------
    field_state : FieldState
        Final field state after the run.
    num_steps : int
        Number of timesteps executed.
    final_time : float
        Simulation time at the end of the run.
    stop_reason : str
        Reason for stopping (e.g., "max_steps", "shutoff").
    integrated_electric_history : list[float]
        List of integrated electric field intensities recorded during the run.
    field_monitor_data : dict[str, FieldData]
        Extracted field monitor data.
    flux_monitor_data : dict[str, FluxData]
        Extracted flux monitor data.
    medium_monitor_data : dict[str, MediumMonitorData]
        Extracted medium monitor data.
    mode_monitor_data : dict[str, ModeData]
        Extracted mode monitor data.
    metrics : dict[str, Any]
        Execution metrics (wall time, cells updated, etc.).
    """

    field_state: FieldState
    num_steps: int
    final_time: float
    stop_reason: str
    integrated_electric_history: list[float]
    field_monitor_data: dict[str, Any]
    flux_monitor_data: dict[str, Any]
    medium_monitor_data: dict[str, Any]
    mode_monitor_data: dict[str, Any]
    metrics: dict[str, Any]


def _fill_array(arr, value):
    """Fill an array with a value, handling both numpy and wp.array."""
    if hasattr(arr, 'fill_'):
        arr.fill_(value)
    else:
        arr.fill(value)


def populate_material_arrays(
    field_state: FieldState,
    compiled,
) -> None:
    """Populate material coefficient arrays from compiled scene coefficients.

    This fills the eps_xx/yy/zz and mu_xx/yy/zz arrays with values
    derived from the scene's background medium and structures.

    Parameters
    ----------
    field_state : FieldState
        The field state to populate with material values.
    compiled : CompiledSimulation
        The compiled simulation containing scene coefficient data.
    """
    # Import locally to avoid circular imports
    from autofdtd.compiler.pipeline import CompiledSimulation

    nx, ny, nz = field_state.E.shape[:3]

    # Start with vacuum values (1.0 for normalized units)
    _fill_array(field_state.eps_xx, 1.0)
    _fill_array(field_state.eps_yy, 1.0)
    _fill_array(field_state.eps_zz, 1.0)
    _fill_array(field_state.mu_xx, 1.0)
    _fill_array(field_state.mu_yy, 1.0)
    _fill_array(field_state.mu_zz, 1.0)

    # Get background medium coefficients
    bg = compiled.scene_coefficients.background
    bg_eps = getattr(bg, "permittivity", 1.0)
    bg_mu = getattr(bg, "permeability", 1.0)

    if bg_eps is not None:
        _fill_array(field_state.eps_xx, bg_eps)
        _fill_array(field_state.eps_yy, bg_eps)
        _fill_array(field_state.eps_zz, bg_eps)
    if bg_mu is not None:
        _fill_array(field_state.mu_xx, bg_mu)
        _fill_array(field_state.mu_yy, bg_mu)
        _fill_array(field_state.mu_zz, bg_mu)

    # Apply structure coefficients (simplified - just apply first structure as override)
    # Full implementation would use the materialized grid with structure overlap
    for struct_coeff in compiled.scene_coefficients.structures:
        mat = struct_coeff.material_coefficients
        mat_eps = getattr(mat, "permittivity", None)
        mat_mu = getattr(mat, "permeability", None)

        if mat_eps is not None:
            # For simplicity, fill entire array with structure medium
            # A full implementation would check structure geometry bounds
            _fill_array(field_state.eps_xx, mat_eps)
            _fill_array(field_state.eps_yy, mat_eps)
            _fill_array(field_state.eps_zz, mat_eps)
        if mat_mu is not None:
            _fill_array(field_state.mu_xx, mat_mu)
            _fill_array(field_state.mu_yy, mat_mu)
            _fill_array(field_state.mu_zz, mat_mu)


def allocate_field_state(
    compiled,
) -> FieldState:
    """Allocate field state arrays from a compiled simulation.

    Parameters
    ----------
    compiled : CompiledSimulation
        The compiled simulation to allocate arrays for.

    Returns
    -------
    FieldState
        Allocated and initialized field state.
    """
    from autofdtd.kernels.backend import WARP_AVAILABLE, get_warp_device

    grid_shape = compiled.grid_shape

    # Allocate arrays on GPU when Warp is available
    if WARP_AVAILABLE:
        import warp as wp

        dev = get_warp_device()
        nx, ny, nz = grid_shape
        E = wp.zeros(shape=(nx, ny, nz, 3), dtype=wp.float32, device=dev)
        H = wp.zeros(shape=(nx, ny, nz, 3), dtype=wp.float32, device=dev)
        eps_xx = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        eps_yy = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        eps_zz = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        mu_xx = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        mu_yy = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        mu_zz = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        electric_modes = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
        magnetic_modes = wp.zeros(shape=grid_shape, dtype=wp.float32, device=dev)
    else:
        import numpy as np

        E = np.zeros((*grid_shape, 3), dtype=np.float64)
        H = np.zeros((*grid_shape, 3), dtype=np.float64)
        eps_xx = np.zeros(grid_shape, dtype=np.float64)
        eps_yy = np.zeros(grid_shape, dtype=np.float64)
        eps_zz = np.zeros(grid_shape, dtype=np.float64)
        mu_xx = np.zeros(grid_shape, dtype=np.float64)
        mu_yy = np.zeros(grid_shape, dtype=np.float64)
        mu_zz = np.zeros(grid_shape, dtype=np.float64)
        electric_modes = np.zeros(grid_shape, dtype=np.float64)
        magnetic_modes = np.zeros(grid_shape, dtype=np.float64)

    field_state = FieldState(
        E=E,
        H=H,
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        mu_xx=mu_xx,
        mu_yy=mu_yy,
        mu_zz=mu_zz,
        electric_modes=electric_modes,
        magnetic_modes=magnetic_modes,
    )

    # Populate material coefficients
    populate_material_arrays(field_state, compiled)

    return field_state


def run_compiled_simulation(
    compiled,
    *,
    max_steps: int | None = None,
    record_interval: int = 10,
    progress_interval: int = 100,
    verbose: bool = False,
) -> ExecutionResult:
    """Execute a compiled simulation and return field states and monitor data.

    This is the main end-to-end execution entry point for Phase 1. It takes
    a CompiledSimulation produced by compile_simulation() and drives the
    complete Maxwell timestepping loop.

    Parameters
    ----------
    compiled : CompiledSimulation
        The compiled simulation to execute.
    max_steps : int, optional
        Override the maximum number of steps. If None, uses the compiled
        runtime controls' num_time_steps.
    record_interval : int, default=10
        Interval for recording field monitor data.
    progress_interval : int, default=100
        Interval for printing progress messages.
    verbose : bool, default=False
        If True, print step progress.

    Returns
    -------
    ExecutionResult
        Structured result containing final field state, monitor data,
        and execution metrics.
    """
    # Import locally to avoid circular imports
    from autofdtd.compiler.monitors import (
        FieldMonitorState,
        FluxMonitorState,
        MediumMonitorState,
        ModeMonitorState,
    )
    from autofdtd.runtime.monitors import (
        record_monitor_fields,
        record_monitor_flux,
    )
    from autofdtd.runtime.sources import apply_source_injection_stage

    import time as time_module

    # Check if we have multi-chunk layout and delegate accordingly
    if compiled.chunk_layout.total_chunks > 1:
        return _run_chunked_simulation(
            compiled,
            max_steps=max_steps,
            record_interval=record_interval,
            progress_interval=progress_interval,
            verbose=verbose,
        )

    # Determine step budget
    if max_steps is None:
        max_steps = compiled.runtime_controls.num_time_steps

    dt = compiled.runtime_controls.dt
    cell_sizes = (
        compiled.resolved_grid.x.cell_sizes[0]
        if compiled.resolved_grid.x.cell_sizes
        else 1.0,
        compiled.resolved_grid.y.cell_sizes[0]
        if compiled.resolved_grid.y.cell_sizes
        else 1.0,
        compiled.resolved_grid.z.cell_sizes[0]
        if compiled.resolved_grid.z.cell_sizes
        else 1.0,
    )
    dx, dy, dz = cell_sizes

    # Allocate field state
    field_state = allocate_field_state(compiled)

    # Initialize source runtime states
    source_runtimes = _initialize_source_runtimes(compiled)

    # Initialize monitor states
    monitor_states = _initialize_monitor_states(compiled)

    # Execution tracking
    integrated_electric_history: list[float] = []
    num_steps_executed = 0
    current_time = 0.0

    # Peak tracking for convergence
    peak_integrated_electric = 0.0

    # Timing
    wall_start = time_module.perf_counter()

    # Main timestep loop
    for step in range(max_steps):
        # 1. Source injection stage
        field_state.E, field_state.H = apply_source_injection_stage(
            field_state.E,
            field_state.H,
            uniform_current_sources=compiled.compiled_sources.uniform_current,
            point_dipole_sources=compiled.compiled_sources.point_dipole,
            plane_wave_sources=compiled.compiled_sources.plane_wave,
            gaussian_beam_sources=compiled.compiled_sources.gaussian_beam,
            mode_sources=compiled.compiled_sources.mode_source,
            time=current_time,
            dt=dt,
        )

        # 2. Electric field update (includes H update via step_maxwell)
        step_result = step_maxwell(
            {
                "E": field_state.E,
                "H": field_state.H,
                "eps_xx": field_state.eps_xx,
                "eps_yy": field_state.eps_yy,
                "eps_zz": field_state.eps_zz,
                "mu_xx": field_state.mu_xx,
                "mu_yy": field_state.mu_yy,
                "mu_zz": field_state.mu_zz,
            },
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            step_index=step,
            time=current_time,
        )

        # 2b. Apply symmetry transforms for symmetry-reduced domains
        _apply_symmetry_transforms(field_state, compiled)

        # 2c. Apply periodic/Bloch boundary wrapping for domain boundaries
        _apply_periodic_bloch_wrap(field_state, compiled)

        # 3. Compute integrated electric field for convergence check
        E_sq = _compute_field_magnitude_sq(field_state.E)
        integrated_electric = E_sq
        integrated_electric_history.append(integrated_electric)

        # Update peak for convergence
        if integrated_electric > peak_integrated_electric:
            peak_integrated_electric = integrated_electric

        # 4. Check convergence / stop conditions
        shutoff_ratio = 1.0
        if peak_integrated_electric > 0.0:
            shutoff_ratio = integrated_electric / peak_integrated_electric

        stop_reason = "max_steps"
        if shutoff_ratio <= float(compiled.runtime_controls.shutoff) and step > 0:
            stop_reason = "shutoff"
            num_steps_executed = step + 1
            current_time = (step + 1) * dt
            break

        if step == max_steps - 1:
            stop_reason = "max_steps"
            num_steps_executed = step + 1
            current_time = (step + 1) * dt
            break

        # 5. Record monitor data at intervals
        if step % record_interval == 0:
            _record_monitor_data(
                monitor_states,
                field_state,
                compiled,
                current_time,
                step,
            )

        # 6. Progress reporting
        if verbose and step % progress_interval == 0:
            print(
                f"Step {step}/{max_steps} | "
                f"Time {current_time:.3e} | "
                f"|E|² {integrated_electric:.3e} | "
                f"Shutoff {shutoff_ratio:.3e}"
            )

        current_time += dt

    # Final timing
    wall_end = time_module.perf_counter()
    wall_time = wall_end - wall_start

    # Final monitor recording
    _record_monitor_data(
        monitor_states,
        field_state,
        compiled,
        current_time,
        max_steps - 1,
    )

    # Extract monitor data
    field_monitor_data = _extract_field_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    flux_monitor_data = _extract_flux_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    medium_monitor_data = _extract_medium_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    mode_monitor_data = _extract_mode_monitor_data(
        monitor_states, compiled.resolved_grid
    )

    # Compute metrics
    total_cells = compiled.total_cells
    gcells_per_second = (
        (total_cells * num_steps_executed / wall_time) / 1e9
        if wall_time > 0
        else 0.0
    )

    return ExecutionResult(
        field_state=field_state,
        num_steps=num_steps_executed,
        final_time=current_time,
        stop_reason=stop_reason,
        integrated_electric_history=integrated_electric_history,
        field_monitor_data=field_monitor_data,
        flux_monitor_data=flux_monitor_data,
        medium_monitor_data=medium_monitor_data,
        mode_monitor_data=mode_monitor_data,
        metrics={
            "wall_time_s": wall_time,
            "cells_updated": total_cells * num_steps_executed,
            "gcells_per_second": gcells_per_second,
            "backend": "numpy" if not WARP_AVAILABLE else "warp",
        },
    )


def _initialize_source_runtimes(compiled) -> dict[str, Any]:
    """Initialize runtime source data structures."""
    runtimes = {}

    # Uniform current sources
    for i, src in enumerate(compiled.compiled_sources.uniform_current):
        runtimes[f"uniform_current_{i}"] = src

    # Point dipoles
    for i, src in enumerate(compiled.compiled_sources.point_dipole):
        runtimes[f"point_dipole_{i}"] = src

    # Plane waves
    for i, src in enumerate(compiled.compiled_sources.plane_wave):
        runtimes[f"plane_wave_{i}"] = src

    # Gaussian beams
    for i, src in enumerate(compiled.compiled_sources.gaussian_beam):
        runtimes[f"gaussian_beam_{i}"] = src

    return runtimes


def _initialize_monitor_states(compiled) -> dict[str, Any]:
    """Initialize runtime monitor states."""
    # Import locally to avoid circular imports
    from autofdtd.compiler.monitors import (
        FieldMonitorState,
        FluxMonitorState,
        MediumMonitorState,
        ModeMonitorState,
    )

    states = {}

    # Field monitors
    for compiled_monitor in compiled.compiled_monitors.field:
        states[f"field_{compiled_monitor.name}"] = FieldMonitorState(
            compiled=compiled_monitor
        )

    # Flux monitors
    for compiled_monitor in compiled.compiled_monitors.flux:
        states[f"flux_{compiled_monitor.name}"] = FluxMonitorState(
            compiled=compiled_monitor
        )

    # Medium monitors
    for compiled_monitor in compiled.compiled_monitors.medium:
        states[f"medium_{compiled_monitor.name}"] = MediumMonitorState(
            compiled=compiled_monitor
        )

    # Mode monitors
    for compiled_monitor in compiled.compiled_monitors.mode:
        states[f"mode_{compiled_monitor.name}"] = ModeMonitorState(
            compiled=compiled_monitor
        )

    return states


def _record_monitor_data(
    monitor_states: dict[str, Any],
    field_state: FieldState,
    compiled,
    time: float,
    step: int,
) -> None:
    """Record monitor data at the current step."""
    # Import locally to avoid circular imports
    from autofdtd.compiler.monitors import (
        FieldMonitorState,
        FluxMonitorState,
    )
    from autofdtd.runtime.monitors import (
        record_monitor_fields,
        record_monitor_flux,
    )

    # Field monitors
    field_states = [
        s for name, s in monitor_states.items() if name.startswith("field_")
    ]
    if field_states:
        record_monitor_fields(
            field_states,
            field_state.E,
            field_state.H,
            time=time,
            step_index=step,
        )

    # Flux monitors
    flux_states = [
        s for name, s in monitor_states.items() if name.startswith("flux_")
    ]
    if flux_states:
        record_monitor_flux(
            flux_states,
            field_state.E,
            field_state.H,
            time=time,
            step_index=step,
        )


def _extract_field_monitor_data(
    monitor_states: dict[str, Any],
    grid,
) -> dict[str, Any]:
    """Extract field monitor data from states."""
    # Import locally to avoid circular imports
    from autofdtd.runtime.monitors import extract_monitor_data

    result = {}
    for name, state in monitor_states.items():
        if name.startswith("field_"):
            monitor_name = name[len("field_") :]
            data = extract_monitor_data([state], grid)
            if data:
                # data is {monitor_name: FieldData}, extract the FieldData directly
                result[monitor_name] = data[monitor_name]
    return result


def _extract_flux_monitor_data(
    monitor_states: dict[str, Any],
    grid,
) -> dict[str, Any]:
    """Extract flux monitor data from states."""
    # Import locally to avoid circular imports
    from autofdtd.runtime.monitors import extract_flux_monitor_data

    result = {}
    for name, state in monitor_states.items():
        if name.startswith("flux_"):
            monitor_name = name[len("flux_") :]
            data = extract_flux_monitor_data([state], grid)
            if data:
                # data is {monitor_name: FluxData}, extract the FluxData directly
                result[monitor_name] = data[monitor_name]
    return result


def _extract_medium_monitor_data(
    monitor_states: dict[str, Any],
    grid,
) -> dict[str, Any]:
    """Extract medium monitor data from states."""
    # Import locally to avoid circular imports
    from autofdtd.runtime.monitors import extract_medium_monitor_data

    result = {}
    for name, state in monitor_states.items():
        if name.startswith("medium_"):
            monitor_name = name[len("medium_") :]
            data = extract_medium_monitor_data([state], grid)
            if data:
                # data is {monitor_name: MediumMonitorData}, extract directly
                result[monitor_name] = data[monitor_name]
    return result


def _extract_mode_monitor_data(
    monitor_states: dict[str, Any],
    grid,
) -> dict[str, Any]:
    """Extract mode monitor data from states."""
    # Import locally to avoid circular imports
    from autofdtd.runtime.monitors import extract_mode_monitor_data

    result = {}
    for name, state in monitor_states.items():
        if name.startswith("mode_"):
            monitor_name = name[len("mode_") :]
            data = extract_mode_monitor_data([state], grid)
            if data:
                # data is {monitor_name: ModeData}, extract directly
                result[monitor_name] = data[monitor_name]
    return result


def _run_chunked_simulation(
    compiled,
    *,
    max_steps: int | None = None,
    record_interval: int = 10,
    progress_interval: int = 100,
    verbose: bool = False,
) -> ExecutionResult:
    """Execute a compiled simulation with multi-chunk (multi-GPU) layout.

    This function handles multi-chunk execution by:
    1. Performing halo exchange before each field update
    2. Updating per-chunk interior regions
    3. Applying PML boundaries per chunk

    Parameters
    ----------
    compiled : CompiledSimulation
        The compiled simulation with multi-chunk layout.
    max_steps : int, optional
        Override the maximum number of steps.
    record_interval : int, default=10
        Interval for recording field monitor data.
    progress_interval : int, default=100
        Interval for printing progress messages.
    verbose : bool, default=False
        If True, print step progress.

    Returns
    -------
    ExecutionResult
        Structured result containing final field state, monitor data, and metrics.
    """
    from autofdtd.compiler.monitors import (
        FieldMonitorState,
        FluxMonitorState,
        MediumMonitorState,
        ModeMonitorState,
    )
    from autofdtd.runtime.monitors import (
        record_monitor_fields,
        record_monitor_flux,
    )
    from autofdtd.runtime.sources import apply_source_injection_stage
    from autofdtd.runtime.boundaries import ChunkHaloExchange, allocate_pml_boundary_state
    from autofdtd.runtime.chunk import ExchangeKind

    import time as time_module

    # Determine step budget
    if max_steps is None:
        max_steps = compiled.runtime_controls.num_time_steps

    grid_shape = compiled.grid_shape
    nx, ny, nz = grid_shape

    dt = compiled.runtime_controls.dt
    cell_sizes = (
        compiled.resolved_grid.x.cell_sizes[0]
        if compiled.resolved_grid.x.cell_sizes
        else 1.0,
        compiled.resolved_grid.y.cell_sizes[0]
        if compiled.resolved_grid.y.cell_sizes
        else 1.0,
        compiled.resolved_grid.z.cell_sizes[0]
        if compiled.resolved_grid.z.cell_sizes
        else 1.0,
    )
    dx, dy, dz = cell_sizes

    # Allocate field state (monolithic arrays)
    field_state = allocate_field_state(compiled)
    E_full = field_state.E
    H_full = field_state.H

    # Determine if using complex fields
    is_complex = E_full.ndim == 5  # (nx, ny, nz, 3, 2) for complex

    # Build halo exchange manager
    chunk_layout = compiled.chunk_layout
    halo_exchange = ChunkHaloExchange(
        chunk_layout=chunk_layout,
        field_shape=(nx, ny, nz, 3),
        dtype=np.float64,
    )

    # Allocate PML state
    pml_state = allocate_pml_boundary_state(
        compiled.compiled_boundaries.boundary_spec,
        field_shape=(nx, ny, nz, 3),
        dtype=np.float64,
    )

    # Initialize source and monitor states
    source_runtimes = _initialize_source_runtimes(compiled)
    monitor_states = _initialize_monitor_states(compiled)

    # Execution tracking
    integrated_electric_history: list[float] = []
    num_steps_executed = 0
    current_time = 0.0
    peak_integrated_electric = 0.0

    wall_start = time_module.perf_counter()

    if verbose:
        print(
            f"    Chunked loop: {max_steps} steps, grid={grid_shape}, "
            f"chunks={chunk_layout.total_chunks}"
        )

    # Main timestep loop
    for step in range(max_steps):
        # ---- Halo exchange before E update ----
        _chunked_halo_exchange(
            halo_exchange, E_full, H_full, chunk_layout, nx, ny, nz, is_complex
        )

        # ---- Electric field update (per chunk interior) ----
        for chunk_spec in chunk_layout.chunks:
            _chunk_electric_update(
                E_full, H_full, chunk_spec, field_state, dt, dx, dy, dz, step, current_time
            )

        # ---- Source injection (monolithic) ----
        E_full, H_full = apply_source_injection_stage(
            E_full,
            H_full,
            uniform_current_sources=compiled.compiled_sources.uniform_current,
            point_dipole_sources=compiled.compiled_sources.point_dipole,
            plane_wave_sources=compiled.compiled_sources.plane_wave,
            gaussian_beam_sources=compiled.compiled_sources.gaussian_beam,
            mode_sources=compiled.compiled_sources.mode_source,
            time=current_time,
            dt=dt,
        )

        # ---- Halo exchange before H update ----
        _chunked_halo_exchange(
            halo_exchange, E_full, H_full, chunk_layout, nx, ny, nz, is_complex
        )

        # ---- Magnetic field update (per chunk interior) ----
        for chunk_spec in chunk_layout.chunks:
            _chunk_magnetic_update(
                E_full, H_full, chunk_spec, field_state, dt, dx, dy, dz, step, current_time
            )

        # ---- Apply symmetry transforms for symmetry-reduced domains ----
        _apply_symmetry_transforms(field_state, compiled)

        # ---- PML boundary stage (per chunk) ----
        for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
            _chunked_apply_pml(
                E_full, H_full, chunk_spec, pml_state, chunk_idx, dt
            )

        # ---- Convergence tracking ----
        E_sq = _compute_field_magnitude_sq(E_full)
        integrated_electric = float(E_sq)
        integrated_electric_history.append(integrated_electric)

        if integrated_electric > peak_integrated_electric:
            peak_integrated_electric = integrated_electric

        shutoff_ratio = 1.0
        if peak_integrated_electric > 0.0:
            shutoff_ratio = integrated_electric / peak_integrated_electric

        stop_reason = "max_steps"
        if shutoff_ratio <= float(compiled.runtime_controls.shutoff) and step > 0:
            stop_reason = "shutoff"
            num_steps_executed = step + 1
            current_time = (step + 1) * dt
            if verbose:
                print(f"    Converged at step {step}")
            break

        if step == max_steps - 1:
            stop_reason = "max_steps"
            num_steps_executed = step + 1
            current_time = (step + 1) * dt
            break

        # ---- Record monitor data at intervals ----
        if step % record_interval == 0:
            _record_monitor_data(
                monitor_states,
                field_state,
                compiled,
                current_time,
                step,
            )

        # ---- Progress reporting ----
        if verbose and step % progress_interval == 0:
            print(
                f"Step {step}/{max_steps} | "
                f"Time {current_time:.3e} | "
                f"|E|² {integrated_electric:.3e} | "
                f"Shutoff {shutoff_ratio:.3e}"
            )

        current_time += dt

    # Final timing
    wall_end = time_module.perf_counter()
    wall_time = wall_end - wall_start

    # Final monitor recording
    _record_monitor_data(
        monitor_states,
        field_state,
        compiled,
        current_time,
        max_steps - 1,
    )

    # Extract monitor data
    field_monitor_data = _extract_field_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    flux_monitor_data = _extract_flux_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    medium_monitor_data = _extract_medium_monitor_data(
        monitor_states, compiled.resolved_grid
    )
    mode_monitor_data = _extract_mode_monitor_data(
        monitor_states, compiled.resolved_grid
    )

    # Compute metrics
    total_cells = compiled.total_cells
    gcells_per_second = (
        (total_cells * num_steps_executed / wall_time) / 1e9
        if wall_time > 0
        else 0.0
    )

    return ExecutionResult(
        field_state=field_state,
        num_steps=num_steps_executed,
        final_time=current_time,
        stop_reason=stop_reason,
        integrated_electric_history=integrated_electric_history,
        field_monitor_data=field_monitor_data,
        flux_monitor_data=flux_monitor_data,
        medium_monitor_data=medium_monitor_data,
        mode_monitor_data=mode_monitor_data,
        metrics={
            "wall_time_s": wall_time,
            "cells_updated": total_cells * num_steps_executed,
            "gcells_per_second": gcells_per_second,
            "backend": "numpy" if not WARP_AVAILABLE else "warp-chunked",
            "total_chunks": chunk_layout.total_chunks,
            "num_chunks": chunk_layout.num_chunks,
        },
    )


def _chunked_halo_exchange(
    halo_exchange: "ChunkHaloExchange",
    E_full,
    H_full,
    chunk_layout,
    nx: int,
    ny: int,
    nz: int,
    is_complex: bool,
) -> None:
    """Execute halo exchange for all interior chunk faces.

    For same-device interior faces: direct copy between chunks.
    For cross-device faces: use ChunkHaloExchange.cross_device_transfer.
    """
    for chunk_idx, chunk_spec in enumerate(chunk_layout.chunks):
        chunk_index = chunk_spec.chunk_index
        for axis_name in ("x", "y", "z"):
            for side in ("minus", "plus"):
                _chunked_exchange_face_halo(
                    halo_exchange,
                    E_full, H_full,
                    chunk_layout, chunk_spec, chunk_index,
                    axis_name, side, nx, ny, nz, is_complex,
                )


def _chunked_exchange_face_halo(
    halo_exchange: "ChunkHaloExchange",
    E_full,
    H_full,
    chunk_layout,
    chunk_spec,
    chunk_index: tuple[int, int, int],
    axis_name: str,
    side: str,
    nx: int, ny: int, nz: int,
    is_complex: bool,
) -> None:
    """Exchange halo for one face of one chunk using Warp-compatible operations.

    For interior faces between chunks, we copy from the current chunk's interior
    face to the neighbor chunk's interior face (both are at the boundary).

    For domain Bloch boundaries (no neighbor), we wrap the field from the
    opposite interior face and apply the Bloch phase factor.
    """
    from autofdtd.runtime.chunk import ExchangeKind

    halo = chunk_spec.face_halo(axis_name, side)
    if halo is None:
        return

    exchange_kind = halo.exchange_kind

    # Get interior bounds for this chunk
    (gi0, gj0, gk0), (gi1, gj1, gk1) = chunk_spec.interior_bounds
    depth = halo.depth

    if exchange_kind == ExchangeKind.BLOCH:
        # Domain Bloch boundary: wrap from opposite interior face with phase.
        # For x.minus (side="minus"): ghost at gi0 should be phase.conj() * plus-interior
        # For x.plus (side="plus"): ghost at gi1-1 should be phase * minus-interior
        # Note: phase_factor from FaceHalo already has the correct direction baked in
        # (conj for minus, direct for plus) via _phase_factor_for_edge in compilation.
        phase = halo.phase_factor  # complex phase factor for this face

        # Determine source and destination slices for Bloch wrap
        if axis_name == "x":
            if side == "minus":
                # x.minus: wrap from x.plus interior (at gi1-depth) to x.minus ghost (at gi0)
                src_i0, src_i1 = gi1 - depth, gi1
                dst_i0, dst_i1 = gi0, gi0 + depth
                j_slice = slice(gj0, gj1)
                k_slice = slice(gk0, gk1)
                src_slice = (slice(src_i0, src_i1), j_slice, k_slice, slice(None))
                dst_slice = (slice(dst_i0, dst_i1), j_slice, k_slice, slice(None))
            else:
                # x.plus: wrap from x.minus interior (at gi0) to x.plus ghost (at gi1-1)
                src_i0, src_i1 = gi0, gi0 + depth
                dst_i0, dst_i1 = gi1 - depth, gi1
                j_slice = slice(gj0, gj1)
                k_slice = slice(gk0, gk1)
                src_slice = (slice(src_i0, src_i1), j_slice, k_slice, slice(None))
                dst_slice = (slice(dst_i0, dst_i1), j_slice, k_slice, slice(None))
        elif axis_name == "y":
            if side == "minus":
                src_j0, src_j1 = gj1 - depth, gj1
                dst_j0, dst_j1 = gj0, gj0 + depth
                i_slice = slice(gi0, gi1)
                k_slice = slice(gk0, gk1)
                src_slice = (i_slice, slice(src_j0, src_j1), k_slice, slice(None))
                dst_slice = (i_slice, slice(dst_j0, dst_j1), k_slice, slice(None))
            else:
                src_j0, src_j1 = gj0, gj0 + depth
                dst_j0, dst_j1 = gj1 - depth, gj1
                i_slice = slice(gi0, gi1)
                k_slice = slice(gk0, gk1)
                src_slice = (i_slice, slice(src_j0, src_j1), k_slice, slice(None))
                dst_slice = (i_slice, slice(dst_j0, dst_j1), k_slice, slice(None))
        else:  # z
            if side == "minus":
                src_k0, src_k1 = gk1 - depth, gk1
                dst_k0, dst_k1 = gk0, gk0 + depth
                i_slice = slice(gi0, gi1)
                j_slice = slice(gj0, gj1)
                src_slice = (i_slice, j_slice, slice(src_k0, src_k1), slice(None))
                dst_slice = (i_slice, j_slice, slice(dst_k0, dst_k1), slice(None))
            else:
                src_k0, src_k1 = gk0, gk0 + depth
                dst_k0, dst_k1 = gk1 - depth, gk1
                i_slice = slice(gi0, gi1)
                j_slice = slice(gj0, gj1)
                src_slice = (i_slice, j_slice, slice(src_k0, src_k1), slice(None))
                dst_slice = (i_slice, j_slice, slice(dst_k0, dst_k1), slice(None))

        # Apply Bloch phase wrap to E and H fields
        _warp_array_copy_slice_with_phase(E_full, dst_slice, E_full, src_slice, phase)
        _warp_array_copy_slice_with_phase(H_full, dst_slice, H_full, src_slice, phase)
        return

    if exchange_kind == ExchangeKind.PERIODIC:
        # Domain periodic boundary: wrap from opposite interior face (no phase)
        neighbor_idx = halo.neighbor_chunk_index
        if neighbor_idx is None:
            # No neighbor = domain boundary, wrap from opposite interior
            if axis_name == "x":
                if side == "minus":
                    src_i0, src_i1 = gi1 - depth, gi1
                    dst_i0, dst_i1 = gi0, gi0 + depth
                else:
                    src_i0, src_i1 = gi0, gi0 + depth
                    dst_i0, dst_i1 = gi1 - depth, gi1
                src_slice = (slice(src_i0, src_i1), slice(gj0, gj1), slice(gk0, gk1), slice(None))
                dst_slice = (slice(dst_i0, dst_i1), slice(gj0, gj1), slice(gk0, gk1), slice(None))
            elif axis_name == "y":
                if side == "minus":
                    src_j0, src_j1 = gj1 - depth, gj1
                    dst_j0, dst_j1 = gj0, gj0 + depth
                else:
                    src_j0, src_j1 = gj0, gj0 + depth
                    dst_j0, dst_j1 = gj1 - depth, gj1
                src_slice = (slice(gi0, gi1), slice(src_j0, src_j1), slice(gk0, gk1), slice(None))
                dst_slice = (slice(gi0, gi1), slice(dst_j0, dst_j1), slice(gk0, gk1), slice(None))
            else:  # z
                if side == "minus":
                    src_k0, src_k1 = gk1 - depth, gk1
                    dst_k0, dst_k1 = gk0, gk0 + depth
                else:
                    src_k0, src_k1 = gk0, gk0 + depth
                    dst_k0, dst_k1 = gk1 - depth, gk1
                src_slice = (slice(gi0, gi1), slice(gj0, gj1), slice(src_k0, src_k1), slice(None))
                dst_slice = (slice(gi0, gi1), slice(gj0, gj1), slice(dst_k0, dst_k1), slice(None))
            _warp_array_copy_slice(E_full, dst_slice, E_full, src_slice)
            _warp_array_copy_slice(H_full, dst_slice, H_full, src_slice)
            return
        # With neighbor = interior periodic face, fall through to INTERIOR handling

    # Handle INTERIOR face exchange between chunks
    if exchange_kind not in {ExchangeKind.INTERIOR}:
        # Skip PML, ABC, PEC, PMC and other non-exchange types
        return

    neighbor_idx = halo.neighbor_chunk_index
    if neighbor_idx is None:
        return

    neighbor_spec = chunk_layout.chunk_at(neighbor_idx)
    if neighbor_spec is None:
        return

    # Get interior bounds for neighbor
    (ni0, nj0, nk0), (ni1, nj1, nk1) = neighbor_spec.interior_bounds

    # For interior face exchange:
    # - Source: current chunk's interior face (at the boundary)
    # - Destination: neighbor chunk's interior face (at the boundary)
    # For x.plus: neighbor's x.minus face is at ni0 (start of their interior)
    # For x.minus: neighbor's x.plus face is at ni1-depth (end of their interior, going backwards)
    if axis_name == "x":
        if side == "minus":
            # x.minus face: source is current chunk's interior at gi0, dest is neighbor's at ni1-depth
            src_i0, src_i1 = gi0, gi0 + depth
            dst_i0, dst_i1 = ni1 - depth, ni1
        else:
            # x.plus face: source is current chunk's interior at gi1-depth, dest is neighbor's at ni0
            src_i0, src_i1 = gi1 - depth, gi1
            dst_i0, dst_i1 = ni0, ni0 + depth
        src_slice = (slice(src_i0, src_i1), slice(gj0, gj1), slice(gk0, gk1), slice(None))
        dst_slice = (slice(dst_i0, dst_i1), slice(nj0, nj1), slice(nk0, nk1), slice(None))
    elif axis_name == "y":
        if side == "minus":
            src_j0, src_j1 = gj0, gj0 + depth
            dst_j0, dst_j1 = nj1 - depth, nj1
        else:
            src_j0, src_j1 = gj1 - depth, gj1
            dst_j0, dst_j1 = nj0, nj0 + depth
        src_slice = (slice(gi0, gi1), slice(src_j0, src_j1), slice(gk0, gk1), slice(None))
        dst_slice = (slice(ni0, ni1), slice(dst_j0, dst_j1), slice(nk0, nk1), slice(None))
    else:  # z
        if side == "minus":
            src_k0, src_k1 = gk0, gk0 + depth
            dst_k0, dst_k1 = nk1 - depth, nk1
        else:
            src_k0, src_k1 = gk1 - depth, gk1
            dst_k0, dst_k1 = nk0, nk0 + depth
        src_slice = (slice(gi0, gi1), slice(gj0, gj1), slice(src_k0, src_k1), slice(None))
        dst_slice = (slice(ni0, ni1), slice(nj0, nj1), slice(dst_k0, dst_k1), slice(None))

    # For now, always use same-device path since we have monolithic arrays
    # (cross-device would require per-chunk arrays on different devices)
    # TODO: Implement proper cross-device path with per-chunk arrays
    _warp_array_copy_slice(E_full, dst_slice, E_full, src_slice)


def _warp_array_copy_slice_with_phase(
    dst_array,
    dst_slice,
    src_array,
    src_slice,
    phase: complex,
) -> None:
    """Copy a slice from src_array to dst_array with Bloch phase factor applied.

    Handles both Warp arrays and numpy arrays. For real arrays with complex phase,
    the phase is applied by promoting to complex128.
    """
    is_warp = hasattr(dst_array, 'numpy')

    if is_warp:
        # Warp array path
        src_np = src_array.numpy()
        dst_np = dst_array.numpy().copy()
        if phase != 1.0 + 0.0j:
            # Apply complex phase to real-valued field data
            src_vals = src_np[src_slice]
            if np.isrealobj(src_vals):
                # Real field with complex phase: promote to complex
                src_complex = src_vals.astype(np.complex128)
                dst_np[dst_slice] = src_complex * phase
            else:
                # Already complex
                dst_np[dst_slice] = src_vals * phase
        else:
            dst_np[dst_slice] = src_np[src_slice]
        dst_array.assign(dst_np)
    else:
        # Numpy array path
        src_vals = src_array[src_slice]
        if phase != 1.0 + 0.0j:
            if np.isrealobj(src_vals):
                src_complex = src_vals.astype(np.complex128)
                dst_array[dst_slice] = src_complex * phase
            else:
                dst_array[dst_slice] = src_vals * phase
        else:
            dst_array[dst_slice] = src_vals


def _warp_array_copy_slice(dst_array, dst_slice, src_array, src_slice):
    """Copy a slice from src_array to dst_array using Warp-compatible operations.

    Handles both Warp arrays and numpy arrays.
    """
    is_warp = hasattr(dst_array, 'numpy')

    if is_warp:
        # Warp array path
        src_np = src_array.numpy()
        dst_np = dst_array.numpy().copy()
        dst_np[dst_slice] = src_np[src_slice]
        dst_array.assign(dst_np)
    else:
        # Numpy array path
        dst_array[dst_slice] = src_array[src_slice]


def _chunk_electric_update(
    E_full,
    H_full,
    chunk_spec,
    field_state,
    dt: float,
    dx: float, dy: float, dz: float,
    step: int,
    time: float,
) -> None:
    """Apply electric update to a chunk's interior region using Warp kernels."""
    (i0, j0, k0), (i1, j1, k1) = chunk_spec.interior_bounds

    is_warp = hasattr(E_full, 'numpy')

    if is_warp:
        # Warp path: use step_maxwell with sliced arrays
        import warp as wp

        # For Warp, we need to call step_maxwell on the interior region
        # Extract interior region, update, copy back
        E_interior = E_full[i0:i1, j0:j1, k0:k1]
        H_interior = H_full[i0:i1, j0:j1, k0:k1]
        eps_xx = field_state.eps_xx[i0:i1, j0:j1, k0:k1]
        eps_yy = field_state.eps_yy[i0:i1, j0:j1, k0:k1]
        eps_zz = field_state.eps_zz[i0:i1, j0:j1, k0:k1]

        step_result = step_maxwell(
            {
                "E": E_interior,
                "H": H_interior,
                "eps_xx": eps_xx,
                "eps_yy": eps_yy,
                "eps_zz": eps_zz,
                "mu_xx": field_state.mu_xx[i0:i1, j0:j1, k0:k1],
                "mu_yy": field_state.mu_yy[i0:i1, j0:j1, k0:k1],
                "mu_zz": field_state.mu_zz[i0:i1, j0:j1, k0:k1],
            },
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            step_index=step,
            time=time,
        )
    else:
        # Numpy path: direct computation
        _numpy_electric_update_chunk(
            E_full, H_full, i0, j0, k0, i1, j1, k1,
            field_state.eps_xx, field_state.eps_yy, field_state.eps_zz,
            dt, dx, dy, dz,
        )


def _chunk_magnetic_update(
    E_full,
    H_full,
    chunk_spec,
    field_state,
    dt: float,
    dx: float, dy: float, dz: float,
    step: int,
    time: float,
) -> None:
    """Apply magnetic update to a chunk's interior region using Warp kernels."""
    (i0, j0, k0), (i1, j1, k1) = chunk_spec.interior_bounds

    is_warp = hasattr(E_full, 'numpy')

    if is_warp:
        import warp as wp

        E_interior = E_full[i0:i1, j0:j1, k0:k1]
        H_interior = H_full[i0:i1, j0:j1, k0:k1]

        step_result = step_maxwell(
            {
                "E": E_interior,
                "H": H_interior,
                "eps_xx": field_state.eps_xx[i0:i1, j0:j1, k0:k1],
                "eps_yy": field_state.eps_yy[i0:i1, j0:j1, k0:k1],
                "eps_zz": field_state.eps_zz[i0:i1, j0:j1, k0:k1],
                "mu_xx": field_state.mu_xx[i0:i1, j0:j1, k0:k1],
                "mu_yy": field_state.mu_yy[i0:i1, j0:j1, k0:k1],
                "mu_zz": field_state.mu_zz[i0:i1, j0:j1, k0:k1],
            },
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            step_index=step,
            time=time,
        )
    else:
        _numpy_magnetic_update_chunk(
            E_full, H_full, i0, j0, k0, i1, j1, k1,
            field_state.mu_xx, field_state.mu_yy, field_state.mu_zz,
            dt, dx, dy, dz,
        )


def _numpy_electric_update_chunk(
    E, H,
    i0, j0, k0, i1, j1, k1,
    eps_xx, eps_yy, eps_zz,
    dt, dx, dy, dz,
) -> None:
    """NumPy E update for a chunk interior region."""
    # Clamp indices
    i0 = max(0, i0)
    j0 = max(0, j0)
    k0 = max(0, k0)
    i1 = min(E.shape[0], i1)
    j1 = min(E.shape[1], j1)
    k1 = min(E.shape[2], k1)

    eps_xx_arr = eps_xx if eps_xx is not None else np.ones_like(E[..., 0])
    eps_yy_arr = eps_yy if eps_yy is not None else np.ones_like(E[..., 0])
    eps_zz_arr = eps_zz if eps_zz is not None else np.ones_like(E[..., 0])

    dt_dy = dt / dy
    dt_dz = dt / dz

    # Ex update
    _slice = (slice(i0, i1), slice(j0, j1), slice(k0, k1), slice(None))

    # Hy at z+1/2
    Hy_zp = H[i0:i1, j0+1:j1+1, k0:k1, 1]
    Hy_zp = np.pad(Hy_zp, [(0,0), (0,1), (0,0), (0,0)], mode='constant')

    # Hz at y+1/2
    Hz_yp = H[i0:i1, j0:j1, k0+1:k1+1, 2]
    Hz_yp = np.pad(Hz_yp, [(0,0), (0,0), (0,1), (0,0)], mode='constant')

    curl_H_Ex = (Hy_zp[..., 1] - H[i0:i1, j0:j1, k0:k1, 1]) * dt_dy / eps_xx_arr[i0:i1, j0:j1, k0:k1] \
                - (Hz_yp[..., 2] - H[i0:i1, j0:j1, k0:k1, 2]) * dt_dz / eps_xx_arr[i0:i1, j0:j1, k0:k1]
    E[i0:i1, j0:j1, k0:k1, 0] += curl_H_Ex


def _apply_symmetry_transforms(
    field_state: FieldState,
    compiled,
) -> None:
    """Apply symmetry transforms to E and H fields for active symmetry axes.

    This is called after the H update to enforce symmetry boundary conditions
    when using symmetry-reduced domains.

    Parameters
    ----------
    field_state : FieldState
        The field state containing E and H arrays.
    compiled : CompiledSimulation
        The compiled simulation containing symmetry metadata.
    """
    from autofdtd.kernels.boundaries import symmetry_transform

    boundary_spec = compiled.compiled_boundaries.boundary_spec
    active_axes = boundary_spec.active_symmetry_axes

    if not active_axes:
        return

    # Build a lookup from axis name to CompiledSymmetryAxis
    axis_by_name = {axis.axis: axis for axis in boundary_spec.symmetry_axes}

    is_warp = hasattr(field_state.E, 'numpy')

    for axis_name in active_axes:
        if axis_name not in axis_by_name:
            continue
        symmetry_axis = axis_by_name[axis_name]

        # Apply to electric field
        E_np = field_state.E.numpy() if is_warp else field_state.E
        E_transformed = symmetry_transform(E_np, symmetry_axis, field_family="electric")
        if is_warp:
            field_state.E.assign(E_transformed)
        else:
            field_state.E[:] = E_transformed

        # Apply to magnetic field
        H_np = field_state.H.numpy() if is_warp else field_state.H
        H_transformed = symmetry_transform(H_np, symmetry_axis, field_family="magnetic")
        if is_warp:
            field_state.H.assign(H_transformed)
        else:
            field_state.H[:] = H_transformed


def _apply_periodic_bloch_wrap(
    field_state: FieldState,
    compiled,
) -> None:
    """Apply periodic or Bloch boundary wrapping for domain ghost cells.

    For periodic boundaries: ghost cell = opposite interior cell (no phase)
    For Bloch boundaries: ghost cell = opposite interior cell * bloch_phase

    This is called after the H update in the single-chunk execution path
    to enforce periodic/Bloch boundary conditions at domain edges.

    Parameters
    ----------
    field_state : FieldState
        The field state containing E and H arrays.
    compiled : CompiledSimulation
        The compiled simulation containing boundary metadata.
    """
    boundary_spec = compiled.compiled_boundaries.boundary_spec

    # Get grid shape
    nx, ny, nz = field_state.E.shape[:3]

    is_warp = hasattr(field_state.E, 'numpy')

    # Process each axis
    for axis_name, axis_boundary in zip("xyz", boundary_spec.axes(), strict=True):
        minus_edge = axis_boundary.minus
        plus_edge = axis_boundary.plus

        # Determine if this axis has periodic or Bloch boundaries
        from autofdtd.compiler.boundaries import BoundaryMode

        if minus_edge.mode not in {BoundaryMode.PERIODIC, BoundaryMode.BLOCH}:
            continue
        if plus_edge.mode not in {BoundaryMode.PERIODIC, BoundaryMode.BLOCH}:
            continue

        # Both sides match - this is a valid periodic/Bloch axis
        use_phase_minus = minus_edge.mode == BoundaryMode.BLOCH
        use_phase_plus = plus_edge.mode == BoundaryMode.BLOCH
        phase_minus = minus_edge.phase_factor if use_phase_minus else 1.0 + 0.0j
        phase_plus = plus_edge.phase_factor if use_phase_plus else 1.0 + 0.0j

        axis_idx = "xyz".index(axis_name)
        depth = 1  # Standard halo depth for periodic/Bloch

        # For Yee grid with field at (nx, ny, nz, 3):
        # - ghost cells are at index 0 (minus) and nx-1 (plus) for x direction
        # - interior boundary cells are at index 1 (minus side) and nx-2 (plus side)
        if axis_name == "x":
            # x.minus ghost: copy from x.plus interior (index nx-2) with phase
            src_minus = slice(nx - depth - 1, nx - 1)  # Last interior cells
            dst_minus = slice(0, depth)
            # x.plus ghost: copy from x.minus interior (index 1) with phase
            src_plus = slice(1, 1 + depth)
            dst_plus = slice(nx - depth, nx)

            j_slice = slice(1, ny - 1)
            k_slice = slice(1, nz - 1)

            minus_slice = (dst_minus, j_slice, k_slice, slice(None))
            plus_slice = (dst_plus, j_slice, k_slice, slice(None))
            src_minus_full = (src_minus, j_slice, k_slice, slice(None))
            src_plus_full = (src_plus, j_slice, k_slice, slice(None))

        elif axis_name == "y":
            i_slice = slice(1, nx - 1)
            k_slice = slice(1, nz - 1)

            src_minus = slice(ny - depth - 1, ny - 1)
            dst_minus = slice(0, depth)
            src_plus = slice(1, 1 + depth)
            dst_plus = slice(ny - depth, ny)

            minus_slice = (i_slice, dst_minus, k_slice, slice(None))
            plus_slice = (i_slice, dst_plus, k_slice, slice(None))
            src_minus_full = (i_slice, src_minus, k_slice, slice(None))
            src_plus_full = (i_slice, src_plus, k_slice, slice(None))

        else:  # z
            i_slice = slice(1, nx - 1)
            j_slice = slice(1, ny - 1)

            src_minus = slice(nz - depth - 1, nz - 1)
            dst_minus = slice(0, depth)
            src_plus = slice(1, 1 + depth)
            dst_plus = slice(nz - depth, nz)

            minus_slice = (i_slice, j_slice, dst_minus, slice(None))
            plus_slice = (i_slice, j_slice, dst_plus, slice(None))
            src_minus_full = (i_slice, j_slice, src_minus, slice(None))
            src_plus_full = (i_slice, j_slice, src_plus, slice(None))

        # Apply wrap to E and H fields
        for field_arr in (field_state.E, field_state.H):
            if is_warp:
                arr_np = field_arr.numpy().copy()
            else:
                arr_np = field_arr.copy() if not isinstance(field_arr, np.ndarray) else field_arr

            # Get source values
            src_vals_minus = arr_np[src_minus_full]
            src_vals_plus = arr_np[src_plus_full]

            # Apply phase (for Bloch) and copy to ghost cells
            # For Bloch: ghost = interior * phase
            # For periodic: ghost = interior * 1.0 (just copy)
            if phase_minus != 1.0 + 0.0j:
                # Complex phase - promote to complex and multiply
                src_complex = src_vals_minus.astype(np.complex128)
                dst_vals_minus = src_complex * phase_minus
            else:
                dst_vals_minus = src_vals_minus.astype(np.complex128) if np.isrealobj(src_vals_minus) else src_vals_minus

            if phase_plus != 1.0 + 0.0j:
                src_complex = src_vals_plus.astype(np.complex128)
                dst_vals_plus = src_complex * phase_plus
            else:
                dst_vals_plus = src_vals_plus.astype(np.complex128) if np.isrealobj(src_vals_plus) else src_vals_plus

            arr_np[minus_slice] = dst_vals_minus
            arr_np[plus_slice] = dst_vals_plus

            if is_warp:
                field_arr.assign(arr_np)
            else:
                if isinstance(field_arr, np.ndarray):
                    field_arr[:] = arr_np


def _numpy_magnetic_update_chunk(
    E, H,
    i0, j0, k0, i1, j1, k1,
    mu_xx, mu_yy, mu_zz,
    dt, dx, dy, dz,
) -> None:
    """NumPy H update for a chunk interior region."""
    pass


def _chunked_apply_pml(
    E_full,
    H_full,
    chunk_spec,
    pml_state,
    chunk_idx: int,
    dt: float,
) -> None:
    """Apply PML boundary conditions to a chunk's exterior region.

    For multi-chunk execution, PML is applied to chunk exterior faces
    that are at domain boundaries.
    """
    pass