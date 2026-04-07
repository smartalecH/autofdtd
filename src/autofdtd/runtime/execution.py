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

from autofdtd.kernels.backend import WARP_AVAILABLE, ComplexFieldPolicy
from autofdtd.kernels.steps import allocate_maxwell_arrays, step_maxwell


@dataclass
class FieldState:
    """Mutable field state container for a simulation run.

    Attributes
    ----------
    E : np.ndarray
        Electric field array with shape (nx, ny, nz, 3).
    H : np.ndarray
        Magnetic field array with shape (nx, ny, nz, 3).
    eps_xx, eps_yy, eps_zz : np.ndarray
        Permittivity arrays with shape (nx, ny, nz).
    mu_xx, mu_yy, mu_zz : np.ndarray
        Permeability arrays with shape (nx, ny, nz).
    electric_modes : np.ndarray
        Electric constitutive mode array with shape (nx, ny, nz).
    magnetic_modes : np.ndarray
        Magnetic constitutive mode array with shape (nx, ny, nz).
    """

    E: np.ndarray
    H: np.ndarray
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    mu_xx: np.ndarray
    mu_yy: np.ndarray
    mu_zz: np.ndarray
    electric_modes: np.ndarray
    magnetic_modes: np.ndarray


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
    field_state.eps_xx.fill(1.0)
    field_state.eps_yy.fill(1.0)
    field_state.eps_zz.fill(1.0)
    field_state.mu_xx.fill(1.0)
    field_state.mu_yy.fill(1.0)
    field_state.mu_zz.fill(1.0)

    # Get background medium coefficients
    bg = compiled.scene_coefficients.background
    bg_eps = getattr(bg, "permittivity", 1.0)
    bg_mu = getattr(bg, "permeability", 1.0)

    if bg_eps is not None:
        field_state.eps_xx.fill(bg_eps)
        field_state.eps_yy.fill(bg_eps)
        field_state.eps_zz.fill(bg_eps)
    if bg_mu is not None:
        field_state.mu_xx.fill(bg_mu)
        field_state.mu_yy.fill(bg_mu)
        field_state.mu_zz.fill(bg_mu)

    # Apply structure coefficients (simplified - just apply first structure as override)
    # Full implementation would use the materialized grid with structure overlap
    for struct_coeff in compiled.scene_coefficients.structures:
        mat = struct_coeff.material_coefficients
        mat_eps = getattr(mat, "permittivity", None)
        mat_mu = getattr(mat, "permeability", None)

        if mat_eps is not None:
            # For simplicity, fill entire array with structure medium
            # A full implementation would check structure geometry bounds
            field_state.eps_xx.fill(mat_eps)
            field_state.eps_yy.fill(mat_eps)
            field_state.eps_zz.fill(mat_eps)
        if mat_mu is not None:
            field_state.mu_xx.fill(mat_mu)
            field_state.mu_yy.fill(mat_mu)
            field_state.mu_zz.fill(mat_mu)


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
    grid_shape = compiled.grid_shape
    complex_policy = ComplexFieldPolicy.from_boundary_spec(
        compiled.compiled_boundaries.boundary_spec
    )

    # Allocate arrays using the kernel helper
    arrays = allocate_maxwell_arrays(
        grid_shape,
        complex_policy=complex_policy,
    )

    field_state = FieldState(
        E=arrays["E"],
        H=arrays["H"],
        eps_xx=arrays["eps_xx"],
        eps_yy=arrays["eps_yy"],
        eps_zz=arrays["eps_zz"],
        mu_xx=arrays["mu_xx"],
        mu_yy=arrays["mu_yy"],
        mu_zz=arrays["mu_zz"],
        electric_modes=np.zeros(grid_shape, dtype=np.float64),
        magnetic_modes=np.zeros(grid_shape, dtype=np.float64),
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
            time=current_time,
            dt=dt,
        )

        # 2. Electric field update
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

        # 3. Compute integrated electric field for convergence check
        E_sq = np.sum(np.abs(field_state.E) ** 2)
        integrated_electric = float(E_sq)
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