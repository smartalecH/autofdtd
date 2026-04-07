"""Examples for the Phase 1 simulation runtime-control surface."""

from __future__ import annotations

from autofdtd.api import GridSpec, Simulation, UniformGrid
from autofdtd.compiler import compile_runtime_controls
from autofdtd.diagnostics.metrics import (
    BenchmarkBoundaryClass,
    BenchmarkMaterialClass,
    BenchmarkPrecision,
    BenchmarkResult,
    BenchmarkTimer,
    EndToEndTiming,
    InitializationTiming,
    JitTiming,
    SteadyStateTiming,
    build_benchmark_metadata,
    build_steady_state_timing,
    compute_gcells_per_second,
)
from autofdtd.runtime import build_runtime_controller, run_until_stop_with_logging


def runtime_control_snapshot() -> dict[str, object]:
    """Return a compact snapshot of compiled stop-policy metadata."""

    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=1.0e-13,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        shutoff=1.0e-4,
    )
    compiled = compile_runtime_controls(simulation, shutoff_check_interval=4)
    return compiled.to_payload()


def shutoff_demo() -> dict[str, object]:
    """Return a simple early-stop demonstration using integrated-field intensity samples."""

    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=1.0e-13,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        shutoff=1.0e-3,
    )
    controller = build_runtime_controller(simulation, dt=1.0e-14, shutoff_check_interval=1)
    decision = None
    for step, intensity in enumerate((8.0, 2.0, 5.0e-4)):
        decision = controller.evaluate(step, integrated_electric=float(intensity))
        if decision.should_stop:
            break
    assert decision is not None
    return decision.to_payload()


def runtime_logging_demo() -> dict[str, object]:
    """Return a structured runtime log bundle for a simple shutoff-limited run."""

    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=1.0e-13,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        shutoff=1.0e-3,
    )
    log = run_until_stop_with_logging(
        simulation,
        [12.0, 5.0, 2.0, 5.0e-4],
        dt=1.0e-14,
        shutoff_check_interval=1,
        progress_interval=2,
    )
    return log.to_payload()


def benchmark_snapshot() -> dict[str, object]:
    """Return a structured benchmark result snapshot for a simple vacuum PEC run.

    This demonstrates the benchmark metadata and timing capture surface
    without requiring actual GPU kernel execution.
    """
    grid_shape = (20, 10, 10)
    num_steps = 50

    # Build metadata for a vacuum PEC benchmark
    metadata = build_benchmark_metadata(
        grid_shape=grid_shape,
        num_steps=num_steps,
        precision=BenchmarkPrecision.FLOAT32,
        boundary_class=BenchmarkBoundaryClass.PEC,
        material_class=BenchmarkMaterialClass.VACUUM,
    )

    # Simulate steady-state step timings (1ms per step)
    step_times = [0.001] * num_steps
    steady_timing = build_steady_state_timing(step_times, num_cells=metadata.total_cells)

    # Simulate initialization timing
    init_timing = InitializationTiming(
        wall_time_s=0.5,
        grid_resolution_s=0.1,
        scene_materialization_s=0.2,
        coefficient_compilation_s=0.1,
        boundary_compilation_s=0.1,
    )

    # Build JIT timing
    jit_timing = JitTiming(wall_time_s=0.3, module_stable=True)

    # Build end-to-end timing
    end_to_end = EndToEndTiming(
        initialization=init_timing,
        jit=jit_timing,
        steady_state=steady_timing,
        total_wall_time_s=0.5 + 0.3 + 0.050,  # init + jit + steady
    )

    # Build benchmark result
    result = BenchmarkResult(
        name="vacuum_pec_20x10x10",
        metadata=metadata,
        end_to_end_timing=end_to_end,
        steady_state_gcells_per_second=compute_gcells_per_second(
            metadata.total_cells, steady_timing.wall_time_s / num_steps
        ),
        end_to_end_gcells_per_second=compute_gcells_per_second(
            metadata.total_cells * num_steps, end_to_end.total_wall_time_s
        ),
        initialization_time_s=init_timing.wall_time_s,
        jit_time_s=jit_timing.wall_time_s,
        stop_reason="step_count",
        converged=False,
    )

    return result.to_payload()
