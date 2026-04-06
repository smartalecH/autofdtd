"""Examples for the Phase 1 simulation runtime-control surface."""

from __future__ import annotations

from autofdtd.api import GridSpec, Simulation, UniformGrid
from autofdtd.compiler import compile_runtime_controls
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
