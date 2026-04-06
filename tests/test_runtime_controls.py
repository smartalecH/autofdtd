from __future__ import annotations

import pytest

from autofdtd.api import GridSpec, Simulation, UniformGrid
from autofdtd.compiler import (
    RuntimeConvergencePolicy,
    RuntimeStopReason,
    compile_runtime_controls,
)
from autofdtd.runtime import build_runtime_controller, run_until_stop


def build_simulation(*, shutoff: float = 1.0e-5) -> Simulation:
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=1.0e-13,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        shutoff=shutoff,
    )


def test_compile_runtime_controls_derives_tidy3d_style_step_count() -> None:
    simulation = build_simulation()

    compiled = compile_runtime_controls(simulation, shutoff_check_interval=8)

    assert compiled.primary_stop_reason is RuntimeStopReason.RUN_TIME
    assert compiled.convergence_policy is RuntimeConvergencePolicy.INTEGRATED_ELECTRIC_FIELD
    assert compiled.shutoff == pytest.approx(1.0e-5)
    assert compiled.num_time_steps == simulation.num_time_steps()
    assert compiled.max_step_index == compiled.num_time_steps - 1
    assert compiled.shutoff_check_interval == 8


def test_compile_runtime_controls_allows_explicit_step_count_override() -> None:
    simulation = build_simulation()

    compiled = compile_runtime_controls(simulation, max_steps=12)

    assert compiled.primary_stop_reason is RuntimeStopReason.STEP_COUNT
    assert compiled.user_step_limit == 12
    assert compiled.num_time_steps == 12
    assert compiled.max_step_index == 11


def test_runtime_controller_stops_early_on_shutoff_ratio() -> None:
    decision = run_until_stop(
        build_simulation(shutoff=1.0e-3),
        [12.0, 3.0, 5.0e-4],
        dt=1.0e-14,
        shutoff_check_interval=1,
    )

    assert decision.should_stop
    assert decision.reason is RuntimeStopReason.SHUTOFF
    assert decision.step == 2
    assert decision.shutoff_ratio == pytest.approx(5.0e-4 / 12.0)
    assert decision.checks_performed == 3


def test_runtime_controller_can_disable_shutoff_and_stop_on_step_limit() -> None:
    controller = build_runtime_controller(
        build_simulation(shutoff=0.0),
        dt=1.0e-14,
        max_steps=3,
    )

    decisions = [controller.evaluate(step, integrated_electric=1.0) for step in range(3)]

    assert decisions[-1].should_stop
    assert decisions[-1].reason is RuntimeStopReason.STEP_COUNT
    assert decisions[-1].convergence_policy is RuntimeConvergencePolicy.NONE
    assert decisions[-1].shutoff_ratio is None


def test_compile_runtime_controls_rejects_invalid_step_override() -> None:
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        compile_runtime_controls(build_simulation(), max_steps=0)
