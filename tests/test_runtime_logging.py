from __future__ import annotations

import json

import pytest

from autofdtd.api import GridSpec, Simulation, UniformGrid
from autofdtd.compiler import RuntimeConvergencePolicy, RuntimeStopReason
from autofdtd.diagnostics import RuntimeLogEventKind, RuntimeLogTrigger
from autofdtd.runtime import run_until_stop_with_logging


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


def test_run_until_stop_with_logging_emits_interval_and_final_records() -> None:
    log = run_until_stop_with_logging(
        build_simulation(shutoff=1.0e-3),
        [12.0, 3.0, 5.0e-4],
        dt=1.0e-14,
        shutoff_check_interval=1,
        progress_interval=2,
    )

    assert log.final_decision["reason"] == RuntimeStopReason.SHUTOFF
    assert log.emitted_event_count == 3
    assert log.executed_steps == 3
    assert log.convergence_evidence.triggered
    assert log.convergence_evidence.checks_performed == 3
    assert log.convergence_evidence.minimum_shutoff_ratio == pytest.approx(5.0e-4 / 12.0)
    assert len(log.convergence_evidence.samples) == 3
    assert log.events[0].kind is RuntimeLogEventKind.PROGRESS
    assert log.events[0].trigger is RuntimeLogTrigger.INITIAL
    assert log.events[1].trigger is RuntimeLogTrigger.INTERVAL
    assert log.events[-1].kind is RuntimeLogEventKind.STOP
    assert log.events[-1].trigger is RuntimeLogTrigger.FINAL
    assert log.events[-1].stop_reason is RuntimeStopReason.SHUTOFF
    assert log.events[-1].progress_fraction == pytest.approx(3 / log.num_time_steps)


def test_run_until_stop_with_logging_preserves_step_limit_stop_metadata() -> None:
    log = run_until_stop_with_logging(
        build_simulation(shutoff=0.0),
        [1.0, 1.0, 1.0],
        dt=1.0e-14,
        max_steps=3,
        progress_interval=10,
    )

    assert log.final_decision["reason"] == RuntimeStopReason.STEP_COUNT
    assert log.primary_stop_reason is RuntimeStopReason.STEP_COUNT
    assert log.convergence_policy is RuntimeConvergencePolicy.NONE
    assert not log.convergence_evidence.triggered
    assert log.convergence_evidence.samples == ()
    assert log.events[-1].kind is RuntimeLogEventKind.STOP
    assert log.events[-1].stop_reason is RuntimeStopReason.STEP_COUNT


def test_runtime_execution_log_writes_jsonl_event_stream(tmp_path) -> None:
    log = run_until_stop_with_logging(
        build_simulation(shutoff=1.0e-3),
        [12.0, 3.0, 5.0e-4],
        dt=1.0e-14,
        shutoff_check_interval=1,
        progress_interval=1,
    )

    path = tmp_path / "runtime-log.jsonl"
    log.write_jsonl(path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == log.emitted_event_count
    payload = json.loads(lines[-1])
    assert payload["kind"] == RuntimeLogEventKind.STOP
    assert payload["stop_reason"] == RuntimeStopReason.SHUTOFF
