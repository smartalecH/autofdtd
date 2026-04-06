"""Structured runtime logging and convergence evidence for Phase 1 execution."""

from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal, Protocol

from autofdtd.compiler.runtime import (
    CompiledRuntimeControls,
    RuntimeConvergencePolicy,
    RuntimeStopReason,
)
from autofdtd.core.models import AutoFDTDModel


class RuntimeStopDecisionLike(Protocol):
    """Structural typing contract for stop decisions consumed by the logger."""

    should_stop: bool
    reason: RuntimeStopReason | None
    step: int
    step_time: float
    convergence_policy: RuntimeConvergencePolicy
    integrated_electric: float | None
    peak_integrated_electric: float | None
    shutoff_ratio: float | None
    shutoff_threshold: float | None
    checks_performed: int

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-ready payload."""


class RuntimeLogEventKind(StrEnum):
    """Structured runtime-log event kinds emitted by the Phase 1 controller seam."""

    PROGRESS = "progress"
    STOP = "stop"


class RuntimeLogTrigger(StrEnum):
    """Reasons why a progress event was emitted."""

    INITIAL = "initial"
    INTERVAL = "interval"
    FINAL = "final"


class RuntimeConvergenceSample(AutoFDTDModel):
    """One convergence-check sample captured during a run."""

    type: Literal["RuntimeConvergenceSample"] = "RuntimeConvergenceSample"
    step: int
    step_time: float
    integrated_electric: float
    peak_integrated_electric: float
    shutoff_ratio: float
    shutoff_threshold: float
    below_threshold: bool
    checks_performed: int


class RuntimeConvergenceEvidence(AutoFDTDModel):
    """Structured convergence evidence accumulated across a run."""

    type: Literal["RuntimeConvergenceEvidence"] = "RuntimeConvergenceEvidence"
    convergence_policy: RuntimeConvergencePolicy
    shutoff_threshold: float | None = None
    checks_performed: int = 0
    peak_integrated_electric: float | None = None
    last_integrated_electric: float | None = None
    last_shutoff_ratio: float | None = None
    minimum_shutoff_ratio: float | None = None
    triggered: bool = False
    samples: tuple[RuntimeConvergenceSample, ...] = ()


class RuntimeLogEvent(AutoFDTDModel):
    """Structured progress or stop record for one emitted timestep event."""

    type: Literal["RuntimeLogEvent"] = "RuntimeLogEvent"
    kind: RuntimeLogEventKind
    trigger: RuntimeLogTrigger
    step: int
    step_time: float
    completed_steps: int
    total_steps: int
    progress_fraction: float
    steps_remaining: int
    elapsed_wall_time_s: float
    convergence_policy: RuntimeConvergencePolicy
    integrated_electric: float | None = None
    peak_integrated_electric: float | None = None
    shutoff_ratio: float | None = None
    shutoff_threshold: float | None = None
    checks_performed: int = 0
    should_stop: bool = False
    stop_reason: RuntimeStopReason | None = None


class RuntimeExecutionLog(AutoFDTDModel):
    """End-to-end structured logging bundle for one Phase 1 runtime execution."""

    type: Literal["RuntimeExecutionLog"] = "RuntimeExecutionLog"
    dt: float
    run_time: float
    num_time_steps: int
    max_step_index: int
    primary_stop_reason: RuntimeStopReason
    convergence_policy: RuntimeConvergencePolicy
    shutoff_threshold: float | None = None
    shutoff_check_interval: int
    progress_interval: int
    emitted_event_count: int
    executed_steps: int
    elapsed_wall_time_s: float
    final_decision: dict[str, object]
    convergence_evidence: RuntimeConvergenceEvidence
    events: tuple[RuntimeLogEvent, ...]

    def to_json_lines(self) -> str:
        """Return the emitted event stream as JSON Lines text."""

        return "".join(
            event.model_dump_json(exclude_none=True) + "\n" for event in self.events
        )

    def write_jsonl(self, path: str | Path) -> None:
        """Write the emitted event stream to disk as JSON Lines."""

        Path(path).write_text(self.to_json_lines(), encoding="utf-8")


class RuntimeProgressLogger:
    """Collect structured progress logs and convergence evidence for a run."""

    def __init__(
        self,
        compiled: CompiledRuntimeControls,
        *,
        progress_interval: int = 10,
        wall_time_source: Callable[[], float] | None = None,
    ) -> None:
        if progress_interval <= 0:
            raise ValueError("progress_interval must be a positive integer")
        self.compiled = compiled
        self.progress_interval = progress_interval
        self._wall_time = wall_time_source or time.perf_counter
        self._started_at = self._wall_time()
        self._events: list[RuntimeLogEvent] = []
        self._convergence_samples: list[RuntimeConvergenceSample] = []
        self._last_decision: RuntimeStopDecisionLike | None = None

    @property
    def events(self) -> tuple[RuntimeLogEvent, ...]:
        """Return the currently collected event stream."""

        return tuple(self._events)

    def record(self, decision: RuntimeStopDecisionLike) -> RuntimeLogEvent | None:
        """Record convergence evidence and maybe emit a structured log event."""

        self._last_decision = decision
        if decision.shutoff_ratio is not None and decision.integrated_electric is not None:
            self._convergence_samples.append(
                RuntimeConvergenceSample(
                    step=decision.step,
                    step_time=decision.step_time,
                    integrated_electric=decision.integrated_electric,
                    peak_integrated_electric=float(
                        decision.peak_integrated_electric or decision.integrated_electric
                    ),
                    shutoff_ratio=decision.shutoff_ratio,
                    shutoff_threshold=float(decision.shutoff_threshold or 0.0),
                    below_threshold=decision.should_stop
                    and decision.reason is RuntimeStopReason.SHUTOFF,
                    checks_performed=decision.checks_performed,
                )
            )

        trigger = self._emission_trigger(decision)
        if trigger is None:
            return None

        event = RuntimeLogEvent(
            kind=RuntimeLogEventKind.STOP if decision.should_stop else RuntimeLogEventKind.PROGRESS,
            trigger=trigger,
            step=decision.step,
            step_time=decision.step_time,
            completed_steps=min(decision.step + 1, self.compiled.num_time_steps),
            total_steps=self.compiled.num_time_steps,
            progress_fraction=min(
                (decision.step + 1) / max(self.compiled.num_time_steps, 1),
                1.0,
            ),
            steps_remaining=max(self.compiled.max_step_index - decision.step, 0),
            elapsed_wall_time_s=max(self._wall_time() - self._started_at, 0.0),
            convergence_policy=decision.convergence_policy,
            integrated_electric=decision.integrated_electric,
            peak_integrated_electric=decision.peak_integrated_electric,
            shutoff_ratio=decision.shutoff_ratio,
            shutoff_threshold=decision.shutoff_threshold,
            checks_performed=decision.checks_performed,
            should_stop=decision.should_stop,
            stop_reason=decision.reason,
        )
        self._events.append(event)
        return event

    def finalize(
        self,
        final_decision: RuntimeStopDecisionLike | None = None,
    ) -> RuntimeExecutionLog:
        """Build the final structured log bundle for the run."""

        decision = final_decision or self._last_decision
        if decision is None:
            raise ValueError("cannot finalize runtime logging before at least one decision")

        evidence = self._build_convergence_evidence(decision)
        return RuntimeExecutionLog(
            dt=self.compiled.dt,
            run_time=self.compiled.run_time,
            num_time_steps=self.compiled.num_time_steps,
            max_step_index=self.compiled.max_step_index,
            primary_stop_reason=self.compiled.primary_stop_reason,
            convergence_policy=self.compiled.convergence_policy,
            shutoff_threshold=self.compiled.shutoff,
            shutoff_check_interval=self.compiled.shutoff_check_interval,
            progress_interval=self.progress_interval,
            emitted_event_count=len(self._events),
            executed_steps=min(decision.step + 1, self.compiled.num_time_steps),
            elapsed_wall_time_s=max(self._wall_time() - self._started_at, 0.0),
            final_decision=decision.to_payload(),
            convergence_evidence=evidence,
            events=tuple(self._events),
        )

    def _emission_trigger(
        self,
        decision: RuntimeStopDecisionLike,
    ) -> RuntimeLogTrigger | None:
        if decision.should_stop:
            return RuntimeLogTrigger.FINAL
        if decision.step == 0:
            return RuntimeLogTrigger.INITIAL
        if (decision.step + 1) % self.progress_interval == 0:
            return RuntimeLogTrigger.INTERVAL
        return None

    def _build_convergence_evidence(
        self,
        final_decision: RuntimeStopDecisionLike,
    ) -> RuntimeConvergenceEvidence:
        minimum_ratio = None
        if self._convergence_samples:
            minimum_ratio = min(sample.shutoff_ratio for sample in self._convergence_samples)
        return RuntimeConvergenceEvidence(
            convergence_policy=self.compiled.convergence_policy,
            shutoff_threshold=self.compiled.shutoff,
            checks_performed=final_decision.checks_performed,
            peak_integrated_electric=final_decision.peak_integrated_electric,
            last_integrated_electric=final_decision.integrated_electric,
            last_shutoff_ratio=final_decision.shutoff_ratio,
            minimum_shutoff_ratio=minimum_ratio,
            triggered=final_decision.reason is RuntimeStopReason.SHUTOFF,
            samples=tuple(self._convergence_samples),
        )
