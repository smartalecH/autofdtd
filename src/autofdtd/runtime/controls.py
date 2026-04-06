"""Runtime stop-policy execution helpers for the Phase 1 timestep loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autofdtd.compiler.runtime import (
    CompiledRuntimeControls,
    RuntimeConvergencePolicy,
    RuntimeStopReason,
    compile_runtime_controls,
)
from autofdtd.core.containers import Simulation
from autofdtd.core.models import AutoFDTDModel
from autofdtd.diagnostics.runtime_logging import RuntimeExecutionLog, RuntimeProgressLogger


def integrated_electric_field_intensity(electric_field: np.ndarray) -> float:
    """Return the instantaneous integrated electric-field intensity."""

    field = np.asarray(electric_field)
    return float(np.sum(np.abs(field) ** 2))


class RuntimeStopDecision(AutoFDTDModel):
    """Structured stop-decision record for one timestep-loop evaluation."""

    type: str = "RuntimeStopDecision"
    should_stop: bool
    reason: RuntimeStopReason | None = None
    step: int
    step_time: float
    convergence_policy: RuntimeConvergencePolicy
    integrated_electric: float | None = None
    peak_integrated_electric: float | None = None
    shutoff_ratio: float | None = None
    shutoff_threshold: float | None = None
    checks_performed: int = 0


@dataclass(slots=True)
class _RuntimeConvergenceState:
    peak_integrated_electric: float = 0.0
    checks_performed: int = 0


class RuntimeController:
    """Stateful evaluator for the compiled Phase 1 stop-policy metadata."""

    def __init__(self, compiled: CompiledRuntimeControls) -> None:
        self.compiled = compiled
        self._convergence = _RuntimeConvergenceState()

    def reset(self) -> None:
        """Reset peak-history state for a fresh run."""

        self._convergence = _RuntimeConvergenceState()

    def evaluate(
        self,
        step: int,
        *,
        electric_field: np.ndarray | None = None,
        integrated_electric: float | None = None,
    ) -> RuntimeStopDecision:
        """Evaluate the configured stop conditions at one zero-based step index."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if electric_field is not None and integrated_electric is not None:
            raise ValueError("provide either electric_field or integrated_electric, not both")
        if electric_field is not None:
            integrated_electric = integrated_electric_field_intensity(electric_field)

        step_time = self.compiled.step_time(step)
        peak = self._convergence.peak_integrated_electric
        if integrated_electric is not None:
            peak = max(peak, integrated_electric)
            self._convergence.peak_integrated_electric = peak

        shutoff_ratio: float | None = None
        shutoff_reason = False
        if (
            self.compiled.convergence_policy
            is RuntimeConvergencePolicy.INTEGRATED_ELECTRIC_FIELD
            and integrated_electric is not None
            and step % self.compiled.shutoff_check_interval == 0
        ):
            self._convergence.checks_performed += 1
            if peak > 0.0:
                shutoff_ratio = integrated_electric / peak
            else:
                shutoff_ratio = 1.0
            shutoff_reason = step > 0 and shutoff_ratio <= float(self.compiled.shutoff)

        if shutoff_reason:
            return RuntimeStopDecision(
                should_stop=True,
                reason=RuntimeStopReason.SHUTOFF,
                step=step,
                step_time=step_time,
                convergence_policy=self.compiled.convergence_policy,
                integrated_electric=integrated_electric,
                peak_integrated_electric=peak if peak > 0.0 else None,
                shutoff_ratio=shutoff_ratio,
                shutoff_threshold=self.compiled.shutoff,
                checks_performed=self._convergence.checks_performed,
            )

        if step >= self.compiled.max_step_index:
            return RuntimeStopDecision(
                should_stop=True,
                reason=self.compiled.primary_stop_reason,
                step=step,
                step_time=step_time,
                convergence_policy=self.compiled.convergence_policy,
                integrated_electric=integrated_electric,
                peak_integrated_electric=peak if peak > 0.0 else None,
                shutoff_ratio=shutoff_ratio,
                shutoff_threshold=self.compiled.shutoff,
                checks_performed=self._convergence.checks_performed,
            )

        return RuntimeStopDecision(
            should_stop=False,
            step=step,
            step_time=step_time,
            convergence_policy=self.compiled.convergence_policy,
            integrated_electric=integrated_electric,
            peak_integrated_electric=peak if peak > 0.0 else None,
            shutoff_ratio=shutoff_ratio,
            shutoff_threshold=self.compiled.shutoff,
            checks_performed=self._convergence.checks_performed,
        )


def build_runtime_controller(
    simulation: Simulation,
    *,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
) -> RuntimeController:
    """Build the default Phase 1 stop-policy controller for a simulation."""

    return RuntimeController(
        compile_runtime_controls(
            simulation,
            dt=dt,
            max_steps=max_steps,
            shutoff_check_interval=shutoff_check_interval,
        )
    )


def run_until_stop(
    simulation: Simulation,
    integrated_electric_history: list[float] | tuple[float, ...],
    *,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
) -> RuntimeStopDecision:
    """Drive the controller over precomputed intensity history and return the stop decision."""

    controller = build_runtime_controller(
        simulation,
        dt=dt,
        max_steps=max_steps,
        shutoff_check_interval=shutoff_check_interval,
    )
    decision: RuntimeStopDecision | None = None
    for step, intensity in enumerate(integrated_electric_history):
        decision = controller.evaluate(step, integrated_electric=float(intensity))
        if decision.should_stop:
            return decision
    if decision is None:
        raise ValueError("integrated_electric_history must contain at least one sample")
    return decision


def run_until_stop_with_logging(
    simulation: Simulation,
    integrated_electric_history: list[float] | tuple[float, ...],
    *,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
    progress_interval: int = 10,
) -> RuntimeExecutionLog:
    """Drive the controller over precomputed intensity history and collect structured logs."""

    controller = build_runtime_controller(
        simulation,
        dt=dt,
        max_steps=max_steps,
        shutoff_check_interval=shutoff_check_interval,
    )
    logger = RuntimeProgressLogger(
        controller.compiled,
        progress_interval=progress_interval,
    )
    decision: RuntimeStopDecision | None = None
    for step, intensity in enumerate(integrated_electric_history):
        decision = controller.evaluate(step, integrated_electric=float(intensity))
        logger.record(decision)
        if decision.should_stop:
            return logger.finalize(decision)
    if decision is None:
        raise ValueError("integrated_electric_history must contain at least one sample")
    return logger.finalize(decision)
