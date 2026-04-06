"""Runtime-control compilation helpers for Phase 1 simulation execution."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from autofdtd.core.containers import Simulation
from autofdtd.core.models import AutoFDTDModel

C_0 = 299_792_458.0
_TIME_TOLERANCE = 1.0e-18


class RuntimeStopReason(StrEnum):
    """Primary stop conditions supported by the Phase 1 runtime loop."""

    RUN_TIME = "run_time"
    STEP_COUNT = "step_count"
    SHUTOFF = "shutoff"


class RuntimeConvergencePolicy(StrEnum):
    """Explicit convergence policies supported by the Phase 1 runtime loop."""

    NONE = "none"
    INTEGRATED_ELECTRIC_FIELD = "integrated_electric_field"


class CompiledRuntimeControls(AutoFDTDModel):
    """Compiled stop-policy metadata consumed by the runtime stepping loop."""

    type: Literal["CompiledRuntimeControls"] = "CompiledRuntimeControls"
    run_time: float
    dt: float
    num_time_steps: int
    max_step_index: int
    scaled_courant: float
    cfl_spacings: tuple[float, ...]
    primary_stop_reason: RuntimeStopReason
    convergence_policy: RuntimeConvergencePolicy
    shutoff: float | None = None
    shutoff_check_interval: int = 1
    normalize_index: int | None = None
    user_step_limit: int | None = None

    @property
    def shutoff_enabled(self) -> bool:
        """Return whether the compiled controls request shutoff-based early stop."""

        return self.convergence_policy is RuntimeConvergencePolicy.INTEGRATED_ELECTRIC_FIELD

    def step_time(self, step: int) -> float:
        """Return the physical time associated with a zero-based step index."""

        if step < 0:
            raise ValueError("step must be non-negative")
        return step * self.dt


def effective_courant(simulation: Simulation) -> float:
    """Return the Courant factor after applying Phase 1 subpixel scaling."""

    subpixel_ratio = 1.0
    if simulation.subpixel is not None:
        subpixel_ratio = simulation.subpixel.courant_ratio()
    return simulation.courant * subpixel_ratio


def cfl_spacings(simulation: Simulation) -> tuple[float, ...]:
    """Return the active Yee-cell spacings that participate in the CFL limit."""

    resolved_grid = simulation.resolved_grid()
    if resolved_grid is None:
        raise ValueError("runtime control compilation requires a resolved grid_spec")
    spacings = tuple(
        axis.min_step
        for axis in (resolved_grid.x, resolved_grid.y, resolved_grid.z)
        if axis.num_cells > 0 and axis.min_step > 0.0
    )
    if not spacings:
        raise ValueError("runtime control compilation requires at least one non-zero grid spacing")
    return spacings


def estimate_time_step(simulation: Simulation) -> float:
    """Estimate the explicit Yee time step from the resolved grid and Courant factor."""

    spacings = cfl_spacings(simulation)
    inv_sq_sum = sum(1.0 / (spacing * spacing) for spacing in spacings)
    if inv_sq_sum <= 0.0:
        raise ValueError("runtime control compilation requires positive grid spacing")
    return effective_courant(simulation) / (C_0 * math.sqrt(inv_sq_sum))


def estimate_num_time_steps(simulation: Simulation, *, dt: float | None = None) -> int:
    """Return the Tidy3D-style number of time points in the simulation tmesh."""

    timestep = estimate_time_step(simulation) if dt is None else float(dt)
    if timestep <= 0.0:
        raise ValueError("dt must be positive")
    return int(math.ceil(simulation.run_time / timestep - _TIME_TOLERANCE)) + 1


def compile_runtime_controls(
    simulation: Simulation,
    *,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
) -> CompiledRuntimeControls:
    """Compile the public simulation stop controls into runtime-ready metadata."""

    timestep = estimate_time_step(simulation) if dt is None else float(dt)
    if timestep <= 0.0:
        raise ValueError("runtime control compilation requires dt > 0")
    if shutoff_check_interval <= 0:
        raise ValueError("shutoff_check_interval must be a positive integer")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be a positive integer when provided")

    run_time_steps = estimate_num_time_steps(simulation, dt=timestep)
    num_time_steps = min(run_time_steps, max_steps) if max_steps is not None else run_time_steps
    primary_stop_reason = (
        RuntimeStopReason.STEP_COUNT
        if max_steps is not None and max_steps < run_time_steps
        else RuntimeStopReason.RUN_TIME
    )
    convergence_policy = (
        RuntimeConvergencePolicy.INTEGRATED_ELECTRIC_FIELD
        if simulation.shutoff not in (None, 0.0)
        else RuntimeConvergencePolicy.NONE
    )

    return CompiledRuntimeControls(
        run_time=simulation.run_time,
        dt=timestep,
        num_time_steps=num_time_steps,
        max_step_index=num_time_steps - 1,
        scaled_courant=effective_courant(simulation),
        cfl_spacings=cfl_spacings(simulation),
        primary_stop_reason=primary_stop_reason,
        convergence_policy=convergence_policy,
        shutoff=simulation.shutoff if convergence_policy is not RuntimeConvergencePolicy.NONE else None,
        shutoff_check_interval=shutoff_check_interval,
        normalize_index=simulation.normalize_index,
        user_step_limit=max_steps,
    )
