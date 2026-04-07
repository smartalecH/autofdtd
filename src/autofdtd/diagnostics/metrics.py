"""Structured runtime metrics, benchmark helpers, and execution evidence capture for Phase 1.

This module provides:

- Benchmark metadata (grid shape, cell count, precision, boundary class)
- Initialization, JIT, and steady-state timing
- Cells-updated-per-second and Gcells/s metrics
- Structured benchmark result bundles
- End-to-end vs steady-state timing separation

References
----------
- GPU Benchmarking: ``../../papers/gpu_benchmarking.pdf``
- Warp backend conventions: ``autofdtd/kernels/backend.py``
- Runtime logging: ``autofdtd/diagnostics/runtime_logging.py``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from autofdtd.core.models import AutoFDTDModel


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------


class BenchmarkPrecision(StrEnum):
    """Numerical precision used in a benchmark run."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


class BenchmarkBoundaryClass(StrEnum):
    """Boundary condition class for benchmark categorization."""

    PEC = "pec"
    PMC = "pmc"
    PERIODIC = "periodic"
    BLOCH = "bloch"
    PML = "pml"
    ABSORBER = "absorber"
    MIXED = "mixed"


class BenchmarkMaterialClass(StrEnum):
    """Material complexity class for benchmark categorization."""

    VACUUM = "vacuum"
    DIELECTRIC = "dielectric"
    DISPERSIVE = "dispersive"
    ANISOTROPIC = "anisotropic"
    PEC = "pec"


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Static metadata describing one benchmark run configuration."""

    grid_shape: tuple[int, int, int]
    total_cells: int
    num_steps: int
    precision: BenchmarkPrecision
    boundary_class: BenchmarkBoundaryClass
    material_class: BenchmarkMaterialClass
    single_gpu: bool = True
    multi_gpu_count: int | None = None
    has_bloch: bool = False
    has_pml: bool = False
    has_dispersive: bool = False
    has_anisotropic: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-ready payload."""
        return {
            "grid_shape": self.grid_shape,
            "total_cells": self.total_cells,
            "num_steps": self.num_steps,
            "precision": self.precision.value,
            "boundary_class": self.boundary_class.value,
            "material_class": self.material_class.value,
            "single_gpu": self.single_gpu,
            "multi_gpu_count": self.multi_gpu_count,
            "has_bloch": self.has_bloch,
            "has_pml": self.has_pml,
            "has_dispersive": self.has_dispersive,
            "has_anisotropic": self.has_anisotropic,
        }


# ---------------------------------------------------------------------------
# Timing phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitializationTiming:
    """Timing for simulation initialization phase."""

    wall_time_s: float
    grid_resolution_s: float | None = None
    scene_materialization_s: float | None = None
    coefficient_compilation_s: float | None = None
    source_planning_s: float | None = None
    monitor_planning_s: float | None = None
    boundary_compilation_s: float | None = None
    chunk_layout_s: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "grid_resolution_s": self.grid_resolution_s,
            "scene_materialization_s": self.scene_materialization_s,
            "coefficient_compilation_s": self.coefficient_compilation_s,
            "source_planning_s": self.source_planning_s,
            "monitor_planning_s": self.monitor_planning_s,
            "boundary_compilation_s": self.boundary_compilation_s,
            "chunk_layout_s": self.chunk_layout_s,
        }


@dataclass(frozen=True)
class JitTiming:
    """Timing for JIT compilation phase."""

    wall_time_s: float
    kernel_compilation_s: float | None = None
    module_stable: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "kernel_compilation_s": self.kernel_compilation_s,
            "module_stable": self.module_stable,
        }


@dataclass(frozen=True)
class SteadyStateTiming:
    """Timing for the steady-state timestep window (after initialization and JIT)."""

    wall_time_s: float
    steps_executed: int
    cells_updated_per_step: int
    gcells_per_second: float | None = None
    end_to_end_gcells_per_second: float | None = None
    min_step_time_s: float | None = None
    max_step_time_s: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "steps_executed": self.steps_executed,
            "cells_updated_per_step": self.cells_updated_per_step,
            "gcells_per_second": self.gcells_per_second,
            "end_to_end_gcells_per_second": self.end_to_end_gcells_per_second,
            "min_step_time_s": self.min_step_time_s,
            "max_step_time_s": self.max_step_time_s,
        }


@dataclass(frozen=True)
class EndToEndTiming:
    """Complete end-to-end timing for a benchmark run."""

    initialization: InitializationTiming
    jit: JitTiming | None
    steady_state: SteadyStateTiming | None
    total_wall_time_s: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "initialization": self.initialization.to_payload(),
            "jit": self.jit.to_payload() if self.jit else None,
            "steady_state": self.steady_state.to_payload() if self.steady_state else None,
            "total_wall_time_s": self.total_wall_time_s,
        }


# ---------------------------------------------------------------------------
# Execution evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepEvidence:
    """Per-step evidence from a benchmark timestep window."""

    step_index: int
    wall_time_s: float
    integrated_electric: float | None = None
    gcells_per_second: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "wall_time_s": self.wall_time_s,
            "integrated_electric": self.integrated_electric,
            "gcells_per_second": self.gcells_per_second,
        }


@dataclass(frozen=True)
class BenchmarkExecutionEvidence:
    """Execution evidence captured during a benchmark run."""

    benchmark_metadata: dict[str, Any]
    end_to_end: dict[str, Any]
    step_evidence: tuple[StepEvidence, ...] = ()
    converged: bool = False
    stop_reason: str | None = None
    initialization_timing: dict[str, Any] | None = None
    jit_timing: dict[str, Any] | None = None
    steady_state_timing: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "benchmark_metadata": self.benchmark_metadata,
            "end_to_end": self.end_to_end,
            "step_evidence": [e.to_payload() for e in self.step_evidence],
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "initialization_timing": self.initialization_timing,
            "jit_timing": self.jit_timing,
            "steady_state_timing": self.steady_state_timing,
        }


# ---------------------------------------------------------------------------
# Benchmark result bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured benchmark result bundle for one execution configuration."""

    name: str
    metadata: BenchmarkMetadata
    end_to_end_timing: EndToEndTiming
    steady_state_gcells_per_second: float | None = None
    end_to_end_gcells_per_second: float | None = None
    initialization_time_s: float | None = None
    jit_time_s: float | None = None
    stop_reason: str | None = None
    converged: bool = False
    backend_info: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": self.metadata.to_payload(),
            "end_to_end_timing": self.end_to_end_timing.to_payload(),
            "steady_state_gcells_per_second": self.steady_state_gcells_per_second,
            "end_to_end_gcells_per_second": self.end_to_end_gcells_per_second,
            "initialization_time_s": self.initialization_time_s,
            "jit_time_s": self.jit_time_s,
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "backend_info": self.backend_info,
        }


# ---------------------------------------------------------------------------
# Benchmark timer helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkTimer:
    """Hierarchical timer for benchmark phase tracking.

    Provides a context-manager interface for timing initialization, JIT,
    and steady-state phases independently.

    Example
    -------
    >>> timer = BenchmarkTimer()
    >>> with timer.phase("initialization"):
    ...     # initialization work
    ...     timer.mark("grid_resolution")
    ...     # more init work
    ...     timer.mark("scene_materialization")
    >>> init_timing = timer.get_timing("initialization")
    """

    _phase_starts: dict[str, float] = field(default_factory=dict)
    _phase_marks: dict[str, dict[str, float]] = field(default_factory=dict)
    _active_phase: str | None = None
    _active_start: float | None = None
    _wall_time_source: callable = field(default_factory=lambda: time.perf_counter)

    def phase(self, name: str) -> "BenchmarkTimer":
        """Start a named timing phase.

        Can be used as a context manager or called directly.

        Example
        -------
        >>> timer.phase("initialization")  # start
        >>> # work
        >>> timer.mark("grid_resolution")  # sub-marker
        >>> # work
        >>> timer.mark("scene_materialization")  # sub-marker
        >>> timer.phase(None)  # end phase
        """
        self._start_phase(name)
        return self

    def mark(self, label: str) -> None:
        """Record a named sub-marker within the active phase."""
        if self._active_phase is None:
            return
        if self._active_phase not in self._phase_marks:
            self._phase_marks[self._active_phase] = {}
        self._phase_marks[self._active_phase][label] = self._wall_time_source()

    def get_timing(self, phase: str) -> dict[str, float]:
        """Return timing data for a completed phase.

        Returns a dict with 'wall_time_s' and any recorded sub-markers
        as 'label_s' entries.
        """
        if phase not in self._phase_starts:
            return {}
        end = self._phase_marks.get(phase, {})
        phase_end = max(
            end[label] for label in end
        ) if end else self._wall_time_source()
        base = self._phase_starts[phase]
        result = {"wall_time_s": phase_end - base}
        for label, marked in end.items():
            result[f"{label}_s"] = marked - base
        return result

    def _start_phase(self, name: str | None) -> None:
        if name is not None:
            self._phase_starts[name] = self._wall_time_source()
            self._active_phase = name
        else:
            self._active_phase = None
            self._active_start = None

    def __enter__(self) -> "BenchmarkTimer":
        return self

    def __exit__(self, *args: Any) -> None:
        self._start_phase(None)


# ---------------------------------------------------------------------------
# Benchmark runner helpers
# ---------------------------------------------------------------------------


def compute_gcells_per_second(num_cells: int, wall_time_s: float) -> float | None:
    """Compute Gcells/s from cell count and wall time.

    Returns None if wall_time_s is zero or negative.
    """
    if wall_time_s <= 0.0:
        return None
    return (num_cells / wall_time_s) / 1e9


def compute_cells_per_second(num_cells: int, wall_time_s: float) -> float | None:
    """Compute cells/s from cell count and wall time.

    Returns None if wall_time_s is zero or negative.
    """
    if wall_time_s <= 0.0:
        return None
    return num_cells / wall_time_s


def build_benchmark_metadata(
    grid_shape: tuple[int, int, int],
    num_steps: int,
    *,
    precision: BenchmarkPrecision = BenchmarkPrecision.FLOAT32,
    boundary_class: BenchmarkBoundaryClass = BenchmarkBoundaryClass.PEC,
    material_class: BenchmarkMaterialClass = BenchmarkMaterialClass.VACUUM,
    has_bloch: bool = False,
    has_pml: bool = False,
    has_dispersive: bool = False,
    has_anisotropic: bool = False,
    multi_gpu_count: int | None = None,
) -> BenchmarkMetadata:
    """Build a benchmark metadata record from grid and execution parameters.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Number of cells per axis (Nx, Ny, Nz).
    num_steps : int
        Number of timesteps executed in the benchmark.
    precision : BenchmarkPrecision
        Numerical precision used.
    boundary_class : BenchmarkBoundaryClass
        Category of boundary conditions.
    material_class : BenchmarkMaterialClass
        Category of material complexity.
    has_bloch : bool
        Whether Bloch boundaries are present.
    has_pml : bool
        Whether PML absorbers are present.
    has_dispersive : bool
        Whether dispersive media are present.
    has_anisotropic : bool
        Whether anisotropic media are present.
    multi_gpu_count : int, optional
        Number of GPUs if multi-GPU execution.

    Returns
    -------
    BenchmarkMetadata
        Structured metadata for the benchmark run.
    """
    total_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    single_gpu = multi_gpu_count is None or multi_gpu_count <= 1
    return BenchmarkMetadata(
        grid_shape=grid_shape,
        total_cells=total_cells,
        num_steps=num_steps,
        precision=precision,
        boundary_class=boundary_class,
        material_class=material_class,
        single_gpu=single_gpu,
        multi_gpu_count=multi_gpu_count,
        has_bloch=has_bloch,
        has_pml=has_pml,
        has_dispersive=has_dispersive,
        has_anisotropic=has_anisotropic,
    )


def build_steady_state_timing(
    step_times: list[float],
    num_cells: int,
    *,
    warm_up_steps: int = 0,
) -> SteadyStateTiming:
    """Build steady-state timing from per-step wall times.

    Parameters
    ----------
    step_times : list[float]
        Wall-clock times for each step (in seconds).
    num_cells : int
        Total number of grid cells updated per step.
    warm_up_steps : int
        Number of initial steps to exclude from steady-state window.

    Returns
    -------
    SteadyStateTiming
        Structured steady-state metrics excluding initialization overhead.
    """
    if not step_times:
        raise ValueError("step_times must not be empty")

    valid_times = step_times[warm_up_steps:] if warm_up_steps else step_times
    if not valid_times:
        raise ValueError("warm_up_steps too large for step_times")

    total_time = sum(valid_times)
    steps_executed = len(valid_times)
    cells_updated = num_cells * steps_executed

    gcells_s = None
    if total_time > 0.0:
        gcells_s = (cells_updated / total_time) / 1e9

    return SteadyStateTiming(
        wall_time_s=total_time,
        steps_executed=steps_executed,
        cells_updated_per_step=num_cells,
        gcells_per_second=gcells_s,
        min_step_time_s=min(valid_times) if valid_times else None,
        max_step_time_s=max(valid_times) if valid_times else None,
    )


def benchmark_from_execution_log(
    name: str,
    metadata: BenchmarkMetadata,
    execution_log: Any,
    *,
    steady_state_start_step: int = 0,
) -> BenchmarkResult:
    """Build a BenchmarkResult from a RuntimeExecutionLog.

    Parameters
    ----------
    name : str
        Human-readable name for this benchmark.
    metadata : BenchmarkMetadata
        Benchmark metadata record.
    execution_log : RuntimeExecutionLog
        Runtime execution log from diagnostics.
    steady_state_start_step : int
        Step index at which steady-state window begins.

    Returns
    -------
    BenchmarkResult
        Structured benchmark result bundle.
    """
    total_time = execution_log.elapsed_wall_time_s or 0.0
    num_cells = metadata.total_cells

    # Estimate end-to-end Gcells/s
    end_to_end_gcells = None
    if total_time > 0 and metadata.num_steps > 0:
        end_to_end_gcells = (num_cells * metadata.num_steps) / total_time / 1e9

    # Steady-state from step evidence
    steady_state_timing = None
    steady_state_gcells = None
    if execution_log.events:
        step_times = []
        for event in execution_log.events:
            if event.step >= steady_state_start_step:
                step_duration = (
                    (event.elapsed_wall_time_s / max(event.step + 1, 1))
                    if event.step > 0
                    else (total_time / max(metadata.num_steps, 1))
                )
                step_times.append(step_duration)
        if step_times:
            steady_state_timing = build_steady_state_timing(
                step_times, num_cells, warm_up_steps=0
            )
            steady_state_gcells = steady_state_timing.gcells_per_second

    # Extract initialization and JIT from final decision payload
    init_timing = None
    jit_timing = None
    init_time = None
    jit_time = None

    if execution_log.final_decision:
        fd = execution_log.final_decision
        init_time = fd.get("step_time", 0.0) if fd else None

    # Build end-to-end timing
    end_to_end_timing = EndToEndTiming(
        initialization=InitializationTiming(wall_time_s=init_time or 0.0),
        jit=jit_timing,
        steady_state=steady_state_timing,
        total_wall_time_s=total_time,
    )

    return BenchmarkResult(
        name=name,
        metadata=metadata,
        end_to_end_timing=end_to_end_timing,
        steady_state_gcells_per_second=steady_state_gcells,
        end_to_end_gcells_per_second=end_to_end_gcells,
        initialization_time_s=init_time,
        jit_time_s=jit_time,
        stop_reason=str(execution_log.primary_stop_reason.value)
        if execution_log.primary_stop_reason
        else None,
        converged=execution_log.convergence_evidence.triggered
        if execution_log.convergence_evidence
        else False,
        backend_info=None,
    )