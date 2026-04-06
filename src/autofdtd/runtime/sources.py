"""Runtime helpers for compiled source injection."""

from __future__ import annotations

import numpy as np

from autofdtd.compiler.sources import CompiledUniformCurrentSource, compile_uniform_current_source
from autofdtd.grid import ResolvedGrid
from autofdtd.kernels.sources import inject_uniform_current_source
from autofdtd.sources import UniformCurrentSource


def build_uniform_current_runtime(
    source: UniformCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledUniformCurrentSource:
    """Compile runtime placement data for a uniform current source."""

    return compile_uniform_current_source(source, grid=grid)


def apply_uniform_current_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledUniformCurrentSource, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled uniform current sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_uniform_current_source(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic
