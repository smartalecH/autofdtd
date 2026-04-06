"""Source injection helpers and backend conventions for Phase 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from autofdtd.compiler.sources import CompiledUniformCurrentSource

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


def uniform_current_source_kernel_metadata() -> dict[str, Any]:
    """Expose the current-source kernel backend choices for diagnostics."""

    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("source_injection",),
    }


def uniform_current_density(
    compiled_source: CompiledUniformCurrentSource,
    *,
    shape: tuple[int, int, int],
    time: float,
) -> np.ndarray:
    """Return the spatial current-density term on the primal-cell grid."""

    density = np.zeros((*shape, 3), dtype=np.complex128)
    amplitude = compiled_source.amplitude_at_time(time)
    axis = compiled_source.component_axis
    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        density[placement][axis] += amplitude * weight
    return density


def inject_uniform_current_source(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledUniformCurrentSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled uniform current source to the supplied field buffers.

    Phase 1 keeps source injection explicit and staged: electric current adds to ``E``,
    magnetic current adds to ``H``. Full constitutive scaling remains the responsibility of
    the future stepper.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)
    target = updated_electric if compiled_source.field_kind == "electric" else updated_magnetic
    axis = compiled_source.component_axis

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        target[placement][axis] += amplitude * weight

    return updated_electric, updated_magnetic
