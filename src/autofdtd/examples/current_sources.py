"""Examples for Phase 1 current-source placement and injection."""

from __future__ import annotations

import numpy as np

from autofdtd.api import (
    ContinuousWave,
    CustomCurrentSource,
    GridSpec,
    PointDipole,
    Simulation,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.runtime import (
    apply_custom_current_sources,
    apply_point_dipole_sources,
    apply_uniform_current_sources,
    build_custom_current_source_runtime,
    build_point_dipole_runtime,
    build_uniform_current_runtime,
)


def uniform_current_source_snapshot() -> dict[str, object]:
    """Return a compact snapshot of source placement and one injection step."""

    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
        sources=(
            UniformCurrentSource(
                center=(0.0, 0.0, 0.5),
                size=(0.0, 2.0, 0.0),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
                name="sheet_drive",
            ),
        ),
    )
    compiled = build_uniform_current_runtime(
        simulation.sources[0],
        grid=simulation.resolved_grid(),
    )
    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    injected_electric, _ = apply_uniform_current_sources(
        electric,
        magnetic,
        (compiled,),
        time=0.0,
        dt=0.25,
    )
    return {
        "placement_kind": compiled.placement_kind,
        "support_point_count": compiled.support_point_count,
        "placements": compiled.placements,
        "amplitude_scale": compiled.amplitude_scale,
        "sum_abs_injected_e": float(np.sum(np.abs(injected_electric[..., 2]))),
    }


def point_dipole_snapshot() -> dict[str, object]:
    """Return a compact snapshot of point dipole placement and one injection step."""

    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
        sources=(
            PointDipole(
                center=(2.0, 2.0, 2.0),
                polarization="Ez",
                source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
                name="point_dipole",
            ),
        ),
    )
    compiled = build_point_dipole_runtime(
        simulation.sources[0],
        grid=simulation.resolved_grid(),
    )
    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    injected_electric, _ = apply_point_dipole_sources(
        electric,
        magnetic,
        (compiled,),
        time=0.0,
        dt=0.25,
    )
    return {
        "placement_kind": compiled.placement_kind,
        "support_point_count": compiled.support_point_count,
        "placements": compiled.placements,
        "sum_abs_injected_e": float(np.sum(np.abs(injected_electric[..., 2]))),
    }


def custom_current_source_snapshot() -> dict[str, object]:
    """Return a compact snapshot of custom current source placement and injection."""

    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
        sources=(
            CustomCurrentSource(
                center=(0.5, 0.5, 0.5),
                size=(0.0, 0.0, 0.0),
                source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=2.0),
                name="custom_current",
                e_fields={"Ex": (1.0,), "Ey": (1.0,), "Ez": (1.0,)},
            ),
        ),
    )
    compiled = build_custom_current_source_runtime(
        simulation.sources[0],
        grid=simulation.resolved_grid(),
    )
    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    injected_electric, _ = apply_custom_current_sources(
        electric,
        magnetic,
        (compiled,),
        time=0.0,
        dt=0.25,
    )
    return {
        "support_point_count": compiled.support_point_count,
        "placements": compiled.placements,
        "has_electric": compiled.has_electric,
        "sum_abs_injected_e": float(np.sum(np.abs(injected_electric))),
    }
