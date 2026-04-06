from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    CustomSourceTime,
    GridSpec,
    Simulation,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.compiler import compile_uniform_current_source
from autofdtd.examples import uniform_current_source_snapshot
from autofdtd.ir import UniformCurrentSourceIR, simulation_to_ir
from autofdtd.kernels import inject_uniform_current_source, uniform_current_density
from autofdtd.runtime import apply_uniform_current_sources, build_uniform_current_runtime


def _resolved_grid():
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )
    return simulation.resolved_grid()


def test_uniform_current_source_exposes_phase1_placement_semantics() -> None:
    source = UniformCurrentSource(
        center=(0.0, 0.0, 0.5),
        size=(0.0, 2.0, 0.0),
        polarization="Ez",
        current_amplitude_definition="total",
        source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
        name=" drive ",
    )

    assert source.name == "drive"
    assert source.field_kind == "electric"
    assert source.component_axis == 2
    assert source.placement_kind == "line"
    assert source.zero_size_axes == (0, 2)
    assert source.support_bounds == ((0.0, -1.0, 0.5), (0.0, 1.0, 0.5))


def test_simulation_normalizes_uniform_current_source_payloads() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        sources=(
            {
                "type": "UniformCurrentSource",
                "center": [0.0, 0.0, 0.5],
                "size": [0.0, 2.0, 0.0],
                "polarization": "Ez",
                "source_time": {
                    "type": "ContinuousWave",
                    "freq0": 1.0,
                    "fwidth": 1.0,
                    "offset": 2.5,
                },
            },
        ),
    )

    normalized = simulation.sources[0]
    assert normalized["type"] == "UniformCurrentSource"
    assert normalized["polarization"] == "Ez"
    assert normalized["source_time"]["type"] == "ContinuousWave"


def test_uniform_current_source_lowers_to_typed_ir() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        sources=(
            UniformCurrentSource(
                center=(0.0, 0.0, 0.5),
                size=(0.0, 2.0, 0.0),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
                name="drive",
            ),
        ),
    )

    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.sources[0], UniformCurrentSourceIR)
    source_ir = simulation_ir.sources[0]
    assert source_ir.placement_kind == "line"
    assert source_ir.component_axis == 2
    assert source_ir.field_kind == "electric"
    assert source_ir.source_time.component_type == "ContinuousWave"


def test_uniform_current_source_compilation_and_injection_kernel() -> None:
    source = UniformCurrentSource(
        center=(0.0, 0.0, 0.5),
        size=(0.0, 2.0, 0.0),
        polarization="Ez",
        current_amplitude_definition="total",
        source_time=CustomSourceTime(
            freq0=1.0,
            fwidth=1.0,
            amplitude=2.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
    )
    compiled = compile_uniform_current_source(source, grid=_resolved_grid())

    assert compiled.placement_kind == "line"
    assert compiled.support_point_count == 4
    assert compiled.placements == ((1, 1, 2), (1, 2, 2), (2, 1, 2), (2, 2, 2))
    assert compiled.placement_weights == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert compiled.amplitude_scale == pytest.approx(0.5)

    density = uniform_current_density(compiled, shape=(4, 4, 4), time=0.0)
    assert density.shape == (4, 4, 4, 3)
    assert np.sum(density[..., 2]) == pytest.approx(2.0)
    assert np.count_nonzero(np.abs(density[..., 2])) == 4

    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    injected_electric, injected_magnetic = inject_uniform_current_source(
        electric,
        magnetic,
        compiled,
        time=0.0,
        dt=0.25,
    )
    assert np.allclose(injected_magnetic, 0.0)
    assert injected_electric[1, 1, 2, 2] == pytest.approx(0.125)
    assert injected_electric[2, 2, 2, 2] == pytest.approx(0.125)


def test_uniform_current_source_runtime_batches_sources_by_field_kind() -> None:
    electric_source = build_uniform_current_runtime(
        UniformCurrentSource(
            center=(0.5, 0.5, 0.5),
            size=(0.0, 0.0, 0.0),
            polarization="Ey",
            source_time=CustomSourceTime(
                freq0=1.0,
                fwidth=1.0,
                amplitude=1.0,
                offset=0.0,
                time_samples=(0.0, 1.0),
                envelope_values=((1.0, 0.0), (1.0, 0.0)),
            ),
        ),
        grid=_resolved_grid(),
    )
    magnetic_source = build_uniform_current_runtime(
        UniformCurrentSource(
            center=(0.5, 0.5, 0.5),
            size=(0.0, 0.0, 0.0),
            polarization="Hx",
            source_time=CustomSourceTime(
                freq0=1.0,
                fwidth=1.0,
                amplitude=1.0,
                offset=0.0,
                time_samples=(0.0, 1.0),
                envelope_values=((1.0, 0.0), (1.0, 0.0)),
            ),
        ),
        grid=_resolved_grid(),
    )

    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    injected_electric, injected_magnetic = apply_uniform_current_sources(
        electric,
        magnetic,
        (electric_source, magnetic_source),
        time=0.0,
        dt=0.5,
    )

    assert injected_electric[2, 2, 2, 1] == pytest.approx(0.5)
    assert injected_magnetic[2, 2, 2, 0] == pytest.approx(0.5)


def test_uniform_current_source_example_snapshot_is_stable() -> None:
    snapshot = uniform_current_source_snapshot()

    assert snapshot["placement_kind"] == "line"
    assert snapshot["support_point_count"] == 4
    assert snapshot["amplitude_scale"] == pytest.approx(0.5)
