"""Tests for field monitor families, compilation, and recording."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.compiler.monitors import (
    CompiledFieldMonitor,
    FieldMonitorState,
    compile_field_monitor,
)
from autofdtd.grid import ResolvedGrid, ResolvedGridAxis, UniformGrid
from autofdtd.kernels.monitors import (
    accumulate_flux,
    record_field_frequency_domain,
    record_field_time_domain,
    sample_field_at_points,
)
from autofdtd.monitors import (
    AuxFieldTimeMonitor,
    FieldData,
    FieldMonitor,
    FieldTimeMonitor,
)
from autofdtd.runtime.monitors import (
    build_field_monitor_runtime,
    extract_monitor_data,
    record_monitor_fields,
)


# ---------------------------------------------------------------------------
# Test grid fixture
# ---------------------------------------------------------------------------


def make_test_grid(nx: int = 10, ny: int = 10, nz: int = 10) -> ResolvedGrid:
    """Create a simple test grid."""
    x_boundaries = tuple(float(i) for i in range(nx + 1))
    y_boundaries = tuple(float(i) for i in range(ny + 1))
    z_boundaries = tuple(float(i) for i in range(nz + 1))

    x_axis = ResolvedGridAxis(
        axis="x",
        boundaries=x_boundaries,
    )
    y_axis = ResolvedGridAxis(
        axis="y",
        boundaries=y_boundaries,
    )
    z_axis = ResolvedGridAxis(
        axis="z",
        boundaries=z_boundaries,
    )

    return ResolvedGrid(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 10.0),
        x=x_axis,
        y=y_axis,
        z=z_axis,
    )


# ---------------------------------------------------------------------------
# Monitor compilation tests
# ---------------------------------------------------------------------------


def test_compile_field_monitor_point() -> None:
    """Point field monitor compiles to single cell placement."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="point_monitor",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex", "Ey", "Ez"),
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.name == "point_monitor"
    assert compiled.monitor_type == "FieldTimeMonitor"
    assert len(compiled.placements) == 1
    # Point monitor places at or near the center cell
    assert compiled.placements[0][0] in (4, 5)  # Near center
    assert compiled.fields == ("Ex", "Ey", "Ez")
    assert compiled.interval == 1
    assert compiled.is_time_domain is True


def test_compile_field_monitor_volume() -> None:
    """Volume field monitor compiles to multiple cell placements."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="volume_monitor",
        monitor_type="FieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 2.0),
        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        interval=10,
        start=5,
        grid=grid,
    )

    assert compiled.name == "volume_monitor"
    assert compiled.monitor_type == "FieldMonitor"
    # Volume should include multiple cells
    assert len(compiled.placements) > 1
    assert compiled.interval == 10
    assert compiled.start == 5
    assert compiled.is_time_domain is False
    # FieldMonitor without explicit freqs is NOT frequency_domain
    assert compiled.is_frequency_domain is False


def test_compile_field_monitor_plane() -> None:
    """Planar field monitor compiles to line of cells."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="plane_monitor",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.name == "plane_monitor"
    # Size 0 in z means it's a plane
    assert len(compiled.placements) > 1


def test_field_monitor_frequency_domain() -> None:
    """FieldMonitor with freqs is treated as frequency-domain."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="freq_monitor",
        monitor_type="FieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        freqs=(1e14, 2e14),
        grid=grid,
    )

    assert compiled.is_frequency_domain is True
    assert compiled.num_freqs == 2
    assert len(compiled.freqs) == 2


def test_field_monitor_state_time_domain() -> None:
    """FieldMonitorState initializes correctly for time-domain."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="time_monitor",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    assert state.compiled.name == "time_monitor"
    assert state.time_series is not None
    assert "Ex" in state.time_series
    assert "Ey" in state.time_series
    assert state.time_stamps is not None


def test_field_monitor_state_frequency_domain() -> None:
    """FieldMonitorState initializes correctly for frequency-domain."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="freq_monitor",
        monitor_type="FieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        freqs=(1e14,),
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    assert state.compiled.name == "freq_monitor"
    assert state.time_series is None
    assert state.dft_data is not None
    assert "Ex" in state.dft_data


# ---------------------------------------------------------------------------
# Recording tests
# ---------------------------------------------------------------------------


def test_record_time_domain_fields() -> None:
    """Time-domain recording accumulates field values."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="time_monitor",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    # Create synthetic field buffers
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set a known value at the monitor cell
    electric[5, 5, 5, 0] = 1.0 + 0.5j

    # Record at two time steps
    record_field_time_domain(electric, magnetic, state, time=0.0)
    record_field_time_domain(electric, magnetic, state, time=1e-15)

    assert len(state.time_series["Ex"]) == 2
    assert len(state.time_stamps) == 2
    assert state.time_stamps[0] == 0.0
    assert state.time_stamps[1] == 1e-15


def test_record_time_domain_skip() -> None:
    """Recording respects interval and start parameters."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="time_monitor",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=2,
        start=1,
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Use record_monitor_fields which handles step_index filtering
    # Step 0 - should skip (before start)
    record_monitor_fields([state], electric, magnetic, time=0.0, step_index=0)
    assert len(state.time_series["Ex"]) == 0

    # Step 1 - should record (start)
    record_monitor_fields([state], electric, magnetic, time=1e-15, step_index=1)
    assert len(state.time_series["Ex"]) == 1

    # Step 2 - should skip (interval)
    record_monitor_fields([state], electric, magnetic, time=2e-15, step_index=2)
    assert len(state.time_series["Ex"]) == 1

    # Step 3 - should record (interval)
    record_monitor_fields([state], electric, magnetic, time=3e-15, step_index=3)
    assert len(state.time_series["Ex"]) == 2


def test_record_frequency_domain_fields() -> None:
    """Frequency-domain recording accumulates DFT terms."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="freq_monitor",
        monitor_type="FieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        freqs=(1e15,),
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    electric[5, 5, 5, 0] = 1.0 + 0.0j

    # Record at two time steps
    record_field_frequency_domain(electric, magnetic, state, time=0.0)
    record_field_frequency_domain(electric, magnetic, state, time=1e-15)

    assert state.dft_count == 2
    # The DFT should have accumulated some value
    assert state.dft_data is not None


# ---------------------------------------------------------------------------
# Runtime helpers tests
# ---------------------------------------------------------------------------


def test_build_field_monitor_runtime() -> None:
    """build_field_monitor_runtime creates compiled and state objects."""
    grid = make_test_grid()

    monitor = FieldTimeMonitor(name="test_monitor", size=(0.0, 0.0, 0.0))

    compiled, state = build_field_monitor_runtime(monitor, grid=grid)

    assert isinstance(compiled, CompiledFieldMonitor)
    assert isinstance(state, FieldMonitorState)
    assert compiled.name == "test_monitor"
    assert compiled.is_time_domain is True


def test_record_monitor_fields() -> None:
    """record_monitor_fields updates all active monitors."""
    grid = make_test_grid()

    monitor1 = FieldTimeMonitor(name="m1", size=(0.0, 0.0, 0.0))
    monitor2 = FieldTimeMonitor(name="m2", size=(1.0, 1.0, 1.0))

    _, state1 = build_field_monitor_runtime(monitor1, grid=grid)
    _, state2 = build_field_monitor_runtime(monitor2, grid=grid)

    states = [state1, state2]

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    record_monitor_fields(states, electric, magnetic, time=0.0, step_index=0)

    # Both should have recorded
    assert len(state1.time_series["Ex"]) == 1
    assert len(state2.time_series["Ex"]) == 1


# ---------------------------------------------------------------------------
# Monitor model construction tests
# ---------------------------------------------------------------------------


def test_field_monitor_model() -> None:
    """FieldMonitor model constructs with correct defaults."""
    monitor = FieldMonitor(name="field", size=(1.0, 1.0, 0.0))

    assert monitor.name == "field"
    assert monitor.type == "FieldMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert monitor.interval == 1
    assert monitor.start == 0
    assert monitor.overwrite is True


def test_field_time_monitor_model() -> None:
    """FieldTimeMonitor model constructs with correct defaults."""
    monitor = FieldTimeMonitor(name="time", size=(1.0, 1.0, 0.0))

    assert monitor.name == "time"
    assert monitor.type == "FieldTimeMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert monitor.colocated() is True


def test_aux_field_time_monitor_model() -> None:
    """AuxFieldTimeMonitor model constructs with correct defaults."""
    monitor = AuxFieldTimeMonitor(name="aux", size=(1.0, 1.0, 0.0))

    assert monitor.name == "aux"
    assert monitor.type == "AuxFieldTimeMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def test_field_monitor_custom_fields() -> None:
    """FieldMonitor accepts custom field list."""
    monitor = FieldMonitor(
        name="custom_fields",
        size=(1.0, 1.0, 0.0),
        fields=("Ex", "Hy"),
    )

    assert monitor.fields == ("Ex", "Hy")


# ---------------------------------------------------------------------------
# Flux accumulation test
# ---------------------------------------------------------------------------


def test_accumulate_flux() -> None:
    """accumulate_flux computes surface integral of Poynting vector."""
    # Create a simple grid
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    # Set uniform fields in z-direction
    for i in range(5):
        for j in range(5):
            # E_x = 1, H_y = 0.5 -> S_z = E_x * H_y = 0.5
            electric[i, j, 0, 0] = 1.0  # Ex
            magnetic[i, j, 0, 1] = 0.5  # Hy

    placements = [(i, j, 0) for i in range(5) for j in range(5)]

    flux_plus = accumulate_flux(electric, magnetic, "+", 2, placements)
    flux_minus = accumulate_flux(electric, magnetic, "-", 2, placements)

    # Flux should be positive for + direction
    assert flux_plus > 0
    # Flux should be negative for - direction
    assert flux_minus < 0
    # Magnitudes should be equal
    assert abs(flux_plus + flux_minus) < 1e-10


# ---------------------------------------------------------------------------
# Sample field at points
# ---------------------------------------------------------------------------


def test_sample_field_at_points() -> None:
    """sample_field_at_points extracts field values at indices."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    electric[2, 2, 2, 0] = 1.0 + 2.0j  # Ex at center
    electric[2, 2, 2, 1] = 3.0 + 4.0j  # Ey at center
    magnetic[2, 2, 2, 2] = 0.5j  # Hz at center

    points = ((2, 2, 2),)

    result = sample_field_at_points(
        electric, magnetic, points, ("Ex", "Ey", "Hz")
    )

    assert result["Ex"][0] == 1.0 + 2.0j
    assert result["Ey"][0] == 3.0 + 4.0j
    assert result["Hz"][0] == 0.5j


def test_sample_field_at_multiple_points() -> None:
    """sample_field_at_points handles multiple points."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    electric[0, 0, 0, 0] = 1.0
    electric[4, 4, 4, 0] = 2.0

    points = ((0, 0, 0), (4, 4, 4))

    result = sample_field_at_points(electric, magnetic, points, ("Ex",))

    assert result["Ex"][0] == 1.0
    assert result["Ex"][1] == 2.0


# ---------------------------------------------------------------------------
# Field data conversion tests
# ---------------------------------------------------------------------------


def test_field_monitor_state_to_field_data() -> None:
    """FieldMonitorState.to_field_data produces (real, imag) tuples."""
    grid = make_test_grid()

    compiled = compile_field_monitor(
        name="test",
        monitor_type="FieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    electric[5, 5, 5, 0] = 1.5 + 0.5j

    record_field_time_domain(electric, magnetic, state, time=0.0)

    result = state.to_field_data()

    assert "Ex" in result
    # Should be tuple of (real, imag) tuples
    assert len(result["Ex"]) == 1


def test_field_data_model_construction() -> None:
    """FieldData model stores (real, imag) tuples correctly."""
    data = FieldData(
        monitor_name="test",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.5), (2.0, 0.3)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )

    assert data.monitor_name == "test"
    assert len(data.Ex) == 2
    assert data.Ex[0] == (1.0, 0.5)
    assert data.Ex[1] == (2.0, 0.3)


def test_field_data_json_roundtrip() -> None:
    """FieldData roundtrips through JSON serialization."""
    data = FieldData(
        monitor_name="test",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.5), (2.0, 0.3)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )

    json_text = data.to_json_text()
    restored = FieldData.model_validate_json(json_text)

    assert restored.monitor_name == "test"
    assert len(restored.Ex) == 2
    assert restored.Ex[0] == (1.0, 0.5)


# ---------------------------------------------------------------------------
# Monitor IR lowering tests
# ---------------------------------------------------------------------------


def test_field_monitor_ir_lowering_with_time_domain() -> None:
    """FieldTimeMonitor lowers to FieldTimeMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = FieldTimeMonitor(
        name="time_field",
        size=(2.0, 2.0, 0.0),
        fields=("Ex", "Ey"),
        interval=5,
        start=10,
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "FieldTimeMonitor"
    assert ir.name == "time_field"
    assert ir.size == (2.0, 2.0, 0.0)
    assert ir.fields == ("Ex", "Ey")
    assert ir.interval == 5
    assert ir.start == 10


def test_field_monitor_ir_lowering_aux() -> None:
    """AuxFieldTimeMonitor lowers to AuxFieldTimeMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = AuxFieldTimeMonitor(
        name="aux_monitor",
        size=(1.0, 1.0, 1.0),
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "AuxFieldTimeMonitor"
    assert ir.name == "aux_monitor"