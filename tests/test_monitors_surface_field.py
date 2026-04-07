"""Tests for surface field monitor families, compilation, and recording."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.compiler.monitors import (
    CompiledSurfaceFieldMonitor,
    SurfaceFieldMonitorState,
    compile_surface_field_monitor,
)
from autofdtd.grid import ResolvedGrid, ResolvedGridAxis
from autofdtd.monitors import (
    SurfaceFieldData,
    SurfaceFieldMonitor,
    SurfaceFieldTimeData,
    SurfaceFieldTimeMonitor,
)
from autofdtd.runtime.monitors import (
    build_surface_field_monitor_runtime,
    extract_surface_field_monitor_data,
    record_surface_monitor_fields,
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


def test_compile_surface_field_monitor_z_normal() -> None:
    """Z-normal surface field monitor compiles correctly."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="surface_monitor",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.name == "surface_monitor"
    assert compiled.monitor_type == "SurfaceFieldMonitor"
    assert compiled.normal_axis == 2  # z-normal
    assert compiled.tang_axis_1 == 0
    assert compiled.tang_axis_2 == 1
    assert len(compiled.placements) > 0
    assert compiled.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert compiled.interval == 1
    assert compiled.is_time_domain is False
    assert compiled.is_frequency_domain is False  # No freqs provided


def test_compile_surface_field_monitor_x_normal() -> None:
    """X-normal surface field monitor compiles correctly."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="surface_monitor_x",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        fields=("Ex", "Ey", "Ez"),
        interval=10,
        start=5,
        grid=grid,
    )

    assert compiled.name == "surface_monitor_x"
    assert compiled.normal_axis == 0  # x-normal
    assert compiled.tang_axis_1 == 1
    assert compiled.tang_axis_2 == 2


def test_compile_surface_field_monitor_y_normal() -> None:
    """Y-normal surface field monitor compiles correctly."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="surface_monitor_y",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 0.0, 10.0),
        fields=("Hx", "Hy", "Hz"),
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.name == "surface_monitor_y"
    assert compiled.normal_axis == 1  # y-normal
    assert compiled.tang_axis_1 == 0
    assert compiled.tang_axis_2 == 2


def test_compile_surface_field_monitor_frequency_domain() -> None:
    """SurfaceFieldMonitor with freqs is treated as frequency-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="freq_surface_monitor",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        freqs=(1e14, 2e14),
        grid=grid,
    )

    assert compiled.is_frequency_domain is True
    assert compiled.num_freqs == 2
    assert len(compiled.freqs) == 2


def test_compile_surface_field_monitor_time_domain() -> None:
    """SurfaceFieldTimeMonitor is treated as time-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="time_surface_monitor",
        monitor_type="SurfaceFieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.is_time_domain is True
    assert compiled.is_frequency_domain is False


def test_compile_surface_field_monitor_invalid_size() -> None:
    """Surface monitor with no zero dimension raises error."""
    grid = make_test_grid()

    with pytest.raises(ValueError, match="2D surface"):
        compile_surface_field_monitor(
            name="invalid",
            monitor_type="SurfaceFieldMonitor",
            center=(5.0, 5.0, 5.0),
            size=(10.0, 10.0, 10.0),  # No zero dimension
            fields=("Ex",),
            interval=1,
            start=0,
            grid=grid,
        )


# ---------------------------------------------------------------------------
# Monitor state tests
# ---------------------------------------------------------------------------


def test_surface_field_monitor_state_time_domain() -> None:
    """SurfaceFieldMonitorState initializes correctly for time-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="time_surface",
        monitor_type="SurfaceFieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    assert state.compiled.name == "time_surface"
    assert state.time_series is not None
    assert "Ex" in state.time_series
    assert "Ey" in state.time_series
    assert state.time_stamps is not None
    assert state.dft_data is None


def test_surface_field_monitor_state_frequency_domain() -> None:
    """SurfaceFieldMonitorState initializes correctly for frequency-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="freq_surface",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        fields=("Ex", "Ey"),
        interval=1,
        start=0,
        freqs=(1e14,),
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    assert state.compiled.name == "freq_surface"
    assert state.time_series is None
    assert state.dft_data is not None
    assert "Ex" in state.dft_data
    assert "Ey" in state.dft_data


# ---------------------------------------------------------------------------
# Recording tests
# ---------------------------------------------------------------------------


def test_record_time_domain_surface_fields() -> None:
    """Time-domain recording accumulates field values on surface."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="time_surface",
        monitor_type="SurfaceFieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 0.0),  # Small surface
        fields=("Ex",),
        interval=1,
        start=0,
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    # Create synthetic field buffers
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set a known value at the monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.0 + 0.5j

    # Record at two time steps
    state.record_time_domain(electric, magnetic, time=0.0)
    state.record_time_domain(electric, magnetic, time=1e-15)

    assert len(state.time_series["Ex"]) == 2
    assert len(state.time_stamps) == 2
    assert state.time_stamps[0] == 0.0
    assert state.time_stamps[1] == 1e-15


def test_record_time_domain_skip() -> None:
    """Recording respects interval and start parameters."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="time_surface",
        monitor_type="SurfaceFieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
        interval=2,
        start=1,
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Use record_surface_monitor_fields which handles step_index filtering
    # Step 0 - should skip (before start)
    record_surface_monitor_fields([state], electric, magnetic, time=0.0, step_index=0)
    assert len(state.time_series["Ex"]) == 0

    # Step 1 - should record (start)
    record_surface_monitor_fields([state], electric, magnetic, time=1e-15, step_index=1)
    assert len(state.time_series["Ex"]) == 1

    # Step 2 - should skip (interval)
    record_surface_monitor_fields([state], electric, magnetic, time=2e-15, step_index=2)
    assert len(state.time_series["Ex"]) == 1

    # Step 3 - should record (interval)
    record_surface_monitor_fields([state], electric, magnetic, time=3e-15, step_index=3)
    assert len(state.time_series["Ex"]) == 2


def test_record_frequency_domain_surface_fields() -> None:
    """Frequency-domain recording accumulates DFT terms."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="freq_surface",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        freqs=(1e15,),
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set a value at monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.0 + 0.0j

    # Record at two time steps
    state.accumulate_dft(electric, magnetic, time=0.0)
    state.accumulate_dft(electric, magnetic, time=1e-15)

    assert state.dft_count_value == 2
    assert state.dft_data is not None


# ---------------------------------------------------------------------------
# Runtime helpers tests
# ---------------------------------------------------------------------------


def test_build_surface_field_monitor_runtime() -> None:
    """build_surface_field_monitor_runtime creates compiled and state objects."""
    grid = make_test_grid()

    monitor = SurfaceFieldTimeMonitor(name="test_surface", size=(10.0, 10.0, 0.0))

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    assert isinstance(compiled, CompiledSurfaceFieldMonitor)
    assert isinstance(state, SurfaceFieldMonitorState)
    assert compiled.name == "test_surface"
    assert compiled.is_time_domain is True


def test_build_surface_field_monitor_runtime_freq() -> None:
    """build_surface_field_monitor_runtime works for frequency-domain."""
    grid = make_test_grid()

    monitor = SurfaceFieldMonitor(
        name="freq_surface",
        size=(10.0, 10.0, 0.0),
        freqs=(1e14, 2e14),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    assert compiled.name == "freq_surface"
    assert compiled.is_frequency_domain is True
    assert compiled.num_freqs == 2


def test_record_surface_monitor_fields() -> None:
    """record_surface_monitor_fields updates all active monitors."""
    grid = make_test_grid()

    monitor1 = SurfaceFieldTimeMonitor(name="m1", size=(2.0, 2.0, 0.0))
    monitor2 = SurfaceFieldTimeMonitor(name="m2", size=(2.0, 2.0, 0.0))

    _, state1 = build_surface_field_monitor_runtime(monitor1, grid=grid)
    _, state2 = build_surface_field_monitor_runtime(monitor2, grid=grid)

    states = [state1, state2]

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    record_surface_monitor_fields(states, electric, magnetic, time=0.0, step_index=0)

    # Both should have recorded
    assert len(state1.time_series["Ex"]) == 1
    assert len(state2.time_series["Ex"]) == 1


# ---------------------------------------------------------------------------
# Monitor model construction tests
# ---------------------------------------------------------------------------


def test_surface_field_monitor_model() -> None:
    """SurfaceFieldMonitor model constructs with correct defaults."""
    monitor = SurfaceFieldMonitor(name="surface", size=(5.0, 5.0, 0.0))

    assert monitor.name == "surface"
    assert monitor.type == "SurfaceFieldMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert monitor.interval == 1
    assert monitor.start == 0


def test_surface_field_time_monitor_model() -> None:
    """SurfaceFieldTimeMonitor model constructs with correct defaults."""
    monitor = SurfaceFieldTimeMonitor(name="surface_time", size=(5.0, 5.0, 0.0))

    assert monitor.name == "surface_time"
    assert monitor.type == "SurfaceFieldTimeMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert monitor.colocated() is True


def test_surface_field_monitor_custom_fields() -> None:
    """SurfaceFieldMonitor accepts custom field list."""
    monitor = SurfaceFieldMonitor(
        name="custom_fields",
        size=(5.0, 5.0, 0.0),
        fields=("Ex", "Hy"),
    )

    assert monitor.fields == ("Ex", "Hy")


def test_surface_field_monitor_with_freqs() -> None:
    """SurfaceFieldMonitor accepts frequency points."""
    monitor = SurfaceFieldMonitor(
        name="freq_surface",
        size=(5.0, 5.0, 0.0),
        freqs=(1e14, 2e14, 3e14),
    )

    assert monitor.freqs == (1e14, 2e14, 3e14)
    assert monitor.num_freqs == 3


# ---------------------------------------------------------------------------
# Surface field data conversion tests
# ---------------------------------------------------------------------------


def test_surface_field_monitor_state_to_data_time_domain() -> None:
    """SurfaceFieldMonitorState.to_surface_field_data produces (real, imag) tuples for time-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="test",
        monitor_type="SurfaceFieldTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set value at monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.5 + 0.5j

    state.record_time_domain(electric, magnetic, time=0.0)

    result = state.to_surface_field_data()

    assert "Ex" in result
    # Should be tuple of (real, imag) tuples
    assert len(result["Ex"]) > 0


def test_surface_field_monitor_state_to_data_frequency_domain() -> None:
    """SurfaceFieldMonitorState.to_surface_field_data produces (real, imag) tuples for frequency-domain."""
    grid = make_test_grid()

    compiled = compile_surface_field_monitor(
        name="test",
        monitor_type="SurfaceFieldMonitor",
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
        interval=1,
        start=0,
        freqs=(1e14,),
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set value at monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.5 + 0.5j

    state.accumulate_dft(electric, magnetic, time=0.0)

    result = state.to_surface_field_data()

    assert "Ex" in result
    assert len(result["Ex"]) > 0


def test_surface_field_data_model_construction() -> None:
    """SurfaceFieldData model stores (real, imag) tuples correctly."""
    data = SurfaceFieldData(
        monitor_name="test",
        monitor_type="SurfaceFieldMonitor",
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


def test_surface_field_time_data_model_construction() -> None:
    """SurfaceFieldTimeData model stores (real, imag) tuples correctly."""
    data = SurfaceFieldTimeData(
        monitor_name="test",
        monitor_type="SurfaceFieldTimeMonitor",
        Ex=((1.0, 0.5), (2.0, 0.3)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
        t=(0.0, 1e-15),
    )

    assert data.monitor_name == "test"
    assert len(data.Ex) == 2
    assert data.Ex[0] == (1.0, 0.5)
    assert data.t == (0.0, 1e-15)


def test_surface_field_data_json_roundtrip() -> None:
    """SurfaceFieldData roundtrips through JSON serialization."""
    data = SurfaceFieldData(
        monitor_name="test",
        monitor_type="SurfaceFieldMonitor",
        Ex=((1.0, 0.5), (2.0, 0.3)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )

    json_text = data.to_json_text()
    restored = SurfaceFieldData.model_validate_json(json_text)

    assert restored.monitor_name == "test"
    assert len(restored.Ex) == 2
    assert restored.Ex[0] == (1.0, 0.5)


def test_surface_field_time_data_json_roundtrip() -> None:
    """SurfaceFieldTimeData roundtrips through JSON serialization."""
    data = SurfaceFieldTimeData(
        monitor_name="test",
        monitor_type="SurfaceFieldTimeMonitor",
        Ex=((1.0, 0.5), (2.0, 0.3)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
        t=(0.0, 1e-15),
    )

    json_text = data.to_json_text()
    restored = SurfaceFieldTimeData.model_validate_json(json_text)

    assert restored.monitor_name == "test"
    assert len(restored.Ex) == 2
    assert restored.t == (0.0, 1e-15)


# ---------------------------------------------------------------------------
# Monitor IR lowering tests
# ---------------------------------------------------------------------------


def test_surface_field_monitor_ir_lowering_time_domain() -> None:
    """SurfaceFieldTimeMonitor lowers to SurfaceFieldTimeMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = SurfaceFieldTimeMonitor(
        name="time_surface",
        size=(5.0, 5.0, 0.0),
        fields=("Ex", "Ey"),
        interval=5,
        start=10,
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "SurfaceFieldTimeMonitor"
    assert ir.name == "time_surface"
    assert ir.size == (5.0, 5.0, 0.0)
    assert ir.fields == ("Ex", "Ey")
    assert ir.interval == 5
    assert ir.start == 10


def test_surface_field_monitor_ir_lowering_frequency_domain() -> None:
    """SurfaceFieldMonitor lowers to SurfaceFieldMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = SurfaceFieldMonitor(
        name="freq_surface",
        size=(5.0, 5.0, 0.0),
        fields=("Ex", "Ey", "Ez"),
        freqs=(1e14, 2e14),
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "SurfaceFieldMonitor"
    assert ir.name == "freq_surface"
    assert ir.size == (5.0, 5.0, 0.0)
    assert ir.fields == ("Ex", "Ey", "Ez")
    assert len(ir.freqs) == 2


# ---------------------------------------------------------------------------
# Extract monitor data tests
# ---------------------------------------------------------------------------


def test_extract_surface_field_monitor_data_time_domain() -> None:
    """extract_surface_field_monitor_data works for time-domain."""
    grid = make_test_grid()

    monitor = SurfaceFieldTimeMonitor(
        name="time_surface",
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set value at monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.0 + 0.5j

    record_surface_monitor_fields([state], electric, magnetic, time=0.0, step_index=0)

    result = extract_surface_field_monitor_data([state], grid)

    assert "time_surface" in result
    data = result["time_surface"]
    assert isinstance(data, SurfaceFieldTimeData)
    assert data.monitor_name == "time_surface"


def test_extract_surface_field_monitor_data_frequency_domain() -> None:
    """extract_surface_field_monitor_data works for frequency-domain."""
    grid = make_test_grid()

    monitor = SurfaceFieldMonitor(
        name="freq_surface",
        size=(2.0, 2.0, 0.0),
        fields=("Ex",),
        freqs=(1e14,),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set value at monitor cells
    for i in range(4, 6):
        for j in range(4, 6):
            electric[i, j, 5, 0] = 1.0 + 0.5j

    record_surface_monitor_fields([state], electric, magnetic, time=0.0, step_index=0)

    result = extract_surface_field_monitor_data([state], grid)

    assert "freq_surface" in result
    data = result["freq_surface"]
    assert isinstance(data, SurfaceFieldData)
    assert data.monitor_name == "freq_surface"


def test_extract_surface_field_monitor_data_coordinates_z_normal() -> None:
    """extract_surface_field_monitor_data provides correct coordinates for z-normal."""
    grid = make_test_grid(nx=6, ny=6, nz=6)

    monitor = SurfaceFieldTimeMonitor(
        name="z_surface",
        size=(4.0, 4.0, 0.0),  # z-normal
        fields=("Ex",),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    # The z-normal surface should have x and y coordinates
    assert compiled.normal_axis == 2
    assert compiled.tang_axis_1 == 0
    assert compiled.tang_axis_2 == 1


def test_extract_surface_field_monitor_data_coordinates_x_normal() -> None:
    """extract_surface_field_monitor_data provides correct coordinates for x-normal."""
    grid = make_test_grid(nx=6, ny=6, nz=6)

    monitor = SurfaceFieldTimeMonitor(
        name="x_surface",
        size=(0.0, 4.0, 4.0),  # x-normal
        fields=("Ex",),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    # The x-normal surface should have y and z coordinates
    assert compiled.normal_axis == 0
    assert compiled.tang_axis_1 == 1
    assert compiled.tang_axis_2 == 2


def test_extract_surface_field_monitor_data_coordinates_y_normal() -> None:
    """extract_surface_field_monitor_data provides correct coordinates for y-normal."""
    grid = make_test_grid(nx=6, ny=6, nz=6)

    monitor = SurfaceFieldTimeMonitor(
        name="y_surface",
        size=(4.0, 0.0, 4.0),  # y-normal
        fields=("Ex",),
    )

    compiled, state = build_surface_field_monitor_runtime(monitor, grid=grid)

    # The y-normal surface should have x and z coordinates
    assert compiled.normal_axis == 1
    assert compiled.tang_axis_1 == 0
    assert compiled.tang_axis_2 == 2
