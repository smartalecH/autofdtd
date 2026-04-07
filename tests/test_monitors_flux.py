"""Tests for flux monitor families, compilation, and recording."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.compiler.monitors import (
    CompiledFluxMonitor,
    FluxMonitorState,
    compile_flux_monitor,
    _infer_flux_normal_axis,
)
from autofdtd.grid import ResolvedGrid, ResolvedGridAxis, UniformGrid
from autofdtd.kernels.monitors import (
    accumulate_flux,
    flux_monitor_kernel_metadata,
    record_flux_frequency_domain,
    record_flux_time_domain,
)
from autofdtd.monitors import (
    FluxData,
    FluxMonitor,
    FluxTimeMonitor,
)
from autofdtd.runtime.monitors import (
    build_flux_monitor_runtime,
    extract_flux_monitor_data,
    record_monitor_flux,
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
# Normal axis inference tests
# ---------------------------------------------------------------------------


def test_infer_flux_normal_axis_x() -> None:
    """Zero in x-size means normal is x-axis."""
    axis = _infer_flux_normal_axis((0.0, 2.0, 3.0))
    assert axis == 0


def test_infer_flux_normal_axis_y() -> None:
    """Zero in y-size means normal is y-axis."""
    axis = _infer_flux_normal_axis((2.0, 0.0, 3.0))
    assert axis == 1


def test_infer_flux_normal_axis_z() -> None:
    """Zero in z-size means normal is z-axis."""
    axis = _infer_flux_normal_axis((2.0, 3.0, 0.0))
    assert axis == 2


def test_infer_flux_normal_axis_near_zero() -> None:
    """Near-zero size also counts as zero."""
    axis = _infer_flux_normal_axis((1e-14, 2.0, 3.0))
    assert axis == 0


def test_infer_flux_normal_axis_error() -> None:
    """Non-zero size in all dimensions raises error."""
    with pytest.raises(ValueError, match="2D surface"):
        _infer_flux_normal_axis((2.0, 3.0, 4.0))


# ---------------------------------------------------------------------------
# FluxMonitor model tests
# ---------------------------------------------------------------------------


def test_flux_monitor_model() -> None:
    """FluxMonitor model constructs with correct defaults."""
    monitor = FluxMonitor(
        name="flux_monitor",
        size=(2.0, 2.0, 0.0),  # z-normal surface
    )

    assert monitor.name == "flux_monitor"
    assert monitor.type == "FluxMonitor"
    assert monitor.direction == "+"
    assert monitor.num_freqs == 1
    assert monitor.freqs == ()


def test_flux_monitor_with_freqs() -> None:
    """FluxMonitor accepts frequency points for frequency-domain."""
    monitor = FluxMonitor(
        name="flux_freq",
        size=(2.0, 0.0, 2.0),  # y-normal surface
        direction="-",
        num_freqs=3,
        freqs=(1e14, 2e14, 3e14),
    )

    assert monitor.direction == "-"
    assert monitor.num_freqs == 3
    assert len(monitor.freqs) == 3


def test_flux_time_monitor_model() -> None:
    """FluxTimeMonitor model constructs with correct defaults."""
    monitor = FluxTimeMonitor(
        name="flux_time",
        size=(0.0, 3.0, 3.0),  # x-normal surface
        direction="-",
    )

    assert monitor.name == "flux_time"
    assert monitor.type == "FluxTimeMonitor"
    assert monitor.direction == "-"


# ---------------------------------------------------------------------------
# FluxMonitor compilation tests
# ---------------------------------------------------------------------------


def test_compile_flux_monitor_z_normal() -> None:
    """Z-normal flux monitor compiles with correct axis."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_z",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),  # z-normal
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.name == "flux_z"
    assert compiled.monitor_type == "FluxMonitor"
    assert compiled.normal_axis == 2  # z
    assert compiled.direction == "+"
    assert compiled.is_time_domain is False
    assert compiled.is_frequency_domain is False


def test_compile_flux_monitor_y_normal() -> None:
    """Y-normal flux monitor compiles correctly."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_y",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 0.0, 10.0),  # y-normal
        direction="-",
        interval=5,
        start=2,
        grid=grid,
    )

    assert compiled.normal_axis == 1  # y
    assert compiled.direction == "-"
    assert compiled.interval == 5
    assert compiled.start == 2


def test_compile_flux_monitor_x_normal() -> None:
    """X-normal flux monitor compiles correctly."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_x",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),  # x-normal
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    assert compiled.normal_axis == 0  # x


def test_compile_flux_monitor_frequency_domain() -> None:
    """FluxMonitor with freqs is frequency-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_freq",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        freqs=(1e14, 2e14),
        grid=grid,
    )

    assert compiled.is_frequency_domain is True
    assert compiled.num_freqs == 2
    assert len(compiled.freqs) == 2


def test_compile_flux_time_monitor() -> None:
    """FluxTimeMonitor compiles as time-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_time",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=10,
        start=5,
        grid=grid,
    )

    assert compiled.monitor_type == "FluxTimeMonitor"
    assert compiled.is_time_domain is True
    assert compiled.is_frequency_domain is False


def test_compile_flux_monitor_invalid_size() -> None:
    """FluxMonitor with non-2D size raises error."""
    grid = make_test_grid()

    with pytest.raises(ValueError, match="2D surface"):
        compile_flux_monitor(
            name="invalid",
            monitor_type="FluxMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 3.0, 4.0),  # Not a surface
            direction="+",
            interval=1,
            start=0,
            grid=grid,
        )


# ---------------------------------------------------------------------------
# FluxMonitorState tests
# ---------------------------------------------------------------------------


def test_flux_monitor_state_time_domain() -> None:
    """FluxMonitorState initializes for time-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_time",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    assert state.flux_series is not None
    assert state.time_stamps is not None
    assert state.dft_flux is None


def test_flux_monitor_state_frequency_domain() -> None:
    """FluxMonitorState initializes for frequency-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_freq",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        freqs=(1e14, 2e14),
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    assert state.flux_series is None
    assert state.time_stamps is None
    assert state.dft_flux is not None
    assert len(state.dft_flux) == 2


# ---------------------------------------------------------------------------
# Flux integration (Poynting vector) tests
# ---------------------------------------------------------------------------


def test_accumulate_flux_z_normal_plus() -> None:
    """Flux through z-normal surface with + direction."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    # Set uniform fields: E_x = 1, H_y = 0.5
    # S_z = E_x * H_y = 0.5
    for i in range(5):
        for j in range(5):
            electric[i, j, 0, 0] = 1.0  # Ex
            magnetic[i, j, 0, 1] = 0.5  # Hy

    placements = [(i, j, 0) for i in range(5) for j in range(5)]

    flux = accumulate_flux(electric, magnetic, "+", 2, placements)

    # Expected: sum of 25 cells * 0.5 = 12.5
    assert abs(flux - 12.5) < 1e-10


def test_accumulate_flux_z_normal_minus() -> None:
    """Flux through z-normal surface with - direction gives negative."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    for i in range(5):
        for j in range(5):
            electric[i, j, 0, 0] = 1.0
            magnetic[i, j, 0, 1] = 0.5

    placements = [(i, j, 0) for i in range(5) for j in range(5)]

    flux_plus = accumulate_flux(electric, magnetic, "+", 2, placements)
    flux_minus = accumulate_flux(electric, magnetic, "-", 2, placements)

    assert flux_plus > 0
    assert flux_minus < 0
    assert abs(flux_plus + flux_minus) < 1e-10


def test_accumulate_flux_x_normal() -> None:
    """Flux through x-normal surface."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    # E_y = 1, H_z = 0.5 -> S_x = E_y * H_z - E_z * H_y = 0.5
    for j in range(5):
        for k in range(5):
            electric[0, j, k, 1] = 1.0  # Ey
            magnetic[0, j, k, 2] = 0.5  # Hz

    placements = [(0, j, k) for j in range(5) for k in range(5)]

    flux = accumulate_flux(electric, magnetic, "+", 0, placements)

    # Expected: 25 * 0.5 = 12.5
    assert abs(flux - 12.5) < 1e-10


def test_accumulate_flux_y_normal() -> None:
    """Flux through y-normal surface."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    # E_z = 1, H_x = 0.5 -> S_y = E_z * H_x - E_x * H_z = 0.5
    for i in range(5):
        for k in range(5):
            electric[i, 0, k, 2] = 1.0  # Ez
            magnetic[i, 0, k, 0] = 0.5  # Hx

    placements = [(i, 0, k) for i in range(5) for k in range(5)]

    flux = accumulate_flux(electric, magnetic, "+", 1, placements)

    # Expected: 25 * 0.5 = 12.5
    assert abs(flux - 12.5) < 1e-10


def test_accumulate_flux_with_complex_fields() -> None:
    """Flux accumulation works with complex field values (uses real part)."""
    electric = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    magnetic = np.zeros((3, 3, 3, 3), dtype=np.complex128)

    # Set complex fields - use z-normal to get ex * hy
    for i in range(3):
        for j in range(3):
            electric[i, j, 0, 0] = 1.0 + 0.0j  # Ex
            magnetic[i, j, 0, 1] = 0.5 + 0.0j  # Hy

    placements = [(i, j, 0) for i in range(3) for j in range(3)]

    flux = accumulate_flux(electric, magnetic, "+", 2, placements)

    # Uses real part: sum of 9 cells * 0.5 = 4.5
    assert abs(flux - 4.5) < 1e-10


# ---------------------------------------------------------------------------
# Time-domain recording tests
# ---------------------------------------------------------------------------


def test_record_flux_time_domain() -> None:
    """Time-domain flux recording accumulates flux values."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_time",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    # Create synthetic field buffers
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set uniform fields
    for i in range(10):
        for j in range(10):
            electric[i, j, 5, 0] = 1.0  # Ex
            magnetic[i, j, 5, 1] = 0.5  # Hy

    # Record at two timesteps
    record_flux_time_domain(electric, magnetic, state, time=0.0)
    record_flux_time_domain(electric, magnetic, state, time=1e-15)

    assert len(state.flux_series) == 2
    assert len(state.time_stamps) == 2
    assert state.time_stamps[0] == 0.0
    assert state.time_stamps[1] == 1e-15


def test_record_flux_time_domain_skip() -> None:
    """Recording respects interval and start parameters."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_time",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=2,
        start=1,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Step 0 - should skip (before start)
    record_monitor_flux([state], electric, magnetic, time=0.0, step_index=0)
    assert len(state.flux_series) == 0

    # Step 1 - should record (start)
    record_monitor_flux([state], electric, magnetic, time=1e-15, step_index=1)
    assert len(state.flux_series) == 1

    # Step 2 - should skip (interval)
    record_monitor_flux([state], electric, magnetic, time=2e-15, step_index=2)
    assert len(state.flux_series) == 1

    # Step 3 - should record (interval)
    record_monitor_flux([state], electric, magnetic, time=3e-15, step_index=3)
    assert len(state.flux_series) == 2


# ---------------------------------------------------------------------------
# Frequency-domain recording tests
# ---------------------------------------------------------------------------


def test_record_flux_frequency_domain() -> None:
    """Frequency-domain flux recording accumulates DFT terms."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_freq",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        freqs=(1e15,),
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set uniform fields
    for i in range(10):
        for j in range(10):
            electric[i, j, 5, 0] = 1.0
            magnetic[i, j, 5, 1] = 0.5

    # Record at two timesteps
    record_flux_frequency_domain(electric, magnetic, state, time=0.0)
    record_flux_frequency_domain(electric, magnetic, state, time=1e-15)

    assert state.dft_count == 2
    assert state.dft_flux is not None


# ---------------------------------------------------------------------------
# Runtime helpers tests
# ---------------------------------------------------------------------------


def test_build_flux_monitor_runtime_time_domain() -> None:
    """build_flux_monitor_runtime creates correct runtime objects."""
    grid = make_test_grid()

    monitor = FluxTimeMonitor(
        name="flux_time",
        size=(10.0, 10.0, 0.0),
        direction="-",
    )

    compiled, state = build_flux_monitor_runtime(monitor, grid=grid)

    assert isinstance(compiled, CompiledFluxMonitor)
    assert isinstance(state, FluxMonitorState)
    assert compiled.name == "flux_time"
    assert compiled.direction == "-"
    assert compiled.is_time_domain is True


def test_build_flux_monitor_runtime_frequency_domain() -> None:
    """build_flux_monitor_runtime handles frequency-domain."""
    grid = make_test_grid()

    monitor = FluxMonitor(
        name="flux_freq",
        size=(10.0, 10.0, 0.0),
        freqs=(1e14, 2e14),
    )

    compiled, state = build_flux_monitor_runtime(monitor, grid=grid)

    assert compiled.is_frequency_domain is True
    assert compiled.num_freqs == 2


def test_record_monitor_flux() -> None:
    """record_monitor_flux updates all active flux monitors."""
    grid = make_test_grid()

    monitor1 = FluxTimeMonitor(name="flux1", size=(10.0, 10.0, 0.0))
    monitor2 = FluxTimeMonitor(name="flux2", size=(0.0, 10.0, 10.0))

    _, state1 = build_flux_monitor_runtime(monitor1, grid=grid)
    _, state2 = build_flux_monitor_runtime(monitor2, grid=grid)

    states = [state1, state2]

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Set some fields for flux computation
    for i in range(10):
        for j in range(10):
            electric[i, j, 5, 0] = 1.0
            magnetic[i, j, 5, 1] = 0.5

    record_monitor_flux(states, electric, magnetic, time=0.0, step_index=0)

    # Both should have recorded
    assert len(state1.flux_series) == 1
    assert len(state2.flux_series) == 1


# ---------------------------------------------------------------------------
# Extract flux data tests
# ---------------------------------------------------------------------------


def test_extract_flux_monitor_data_time_domain() -> None:
    """extract_flux_monitor_data produces FluxData for time-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_time",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    for i in range(10):
        for j in range(10):
            electric[i, j, 5, 0] = 1.0
            magnetic[i, j, 5, 1] = 0.5

    record_flux_time_domain(electric, magnetic, state, time=0.0)
    record_flux_time_domain(electric, magnetic, state, time=1e-15)

    result = extract_flux_monitor_data([state], grid)

    assert "flux_time" in result
    assert isinstance(result["flux_time"], FluxData)
    assert result["flux_time"].monitor_type == "FluxTimeMonitor"
    assert len(result["flux_time"].flux) == 2


def test_extract_flux_monitor_data_frequency_domain() -> None:
    """extract_flux_monitor_data produces FluxData for frequency-domain."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_freq",
        monitor_type="FluxMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        freqs=(1e14,),
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    for i in range(10):
        for j in range(10):
            electric[i, j, 5, 0] = 1.0
            magnetic[i, j, 5, 1] = 0.5

    record_flux_frequency_domain(electric, magnetic, state, time=0.0)
    record_flux_frequency_domain(electric, magnetic, state, time=1e-15)

    result = extract_flux_monitor_data([state], grid)

    assert "flux_freq" in result
    assert isinstance(result["flux_freq"], FluxData)
    assert result["flux_freq"].monitor_type == "FluxMonitor"
    # flux should be complex (real, imag) tuples for frequency-domain
    assert result["flux_freq"].flux is not None
    # DFT accumulates with phase, so the flux should be complex


# ---------------------------------------------------------------------------
# FluxData model tests
# ---------------------------------------------------------------------------


def test_flux_data_model() -> None:
    """FluxData model stores flux values correctly."""
    data = FluxData(
        monitor_name="test",
        monitor_type="FluxMonitor",
        flux=(1.0, 2.0, 3.0),
        t=(0.0, 1e-15, 2e-15),
    )

    assert data.monitor_name == "test"
    assert len(data.flux) == 3
    assert data.flux[0] == 1.0
    assert data.t[0] == 0.0


def test_flux_data_json_roundtrip() -> None:
    """FluxData roundtrips through JSON serialization."""
    data = FluxData(
        monitor_name="test",
        monitor_type="FluxTimeMonitor",
        flux=(1.5, 2.5),
        t=(0.0, 1e-15),
    )

    json_text = data.to_json_text()
    restored = FluxData.model_validate_json(json_text)

    assert restored.monitor_name == "test"
    assert len(restored.flux) == 2


# ---------------------------------------------------------------------------
# Kernel metadata test
# ---------------------------------------------------------------------------


def test_flux_monitor_kernel_metadata() -> None:
    """flux_monitor_kernel_metadata returns expected backend info."""
    metadata = flux_monitor_kernel_metadata()

    assert metadata["backend"] == "numpy"
    assert "warp_available" in metadata
    assert metadata["staging"] == ("flux_integration",)


# ---------------------------------------------------------------------------
# Monitor IR lowering tests
# ---------------------------------------------------------------------------


def test_flux_monitor_ir_lowering() -> None:
    """FluxMonitor lowers to FluxMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = FluxMonitor(
        name="flux_monitor",
        size=(2.0, 2.0, 0.0),
        direction="-",
        interval=5,
        start=10,
        num_freqs=2,
        freqs=(1e14, 2e14),
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "FluxMonitor"
    assert ir.name == "flux_monitor"
    assert ir.size == (2.0, 2.0, 0.0)
    assert ir.direction == "-"
    assert ir.interval == 5
    assert ir.start == 10
    assert ir.num_freqs == 2


def test_flux_time_monitor_ir_lowering() -> None:
    """FluxTimeMonitor lowers to FluxTimeMonitorIR correctly."""
    from autofdtd.ir import monitor_to_ir

    monitor = FluxTimeMonitor(
        name="flux_time",
        size=(0.0, 3.0, 3.0),
        direction="+",
        interval=1,
        start=0,
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "FluxTimeMonitor"
    assert ir.name == "flux_time"
    assert ir.size == (0.0, 3.0, 3.0)
    assert ir.direction == "+"


# ---------------------------------------------------------------------------
# Energy flow validation tests
# ---------------------------------------------------------------------------


def test_flux_reflects_poynting_direction() -> None:
    """Flux sign correctly reflects Poynting vector direction."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    # Forward propagating wave: E_x > 0, H_y > 0 -> S_z > 0
    for i in range(5):
        for j in range(5):
            electric[i, j, 2, 0] = 1.0
            magnetic[i, j, 2, 1] = 0.5

    placements = [(i, j, 2) for i in range(5) for j in range(5)]

    flux_z_plus = accumulate_flux(electric, magnetic, "+", 2, placements)
    flux_z_minus = accumulate_flux(electric, magnetic, "-", 2, placements)

    # Positive direction sees positive Poynting
    assert flux_z_plus > 0
    # Negative direction sees negative Poynting
    assert flux_z_minus < 0


def test_flux_zero_when_no_fields() -> None:
    """Flux is zero when fields are zero."""
    electric = np.zeros((5, 5, 5, 3), dtype=np.complex128)
    magnetic = np.zeros((5, 5, 5, 3), dtype=np.complex128)

    placements = [(i, j, 0) for i in range(5) for j in range(5)]

    flux = accumulate_flux(electric, magnetic, "+", 2, placements)

    assert abs(flux) < 1e-20


def test_flux_time_history_tracks_propagation() -> None:
    """Flux time-series correctly tracks field propagation."""
    grid = make_test_grid()

    compiled = compile_flux_monitor(
        name="flux_tracking",
        monitor_type="FluxTimeMonitor",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        direction="+",
        interval=1,
        start=0,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    # Simulate a pulse propagating through
    # Note: z-placement is at index 4 (center 4.5) since that's nearest to z=5.0
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Record at multiple times with changing field amplitude
    for t_idx, amp in enumerate([0.0, 0.5, 1.0, 0.5, 0.0]):
        for i in range(10):
            for j in range(10):
                electric[i, j, 4, 0] = amp  # z=4 is where placement is
                magnetic[i, j, 4, 1] = amp * 0.5

        record_flux_time_domain(electric, magnetic, state, time=float(t_idx))

    assert len(state.flux_series) == 5
    # Flux should follow amplitude squared pattern: 0, small, max, small, 0
    assert abs(state.flux_series[0]) < 1e-20  # 0.0 amplitude
    assert state.flux_series[2] > state.flux_series[1]  # peak at middle
    assert abs(state.flux_series[4]) < 1e-20  # back to 0


def test_symmetric_flux_surface_integral() -> None:
    """Flux integral is additive over surface cells."""
    electric = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    magnetic = np.zeros((3, 3, 3, 3), dtype=np.complex128)

    # Constant field across surface
    for i in range(3):
        for j in range(3):
            electric[i, j, 0, 0] = 1.0
            magnetic[i, j, 0, 1] = 1.0

    # Full surface
    full_placements = [(i, j, 0) for i in range(3) for j in range(3)]
    flux_full = accumulate_flux(electric, magnetic, "+", 2, full_placements)

    # Quarter surface
    quarter_placements = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    flux_quarter = accumulate_flux(electric, magnetic, "+", 2, quarter_placements)

    # Full should be 9x the quarter (9 cells vs 4 cells, but full has 9)
    # Actually: full = 9 * 1.0 * 1.0 = 9, quarter = 4 * 1.0 * 1.0 = 4
    assert abs(flux_full - 9.0) < 1e-10
    assert abs(flux_quarter - 4.0) < 1e-10
    assert abs(flux_full / flux_quarter - 9.0 / 4.0) < 1e-10


# ---------------------------------------------------------------------------
# SimulationData integration test
# ---------------------------------------------------------------------------


def test_flux_monitor_in_simulation_data() -> None:
    """FluxData can be registered in SimulationData."""
    from autofdtd.monitors import SimulationData

    sim_data = SimulationData(simulation_name="test_sim")
    flux_data = FluxData(
        monitor_name="flux1",
        monitor_type="FluxTimeMonitor",
        flux=(1.0, 2.0),
        t=(0.0, 1e-15),
    )

    sim_data.register_flux_monitor("flux1", flux_data)

    assert "flux1" in sim_data
    assert sim_data.flux1.flux == (1.0, 2.0)

    # Dictionary access
    assert sim_data["flux1"].flux == (1.0, 2.0)
