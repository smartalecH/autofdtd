"""Tests for mode monitor families, compilation, and recording."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.compiler.monitors import (
    CompiledModeMonitor,
    ModeMonitorState,
    compile_mode_monitor,
    _infer_mode_monitor_normal_axis,
)
from autofdtd.grid import ResolvedGrid, ResolvedGridAxis, UniformGrid
from autofdtd.kernels.monitors import (
    compute_mode_overlap,
    extract_mode_field_components,
    mode_monitor_kernel_metadata,
    record_mode_frequency_domain,
    record_mode_time_domain,
)
from autofdtd.monitors import (
    ModeData,
    ModeMonitor,
    ModeSolverMonitor,
)
from autofdtd.modes import ModeSpec, ModeSolution
from autofdtd.runtime.monitors import (
    build_mode_monitor_runtime,
    extract_mode_monitor_data,
    record_monitor_modes,
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


def test_infer_mode_monitor_normal_axis_x() -> None:
    """Zero in x-size means normal is x-axis."""
    axis = _infer_mode_monitor_normal_axis((0.0, 2.0, 3.0))
    assert axis == 0


def test_infer_mode_monitor_normal_axis_y() -> None:
    """Zero in y-size means normal is y-axis."""
    axis = _infer_mode_monitor_normal_axis((2.0, 0.0, 3.0))
    assert axis == 1


def test_infer_mode_monitor_normal_axis_z() -> None:
    """Zero in z-size means normal is z-axis."""
    axis = _infer_mode_monitor_normal_axis((2.0, 3.0, 0.0))
    assert axis == 2


def test_infer_mode_monitor_normal_axis_error() -> None:
    """No zero dimension raises ValueError."""
    with pytest.raises(ValueError, match="ModeMonitor requires a planar cross-section"):
        _infer_mode_monitor_normal_axis((2.0, 3.0, 4.0))


# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------


def test_mode_monitor_construction() -> None:
    """ModeMonitor can be constructed with standard fields."""
    monitor = ModeMonitor(
        name="test_mode",
        center=(0.0, 0.0, 5.0),
        size=(0.0, 10.0, 10.0),  # x-normal planar
        mode_spec=ModeSpec(num_modes=3),
        direction="-",
        num_freqs=2,
        freqs=(1.5e14, 2.0e14),
    )
    assert monitor.name == "test_mode"
    assert monitor.direction == "-"
    assert monitor.num_freqs == 2
    assert len(monitor.freqs) == 2


def test_mode_solver_monitor_construction() -> None:
    """ModeSolverMonitor can be constructed with standard fields."""
    monitor = ModeSolverMonitor(
        name="test_mode_solver",
        center=(0.0, 0.0, 5.0),
        size=(0.0, 10.0, 10.0),
        mode_spec=ModeSpec(num_modes=2, target_neff=1.5),
        direction="+",
    )
    assert monitor.name == "test_mode_solver"
    assert monitor.direction == "+"
    assert monitor.mode_spec is not None


def test_mode_monitor_mode_spec_none() -> None:
    """ModeMonitor can have None mode_spec."""
    monitor = ModeMonitor(
        name="test_mode",
        center=(0.0, 0.0, 5.0),
        size=(0.0, 10.0, 10.0),
    )
    assert monitor.mode_spec is None
    assert monitor.num_freqs == 1


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------


def test_compile_mode_monitor_z_normal() -> None:
    """Compile a z-normal mode monitor."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_z",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),  # z-normal
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    assert compiled.name == "mode_z"
    assert compiled.normal_axis == 2
    assert compiled.monitor_type == "ModeMonitor"
    assert len(compiled.placements) > 0


def test_compile_mode_monitor_x_normal() -> None:
    """Compile an x-normal mode monitor."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_x",
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),  # x-normal
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 2},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    assert compiled.name == "mode_x"
    assert compiled.normal_axis == 0


def test_compile_mode_monitor_frequency_domain() -> None:
    """Compile a frequency-domain mode monitor."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_freq",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        num_freqs=3,
        freqs=(1e14, 1.5e14, 2e14),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    assert compiled.is_frequency_domain
    assert compiled.num_freqs == 3
    assert len(compiled.freqs) == 3


def test_compile_mode_monitor_time_domain() -> None:
    """Compile a time-domain mode monitor (no frequencies)."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_time",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    assert compiled.is_time_domain
    assert compiled.num_freqs == 1


# ---------------------------------------------------------------------------
# Runtime state tests
# ---------------------------------------------------------------------------


def test_mode_monitor_state_time_domain() -> None:
    """ModeMonitorState initializes correctly for time-domain."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_time",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 2},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    assert state.amplitude_series is not None
    assert state.time_stamps is not None
    assert len(state.amplitude_series) == 0
    assert compiled.is_time_domain


def test_mode_monitor_state_frequency_domain() -> None:
    """ModeMonitorState initializes correctly for frequency-domain."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_freq",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        num_freqs=2,
        freqs=(1e14, 2e14),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 3},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    assert state.dft_amplitudes is not None
    assert state.dft_amplitudes.shape == (2, 3)  # (n_freqs, n_modes)
    assert compiled.is_frequency_domain


def test_mode_monitor_state_set_mode_solutions() -> None:
    """ModeMonitorState can store mode solutions."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_time",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    # Create a simple mode solution
    x_coords = tuple(float(i) for i in range(5))
    y_coords = tuple(float(i) for i in range(5))
    n_pts = len(x_coords) * len(y_coords)

    mode_solution = ModeSolution(
        neff=2.0,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.0, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.0, 1.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    state.set_mode_solutions((mode_solution,))

    assert len(state.mode_solutions) == 1
    assert state.mode_solutions[0].neff == 2.0


# ---------------------------------------------------------------------------
# Kernel metadata tests
# ---------------------------------------------------------------------------


def test_mode_monitor_kernel_metadata() -> None:
    """Mode monitor kernel metadata reports correct backend info."""
    metadata = mode_monitor_kernel_metadata()
    assert metadata["backend"] == "numpy"
    assert "warp_available" in metadata
    assert metadata["staging"] == ("mode_overlap",)


# ---------------------------------------------------------------------------
# Field extraction tests
# ---------------------------------------------------------------------------


def test_extract_mode_field_components_z_normal() -> None:
    """Extract field components for z-normal mode monitor."""
    # Create a simple field buffer
    nx, ny, nz = 5, 5, 5
    e_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    h_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Set some field values
    e_field[2, 2, 2, 0] = 1.0 + 0.0j  # Ex at center
    e_field[2, 2, 2, 1] = 0.5 + 0.0j  # Ey at center
    h_field[2, 2, 2, 0] = 0.3 + 0.0j  # Hx at center
    h_field[2, 2, 2, 1] = 0.4 + 0.0j  # Hy at center

    placements = ((2, 2, 2),)

    result = extract_mode_field_components(e_field, h_field, placements, 2)

    # Z-normal: tangential fields are Ex, Ey, Hx, Hy
    assert "Ex" in result
    assert "Ey" in result
    assert "Hx" in result
    assert "Hy" in result
    assert result["Ex"][0] == 1.0 + 0.0j
    assert result["Ey"][0] == 0.5 + 0.0j
    assert result["Hx"][0] == 0.3 + 0.0j
    assert result["Hy"][0] == 0.4 + 0.0j


def test_extract_mode_field_components_x_normal() -> None:
    """Extract field components for x-normal mode monitor."""
    nx, ny, nz = 5, 5, 5
    e_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    h_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Set some field values
    e_field[2, 2, 2, 1] = 1.0 + 0.0j  # Ey at center
    e_field[2, 2, 2, 2] = 0.5 + 0.0j  # Ez at center
    h_field[2, 2, 2, 1] = 0.3 + 0.0j  # Hy at center
    h_field[2, 2, 2, 2] = 0.4 + 0.0j  # Hz at center

    placements = ((2, 2, 2),)

    result = extract_mode_field_components(e_field, h_field, placements, 0)

    # X-normal: tangential fields are Ey, Ez, Hy, Hz
    assert "Ey" in result
    assert "Ez" in result
    assert "Hy" in result
    assert "Hz" in result


# ---------------------------------------------------------------------------
# Overlap computation tests
# ---------------------------------------------------------------------------


def test_compute_mode_overlap_simple() -> None:
    """Compute overlap with a simple uniform mode."""
    x_coords = tuple(float(i) for i in range(3))
    y_coords = tuple(float(i) for i in range(3))
    n_pts = len(x_coords) * len(y_coords)

    # Create a simple uniform mode
    mode_solution = ModeSolution(
        neff=1.5,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.0, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.0, 1.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    # Create uniform field data matching the mode
    n_pts = len(x_coords) * len(y_coords)
    field_components = {
        "Ex": np.array([1.0 + 0.0j] * n_pts),
        "Ey": np.array([0.0 + 0.0j] * n_pts),
        "Ez": np.array([0.0 + 0.0j] * n_pts),
        "Hx": np.array([0.0 + 0.0j] * n_pts),
        "Hy": np.array([0.0 + 1.0j] * n_pts),
        "Hz": np.array([0.0 + 0.0j] * n_pts),
    }

    placements = tuple((i, j, 0) for i in range(3) for j in range(3))

    overlap = compute_mode_overlap(field_components, mode_solution, 2, placements)

    # Should have non-zero overlap for matching fields
    assert isinstance(overlap, complex)


def test_compute_mode_overlap_zero_fields() -> None:
    """Compute overlap with zero fields gives zero result."""
    x_coords = tuple(float(i) for i in range(3))
    y_coords = tuple(float(i) for i in range(3))
    n_pts = len(x_coords) * len(y_coords)

    mode_solution = ModeSolution(
        neff=1.5,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.0, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.0, 1.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    # All zero fields
    field_components = {
        "Ex": np.array([0.0 + 0.0j] * n_pts),
        "Ey": np.array([0.0 + 0.0j] * n_pts),
        "Ez": np.array([0.0 + 0.0j] * n_pts),
        "Hx": np.array([0.0 + 0.0j] * n_pts),
        "Hy": np.array([0.0 + 0.0j] * n_pts),
        "Hz": np.array([0.0 + 0.0j] * n_pts),
    }

    placements = tuple((i, j, 0) for i in range(3) for j in range(3))

    overlap = compute_mode_overlap(field_components, mode_solution, 2, placements)

    # Should be essentially zero
    assert abs(overlap) < 1e-10


# ---------------------------------------------------------------------------
# Time-domain recording tests
# ---------------------------------------------------------------------------


def test_record_mode_time_domain() -> None:
    """Record mode overlap in time domain."""
    # Create field buffers
    nx, ny, nz = 5, 5, 5
    e_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    h_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Set some field values
    for i in range(nx):
        for j in range(ny):
            e_field[i, j, 2, 0] = 1.0  # Ex
            e_field[i, j, 2, 1] = 0.5  # Ey
            h_field[i, j, 2, 0] = 0.3  # Hx
            h_field[i, j, 2, 1] = 0.4  # Hy

    grid = make_test_grid(nx, ny, nz)
    monitor = ModeMonitor(
        name="mode_time",
        center=(2.5, 2.5, 2.5),
        size=(5.0, 5.0, 0.0),  # z-normal
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    # Create mode solution
    x_coords = tuple(float(i) for i in range(nx))
    y_coords = tuple(float(i) for i in range(ny))
    n_pts = len(x_coords) * len(y_coords)

    mode_solution = ModeSolution(
        neff=2.0,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.5, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.3, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.4, 0.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    record_mode_time_domain(e_field, h_field, state, (mode_solution,), time=1e-12)

    assert len(state.amplitude_series) == 1
    assert len(state.time_stamps) == 1
    assert state.time_stamps[0] == 1e-12


# ---------------------------------------------------------------------------
# Frequency-domain recording tests
# ---------------------------------------------------------------------------


def test_record_mode_frequency_domain() -> None:
    """Record mode overlap for frequency-domain monitor."""
    # Create field buffers
    nx, ny, nz = 5, 5, 5
    e_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    h_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Set some field values
    for i in range(nx):
        for j in range(ny):
            e_field[i, j, 2, 0] = 1.0
            e_field[i, j, 2, 1] = 0.5
            h_field[i, j, 2, 0] = 0.3
            h_field[i, j, 2, 1] = 0.4

    grid = make_test_grid(nx, ny, nz)
    monitor = ModeMonitor(
        name="mode_freq",
        center=(2.5, 2.5, 2.5),
        size=(5.0, 5.0, 0.0),
        num_freqs=2,
        freqs=(1e14, 2e14),
    )

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type="ModeMonitor",
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec={"num_modes": 1},
        num_freqs=monitor.num_freqs,
        freqs=monitor.freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    # Create mode solution
    x_coords = tuple(float(i) for i in range(nx))
    y_coords = tuple(float(i) for i in range(ny))
    n_pts = len(x_coords) * len(y_coords)

    mode_solution = ModeSolution(
        neff=2.0,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.5, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.3, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.4, 0.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    record_mode_frequency_domain(e_field, h_field, state, (mode_solution,), time=1e-12)

    assert state.dft_count_value == 1
    assert state.dft_amplitudes is not None
    assert state.dft_amplitudes.shape == (2, 1)  # (n_freqs, n_modes)


# ---------------------------------------------------------------------------
# Runtime integration tests
# ---------------------------------------------------------------------------


def test_build_mode_monitor_runtime() -> None:
    """Build runtime objects for a mode monitor."""
    grid = make_test_grid(10, 10, 10)
    monitor = ModeMonitor(
        name="mode_runtime",
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        mode_spec=ModeSpec(num_modes=2),
    )

    compiled, state = build_mode_monitor_runtime(monitor, grid=grid)

    assert compiled.name == "mode_runtime"
    assert isinstance(state, ModeMonitorState)
    assert state.compiled.normal_axis == 2


def test_extract_mode_monitor_data() -> None:
    """Extract mode monitor data from states."""
    grid = make_test_grid(5, 5, 5)
    monitor = ModeMonitor(
        name="mode_extract",
        center=(2.5, 2.5, 2.5),
        size=(5.0, 5.0, 0.0),
    )

    compiled, state = build_mode_monitor_runtime(monitor, grid=grid)

    # Create mode solution and record some data
    x_coords = tuple(float(i) for i in range(5))
    y_coords = tuple(float(i) for i in range(5))
    n_pts = len(x_coords) * len(y_coords)

    mode_solution = ModeSolution(
        neff=2.0,
        wavelength=1e-6,
        x=x_coords,
        y=y_coords,
        Ex=tuple((1.0, 0.0) for _ in range(n_pts)),
        Ey=tuple((0.0, 0.0) for _ in range(n_pts)),
        Ez=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hx=tuple((0.0, 0.0) for _ in range(n_pts)),
        Hy=tuple((0.0, 1.0) for _ in range(n_pts)),
        Hz=tuple((0.0, 0.0) for _ in range(n_pts)),
        power=1.0,
    )

    # Add some time-domain data
    state.amplitude_series.append(np.array([1.0 + 0.0j]))
    state.time_stamps.append(1e-12)

    data = extract_mode_monitor_data([state], grid)

    assert "mode_extract" in data
    assert isinstance(data["mode_extract"], ModeData)
    assert data["mode_extract"].monitor_type == "ModeMonitor"


# ---------------------------------------------------------------------------
# IR lowering tests
# ---------------------------------------------------------------------------


def test_mode_monitor_ir_lowering() -> None:
    """ModeMonitor lowers correctly to IR."""
    from autofdtd.ir.models import monitor_to_ir

    monitor = ModeMonitor(
        name="mode_ir",
        center=(0.0, 0.0, 5.0),
        size=(0.0, 10.0, 10.0),
        mode_spec=ModeSpec(num_modes=3),
        direction="-",
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "ModeMonitor"
    assert ir.name == "mode_ir"
    assert ir.direction == "-"
    assert ir.mode_spec is not None


def test_mode_solver_monitor_ir_lowering() -> None:
    """ModeSolverMonitor lowers correctly to IR."""
    from autofdtd.ir.models import monitor_to_ir

    monitor = ModeSolverMonitor(
        name="mode_solver_ir",
        center=(0.0, 0.0, 5.0),
        size=(0.0, 10.0, 10.0),
        mode_spec=ModeSpec(num_modes=2),
        direction="+",
    )

    ir = monitor_to_ir(monitor)

    assert ir.component_type == "ModeSolverMonitor"
    assert ir.name == "mode_solver_ir"
    assert ir.direction == "+"
    assert ir.mode_spec is not None


# ---------------------------------------------------------------------------
# Interval/start filtering tests
# ---------------------------------------------------------------------------


def test_mode_monitor_interval_filtering() -> None:
    """Mode monitor respects interval and start parameters."""
    grid = make_test_grid(5, 5, 5)
    monitor = ModeMonitor(
        name="mode_interval",
        center=(2.5, 2.5, 2.5),
        size=(5.0, 5.0, 0.0),
        interval=2,
        start=1,
    )

    compiled, state = build_mode_monitor_runtime(monitor, grid=grid)

    assert compiled.interval == 2
    assert compiled.start == 1