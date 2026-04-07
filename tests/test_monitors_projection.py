"""Tests for projection and far-field monitor families."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.monitors import (
    DiffractionMonitor,
    DirectivityMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    SimulationData,
    monitor_model_from_value,
)
from autofdtd.monitors.models import (
    DiffractionData,
    DirectivityData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
)


# ---------------------------------------------------------------------------
# Projection monitor model construction
# ---------------------------------------------------------------------------


def test_field_projection_angle_monitor_defaults() -> None:
    """FieldProjectionAngleMonitor has sensible defaults."""
    monitor = FieldProjectionAngleMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
    )
    assert monitor.type == "FieldProjectionAngleMonitor"
    assert monitor.normal_vector == (0.0, 0.0, 1.0)
    assert monitor.projection_distance == 1e5
    assert monitor.num_freqs == 1
    assert monitor.phi == (-90.0, 90.0, 181)
    assert monitor.theta == (0.0, 180.0, 181)


def test_field_projection_cartesian_monitor_defaults() -> None:
    """FieldProjectionCartesianMonitor has sensible defaults."""
    monitor = FieldProjectionCartesianMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
    )
    assert monitor.type == "FieldProjectionCartesianMonitor"
    assert monitor.normal_vector == (0.0, 0.0, 1.0)
    assert monitor.projection_distance == 1e5
    assert monitor.x == (-50.0, 50.0, 201)
    assert monitor.y == (-50.0, 50.0, 201)


def test_field_projection_kspace_monitor_defaults() -> None:
    """FieldProjectionKSpaceMonitor has sensible defaults."""
    monitor = FieldProjectionKSpaceMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
    )
    assert monitor.type == "FieldProjectionKSpaceMonitor"
    assert monitor.num_k == 1
    assert monitor.kx == (-10.0, 10.0, 21)
    assert monitor.ky == (-10.0, 10.0, 21)


def test_diffraction_monitor_defaults() -> None:
    """DiffractionMonitor has sensible defaults."""
    monitor = DiffractionMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
    )
    assert monitor.type == "DiffractionMonitor"
    assert monitor.normal_vector == (0.0, 0.0, 1.0)


def test_directivity_monitor_defaults() -> None:
    """DirectivityMonitor has sensible defaults."""
    monitor = DirectivityMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
    )
    assert monitor.type == "DirectivityMonitor"
    assert monitor.projection_distance == 1e5


def test_projection_monitor_custom_params() -> None:
    """Projection monitors accept custom parameters."""
    monitor = FieldProjectionAngleMonitor(
        size=(0, 2.0, 2.0),
        freqs=[1e14, 2e14, 3e14],
        normal_vector=(1.0, 0.0, 0.0),
        projection_distance=5e4,
        phi=(-45.0, 45.0, 91),
        theta=(0.0, 90.0, 91),
    )
    assert monitor.normal_vector == (1.0, 0.0, 0.0)
    assert monitor.projection_distance == 5e4
    assert len(monitor.freqs) == 3
    assert monitor.phi == (-45.0, 45.0, 91)


def test_projection_monitor_factory() -> None:
    """monitor_model_from_value constructs concrete projection monitor models."""
    data = {
        "type": "FieldProjectionAngleMonitor",
        "name": "angle_proj",
        "size": [0, 1.0, 1.0],
        "freqs": [2e14],
    }
    monitor = monitor_model_from_value(data)
    assert isinstance(monitor, FieldProjectionAngleMonitor)
    assert monitor.name == "angle_proj"
    assert monitor.projection_distance == 1e5


# ---------------------------------------------------------------------------
# Projection monitor data types
# ---------------------------------------------------------------------------


def test_field_projection_angle_data() -> None:
    """FieldProjectionAngleData stores projection results."""
    data = FieldProjectionAngleData(
        monitor_name="angle_proj",
        monitor_type="FieldProjectionAngleMonitor",
        e_theta=(((1.0, 0.0),),),
        e_phi=(((0.5, 0.0),),),
        phi=(0.0, 90.0, 2),
        theta=(0.0, 180.0, 3),
    )
    assert data.monitor_name == "angle_proj"
    assert data.type == "FieldProjectionAngleData"
    assert data.e_theta is not None


def test_field_projection_cartesian_data() -> None:
    """FieldProjectionCartesianData stores projection results."""
    data = FieldProjectionCartesianData(
        monitor_name="cart_proj",
        monitor_type="FieldProjectionCartesianMonitor",
        ex=(((1.0, 0.0), (2.0, 0.0)),),
        ey=None,
        ez=None,
        x=(-50.0, 50.0, 2),
        y=(-50.0, 50.0, 2),
    )
    assert data.monitor_name == "cart_proj"
    assert data.type == "FieldProjectionCartesianData"
    assert data.ex is not None


def test_field_projection_kspace_data() -> None:
    """FieldProjectionKSpaceData stores projection results."""
    data = FieldProjectionKSpaceData(
        monitor_name="kspace_proj",
        monitor_type="FieldProjectionKSpaceMonitor",
        ex=(((1.0, 0.0),),),
        ey=None,
        ez=None,
        kx=(-10.0, 10.0, 2),
        ky=(-10.0, 10.0, 2),
    )
    assert data.monitor_name == "kspace_proj"
    assert data.type == "FieldProjectionKSpaceData"
    assert data.ex is not None


def test_diffraction_data() -> None:
    """DiffractionData stores diffraction orders."""
    data = DiffractionData(
        monitor_name="diff",
        monitor_type="DiffractionMonitor",
        # Shape: (num_orders, num_freqs) = (2, 1), each with (real, imag) complex tuple
        orders=(((1.0, 0.0),), ((0.5, 0.0),)),
        mx=(0, 1),
        my=(0, -1),
    )
    assert data.monitor_name == "diff"
    assert data.type == "DiffractionData"
    assert data.orders is not None


def test_directivity_data() -> None:
    """DirectivityData stores directivity values."""
    data = DirectivityData(
        monitor_name="dir",
        monitor_type="DirectivityMonitor",
        directivity=((0.0, 1.0, 2.0), (1.0, 2.0, 3.0)),
        theta=(0.0, 90.0, 2),
        phi=(-90.0, 90.0, 3),
    )
    assert data.monitor_name == "dir"
    assert data.type == "DirectivityData"
    assert data.directivity is not None


# ---------------------------------------------------------------------------
# SimulationData registration for projection monitors
# ---------------------------------------------------------------------------


def test_simulation_data_register_field_projection_angle_monitor() -> None:
    """SimulationData.register_field_projection_angle_monitor stores data."""
    sim_data = SimulationData()

    data = FieldProjectionAngleData(
        monitor_name="angle",
        monitor_type="FieldProjectionAngleMonitor",
        e_theta=(((1.0, 0.0),),),
        e_phi=(((0.5, 0.0),),),
        phi=(0.0, 90.0, 2),
        theta=(0.0, 180.0, 3),
    )
    sim_data.register_field_projection_angle_monitor("angle", data)

    assert "angle" in sim_data
    retrieved = sim_data["angle"]
    assert isinstance(retrieved, FieldProjectionAngleData)


def test_simulation_data_register_field_projection_cartesian_monitor() -> None:
    """SimulationData.register_field_projection_cartesian_monitor stores data."""
    sim_data = SimulationData()

    data = FieldProjectionCartesianData(
        monitor_name="cart",
        monitor_type="FieldProjectionCartesianMonitor",
        ex=(((1.0, 0.0),),),
        x=(-50.0, 50.0, 2),
        y=(-50.0, 50.0, 2),
    )
    sim_data.register_field_projection_cartesian_monitor("cart", data)

    assert "cart" in sim_data
    retrieved = sim_data["cart"]
    assert isinstance(retrieved, FieldProjectionCartesianData)


def test_simulation_data_register_field_projection_kspace_monitor() -> None:
    """SimulationData.register_field_projection_kspace_monitor stores data."""
    sim_data = SimulationData()

    data = FieldProjectionKSpaceData(
        monitor_name="kspace",
        monitor_type="FieldProjectionKSpaceMonitor",
        ex=(((1.0, 0.0),),),
        kx=(-10.0, 10.0, 2),
        ky=(-10.0, 10.0, 2),
    )
    sim_data.register_field_projection_kspace_monitor("kspace", data)

    assert "kspace" in sim_data
    retrieved = sim_data["kspace"]
    assert isinstance(retrieved, FieldProjectionKSpaceData)


def test_simulation_data_register_diffraction_monitor() -> None:
    """SimulationData.register_diffraction_monitor stores data."""
    sim_data = SimulationData()

    data = DiffractionData(
        monitor_name="diff",
        monitor_type="DiffractionMonitor",
        # Shape: (num_orders, num_freqs) = (2, 1)
        orders=(((1.0, 0.0),), ((0.5, 0.0),)),
        mx=(0, 1),
        my=(0, -1),
    )
    sim_data.register_diffraction_monitor("diff", data)

    assert "diff" in sim_data
    retrieved = sim_data["diff"]
    assert isinstance(retrieved, DiffractionData)


def test_simulation_data_register_directivity_monitor() -> None:
    """SimulationData.register_directivity_monitor stores data."""
    sim_data = SimulationData()

    data = DirectivityData(
        monitor_name="dir",
        monitor_type="DirectivityMonitor",
        directivity=((0.0, 1.0),),
        theta=(0.0, 90.0, 2),
        phi=(-90.0, 90.0, 3),
    )
    sim_data.register_directivity_monitor("dir", data)

    assert "dir" in sim_data
    retrieved = sim_data["dir"]
    assert isinstance(retrieved, DirectivityData)


# ---------------------------------------------------------------------------
# Projection monitor IR lowering
# ---------------------------------------------------------------------------


def test_field_projection_angle_monitor_ir_lowering() -> None:
    """FieldProjectionAngleMonitor lowers to FieldProjectionAngleMonitorIR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            FieldProjectionAngleMonitor(
                name="angle_proj",
                size=(0, 2.0, 2.0),
                freqs=[2e14],
                normal_vector=(0.0, 0.0, 1.0),
                projection_distance=1e5,
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    assert len(ir.monitors) == 1
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "FieldProjectionAngleMonitor"
    assert monitor_ir.name == "angle_proj"
    assert monitor_ir.normal_vector == (0.0, 0.0, 1.0)
    assert monitor_ir.projection_distance == 1e5


def test_field_projection_cartesian_monitor_ir_lowering() -> None:
    """FieldProjectionCartesianMonitor lowers to Cartesian IR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            FieldProjectionCartesianMonitor(
                name="cart_proj",
                size=(0, 2.0, 2.0),
                freqs=[2e14],
                x=(-25.0, 25.0, 101),
                y=(-25.0, 25.0, 101),
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "FieldProjectionCartesianMonitor"
    assert monitor_ir.name == "cart_proj"
    assert monitor_ir.x == (-25.0, 25.0, 101)


def test_field_projection_kspace_monitor_ir_lowering() -> None:
    """FieldProjectionKSpaceMonitor lowers to KSpace IR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            FieldProjectionKSpaceMonitor(
                name="kspace_proj",
                size=(0, 2.0, 2.0),
                freqs=[2e14],
                kx=(-5.0, 5.0, 11),
                ky=(-5.0, 5.0, 11),
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "FieldProjectionKSpaceMonitor"
    assert monitor_ir.name == "kspace_proj"
    assert monitor_ir.kx == (-5.0, 5.0, 11)


def test_diffraction_monitor_ir_lowering() -> None:
    """DiffractionMonitor lowers to DiffractionMonitorIR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            DiffractionMonitor(
                name="diff",
                size=(0, 2.0, 2.0),
                freqs=[2e14],
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "DiffractionMonitor"
    assert monitor_ir.name == "diff"


def test_directivity_monitor_ir_lowering() -> None:
    """DirectivityMonitor lowers to DirectivityMonitorIR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            DirectivityMonitor(
                name="dir",
                size=(0, 2.0, 2.0),
                freqs=[2e14],
                projection_distance=5e4,
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "DirectivityMonitor"
    assert monitor_ir.name == "dir"
    assert monitor_ir.projection_distance == 5e4


# ---------------------------------------------------------------------------
# Projection monitor compilation
# ---------------------------------------------------------------------------


def test_compile_projection_monitor() -> None:
    """compile_projection_monitor produces valid compiled monitors."""
    from autofdtd.compiler.monitors import (
        CompiledProjectionMonitor,
        ProjectionMonitorState,
        compile_projection_monitor,
    )
    from autofdtd.grid import GridSpec, UniformGrid

    spec = GridSpec(
        grid_x=UniformGrid(dl=0.1),
        grid_y=UniformGrid(dl=0.1),
        grid_z=UniformGrid(dl=0.1),
    )
    grid = spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))

    compiled = compile_projection_monitor(
        name="test_proj",
        monitor_type="FieldProjectionAngleMonitor",
        center=(0.0, 0.0, 0.0),
        size=(0, 1.0, 1.0),
        normal_vector=(0.0, 0.0, 1.0),
        projection_distance=1e5,
        interval=1,
        start=0,
        num_freqs=2,
        freqs=(1e14, 2e14),
        phi=(-90.0, 90.0, 181),
        theta=(0.0, 180.0, 181),
        grid=grid,
    )

    assert isinstance(compiled, CompiledProjectionMonitor)
    assert compiled.name == "test_proj"
    assert compiled.monitor_type == "FieldProjectionAngleMonitor"
    assert compiled.normal_axis == 0
    assert compiled.num_freqs == 2
    assert len(compiled.placements) > 0

    state = ProjectionMonitorState(compiled=compiled)
    assert state.compiled is compiled
    assert state.dft_e is not None
    assert state.dft_h is not None


def test_projection_monitor_dft_accumulation() -> None:
    """ProjectionMonitorState accumulates DFT terms correctly."""
    from autofdtd.compiler.monitors import (
        ProjectionMonitorState,
        compile_projection_monitor,
    )
    from autofdtd.grid import GridSpec, UniformGrid

    spec = GridSpec(
        grid_x=UniformGrid(dl=0.1),
        grid_y=UniformGrid(dl=0.1),
        grid_z=UniformGrid(dl=0.1),
    )
    grid = spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))

    compiled = compile_projection_monitor(
        name="test_proj",
        monitor_type="FieldProjectionAngleMonitor",
        center=(0.0, 0.0, 0.0),
        size=(0, 1.0, 1.0),
        num_freqs=1,
        freqs=(1e14,),
        grid=grid,
    )

    state = ProjectionMonitorState(compiled=compiled)

    # Simulate a field buffer
    nx, ny, nz = grid.shape
    electric_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    magnetic_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Add some field values at monitor placements
    for idx in compiled.placements:
        electric_field[idx] = (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
        magnetic_field[idx] = (0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j)

    # Accumulate DFT at two times
    state.accumulate_dft(electric_field, magnetic_field, time=0.0)
    state.accumulate_dft(electric_field, magnetic_field, time=1e-14)

    assert state.dft_count_value == 2

    # DFT should have accumulated values
    data_dict = state.to_projection_data()
    assert data_dict["dft_e"] is not None
    assert data_dict["dft_h"] is not None


# ---------------------------------------------------------------------------
# DiffractionMonitor validation tests
# ---------------------------------------------------------------------------


def test_diffraction_monitor_unit_vector_validation() -> None:
    """DiffractionMonitor validates normal_vector is a unit vector."""
    import math

    # Valid unit vector should work
    monitor = DiffractionMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
        normal_vector=(0.0, 0.0, 1.0),
    )
    assert monitor.normal_vector == (0.0, 0.0, 1.0)

    # Non-unit vector should raise
    with pytest.raises(ValueError, match="normal_vector must be a unit vector"):
        DiffractionMonitor(
            size=(0, 1.0, 1.0),
            freqs=[2e14],
            normal_vector=(1.0, 1.0, 1.0),  # Not normalized
        )

    # Zero vector is allowed as placeholder
    monitor_zero = DiffractionMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
        normal_vector=(0.0, 0.0, 0.0),
    )
    assert monitor_zero.normal_vector == (0.0, 0.0, 0.0)


def test_diffraction_monitor_num_freqs_validation() -> None:
    """DiffractionMonitor validates num_freqs >= 1."""
    # Valid num_freqs
    monitor = DiffractionMonitor(
        size=(0, 1.0, 1.0),
        freqs=[2e14],
        num_freqs=3,
    )
    assert monitor.num_freqs == 3

    # Invalid num_freqs (0)
    with pytest.raises(ValueError, match="num_freqs must be at least 1"):
        DiffractionMonitor(
            size=(0, 1.0, 1.0),
            freqs=[2e14],
            num_freqs=0,
        )

    # Invalid num_freqs (negative)
    with pytest.raises(ValueError, match="num_freqs must be at least 1"):
        DiffractionMonitor(
            size=(0, 1.0, 1.0),
            freqs=[2e14],
            num_freqs=-1,
        )


def test_diffraction_monitor_freqs_validation() -> None:
    """DiffractionMonitor validates freqs are positive."""
    # Valid freqs
    monitor = DiffractionMonitor(
        size=(0, 1.0, 1.0),
        freqs=[1e14, 2e14, 3e14],
    )
    assert len(monitor.freqs) == 3

    # Zero frequency should raise
    with pytest.raises(ValueError, match="freqs must be positive"):
        DiffractionMonitor(
            size=(0, 1.0, 1.0),
            freqs=[0.0],
        )

    # Negative frequency should raise
    with pytest.raises(ValueError, match="freqs must be positive"):
        DiffractionMonitor(
            size=(0, 1.0, 1.0),
            freqs=[-1e14],
        )


# ---------------------------------------------------------------------------
# Diffraction order computation kernel tests
# ---------------------------------------------------------------------------


def test_compute_diffraction_orders_basic() -> None:
    """compute_diffraction_orders produces order indices and shapes."""
    from autofdtd.kernels.monitors import compute_diffraction_orders

    # Create minimal DFT data
    n_pts = 10
    n_freqs = 2
    dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    # Add uniform field
    for i in range(n_pts):
        dft_e[i, :, 0] = 1.0 + 0.0j

    placements = tuple((0, i, 5) for i in range(n_pts))
    freqs = (1e14, 2e14)

    orders, mx, my = compute_diffraction_orders(
        dft_e=dft_e,
        dft_h=dft_h,
        placements=placements,
        normal_axis=0,  # x-normal surface
        freqs=freqs,
        period_y=1e-6,  # 1 micron period
        period_z=1e-6,
        medium_eps=1.0 + 0.0j,
        num_orders=1,
    )

    # Should have (2*1+1)^2 = 9 orders
    assert orders.shape[0] == 9
    assert orders.shape[1] == 2  # 2 frequencies
    assert len(mx) == 9
    assert len(my) == 9


def test_compute_diffraction_orders_no_period() -> None:
    """compute_diffraction_orders handles missing periods gracefully."""
    from autofdtd.kernels.monitors import compute_diffraction_orders

    n_pts = 10
    n_freqs = 1
    dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    placements = tuple((0, i, 5) for i in range(n_pts))
    freqs = (1e14,)

    # No period provided - should return zeros
    orders, mx, my = compute_diffraction_orders(
        dft_e=dft_e,
        dft_h=dft_h,
        placements=placements,
        normal_axis=0,
        freqs=freqs,
        period_y=None,  # No period
        period_z=None,
        num_orders=2,
    )

    # Should still return shape
    assert orders.shape[0] == 25  # (2*2+1)^2 = 25
    assert orders.shape[1] == 1
    # All zeros since no period
    assert np.allclose(orders, 0.0)


def test_compute_diffraction_orders_evanescent() -> None:
    """compute_diffraction_orders suppresses evanescent orders."""
    from autofdtd.kernels.monitors import compute_diffraction_orders

    n_pts = 10
    n_freqs = 1
    dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    # Add some field
    for i in range(n_pts):
        dft_e[i, 0, 0] = 1.0 + 0.0j

    placements = tuple((0, i, 5) for i in range(n_pts))
    freqs = (1e14,)

    # Very small period - all high orders should be evanescent
    orders, mx, my = compute_diffraction_orders(
        dft_e=dft_e,
        dft_h=dft_h,
        placements=placements,
        normal_axis=0,
        freqs=freqs,
        period_y=1e-9,  # Very small - high spatial freq
        period_z=1e-9,
        medium_eps=1.0 + 0.0j,
        num_orders=3,
    )

    # Should still have output (25 orders), but most should be zero
    assert orders.shape[0] == 49  # (2*3+1)^2 = 49
    # Higher order terms should be suppressed as evanescent
    # (0, 0) order should be non-zero
    idx_00 = mx.index(0) * 7 + my.index(0)
    # At least some propagating orders should have non-zero amplitude