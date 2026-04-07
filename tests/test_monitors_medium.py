"""Tests for medium-property monitors (MediumMonitor, PermittivityMonitor)."""

import numpy as np
import pytest

from autofdtd.monitors import (
    MediumMonitor,
    PermittivityMonitor,
    MediumMonitorData,
    PermittivityData,
    SimulationData,
)
from autofdtd.compiler.monitors import (
    CompiledMediumMonitor,
    MediumMonitorState,
    compile_medium_monitor,
)
from autofdtd.grid import ResolvedGrid, ResolvedGridAxis, UniformGrid
from autofdtd.core.containers import Scene, Structure
from autofdtd.geometry import Box
from autofdtd.materials import Medium


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
        center=(float(nx) / 2.0, float(ny) / 2.0, float(nz) / 2.0),
        size=(float(nx), float(ny), float(nz)),
        x=x_axis,
        y=y_axis,
        z=z_axis,
    )


class TestMediumMonitorModels:
    """Test MediumMonitor and PermittivityMonitor model properties."""

    def test_medium_monitor_basic_fields(self):
        """Test that MediumMonitor stores basic fields correctly."""
        monitor = MediumMonitor(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            freqs=(250e12, 300e12),
            name="medium_monitor",
        )
        assert monitor.type == "MediumMonitor"
        assert monitor.name == "medium_monitor"
        assert monitor.center == (0.0, 0.0, 0.0)
        assert monitor.size == (2.0, 2.0, 2.0)
        assert monitor.num_freqs == 1
        assert monitor.freqs == (250e12, 300e12)

    def test_medium_monitor_default_values(self):
        """Test that MediumMonitor default values are correct."""
        monitor = MediumMonitor(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
        )
        assert monitor.interval == 1
        assert monitor.start == 0
        assert monitor.num_freqs == 1
        assert monitor.freqs == ()

    def test_permittivity_monitor_basic_fields(self):
        """Test that PermittivityMonitor stores basic fields correctly."""
        monitor = PermittivityMonitor(
            center=(1.0, 2.0, 3.0),
            size=(1.0, 1.0, 1.0),
            name="eps_monitor",
        )
        assert monitor.type == "PermittivityMonitor"
        assert monitor.name == "eps_monitor"
        assert monitor.center == (1.0, 2.0, 3.0)
        assert monitor.size == (1.0, 1.0, 1.0)

    def test_medium_monitor_interval_validation(self):
        """Test that MediumMonitor validates interval correctly."""
        with pytest.raises(ValueError):
            MediumMonitor(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 0.0, 0.0),
                interval=0,
            )

    def test_medium_monitor_start_validation(self):
        """Test that MediumMonitor validates start correctly."""
        with pytest.raises(ValueError):
            MediumMonitor(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 0.0, 0.0),
                start=-1,
            )


class TestCompiledMediumMonitor:
    """Test CompiledMediumMonitor creation from grid."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple uniform grid for testing."""
        return make_test_grid()

    def test_compile_medium_monitor_volume(self, simple_grid):
        """Test compiling a volume medium monitor."""
        compiled = compile_medium_monitor(
            name="test_medium",
            monitor_type="MediumMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            interval=1,
            start=0,
            num_freqs=2,
            freqs=(250e12, 300e12),
            grid=simple_grid,
        )

        assert compiled.name == "test_medium"
        assert compiled.monitor_type == "MediumMonitor"
        assert compiled.center == (5.0, 5.0, 5.0)
        assert compiled.size == (2.0, 2.0, 2.0)
        assert compiled.interval == 1
        assert compiled.start == 0
        assert compiled.num_freqs == 2
        assert compiled.freqs == (250e12, 300e12)
        assert len(compiled.placements) > 0
        assert not compiled.is_permittivity_only

    def test_compile_medium_monitor_point(self, simple_grid):
        """Test compiling a point medium monitor (zero size)."""
        compiled = compile_medium_monitor(
            name="test_point",
            monitor_type="MediumMonitor",
            center=(5.0, 5.0, 5.0),
            size=(0.0, 0.0, 0.0),
            interval=1,
            start=0,
            grid=simple_grid,
        )

        assert len(compiled.placements) == 1
        assert compiled.placements[0] is not None

    def test_compile_permittivity_monitor(self, simple_grid):
        """Test compiling a permittivity monitor."""
        compiled = compile_medium_monitor(
            name="test_eps",
            monitor_type="PermittivityMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            interval=1,
            start=0,
            grid=simple_grid,
        )

        assert compiled.monitor_type == "PermittivityMonitor"
        assert compiled.is_permittivity_only
        assert compiled.num_freqs == 1  # default


class TestMediumMonitorState:
    """Test MediumMonitorState for medium data recording."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple uniform grid for testing."""
        return make_test_grid()

    @pytest.fixture
    def simple_scene(self):
        """Create a simple scene with one structure."""
        scene = Scene()
        box = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        medium = Medium(permittivity=3.5)
        structure = Structure(geometry=box, medium=medium, name="box1")
        scene.add_structure(structure)
        return scene

    def test_medium_monitor_state_initialization(self, simple_grid):
        """Test that MediumMonitorState initializes correctly."""
        compiled = compile_medium_monitor(
            name="test",
            monitor_type="MediumMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            num_freqs=2,
            freqs=(250e12, 300e12),
            grid=simple_grid,
        )

        state = MediumMonitorState(compiled=compiled)

        assert state.compiled is compiled
        assert state.medium_data is not None
        assert state.medium_data.shape == (len(compiled.placements), 2, 6)

    def test_permittivity_monitor_state_initialization(self, simple_grid):
        """Test that MediumMonitorState for PermittivityMonitor initializes correctly."""
        compiled = compile_medium_monitor(
            name="test",
            monitor_type="PermittivityMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            grid=simple_grid,
        )

        state = MediumMonitorState(compiled=compiled)

        assert state.compiled is compiled
        assert state.medium_data is not None
        assert state.medium_data.shape == (len(compiled.placements), 1, 3)

    def test_to_medium_monitor_data_format(self, simple_grid):
        """Test that to_medium_monitor_data returns correct format."""
        compiled = compile_medium_monitor(
            name="test",
            monitor_type="MediumMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            num_freqs=2,
            freqs=(250e12, 300e12),
            grid=simple_grid,
        )

        state = MediumMonitorState(compiled=compiled)

        # Set some test data
        state.medium_data[0, 0, 0] = complex(2.0, 0.0)
        state.medium_data[0, 0, 3] = complex(1.0, 0.0)

        data = state.to_medium_monitor_data()

        assert "eps_xx" in data
        assert "mu_xx" in data
        # Should be flattened (n_pts * n_freqs) tuples
        assert len(data["eps_xx"]) == len(compiled.placements) * 2

    def test_to_permittivity_monitor_data_format(self, simple_grid):
        """Test that to_permittivity_monitor_data returns correct format."""
        compiled = compile_medium_monitor(
            name="test",
            monitor_type="PermittivityMonitor",
            center=(5.0, 5.0, 5.0),
            size=(2.0, 2.0, 2.0),
            grid=simple_grid,
        )

        state = MediumMonitorState(compiled=compiled)

        # Set some test data
        state.medium_data[0, 0, 0] = complex(2.5, 0.0)

        data = state.to_permittivity_monitor_data()

        assert "eps_xx" in data
        assert "eps_yy" in data
        assert "eps_zz" in data
        assert "mu_xx" not in data  # permittivity-only should not have mu


class TestMediumMonitorData:
    """Test MediumMonitorData container."""

    def test_medium_monitor_data_creation(self):
        """Test creating a MediumMonitorData object."""
        data = MediumMonitorData(
            monitor_name="test",
            monitor_type="MediumMonitor",
            eps_xx=((2.0, 0.0), (2.1, 0.0)),
            eps_yy=((2.0, 0.0), (2.1, 0.0)),
            eps_zz=((2.0, 0.0), (2.1, 0.0)),
            mu_xx=((1.0, 0.0), (1.0, 0.0)),
            mu_yy=((1.0, 0.0), (1.0, 0.0)),
            mu_zz=((1.0, 0.0), (1.0, 0.0)),
        )

        assert data.monitor_name == "test"
        assert data.monitor_type == "MediumMonitor"
        assert data.eps_xx == ((2.0, 0.0), (2.1, 0.0))

    def test_medium_monitor_data_with_none(self):
        """Test MediumMonitorData with None values."""
        data = MediumMonitorData(
            monitor_name="test",
            monitor_type="MediumMonitor",
        )

        assert data.eps_xx is None
        assert data.mu_xx is None


class TestPermittivityData:
    """Test PermittivityData container."""

    def test_permittivity_data_creation(self):
        """Test creating a PermittivityData object."""
        data = PermittivityData(
            monitor_name="test",
            monitor_type="PermittivityMonitor",
            eps_xx=(2.0, 2.1, 2.2),
            eps_yy=(2.0, 2.1, 2.2),
            eps_zz=(2.0, 2.1, 2.2),
        )

        assert data.monitor_name == "test"
        assert data.monitor_type == "PermittivityMonitor"
        assert data.eps_xx == (2.0, 2.1, 2.2)


class TestSimulationDataMediumRegistration:
    """Test SimulationData registration methods for medium monitors."""

    def test_register_medium_monitor(self):
        """Test registering a medium monitor in SimulationData."""
        sim_data = SimulationData(simulation_name="test_sim")
        data = MediumMonitorData(
            monitor_name="medium",
            monitor_type="MediumMonitor",
            eps_xx=((2.0, 0.0),),
        )

        sim_data.register_medium_monitor("medium", data)
        assert "medium" in sim_data
        assert sim_data["medium"].monitor_name == "medium"

    def test_register_permittivity_monitor(self):
        """Test registering a permittivity monitor in SimulationData."""
        sim_data = SimulationData(simulation_name="test_sim")
        data = PermittivityData(
            monitor_name="eps",
            monitor_type="PermittivityMonitor",
            eps_xx=(2.0,),
        )

        sim_data.register_permittivity_monitor("eps", data)
        assert "eps" in sim_data
        assert sim_data["eps"].monitor_name == "eps"

    def test_register_medium_monitor_name_mismatch(self):
        """Test that register_medium_monitor corrects name mismatches."""
        sim_data = SimulationData(simulation_name="test_sim")
        data = MediumMonitorData(
            monitor_name="wrong_name",
            monitor_type="MediumMonitor",
            eps_xx=((2.0, 0.0),),
        )

        sim_data.register_medium_monitor("correct_name", data)
        assert "correct_name" in sim_data
        assert sim_data["correct_name"].monitor_name == "correct_name"

    def test_register_medium_monitor_duplicate(self):
        """Test that registering duplicate medium monitor raises error."""
        sim_data = SimulationData(simulation_name="test_sim")
        data = MediumMonitorData(
            monitor_name="medium",
            monitor_type="MediumMonitor",
            eps_xx=((2.0, 0.0),),
        )

        sim_data.register_medium_monitor("medium", data)
        with pytest.raises(KeyError):
            sim_data.register_medium_monitor("medium", data)


class TestMonitorDataModelFromValue:
    """Test monitor_data_model_from_value factory."""

    def test_medium_monitor_data_from_dict(self):
        """Test creating MediumMonitorData from a dict."""
        from autofdtd.monitors.models import monitor_data_model_from_value

        value = {
            "type": "MediumMonitorData",
            "monitor_name": "test",
            "monitor_type": "MediumMonitor",
            "eps_xx": [(2.0, 0.0)],
        }

        data = monitor_data_model_from_value(value)
        assert isinstance(data, MediumMonitorData)
        assert data.monitor_name == "test"

    def test_permittivity_data_from_dict(self):
        """Test creating PermittivityData from a dict."""
        from autofdtd.monitors.models import monitor_data_model_from_value

        value = {
            "type": "PermittivityData",
            "monitor_name": "test",
            "monitor_type": "PermittivityMonitor",
            "eps_xx": [2.0],
        }

        data = monitor_data_model_from_value(value)
        assert isinstance(data, PermittivityData)
        assert data.monitor_name == "test"