"""Tests for monitor naming, SimulationData access patterns, and monitor IR lowering."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from autofdtd.monitors import (
    AstigmaticGaussianOverlapData,
    AstigmaticGaussianOverlapMonitor,
    FieldData,
    FieldMonitor,
    FieldTimeMonitor,
    FluxData,
    FluxMonitor,
    FluxTimeMonitor,
    GaussianOverlapData,
    GaussianOverlapMonitor,
    MediumMonitorData,
    Monitor,
    MonitorData,
    ModeData,
    ModeMonitor,
    PermittivityData,
    PermittivityMonitor,
    SimulationData,
    monitor_model_from_value,
)


# ---------------------------------------------------------------------------
# Monitor model construction and naming
# ---------------------------------------------------------------------------


def test_monitor_name_normalization() -> None:
    """Monitor names are normalized (trimmed) and empty names are rejected."""
    monitor = FieldMonitor(name=" field_monitor ", size=(1.0, 1.0, 0.0))
    assert monitor.name == "field_monitor"

    with pytest.raises(ValueError, match="names must not be empty"):
        FieldMonitor(name="   ", size=(1.0, 1.0, 0.0))


def test_monitor_unique_name_validation() -> None:
    """Simulation validates that monitor names are unique."""
    from autofdtd.api import Simulation

    with pytest.raises(ValidationError, match="monitors names must be unique"):
        Simulation(
            size=(4.0, 4.0, 4.0),
            run_time=1.0,
            monitors=(
                FieldMonitor(name="m1", size=(1.0, 1.0, 0.0)),
                FieldMonitor(name="m1", size=(1.0, 1.0, 0.0)),
            ),
        )


def test_monitor_model_factory() -> None:
    """monitor_model_from_value constructs concrete monitor models."""
    data = {"type": "FieldMonitor", "name": "test", "size": [1.0, 1.0, 0.0]}
    monitor = monitor_model_from_value(data)
    assert isinstance(monitor, FieldMonitor)
    assert monitor.name == "test"
    assert monitor.size == (1.0, 1.0, 0.0)


def test_monitor_model_factory_unknown_type() -> None:
    """Unknown monitor types return a generic Monitor."""
    data = {"type": "CustomMonitor", "name": "test", "size": [1.0, 1.0, 0.0]}
    monitor = monitor_model_from_value(data)
    assert isinstance(monitor, Monitor)
    assert monitor.type == "CustomMonitor"


def test_field_monitor_defaults() -> None:
    """FieldMonitor has sensible defaults for fields and interval."""
    monitor = FieldMonitor(size=(1.0, 1.0, 0.0))
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert monitor.interval == 1
    assert monitor.start == 0
    assert monitor.colocated() is True


def test_flux_monitor_direction() -> None:
    """FluxMonitor records direction (+ or -)."""
    monitor = FluxMonitor(size=(1.0, 1.0, 0.0), direction="-")
    assert monitor.direction == "-"
    assert monitor.num_freqs == 1


def test_field_time_monitor() -> None:
    """FieldTimeMonitor captures time-domain fields."""
    monitor = FieldTimeMonitor(size=(1.0, 1.0, 0.0))
    assert monitor.type == "FieldTimeMonitor"
    assert monitor.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# ---------------------------------------------------------------------------
# SimulationData named access
# ---------------------------------------------------------------------------


def test_simulation_data_empty() -> None:
    """SimulationData starts empty and reports no monitors."""
    data = SimulationData()
    assert len(data) == 0
    assert list(data) == []
    assert data.monitor_names == ()
    assert "field_monitor" not in data


def test_simulation_data_register_field_monitor() -> None:
    """SimulationData.register_field_monitor stores FieldData by name."""
    sim_data = SimulationData()

    field_data = FieldData(
        monitor_name="field_monitor",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0), (2.0, 0.0)),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )

    sim_data.register_field_monitor("field_monitor", field_data)

    assert len(sim_data) == 1
    assert "field_monitor" in sim_data
    assert "other_monitor" not in sim_data


def test_simulation_data_dictionary_access() -> None:
    """SimulationData supports dictionary-style access via __getitem__."""
    sim_data = SimulationData()

    field_data = FieldData(
        monitor_name="field_monitor",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0),),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )
    sim_data.register_field_monitor("field_monitor", field_data)

    # Dictionary access
    retrieved = sim_data["field_monitor"]
    assert retrieved.monitor_name == "field_monitor"
    assert isinstance(retrieved, FieldData)

    # KeyError for unknown name
    with pytest.raises(KeyError, match="no monitor data found"):
        _ = sim_data["unknown"]


def test_simulation_data_attribute_access() -> None:
    """SimulationData supports attribute-style access via __getattr__."""
    sim_data = SimulationData()

    flux_data = FluxData(
        monitor_name="flux_monitor",
        monitor_type="FluxMonitor",
        flux=(1.0, 2.0, 3.0),
    )
    sim_data.register_flux_monitor("flux_monitor", flux_data)

    # Attribute access
    retrieved = sim_data.flux_monitor
    assert retrieved.monitor_name == "flux_monitor"
    assert isinstance(retrieved, FluxData)

    # AttributeError for unknown name
    with pytest.raises(AttributeError, match="no monitor data found"):
        _ = sim_data.unknown_monitor


def test_simulation_data_iteration() -> None:
    """SimulationData is iterable over monitor names."""
    sim_data = SimulationData()

    field_data = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0),),
    )
    flux_data = FluxData(monitor_name="flux", monitor_type="FluxMonitor", flux=(1.0,))

    sim_data.register_field_monitor("field", field_data)
    sim_data.register_flux_monitor("flux", flux_data)

    names = list(sim_data)
    assert set(names) == {"field", "flux"}


def test_simulation_data_keys_values_items() -> None:
    """SimulationData exposes keys(), values(), and items()."""
    sim_data = SimulationData()

    field_data = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0),),
    )
    sim_data.register_field_monitor("field", field_data)

    assert list(sim_data.keys()) == ["field"]
    assert list(sim_data.values()) == [field_data]
    assert list(sim_data.items()) == [("field", field_data)]


def test_simulation_data_get_with_default() -> None:
    """SimulationData.get returns a default for unknown keys."""
    sim_data = SimulationData()
    assert sim_data.get("unknown") is None
    assert sim_data.get("unknown", "default") == "default"


def test_simulation_data_register_duplicate_raises() -> None:
    """Registering data for an already-registered name raises KeyError."""
    sim_data = SimulationData()

    field_data1 = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0),),
    )
    field_data2 = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((2.0, 0.0),),
    )

    sim_data.register_field_monitor("field", field_data1)
    with pytest.raises(KeyError, match="already registered"):
        sim_data.register_field_monitor("field", field_data2)


def test_simulation_data_to_dict() -> None:
    """SimulationData.to_dict returns a plain dict with all monitor data."""
    sim_data = SimulationData()

    field_data = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0),),
    )
    sim_data.register_field_monitor("field", field_data)

    result = sim_data.to_dict()
    assert "monitor_names" in result
    assert "field" in result
    assert result["field"]["monitor_type"] == "FieldMonitor"


def test_simulation_data_register_custom_monitor() -> None:
    """SimulationData.register_custom_monitor accepts arbitrary MonitorData."""
    sim_data = SimulationData()

    custom_data = MonitorData(
        monitor_name="custom",
        monitor_type="CustomMonitor",
        data={"custom_field": (1.0, 2.0)},
    )
    sim_data.register_custom_monitor("custom", custom_data)

    assert sim_data["custom"].monitor_name == "custom"
    assert sim_data["custom"].data["custom_field"] == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Monitor IR lowering
# ---------------------------------------------------------------------------


def test_field_monitor_ir_lowering() -> None:
    """FieldMonitor lowers to FieldMonitorIR with correct fields."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(FieldMonitor(name="field", size=(1.0, 1.0, 0.0)),),
    )

    ir = simulation_to_ir(sim)
    assert len(ir.monitors) == 1
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "FieldMonitor"
    assert monitor_ir.name == "field"
    assert monitor_ir.size == (1.0, 1.0, 0.0)
    assert monitor_ir.fields == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def test_flux_monitor_ir_lowering() -> None:
    """FluxMonitor lowers to FluxMonitorIR with direction and freqs."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(FluxMonitor(name="flux", size=(1.0, 1.0, 0.0), direction="-"),),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "FluxMonitor"
    assert monitor_ir.direction == "-"


def test_mode_monitor_ir_lowering() -> None:
    """ModeMonitor lowers to ModeMonitorIR with mode_spec."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import ModeSpec, Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            ModeMonitor(
                name="mode",
                size=(1.0, 1.0, 0.0),
                mode_spec=ModeSpec(num_modes=3),
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "ModeMonitor"
    assert monitor_ir.name == "mode"
    assert monitor_ir.mode_spec is not None


def test_permittivity_monitor_ir_lowering() -> None:
    """PermittivityMonitor lowers to PermittivityMonitorIR."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(PermittivityMonitor(name="eps", size=(1.0, 1.0, 0.0)),),
    )

    ir = simulation_to_ir(sim)
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "PermittivityMonitor"
    assert monitor_ir.name == "eps"


def test_multiple_monitors_ir_lowering() -> None:
    """Simulation with multiple monitors lowers each to the correct IR type."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            FieldMonitor(name="field", size=(1.0, 1.0, 0.0)),
            FluxMonitor(name="flux", size=(1.0, 1.0, 0.0)),
            PermittivityMonitor(name="eps", size=(1.0, 1.0, 0.0)),
        ),
    )

    ir = simulation_to_ir(sim)
    assert len(ir.monitors) == 3

    types = {m.component_type for m in ir.monitors}
    assert types == {"FieldMonitor", "FluxMonitor", "PermittivityMonitor"}

    names = {m.name for m in ir.monitors}
    assert names == {"field", "flux", "eps"}


# ---------------------------------------------------------------------------
# Monitor data JSON safety
# ---------------------------------------------------------------------------


def test_field_data_complex_json_safety() -> None:
    """FieldData stores complex values as (real, imag) tuples for JSON safety."""
    data = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.5, 0.5),),
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )

    # Check JSON serialization works
    json_text = data.to_json_text()
    assert '"Ex"' in json_text
    assert "1.5" in json_text


def test_simulation_data_json_roundtrip() -> None:
    """SimulationData JSON roundtrips correctly for metadata.

    Note: The monitor_store (runtime data) is excluded from serialization
    by design since it is populated during simulation execution.
    """
    sim_data = SimulationData(simulation_name="test_sim")

    # Register some data
    field_data = FieldData(
        monitor_name="field",
        monitor_type="FieldMonitor",
        Ex=((1.0, 0.0), (2.0, 0.0)),
    )
    sim_data.register_field_monitor("field", field_data)

    # Serialize only the metadata (not runtime data)
    json_text = sim_data.to_json_text()
    restored = SimulationData.model_validate_json(json_text)

    # Metadata roundtrips
    assert restored.simulation_name == "test_sim"
    assert restored.monitor_names == ()

    # Runtime data is not serialized (by design)
    assert len(restored) == 0


# ---------------------------------------------------------------------------
# Gaussian overlap monitor tests
# ---------------------------------------------------------------------------


def test_gaussian_overlap_monitor_defaults() -> None:
    """GaussianOverlapMonitor has sensible defaults for beam parameters."""
    monitor = GaussianOverlapMonitor(size=(0, 3, 3), freqs=[2e14])
    assert monitor.type == "GaussianOverlapMonitor"
    assert monitor.direction == "+"
    assert monitor.num_freqs == 1
    assert monitor.waist_radius == 1.0
    assert monitor.waist_distance == 0.0
    assert monitor.angle_theta == 0.0
    assert monitor.angle_phi == 0.0
    assert monitor.pol_angle == 0.0
    assert monitor._angles == (0.0, 0.0)


def test_gaussian_overlap_monitor_beam_params() -> None:
    """GaussianOverlapMonitor accepts beam parameters."""
    monitor = GaussianOverlapMonitor(
        size=(0, 3, 3),
        freqs=[2e14],
        pol_angle=1.57,
        waist_radius=2.0,
        waist_distance=-1.0,
        angle_theta=0.5,
        angle_phi=0.3,
    )
    assert monitor.waist_radius == 2.0
    assert monitor.waist_distance == -1.0
    assert monitor.pol_angle == 1.57
    assert monitor.angle_theta == 0.5
    assert monitor.angle_phi == 0.3


def test_gaussian_overlap_monitor_ir_lowering() -> None:
    """GaussianOverlapMonitor lowers to GaussianOverlapMonitorIR with beam params."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            GaussianOverlapMonitor(
                name="gauss",
                size=(0, 3, 3),
                freqs=[2e14],
                pol_angle=1.57,
                waist_radius=2.0,
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    assert len(ir.monitors) == 1
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "GaussianOverlapMonitor"
    assert monitor_ir.name == "gauss"
    assert monitor_ir.pol_angle == 1.57
    assert monitor_ir.waist_radius == 2.0


# ---------------------------------------------------------------------------
# Astigmatic Gaussian overlap monitor tests
# ---------------------------------------------------------------------------


def test_astigmatic_gaussian_overlap_monitor_defaults() -> None:
    """AstigmaticGaussianOverlapMonitor has sensible defaults for beam parameters."""
    monitor = AstigmaticGaussianOverlapMonitor(size=(0, 3, 3), freqs=[2e14])
    assert monitor.type == "AstigmaticGaussianOverlapMonitor"
    assert monitor.direction == "+"
    assert monitor.num_freqs == 1
    assert monitor.waist_sizes == (1.0, 1.0)
    assert monitor.waist_distances == (0.0, 0.0)
    assert monitor.angle_theta == 0.0
    assert monitor.angle_phi == 0.0
    assert monitor.pol_angle == 0.0
    assert monitor._angles == (0.0, 0.0)


def test_astigmatic_gaussian_overlap_monitor_beam_params() -> None:
    """AstigmaticGaussianOverlapMonitor accepts beam parameters."""
    monitor = AstigmaticGaussianOverlapMonitor(
        size=(0, 3, 3),
        freqs=[2e14],
        pol_angle=1.57,
        waist_sizes=(1.0, 2.0),
        waist_distances=(3.0, 4.0),
        angle_theta=0.5,
        angle_phi=0.3,
    )
    assert monitor.waist_sizes == (1.0, 2.0)
    assert monitor.waist_distances == (3.0, 4.0)
    assert monitor.pol_angle == 1.57


def test_astigmatic_gaussian_overlap_monitor_waist_sizes_validation() -> None:
    """AstigmaticGaussianOverlapMonitor validates waist_sizes are positive."""
    with pytest.raises(ValueError, match="waist_sizes must be positive"):
        AstigmaticGaussianOverlapMonitor(
            size=(0, 3, 3),
            waist_sizes=(0.0, 1.0),
        )


def test_astigmatic_gaussian_overlap_monitor_ir_lowering() -> None:
    """AstigmaticGaussianOverlapMonitor lowers with beam params."""
    from autofdtd.ir import simulation_to_ir

    from autofdtd.api import Simulation

    sim = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        monitors=(
            AstigmaticGaussianOverlapMonitor(
                name="astig",
                size=(0, 3, 3),
                freqs=[2e14],
                pol_angle=1.57,
                waist_sizes=(1.0, 2.0),
                waist_distances=(3.0, 4.0),
            ),
        ),
    )

    ir = simulation_to_ir(sim)
    assert len(ir.monitors) == 1
    monitor_ir = ir.monitors[0]
    assert monitor_ir.component_type == "AstigmaticGaussianOverlapMonitor"
    assert monitor_ir.name == "astig"
    assert monitor_ir.pol_angle == 1.57
    assert monitor_ir.waist_sizes == (1.0, 2.0)
    assert monitor_ir.waist_distances == (3.0, 4.0)


# ---------------------------------------------------------------------------
# Gaussian overlap data tests
# ---------------------------------------------------------------------------


def test_gaussian_overlap_data() -> None:
    """GaussianOverlapData stores complex overlap amplitudes."""
    data = GaussianOverlapData(
        monitor_name="gauss",
        monitor_type="GaussianOverlapMonitor",
        amplitudes=(
            # + direction amplitudes for 2 freqs
            ((1.0, 0.0), (2.0, 0.0)),
            # - direction amplitudes for 2 freqs
            ((0.5, 0.0), (1.5, 0.0)),
        ),
    )
    assert data.monitor_name == "gauss"
    assert data.type == "GaussianOverlapData"
    assert len(data.amplitudes) == 2  # two directions


def test_astigmatic_gaussian_overlap_data() -> None:
    """AstigmaticGaussianOverlapData stores complex overlap amplitudes."""
    data = AstigmaticGaussianOverlapData(
        monitor_name="astig",
        monitor_type="AstigmaticGaussianOverlapMonitor",
        amplitudes=(
            # + direction amplitudes for 2 freqs
            ((1.0, 0.0), (2.0, 0.0)),
            # - direction amplitudes for 2 freqs
            ((0.5, 0.0), (1.5, 0.0)),
        ),
    )
    assert data.monitor_name == "astig"
    assert data.type == "AstigmaticGaussianOverlapData"
    assert len(data.amplitudes) == 2  # two directions


def test_simulation_data_register_gaussian_overlap_monitor() -> None:
    """SimulationData.register_gaussian_overlap_monitor stores GaussianOverlapData."""
    sim_data = SimulationData()

    data = GaussianOverlapData(
        monitor_name="gauss",
        monitor_type="GaussianOverlapMonitor",
        amplitudes=(((1.0, 0.0),),),
    )
    sim_data.register_gaussian_overlap_monitor("gauss", data)

    assert "gauss" in sim_data
    retrieved = sim_data["gauss"]
    assert isinstance(retrieved, GaussianOverlapData)


def test_simulation_data_register_astigmatic_gaussian_overlap_monitor() -> None:
    """SimulationData.register_astigmatic_gaussian_overlap_monitor stores data."""
    sim_data = SimulationData()

    data = AstigmaticGaussianOverlapData(
        monitor_name="astig",
        monitor_type="AstigmaticGaussianOverlapMonitor",
        amplitudes=(((1.0, 0.0),),),
    )
    sim_data.register_astigmatic_gaussian_overlap_monitor("astig", data)

    assert "astig" in sim_data
    retrieved = sim_data["astig"]
    assert isinstance(retrieved, AstigmaticGaussianOverlapData)
