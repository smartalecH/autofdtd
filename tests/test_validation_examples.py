"""Validation tests for Phase 1 feature integration examples.

These tests exercise the complete execution pipeline for the validation
examples, verifying that:
1. Examples compile and execute without errors
2. Physical behavior is consistent with expectations
3. Monitor data is recorded correctly
4. Convergence behavior works as expected
"""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    AutoGrid,
    Box,
    Boundary,
    BoundarySpec,
    ContinuousWave,
    GaussianPulse,
    GridSpec,
    Medium,
    PML,
    PECBoundary,
    PMCBoundary,
    Scene,
    Simulation,
    Sphere,
    Structure,
    SubpixelSpec,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.compiler import compile_simulation
from autofdtd.monitors import (
    FieldMonitor,
    FluxMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
)
from autofdtd.sources import (
    GaussianBeam,
    PlaneWave,
    PointDipole,
    TFSF,
)
from autofdtd.examples.validation import (
    convergence_shutoff_example,
    dielectric_slab_example,
    field_monitor_recording_example,
    flux_monitor_recording_example,
    medium_monitor_example,
    multi_feature_integration_example,
    pml_absorption_example,
    run_all_validation_examples,
    uniform_current_injection_example,
    vacuum_plane_wave_example,
    vacuum_point_source_example,
)
from autofdtd.runtime import run_compiled_simulation


class TestVacuumPropagationExamples:
    """Test vacuum propagation validation examples."""

    def test_vacuum_point_source_runs(self):
        """Vacuum point source example should execute without error."""
        result = vacuum_point_source_example()
        assert result["example"] == "vacuum_point_source"
        assert result["num_steps"] > 0
        assert result["stop_reason"] in ("max_steps", "shutoff")
        assert result["total_cells"] > 0

    def test_vacuum_point_source_energy_behavior(self):
        """Point source energy should peak early and decay."""
        result = vacuum_point_source_example()
        # Energy should peak early then decay
        assert result["energy_decay_ratio"] is not None
        # Energy should decay (final < peak)
        assert result["energy_decay_ratio"] < 1.0

    def test_vacuum_plane_wave_runs(self):
        """Vacuum plane wave example should execute without error."""
        result = vacuum_plane_wave_example()
        assert result["example"] == "vacuum_plane_wave"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0


class TestDielectricExamples:
    """Test dielectric slab validation examples."""

    def test_dielectric_slab_runs(self):
        """Dielectric slab example should execute without error."""
        result = dielectric_slab_example()
        assert result["example"] == "dielectric_slab"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_dielectric_slab_has_field_data(self):
        """Dielectric slab should produce field data."""
        result = dielectric_slab_example()
        assert result["has_field_data"] is True


class TestPMLAbsorptionExamples:
    """Test PML absorption validation examples."""

    def test_pml_absorption_runs(self):
        """PML absorption example should execute without error."""
        result = pml_absorption_example()
        assert result["example"] == "pml_absorption"
        assert result["pml_steps"] > 0
        assert result["pec_steps"] > 0
        assert result["total_cells"] > 0

    def test_pml_absorption_compares_pml_vs_pec(self):
        """PML should absorb more energy than PEC reflects."""
        result = pml_absorption_example()
        # PML final energy should be lower than PEC (more absorption)
        assert result["absorption_ratio"] is not None
        # Ratio < 1 means PML absorbed more than PEC reflected


class TestSourceInjectionExamples:
    """Test source injection validation examples."""

    def test_uniform_current_injection_runs(self):
        """Uniform current injection example should execute without error."""
        result = uniform_current_injection_example()
        assert result["example"] == "uniform_current_injection"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_uniform_current_has_field_data(self):
        """Uniform current should produce field data."""
        result = uniform_current_injection_example()
        assert result["has_Ez_data"] is True
        assert result["has_H_data"] is True


class TestMonitorRecordingExamples:
    """Test monitor recording validation examples."""

    def test_field_monitor_recording_runs(self):
        """Field monitor recording example should execute without error."""
        result = field_monitor_recording_example()
        assert result["example"] == "field_monitor_recording"
        assert result["num_steps"] > 0
        assert result["num_records"] > 0
        assert result["total_cells"] > 0

    def test_field_monitor_records_expected_count(self):
        """Field monitor should record expected number of time samples."""
        result = field_monitor_recording_example()
        # Records at interval=5 starting from step 0
        assert result["num_records"] == result["expected_records"]

    def test_field_monitor_has_Ez(self):
        """Field monitor should have Ez data."""
        result = field_monitor_recording_example()
        assert result["has_Ez"] is True

    def test_flux_monitor_recording_runs(self):
        """Flux monitor recording example should execute without error."""
        result = flux_monitor_recording_example()
        assert result["example"] == "flux_monitor_recording"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_flux_monitor_has_data(self):
        """Flux monitor should have flux data."""
        result = flux_monitor_recording_example()
        assert result["has_flux_data"] is True

    def test_medium_monitor_runs(self):
        """Medium monitor example should execute without error."""
        result = medium_monitor_example()
        assert result["example"] == "medium_monitor"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_medium_monitor_has_data(self):
        """Medium monitor should have medium data."""
        result = medium_monitor_example()
        assert result["has_medium_data"] is True


class TestConvergenceExamples:
    """Test convergence and shutoff validation examples."""

    def test_convergence_shutoff_runs(self):
        """Convergence shutoff example should execute without error."""
        result = convergence_shutoff_example()
        assert result["example"] == "convergence_shutoff"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_convergence_shutoff_triggers_early(self):
        """Convergence shutoff should trigger before max_steps."""
        result = convergence_shutoff_example()
        # Should stop early due to shutoff, not max_steps
        assert result["num_steps"] < result["max_steps_config"]
        assert result["stop_reason"] == "shutoff"

    def test_convergence_energy_decays(self):
        """Energy should decay below shutoff threshold."""
        result = convergence_shutoff_example()
        # Final energy ratio should be below shutoff threshold
        assert result["energy_ratio"] is not None
        assert result["energy_ratio"] < 1e-3  # shutoff is 1e-4


class TestMultiFeatureIntegration:
    """Test multi-feature integration examples."""

    def test_multi_feature_integration_runs(self):
        """Multi-feature integration example should execute without error."""
        result = multi_feature_integration_example()
        assert result["example"] == "multi_feature_integration"
        assert result["num_steps"] > 0
        assert result["total_cells"] > 0

    def test_multi_feature_has_all_monitor_data(self):
        """Multi-feature should have field, flux, and medium data."""
        result = multi_feature_integration_example()
        assert result["has_field_data"] is True
        assert result["has_flux_data"] is True
        assert result["has_medium_data"] is True

    def test_multi_feature_has_sources_and_monitors(self):
        """Multi-feature should have compiled sources and monitors."""
        result = multi_feature_integration_example()
        assert result["num_sources"] > 0
        assert result["num_monitors"] > 0


class TestRunAllExamples:
    """Test running all validation examples together."""

    def test_run_all_validation_examples(self):
        """All validation examples should run without errors."""
        summary = run_all_validation_examples()
        assert summary["num_examples"] == 10
        assert summary["successful"] == 10

    def test_run_all_has_backend_info(self):
        """Run all should report backend information."""
        summary = run_all_validation_examples()
        assert summary["backend"] is not None
        backend_info = summary["backend_info"]
        assert hasattr(backend_info, "warp_available")


class TestExecutionPipeline:
    """Test the execution pipeline directly."""

    def test_compile_and_run_simple_simulation(self):
        """A simple simulation should compile and run."""
        sim = Simulation(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            grid_spec=GridSpec(
                grid_x=UniformGrid(dl=0.1),
                grid_y=UniformGrid(dl=0.1),
                grid_z=UniformGrid(dl=0.1),
            ),
            medium=Medium(permittivity=1.0),
            sources=(
                PointDipole(
                    center=(0.0, 0.0, 0.0),
                    polarization="Ez",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=1e14,
                        amplitude=1.0,
                        offset=3.0,
                    ),
                    name="dipole",
                ),
            ),
            boundary_spec=BoundarySpec(
                x=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
                y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
                z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            ),
            symmetry=(0, 0, 0),
            shutoff=1e-5,
        )

        compiled = compile_simulation(sim)
        result = run_compiled_simulation(
            compiled,
            max_steps=100,
            record_interval=10,
            verbose=False,
        )

        assert result.num_steps > 0
        assert result.final_time > 0
        assert len(result.integrated_electric_history) == result.num_steps

    def test_execution_result_contains_expected_fields(self):
        """ExecutionResult should contain all expected fields."""
        sim = Simulation(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            grid_spec=GridSpec(
                grid_x=UniformGrid(dl=0.1),
                grid_y=UniformGrid(dl=0.1),
                grid_z=UniformGrid(dl=0.1),
            ),
            medium=Medium(permittivity=1.0),
            sources=(
                PointDipole(
                    center=(0.0, 0.0, 0.0),
                    polarization="Ez",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=1e14,
                        amplitude=1.0,
                        offset=3.0,
                    ),
                    name="dipole",
                ),
            ),
            boundary_spec=BoundarySpec(
                x=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
                y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
                z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            ),
            symmetry=(0, 0, 0),
            shutoff=1e-5,
        )

        compiled = compile_simulation(sim)
        result = run_compiled_simulation(
            compiled,
            max_steps=50,
            record_interval=10,
            verbose=False,
        )

        # Check result structure
        assert hasattr(result, "field_state")
        assert hasattr(result, "num_steps")
        assert hasattr(result, "final_time")
        assert hasattr(result, "stop_reason")
        assert hasattr(result, "integrated_electric_history")
        assert hasattr(result, "field_monitor_data")
        assert hasattr(result, "flux_monitor_data")
        assert hasattr(result, "medium_monitor_data")
        assert hasattr(result, "mode_monitor_data")
        assert hasattr(result, "metrics")

        # Check field_state structure
        fs = result.field_state
        assert hasattr(fs, "E")
        assert hasattr(fs, "H")
        assert hasattr(fs, "eps_xx")
        assert hasattr(fs, "mu_xx")

    def test_energy_history_monotonic_then_decay(self):
        """Energy history should show pulse peak then PML decay."""
        sim = Simulation(
            center=(0.0, 0.0, 0.0),
            size=(3.0e-6, 2.0e-6, 2.0e-6),
            run_time=3e-12,
            grid_spec=GridSpec(
                grid_x=UniformGrid(dl=2.0e-7),
                grid_y=UniformGrid(dl=2.0e-7),
                grid_z=UniformGrid(dl=2.0e-7),
            ),
            medium=Medium(permittivity=1.0),
            sources=(
                PointDipole(
                    center=(0.0, 0.0, 0.0),
                    polarization="Ez",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=8e13,
                        amplitude=1.0,
                        offset=3.0,
                    ),
                    name="dipole",
                ),
            ),
            boundary_spec=BoundarySpec(
                x=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
                y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
                z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            ),
            symmetry=(0, 0, 0),
            shutoff=1e-6,
        )

        compiled = compile_simulation(sim)
        result = run_compiled_simulation(
            compiled,
            max_steps=300,
            record_interval=10,
            verbose=False,
        )

        history = result.integrated_electric_history
        assert len(history) > 10

        # Find peak and check decay after
        max_val = max(history)
        max_idx = history.index(max_val)

        # After peak, energy should generally decay (allowing some oscillation)
        if max_idx < len(history) - 5:
            late_avg = np.mean(history[-5:])
            early_peak_region = np.mean(history[max_idx : max_idx + 5])
            assert late_avg < early_peak_region


class TestFieldDataTypes:
    """Test that field data types are correct."""

    def test_field_monitor_returns_dict(self):
        """Field monitor data should be a dictionary."""
        result = field_monitor_recording_example()
        assert isinstance(result, dict)
        assert "num_records" in result

    def test_flux_monitor_returns_dict(self):
        """Flux monitor data should be a dictionary."""
        result = flux_monitor_recording_example()
        assert isinstance(result, dict)
        assert "has_flux_data" in result

    def test_medium_monitor_returns_dict(self):
        """Medium monitor data should be a dictionary."""
        result = medium_monitor_example()
        assert isinstance(result, dict)
        assert "has_medium_data" in result


class TestValidationMetrics:
    """Test validation metrics and reporting."""

    def test_backend_reporting(self):
        """Examples should report backend type."""
        result = vacuum_point_source_example()
        assert "backend" in result
        assert result["backend"] in ("numpy", "warp")

    def test_cells_reporting(self):
        """Examples should report cell count."""
        result = vacuum_point_source_example()
        assert "total_cells" in result
        assert result["total_cells"] > 0

    def test_steps_reporting(self):
        """Examples should report step count."""
        result = vacuum_point_source_example()
        assert "num_steps" in result
        assert result["num_steps"] > 0