"""Tests for diagnostics.metrics module."""

from __future__ import annotations

import pytest

from autofdtd.diagnostics.metrics import (
    BenchmarkBoundaryClass,
    BenchmarkMaterialClass,
    BenchmarkMetadata,
    BenchmarkPrecision,
    BenchmarkResult,
    BenchmarkTimer,
    EndToEndTiming,
    InitializationTiming,
    JitTiming,
    SteadyStateTiming,
    StepEvidence,
    build_benchmark_metadata,
    build_steady_state_timing,
    compute_cells_per_second,
    compute_gcells_per_second,
)


class TestComputeGcellsPerSecond:
    """Test Gcells/s computation helper."""

    def test_computes_gcells_per_second_from_cells_and_time(self):
        """compute_gcells_per_second returns correct Gcells/s."""
        # 1M cells in 1ms = 1e6/1e-3 = 1e9 cells/s = 1 Gcell/s
        result = compute_gcells_per_second(1_000_000, 0.001)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_handles_zero_time(self):
        """compute_gcells_per_second returns None for zero time."""
        result = compute_gcells_per_second(1_000_000, 0.0)
        assert result is None

    def test_handles_negative_time(self):
        """compute_gcells_per_second returns None for negative time."""
        result = compute_gcells_per_second(1_000_000, -1.0)
        assert result is None

    def test_large_grid_gcells_per_second(self):
        """compute_gcells_per_second handles large grids correctly."""
        # 100M cells in 0.01s = 1e8/1e-2 = 1e10 cells/s = 10 Gcells/s
        result = compute_gcells_per_second(100_000_000, 0.01)
        assert result == pytest.approx(10.0, rel=0.01)


class TestComputeCellsPerSecond:
    """Test cells/s computation helper."""

    def test_computes_cells_per_second(self):
        """compute_cells_per_second returns correct cells/s."""
        result = compute_cells_per_second(1_000_000, 0.001)
        assert result == pytest.approx(1e9, rel=0.01)

    def test_handles_zero_time(self):
        """compute_cells_per_second returns None for zero time."""
        result = compute_cells_per_second(1_000_000, 0.0)
        assert result is None


class TestBuildBenchmarkMetadata:
    """Test benchmark metadata builder."""

    def test_builds_basic_metadata(self):
        """build_benchmark_metadata creates a valid BenchmarkMetadata."""
        meta = build_benchmark_metadata(
            grid_shape=(10, 20, 30),
            num_steps=100,
        )
        assert meta.grid_shape == (10, 20, 30)
        assert meta.total_cells == 10 * 20 * 30
        assert meta.num_steps == 100
        assert meta.precision == BenchmarkPrecision.FLOAT32
        assert meta.single_gpu is True

    def test_respects_precision_parameter(self):
        """build_benchmark_metadata accepts precision parameter."""
        meta = build_benchmark_metadata(
            grid_shape=(10, 10, 10),
            num_steps=50,
            precision=BenchmarkPrecision.FLOAT64,
        )
        assert meta.precision == BenchmarkPrecision.FLOAT64

    def test_respects_boundary_class(self):
        """build_benchmark_metadata accepts boundary_class."""
        meta = build_benchmark_metadata(
            grid_shape=(10, 10, 10),
            num_steps=50,
            boundary_class=BenchmarkBoundaryClass.PML,
        )
        assert meta.boundary_class == BenchmarkBoundaryClass.PML

    def test_respects_material_class(self):
        """build_benchmark_metadata accepts material_class."""
        meta = build_benchmark_metadata(
            grid_shape=(10, 10, 10),
            num_steps=50,
            material_class=BenchmarkMaterialClass.DISPERSIVE,
        )
        assert meta.material_class == BenchmarkMaterialClass.DISPERSIVE

    def test_single_gpu_flag(self):
        """single_gpu is True when multi_gpu_count is None or <= 1."""
        meta1 = build_benchmark_metadata((10, 10, 10), 50)
        assert meta1.single_gpu is True

        meta2 = build_benchmark_metadata((10, 10, 10), 50, multi_gpu_count=1)
        assert meta2.single_gpu is True

        meta3 = build_benchmark_metadata((10, 10, 10), 50, multi_gpu_count=4)
        assert meta3.single_gpu is False
        assert meta3.multi_gpu_count == 4

    def test_flag_fields(self):
        """has_bloch, has_pml, has_dispersive, has_anisotropic are set."""
        meta = build_benchmark_metadata(
            (10, 10, 10),
            50,
            has_bloch=True,
            has_pml=True,
            has_dispersive=True,
            has_anisotropic=True,
        )
        assert meta.has_bloch is True
        assert meta.has_pml is True
        assert meta.has_dispersive is True
        assert meta.has_anisotropic is True

    def test_metadata_to_payload(self):
        """BenchmarkMetadata.to_payload() returns JSON-ready dict."""
        meta = build_benchmark_metadata(
            grid_shape=(5, 5, 5),
            num_steps=10,
        )
        payload = meta.to_payload()
        assert isinstance(payload, dict)
        assert payload["grid_shape"] == (5, 5, 5)
        assert payload["total_cells"] == 125
        assert payload["num_steps"] == 10


class TestBuildSteadyStateTiming:
    """Test steady-state timing builder."""

    def test_builds_steady_state_timing(self):
        """build_steady_state_timing computes correct timing."""
        step_times = [0.001, 0.001, 0.001, 0.001]  # 1ms per step, 4 steps
        timing = build_steady_state_timing(step_times, num_cells=1_000_000)

        assert timing.wall_time_s == pytest.approx(0.004, rel=0.01)
        assert timing.steps_executed == 4
        assert timing.cells_updated_per_step == 1_000_000
        assert timing.gcells_per_second == pytest.approx(1.0, rel=0.01)
        assert timing.min_step_time_s == pytest.approx(0.001, rel=0.01)
        assert timing.max_step_time_s == pytest.approx(0.001, rel=0.01)

    def test_warm_up_steps_excludes_initial_steps(self):
        """build_steady_state_timing supports warm_up_steps."""
        step_times = [0.010, 0.010, 0.001, 0.001, 0.001, 0.001]  # first 2 slow
        timing = build_steady_state_timing(step_times, num_cells=1_000_000, warm_up_steps=2)

        # Should only count the 4 fast steps
        assert timing.steps_executed == 4
        assert timing.wall_time_s == pytest.approx(0.004, rel=0.01)

    def test_empty_step_times_raises(self):
        """build_steady_state_timing raises on empty step_times."""
        with pytest.raises(ValueError, match="step_times must not be empty"):
            build_steady_state_timing([], num_cells=1_000_000)

    def test_warm_up_too_large_raises(self):
        """build_steady_state_timing raises when warm_up_steps too large."""
        with pytest.raises(ValueError, match="warm_up_steps too large"):
            build_steady_state_timing([0.001], num_cells=1_000_000, warm_up_steps=5)


class TestBenchmarkTimer:
    """Test benchmark timer context manager."""

    def test_phase_context_manager(self):
        """BenchmarkTimer.phase() works as a context manager."""
        timer = BenchmarkTimer()
        with timer.phase("test_phase"):
            pass
        timing = timer.get_timing("test_phase")
        assert "wall_time_s" in timing
        assert timing["wall_time_s"] >= 0.0

    def test_mark_records_sub_markers(self):
        """BenchmarkTimer.mark() records sub-markers within a phase."""
        timer = BenchmarkTimer()
        with timer.phase("init"):
            timer.mark("sub1")
            timer.mark("sub2")
        timing = timer.get_timing("init")
        assert "sub1_s" in timing
        assert "sub2_s" in timing

    def test_no_active_phase_mark_is_noop(self):
        """BenchmarkTimer.mark() is no-op when no phase is active."""
        timer = BenchmarkTimer()
        timer.mark("something")  # no active phase
        # should not raise

    def test_get_timing_unknown_phase(self):
        """BenchmarkTimer.get_timing() returns empty dict for unknown phase."""
        timer = BenchmarkTimer()
        result = timer.get_timing("nonexistent")
        assert result == {}


class TestBenchmarkMetadataEnums:
    """Test benchmark enum classes."""

    def test_benchmark_precision_values(self):
        """BenchmarkPrecision has expected values."""
        assert BenchmarkPrecision.FLOAT32.value == "float32"
        assert BenchmarkPrecision.FLOAT64.value == "float64"
        assert BenchmarkPrecision.COMPLEX64.value == "complex64"
        assert BenchmarkPrecision.COMPLEX128.value == "complex128"

    def test_benchmark_boundary_class_values(self):
        """BenchmarkBoundaryClass has expected values."""
        assert BenchmarkBoundaryClass.PEC.value == "pec"
        assert BenchmarkBoundaryClass.PMC.value == "pmc"
        assert BenchmarkBoundaryClass.PERIODIC.value == "periodic"
        assert BenchmarkBoundaryClass.BLOCH.value == "bloch"
        assert BenchmarkBoundaryClass.PML.value == "pml"
        assert BenchmarkBoundaryClass.ABSORBER.value == "absorber"
        assert BenchmarkBoundaryClass.MIXED.value == "mixed"

    def test_benchmark_material_class_values(self):
        """BenchmarkMaterialClass has expected values."""
        assert BenchmarkMaterialClass.VACUUM.value == "vacuum"
        assert BenchmarkMaterialClass.DIELECTRIC.value == "dielectric"
        assert BenchmarkMaterialClass.DISPERSIVE.value == "dispersive"
        assert BenchmarkMaterialClass.ANISOTROPIC.value == "anisotropic"
        assert BenchmarkMaterialClass.PEC.value == "pec"


class TestInitializationTiming:
    """Test InitializationTiming dataclass."""

    def test_initialization_timing_to_payload(self):
        """InitializationTiming.to_payload() works."""
        timing = InitializationTiming(
            wall_time_s=1.5,
            grid_resolution_s=0.3,
            scene_materialization_s=0.5,
        )
        payload = timing.to_payload()
        assert payload["wall_time_s"] == 1.5
        assert payload["grid_resolution_s"] == 0.3
        assert payload["scene_materialization_s"] == 0.5


class TestJitTiming:
    """Test JitTiming dataclass."""

    def test_jit_timing_to_payload(self):
        """JitTiming.to_payload() works."""
        timing = JitTiming(wall_time_s=0.5, module_stable=True)
        payload = timing.to_payload()
        assert payload["wall_time_s"] == 0.5
        assert payload["module_stable"] is True


class TestSteadyStateTiming:
    """Test SteadyStateTiming dataclass."""

    def test_steady_state_timing_to_payload(self):
        """SteadyStateTiming.to_payload() works."""
        timing = SteadyStateTiming(
            wall_time_s=10.0,
            steps_executed=100,
            cells_updated_per_step=1_000_000,
            gcells_per_second=1.0,
            min_step_time_s=0.09,
            max_step_time_s=0.11,
        )
        payload = timing.to_payload()
        assert payload["wall_time_s"] == 10.0
        assert payload["steps_executed"] == 100
        assert payload["gcells_per_second"] == 1.0


class TestEndToEndTiming:
    """Test EndToEndTiming dataclass."""

    def test_end_to_end_timing_to_payload(self):
        """EndToEndTiming.to_payload() works."""
        init_timing = InitializationTiming(wall_time_s=1.0)
        steady = SteadyStateTiming(
            wall_time_s=10.0,
            steps_executed=100,
            cells_updated_per_step=1_000_000,
        )
        timing = EndToEndTiming(
            initialization=init_timing,
            jit=JitTiming(wall_time_s=0.2),
            steady_state=steady,
            total_wall_time_s=11.2,
        )
        payload = timing.to_payload()
        assert payload["total_wall_time_s"] == 11.2
        assert payload["initialization"]["wall_time_s"] == 1.0
        assert payload["jit"]["wall_time_s"] == 0.2


class TestStepEvidence:
    """Test StepEvidence dataclass."""

    def test_step_evidence_to_payload(self):
        """StepEvidence.to_payload() works."""
        evidence = StepEvidence(
            step_index=5,
            wall_time_s=0.001,
            integrated_electric=0.5,
            gcells_per_second=1.0,
        )
        payload = evidence.to_payload()
        assert payload["step_index"] == 5
        assert payload["wall_time_s"] == 0.001
        assert payload["integrated_electric"] == 0.5
        assert payload["gcells_per_second"] == 1.0