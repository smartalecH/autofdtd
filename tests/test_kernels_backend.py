"""Tests for Warp backend conventions in kernels.backend."""

import numpy as np
import pytest

from autofdtd.kernels.backend import (
    ArrayTags,
    AuxiliaryArrayTag,
    ComplexFieldPolicy,
    CoefficientArrayTag,
    FieldArrayTag,
    ModuleStability,
    StepMetrics,
    WarpBackendInfo,
    WarpTimer,
    allocate_auxiliary_array,
    allocate_coefficient_array,
    allocate_field_array,
    backend_info,
    backend_summary,
    get_default_device,
    get_warp_device,
    is_module_stable,
    mark_module_stable,
    nvtx_range,
    step_metrics,
    WARP_AVAILABLE,
)


class TestBackendAvailability:
    """Test Warp availability detection."""

    def test_backend_info_warp_available(self):
        """backend_info() returns WarpBackendInfo with correct warp_available field."""
        info = backend_info()
        assert isinstance(info, WarpBackendInfo)
        assert info.warp_available == WARP_AVAILABLE

    def test_backend_info_structured(self):
        """backend_info() returns a fully structured WarpBackendInfo."""
        info = backend_info()
        # All fields should be present even if Warp is unavailable
        assert hasattr(info, "warp_available")
        assert hasattr(info, "warp_version")
        assert hasattr(info, "cuda_available")
        assert hasattr(info, "num_devices")
        assert hasattr(info, "current_device")
        assert hasattr(info, "supports_graph_capture")
        assert hasattr(info, "supports_float64")

    def test_backend_summary_keys(self):
        """backend_summary() returns a dict with expected keys."""
        summary = backend_summary()
        assert isinstance(summary, dict)
        assert "warp_available" in summary
        assert "stable_modules" in summary


class TestDeviceManagement:
    """Test device selection and management."""

    def test_get_warp_device_returns_none_when_unavailable(self):
        """get_warp_device() returns None when Warp is unavailable."""
        if not WARP_AVAILABLE:
            result = get_warp_device()
            assert result is None

    def test_get_default_device_returns_none_when_unavailable(self):
        """get_default_device() returns None when Warp is unavailable."""
        if not WARP_AVAILABLE:
            result = get_default_device()
            assert result is None


class TestArrayTags:
    """Test array classification tags."""

    def test_field_array_tag_fields(self):
        """FieldArrayTag stores family and chunk_index."""
        tag = FieldArrayTag(family="electric")
        assert tag.family == "electric"
        assert tag.chunk_index is None

        tag_chunked = FieldArrayTag(family="magnetic", chunk_index=(0, 0, 0))
        assert tag_chunked.family == "magnetic"
        assert tag_chunked.chunk_index == (0, 0, 0)

    def test_coefficient_array_tag_fields(self):
        """CoefficientArrayTag stores component, axis, and chunk_index."""
        tag = CoefficientArrayTag(component="eps", axis="xx")
        assert tag.component == "eps"
        assert tag.axis == "xx"
        assert tag.chunk_index is None

        tag_chunked = CoefficientArrayTag(
            component="mu", axis="zz", chunk_index=(1, 2, 3)
        )
        assert tag_chunked.component == "mu"
        assert tag_chunked.axis == "zz"
        assert tag_chunked.chunk_index == (1, 2, 3)

    def test_auxiliary_array_tag_fields(self):
        """AuxiliaryArrayTag stores medium_family and chunk_index."""
        tag = AuxiliaryArrayTag(medium_family="PoleResidue")
        assert tag.medium_family == "PoleResidue"
        assert tag.chunk_index is None

        tag_chunked = AuxiliaryArrayTag(
            medium_family="Lorentz", chunk_index=(0, 1, 0)
        )
        assert tag_chunked.medium_family == "Lorentz"
        assert tag_chunked.chunk_index == (0, 1, 0)

    def test_array_tags_container(self):
        """ArrayTags bundles all tag types."""
        field_tag = FieldArrayTag(family="electric")
        coeff_tag = CoefficientArrayTag(component="eps", axis="xx")
        aux_tag = AuxiliaryArrayTag(medium_family="PoleResidue")

        tags = ArrayTags(
            fields=(field_tag,),
            coefficients=(coeff_tag,),
            auxiliary=(aux_tag,),
        )

        assert len(tags.fields) == 1
        assert len(tags.coefficients) == 1
        assert len(tags.auxiliary) == 1
        assert tags.fields[0].family == "electric"


class TestArrayAllocation:
    """Test array allocation helpers return correct types."""

    def test_allocate_field_array_returns_numpy_when_unavailable(self):
        """allocate_field_array returns np.ndarray when Warp unavailable."""
        if not WARP_AVAILABLE:
            arr = allocate_field_array((10, 10, 10), family="electric")
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (10, 10, 10, 3)
            assert arr.dtype == np.float64

    def test_allocate_field_array_shape(self):
        """allocate_field_array returns shape (*shape, 3) for vector fields."""
        if not WARP_AVAILABLE:
            arr = allocate_field_array((5, 6, 7), family="magnetic")
            assert arr.shape == (5, 6, 7, 3)

    def test_allocate_coefficient_array_returns_numpy_when_unavailable(self):
        """allocate_coefficient_array returns np.ndarray when Warp unavailable."""
        if not WARP_AVAILABLE:
            arr = allocate_coefficient_array(
                (8, 8, 8), component="eps", axis="yy"
            )
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (8, 8, 8)
            assert arr.dtype == np.float64

    def test_allocate_auxiliary_array_returns_numpy_when_unavailable(self):
        """allocate_auxiliary_array returns np.ndarray when Warp unavailable."""
        if not WARP_AVAILABLE:
            arr = allocate_auxiliary_array((4, 4, 4), num_poles=3)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (3, 4, 4, 4)
            assert arr.dtype == np.complex128

    def test_allocate_auxiliary_array_tag_attached(self):
        """allocate_auxiliary_array attaches AuxiliaryArrayTag when tag=True and Warp available."""
        # When Warp is unavailable, the function returns a plain numpy array without a tag
        # The tag is only attached to wp.array objects when Warp is available
        if WARP_AVAILABLE:
            arr = allocate_auxiliary_array((4, 4, 4), num_poles=2, tag=True)
            assert hasattr(arr, "tag")
            assert isinstance(arr.tag, AuxiliaryArrayTag)
        else:
            # When Warp is unavailable, numpy array is returned without tag
            arr = allocate_auxiliary_array((4, 4, 4), num_poles=2, tag=True)
            assert isinstance(arr, np.ndarray)
            assert not hasattr(arr, "tag")


class TestComplexFieldPolicy:
    """Test complex dtype policy determination."""

    def test_complex_field_policy_defaults(self):
        """ComplexFieldPolicy defaults to scalar (not complex)."""
        policy = ComplexFieldPolicy()
        assert policy.force_complex is False
        assert policy.has_bloch_axis is False
        assert policy.requires_complex is False

    def test_complex_field_policy_force_complex(self):
        """ComplexFieldPolicy.force_complex forces complex dtype."""
        policy = ComplexFieldPolicy(force_complex=True)
        assert policy.requires_complex is True

    def test_complex_field_policy_bloch_infers_complex(self):
        """ComplexFieldPolicy.has_bloch_axis=True forces complex dtype."""
        policy = ComplexFieldPolicy(has_bloch_axis=True)
        assert policy.requires_complex is True


class TestModuleStability:
    """Test module stability tracking."""

    def test_mark_and_check_module_stable(self):
        """mark_module_stable() and is_module_stable() work as a pair."""
        test_module = "test_kernel_module_v1"

        # Initially not stable
        assert is_module_stable(test_module) is False

        # Mark as stable
        mark_module_stable(test_module)
        assert is_module_stable(test_module) is True

    def test_unmarked_module_is_unstable(self):
        """Unmarked modules are reported as unstable."""
        result = is_module_stable("definitely_new_module_xyz")
        assert result is False

    def test_stable_modules_accumulate(self):
        """Multiple modules can be marked stable independently."""
        mark_module_stable("module_a")
        mark_module_stable("module_b")

        assert is_module_stable("module_a") is True
        assert is_module_stable("module_b") is True


class TestProfilingHooks:
    """Test profiling hooks and metrics."""

    def test_step_metrics_computes_gcells_per_second(self):
        """step_metrics computes Gcells/s from wall time and cell count."""
        metrics = step_metrics(
            step_index=0,
            wall_time_s=0.001,  # 1ms
            num_cells=1_000_000,  # 1 million cells
        )

        assert isinstance(metrics, StepMetrics)
        assert metrics.step_index == 0
        assert metrics.gcells_per_second is not None
        # 1M cells in 1ms = 1e6/1e-3 = 1e9 cells/s = 1 Gcell/s
        assert metrics.gcells_per_second == pytest.approx(1.0, rel=0.01)

    def test_step_metrics_zero_time_handled(self):
        """step_metrics handles zero wall time gracefully."""
        metrics = step_metrics(
            step_index=0,
            wall_time_s=0.0,
            num_cells=1_000_000,
        )
        # gcells_per_second should be None when time is zero (avoid div/0)
        assert metrics.gcells_per_second is None

    def test_step_metrics_initial_step_flag(self):
        """step_metrics preserves initial_step flag."""
        metrics = step_metrics(
            step_index=0,
            wall_time_s=0.1,
            num_cells=1_000_000,
            initial_step=True,
        )
        assert metrics.initial_step is True

    def test_nvtx_range_context_manager(self):
        """nvtx_range context manager is callable without error."""
        # Should not raise even when Warp is unavailable
        with nvtx_range("test_region", color="blue"):
            pass  # empty region

    def test_warp_timer_context_manager(self):
        """WarpTimer context manager is callable without error."""
        # Should not raise even when Warp is unavailable
        with WarpTimer("test_timer"):
            pass  # empty timed region


class TestBackendSummary:
    """Test backend summary diagnostics."""

    def test_backend_summary_returns_dict(self):
        """backend_summary() returns a dict."""
        summary = backend_summary()
        assert isinstance(summary, dict)

    def test_backend_summary_includes_stable_modules(self):
        """backend_summary includes stable_modules list."""
        summary = backend_summary()
        assert "stable_modules" in summary
        assert isinstance(summary["stable_modules"], list)
