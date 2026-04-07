"""Tests for wp.array preservation in source injection kernels.

This module tests that source injection functions preserve wp.array inputs
instead of silently converting them to NumPy. When a wp.array is passed in,
the function should return a wp.array on the same device.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import warp as wp
    WARP_AVAILABLE = True
except ModuleNotFoundError:
    WARP_AVAILABLE = False
    wp = None


def _resolved_grid():
    """Create a test grid for source compilation."""
    from autofdtd.api import GridSpec, Simulation, UniformGrid

    simulation = Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )
    return simulation.resolved_grid()


class TestWarpArrayPreservation:
    """Test that wp.array inputs are preserved through injection."""

    @pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp not available")
    def test_inject_uniform_current_source_preserves_wp_array(self):
        """inject_uniform_current_source returns wp.array when given wp.array input."""
        from autofdtd.compiler.sources import compile_uniform_current_source
        from autofdtd.kernels.sources import inject_uniform_current_source
        from autofdtd.sources import GaussianPulse, UniformCurrentSource

        grid = _resolved_grid()

        source = UniformCurrentSource(
            center=(5.0, 5.0, 5.0),
            size=(0.0, 0.0, 2.0),
            polarization="Ez",
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        )

        compiled = compile_uniform_current_source(source, grid=grid)

        # Create wp.array inputs
        dev = wp.get_device()
        electric_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)
        magnetic_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)

        # Apply injection
        e_result, h_result = inject_uniform_current_source(
            electric_wp,
            magnetic_wp,
            compiled,
            time=0.0,
            dt=0.1,
        )

        # Result should be wp.array on same device
        assert isinstance(e_result, wp.array), f"Expected wp.array, got {type(e_result)}"
        assert isinstance(h_result, wp.array), f"Expected wp.array, got {type(h_result)}"
        assert e_result.device == dev
        assert h_result.device == dev

    @pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp not available")
    def test_inject_sources_stage_preserves_wp_array(self):
        """inject_sources_stage returns wp.array when given wp.array inputs."""
        from autofdtd.compiler.sources import compile_plane_wave
        from autofdtd.kernels.sources import inject_sources_stage
        from autofdtd.sources import GaussianPulse, PlaneWave

        grid = _resolved_grid()

        source = PlaneWave(
            center=(5.0, 5.0, 5.0),
            size=(0.0, 10.0, 10.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
            direction="+",
        )

        compiled = compile_plane_wave(source, grid=grid)

        # Create wp.array inputs
        dev = wp.get_device()
        electric_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)
        magnetic_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)

        # Apply injection
        e_result, h_result = inject_sources_stage(
            electric_wp,
            magnetic_wp,
            plane_wave_sources=(compiled,),
            time=0.0,
            dt=0.1,
        )

        # Result should be wp.array on same device
        assert isinstance(e_result, wp.array), f"Expected wp.array, got {type(e_result)}"
        assert isinstance(h_result, wp.array), f"Expected wp.array, got {type(h_result)}"
        assert e_result.device == dev
        assert h_result.device == dev

    @pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp not available")
    def test_apply_source_injection_stage_preserves_wp_array(self):
        """apply_source_injection_stage returns wp.array when given wp.array inputs."""
        from autofdtd.compiler.sources import compile_uniform_current_source
        from autofdtd.runtime.sources import apply_source_injection_stage
        from autofdtd.sources import GaussianPulse, UniformCurrentSource

        grid = _resolved_grid()

        source = UniformCurrentSource(
            center=(5.0, 5.0, 5.0),
            size=(0.0, 0.0, 2.0),
            polarization="Ez",
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        )

        compiled = compile_uniform_current_source(source, grid=grid)

        # Create wp.array inputs
        dev = wp.get_device()
        electric_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)
        magnetic_wp = wp.zeros(shape=(10, 10, 10, 3), dtype=wp.float32, device=dev)

        # Apply injection via runtime function
        e_result, h_result = apply_source_injection_stage(
            electric_wp,
            magnetic_wp,
            uniform_current_sources=(compiled,),
            time=0.0,
            dt=0.1,
        )

        # Result should be wp.array on same device
        assert isinstance(e_result, wp.array), f"Expected wp.array, got {type(e_result)}"
        assert isinstance(h_result, wp.array), f"Expected wp.array, got {type(h_result)}"
        assert e_result.device == dev
        assert h_result.device == dev

    def test_inject_uniform_current_source_numpy_still_works(self):
        """inject_uniform_current_source still works with NumPy inputs."""
        from autofdtd.compiler.sources import compile_uniform_current_source
        from autofdtd.kernels.sources import inject_uniform_current_source
        from autofdtd.sources import GaussianPulse, UniformCurrentSource

        grid = _resolved_grid()

        source = UniformCurrentSource(
            center=(5.0, 5.0, 5.0),
            size=(0.0, 0.0, 2.0),
            polarization="Ez",
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        )

        compiled = compile_uniform_current_source(source, grid=grid)

        # Create NumPy inputs
        electric_np = np.zeros((10, 10, 10, 3), dtype=np.complex128)
        magnetic_np = np.zeros((10, 10, 10, 3), dtype=np.complex128)

        # Apply injection
        e_result, h_result = inject_uniform_current_source(
            electric_np,
            magnetic_np,
            compiled,
            time=0.0,
            dt=0.1,
        )

        # Result should be numpy array
        assert isinstance(e_result, np.ndarray), f"Expected numpy array, got {type(e_result)}"
        assert isinstance(h_result, np.ndarray), f"Expected numpy array, got {type(h_result)}"

        # Should have some injected values
        assert np.any(e_result != 0.0) or np.any(h_result != 0.0)

    def test_inject_plane_wave_numpy_still_works(self):
        """inject_plane_wave still works with NumPy inputs."""
        from autofdtd.compiler.sources import compile_plane_wave
        from autofdtd.kernels.sources import inject_plane_wave
        from autofdtd.sources import GaussianPulse, PlaneWave

        grid = _resolved_grid()

        source = PlaneWave(
            center=(5.0, 5.0, 5.0),
            size=(0.0, 10.0, 10.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
            direction="+",
        )

        compiled = compile_plane_wave(source, grid=grid)

        # Create NumPy inputs
        electric_np = np.zeros((10, 10, 10, 3), dtype=np.complex128)
        magnetic_np = np.zeros((10, 10, 10, 3), dtype=np.complex128)

        # Apply injection
        e_result, h_result = inject_plane_wave(
            electric_np,
            magnetic_np,
            compiled,
            time=0.0,
            dt=0.1,
        )

        # Result should be numpy array
        assert isinstance(e_result, np.ndarray), f"Expected numpy array, got {type(e_result)}"
        assert isinstance(h_result, np.ndarray), f"Expected numpy array, got {type(h_result)}"

        # Should have some injected values
        assert np.any(e_result != 0.0) or np.any(h_result != 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
