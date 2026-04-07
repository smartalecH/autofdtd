"""Tests for near-to-far (N2F) postprocessing kernels and runtime."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autofdtd.kernels.near2far import (
    C0,
    Z0,
    compute_diffraction_orders_n2f,
    compute_surface_currents,
    compute_surface_positions,
    far_field_power,
    near2far_kernel_metadata,
    near2far_transform,
    extract_radiation_pattern,
    project_to_cartesian_plane,
)
from autofdtd.runtime.near2far import (
    build_near2far_surface,
    compute_diffraction_from_dft,
    compute_radiation_from_dft,
    compute_total_radiated_power,
    far_field_directivity,
    near2far_runtime_metadata,
    project_far_field_to_cartesian,
)


class TestNear2farKernelMetadata:
    """Test near2far kernel metadata reporting."""

    def test_near2far_kernel_metadata_keys(self):
        """Metadata should contain expected backend keys."""
        meta = near2far_kernel_metadata()
        assert "backend" in meta
        assert "warp_available" in meta
        assert "staging" in meta
        assert meta["backend"] == "numpy"

    def test_near2far_staging(self):
        """Staging should include all N2F stages."""
        meta = near2far_kernel_metadata()
        assert "surface_dft" in meta["staging"]
        assert "greens_function" in meta["staging"]
        assert "far_field_projection" in meta["staging"]


class TestSurfaceCurrents:
    """Test surface current computation from DFT fields."""

    def test_surface_currents_z_normal(self):
        """Test J_s and M_s computation for z-normal surface."""
        n_pts = 10
        n_freqs = 3

        # Create sample DFT data at z-normal surface
        dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

        # Set some non-zero fields
        dft_e[:, :, 0] = 1.0  # Ex
        dft_e[:, :, 1] = 2.0  # Ey
        dft_h[:, :, 0] = 0.1  # Hx
        dft_h[:, :, 1] = 0.2  # Hy

        placements = tuple((0, 0, i) for i in range(n_pts))
        cell_sizes = (0.1, 0.1, 0.1)

        Js, Ms = compute_surface_currents(dft_e, dft_h, placements, normal_axis=2, cell_sizes=cell_sizes)

        # For z-normal surface (n = (0, 0, 1)):
        # J_s = n × H = (Hy, -Hx, 0)
        # M_s = -n × E = (-Ey, Ex, 0)
        assert Js.shape == (n_pts, n_freqs, 3)
        assert Ms.shape == (n_pts, n_freqs, 3)

        # J_sz and M_sz should be zero for z-normal
        np.testing.assert_allclose(Js[:, :, 2], 0.0)
        np.testing.assert_allclose(Ms[:, :, 2], 0.0)

    def test_surface_currents_x_normal(self):
        """Test J_s and M_s computation for x-normal surface."""
        n_pts = 5
        n_freqs = 2

        dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

        dft_h[:, :, 1] = 1.0  # Hy
        dft_h[:, :, 2] = 2.0  # Hz
        dft_e[:, :, 1] = 3.0  # Ey
        dft_e[:, :, 2] = 4.0  # Ez

        placements = tuple((i, 0, 0) for i in range(n_pts))
        cell_sizes = (0.1, 0.1, 0.1)

        Js, Ms = compute_surface_currents(dft_e, dft_h, placements, normal_axis=0, cell_sizes=cell_sizes)

        # For x-normal (n = (1, 0, 0)):
        # J_s = n × H = (0, Hz, -Hy)
        # M_s = -n × E = (0, -Ez, Ey)
        np.testing.assert_allclose(Js[:, :, 0], 0.0)
        np.testing.assert_allclose(Ms[:, :, 0], 0.0)


class TestSurfacePositions:
    """Test physical position computation for surface cells."""

    def test_positions_single_cell(self):
        """Test position computation for single cell."""
        placements = ((5, 3, 2),)
        center = (1.0, 2.0, 3.0)
        size = (0.0, 0.5, 0.5)
        cell_sizes = (0.1, 0.1, 0.1)

        positions = compute_surface_positions(placements, center, size, cell_sizes)

        assert positions.shape == (1, 3)
        # Position should be lower_bound + (index + 0.5) * cell_size
        # lower_bound = center - size/2 = (1.0, 2.0, 3.0) - (0.0, 0.25, 0.25) = (1.0, 1.75, 2.75)
        # Position x: 1.0 + (5 + 0.5) * 0.1 = 1.55
        # Position y: 1.75 + (3 + 0.5) * 0.1 = 2.1
        # Position z: 2.75 + (2 + 0.5) * 0.1 = 3.0
        np.testing.assert_allclose(positions[0, 0], 1.55)
        np.testing.assert_allclose(positions[0, 1], 2.1)
        np.testing.assert_allclose(positions[0, 2], 3.0)

    def test_positions_multiple_cells(self):
        """Test position computation for multiple cells."""
        placements = ((0, 0, 0), (1, 0, 0), (2, 0, 0))
        center = (0.0, 0.0, 0.0)
        size = (0.0, 0.3, 0.3)
        cell_sizes = (0.1, 0.1, 0.1)

        positions = compute_surface_positions(placements, center, size, cell_sizes)

        assert positions.shape == (3, 3)
        # x positions should be at 0.05, 0.15, 0.25 (index + 0.5) * dx
        np.testing.assert_allclose(positions[:, 0], [0.05, 0.15, 0.25])


class TestNear2farTransform:
    """Test far-field transformation from surface currents."""

    def test_far_field_zeros_when_no_currents(self):
        """Zero currents should produce zero far field."""
        n_pts = 5
        n_freqs = 1
        n_phi = 10
        n_theta = 10

        Js = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        Ms = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        positions = np.zeros((n_pts, 3), dtype=np.float64)
        cell_areas = np.ones(n_pts)

        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0, np.pi, n_theta)
        freqs = (1e14,)

        e_theta, e_phi = near2far_transform(
            Js, Ms, positions, obs_distance=1e5,
            phi_vals=phi_vals, theta_vals=theta_vals, freqs=freqs,
            cell_areas=cell_areas,
        )

        assert e_theta.shape == (n_phi, n_theta, n_freqs)
        assert e_phi.shape == (n_phi, n_theta, n_freqs)
        np.testing.assert_allclose(e_theta, 0.0)
        np.testing.assert_allclose(e_phi, 0.0)

    def test_far_field_propagation_direction(self):
        """Test that far field is computed correctly for z-directed current."""
        n_pts = 20
        n_freqs = 1
        n_phi = 36
        n_theta = 19

        # Create a uniform current distribution
        Js = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        Ms = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        Js[:, 0, 2] = 1.0  # z-directed current (radiates in xy plane)

        # Create planar positions in xy plane
        positions = np.zeros((n_pts, 3), dtype=np.float64)
        for i in range(n_pts):
            positions[i, 0] = (i % 4) * 0.1
            positions[i, 1] = (i // 4) * 0.1
            positions[i, 2] = 0.0

        cell_areas = np.full(n_pts, 0.01)

        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0, np.pi, n_theta)
        freqs = (1e14,)

        e_theta, e_phi = near2far_transform(
            Js, Ms, positions, obs_distance=1e5,
            phi_vals=phi_vals, theta_vals=theta_vals, freqs=freqs,
            cell_areas=cell_areas,
        )

        # Far field should be computed and non-zero for z-directed current
        assert e_theta.shape == (n_phi, n_theta, n_freqs)
        assert e_phi.shape == (n_phi, n_theta, n_freqs)
        # The total power should be non-zero for z-directed current
        power = np.abs(e_theta)**2 + np.abs(e_phi)**2
        assert np.sum(power) > 0


class TestFarFieldPower:
    """Test radiated power computation."""

    def test_power_zeros_for_zero_fields(self):
        """Zero fields should produce zero power."""
        e_theta = np.zeros((5, 5, 2))
        e_phi = np.zeros((5, 5, 2))
        freqs = (1e14, 2e14)

        power = far_field_power(e_theta, e_phi, freqs)

        assert power.shape == (5, 5, 2)
        np.testing.assert_allclose(power, 0.0)

    def test_power_formula(self):
        """Test power formula S = |E|²/(2Z0)."""
        e_theta = np.array([[[1.0 + 0.0j]]])  # Single point
        e_phi = np.array([[[0.0 + 0.0j]]])
        freqs = (1e14,)

        power = far_field_power(e_theta, e_phi, freqs)

        expected = np.abs(1.0)**2 / (2.0 * Z0)
        np.testing.assert_allclose(power[0, 0, 0], expected)


class TestDiffractionOrders:
    """Test diffraction order computation for periodic structures."""

    def test_diffraction_no_periods(self):
        """Without periods, should return zero amplitudes."""
        n_pts = 10
        n_freqs = 3

        dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        placements = tuple((0, i, 0) for i in range(n_pts))
        freqs = (1e14, 1.5e14, 2e14)

        orders, mx, my = compute_diffraction_orders_n2f(
            dft_e, dft_h, placements, normal_axis=0,
            freqs=freqs, period_y=None, period_z=None,
        )

        # Without valid periods, orders should be zero
        np.testing.assert_allclose(orders, 0.0)

    def test_diffraction_propagating_criterion(self):
        """Test propagating vs evanescent order determination."""
        n_pts = 5
        n_freqs = 1

        dft_e = np.ones((n_pts, n_freqs, 3), dtype=np.complex128) * 0.1
        dft_h = np.ones((n_pts, n_freqs, 3), dtype=np.complex128) * 0.01
        placements = tuple((0, i, 0) for i in range(n_pts))
        freqs = (1e14,)

        orders, mx, my = compute_diffraction_orders_n2f(
            dft_e, dft_h, placements, normal_axis=0,
            freqs=freqs, period_y=1e-6, period_z=1e-6,
            medium_eps=1.0 + 0.0j, num_orders=1,
        )

        assert orders.shape[0] == 9  # 3x3 orders (mx=-1,0,1 x my=-1,0,1)
        assert len(mx) == 9
        assert len(my) == 9


class TestExtractRadiationPattern:
    """Test radiation pattern extraction."""

    def test_pattern_keys(self):
        """Pattern should contain expected components."""
        n_phi = 10
        n_theta = 10
        n_freqs = 2

        e_theta = np.zeros((n_phi, n_theta, n_freqs))
        e_phi = np.zeros((n_phi, n_theta, n_freqs))
        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0, np.pi, n_theta)
        freqs = (1e14, 2e14)

        pattern = extract_radiation_pattern(e_theta, e_phi, phi_vals, theta_vals, freqs)

        assert "e_theta" in pattern
        assert "e_phi" in pattern
        assert "power" in pattern
        assert "phi_vals" in pattern
        assert "theta_vals" in pattern
        assert "freqs" in pattern


class TestProjectToCartesian:
    """Test Cartesian plane projection."""

    def test_projection_output_shape(self):
        """Output shape should match observation grid."""
        n_phi = 20
        n_theta = 10
        n_freqs = 2

        e_theta = np.random.randn(n_phi, n_theta, n_freqs) + 1j * np.random.randn(n_phi, n_theta, n_freqs)
        e_phi = np.random.randn(n_phi, n_theta, n_freqs) + 1j * np.random.randn(n_phi, n_theta, n_freqs)
        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0.1, np.pi - 0.1, n_theta)
        freqs = (1e14, 2e14)

        x_obs = np.linspace(-1, 1, 5)
        y_obs = np.linspace(-1, 1, 7)

        ex, ey, ez = project_to_cartesian_plane(
            e_theta, e_phi, phi_vals, theta_vals, freqs,
            x_obs, y_obs, obs_distance=1e5,
        )

        assert ex.shape == (5, 7, n_freqs)
        assert ey.shape == (5, 7, n_freqs)
        assert ez.shape == (5, 7, n_freqs)


class TestNear2farRuntimeMetadata:
    """Test N2F runtime metadata."""

    def test_runtime_metadata_keys(self):
        """Runtime metadata should include expected fields."""
        meta = near2far_runtime_metadata()
        assert "equivalence_principle" in meta
        assert "greens_function" in meta
        assert meta["equivalence_principle"] == "J_s = n × H, M_s = -n × E"


class TestBuildNear2farSurface:
    """Test N2F surface building."""

    def test_cell_area_computation_z_normal(self):
        """Cell area should be product of tangential dimensions."""
        monitor_center = (0.0, 0.0, 0.0)
        monitor_size = (0.0, 1.0, 1.0)  # z-normal surface
        placements = tuple((0, i, j) for i in range(10) for j in range(10))
        normal_axis = 2
        cell_sizes = (0.1, 0.1, 0.1)

        surface_info = build_near2far_surface(
            monitor_center, monitor_size, placements,
            normal_axis, cell_sizes,
        )

        # Area should be dy * dz = 0.1 * 0.1 = 0.01
        assert surface_info["cell_areas"][0] == pytest.approx(0.01)
        assert surface_info["normal_axis"] == 2

    def test_positions_from_surface_info(self):
        """Positions should be stored in surface info."""
        monitor_center = (0.5, 0.0, 0.0)
        monitor_size = (0.0, 0.2, 0.2)
        placements = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
        normal_axis = 0
        cell_sizes = (0.1, 0.1, 0.1)

        surface_info = build_near2far_surface(
            monitor_center, monitor_size, placements,
            normal_axis, cell_sizes,
        )

        assert surface_info["positions"].shape == (3, 3)


class TestComputeRadiationFromDft:
    """Test radiation computation from DFT data."""

    def test_radiation_with_synthetic_data(self):
        """Test with simple synthetic DFT data."""
        n_pts = 5
        n_freqs = 1

        dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
        dft_e[:, :, 0] = 1.0  # x-polarized E
        dft_h[:, :, 1] = 0.01  # y-polarized H

        # Create proper surface info with positions
        positions = np.zeros((n_pts, 3), dtype=np.float64)
        positions[:, 0] = np.arange(n_pts) * 0.1
        surface_info = {
            "positions": positions,
            "cell_areas": np.ones(n_pts) * 0.01,
            "normal_axis": 2,
            "cell_sizes": (0.1, 0.1, 0.1),
        }

        phi_vals = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
        theta_vals = (np.pi / 2,)  # Equatorial plane
        freqs = (1e14,)

        pattern = compute_radiation_from_dft(
            dft_e, dft_h, surface_info,
            obs_distance=1e5, phi_vals=phi_vals, theta_vals=theta_vals, freqs=freqs,
        )

        assert "power" in pattern
        assert pattern["power"].shape[2] == n_freqs


class TestComputeDiffractionFromDft:
    """Test diffraction computation from DFT data."""

    def test_diffraction_output_structure(self):
        """Output should have correct structure."""
        n_pts = 10
        n_freqs = 2

        dft_e = np.random.randn(n_pts, n_freqs, 3) + 1j * np.random.randn(n_pts, n_freqs, 3)
        dft_h = np.random.randn(n_pts, n_freqs, 3) + 1j * np.random.randn(n_pts, n_freqs, 3)
        freqs = (1e14, 2e14)

        orders, mx, my = compute_diffraction_from_dft(
            dft_e, dft_h, normal_axis=2, freqs=freqs,
            period_y=1e-6, period_z=1e-6,
        )

        assert orders.shape[1] == n_freqs  # num_freqs dimension
        assert len(mx) == len(my)


class TestComputeTotalRadiatedPower:
    """Test total radiated power integration."""

    def test_total_power_with_uniform_pattern(self):
        """Uniform pattern should give non-zero total power."""
        n_phi = 20
        n_theta = 10
        n_freqs = 1

        # Uniform field in theta direction
        e_theta = np.ones((n_phi, n_theta, n_freqs)) * (1.0 + 0.0j)
        e_phi = np.zeros((n_phi, n_theta, n_freqs))
        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0.1, np.pi - 0.1, n_theta)
        freqs = (1e14,)

        total_power = compute_total_radiated_power(e_theta, e_phi, phi_vals, theta_vals, freqs)

        assert total_power.shape == (n_freqs,)
        assert total_power[0] > 0

    def test_total_power_zero_for_zero_fields(self):
        """Zero fields should give zero total power."""
        e_theta = np.zeros((5, 5, 2))
        e_phi = np.zeros((5, 5, 2))
        phi_vals = np.linspace(0, 2 * np.pi, 5)
        theta_vals = np.linspace(0.1, np.pi - 0.1, 5)
        freqs = (1e14, 2e14)

        total_power = compute_total_radiated_power(e_theta, e_phi, phi_vals, theta_vals, freqs)

        np.testing.assert_allclose(total_power, 0.0)


class TestFarFieldDirectivity:
    """Test far-field directivity computation."""

    def test_directivity_dbi(self):
        """Directivity should be in dBi."""
        n_phi = 10
        n_theta = 10
        n_freqs = 1

        # Create simple pattern with complex dtype
        e_theta = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)
        e_phi = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)
        # Add some non-zero component
        e_theta[5, 5, 0] = 1.0 + 0.0j

        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0.1, np.pi - 0.1, n_theta)
        freqs = (1e14,)

        directivity_db = far_field_directivity(e_theta, e_phi, phi_vals, theta_vals, freqs)

        assert directivity_db.shape == (n_phi, n_theta, n_freqs)
        # Peak should be positive dBi (directivity > 1)
        assert np.max(directivity_db) > 0


class TestProjectFarFieldToCartesian:
    """Test spherical to Cartesian far-field projection."""

    def test_projection_output_shapes(self):
        """Output should have correct shapes."""
        n_phi = 20
        n_theta = 10
        n_freqs = 2

        e_theta = np.random.randn(n_phi, n_theta, n_freqs) + 1j * np.random.randn(n_phi, n_theta, n_freqs)
        e_phi = np.random.randn(n_phi, n_theta, n_freqs) + 1j * np.random.randn(n_phi, n_theta, n_freqs)
        phi_vals = np.linspace(0, 2 * np.pi, n_phi)
        theta_vals = np.linspace(0.1, np.pi - 0.1, n_theta)
        freqs = (1e14, 2e14)

        ex, ey, ez = project_far_field_to_cartesian(
            e_theta, e_phi, phi_vals, theta_vals, freqs,
            x_range=(-1, 1, 5), y_range=(-1, 1, 7),
        )

        assert ex.shape == (5, 7, n_freqs)
        assert ey.shape == (5, 7, n_freqs)
        assert ez.shape == (5, 7, n_freqs)


class TestPhysicalConstants:
    """Test physical constant values."""

    def test_speed_of_light(self):
        """C0 should be approximately 3e8 m/s."""
        assert C0 == pytest.approx(299792458.0)

    def test_impedance_of_free_space(self):
        """Z0 should be approximately 377 ohms."""
        assert Z0 == pytest.approx(377.0, rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])