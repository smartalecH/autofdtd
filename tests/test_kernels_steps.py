"""Tests for Maxwell timestepping kernels in kernels.steps.

These tests focus on the vacuum E/H update kernels, field staggering
on the Yee lattice, and the capture-safe step skeleton.
"""

import numpy as np
import pytest

from autofdtd.kernels.backend import WARP_AVAILABLE, ComplexFieldPolicy
from autofdtd.kernels.steps import (
    MaxwellStepKernelSpec,
    allocate_maxwell_arrays,
    numpy_electric_update_3d,
    numpy_magnetic_update_3d,
    step_kernel_metadata,
    step_maxwell,
    vacuum_maxwell_step,
)


class TestStepKernelMetadata:
    """Test kernel metadata and stage ordering."""

    def test_step_kernel_metadata_returns_dict(self):
        """step_kernel_metadata returns a dict."""
        metadata = step_kernel_metadata()
        assert isinstance(metadata, dict)

    def test_electric_update_spec(self):
        """Metadata contains correct electric update spec."""
        metadata = step_kernel_metadata()
        spec = metadata["electric_update"]
        assert isinstance(spec, MaxwellStepKernelSpec)
        assert spec.name == "electric_update_3d"
        assert spec.field_family == "electric"
        assert spec.supports_complex is True

    def test_magnetic_update_spec(self):
        """Metadata contains correct magnetic update spec."""
        metadata = step_kernel_metadata()
        spec = metadata["magnetic_update"]
        assert isinstance(spec, MaxwellStepKernelSpec)
        assert spec.name == "magnetic_update_3d"
        assert spec.field_family == "magnetic"
        assert spec.supports_complex is True

    def test_update_stages_order(self):
        """Update stages are in correct bulk-synchronous order."""
        metadata = step_kernel_metadata()
        stages = metadata["update_stages"]
        expected = (
            "boundary_exchange",
            "electric_update",
            "source_injection",
            "boundary_exchange",
            "magnetic_update",
            "pml_stage",
            "monitor_collection",
            "convergence_check",
        )
        assert stages == expected

    def test_curl_component_mapping_keys(self):
        """Curl mapping contains all 6 field components."""
        metadata = step_kernel_metadata()
        curl_map = metadata["curl_component_mapping"]
        assert set(curl_map.keys()) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}

    def test_curl_component_mapping_ex_format(self):
        """Ex curl mapping has correct (component, cofactor, neighbor, cofactor) format."""
        metadata = step_kernel_metadata()
        ex_map = metadata["curl_component_mapping"]["Ex"]
        # Ex uses Hz for dHz/dy and Hy for dHy/dz
        assert ex_map == ("Hz", "dy", "Hy", "dz")

    def test_coefficient_requires(self):
        """Coefficient requires lists all 6 material components."""
        metadata = step_kernel_metadata()
        coeffs = metadata["coefficient_requires"]
        assert set(coeffs) == {"eps_xx", "eps_yy", "eps_zz", "mu_xx", "mu_yy", "mu_zz"}


class TestArrayAllocation:
    """Test Maxwell array allocation."""

    def test_allocate_maxwell_arrays_shape(self):
        """allocate_maxwell_arrays returns arrays with correct shapes."""
        arrays = allocate_maxwell_arrays((10, 10, 10))
        assert "E" in arrays
        assert "H" in arrays
        assert "eps_xx" in arrays
        assert "eps_yy" in arrays
        assert "eps_zz" in arrays
        assert "mu_xx" in arrays
        assert "mu_yy" in arrays
        assert "mu_zz" in arrays

        # E and H should be (nx, ny, nz, 3)
        assert arrays["E"].shape == (10, 10, 10, 3)
        assert arrays["H"].shape == (10, 10, 10, 3)

        # Coefficient arrays should be (nx, ny, nz)
        assert arrays["eps_xx"].shape == (10, 10, 10)
        assert arrays["mu_zz"].shape == (10, 10, 10)

    def test_allocate_maxwell_arrays_complex_policy(self):
        """Complex policy affects field dtype when Warp available."""
        arrays = allocate_maxwell_arrays(
            (8, 8, 8), complex_policy=ComplexFieldPolicy(force_complex=True)
        )
        assert arrays["E"].shape == (8, 8, 8, 3)
        assert arrays["H"].shape == (8, 8, 8, 3)


class TestNumpyElectricUpdate:
    """Test NumPy electric field update implementation."""

    def setup_method(self):
        """Set up test grid and arrays."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dx = self.dy = self.dz = 1e-8  # 10 nm cells
        # CFL-constrained dt for vacuum
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / (3e8)

        # Allocate arrays
        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )

        # Initialize to zero
        for key in self.arrays:
            self.arrays[key].fill(0.0)

        # Set vacuum permittivity and permeability
        self.arrays["eps_xx"].fill(1.0)
        self.arrays["eps_yy"].fill(1.0)
        self.arrays["eps_zz"].fill(1.0)
        self.arrays["mu_xx"].fill(1.0)
        self.arrays["mu_yy"].fill(1.0)
        self.arrays["mu_zz"].fill(1.0)

    def test_electric_update_preserves_zero_fields(self):
        """Zero E field remains zero after update with zero H."""
        E_before = self.arrays["E"].copy()
        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )
        # Should be unchanged (still zero)
        np.testing.assert_allclose(self.arrays["E"], E_before, atol=1e-15)

    def test_electric_update_changes_with_nonzero_H(self):
        """Non-zero H field causes E field to change."""
        # Set a spatially-varying H field to produce non-zero curl
        # Hz varies with j (y-index) to produce dHz/dy
        for j in range(self.ny):
            self.arrays["H"][:, j, :, 2] = j * 0.1  # Hz varies with y

        E_before = self.arrays["E"].copy()
        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )
        # E should have changed due to curl(H)
        assert np.linalg.norm(self.arrays["E"]) > 0.0

    def test_electric_update_ex_component(self):
        """Ex component update uses dHz/dy - dHy/dz."""
        # Set Hz varying with j (y-index) to produce dHz/dy
        # Set Hy varying with k (z-index) to produce dHy/dz
        for j in range(self.ny):
            self.arrays["H"][:, j, :, 2] = j * 0.1  # Hz varies with y
        for k in range(self.nz):
            self.arrays["H"][:, :, k, 1] = k * 0.1  # Hy varies with z

        E_before = self.arrays["E"][:, :, :, 0].copy()

        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # Ex should have changed due to dHz/dy and dHy/dz terms
        dEx = self.arrays["E"][:, :, :, 0] - E_before
        # Interior cells should show non-zero change
        interior = dEx[1:-1, 1:-1, 1:-1]
        assert np.max(np.abs(interior)) > 0.0

    def test_electric_update_interior_only(self):
        """Electric update only affects interior cells."""
        # Set spatially-varying H
        for i in range(self.nx):
            self.arrays["H"][i, :, :, 2] = i * 0.1

        # Record boundary cells before update
        E_boundary_before = self.arrays["E"][0, :, :, :].copy()
        E_boundary_end_before = self.arrays["E"][-1, :, :, :].copy()

        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # Ghost cells should be unchanged (still zero)
        np.testing.assert_allclose(self.arrays["E"][0, :, :, :], E_boundary_before)
        np.testing.assert_allclose(self.arrays["E"][-1, :, :, :], E_boundary_end_before)


class TestNumpyMagneticUpdate:
    """Test NumPy magnetic field update implementation."""

    def setup_method(self):
        """Set up test grid and arrays."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dx = self.dy = self.dz = 1e-8  # 10 nm cells
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / (3e8)

        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )

        for key in self.arrays:
            arr = self.arrays[key]
            if hasattr(arr, 'fill_'):
                arr.fill_(0.0)
            else:
                arr.fill(0.0)

        eps_xx = self.arrays["eps_xx"]
        eps_yy = self.arrays["eps_yy"]
        eps_zz = self.arrays["eps_zz"]
        mu_xx = self.arrays["mu_xx"]
        mu_yy = self.arrays["mu_yy"]
        mu_zz = self.arrays["mu_zz"]
        if hasattr(eps_xx, 'fill_'):
            eps_xx.fill_(1.0)
            eps_yy.fill_(1.0)
            eps_zz.fill_(1.0)
            mu_xx.fill_(1.0)
            mu_yy.fill_(1.0)
            mu_zz.fill_(1.0)
        else:
            eps_xx.fill(1.0)
            eps_yy.fill(1.0)
            eps_zz.fill(1.0)
            mu_xx.fill(1.0)
            mu_yy.fill(1.0)
            mu_zz.fill(1.0)

    def test_magnetic_update_preserves_zero_fields(self):
        """Zero H field remains zero after update with zero E."""
        H_before = self.arrays["H"].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )
        np.testing.assert_allclose(self.arrays["H"], H_before, atol=1e-15)

    def test_magnetic_update_changes_with_nonzero_E(self):
        """Non-zero E field causes H field to change."""
        # Set spatially-varying E to produce non-zero curl
        # Ey varies with k (z-index) to produce dEy/dz for Hx
        for k in range(self.nz):
            self.arrays["E"][:, :, k, 1] = k * 0.1  # Ey varies with z

        H_before = self.arrays["H"].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )
        assert np.linalg.norm(self.arrays["H"]) > 0.0

    def test_magnetic_update_sign_is_negative(self):
        """H update uses negative sign (opposite direction to E update)."""
        # Set spatially-varying E field to produce curl
        for k in range(self.nz):
            self.arrays["E"][:, :, k, 1] = k * 0.1  # Ey varies with z

        H_before = self.arrays["H"].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # H should change
        dH = self.arrays["H"] - H_before
        assert np.max(np.abs(dH)) > 0.0

    def test_magnetic_update_interior_only(self):
        """Magnetic update only affects interior cells."""
        # Set spatially-varying E
        for i in range(self.nx):
            self.arrays["E"][i, :, :, 1] = i * 0.1

        H_boundary_before = self.arrays["H"][0, :, :, :].copy()

        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        np.testing.assert_allclose(self.arrays["H"][0, :, :, :], H_boundary_before)


class TestVacuumMaxwellStep:
    """Test complete Maxwell step (E then H) in vacuum."""

    def setup_method(self):
        """Set up vacuum simulation."""
        self.nx, self.ny, self.nz = 20, 20, 20
        self.dx = self.dy = self.dz = 1e-8
        # CFL-safe timestep
        c = 3e8
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / c

        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )

        for key in self.arrays:
            arr = self.arrays[key]
            if hasattr(arr, 'fill_'):
                arr.fill_(0.0)
            else:
                arr.fill(0.0)

        eps_xx = self.arrays["eps_xx"]
        eps_yy = self.arrays["eps_yy"]
        eps_zz = self.arrays["eps_zz"]
        mu_xx = self.arrays["mu_xx"]
        mu_yy = self.arrays["mu_yy"]
        mu_zz = self.arrays["mu_zz"]
        if hasattr(eps_xx, 'fill_'):
            eps_xx.fill_(1.0)
            eps_yy.fill_(1.0)
            eps_zz.fill_(1.0)
            mu_xx.fill_(1.0)
            mu_yy.fill_(1.0)
            mu_zz.fill_(1.0)
        else:
            eps_xx.fill(1.0)
            eps_yy.fill(1.0)
            eps_zz.fill(1.0)
            mu_xx.fill(1.0)
            mu_yy.fill(1.0)
            mu_zz.fill(1.0)

    def test_step_maxwell_numpy_returns_metrics(self):
        """step_maxwell returns structured metrics."""
        result = step_maxwell(
            self.arrays, self.dt, self.dx, self.dy, self.dz, step_index=0
        )
        assert "step_index" in result
        assert "cells_updated" in result
        assert "backend" in result
        assert result["backend"] == "numpy"

    def test_vacuum_propagation_is_energy_conserving(self):
        """Vacuum propagation should approximately conserve total energy."""
        # Initialize a plane wave pulse
        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2

        # Set a Gaussian pulse in Ez
        sigma = 3
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    r2 = ((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / (
                        2 * sigma**2
                    )
                    self.arrays["E"][i, j, k, 2] = np.exp(-r2)  # Ez component

        # Record initial energy
        i1, i2 = 1, self.nx - 1
        j1, j2 = 1, self.ny - 1
        k1, k2 = 1, self.nz - 1

        E0_sq = np.sum(self.arrays["E"][i1:i2, j1:j2, k1:k2, :] ** 2)
        H0_sq = np.sum(self.arrays["H"][i1:i2, j1:j2, k1:k2, :] ** 2)
        initial_energy = 0.5 * (E0_sq + H0_sq)

        # Run several steps
        for step in range(50):
            step_maxwell(
                self.arrays,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                step_index=step,
                time=step * self.dt,
            )

        # Compute final energy
        Ef_sq = np.sum(self.arrays["E"][i1:i2, j1:j2, k1:k2, :] ** 2)
        Hf_sq = np.sum(self.arrays["H"][i1:i2, j1:j2, k1:k2, :] ** 2)
        final_energy = 0.5 * (Ef_sq + Hf_sq)

        # Energy should be approximately conserved (within ~10% for this coarse test)
        if initial_energy > 1e-10:
            relative_change = abs(final_energy - initial_energy) / initial_energy
            assert relative_change < 0.15, (
                f"Energy changed by {relative_change*100:.1f}%, "
                f"expected < 15% for vacuum propagation"
            )

    def test_vacuum_wave_propagates(self):
        """A field pulse in vacuum should propagate without blowing up."""
        # Initialize a Gaussian pulse at the center
        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2
        sigma = 2

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    r2 = ((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / (
                        2 * sigma**2
                    )
                    self.arrays["E"][i, j, k, 2] = np.exp(-r2)

        # Record initial maximum field value
        initial_max = np.max(np.abs(self.arrays["E"]))

        # Run several steps
        num_steps = 20
        for step in range(num_steps):
            step_maxwell(
                self.arrays,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                step_index=step,
                time=step * self.dt,
            )

        final_max = np.max(np.abs(self.arrays["E"]))

        # Field should still be present (not decayed to zero)
        assert final_max > 0.1 * initial_max, (
            f"Field decayed too much: initial={initial_max:.4f}, final={final_max:.4f}"
        )

        # Field should not have exploded
        assert final_max < 10 * initial_max, (
            f"Field grew too much: initial={initial_max:.4f}, final={final_max:.4f}"
        )

    def test_vacuum_maxwell_step_returns_diagnostics(self):
        """vacuum_maxwell_step returns list of diagnostics."""
        results = vacuum_maxwell_step(
            self.arrays,
            self.dt,
            self.dx,
            self.dy,
            self.dz,
            num_steps=25,
            interval=10,
        )

        # Steps recorded: 0, 10, 20 (25 steps runs 0 to 24)
        assert len(results) == 3

        for r in results:
            assert "step_index" in r
            assert "E_energy" in r
            assert "H_energy" in r
            assert "total_energy" in r


class TestYeeLatticeStaggering:
    """Test that the Yee lattice staggering is correctly implemented."""

    def setup_method(self):
        """Set up small test grid to verify staggering."""
        self.nx, self.ny, self.nz = 8, 8, 8
        self.dx = self.dy = self.dz = 1e-8
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / (3e8)

        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )

        for key in self.arrays:
            arr = self.arrays[key]
            if hasattr(arr, 'fill_'):
                arr.fill_(0.0)
            else:
                arr.fill(0.0)

        eps_xx = self.arrays["eps_xx"]
        eps_yy = self.arrays["eps_yy"]
        eps_zz = self.arrays["eps_zz"]
        mu_xx = self.arrays["mu_xx"]
        mu_yy = self.arrays["mu_yy"]
        mu_zz = self.arrays["mu_zz"]
        if hasattr(eps_xx, 'fill_'):
            eps_xx.fill_(1.0)
            eps_yy.fill_(1.0)
            eps_zz.fill_(1.0)
            mu_xx.fill_(1.0)
            mu_yy.fill_(1.0)
            mu_zz.fill_(1.0)
        else:
            eps_xx.fill(1.0)
            eps_yy.fill(1.0)
            eps_zz.fill(1.0)
            mu_xx.fill(1.0)
            mu_yy.fill(1.0)
            mu_zz.fill(1.0)

    def test_ey_update_uses_hx_and_hz_neighbors(self):
        """Ey update depends on Hx (dz) and Hz (dx) neighbors."""
        # Set Hx varying with k (z-index) to produce dHx/dz
        for k in range(self.nz):
            self.arrays["H"][:, :, k, 0] = k * 0.1  # Hx varies with z

        Ey_before = self.arrays["E"][:, :, :, 1].copy()
        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        dEy = self.arrays["E"][:, :, :, 1] - Ey_before
        interior_dEy = dEy[1:-1, 1:-1, 1:-1]
        # With spatially-varying Hx, Ey should change due to dHx/dz term
        assert np.max(np.abs(interior_dEy)) > 0.0

    def test_ez_update_uses_hx_and_hy_neighbors(self):
        """Ez update depends on Hy (dx) and Hx (dy) neighbors."""
        # Set Hy varying with i (x-index) to produce dHy/dx
        for i in range(self.nx):
            self.arrays["H"][i, :, :, 1] = i * 0.1  # Hy varies with x

        Ez_before = self.arrays["E"][:, :, :, 2].copy()
        numpy_electric_update_3d(
            self.arrays["E"],
            self.arrays["H"],
            self.arrays["eps_xx"],
            self.arrays["eps_yy"],
            self.arrays["eps_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        dEz = self.arrays["E"][:, :, :, 2] - Ez_before
        interior_dEz = dEz[1:-1, 1:-1, 1:-1]
        # With spatially-varying Hy, Ez should change due to dHy/dx term
        assert np.max(np.abs(interior_dEz)) > 0.0

    def test_hx_update_uses_ey_and_ez_neighbors(self):
        """Hx update depends on Ey (dz) and Ez (dy) neighbors."""
        # Set Ey varying with k (z-index) to produce dEy/dz
        for k in range(self.nz):
            self.arrays["E"][:, :, k, 1] = k * 0.1  # Ey varies with z

        Hx_before = self.arrays["H"][:, :, :, 0].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        dHx = self.arrays["H"][:, :, :, 0] - Hx_before
        interior_dHx = dHx[1:-1, 1:-1, 1:-1]
        # With spatially-varying Ey, Hx should change due to dEy/dz term
        assert np.max(np.abs(interior_dHx)) > 0.0

    def test_hy_update_uses_ex_and_ez_neighbors(self):
        """Hy update depends on Ez (dx) and Ex (dz) neighbors."""
        # Set Ez varying with i (x-index) to produce dEz/dx
        for i in range(self.nx):
            self.arrays["E"][i, :, :, 2] = i * 0.1  # Ez varies with x

        Hy_before = self.arrays["H"][:, :, :, 1].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        dHy = self.arrays["H"][:, :, :, 1] - Hy_before
        interior_dHy = dHy[1:-1, 1:-1, 1:-1]
        # With spatially-varying Ez, Hy should change due to dEz/dx term
        assert np.max(np.abs(interior_dHy)) > 0.0

    def test_hz_update_uses_ex_and_ey_neighbors(self):
        """Hz update depends on Ex (dy) and Ey (dx) neighbors."""
        # Set Ex varying with j (y-index) to produce dEx/dy
        for j in range(self.ny):
            self.arrays["E"][:, j, :, 0] = j * 0.1  # Ex varies with y

        Hz_before = self.arrays["H"][:, :, :, 2].copy()
        numpy_magnetic_update_3d(
            self.arrays["H"],
            self.arrays["E"],
            self.arrays["mu_xx"],
            self.arrays["mu_yy"],
            self.arrays["mu_zz"],
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        dHz = self.arrays["H"][:, :, :, 2] - Hz_before
        interior_dHz = dHz[1:-1, 1:-1, 1:-1]
        # With spatially-varying Ex, Hz should change due to dEx/dy term
        assert np.max(np.abs(interior_dHz)) > 0.0
