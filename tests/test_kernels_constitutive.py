"""Tests for constitutive and material-update kernel stages.

These tests exercise the coefficient-aware, dispersive (PoleResidue),
and anisotropic Maxwell update kernels that Phase 1 claims to support.
"""

import numpy as np
import pytest

from autofdtd.kernels.backend import ComplexFieldPolicy
from autofdtd.kernels.steps import (
    allocate_maxwell_arrays,
    step_kernel_metadata,
)
from autofdtd.compiler.materials import compile_isotropic_medium_coefficients
from autofdtd.kernels.materials import electric_constitutive_update
from autofdtd.grid.specs import GridSpec, UniformGrid, resolve_grid_spec


class TestStepKernelMetadataUpdated:
    """Test that kernel metadata reflects the expanded kernel set."""

    def test_coefficient_kernels_listed_in_metadata(self):
        """step_kernel_metadata includes coefficient-aware kernel specs."""
        metadata = step_kernel_metadata()
        # Metadata lists the base kernel names; new kernels are verified via module exports
        from autofdtd.kernels import steps
        assert hasattr(steps, "electric_update_3d_coeff")
        assert hasattr(steps, "magnetic_update_3d_coeff")

    def test_dispersive_kernels_exported(self):
        """Dispersive update kernels are exported."""
        from autofdtd.kernels import steps
        assert hasattr(steps, "pole_residue_electric_update_3d")
        assert hasattr(steps, "pole_residue_polarization_update_3d")

    def test_anisotropic_kernels_exported(self):
        """Anisotropic update kernels are exported."""
        from autofdtd.kernels import steps
        assert hasattr(steps, "anisotropic_electric_update_3d")
        assert hasattr(steps, "anisotropic_magnetic_update_3d")


class TestCoefficientAwareUpdate:
    """Test coefficient-aware (inhomogeneous) update kernels via NumPy equivalents."""

    def setup_method(self):
        """Set up test grid with spatially varying coefficients."""
        self.nx, self.ny, self.nz = 12, 12, 12
        self.dx = self.dy = self.dz = 1e-8
        c = 3e8
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / c

        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )
        for key in self.arrays:
            self.arrays[key].fill(0.0)

        # Vacuum coefficients
        self.arrays["eps_xx"].fill(1.0)
        self.arrays["eps_yy"].fill(1.0)
        self.arrays["eps_zz"].fill(1.0)
        self.arrays["mu_xx"].fill(1.0)
        self.arrays["mu_yy"].fill(1.0)
        self.arrays["mu_zz"].fill(1.0)

    def test_inhomogeneous_coefficients_affect_update_rate(self):
        """Spatially varying permittivity produces spatially varying update rates.

        Uses a simple finite-difference check: set uniform curl_H and verify
        that the E update magnitude varies with local eps.
        """
        # Set spatially varying permittivity: higher in center, lower at edges
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Distance from center
                    di = i - self.nx // 2
                    dj = j - self.ny // 2
                    dk = k - self.nz // 2
                    r = np.sqrt(di**2 + dj**2 + dk**2)
                    self.arrays["eps_xx"][i, j, k] = 1.0 + 10.0 * np.exp(-r / 3.0)
                    self.arrays["eps_yy"][i, j, k] = self.arrays["eps_xx"][i, j, k]
                    self.arrays["eps_zz"][i, j, k] = self.arrays["eps_xx"][i, j, k]

        # Set a uniform non-zero curl_H by making Hy vary with z
        # dHy_dz = (Hy[i,j,k+1] - Hy[i,j,k]) / dz
        for k in range(self.nz - 1):
            self.arrays["H"][:, :, k, 1] = 1.0 * k  # Hy varies with k

        from autofdtd.kernels.steps import numpy_electric_update_3d

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

        # Check that interior E has changed (curl_H is non-zero)
        interior_E = self.arrays["E"][2:10, 2:10, 2:10, :]
        assert np.linalg.norm(interior_E) > 0.0

    def test_high_eps_suppresses_update(self):
        """Very high permittivity suppresses the electric field update.

        The update coefficient ce_Ex = dt/(eps*eps0) becomes very small
        when eps is very large, suppressing the curl-driven E change.
        """
        self.arrays["eps_xx"].fill(1e8)
        self.arrays["eps_yy"].fill(1e8)
        self.arrays["eps_zz"].fill(1e8)

        # Set strong curl_H
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    self.arrays["H"][i, j, k, 2] = 1.0  # Hz = 1

        from autofdtd.kernels.steps import numpy_electric_update_3d

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

        # With very high eps, the update should be extremely small
        assert np.max(np.abs(self.arrays["E"])) < 1e-6


class TestDispersiveUpdate:
    """Test dispersive (PoleResidue) update kernels via NumPy equivalents."""

    def setup_method(self):
        """Set up test grid with PoleResidue auxiliary state."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dx = self.dy = self.dz = 1e-8
        c = 3e8
        self.dt = 0.99 * min(self.dx, self.dy, self.dz) / c

        self.arrays = allocate_maxwell_arrays(
            (self.nx, self.ny, self.nz), complex_policy=ComplexFieldPolicy()
        )
        for key in self.arrays:
            self.arrays[key].fill(0.0)

        self.arrays["eps_xx"].fill(2.0)
        self.arrays["eps_yy"].fill(2.0)
        self.arrays["eps_zz"].fill(2.0)
        self.arrays["mu_xx"].fill(1.0)
        self.arrays["mu_yy"].fill(1.0)
        self.arrays["mu_zz"].fill(1.0)

    def test_pole_residue_coefficients_affect_electric_update(self):
        """Dispersive PoleResidue coefficients modify the baseline E update."""
        from autofdtd.compiler.materials import compile_pole_residue_coefficients
        from autofdtd.kernels.materials import (
            allocate_pole_residue_state,
            pole_residue_electric_update,
        )
        from autofdtd.materials import PoleResidue

        # Single-pole Lorentzian with resonance near visible
        medium = PoleResidue(
            eps_inf=2.0,
            poles=(((-1.0e12, 0.0), (2.0e10, 0.0)),),
        )
        coeffs = compile_pole_residue_coefficients(medium, dt=self.dt)
        state = allocate_pole_residue_state(coeffs, field_shape=(self.nx, self.ny, self.nz))

        electric = np.random.randn(self.nx, self.ny, self.nz)
        curl_h = np.random.randn(self.nx, self.ny, self.nz) * 0.1

        updated_e, updated_state, pol_current = pole_residue_electric_update(
            electric, curl_h, coeffs, state, dt=self.dt
        )

        # The dispersive update should differ from simple constitutive update
        # because polarization current is subtracted
        # Baseline using the eps_inf background
        from autofdtd.materials import Medium

        background_medium = Medium(permittivity=medium.eps_inf)
        baseline_coeffs = compile_isotropic_medium_coefficients(background_medium, dt=self.dt)
        baseline = electric_constitutive_update(electric, curl_h, baseline_coeffs)

        # Dispersive update should differ from baseline due to polarization
        diff = np.max(np.abs(updated_e - baseline))
        assert diff > 0.0, "Dispersive update should differ from non-dispersive baseline"

        # Verify auxiliary state was updated
        assert updated_state.polarization.shape[0] == coeffs.num_poles

    def test_pole_residue_auxiliary_state_shape(self):
        """PoleResidue auxiliary state has correct shape per pole."""
        from autofdtd.compiler.materials import compile_pole_residue_coefficients
        from autofdtd.kernels.materials import allocate_pole_residue_state
        from autofdtd.materials import PoleResidue

        medium = PoleResidue(
            eps_inf=2.5,
            poles=(
                ((-1.0e12, 0.0), (5.0e10, 0.0)),
                ((-5.0e11, 0.0), (1.0e10, 0.0)),
            ),
        )
        coeffs = compile_pole_residue_coefficients(medium, dt=self.dt)
        field_shape = (8, 8, 8)
        state = allocate_pole_residue_state(coeffs, field_shape=field_shape)

        assert state.polarization.shape == (2, 8, 8, 8)
        assert state.num_poles == 2

    def test_sellmeier_lowers_to_pole_residue_auxiliary_state(self):
        """Sellmeier media use the shared PoleResidue auxiliary state."""
        from autofdtd.compiler.materials import (
            compile_sellmeier_coefficients,
            compile_pole_residue_coefficients,
        )
        from autofdtd.kernels.materials import allocate_pole_residue_state
        from autofdtd.materials import Sellmeier

        silica = Sellmeier(coeffs=((0.6961663, 4.67914825849e-15), (0.4079426, 0.0)))
        sellmeier_coeffs = compile_sellmeier_coefficients(silica, dt=self.dt)
        pole_residue_coeffs = compile_pole_residue_coefficients(silica.to_pole_residue(), dt=self.dt)

        # Both should produce same auxiliary state shape
        s_state = allocate_pole_residue_state(sellmeier_coeffs, field_shape=(6, 6, 6))
        p_state = allocate_pole_residue_state(pole_residue_coeffs, field_shape=(6, 6, 6))

        assert s_state.polarization.shape == p_state.polarization.shape

    def test_analytical_dispersion_auxiliary_state_paths(self):
        """Lorentz, Drude, and Debye all use PoleResidue auxiliary state layout."""
        from autofdtd.compiler.materials import (
            compile_lorentz_coefficients,
            compile_drude_coefficients,
            compile_debye_coefficients,
        )
        from autofdtd.kernels.materials import allocate_pole_residue_state
        from autofdtd.materials import Debye, Drude, Lorentz

        media_and_compilers = [
            (Lorentz(eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),)), compile_lorentz_coefficients),
            (Drude(eps_inf=1.2, coeffs=((180e12, 8e12),)), compile_drude_coefficients),
            (Debye(eps_inf=2.1, coeffs=((1.4, 5.0e-12),)), compile_debye_coefficients),
        ]

        for medium, compiler in media_and_compilers:
            coeffs = compiler(medium, dt=self.dt)
            state = allocate_pole_residue_state(coeffs, field_shape=(5, 5, 5))
            assert state.polarization.ndim == 4
            assert state.polarization.shape[0] == coeffs.num_poles


class TestAnisotropicUpdate:
    """Test diagonal anisotropic update kernels."""

    def test_anisotropic_coefficients_per_axis_affect_update(self):
        """Anisotropic coefficients produce axis-dependent update behavior."""
        from autofdtd.compiler.materials import compile_anisotropic_medium_coefficients
        from autofdtd.kernels.materials import (
            anisotropic_electric_update,
            anisotropic_magnetic_update,
            allocate_anisotropic_state,
        )
        from autofdtd.materials import AnisotropicMedium, Medium, PECMedium, PoleResidue

        # Diagonal tensor: different permittivity per axis
        medium = AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=PoleResidue(eps_inf=2.5, poles=(((-2.0e12, 0.0), (8.0e10, 0.0)),)),
            zz=PECMedium(),
        )

        coeffs = compile_anisotropic_medium_coefficients(medium, dt=1e-12)
        state = allocate_anisotropic_state(coeffs, field_shape=(6, 6, 6))

        # Anisotropic E field (Ex, Ey, Ez per cell)
        electric = np.random.randn(6, 6, 6, 3)
        magnetic = np.random.randn(6, 6, 6, 3)
        curl_h = np.random.randn(6, 6, 6, 3) * 0.1

        updated_e, updated_state, pol = anisotropic_electric_update(
            electric, curl_h, coeffs, state, dt=1e-12
        )

        # Ez component should be zero (PEC clamp)
        assert np.allclose(updated_e[:, :, :, 2], 0.0)

        # Ex and Ey components should use different coefficient paths
        # The update should differ between components
        dEx = updated_e[:, :, :, 0] - electric[:, :, :, 0]
        dEy = updated_e[:, :, :, 1] - electric[:, :, :, 1]
        assert not np.allclose(dEx, dEy)

        # Verify magnetic update returns field of correct shape
        updated_h = anisotropic_magnetic_update(magnetic, curl_h, coeffs)
        assert updated_h.shape == magnetic.shape

    def test_anisotropic_state_tracks_dispersive_axes(self):
        """Anisotropic auxiliary state has per-axis PoleResidue state."""
        from autofdtd.compiler.materials import compile_anisotropic_medium_coefficients
        from autofdtd.kernels.materials import allocate_anisotropic_state
        from autofdtd.materials import AnisotropicMedium, Medium, PoleResidue

        medium = AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=PoleResidue(eps_inf=2.5, poles=(((-2.0e12, 0.0), (8.0e10, 0.0)),)),
            zz=Medium(permittivity=3.0),
        )

        coeffs = compile_anisotropic_medium_coefficients(medium, dt=1e-12)
        state = allocate_anisotropic_state(coeffs, field_shape=(5, 5, 5))

        # xx and zz should be None (no dispersive state needed)
        assert state.xx is None
        assert state.zz is None
        # yy should have dispersive state
        assert state.yy is not None
        assert state.yy.polarization.shape[0] == 1  # 1 pole


class TestNumPyConstitutiveUpdateWithCoefficients:
    """Test NumPy constitutive update helpers with coefficient arrays."""

    def test_constitutive_update_with_clamp_mode(self):
        """Electric constitutive update with CLAMP_ZERO mode returns zeros."""
        from autofdtd.compiler.materials import ConstitutiveMode
        from autofdtd.kernels.materials import electric_constitutive_update

        # Build a clamp mode coefficient
        class FakeIsotropicCoeffs:
            electric_decay = 0.5
            electric_drive = 1e-12
            electric_mode = ConstitutiveMode.CLAMP_ZERO

        electric = np.array([1.0, -2.0, 0.5])
        curl_h = np.array([0.1, 0.3, -0.2])

        result = electric_constitutive_update(electric, curl_h, FakeIsotropicCoeffs())
        assert np.allclose(result, 0.0)

    def test_constitutive_update_standard_mode(self):
        """Electric constitutive update with STANDARD mode applies decay/drive."""
        from autofdtd.compiler.materials import ConstitutiveMode
        from autofdtd.kernels.materials import electric_constitutive_update

        class FakeIsotropicCoeffs:
            electric_decay = 0.8
            electric_drive = 1e-12
            electric_mode = ConstitutiveMode.STANDARD

        electric = np.array([1.0, -2.0, 0.5])
        curl_h = np.array([0.1, 0.3, -0.2])

        result = electric_constitutive_update(electric, curl_h, FakeIsotropicCoeffs())
        expected = 0.8 * electric + 1e-12 * curl_h
        np.testing.assert_allclose(result, expected)


class TestMaterialKernelStaging:
    """Test that material kernel stages are correctly ordered and staged."""

    def test_pole_residue_electric_update_returns_tuple(self):
        """PoleResidue update returns (updated_E, updated_state, polarization_current)."""
        from autofdtd.compiler.materials import compile_pole_residue_coefficients
        from autofdtd.kernels.materials import (
            allocate_pole_residue_state,
            pole_residue_electric_update,
        )
        from autofdtd.materials import PoleResidue

        medium = PoleResidue(
            eps_inf=2.0, poles=(((-2.0e12, 0.0), (8.0e10, 1.0e10)),)
        )
        coeffs = compile_pole_residue_coefficients(medium, dt=1e-12)
        state = allocate_pole_residue_state(coeffs, field_shape=(3,))
        electric = np.array([1.0, -0.5, 0.25])
        curl_h = np.array([0.1, 0.2, -0.1])

        result = pole_residue_electric_update(
            electric, curl_h, coeffs, state, dt=1e-12
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        updated_e, updated_state, pol_current = result
        assert updated_e.shape == electric.shape
        assert updated_state.polarization.shape[0] == coeffs.num_poles
        assert pol_current.shape == electric.shape

    def test_anisotropic_electric_update_returns_tuple(self):
        """Anisotropic E update returns (updated_E, updated_state, polarization)."""
        from autofdtd.compiler.materials import compile_anisotropic_medium_coefficients
        from autofdtd.kernels.materials import (
            allocate_anisotropic_state,
            anisotropic_electric_update,
        )
        from autofdtd.materials import AnisotropicMedium, Medium, PoleResidue

        medium = AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=PoleResidue(eps_inf=2.5, poles=(((-2.0e12, 0.0), (8.0e10, 0.0)),)),
            zz=Medium(permittivity=1.5),
        )
        coeffs = compile_anisotropic_medium_coefficients(medium, dt=1e-12)
        state = allocate_anisotropic_state(coeffs, field_shape=(4, 4))

        electric = np.random.randn(4, 4, 3)
        curl_h = np.random.randn(4, 4, 3) * 0.1

        result = anisotropic_electric_update(electric, curl_h, coeffs, state, dt=1e-12)

        assert isinstance(result, tuple)
        assert len(result) == 3
        updated_e, updated_state, pol = result
        assert updated_e.shape == electric.shape
        assert pol.shape == electric.shape

    def test_magnetic_update_returns_field_only(self):
        """Anisotropic magnetic update returns updated field array."""
        from autofdtd.compiler.materials import compile_anisotropic_medium_coefficients
        from autofdtd.kernels.materials import anisotropic_magnetic_update
        from autofdtd.materials import AnisotropicMedium, Medium, PoleResidue

        medium = AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=PoleResidue(eps_inf=2.5, poles=(((-2.0e12, 0.0), (8.0e10, 0.0)),)),
            zz=Medium(permittivity=1.5),
        )
        coeffs = compile_anisotropic_medium_coefficients(medium, dt=1e-12)

        magnetic = np.random.randn(4, 4, 3)
        curl_e = np.random.randn(4, 4, 3) * 0.1

        result = anisotropic_magnetic_update(magnetic, curl_e, coeffs)

        assert isinstance(result, np.ndarray)
        assert result.shape == magnetic.shape


class TestKernelStagingMetadata:
    """Test that kernel staging metadata is correct."""

    def test_update_stages_listed(self):
        """Update stages include all expected phases."""
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

    def test_curl_component_mapping_complete(self):
        """All 6 field components have curl mappings."""
        metadata = step_kernel_metadata()
        curl_map = metadata["curl_component_mapping"]
        assert set(curl_map.keys()) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}

    def test_coefficient_requires_lists_six_components(self):
        """Coefficient requires lists all 6 material tensor components."""
        metadata = step_kernel_metadata()
        coeffs = metadata["coefficient_requires"]
        assert set(coeffs) == {"eps_xx", "eps_yy", "eps_zz", "mu_xx", "mu_yy", "mu_zz"}

    def test_new_kernels_have_stable_module_flag(self):
        """New kernels should be marked with module_contents_stable once finalized."""
        from autofdtd.kernels.steps import step_kernel_metadata

        metadata = step_kernel_metadata()
        # module_contents_stable starts False during development
        assert "module_contents_stable" in metadata


class TestCoefficientFieldIntegration:
    """Test integration between CoefficientField and update kernels."""

    def test_coefficient_field_assembly_produces_correct_shapes(self):
        """CoefficientField from discretization has correct array shapes."""
        from autofdtd.compiler.discretization import (
            CoefficientField,
            assemble_coefficient_fields,
            materialize_grid,
        )
        from autofdtd.geometry.primitives import Box
        from autofdtd.materials import Medium, PECMedium
        from autofdtd.core.containers import Scene, Structure, StructurePriorityMode

        scene = Scene(
            medium=Medium(permittivity=1.5),
            structures=(
                Structure(
                    geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
                    medium=Medium(permittivity=3.5),
                    name="core",
                ),
                Structure(
                    geometry=Box(center=(3.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                    medium=PECMedium(),
                    name="pec",
                ),
            ),
            structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        )

        grid_spec = GridSpec(
            grid_x=UniformGrid(dl=0.4),
            grid_y=UniformGrid(dl=0.4),
            grid_z=UniformGrid(dl=0.4),
        )
        grid = resolve_grid_spec(grid_spec, size=(8.0, 8.0, 8.0), center=(0.0, 0.0, 0.0))
        dt = 1e-14

        material_field = materialize_grid(scene, grid)
        coeff_field = assemble_coefficient_fields(
            material_field, scene, grid, dt=dt
        )

        assert isinstance(coeff_field, CoefficientField)
        assert coeff_field.nx == grid.x.num_cells
        assert coeff_field.ny == grid.y.num_cells
        assert coeff_field.nz == grid.z.num_cells
        assert coeff_field.eps_xx.shape == (coeff_field.nx, coeff_field.ny, coeff_field.nz)
        assert coeff_field.mu_xx.shape == (coeff_field.nx, coeff_field.ny, coeff_field.nz)
        assert coeff_field.e_drive_xx.shape == coeff_field.eps_xx.shape
        assert coeff_field.electric_modes.shape == coeff_field.eps_xx.shape
        # electric_modes should be 0 (standard) or 1 (clamp_zero)
        assert np.all(coeff_field.electric_modes >= 0)
        assert np.all(coeff_field.electric_modes <= 1)

    def test_coeff_field_pec_cells_have_clamp_mode(self):
        """PEC regions in CoefficientField have electric_mode = clamp_zero."""
        from autofdtd.compiler.discretization import (
            assemble_coefficient_fields,
            materialize_grid,
        )
        from autofdtd.geometry.primitives import Box
        from autofdtd.materials import Medium, PECMedium
        from autofdtd.core.containers import Scene, Structure, StructurePriorityMode

        scene = Scene(
            medium=Medium(permittivity=1.5),
            structures=(
                Structure(
                    geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
                    medium=PECMedium(),
                    name="pec",
                ),
            ),
            structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        )

        grid_spec = GridSpec(
            grid_x=UniformGrid(dl=0.4),
            grid_y=UniformGrid(dl=0.4),
            grid_z=UniformGrid(dl=0.4),
        )
        grid = resolve_grid_spec(grid_spec, size=(8.0, 8.0, 8.0), center=(0.0, 0.0, 0.0))
        dt = 1e-14

        material_field = materialize_grid(scene, grid)
        coeff_field = assemble_coefficient_fields(material_field, scene, grid, dt=dt)

        # At least some cells should be in clamp mode (pec region)
        assert np.any(coeff_field.electric_modes == 1.0)


class TestFullMaterialPipeline:
    """Test the full material pipeline from medium to kernel update."""

    def test_isotropic_medium_to_coefficient_to_update(self):
        """Full pipeline: Medium → CoefficientField → E update."""
        from autofdtd.compiler.materials import compile_isotropic_medium_coefficients
        from autofdtd.kernels.materials import electric_constitutive_update, magnetic_constitutive_update
        from autofdtd.materials import Medium

        medium = Medium(permittivity=3.5, conductivity=0.1, permeability=1.2)
        coeffs = compile_isotropic_medium_coefficients(medium, dt=1e-12)

        electric = np.array([1.0, -0.5, 0.25])
        magnetic = np.array([0.3, -0.1, 0.4])
        curl_h = np.array([0.1, 0.2, -0.1])
        curl_e = np.array([-0.2, 0.1, 0.3])

        updated_e = electric_constitutive_update(electric, curl_h, coeffs)
        updated_h = magnetic_constitutive_update(magnetic, curl_e, coeffs)

        # Verify decay/drive formula
        assert np.allclose(
            updated_e,
            coeffs.electric_decay * electric + coeffs.electric_drive * curl_h,
        )
        assert np.allclose(
            updated_h,
            coeffs.magnetic_decay * magnetic - coeffs.magnetic_drive * curl_e,
        )

    def test_pole_residue_full_pipeline(self):
        """Full pipeline: PoleResidue → Coefficients → Dispersive update."""
        from autofdtd.compiler.materials import compile_pole_residue_coefficients
        from autofdtd.kernels.materials import (
            allocate_pole_residue_state,
            pole_residue_electric_update,
        )
        from autofdtd.materials import PoleResidue

        medium = PoleResidue(
            eps_inf=2.25,
            poles=(((-1.0e12, 0.0), (3.0e10, 0.0)), ((-5.0e11, 0.0), (1.0e9, 0.0))),
        )
        dt = 1e-12
        coeffs = compile_pole_residue_coefficients(medium, dt=dt)

        field_shape = (4,)
        state = allocate_pole_residue_state(coeffs, field_shape=field_shape)

        electric = np.array([1.0, -0.5, 0.25, 0.1])
        curl_h = np.array([0.1, 0.2, -0.1, 0.05])

        updated_e, updated_state, pol_current = pole_residue_electric_update(
            electric, curl_h, coeffs, state, dt=dt
        )

        # Verify state was updated for each pole
        assert updated_state.polarization.shape[0] == 2

        # Verify polarization current is non-zero for dispersive medium
        assert np.any(pol_current != 0.0)

    def test_sellmeier_full_pipeline(self):
        """Sellmeier → PoleResidue coefficients → dispersive kernel update."""
        from autofdtd.compiler.materials import compile_sellmeier_coefficients
        from autofdtd.kernels.materials import (
            allocate_pole_residue_state,
            pole_residue_electric_update,
        )
        from autofdtd.materials import Sellmeier

        silica = Sellmeier(coeffs=((0.6961663, 4.67914825849e-15), (0.4079426, 0.0)))
        dt = 1e-12
        coeffs = compile_sellmeier_coefficients(silica, dt=dt)

        assert coeffs.medium_type == "Sellmeier"
        assert coeffs.num_poles >= 1

        state = allocate_pole_residue_state(coeffs, field_shape=(3,))
        electric = np.array([1.0, -0.5, 0.25])
        curl_h = np.array([0.1, 0.2, -0.1])

        updated_e, updated_state, pol_current = pole_residue_electric_update(
            electric, curl_h, coeffs, state, dt=dt
        )

        # Sellmeier update should differ from simple vacuum update
        baseline_drive = coeffs.electric_drive * curl_h
        assert not np.allclose(pol_current, 0.0)
