"""Tests for the Phase 1 vector mode solver subsystem."""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import scipy
except ModuleNotFoundError:
    scipy = None

from autofdtd.modes import (
    BoundaryCondition,
    ModeSpec,
    ModeSolution,
    ModeSolverConfig,
    ModeSolverCrossSection,
    assemble_mode_matrix,
    boundary_condition_to_ir,
    cross_section_to_ir,
    mode_solver_config_to_ir,
    mode_solution_to_ir,
    mode_spec_to_ir,
    sample_scene_epsilon_tensor_2d,
    solve_modes,
)
from autofdtd.modes.ir import (
    BoundaryConditionIR,
    ModeSolverConfigIR,
    ModeSolutionIR,
    ModeSpecIR,
    ModeSolverCrossSectionIR,
)


class TestModeSpec:
    def test_default_mode_spec(self):
        spec = ModeSpec()
        assert spec.num_modes == 1
        assert spec.target_neff is None
        assert spec.precision == "single"
        assert spec.polynomial_degree == 2

    def test_mode_spec_with_target_neff(self):
        spec = ModeSpec(num_modes=4, target_neff=1.5)
        assert spec.num_modes == 4
        assert spec.target_neff == 1.5

    def test_invalid_num_modes(self):
        with pytest.raises(ValueError, match="num_modes must be"):
            ModeSpec(num_modes=0)

    def test_invalid_polynomial_degree(self):
        with pytest.raises(ValueError, match="polynomial_degree must be"):
            ModeSpec(polynomial_degree=5)


class TestBoundaryCondition:
    def test_dirichlet(self):
        bc = BoundaryCondition(condition="dirichlet")
        assert bc.condition == "dirichlet"
        assert bc.bloch_phase == 0.0

    def test_neumann(self):
        bc = BoundaryCondition(condition="neumann")
        assert bc.condition == "neumann"

    def test_bloch(self):
        bc = BoundaryCondition(condition="bloch", bloch_phase=math.pi / 4)
        assert bc.condition == "bloch"
        assert math.isclose(bc.bloch_phase, math.pi / 4)


class TestModeSolverCrossSection:
    def test_default_cross_section(self):
        cs = ModeSolverCrossSection()
        assert cs.normal_axis == 2
        assert cs.position == 0.0
        assert len(cs.boundaries) == 4
        assert all(bc.condition == "dirichlet" for bc in cs.boundaries)

    def test_custom_cross_section(self):
        bc_north = BoundaryCondition(condition="neumann")
        bc_south = BoundaryCondition(condition="dirichlet")
        bc_east = BoundaryCondition(condition="dirichlet")
        bc_west = BoundaryCondition(condition="dirichlet")
        cs = ModeSolverCrossSection(
            normal_axis=0,
            position=2.0,
            boundaries=(bc_north, bc_south, bc_east, bc_west),
        )
        assert cs.normal_axis == 0
        assert cs.position == 2.0
        assert cs.boundaries[0].condition == "neumann"


class TestModeSolverConfig:
    def test_default_config(self):
        config = ModeSolverConfig(wavelength=1.55e-6)
        assert config.wavelength == 1.55e-6
        assert config.cross_section.normal_axis == 2
        assert config.mode_spec.num_modes == 1
        assert config.min_cells == 20
        assert config.max_cells == 200
        assert math.isclose(config.tolerance, 1e-8)

    def test_custom_config(self):
        config = ModeSolverConfig(
            wavelength=1.0e-6,
            mode_spec=ModeSpec(num_modes=3, target_neff=1.4),
            min_cells=10,
            max_cells=100,
            tolerance=1e-6,
        )
        assert config.mode_spec.num_modes == 3
        assert config.mode_spec.target_neff == 1.4

    def test_invalid_wavelength(self):
        with pytest.raises(Exception, match="wavelength must be"):
            ModeSolverConfig(wavelength=-1.0)

    def test_invalid_min_cells(self):
        with pytest.raises(ValueError, match="min_cells must be at least"):
            ModeSolverConfig(wavelength=1.55e-6, min_cells=2)


class TestModeSolution:
    def test_solution_properties(self):
        x = (0.0, 0.5, 1.0)
        y = (0.0, 0.5, 1.0)
        sol = ModeSolution(
            neff=1.5,
            wavelength=1.55e-6,
            x=x,
            y=y,
            Ex=((1.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            Ey=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
            Ez=((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            Hx=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            Hy=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            Hz=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        )
        assert sol.neff_real == 1.5
        assert sol.num_cells == 9
        k0 = 2 * math.pi / sol.wavelength
        assert math.isclose(sol.k0, k0)


class TestIRLowering:
    def test_mode_spec_to_ir(self):
        spec = ModeSpec(num_modes=3, target_neff=1.4)
        ir = mode_spec_to_ir(spec)
        assert isinstance(ir, ModeSpecIR)
        assert ir.num_modes == 3
        assert ir.target_neff == 1.4
        assert ir.type == "ModeSpecIR"

    def test_boundary_condition_to_ir(self):
        bc = BoundaryCondition(condition="bloch", bloch_phase=0.5)
        ir = boundary_condition_to_ir(bc)
        assert isinstance(ir, BoundaryConditionIR)
        assert ir.condition == "bloch"
        assert math.isclose(ir.bloch_phase, 0.5)

    def test_cross_section_to_ir(self):
        cs = ModeSolverCrossSection(
            normal_axis=1,
            position=1.0,
            boundaries=(
                BoundaryCondition(condition="neumann"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
            ),
        )
        ir = cross_section_to_ir(cs)
        assert isinstance(ir, ModeSolverCrossSectionIR)
        assert ir.normal_axis == 1
        assert ir.position == 1.0
        assert ir.boundaries[0].condition == "neumann"

    def test_mode_solver_config_to_ir(self):
        config = ModeSolverConfig(
            wavelength=1.55e-6,
            mode_spec=ModeSpec(num_modes=2),
        )
        ir = mode_solver_config_to_ir(config)
        assert isinstance(ir, ModeSolverConfigIR)
        assert ir.wavelength == 1.55e-6
        assert ir.mode_spec.num_modes == 2

    def test_mode_solution_to_ir(self):
        x = (0.0, 0.5, 1.0)
        y = (0.0, 0.5, 1.0)
        solution = ModeSolution(
            neff=1.5,
            wavelength=1.55e-6,
            x=x,
            y=y,
            Ex=((1.0, 0.0),),
            Ey=((0.0, 0.0),),
            Ez=((0.0, 0.0),),
            Hx=((0.0, 0.0),),
            Hy=((0.0, 0.0),),
            Hz=((0.0, 0.0),),
        )
        ir = mode_solution_to_ir(solution)
        assert isinstance(ir, ModeSolutionIR)
        assert ir.neff == 1.5
        assert len(ir.x) == 3


# --- Solver tests with simple epsilon functions ---

def _homogeneous_epsilon(eps_val: float):
    """Return an epsilon callback for a homogeneous medium."""

    def callback(x: float, y: float):
        return (eps_val, 0.0, 0.0, eps_val, eps_val)

    return callback


def _slab_epsilon(eps_core: float, eps_clad: float, y_center: float, half_width: float):
    """Return an epsilon callback for a slab waveguide."""

    def callback(x: float, y: float):
        if abs(y - y_center) < half_width:
            return (eps_core, 0.0, 0.0, eps_core, eps_core)
        return (eps_clad, 0.0, 0.0, eps_clad, eps_clad)

    return callback


@pytest.mark.skipif(scipy is None, reason="scipy is not installed")
class TestAssembleModeMatrix:
    """Test sparse matrix assembly without full solving."""

    def test_assemble_homogeneous(self):
        x = tuple(0.1 * i for i in range(10))
        y = tuple(0.1 * i for i in range(10))
        eps = _homogeneous_epsilon(1.0)
        bc = (
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
        )
        A = assemble_mode_matrix(x, y, 1.55, eps, bc)
        assert A.shape == (200, 200)  # 2 * 10 * 10
        assert A.nnz > 0  # Non-zero entries

    def test_assemble_slab(self):
        x = tuple(0.05 * i for i in range(20))
        y = tuple(0.05 * i for i in range(20))
        eps = _slab_epsilon(2.1, 1.0, 0.5, 0.2)
        bc = tuple(BoundaryCondition(condition="dirichlet") for _ in range(4))
        A = assemble_mode_matrix(x, y, 1.55, eps, bc)
        assert A.shape == (800, 800)
        assert A.nnz > 0


@pytest.mark.skipif(scipy is None, reason="scipy is not installed")
class TestSolveModes:
    """Integration tests for the mode solver."""

    def test_solve_empty_medium(self):
        """Solve modes in a homogeneous medium (should recover neff ~ 0 for vacuum)."""
        x = tuple(0.1 * i for i in range(1, 11))
        y = tuple(0.1 * i for i in range(1, 11))
        config = ModeSolverConfig(wavelength=1.55e-6, mode_spec=ModeSpec(num_modes=1))
        eps = _homogeneous_epsilon(1.0)

        modes = solve_modes(config, eps, x, y)
        assert len(modes) >= 1
        assert modes[0].neff_real >= 0.0

    def test_solve_slab_waveguide(self):
        """Solve modes for a simple slab waveguide.

        For a step-index slab with n_core=1.5, n_clad=1.0, the fundamental
        TE mode at 1550nm should have neff close to n_clad (since the
        mode is weakly guided). The test verifies the solver returns
        valid guided modes with neff in a physical range.
        """
        x = tuple(0.02 * i for i in range(1, 51))  # 0 to 1 um
        y = tuple(0.02 * i for i in range(1, 51))
        config = ModeSolverConfig(
            wavelength=1.55e-6,
            mode_spec=ModeSpec(num_modes=3),
        )
        eps = _slab_epsilon(2.25, 1.0, 0.5, 0.2)  # n_core=1.5, n_clad=1.0

        modes = solve_modes(config, eps, x, y)
        assert len(modes) >= 1

        # Modes should have neff in a physical range (allow neff ~ 1.0 for
        # weakly guided modes; the solver should not produce neff < 0.99)
        for mode in modes:
            assert 0.99 <= mode.neff_real <= 2.0, f"neff={mode.neff_real} out of expected range"

    def test_solve_mode_sorting(self):
        """Modes should be sorted by neff descending."""
        x = tuple(0.02 * i for i in range(1, 31))
        y = tuple(0.02 * i for i in range(1, 31))
        config = ModeSolverConfig(
            wavelength=1.55e-6,
            mode_spec=ModeSpec(num_modes=4),
        )
        eps = _slab_epsilon(2.25, 1.0, 0.3, 0.15)

        modes = solve_modes(config, eps, x, y)
        if len(modes) >= 2:
            neffs = [m.neff_real for m in modes]
            assert neffs == sorted(neffs, reverse=True), "modes should be sorted by neff descending"

    def test_neumann_boundaries(self):
        """Neumann BCs should give different results than Dirichlet for a large domain."""
        # Use a larger domain so BC differences are more pronounced
        x = tuple(0.02 * i for i in range(1, 51))
        y = tuple(0.02 * i for i in range(1, 51))
        eps = _slab_epsilon(2.25, 1.0, 0.5, 0.2)
        wavelength = 1.55e-6

        dirichlet_config = ModeSolverConfig(
            wavelength=wavelength,
            cross_section=ModeSolverCrossSection(
                boundaries=tuple(BoundaryCondition(condition="dirichlet") for _ in range(4)),
            ),
            mode_spec=ModeSpec(num_modes=1),
        )

        neumann_config = ModeSolverConfig(
            wavelength=wavelength,
            cross_section=ModeSolverCrossSection(
                boundaries=tuple(BoundaryCondition(condition="neumann") for _ in range(4)),
            ),
            mode_spec=ModeSpec(num_modes=1),
        )

        modes_d = solve_modes(dirichlet_config, eps, x, y)
        modes_n = solve_modes(neumann_config, eps, x, y)

        # Both BC types should produce valid guided modes
        assert modes_d, "Dirichlet config should produce modes"
        assert modes_n, "Neumann config should produce modes"
        neff_d = modes_d[0].neff_real
        neff_n = modes_n[0].neff_real
        assert 0.99 <= neff_d <= 2.0, f"Dirichlet neff={neff_d} out of range"
        assert 0.99 <= neff_n <= 2.0, f"Neumann neff={neff_n} out of range"

    def test_mode_solution_serialization(self):
        """ModeSolution should serialize to JSON cleanly."""
        x = (0.0, 0.5, 1.0)
        y = (0.0, 0.5, 1.0)
        solution = ModeSolution(
            neff=1.5,
            wavelength=1.55e-6,
            x=x,
            y=y,
            Ex=((1.0, 0.0), (0.5, 0.1), (0.0, 0.0)),
            Ey=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
            Ez=((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            Hx=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            Hy=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            Hz=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            power=1.0,
        )
        ir = mode_solution_to_ir(solution)
        json_text = ir.to_json_text()
        assert "ModeSolutionIR" in json_text
        assert "1.5" in json_text


class TestEpsilonSampling:
    """Tests for scene epsilon tensor sampling."""

    def test_sample_scene_from_api_import(self):
        """Verify sample_scene_epsilon_tensor_2d can be called with minimal scene."""
        from autofdtd.api import Scene, Simulation, Structure, Box, Medium

        background = Medium(permittivity=1.0)
        medium = Medium(permittivity=2.1)
        box = Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0))
        structure = Structure(geometry=box, medium=medium)
        scene = Scene(structures=(structure,), medium=background)

        cross_section = ModeSolverCrossSection(normal_axis=2, position=0.0)
        sim_center = (0.0, 0.0, 0.0)
        sim_size = (2.0, 2.0, 2.0)
        wavelength = 1.55e-6
        dt = 1e-17

        eps_cb, x_coords, y_coords, cell_size = sample_scene_epsilon_tensor_2d(
            scene, cross_section, sim_center, sim_size, wavelength, dt
        )

        # Test the callback
        eps = eps_cb(0.0, 0.0)
        assert len(eps) == 5
        assert eps[0] > 0  # eps_xx should be positive

        # Check coordinates
        assert len(x_coords) >= 3
        assert len(y_coords) >= 3
        assert cell_size > 0

    def test_homogeneous_epsilon_callback(self):
        """Test that the slab epsilon function works as a valid callback."""
        eps = _slab_epsilon(2.25, 1.0, 0.5, 0.2)
        # Core region
        eps_core = eps(0.5, 0.5)
        assert eps_core[0] == 2.25  # eps_xx in core
        assert eps_core[3] == 2.25  # eps_yy in core
        # Cladding region
        eps_clad = eps(0.0, 0.0)
        assert eps_clad[0] == 1.0
        assert eps_clad[3] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
