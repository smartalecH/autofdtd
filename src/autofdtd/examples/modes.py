"""Example usage of the Phase 1 vector mode solver.

Demonstrates solving for waveguide modes using the VectorModesolver.jl-inspired
finite-difference eigenvalue formulation, with both simple analytic epsilon
callbacks and scene-based epsilon sampling from structures.
"""

from __future__ import annotations

import math
from typing import Callable

from autofdtd.api import Box, Medium, Scene, Simulation, Structure
from autofdtd.grid import GridSpec, UniformGrid
from autofdtd.modes import (
    BoundaryCondition,
    ModeSpec,
    ModeSolverConfig,
    ModeSolverCrossSection,
    sample_scene_epsilon_tensor_2d,
    solve_modes,
)


# --- Analytic epsilon callbacks ---

def make_slab_epsilon(
    n_core: float = 1.5,
    n_clad: float = 1.0,
    y_center: float = 0.5,
    half_width: float = 0.2,
) -> Callable[[float, float], tuple[float, float, float, float, float]]:
    """Create a step-index slab waveguide epsilon callback.

    Args:
        n_core: Refractive index of the core
        n_clad: Refractive index of the cladding
        y_center: Y-coordinate of the slab center
        half_width: Half-width of the core region

    Returns:
        Epsilon callback ε(x, y) → (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
    """

    def epsilon(x: float, y: float) -> tuple[float, float, float, float, float]:
        if abs(y - y_center) < half_width:
            eps_core = n_core**2
            return (eps_core, 0.0, 0.0, eps_core, eps_core)
        eps_clad = n_clad**2
        return (eps_clad, 0.0, 0.0, eps_clad, eps_clad)

    return epsilon


def make_rectangular_waveguide_epsilon(
    n_core: float = 1.5,
    n_clad: float = 1.0,
    x_center: float = 0.5,
    y_center: float = 0.5,
    half_width_x: float = 0.1,
    half_width_y: float = 0.1,
) -> Callable[[float, float], tuple[float, float, float, float, float]]:
    """Create a rectangular waveguide epsilon callback."""

    def epsilon(x: float, y: float) -> tuple[float, float, float, float, float]:
        in_core = abs(x - x_center) < half_width_x and abs(y - y_center) < half_width_y
        if in_core:
            eps_core = n_core**2
            return (eps_core, 0.0, 0.0, eps_core, eps_core)
        eps_clad = n_clad**2
        return (eps_clad, 0.0, 0.0, eps_clad, eps_clad)

    return epsilon


def example_slab_waveguide():
    """Solve for modes in a step-index slab waveguide.

    This reproduces the classic step-index slab waveguide problem.
    For a step-index slab with n_core=1.5, n_clad=1.0 at 1550nm,
    the fundamental TE mode should have neff ≈ 1.45.
    """
    print("=== Slab Waveguide Mode Solver ===")

    wavelength = 1.55e-6  # 1550 nm

    # Cross-section grid (x is transverse, y is propagation direction in 2D)
    num_cells = 40
    domain_size = 2.0e-5  # 20 microns
    x = tuple(domain_size * i / num_cells for i in range(1, num_cells + 1))

    # Use a 2D solver: x is the in-plane direction, y is through the slab
    y = tuple(domain_size * i / num_cells for i in range(1, num_cells + 1))

    config = ModeSolverConfig(
        wavelength=wavelength,
        cross_section=ModeSolverCrossSection(
            normal_axis=2,  # z-propagating modes
            position=0.0,
            boundaries=(
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
            ),
        ),
        mode_spec=ModeSpec(num_modes=4),
        tolerance=1e-8,
    )

    eps = make_slab_epsilon(n_core=1.5, n_clad=1.0, y_center=domain_size / 2, half_width=domain_size / 6)

    print(f"Grid: {len(x)} x {len(y)} cells")
    print(f"Wavelength: {wavelength * 1e9:.1f} nm")
    print("Solving for modes...")

    modes = solve_modes(config, eps, x, y)

    print(f"\nFound {len(modes)} modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i}: neff = {mode.neff_real:.6f}, power = {mode.power:.6e}")

    # Verify fundamental mode is guided
    assert len(modes) >= 1
    neff_fundamental = modes[0].neff_real
    assert 1.0 < neff_fundamental < 2.0, f"neff {neff_fundamental} out of physical range"

    return modes


def example_rectangular_waveguide():
    """Solve for modes in a rectangular dielectric waveguide."""
    print("\n=== Rectangular Waveguide Mode Solver ===")

    wavelength = 1.55e-6
    domain_size = 2.0e-5
    num_cells = 30
    x = tuple(domain_size * i / num_cells for i in range(1, num_cells + 1))
    y = tuple(domain_size * i / num_cells for i in range(1, num_cells + 1))

    config = ModeSolverConfig(
        wavelength=wavelength,
        cross_section=ModeSolverCrossSection(
            normal_axis=2,
            position=0.0,
            boundaries=(
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
                BoundaryCondition(condition="dirichlet"),
            ),
        ),
        mode_spec=ModeSpec(num_modes=3),
        tolerance=1e-8,
    )

    eps = make_rectangular_waveguide_epsilon(
        n_core=1.5,
        n_clad=1.0,
        x_center=domain_size / 2,
        y_center=domain_size / 2,
        half_width_x=domain_size / 10,
        half_width_y=domain_size / 10,
    )

    modes = solve_modes(config, eps, x, y)

    print(f"Found {len(modes)} modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i}: neff = {mode.neff_real:.6f}")

    return modes


def example_scene_based_modes():
    """Solve modes using epsilon sampled from an autofdtd Scene.

    This demonstrates the full pipeline: create structures, sample the
    scene permittivity onto a cross-sectional grid, and solve for modes.
    """
    print("\n=== Scene-Based Mode Solver ===")

    wavelength = 1.55e-6

    # Create a simulation with a waveguide structure
    sim_size = (2.0e-5, 2.0e-5, 2.0e-5)
    sim_center = (0.0, 0.0, 0.0)

    # Build a scene with a box waveguide
    medium_core = Medium(permittivity=2.25)  # n ≈ 1.5
    medium_clad = Medium(permittivity=1.0)  # n = 1.0

    # The waveguide is a thin box in the x-y plane
    # For z-propagating modes, the cross-section is at z = 0
    waveguide = Box(
        center=(0.0, sim_size[1] / 2, 0.0),  # Center in y, at z=0
        size=(sim_size[0] / 4, sim_size[1] / 5, sim_size[2]),
    )
    structure = Structure(geometry=waveguide, medium=medium_core)

    scene = Scene(structures=(structure,), medium=medium_clad)

    # Configure cross-section
    cross_section = ModeSolverCrossSection(
        normal_axis=2,  # z-normal: solve for z-propagating modes
        position=0.0,  # At z=0
        boundaries=(
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
            BoundaryCondition(condition="dirichlet"),
        ),
    )

    # Sample scene epsilon onto grid
    print("Sampling scene epsilon tensor...")
    dt = 1e-17  # Small timestep for coefficient compilation
    eps_callback, x_coords, y_coords, cell_size = sample_scene_epsilon_tensor_2d(
        scene=scene,
        cross_section=cross_section,
        sim_center=sim_center,
        sim_size=sim_size,
        wavelength=wavelength,
        dt=dt,
    )

    print(f"Grid: {len(x_coords)} x {len(y_coords)} cells, cell size = {cell_size:.2e} m")

    # Configure and run solver
    config = ModeSolverConfig(
        wavelength=wavelength,
        cross_section=cross_section,
        mode_spec=ModeSpec(num_modes=3),
        tolerance=1e-8,
    )

    print("Solving for modes...")
    modes = solve_modes(config, eps_callback, x_coords, y_coords)

    print(f"\nFound {len(modes)} modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i}: neff = {mode.neff_real:.6f}")

    return modes


def main():
    """Run all mode-solver examples."""
    modes_slab = example_slab_waveguide()
    modes_rect = example_rectangular_waveguide()
    modes_scene = example_scene_based_modes()

    print("\n=== Summary ===")
    print(f"Slab waveguide fundamental neff: {modes_slab[0].neff_real:.6f}")
    print(f"Rectangular waveguide fundamental neff: {modes_rect[0].neff_real:.6f}")
    print(f"Scene-based fundamental neff: {modes_scene[0].neff_real:.6f}")
    print("\nAll mode-solver examples completed successfully.")


if __name__ == "__main__":
    main()
