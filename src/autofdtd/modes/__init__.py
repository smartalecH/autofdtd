"""Mode-solver and mode-interaction namespace for Phase 1.

Provides a vector finite-difference mode solver for electromagnetic waveguide
modes, compatible with sampled epsilon tensors from scenes and cross-sectional
grids. The solver implements the eigenvalue formulation from VectorModesolver.jl
adapted for Python with NumPy/SciPy and Warp-ready kernel conventions.

Public API
----------
autofdtd.modes.ModeSpec
    Mode-specification controls for solver and mode-source workflows.
autofdtd.modes.ModeSolverConfig
    Full configuration for the vector mode solver.
autofdtd.modes.ModeSolverCrossSection
    Cross-sectional plane specification.
autofdtd.modes.BoundaryCondition
    Boundary condition for each cross-section edge.
autofdtd.modes.ModeSolution
    Solved mode with field components and effective index.
autofdtd.modes.solve_modes
    Primary entry point for solving waveguide modes.
autofdtd.modes.epsilon
    Utilities for sampling scene epsilon tensors onto cross-sectional grids.

IR Lowering
-----------
autofdtd.modes.ir.ModeSolverConfigIR
    Versioned IR for solver configuration bundles.
autofdtd.modes.ir.ModeSolutionIR
    Versioned IR for solved mode profiles.
"""

from autofdtd.modes.epsilon import EpsilonCallback, sample_scene_epsilon_tensor_2d
from autofdtd.modes.ir import (
    ModeSolverConfigIR,
    ModeSolutionIR,
    ModeSpecIR,
    boundary_condition_to_ir,
    cross_section_to_ir,
    mode_solver_config_to_ir,
    mode_solution_to_ir,
    mode_spec_to_ir,
)
from autofdtd.modes.models import (
    BoundaryCondition,
    ModeSolution,
    ModeSolverConfig,
    ModeSolverCrossSection,
    ModeSpec,
)
from autofdtd.modes.solver import EpsilonCallback as SolverEpsilonCallback
from autofdtd.modes.solver import assemble_mode_matrix, solve_modes

__all__ = [
    # Public models
    "ModeSpec",
    "ModeSolverConfig",
    "ModeSolverCrossSection",
    "BoundaryCondition",
    "ModeSolution",
    # Solver entry points
    "solve_modes",
    "assemble_mode_matrix",
    # Epsilon sampling
    "EpsilonCallback",
    "sample_scene_epsilon_tensor_2d",
    # IR
    "ModeSpecIR",
    "ModeSolverConfigIR",
    "ModeSolutionIR",
    "mode_spec_to_ir",
    "cross_section_to_ir",
    "boundary_condition_to_ir",
    "mode_solver_config_to_ir",
    "mode_solution_to_ir",
]
