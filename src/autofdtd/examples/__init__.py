"""Validation example and benchmark namespace."""

from autofdtd.examples.boundaries import (
    abc_boundary_snapshot,
    boundary_runtime_snapshot,
    pml_absorption_snapshot,
)
from autofdtd.examples.composite_geometry import rotated_ring_resonator
from autofdtd.examples.current_sources import uniform_current_source_snapshot
from autofdtd.examples.geometry_array import rotated_post_array_scene
from autofdtd.examples.grids import autogrid_example, grid_spec_examples, resolved_grid_example
from autofdtd.examples.materials import (
    build_material_scene,
    compiled_anisotropic_example,
    compiled_lorentz_drude_debye_examples,
    compiled_material_example,
)
from autofdtd.examples.polyslab import directional_coupler_scene, strip_waveguide_simulation
from autofdtd.examples.runtime_controls import (
    runtime_control_snapshot,
    runtime_logging_demo,
    shutoff_demo,
)
from autofdtd.examples.subpixel import build_subpixel_demo

__all__ = [
    "abc_boundary_snapshot",
    "boundary_runtime_snapshot",
    "pml_absorption_snapshot",
    "directional_coupler_scene",
    "autogrid_example",
    "build_subpixel_demo",
    "compiled_anisotropic_example",
    "compiled_lorentz_drude_debye_examples",
    "build_material_scene",
    "compiled_material_example",
    "grid_spec_examples",
    "resolved_grid_example",
    "rotated_post_array_scene",
    "rotated_ring_resonator",
    "runtime_control_snapshot",
    "runtime_logging_demo",
    "shutoff_demo",
    "strip_waveguide_simulation",
    "uniform_current_source_snapshot",
]
