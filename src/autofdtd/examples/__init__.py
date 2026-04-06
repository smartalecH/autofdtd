"""Validation example and benchmark namespace."""

from autofdtd.examples.composite_geometry import rotated_ring_resonator
from autofdtd.examples.geometry_array import rotated_post_array_scene
from autofdtd.examples.grids import autogrid_example, grid_spec_examples, resolved_grid_example
from autofdtd.examples.materials import (
    build_material_scene,
    compiled_lorentz_drude_debye_examples,
    compiled_material_example,
)
from autofdtd.examples.polyslab import directional_coupler_scene, strip_waveguide_simulation
from autofdtd.examples.subpixel import build_subpixel_demo

__all__ = [
    "directional_coupler_scene",
    "autogrid_example",
    "build_subpixel_demo",
    "compiled_lorentz_drude_debye_examples",
    "build_material_scene",
    "compiled_material_example",
    "grid_spec_examples",
    "resolved_grid_example",
    "rotated_post_array_scene",
    "rotated_ring_resonator",
    "strip_waveguide_simulation",
]
