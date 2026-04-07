"""Validation example and benchmark namespace."""

from autofdtd.examples.boundaries import (
    abc_boundary_snapshot,
    boundary_runtime_snapshot,
    pml_absorption_snapshot,
)
from autofdtd.examples.composite_geometry import rotated_ring_resonator
from autofdtd.examples.current_sources import (
    custom_current_source_snapshot,
    point_dipole_snapshot,
    uniform_current_source_snapshot,
)
from autofdtd.examples.geometry_array import rotated_post_array_scene
from autofdtd.examples.grids import autogrid_example, grid_spec_examples, resolved_grid_example
from autofdtd.examples.integration import full_pipeline_snapshot
from autofdtd.examples.materials import (
    build_material_scene,
    compiled_anisotropic_example,
    compiled_lorentz_drude_debye_examples,
    compiled_material_example,
)
from autofdtd.examples.near2far import (
    define_grating_simulation,
    define_radiator_simulation,
    demonstrate_diffraction,
    demonstrate_directivity,
    demonstrate_near2far_transform,
    demonstrate_radiation_pattern,
    demonstrate_surface_currents,
)
from autofdtd.examples.polyslab import directional_coupler_scene, strip_waveguide_simulation
from autofdtd.examples.runtime_controls import (
    benchmark_snapshot,
    runtime_control_snapshot,
    runtime_logging_demo,
    shutoff_demo,
)
from autofdtd.examples.subpixel import build_subpixel_demo
from autofdtd.examples.validation import (
    convergence_shutoff_example,
    dielectric_slab_example,
    field_monitor_recording_example,
    flux_monitor_recording_example,
    medium_monitor_example,
    multi_feature_integration_example,
    pml_absorption_example,
    run_all_validation_examples,
    uniform_current_injection_example,
    vacuum_plane_wave_example,
    vacuum_point_source_example,
)

__all__ = [
    "abc_boundary_snapshot",
    "autogrid_example",
    "benchmark_snapshot",
    "boundary_runtime_snapshot",
    "build_subpixel_demo",
    "compiled_anisotropic_example",
    "compiled_lorentz_drude_debye_examples",
    "build_material_scene",
    "compiled_material_example",
    "convergence_shutoff_example",
    "custom_current_source_snapshot",
    "define_grating_simulation",
    "define_radiator_simulation",
    "demonstrate_diffraction",
    "demonstrate_directivity",
    "demonstrate_near2far_transform",
    "demonstrate_radiation_pattern",
    "demonstrate_surface_currents",
    "dielectric_slab_example",
    "directional_coupler_scene",
    "field_monitor_recording_example",
    "flux_monitor_recording_example",
    "full_pipeline_snapshot",
    "grid_spec_examples",
    "medium_monitor_example",
    "multi_feature_integration_example",
    "pml_absorption_example",
    "pml_absorption_snapshot",
    "point_dipole_snapshot",
    "resolved_grid_example",
    "rotated_post_array_scene",
    "rotated_ring_resonator",
    "run_all_validation_examples",
    "runtime_control_snapshot",
    "runtime_logging_demo",
    "shutoff_demo",
    "strip_waveguide_simulation",
    "uniform_current_injection_example",
    "uniform_current_source_snapshot",
    "vacuum_plane_wave_example",
    "vacuum_point_source_example",
]
