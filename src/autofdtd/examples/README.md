# autofdtd Examples

This package contains runnable examples exercising the Phase 1 feature families end-to-end.

## Overview

| Module | Focus |
|--------|-------|
| `validation.py` | 10 canonical physics examples (vacuum, plane wave, slab, PML, sources, monitors, shutoff) |
| `integration.py` | Full public API → IR → compilation → execution pipeline |
| `current_sources.py` | UniformCurrentSource, PointDipole, CustomCurrentSource placement and injection |
| `runtime_controls.py` | Runtime control surface, stop-policy, shutoff, logging |
| `near2far.py` | Near-to-far postprocessing, equivalence principle, diffraction orders |
| `modes.py` | Vector mode solver for waveguides using finite-difference eigenvalue |
| `boundaries.py` | PML, ABC, StablePML, Absorber, PEC, PMC boundary conditions |
| `grids.py` | UniformGrid, CustomGrid, AutoGrid with layer refinement |
| `materials.py` | Isotropic, anisotropic, Lorentz/Drude/Debye dispersive media |
| `composite_geometry.py` | GeometryGroup, ClipOperation, Transformed (ring resonator) |
| `geometry_array.py` | GeometryArray with repeated, transformed placements |
| `polyslab.py` | PolySlab for planar photonics (strip waveguide, directional coupler) |
| `subpixel.py` | Subpixel averaging policies for dielectric/metal staircasing |

## Running Examples

```python
# Run all validation examples
from autofdtd.examples import run_all_validation_examples
summary = run_all_validation_examples()

# Run a specific example
from autofdtd.examples.validation import vacuum_point_source_example
result = vacuum_point_source_example()

# Run the integration pipeline demo
from autofdtd.examples.integration import full_pipeline_snapshot
result = full_pipeline_snapshot()

# Run the mode solver examples
from autofdtd.examples.modes import main
main()

# Run the near-to-far demonstrations
from autofdtd.examples.near2far import main
main()

# Run subpixel demo
from autofdtd.examples.subpixel import build_subpixel_demo
sim = build_subpixel_demo()
```

## Validation Examples (`validation.py`)

Each example compiles and runs a small FDTD simulation, then returns a dict of physical metrics.

| Example | What it exercises |
|---------|-------------------|
| `vacuum_point_source` | Point dipole in PML-backed box; energy decays as waves leave domain |
| `vacuum_plane_wave` | Broadband plane wave in vacuum with PML absorption |
| `dielectric_slab` | Slab waveguide with confined mode; validates field confinement |
| `pml_absorption` | Compares PML vs PEC boundaries; validates energy absorption |
| `uniform_current_injection` | Uniform current sheet source; validates outward propagation |
| `field_monitor_recording` | FieldTimeMonitor time-domain recording |
| `flux_monitor_recording` | FluxMonitor power-flow through a surface |
| `medium_monitor` | PermittivityMonitor static recording |
| `convergence_shutoff` | Early stop when integrated field drops below `shutoff` threshold |
| `multi_feature_integration` | Combined: dielectric + PML + source + monitors + shutoff |

## Boundary Types (`boundaries.py`)

| Function | Boundary Type |
|----------|--------------|
| `boundary_runtime_snapshot()` | Periodic, PEC, PMC |
| `pml_absorption_snapshot()` | PML with staged attenuation path |
| `absorbing_boundary_snapshot()` | StablePML, Absorber with custom sigma |
| `abc_boundary_snapshot()` | First-order absorbing boundary condition |

## Grid Types (`grids.py`)

- `grid_spec_examples()` — uniform and mixed explicit grids
- `autogrid_example()` — AutoGrid with layer refinement around a feature
- `resolved_grid_example()` — explicit axis boundaries from a mixed grid

## Material Types (`materials.py`)

| Function | Material |
|----------|----------|
| `compiled_material_example()` | Isotropic dielectric |
| `compiled_pole_residue_example()` | PoleResidue dispersive |
| `compiled_sellmeier_example()` | Sellmeier (fused silica) |
| `compiled_lorentz_drude_debye_examples()` | Lorentz, Drude, Debye dispersive models |
| `compiled_anisotropic_example()` | Diagonal anisotropic (uniaxial) |

## Geometry Types

- `composite_geometry.py` — `rotated_ring_resonator()`: difference of two spheres + rotated bus waveguide
- `geometry_array.py` — `rotated_post_array_scene()`: repeated posts with overlap precedence
- `polyslab.py` — `strip_waveguide_simulation()`: rectangular core; `directional_coupler_scene()`: two parallel cores
- `subpixel.py` — `build_subpixel_demo()`: explicit dielectric/metal/pec averaging policies
