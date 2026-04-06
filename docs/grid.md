# Grid Scope

Phase 1 grid support now covers both explicit manual discretization and a first
AutoGrid-style planning path with inspectable layer-refinement metadata.

## Implemented Models

- `UniformGrid` for constant step size on one axis
- `CustomGrid` for explicit cell sizes with edge-cell extension to cover the full domain
- `CustomGridBoundaries` for fully explicit axis boundary coordinates
- `GridSpec` as the top-level container that resolves all three axes together
- `AutoGrid` for vacuum-scale-driven meshing with wavelength-aware spacing
- `GridRefinement` for local refinement targets used by AutoGrid planning
- `LayerRefinementSpec` for axis-aligned layer snapping and bounds refinement

```python
from autofdtd.api import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    UniformGrid,
)

grid_spec = GridSpec(
    grid_x=UniformGrid(dl=0.02),
    grid_y=CustomGrid(dl=(0.03, 0.02, 0.03)),
    grid_z=CustomGridBoundaries(coords=(-2.0, -0.5, 0.5, 2.0)),
)
resolved = grid_spec.make_grid(center=(0.0, 0.0, 0.0), size=(12.0, 8.0, 4.0))

auto = GridSpec(
    grid_x=AutoGrid(min_steps_per_wvl=16, max_scale=1.6),
    grid_y=UniformGrid(dl=0.05),
    grid_z=UniformGrid(dl=0.05),
    wavelength=1.55,
    layer_refinement_specs=(
        LayerRefinementSpec(
            axis=0,
            center=(0.0, 0.0, 0.0),
            size=(0.22, 1.0, 1.0),
            min_steps_along_axis=6,
            bounds_refinement=GridRefinement(refinement_factor=2.0, num_cells=5),
            bounds_snapping="bounds",
        ),
    ),
)
```

## Resolution Behavior

- `UniformGrid` snaps the final step size so the axis length is covered exactly
- `CustomGrid` centers the supplied pattern by default and repeats the first and last
  cell sizes as needed to reach the domain bounds
- `CustomGridBoundaries` must already start and end on the simulation axis bounds
- `AutoGrid` currently derives a vacuum baseline from `wavelength`,
  `min_steps_per_wvl`, and `min_steps_per_sim_size`, then applies a simplified
  graded piecewise discretization
- `LayerRefinementSpec` currently affects only matching `AutoGrid` axes and only
  through layer-thickness refinement plus snapping or refinement around layer bounds
- `Simulation.resolved_grid()` now validates and exposes the concrete Phase 1 grid
  layout for later compiler, runtime, and mode-solver work

## Explicit Phase 1 Limits

- `grid_spec` must use a top-level `GridSpec` container
- `GridSpec.wavelength` is required whenever any axis uses `AutoGrid`
- `QuasiUniformGrid` remains planned and is still rejected explicitly
- `LayerRefinementSpec` is limited to axis-aligned layer boxes; corner finding,
  geometry-driven feature detection, and mesh override structures are not implemented yet
- zero-size axes resolve to zero cells with duplicate boundary endpoints so 2D and
  1D simulation shells can still carry inspectable grid metadata
