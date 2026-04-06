"""Representative grid planning examples for Phase 1 discretization work."""

from __future__ import annotations

from autofdtd.api import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    ResolvedGrid,
    UniformGrid,
)


def grid_spec_examples() -> tuple[GridSpec, GridSpec]:
    """Return representative uniform and mixed explicit grid specifications."""
    uniform = GridSpec.uniform(0.05)
    mixed = GridSpec(
        grid_x=UniformGrid(dl=0.04),
        grid_y=CustomGrid(dl=(0.1, 0.05, 0.05, 0.1)),
        grid_z=CustomGridBoundaries(coords=(-0.5, -0.2, 0.0, 0.2, 0.5)),
    )
    return uniform, mixed


def autogrid_example() -> GridSpec:
    """Return a representative AutoGrid configuration with layer refinement metadata."""
    return GridSpec(
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


def resolved_grid_example() -> ResolvedGrid:
    """Return a resolved grid with explicit axis boundaries."""
    _, mixed = grid_spec_examples()
    return mixed.make_grid(center=(0.0, 0.0, 0.0), size=(1.2, 0.6, 1.0))


__all__ = ["autogrid_example", "grid_spec_examples", "resolved_grid_example"]
