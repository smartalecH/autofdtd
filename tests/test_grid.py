from __future__ import annotations

import pytest

from autofdtd.api import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    PolarizedAveraging,
    Staircasing,
    SubpixelSpec,
    UniformGrid,
)


def test_uniform_grid_snaps_to_domain_extent() -> None:
    grid = UniformGrid(dl=0.3)

    boundaries = grid.make_boundaries(center=0.0, size=1.0)

    assert boundaries == pytest.approx((-0.5, -0.25, 0.0, 0.25, 0.5))
    assert grid.estimated_min_dl() == pytest.approx(0.3)


def test_custom_grid_extends_edge_cells_to_cover_domain() -> None:
    grid = CustomGrid(dl=(0.2, 0.1, 0.1))

    boundaries = grid.make_boundaries(center=0.0, size=1.0)

    assert boundaries == pytest.approx((-0.5, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5))


def test_custom_grid_boundaries_must_match_simulation_domain() -> None:
    grid = CustomGridBoundaries(coords=(-0.4, 0.0, 0.5))

    with pytest.raises(ValueError, match="start and end on the simulation axis bounds"):
        grid.make_boundaries(center=0.0, size=1.0)


def test_gridspec_resolves_all_axes_into_concrete_shape() -> None:
    grid_spec = GridSpec(
        grid_x=UniformGrid(dl=0.25),
        grid_y=CustomGrid(dl=(0.2, 0.1)),
        grid_z=CustomGridBoundaries(coords=(-0.5, -0.1, 0.2, 0.5)),
    )

    resolved = grid_spec.make_grid(center=(0.0, 0.0, 0.0), size=(1.0, 0.8, 1.0))

    assert resolved.shape == (4, 7, 3)
    assert resolved.total_cells == 84
    assert resolved.x.boundaries == pytest.approx((-0.5, -0.25, 0.0, 0.25, 0.5))
    assert resolved.y.boundaries == pytest.approx((-0.4, -0.35, -0.15, 0.05, 0.15, 0.25, 0.35, 0.4))
    assert resolved.z.cell_sizes == pytest.approx((0.4, 0.3, 0.3))
    assert resolved.min_step == pytest.approx(0.05)
    assert resolved.max_step == pytest.approx(0.4)


def test_gridspec_uniform_helper_sets_all_axes() -> None:
    grid_spec = GridSpec.uniform(0.125)

    assert isinstance(grid_spec.grid_x, UniformGrid)
    assert isinstance(grid_spec.grid_y, UniformGrid)
    assert isinstance(grid_spec.grid_z, UniformGrid)
    assert grid_spec.grid_x.dl == pytest.approx(0.125)


def test_autogrid_uses_wavelength_based_spacing() -> None:
    grid_spec = GridSpec(
        grid_x=AutoGrid(min_steps_per_wvl=10, min_steps_per_sim_size=4),
        grid_y=UniformGrid(dl=0.5),
        grid_z=UniformGrid(dl=0.5),
        wavelength=1.0,
    )

    resolved = grid_spec.make_grid(center=(0.0, 0.0, 0.0), size=(4.0, 1.0, 1.0))

    assert resolved.x.cell_sizes[0] == pytest.approx(0.1)
    assert resolved.x.num_cells == 40


def test_autogrid_applies_layer_refinement_and_snapping() -> None:
    grid_spec = GridSpec(
        grid_x=AutoGrid(min_steps_per_wvl=10, max_scale=2.0),
        grid_y=UniformGrid(dl=1.0),
        grid_z=UniformGrid(dl=1.0),
        wavelength=1.0,
        layer_refinement_specs=(
            LayerRefinementSpec(
                axis=0,
                center=(0.0, 0.0, 0.0),
                size=(0.6, 1.0, 1.0),
                min_steps_along_axis=6,
                bounds_refinement=GridRefinement(refinement_factor=2.0, num_cells=4),
                bounds_snapping="bounds",
            ),
        ),
    )

    resolved = grid_spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 1.0, 1.0))

    assert any(abs(value + 0.3) <= 1e-9 for value in resolved.x.boundaries)
    assert any(abs(value - 0.3) <= 1e-9 for value in resolved.x.boundaries)
    assert resolved.x.min_step <= 0.05 + 1e-9


def test_autogrid_requires_wavelength() -> None:
    grid_spec = GridSpec(grid_x=AutoGrid())

    with pytest.raises(ValueError, match="wavelength is required"):
        grid_spec.make_grid(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))


def test_subpixel_spec_defaults_match_phase1_policy_scope() -> None:
    spec = SubpixelSpec()

    assert isinstance(spec.dielectric, PolarizedAveraging)
    assert isinstance(spec.metal, Staircasing)
    assert spec.averaging_targets() == ("dielectric",)
    assert spec.staircasing_targets() == ("metal", "pec", "pmc", "lossy_metal")
    assert spec.requires_anisotropic_materialization()


def test_subpixel_spec_staircasing_helper_disables_averaging() -> None:
    spec = SubpixelSpec.staircasing()

    assert isinstance(spec.dielectric, Staircasing)
    assert spec.averaging_targets() == ()
    assert spec.staircasing_targets() == (
        "dielectric",
        "metal",
        "pec",
        "pmc",
        "lossy_metal",
    )
    assert not spec.requires_anisotropic_materialization()


def test_subpixel_spec_rejects_deferred_or_invalid_policy_selection() -> None:
    with pytest.raises(ValueError, match="ContourPathAveraging"):
        SubpixelSpec(dielectric={"type": "ContourPathAveraging"})

    with pytest.raises(ValueError, match="only supports Staircasing"):
        SubpixelSpec(pec={"type": "PolarizedAveraging"})
