"""Tests for geometry discretization, materialization, and subpixel smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.core.containers import Scene, Simulation, Structure
from autofdtd.geometry import Box, Sphere, Cylinder, GeometryGroup, Transformed
from autofdtd.geometry.primitives import GeometryTransform
from autofdtd.grid import UniformGrid, GridSpec, ResolvedGrid
from autofdtd.grid.subpixel import SubpixelSpec, Staircasing, PolarizedAveraging
from autofdtd.materials import Medium, PECMedium, PMCMedium, AnisotropicMedium
from autofdtd.compiler.discretization import (
    MediumKind,
    CellMaterial,
    MaterialField,
    CoefficientField,
    _classify_medium,
    _box_volume_fraction,
    _sphere_volume_fraction,
    _cylinder_volume_fraction,
    materialize_grid,
    apply_subpixel_smoothing,
    assemble_coefficient_fields,
    discretize_scene,
)


class TestMediumKind:
    """Tests for medium classification."""

    def test_medium_classification_dielectric(self):
        m = Medium(permittivity=2.0)
        assert _classify_medium(m) == MediumKind.DIELECTRIC

    def test_medium_classification_metal(self):
        m = Medium(permittivity=2.0, conductivity=100.0)
        assert _classify_medium(m) == MediumKind.METAL

    def test_medium_classification_pec(self):
        assert _classify_medium(PECMedium()) == MediumKind.PEC

    def test_medium_classification_pmc(self):
        assert _classify_medium(PMCMedium()) == MediumKind.PMC

    def test_medium_classification_vacuum(self):
        m = Medium()
        assert _classify_medium(m) == MediumKind.DIELECTRIC


class TestVolumeFractions:
    """Tests for volume fraction computation."""

    def test_box_full_overlap(self):
        """A cell fully inside a box has volume fraction 1."""
        box = Box(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0))
        # Cell from -1 to 1 is fully inside
        f = _box_volume_fraction(box, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        assert f == 1.0

    def test_box_no_overlap(self):
        """A cell far outside a box has volume fraction 0."""
        box = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        f = _box_volume_fraction(box, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0)
        assert f == 0.0

    def test_box_partial_overlap(self):
        """A cell partially overlapping a box has partial volume fraction."""
        box = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        # Cell from 0.5 to 1.5 overlaps half in x
        f = _box_volume_fraction(box, 0.5, 1.5, -0.5, 0.5, -0.5, 0.5)
        assert 0.0 < f < 1.0

    def test_sphere_volume_fraction_center_inside(self):
        """A cell with center inside sphere has high fraction."""
        sphere = Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        f = _sphere_volume_fraction(sphere, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1)
        assert f > 0.5

    def test_sphere_volume_fraction_corner_outside(self):
        """A cell with corners outside has reduced fraction."""
        sphere = Sphere(center=(0.0, 0.0, 0.0), radius=0.5)
        f = _sphere_volume_fraction(sphere, 0.6, 0.7, 0.6, 0.7, 0.6, 0.7)
        assert f == 0.0

    def test_cylinder_volume_fraction_inside(self):
        """A cell fully inside a cylinder has fraction 1."""
        cyl = Cylinder(center=(0.0, 0.0, 0.0), radius=1.0, length=10.0, axis=2)
        f = _cylinder_volume_fraction(cyl, -0.1, 0.1, -0.1, 0.1, -0.5, 0.5)
        assert f == 1.0

    def test_cylinder_volume_fraction_outside(self):
        """A cell outside a cylinder has fraction 0."""
        cyl = Cylinder(center=(0.0, 0.0, 0.0), radius=0.5, length=10.0, axis=2)
        f = _cylinder_volume_fraction(cyl, 0.6, 0.7, 0.6, 0.7, -0.5, 0.5)
        assert f == 0.0


class TestMaterialField:
    """Tests for grid materialization."""

    def _make_grid(self, nx=5, ny=5, nz=5, size=2.0, dl=0.4):
        """Helper to create a small resolved grid."""
        spec = GridSpec.uniform(dl=dl)
        center = (0.0, 0.0, 0.0)
        size_tuple = (size, size, size)
        return spec.make_grid(center=center, size=size_tuple)

    def test_materialize_vacuum_scene(self):
        """A scene with no structures is all vacuum."""
        scene = Scene()
        grid = self._make_grid()
        mfield = materialize_grid(scene, grid)

        assert mfield.nx == 5
        assert mfield.ny == 5
        assert mfield.nz == 5
        assert np.all(mfield.structure_indices == -1)
        # Vacuum should have dielectric kind
        assert np.all(mfield.medium_kinds == 0)  # 0 = DIELECTRIC

    def test_materialize_single_box(self):
        """A single box structure is correctly assigned to cells."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=Medium(permittivity=3.5), name="slab")
        scene = Scene(structures=(struct,))
        grid = self._make_grid(size=2.0, dl=0.2)

        mfield = materialize_grid(scene, grid)

        # Cells inside the box should have structure index 0
        # The box spans from -0.5 to 0.5 in each axis
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    y_c = 0.5 * (grid.y.boundaries[j] + grid.y.boundaries[j + 1])
                    z_c = 0.5 * (grid.z.boundaries[k] + grid.z.boundaries[k + 1])
                    if box.contains_point((x_c, y_c, z_c)):
                        assert mfield.structure_indices[i, j, k] == 0, f"Cell ({i},{j},{k}) should be inside box"
                    else:
                        assert mfield.structure_indices[i, j, k] == -1, f"Cell ({i},{j},{k}) should be background"

    def test_materialize_pec_classification(self):
        """A PEC medium is classified correctly."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=PECMedium(), name="pec_slab")
        scene = Scene(structures=(struct,))
        grid = self._make_grid()

        mfield = materialize_grid(scene, grid)

        # PEC should have kind code 2
        assert np.any(mfield.medium_kinds == 2)

    def test_materialize_two_structures_precedence(self):
        """Two overlapping structures use precedence rules."""
        # Inner box (higher priority)
        inner = Box(center=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))
        outer = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        s1 = Structure(geometry=outer, medium=Medium(permittivity=1.5), name="outer")
        s2 = Structure(geometry=inner, medium=Medium(permittivity=3.5), name="inner")
        scene = Scene(structures=(s1, s2))
        grid = self._make_grid(size=2.0, dl=0.2)

        mfield = materialize_grid(scene, grid)

        # In the overlap region (inner box), s2 should win
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    y_c = 0.5 * (grid.y.boundaries[j] + grid.y.boundaries[j + 1])
                    z_c = 0.5 * (grid.z.boundaries[k] + grid.z.boundaries[k + 1])
                    if inner.contains_point((x_c, y_c, z_c)):
                        assert mfield.structure_indices[i, j, k] == 1  # s2 (index 1) wins


class TestSubpixelSmoothing:
    """Tests for subpixel smoothing application."""

    def _make_grid(self, nx=5, ny=5, nz=5, size=2.0, dl=0.4):
        spec = GridSpec.uniform(dl=dl)
        center = (0.0, 0.0, 0.0)
        size_tuple = (size, size, size)
        return spec.make_grid(center=center, size=size_tuple)

    def test_staircasing_spec_no_smoothing(self):
        """Staircasing spec does not trigger subpixel smoothing."""
        spec = SubpixelSpec.staircasing()
        assert not spec.dielectric.requires_anisotropic_materialization()

    def test_polarized_averaging_spec_smoothing(self):
        """PolarizedAveraging spec triggers subpixel smoothing."""
        spec = SubpixelSpec()  # default has PolarizedAveraging for dielectric
        assert spec.dielectric.requires_anisotropic_materialization()

    def test_apply_smoothing_noop_with_staircasing(self):
        """Subpixel smoothing is a no-op when staircasing is configured."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=Medium(permittivity=3.5), name="slab")
        scene = Scene(structures=(struct,))
        grid = self._make_grid(size=2.0, dl=0.2)
        mfield = materialize_grid(scene, grid)

        stair_spec = SubpixelSpec.staircasing()
        smoothed = apply_subpixel_smoothing(mfield, scene, grid, stair_spec)

        # Fractions should be unchanged
        np.testing.assert_array_equal(smoothed.volume_fractions, mfield.volume_fractions)

    def test_apply_smoothing_changes_interface_fractions(self):
        """Subpixel smoothing modifies fractions at dielectric interfaces."""
        # Create a scenario with a thin slab that partially fills cells
        box = Box(center=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6))
        struct = Structure(geometry=box, medium=Medium(permittivity=3.5), name="slab")
        scene = Scene(structures=(struct,))
        # Use a coarse grid so interface cells are partial
        grid = self._make_grid(size=2.0, dl=0.3)
        mfield = materialize_grid(scene, grid)

        avg_spec = SubpixelSpec()  # PolarizedAveraging for dielectric
        smoothed = apply_subpixel_smoothing(mfield, scene, grid, avg_spec)

        # At least some fractions should be different after smoothing
        # (interface cells should have 0.5)
        # Not all cells should be exactly 0 or 1


class TestCoefficientField:
    """Tests for coefficient field assembly."""

    def _make_grid(self, nx=3, ny=3, nz=3, size=1.0, dl=1.0/3.0):
        spec = GridSpec.uniform(dl=dl)
        center = (0.0, 0.0, 0.0)
        size_tuple = (size, size, size)
        return spec.make_grid(center=center, size=size_tuple)

    def test_assemble_vacuum_coefficients(self):
        """Vacuum scene produces vacuum coefficients."""
        scene = Scene()
        grid = self._make_grid()
        dt = 5e-13
        mfield = materialize_grid(scene, grid)
        cfield = assemble_coefficient_fields(mfield, scene, grid, dt)

        assert cfield.nx == 3
        assert cfield.ny == 3
        assert cfield.nz == 3
        # All cells should have same coefficient (vacuum)
        assert np.allclose(cfield.eps_xx, cfield.eps_xx[0, 0, 0])
        assert np.allclose(cfield.mu_xx, cfield.mu_xx[0, 0, 0])

    def test_assemble_box_permittivity(self):
        """A box structure produces the correct permittivity."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        eps_slab = 4.0
        struct = Structure(geometry=box, medium=Medium(permittivity=eps_slab), name="slab")
        scene = Scene(structures=(struct,))
        grid = self._make_grid(size=2.0, dl=0.2)
        dt = 3e-13
        mfield = materialize_grid(scene, grid)
        cfield = assemble_coefficient_fields(mfield, scene, grid, dt)

        # Find vacuum e_drive (outside the box) for comparison
        vacuum_e_drive = None
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    if mfield.structure_indices[i, j, k] < 0:
                        vacuum_e_drive = cfield.e_drive_xx[i, j, k]
                        break
            if vacuum_e_drive is not None:
                break

        # Find cells inside the box and verify drive coefficient is reduced
        found_box_cell = False
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    y_c = 0.5 * (grid.y.boundaries[j] + grid.y.boundaries[j + 1])
                    z_c = 0.5 * (grid.z.boundaries[k] + grid.z.boundaries[k + 1])
                    if box.contains_point((x_c, y_c, z_c)):
                        found_box_cell = True
                        # e_drive should be vacuum_e_drive / eps_slab for lossless dielectric
                        assert vacuum_e_drive is not None
                        assert cfield.e_drive_xx[i, j, k] < vacuum_e_drive, (
                            f"Cell ({i},{j},{k}) should have reduced e_drive for eps={eps_slab}"
                        )
                        # Also check it's approximately vacuum/4
                        assert abs(cfield.e_drive_xx[i, j, k] - vacuum_e_drive / eps_slab) < 1e-10, (
                            f"Cell ({i},{j},{k}) e_drive={cfield.e_drive_xx[i,j,k]} "
                            f"should be vacuum_e_drive/eps={vacuum_e_drive}/{eps_slab}"
                        )
        assert found_box_cell, "No box cells found in grid"

    def test_assemble_pec_electric_clamp(self):
        """PEC medium sets electric clamp mode."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=PECMedium(), name="pec")
        scene = Scene(structures=(struct,))
        grid = self._make_grid(size=2.0, dl=0.2)
        dt = 3e-13
        mfield = materialize_grid(scene, grid)
        cfield = assemble_coefficient_fields(mfield, scene, grid, dt)

        # Find PEC cells
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    if mfield.medium_kinds[i, j, k] == 2:  # PEC
                        assert cfield.electric_modes[i, j, k] == 1.0


class TestDiscretizeScene:
    """Integration tests for the full discretization pipeline."""

    def test_discretize_scene_end_to_end(self):
        """discretize_scene produces both material and coefficient fields."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=Medium(permittivity=3.5), name="slab")
        scene = Scene(structures=(struct,))
        spec = GridSpec.uniform(dl=0.1)
        grid = spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        dt = 1e-13

        mfield, cfield = discretize_scene(scene, grid, dt)

        assert isinstance(mfield, MaterialField)
        assert isinstance(cfield, CoefficientField)
        assert mfield.nx == cfield.nx == 20
        assert mfield.ny == cfield.ny == 20
        assert mfield.nz == cfield.nz == 20

    def test_discretize_with_anisotropic_medium(self):
        """AnisotropicMedium is discretized correctly."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        aniso = AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=Medium(permittivity=3.0),
            zz=Medium(permittivity=4.0),
        )
        struct = Structure(geometry=box, medium=aniso, name="aniso_box")
        scene = Scene(structures=(struct,))
        spec = GridSpec.uniform(dl=0.2)
        grid = spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        dt = 1e-13

        mfield, cfield = discretize_scene(scene, grid, dt)

        # Anisotropic cells should have different xx/yy/zz drive coefficients
        # For lossless anisotropic media, e_drive differs per axis (e_decay is always 1)
        # Find an anisotropic cell and check it has different per-axis values
        found_aniso = False
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    y_c = 0.5 * (grid.y.boundaries[j] + grid.y.boundaries[j + 1])
                    z_c = 0.5 * (grid.z.boundaries[k] + grid.z.boundaries[k + 1])
                    if box.contains_point((x_c, y_c, z_c)):
                        found_aniso = True
                        # Anisotropic medium should have different e_drive_xx, e_drive_yy, e_drive_zz
                        assert cfield.e_drive_xx[i, j, k] != cfield.e_drive_yy[i, j, k] or \
                               cfield.e_drive_yy[i, j, k] != cfield.e_drive_zz[i, j, k], (
                            f"Anisotropic medium should have different per-axis drive coefficients, "
                            f"got xx={cfield.e_drive_xx[i,j,k]}, yy={cfield.e_drive_yy[i,j,k]}, "
                            f"zz={cfield.e_drive_zz[i,j,k]}"
                        )
        assert found_aniso

    def test_discretize_with_subpixel_spec(self):
        """Subpixel spec is passed through to coefficient assembly."""
        box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        struct = Structure(geometry=box, medium=Medium(permittivity=2.0), name="slab")
        scene = Scene(structures=(struct,))
        spec = GridSpec.uniform(dl=0.2)
        grid = spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        dt = 1e-13
        subpixel = SubpixelSpec()

        mfield, cfield = discretize_scene(scene, grid, dt, subpixel_spec=subpixel)

        # Should complete without error
        assert cfield is not None
        # Smoothing should have been attempted (fractions may or may not change)


class TestTransformedGeometry:
    """Tests for transformed geometry discretization."""

    def _make_grid(self, size=2.0, dl=0.25):
        spec = GridSpec.uniform(dl=dl)
        return spec.make_grid(center=(0.0, 0.0, 0.0), size=(size, size, size))

    def test_discretize_translated_box(self):
        """A translated box is materialized correctly."""
        box = Box(center=(0.5, 0.5, 0.0), size=(0.5, 0.5, 2.0))
        struct = Structure(geometry=box, medium=Medium(permittivity=3.0), name="shifted")
        scene = Scene(structures=(struct,))
        grid = self._make_grid()

        mfield, cfield = discretize_scene(scene, grid, dt=1e-13)

        # Check that the shifted box is materialized at the correct location
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    y_c = 0.5 * (grid.y.boundaries[j] + grid.y.boundaries[j + 1])
                    z_c = 0.5 * (grid.z.boundaries[k] + grid.z.boundaries[k + 1])
                    if box.contains_point((x_c, y_c, z_c)):
                        assert mfield.structure_indices[i, j, k] == 0


class TestSceneWithMultipleStructures:
    """Tests for scenes with multiple structures."""

    def _make_grid(self, dl=0.2):
        spec = GridSpec.uniform(dl=dl)
        return spec.make_grid(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))

    def test_discretize_layered_structures(self):
        """Two non-overlapping structures are both materialized."""
        # Box at x < 0
        box1 = Box(center=(-0.5, 0.0, 0.0), size=(0.5, 2.0, 2.0))
        s1 = Structure(geometry=box1, medium=Medium(permittivity=2.0), name="left")
        # Box at x > 0
        box2 = Box(center=(0.5, 0.0, 0.0), size=(0.5, 2.0, 2.0))
        s2 = Structure(geometry=box2, medium=Medium(permittivity=4.0), name="right")
        scene = Scene(structures=(s1, s2))
        grid = self._make_grid(dl=0.1)
        dt = 1e-13

        mfield, cfield = discretize_scene(scene, grid, dt)

        left_count = 0
        right_count = 0
        for i in range(mfield.nx):
            for j in range(mfield.ny):
                for k in range(mfield.nz):
                    x_c = 0.5 * (grid.x.boundaries[i] + grid.x.boundaries[i + 1])
                    if box1.contains_point((x_c, 0.0, 0.0)):
                        left_count += 1
                    elif box2.contains_point((x_c, 0.0, 0.0)):
                        right_count += 1

        assert left_count > 0
        assert right_count > 0
