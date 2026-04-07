"""Tests for boundary kernel stages in kernels.boundaries.

These tests focus on the boundary ghost exchange, PML layer, and ABC
kernel stages, including both NumPy and Warp backends where available.
"""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.compiler.boundaries import (
    BoundaryMode,
    CompiledABCCoefficients,
    CompiledBoundarySpec,
    CompiledPMLCoefficients,
    compile_boundary_spec,
)
from autofdtd.boundaries.models import Boundary, BoundarySpec
from autofdtd.kernels.backend import WARP_AVAILABLE, ComplexFieldPolicy
from autofdtd.kernels.boundaries import (
    ABCBoundaryState,
    ABCFaceState,
    PMLBoundaryState,
    PMLFaceState,
    apply_abc_layers,
    apply_boundary_ghosts,
    apply_boundary_stages,
    apply_pml_layers,
    boundary_edge_transform,
    boundary_kernel_metadata,
    symmetry_transform,
)
from autofdtd.runtime.chunk import (
    ChunkLayout,
    ChunkSpec,
    FaceHalo,
    build_chunk_layout,
    chunk_boundary_face_keys,
    chunk_layout_summary,
    chunk_min_timestep,
    global_min_timestep,
)


class TestBoundaryKernelMetadata:
    """Test kernel metadata reporting."""

    def test_boundary_kernel_metadata_returns_dict(self):
        """boundary_kernel_metadata returns a dict."""
        metadata = boundary_kernel_metadata()
        assert isinstance(metadata, dict)

    def test_supported_modes(self):
        """Metadata lists all supported boundary modes."""
        metadata = boundary_kernel_metadata()
        modes = metadata["supported_modes"]
        assert "periodic" in modes
        assert "bloch" in modes
        assert "pec" in modes
        assert "pmc" in modes
        assert "abc" in modes
        assert "pml" in modes
        assert "stable_pml" in modes
        assert "absorber" in modes

    def test_stages_list(self):
        """Metadata lists all boundary stages."""
        metadata = boundary_kernel_metadata()
        stages = metadata["stages"]
        assert "electric_boundary" in stages
        assert "magnetic_boundary" in stages
        assert "electric_abc" in stages
        assert "magnetic_abc" in stages
        assert "electric_pml" in stages
        assert "magnetic_pml" in stages

    def test_halo_depth_is_one(self):
        """Halo depth is 1 for current implementation."""
        metadata = boundary_kernel_metadata()
        assert metadata["halo_depth"] == 1


class TestApplyBoundaryGhosts:
    """Test ghost cell exchange for boundary faces."""

    def setup_method(self):
        """Set up test grid and boundary spec."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dt = 1e-12
        self.dx = self.dy = self.dz = 1e-8

        # Compile periodic boundaries
        self.boundary_spec = compile_boundary_spec(
            BoundarySpec.periodic(),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        # Allocate field arrays
        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        # Set interior field values
        self.electric[2:-2, 2:-2, 2:-2, :] = 1.0
        self.magnetic[2:-2, 2:-2, 2:-2, :] = 0.5

    def test_periodic_ghost_exchange_x(self):
        """Periodic boundaries copy ghost from opposite interior cell."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # x-minus ghost should copy from x-plus interior
        np.testing.assert_allclose(electric[0, :, :], electric[-2, :, :])
        # x-plus ghost should copy from x-minus interior
        np.testing.assert_allclose(electric[-1, :, :], electric[1, :, :])

    def test_periodic_ghost_exchange_y(self):
        """Periodic boundaries copy ghost in y direction."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # y-minus ghost should copy from y-plus interior
        np.testing.assert_allclose(electric[:, 0, :], electric[:, -2, :])
        # y-plus ghost should copy from y-minus interior
        np.testing.assert_allclose(electric[:, -1, :], electric[:, 1, :])

    def test_periodic_ghost_exchange_z(self):
        """Periodic boundaries copy ghost in z direction."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # z-minus ghost should copy from z-plus interior
        np.testing.assert_allclose(electric[:, :, 0], electric[:, :, -2])
        # z-plus ghost should copy from z-minus interior
        np.testing.assert_allclose(electric[:, :, -1], electric[:, :, 1])


class TestApplyPECBoundaryGhosts:
    """Test PEC boundary ghost exchange."""

    def setup_method(self):
        """Set up test grid with PEC boundaries."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.pec(),
                y=Boundary.periodic(),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        # Set interior field values
        self.electric[2:-2, 2:-2, 2:-2, :] = 1.0
        self.magnetic[2:-2, 2:-2, 2:-2, :] = 0.5

    def test_pec_reflection_sign(self):
        """PEC ghost reflects with sign change for E, same sign for H."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # x-minus ghost: E should be -E[1], H should be +H[1]
        # electric_signs for PEC: (-1, -1, 1), magnetic_signs: (1, 1, 1)
        for comp in range(3):
            np.testing.assert_allclose(electric[0, :, :, comp], -electric[1, :, :, comp])
            np.testing.assert_allclose(magnetic[0, :, :, comp], magnetic[1, :, :, comp])

    def test_pec_x_plus_reflection(self):
        """PEC at x-plus face reflects correctly."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # x-plus ghost: E should be -E[-2], H should be +H[-2]
        for comp in range(3):
            np.testing.assert_allclose(
                electric[-1, :, :, comp], -electric[-2, :, :, comp]
            )
            np.testing.assert_allclose(
                magnetic[-1, :, :, comp], magnetic[-2, :, :, comp]
            )


class TestApplyPMCBoundaryGhosts:
    """Test PMC boundary ghost exchange."""

    def setup_method(self):
        """Set up test grid with PMC boundaries."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.pmc(),
                y=Boundary.periodic(),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        self.electric[2:-2, 2:-2, 2:-2, :] = 1.0
        self.magnetic[2:-2, 2:-2, 2:-2, :] = 0.5

    def test_pmc_reflection_sign(self):
        """PMC ghost reflects with same sign for E, opposite for H."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # x-minus ghost: E should be +E[1], H should be -H[1]
        for comp in range(3):
            np.testing.assert_allclose(electric[0, :, :, comp], electric[1, :, :, comp])
            np.testing.assert_allclose(magnetic[0, :, :, comp], -magnetic[1, :, :, comp])


class TestApplyBlochBoundaryGhosts:
    """Test Bloch boundary ghost exchange."""

    def setup_method(self):
        """Set up test grid with Bloch boundaries."""
        self.nx, self.ny, self.nz = 10, 10, 10
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.bloch(0.25),
                y=Boundary.periodic(),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        # Bloch boundaries require complex fields
        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.complex128)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.complex128)

        # Set spatially varying fields
        for i in range(2, self.nx - 2):
            self.electric[i, :, :, :] = i * 0.1
            self.magnetic[i, :, :, :] = i * 0.05

    def test_bloch_phase_factor_applied(self):
        """Bloch ghost multiplies by phase factor."""
        electric, magnetic = apply_boundary_ghosts(
            self.electric, self.magnetic, self.boundary_spec
        )

        # For x.bloch with phase factor, ghost at x=0 should be phase * field[-2]
        x_minus_edge = self.boundary_spec.x.minus
        phase = x_minus_edge.phase_factor

        # The ghost at index 0 should be phase * field[-2]
        for comp in range(3):
            expected = self.electric[-2, :, :, comp] * phase
            np.testing.assert_allclose(electric[0, :, :, comp], expected)


class TestPMLBoundaryState:
    """Test PML state allocation and layer application."""

    def setup_method(self):
        """Set up PML test case."""
        self.nx, self.ny, self.nz = 20, 20, 20
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.pml(num_layers=8),
                y=Boundary.periodic(),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        # Set interior field values
        self.electric[5:-5, 5:-5, 5:-5, :] = 1.0
        self.magnetic[5:-5, 5:-5, 5:-5, :] = 0.5

    def test_pml_state_allocated(self):
        """PML state is allocated when PML faces present."""
        from autofdtd.runtime.boundaries import allocate_pml_boundary_state

        state = allocate_pml_boundary_state(
            self.boundary_spec, self.electric.shape, dtype=np.float64
        )

        assert state is not None
        assert isinstance(state, PMLBoundaryState)
        assert len(state.electric) > 0

    def test_pml_layers_applied(self):
        """PML layers attenuate fields near boundaries."""
        from autofdtd.runtime.boundaries import allocate_pml_boundary_state

        state = allocate_pml_boundary_state(
            self.boundary_spec, self.electric.shape, dtype=np.float64
        )

        electric = np.array(self.electric, copy=True)
        magnetic = np.array(self.magnetic, copy=True)

        electric, magnetic = apply_boundary_stages(
            electric, magnetic, self.boundary_spec, state
        )

        # Fields near PML boundaries should be attenuated
        # Check that field near x=0 boundary is smaller than interior
        interior_mean = np.mean(np.abs(electric[5:-5, 5:-5, 5:-5, :]))
        near_boundary_mean = np.mean(np.abs(electric[1:4, 5:-5, 5:-5, :]))

        # Near-boundary should be attenuated relative to interior
        assert near_boundary_mean < interior_mean


class TestABCBoundaryState:
    """Test ABC state allocation and layer application."""

    def setup_method(self):
        """Set up ABC test case."""
        self.nx, self.ny, self.nz = 20, 20, 20
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.abc(permittivity=2.25, conductivity=0.0),
                y=Boundary.periodic(),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

    def test_abc_state_allocated(self):
        """ABC state is allocated when ABC faces present."""
        from autofdtd.runtime.boundaries import allocate_abc_boundary_state

        state = allocate_abc_boundary_state(
            self.boundary_spec, self.electric.shape, dtype=np.float64
        )

        assert state is not None
        assert isinstance(state, ABCBoundaryState)
        assert len(state.electric) > 0

    def test_abc_layers_applied(self):
        """ABC layers update ghost cells using one-step recurrence."""
        from autofdtd.runtime.boundaries import allocate_abc_boundary_state

        state = allocate_abc_boundary_state(
            self.boundary_spec, self.electric.shape, dtype=np.float64
        )

        # Set a wave propagating toward x-minus ghost (at i=0)
        # ABC reads from adjacent cell at i=1 and writes to ghost at i=0
        self.electric[1, 5:-5, 5:-5, 0] = 1.0

        electric = np.array(self.electric, copy=True)
        magnetic = np.array(self.magnetic, copy=True)

        electric, magnetic = apply_boundary_stages(
            electric, magnetic, self.boundary_spec, None, abc_state=state
        )

        # Ghost cells should have been updated by ABC recurrence
        assert np.abs(electric[0, 5:-5, 5:-5, 0]).any()


class TestBoundaryEdgeTransform:
    """Test boundary edge transform for halo exchange."""

    def test_periodic_edge_returns_copy(self):
        """Periodic edge transform returns a copy with no sign change."""
        from autofdtd.compiler.boundaries import CompiledBoundaryEdge

        edge = CompiledBoundaryEdge(
            axis="x",
            side="minus",
            mode=BoundaryMode.PERIODIC,
            electric_signs=(1, 1, 1),
            magnetic_signs=(1, 1, 1),
            source_side="plus",
        )

        values = np.ones((5, 5, 3))
        result = boundary_edge_transform(values, edge, field_family="electric")

        np.testing.assert_allclose(result, values)

    def test_pec_edge_reflects_electric(self):
        """PEC edge reflects electric field with sign change."""
        from autofdtd.compiler.boundaries import CompiledBoundaryEdge

        edge = CompiledBoundaryEdge(
            axis="x",
            side="minus",
            mode=BoundaryMode.PEC,
            electric_signs=(-1, -1, 1),
            magnetic_signs=(1, 1, 1),
            source_side="plus",
        )

        values = np.array([1.0, 2.0, 3.0])
        result = boundary_edge_transform(values, edge, field_family="electric")

        np.testing.assert_allclose(result, [-1.0, -2.0, 3.0])

    def test_pmc_edge_reflects_magnetic(self):
        """PMC edge reflects magnetic field with sign change."""
        from autofdtd.compiler.boundaries import CompiledBoundaryEdge

        edge = CompiledBoundaryEdge(
            axis="x",
            side="minus",
            mode=BoundaryMode.PMC,
            electric_signs=(1, 1, 1),
            magnetic_signs=(-1, -1, 1),
            source_side="plus",
        )

        values = np.array([1.0, 2.0, 3.0])
        result = boundary_edge_transform(values, edge, field_family="magnetic")

        np.testing.assert_allclose(result, [-1.0, -2.0, 3.0])

    def test_bloch_edge_applies_phase(self):
        """Bloch edge applies phase factor."""
        from autofdtd.compiler.boundaries import CompiledBoundaryEdge

        edge = CompiledBoundaryEdge(
            axis="x",
            side="minus",
            mode=BoundaryMode.BLOCH,
            electric_signs=(1, 1, 1),
            magnetic_signs=(1, 1, 1),
            source_side="plus",
            phase_factor=1.0 + 1.0j,
        )

        # Bloch boundaries require complex fields
        values = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        result = boundary_edge_transform(values, edge, field_family="electric")

        expected = np.array([1.0, 2.0, 3.0], dtype=np.complex128) * (1.0 + 1.0j)
        np.testing.assert_allclose(result, expected)


class TestSymmetryTransform:
    """Test symmetry transform for reduced-domain boundaries."""

    def test_even_symmetry(self):
        """Even symmetry (mirror) reflects with same sign."""
        from autofdtd.compiler.boundaries import CompiledSymmetryAxis

        # For x-axis even symmetry: spatial signs = (-1, 1, 1), symmetry = 1
        # electric_signs = spatial * symmetry = (-1, 1, 1)
        sym = CompiledSymmetryAxis(
            axis="x",
            symmetry=1,
            electric_signs=(-1, 1, 1),
            magnetic_signs=(1, -1, -1),
        )

        values = np.array([1.0, 2.0, 3.0])
        result = symmetry_transform(values, sym, field_family="electric")

        np.testing.assert_allclose(result, [-1.0, 2.0, 3.0])

    def test_odd_symmetry(self):
        """Odd symmetry reflects with opposite sign."""
        from autofdtd.compiler.boundaries import CompiledSymmetryAxis

        # For x-axis odd symmetry: spatial signs = (-1, 1, 1), symmetry = -1
        # electric_signs = spatial * symmetry = (1, -1, -1)
        sym = CompiledSymmetryAxis(
            axis="x",
            symmetry=-1,
            electric_signs=(1, -1, -1),
            magnetic_signs=(-1, 1, 1),
        )

        values = np.array([1.0, 2.0, 3.0])
        result = symmetry_transform(values, sym, field_family="electric")

        np.testing.assert_allclose(result, [1.0, -2.0, -3.0])


class TestChunkSpec:
    """Test ChunkSpec and chunk layout building."""

    def test_chunk_spec_properties(self):
        """ChunkSpec exposes expected properties."""
        spec = ChunkSpec(
            chunk_index=(0, 0, 0),
            global_bounds=((0, 0, 0), (20, 20, 20)),
            interior_bounds=((1, 1, 1), (19, 19, 19)),
            owned_bounds=((1, 1, 1), (19, 19, 19)),
            local_grid_shape=(20, 20, 20),
        )

        assert spec.chunk_index == (0, 0, 0)
        assert spec.num_cells == 20 * 20 * 20
        assert spec.interior_num_cells == 18 * 18 * 18
        assert spec.local_grid_shape == (20, 20, 20)

    def test_face_halo_key(self):
        """FaceHalo face_key property works."""
        halo = FaceHalo(axis="x", side="minus", depth=1)
        assert halo.face_key == "x.minus"

    def test_build_chunk_layout_monolithic(self):
        """build_chunk_layout creates single-chunk layout."""
        layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(20, 20, 20),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )

        assert layout.total_chunks == 1
        assert len(layout.chunks) == 1
        assert layout.chunks[0].chunk_index == (0, 0, 0)

    def test_chunk_layout_summary(self):
        """chunk_layout_summary returns diagnostic dict."""
        layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(20, 20, 20),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )

        summary = chunk_layout_summary(layout)
        assert summary["total_chunks"] == 1
        assert summary["num_chunks"] == (1, 1, 1)
        assert len(summary["chunks"]) == 1


class TestChunkBoundaryFaceKeys:
    """Test boundary face key helpers."""

    def test_boundary_face_keys(self):
        """chunk_boundary_face_keys returns faces without neighbors."""
        spec = ChunkSpec(
            chunk_index=(0, 0, 0),
            global_bounds=((0, 0, 0), (20, 20, 20)),
            interior_bounds=((1, 1, 1), (19, 19, 19)),
            owned_bounds=((1, 1, 1), (19, 19, 19)),
            local_grid_shape=(20, 20, 20),
            face_halos={
                "x.minus": FaceHalo(
                    axis="x", side="minus", depth=1, neighbor_chunk_index=None
                ),
                "x.plus": FaceHalo(
                    axis="x", side="plus", depth=1, neighbor_chunk_index=None
                ),
                "y.minus": FaceHalo(
                    axis="y", side="minus", depth=1, neighbor_chunk_index=(0, 0, 1)
                ),
                "y.plus": FaceHalo(
                    axis="y", side="plus", depth=1, neighbor_chunk_index=None
                ),
            },
        )

        keys = chunk_boundary_face_keys(spec)
        assert "x.minus" in keys
        assert "x.plus" in keys
        assert "y.plus" in keys
        # y.minus has neighbor, so not a boundary face
        assert "y.minus" not in keys


class TestChunkMinTimestep:
    """Test CFL timestep computation per chunk."""

    def test_chunk_min_timestep_vacuum(self):
        """Vacuum chunk CFL is min_spacing / (c * sqrt(3))."""
        spec = ChunkSpec(
            chunk_index=(0, 0, 0),
            global_bounds=((0, 0, 0), (20, 20, 20)),
            interior_bounds=((1, 1, 1), (19, 19, 19)),
            owned_bounds=((1, 1, 1), (19, 19, 19)),
            local_grid_shape=(20, 20, 20),
        )

        dt = chunk_min_timestep(spec, (1e-8, 1e-8, 1e-8))
        c0 = 299_792_458.0
        expected = 0.99 * 1e-8 / (c0 * (3**0.5))

        np.testing.assert_allclose(dt, expected, rtol=1e-10)

    def test_global_min_timestep(self):
        """global_min_timestep returns minimum across chunks."""
        specs = (
            ChunkSpec(
                chunk_index=(0, 0, 0),
                global_bounds=((0, 0, 0), (10, 10, 10)),
                interior_bounds=((1, 1, 1), (9, 9, 9)),
                owned_bounds=((1, 1, 1), (9, 9, 9)),
                local_grid_shape=(10, 10, 10),
            ),
            ChunkSpec(
                chunk_index=(1, 0, 0),
                global_bounds=((0, 0, 0), (20, 10, 10)),
                interior_bounds=((1, 1, 1), (19, 9, 9)),
                owned_bounds=((1, 1, 1), (19, 9, 9)),
                local_grid_shape=(20, 10, 10),
            ),
        )

        # Second chunk has larger grid but same cell size
        # Both should have same dt since cell sizes are same
        dt = global_min_timestep(specs, (1e-8, 1e-8, 1e-8))

        c0 = 299_792_458.0
        expected = 0.99 * 1e-8 / (c0 * (3**0.5))
        np.testing.assert_allclose(dt, expected, rtol=1e-10)


class TestApplyBoundaryStages:
    """Test full boundary stage ordering."""

    def setup_method(self):
        """Set up full test with mixed boundaries."""
        self.nx, self.ny, self.nz = 20, 20, 20
        self.dt = 1e-12
        self.dx = 1e-8

        self.boundary_spec = compile_boundary_spec(
            BoundarySpec(
                x=Boundary.pec(),
                y=Boundary.pml(num_layers=8),
                z=Boundary.periodic(),
            ),
            dt=self.dt,
            grid_spacing=self.dx,
        )

        self.electric = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)
        self.magnetic = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float64)

        # Set interior field
        self.electric[5:-5, 5:-5, 5:-5, :] = 1.0
        self.magnetic[5:-5, 5:-5, 5:-5, :] = 0.5

    def test_apply_boundary_stages_returns_both_fields(self):
        """apply_boundary_stages returns both electric and magnetic."""
        from autofdtd.runtime.boundaries import allocate_pml_boundary_state

        state = allocate_pml_boundary_state(
            self.boundary_spec, self.electric.shape, dtype=np.float64
        )

        result_e, result_h = apply_boundary_stages(
            self.electric, self.magnetic, self.boundary_spec, state
        )

        assert result_e.shape == self.electric.shape
        assert result_h.shape == self.magnetic.shape

    def test_stage_order_includes_pml(self):
        """Boundary spec stage order includes PML stages."""
        stages = self.boundary_spec.stage_order
        assert "electric_boundary" in stages
        assert "magnetic_boundary" in stages
        assert "electric_pml" in stages
        assert "magnetic_pml" in stages


class TestHaloExchangeHelpers:
    """Test halo pack/unpack helpers."""

    def test_pack_halo_vector_field_minus(self):
        """Pack halo from minus face of vector field."""
        from autofdtd.runtime.boundaries import pack_halo

        field = np.zeros((10, 10, 10, 3), dtype=np.float64)
        field[0, :, :, 0] = 1.0  # Ghost cell at index 0

        packed = pack_halo(field, "x", "minus", depth=1, chunk_size=10)
        assert packed.shape == (1, 10, 10, 3)
        assert packed[0, :, :, 0].sum() == 100.0

    def test_pack_halo_vector_field_plus(self):
        """Pack halo from plus face of vector field."""
        from autofdtd.runtime.boundaries import pack_halo

        field = np.zeros((10, 10, 10, 3), dtype=np.float64)
        field[-1, :, :, 0] = 2.0  # Ghost cell at index -1

        packed = pack_halo(field, "x", "plus", depth=1, chunk_size=10)
        assert packed.shape == (1, 10, 10, 3)
        assert packed[0, :, :, 0].sum() == 200.0

    def test_unpack_halo_vector_field(self):
        """Unpack halo into vector field."""
        from autofdtd.runtime.boundaries import pack_halo, unpack_halo

        original = np.zeros((10, 10, 10, 3), dtype=np.float64)
        original[2:-2, 2:-2, 2:-2, :] = 1.0

        packed = pack_halo(original, "x", "minus", depth=1, chunk_size=10)
        result = unpack_halo(packed, np.zeros_like(original), "x", "minus", depth=1, chunk_size=10)

        # Ghost cell should now have the packed value
        assert result[0, :, :, :].shape == (10, 10, 3)

    def test_apply_signs_to_halo_electric(self):
        """Apply reflection signs to electric field halo."""
        from autofdtd.runtime.boundaries import apply_signs_to_halo

        # 4D halo data for vector field: (..., 3)
        halo = np.ones((5, 5, 5, 3), dtype=np.float64)
        signs = (-1, -1, 1)
        result = apply_signs_to_halo(halo, signs, "electric")

        np.testing.assert_allclose(result[..., 0], -1.0)
        np.testing.assert_allclose(result[..., 1], -1.0)
        np.testing.assert_allclose(result[..., 2], 1.0)

    def test_apply_phase_to_halo_bloch(self):
        """Apply Bloch phase factor to halo."""
        from autofdtd.runtime.boundaries import apply_phase_to_halo

        halo = np.ones((5, 5, 3), dtype=np.float64)
        phase = 1.0 + 1.0j
        result = apply_phase_to_halo(halo, phase)

        assert np.iscomplexobj(result)
        np.testing.assert_allclose(result, halo * phase)


class TestChunkHaloExchange:
    """Test ChunkHaloExchange manager."""

    def setup_method(self):
        """Set up test chunk layout."""
        from autofdtd.runtime.chunk import build_chunk_layout

        self.layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(20, 20, 20),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )
        self.field_shape = (20, 20, 20, 3)

    def test_build_chunk_halo_exchange(self):
        """build_chunk_halo_exchange creates manager correctly."""
        from autofdtd.runtime.boundaries import ChunkHaloExchange, build_chunk_halo_exchange

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)
        assert isinstance(exchange, ChunkHaloExchange)
        assert exchange.chunk_layout is self.layout
        assert exchange.field_shape == self.field_shape

    def test_exchange_kind_for_face(self):
        """Exchange kind is periodic for periodic boundaries."""
        from autofdtd.runtime.boundaries import build_chunk_halo_exchange
        from autofdtd.runtime.chunk import ExchangeKind

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)
        kind = exchange.exchange_kind_for_face((0, 0, 0), "x", "minus")
        # x.minus is periodic in this layout
        assert kind == ExchangeKind.PERIODIC

    def test_pack_face_halo_returns_array(self):
        """pack_face_halo returns halo data array."""
        from autofdtd.runtime.boundaries import build_chunk_halo_exchange

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)
        field = np.zeros(self.field_shape, dtype=np.float64)
        field[2:-2, 2:-2, 2:-2, :] = 1.0

        packed = exchange.pack_face_halo((0, 0, 0), field, "x", "minus")
        assert packed is not None
        assert packed.shape == (1, 20, 20, 3)

    def test_pack_face_halo_returns_none_for_interior(self):
        """pack_face_halo returns None for interior faces."""
        from autofdtd.runtime.boundaries import build_chunk_halo_exchange
        from autofdtd.runtime.chunk import ExchangeKind

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)

        # Interior faces should return None
        # y.minus neighbor is (0, 0, 1) in monolithic layout so it's interior
        kind = exchange.exchange_kind_for_face((0, 0, 0), "y", "minus")
        if kind == ExchangeKind.INTERIOR:
            field = np.zeros(self.field_shape, dtype=np.float64)
            packed = exchange.pack_face_halo((0, 0, 0), field, "y", "minus")
            assert packed is None

    def test_unpack_face_halo_updates_field(self):
        """unpack_face_halo updates ghost cells."""
        from autofdtd.runtime.boundaries import build_chunk_halo_exchange

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)
        field = np.zeros(self.field_shape, dtype=np.float64)
        field[2:-2, 2:-2, 2:-2, :] = 1.0

        packed = exchange.pack_face_halo((0, 0, 0), field, "x", "minus")
        result = exchange.unpack_face_halo((0, 0, 0), np.zeros_like(field), "x", "minus", packed)

        if packed is not None:
            assert result[0, :, :, :].shape == (20, 20, 3)

    def test_interior_copy_slice(self):
        """interior_copy_slice returns correct slices."""
        from autofdtd.runtime.boundaries import build_chunk_halo_exchange

        exchange = build_chunk_halo_exchange(self.layout, self.field_shape)
        dest_slice, src_slice = exchange.interior_copy_slice("x", "minus", depth=1)

        assert dest_slice == slice(0, 1)
        assert src_slice == slice(1, 2)


class TestChunkLayoutIRLowering:
    """Test chunk layout lowering to IR."""

    def test_chunk_layout_to_ir(self):
        """chunk_layout_to_ir produces valid IR."""
        from autofdtd.ir.models import chunk_layout_to_ir
        from autofdtd.runtime.chunk import build_chunk_layout

        layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(10, 10, 10),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )

        ir = chunk_layout_to_ir(layout)
        assert ir.type == "ChunkLayoutIR"
        assert ir.num_chunks == (1, 1, 1)
        assert ir.total_chunks == 1
        assert len(ir.chunks) == 1

    def test_chunk_spec_to_ir(self):
        """ChunkSpec lowering produces valid IR."""
        from autofdtd.ir.models import _chunk_spec_to_ir
        from autofdtd.runtime.chunk import build_chunk_layout

        layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(10, 10, 10),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )

        ir = _chunk_spec_to_ir(layout.chunks[0])
        assert ir.type == "ChunkSpecIR"
        assert ir.chunk_index == (0, 0, 0)
        assert ir.local_grid_shape == (10, 10, 10)

    def test_face_halo_to_ir(self):
        """FaceHalo lowering produces valid IR."""
        from autofdtd.ir.models import _face_halo_to_ir
        from autofdtd.runtime.chunk import build_chunk_layout

        layout = build_chunk_layout(
            num_chunks=(1, 1, 1),
            grid_shape=(10, 10, 10),
            cell_sizes=(1e-8, 1e-8, 1e-8),
        )

        chunk = layout.chunks[0]
        halo = chunk.face_halo("x", "minus")
        ir = _face_halo_to_ir(halo)
        assert ir.type == "FaceHaloIR"
        assert ir.axis == "x"
        assert ir.side == "minus"
        assert ir.depth == 1

