"""Tests for GaussianBeam and AstigmaticGaussianBeam source implementation."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    GaussianPulse,
    GridSpec,
    Simulation,
    UniformGrid,
)
from autofdtd.compiler import (
    compile_gaussian_beam,
    compile_astigmatic_gaussian_beam,
)
from autofdtd.ir import (
    AstigmaticGaussianBeamIR,
    GaussianBeamIR,
    simulation_to_ir,
)
from autofdtd.kernels import (
    inject_gaussian_beam,
    inject_astigmatic_gaussian_beam,
)
from autofdtd.runtime import (
    apply_gaussian_beam_sources,
    apply_astigmatic_gaussian_beam_sources,
    build_gaussian_beam_runtime,
    build_astigmatic_gaussian_beam_runtime,
)


def _resolved_grid():
    simulation = Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )
    return simulation.resolved_grid()


def test_gaussian_beam_requires_planar_geometry():
    """GaussianBeam must have exactly one zero-size dimension (planar source)."""
    # Should succeed - yz plane (x-size = 0)
    gb = GaussianBeam(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=0.0,
        direction="+",
    )
    assert gb.placement_kind == "sheet"
    assert gb.injection_axis == 0  # x-axis

    # Should fail - all dimensions non-zero
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        GaussianBeam(
            center=(0.0, 0.0, 0.0),
            size=(10.0, 10.0, 10.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            pol_angle=0.0,
            direction="+",
        )

    # Should fail - all dimensions zero
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        GaussianBeam(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            pol_angle=0.0,
            direction="+",
        )


def test_astigmatic_gaussian_beam_requires_planar_geometry():
    """AstigmaticGaussianBeam must have exactly one zero-size dimension."""
    # Should succeed - yz plane (x-size = 0)
    agb = AstigmaticGaussianBeam(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=0.0,
        direction="+",
    )
    assert agb.placement_kind == "sheet"
    assert agb.injection_axis == 0  # x-axis

    # Should fail - all dimensions non-zero
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        AstigmaticGaussianBeam(
            center=(0.0, 0.0, 0.0),
            size=(10.0, 10.0, 10.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            pol_angle=0.0,
            direction="+",
        )


def test_gaussian_beam_exposes_phase1_properties():
    """GaussianBeam exposes the expected Phase 1 properties."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=ContinuousWave(freq0=2.0, fwidth=0.5, amplitude=1.5, phase=0.1),
        pol_angle=0.3,
        direction="-",
        waist_radius=3.0,
        waist_distance=0.5,
        name="test_gb",
        interpolate=True,
        confine_to_bounds=False,
    )

    assert gb.name == "test_gb"
    assert gb.direction == "-"
    assert gb.injection_axis == 0  # x-axis
    assert gb.placement_kind == "sheet"
    assert gb.angle_theta == 0.0
    assert gb.angle_phi == 0.0
    assert gb.pol_angle == 0.3
    assert gb.waist_radius == 3.0
    assert gb.waist_distance == 0.5
    assert gb.support_bounds == ((5.0, 0.0, 0.0), (5.0, 10.0, 10.0))


def test_gaussian_beam_direction_vectors():
    """GaussianBeam computes direction and polarization vectors correctly."""
    # Normal incidence along z-axis
    gb_z = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=0.0,
        direction="+",
    )
    assert gb_z.injection_axis == 2
    assert gb_z._dir_vector == pytest.approx((0.0, 0.0, 1.0))

    # pol_angle = 0 means Ex polarization for z-incidence
    assert gb_z._pol_vector == pytest.approx((1.0, 0.0, 0.0))

    # pol_angle = pi/2 means Ey polarization for z-incidence
    gb_ey = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=np.pi / 2,
        direction="+",
    )
    assert gb_ey._pol_vector == pytest.approx((0.0, 1.0, 0.0))


def test_gaussian_beam_waist_validation():
    """GaussianBeam validates waist parameters correctly."""
    # Valid waist_radius
    gb = GaussianBeam(
        size=(0, 10, 10),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        waist_radius=5.0,
        direction="+",
    )
    assert gb.waist_radius == 5.0

    # Invalid waist_radius (zero or negative)
    with pytest.raises(ValueError, match="waist_radius must be positive"):
        GaussianBeam(
            size=(0, 10, 10),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            waist_radius=0.0,
            direction="+",
        )

    with pytest.raises(ValueError, match="waist_radius must be positive"):
        GaussianBeam(
            size=(0, 10, 10),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            waist_radius=-1.0,
            direction="+",
        )


def test_gaussian_beam_lowers_to_typed_ir():
    """GaussianBeam lowers to GaussianBeamIR correctly."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=2.0, fwidth=0.5, amplitude=1.5),
        pol_angle=0.3,
        direction="-",
        waist_radius=3.0,
        waist_distance=0.5,
        name="test_gb",
    )

    sim = Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1.0,
        sources=[gb],
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )

    sim_ir = simulation_to_ir(sim)
    assert len(sim_ir.sources) == 1

    src_ir = sim_ir.sources[0]
    assert isinstance(src_ir, GaussianBeamIR)
    assert src_ir.name == "test_gb"
    assert src_ir.placement_kind == "sheet"
    assert src_ir.injection_axis == 0
    assert src_ir.direction == "-"
    assert src_ir.pol_angle == 0.3
    assert src_ir.waist_radius == 3.0
    assert src_ir.waist_distance == 0.5


def test_astigmatic_gaussian_beam_lowers_to_typed_ir():
    """AstigmaticGaussianBeam lowers to AstigmaticGaussianBeamIR correctly."""
    agb = AstigmaticGaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=2.0, fwidth=0.5, amplitude=1.5),
        pol_angle=0.3,
        direction="-",
        waist_radius_x=3.0,
        waist_radius_y=2.0,
        waist_distance_x=0.5,
        waist_distance_y=0.3,
        name="test_agb",
    )

    sim = Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1.0,
        sources=[agb],
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )

    sim_ir = simulation_to_ir(sim)
    assert len(sim_ir.sources) == 1

    src_ir = sim_ir.sources[0]
    assert isinstance(src_ir, AstigmaticGaussianBeamIR)
    assert src_ir.name == "test_agb"
    assert src_ir.placement_kind == "sheet"
    assert src_ir.injection_axis == 0
    assert src_ir.direction == "-"
    assert src_ir.pol_angle == 0.3
    assert src_ir.waist_radius_x == 3.0
    assert src_ir.waist_radius_y == 2.0


def test_gaussian_beam_compilation():
    """GaussianBeam compiles correctly to placement data."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=2.0),
        direction="+",
        waist_radius=5.0,
        waist_distance=0.0,
    )

    compiled = compile_gaussian_beam(gb, grid=_resolved_grid())

    assert compiled.placement_kind == "sheet"
    assert compiled.injection_axis == 0
    assert compiled.direction == "+"
    # The resolved grid has 10 cells in each direction (dl=1.0, size=10)
    assert compiled.support_point_count == 25
    assert compiled.waist_radius == 5.0
    assert compiled.waist_distance == 0.0
    # Beam weights should be present and normalized
    assert len(compiled.beam_weights) == compiled.support_point_count
    # Center should have highest weight (Gaussian peak)
    # Note: beam center at (5,5) may not align with cell centers exactly,
    # so peak weight is close to but not exactly 1.0
    assert max(compiled.beam_weights) == pytest.approx(1.0, rel=0.1)


def test_astigmatic_gaussian_beam_compilation():
    """AstigmaticGaussianBeam compiles correctly to placement data."""
    agb = AstigmaticGaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=2.0),
        direction="+",
        waist_radius_x=5.0,
        waist_radius_y=3.0,
        waist_distance_x=0.0,
        waist_distance_y=0.0,
    )

    compiled = compile_astigmatic_gaussian_beam(agb, grid=_resolved_grid())

    assert compiled.placement_kind == "sheet"
    assert compiled.injection_axis == 0
    assert compiled.direction == "+"
    assert compiled.support_point_count == 25
    assert compiled.waist_radius_x == 5.0
    assert compiled.waist_radius_y == 3.0
    # Beam weights should be present
    assert len(compiled.beam_weights) == compiled.support_point_count


def test_gaussian_beam_injection_kernel():
    """GaussianBeam injection works correctly."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
        waist_radius=5.0,
        waist_distance=0.0,
    )

    compiled = compile_gaussian_beam(gb, grid=_resolved_grid())

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    # At time=0, the source should have some amplitude
    e2, m2 = inject_gaussian_beam(electric, magnetic, compiled, time=0.0, dt=0.1)

    # Something should be injected on the plane
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_gaussian_beam_runtime_integration():
    """GaussianBeam integrates with the runtime injection pipeline."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
        waist_radius=5.0,
        waist_distance=0.0,
    )

    compiled_source = build_gaussian_beam_runtime(gb, grid=_resolved_grid())

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    e2, m2 = apply_gaussian_beam_sources(
        electric,
        magnetic,
        (compiled_source,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_astigmatic_gaussian_beam_runtime_integration():
    """AstigmaticGaussianBeam integrates with the runtime injection pipeline."""
    agb = AstigmaticGaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
        waist_radius_x=5.0,
        waist_radius_y=3.0,
        waist_distance_x=0.0,
        waist_distance_y=0.0,
    )

    compiled_source = build_astigmatic_gaussian_beam_runtime(agb, grid=_resolved_grid())

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    e2, m2 = apply_astigmatic_gaussian_beam_sources(
        electric,
        magnetic,
        (compiled_source,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_gaussian_beam_with_angles():
    """GaussianBeam with non-zero angles."""
    gb = GaussianBeam(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        angle_theta=0.5,
        angle_phi=0.3,
        pol_angle=0.2,
        direction="+",
        waist_radius=5.0,
    )

    assert gb.injection_axis == 0
    assert gb.is_fixed_angle is False

    # Direction vector should reflect the angles
    dir_vec = gb._dir_vector
    # For x-injection with theta=0.5, the k should have significant x-component
    assert abs(dir_vec[0]) > 0.5

    compiled = compile_gaussian_beam(gb, grid=_resolved_grid())
    assert compiled.angle_theta == 0.5
    assert compiled.angle_phi == 0.3
    assert compiled.pol_angle == 0.2


# Import GaussianBeam and AstigmaticGaussianBeam at module level for test access
import numpy as np  # noqa: E402
from autofdtd.sources import AstigmaticGaussianBeam, GaussianBeam  # noqa: E402
