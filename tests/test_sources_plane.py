"""Tests for PlaneWave source implementation."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    GridSpec,
    Simulation,
    UniformGrid,
)
from autofdtd.compiler import (
    compile_plane_wave,
)
from autofdtd.ir import (
    PlaneWaveIR,
    simulation_to_ir,
)
from autofdtd.kernels import (
    inject_plane_wave,
)
from autofdtd.runtime import (
    apply_plane_wave_sources,
    build_plane_wave_runtime,
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


def test_plane_wave_requires_planar_geometry():
    """PlaneWave must have exactly one zero-size dimension (planar source)."""
    # Should succeed - yz plane (x-size = 0)
    pw = PlaneWave(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=0.0,
        direction="+",
    )
    assert pw.placement_kind == "sheet"
    assert pw.injection_axis == 0  # x-axis

    # Should fail - all dimensions non-zero
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        PlaneWave(
            center=(0.0, 0.0, 0.0),
            size=(10.0, 10.0, 10.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            pol_angle=0.0,
            direction="+",
        )

    # Should fail - all dimensions zero
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        PlaneWave(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            pol_angle=0.0,
            direction="+",
        )


def test_plane_wave_exposes_phase1_properties():
    """PlaneWave exposes the expected Phase 1 properties."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=ContinuousWave(freq0=2.0, fwidth=0.5, amplitude=1.5, phase=0.1),
        pol_angle=0.3,
        direction="-",
        name="test_plane_wave",
        interpolate=True,
        confine_to_bounds=False,
    )

    assert pw.name == "test_plane_wave"
    assert pw.direction == "-"
    assert pw.injection_axis == 0  # x-axis
    assert pw.placement_kind == "sheet"
    assert pw.angle_theta == 0.0
    assert pw.angle_phi == 0.0
    assert pw.pol_angle == 0.3
    assert pw.support_bounds == ((5.0, 0.0, 0.0), (5.0, 10.0, 10.0))


def test_plane_wave_direction_vectors():
    """PlaneWave computes direction and polarization vectors correctly."""
    # Normal incidence along z-axis
    pw_z = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=0.0,
        direction="+",
    )
    assert pw_z.injection_axis == 2
    assert pw_z._dir_vector == pytest.approx((0.0, 0.0, 1.0))

    # pol_angle = 0 means Ex polarization for z-incidence (Tidy3D convention)
    assert pw_z._pol_vector == pytest.approx((1.0, 0.0, 0.0))

    # pol_angle = pi/2 means Ey polarization for z-incidence
    pw_ey = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        pol_angle=np.pi / 2,
        direction="+",
    )
    assert pw_ey._pol_vector == pytest.approx((0.0, 1.0, 0.0))


def test_plane_wave_angular_spec():
    """PlaneWave with different angular specs."""
    # Default is FixedInPlaneKSpec
    pw_default = PlaneWave(
        size=(0, 10, 10),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        direction="+",
    )
    assert isinstance(pw_default.angular_spec, FixedInPlaneKSpec)
    assert pw_default.is_fixed_angle is False

    # Explicit FixedInPlaneKSpec
    pw_fixed_k = PlaneWave(
        size=(0, 10, 10),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        angular_spec=FixedInPlaneKSpec(),
        direction="+",
    )
    assert isinstance(pw_fixed_k.angular_spec, FixedInPlaneKSpec)

    # FixedAngleSpec at non-zero angle
    pw_fixed_angle = PlaneWave(
        size=(0, 10, 10),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        angular_spec=FixedAngleSpec(),
        angle_theta=0.5,
        direction="+",
    )
    assert isinstance(pw_fixed_angle.angular_spec, FixedAngleSpec)
    assert pw_fixed_angle.is_fixed_angle is True


def test_plane_wave_lowers_to_typed_ir():
    """PlaneWave lowers to PlaneWaveIR correctly."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=2.0, fwidth=0.5, amplitude=1.5),
        pol_angle=0.3,
        direction="-",
        name="test_pw",
    )

    sim = Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1.0,
        sources=[pw],
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )

    sim_ir = simulation_to_ir(sim)
    assert len(sim_ir.sources) == 1

    src_ir = sim_ir.sources[0]
    assert isinstance(src_ir, PlaneWaveIR)
    assert src_ir.name == "test_pw"
    assert src_ir.placement_kind == "sheet"
    assert src_ir.injection_axis == 0
    assert src_ir.direction == "-"
    assert src_ir.pol_angle == 0.3
    assert src_ir.angular_spec.component_type == "FixedInPlaneKSpec"
    # For direction="-" at theta=0, dir_vector is (-1, 0, 0) along injection axis
    assert src_ir.dir_vector == pytest.approx((-1.0, 0.0, 0.0))


def test_plane_wave_compilation():
    """PlaneWave compiles correctly to placement data."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=CustomSourceTime(
            freq0=1.0,
            fwidth=1.0,
            amplitude=2.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        direction="+",
    )

    compiled = compile_plane_wave(pw, grid=_resolved_grid())

    assert compiled.placement_kind == "sheet"
    assert compiled.injection_axis == 0
    assert compiled.direction == "+"
    # The resolved grid has 10 cells in each direction (dl=1.0, size=10)
    # But the plane at center=5 with size=10 spans y=0 to y=10, z=0 to z=10
    # The grid cell centers in those ranges are: y=[0.5,1.5,2.5,3.5,4.5] and z=[0.5,1.5,2.5,3.5,4.5]
    # So we get 5x5 = 25 placements
    assert compiled.support_point_count == 25
    assert compiled.angle_theta == 0.0
    assert compiled.angle_phi == 0.0
    assert compiled.pol_angle == 0.0
    assert compiled.is_fixed_angle is False


def test_plane_wave_injection_kernel():
    """PlaneWave injection works correctly."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=CustomSourceTime(
            freq0=1.0,
            fwidth=1.0,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        direction="+",
    )

    compiled = compile_plane_wave(pw, grid=_resolved_grid())

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    # At time=0, the source should have some amplitude
    e2, m2 = inject_plane_wave(electric, magnetic, compiled, time=0.0, dt=0.1)

    # Something should be injected on the plane
    # The injection axis is x (0), so we should see changes in Ey and Ez components
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_plane_wave_runtime_integration():
    """PlaneWave integrates with the runtime injection pipeline."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=CustomSourceTime(
            freq0=1.0,
            fwidth=1.0,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        direction="+",
    )

    compiled_source = build_plane_wave_runtime(pw, grid=_resolved_grid())

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    e2, m2 = apply_plane_wave_sources(
        electric,
        magnetic,
        (compiled_source,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_plane_wave_with_angles():
    """PlaneWave with non-zero angles."""
    # Default angular spec is FixedInPlaneKSpec
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        angle_theta=0.5,
        angle_phi=0.3,
        pol_angle=0.2,
        direction="+",
    )

    assert pw.injection_axis == 0
    # is_fixed_angle is True only with FixedAngleSpec AND non-zero theta
    # Default is FixedInPlaneKSpec, so is_fixed_angle is False even with theta=0.5
    assert pw.is_fixed_angle is False

    # Direction vector should reflect the angles
    dir_vec = pw._dir_vector
    # For x-injection with theta=0.5, the k should have significant x-component
    assert abs(dir_vec[0]) > 0.5

    compiled = compile_plane_wave(pw, grid=_resolved_grid())
    assert compiled.angle_theta == 0.5
    assert compiled.angle_phi == 0.3
    assert compiled.pol_angle == 0.2


def test_plane_wave_fixed_angle_spec():
    """PlaneWave with FixedAngleSpec at non-zero angle."""
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        angle_theta=0.5,
        angle_phi=0.3,
        pol_angle=0.2,
        angular_spec=FixedAngleSpec(),
        direction="+",
    )

    assert pw.injection_axis == 0
    # With FixedAngleSpec AND non-zero theta, is_fixed_angle is True
    assert pw.is_fixed_angle is True


# Import PlaneWave and FixedAngleSpec at module level for test access
import numpy as np  # noqa: E402
from autofdtd.sources import FixedAngleSpec, FixedInPlaneKSpec, PlaneWave  # noqa: E402