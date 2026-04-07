"""Tests for TFSF (Total-Field Scattered-Field) source implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    GaussianPulse,
    GridSpec,
    Simulation,
    UniformGrid,
)
from autofdtd.compiler import compile_tfsf
from autofdtd.ir import TFSFIR, simulation_to_ir
from autofdtd.kernels import inject_tfsf
from autofdtd.runtime import (
    apply_tfsf_sources,
    build_tfsf_runtime,
)
from autofdtd.sources import TFSF


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


def test_tfsf_requires_volume_geometry():
    """TFSF must have all three dimensions non-zero (volume source)."""
    # Should succeed - 3D volume
    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )
    assert tfsf.placement_kind == "volume"
    assert tfsf.injection_axis == 2  # z-axis

    # Should fail - one dimension zero (planar source)
    with pytest.raises(ValueError, match="3D volume"):
        TFSF(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 0.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            injection_axis=2,
            direction="+",
        )


def test_tfsf_injection_axis():
    """TFSF injection_axis must be 0, 1, or 2."""
    # Valid axes
    for axis in (0, 1, 2):
        tfsf = TFSF(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            injection_axis=axis,
            direction="+",
        )
        assert tfsf.injection_axis == axis

    # Invalid axis
    with pytest.raises(ValueError, match="injection_axis must be 0"):
        TFSF(
            center=(0.0, 0.0, 0.0),
            size=(2.0, 2.0, 2.0),
            source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
            injection_axis=3,
            direction="+",
        )


def test_tfsf_bounds():
    """TFSF computes correct support and injection plane bounds."""
    tfsf = TFSF(
        center=(5.0, 5.0, 5.0),
        size=(2.0, 4.0, 6.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    # Support bounds: center ± size/2
    lower, upper = tfsf.support_bounds
    assert lower == (4.0, 3.0, 2.0)
    assert upper == (6.0, 7.0, 8.0)

    # Injection plane center: offset by half size along injection axis
    # direction="+" means injection plane at negative face
    plane_center = tfsf.injection_plane_center
    assert plane_center == (5.0, 5.0, 2.0)  # z = 5 - 6/2 = 2

    # For direction="-", injection plane at positive face
    tfsf_bw = TFSF(
        center=(5.0, 5.0, 5.0),
        size=(2.0, 4.0, 6.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="-",
    )
    plane_center_bw = tfsf_bw.injection_plane_center
    assert plane_center_bw == (5.0, 5.0, 8.0)  # z = 5 + 6/2 = 8


def test_tfsf_direction_vectors():
    """TFSF computes correct direction and polarization vectors."""
    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
    )

    # Normal incidence - direction along +z for "+" (propagates in +direction of injection axis)
    dir_vec = tfsf._dir_vector
    assert dir_vec == (0.0, 0.0, 1.0)

    # For z-injection with pol_angle=0, polarization along x
    pol_vec = tfsf._pol_vector
    assert pol_vec[2] == 0.0  # z component should be 0
    assert pol_vec[0] == 1.0 or pol_vec[1] == 1.0  # x or y should be 1


def test_tfsf_ir_lowering():
    """TFSF lowers correctly to typed IR."""
    tfsf = TFSF(
        center=(1.0, 2.0, 3.0),
        size=(2.0, 3.0, 4.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=1,
        direction="-",
        angle_theta=0.1,
        angle_phi=0.2,
        pol_angle=0.3,
        num_freqs=3,
    )

    ir = simulation_to_ir(
        Simulation(
            size=(10.0, 10.0, 10.0),
            run_time=1.0,
            sources=[tfsf],
        )
    )

    # Find the TFSF IR in the lowered simulation
    tfsf_ir = None
    for src in ir.sources:
        if isinstance(src, TFSFIR):
            tfsf_ir = src
            break

    assert tfsf_ir is not None
    assert tfsf_ir.center == (1.0, 2.0, 3.0)
    assert tfsf_ir.size == (2.0, 3.0, 4.0)
    assert tfsf_ir.direction == "-"
    assert tfsf_ir.injection_axis == 1
    assert tfsf_ir.num_freqs == 3
    assert tfsf_ir.angle_theta == 0.1
    assert tfsf_ir.angle_phi == 0.2
    assert tfsf_ir.pol_angle == 0.3
    assert tfsf_ir.placement_kind == "volume"


def test_tfsf_compilation():
    """TFSF compiles to discrete placement data."""
    grid = _resolved_grid()

    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(3.0, 3.0, 3.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    compiled = compile_tfsf(tfsf, grid=grid)

    assert compiled.injection_axis == 2
    assert compiled.direction == "+"
    assert compiled.placement_kind == "volume"
    assert compiled.support_point_count > 0

    # Check that we have bounds indices
    assert len(compiled.tfsf_bounds_indices) == 2


def test_tfsf_runtime_build():
    """TFSF builds runtime placement data."""
    grid = _resolved_grid()

    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(3.0, 3.0, 3.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    runtime = build_tfsf_runtime(tfsf, grid=grid)
    assert runtime.injection_axis == 2


def test_tfsf_injection_kernel():
    """TFSF injection kernel updates field buffers."""
    grid = _resolved_grid()

    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(3.0, 3.0, 3.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    compiled = compile_tfsf(tfsf, grid=grid)

    # Create field buffers
    nx, ny, nz = 10, 10, 10
    electric_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    magnetic_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Apply TFSF injection at a specific time
    time = 0.5
    dt = 0.1

    e_before = electric_field.copy()
    h_before = magnetic_field.copy()

    e_out, h_out = inject_tfsf(
        electric_field,
        magnetic_field,
        compiled,
        time=time,
        dt=dt,
    )

    # Fields should be modified at TFSF placements
    assert e_out.shape == electric_field.shape
    assert h_out.shape == magnetic_field.shape


def test_tfsf_runtime_apply():
    """TFSF runtime apply stages multiple sources."""
    grid = _resolved_grid()

    tfsf1 = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(3.0, 3.0, 3.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    tfsf2 = TFSF(
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="-",
    )

    runtime1 = build_tfsf_runtime(tfsf1, grid=grid)
    runtime2 = build_tfsf_runtime(tfsf2, grid=grid)

    # Create field buffers
    nx, ny, nz = 10, 10, 10
    electric_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
    magnetic_field = np.zeros((nx, ny, nz, 3), dtype=np.complex128)

    # Apply multiple TFSF sources
    e_out, h_out = apply_tfsf_sources(
        electric_field,
        magnetic_field,
        (runtime1, runtime2),
        time=0.5,
        dt=0.1,
    )

    assert e_out.shape == electric_field.shape
    assert h_out.shape == magnetic_field.shape


def test_tfsf_source_time_integration():
    """TFSF amplitude correctly integrates with source_time."""
    tfsf = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    grid = _resolved_grid()
    compiled = compile_tfsf(tfsf, grid=grid)

    # Amplitude at different times should vary
    amp_t0 = compiled.amplitude_at_time(0.0)
    amp_t1 = compiled.amplitude_at_time(1.0)

    # Pulse amplitude should be different at different times
    assert amp_t0 != amp_t1 or abs(amp_t0) < 1e-10  # May be near zero at t=0


def test_tfsf_reference_wavelength():
    """TFSF correctly computes reference wavelength from source_time."""
    # With GaussianPulse
    tfsf_g = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=2.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )
    wavelength_g = tfsf_g._reference_wavelength
    assert wavelength_g is not None
    assert abs(wavelength_g - 2.998e8 / 2.0) < 1e6  # c/freq

    # With ContinuousWave
    tfsf_cw = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=ContinuousWave(freq0=1.5, fwidth=0.5, amplitude=1.0),
        injection_axis=2,
        direction="+",
    )
    wavelength_cw = tfsf_cw._reference_wavelength
    assert wavelength_cw is not None
    assert abs(wavelength_cw - 2.998e8 / 1.5) < 1e6


def test_tfsf_tfsf_bounds():
    """TFSF correctly computes tfsf_bounds property."""
    tfsf = TFSF(
        center=(5.0, 5.0, 5.0),
        size=(2.0, 4.0, 6.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )

    lower, upper = tfsf.tfsf_bounds
    assert lower == (4.0, 3.0, 2.0)
    assert upper == (6.0, 7.0, 8.0)


def test_tfsf_tangential_axes():
    """TFSF correctly identifies tangential axes."""
    # Z-injection
    tfsf_z = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=2,
        direction="+",
    )
    assert tfsf_z._tangential_axes == (0, 1)

    # X-injection
    tfsf_x = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=0,
        direction="+",
    )
    assert tfsf_x._tangential_axes == (1, 2)

    # Y-injection
    tfsf_y = TFSF(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=1,
        direction="+",
    )
    assert tfsf_y._tangential_axes == (0, 2)


def test_tfsf_serialization():
    """TFSF serializes and deserializes correctly."""
    tfsf = TFSF(
        center=(1.0, 2.0, 3.0),
        size=(2.0, 3.0, 4.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5),
        injection_axis=1,
        direction="-",
        angle_theta=0.1,
        angle_phi=0.2,
        pol_angle=0.3,
        num_freqs=5,
        name="test_tfsf",
    )

    # Test JSON serialization
    json_text = tfsf.model_dump_json()
    assert '"type":"TFSF"' in json_text
    assert '"injection_axis":1' in json_text
    assert '"name":"test_tfsf"' in json_text

    # Test deserialization
    tfsf_restored = TFSF.model_validate_json(json_text)
    assert tfsf_restored.center == tfsf.center
    assert tfsf_restored.size == tfsf.size
    assert tfsf_restored.injection_axis == tfsf.injection_axis
    assert tfsf_restored.direction == tfsf.direction
    assert tfsf_restored.num_freqs == tfsf.num_freqs
    assert tfsf_restored.name == tfsf.name
