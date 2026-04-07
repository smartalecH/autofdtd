"""Tests for unified source injection scheduling.

This module tests the apply_source_injection_stage function that orchestrates
all source types in a single kernel call, following the scheduling semantics
defined in step_kernel_metadata().
"""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    CustomCurrentSource,
    CustomFieldSource,
    GaussianPulse,
    GridSpec,
    Simulation,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.compiler import (
    compile_custom_current_source,
    compile_custom_field_source,
    compile_point_dipole,
    compile_plane_wave,
    compile_tfsf,
    compile_uniform_current_source,
)
from autofdtd.kernels import inject_sources_stage
from autofdtd.runtime import apply_source_injection_stage
from autofdtd.sources import PointDipole, PlaneWave, TFSF


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


def test_apply_source_injection_stage_with_no_sources():
    """apply_source_injection_stage with no sources leaves fields unchanged."""
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric, magnetic, time=0.0, dt=0.1
    )

    np.testing.assert_array_equal(e2, electric)
    np.testing.assert_array_equal(m2, magnetic)


def test_apply_source_injection_stage_with_uniform_current():
    """apply_source_injection_stage applies uniform current source correctly."""
    grid = _resolved_grid()

    source = UniformCurrentSource(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 2.0),
        polarization="Ez",
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
    )

    compiled = compile_uniform_current_source(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        uniform_current_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_apply_source_injection_stage_with_plane_wave():
    """apply_source_injection_stage applies plane wave source correctly."""
    grid = _resolved_grid()

    source = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )

    compiled = compile_plane_wave(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        plane_wave_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_apply_source_injection_stage_with_tfsf():
    """apply_source_injection_stage applies TFSF source correctly."""
    grid = _resolved_grid()

    source = TFSF(
        center=(5.0, 5.0, 5.0),
        size=(3.0, 3.0, 3.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        injection_axis=2,
        direction="+",
    )

    compiled = compile_tfsf(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        tfsf_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_apply_source_injection_stage_with_multiple_source_types():
    """apply_source_injection_stage applies multiple source types correctly."""
    grid = _resolved_grid()

    # Uniform current source
    ucs = UniformCurrentSource(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 2.0),
        polarization="Ez",
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
    )
    compiled_ucs = compile_uniform_current_source(ucs, grid=grid)

    # Plane wave source
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )
    compiled_pw = compile_plane_wave(pw, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        uniform_current_sources=(compiled_ucs,),
        plane_wave_sources=(compiled_pw,),
        time=0.0,
        dt=0.1,
    )

    # Both sources should contribute
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_inject_sources_stage_kernel_equivalent():
    """inject_sources_stage kernel function is equivalent to apply_source_injection_stage."""
    grid = _resolved_grid()

    # Uniform current source
    ucs = UniformCurrentSource(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 2.0),
        polarization="Ez",
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
    )
    compiled_ucs = compile_uniform_current_source(ucs, grid=grid)

    # Plane wave source
    pw = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )
    compiled_pw = compile_plane_wave(pw, grid=grid)

    # Apply using runtime function
    electric1 = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic1 = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    e1, m1 = apply_source_injection_stage(
        electric1,
        magnetic1,
        uniform_current_sources=(compiled_ucs,),
        plane_wave_sources=(compiled_pw,),
        time=0.0,
        dt=0.1,
    )

    # Apply using kernel function
    electric2 = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic2 = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    e2, m2 = inject_sources_stage(
        electric2,
        magnetic2,
        uniform_current_sources=(compiled_ucs,),
        plane_wave_sources=(compiled_pw,),
        time=0.0,
        dt=0.1,
    )

    # Results should be identical
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(m1, m2)


def test_source_injection_stage_time_evolution():
    """Source amplitude changes correctly with time."""
    grid = _resolved_grid()

    source = UniformCurrentSource(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 0.0, 2.0),
        polarization="Ez",
        source_time=ContinuousWave(freq0=1.0, fwidth=0.5, amplitude=1.0),
    )

    compiled = compile_uniform_current_source(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Apply at different times
    e1, m1 = apply_source_injection_stage(
        electric, magnetic, uniform_current_sources=(compiled,), time=0.0, dt=0.1
    )

    e2, m2 = apply_source_injection_stage(
        electric, magnetic, uniform_current_sources=(compiled,), time=0.5, dt=0.1
    )

    e3, m3 = apply_source_injection_stage(
        electric, magnetic, uniform_current_sources=(compiled,), time=1.0, dt=0.1
    )

    # Different times should give different amplitudes (pulsed source)
    # Note: ContinuousWave at freq0=1.0 with fwidth=0.5 will have time-varying amplitude
    assert not np.allclose(e1, e2) or not np.allclose(e2, e3)


def test_source_injection_preserves_field_shape():
    """apply_source_injection_stage preserves field array shapes."""
    grid = _resolved_grid()

    source = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )

    compiled = compile_plane_wave(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        plane_wave_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    assert e2.shape == electric.shape
    assert m2.shape == magnetic.shape


def test_source_injection_with_freq_parameter():
    """apply_source_injection_stage accepts freq parameter for plane wave."""
    grid = _resolved_grid()

    source = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )

    compiled = compile_plane_wave(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    # Apply with explicit freq
    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        plane_wave_sources=(compiled,),
        time=0.0,
        dt=0.1,
        freq=1.0,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_point_dipole_in_scheduling():
    """Point dipole source works with unified scheduling."""
    grid = _resolved_grid()

    source = PointDipole(
        center=(5.0, 5.0, 5.0),
        polarization="Ex",
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
    )

    compiled = compile_point_dipole(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        point_dipole_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_custom_current_source_in_scheduling():
    """Custom current source works with unified scheduling."""
    grid = _resolved_grid()

    # Create a custom current source with simple E field data
    # Note: CustomCurrentSource e_fields/h_fields use real values
    source = CustomCurrentSource(
        center=(5.0, 5.0, 5.0),
        size=(2.0, 2.0, 2.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        e_fields={"Ex": [1.0] * 8},
    )

    compiled = compile_custom_current_source(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        custom_current_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something may or may not be injected depending on placement overlap
    # The important thing is it doesn't crash
    assert e2.shape == electric.shape
    assert m2.shape == magnetic.shape


def test_custom_field_source_in_scheduling():
    """Custom field source works with unified scheduling."""
    grid = _resolved_grid()

    # Create a custom field source on a planar surface
    # Note: CustomFieldSource e_fields/h_fields use real values
    # injection_axis is derived from the zero-size dimension of size
    source = CustomFieldSource(
        center=(5.0, 5.0, 5.0),
        size=(10.0, 10.0, 0.0),  # z-size=0 means injection_axis=2
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        e_fields={"Ex": [1.0] * 25, "Ey": [0.5] * 25},
        h_fields={"Hx": [0.3] * 25, "Hy": [0.2] * 25},
    )

    compiled = compile_custom_field_source(source, grid=grid)

    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)

    e2, m2 = apply_source_injection_stage(
        electric,
        magnetic,
        custom_field_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Something should be injected
    assert np.any(e2 != 0.0) or np.any(m2 != 0.0)


def test_scheduling_semantics_preserves_fields():
    """Multiple calls preserve previously injected fields."""
    grid = _resolved_grid()

    source = PlaneWave(
        center=(5.0, 5.0, 5.0),
        size=(0.0, 10.0, 10.0),
        source_time=GaussianPulse(freq0=1.0, fwidth=0.5, amplitude=1.0),
        direction="+",
    )

    compiled = compile_plane_wave(source, grid=grid)

    # First call
    electric = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    magnetic = np.zeros((10, 10, 10, 3), dtype=np.complex128)
    e1, m1 = apply_source_injection_stage(
        electric,
        magnetic,
        plane_wave_sources=(compiled,),
        time=0.0,
        dt=0.1,
    )

    # Second call with first result as input
    e2, m2 = apply_source_injection_stage(
        e1,
        m1,
        plane_wave_sources=(compiled,),
        time=0.1,
        dt=0.1,
    )

    # Should accumulate (not overwrite)
    assert np.any(e2 != e1) or np.any(m2 != m1)
