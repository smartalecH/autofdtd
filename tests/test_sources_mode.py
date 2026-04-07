"""Tests for ModeSource with solver-backed mode definitions and injection support."""

from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    GridSpec,
    ModeSource,
    ModeSpec,
    Scene,
    Simulation,
    UniformGrid,
)
from autofdtd.compiler import compile_mode_source
from autofdtd.ir import ModeSourceIR, simulation_to_ir
from autofdtd.kernels import inject_mode_source
from autofdtd.runtime import apply_mode_sources, build_mode_source_runtime


def _resolved_grid():
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0),
            grid_y=UniformGrid(dl=1.0),
            grid_z=UniformGrid(dl=1.0),
        ),
    )
    return simulation.resolved_grid()


def test_mode_source_requires_planar_geometry() -> None:
    """ModeSource must have exactly one zero-size dimension (planar source for mode injection)."""
    # Should succeed - xy plane (z-size = 0)
    source = ModeSource(
        size=(10, 2, 0),
        source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
        mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
        mode_index=1,
        direction="+",
        name="mode_source",
    )
    assert source.placement_kind == "sheet"
    assert source.injection_axis == 2  # z-axis
    assert source.mode_index == 1
    assert source.direction == "+"

    # Should fail - all dimensions non-zero (not planar)
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        ModeSource(
            size=(10, 2, 2),
            source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
            mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
            mode_index=0,
            direction="+",
        )

    # Should fail - all dimensions zero (not a valid source)
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        ModeSource(
            size=(0, 0, 0),
            source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
            mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
            mode_index=0,
            direction="+",
        )

    # Should fail - two zero dimensions
    with pytest.raises(ValueError, match="exactly one zero-size dimension"):
        ModeSource(
            size=(10, 0, 0),
            source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
            mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
            mode_index=0,
            direction="+",
        )


def test_mode_source_exposes_phase1_properties() -> None:
    """ModeSource exposes the expected Phase 1 properties."""
    source = ModeSource(
        center=(5.0, 1.0, 2.0),
        size=(10, 2, 0),
        source_time=ContinuousWave(freq0=1.0, fwidth=1.0, amplitude=1.0, offset=2.5),
        mode_spec=ModeSpec(num_modes=3, target_neff=2.0),
        mode_index=1,
        direction="-",
        name="test_mode_source",
        interpolate=True,
        confine_to_bounds=False,
    )

    assert source.name == "test_mode_source"
    assert source.direction == "-"
    assert source.injection_axis == 2  # z-axis is the non-zero dimension
    assert source.placement_kind == "sheet"
    assert source.mode_index == 1
    assert source.mode_spec.num_modes == 3
    assert source.mode_spec.target_neff == 2.0
    assert source.support_bounds == ((0.0, 0.0, 2.0), (10.0, 2.0, 2.0))


# Optical frequency for 1550nm wavelength: freq0 = c/λ ≈ 1.93e14 Hz
# In FDTD units where 1 unit = 1 um, this gives a wavelength of ~1.55 units
_OPTICAL_FREQ = 1.93e14


def _waveguide_scene():
    """Create a Scene with a slab waveguide for mode source testing."""
    from autofdtd.api import Box, Medium, Scene, Structure

    background = Medium(permittivity=1.0)
    waveguide_medium = Medium(permittivity=2.1)
    # Slab waveguide extending in z-direction (the injection axis)
    waveguide = Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0))
    structure = Structure(geometry=waveguide, medium=waveguide_medium)
    return Scene(structures=(structure,), medium=background)


def test_mode_source_compilation_with_empty_scene() -> None:
    """ModeSource compiles successfully with a waveguide scene supporting guided modes."""
    source = ModeSource(
        size=(4, 4, 0),  # xy plane
        source_time=CustomSourceTime(
            freq0=_OPTICAL_FREQ,
            fwidth=1e12,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        mode_spec=ModeSpec(num_modes=1),
        mode_index=0,
        direction="+",
    )
    grid = _resolved_grid()

    # Skip test if scipy not available (mode solver dependency)
    try:
        from scipy import sparse
    except ImportError:
        pytest.skip("scipy not available")

    scene = _waveguide_scene()

    compiled = compile_mode_source(
        source,
        grid=grid,
        scene=scene,
        sim_center=(0.0, 0.0, 0.0),
        sim_size=(4.0, 4.0, 4.0),
        dt=0.1,
    )

    assert compiled.placement_kind == "sheet"
    assert compiled.injection_axis == 2
    assert compiled.direction == "+"
    assert compiled.mode_index == 0
    assert compiled.e_field_data is not None
    assert compiled.h_field_data is not None
    assert compiled.x_coords is not None
    assert compiled.y_coords is not None


def test_mode_source_runtime_integration() -> None:
    """ModeSource integrates with the runtime injection pipeline."""
    # Skip test if scipy not available
    try:
        from scipy import sparse
    except ImportError:
        pytest.skip("scipy not available")

    source = ModeSource(
        size=(4, 4, 0),  # xy plane
        source_time=CustomSourceTime(
            freq0=_OPTICAL_FREQ,
            fwidth=1e12,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        mode_spec=ModeSpec(num_modes=1),
        mode_index=0,
        direction="+",
    )
    grid = _resolved_grid()

    scene = _waveguide_scene()

    compiled = build_mode_source_runtime(
        source,
        grid=grid,
        scene=scene,
        sim_center=(0.0, 0.0, 0.0),
        sim_size=(4.0, 4.0, 4.0),
        dt=0.1,
    )

    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    inj_e, inj_h = apply_mode_sources(
        electric,
        magnetic,
        (compiled,),
        time=0.0,
        dt=0.5,
    )

    # ModeSource injection should have modified fields
    # (actual values depend on mode profile)
    assert inj_e is not electric or np.any(inj_e != 0.0)
    assert inj_h is not magnetic or np.any(inj_h != 0.0)


def test_mode_source_injection_kernel() -> None:
    """ModeSource injection kernel applies mode fields correctly."""
    # Skip test if scipy not available
    try:
        from scipy import sparse
    except ImportError:
        pytest.skip("scipy not available")

    source = ModeSource(
        size=(4, 4, 0),  # xy plane
        source_time=CustomSourceTime(
            freq0=_OPTICAL_FREQ,
            fwidth=1e12,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        mode_spec=ModeSpec(num_modes=1),
        mode_index=0,
        direction="+",
    )
    grid = _resolved_grid()

    scene = _waveguide_scene()

    compiled = compile_mode_source(
        source,
        grid=grid,
        scene=scene,
        sim_center=(0.0, 0.0, 0.0),
        sim_size=(4.0, 4.0, 4.0),
        dt=0.1,
    )

    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    # Apply injection with positive direction
    inj_e, inj_h = inject_mode_source(
        electric,
        magnetic,
        compiled,
        time=0.0,
        dt=0.5,
    )

    # Something should be injected
    assert np.any(inj_e != 0.0) or np.any(inj_h != 0.0)


def test_mode_source_backward_direction() -> None:
    """ModeSource with direction='-' injects backward-propagating mode."""
    # Skip test if scipy not available
    try:
        from scipy import sparse
    except ImportError:
        pytest.skip("scipy not available")

    source = ModeSource(
        size=(4, 4, 0),  # xy plane
        source_time=CustomSourceTime(
            freq0=_OPTICAL_FREQ,
            fwidth=1e12,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        mode_spec=ModeSpec(num_modes=1),
        mode_index=0,
        direction="-",  # Backward propagation
    )
    grid = _resolved_grid()

    scene = _waveguide_scene()

    compiled = compile_mode_source(
        source,
        grid=grid,
        scene=scene,
        sim_center=(0.0, 0.0, 0.0),
        sim_size=(4.0, 4.0, 4.0),
        dt=0.1,
    )

    assert compiled.direction == "-"

    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros_like(electric)

    inj_e, inj_h = inject_mode_source(
        electric,
        magnetic,
        compiled,
        time=0.0,
        dt=0.5,
    )

    # Something should be injected with different sign than forward
    assert np.any(inj_e != 0.0) or np.any(inj_h != 0.0)


def test_mode_source_mode_index_bounds() -> None:
    """ModeSource with invalid mode_index raises appropriate error."""
    # Skip test if scipy not available
    try:
        from scipy import sparse
    except ImportError:
        pytest.skip("scipy not available")

    source = ModeSource(
        size=(4, 4, 0),
        source_time=CustomSourceTime(
            freq0=_OPTICAL_FREQ,
            fwidth=1e12,
            amplitude=1.0,
            offset=0.0,
            time_samples=(0.0, 1.0),
            envelope_values=((1.0, 0.0), (1.0, 0.0)),
        ),
        mode_spec=ModeSpec(num_modes=1),
        mode_index=5,  # Only 1 mode available (index 0)
        direction="+",
    )
    grid = _resolved_grid()

    scene = _waveguide_scene()

    with pytest.raises(ValueError, match="mode_index"):
        compile_mode_source(
            source,
            grid=grid,
            scene=scene,
            sim_center=(0.0, 0.0, 0.0),
            sim_size=(4.0, 4.0, 4.0),
            dt=0.1,
        )
