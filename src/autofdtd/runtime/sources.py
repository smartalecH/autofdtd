"""Runtime helpers for compiled source injection."""

from __future__ import annotations

import numpy as np

from autofdtd.compiler.sources import (
    CompiledAstigmaticGaussianBeam,
    CompiledCustomCurrentSource,
    CompiledCustomFieldSource,
    CompiledGaussianBeam,
    CompiledModeSource,
    CompiledPlaneWave,
    CompiledPointDipole,
    CompiledTFSF,
    CompiledUniformCurrentSource,
    compile_astigmatic_gaussian_beam,
    compile_custom_current_source,
    compile_custom_field_source,
    compile_gaussian_beam,
    compile_mode_source,
    compile_plane_wave,
    compile_point_dipole,
    compile_tfsf,
    compile_uniform_current_source,
)
from autofdtd.grid import ResolvedGrid
from autofdtd.kernels.sources import (
    inject_astigmatic_gaussian_beam,
    inject_custom_current_source,
    inject_custom_field_source,
    inject_gaussian_beam,
    inject_mode_source,
    inject_plane_wave,
    inject_point_dipole,
    inject_tfsf,
    inject_uniform_current_source,
)
from autofdtd.sources import (
    AstigmaticGaussianBeam,
    CustomCurrentSource,
    CustomFieldSource,
    GaussianBeam,
    PlaneWave,
    PointDipole,
    TFSF,
    UniformCurrentSource,
)
from autofdtd.sources.mode import ModeSource


def build_uniform_current_runtime(
    source: UniformCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledUniformCurrentSource:
    """Compile runtime placement data for a uniform current source."""

    return compile_uniform_current_source(source, grid=grid)


def build_point_dipole_runtime(
    source: PointDipole,
    *,
    grid: ResolvedGrid,
) -> CompiledPointDipole:
    """Compile runtime placement data for a point dipole source."""

    return compile_point_dipole(source, grid=grid)


def build_custom_current_source_runtime(
    source: CustomCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledCustomCurrentSource:
    """Compile runtime placement data for a custom current source."""

    return compile_custom_current_source(source, grid=grid)


def apply_uniform_current_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledUniformCurrentSource, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled uniform current sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_uniform_current_source(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


def apply_point_dipole_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledPointDipole, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled point dipole sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_point_dipole(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


def apply_custom_current_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledCustomCurrentSource, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled custom current sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_custom_current_source(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


def build_custom_field_source_runtime(
    source: CustomFieldSource,
    *,
    grid: ResolvedGrid,
) -> CompiledCustomFieldSource:
    """Compile runtime placement data for a custom field source."""

    return compile_custom_field_source(source, grid=grid)


def apply_custom_field_source_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledCustomFieldSource, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled custom field sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_custom_field_source(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


def build_mode_source_runtime(
    source: ModeSource,
    *,
    grid: ResolvedGrid,
    scene,
    sim_center: tuple[float, float, float],
    sim_size: tuple[float, float, float],
    dt: float,
) -> CompiledModeSource:
    """Compile runtime placement data for a mode source.

    Uses the mode solver to compute the mode field profile at the source's
    central frequency.
    """

    return compile_mode_source(
        source,
        grid=grid,
        scene=scene,
        sim_center=sim_center,
        sim_size=sim_size,
        dt=dt,
    )


def apply_mode_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledModeSource, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled mode sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_mode_source(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


def build_plane_wave_runtime(
    source: PlaneWave,
    *,
    grid: ResolvedGrid,
) -> CompiledPlaneWave:
    """Compile runtime placement data for a plane wave source."""

    return compile_plane_wave(source, grid=grid)


def apply_plane_wave_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledPlaneWave, ...],
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled plane wave sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_plane_wave(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
            freq=freq,
        )
    return electric, magnetic


def build_gaussian_beam_runtime(
    source: GaussianBeam,
    *,
    grid: ResolvedGrid,
) -> CompiledGaussianBeam:
    """Compile runtime placement data for a Gaussian beam source."""

    return compile_gaussian_beam(source, grid=grid)


def apply_gaussian_beam_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledGaussianBeam, ...],
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled Gaussian beam sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_gaussian_beam(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
            freq=freq,
        )
    return electric, magnetic


def build_astigmatic_gaussian_beam_runtime(
    source: AstigmaticGaussianBeam,
    *,
    grid: ResolvedGrid,
) -> CompiledAstigmaticGaussianBeam:
    """Compile runtime placement data for an astigmatic Gaussian beam source."""

    return compile_astigmatic_gaussian_beam(source, grid=grid)


def apply_astigmatic_gaussian_beam_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledAstigmaticGaussianBeam, ...],
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled astigmatic Gaussian beam sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_astigmatic_gaussian_beam(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
            freq=freq,
        )
    return electric, magnetic


def build_tfsf_runtime(
    source: TFSF,
    *,
    grid: ResolvedGrid,
) -> CompiledTFSF:
    """Compile runtime placement data for a TFSF source."""

    return compile_tfsf(source, grid=grid)


def apply_tfsf_sources(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    sources: tuple[CompiledTFSF, ...],
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a staged batch of compiled TFSF sources."""

    electric = electric_field
    magnetic = magnetic_field
    for source in sources:
        electric, magnetic = inject_tfsf(
            electric,
            magnetic,
            source,
            time=time,
            dt=dt,
        )
    return electric, magnetic


# ---------------------------------------------------------------------------
# Unified source injection stage
# ---------------------------------------------------------------------------


def apply_source_injection_stage(
    electric_field,
    magnetic_field,
    *,
    uniform_current_sources: tuple[CompiledUniformCurrentSource, ...] = (),
    point_dipole_sources: tuple[CompiledPointDipole, ...] = (),
    custom_current_sources: tuple[CompiledCustomCurrentSource, ...] = (),
    custom_field_sources: tuple[CompiledCustomFieldSource, ...] = (),
    mode_sources: tuple[CompiledModeSource, ...] = (),
    plane_wave_sources: tuple[CompiledPlaneWave, ...] = (),
    gaussian_beam_sources: tuple[CompiledGaussianBeam, ...] = (),
    astigmatic_gaussian_beam_sources: tuple[CompiledAstigmaticGaussianBeam, ...] = (),
    tfsf_sources: tuple[CompiledTFSF, ...] = (),
    time: float = 0.0,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple:
    """Apply all compiled sources in the source_injection stage.

    This is the unified entry point for source injection during a Maxwell
    timestep. It applies all source types in the following order:

    1. Uniform current sources (electric and magnetic currents)
    2. Point dipole sources
    3. Custom current sources
    4. Custom field sources (equivalence principle)
    5. Mode sources (equivalence principle with mode profile)
    6. Plane wave sources (equivalence principle)
    7. Gaussian beam sources (equivalence principle with Gaussian envelope)
    8. Astigmatic Gaussian beam sources
    9. TFSF sources (volume injection with TF/SF correction)

    Scheduling Semantics
    --------------------
    The source_injection stage runs after the first boundary_exchange and
    electric_update, and before the second boundary_exchange and magnetic_update::

        boundary_exchange → electric_update → source_injection
        → boundary_exchange → magnetic_update → pml_stage

    All sources receive the same (time, dt, freq) arguments so that sources
    of different types remain synchronized. The caller is responsible for
    passing consistent time values across the stage sequence.

    GPU path (wp.array input): preserved and passed directly to each inject
    function, which launches Warp kernels for in-place injection with no CPU
    roundtrip.

    CPU path (ndarray input): NumPy fallback with complex128 arithmetic.
    """
    # Import here to avoid circular import issues with warp detection
    from autofdtd.kernels.sources import _is_warp_array

    is_warp = _is_warp_array(electric_field)

    # Uniform current sources (electric/magnetic currents)
    if uniform_current_sources:
        electric_field, magnetic_field = apply_uniform_current_sources(
            electric_field, magnetic_field, uniform_current_sources, time=time, dt=dt
        )

    # Point dipole sources
    if point_dipole_sources:
        electric_field, magnetic_field = apply_point_dipole_sources(
            electric_field, magnetic_field, point_dipole_sources, time=time, dt=dt
        )

    # Custom current sources
    if custom_current_sources:
        electric_field, magnetic_field = apply_custom_current_sources(
            electric_field, magnetic_field, custom_current_sources, time=time, dt=dt
        )

    # Custom field sources (equivalence principle)
    if custom_field_sources:
        electric_field, magnetic_field = apply_custom_field_source_sources(
            electric_field, magnetic_field, custom_field_sources, time=time, dt=dt
        )

    # Mode sources (equivalence principle with mode profile)
    if mode_sources:
        electric_field, magnetic_field = apply_mode_sources(
            electric_field, magnetic_field, mode_sources, time=time, dt=dt
        )

    # Plane wave sources (equivalence principle)
    if plane_wave_sources:
        electric_field, magnetic_field = apply_plane_wave_sources(
            electric_field, magnetic_field, plane_wave_sources, time=time, dt=dt, freq=freq
        )

    # Gaussian beam sources (equivalence principle with Gaussian envelope)
    if gaussian_beam_sources:
        electric_field, magnetic_field = apply_gaussian_beam_sources(
            electric_field, magnetic_field, gaussian_beam_sources, time=time, dt=dt, freq=freq
        )

    # Astigmatic Gaussian beam sources
    if astigmatic_gaussian_beam_sources:
        electric_field, magnetic_field = apply_astigmatic_gaussian_beam_sources(
            electric_field, magnetic_field, astigmatic_gaussian_beam_sources, time=time, dt=dt, freq=freq
        )

    # TFSF sources (volume injection with TF/SF correction)
    if tfsf_sources:
        electric_field, magnetic_field = apply_tfsf_sources(
            electric_field, magnetic_field, tfsf_sources, time=time, dt=dt
        )

    # For GPU path, each inject function returns the same wp.array (modified in-place)
    # For CPU path, each inject function returns modified numpy arrays
    return electric_field, magnetic_field
