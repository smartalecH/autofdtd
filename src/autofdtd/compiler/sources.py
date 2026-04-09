"""Source compilation helpers for Phase 1 current injection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from autofdtd.grid import ResolvedGrid, ResolvedGridAxis
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


@dataclass(frozen=True)
class CompiledAxisPlacement:
    """Discrete support metadata for one source axis."""

    indices: tuple[int, ...]
    weights: tuple[float, ...]
    cell_sizes: tuple[float, ...]
    centers: tuple[float, ...]


@dataclass(frozen=True)
class CompiledUniformCurrentSource:
    """Discrete current-source placement and amplitude metadata."""

    source: UniformCurrentSource
    field_kind: Literal["electric", "magnetic"]
    component_axis: int
    placement_kind: Literal["point", "line", "sheet", "volume"]
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    amplitude_scale: float

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0]) * self.amplitude_scale


@dataclass(frozen=True)
class CompiledPointDipole:
    """Discrete point dipole placement and amplitude metadata."""

    source: PointDipole
    field_kind: Literal["electric", "magnetic"]
    component_axis: int
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    placement_kind: Literal["point"] = "point"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


@dataclass(frozen=True)
class CompiledCustomCurrentSource:
    """Discrete custom current source placement and amplitude metadata."""

    source: CustomCurrentSource
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    has_electric: bool
    has_magnetic: bool
    e_field_data: dict[str, np.ndarray] | None
    h_field_data: dict[str, np.ndarray] | None

    @property
    def placement_kind(self) -> str:
        """Determine placement kind based on source size."""
        return "point"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


@dataclass(frozen=True)
class CompiledCustomFieldSource:
    """Discrete custom field source placement and amplitude metadata.

    CustomFieldSource uses the equivalence principle on a planar surface,
    deriving J = n × H and M = -n × E from tangential field components.
    """

    source: CustomFieldSource
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    has_electric: bool
    has_magnetic: bool
    has_tangential_fields: bool
    e_field_data: dict[str, np.ndarray] | None
    h_field_data: dict[str, np.ndarray] | None

    @property
    def placement_kind(self) -> str:
        """Determine placement kind based on source size."""
        return "sheet"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def _axis_centers(axis: ResolvedGridAxis) -> np.ndarray:
    boundaries = np.asarray(axis.boundaries, dtype=float)
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def _axis_sizes(axis: ResolvedGridAxis) -> np.ndarray:
    return np.asarray(axis.cell_sizes, dtype=float)


def _bracket_indices(values: np.ndarray, position: float) -> tuple[int, int]:
    if values.size == 1:
        return (0, 0)
    if position <= values[0]:
        return (0, 0)
    if position >= values[-1]:
        last = values.size - 1
        return (last, last)
    upper = int(np.searchsorted(values, position, side="right"))
    lower = max(0, upper - 1)
    return (lower, min(values.size - 1, upper))


def _compile_axis_placement(
    axis: ResolvedGridAxis,
    *,
    center: float,
    size: float,
    interpolate: bool,
    confine_to_bounds: bool,
) -> CompiledAxisPlacement:
    if axis.num_cells <= 0:
        raise ValueError("UniformCurrentSource requires a non-collapsed grid axis")

    centers = _axis_centers(axis)
    cell_sizes = _axis_sizes(axis)
    boundaries = np.asarray(axis.boundaries, dtype=float)
    tolerance = max(1e-12, axis.max_step * 1e-9)

    if math.isclose(size, 0.0, abs_tol=tolerance):
        exact = np.nonzero(np.isclose(centers, center, rtol=0.0, atol=tolerance))[0]
        if exact.size:
            index = int(exact[0])
            return CompiledAxisPlacement(
                indices=(index,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[index]),),
                centers=(float(centers[index]),),
            )
        if not interpolate:
            index = int(np.argmin(np.abs(centers - center)))
            return CompiledAxisPlacement(
                indices=(index,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[index]),),
                centers=(float(centers[index]),),
            )

        lower, upper = _bracket_indices(centers, center)
        if lower == upper:
            return CompiledAxisPlacement(
                indices=(lower,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[lower]),),
                centers=(float(centers[lower]),),
            )
        left_center = centers[lower]
        right_center = centers[upper]
        fraction = float((center - left_center) / (right_center - left_center))
        return CompiledAxisPlacement(
            indices=(lower, upper),
            weights=(1.0 - fraction, fraction),
            cell_sizes=(float(cell_sizes[lower]), float(cell_sizes[upper])),
            centers=(float(left_center), float(right_center)),
        )

    lower = center - size / 2.0
    upper = center + size / 2.0
    if confine_to_bounds:
        overlaps = np.minimum(boundaries[1:], upper) - np.maximum(boundaries[:-1], lower)
        mask = overlaps > tolerance
        if not np.any(mask):
            raise ValueError("UniformCurrentSource support does not overlap the resolved grid")
        weights = overlaps[mask] / cell_sizes[mask]
        indices = np.nonzero(mask)[0]
    else:
        mask = (centers >= lower - tolerance) & (centers <= upper + tolerance)
        if not np.any(mask):
            raise ValueError("UniformCurrentSource support does not include any resolved grid cells")
        weights = np.ones(int(np.count_nonzero(mask)), dtype=float)
        indices = np.nonzero(mask)[0]
    return CompiledAxisPlacement(
        indices=tuple(int(index) for index in indices),
        weights=tuple(float(weight) for weight in weights),
        cell_sizes=tuple(float(cell_sizes[index]) for index in indices),
        centers=tuple(float(centers[index]) for index in indices),
    )


def _transverse_extent(source: UniformCurrentSource, grid: ResolvedGrid) -> float:
    placements = []
    axes = (grid.x, grid.y, grid.z)
    for axis in range(3):
        if axis == source.component_axis:
            continue
        extent = source.size[axis]
        if extent > 0.0:
            placements.append(extent)
            continue
        centers = _axis_centers(axes[axis])
        cell_sizes = _axis_sizes(axes[axis])
        index = int(np.argmin(np.abs(centers - source.center[axis])))
        placements.append(float(cell_sizes[index]))
    measure = 1.0
    for extent in placements:
        measure *= extent
    return measure


def compile_uniform_current_source(
    source: UniformCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledUniformCurrentSource:
    """Compile a uniform current source onto the resolved primal-cell grid."""

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    amplitude_scale = 1.0
    if source.current_amplitude_definition == "total":
        transverse_extent = _transverse_extent(source, grid)
        if transverse_extent <= 0.0:
            raise ValueError("UniformCurrentSource total-current normalization needs positive extent")
        amplitude_scale = 1.0 / transverse_extent

    return CompiledUniformCurrentSource(
        source=source,
        field_kind=source.field_kind,
        component_axis=source.component_axis,
        placement_kind=source.placement_kind,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        amplitude_scale=amplitude_scale,
    )


def compile_point_dipole(
    source: PointDipole,
    *,
    grid: ResolvedGrid,
) -> CompiledPointDipole:
    """Compile a point dipole source onto the resolved primal-cell grid.

    PointDipole always has zero size, so it uses interpolation to distribute
    amplitude across neighboring cell centers when interpolate=True.
    """

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=0.0,  # PointDipole is always zero size
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    return CompiledPointDipole(
        source=source,
        field_kind=source.field_kind,
        component_axis=source.component_axis,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
    )


def _compile_custom_current_source_data(
    source: CustomCurrentSource,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Compile field data arrays from a CustomCurrentSource into numpy arrays."""

    e_field_data = None
    h_field_data = None

    if source.e_fields:
        e_field_data = {}
        for key, values in source.e_fields.items():
            e_field_data[key] = np.array(values, dtype=np.complex128)

    if source.h_fields:
        h_field_data = {}
        for key, values in source.h_fields.items():
            h_field_data[key] = np.array(values, dtype=np.complex128)

    return e_field_data, h_field_data


def compile_custom_current_source(
    source: CustomCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledCustomCurrentSource:
    """Compile a custom current source onto the resolved primal-cell grid."""

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    e_field_data, h_field_data = _compile_custom_current_source_data(source)

    return CompiledCustomCurrentSource(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        has_electric=source.has_electric,
        has_magnetic=source.has_magnetic,
        e_field_data=e_field_data,
        h_field_data=h_field_data,
    )


def _compile_custom_field_source_data(
    source: CustomFieldSource,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Compile field data arrays from a CustomFieldSource into numpy arrays."""

    e_field_data = None
    h_field_data = None

    if source.e_fields:
        e_field_data = {}
        for key, values in source.e_fields.items():
            e_field_data[key] = np.array(values, dtype=np.complex128)

    if source.h_fields:
        h_field_data = {}
        for key, values in source.h_fields.items():
            h_field_data[key] = np.array(values, dtype=np.complex128)

    return e_field_data, h_field_data


def compile_custom_field_source(
    source: CustomFieldSource,
    *,
    grid: ResolvedGrid,
) -> CompiledCustomFieldSource:
    """Compile a custom field source onto the resolved primal-cell grid.

    CustomFieldSource is a planar source using the equivalence principle.
    """

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    e_field_data, h_field_data = _compile_custom_field_source_data(source)

    return CompiledCustomFieldSource(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        has_electric=source.has_electric,
        has_magnetic=source.has_magnetic,
        has_tangential_fields=source.has_tangential_fields,
        e_field_data=e_field_data,
        h_field_data=h_field_data,
    )


@dataclass(frozen=True)
class CompiledModeSource:
    """Discrete mode source placement and amplitude metadata.

    ModeSource is a planar source that injects a guided mode field profile.
    The mode profile is obtained from the mode solver at the source's
    frequency, and the injection uses the equivalence principle.
    """

    source: ModeSource
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    mode_index: int
    mode_neff: float
    mode_power: float
    # Field component data as numpy arrays (from mode solver)
    e_field_data: dict[str, np.ndarray]
    h_field_data: dict[str, np.ndarray]
    # In-plane coordinates for field interpolation
    x_coords: tuple[float, ...]
    y_coords: tuple[float, ...]

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def compile_mode_source(
    source: "ModeSource",
    *,
    grid: ResolvedGrid,
    scene,
    sim_center: tuple[float, float, float],
    sim_size: tuple[float, float, float],
    dt: float,
) -> CompiledModeSource:
    """Compile a mode source onto the resolved primal-cell grid.

    Uses the mode solver to compute the mode field profile at the
    source's central frequency, then compiles placement and field data.

    Args:
        source: The ModeSource to compile
        grid: The resolved simulation grid
        scene: The scene containing structures for epsilon sampling
        sim_center: Simulation domain center
        sim_size: Simulation domain size
        dt: Timestep for coefficient compilation

    Returns:
        CompiledModeSource with mode profile and placement data
    """
    # Local imports to avoid circular dependency
    from autofdtd.modes.epsilon import sample_scene_epsilon_tensor_2d
    from autofdtd.modes.models import ModeSolverConfig, ModeSolverCrossSection
    from autofdtd.modes.solver import solve_modes

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    # Build cross-section for mode solver
    normal_axis = source.injection_axis
    cross_section = ModeSolverCrossSection(
        normal_axis=normal_axis,
        position=source.center[normal_axis],
        bend_radius=source.bend_radius,
        bend_axis=source.bend_axis,
    )

    # Build mode solver config
    wavelength = 2.998e8 / source.source_time.freq0 if source.source_time.freq0 > 0 else 1.55e-6
    mode_spec = source.mode_spec
    config = ModeSolverConfig(
        wavelength=wavelength,
        cross_section=cross_section,
        mode_spec=mode_spec,
    )

    # Get epsilon callback for the cross-section
    epsilon_cb, x_coords, y_coords, cell_size = sample_scene_epsilon_tensor_2d(
        scene=scene,
        cross_section=cross_section,
        sim_center=sim_center,
        sim_size=sim_size,
        wavelength=wavelength,
        dt=dt,
    )

    # Solve for modes
    try:
        modes = solve_modes(config, epsilon_cb, x_coords, y_coords)
    except Exception as exc:
        raise RuntimeError(f"ModeSource mode solver failed: {exc}") from exc

    if not modes:
        raise RuntimeError("ModeSource mode solver found no modes")

    # Select the target mode
    if source.mode_index >= len(modes):
        raise ValueError(
            f"ModeSource mode_index={source.mode_index} but only {len(modes)} modes available"
        )

    selected_mode = modes[source.mode_index]
    mode_neff = selected_mode.neff_real
    mode_power = selected_mode.power

    # Extract field components
    e_field_data: dict[str, np.ndarray] = {}
    h_field_data: dict[str, np.ndarray] = {}

    # The field arrays in ModeSolution are flattened (real, imag) tuples
    # The mode solver produces fields on an nx x ny grid matching the
    # coordinate arrays, then downsamples to (nx-1, ny-1) cell-centered grids
    nx = len(x_coords)
    ny = len(y_coords)
    # All field components (E and H) are downsampled to cell-centered (nx-1, ny-1)
    # Note: numpy arrays are indexed as [ny, nx]; reshape order is (ny, nx)
    field_nx = ny - 1
    field_ny = nx - 1

    def _parse_field_component(
        component_data: tuple[tuple[float, float], ...],
        name: str = "unknown",
    ) -> np.ndarray:
        """Parse (real, imag) tuple array into complex numpy array."""
        actual_len = len(component_data)
        parsed = np.zeros(actual_len, dtype=np.complex128)
        for i, (re, im) in enumerate(component_data):
            parsed[i] = complex(re, im)
        # numpy arrays are indexed as [ny, nx]; mode solver returns field_nx * field_ny elements
        # with x varying fastest (C order), so reshape to (field_ny, field_nx) = (nx-1, ny-1)
        return parsed.reshape((field_ny, field_nx))

    e_field_data["Ex"] = _parse_field_component(selected_mode.Ex, "Ex")
    e_field_data["Ey"] = _parse_field_component(selected_mode.Ey, "Ey")
    e_field_data["Ez"] = _parse_field_component(selected_mode.Ez, "Ez")
    h_field_data["Hx"] = _parse_field_component(selected_mode.Hx, "Hx")
    h_field_data["Hy"] = _parse_field_component(selected_mode.Hy, "Hy")
    h_field_data["Hz"] = _parse_field_component(selected_mode.Hz, "Hz")

    return CompiledModeSource(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        mode_index=source.mode_index,
        mode_neff=mode_neff,
        mode_power=mode_power,
        e_field_data=e_field_data,
        h_field_data=h_field_data,
        x_coords=x_coords,
        y_coords=y_coords,
    )


@dataclass(frozen=True)
class CompiledPlaneWave:
    """Discrete plane wave source placement and amplitude metadata.

    A PlaneWave is a planar source that injects a uniform electromagnetic
    wave with a defined propagation direction and polarization.
    """

    source: PlaneWave
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    is_fixed_angle: bool
    # Frequency-dependent k-vector components for injection
    kx_func: object  # Callable that returns kx at a given frequency
    ky_func: object  # Callable that returns ky at a given frequency
    kz_func: object  # Callable that returns kz at a given frequency
    # Angular spec type
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"]

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def _plane_wave_k_from_angles(
    angle_theta: float,
    angle_phi: float,
    direction: Literal["+", "-"],
    injection_axis: int,
    n_medium: float = 1.0,
) -> tuple[float, float, float]:
    """Compute k-vector components from angles for a plane wave.

    Args:
        angle_theta: Polar angle from injection axis
        angle_phi: Azimuth angle in plane perpendicular to injection axis
        direction: Propagation direction along injection axis (+ or -)
        injection_axis: Which axis is the injection normal (0=x, 1=y, 2=z)
        n_medium: Refractive index of the background medium

    Returns:
        kx, ky, kz components of the wavevector
    """
    c0 = 2.998e8  # speed of light in vacuum

    # Direction multiplier
    dir_mult = 1.0 if direction == "+" else -1.0

    # For a plane wave propagating at angles theta, phi relative to injection axis:
    # kx = k * sin(theta) * cos(phi)
    # ky = k * sin(theta) * sin(phi)
    # kz = k * cos(theta) for injection along z
    # Then we rotate to the actual injection axis

    sin_theta = math.sin(angle_theta)
    cos_theta = math.cos(angle_theta)
    sin_phi = math.sin(angle_phi)
    cos_phi = math.cos(angle_phi)

    # k magnitude in the medium
    k_mag = dir_mult * n_medium * 2 * math.pi / c0

    # Temporary k vector assuming injection along z
    kx_t = k_mag * sin_theta * cos_phi
    ky_t = k_mag * sin_theta * sin_phi
    kz_t = k_mag * cos_theta

    # Rotate to actual injection axis
    if injection_axis == 0:
        # Injection along x: kx=z, ky=y, kz=x -> we need kz=kx_t, kx=ky_t, ky=kz_t
        return (kz_t, ky_t, kx_t)
    elif injection_axis == 1:
        # Injection along y: kx=x, ky=z, kz=y -> we need kx=kx_t, ky=kz_t, kz=ky_t
        return (kx_t, kz_t, ky_t)
    else:
        # Injection along z: kx=x, ky=y, kz=z
        return (kx_t, ky_t, kz_t)


def compile_plane_wave(
    source: PlaneWave,
    *,
    grid: ResolvedGrid,
) -> CompiledPlaneWave:
    """Compile a plane wave source onto the resolved primal-cell grid.

    Args:
        source: The PlaneWave to compile
        grid: The resolved simulation grid

    Returns:
        CompiledPlaneWave with placement and direction data
    """
    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    # Angular spec type for frequency handling
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"] = (
        "FixedAngle" if source.is_fixed_angle else "FixedInPlaneK"
    )

    # The k-vector functions will be evaluated at runtime per frequency
    # For now, store the parameters for later evaluation
    def _make_k_funcs(inj_axis: int, direction: Literal["+", "-"]):
        """Create k-vector functions for the plane wave at a given frequency."""

        def kx_func(freq: float, n: float = 1.0) -> float:
            kx, _, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kx / (2 * math.pi * freq) if freq > 0 else 0.0

        def ky_func(freq: float, n: float = 1.0) -> float:
            _, ky, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return ky / (2 * math.pi * freq) if freq > 0 else 0.0

        def kz_func(freq: float, n: float = 1.0) -> float:
            _, _, kz = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kz / (2 * math.pi * freq) if freq > 0 else 0.0

        return kx_func, ky_func, kz_func

    kx_func, ky_func, kz_func = _make_k_funcs(source.injection_axis, source.direction)

    return CompiledPlaneWave(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        dir_vector=source._dir_vector,
        pol_vector=source._pol_vector,
        is_fixed_angle=source.is_fixed_angle,
        kx_func=kx_func,
        ky_func=ky_func,
        kz_func=kz_func,
        angular_spec_type=angular_spec_type,
    )


@dataclass(frozen=True)
class CompiledGaussianBeam:
    """Discrete Gaussian beam source placement and amplitude metadata.

    A GaussianBeam is a planar source that injects a spatially Gaussian-shaped
    electromagnetic wave with a defined propagation direction, polarization,
    and waist parameters.
    """

    source: GaussianBeam
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    beam_weights: tuple[float, ...]  # Gaussian envelope weights
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    is_fixed_angle: bool
    waist_radius: float
    waist_distance: float
    # Frequency-dependent k-vector components for injection
    kx_func: object  # Callable that returns kx at a given frequency
    ky_func: object  # Callable that returns ky at a given frequency
    kz_func: object  # Callable that returns kz at a given frequency
    # Angular spec type
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"]

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def _compute_gaussian_weights(
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement],
    injection_axis: int,
    waist_radius: float,
    waist_distance: float,
) -> tuple[float, ...]:
    """Compute Gaussian envelope weights for beam cross-section.

    Args:
        axis_placements: The compiled axis placements for x, y, z
        injection_axis: The injection axis (0=x, 1=y, 2=z)
        waist_radius: Beam waist radius
        waist_distance: Distance from waist to injection plane

    Returns:
        Tuple of Gaussian weights for each placement (matches placements length)
    """
    # For beam injection, we need to compute the Gaussian amplitude at each
    # cell center in the tangential plane, multiplied by the interpolation weights
    # from all three axes (to match the number of placements)
    weights: list[float] = []

    # axis_placements[axis].centers is already the subset of centers for this placement
    # (same length as indices and weights for that axis)
    # We iterate by position (0, 1, 2, ...) not by cell index
    for x_pos in range(len(axis_placements[0].weights)):
        for y_pos in range(len(axis_placements[1].weights)):
            for z_pos in range(len(axis_placements[2].weights)):
                # Get centers for Gaussian computation
                x_c = axis_placements[0].centers[x_pos]
                y_c = axis_placements[1].centers[y_pos]
                z_c = axis_placements[2].centers[z_pos]

                # Get interpolation weights
                x_w = axis_placements[0].weights[x_pos]
                y_w = axis_placements[1].weights[y_pos]
                z_w = axis_placements[2].weights[z_pos]

                # Radial distance from beam center in the transverse plane
                # For injection axis, the position is along injection direction
                # For tangential axes, we compute distance from beam center
                r_squared = 0.0
                if injection_axis != 0:
                    r_squared += x_c * x_c
                if injection_axis != 1:
                    r_squared += y_c * y_c
                if injection_axis != 2:
                    r_squared += z_c * z_c

                # Gaussian envelope: exp(-2 * r^2 / w^2)
                gaussian = math.exp(-2.0 * r_squared / (waist_radius * waist_radius))

                # Combined weight: Gaussian * all interpolation weights
                weight = gaussian * x_w * y_w * z_w
                weights.append(weight)

    return tuple(weights)


def compile_gaussian_beam(
    source: GaussianBeam,
    *,
    grid: ResolvedGrid,
) -> CompiledGaussianBeam:
    """Compile a Gaussian beam source onto the resolved primal-cell grid.

    Args:
        source: The GaussianBeam to compile
        grid: The resolved simulation grid

    Returns:
        CompiledGaussianBeam with placement and direction data
    """
    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    # Compute Gaussian beam weights
    beam_weights = _compute_gaussian_weights(
        axis_placements,
        source.injection_axis,
        source.waist_radius,
        source.waist_distance,
    )

    # Angular spec type for frequency handling
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"] = (
        "FixedAngle" if source.is_fixed_angle else "FixedInPlaneK"
    )

    def _make_k_funcs(inj_axis: int, direction: Literal["+", "-"]):
        """Create k-vector functions for the beam at a given frequency."""

        def kx_func(freq: float, n: float = 1.0) -> float:
            kx, _, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kx / (2 * math.pi * freq) if freq > 0 else 0.0

        def ky_func(freq: float, n: float = 1.0) -> float:
            _, ky, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return ky / (2 * math.pi * freq) if freq > 0 else 0.0

        def kz_func(freq: float, n: float = 1.0) -> float:
            _, _, kz = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kz / (2 * math.pi * freq) if freq > 0 else 0.0

        return kx_func, ky_func, kz_func

    kx_func, ky_func, kz_func = _make_k_funcs(source.injection_axis, source.direction)

    return CompiledGaussianBeam(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        beam_weights=beam_weights,
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        dir_vector=source._dir_vector,
        pol_vector=source._pol_vector,
        is_fixed_angle=source.is_fixed_angle,
        waist_radius=source.waist_radius,
        waist_distance=source.waist_distance,
        kx_func=kx_func,
        ky_func=ky_func,
        kz_func=kz_func,
        angular_spec_type=angular_spec_type,
    )


@dataclass(frozen=True)
class CompiledAstigmaticGaussianBeam:
    """Discrete astigmatic Gaussian beam source placement and amplitude metadata.

    An AstigmaticGaussianBeam is a planar source that injects a spatially
    Gaussian-shaped electromagnetic wave with separate waist radii in x and y.
    """

    source: AstigmaticGaussianBeam
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    beam_weights: tuple[float, ...]  # Gaussian envelope weights
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    is_fixed_angle: bool
    waist_radius_x: float
    waist_radius_y: float
    waist_distance_x: float
    waist_distance_y: float
    # Frequency-dependent k-vector components for injection
    kx_func: object  # Callable that returns kx at a given frequency
    ky_func: object  # Callable that returns ky at a given frequency
    kz_func: object  # Callable that returns kz at a given frequency
    # Angular spec type
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"]

    @property
    def placement_kind(self) -> Literal["sheet"]:
        """Always a sheet (planar) placement."""
        return "sheet"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def _compute_astigmatic_gaussian_weights(
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement],
    injection_axis: int,
    waist_radius_x: float,
    waist_radius_y: float,
    waist_distance_x: float,
    waist_distance_y: float,
) -> tuple[float, ...]:
    """Compute astigmatic Gaussian envelope weights for beam cross-section.

    Args:
        axis_placements: The compiled axis placements for x, y, z
        injection_axis: The injection axis (0=x, 1=y, 2=z)
        waist_radius_x: Beam waist radius in x direction
        waist_radius_y: Beam waist radius in y direction
        waist_distance_x: Distance from waist to injection plane in x
        waist_distance_y: Distance from waist to injection plane in y

    Returns:
        Tuple of Gaussian weights for each placement
    """
    # Get the two tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)

    # Extract centers for tangential axes
    centers = [
        axis_placements[tang_axes[0]].centers,
        axis_placements[tang_axes[1]].centers,
    ]

    weights: list[float] = []

    # For beam injection, we compute the astigmatic Gaussian amplitude
    # at each cell center in the tangential plane
    for x_c in centers[0]:
        for y_c in centers[1]:
            # Astigmatic Gaussian: exp(-2 * (x^2/wx^2 + y^2/wy^2))
            weight = math.exp(
                -2.0 * x_c * x_c / (waist_radius_x * waist_radius_x)
                - 2.0 * y_c * y_c / (waist_radius_y * waist_radius_y)
            )
            weights.append(weight)

    return tuple(weights)


def compile_astigmatic_gaussian_beam(
    source: AstigmaticGaussianBeam,
    *,
    grid: ResolvedGrid,
) -> CompiledAstigmaticGaussianBeam:
    """Compile an astigmatic Gaussian beam source onto the resolved primal-cell grid.

    Args:
        source: The AstigmaticGaussianBeam to compile
        grid: The resolved simulation grid

    Returns:
        CompiledAstigmaticGaussianBeam with placement and direction data
    """
    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    # Compute astigmatic Gaussian beam weights
    beam_weights = _compute_astigmatic_gaussian_weights(
        axis_placements,
        source.injection_axis,
        source.waist_radius_x,
        source.waist_radius_y,
        source.waist_distance_x,
        source.waist_distance_y,
    )

    # Angular spec type for frequency handling
    angular_spec_type: Literal["FixedInPlaneK", "FixedAngle"] = (
        "FixedAngle" if source.is_fixed_angle else "FixedInPlaneK"
    )

    def _make_k_funcs(inj_axis: int, direction: Literal["+", "-"]):
        """Create k-vector functions for the beam at a given frequency."""

        def kx_func(freq: float, n: float = 1.0) -> float:
            kx, _, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kx / (2 * math.pi * freq) if freq > 0 else 0.0

        def ky_func(freq: float, n: float = 1.0) -> float:
            _, ky, _ = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return ky / (2 * math.pi * freq) if freq > 0 else 0.0

        def kz_func(freq: float, n: float = 1.0) -> float:
            _, _, kz = _plane_wave_k_from_angles(
                source.angle_theta, source.angle_phi, direction, inj_axis, n
            )
            return kz / (2 * math.pi * freq) if freq > 0 else 0.0

        return kx_func, ky_func, kz_func

    kx_func, ky_func, kz_func = _make_k_funcs(source.injection_axis, source.direction)

    return CompiledAstigmaticGaussianBeam(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        beam_weights=beam_weights,
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        dir_vector=source._dir_vector,
        pol_vector=source._pol_vector,
        is_fixed_angle=source.is_fixed_angle,
        waist_radius_x=source.waist_radius_x,
        waist_radius_y=source.waist_radius_y,
        waist_distance_x=source.waist_distance_x,
        waist_distance_y=source.waist_distance_y,
        kx_func=kx_func,
        ky_func=ky_func,
        kz_func=kz_func,
        angular_spec_type=angular_spec_type,
    )


@dataclass(frozen=True)
class CompiledTFSF:
    """Discrete TFSF source placement and amplitude metadata.

    A TFSF source is a volume source that separates total-field (inside the source
    region) from scattered-field (outside the source region). The incident plane wave
    is injected at the injection plane and the boundary interaction cancels the incident
    field at the TFSF box edges.
    """

    source: TFSF
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    injection_axis: int
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    num_freqs: int
    injection_plane_center: tuple[float, float, float]
    # TFSF boundary indices for field splitting
    # The six faces of the TFSF box
    tfsf_bounds_indices: tuple[
        tuple[int, int, int],  # lower corner (imin, jmin, kmin)
        tuple[int, int, int],  # upper corner (imax, jmax, kmax)
    ]

    @property
    def placement_kind(self) -> Literal["volume"]:
        """Always a volume (3D) placement for TFSF."""
        return "volume"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0])


def compile_tfsf(
    source: TFSF,
    *,
    grid: ResolvedGrid,
) -> CompiledTFSF:
    """Compile a TFSF source onto the resolved primal-cell grid.

    Args:
        source: The TFSF to compile
        grid: The resolved simulation grid

    Returns:
        CompiledTFSF with placement and TFSF boundary data
    """
    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    # Compute TFSF boundary indices
    # The TFSF bounds indices define the corners of the TFSF volume
    tfsf_bounds_indices = (
        (
            axis_placements[0].indices[0],
            axis_placements[1].indices[0],
            axis_placements[2].indices[0],
        ),
        (
            axis_placements[0].indices[-1],
            axis_placements[1].indices[-1],
            axis_placements[2].indices[-1],
        ),
    )

    return CompiledTFSF(
        source=source,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        injection_axis=source.injection_axis,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        dir_vector=source._dir_vector,
        pol_vector=source._pol_vector,
        num_freqs=source.num_freqs,
        injection_plane_center=source.injection_plane_center,
        tfsf_bounds_indices=tfsf_bounds_indices,
    )
