"""Monitor compilation helpers for Phase 1 field recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from autofdtd.grid import ResolvedGrid
from autofdtd.kernels.backend import WARP_AVAILABLE


@dataclass(frozen=True)
class CompiledFieldMonitor:
    """Discrete field monitor placement and recording metadata."""

    name: str
    monitor_type: Literal["FieldMonitor", "FieldTimeMonitor", "AuxFieldTimeMonitor"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Cell indices and weights for the monitored region
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    # Fields to record
    fields: tuple[str, ...]
    # Interval and start for time-domain monitors
    interval: int
    start: int
    # Whether to overwrite existing data
    overwrite: bool = True
    # For frequency-domain monitors, the frequency points
    freqs: tuple[float, ...] = ()
    num_freqs: int = 1

    @property
    def is_time_domain(self) -> bool:
        return self.monitor_type in {"FieldTimeMonitor", "AuxFieldTimeMonitor"}

    @property
    def is_frequency_domain(self) -> bool:
        # Only true when FieldMonitor has explicit frequency points
        return self.monitor_type == "FieldMonitor" and len(self.freqs) > 0

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def needs_dft(self) -> bool:
        """Whether this monitor needs DFT accumulation for frequency-domain output."""
        return self.monitor_type == "FieldMonitor" and len(self.freqs) > 0


@dataclass(frozen=True)
class CompiledAxisPlacementMonitor:
    """Discrete support metadata for one monitor axis."""

    indices: tuple[int, ...]
    weights: tuple[float, ...]


def _axis_centers_monitor(axis_boundaries: tuple[float, ...]) -> np.ndarray:
    """Compute cell centers from boundaries."""
    boundaries = np.asarray(axis_boundaries, dtype=float)
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def _compile_monitor_axis_placement(
    axis_boundaries: tuple[float, ...],
    *,
    center: float,
    size: float,
) -> CompiledAxisPlacementMonitor:
    """Compile monitor axis placement onto a grid axis."""
    boundaries = np.asarray(axis_boundaries, dtype=float)
    centers = _axis_centers_monitor(axis_boundaries)

    tolerance = 1e-12

    if size <= tolerance:
        # Zero-size: record at center cell
        index = int(np.argmin(np.abs(centers - center)))
        return CompiledAxisPlacementMonitor(
            indices=(index,),
            weights=(1.0,),
        )

    # Non-zero size: record over volume
    lower = center - size / 2.0
    upper = center + size / 2.0
    mask = (centers >= lower - tolerance) & (centers <= upper + tolerance)
    if not np.any(mask):
        # Fall back to center cell if no overlap
        index = int(np.argmin(np.abs(centers - center)))
        return CompiledAxisPlacementMonitor(
            indices=(index,),
            weights=(1.0,),
        )
    indices = np.nonzero(mask)[0]
    weights = np.ones(len(indices), dtype=float)
    return CompiledAxisPlacementMonitor(
        indices=tuple(int(i) for i in indices),
        weights=tuple(float(w) for w in weights),
    )


def compile_field_monitor(
    name: str,
    monitor_type: Literal["FieldMonitor", "FieldTimeMonitor", "AuxFieldTimeMonitor"],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    fields: tuple[str, ...],
    *,
    interval: int = 1,
    start: int = 0,
    overwrite: bool = True,
    freqs: tuple[float, ...] = (),
    grid: ResolvedGrid,
) -> CompiledFieldMonitor:
    """Compile a field monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: FieldMonitor, FieldTimeMonitor, or AuxFieldTimeMonitor
        center: Monitor center position
        size: Monitor size (0 for point, non-zero for volume/plane)
        fields: Field components to record
        interval: Recording interval in timesteps
        start: First timestep to start recording
        overwrite: Whether to overwrite existing data
        freqs: Frequency points for frequency-domain monitors
        grid: The resolved simulation grid

    Returns:
        CompiledFieldMonitor with placement and metadata
    """
    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    num_freqs = len(freqs) if freqs else 1

    return CompiledFieldMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        fields=fields,
        interval=interval,
        start=start,
        overwrite=overwrite,
        freqs=freqs,
        num_freqs=num_freqs,
    )


@dataclass(frozen=True)
class FieldMonitorState:
    """Runtime state for a field monitor.

    For time-domain monitors, stores field time-series.
    For frequency-domain monitors, accumulates DFT terms.
    """

    compiled: CompiledFieldMonitor
    # Time-domain storage: list of field snapshots with timestamps
    time_series: dict[str, list[np.ndarray]] | None = None
    time_stamps: list[float] | None = None
    # Frequency-domain storage: DFT accumulator arrays
    dft_data: dict[str, np.ndarray] | None = None
    dft_count: int = 0

    def __post_init__(self):
        if self.compiled.is_time_domain:
            object.__setattr__(
                self,
                "time_series",
                {field: [] for field in self.compiled.fields},
            )
            object.__setattr__(self, "time_stamps", [])
        else:
            # Frequency-domain: DFT accumulator
            n_pts = len(self.compiled.placements)
            n_freqs = self.compiled.num_freqs
            object.__setattr__(
                self,
                "dft_data",
                {
                    field: np.zeros((n_pts, n_freqs), dtype=np.complex128)
                    for field in self.compiled.fields
                },
            )
            object.__setattr__(self, "dft_count", 0)

    @property
    def dft_count_value(self) -> int:
        """Access dft_count without triggering frozen dataclass issue."""
        return object.__getattribute__(self, "dft_count")

    def _set_dft_count(self, value: int) -> None:
        """Set dft_count via object.__setattr__ to bypass frozen."""
        object.__setattr__(self, "dft_count", value)

    def record_time_domain(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Record field values at the current timestep for time-domain monitor."""
        if not self.compiled.is_time_domain:
            return

        for field in self.compiled.fields:
            values = self._extract_field_values(
                electric_field, magnetic_field, field
            )
            self.time_series[field].append(values.copy())

        self.time_stamps.append(time)

    def accumulate_dft(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Accumulate DFT terms for frequency-domain monitor."""
        if not self.compiled.is_frequency_domain:
            return

        for field in self.compiled.fields:
            values = self._extract_field_values(
                electric_field, magnetic_field, field
            )
            # DFT accumulation: sum over timesteps
            omega_t = 2.0 * np.pi * np.asarray(self.compiled.freqs) * time
            phase = np.cos(omega_t) - 1j * np.sin(omega_t)
            # dft_data shape: (n_pts, n_freqs), values shape: (n_pts,), phase shape: (n_freqs,)
            # Always expand for proper broadcasting: values[:, None] * phase[None, :]
            self.dft_data[field] += values[:, None] * phase[None, :]

        self._set_dft_count(self.dft_count_value + 1)

    def _extract_field_values(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        field: str,
    ) -> np.ndarray:
        """Extract field component values at monitor placements."""
        if field not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            return np.zeros(len(self.compiled.placements), dtype=np.complex128)

        is_electric = field[1].lower() == "x"  # Ex, Ey, Ez are electric

        # Convert Warp arrays to numpy if needed
        if WARP_AVAILABLE:
            import warp as wp

            if isinstance(electric_field, wp.array):
                electric_field = electric_field.numpy()
            if isinstance(magnetic_field, wp.array):
                magnetic_field = magnetic_field.numpy()

        if len(self.compiled.placements) == 1:
            idx = self.compiled.placements[0]
            if is_electric:
                return np.array([electric_field[idx][0 if field == "Ex" else 1 if field == "Ey" else 2]])
            else:
                return np.array([magnetic_field[idx][0 if field == "Hx" else 1 if field == "Hy" else 2]])

        values = np.zeros(len(self.compiled.placements), dtype=np.complex128)
        for i, idx in enumerate(self.compiled.placements):
            comp = 0 if field[1] == "x" else 1 if field[1] == "y" else 2
            if is_electric:
                values[i] = electric_field[idx][comp]
            else:
                values[i] = magnetic_field[idx][comp]

        return values

    def to_field_data(self) -> dict[str, tuple[tuple[float, float], ...]]:
        """Convert recorded data to FieldData format with (real, imag) tuples."""
        n_pts = len(self.compiled.placements)

        if self.compiled.is_time_domain:
            result = {}
            for field in self.compiled.fields:
                series = self.time_series[field]
                if not series:
                    result[field] = ()
                    continue
                # Stack all timesteps: shape (n_timesteps, n_pts)
                stacked = np.stack(series, axis=0)
                # Convert to (real, imag) tuples
                tuples = tuple(
                    (float(v.real), float(v.imag))
                    for v in stacked.flatten()
                )
                result[field] = tuples
            return result
        else:
            # Frequency-domain: normalize DFT
            # dft_data shape: (n_pts, n_freqs)
            result = {}
            if self.dft_count_value > 0:
                for field in self.compiled.fields:
                    normalized = self.dft_data[field] / float(self.dft_count_value)
                    tuples = tuple(
                        (float(v.real), float(v.imag))
                        for v in normalized.flatten()
                    )
                    result[field] = tuples
            else:
                for field in self.compiled.fields:
                    result[field] = ()
            return result

    def time_data(self) -> tuple[tuple[float, ...], dict[str, tuple[tuple[float, float], ...]]]:
        """Return time stamps and field data for time-domain monitors."""
        if not self.compiled.is_time_domain:
            return (), {}

        time_stamps = tuple(self.time_stamps) if self.time_stamps else ()

        result = {}
        for field in self.compiled.fields:
            series = self.time_series[field]
            if not series:
                result[field] = ()
                continue
            stacked = np.stack(series, axis=0)
            tuples = tuple(
                (float(v.real), float(v.imag))
                for v in stacked.flatten()
            )
            result[field] = tuples

        return time_stamps, result


# --------------------------------------------------------------------
# Flux monitor compilation
# --------------------------------------------------------------------


def _infer_flux_normal_axis(size: tuple[float, float, float]) -> int:
    """Infer the normal axis for a flux monitor from its size.

    A flux monitor measures flux through a 2D surface. The normal axis
    is the one with zero extent.

    Args:
        size: Monitor size tuple

    Returns:
        Axis index (0=x, 1=y, 2=z)

    Raises:
        ValueError: If size has no zero extent (not a surface)
    """
    for axis, s in enumerate(size):
        if s <= 1e-12:
            return axis
    raise ValueError(
        f"FluxMonitor requires a 2D surface (one zero-size dimension), got size={size}"
    )


@dataclass(frozen=True)
class CompiledFluxMonitor:
    """Discrete flux monitor placement and integration metadata."""

    name: str
    monitor_type: Literal["FluxMonitor", "FluxTimeMonitor"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Normal axis for flux integration (0=x, 1=y, 2=z)
    normal_axis: int
    # Direction for flux sign convention
    direction: Literal["+", "-"]
    # Cell indices and weights for the surface
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    # Interval and start for time-domain monitors
    interval: int
    start: int
    # For frequency-domain monitors (FluxMonitor with freqs)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @property
    def is_time_domain(self) -> bool:
        return self.monitor_type == "FluxTimeMonitor"

    @property
    def is_frequency_domain(self) -> bool:
        return self.monitor_type == "FluxMonitor" and len(self.freqs) > 0

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def needs_dft(self) -> bool:
        """Whether this monitor needs DFT accumulation for frequency-domain output."""
        return self.is_frequency_domain


def compile_flux_monitor(
    name: str,
    monitor_type: Literal["FluxMonitor", "FluxTimeMonitor"],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    direction: Literal["+", "-"] = "+",
    interval: int = 1,
    start: int = 0,
    freqs: tuple[float, ...] = (),
    grid: ResolvedGrid,
) -> CompiledFluxMonitor:
    """Compile a flux monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: FluxMonitor or FluxTimeMonitor
        center: Monitor center position
        size: Monitor size (must have exactly one zero dimension for surface)
        direction: Flux direction (+ or -)
        interval: Recording interval in timesteps
        start: First timestep to start recording
        freqs: Frequency points for frequency-domain monitors
        grid: The resolved simulation grid

    Returns:
        CompiledFluxMonitor with placement and metadata
    """
    normal_axis = _infer_flux_normal_axis(size)

    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    num_freqs = len(freqs) if freqs else 1

    return CompiledFluxMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        normal_axis=normal_axis,
        direction=direction,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        interval=interval,
        start=start,
        num_freqs=num_freqs,
        freqs=freqs,
    )


@dataclass(frozen=True)
class FluxMonitorState:
    """Runtime state for a flux monitor.

    For time-domain monitors, stores flux time-series.
    For frequency-domain monitors, accumulates DFT terms.
    """

    compiled: CompiledFluxMonitor
    # Time-domain storage: flux values and time stamps
    flux_series: list[float] | None = None
    time_stamps: list[float] | None = None
    # Frequency-domain storage: DFT accumulator
    dft_flux: np.ndarray | None = None
    dft_count: int = 0

    def __post_init__(self):
        if self.compiled.is_time_domain:
            object.__setattr__(self, "flux_series", [])
            object.__setattr__(self, "time_stamps", [])
        else:
            # Frequency-domain: DFT accumulator
            n_freqs = self.compiled.num_freqs
            object.__setattr__(
                self,
                "dft_flux",
                np.zeros(n_freqs, dtype=np.complex128),
            )
            object.__setattr__(self, "dft_count", 0)

    @property
    def dft_count_value(self) -> int:
        """Access dft_count without triggering frozen dataclass issue."""
        return object.__getattribute__(self, "dft_count")

    def _set_dft_count(self, value: int) -> None:
        """Set dft_count via object.__setattr__ to bypass frozen."""
        object.__setattr__(self, "dft_count", value)

    def record_flux_time_domain(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Record flux value at the current timestep for time-domain monitor."""
        if not self.compiled.is_time_domain:
            return

        from autofdtd.kernels.monitors import accumulate_flux

        flux_value = accumulate_flux(
            electric_field,
            magnetic_field,
            self.compiled.direction,
            self.compiled.normal_axis,
            self.compiled.placements,
        )

        self.flux_series.append(flux_value)
        self.time_stamps.append(time)

    def accumulate_flux_dft(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Accumulate DFT terms for frequency-domain flux monitor."""
        if not self.compiled.is_frequency_domain:
            return

        from autofdtd.kernels.monitors import accumulate_flux

        flux_value = accumulate_flux(
            electric_field,
            magnetic_field,
            self.compiled.direction,
            self.compiled.normal_axis,
            self.compiled.placements,
        )

        # DFT accumulation
        omega_t = 2.0 * np.pi * np.asarray(self.compiled.freqs) * time
        for j, freq in enumerate(self.compiled.freqs):
            phase = np.exp(-1j * omega_t[j])
            self.dft_flux[j] += flux_value * phase

        self._set_dft_count(self.dft_count_value + 1)

    def to_flux_data(self) -> dict[str, Any]:
        """Convert recorded data to FluxData format."""
        if self.compiled.is_time_domain:
            return {
                "flux": tuple(self.flux_series) if self.flux_series else (),
                "t": tuple(self.time_stamps) if self.time_stamps else (),
            }
        else:
            # Frequency-domain: normalize DFT
            if self.dft_count_value > 0:
                normalized = self.dft_flux / float(self.dft_count_value)
                flux_tuples = tuple(
                    (float(v.real), float(v.imag))
                    for v in normalized
                )
            else:
                flux_tuples = ()
            return {"flux": flux_tuples, "t": ()}


# --------------------------------------------------------------------
# Medium/Permittivity monitor compilation
# --------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledMediumMonitor:
    """Discrete medium monitor placement and frequency metadata."""

    name: str
    monitor_type: Literal["MediumMonitor", "PermittivityMonitor"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Cell indices and weights for the monitored region
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    # Interval and start
    interval: int
    start: int
    # Frequency-domain params
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @property
    def is_permittivity_only(self) -> bool:
        return self.monitor_type == "PermittivityMonitor"

    @property
    def support_point_count(self) -> int:
        return len(self.placements)


def compile_medium_monitor(
    name: str,
    monitor_type: Literal["MediumMonitor", "PermittivityMonitor"],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    interval: int = 1,
    start: int = 0,
    num_freqs: int = 1,
    freqs: tuple[float, ...] = (),
    grid: ResolvedGrid,
) -> CompiledMediumMonitor:
    """Compile a medium/permittivity monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: MediumMonitor or PermittivityMonitor
        center: Monitor center position
        size: Monitor size (0 for point, non-zero for volume/plane)
        interval: Recording interval in timesteps
        start: First timestep to start recording
        num_freqs: Number of frequency points
        freqs: Frequency points for frequency-domain monitors
        grid: The resolved simulation grid

    Returns:
        CompiledMediumMonitor with placement and metadata
    """
    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    return CompiledMediumMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        interval=interval,
        start=start,
        num_freqs=num_freqs,
        freqs=freqs,
    )


@dataclass(frozen=True)
class MediumMonitorState:
    """Runtime state for a medium/permittivity monitor.

    Stores the compiled medium coefficients and scene reference for
    computing permittivity and permeability values at each frequency.
    """

    compiled: CompiledMediumMonitor
    # Resolved medium coefficients at each placement for each frequency
    # Structure: (n_pts, num_freqs, 6) where 6 = [eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz]
    medium_data: np.ndarray | None = None
    # For permittivity-only monitors, only eps components are populated

    def __post_init__(self):
        n_pts = self.compiled.support_point_count
        n_freqs = self.compiled.num_freqs
        n_components = 3 if self.compiled.is_permittivity_only else 6
        object.__setattr__(
            self,
            "medium_data",
            np.zeros((n_pts, n_freqs, n_components), dtype=np.complex128),
        )

    def populate_from_scene(
        self,
        scene,
        scene_compiled: object,
    ) -> None:
        """Populate medium_data by sampling the scene at monitor placements.

        This should be called once after scene compilation, before the
        timestep loop starts.

        Args:
            scene: The Scene object
            scene_compiled: CompiledScene from the runtime compiler
        """
        from autofdtd.compiler.materials import sample_scene_mediums, compile_medium_coefficients
        from autofdtd.materials import medium_model_from_value

        placements = self.compiled.placements
        freqs = self.compiled.freqs
        grid_extent = self.compiled.size
        n_pts = len(placements)

        if len(placements) == 0:
            return

        # Get coordinate information from placements
        # For each placement, we need to compute the actual 3D position
        # We approximate using the center and size of the monitor
        center = self.compiled.center
        size = self.compiled.size

        # Build sample points at the center of each monitor cell
        sample_points = []
        for idx in placements:
            # Use grid cell center - approximate from index
            # This is a simplification; real implementation would need grid boundaries
            px = center[0]  # placeholder - would need actual grid
            py = center[1]
            pz = center[2]
            sample_points.append((px, py, pz))

        # Sample scene media at these points
        samples = sample_scene_mediums(scene, points=sample_points)

        # For each sample point, compute the complex permittivity and permeability
        # at each frequency
        n_freqs = self.compiled.num_freqs
        for i, (idx, sample) in enumerate(zip(placements, samples, strict=False)):
            if i >= n_pts:
                break

            medium = sample.medium

            # Try to compile medium coefficients
            try:
                # Use dt=1e-15 as a placeholder; the actual dt will be used at runtime
                coeffs = compile_medium_coefficients(medium, dt=1e-15)
            except Exception:
                # Fall back to simple permittivity
                normalized = medium_model_from_value(medium)
                eps_val = float(getattr(normalized, "permittivity", 1.0))
                mu_val = float(getattr(normalized, "permeability", 1.0))
                for f_idx in range(n_freqs):
                    self.medium_data[i, f_idx, 0] = complex(eps_val, 0.0)
                    self.medium_data[i, f_idx, 1] = complex(eps_val, 0.0)
                    self.medium_data[i, f_idx, 2] = complex(eps_val, 0.0)
                    if not self.compiled.is_permittivity_only:
                        self.medium_data[i, f_idx, 3] = complex(mu_val, 0.0)
                        self.medium_data[i, f_idx, 4] = complex(mu_val, 0.0)
                        self.medium_data[i, f_idx, 5] = complex(mu_val, 0.0)
                continue

            # Extract permittivity from coefficients
            if hasattr(coeffs, "permittivity"):
                eps_xx = eps_yy = eps_zz = complex(float(coeffs.permittivity), 0.0)
            elif hasattr(coeffs, "xx"):
                # Anisotropic
                eps_xx = complex(float(getattr(coeffs.xx, "permittivity", 1.0)), 0.0)
                eps_yy = complex(float(getattr(coeffs.yy, "permittivity", 1.0)), 0.0)
                eps_zz = complex(float(getattr(coeffs.zz, "permittivity", 1.0)), 0.0)
            else:
                eps_xx = eps_yy = eps_zz = complex(1.0, 0.0)

            # Extract permeability if available
            if hasattr(coeffs, "permeability"):
                mu_xx = mu_yy = mu_zz = complex(float(coeffs.permeability), 0.0)
            else:
                mu_xx = mu_yy = mu_zz = complex(1.0, 0.0)

            for f_idx in range(n_freqs):
                self.medium_data[i, f_idx, 0] = eps_xx
                self.medium_data[i, f_idx, 1] = eps_yy
                self.medium_data[i, f_idx, 2] = eps_zz
                if not self.compiled.is_permittivity_only:
                    self.medium_data[i, f_idx, 3] = mu_xx
                    self.medium_data[i, f_idx, 4] = mu_yy
                    self.medium_data[i, f_idx, 5] = mu_zz

    def to_medium_monitor_data(self) -> dict[str, Any]:
        """Convert recorded data to MediumMonitorData format.

        Returns permittivity and permeability as (real, imag) tuples.
        """
        n_pts = self.compiled.support_point_count
        n_freqs = self.compiled.num_freqs

        if self.medium_data is None:
            return {}

        result = {}
        n_components = 3 if self.compiled.is_permittivity_only else 6
        field_names = ["eps_xx", "eps_yy", "eps_zz"] if self.compiled.is_permittivity_only else ["eps_xx", "eps_yy", "eps_zz", "mu_xx", "mu_yy", "mu_zz"]

        for c_idx, field_name in enumerate(field_names):
            values = []
            for f_idx in range(n_freqs):
                for p_idx in range(n_pts):
                    val = self.medium_data[p_idx, f_idx, c_idx]
                    values.append((float(val.real), float(val.imag)))
            result[field_name] = tuple(values)

        return result

    def to_permittivity_monitor_data(self) -> dict[str, Any]:
        """Convert recorded data to PermittivityData format.

        Returns only permittivity as float tuples.
        """
        n_pts = self.compiled.support_point_count
        n_freqs = self.compiled.num_freqs

        if self.medium_data is None:
            return {}

        result = {}
        for field_name, c_idx in [("eps_xx", 0), ("eps_yy", 1), ("eps_zz", 2)]:
            values = []
            for f_idx in range(n_freqs):
                for p_idx in range(n_pts):
                    val = self.medium_data[p_idx, f_idx, c_idx]
                    values.append(float(val.real))  # permittivity is real for Phase 1
            result[field_name] = tuple(values)

        return result


# --------------------------------------------------------------------
# Mode monitor compilation
# --------------------------------------------------------------------


def _infer_mode_monitor_normal_axis(size: tuple[float, float, float]) -> int:
    """Infer the normal axis for a mode monitor from its size.

    A mode monitor projects fields onto mode profiles at a planar cross-section.
    The normal axis is the one with zero extent.

    Args:
        size: Monitor size tuple

    Returns:
        Axis index (0=x, 1=y, 2=z)

    Raises:
        ValueError: If size has no zero extent (not a planar monitor)
    """
    for axis, s in enumerate(size):
        if s <= 1e-12:
            return axis
    raise ValueError(
        f"ModeMonitor requires a planar cross-section (one zero-size dimension), got size={size}"
    )


@dataclass(frozen=True)
class CompiledModeMonitor:
    """Discrete mode monitor placement and mode-overlap metadata."""

    name: str
    monitor_type: Literal["ModeMonitor", "ModeSolverMonitor"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Normal axis for the cross-section (0=x, 1=y, 2=z)
    normal_axis: int
    # Direction for overlap sign convention
    direction: Literal["+", "-"]
    # Cell indices and weights for the surface
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    # Interval and start
    interval: int
    start: int
    # Mode specification
    mode_spec: dict[str, Any] | None = None
    # Frequency parameters
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @property
    def is_time_domain(self) -> bool:
        # ModeMonitor with freqs is frequency-domain, without is time-domain
        return len(self.freqs) == 0

    @property
    def is_frequency_domain(self) -> bool:
        return len(self.freqs) > 0

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def needs_dft(self) -> bool:
        """Whether this monitor needs DFT accumulation for frequency-domain output."""
        return self.is_frequency_domain


def compile_mode_monitor(
    name: str,
    monitor_type: Literal["ModeMonitor", "ModeSolverMonitor"],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    direction: Literal["+", "-"] = "+",
    interval: int = 1,
    start: int = 0,
    mode_spec: dict[str, Any] | None = None,
    num_freqs: int = 1,
    freqs: tuple[float, ...] = (),
    grid: ResolvedGrid,
) -> CompiledModeMonitor:
    """Compile a mode monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: ModeMonitor or ModeSolverMonitor
        center: Monitor center position
        size: Monitor size (must have exactly one zero dimension for planar cross-section)
        direction: Mode direction (+ or -)
        interval: Recording interval in timesteps
        start: First timestep to start recording
        mode_spec: Mode specification dict
        num_freqs: Number of frequency points
        freqs: Frequency points for frequency-domain monitors
        grid: The resolved simulation grid

    Returns:
        CompiledModeMonitor with placement and metadata
    """
    normal_axis = _infer_mode_monitor_normal_axis(size)

    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    return CompiledModeMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        normal_axis=normal_axis,
        direction=direction,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        interval=interval,
        start=start,
        mode_spec=mode_spec,
        num_freqs=num_freqs,
        freqs=freqs,
    )


@dataclass(frozen=True)
class ModeMonitorState:
    """Runtime state for a mode monitor.

    Stores mode overlap integrals computed from recorded fields and
    solved mode profiles.
    """

    compiled: CompiledModeMonitor
    # Resolved mode solutions at each frequency
    mode_solutions: dict[int, tuple] | None = None
    # Time-domain storage: mode amplitude time-series
    amplitude_series: list[np.ndarray] | None = None
    time_stamps: list[float] | None = None
    # Frequency-domain storage: DFT accumulator for mode amplitudes
    dft_amplitudes: np.ndarray | None = None
    dft_count: int = 0

    def __post_init__(self):
        if self.compiled.is_time_domain:
            object.__setattr__(self, "amplitude_series", [])
            object.__setattr__(self, "time_stamps", [])
            object.__setattr__(self, "mode_solutions", {})
        else:
            # Frequency-domain: DFT accumulator
            n_freqs = self.compiled.num_freqs
            n_modes = self.compiled.mode_spec.get("num_modes", 1) if self.compiled.mode_spec else 1
            object.__setattr__(
                self,
                "dft_amplitudes",
                np.zeros((n_freqs, n_modes), dtype=np.complex128),
            )
            object.__setattr__(self, "dft_count", 0)
            object.__setattr__(self, "mode_solutions", {})

    @property
    def dft_count_value(self) -> int:
        """Access dft_count without triggering frozen dataclass issue."""
        return object.__getattribute__(self, "dft_count")

    def _set_dft_count(self, value: int) -> None:
        """Set dft_count via object.__setattr__ to bypass frozen."""
        object.__setattr__(self, "dft_count", value)

    def set_mode_solutions(self, solutions: tuple) -> None:
        """Set the solved mode profiles for overlap computation.

        Args:
            solutions: Tuple of ModeSolution objects
        """
        object.__setattr__(self, "mode_solutions", {i: s for i, s in enumerate(solutions)})

    def compute_overlap(
        self,
        field_data: np.ndarray,
        mode_solution,
        normal_axis: int,
    ) -> complex:
        """Compute the overlap integral between field data and a mode profile.

        The overlap integral is:
            overlap = integral(E_field × H_mode* - H_field × E_mode*) dA

        For a planar monitor at cross-section, this reduces to a discrete
        surface integral.

        Args:
            field_data: Dict of field component arrays at monitor placements
            mode_solution: ModeSolution with field components
            normal_axis: Normal axis for the cross-section

        Returns:
            Complex overlap amplitude
        """
        placements = self.compiled.placements
        n_pts = len(placements)

        # Get tangential axes
        tang_axes = tuple(a for a in range(3) if a != normal_axis)

        overlap = 0.0 + 0.0j

        for i, idx in enumerate(placements):
            # Extract fields at this point
            ex = field_data.get("Ex", np.zeros(n_pts))[i] if "Ex" in field_data else 0.0
            ey = field_data.get("Ey", np.zeros(n_pts))[i] if "Ey" in field_data else 0.0
            ez = field_data.get("Ez", np.zeros(n_pts))[i] if "Ez" in field_data else 0.0
            hx = field_data.get("Hx", np.zeros(n_pts))[i] if "Hx" in field_data else 0.0
            hy = field_data.get("Hy", np.zeros(n_pts))[i] if "Hy" in field_data else 0.0
            hz = field_data.get("Hz", np.zeros(n_pts))[i] if "Hz" in field_data else 0.0

            # Get mode fields at this point (mode fields are on cross-sectional grid)
            # Mode coordinates map to monitor placements
            mode_x = mode_solution.x
            mode_y = mode_solution.y

            # Simple nearest-neighbor lookup in mode field grid
            # This is approximate - a real implementation would interpolate
            mx_idx = min(i % len(mode_x), len(mode_x) - 1) if len(mode_x) > 0 else 0
            my_idx = min(i // len(mode_x), len(mode_y) - 1) if len(mode_y) > 0 else 0

            # Get mode fields at this grid point
            mode_Ex = mode_solution.Ex[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Ex) else (0.0, 0.0)
            mode_Ey = mode_solution.Ey[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Ey) else (0.0, 0.0)
            mode_Ez = mode_solution.Ez[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Ez) else (0.0, 0.0)
            mode_Hx = mode_solution.Hx[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Hx) else (0.0, 0.0)
            mode_Hy = mode_solution.Hy[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Hy) else (0.0, 0.0)
            mode_Hz = mode_solution.Hz[mx_idx * len(mode_y) + my_idx] if mx_idx * len(mode_y) + my_idx < len(mode_solution.Hz) else (0.0, 0.0)

            # Convert mode complex tuples to complex values
            def to_complex(c: tuple) -> complex:
                return complex(c[0], c[1]) if isinstance(c, tuple) else complex(c, 0.0)

            mEx = to_complex(mode_Ex)
            mEy = to_complex(mode_Ey)
            mEz = to_complex(mode_Ez)
            mHx = to_complex(mode_Hx)
            mHy = to_complex(mode_Hy)
            mHz = to_complex(mode_Hz)

            # Poynting-like overlap: E × H* for the mode
            # For cross-section normal to axis:
            # - axis=0 (x-normal): S_y = E_z*H_y* - E_y*H_z*, S_z = E_y*H_x* - E_x*H_y*
            # - axis=1 (y-normal): S_x = E_y*H_z* - E_z*H_y*, S_z = E_z*H_x* - E_x*H_z*
            # - axis=2 (z-normal): S_x = E_y*H_z* - E_z*H_y*, S_y = E_z*H_x* - E_x*H_z*

            if normal_axis == 0:
                s_mode = (mEz * np.conj(mHy) - mEy * np.conj(mHz),
                         mEx * np.conj(mHz) - mEz * np.conj(mHx))
                s_field = (ez * np.conj(hy) - ey * np.conj(hz),
                          ex * np.conj(hz) - ez * np.conj(hx))
            elif normal_axis == 1:
                s_mode = (mEz * np.conj(mHy) - mEy * np.conj(mHz),
                          mEx * np.conj(mHz) - mEz * np.conj(mHx))
                s_field = (ez * np.conj(hy) - ey * np.conj(hz),
                          ex * np.conj(hz) - ez * np.conj(hx))
            else:  # axis == 2
                s_mode = (mEz * np.conj(mHy) - mEy * np.conj(mHz),
                          mEx * np.conj(mHz) - mEz * np.conj(mHx))
                s_field = (ez * np.conj(hy) - ey * np.conj(hz),
                          ex * np.conj(hz) - ez * np.conj(hx))

            overlap += s_field[0] * np.conj(s_mode[0]) + s_field[1] * np.conj(s_mode[1])

        return overlap

    def record_mode_time_domain(
        self,
        field_data: dict[str, np.ndarray],
        mode_solutions: tuple,
        time: float,
    ) -> None:
        """Record mode overlap amplitudes at the current timestep.

        Args:
            field_data: Dict of field component arrays
            mode_solutions: Tuple of ModeSolution objects
            time: Current simulation time
        """
        if not self.compiled.is_time_domain:
            return

        # Compute overlap for each mode
        n_modes = len(mode_solutions)
        amplitudes = np.zeros(n_modes, dtype=np.complex128)

        for m_idx, mode_sol in enumerate(mode_solutions):
            amplitudes[m_idx] = self.compute_overlap(
                field_data, mode_sol, self.compiled.normal_axis
            )

        self.amplitude_series.append(amplitudes)
        self.time_stamps.append(time)

    def accumulate_mode_dft(
        self,
        field_data: dict[str, np.ndarray],
        mode_solutions: tuple,
        time: float,
    ) -> None:
        """Accumulate DFT terms for frequency-domain mode monitor.

        Args:
            field_data: Dict of field component arrays
            mode_solutions: Tuple of ModeSolution objects
            time: Current simulation time
        """
        if not self.compiled.is_frequency_domain:
            return

        # Compute overlap for each mode at each frequency
        n_freqs = self.compiled.num_freqs
        n_modes = len(mode_solutions)

        for m_idx, mode_sol in enumerate(mode_solutions):
            overlap = self.compute_overlap(
                field_data, mode_sol, self.compiled.normal_axis
            )

            omega_t = 2.0 * np.pi * np.asarray(self.compiled.freqs) * time
            for j, freq in enumerate(self.compiled.freqs):
                phase = np.exp(-1j * omega_t[j])
                self.dft_amplitudes[j, m_idx] += overlap * phase

        self._set_dft_count(self.dft_count_value + 1)

    def to_mode_data(self) -> dict[str, Any]:
        """Convert recorded data to ModeData format."""
        if self.compiled.is_time_domain:
            if not self.amplitude_series:
                return {"amplitudes": (), "t": ()}

            # Stack amplitude series: shape (n_timesteps, n_modes)
            stacked = np.stack(self.amplitude_series, axis=0)
            # Convert to (real, imag) tuples
            amplitude_tuples = tuple(
                tuple((float(v.real), float(v.imag)) for v in stacked[:, m_idx])
                for m_idx in range(stacked.shape[1])
            )
            return {
                "amplitudes": tuple(amplitude_tuples),
                "t": tuple(self.time_stamps) if self.time_stamps else (),
            }
        else:
            # Frequency-domain: normalize DFT
            if self.dft_count_value > 0:
                normalized = self.dft_amplitudes / float(self.dft_count_value)
                amplitude_tuples = tuple(
                    tuple((float(v.real), float(v.imag)) for v in normalized[:, m_idx])
                    for m_idx in range(normalized.shape[1])
                )
            else:
                amplitude_tuples = ()
            return {"amplitudes": amplitude_tuples, "t": ()}


# --------------------------------------------------------------------
# Projection monitor compilation
# --------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledProjectionMonitor:
    """Discrete projection monitor placement and projection metadata.

    Projection monitors compute far-field radiation patterns by applying
    the equivalence principle to fields on a surface and projecting to
    observation points at a specified distance.
    """

    name: str
    monitor_type: Literal[
        "FieldProjectionAngleMonitor",
        "FieldProjectionCartesianMonitor",
        "FieldProjectionKSpaceMonitor",
        "DiffractionMonitor",
        "DirectivityMonitor",
    ]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Normal axis for the source surface (axis with zero extent)
    normal_axis: int
    # Projection distance
    projection_distance: float
    # Interval and start
    interval: int
    start: int
    # Frequency parameters
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    # For angle-space projection
    phi: tuple[float, float, int] = (-90.0, 90.0, 181)
    theta: tuple[float, float, int] = (0.0, 180.0, 181)
    # For Cartesian projection
    x: tuple[float, float, int] = (-50.0, 50.0, 201)
    y: tuple[float, float, int] = (-50.0, 50.0, 201)
    # For k-space projection
    num_k: int = 1
    kx: tuple[float, float, int] = (-10.0, 10.0, 21)
    ky: tuple[float, float, int] = (-10.0, 10.0, 21)
    # Cell indices and weights for the source surface
    placements: tuple[tuple[int, int, int], ...] = ()
    placement_weights: tuple[float, ...] = ()

    @property
    def is_time_domain(self) -> bool:
        return len(self.freqs) == 0

    @property
    def is_frequency_domain(self) -> bool:
        return len(self.freqs) > 0

    @property
    def support_point_count(self) -> int:
        return len(self.placements)


def _infer_projection_normal_axis(size: tuple[float, float, float]) -> int:
    """Infer the normal axis for a projection monitor from its size.

    A projection monitor records fields on a 2D surface (one zero extent).
    The normal axis is the one with zero extent.
    """
    for axis, s in enumerate(size):
        if s <= 1e-12:
            return axis
    raise ValueError(
        f"Projection monitor requires a 2D surface (one zero-size dimension), got size={size}"
    )


def compile_projection_monitor(
    name: str,
    monitor_type: Literal[
        "FieldProjectionAngleMonitor",
        "FieldProjectionCartesianMonitor",
        "FieldProjectionKSpaceMonitor",
        "DiffractionMonitor",
        "DirectivityMonitor",
    ],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
    projection_distance: float = 1e5,
    interval: int = 1,
    start: int = 0,
    num_freqs: int = 1,
    freqs: tuple[float, ...] = (),
    phi: tuple[float, float, int] = (-90.0, 90.0, 181),
    theta: tuple[float, float, int] = (0.0, 180.0, 181),
    x: tuple[float, float, int] = (-50.0, 50.0, 201),
    y: tuple[float, float, int] = (-50.0, 50.0, 201),
    num_k: int = 1,
    kx: tuple[float, float, int] = (-10.0, 10.0, 21),
    ky: tuple[float, float, int] = (-10.0, 10.0, 21),
    grid: ResolvedGrid,
) -> CompiledProjectionMonitor:
    """Compile a projection monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: Type of projection monitor
        center: Monitor center position
        size: Monitor size (must have exactly one zero dimension for surface)
        normal_vector: Normal vector of the projection surface
        projection_distance: Distance for far-field projection
        interval: Recording interval in timesteps
        start: First timestep to start recording
        num_freqs: Number of frequency points
        freqs: Frequency points for frequency-domain monitors
        phi: Phi angle range (min, max, count) for angle projection
        theta: Theta angle range (min, max, count) for angle projection
        x: X range (min, max, count) for Cartesian projection
        y: Y range (min, max, count) for Cartesian projection
        num_k: Number of k points
        kx: Kx range (min, max, count) for k-space projection
        ky: Ky range (min, max, count) for k-space projection
        grid: The resolved simulation grid

    Returns:
        CompiledProjectionMonitor with placement and metadata
    """
    normal_axis = _infer_projection_normal_axis(size)

    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    return CompiledProjectionMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        normal_axis=normal_axis,
        projection_distance=projection_distance,
        interval=interval,
        start=start,
        num_freqs=num_freqs,
        freqs=freqs,
        phi=phi,
        theta=theta,
        x=x,
        y=y,
        num_k=num_k,
        kx=kx,
        ky=ky,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
    )


@dataclass(frozen=True)
class ProjectionMonitorState:
    """Runtime state for a projection monitor.

    Stores the DFT of field values on the source surface for later
    far-field projection computation.
    """

    compiled: CompiledProjectionMonitor
    # DFT accumulator for E and H fields at each surface point
    dft_e: np.ndarray | None = None
    dft_h: np.ndarray | None = None
    dft_count: int = 0

    def __post_init__(self):
        n_pts = self.compiled.support_point_count
        n_freqs = self.compiled.num_freqs
        # 6 components: Ex, Ey, Ez, Hx, Hy, Hz
        object.__setattr__(
            self,
            "dft_e",
            np.zeros((n_pts, n_freqs, 3), dtype=np.complex128),
        )
        object.__setattr__(
            self,
            "dft_h",
            np.zeros((n_pts, n_freqs, 3), dtype=np.complex128),
        )
        object.__setattr__(self, "dft_count", 0)

    @property
    def dft_count_value(self) -> int:
        return object.__getattribute__(self, "dft_count")

    def _set_dft_count(self, value: int) -> None:
        object.__setattr__(self, "dft_count", value)

    def accumulate_dft(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Accumulate DFT terms for frequency-domain projection monitor.

        Args:
            electric_field: The E field buffer with shape (nx, ny, nz, 3)
            magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
            time: Current simulation time
        """
        if not self.compiled.is_frequency_domain:
            return

        placements = self.compiled.placements
        freqs = self.compiled.freqs
        n_pts = len(placements)

        for i, idx in enumerate(placements):
            ex = electric_field[idx][0]
            ey = electric_field[idx][1]
            ez = electric_field[idx][2]
            hx = magnetic_field[idx][0]
            hy = magnetic_field[idx][1]
            hz = magnetic_field[idx][2]

            for j, freq in enumerate(freqs):
                omega_t = 2.0 * np.pi * freq * time
                phase = np.exp(-1j * omega_t)
                self.dft_e[i, j, 0] += ex * phase
                self.dft_e[i, j, 1] += ey * phase
                self.dft_e[i, j, 2] += ez * phase
                self.dft_h[i, j, 0] += hx * phase
                self.dft_h[i, j, 1] += hy * phase
                self.dft_h[i, j, 2] += hz * phase

        self._set_dft_count(self.dft_count_value + 1)

    def to_projection_data(self) -> dict[str, Any]:
        """Convert recorded DFT data to projection monitor data format.

        For Phase 1, projection computation is deferred. This returns
        the raw DFT data that would be used for projection.
        """
        if self.dft_count_value == 0:
            return {"dft_e": (), "dft_h": (), "t": ()}

        # Normalize DFT
        normalized_e = self.dft_e / float(self.dft_count_value)
        normalized_h = self.dft_h / float(self.dft_count_value)

        return {
            "dft_e": normalized_e,
            "dft_h": normalized_h,
            "t": (),
        }


# --------------------------------------------------------------------
# Surface field monitor compilation
# --------------------------------------------------------------------


def _infer_surface_normal_axis(size: tuple[float, float, float]) -> int:
    """Infer the normal axis for a surface monitor from its size.

    A surface monitor records fields on a 2D surface. The normal axis
    is the one with zero extent.

    Args:
        size: Monitor size tuple

    Returns:
        Axis index (0=x, 1=y, 2=z)

    Raises:
        ValueError: If size has no zero extent (not a surface)
    """
    for axis, s in enumerate(size):
        if s <= 1e-12:
            return axis
    raise ValueError(
        f"Surface monitor requires a 2D surface (one zero-size dimension), got size={size}"
    )


@dataclass(frozen=True)
class CompiledSurfaceFieldMonitor:
    """Discrete surface field monitor placement and recording metadata.

    A surface monitor records field values over a 2D surface (one dimension
    has zero extent). Unlike a flux monitor which computes an integrated
    quantity, a surface field monitor records field values at each point
    on the surface.
    """

    name: str
    monitor_type: Literal["SurfaceFieldMonitor", "SurfaceFieldTimeMonitor"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    # Normal axis for the surface (axis with zero extent)
    normal_axis: int
    # Cell indices and weights for the surface
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    # Tangential axes (non-normal axes) for coordinate extraction
    tang_axis_1: int
    tang_axis_2: int
    # Fields to record
    fields: tuple[str, ...]
    # Interval and start for time-domain monitors
    interval: int
    start: int
    # For frequency-domain monitors (SurfaceFieldMonitor with freqs)
    freqs: tuple[float, ...] = ()
    num_freqs: int = 1

    @property
    def is_time_domain(self) -> bool:
        return self.monitor_type == "SurfaceFieldTimeMonitor"

    @property
    def is_frequency_domain(self) -> bool:
        return self.monitor_type == "SurfaceFieldMonitor" and len(self.freqs) > 0

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def needs_dft(self) -> bool:
        """Whether this monitor needs DFT accumulation for frequency-domain output."""
        return self.is_frequency_domain


def compile_surface_field_monitor(
    name: str,
    monitor_type: Literal["SurfaceFieldMonitor", "SurfaceFieldTimeMonitor"],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    fields: tuple[str, ...],
    *,
    interval: int = 1,
    start: int = 0,
    freqs: tuple[float, ...] = (),
    grid: ResolvedGrid,
) -> CompiledSurfaceFieldMonitor:
    """Compile a surface field monitor onto the resolved grid.

    Args:
        name: Monitor name
        monitor_type: SurfaceFieldMonitor or SurfaceFieldTimeMonitor
        center: Monitor center position
        size: Monitor size (must have exactly one zero dimension for surface)
        fields: Field components to record
        interval: Recording interval in timesteps
        start: First timestep to start recording
        freqs: Frequency points for frequency-domain monitors
        grid: The resolved simulation grid

    Returns:
        CompiledSurfaceFieldMonitor with placement and metadata
    """
    normal_axis = _infer_surface_normal_axis(size)

    # Determine tangential axes
    tang_axes = tuple(a for a in range(3) if a != normal_axis)
    tang_axis_1 = tang_axes[0]
    tang_axis_2 = tang_axes[1]

    axis_boundaries = (
        grid.x.boundaries,
        grid.y.boundaries,
        grid.z.boundaries,
    )

    axis_placements = tuple(
        _compile_monitor_axis_placement(
            axis_boundaries[axis],
            center=center[axis],
            size=size[axis],
        )
        for axis in range(3)
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

    num_freqs = len(freqs) if freqs else 1

    return CompiledSurfaceFieldMonitor(
        name=name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        normal_axis=normal_axis,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        tang_axis_1=tang_axis_1,
        tang_axis_2=tang_axis_2,
        fields=fields,
        interval=interval,
        start=start,
        freqs=freqs,
        num_freqs=num_freqs,
    )


@dataclass(frozen=True)
class SurfaceFieldMonitorState:
    """Runtime state for a surface field monitor.

    For time-domain monitors, stores field time-series on the surface.
    For frequency-domain monitors, accumulates DFT terms.
    """

    compiled: CompiledSurfaceFieldMonitor
    # Time-domain storage: list of field snapshots with timestamps
    time_series: dict[str, list[np.ndarray]] | None = None
    time_stamps: list[float] | None = None
    # Frequency-domain storage: DFT accumulator arrays
    dft_data: dict[str, np.ndarray] | None = None
    dft_count: int = 0

    def __post_init__(self):
        if self.compiled.is_time_domain:
            object.__setattr__(
                self,
                "time_series",
                {field: [] for field in self.compiled.fields},
            )
            object.__setattr__(self, "time_stamps", [])
        else:
            # Frequency-domain: DFT accumulator
            n_pts = len(self.compiled.placements)
            n_freqs = self.compiled.num_freqs
            object.__setattr__(
                self,
                "dft_data",
                {
                    field: np.zeros((n_pts, n_freqs), dtype=np.complex128)
                    for field in self.compiled.fields
                },
            )
            object.__setattr__(self, "dft_count", 0)

    @property
    def dft_count_value(self) -> int:
        """Access dft_count without triggering frozen dataclass issue."""
        return object.__getattribute__(self, "dft_count")

    def _set_dft_count(self, value: int) -> None:
        """Set dft_count via object.__setattr__ to bypass frozen."""
        object.__setattr__(self, "dft_count", value)

    def record_time_domain(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Record field values at the current timestep for time-domain monitor."""
        if not self.compiled.is_time_domain:
            return

        for field in self.compiled.fields:
            values = self._extract_field_values(
                electric_field, magnetic_field, field
            )
            self.time_series[field].append(values.copy())

        self.time_stamps.append(time)

    def accumulate_dft(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        time: float,
    ) -> None:
        """Accumulate DFT terms for frequency-domain surface monitor."""
        if not self.compiled.is_frequency_domain:
            return

        for field in self.compiled.fields:
            values = self._extract_field_values(
                electric_field, magnetic_field, field
            )
            # DFT accumulation: sum over timesteps
            omega_t = 2.0 * np.pi * np.asarray(self.compiled.freqs) * time
            phase = np.cos(omega_t) - 1j * np.sin(omega_t)
            # dft_data shape: (n_pts, n_freqs), values shape: (n_pts,), phase shape: (n_freqs,)
            # Always expand for proper broadcasting: values[:, None] * phase[None, :]
            self.dft_data[field] += values[:, None] * phase[None, :]

        self._set_dft_count(self.dft_count_value + 1)

    def _extract_field_values(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        field: str,
    ) -> np.ndarray:
        """Extract field component values at monitor placements."""
        if field not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            return np.zeros(len(self.compiled.placements), dtype=np.complex128)

        is_electric = field[0] == "E"
        component_axis = {"x": 0, "y": 1, "z": 2}[field[1]]

        if len(self.compiled.placements) == 1:
            idx = self.compiled.placements[0]
            if is_electric:
                return np.array([electric_field[idx][component_axis]])
            else:
                return np.array([magnetic_field[idx][component_axis]])

        values = np.zeros(len(self.compiled.placements), dtype=np.complex128)
        for i, idx in enumerate(self.compiled.placements):
            if is_electric:
                values[i] = electric_field[idx][component_axis]
            else:
                values[i] = magnetic_field[idx][component_axis]

        return values

    def to_surface_field_data(self) -> dict[str, tuple[tuple[float, float], ...]]:
        """Convert recorded data to SurfaceFieldData format with (real, imag) tuples."""
        n_pts = len(self.compiled.placements)

        if self.compiled.is_time_domain:
            result = {}
            for field in self.compiled.fields:
                series = self.time_series[field]
                if not series:
                    result[field] = ()
                    continue
                # Stack all timesteps: shape (n_timesteps, n_pts)
                stacked = np.stack(series, axis=0)
                # Convert to (real, imag) tuples
                tuples = tuple(
                    (float(v.real), float(v.imag))
                    for v in stacked.flatten()
                )
                result[field] = tuples
            return result
        else:
            # Frequency-domain: normalize DFT
            # dft_data shape: (n_pts, n_freqs)
            result = {}
            if self.dft_count_value > 0:
                for field in self.compiled.fields:
                    normalized = self.dft_data[field] / float(self.dft_count_value)
                    tuples = tuple(
                        (float(v.real), float(v.imag))
                        for v in normalized.flatten()
                    )
                    result[field] = tuples
            else:
                for field in self.compiled.fields:
                    result[field] = ()
            return result

    def time_data(self) -> tuple[tuple[float, ...], dict[str, tuple[tuple[float, float], ...]]]:
        """Return time stamps and field data for time-domain monitors."""
        if not self.compiled.is_time_domain:
            return (), {}

        time_stamps = tuple(self.time_stamps) if self.time_stamps else ()

        result = {}
        for field in self.compiled.fields:
            series = self.time_series[field]
            if not series:
                result[field] = ()
                continue
            stacked = np.stack(series, axis=0)
            tuples = tuple(
                (float(v.real), float(v.imag))
                for v in stacked.flatten()
            )
            result[field] = tuples

        return time_stamps, result