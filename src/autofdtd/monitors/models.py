"""Monitor-family models for Phase 1 field sampling and result storage."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel

Vec3 = tuple[float, float, float]

# Supported Phase 1 monitor types
_MONITOR_TYPES: frozenset[str] = frozenset({
    "FieldMonitor",
    "FieldTimeMonitor",
    "AuxFieldTimeMonitor",
    "FluxMonitor",
    "FluxTimeMonitor",
    "ModeMonitor",
    "ModeSolverMonitor",
    "MediumMonitor",
    "PermittivityMonitor",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceMonitor",
    "DiffractionMonitor",
    "DirectivityMonitor",
    "GaussianOverlapMonitor",
    "AstigmaticGaussianOverlapMonitor",
    "SurfaceFieldMonitor",
    "SurfaceFieldTimeMonitor",
})


def _normalize_monitor_name(name: str | None) -> str | None:
    """Normalize a monitor name by trimming whitespace and rejecting empties."""
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        raise ValueError("monitor names must not be empty or whitespace-only")
    return stripped


def _normalize_vec3(value: Sequence[object], *, field_name: str) -> Vec3:
    """Coerce a 3-vector to a finite float tuple."""
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three components")
    normalized = (float(value[0]), float(value[1]), float(value[2]))
    if any(not math.isfinite(component) for component in normalized):
        raise ValueError(f"{field_name} components must be finite")
    return normalized


def _normalize_size(value: Sequence[object]) -> Vec3:
    """Coerce a 3-vector to a non-negative float tuple for monitor size."""
    size = _normalize_vec3(value, field_name="size")
    if any(component < 0.0 for component in size):
        raise ValueError("size components must be non-negative")
    return size


class FieldRegion(str):
    """Monitor field region specification."""

    VOLUME = "volume"
    SURFACE = "surface"
    PLANE = "plane"
    LINE = "line"
    POINT = "point"


class Monitor(TaggedModel):
    """Base monitor shell with name, placement, and interval semantics.

    Phase 1 monitors support:
    - ``FieldMonitor``: time-averaged complex field recording over a 3D volume or 2D plane
    - ``FieldTimeMonitor``: time-domain field recording over a 3D volume or 2D plane
    - ``AuxFieldTimeMonitor``: auxiliary field time-domain recording for观音观音
    - ``FluxMonitor``: flux-through integration over a 2D surface
    - ``FluxTimeMonitor``: time-domain flux recording
    - ``ModeMonitor``: mode overlap projection at a planar cross-section
    - ``ModeSolverMonitor``: mode solver backing a mode monitor
    - ``MediumMonitor``: material property sampling at points
    - ``PermittivityMonitor``: permittivity distribution recording
    - ``FieldProjectionAngleMonitor``: far-field projection in angle space
    - ``FieldProjectionCartesianMonitor``: far-field projection in Cartesian coords
    - ``FieldProjectionKSpaceMonitor``: k-space field projection
    - ``DiffractionMonitor``: diffraction order recording
    - ``DirectivityMonitor``: directivity computation via projection
    - ``GaussianOverlapMonitor``: Gaussian beam overlap integral
    - ``AstigmaticGaussianOverlapMonitor``: astigmatic beam overlap
    - ``SurfaceFieldMonitor``: surface field recording
    - ``SurfaceFieldTimeMonitor``: surface field time-domain recording
    """

    type: str = Field(default="Monitor")
    name: str | None = None
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (0.0, 0.0, 0.0)
    interval: int = 1
    start: int = 0

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_monitor_name(value)

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Sequence[object]) -> Vec3:
        return _normalize_vec3(value, field_name="center")

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Sequence[object]) -> Vec3:
        return _normalize_size(value)

    @field_validator("interval")
    @classmethod
    def _validate_interval(cls, value: int) -> int:
        if value < 1:
            raise ValueError("interval must be at least 1")
        return value

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: int) -> int:
        if value < 0:
            raise ValueError("start must be non-negative")
        return value

    def monitor_type(self) -> str:
        """Return the concrete monitor family tag."""
        return self.type

    def colocated(self) -> bool:
        """Return whether this monitor records colocated (vs Yee-staggered) fields."""
        return True


class FieldMonitor(Monitor):
    """Time-averaged complex field recording over a 3D volume or 2D plane."""

    type: Literal["FieldMonitor"] = "FieldMonitor"
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    overwrite: bool = True

    def colocated(self) -> bool:
        return True


class FieldTimeMonitor(Monitor):
    """Time-domain field recording over a 3D volume or 2D plane."""

    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class AuxFieldTimeMonitor(Monitor):
    """Auxiliary field time-domain recording for观音观音."""

    type: Literal["AuxFieldTimeMonitor"] = "AuxFieldTimeMonitor"
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class FluxMonitor(Monitor):
    """Flux-through integration over a 2D surface."""

    type: Literal["FluxMonitor"] = "FluxMonitor"
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class FluxTimeMonitor(Monitor):
    """Time-domain flux recording."""

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"
    direction: Literal["+", "-"] = "+"


class ModeMonitor(Monitor):
    """Mode overlap projection at a planar cross-section."""

    type: Literal["ModeMonitor"] = "ModeMonitor"
    mode_spec: object | None = None
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class ModeSolverMonitor(Monitor):
    """Mode solver backing a mode monitor."""

    type: Literal["ModeSolverMonitor"] = "ModeSolverMonitor"
    mode_spec: object | None = None
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class MediumMonitor(Monitor):
    """Material property sampling at points."""

    type: Literal["MediumMonitor"] = "MediumMonitor"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class PermittivityMonitor(Monitor):
    """Permittivity distribution recording."""

    type: Literal["PermittivityMonitor"] = "PermittivityMonitor"


class FieldProjectionAngleMonitor(Monitor):
    """Far-field projection in angle space."""

    type: Literal["FieldProjectionAngleMonitor"] = "FieldProjectionAngleMonitor"
    normal_vector: Vec3 = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    phi: tuple[float, float, float] = (-90.0, 90.0, 181)
    theta: tuple[float, float, float] = (0.0, 180.0, 181)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class FieldProjectionCartesianMonitor(Monitor):
    """Far-field projection in Cartesian coords."""

    type: Literal["FieldProjectionCartesianMonitor"] = "FieldProjectionCartesianMonitor"
    normal_vector: Vec3 = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    x: tuple[float, float, int] = (-50.0, 50.0, 201)
    y: tuple[float, float, int] = (-50.0, 50.0, 201)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class FieldProjectionKSpaceMonitor(Monitor):
    """K-space field projection."""

    type: Literal["FieldProjectionKSpaceMonitor"] = "FieldProjectionKSpaceMonitor"
    normal_vector: Vec3 = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    num_k: int = 1
    kx: tuple[float, float, int] = (-10.0, 10.0, 21)
    ky: tuple[float, float, int] = (-10.0, 10.0, 21)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class DiffractionMonitor(Monitor):
    """Diffraction order recording.

    Computes diffraction orders from fields recorded on a 2D surface.
    The diffraction orders are determined by the grating equation applied
    to the periodic structure in the scene. Phase 1 supports diffraction
    from uniform or piecewise-uniform media with known periodicity.

    For diffraction to produce meaningful order amplitudes, the scene should
    contain periodic structures (e.g., grating geometries). Without explicit
    grating periodicity, the monitor records the angular spectrum of the
    transmitted field.

    Example
    -------
    >>> monitor = DiffractionMonitor(
    ...     size=(0, 5.0, 5.0),  # 2D surface in y-z plane
    ...     freqs=[3e14],
    ...     normal_vector=(1.0, 0.0, 0.0),  # x-normal surface
    ...     name="diffraction_monitor",
    ... )
    """

    type: Literal["DiffractionMonitor"] = "DiffractionMonitor"
    normal_vector: Vec3 = (0.0, 0.0, 1.0)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("normal_vector", mode="before")
    @classmethod
    def _validate_normal_vector(cls, value: Vec3) -> Vec3:
        """Ensure normal_vector is a unit vector (optionally allowing zero vector)."""
        nv = value
        norm_sq = nv[0] ** 2 + nv[1] ** 2 + nv[2] ** 2
        if norm_sq < 1e-12:
            # Zero vector is allowed as a placeholder
            return nv
        if abs(norm_sq - 1.0) > 1e-6:
            norm = norm_sq ** 0.5
            raise ValueError(
                f"normal_vector must be a unit vector, got norm={norm:.6f}. "
                "Normalize the vector or use (0, 0, 0) for auto-detection."
            )
        return nv

    @field_validator("num_freqs")
    @classmethod
    def _validate_num_freqs(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"num_freqs must be at least 1, got {value}")
        return value

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        for f in value:
            if f <= 0:
                raise ValueError(f"freqs must be positive, got {f}")
        return value


class DirectivityMonitor(Monitor):
    """Directivity computation via projection."""

    type: Literal["DirectivityMonitor"] = "DirectivityMonitor"
    normal_vector: Vec3 = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value


class GaussianOverlapMonitor(Monitor):
    """Gaussian beam overlap integral monitor.

    Projects recorded fields onto a Gaussian beam basis defined by
    propagation direction, polarization, waist radius, and waist distance.

    Example
    -------
    >>> monitor = GaussianOverlapMonitor(
    ...     size=(0, 3, 3),
    ...     freqs=[2e14],
    ...     pol_angle=np.pi / 2,
    ...     waist_radius=1.0,
    ...     name="gaussian_monitor",
    ... )
    """

    type: Literal["GaussianOverlapMonitor"] = "GaussianOverlapMonitor"
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    # Beam parameters (matching Tidy3D GaussianOverlapMonitor)
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_radius: float = 1.0
    waist_distance: float = 0.0

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value

    @property
    def _angles(self) -> tuple[float, float]:
        """Return the propagation angles (theta, phi)."""
        return (self.angle_theta, self.angle_phi)


class AstigmaticGaussianOverlapMonitor(Monitor):
    """Astigmatic Gaussian beam overlap monitor.

    Projects recorded fields onto an astigmatic Gaussian beam basis with
    separate waist parameters for the two principal axes.

    When equal waist sizes and equal waist distances are specified in the two
    directions, this monitor becomes equivalent to GaussianOverlapMonitor.

    Example
    -------
    >>> monitor = AstigmaticGaussianOverlapMonitor(
    ...     size=(0, 3, 3),
    ...     freqs=[2e14],
    ...     pol_angle=np.pi / 2,
    ...     waist_sizes=(1.0, 2.0),
    ...     waist_distances=(3.0, 4.0),
    ...     name="astigmatic_monitor",
    ... )
    """

    type: Literal["AstigmaticGaussianOverlapMonitor"] = "AstigmaticGaussianOverlapMonitor"
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    # Beam parameters (matching Tidy3D AstigmaticGaussianOverlapMonitor)
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_sizes: tuple[float, float] = (1.0, 1.0)
    waist_distances: tuple[float, float] = (0.0, 0.0)

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value

    @field_validator("waist_sizes")
    @classmethod
    def _validate_waist_sizes(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("waist_sizes must be positive")
        return value

    @property
    def _angles(self) -> tuple[float, float]:
        """Return the propagation angles (theta, phi)."""
        return (self.angle_theta, self.angle_phi)


class SurfaceFieldMonitor(Monitor):
    """Surface field recording in the frequency domain.

    Records complex field values on a 2D surface at specified frequency points.
    """

    type: Literal["SurfaceFieldMonitor"] = "SurfaceFieldMonitor"
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()

    @field_validator("freqs", mode="before")
    @classmethod
    def _validate_freqs(cls, value: tuple[float, ...] | None) -> tuple[float, ...]:
        if value is None:
            return ()
        return value

    @model_validator(mode="after")
    def _update_num_freqs(self) -> "SurfaceFieldMonitor":
        """Derive num_freqs from len(freqs)."""
        if self.freqs:
            object.__setattr__(self, "num_freqs", len(self.freqs))
        return self


class SurfaceFieldTimeMonitor(Monitor):
    """Surface field time-domain recording."""

    type: Literal["SurfaceFieldTimeMonitor"] = "SurfaceFieldTimeMonitor"
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# --------------------------------------------------------------------
# Monitor result containers (simulation output data)
# --------------------------------------------------------------------


class MonitorData(TaggedModel):
    """Base container for monitor output data."""

    type: str = Field(default="MonitorData")
    monitor_name: str
    monitor_type: str
    data: dict[str, Any] = Field(default_factory=dict)

    def field_components(self) -> tuple[str, ...]:
        """Return the field components available in this monitor data."""
        return tuple(self.data.keys())


class FieldData(MonitorData):
    """Complex field data from a field monitor."""

    type: Literal["FieldData"] = "FieldData"
    Ex: tuple[tuple[float, float], ...] | None = None
    Ey: tuple[tuple[float, float], ...] | None = None
    Ez: tuple[tuple[float, float], ...] | None = None
    Hx: tuple[tuple[float, float], ...] | None = None
    Hy: tuple[tuple[float, float], ...] | None = None
    Hz: tuple[tuple[float, float], ...] | None = None
    # Grid coordinates stored as (real, imag) tuples for JSON safety
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    z: tuple[float, ...] | None = None
    # Time steps for time-domain monitors
    t: tuple[float, ...] | None = None


class FluxData(MonitorData):
    """Flux data from a flux monitor.

    For time-domain flux monitors (FluxTimeMonitor), flux contains real-valued
    time-series. For frequency-domain flux monitors (FluxMonitor with freqs),
    flux contains complex values as (real, imag) tuples representing the
    Fourier transform at each frequency point.
    """

    type: Literal["FluxData"] = "FluxData"
    # For time-domain: real flux values. For frequency-domain: (real, imag) tuples.
    flux: tuple[float, ...] | tuple[tuple[float, float], ...] | None = None
    t: tuple[float, ...] | None = None


class ModeData(MonitorData):
    """Mode monitor data with overlap integrals."""

    type: Literal["ModeData"] = "ModeData"
    # Complex mode amplitudes stored as (real, imag) pairs
    amplitudes: tuple[tuple[tuple[float, float], ...], ...] | None = None
    flux: tuple[float, ...] | None = None
    t: tuple[float, ...] | None = None


class PermittivityData(MonitorData):
    """Permittivity data from a permittivity monitor."""

    type: Literal["PermittivityData"] = "PermittivityData"
    eps_xx: tuple[float, ...] | None = None
    eps_yy: tuple[float, ...] | None = None
    eps_zz: tuple[float, ...] | None = None
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    z: tuple[float, ...] | None = None


class MediumMonitorData(MonitorData):
    """Medium property data from a medium monitor.

    Stores diagonal components of the complex-valued relative permittivity
    and permeability tensors in the frequency domain.
    """

    type: Literal["MediumMonitorData"] = "MediumMonitorData"
    eps_xx: tuple[tuple[float, float], ...] | None = None
    eps_yy: tuple[tuple[float, float], ...] | None = None
    eps_zz: tuple[tuple[float, float], ...] | None = None
    mu_xx: tuple[tuple[float, float], ...] | None = None
    mu_yy: tuple[tuple[float, float], ...] | None = None
    mu_zz: tuple[tuple[float, float], ...] | None = None
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    z: tuple[float, ...] | None = None


class GaussianOverlapData(MonitorData):
    """Gaussian beam overlap monitor data.

    Stores complex overlap amplitudes for the + and - directions
    at each frequency point. Complex values are stored as (real, imag)
    tuples for JSON safety.
    """

    type: Literal["GaussianOverlapData"] = "GaussianOverlapData"
    # Complex amplitudes stored as (real, imag) pairs; first dim is direction (+/-), second is frequency
    amplitudes: tuple[tuple[tuple[float, float], ...], ...] | None = None
    t: tuple[float, ...] | None = None


class AstigmaticGaussianOverlapData(MonitorData):
    """Astigmatic Gaussian beam overlap monitor data.

    Stores complex overlap amplitudes for the + and - directions
    at each frequency point, with separate amplitude data for the
    x and y waist directions. Complex values are stored as (real, imag)
    tuples for JSON safety.
    """

    type: Literal["AstigmaticGaussianOverlapData"] = "AstigmaticGaussianOverlapData"
    # Complex amplitudes stored as (real, imag) pairs; first dim is direction (+/-), second is frequency
    amplitudes: tuple[tuple[tuple[float, float], ...], ...] | None = None
    t: tuple[float, ...] | None = None


class FieldProjectionAngleData(MonitorData):
    """Angle-space field projection monitor data.

    Stores far-field projections as a function of (phi, theta) angles.
    Complex values are stored as (real, imag) tuples for JSON safety.

    The projection is computed from fields on a surface using the
    equivalence principle to radiate to observation points in the
    far field at a specified distance.
    """

    type: Literal["FieldProjectionAngleData"] = "FieldProjectionAngleData"
    # Far-field values E_theta and E_phi at each (phi, theta, freq) point
    # Shape: (num_phi_points, num_theta_points, num_freqs, 2) where 2 = (E_theta, E_phi)
    e_theta: tuple[tuple[tuple[float, float], ...], ...] | None = None
    e_phi: tuple[tuple[tuple[float, float], ...], ...] | None = None
    # Grid coordinates
    phi: tuple[float, ...] | None = None
    theta: tuple[float, ...] | None = None
    # Time stamps (empty for frequency-domain)
    t: tuple[float, ...] | None = None


class FieldProjectionCartesianData(MonitorData):
    """Cartesian far-field projection monitor data.

    Stores far-field projections on a Cartesian (x, y) observation plane
    at a specified distance. Complex values are stored as (real, imag)
    tuples for JSON safety.
    """

    type: Literal["FieldProjectionCartesianData"] = "FieldProjectionCartesianData"
    # Far-field E components at each (x, y, freq) point
    ex: tuple[tuple[tuple[float, float], ...], ...] | None = None
    ey: tuple[tuple[tuple[float, float], ...], ...] | None = None
    ez: tuple[tuple[tuple[float, float], ...], ...] | None = None
    # Grid coordinates
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    # Time stamps
    t: tuple[float, ...] | None = None


class FieldProjectionKSpaceData(MonitorData):
    """K-space field projection monitor data.

    Stores far-field projections in k-space (kx, ky) at a fixed
    propagation distance. Complex values are stored as (real, imag)
    tuples for JSON safety.
    """

    type: Literal["FieldProjectionKSpaceData"] = "FieldProjectionKSpaceData"
    # Far-field E components at each (kx, ky, freq) point
    ex: tuple[tuple[tuple[float, float], ...], ...] | None = None
    ey: tuple[tuple[tuple[float, float], ...], ...] | None = None
    ez: tuple[tuple[tuple[float, float], ...], ...] | None = None
    # Grid coordinates
    kx: tuple[float, ...] | None = None
    ky: tuple[float, ...] | None = None
    # Time stamps
    t: tuple[float, ...] | None = None


class DiffractionData(MonitorData):
    """Diffraction monitor data.

    Stores diffraction order amplitudes. Complex values are stored
    as (real, imag) tuples for JSON safety.
    """

    type: Literal["DiffractionData"] = "DiffractionData"
    # Diffraction orders: (num_orders, num_freqs, 2) where 2 = (real, imag)
    orders: tuple[tuple[tuple[float, float], ...], ...] | None = None
    # Diffraction order indices (mx, my) for each order
    mx: tuple[int, ...] | None = None
    my: tuple[int, ...] | None = None
    # Time stamps
    t: tuple[float, ...] | None = None


class DirectivityData(MonitorData):
    """Directivity monitor data.

    Stores radiated power directivity as a function of (theta, phi) angles.
    Real values. Time stamps stored for time-domain variants.
    """

    type: Literal["DirectivityData"] = "DirectivityData"
    # Directivity values at each (theta, phi) point (real, in dBi)
    directivity: tuple[tuple[float, ...], ...] | None = None
    # Grid coordinates
    theta: tuple[float, ...] | None = None
    phi: tuple[float, ...] | None = None
    # Time stamps
    t: tuple[float, ...] | None = None


class SurfaceFieldData(MonitorData):
    """Surface field data from a surface field monitor.

    Stores complex field values on a surface (2D grid) in the frequency domain.
    Complex values are stored as (real, imag) tuples for JSON safety.

    A surface monitor records fields over a 2D surface (one dimension has zero
    extent). The fields are recorded at specified frequency points.
    """

    type: Literal["SurfaceFieldData"] = "SurfaceFieldData"
    # Complex field values stored as (real, imag) tuples
    Ex: tuple[tuple[float, float], ...] | None = None
    Ey: tuple[tuple[float, float], ...] | None = None
    Ez: tuple[tuple[float, float], ...] | None = None
    Hx: tuple[tuple[float, float], ...] | None = None
    Hy: tuple[tuple[float, float], ...] | None = None
    Hz: tuple[tuple[float, float], ...] | None = None
    # Grid coordinates along the surface tangents
    # For x-normal surface: y, z coordinates
    # For y-normal surface: x, z coordinates
    # For z-normal surface: x, y coordinates
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    z: tuple[float, ...] | None = None
    # Time stamps (empty for frequency-domain SurfaceFieldMonitor)
    t: tuple[float, ...] | None = None


class SurfaceFieldTimeData(MonitorData):
    """Surface field time-domain data from a surface field time monitor.

    Stores complex field time-series on a surface (2D grid).
    Complex values are stored as (real, imag) tuples for JSON safety.

    A surface time monitor records fields at every timestep (or interval)
    over a 2D surface.
    """

    type: Literal["SurfaceFieldTimeData"] = "SurfaceFieldTimeData"
    # Complex field values stored as (real, imag) tuples; first dim is time
    Ex: tuple[tuple[float, float], ...] | None = None
    Ey: tuple[tuple[float, float], ...] | None = None
    Ez: tuple[tuple[float, float], ...] | None = None
    Hx: tuple[tuple[float, float], ...] | None = None
    Hy: tuple[tuple[float, float], ...] | None = None
    Hz: tuple[tuple[float, float], ...] | None = None
    # Grid coordinates along the surface tangents
    x: tuple[float, ...] | None = None
    y: tuple[float, ...] | None = None
    z: tuple[float, ...] | None = None
    # Time stamps
    t: tuple[float, ...] | None = None


# --------------------------------------------------------------------
# SimulationData - named access container for all monitor results
# --------------------------------------------------------------------

# Mapping from monitor type name to concrete model class
_MONITOR_CONSTRUCTORS: dict[str, type[Monitor]] = {
    "FieldMonitor": FieldMonitor,
    "FieldTimeMonitor": FieldTimeMonitor,
    "AuxFieldTimeMonitor": AuxFieldTimeMonitor,
    "FluxMonitor": FluxMonitor,
    "FluxTimeMonitor": FluxTimeMonitor,
    "ModeMonitor": ModeMonitor,
    "ModeSolverMonitor": ModeSolverMonitor,
    "MediumMonitor": MediumMonitor,
    "PermittivityMonitor": PermittivityMonitor,
    "FieldProjectionAngleMonitor": FieldProjectionAngleMonitor,
    "FieldProjectionCartesianMonitor": FieldProjectionCartesianMonitor,
    "FieldProjectionKSpaceMonitor": FieldProjectionKSpaceMonitor,
    "DiffractionMonitor": DiffractionMonitor,
    "DirectivityMonitor": DirectivityMonitor,
    "GaussianOverlapMonitor": GaussianOverlapMonitor,
    "AstigmaticGaussianOverlapMonitor": AstigmaticGaussianOverlapMonitor,
    "SurfaceFieldMonitor": SurfaceFieldMonitor,
    "SurfaceFieldTimeMonitor": SurfaceFieldTimeMonitor,
}


class SimulationData(TaggedModel):
    """Container for simulation results with Tidy3D-style named monitor access.

    Provides dictionary-like access via ``sim_data['monitor_name']`` and
    attribute access via ``sim_data.monitor_name`` for all monitor data
    produced by a simulation run.

    Example::

        sim_data = SimulationData(simulation=sim, runtime_log=log)
        sim_data.register_field_monitor("field_monitor", field_data)
        sim_data.register_flux_monitor("flux_monitor", flux_data)

        # Dictionary-style access
        field_result = sim_data["field_monitor"]
        flux_result = sim_data["flux_monitor"]

        # Attribute-style access
        field_result = sim_data.field_monitor
        flux_result = sim_data.flux_monitor

        # Monitor iteration
        for name in sim_data.monitor_names:
            print(f"{name}: {type(sim_data[name])}")
    """

    type: Literal["SimulationData"] = "SimulationData"
    simulation_name: str | None = None
    # Monitor metadata from the original simulation
    monitor_names: tuple[str, ...] = ()

    # Internal mutable storage - excluded from serialization
    monitor_store: dict[str, MonitorData] = Field(default_factory=dict, exclude=True)

    def monitor_data(self) -> dict[str, MonitorData]:
        """Return the full dictionary of registered monitor data."""
        return self.monitor_store

    def register_monitor_data(self, data: MonitorData) -> None:
        """Register monitor data by the monitor's name.

        Args:
            data: MonitorData with a monitor_name attribute

        Raises:
            KeyError: If a monitor with the same name is already registered
        """
        name = data.monitor_name
        if name in self.monitor_store:
            raise KeyError(f"monitor data for {name!r} is already registered")
        self.monitor_store[name] = data

    def register_field_monitor(
        self, name: str, data: FieldData
    ) -> None:
        """Register field monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_flux_monitor(
        self, name: str, data: FluxData
    ) -> None:
        """Register flux monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_mode_monitor(
        self, name: str, data: ModeData
    ) -> None:
        """Register mode monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_permittivity_monitor(
        self, name: str, data: PermittivityData
    ) -> None:
        """Register permittivity monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_medium_monitor(
        self, name: str, data: MediumMonitorData
    ) -> None:
        """Register medium monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_gaussian_overlap_monitor(
        self, name: str, data: GaussianOverlapData
    ) -> None:
        """Register Gaussian overlap monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_astigmatic_gaussian_overlap_monitor(
        self, name: str, data: AstigmaticGaussianOverlapData
    ) -> None:
        """Register astigmatic Gaussian overlap monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_field_projection_angle_monitor(
        self, name: str, data: FieldProjectionAngleData
    ) -> None:
        """Register angle-space field projection monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_field_projection_cartesian_monitor(
        self, name: str, data: FieldProjectionCartesianData
    ) -> None:
        """Register Cartesian field projection monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_field_projection_kspace_monitor(
        self, name: str, data: FieldProjectionKSpaceData
    ) -> None:
        """Register k-space field projection monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_diffraction_monitor(
        self, name: str, data: DiffractionData
    ) -> None:
        """Register diffraction monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_directivity_monitor(
        self, name: str, data: DirectivityData
    ) -> None:
        """Register directivity monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_surface_field_monitor(
        self, name: str, data: SurfaceFieldData
    ) -> None:
        """Register surface field monitor data (frequency-domain)."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_surface_field_time_monitor(
        self, name: str, data: SurfaceFieldTimeData
    ) -> None:
        """Register surface field time monitor data (time-domain)."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def register_custom_monitor(
        self, name: str, data: MonitorData
    ) -> None:
        """Register arbitrary monitor data."""
        if data.monitor_name != name:
            data = data.model_copy(update={"monitor_name": name})
        self.register_monitor_data(data)

    def __getitem__(self, key: str) -> MonitorData:
        """Dictionary-style access: sim_data['monitor_name']."""
        try:
            return self.monitor_store[key]
        except KeyError:
            raise KeyError(f"no monitor data found for {key!r}") from None

    def __getattr__(self, name: str) -> MonitorData:
        """Attribute-style access: sim_data.monitor_name.

        Only called for names not found in instance __dict__.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.monitor_store[name]
        except KeyError:
            raise AttributeError(f"no monitor data found for {name!r}") from None

    def __contains__(self, key: str) -> bool:
        """Return True if monitor data exists for the given name."""
        return key in self.monitor_store

    def __iter__(self):
        """Iterate over registered monitor data names."""
        return iter(self.monitor_store)

    def __len__(self) -> int:
        """Return the number of registered monitor data entries."""
        return len(self.monitor_store)

    def keys(self):
        """Return monitor data names."""
        return self.monitor_store.keys()

    def values(self):
        """Return monitor data values."""
        return self.monitor_store.values()

    def items(self):
        """Return (name, data) pairs."""
        return self.monitor_store.items()

    def get(self, key: str, default=None) -> MonitorData | None:
        """Return monitor data or a default if not found."""
        return self.monitor_store.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation with all monitor data."""
        result = {"monitor_names": self.monitor_names}
        for name, data in self.monitor_store.items():
            result[name] = data.model_dump(mode="json", exclude_none=True)
        return result


# --------------------------------------------------------------------
# Monitor model factory
# --------------------------------------------------------------------


def monitor_model_from_value(value: object) -> Monitor:
    """Construct a concrete monitor model from a tagged dict or model."""
    if isinstance(value, Monitor):
        return value
    # Handle Pydantic BaseModel instances
    if isinstance(value, BaseModel):
        # If it's already a concrete Monitor subclass, return it
        if value.type in _MONITOR_CONSTRUCTORS:
            return _MONITOR_CONSTRUCTORS[value.type].model_validate(value.model_dump())
        # Otherwise return as generic Monitor
        return Monitor.model_validate(value.model_dump())
    if not isinstance(value, Mapping):
        raise TypeError(f"monitor must be a mapping or Monitor instance, got {type(value)!r}")

    monitor_type = str(value.get("type", "Monitor"))

    if monitor_type == "Monitor":
        # Return generic Monitor for unknown types
        return Monitor.model_validate(value)

    constructor = _MONITOR_CONSTRUCTORS.get(monitor_type)
    if constructor is None:
        # Unknown monitor type - return as generic Monitor
        return Monitor.model_validate(value)

    return constructor.model_validate(value)


def monitor_data_model_from_value(value: object) -> MonitorData:
    """Construct a concrete monitor data model from a tagged dict or model."""
    if isinstance(value, MonitorData):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"monitor data must be a mapping or MonitorData instance, got {type(value)!r}")

    data_type = str(value.get("type", "MonitorData"))

    _DATA_CONSTRUCTORS: dict[str, type[MonitorData]] = {
        "FieldData": FieldData,
        "FluxData": FluxData,
        "ModeData": ModeData,
        "PermittivityData": PermittivityData,
        "MediumMonitorData": MediumMonitorData,
        "GaussianOverlapData": GaussianOverlapData,
        "AstigmaticGaussianOverlapData": AstigmaticGaussianOverlapData,
        "FieldProjectionAngleData": FieldProjectionAngleData,
        "FieldProjectionCartesianData": FieldProjectionCartesianData,
        "FieldProjectionKSpaceData": FieldProjectionKSpaceData,
        "DiffractionData": DiffractionData,
        "DirectivityData": DirectivityData,
        "SurfaceFieldData": SurfaceFieldData,
        "SurfaceFieldTimeData": SurfaceFieldTimeData,
    }

    constructor = _DATA_CONSTRUCTORS.get(data_type)
    if constructor is None:
        return MonitorData.model_validate(value)

    return constructor.model_validate(value)
