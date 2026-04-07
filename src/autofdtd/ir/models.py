"""Versioned execution IR models and packaging helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from autofdtd.boundaries import (
    ABCBoundary,
    Absorber,
    AbsorberParams,
    BlochBoundary,
    BroadbandModeABCSpec,
    Boundary,
    BoundarySpec,
    ModeABCBoundary,
    PML,
    PMLParams,
    PECBoundary,
    PMCBoundary,
    Periodic,
    StablePML,
    boundary_edge_model_from_value,
    boundary_model_from_value,
    boundary_spec_model_from_value,
)
from autofdtd.core.containers import Scene, Simulation, Structure, StructurePriorityMode
from autofdtd.core.models import TaggedModel, json_ready
from autofdtd.compiler.runtime import compile_runtime_controls
from autofdtd.geometry import (
    Box,
    ClipOperation,
    Cylinder,
    GeometryArray,
    GeometryGroup,
    GeometryTransform,
    PolySlab,
    Sphere,
    Transformed,
    geometry_model_from_value,
)
from autofdtd.grid import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    PolarizedAveraging,
    ResolvedGrid,
    ResolvedGridAxis,
    Staircasing,
    UniformGrid,
    grid_model_from_value,
    subpixel_model_from_value,
)
from autofdtd.materials import (
    AnisotropicMedium,
    Debye,
    Drude,
    Lorentz,
    Medium,
    PECMedium,
    PMCMedium,
    PoleResidue,
    Sellmeier,
    medium_model_from_value,
)
from autofdtd.sources import (
    AstigmaticGaussianBeam,
    BroadbandPulse,
    ContinuousWave,
    CustomCurrentSource,
    CustomFieldSource,
    CustomSourceTime,
    FixedAngleSpec,
    FixedInPlaneKSpec,
    GaussianBeam,
    GaussianPulse,
    PointDipole,
    PlaneWave,
    TFSF,
    UniformCurrentSource,
    angular_spec_model_from_value,
    source_time_model_from_value,
)
from autofdtd.sources.mode import ModeSource
from autofdtd.version import __version__

IR_SCHEMA_VERSION = "phase1.v1"
EXECUTION_PACKAGE_FORMAT = "autofdtd.execution-package.v1"


class IRModel(TaggedModel):
    """Frozen tagged base for inspectable execution-IR payloads."""

    schema_version: str = Field(default=IR_SCHEMA_VERSION)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != IR_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {IR_SCHEMA_VERSION!r}")
        return value


class ComponentFamily(str):
    """Execution-IR family namespace for tagged components."""

    GEOMETRY = "geometry"
    MEDIUM = "medium"
    SOURCE = "source"
    MONITOR = "monitor"
    BOUNDARY = "boundary"
    GRID = "grid"
    SUBPIXEL = "subpixel"


class ComponentIR(IRModel):
    """Generic tagged envelope for feature-family payloads."""

    type: Literal["ComponentIR"] = "ComponentIR"
    family: str
    component_type: str
    name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family", "component_type")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("component family and type must be non-empty")
        return stripped

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("component names must not be empty or whitespace-only")
        return stripped


class GeometryTransformIR(IRModel):
    """Scene-compilation transform metadata for primitive geometries."""

    type: Literal["GeometryTransformIR"] = "GeometryTransformIR"
    origin: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class GaussianPulseIR(IRModel):
    """Typed execution IR for a Gaussian source-time profile."""

    type: Literal["GaussianPulseIR"] = "GaussianPulseIR"
    component_type: Literal["GaussianPulse"] = "GaussianPulse"
    amplitude: float
    phase: float
    freq0: float
    fwidth: float
    offset: float
    remove_dc_component: bool
    twidth: float
    offset_time: float
    peak_time: float
    peak_frequency: float
    end_time: float
    frequency_range: tuple[float, float]


class ContinuousWaveIR(IRModel):
    """Typed execution IR for a continuous-wave source-time profile."""

    type: Literal["ContinuousWaveIR"] = "ContinuousWaveIR"
    component_type: Literal["ContinuousWave"] = "ContinuousWave"
    amplitude: float
    phase: float
    freq0: float
    fwidth: float
    offset: float
    twidth: float
    offset_time: float
    end_time: None = None
    frequency_range: tuple[float, float]


class BroadbandPulseIR(IRModel):
    """Typed execution IR for the supported broadband-pulse subset."""

    type: Literal["BroadbandPulseIR"] = "BroadbandPulseIR"
    component_type: Literal["BroadbandPulse"] = "BroadbandPulse"
    amplitude: float
    phase: float
    freq_range: tuple[float, float]
    minimum_amplitude: float
    offset: float
    freq0: float
    bandwidth: float
    fwidth: float
    twidth: float
    offset_time: float
    peak_time: float
    end_time: float
    frequency_range: tuple[float, float]


class CustomSourceTimeIR(IRModel):
    """Typed execution IR for interpolated custom source-time envelopes."""

    type: Literal["CustomSourceTimeIR"] = "CustomSourceTimeIR"
    component_type: Literal["CustomSourceTime"] = "CustomSourceTime"
    amplitude: float
    phase: float
    freq0: float
    fwidth: float
    offset: float
    twidth: float
    offset_time: float
    time_samples: tuple[float, ...]
    envelope_values: tuple[tuple[float, float], ...]
    sample_count: int
    end_time: float | None
    frequency_range: tuple[float, float]


SourceTimeIR = GaussianPulseIR | ContinuousWaveIR | BroadbandPulseIR | CustomSourceTimeIR


class UniformCurrentSourceIR(IRModel):
    """Typed execution IR for a uniform current source."""

    type: Literal["UniformCurrentSourceIR"] = "UniformCurrentSourceIR"
    component_type: Literal["UniformCurrentSource"] = "UniformCurrentSource"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    polarization: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    field_kind: Literal["electric", "magnetic"]
    component_axis: Literal[0, 1, 2]
    placement_kind: Literal["point", "line", "sheet", "volume"]
    zero_size_axes: tuple[int, ...]
    interpolate: bool
    confine_to_bounds: bool
    current_amplitude_definition: Literal["density", "total"]
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]


class PointDipoleIR(IRModel):
    """Typed execution IR for a point dipole source."""

    type: Literal["PointDipoleIR"] = "PointDipoleIR"
    component_type: Literal["PointDipole"] = "PointDipole"
    center: tuple[float, float, float]
    polarization: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    field_kind: Literal["electric", "magnetic"]
    component_axis: Literal[0, 1, 2]
    placement_kind: Literal["point"] = "point"
    zero_size_axes: tuple[int, ...] = (0, 1, 2)
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]


class CustomCurrentSourceIR(IRModel):
    """Typed execution IR for a custom current source with explicit field data."""

    type: Literal["CustomCurrentSourceIR"] = "CustomCurrentSourceIR"
    component_type: Literal["CustomCurrentSource"] = "CustomCurrentSource"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Field component data as typed tuples
    e_fields: dict[str, tuple[float, ...]] | None = None
    h_fields: dict[str, tuple[float, ...]] | None = None
    coordinates: dict[str, tuple[float, ...]] | None = None
    has_electric: bool = False
    has_magnetic: bool = False


class CustomFieldSourceIR(IRModel):
    """Typed execution IR for a custom field source using equivalence principle."""

    type: Literal["CustomFieldSourceIR"] = "CustomFieldSourceIR"
    component_type: Literal["CustomFieldSource"] = "CustomFieldSource"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["sheet"] = "sheet"
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Field component data as typed tuples
    e_fields: dict[str, tuple[float, ...]] | None = None
    h_fields: dict[str, tuple[float, ...]] | None = None
    coordinates: dict[str, tuple[float, ...]] | None = None
    has_electric: bool = False
    has_magnetic: bool = False
    has_tangential_fields: bool = False


class ModeSourceIR(IRModel):
    """Typed execution IR for a mode source backed by the mode solver."""

    type: Literal["ModeSourceIR"] = "ModeSourceIR"
    component_type: Literal["ModeSource"] = "ModeSource"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["sheet"] = "sheet"
    mode_index: int
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Mode profile data as typed tuples (from mode solver)
    e_fields: dict[str, tuple[tuple[float, float], ...]] | None = None
    h_fields: dict[str, tuple[tuple[float, float], ...]] | None = None
    mode_neff: float | None = None
    mode_power: float = 1.0


class FixedInPlaneKSpecIR(IRModel):
    """Typed execution IR for fixed in-plane k specification."""

    type: Literal["FixedInPlaneKSpecIR"] = "FixedInPlaneKSpecIR"
    component_type: Literal["FixedInPlaneKSpec"] = "FixedInPlaneKSpec"


class FixedAngleSpecIR(IRModel):
    """Typed execution IR for fixed angle specification."""

    type: Literal["FixedAngleSpecIR"] = "FixedAngleSpecIR"
    component_type: Literal["FixedAngleSpec"] = "FixedAngleSpec"


AngularSpecIR = FixedInPlaneKSpecIR | FixedAngleSpecIR


class PlaneWaveIR(IRModel):
    """Typed execution IR for a plane wave source."""

    type: Literal["PlaneWaveIR"] = "PlaneWaveIR"
    component_type: Literal["PlaneWave"] = "PlaneWave"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["sheet"] = "sheet"
    angular_spec: AngularSpecIR
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Direction and polarization vectors for injection
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]


class GaussianBeamIR(IRModel):
    """Typed execution IR for a Gaussian beam source."""

    type: Literal["GaussianBeamIR"] = "GaussianBeamIR"
    component_type: Literal["GaussianBeam"] = "GaussianBeam"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["sheet"] = "sheet"
    angular_spec: AngularSpecIR
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Direction and polarization vectors for injection
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    # Beam properties
    waist_radius: float
    waist_distance: float
    reference_wavelength: float | None = None


class AstigmaticGaussianBeamIR(IRModel):
    """Typed execution IR for an astigmatic Gaussian beam source."""

    type: Literal["AstigmaticGaussianBeamIR"] = "AstigmaticGaussianBeamIR"
    component_type: Literal["AstigmaticGaussianBeam"] = "AstigmaticGaussianBeam"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["sheet"] = "sheet"
    angular_spec: AngularSpecIR
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Direction and polarization vectors for injection
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    # Astigmatic beam properties
    waist_radius_x: float
    waist_radius_y: float
    waist_distance_x: float
    waist_distance_y: float
    reference_wavelength: float | None = None


class TFSFIR(IRModel):
    """Typed execution IR for a total-field scattered-field source."""

    type: Literal["TFSFIR"] = "TFSFIR"
    component_type: Literal["TFSF"] = "TFSF"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    direction: Literal["+", "-"]
    angle_theta: float
    angle_phi: float
    pol_angle: float
    injection_axis: Literal[0, 1, 2]
    placement_kind: Literal["volume"] = "volume"
    interpolate: bool
    confine_to_bounds: bool
    source_time: SourceTimeIR
    name: str | None = None
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    # Direction and polarization vectors for injection
    dir_vector: tuple[float, float, float]
    pol_vector: tuple[float, float, float]
    num_freqs: int
    injection_plane_center: tuple[float, float, float]
    reference_wavelength: float | None = None


class BoxIR(IRModel):
    """Typed execution IR for an axis-aligned box primitive."""

    type: Literal["BoxIR"] = "BoxIR"
    component_type: Literal["Box"] = "Box"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    transform: GeometryTransformIR


class SphereIR(IRModel):
    """Typed execution IR for a sphere primitive."""

    type: Literal["SphereIR"] = "SphereIR"
    component_type: Literal["Sphere"] = "Sphere"
    center: tuple[float, float, float]
    radius: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    transform: GeometryTransformIR


class CylinderIR(IRModel):
    """Typed execution IR for an axis-aligned cylinder primitive."""

    type: Literal["CylinderIR"] = "CylinderIR"
    component_type: Literal["Cylinder"] = "Cylinder"
    center: tuple[float, float, float]
    radius: float
    length: float
    axis: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    transform: GeometryTransformIR


class PolySlabIR(IRModel):
    """Typed execution IR for a constant-cross-section polygon extrusion."""

    type: Literal["PolySlabIR"] = "PolySlabIR"
    component_type: Literal["PolySlab"] = "PolySlab"
    vertices: tuple[tuple[float, float], ...]
    slab_bounds: tuple[float, float]
    axis: int
    sidewall_angle: float
    dilation: float
    reference_plane: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    transform: GeometryTransformIR


class GeometryGroupIR(IRModel):
    """Typed execution IR for a grouped geometry collection."""

    type: Literal["GeometryGroupIR"] = "GeometryGroupIR"
    component_type: Literal["GeometryGroup"] = "GeometryGroup"
    geometries: tuple[GeometryIR, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


class TransformedIR(IRModel):
    """Typed execution IR for a transformed geometry wrapper."""

    type: Literal["TransformedIR"] = "TransformedIR"
    component_type: Literal["Transformed"] = "Transformed"
    geometry: GeometryIR
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    transform: GeometryTransformIR


class ClipOperationIR(IRModel):
    """Typed execution IR for Phase 1 clip-operation wrappers."""

    type: Literal["ClipOperationIR"] = "ClipOperationIR"
    component_type: Literal["ClipOperation"] = "ClipOperation"
    operation: Literal["union", "intersection", "difference"]
    geometry_a: GeometryIR
    geometry_b: GeometryIR
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


GeometryIR = (
    BoxIR
    | SphereIR
    | CylinderIR
    | PolySlabIR
    | GeometryGroupIR
    | TransformedIR
    | ClipOperationIR
    | ComponentIR
)


class ResolvedGridAxisIR(IRModel):
    """Resolved axis-grid metadata included in the execution IR."""

    type: Literal["ResolvedGridAxisIR"] = "ResolvedGridAxisIR"
    axis: Literal["x", "y", "z"]
    boundaries: tuple[float, ...]
    cell_sizes: tuple[float, ...]
    num_cells: int


class ResolvedGridIR(IRModel):
    """Resolved 3D grid metadata included in the execution IR."""

    type: Literal["ResolvedGridIR"] = "ResolvedGridIR"
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    x: ResolvedGridAxisIR
    y: ResolvedGridAxisIR
    z: ResolvedGridAxisIR
    shape: tuple[int, int, int]
    total_cells: int
    min_step: float
    max_step: float


class UniformGridIR(IRModel):
    """Typed execution IR for a uniform axis-grid specification."""

    type: Literal["UniformGridIR"] = "UniformGridIR"
    component_type: Literal["UniformGrid"] = "UniformGrid"
    dl: float


class CustomGridBoundariesIR(IRModel):
    """Typed execution IR for explicit axis-grid boundaries."""

    type: Literal["CustomGridBoundariesIR"] = "CustomGridBoundariesIR"
    component_type: Literal["CustomGridBoundaries"] = "CustomGridBoundaries"
    coords: tuple[float, ...]


class CustomGridIR(IRModel):
    """Typed execution IR for explicit axis-grid cell sizes."""

    type: Literal["CustomGridIR"] = "CustomGridIR"
    component_type: Literal["CustomGrid"] = "CustomGrid"
    dl: tuple[float, ...]
    custom_offset: float | None = None


class AutoGridIR(IRModel):
    """Typed execution IR for a simplified AutoGrid axis specification."""

    type: Literal["AutoGridIR"] = "AutoGridIR"
    component_type: Literal["AutoGrid"] = "AutoGrid"
    min_steps_per_wvl: float
    min_steps_per_sim_size: float
    max_scale: float
    dl_min: float | None = None


class GridRefinementIR(IRModel):
    """Typed execution IR for local AutoGrid refinement metadata."""

    type: Literal["GridRefinementIR"] = "GridRefinementIR"
    component_type: Literal["GridRefinement"] = "GridRefinement"
    refinement_factor: float | None = None
    dl: float | None = None
    num_cells: int


class LayerRefinementSpecIR(IRModel):
    """Typed execution IR for the Phase 1 layer-refinement subset."""

    type: Literal["LayerRefinementSpecIR"] = "LayerRefinementSpecIR"
    component_type: Literal["LayerRefinementSpec"] = "LayerRefinementSpec"
    axis: int
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    min_steps_along_axis: float | None = None
    bounds_refinement: GridRefinementIR | None = None
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] | None = None


GridAxisIR = UniformGridIR | CustomGridBoundariesIR | CustomGridIR | AutoGridIR | ComponentIR


class GridSpecIR(IRModel):
    """Typed execution IR for the Phase 1 grid-spec container."""

    type: Literal["GridSpecIR"] = "GridSpecIR"
    component_type: Literal["GridSpec"] = "GridSpec"
    grid_x: GridAxisIR
    grid_y: GridAxisIR
    grid_z: GridAxisIR
    wavelength: float | None = None
    layer_refinement_specs: tuple[LayerRefinementSpecIR, ...] = ()
    resolved: ResolvedGridIR


class StaircasingIR(IRModel):
    """Typed execution IR for staircased interface assignment."""

    type: Literal["StaircasingIR"] = "StaircasingIR"
    component_type: Literal["Staircasing"] = "Staircasing"
    materialization_mode: Literal["staircasing"] = "staircasing"
    requires_anisotropic_materialization: bool = False
    courant_ratio: float = 1.0


class PolarizedAveragingIR(IRModel):
    """Typed execution IR for dielectric polarized averaging."""

    type: Literal["PolarizedAveragingIR"] = "PolarizedAveragingIR"
    component_type: Literal["PolarizedAveraging"] = "PolarizedAveraging"
    materialization_mode: Literal["averaging"] = "averaging"
    requires_anisotropic_materialization: bool = True
    courant_ratio: float = 1.0


SubpixelPolicyIR = StaircasingIR | PolarizedAveragingIR


class SubpixelSpecIR(IRModel):
    """Typed execution IR for Phase 1 subpixel interface policy selection."""

    type: Literal["SubpixelSpecIR"] = "SubpixelSpecIR"
    component_type: Literal["SubpixelSpec"] = "SubpixelSpec"
    dielectric: SubpixelPolicyIR
    metal: SubpixelPolicyIR
    pec: SubpixelPolicyIR
    pmc: SubpixelPolicyIR
    lossy_metal: SubpixelPolicyIR
    averaging_targets: tuple[str, ...] = ()
    staircasing_targets: tuple[str, ...] = ()
    requires_anisotropic_materialization: bool
    courant_ratio: float


class PeriodicIR(IRModel):
    """Typed execution IR for a periodic boundary edge."""

    type: Literal["PeriodicIR"] = "PeriodicIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["Periodic"] = "Periodic"
    name: str | None = None


class BlochBoundaryIR(IRModel):
    """Typed execution IR for a Bloch boundary edge."""

    type: Literal["BlochBoundaryIR"] = "BlochBoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["BlochBoundary"] = "BlochBoundary"
    name: str | None = None
    bloch_vec: float
    phase_real: float
    phase_imag: float


class PECBoundaryIR(IRModel):
    """Typed execution IR for a PEC boundary edge."""

    type: Literal["PECBoundaryIR"] = "PECBoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["PECBoundary"] = "PECBoundary"
    name: str | None = None


class PMCBoundaryIR(IRModel):
    """Typed execution IR for a PMC boundary edge."""

    type: Literal["PMCBoundaryIR"] = "PMCBoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["PMCBoundary"] = "PMCBoundary"
    name: str | None = None


class ABCBoundaryIR(IRModel):
    """Typed execution IR for a first-order absorbing boundary edge."""

    type: Literal["ABCBoundaryIR"] = "ABCBoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["ABCBoundary"] = "ABCBoundary"
    name: str | None = None
    permittivity: float | None = None
    conductivity: float | None = None
    requires_material_inference: bool = False


class PMLParamsIR(IRModel):
    """Typed execution IR for baseline PML profile parameters."""

    type: Literal["PMLParamsIR"] = "PMLParamsIR"
    sigma_order: int
    sigma_min: float
    sigma_max: float
    kappa_order: int
    kappa_min: float
    kappa_max: float
    alpha_order: int
    alpha_min: float
    alpha_max: float


class PMLIR(IRModel):
    """Typed execution IR for a baseline PML boundary edge."""

    type: Literal["PMLIR"] = "PMLIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["PML"] = "PML"
    name: str | None = None
    num_layers: int
    parameters: PMLParamsIR
    extrude_structures: bool = True


class StablePMLIR(IRModel):
    """Typed execution IR for a stable-PML boundary edge."""

    type: Literal["StablePMLIR"] = "StablePMLIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["StablePML"] = "StablePML"
    name: str | None = None
    num_layers: int
    parameters: PMLParamsIR
    extrude_structures: bool = True


class AbsorberParamsIR(IRModel):
    """Typed execution IR for adiabatic absorber parameters."""

    type: Literal["AbsorberParamsIR"] = "AbsorberParamsIR"
    sigma_order: int
    sigma_min: float
    sigma_max: float


class AbsorberIR(IRModel):
    """Typed execution IR for an adiabatic absorber edge."""

    type: Literal["AbsorberIR"] = "AbsorberIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["Absorber"] = "Absorber"
    name: str | None = None
    num_layers: int
    parameters: AbsorberParamsIR
    extrude_structures: bool = False


class ModeABCBoundaryIR(IRModel):
    """Typed execution IR for the deferred mode-absorbing boundary surface."""

    type: Literal["ModeABCBoundaryIR"] = "ModeABCBoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["ModeABCBoundary"] = "ModeABCBoundary"
    name: str | None = None
    mode_spec: dict[str, Any]
    mode_index: int
    freq_spec: float | dict[str, Any] | None = None
    plane: BoxIR
    phase1_policy: str


class SymmetryAxisIR(IRModel):
    """Typed execution IR for one mirror-symmetry axis."""

    type: Literal["SymmetryAxisIR"] = "SymmetryAxisIR"
    axis: Literal["x", "y", "z"]
    symmetry: Literal[-1, 1]
    electric_signs: tuple[int, int, int]
    magnetic_signs: tuple[int, int, int]


BoundaryEdgeIR = (
    PeriodicIR
    | BlochBoundaryIR
    | PECBoundaryIR
    | PMCBoundaryIR
    | ABCBoundaryIR
    | PMLIR
    | StablePMLIR
    | AbsorberIR
    | ModeABCBoundaryIR
    | ComponentIR
)


class BoundaryIR(IRModel):
    """Typed execution IR for one axis boundary pair."""

    type: Literal["BoundaryIR"] = "BoundaryIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["Boundary"] = "Boundary"
    plus: BoundaryEdgeIR
    minus: BoundaryEdgeIR


class BoundarySpecIR(IRModel):
    """Typed execution IR for the Phase 1 boundary container."""

    type: Literal["BoundarySpecIR"] = "BoundarySpecIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["BoundarySpec"] = "BoundarySpec"
    x: BoundaryIR
    y: BoundaryIR
    z: BoundaryIR
    requires_complex_fields: bool = False
    periodic_axes: tuple[str, ...] = ()
    bloch_axes: tuple[str, ...] = ()
    reflective_axes: tuple[str, ...] = ()
    abc_axes: tuple[str, ...] = ()
    abc_faces: tuple[str, ...] = ()
    pml_axes: tuple[str, ...] = ()
    pml_faces: tuple[str, ...] = ()
    stable_pml_axes: tuple[str, ...] = ()
    stable_pml_faces: tuple[str, ...] = ()
    absorber_axes: tuple[str, ...] = ()
    absorber_faces: tuple[str, ...] = ()
    active_symmetry_axes: tuple[str, ...] = ()
    symmetry_axes: tuple[SymmetryAxisIR, ...] = ()


class FaceHaloIR(IRModel):
    """Typed execution IR for one face halo descriptor."""

    type: Literal["FaceHaloIR"] = "FaceHaloIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["FaceHalo"] = "FaceHalo"
    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    depth: int = 1
    neighbor_chunk_index: tuple[int, int, int] | None = None
    exchange_kind: str = "interior"
    phase_factor: complex = 1.0 + 0.0j
    electric_signs: tuple[int, int, int] = (1, 1, 1)
    magnetic_signs: tuple[int, int, int] = (1, 1, 1)


class ChunkSpecIR(IRModel):
    """Typed execution IR for one chunk's planning metadata."""

    type: Literal["ChunkSpecIR"] = "ChunkSpecIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["ChunkSpec"] = "ChunkSpec"
    chunk_index: tuple[int, int, int]
    global_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    interior_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    owned_bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    local_grid_shape: tuple[int, int, int]
    face_halos: tuple[FaceHaloIR, ...] = ()
    is_reduced: bool = False
    symmetry_multiplicity: int = 1
    source_indices: tuple[int, ...] = ()
    monitor_indices: tuple[int, ...] = ()


class ExchangeDescriptorIR(IRModel):
    """Typed execution IR for one halo exchange operation."""

    type: Literal["ExchangeDescriptorIR"] = "ExchangeDescriptorIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["ExchangeDescriptor"] = "ExchangeDescriptor"
    source_chunk_index: tuple[int, int, int]
    dest_chunk_index: tuple[int, int, int]
    axis: Literal["x", "y", "z"]
    source_side: Literal["minus", "plus"]
    dest_side: Literal["minus", "plus"]
    exchange_kind: str
    phase_factor: complex = 1.0 + 0.0j


class ChunkLayoutIR(IRModel):
    """Typed execution IR for the global chunk decomposition plan."""

    type: Literal["ChunkLayoutIR"] = "ChunkLayoutIR"
    family: Literal["boundary"] = "boundary"
    component_type: Literal["ChunkLayout"] = "ChunkLayout"
    num_chunks: tuple[int, int, int]
    total_chunks: int
    chunks: tuple[ChunkSpecIR, ...]
    periodic_axes: tuple[str, ...] = ()
    bloch_axes: tuple[str, ...] = ()
    exchange_plan: tuple[ExchangeDescriptorIR, ...] = ()
    device_assignment: tuple[int, ...] = ()
    rank_assignment: tuple[int, ...] = ()


class MediumIR(IRModel):
    """Typed execution IR for a homogeneous isotropic dielectric or conductor."""

    type: Literal["MediumIR"] = "MediumIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["Medium"] = "Medium"
    name: str | None = None
    permittivity: float
    conductivity: float
    permeability: float
    magnetic_conductivity: float


class PECMediumIR(IRModel):
    """Typed execution IR for a perfect electric conductor marker."""

    type: Literal["PECMediumIR"] = "PECMediumIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["PECMedium"] = "PECMedium"
    name: str | None = None


class PMCMediumIR(IRModel):
    """Typed execution IR for a perfect magnetic conductor marker."""

    type: Literal["PMCMediumIR"] = "PMCMediumIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["PMCMedium"] = "PMCMedium"
    name: str | None = None


class PoleResidueIR(IRModel):
    """Typed execution IR for a pole-residue dispersive medium."""

    type: Literal["PoleResidueIR"] = "PoleResidueIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["PoleResidue"] = "PoleResidue"
    name: str | None = None
    eps_inf: float
    poles: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


class SellmeierIR(IRModel):
    """Typed execution IR for a Sellmeier optical-dispersion medium."""

    type: Literal["SellmeierIR"] = "SellmeierIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["Sellmeier"] = "Sellmeier"
    name: str | None = None
    coeffs: tuple[tuple[float, float], ...]


class LorentzIR(IRModel):
    """Typed execution IR for a Lorentz dispersive medium."""

    type: Literal["LorentzIR"] = "LorentzIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["Lorentz"] = "Lorentz"
    name: str | None = None
    eps_inf: float
    coeffs: tuple[tuple[float, float, float], ...]


class DrudeIR(IRModel):
    """Typed execution IR for a Drude dispersive medium."""

    type: Literal["DrudeIR"] = "DrudeIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["Drude"] = "Drude"
    name: str | None = None
    eps_inf: float
    coeffs: tuple[tuple[float, float], ...]


class DebyeIR(IRModel):
    """Typed execution IR for a Debye dispersive medium."""

    type: Literal["DebyeIR"] = "DebyeIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["Debye"] = "Debye"
    name: str | None = None
    eps_inf: float
    coeffs: tuple[tuple[float, float], ...]


class AnisotropicMediumIR(IRModel):
    """Typed execution IR for a diagonal anisotropic medium."""

    type: Literal["AnisotropicMediumIR"] = "AnisotropicMediumIR"
    family: Literal["medium"] = "medium"
    component_type: Literal["AnisotropicMedium"] = "AnisotropicMedium"
    name: str | None = None
    xx: (
        MediumIR
        | PECMediumIR
        | PMCMediumIR
        | PoleResidueIR
        | SellmeierIR
        | LorentzIR
        | DrudeIR
        | DebyeIR
    )
    yy: (
        MediumIR
        | PECMediumIR
        | PMCMediumIR
        | PoleResidueIR
        | SellmeierIR
        | LorentzIR
        | DrudeIR
        | DebyeIR
    )
    zz: (
        MediumIR
        | PECMediumIR
        | PMCMediumIR
        | PoleResidueIR
        | SellmeierIR
        | LorentzIR
        | DrudeIR
        | DebyeIR
    )


MediumComponentIR = (
    MediumIR
    | PECMediumIR
    | PMCMediumIR
    | PoleResidueIR
    | SellmeierIR
    | LorentzIR
    | DrudeIR
    | DebyeIR
    | AnisotropicMediumIR
    | ComponentIR
)


class StructureIR(IRModel):
    """Lowered structure shell for scene compilation and materialization."""

    type: Literal["StructureIR"] = "StructureIR"
    name: str | None = None
    original_type: str = "Structure"
    source_index: int = 0
    priority: int | None = None
    resolved_priority: int = 0
    precedence_rank: int = 0
    geometry: GeometryIR
    medium: MediumComponentIR
    background_medium: MediumComponentIR | None = None


class SceneIR(IRModel):
    """Ordered scene IR used by later compilation stages."""

    type: Literal["SceneIR"] = "SceneIR"
    original_type: str = "Scene"
    structure_priority_mode: str = StructurePriorityMode.EQUAL.value
    background_medium: MediumComponentIR
    structures: tuple[StructureIR, ...] = ()


SourceComponentIR = ComponentIR | UniformCurrentSourceIR | PointDipoleIR | CustomCurrentSourceIR | CustomFieldSourceIR | ModeSourceIR | PlaneWaveIR | GaussianBeamIR | AstigmaticGaussianBeamIR | TFSFIR


# --------------------------------------------------------------------
# Monitor IR models
# --------------------------------------------------------------------


class FieldMonitorIR(IRModel):
    """Typed execution IR for a field monitor."""

    type: Literal["FieldMonitorIR"] = "FieldMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FieldMonitor"] = "FieldMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    overwrite: bool = True


class FieldTimeMonitorIR(IRModel):
    """Typed execution IR for a time-domain field monitor."""

    type: Literal["FieldTimeMonitorIR"] = "FieldTimeMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class AuxFieldTimeMonitorIR(IRModel):
    """Typed execution IR for an auxiliary field time monitor."""

    type: Literal["AuxFieldTimeMonitorIR"] = "AuxFieldTimeMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["AuxFieldTimeMonitor"] = "AuxFieldTimeMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class FluxMonitorIR(IRModel):
    """Typed execution IR for a flux monitor."""

    type: Literal["FluxMonitorIR"] = "FluxMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FluxMonitor"] = "FluxMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class FluxTimeMonitorIR(IRModel):
    """Typed execution IR for a time-domain flux monitor."""

    type: Literal["FluxTimeMonitorIR"] = "FluxTimeMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"


class ModeMonitorIR(IRModel):
    """Typed execution IR for a mode monitor."""

    type: Literal["ModeMonitorIR"] = "ModeMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["ModeMonitor"] = "ModeMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    mode_spec: dict[str, Any] | None = None


class ModeSolverMonitorIR(IRModel):
    """Typed execution IR for a mode solver monitor."""

    type: Literal["ModeSolverMonitorIR"] = "ModeSolverMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["ModeSolverMonitor"] = "ModeSolverMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    mode_spec: dict[str, Any] | None = None


class MediumMonitorIR(IRModel):
    """Typed execution IR for a medium monitor."""

    type: Literal["MediumMonitorIR"] = "MediumMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["MediumMonitor"] = "MediumMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class PermittivityMonitorIR(IRModel):
    """Typed execution IR for a permittivity monitor."""

    type: Literal["PermittivityMonitorIR"] = "PermittivityMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["PermittivityMonitor"] = "PermittivityMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0


class FieldProjectionAngleMonitorIR(IRModel):
    """Typed execution IR for an angle-space field projection monitor."""

    type: Literal["FieldProjectionAngleMonitorIR"] = "FieldProjectionAngleMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FieldProjectionAngleMonitor"] = "FieldProjectionAngleMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    phi: tuple[float, float, int] = (-90.0, 90.0, 181)
    theta: tuple[float, float, int] = (0.0, 180.0, 181)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class FieldProjectionCartesianMonitorIR(IRModel):
    """Typed execution IR for a Cartesian field projection monitor."""

    type: Literal["FieldProjectionCartesianMonitorIR"] = "FieldProjectionCartesianMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FieldProjectionCartesianMonitor"] = "FieldProjectionCartesianMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    x: tuple[float, float, int] = (-50.0, 50.0, 201)
    y: tuple[float, float, int] = (-50.0, 50.0, 201)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class FieldProjectionKSpaceMonitorIR(IRModel):
    """Typed execution IR for a k-space field projection monitor."""

    type: Literal["FieldProjectionKSpaceMonitorIR"] = "FieldProjectionKSpaceMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["FieldProjectionKSpaceMonitor"] = "FieldProjectionKSpaceMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    num_k: int = 1
    kx: tuple[float, float, int] = (-10.0, 10.0, 21)
    ky: tuple[float, float, int] = (-10.0, 10.0, 21)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class DiffractionMonitorIR(IRModel):
    """Typed execution IR for a diffraction monitor."""

    type: Literal["DiffractionMonitorIR"] = "DiffractionMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["DiffractionMonitor"] = "DiffractionMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class DirectivityMonitorIR(IRModel):
    """Typed execution IR for a directivity monitor."""

    type: Literal["DirectivityMonitorIR"] = "DirectivityMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["DirectivityMonitor"] = "DirectivityMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    normal_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    projection_distance: float = 1e5
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class GaussianOverlapMonitorIR(IRModel):
    """Typed execution IR for a Gaussian overlap monitor."""

    type: Literal["GaussianOverlapMonitorIR"] = "GaussianOverlapMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["GaussianOverlapMonitor"] = "GaussianOverlapMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    # Beam parameters
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_radius: float = 1.0
    waist_distance: float = 0.0


class AstigmaticGaussianOverlapMonitorIR(IRModel):
    """Typed execution IR for an astigmatic Gaussian overlap monitor."""

    type: Literal["AstigmaticGaussianOverlapMonitorIR"] = "AstigmaticGaussianOverlapMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["AstigmaticGaussianOverlapMonitor"] = "AstigmaticGaussianOverlapMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    direction: Literal["+", "-"] = "+"
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()
    # Beam parameters
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_sizes: tuple[float, float] = (1.0, 1.0)
    waist_distances: tuple[float, float] = (0.0, 0.0)


class SurfaceFieldMonitorIR(IRModel):
    """Typed execution IR for a surface field monitor."""

    type: Literal["SurfaceFieldMonitorIR"] = "SurfaceFieldMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["SurfaceFieldMonitor"] = "SurfaceFieldMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    num_freqs: int = 1
    freqs: tuple[float, ...] = ()


class SurfaceFieldTimeMonitorIR(IRModel):
    """Typed execution IR for a surface field time monitor."""

    type: Literal["SurfaceFieldTimeMonitorIR"] = "SurfaceFieldTimeMonitorIR"
    family: Literal["monitor"] = "monitor"
    component_type: Literal["SurfaceFieldTimeMonitor"] = "SurfaceFieldTimeMonitor"
    name: str | None = None
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    interval: int = 1
    start: int = 0
    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


MonitorComponentIR = (
    ComponentIR
    | FieldMonitorIR
    | FieldTimeMonitorIR
    | AuxFieldTimeMonitorIR
    | FluxMonitorIR
    | FluxTimeMonitorIR
    | ModeMonitorIR
    | ModeSolverMonitorIR
    | MediumMonitorIR
    | PermittivityMonitorIR
    | FieldProjectionAngleMonitorIR
    | FieldProjectionCartesianMonitorIR
    | FieldProjectionKSpaceMonitorIR
    | DiffractionMonitorIR
    | DirectivityMonitorIR
    | GaussianOverlapMonitorIR
    | AstigmaticGaussianOverlapMonitorIR
    | SurfaceFieldMonitorIR
    | SurfaceFieldTimeMonitorIR
)


class SimulationIR(IRModel):
    """Top-level execution IR for a Phase 1 simulation."""

    type: Literal["SimulationIR"] = "SimulationIR"
    original_type: str = "Simulation"
    autofdtd_version: str = __version__
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    run_time: float
    courant: float
    normalize_index: int | None = None
    symmetry: tuple[int, int, int]
    shutoff: float | None = None
    scene: SceneIR
    sources: tuple[SourceComponentIR, ...] = ()
    monitors: tuple[MonitorComponentIR, ...] = ()
    boundary_spec: BoundarySpecIR | ComponentIR | None = None
    grid_spec: GridSpecIR | ComponentIR | None = None
    subpixel: SubpixelSpecIR | ComponentIR | None = None
    runtime_controls: RuntimeControlsIR | None = None


class RuntimeControlsIR(IRModel):
    """Typed execution IR for compiled timestep-count and early-stop behavior."""

    type: Literal["RuntimeControlsIR"] = "RuntimeControlsIR"
    dt: float | None = None
    num_time_steps: int | None = None
    max_step_index: int | None = None
    scaled_courant: float
    cfl_spacings: tuple[float, ...] = ()
    primary_stop_reason: Literal["run_time", "step_count"]
    convergence_policy: Literal["none", "integrated_electric_field"]
    shutoff: float | None = None
    shutoff_check_interval: int | None = None
    normalize_index: int | None = None
    user_step_limit: int | None = None


class ArtifactIR(IRModel):
    """Manifest entry for an external or packaged execution artifact."""

    type: Literal["ArtifactIR"] = "ArtifactIR"
    logical_path: str
    role: str
    media_type: str
    sha256: str
    byte_count: int


class ExecutionManifestIR(IRModel):
    """Transport metadata for a packaged IR bundle."""

    type: Literal["ExecutionManifestIR"] = "ExecutionManifestIR"
    package_format: str = EXECUTION_PACKAGE_FORMAT
    autofdtd_version: str = __version__
    entrypoint: str = "simulation.json"
    artifacts: tuple[ArtifactIR, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("package_format")
    @classmethod
    def _validate_package_format(cls, value: str) -> str:
        if value != EXECUTION_PACKAGE_FORMAT:
            raise ValueError(f"package_format must be {EXECUTION_PACKAGE_FORMAT!r}")
        return value


class ExecutionPackageIR(IRModel):
    """Filesystem- and transport-ready bundle for local or remote execution."""

    type: Literal["ExecutionPackageIR"] = "ExecutionPackageIR"
    manifest: ExecutionManifestIR
    simulation: SimulationIR

    @classmethod
    def from_simulation(
        cls,
        simulation: Simulation,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionPackageIR:
        """Lower a public simulation shell into the versioned execution IR."""
        simulation_ir = simulation_to_ir(simulation)
        manifest = ExecutionManifestIR(metadata=metadata or {})
        return cls(manifest=manifest, simulation=simulation_ir)

    def write_bundle(self, root: str | Path) -> Path:
        """Write a directory bundle with manifest and simulation payloads."""
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        simulation_path = root_path / self.manifest.entrypoint
        simulation_path.write_text(self.simulation.to_json_text() + "\n", encoding="utf-8")
        artifact = ArtifactIR(
            logical_path=self.manifest.entrypoint,
            role="simulation",
            media_type="application/json",
            sha256=_sha256_file(simulation_path),
            byte_count=simulation_path.stat().st_size,
        )
        manifest = self.manifest.model_copy(update={"artifacts": (artifact,)})
        manifest_path = root_path / "manifest.json"
        manifest_path.write_text(manifest.to_json_text() + "\n", encoding="utf-8")
        return manifest_path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _component_payload(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise TypeError(f"component payload must be mapping-like, got {type(value)!r}")
    return {str(key): json_ready(item) for key, item in raw.items()}


def component_to_ir(value: object, *, family: str) -> SourceComponentIR:
    """Normalize a tagged public component into a transport-safe IR envelope."""
    if family == ComponentFamily.SOURCE and isinstance(
        value, (UniformCurrentSource, PointDipole, CustomCurrentSource, CustomFieldSource, ModeSource, PlaneWave, GaussianBeam, AstigmaticGaussianBeam, TFSF)
    ):
        return source_to_ir(value)
    if (
        family == ComponentFamily.SOURCE
        and isinstance(value, Mapping)
        and str(value.get("type")) in {"UniformCurrentSource", "PointDipole", "CustomCurrentSource", "CustomFieldSource", "ModeSource", "PlaneWave", "GaussianBeam", "AstigmaticGaussianBeam", "TFSF"}
    ):
        return source_to_ir(value)
    raw = _component_payload(value)
    if family == ComponentFamily.SOURCE and "source_time" in raw:
        raw["source_time"] = json_ready(source_time_to_ir(raw["source_time"]))
    component_type = str(raw.pop("type", value.__class__.__name__))
    name = raw.get("name")
    return ComponentIR(
        family=family,
        component_type=component_type,
        name=str(name) if isinstance(name, str) else None,
        payload=raw,
    )


def source_to_ir(value: object) -> SourceComponentIR:
    """Lower supported Phase 1 source models into typed execution IR."""

    if isinstance(value, UniformCurrentSource):
        return UniformCurrentSourceIR(
            center=value.center,
            size=value.size,
            polarization=value.polarization,
            field_kind=value.field_kind,
            component_axis=value.component_axis,
            placement_kind=value.placement_kind,
            zero_size_axes=value.zero_size_axes,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            current_amplitude_definition=value.current_amplitude_definition,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
        )
    if isinstance(value, PointDipole):
        return PointDipoleIR(
            center=value.center,
            polarization=value.polarization,
            field_kind=value.field_kind,
            component_axis=value.component_axis,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
        )
    if isinstance(value, CustomCurrentSource):
        return CustomCurrentSourceIR(
            center=value.center,
            size=value.size,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            e_fields=value.e_fields,
            h_fields=value.h_fields,
            coordinates=value.coordinates,
            has_electric=value.has_electric,
            has_magnetic=value.has_magnetic,
        )
    if isinstance(value, CustomFieldSource):
        return CustomFieldSourceIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            e_fields=value.e_fields,
            h_fields=value.h_fields,
            coordinates=value.coordinates,
            has_electric=value.has_electric,
            has_magnetic=value.has_magnetic,
            has_tangential_fields=value.has_tangential_fields,
        )
    if isinstance(value, ModeSource):
        return ModeSourceIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            mode_index=value.mode_index,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            e_fields=None,  # Populated by mode solver during compilation
            h_fields=None,  # Populated by mode solver during compilation
            mode_neff=None,  # Populated by mode solver during compilation
            mode_power=1.0,
        )
    if isinstance(value, PlaneWave):
        return PlaneWaveIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            angle_theta=value.angle_theta,
            angle_phi=value.angle_phi,
            pol_angle=value.pol_angle,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            angular_spec=_angular_spec_to_ir(value.angular_spec),
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            dir_vector=value._dir_vector,
            pol_vector=value._pol_vector,
        )
    if isinstance(value, GaussianBeam):
        return GaussianBeamIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            angle_theta=value.angle_theta,
            angle_phi=value.angle_phi,
            pol_angle=value.pol_angle,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            angular_spec=_angular_spec_to_ir(value.angular_spec),
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            dir_vector=value._dir_vector,
            pol_vector=value._pol_vector,
            waist_radius=value.waist_radius,
            waist_distance=value.waist_distance,
            reference_wavelength=value._reference_wavelength,
        )
    if isinstance(value, AstigmaticGaussianBeam):
        return AstigmaticGaussianBeamIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            angle_theta=value.angle_theta,
            angle_phi=value.angle_phi,
            pol_angle=value.pol_angle,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            angular_spec=_angular_spec_to_ir(value.angular_spec),
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            dir_vector=value._dir_vector,
            pol_vector=value._pol_vector,
            waist_radius_x=value.waist_radius_x,
            waist_radius_y=value.waist_radius_y,
            waist_distance_x=value.waist_distance_x,
            waist_distance_y=value.waist_distance_y,
            reference_wavelength=value._reference_wavelength,
        )
    if isinstance(value, TFSF):
        return TFSFIR(
            center=value.center,
            size=value.size,
            direction=value.direction,
            angle_theta=value.angle_theta,
            angle_phi=value.angle_phi,
            pol_angle=value.pol_angle,
            injection_axis=value.injection_axis,
            placement_kind=value.placement_kind,
            interpolate=value.interpolate,
            confine_to_bounds=value.confine_to_bounds,
            source_time=source_time_to_ir(value.source_time),
            name=value.name,
            support_bounds=value.support_bounds,
            dir_vector=value._dir_vector,
            pol_vector=value._pol_vector,
            num_freqs=value.num_freqs,
            injection_plane_center=value.injection_plane_center,
            reference_wavelength=value._reference_wavelength,
        )
    if isinstance(value, Mapping) and str(value.get("type")) == "UniformCurrentSource":
        return source_to_ir(UniformCurrentSource.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "PointDipole":
        return source_to_ir(PointDipole.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "CustomCurrentSource":
        return source_to_ir(CustomCurrentSource.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "CustomFieldSource":
        return source_to_ir(CustomFieldSource.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "ModeSource":
        return source_to_ir(ModeSource.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "PlaneWave":
        return source_to_ir(PlaneWave.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "GaussianBeam":
        return source_to_ir(GaussianBeam.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "AstigmaticGaussianBeam":
        return source_to_ir(AstigmaticGaussianBeam.model_validate(value))
    if isinstance(value, Mapping) and str(value.get("type")) == "TFSF":
        return source_to_ir(TFSF.model_validate(value))
    raise TypeError(f"unsupported Phase 1 source type for IR lowering: {type(value)!r}")


def _angular_spec_to_ir(value: object) -> AngularSpecIR:
    """Lower an angular spec model into typed execution IR."""
    spec = angular_spec_model_from_value(value)
    if isinstance(spec, FixedInPlaneKSpec):
        return FixedInPlaneKSpecIR()
    if isinstance(spec, FixedAngleSpec):
        return FixedAngleSpecIR()
    raise TypeError(f"unsupported angular spec type: {type(spec)!r}")


def source_time_to_ir(value: object) -> SourceTimeIR:
    """Lower a supported Phase 1 source-time profile into typed execution IR."""

    source_time = source_time_model_from_value(value)
    if isinstance(source_time, GaussianPulse):
        return GaussianPulseIR(
            amplitude=source_time.amplitude,
            phase=source_time.phase,
            freq0=source_time.freq0,
            fwidth=source_time.fwidth,
            offset=source_time.offset,
            remove_dc_component=source_time.remove_dc_component,
            twidth=source_time.twidth,
            offset_time=source_time.offset_time,
            peak_time=source_time.peak_time,
            peak_frequency=source_time.peak_frequency,
            end_time=source_time.end_time(),
            frequency_range=source_time.frequency_range(),
        )
    if isinstance(source_time, ContinuousWave):
        return ContinuousWaveIR(
            amplitude=source_time.amplitude,
            phase=source_time.phase,
            freq0=source_time.freq0,
            fwidth=source_time.fwidth,
            offset=source_time.offset,
            twidth=source_time.twidth,
            offset_time=source_time.offset_time,
            end_time=source_time.end_time(),
            frequency_range=source_time.frequency_range(),
        )
    if isinstance(source_time, BroadbandPulse):
        return BroadbandPulseIR(
            amplitude=source_time.amplitude,
            phase=source_time.phase,
            freq_range=source_time.freq_range,
            minimum_amplitude=source_time.minimum_amplitude,
            offset=source_time.offset,
            freq0=source_time.freq0,
            bandwidth=source_time.bandwidth,
            fwidth=source_time.fwidth,
            twidth=source_time.twidth,
            offset_time=source_time.offset_time,
            peak_time=source_time.peak_time,
            end_time=source_time.end_time(),
            frequency_range=source_time.frequency_range(),
        )
    if isinstance(source_time, CustomSourceTime):
        return CustomSourceTimeIR(
            amplitude=source_time.amplitude,
            phase=source_time.phase,
            freq0=source_time.freq0,
            fwidth=source_time.fwidth,
            offset=source_time.offset,
            twidth=source_time.twidth,
            offset_time=source_time.offset_time,
            time_samples=source_time.time_samples,
            envelope_values=source_time.envelope_values,
            sample_count=len(source_time.time_samples),
            end_time=source_time.end_time(),
            frequency_range=source_time.frequency_range(),
        )
    raise TypeError(f"unsupported source-time type {type(source_time)!r}")


def geometry_transform_to_ir(value: GeometryTransform) -> GeometryTransformIR:
    """Lower a primitive geometry transform into inspectable IR metadata."""
    return GeometryTransformIR(origin=value.origin, axes=value.axes)


def geometry_to_ir(value: object) -> GeometryIR:
    """Lower supported primitive geometries into typed IR, else fall back to ComponentIR."""
    geometry: (
        Box
        | Sphere
        | Cylinder
        | PolySlab
        | GeometryGroup
        | Transformed
        | ClipOperation
        | GeometryArray
        | None
    ) = None
    if isinstance(
        value,
        (
            Box,
            Sphere,
            Cylinder,
            PolySlab,
            GeometryGroup,
            Transformed,
            ClipOperation,
            GeometryArray,
        ),
    ):
        geometry = value
    elif (
        isinstance(value, Mapping)
        and str(value.get("type"))
        in {
            "Box",
            "Sphere",
            "Cylinder",
            "PolySlab",
            "GeometryGroup",
            "Transformed",
            "ClipOperation",
            "GeometryArray",
        }
    ):
        geometry = geometry_model_from_value(value)
    if isinstance(geometry, Box):
        bounds_min, bounds_max = geometry.bounds
        return BoxIR(
            center=geometry.center,
            size=geometry.size,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            transform=geometry_transform_to_ir(geometry.transform),
        )
    if isinstance(geometry, Sphere):
        bounds_min, bounds_max = geometry.bounds
        return SphereIR(
            center=geometry.center,
            radius=geometry.radius,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            transform=geometry_transform_to_ir(geometry.transform),
        )
    if isinstance(geometry, Cylinder):
        bounds_min, bounds_max = geometry.bounds
        return CylinderIR(
            center=geometry.center,
            radius=geometry.radius,
            length=geometry.length,
            axis=geometry.axis,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            transform=geometry_transform_to_ir(geometry.transform),
        )
    if isinstance(geometry, PolySlab):
        bounds_min, bounds_max = geometry.bounds
        return PolySlabIR(
            vertices=geometry.vertices,
            slab_bounds=geometry.slab_bounds,
            axis=geometry.axis,
            sidewall_angle=geometry.sidewall_angle,
            dilation=geometry.dilation,
            reference_plane=geometry.reference_plane,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            transform=geometry_transform_to_ir(geometry.transform),
        )
    if isinstance(geometry, GeometryGroup):
        bounds_min, bounds_max = geometry.bounds
        return GeometryGroupIR(
            geometries=tuple(geometry_to_ir(item) for item in geometry.geometries),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
    if isinstance(geometry, Transformed):
        bounds_min, bounds_max = geometry.bounds
        return TransformedIR(
            geometry=geometry_to_ir(geometry.geometry),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            transform=geometry_transform_to_ir(geometry.transform),
        )
    if isinstance(geometry, ClipOperation):
        bounds_min, bounds_max = geometry.bounds
        return ClipOperationIR(
            operation=geometry.operation,
            geometry_a=geometry_to_ir(geometry.geometry_a),
            geometry_b=geometry_to_ir(geometry.geometry_b),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
    if isinstance(geometry, GeometryArray):
        return geometry_to_ir(geometry.as_geometry_group())
    return component_to_ir(value, family=ComponentFamily.GEOMETRY)


def medium_to_ir(value: object) -> MediumComponentIR:
    """Lower supported isotropic media into typed execution IR."""

    medium: (
        Medium
        | PECMedium
        | PMCMedium
        | PoleResidue
        | Sellmeier
        | Lorentz
        | Drude
        | Debye
        | AnisotropicMedium
        | None
    ) = None
    if isinstance(
        value,
        (
            Medium,
            PECMedium,
            PMCMedium,
            PoleResidue,
            Sellmeier,
            Lorentz,
            Drude,
            Debye,
            AnisotropicMedium,
        ),
    ):
        medium = value
    elif (
        isinstance(value, Mapping)
        and str(value.get("type"))
        in {
            "Medium",
            "PECMedium",
            "PMCMedium",
            "PoleResidue",
            "Sellmeier",
            "Lorentz",
            "Drude",
            "Debye",
            "AnisotropicMedium",
        }
    ):
        medium = medium_model_from_value(value)

    if isinstance(medium, Medium):
        return MediumIR(
            name=medium.name,
            permittivity=medium.permittivity,
            conductivity=medium.conductivity,
            permeability=medium.permeability,
            magnetic_conductivity=medium.magnetic_conductivity,
        )
    if isinstance(medium, PECMedium):
        return PECMediumIR(name=medium.name)
    if isinstance(medium, PMCMedium):
        return PMCMediumIR(name=medium.name)
    if isinstance(medium, PoleResidue):
        return PoleResidueIR(name=medium.name, eps_inf=medium.eps_inf, poles=medium.poles)
    if isinstance(medium, Sellmeier):
        return SellmeierIR(name=medium.name, coeffs=medium.coeffs)
    if isinstance(medium, Lorentz):
        return LorentzIR(name=medium.name, eps_inf=medium.eps_inf, coeffs=medium.coeffs)
    if isinstance(medium, Drude):
        return DrudeIR(name=medium.name, eps_inf=medium.eps_inf, coeffs=medium.coeffs)
    if isinstance(medium, Debye):
        return DebyeIR(name=medium.name, eps_inf=medium.eps_inf, coeffs=medium.coeffs)
    if isinstance(medium, AnisotropicMedium):
        return AnisotropicMediumIR(
            name=medium.name,
            xx=medium_to_ir(medium.xx),
            yy=medium_to_ir(medium.yy),
            zz=medium_to_ir(medium.zz),
        )
    return component_to_ir(value, family=ComponentFamily.MEDIUM)


def boundary_edge_to_ir(value: object) -> BoundaryEdgeIR:
    """Lower a supported boundary edge into typed execution IR."""

    edge: (
        Periodic
        | BlochBoundary
        | PECBoundary
        | PMCBoundary
        | ABCBoundary
        | PML
        | StablePML
        | Absorber
        | ModeABCBoundary
        | None
    ) = None
    if isinstance(
        value,
        (
            Periodic,
            BlochBoundary,
            PECBoundary,
            PMCBoundary,
            ABCBoundary,
            PML,
            StablePML,
            Absorber,
            ModeABCBoundary,
        ),
    ):
        edge = value
    elif isinstance(value, Mapping) and str(value.get("type")) in {
        "Periodic",
        "BlochBoundary",
        "PECBoundary",
        "PMCBoundary",
        "ABCBoundary",
        "PML",
        "StablePML",
        "Absorber",
        "ModeABCBoundary",
    }:
        edge = boundary_edge_model_from_value(value)

    if isinstance(edge, Periodic):
        return PeriodicIR(name=edge.name)
    if isinstance(edge, BlochBoundary):
        return BlochBoundaryIR(
            name=edge.name,
            bloch_vec=edge.bloch_vec,
            phase_real=float(edge.bloch_phase.real),
            phase_imag=float(edge.bloch_phase.imag),
        )
    if isinstance(edge, PECBoundary):
        return PECBoundaryIR(name=edge.name)
    if isinstance(edge, PMCBoundary):
        return PMCBoundaryIR(name=edge.name)
    if isinstance(edge, ABCBoundary):
        return ABCBoundaryIR(
            name=edge.name,
            permittivity=edge.permittivity,
            conductivity=edge.conductivity,
            requires_material_inference=edge.permittivity is None,
        )
    if isinstance(edge, PML):
        params = (
            edge.parameters
            if isinstance(edge.parameters, PMLParams)
            else PMLParams.model_validate(edge.parameters)
        )
        return PMLIR(
            name=edge.name,
            num_layers=edge.num_layers,
            parameters=PMLParamsIR(
                sigma_order=params.sigma_order,
                sigma_min=params.sigma_min,
                sigma_max=params.sigma_max,
                kappa_order=params.kappa_order,
                kappa_min=params.kappa_min,
                kappa_max=params.kappa_max,
                alpha_order=params.alpha_order,
                alpha_min=params.alpha_min,
                alpha_max=params.alpha_max,
            ),
            extrude_structures=edge.extrude_structures,
        )
    if isinstance(edge, StablePML):
        params = (
            edge.parameters
            if isinstance(edge.parameters, PMLParams)
            else PMLParams.model_validate(edge.parameters)
        )
        return StablePMLIR(
            name=edge.name,
            num_layers=edge.num_layers,
            parameters=PMLParamsIR(
                sigma_order=params.sigma_order,
                sigma_min=params.sigma_min,
                sigma_max=params.sigma_max,
                kappa_order=params.kappa_order,
                kappa_min=params.kappa_min,
                kappa_max=params.kappa_max,
                alpha_order=params.alpha_order,
                alpha_min=params.alpha_min,
                alpha_max=params.alpha_max,
            ),
            extrude_structures=edge.extrude_structures,
        )
    if isinstance(edge, Absorber):
        params = (
            edge.parameters
            if isinstance(edge.parameters, AbsorberParams)
            else AbsorberParams.model_validate(edge.parameters)
        )
        return AbsorberIR(
            name=edge.name,
            num_layers=edge.num_layers,
            parameters=AbsorberParamsIR(
                sigma_order=params.sigma_order,
                sigma_min=params.sigma_min,
                sigma_max=params.sigma_max,
            ),
            extrude_structures=edge.extrude_structures,
        )
    if isinstance(edge, ModeABCBoundary):
        freq_payload: float | dict[str, Any] | None
        if isinstance(edge.freq_spec, BroadbandModeABCSpec):
            freq_payload = edge.freq_spec.model_dump(mode="python", exclude_none=True)
        else:
            freq_payload = edge.freq_spec
        return ModeABCBoundaryIR(
            name=edge.name,
            mode_spec=dict(edge.mode_spec),
            mode_index=edge.mode_index,
            freq_spec=freq_payload,
            plane=geometry_to_ir(edge.plane),
            phase1_policy=edge.phase1_policy,
        )
    return component_to_ir(value, family=ComponentFamily.BOUNDARY)


def boundary_to_ir(value: object) -> BoundaryIR | ComponentIR:
    """Lower a supported axis boundary pair into typed execution IR."""

    boundary: Boundary | None = None
    if isinstance(value, Boundary):
        boundary = value
    elif isinstance(value, Mapping) and str(value.get("type")) in {
        "Boundary",
        "Periodic",
        "BlochBoundary",
        "PECBoundary",
        "PMCBoundary",
        "ABCBoundary",
        "PML",
        "StablePML",
        "Absorber",
        "ModeABCBoundary",
    }:
        boundary = boundary_model_from_value(value)
    if boundary is None:
        return component_to_ir(value, family=ComponentFamily.BOUNDARY)
    return BoundaryIR(
        plus=boundary_edge_to_ir(boundary.plus),
        minus=boundary_edge_to_ir(boundary.minus),
    )


def _reflection_signs_for_axis(
    symmetry: int,
    axis_name: Literal["x", "y", "z"],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    axis_index = "xyz".index(axis_name)
    spatial_reflection = [1, 1, 1]
    pseudo_reflection = [-1, -1, -1]
    spatial_reflection[axis_index] = -1
    pseudo_reflection[axis_index] = 1
    electric = tuple(sign * symmetry for sign in spatial_reflection)
    magnetic = tuple(sign * symmetry for sign in pseudo_reflection)
    return electric, magnetic


def _symmetry_axes_to_ir(symmetry: tuple[int, int, int]) -> tuple[SymmetryAxisIR, ...]:
    axes: list[SymmetryAxisIR] = []
    for axis_name, axis_value in zip("xyz", symmetry, strict=True):
        if axis_value == 0:
            continue
        electric_signs, magnetic_signs = _reflection_signs_for_axis(axis_value, axis_name)
        axes.append(
            SymmetryAxisIR(
                axis=axis_name,
                symmetry=axis_value,
                electric_signs=electric_signs,
                magnetic_signs=magnetic_signs,
            )
        )
    return tuple(axes)


def boundary_spec_to_ir(
    value: object,
    *,
    symmetry: tuple[int, int, int] = (0, 0, 0),
) -> BoundarySpecIR | ComponentIR:
    """Lower a supported BoundarySpec container into typed execution IR."""

    boundary_spec: BoundarySpec | None = None
    if isinstance(value, BoundarySpec):
        boundary_spec = value
    elif isinstance(value, Mapping) and str(value.get("type")) == "BoundarySpec":
        try:
            boundary_spec = boundary_spec_model_from_value(value)
        except Exception:  # noqa: BLE001
            boundary_spec = None
    if boundary_spec is None:
        return component_to_ir(value, family=ComponentFamily.BOUNDARY)

    periodic_axes: list[str] = []
    bloch_axes: list[str] = []
    reflective_axes: list[str] = []
    abc_axes: list[str] = []
    abc_faces: list[str] = []
    pml_axes: list[str] = []
    pml_faces: list[str] = []
    stable_pml_axes: list[str] = []
    stable_pml_faces: list[str] = []
    absorber_axes: list[str] = []
    absorber_faces: list[str] = []
    requires_complex_fields = False
    axes = {}
    for axis_name, axis_boundary in zip("xyz", boundary_spec.as_tuple(), strict=True):
        lowered = boundary_to_ir(axis_boundary)
        if not isinstance(lowered, BoundaryIR):
            return component_to_ir(value, family=ComponentFamily.BOUNDARY)
        axes[axis_name] = lowered
        edge_types = {lowered.minus.component_type, lowered.plus.component_type}
        if edge_types == {"Periodic"}:
            periodic_axes.append(axis_name)
        elif edge_types == {"BlochBoundary"}:
            bloch_axes.append(axis_name)
            requires_complex_fields = True
        elif edge_types == {"ABCBoundary"}:
            abc_axes.append(axis_name)
        elif edge_types <= {"PML", "StablePML", "Absorber"}:
            pml_axes.append(axis_name)
            if lowered.minus.component_type in {"PML", "StablePML", "Absorber"}:
                pml_faces.append(f"{axis_name}.minus")
            if lowered.plus.component_type in {"PML", "StablePML", "Absorber"}:
                pml_faces.append(f"{axis_name}.plus")
            reflective_axes.append(axis_name)
        else:
            if lowered.minus.component_type in {"PML", "StablePML", "Absorber"}:
                pml_faces.append(f"{axis_name}.minus")
            if lowered.plus.component_type in {"PML", "StablePML", "Absorber"}:
                pml_faces.append(f"{axis_name}.plus")
            if (
                lowered.minus.component_type in {"PML", "StablePML", "Absorber"}
                or lowered.plus.component_type in {"PML", "StablePML", "Absorber"}
            ) and axis_name not in pml_axes:
                pml_axes.append(axis_name)
            reflective_axes.append(axis_name)
        if lowered.minus.component_type == "ABCBoundary" or lowered.plus.component_type == "ABCBoundary":
            if axis_name not in abc_axes:
                abc_axes.append(axis_name)
        if lowered.minus.component_type == "ABCBoundary":
            abc_faces.append(f"{axis_name}.minus")
        if lowered.plus.component_type == "ABCBoundary":
            abc_faces.append(f"{axis_name}.plus")
        if lowered.minus.component_type == "StablePML" or lowered.plus.component_type == "StablePML":
            stable_pml_axes.append(axis_name)
        if lowered.minus.component_type == "StablePML":
            stable_pml_faces.append(f"{axis_name}.minus")
        if lowered.plus.component_type == "StablePML":
            stable_pml_faces.append(f"{axis_name}.plus")
        if lowered.minus.component_type == "Absorber" or lowered.plus.component_type == "Absorber":
            absorber_axes.append(axis_name)
        if lowered.minus.component_type == "Absorber":
            absorber_faces.append(f"{axis_name}.minus")
        if lowered.plus.component_type == "Absorber":
            absorber_faces.append(f"{axis_name}.plus")
    symmetry_axes = _symmetry_axes_to_ir(symmetry)
    return BoundarySpecIR(
        x=axes["x"],
        y=axes["y"],
        z=axes["z"],
        requires_complex_fields=requires_complex_fields,
        periodic_axes=tuple(periodic_axes),
        bloch_axes=tuple(bloch_axes),
        reflective_axes=tuple(reflective_axes),
        abc_axes=tuple(abc_axes),
        abc_faces=tuple(abc_faces),
        pml_axes=tuple(pml_axes),
        pml_faces=tuple(pml_faces),
        stable_pml_axes=tuple(stable_pml_axes),
        stable_pml_faces=tuple(stable_pml_faces),
        absorber_axes=tuple(absorber_axes),
        absorber_faces=tuple(absorber_faces),
        active_symmetry_axes=tuple(axis.axis for axis in symmetry_axes),
        symmetry_axes=symmetry_axes,
    )


def _face_halo_to_ir(
    face_halo: object,
) -> FaceHaloIR:
    """Lower a FaceHalo into typed execution IR."""
    from autofdtd.runtime.chunk import FaceHalo as RuntimeFaceHalo

    if isinstance(face_halo, RuntimeFaceHalo):
        return FaceHaloIR(
            axis=face_halo.axis,
            side=face_halo.side,
            depth=face_halo.depth,
            neighbor_chunk_index=face_halo.neighbor_chunk_index,
            exchange_kind=face_halo.exchange_kind.value,
            phase_factor=face_halo.phase_factor,
            electric_signs=face_halo.electric_signs,
            magnetic_signs=face_halo.magnetic_signs,
        )
    raise TypeError(f"expected FaceHalo, got {type(face_halo)!r}")


def _chunk_spec_to_ir(
    chunk_spec: object,
) -> ChunkSpecIR:
    """Lower a ChunkSpec into typed execution IR."""
    from autofdtd.runtime.chunk import ChunkSpec as RuntimeChunkSpec

    if not isinstance(chunk_spec, RuntimeChunkSpec):
        raise TypeError(f"expected ChunkSpec, got {type(chunk_spec)!r}")
    return ChunkSpecIR(
        chunk_index=chunk_spec.chunk_index,
        global_bounds=chunk_spec.global_bounds,
        interior_bounds=chunk_spec.interior_bounds,
        owned_bounds=chunk_spec.owned_bounds,
        local_grid_shape=chunk_spec.local_grid_shape,
        face_halos=tuple(_face_halo_to_ir(h) for h in chunk_spec.face_halos.values()),
        is_reduced=chunk_spec.is_reduced,
        symmetry_multiplicity=chunk_spec.symmetry_multiplicity,
        source_indices=chunk_spec.source_indices,
        monitor_indices=chunk_spec.monitor_indices,
    )


def _exchange_descriptor_to_ir(
    exchange_desc: object,
) -> ExchangeDescriptorIR:
    """Lower an ExchangeDescriptor into typed execution IR."""
    from autofdtd.runtime.chunk import ExchangeDescriptor as RuntimeExchangeDescriptor

    if not isinstance(exchange_desc, RuntimeExchangeDescriptor):
        raise TypeError(f"expected ExchangeDescriptor, got {type(exchange_desc)!r}")
    return ExchangeDescriptorIR(
        source_chunk_index=exchange_desc.source_chunk_index,
        dest_chunk_index=exchange_desc.dest_chunk_index,
        axis=exchange_desc.axis,
        source_side=exchange_desc.source_side,
        dest_side=exchange_desc.dest_side,
        exchange_kind=exchange_desc.exchange_kind.value,
        phase_factor=exchange_desc.phase_factor,
    )


def chunk_layout_to_ir(
    chunk_layout: object,
) -> ChunkLayoutIR:
    """Lower a ChunkLayout into typed execution IR."""
    from autofdtd.runtime.chunk import ChunkLayout as RuntimeChunkLayout

    if not isinstance(chunk_layout, RuntimeChunkLayout):
        raise TypeError(f"expected ChunkLayout, got {type(chunk_layout)!r}")
    return ChunkLayoutIR(
        num_chunks=chunk_layout.num_chunks,
        total_chunks=chunk_layout.total_chunks,
        chunks=tuple(_chunk_spec_to_ir(c) for c in chunk_layout.chunks),
        periodic_axes=chunk_layout.periodic_axes,
        bloch_axes=chunk_layout.bloch_axes,
        exchange_plan=tuple(
            _exchange_descriptor_to_ir(e) for e in chunk_layout.exchange_plan.values()
        ),
        device_assignment=chunk_layout.device_assignment,
        rank_assignment=chunk_layout.rank_assignment,
    )


def _resolved_grid_axis_to_ir(value: ResolvedGridAxis) -> ResolvedGridAxisIR:
    return ResolvedGridAxisIR(
        axis=value.axis,
        boundaries=value.boundaries,
        cell_sizes=value.cell_sizes,
        num_cells=value.num_cells,
    )


def _resolved_grid_to_ir(value: ResolvedGrid) -> ResolvedGridIR:
    return ResolvedGridIR(
        center=value.center,
        size=value.size,
        x=_resolved_grid_axis_to_ir(value.x),
        y=_resolved_grid_axis_to_ir(value.y),
        z=_resolved_grid_axis_to_ir(value.z),
        shape=value.shape,
        total_cells=value.total_cells,
        min_step=value.min_step,
        max_step=value.max_step,
    )


def _grid_axis_to_ir(value: object) -> GridAxisIR:
    grid: object
    if isinstance(value, (UniformGrid, CustomGridBoundaries, CustomGrid, AutoGrid)):
        grid = value
    else:
        grid = grid_model_from_value(value)
        if not isinstance(grid, (UniformGrid, CustomGridBoundaries, CustomGrid, AutoGrid)):
            raise TypeError(f"expected an axis-grid model, got {type(grid)!r}")
    if isinstance(grid, UniformGrid):
        return UniformGridIR(dl=grid.dl)
    if isinstance(grid, CustomGridBoundaries):
        return CustomGridBoundariesIR(coords=grid.coords)
    if isinstance(grid, CustomGrid):
        return CustomGridIR(dl=grid.dl, custom_offset=grid.custom_offset)
    if isinstance(grid, AutoGrid):
        return AutoGridIR(
            min_steps_per_wvl=grid.min_steps_per_wvl,
            min_steps_per_sim_size=grid.min_steps_per_sim_size,
            max_scale=grid.max_scale,
            dl_min=grid.dl_min,
        )
    return component_to_ir(value, family=ComponentFamily.GRID)


def _grid_refinement_to_ir(value: GridRefinement) -> GridRefinementIR:
    return GridRefinementIR(
        refinement_factor=value.refinement_factor,
        dl=value.dl,
        num_cells=value.num_cells,
    )


def _layer_refinement_to_ir(value: LayerRefinementSpec) -> LayerRefinementSpecIR:
    return LayerRefinementSpecIR(
        axis=value.axis,
        center=value.center,
        size=value.size,
        min_steps_along_axis=value.min_steps_along_axis,
        bounds_refinement=(
            _grid_refinement_to_ir(value.bounds_refinement)
            if value.bounds_refinement is not None
            else None
        ),
        bounds_snapping=value.bounds_snapping,
    )


def grid_spec_to_ir(
    value: object,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> GridSpecIR | ComponentIR:
    """Lower a public grid specification into typed execution IR."""
    grid = grid_model_from_value(value)
    if not isinstance(grid, GridSpec):
        return component_to_ir(value, family=ComponentFamily.GRID)
    resolved = grid.make_grid(center=center, size=size)
    return GridSpecIR(
        grid_x=_grid_axis_to_ir(grid.grid_x),
        grid_y=_grid_axis_to_ir(grid.grid_y),
        grid_z=_grid_axis_to_ir(grid.grid_z),
        wavelength=grid.wavelength,
        layer_refinement_specs=tuple(
            _layer_refinement_to_ir(item) for item in grid.layer_refinement_specs
        ),
        resolved=_resolved_grid_to_ir(resolved),
    )


def _subpixel_policy_to_ir(value: object) -> SubpixelPolicyIR:
    if isinstance(value, PolarizedAveraging):
        return PolarizedAveragingIR(
            requires_anisotropic_materialization=value.requires_anisotropic_materialization(),
            courant_ratio=value.courant_ratio(),
        )
    if isinstance(value, Staircasing):
        return StaircasingIR(
            requires_anisotropic_materialization=value.requires_anisotropic_materialization(),
            courant_ratio=value.courant_ratio(),
        )
    raise TypeError(f"unsupported subpixel policy type {type(value)!r}")


def subpixel_to_ir(value: object) -> SubpixelSpecIR:
    """Lower public subpixel settings into typed execution IR."""
    subpixel = subpixel_model_from_value(value)
    return SubpixelSpecIR(
        dielectric=_subpixel_policy_to_ir(subpixel.dielectric),
        metal=_subpixel_policy_to_ir(subpixel.metal),
        pec=_subpixel_policy_to_ir(subpixel.pec),
        pmc=_subpixel_policy_to_ir(subpixel.pmc),
        lossy_metal=_subpixel_policy_to_ir(subpixel.lossy_metal),
        averaging_targets=subpixel.averaging_targets(),
        staircasing_targets=subpixel.staircasing_targets(),
        requires_anisotropic_materialization=subpixel.requires_anisotropic_materialization(),
        courant_ratio=subpixel.courant_ratio(),
    )


def structure_to_ir(
    structure: Structure,
    *,
    priority_mode: StructurePriorityMode,
    source_index: int = 0,
    precedence_rank: int = 0,
) -> StructureIR:
    """Lower a public structure into the execution IR."""
    return StructureIR(
        name=structure.name,
        source_index=source_index,
        priority=structure.priority,
        resolved_priority=structure.resolved_priority(priority_mode),
        precedence_rank=precedence_rank,
        geometry=geometry_to_ir(structure.geometry),
        medium=medium_to_ir(structure.medium),
        background_medium=(
            medium_to_ir(structure.background_medium)
            if structure.background_medium is not None
            else None
        ),
    )


def scene_to_ir(scene: Scene) -> SceneIR:
    """Lower an ordered public scene container into the execution IR."""
    precedence_by_index = {
        entry.source_index: entry for entry in scene.structure_precedence()
    }
    structures = tuple(
        structure_to_ir(
            structure,
            priority_mode=scene.structure_priority_mode,
            source_index=source_index,
            precedence_rank=precedence_by_index[source_index].precedence_rank,
        )
        for source_index, structure in enumerate(scene.structures)
    )
    return SceneIR(
        original_type=scene.type,
        structure_priority_mode=scene.structure_priority_mode.value,
        background_medium=medium_to_ir(scene.medium),
        structures=structures,
    )


def _monitor_freqs_to_tuple(freqs: tuple[float, ...] | None) -> tuple[float, ...]:
    """Normalize monitor freqs field to a tuple."""
    if freqs is None:
        return ()
    return freqs


def monitor_to_ir(value: object) -> MonitorComponentIR:
    """Lower a supported Phase 1 monitor model into typed execution IR."""

    # Import here to avoid circular import
    from autofdtd.monitors import (
        FieldMonitor,
        FieldTimeMonitor,
        AuxFieldTimeMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        ModeMonitor,
        ModeSolverMonitor,
        MediumMonitor,
        PermittivityMonitor,
        FieldProjectionAngleMonitor,
        FieldProjectionCartesianMonitor,
        FieldProjectionKSpaceMonitor,
        DiffractionMonitor,
        DirectivityMonitor,
        GaussianOverlapMonitor,
        AstigmaticGaussianOverlapMonitor,
        SurfaceFieldMonitor,
        SurfaceFieldTimeMonitor,
        monitor_model_from_value,
    )

    try:
        monitor = monitor_model_from_value(value)
    except Exception:
        # Fall back to generic ComponentIR for unknown or incomplete monitor types
        raw = _component_payload(value)
        component_type = str(raw.pop("type", value.__class__.__name__))
        name = raw.get("name")
        return ComponentIR(
            family=ComponentFamily.MONITOR,
            component_type=component_type,
            name=str(name) if isinstance(name, str) else None,
            payload=raw,
        )

    if isinstance(monitor, FieldMonitor):
        return FieldMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            fields=monitor.fields,
            overwrite=monitor.overwrite,
        )
    if isinstance(monitor, FieldTimeMonitor):
        return FieldTimeMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            fields=monitor.fields,
        )
    if isinstance(monitor, AuxFieldTimeMonitor):
        return AuxFieldTimeMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            fields=monitor.fields,
        )
    if isinstance(monitor, FluxMonitor):
        return FluxMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, FluxTimeMonitor):
        return FluxTimeMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
        )
    if isinstance(monitor, ModeMonitor):
        mode_spec_dict = None
        if monitor.mode_spec is not None:
            if hasattr(monitor.mode_spec, "model_dump"):
                mode_spec_dict = monitor.mode_spec.model_dump(mode="json", exclude_none=True)
            else:
                mode_spec_dict = dict(monitor.mode_spec)
        return ModeMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
            mode_spec=mode_spec_dict,
        )
    if isinstance(monitor, ModeSolverMonitor):
        mode_spec_dict = None
        if monitor.mode_spec is not None:
            if hasattr(monitor.mode_spec, "model_dump"):
                mode_spec_dict = monitor.mode_spec.model_dump(mode="json", exclude_none=True)
            else:
                mode_spec_dict = dict(monitor.mode_spec)
        return ModeSolverMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
            mode_spec=mode_spec_dict,
        )
    if isinstance(monitor, MediumMonitor):
        return MediumMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, PermittivityMonitor):
        return PermittivityMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
        )
    if isinstance(monitor, FieldProjectionAngleMonitor):
        return FieldProjectionAngleMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            normal_vector=monitor.normal_vector,
            projection_distance=monitor.projection_distance,
            phi=monitor.phi,
            theta=monitor.theta,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, FieldProjectionCartesianMonitor):
        return FieldProjectionCartesianMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            normal_vector=monitor.normal_vector,
            projection_distance=monitor.projection_distance,
            x=monitor.x,
            y=monitor.y,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, FieldProjectionKSpaceMonitor):
        return FieldProjectionKSpaceMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            normal_vector=monitor.normal_vector,
            projection_distance=monitor.projection_distance,
            num_k=monitor.num_k,
            kx=monitor.kx,
            ky=monitor.ky,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, DiffractionMonitor):
        return DiffractionMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            normal_vector=monitor.normal_vector,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, DirectivityMonitor):
        return DirectivityMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            normal_vector=monitor.normal_vector,
            projection_distance=monitor.projection_distance,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
        )
    if isinstance(monitor, GaussianOverlapMonitor):
        return GaussianOverlapMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
            angle_theta=monitor.angle_theta,
            angle_phi=monitor.angle_phi,
            pol_angle=monitor.pol_angle,
            waist_radius=monitor.waist_radius,
            waist_distance=monitor.waist_distance,
        )
    if isinstance(monitor, AstigmaticGaussianOverlapMonitor):
        return AstigmaticGaussianOverlapMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            direction=monitor.direction,
            num_freqs=monitor.num_freqs,
            freqs=_monitor_freqs_to_tuple(monitor.freqs),
            angle_theta=monitor.angle_theta,
            angle_phi=monitor.angle_phi,
            pol_angle=monitor.pol_angle,
            waist_sizes=monitor.waist_sizes,
            waist_distances=monitor.waist_distances,
        )
    if isinstance(monitor, SurfaceFieldMonitor):
        return SurfaceFieldMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            fields=monitor.fields,
            num_freqs=monitor.num_freqs,
            freqs=monitor.freqs,
        )
    if isinstance(monitor, SurfaceFieldTimeMonitor):
        return SurfaceFieldTimeMonitorIR(
            name=monitor.name,
            center=monitor.center,
            size=monitor.size,
            interval=monitor.interval,
            start=monitor.start,
            fields=monitor.fields,
        )

    # Fall back to generic ComponentIR for unknown monitor types
    raw = _component_payload(monitor)
    component_type = str(raw.pop("type", monitor.__class__.__name__))
    name = raw.get("name")
    return ComponentIR(
        family=ComponentFamily.MONITOR,
        component_type=component_type,
        name=str(name) if isinstance(name, str) else None,
        payload=raw,
    )


def simulation_to_ir(simulation: Simulation) -> SimulationIR:
    """Lower a public simulation shell into a versioned execution-IR snapshot."""
    runtime_controls = None
    if simulation.grid_spec is not None:
        compiled_controls = compile_runtime_controls(simulation)
        runtime_controls = RuntimeControlsIR(
            dt=compiled_controls.dt,
            num_time_steps=compiled_controls.num_time_steps,
            max_step_index=compiled_controls.max_step_index,
            scaled_courant=compiled_controls.scaled_courant,
            cfl_spacings=compiled_controls.cfl_spacings,
            primary_stop_reason=compiled_controls.primary_stop_reason.value,
            convergence_policy=compiled_controls.convergence_policy.value,
            shutoff=compiled_controls.shutoff,
            shutoff_check_interval=compiled_controls.shutoff_check_interval,
            normalize_index=compiled_controls.normalize_index,
            user_step_limit=compiled_controls.user_step_limit,
        )
    return SimulationIR(
        original_type=simulation.type,
        autofdtd_version=simulation.version,
        center=simulation.center,
        size=simulation.size,
        run_time=simulation.run_time,
        courant=simulation.courant,
        normalize_index=simulation.normalize_index,
        symmetry=simulation.symmetry,
        shutoff=simulation.shutoff,
        scene=scene_to_ir(simulation),
        sources=tuple(
            component_to_ir(source, family=ComponentFamily.SOURCE) for source in simulation.sources
        ),
        monitors=tuple(
            monitor_to_ir(monitor)
            for monitor in simulation.monitors
        ),
        boundary_spec=(
            boundary_spec_to_ir(simulation.boundary_spec, symmetry=simulation.symmetry)
            if simulation.boundary_spec is not None
            else None
        ),
        grid_spec=(
            grid_spec_to_ir(simulation.grid_spec, center=simulation.center, size=simulation.size)
            if simulation.grid_spec is not None
            else None
        ),
        subpixel=(
            subpixel_to_ir(simulation.subpixel)
            if simulation.subpixel is not None
            else None
        ),
        runtime_controls=runtime_controls,
    )


def execution_package_json_schema() -> dict[str, Any]:
    """Return the JSON schema for the execution package transport format."""
    return ExecutionPackageIR.model_json_schema()


def write_execution_package_schema(path: str | Path, *, indent: int = 2) -> Path:
    """Write the execution-package JSON schema to disk."""
    output_path = Path(path)
    schema_text = json.dumps(execution_package_json_schema(), indent=indent, sort_keys=True)
    output_path.write_text(schema_text + "\n", encoding="utf-8")
    return output_path


GeometryGroupIR.model_rebuild()
TransformedIR.model_rebuild()
ClipOperationIR.model_rebuild()
GridSpecIR.model_rebuild()
SubpixelSpecIR.model_rebuild()
StructureIR.model_rebuild()
SceneIR.model_rebuild()
SimulationIR.model_rebuild()
ExecutionManifestIR.model_rebuild()
ExecutionPackageIR.model_rebuild()
