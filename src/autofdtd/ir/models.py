"""Versioned execution IR models and packaging helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from autofdtd.core.containers import Scene, Simulation, Structure, StructurePriorityMode
from autofdtd.core.models import TaggedModel, json_ready
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


MediumComponentIR = (
    MediumIR
    | PECMediumIR
    | PMCMediumIR
    | PoleResidueIR
    | SellmeierIR
    | LorentzIR
    | DrudeIR
    | DebyeIR
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


class SimulationIR(IRModel):
    """Top-level execution IR for a Phase 1 simulation."""

    type: Literal["SimulationIR"] = "SimulationIR"
    original_type: str = "Simulation"
    autofdtd_version: str = __version__
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    run_time: float
    courant: float
    symmetry: tuple[int, int, int]
    shutoff: float | None = None
    scene: SceneIR
    sources: tuple[ComponentIR, ...] = ()
    monitors: tuple[ComponentIR, ...] = ()
    boundary_spec: ComponentIR | None = None
    grid_spec: GridSpecIR | ComponentIR | None = None
    subpixel: SubpixelSpecIR | ComponentIR | None = None


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


def component_to_ir(value: object, *, family: str) -> ComponentIR:
    """Normalize a tagged public component into a transport-safe IR envelope."""
    raw = _component_payload(value)
    component_type = str(raw.pop("type", value.__class__.__name__))
    name = raw.get("name")
    return ComponentIR(
        family=family,
        component_type=component_type,
        name=str(name) if isinstance(name, str) else None,
        payload=raw,
    )


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
        Medium | PECMedium | PMCMedium | PoleResidue | Sellmeier | Lorentz | Drude | Debye | None
    ) = None
    if isinstance(
        value,
        (Medium, PECMedium, PMCMedium, PoleResidue, Sellmeier, Lorentz, Drude, Debye),
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
    return component_to_ir(value, family=ComponentFamily.MEDIUM)


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


def simulation_to_ir(simulation: Simulation) -> SimulationIR:
    """Lower a public simulation shell into a versioned execution-IR snapshot."""
    return SimulationIR(
        original_type=simulation.type,
        autofdtd_version=simulation.version,
        center=simulation.center,
        size=simulation.size,
        run_time=simulation.run_time,
        courant=simulation.courant,
        symmetry=simulation.symmetry,
        shutoff=simulation.shutoff,
        scene=scene_to_ir(simulation),
        sources=tuple(
            component_to_ir(source, family=ComponentFamily.SOURCE) for source in simulation.sources
        ),
        monitors=tuple(
            component_to_ir(monitor, family=ComponentFamily.MONITOR)
            for monitor in simulation.monitors
        ),
        boundary_spec=(
            component_to_ir(simulation.boundary_spec, family=ComponentFamily.BOUNDARY)
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
