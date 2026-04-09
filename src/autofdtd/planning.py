"""Planning metadata anchored to the Phase 1 feature checklist."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class FeatureStatus(StrEnum):
    """Implementation state for a Phase 1 feature."""

    IMPLEMENT = "Implement"
    DEFER = "Defer"
    REJECT_CLEARLY = "Reject Clearly"


@dataclass(frozen=True, slots=True)
class FeatureEntry:
    """One checklist entry mapped into package planning metadata."""

    family: str
    feature: str
    status: FeatureStatus
    notes: str


@dataclass(frozen=True, slots=True)
class NamespacePlan:
    """Subsystem namespace reserved for Phase 1 implementation work."""

    module: str
    role: str
    planned_contents: tuple[str, ...]


_FEATURE_MATRIX: Final[tuple[FeatureEntry, ...]] = (
    # Core containers
    FeatureEntry("core", "Simulation", FeatureStatus.IMPLEMENT, "Core API and IR entrypoint."),
    FeatureEntry("core", "Scene", FeatureStatus.IMPLEMENT, "Ordered structure container."),
    FeatureEntry("core", "Structure", FeatureStatus.IMPLEMENT, "Material-carrying geometry shell."),
    FeatureEntry(
        "core",
        "normalize_index",
        FeatureStatus.IMPLEMENT,
        "Validation-backed source normalization selector.",
    ),
    FeatureEntry(
        "core",
        "low_freq_smoothing",
        FeatureStatus.DEFER,
        "Mapped but not Phase 1 core.",
    ),
    FeatureEntry("core", "relax_courant", FeatureStatus.DEFER, "Mapped but not Phase 1 core."),
    FeatureEntry(
        "core",
        "lumped_elements",
        FeatureStatus.REJECT_CLEARLY,
        "Out of Phase 1 scope.",
    ),
    # Geometry
    FeatureEntry("geometry", "Box", FeatureStatus.IMPLEMENT, "Core primitive."),
    FeatureEntry("geometry", "Sphere", FeatureStatus.IMPLEMENT, "Core primitive."),
    FeatureEntry("geometry", "Cylinder", FeatureStatus.IMPLEMENT, "Core primitive."),
    FeatureEntry(
        "geometry",
        "PolySlab",
        FeatureStatus.IMPLEMENT,
        "High-priority photonics geometry.",
    ),
    FeatureEntry("geometry", "GeometryGroup", FeatureStatus.IMPLEMENT, "Scene composition."),
    FeatureEntry(
        "geometry",
        "Transformed",
        FeatureStatus.IMPLEMENT,
        "Minimum transform wrapper.",
    ),
    FeatureEntry(
        "geometry",
        "ClipOperation",
        FeatureStatus.DEFER,
        "Map explicitly; only the narrow Phase 1 boolean subset should pass validation.",
    ),
    FeatureEntry(
        "geometry",
        "GeometryArray",
        FeatureStatus.IMPLEMENT,
        "Repeated placement semantics.",
    ),
    FeatureEntry("geometry", "TriangleMesh", FeatureStatus.DEFER, "Advanced geometry bucket."),
    FeatureEntry(
        "geometry",
        "EulerBend",
        FeatureStatus.IMPLEMENT,
        "Euler/clothoid bend for smoothly-curved waveguide integration.",
    ),
    # Grid and subpixel
    FeatureEntry("grid", "UniformGrid", FeatureStatus.IMPLEMENT, "Core grid path."),
    FeatureEntry("grid", "CustomGrid", FeatureStatus.IMPLEMENT, "Explicit grid control."),
    FeatureEntry(
        "grid",
        "CustomGridBoundaries",
        FeatureStatus.IMPLEMENT,
        "Validated explicit boundaries.",
    ),
    FeatureEntry("grid", "GridSpec", FeatureStatus.IMPLEMENT, "Main grid container."),
    FeatureEntry("grid", "AutoGrid", FeatureStatus.IMPLEMENT, "Simplified but real behavior."),
    FeatureEntry("grid", "QuasiUniformGrid", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("grid", "GridRefinement", FeatureStatus.IMPLEMENT, "Refinement metadata."),
    FeatureEntry(
        "grid",
        "LayerRefinementSpec",
        FeatureStatus.IMPLEMENT,
        "Axis-aligned layer refinement metadata.",
    ),
    FeatureEntry("grid", "SubpixelSpec", FeatureStatus.IMPLEMENT, "Policy surface."),
    FeatureEntry("grid", "Staircasing", FeatureStatus.IMPLEMENT, "Baseline interface policy."),
    FeatureEntry(
        "grid",
        "PolarizedAveraging",
        FeatureStatus.IMPLEMENT,
        "First averaging policy for dielectric interfaces.",
    ),
    FeatureEntry("grid", "ContourPathAveraging", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("grid", "VolumetricAveraging", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("grid", "PECConformal", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("grid", "SurfaceImpedance", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("grid", "HeuristicPECStaircasing", FeatureStatus.DEFER, "Mapped for later."),
    # Materials
    FeatureEntry("materials", "Medium", FeatureStatus.IMPLEMENT, "Core isotropic material."),
    FeatureEntry("materials", "PECMedium", FeatureStatus.IMPLEMENT, "Perfect electric conductor."),
    FeatureEntry("materials", "PMCMedium", FeatureStatus.IMPLEMENT, "Perfect magnetic conductor."),
    FeatureEntry("materials", "PoleResidue", FeatureStatus.IMPLEMENT, "Primary dispersive bucket."),
    FeatureEntry("materials", "Sellmeier", FeatureStatus.IMPLEMENT, "Optical dispersion family."),
    FeatureEntry("materials", "Lorentz", FeatureStatus.IMPLEMENT, "Dispersive family."),
    FeatureEntry("materials", "Drude", FeatureStatus.IMPLEMENT, "Dispersive family."),
    FeatureEntry("materials", "Debye", FeatureStatus.IMPLEMENT, "Dispersive family."),
    FeatureEntry(
        "materials",
        "AnisotropicMedium",
        FeatureStatus.IMPLEMENT,
        "Needed for subpixel work.",
    ),
    FeatureEntry("materials", "LossyMetalMedium", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry("materials", "PerturbationMedium", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry(
        "materials",
        "PerturbationPoleResidue",
        FeatureStatus.DEFER,
        "Mapped for later.",
    ),
    FeatureEntry("materials", "Medium2D", FeatureStatus.DEFER, "Mapped for later."),
    FeatureEntry(
        "materials",
        "FullyAnisotropicMedium",
        FeatureStatus.DEFER,
        "Mapped for later.",
    ),
    FeatureEntry(
        "materials",
        "CustomMedium",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomAnisotropicMedium",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom tensor media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomPoleResidue",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom dispersive media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomSellmeier",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom dispersive media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomLorentz",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom dispersive media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomDrude",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom dispersive media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomDebye",
        FeatureStatus.REJECT_CLEARLY,
        "Sampled custom dispersive media are out of Phase 1 scope.",
    ),
    FeatureEntry(
        "materials",
        "CustomMedia",
        FeatureStatus.REJECT_CLEARLY,
        "Out of Phase 1 scope.",
    ),
    # Boundaries
    FeatureEntry("boundaries", "Boundary", FeatureStatus.IMPLEMENT, "Per-axis boundary pair."),
    FeatureEntry("boundaries", "BoundarySpec", FeatureStatus.IMPLEMENT, "Core boundary container."),
    FeatureEntry("boundaries", "Periodic", FeatureStatus.IMPLEMENT, "Core boundary."),
    FeatureEntry("boundaries", "PECBoundary", FeatureStatus.IMPLEMENT, "Conductor boundary."),
    FeatureEntry("boundaries", "PMCBoundary", FeatureStatus.IMPLEMENT, "Conductor boundary."),
    FeatureEntry(
        "boundaries",
        "BlochBoundary",
        FeatureStatus.IMPLEMENT,
        "Phase-aware periodic boundary.",
    ),
    FeatureEntry("boundaries", "PML", FeatureStatus.IMPLEMENT, "Must-have absorber."),
    FeatureEntry(
        "boundaries",
        "StablePML",
        FeatureStatus.IMPLEMENT,
        "Supported subset acceptable.",
    ),
    FeatureEntry("boundaries", "Absorber", FeatureStatus.IMPLEMENT, "Practical fallback."),
    FeatureEntry(
        "boundaries",
        "ABCBoundary",
        FeatureStatus.IMPLEMENT,
        "Supported as an explicit-parameter first-order subset.",
    ),
    FeatureEntry(
        "boundaries",
        "ModeABCBoundary",
        FeatureStatus.DEFER,
        "Parsed surface remains deferred until mode-solver work lands.",
    ),
    # Sources
    FeatureEntry("sources", "GaussianPulse", FeatureStatus.IMPLEMENT, "Core source-time profile."),
    FeatureEntry("sources", "ContinuousWave", FeatureStatus.IMPLEMENT, "Core source-time profile."),
    FeatureEntry(
        "sources",
        "BroadbandPulse",
        FeatureStatus.IMPLEMENT,
        "Supported subset acceptable.",
    ),
    FeatureEntry(
        "sources",
        "UniformCurrentSource",
        FeatureStatus.IMPLEMENT,
        "Core current source.",
    ),
    FeatureEntry("sources", "PointDipole", FeatureStatus.IMPLEMENT, "Core localized source."),
    FeatureEntry(
        "sources",
        "CustomCurrentSource",
        FeatureStatus.IMPLEMENT,
        "Custom current source with explicit field data.",
    ),
    FeatureEntry(
        "sources",
        "CustomFieldSource",
        FeatureStatus.IMPLEMENT,
        "Planar field source using equivalence principle.",
    ),
    FeatureEntry(
        "sources",
        "ModeSource",
        FeatureStatus.IMPLEMENT,
        "High-priority photonics source.",
    ),
    FeatureEntry("sources", "PlaneWave", FeatureStatus.IMPLEMENT, "Core optical source."),
    FeatureEntry(
        "sources",
        "FixedInPlaneKSpec",
        FeatureStatus.IMPLEMENT,
        "Angular spec for frequency-dependent plane wave angle.",
    ),
    FeatureEntry(
        "sources",
        "FixedAngleSpec",
        FeatureStatus.IMPLEMENT,
        "Angular spec for frequency-independent plane wave angle.",
    ),
    FeatureEntry("sources", "GaussianBeam", FeatureStatus.IMPLEMENT, "Core optical source."),
    FeatureEntry(
        "sources",
        "AstigmaticGaussianBeam",
        FeatureStatus.IMPLEMENT,
        "Astigmatic variant with separate x/y waist parameters.",
    ),
    FeatureEntry(
        "sources",
        "TFSF",
        FeatureStatus.IMPLEMENT,
        "Important total-field scattered-field source.",
    ),
    FeatureEntry(
        "sources",
        "MicrowaveTerminalSource",
        FeatureStatus.REJECT_CLEARLY,
        "Out of Phase 1 scope.",
    ),
    # Monitors and data
    FeatureEntry(
        "monitors",
        "SimulationData",
        FeatureStatus.IMPLEMENT,
        "Named result access model.",
    ),
    FeatureEntry("monitors", "FieldMonitor", FeatureStatus.IMPLEMENT, "Core field monitor."),
    FeatureEntry(
        "monitors",
        "FieldTimeMonitor",
        FeatureStatus.IMPLEMENT,
        "Core time-domain monitor.",
    ),
    FeatureEntry("monitors", "FluxMonitor", FeatureStatus.IMPLEMENT, "Core flux monitor."),
    FeatureEntry(
        "monitors",
        "FluxTimeMonitor",
        FeatureStatus.IMPLEMENT,
        "Time-domain flux monitor.",
    ),
    FeatureEntry("monitors", "ModeMonitor", FeatureStatus.IMPLEMENT, "Mode overlap workflow."),
    FeatureEntry(
        "monitors",
        "ModeSolverMonitor",
        FeatureStatus.IMPLEMENT,
        "Mode workflow dependency.",
    ),
    FeatureEntry("monitors", "PermittivityMonitor", FeatureStatus.IMPLEMENT, "Validation monitor."),
    FeatureEntry("monitors", "ModeSpec", FeatureStatus.IMPLEMENT, "Mode specification for mode monitors."),
    FeatureEntry("monitors", "MediumMonitor", FeatureStatus.IMPLEMENT, "Medium property sampling."),
    FeatureEntry(
        "monitors",
        "GaussianOverlapMonitor",
        FeatureStatus.IMPLEMENT,
        "Gaussian beam overlap projection monitor.",
    ),
    FeatureEntry(
        "monitors",
        "AstigmaticGaussianOverlapMonitor",
        FeatureStatus.IMPLEMENT,
        "Astigmatic Gaussian beam overlap with separate x/y waist parameters.",
    ),
    FeatureEntry(
        "monitors",
        "DirectivityMonitor",
        FeatureStatus.IMPLEMENT,
        "Projection-backed subset acceptable.",
    ),
    FeatureEntry(
        "monitors",
        "FieldProjectionAngleMonitor",
        FeatureStatus.IMPLEMENT,
        "Far-field angle-space projection monitor.",
    ),
    FeatureEntry(
        "monitors",
        "FieldProjectionCartesianMonitor",
        FeatureStatus.IMPLEMENT,
        "Far-field Cartesian projection monitor.",
    ),
    FeatureEntry(
        "monitors",
        "FieldProjectionKSpaceMonitor",
        FeatureStatus.IMPLEMENT,
        "K-space field projection monitor.",
    ),
    FeatureEntry(
        "monitors",
        "DiffractionMonitor",
        FeatureStatus.IMPLEMENT,
        "Diffraction order recording monitor.",
    ),
    # Postprocessing
    FeatureEntry(
        "postprocessing",
        "DFT accumulation",
        FeatureStatus.IMPLEMENT,
        "Foundation for outputs.",
    ),
    FeatureEntry(
        "postprocessing",
        "Near-to-far transform",
        FeatureStatus.IMPLEMENT,
        "Built atop DFT monitors.",
    ),
    FeatureEntry(
        "postprocessing",
        "Diffraction postprocessing",
        FeatureStatus.DEFER,
        "Depends on monitor support.",
    ),
)

_NAMESPACE_PLAN: Final[tuple[NamespacePlan, ...]] = (
    NamespacePlan(
        "autofdtd.api",
        "Public API surface",
        ("Simulation", "Scene", "Structure"),
    ),
    NamespacePlan(
        "autofdtd.core",
        "Shared tagged models",
        ("base models", "serialization", "validators"),
    ),
    NamespacePlan(
        "autofdtd.geometry",
        "Geometry families",
        ("primitives", "groups", "transforms"),
    ),
    NamespacePlan(
        "autofdtd.materials",
        "Material families",
        ("isotropic", "dispersive", "anisotropic"),
    ),
    NamespacePlan(
        "autofdtd.boundaries",
        "Boundary families",
        ("periodic", "bloch", "absorbing"),
    ),
    NamespacePlan(
        "autofdtd.sources",
        "Source families",
        ("time profiles", "current", "field sources"),
    ),
    NamespacePlan(
        "autofdtd.monitors",
        "Monitor families",
        ("field", "flux", "mode", "projection"),
    ),
    NamespacePlan(
        "autofdtd.grid",
        "Grid and subpixel controls",
        ("grid specs", "refinement", "subpixel policies"),
    ),
    NamespacePlan(
        "autofdtd.modes",
        "Mode solver integration",
        ("mode specs", "cross-section sampling"),
    ),
    NamespacePlan(
        "autofdtd.ir",
        "Execution IR",
        ("tagged objects", "versioned serialization"),
    ),
    NamespacePlan(
        "autofdtd.compiler",
        "Scene compilation",
        ("normalization", "lowering", "planning"),
    ),
    NamespacePlan(
        "autofdtd.runtime",
        "Chunk runtime",
        ("chunk topology", "scheduling", "execution context"),
    ),
    NamespacePlan(
        "autofdtd.kernels",
        "GPU kernels",
        ("Warp conventions", "update stages", "halo operations"),
    ),
    NamespacePlan(
        "autofdtd.diagnostics",
        "Instrumentation",
        ("logging", "metrics", "validation reports"),
    ),
    NamespacePlan(
        "autofdtd.examples",
        "Validation examples",
        ("smoke cases", "benchmarks", "docs snippets"),
    ),
)


def feature_matrix() -> tuple[FeatureEntry, ...]:
    """Return the Phase 1 feature matrix baked into the package scaffold."""

    return _FEATURE_MATRIX


def feature_entry(feature: str) -> FeatureEntry | None:
    """Look up one feature by its tagged public name."""

    return next((entry for entry in _FEATURE_MATRIX if entry.feature == feature), None)


def feature_status(feature: str) -> FeatureStatus | None:
    """Return the planned status for one feature, if it is tracked."""

    entry = feature_entry(feature)
    return entry.status if entry is not None else None


def planned_namespace_map() -> tuple[NamespacePlan, ...]:
    """Return the reserved namespace layout for Phase 1 implementation work."""

    return _NAMESPACE_PLAN
