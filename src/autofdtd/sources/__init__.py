"""Source-family models for Phase 1 time profiles and injections."""

from autofdtd.sources.beam import (
    AstigmaticGaussianBeam,
    GaussianBeam,
)
from autofdtd.sources.current import (
    CustomCurrentSource,
    CustomFieldSource,
    CurrentSourceModel,
    PointDipole,
    UniformCurrentSource,
    current_source_model_from_value,
)
from autofdtd.sources.plane import (
    AbstractAngularSpec,
    AngularSpec,
    FixedAngleSpec,
    FixedInPlaneKSpec,
    PlaneWave,
    angular_spec_model_from_value,
)
from autofdtd.sources.time import (
    BroadbandPulse,
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    Pulse,
    SourceTime,
    SourceTimeModel,
    source_time_model_from_value,
)
from autofdtd.sources.tfsf import TFSF

# ModeSource is imported lazily to avoid circular import with modes package
# Use: from autofdtd.sources.mode import ModeSource
__all__ = [
    "AbstractAngularSpec",
    "AngularSpec",
    "AstigmaticGaussianBeam",
    "BroadbandPulse",
    "ContinuousWave",
    "CustomCurrentSource",
    "CustomFieldSource",
    "CurrentSourceModel",
    "CustomSourceTime",
    "FixedAngleSpec",
    "FixedInPlaneKSpec",
    "GaussianBeam",
    "GaussianPulse",
    "ModeSource",
    "PlaneWave",
    "PointDipole",
    "Pulse",
    "SourceTime",
    "SourceTimeModel",
    "TFSF",
    "UniformCurrentSource",
    "angular_spec_model_from_value",
    "current_source_model_from_value",
    "source_time_model_from_value",
]
