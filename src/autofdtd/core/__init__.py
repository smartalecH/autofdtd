"""Shared tagged-model foundations and serialization helpers."""

from autofdtd.core.containers import (
    Scene,
    Simulation,
    Structure,
    StructurePrecedence,
    StructurePriorityMode,
    TaggedContainerModel,
)
from autofdtd.core.models import AutoFDTDModel, TaggedModel, json_ready
from autofdtd.core.validation import AutoFDTDValidationWarning, UnsupportedFeatureError

__all__ = [
    "AutoFDTDValidationWarning",
    "AutoFDTDModel",
    "Scene",
    "Simulation",
    "Structure",
    "StructurePrecedence",
    "StructurePriorityMode",
    "TaggedModel",
    "TaggedContainerModel",
    "UnsupportedFeatureError",
    "json_ready",
]
