"""Material-family models for Phase 1 constitutive work."""

from __future__ import annotations

from collections.abc import Mapping

from autofdtd.materials.advanced import (
    CustomAnisotropicMedium,
    CustomMedium,
    GenericCustomMedium,
    LossyMetalMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
)
from autofdtd.materials.anisotropic import (
    AnisotropicMedium,
    FullyAnisotropicMedium,
    Medium2D,
)
from autofdtd.materials.dispersive import (
    Debye,
    DispersiveMedium,
    Drude,
    Lorentz,
    PoleResidue,
    Sellmeier,
    complex_from_pair,
    complex_to_pair,
)
from autofdtd.materials.isotropic import IsotropicMedium, Medium, PECMedium, PMCMedium

Phase1Medium = (
    Medium
    | PECMedium
    | PMCMedium
    | LossyMetalMedium
    | PoleResidue
    | Sellmeier
    | Lorentz
    | Drude
    | Debye
    | AnisotropicMedium
)
DeferredMaterialModel = (
    LossyMetalMedium
    | FullyAnisotropicMedium
    | Medium2D
    | PerturbationMedium
    | PerturbationPoleResidue
)
RejectedMaterialModel = CustomMedium | CustomAnisotropicMedium | GenericCustomMedium
MaterialModel = Phase1Medium | DeferredMaterialModel | RejectedMaterialModel


def medium_model_from_value(value: object) -> MaterialModel:
    """Coerce a mapping payload into a known material model surface."""

    if isinstance(
        value,
        (
            Medium,
            PECMedium,
            PMCMedium,
            LossyMetalMedium,
            PoleResidue,
            Sellmeier,
            Lorentz,
            Drude,
            Debye,
            AnisotropicMedium,
            FullyAnisotropicMedium,
            Medium2D,
            PerturbationMedium,
            PerturbationPoleResidue,
            CustomMedium,
            CustomAnisotropicMedium,
            GenericCustomMedium,
        ),
    ):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or Phase 1 medium model, got {type(value)!r}")

    medium_type = str(value.get("type", ""))
    if medium_type == "Medium":
        return Medium.model_validate(value)
    if medium_type == "PECMedium":
        return PECMedium.model_validate(value)
    if medium_type == "PMCMedium":
        return PMCMedium.model_validate(value)
    if medium_type == "LossyMetalMedium":
        return LossyMetalMedium.model_validate(value)
    if medium_type == "PoleResidue":
        return PoleResidue.model_validate(value)
    if medium_type == "Sellmeier":
        return Sellmeier.model_validate(value)
    if medium_type == "Lorentz":
        return Lorentz.model_validate(value)
    if medium_type == "Drude":
        return Drude.model_validate(value)
    if medium_type == "Debye":
        return Debye.model_validate(value)
    if medium_type == "AnisotropicMedium":
        return AnisotropicMedium.model_validate(value)
    if medium_type == "FullyAnisotropicMedium":
        return FullyAnisotropicMedium.model_validate(value)
    if medium_type == "Medium2D":
        return Medium2D.model_validate(value)
    if medium_type == "PerturbationMedium":
        return PerturbationMedium.model_validate(value)
    if medium_type == "PerturbationPoleResidue":
        return PerturbationPoleResidue.model_validate(value)
    if medium_type == "CustomMedium":
        return CustomMedium.model_validate(value)
    if medium_type == "CustomAnisotropicMedium":
        return CustomAnisotropicMedium.model_validate(value)
    if medium_type.startswith("Custom"):
        return GenericCustomMedium.model_validate(value)
    raise TypeError(f"unsupported Phase 1 medium type {medium_type!r}")


__all__ = [
    "AnisotropicMedium",
    "CustomAnisotropicMedium",
    "CustomMedium",
    "Debye",
    "DeferredMaterialModel",
    "DispersiveMedium",
    "Drude",
    "FullyAnisotropicMedium",
    "GenericCustomMedium",
    "IsotropicMedium",
    "Lorentz",
    "LossyMetalMedium",
    "Medium",
    "MaterialModel",
    "Medium2D",
    "PECMedium",
    "PMCMedium",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "Phase1Medium",
    "PoleResidue",
    "RejectedMaterialModel",
    "Sellmeier",
    "complex_from_pair",
    "complex_to_pair",
    "medium_model_from_value",
]
