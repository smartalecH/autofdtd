"""Material-family models for Phase 1 constitutive work."""

from __future__ import annotations

from collections.abc import Mapping

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

Phase1Medium = Medium | PECMedium | PMCMedium | PoleResidue | Sellmeier | Lorentz | Drude | Debye


def medium_model_from_value(value: object) -> Phase1Medium:
    """Coerce a mapping payload into a supported Phase 1 material model."""

    if isinstance(
        value,
        (Medium, PECMedium, PMCMedium, PoleResidue, Sellmeier, Lorentz, Drude, Debye),
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
    raise TypeError(f"unsupported Phase 1 medium type {medium_type!r}")


__all__ = [
    "Debye",
    "DispersiveMedium",
    "Drude",
    "IsotropicMedium",
    "Lorentz",
    "Medium",
    "PECMedium",
    "PMCMedium",
    "Phase1Medium",
    "PoleResidue",
    "Sellmeier",
    "complex_from_pair",
    "complex_to_pair",
    "medium_model_from_value",
]
