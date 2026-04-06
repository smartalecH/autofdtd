"""Phase 1 isotropic material models and helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import field_validator

from autofdtd.core.models import TaggedModel


def _positive_finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite value")
    return normalized


def _normalize_name(name: str | None) -> str | None:
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        raise ValueError("names must not be empty or whitespace-only")
    return stripped


def _non_negative_finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite value")
    return normalized


class Medium(TaggedModel):
    """Foundational homogeneous isotropic medium."""

    type: Literal["Medium"] = "Medium"
    name: str | None = None
    permittivity: float = 1.0
    conductivity: float = 0.0
    permeability: float = 1.0
    magnetic_conductivity: float = 0.0

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("permittivity")
    @classmethod
    def _validate_permittivity(cls, value: float) -> float:
        return _positive_finite(value, field_name="permittivity")

    @field_validator("permeability")
    @classmethod
    def _validate_permeability(cls, value: float) -> float:
        return _positive_finite(value, field_name="permeability")

    @field_validator("conductivity")
    @classmethod
    def _validate_conductivity(cls, value: float) -> float:
        return _non_negative_finite(value, field_name="conductivity")

    @field_validator("magnetic_conductivity")
    @classmethod
    def _validate_magnetic_conductivity(cls, value: float) -> float:
        return _non_negative_finite(value, field_name="magnetic_conductivity")


class PECMedium(TaggedModel):
    """Perfect electric conductor medium marker."""

    type: Literal["PECMedium"] = "PECMedium"
    name: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)


class PMCMedium(TaggedModel):
    """Perfect magnetic conductor medium marker."""

    type: Literal["PMCMedium"] = "PMCMedium"
    name: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)


IsotropicMedium = Medium | PECMedium | PMCMedium


def medium_model_from_value(value: object) -> IsotropicMedium:
    """Coerce a mapping payload into a supported Phase 1 isotropic material model."""

    if isinstance(value, (Medium, PECMedium, PMCMedium)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or isotropic medium model, got {type(value)!r}")

    medium_type = str(value.get("type", ""))
    if medium_type == "Medium":
        return Medium.model_validate(value)
    if medium_type == "PECMedium":
        return PECMedium.model_validate(value)
    if medium_type == "PMCMedium":
        return PMCMedium.model_validate(value)
    raise TypeError(f"unsupported isotropic medium type {medium_type!r}")
