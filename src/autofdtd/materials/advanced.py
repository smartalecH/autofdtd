"""Explicit Phase 1 policy surfaces for deferred and rejected advanced media."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.materials.dispersive import PoleResidue
from autofdtd.materials.isotropic import Medium, _normalize_name, _positive_finite


def _optional_positive_finite(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, field_name=field_name)


class LossyMetalMedium(Medium):
    """Deferred Phase 1 surface for surface-impedance lossy-metal modeling."""

    type: Literal["LossyMetalMedium"] = "LossyMetalMedium"
    permittivity: Literal[1.0] = 1.0
    conductivity: float
    frequency_range: tuple[float, float]
    allow_gain: Literal[False] = False
    thickness: float | None = None
    roughness: float | None = None

    @field_validator("frequency_range", mode="before")
    @classmethod
    def _validate_frequency_range(cls, value: object) -> tuple[float, float]:
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise TypeError("frequency_range must be a 2-tuple of positive frequencies")
        lower = _positive_finite(float(value[0]), field_name="frequency_range[0]")
        upper = _positive_finite(float(value[1]), field_name="frequency_range[1]")
        if upper <= lower:
            raise ValueError("frequency_range must be strictly increasing")
        return (lower, upper)

    @field_validator("thickness", "roughness")
    @classmethod
    def _validate_optional_positive(cls, value: float | None, info) -> float | None:
        return _optional_positive_finite(value, field_name=info.field_name)

    @property
    def phase1_policy(self) -> str:
        return (
            "LossyMetalMedium is parsed and serialized, but direct Phase 1 runtime support is "
            "deferred pending a dedicated surface-impedance or volumetric metal path."
        )

    def to_medium_approximation(self) -> Medium:
        """Return the homogeneous volumetric fallback medium used by Phase 1 guidance."""

        return Medium(
            name=self.name,
            permittivity=1.0,
            conductivity=self.conductivity,
            permeability=self.permeability,
            magnetic_conductivity=self.magnetic_conductivity,
        )


class PerturbationMedium(Medium):
    """Deferred Phase 1 surface for parameter-perturbed nondispersive media."""

    type: Literal["PerturbationMedium"] = "PerturbationMedium"
    permittivity_perturbation: dict[str, Any] | None = None
    conductivity_perturbation: dict[str, Any] | None = None
    perturbation_spec: dict[str, Any] | None = None
    frequency_range: tuple[float, float] | None = None
    subpixel: bool = True

    @field_validator("frequency_range", mode="before")
    @classmethod
    def _validate_frequency_range(
        cls, value: object | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise TypeError("frequency_range must be a 2-tuple of positive frequencies")
        lower = _positive_finite(float(value[0]), field_name="frequency_range[0]")
        upper = _positive_finite(float(value[1]), field_name="frequency_range[1]")
        if upper <= lower:
            raise ValueError("frequency_range must be strictly increasing")
        return (lower, upper)

    @property
    def phase1_policy(self) -> str:
        return (
            "PerturbationMedium is parsed and serialized, but Phase 1 does not execute "
            "heat-, charge-, or bias-dependent material perturbations."
        )

    def base_medium(self) -> Medium:
        """Drop perturbation metadata and keep the unperturbed homogeneous medium."""

        return Medium(
            name=self.name,
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            permeability=self.permeability,
            magnetic_conductivity=self.magnetic_conductivity,
        )


class PerturbationPoleResidue(PoleResidue):
    """Deferred Phase 1 surface for perturbed pole-residue media."""

    type: Literal["PerturbationPoleResidue"] = "PerturbationPoleResidue"
    eps_inf_perturbation: dict[str, Any] | None = None
    poles_perturbation: tuple[tuple[dict[str, Any] | None, dict[str, Any] | None], ...] | None = None
    perturbation_spec: dict[str, Any] | None = None
    subpixel: bool = True

    @property
    def phase1_policy(self) -> str:
        return (
            "PerturbationPoleResidue is parsed and serialized, but Phase 1 does not execute "
            "runtime perturbations on dispersive pole-residue coefficients."
        )

    def base_medium(self) -> PoleResidue:
        """Drop perturbation metadata and keep the unperturbed dispersive medium."""

        return PoleResidue(name=self.name, eps_inf=self.eps_inf, poles=self.poles)


class CustomMedium(TaggedModel):
    """Rejected Phase 1 surface for sampled spatially varying isotropic media."""

    type: Literal["CustomMedium"] = "CustomMedium"
    name: str | None = None
    permittivity: Any = None
    conductivity: Any = None
    frequency_range: tuple[float, float] | None = None
    derived_from: Any = None
    interp_method: Literal["nearest", "linear"] = "linear"
    subpixel: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("frequency_range", mode="before")
    @classmethod
    def _validate_frequency_range(
        cls, value: object | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise TypeError("frequency_range must be a 2-tuple of positive frequencies")
        lower = _positive_finite(float(value[0]), field_name="frequency_range[0]")
        upper = _positive_finite(float(value[1]), field_name="frequency_range[1]")
        if upper <= lower:
            raise ValueError("frequency_range must be strictly increasing")
        return (lower, upper)

    @property
    def phase1_policy(self) -> str:
        return (
            "CustomMedium is parsed for compatibility inspection only. Phase 1 rejects sampled "
            "or spatially varying custom media because runtime interpolation and storage are out of scope."
        )


class CustomAnisotropicMedium(TaggedModel):
    """Rejected Phase 1 surface for sampled spatially varying diagonal anisotropy."""

    type: Literal["CustomAnisotropicMedium"] = "CustomAnisotropicMedium"
    name: str | None = None
    xx: Any
    yy: Any
    zz: Any
    frequency_range: tuple[float, float] | None = None
    derived_from: Any = None
    interp_method: Literal["nearest", "linear"] | None = "linear"
    subpixel: bool | None = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("frequency_range", mode="before")
    @classmethod
    def _validate_frequency_range(
        cls, value: object | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise TypeError("frequency_range must be a 2-tuple of positive frequencies")
        lower = _positive_finite(float(value[0]), field_name="frequency_range[0]")
        upper = _positive_finite(float(value[1]), field_name="frequency_range[1]")
        if upper <= lower:
            raise ValueError("frequency_range must be strictly increasing")
        return (lower, upper)

    @property
    def phase1_policy(self) -> str:
        return (
            "CustomAnisotropicMedium is parsed for compatibility inspection only. Phase 1 rejects "
            "sampled tensor media because per-voxel interpolation and tensor materialization are out of scope."
        )


class GenericCustomMedium(TaggedModel):
    """Rejected Phase 1 compatibility bucket for other `Custom*` material tags."""

    type: str
    name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type_name(cls, value: str) -> str:
        if not value.startswith("Custom"):
            raise ValueError("GenericCustomMedium only accepts Custom* type tags")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @model_validator(mode="before")
    @classmethod
    def _fold_extra_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "payload" in value:
            return value
        payload = {key: item for key, item in value.items() if key not in {"type", "name"}}
        return {"type": value.get("type"), "name": value.get("name"), "payload": payload}

    @property
    def phase1_policy(self) -> str:
        return (
            f"{self.type} is mapped to the Phase 1 custom-media reject bucket. "
            "Sampled custom material families are not executed in Phase 1."
        )


__all__ = [
    "CustomAnisotropicMedium",
    "CustomMedium",
    "GenericCustomMedium",
    "LossyMetalMedium",
    "PerturbationMedium",
    "PerturbationPoleResidue",
]
