"""Subpixel-policy models for interface materialization planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, cast

from pydantic import Field, ValidationInfo, field_validator

from autofdtd.core.models import TaggedModel
from autofdtd.planning import FeatureStatus, feature_entry

SubpixelTarget = Literal["dielectric", "metal", "pec", "pmc", "lossy_metal"]
_ALL_TARGETS: Final[tuple[SubpixelTarget, ...]] = (
    "dielectric",
    "metal",
    "pec",
    "pmc",
    "lossy_metal",
)


class AbstractSubpixelPolicy(TaggedModel):
    """Base class for subpixel policies applied at structure interfaces."""

    def courant_ratio(self) -> float:
        """Return the Courant scaling required by this policy."""
        return 1.0

    def materialization_mode(self) -> Literal["staircasing", "averaging"]:
        """Return the high-level interface handling mode."""
        return "staircasing"

    def requires_anisotropic_materialization(self) -> bool:
        """Return whether the policy may emit tensor-valued effective media."""
        return False


class Staircasing(AbstractSubpixelPolicy):
    """Assign interface cells without subcell smoothing."""

    type: Literal["Staircasing"] = "Staircasing"


class PolarizedAveraging(AbstractSubpixelPolicy):
    """Apply a polarization-aware dielectric smoothing policy."""

    type: Literal["PolarizedAveraging"] = "PolarizedAveraging"

    def materialization_mode(self) -> Literal["staircasing", "averaging"]:
        return "averaging"

    def requires_anisotropic_materialization(self) -> bool:
        return True


ImplementedDielectricPolicy = Annotated[
    Staircasing | PolarizedAveraging,
    Field(discriminator="type"),
]
ImplementedConductorPolicy = Staircasing


def _deferred_policy_error(type_name: str) -> ValueError:
    entry = feature_entry(type_name)
    if entry is None:
        return ValueError(f"unsupported subpixel policy type {type_name!r}")
    status_text = {
        FeatureStatus.DEFER: "deferred",
        FeatureStatus.REJECT_CLEARLY: "rejected clearly",
        FeatureStatus.IMPLEMENT: "implemented",
    }[entry.status]
    return ValueError(
        f"subpixel policy {type_name!r} is {status_text} in Phase 1. {entry.notes}"
    )


def _coerce_policy(value: object, *, target: SubpixelTarget) -> AbstractSubpixelPolicy:
    if isinstance(value, (Staircasing, PolarizedAveraging)):
        policy = value
    else:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"subpixel policy for {target!r} must be mapping-like, got {type(value)!r}"
            )
        type_name = str(value.get("type", ""))
        model_type = {
            "Staircasing": Staircasing,
            "PolarizedAveraging": PolarizedAveraging,
        }.get(type_name)
        if model_type is None:
            raise _deferred_policy_error(type_name)
        policy = model_type(**dict(value))

    if target == "dielectric":
        return policy
    if not isinstance(policy, Staircasing):
        raise ValueError(
            f"Phase 1 only supports Staircasing for {target!r} interfaces; "
            f"got {policy.type!r}"
        )
    return policy


class SubpixelSpec(TaggedModel):
    """Phase 1 subpixel controls aligned to the local Tidy3D interface families."""

    type: Literal["SubpixelSpec"] = "SubpixelSpec"
    dielectric: ImplementedDielectricPolicy = Field(default_factory=PolarizedAveraging)
    metal: ImplementedConductorPolicy = Field(default_factory=Staircasing)
    pec: ImplementedConductorPolicy = Field(default_factory=Staircasing)
    pmc: ImplementedConductorPolicy = Field(default_factory=Staircasing)
    lossy_metal: ImplementedConductorPolicy = Field(default_factory=Staircasing)

    @field_validator("dielectric", mode="before")
    @classmethod
    def _validate_dielectric(cls, value: object) -> AbstractSubpixelPolicy:
        return _coerce_policy(value, target="dielectric")

    @field_validator("metal", "pec", "pmc", "lossy_metal", mode="before")
    @classmethod
    def _validate_conductors(cls, value: object, info: ValidationInfo) -> Staircasing:
        target = cast(SubpixelTarget, info.field_name)
        return cast(Staircasing, _coerce_policy(value, target=target))

    @classmethod
    def staircasing(cls) -> SubpixelSpec:
        """Return a spec that staircases every interface family."""
        staircase = Staircasing()
        return cls(
            dielectric=staircase,
            metal=staircase,
            pec=staircase,
            pmc=staircase,
            lossy_metal=staircase,
        )

    def policy_for(self, target: SubpixelTarget) -> AbstractSubpixelPolicy:
        """Return the policy assigned to one interface family."""
        return cast(AbstractSubpixelPolicy, getattr(self, target))

    def averaging_targets(self) -> tuple[SubpixelTarget, ...]:
        """Return interface families that request subcell averaging."""
        targets: list[SubpixelTarget] = []
        for target in _ALL_TARGETS:
            if self.policy_for(target).materialization_mode() == "averaging":
                targets.append(target)
        return tuple(targets)

    def staircasing_targets(self) -> tuple[SubpixelTarget, ...]:
        """Return interface families that remain on staircasing."""
        targets: list[SubpixelTarget] = []
        for target in _ALL_TARGETS:
            if self.policy_for(target).materialization_mode() == "staircasing":
                targets.append(target)
        return tuple(targets)

    def courant_ratio(self) -> float:
        """Return the effective Courant scaling across all configured policies."""
        return min(
            self.dielectric.courant_ratio(),
            self.metal.courant_ratio(),
            self.pec.courant_ratio(),
            self.pmc.courant_ratio(),
            self.lossy_metal.courant_ratio(),
        )

    def requires_anisotropic_materialization(self) -> bool:
        """Return whether any configured policy requires tensor-valued materialization."""
        return any(
            self.policy_for(target).requires_anisotropic_materialization()
            for target in _ALL_TARGETS
        )


def subpixel_model_from_value(value: object) -> SubpixelSpec:
    """Normalize a public subpixel payload to the Phase 1 model."""
    if isinstance(value, SubpixelSpec):
        return value
    if isinstance(value, bool):
        return SubpixelSpec() if value else SubpixelSpec.staircasing()
    if not isinstance(value, Mapping):
        raise TypeError(f"subpixel payload must be bool or mapping-like, got {type(value)!r}")
    type_name = str(value.get("type", ""))
    if type_name != "SubpixelSpec":
        raise ValueError(
            "simulation.subpixel must be a bool or a SubpixelSpec payload in Phase 1"
        )
    return SubpixelSpec(**dict(value))
