"""Boundary models for the foundational Phase 1 boundary surface."""

from __future__ import annotations

import cmath
import warnings
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from autofdtd.core.models import TaggedModel
from autofdtd.geometry.primitives import Box


def _normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("names must not be empty or whitespace-only")
    return stripped


class BoundaryEdge(TaggedModel):
    """Base model for one simulation-domain boundary edge."""

    name: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_optional_name(value)


class Periodic(BoundaryEdge):
    """Periodic boundary condition."""

    type: Literal["Periodic"] = "Periodic"


class BlochBoundary(BoundaryEdge):
    """Phase-aware periodic boundary condition."""

    type: Literal["BlochBoundary"] = "BlochBoundary"
    bloch_vec: float

    @property
    def bloch_phase(self) -> complex:
        """Return the forward phase factor for a positive-domain wrap."""

        return cmath.exp(1j * 2.0 * cmath.pi * self.bloch_vec)


class PECBoundary(BoundaryEdge):
    """Perfect electric conductor boundary condition."""

    type: Literal["PECBoundary"] = "PECBoundary"


class PMCBoundary(BoundaryEdge):
    """Perfect magnetic conductor boundary condition."""

    type: Literal["PMCBoundary"] = "PMCBoundary"


class ABCBoundary(BoundaryEdge):
    """First-order absorbing boundary condition with explicit effective medium inputs."""

    type: Literal["ABCBoundary"] = "ABCBoundary"
    permittivity: float | None = None
    conductivity: float | None = None

    @field_validator("permittivity")
    @classmethod
    def _validate_permittivity(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 1.0:
            raise ValueError("ABCBoundary permittivity must be at least 1.0 when provided")
        return normalized

    @field_validator("conductivity")
    @classmethod
    def _validate_conductivity(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0.0:
            raise ValueError("ABCBoundary conductivity must be non-negative when provided")
        return normalized

    @model_validator(mode="after")
    def _validate_parameter_pair(self) -> ABCBoundary:
        if self.conductivity is not None and self.permittivity is None:
            raise ValueError(
                "ABCBoundary conductivity can only be provided together with permittivity"
            )
        return self


class BroadbandModeABCFitterParam(TaggedModel):
    """Deferred fit-parameter surface for broadband mode-ABC configuration."""

    type: Literal["BroadbandModeABCFitterParam"] = "BroadbandModeABCFitterParam"
    max_num_poles: int = 4
    tolerance_rms: float = 1.0e-4
    frequency_sampling_points: int = 11

    @field_validator("max_num_poles", "frequency_sampling_points")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("Broadband mode-ABC fit counts must be positive")
        return normalized

    @field_validator("tolerance_rms")
    @classmethod
    def _validate_non_negative_float(cls, value: float) -> float:
        normalized = float(value)
        if normalized < 0.0:
            raise ValueError("Broadband mode-ABC fit tolerance must be non-negative")
        return normalized


class BroadbandModeABCSpec(TaggedModel):
    """Deferred broadband fitting surface for mode-based absorbing boundaries."""

    type: Literal["BroadbandModeABCSpec"] = "BroadbandModeABCSpec"
    frequency_range: tuple[float, float]
    fit_param: BroadbandModeABCFitterParam = Field(default_factory=BroadbandModeABCFitterParam)

    @field_validator("frequency_range", mode="before")
    @classmethod
    def _validate_frequency_range(cls, value: object) -> tuple[float, float]:
        if not isinstance(value, tuple | list) or len(value) != 2:
            raise TypeError("frequency_range must be a 2-tuple of positive frequencies")
        lower = float(value[0])
        upper = float(value[1])
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("frequency_range values must be positive")
        if upper <= lower:
            raise ValueError("frequency_range must be strictly increasing")
        return (lower, upper)


class ModeABCBoundary(BoundaryEdge):
    """Deferred mode-absorbing boundary surface that depends on the future mode solver."""

    type: Literal["ModeABCBoundary"] = "ModeABCBoundary"
    mode_spec: dict[str, Any] = Field(
        default_factory=lambda: {"type": "ModeSpec", "num_modes": 1}
    )
    mode_index: int = 0
    freq_spec: float | BroadbandModeABCSpec | None = None
    plane: Box

    @field_validator("mode_index")
    @classmethod
    def _validate_mode_index(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 0:
            raise ValueError("ModeABCBoundary mode_index must be non-negative")
        return normalized

    @field_validator("freq_spec")
    @classmethod
    def _validate_freq_spec(
        cls, value: float | BroadbandModeABCSpec | Mapping[str, Any] | None
    ) -> float | BroadbandModeABCSpec | None:
        if value is None or isinstance(value, BroadbandModeABCSpec):
            return value
        if isinstance(value, Mapping):
            return BroadbandModeABCSpec(**value)
        normalized = float(value)
        if normalized <= 0.0:
            raise ValueError("ModeABCBoundary freq_spec must be positive when scalar")
        return normalized

    @field_validator("plane")
    @classmethod
    def _validate_plane(cls, value: Box | Mapping[str, Any]) -> Box:
        plane = value if isinstance(value, Box) else Box(**value)
        if plane.size.count(0.0) != 1:
            raise ValueError(
                f"ModeABCBoundary plane must be planar with one zero extent, got size={plane.size}"
            )
        return plane

    @property
    def phase1_policy(self) -> str:
        return (
            "ModeABCBoundary is parsed and serialized, but runtime compilation is deferred in "
            "Phase 1 until the mode solver and mode-source surfaces exist."
        )


class PMLParams(TaggedModel):
    """Polynomial profile parameters for the Phase 1 baseline PML."""

    type: Literal["PMLParams"] = "PMLParams"
    sigma_order: int = 3
    sigma_min: float = 0.0
    sigma_max: float = 1.5
    kappa_order: int = 3
    kappa_min: float = 1.0
    kappa_max: float = 3.0
    alpha_order: int = 1
    alpha_min: float = 0.0
    alpha_max: float = 0.0

    @field_validator("sigma_order", "kappa_order", "alpha_order")
    @classmethod
    def _validate_order(cls, value: int) -> int:
        if value < 0:
            raise ValueError("PML profile orders must be non-negative")
        return int(value)

    @field_validator(
        "sigma_min",
        "sigma_max",
        "kappa_min",
        "kappa_max",
        "alpha_min",
        "alpha_max",
    )
    @classmethod
    def _validate_non_negative(cls, value: float) -> float:
        normalized = float(value)
        if normalized < 0.0:
            raise ValueError("PML profile coefficients must be non-negative")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> PMLParams:
        if self.sigma_max < self.sigma_min:
            raise ValueError("PML sigma_max must be greater than or equal to sigma_min")
        if self.kappa_min < 1.0:
            raise ValueError("PML kappa_min must be at least 1.0")
        if self.kappa_max < self.kappa_min:
            raise ValueError("PML kappa_max must be greater than or equal to kappa_min")
        if self.alpha_max < self.alpha_min:
            raise ValueError("PML alpha_max must be greater than or equal to alpha_min")
        return self


class AbsorberParams(TaggedModel):
    """Polynomial conductivity profile parameters for adiabatic absorbers."""

    type: Literal["AbsorberParams"] = "AbsorberParams"
    sigma_order: int = 3
    sigma_min: float = 0.0
    sigma_max: float = 6.4

    @field_validator("sigma_order")
    @classmethod
    def _validate_order(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Absorber profile orders must be non-negative")
        return int(value)

    @field_validator("sigma_min", "sigma_max")
    @classmethod
    def _validate_non_negative(cls, value: float) -> float:
        normalized = float(value)
        if normalized < 0.0:
            raise ValueError("Absorber profile coefficients must be non-negative")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> AbsorberParams:
        if self.sigma_max < self.sigma_min:
            raise ValueError("Absorber sigma_max must be greater than or equal to sigma_min")
        return self


DEFAULT_PML_PARAMS = PMLParams()
DEFAULT_STABLE_PML_PARAMS = PMLParams(
    sigma_order=3,
    sigma_min=0.0,
    sigma_max=1.0,
    kappa_order=3,
    kappa_min=1.0,
    kappa_max=5.0,
    alpha_order=1,
    alpha_min=0.0,
    alpha_max=0.9,
)
DEFAULT_ABSORBER_PARAMS = AbsorberParams()


class PML(BoundaryEdge):
    """Complex-frequency-shift-inspired perfectly matched layer edge."""

    type: Literal["PML"] = "PML"
    num_layers: int = 12
    parameters: PMLParams = Field(default_factory=PMLParams)
    extrude_structures: bool = True

    @field_validator("num_layers")
    @classmethod
    def _validate_num_layers(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("PML num_layers must be at least 1")
        if normalized < 6:
            from autofdtd.core.validation import AutoFDTDValidationWarning

            warnings.warn(
                "PML boundaries with fewer than 6 layers are prone to reflection and should "
                "only be used for small smoke tests.",
                AutoFDTDValidationWarning,
                stacklevel=2,
            )
        return normalized


class StablePML(BoundaryEdge):
    """More conservative complex-frequency-shift PML preset."""

    type: Literal["StablePML"] = "StablePML"
    num_layers: int = 40
    parameters: PMLParams = Field(default_factory=lambda: DEFAULT_STABLE_PML_PARAMS.copy_update())
    extrude_structures: bool = True

    @field_validator("num_layers")
    @classmethod
    def _validate_num_layers(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("StablePML num_layers must be at least 1")
        if normalized < 6:
            from autofdtd.core.validation import AutoFDTDValidationWarning

            warnings.warn(
                "StablePML boundaries with fewer than 6 layers are prone to reflection and "
                "should only be used for small smoke tests.",
                AutoFDTDValidationWarning,
                stacklevel=2,
            )
        return normalized


class Absorber(BoundaryEdge):
    """Adiabatic absorbing layer backed by a reflective outer wall."""

    type: Literal["Absorber"] = "Absorber"
    num_layers: int = 40
    parameters: AbsorberParams = Field(default_factory=AbsorberParams)
    extrude_structures: bool = False

    @field_validator("num_layers")
    @classmethod
    def _validate_num_layers(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("Absorber num_layers must be at least 1")
        if normalized < 6:
            from autofdtd.core.validation import AutoFDTDValidationWarning

            warnings.warn(
                "Absorber boundaries with fewer than 6 layers are prone to reflection and "
                "should only be used for small smoke tests.",
                AutoFDTDValidationWarning,
                stacklevel=2,
            )
        return normalized


ImplementedBoundaryEdge = (
    Periodic
    | BlochBoundary
    | PECBoundary
    | PMCBoundary
    | ABCBoundary
    | ModeABCBoundary
    | PML
    | StablePML
    | Absorber
)


def boundary_edge_model_from_value(value: object) -> ImplementedBoundaryEdge:
    """Normalize a supported Phase 1 non-absorbing boundary edge."""

    if isinstance(
        value,
        (
            Periodic,
            BlochBoundary,
            PECBoundary,
            PMCBoundary,
            ABCBoundary,
            ModeABCBoundary,
            PML,
            StablePML,
            Absorber,
        ),
    ):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a boundary-edge mapping, got {type(value)!r}")
    boundary_type = str(value.get("type"))
    if boundary_type == "Periodic":
        return Periodic(**value)
    if boundary_type == "BlochBoundary":
        return BlochBoundary(**value)
    if boundary_type == "PECBoundary":
        return PECBoundary(**value)
    if boundary_type == "PMCBoundary":
        return PMCBoundary(**value)
    if boundary_type == "ABCBoundary":
        return ABCBoundary(**value)
    if boundary_type == "ModeABCBoundary":
        return ModeABCBoundary(**value)
    if boundary_type == "PML":
        return PML(**value)
    if boundary_type == "StablePML":
        return StablePML(**value)
    if boundary_type == "Absorber":
        return Absorber(**value)
    raise TypeError(f"unsupported Phase 1 boundary edge type {boundary_type!r}")


class Boundary(TaggedModel):
    """Boundary pair across one simulation axis."""

    type: Literal["Boundary"] = "Boundary"
    plus: object = Field(default_factory=Periodic)
    minus: object = Field(default_factory=Periodic)

    @field_validator("plus", "minus")
    @classmethod
    def _validate_edge(cls, value: object) -> object:
        if isinstance(
            value,
            (
                Periodic,
                BlochBoundary,
                PECBoundary,
                PMCBoundary,
                ABCBoundary,
                ModeABCBoundary,
                PML,
                StablePML,
                Absorber,
            ),
        ):
            return value
        if isinstance(value, Mapping):
            boundary_type = str(value.get("type"))
            if boundary_type in {
                "Periodic",
                "BlochBoundary",
                "PECBoundary",
                "PMCBoundary",
                "ABCBoundary",
                "ModeABCBoundary",
                "PML",
                "StablePML",
                "Absorber",
            }:
                return boundary_edge_model_from_value(value)
        return value

    @model_validator(mode="after")
    def _validate_edge_composition(self) -> Boundary:
        plus = self.plus
        minus = self.minus

        num_bloch = isinstance(plus, BlochBoundary) + isinstance(minus, BlochBoundary)
        if num_bloch == 1:
            raise ValueError("Bloch boundaries must be applied on both sides of the same axis")
        if isinstance(plus, BlochBoundary) and isinstance(minus, BlochBoundary):
            if plus.bloch_vec != minus.bloch_vec:
                raise ValueError(
                    "Bloch boundaries on opposite sides of one axis must use the same bloch_vec"
                )
            return self

        changed = False

        if isinstance(minus, (PECBoundary, PMCBoundary)) and isinstance(plus, Periodic):
            plus = minus
            changed = True
        elif isinstance(plus, (PECBoundary, PMCBoundary)) and isinstance(minus, Periodic):
            minus = plus
            changed = True

        if changed:
            from autofdtd.core.validation import AutoFDTDValidationWarning

            warnings.warn(
                "A periodic boundary opposite a PEC/PMC boundary is coerced to the conductor "
                "boundary so runtime halo handling stays reflection-only on that axis.",
                AutoFDTDValidationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "plus", plus)
            object.__setattr__(self, "minus", minus)

        return self

    def __getitem__(self, field_name: str) -> object:
        if field_name == "plus":
            return self.plus
        if field_name == "minus":
            return self.minus
        if field_name == "type":
            return self.type
        raise KeyError(field_name)

    @classmethod
    def periodic(cls) -> Boundary:
        return cls(plus=Periodic(), minus=Periodic())

    @classmethod
    def bloch(cls, bloch_vec: float) -> Boundary:
        return cls(
            plus=BlochBoundary(bloch_vec=bloch_vec),
            minus=BlochBoundary(bloch_vec=bloch_vec),
        )

    @classmethod
    def pec(cls) -> Boundary:
        return cls(plus=PECBoundary(), minus=PECBoundary())

    @classmethod
    def pmc(cls) -> Boundary:
        return cls(plus=PMCBoundary(), minus=PMCBoundary())

    @classmethod
    def abc(
        cls,
        *,
        permittivity: float | None = None,
        conductivity: float | None = None,
    ) -> Boundary:
        abc = ABCBoundary(permittivity=permittivity, conductivity=conductivity)
        return cls(plus=abc, minus=abc)

    @classmethod
    def mode_abc(
        cls,
        *,
        plane: Box,
        mode_spec: dict[str, Any] | None = None,
        mode_index: int = 0,
        freq_spec: float | BroadbandModeABCSpec | None = None,
    ) -> Boundary:
        boundary = ModeABCBoundary(
            plane=plane,
            mode_spec={"type": "ModeSpec", "num_modes": 1} if mode_spec is None else mode_spec,
            mode_index=mode_index,
            freq_spec=freq_spec,
        )
        return cls(plus=boundary, minus=boundary)

    @classmethod
    def pml(
        cls,
        *,
        num_layers: int = 12,
        parameters: PMLParams | None = None,
        extrude_structures: bool = True,
    ) -> Boundary:
        pml = PML(
            num_layers=num_layers,
            parameters=DEFAULT_PML_PARAMS if parameters is None else parameters,
            extrude_structures=extrude_structures,
        )
        return cls(plus=pml, minus=pml)

    @classmethod
    def stable_pml(
        cls,
        *,
        num_layers: int = 40,
        parameters: PMLParams | None = None,
        extrude_structures: bool = True,
    ) -> Boundary:
        stable = StablePML(
            num_layers=num_layers,
            parameters=DEFAULT_STABLE_PML_PARAMS if parameters is None else parameters,
            extrude_structures=extrude_structures,
        )
        return cls(plus=stable, minus=stable)

    @classmethod
    def absorber(
        cls,
        *,
        num_layers: int = 40,
        parameters: AbsorberParams | None = None,
        extrude_structures: bool = False,
    ) -> Boundary:
        absorber = Absorber(
            num_layers=num_layers,
            parameters=DEFAULT_ABSORBER_PARAMS if parameters is None else parameters,
            extrude_structures=extrude_structures,
        )
        return cls(plus=absorber, minus=absorber)


def boundary_model_from_value(value: object) -> Boundary:
    """Normalize a supported axis boundary pair."""

    if isinstance(value, Boundary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a boundary mapping, got {type(value)!r}")
    if "plus" not in value and "minus" not in value:
        edge = boundary_edge_model_from_value(value)
        return Boundary(plus=edge, minus=edge)
    return Boundary(**value)


class BoundarySpec(TaggedModel):
    """Boundary conditions across the x, y, and z simulation axes."""

    type: Literal["BoundarySpec"] = "BoundarySpec"
    x: Boundary = Field(default_factory=Boundary.periodic)
    y: Boundary = Field(default_factory=Boundary.periodic)
    z: Boundary = Field(default_factory=Boundary.periodic)

    @field_validator("x", "y", "z", mode="before")
    @classmethod
    def _coerce_axis_boundary(cls, value: object) -> object:
        if isinstance(
            value,
            (
                Boundary,
                Periodic,
                BlochBoundary,
                PECBoundary,
                PMCBoundary,
                ABCBoundary,
                ModeABCBoundary,
                PML,
                StablePML,
                Absorber,
            ),
        ):
            return value
        if isinstance(value, Mapping):
            boundary_type = str(value.get("type"))
            if boundary_type in {
                "Boundary",
                "Periodic",
                "BlochBoundary",
                "PECBoundary",
                "PMCBoundary",
                "ABCBoundary",
                "ModeABCBoundary",
                "PML",
                "StablePML",
                "Absorber",
            }:
                return boundary_model_from_value(value)
        return value

    def __getitem__(self, field_name: str) -> Boundary:
        if field_name == "x":
            return self.x
        if field_name == "y":
            return self.y
        if field_name == "z":
            return self.z
        raise KeyError(field_name)

    def as_tuple(self) -> tuple[Boundary, Boundary, Boundary]:
        return (self.x, self.y, self.z)

    @classmethod
    def all_sides(cls, boundary: ImplementedBoundaryEdge) -> BoundarySpec:
        return cls(
            x=Boundary(plus=boundary, minus=boundary),
            y=Boundary(plus=boundary, minus=boundary),
            z=Boundary(plus=boundary, minus=boundary),
        )

    @classmethod
    def pec(cls, *, x: bool = False, y: bool = False, z: bool = False) -> BoundarySpec:
        return cls(
            x=Boundary.pec() if x else Boundary.periodic(),
            y=Boundary.pec() if y else Boundary.periodic(),
            z=Boundary.pec() if z else Boundary.periodic(),
        )

    @classmethod
    def pmc(cls, *, x: bool = False, y: bool = False, z: bool = False) -> BoundarySpec:
        return cls(
            x=Boundary.pmc() if x else Boundary.periodic(),
            y=Boundary.pmc() if y else Boundary.periodic(),
            z=Boundary.pmc() if z else Boundary.periodic(),
        )

    @classmethod
    def periodic(
        cls, *, x: bool = True, y: bool = True, z: bool = True
    ) -> BoundarySpec:
        return cls(
            x=Boundary.periodic() if x else Boundary.pml(),
            y=Boundary.periodic() if y else Boundary.pml(),
            z=Boundary.periodic() if z else Boundary.pml(),
        )

    @classmethod
    def abc(
        cls,
        *,
        x: bool = False,
        y: bool = False,
        z: bool = False,
        permittivity: float | None = None,
        conductivity: float | None = None,
    ) -> BoundarySpec:
        return cls(
            x=(
                Boundary.abc(permittivity=permittivity, conductivity=conductivity)
                if x
                else Boundary.periodic()
            ),
            y=(
                Boundary.abc(permittivity=permittivity, conductivity=conductivity)
                if y
                else Boundary.periodic()
            ),
            z=(
                Boundary.abc(permittivity=permittivity, conductivity=conductivity)
                if z
                else Boundary.periodic()
            ),
        )

    @classmethod
    def mode_abc(
        cls,
        *,
        x: bool = False,
        y: bool = False,
        z: bool = False,
        plane: Box,
        mode_spec: dict[str, Any] | None = None,
        mode_index: int = 0,
        freq_spec: float | BroadbandModeABCSpec | None = None,
    ) -> BoundarySpec:
        return cls(
            x=(
                Boundary.mode_abc(
                    plane=plane,
                    mode_spec=mode_spec,
                    mode_index=mode_index,
                    freq_spec=freq_spec,
                )
                if x
                else Boundary.periodic()
            ),
            y=(
                Boundary.mode_abc(
                    plane=plane,
                    mode_spec=mode_spec,
                    mode_index=mode_index,
                    freq_spec=freq_spec,
                )
                if y
                else Boundary.periodic()
            ),
            z=(
                Boundary.mode_abc(
                    plane=plane,
                    mode_spec=mode_spec,
                    mode_index=mode_index,
                    freq_spec=freq_spec,
                )
                if z
                else Boundary.periodic()
            ),
        )

    @classmethod
    def pml(
        cls,
        *,
        x: bool = False,
        y: bool = False,
        z: bool = False,
        num_layers: int = 12,
        parameters: PMLParams | None = None,
        extrude_structures: bool = True,
    ) -> BoundarySpec:
        return cls(
            x=(
                Boundary.pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if x
                else Boundary.periodic()
            ),
            y=(
                Boundary.pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if y
                else Boundary.periodic()
            ),
            z=(
                Boundary.pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if z
                else Boundary.periodic()
            ),
        )

    @classmethod
    def stable_pml(
        cls,
        *,
        x: bool = False,
        y: bool = False,
        z: bool = False,
        num_layers: int = 40,
        parameters: PMLParams | None = None,
        extrude_structures: bool = True,
    ) -> BoundarySpec:
        return cls(
            x=(
                Boundary.stable_pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if x
                else Boundary.periodic()
            ),
            y=(
                Boundary.stable_pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if y
                else Boundary.periodic()
            ),
            z=(
                Boundary.stable_pml(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if z
                else Boundary.periodic()
            ),
        )

    @classmethod
    def absorber(
        cls,
        *,
        x: bool = False,
        y: bool = False,
        z: bool = False,
        num_layers: int = 40,
        parameters: AbsorberParams | None = None,
        extrude_structures: bool = False,
    ) -> BoundarySpec:
        return cls(
            x=(
                Boundary.absorber(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if x
                else Boundary.periodic()
            ),
            y=(
                Boundary.absorber(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if y
                else Boundary.periodic()
            ),
            z=(
                Boundary.absorber(
                    num_layers=num_layers,
                    parameters=parameters,
                    extrude_structures=extrude_structures,
                )
                if z
                else Boundary.periodic()
            ),
        )


def boundary_spec_model_from_value(value: object) -> BoundarySpec:
    """Normalize a supported BoundarySpec container."""

    if isinstance(value, BoundarySpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a boundary-spec mapping, got {type(value)!r}")
    return BoundarySpec(**value)
