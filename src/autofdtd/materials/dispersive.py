"""Phase 1 dispersive material models and helpers."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel
from autofdtd.materials.isotropic import Medium, _normalize_name, _positive_finite

ComplexPair: TypeAlias = tuple[float, float]
PoleResiduePair: TypeAlias = tuple[ComplexPair, ComplexPair]
SellmeierCoefficient: TypeAlias = tuple[float, float]
LorentzCoefficient: TypeAlias = tuple[float, float, float]
DrudeCoefficient: TypeAlias = tuple[float, float]
DebyeCoefficient: TypeAlias = tuple[float, float]

EPSILON_0 = 8.8541878128e-12
C_0 = 299_792_458.0


def _finite_float(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _complex_pair_from_value(value: object, *, field_name: str) -> ComplexPair:
    if isinstance(value, complex):
        return (
            _finite_float(value.real, field_name=field_name),
            _finite_float(value.imag, field_name=field_name),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return (
            _finite_float(value[0], field_name=f"{field_name}.real"),
            _finite_float(value[1], field_name=f"{field_name}.imag"),
        )
    raise TypeError(
        f"{field_name} must be a complex number or a 2-tuple of real and imaginary parts"
    )


def complex_from_pair(value: ComplexPair) -> complex:
    """Convert a JSON-safe complex pair into a Python complex number."""

    return complex(value[0], value[1])


def complex_to_pair(value: complex) -> ComplexPair:
    """Convert a Python complex number into a JSON-safe complex pair."""

    return (float(value.real), float(value.imag))


class PoleResidue(TaggedModel):
    """Homogeneous electric-dispersive medium using a pole-residue representation."""

    type: Literal["PoleResidue"] = "PoleResidue"
    name: str | None = None
    eps_inf: float = 1.0
    poles: tuple[PoleResiduePair, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("eps_inf")
    @classmethod
    def _validate_eps_inf(cls, value: float) -> float:
        return _positive_finite(value, field_name="eps_inf")

    @field_validator("poles", mode="before")
    @classmethod
    def _validate_poles(cls, value: object) -> tuple[PoleResiduePair, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("poles must be a sequence of (pole, residue) pairs")

        normalized: list[PoleResiduePair] = []
        for index, item in enumerate(value):
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise TypeError(f"poles[{index}] must be a 2-tuple of (pole, residue)")
            pole_pair = _complex_pair_from_value(item[0], field_name=f"poles[{index}].pole")
            residue_pair = _complex_pair_from_value(item[1], field_name=f"poles[{index}].residue")
            if pole_pair[0] > 0.0:
                raise ValueError("PoleResidue poles must satisfy Re(a) <= 0 for causal stability")
            normalized.append((pole_pair, residue_pair))
        return tuple(normalized)

    @property
    def num_poles(self) -> int:
        """Return the number of dispersive poles."""

        return len(self.poles)

    def eps_model(self, frequency: float) -> complex:
        """Evaluate the complex permittivity at a linear frequency in hertz."""

        omega = 2.0 * math.pi * float(frequency)
        permittivity = complex(self.eps_inf, 0.0)
        for pole_pair, residue_pair in self.poles:
            pole = complex_from_pair(pole_pair)
            residue = complex_from_pair(residue_pair)
            permittivity -= residue / (1j * omega + pole)
            permittivity -= residue.conjugate() / (1j * omega + pole.conjugate())
        return permittivity

    @classmethod
    def from_medium(cls, medium: Medium) -> PoleResidue:
        """Convert a non-dispersive medium into a zero-pole conductivity-only form."""

        return cls(
            name=medium.name,
            eps_inf=medium.permittivity,
            poles=(
                (complex_to_pair(0.0j), complex_to_pair(medium.conductivity / (2.0 * EPSILON_0))),
            ),
        )

    def to_medium(self) -> Medium:
        """Convert a zero-pole conductivity-only model back into a Medium."""

        conductivity = 0.0
        for pole_pair, residue_pair in self.poles:
            pole = complex_from_pair(pole_pair)
            residue = complex_from_pair(residue_pair)
            if abs(pole) > 1e-18:
                raise ValueError(
                    "Cannot convert dispersive PoleResidue with non-zero poles to Medium"
                )
            conductivity += float(2.0 * EPSILON_0 * residue.real)
        return Medium(name=self.name, permittivity=self.eps_inf, conductivity=conductivity)

    def pole_response_factors(self, *, dt: float) -> tuple[tuple[complex, complex], ...]:
        """Return exact-hold recurrence factors for each pole over one timestep."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        factors: list[tuple[complex, complex]] = []
        for pole_pair, residue_pair in self.poles:
            pole = complex_from_pair(pole_pair)
            residue = complex_from_pair(residue_pair)
            decay = cmath.exp(-pole * dt)
            if abs(pole) > 1e-30:
                integral = (1.0 - decay) / pole
            else:
                integral = complex(dt, 0.0)
            drive = -EPSILON_0 * residue * integral
            factors.append((decay, drive))
        return tuple(factors)


class Sellmeier(TaggedModel):
    """Lossless isotropic optical medium using a Sellmeier dispersion law.

    Phase 1 supports homogeneous passive Sellmeier media with coefficients stored as
    ``(B_i, C_i)`` pairs, where ``C_i`` is expressed in square meters.
    """

    type: Literal["Sellmeier"] = "Sellmeier"
    name: str | None = None
    coeffs: tuple[SellmeierCoefficient, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("coeffs", mode="before")
    @classmethod
    def _validate_coeffs(cls, value: object) -> tuple[SellmeierCoefficient, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("coeffs must be a sequence of (B, C) pairs")

        normalized: list[SellmeierCoefficient] = []
        for index, item in enumerate(value):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError(f"coeffs[{index}] must be a 2-tuple of (B, C)")
            b_coeff = _finite_float(item[0], field_name=f"coeffs[{index}].B")
            c_coeff = _finite_float(item[1], field_name=f"coeffs[{index}].C")
            if b_coeff < 0.0:
                raise ValueError("Sellmeier coeffs require B >= 0 in the passive Phase 1 subset")
            if c_coeff < 0.0:
                raise ValueError("Sellmeier coeffs require C >= 0")
            normalized.append((b_coeff, c_coeff))
        return tuple(normalized)

    @property
    def num_terms(self) -> int:
        """Return the number of Sellmeier coefficient pairs."""

        return len(self.coeffs)

    @property
    def eps_inf(self) -> float:
        """Return the epsilon-infinity value after folding zero-wavelength poles."""

        return 1.0 + sum(b_coeff for b_coeff, c_coeff in self.coeffs if c_coeff == 0.0)

    def eps_model(self, frequency: float) -> complex:
        """Evaluate the complex permittivity at a linear frequency in hertz."""

        normalized_frequency = _finite_float(frequency, field_name="frequency")
        if normalized_frequency <= 0.0:
            raise ValueError("frequency must be positive")

        wavelength = C_0 / normalized_frequency
        wavelength_squared = wavelength * wavelength
        permittivity = 1.0
        for b_coeff, c_coeff in self.coeffs:
            if c_coeff == 0.0:
                permittivity += b_coeff
                continue
            permittivity += b_coeff * wavelength_squared / (wavelength_squared - c_coeff)
        return complex(permittivity, 0.0)

    def to_pole_residue(self) -> PoleResidue:
        """Convert the Sellmeier medium into an equivalent PoleResidue model."""

        poles: list[PoleResiduePair] = []
        eps_inf = 1.0
        for b_coeff, c_coeff in self.coeffs:
            if c_coeff == 0.0:
                eps_inf += b_coeff
                continue
            beta = 2.0 * math.pi * C_0 / math.sqrt(c_coeff)
            residue = complex(0.0, -0.5 * beta * b_coeff)
            poles.append((complex_to_pair(complex(0.0, beta)), complex_to_pair(residue)))
        return PoleResidue(name=self.name, eps_inf=eps_inf, poles=tuple(poles))

    @classmethod
    def from_dispersion(
        cls,
        *,
        n: float,
        freq: float,
        dn_dwvl: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> Sellmeier:
        """Create a single-pole passive Sellmeier model from local index data.

        ``dn_dwvl`` uses SI units of inverse meters and must be negative in the supported subset.
        """

        refractive_index = _finite_float(n, field_name="n")
        normalized_frequency = _finite_float(freq, field_name="freq")
        dispersion = _finite_float(dn_dwvl, field_name="dn_dwvl")
        if refractive_index < 1.0:
            raise ValueError("n must be at least 1 for a passive Sellmeier fit")
        if normalized_frequency <= 0.0:
            raise ValueError("freq must be positive")
        if dispersion >= 0.0:
            raise ValueError("dn_dwvl must be negative for the passive Phase 1 subset")

        wavelength = C_0 / normalized_frequency
        nsq_minus_one = refractive_index * refractive_index - 1.0
        c_coeff = -(wavelength**3) * refractive_index * dispersion / (
            nsq_minus_one - wavelength * refractive_index * dispersion
        )
        b_coeff = ((wavelength * wavelength) - c_coeff) / (wavelength * wavelength) * nsq_minus_one
        return cls(name=name, coeffs=((b_coeff, c_coeff),), **kwargs)


class Lorentz(TaggedModel):
    """Homogeneous electric-dispersive medium using a Lorentz oscillator model."""

    type: Literal["Lorentz"] = "Lorentz"
    name: str | None = None
    eps_inf: float = 1.0
    coeffs: tuple[LorentzCoefficient, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("eps_inf")
    @classmethod
    def _validate_eps_inf(cls, value: float) -> float:
        return _positive_finite(value, field_name="eps_inf")

    @field_validator("coeffs", mode="before")
    @classmethod
    def _validate_coeffs(cls, value: object) -> tuple[LorentzCoefficient, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("coeffs must be a sequence of (delta_eps, frequency, damping) triples")

        normalized: list[LorentzCoefficient] = []
        for index, item in enumerate(value):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 3:
                raise TypeError(
                    f"coeffs[{index}] must be a 3-tuple of (delta_eps, frequency, damping)"
                )
            delta_eps = _finite_float(item[0], field_name=f"coeffs[{index}].delta_eps")
            resonance = _positive_finite(item[1], field_name=f"coeffs[{index}].frequency")
            damping = _finite_float(item[2], field_name=f"coeffs[{index}].damping")
            if delta_eps < 0.0:
                raise ValueError(
                    "Lorentz coeffs require delta_eps >= 0 in the passive Phase 1 subset"
                )
            if damping < 0.0:
                raise ValueError("Lorentz coeffs require damping >= 0")
            if math.isclose(resonance * resonance, damping * damping, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("Lorentz coeffs require frequency^2 != damping^2")
            normalized.append((delta_eps, resonance, damping))
        return tuple(normalized)

    @property
    def num_terms(self) -> int:
        return len(self.coeffs)

    def eps_model(self, frequency: float) -> complex:
        """Evaluate the complex permittivity at a linear frequency in hertz."""

        normalized_frequency = _finite_float(frequency, field_name="frequency")
        permittivity = complex(self.eps_inf, 0.0)
        for delta_eps, resonance, damping in self.coeffs:
            permittivity += (delta_eps * resonance * resonance) / (
                resonance * resonance
                - normalized_frequency * normalized_frequency
                - 2.0j * normalized_frequency * damping
            )
        return permittivity

    def to_pole_residue(self) -> PoleResidue:
        """Convert the Lorentz medium into an equivalent PoleResidue model."""

        poles: list[PoleResiduePair] = []
        for delta_eps, resonance, damping in self.coeffs:
            omega_0 = 2.0 * math.pi * resonance
            gamma = 2.0 * math.pi * damping
            if gamma * gamma > omega_0 * omega_0:
                root = complex(math.sqrt(gamma * gamma - omega_0 * omega_0), 0.0)
                pole_a = complex(-gamma, 0.0) + root
                residue_a = complex(delta_eps * omega_0 * omega_0 / 4.0, 0.0) / root
                pole_b = complex(-gamma, 0.0) - root
                residue_b = -residue_a
                poles.append((complex_to_pair(pole_a), complex_to_pair(residue_a)))
                poles.append((complex_to_pair(pole_b), complex_to_pair(residue_b)))
                continue

            root = math.sqrt(omega_0 * omega_0 - gamma * gamma)
            pole = complex(-gamma, -root)
            residue = 1j * delta_eps * omega_0 * omega_0 / (2.0 * root)
            poles.append((complex_to_pair(pole), complex_to_pair(residue)))
        return PoleResidue(name=self.name, eps_inf=self.eps_inf, poles=tuple(poles))


class Drude(TaggedModel):
    """Homogeneous electric-dispersive medium using a damped Drude model."""

    type: Literal["Drude"] = "Drude"
    name: str | None = None
    eps_inf: float = 1.0
    coeffs: tuple[DrudeCoefficient, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("eps_inf")
    @classmethod
    def _validate_eps_inf(cls, value: float) -> float:
        return _positive_finite(value, field_name="eps_inf")

    @field_validator("coeffs", mode="before")
    @classmethod
    def _validate_coeffs(cls, value: object) -> tuple[DrudeCoefficient, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("coeffs must be a sequence of (plasma_frequency, damping) pairs")

        normalized: list[DrudeCoefficient] = []
        for index, item in enumerate(value):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError(
                    f"coeffs[{index}] must be a 2-tuple of (plasma_frequency, damping)"
                )
            plasma_frequency = _positive_finite(
                item[0], field_name=f"coeffs[{index}].plasma_frequency"
            )
            damping = _positive_finite(item[1], field_name=f"coeffs[{index}].damping")
            normalized.append((plasma_frequency, damping))
        return tuple(normalized)

    @property
    def num_terms(self) -> int:
        return len(self.coeffs)

    def eps_model(self, frequency: float) -> complex:
        """Evaluate the complex permittivity at a linear frequency in hertz."""

        normalized_frequency = _finite_float(frequency, field_name="frequency")
        permittivity = complex(self.eps_inf, 0.0)
        for plasma_frequency, damping in self.coeffs:
            permittivity -= (plasma_frequency * plasma_frequency) / (
                normalized_frequency * normalized_frequency + 1.0j * normalized_frequency * damping
            )
        return permittivity

    def to_pole_residue(self) -> PoleResidue:
        """Convert the Drude medium into an equivalent PoleResidue model."""

        poles: list[PoleResiduePair] = []
        for plasma_frequency, damping in self.coeffs:
            omega_p = 2.0 * math.pi * plasma_frequency
            gamma = 2.0 * math.pi * damping
            residue_a = complex(omega_p * omega_p / (2.0 * gamma), 0.0)
            residue_b = -residue_a
            poles.append((complex_to_pair(0.0j), complex_to_pair(residue_a)))
            poles.append((complex_to_pair(complex(-gamma, 0.0)), complex_to_pair(residue_b)))
        return PoleResidue(name=self.name, eps_inf=self.eps_inf, poles=tuple(poles))


class Debye(TaggedModel):
    """Homogeneous electric-dispersive medium using a Debye relaxation model."""

    type: Literal["Debye"] = "Debye"
    name: str | None = None
    eps_inf: float = 1.0
    coeffs: tuple[DebyeCoefficient, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _normalize_name(value)

    @field_validator("eps_inf")
    @classmethod
    def _validate_eps_inf(cls, value: float) -> float:
        return _positive_finite(value, field_name="eps_inf")

    @field_validator("coeffs", mode="before")
    @classmethod
    def _validate_coeffs(cls, value: object) -> tuple[DebyeCoefficient, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("coeffs must be a sequence of (delta_eps, tau) pairs")

        normalized: list[DebyeCoefficient] = []
        for index, item in enumerate(value):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError(f"coeffs[{index}] must be a 2-tuple of (delta_eps, tau)")
            delta_eps = _finite_float(item[0], field_name=f"coeffs[{index}].delta_eps")
            tau = _positive_finite(item[1], field_name=f"coeffs[{index}].tau")
            if delta_eps < 0.0:
                raise ValueError(
                    "Debye coeffs require delta_eps >= 0 in the passive Phase 1 subset"
                )
            normalized.append((delta_eps, tau))
        return tuple(normalized)

    @property
    def num_terms(self) -> int:
        return len(self.coeffs)

    def eps_model(self, frequency: float) -> complex:
        """Evaluate the complex permittivity at a linear frequency in hertz."""

        normalized_frequency = _finite_float(frequency, field_name="frequency")
        permittivity = complex(self.eps_inf, 0.0)
        for delta_eps, tau in self.coeffs:
            permittivity += delta_eps / (1.0 - 1.0j * normalized_frequency * tau)
        return permittivity

    def to_pole_residue(self) -> PoleResidue:
        """Convert the Debye medium into an equivalent PoleResidue model."""

        poles: list[PoleResiduePair] = []
        eps_inf = self.eps_inf
        for delta_eps, tau in self.coeffs:
            pole = complex(-2.0 * math.pi / tau, 0.0)
            residue = -0.5 * delta_eps * pole
            poles.append((complex_to_pair(pole), complex_to_pair(residue)))
        return PoleResidue(name=self.name, eps_inf=eps_inf, poles=tuple(poles))


DispersiveMedium = PoleResidue | Sellmeier | Lorentz | Drude | Debye
