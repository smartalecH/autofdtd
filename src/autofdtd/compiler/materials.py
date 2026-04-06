"""Material sampling and coefficient preparation for Phase 1 material paths."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import field_validator

from autofdtd.core.containers import Scene
from autofdtd.core.models import AutoFDTDModel
from autofdtd.core.validation import normalize_vec3
from autofdtd.materials import (
    Debye,
    Drude,
    Lorentz,
    Medium,
    PECMedium,
    PMCMedium,
    PoleResidue,
    Sellmeier,
    medium_model_from_value,
)
from autofdtd.materials.dispersive import ComplexPair, complex_from_pair, complex_to_pair

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6


class ConstitutiveMode(StrEnum):
    """Runtime behavior for a constitutive update branch."""

    STANDARD = "standard"
    CLAMP_ZERO = "clamp_zero"


class IsotropicMaterialCoefficients(AutoFDTDModel):
    """Scalar constitutive coefficients for one homogeneous isotropic material."""

    type: str = "IsotropicMaterialCoefficients"
    medium_type: str
    permittivity: float | None = None
    conductivity: float | None = None
    permeability: float | None = None
    magnetic_conductivity: float | None = None
    electric_mode: ConstitutiveMode
    magnetic_mode: ConstitutiveMode
    electric_decay: float
    electric_drive: float
    magnetic_decay: float
    magnetic_drive: float


class PoleResidueTermCoefficients(AutoFDTDModel):
    """Per-pole auxiliary-state recurrence coefficients."""

    type: str = "PoleResidueTermCoefficients"
    pole: ComplexPair
    residue: ComplexPair
    decay: ComplexPair
    drive: ComplexPair


class PoleResidueMaterialCoefficients(AutoFDTDModel):
    """Compiled dispersive coefficients plus baseline electric and magnetic updates."""

    type: str = "PoleResidueMaterialCoefficients"
    medium_type: str = "PoleResidue"
    eps_inf: float
    electric_mode: ConstitutiveMode = ConstitutiveMode.STANDARD
    magnetic_mode: ConstitutiveMode = ConstitutiveMode.STANDARD
    electric_decay: float
    electric_drive: float
    magnetic_decay: float
    magnetic_drive: float
    poles: tuple[PoleResidueTermCoefficients, ...]
    auxiliary_layout: str = "complex_polarization_per_pole"

    @property
    def num_poles(self) -> int:
        return len(self.poles)


MaterialCoefficients = IsotropicMaterialCoefficients | PoleResidueMaterialCoefficients


class SceneMaterialSample(AutoFDTDModel):
    """Resolved scene material assignment for one sample point."""

    type: str = "SceneMaterialSample"
    point: tuple[float, float, float]
    structure_name: str | None = None
    medium_type: str
    medium: (
        Medium
        | PECMedium
        | PMCMedium
        | PoleResidue
        | Sellmeier
        | Lorentz
        | Drude
        | Debye
        | MaterialCoefficients
        | dict[str, object]
    )
    precedence_rank: int | None = None

    @field_validator("point")
    @classmethod
    def _validate_point(
        cls, value: Sequence[float] | tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return normalize_vec3(value, field_name="point")


def _compile_nondispersive_coefficients(
    *,
    dt: float,
    permittivity: float,
    conductivity: float,
    permeability: float,
    magnetic_conductivity: float,
    medium_type: str,
) -> IsotropicMaterialCoefficients:
    electric_ratio = conductivity * dt / (2.0 * EPSILON_0 * permittivity)
    magnetic_ratio = magnetic_conductivity * dt / (2.0 * MU_0 * permeability)
    electric_scale = 1.0 / (1.0 + electric_ratio)
    magnetic_scale = 1.0 / (1.0 + magnetic_ratio)
    return IsotropicMaterialCoefficients(
        medium_type=medium_type,
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        magnetic_conductivity=magnetic_conductivity,
        electric_mode=ConstitutiveMode.STANDARD,
        magnetic_mode=ConstitutiveMode.STANDARD,
        electric_decay=(1.0 - electric_ratio) * electric_scale,
        electric_drive=(dt / (EPSILON_0 * permittivity)) * electric_scale,
        magnetic_decay=(1.0 - magnetic_ratio) * magnetic_scale,
        magnetic_drive=(dt / (MU_0 * permeability)) * magnetic_scale,
    )


def compile_isotropic_medium_coefficients(
    medium: object,
    *,
    dt: float,
) -> IsotropicMaterialCoefficients:
    """Compile a supported non-dispersive medium into scalar constitutive coefficients."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if isinstance(normalized, Medium):
        return _compile_nondispersive_coefficients(
            dt=dt,
            permittivity=normalized.permittivity,
            conductivity=normalized.conductivity,
            permeability=normalized.permeability,
            magnetic_conductivity=normalized.magnetic_conductivity,
            medium_type=normalized.type,
        )
    if isinstance(normalized, PECMedium):
        return IsotropicMaterialCoefficients(
            medium_type=normalized.type,
            electric_mode=ConstitutiveMode.CLAMP_ZERO,
            magnetic_mode=ConstitutiveMode.STANDARD,
            electric_decay=0.0,
            electric_drive=0.0,
            magnetic_decay=1.0,
            magnetic_drive=dt / MU_0,
        )
    if isinstance(normalized, PMCMedium):
        return IsotropicMaterialCoefficients(
            medium_type=normalized.type,
            electric_mode=ConstitutiveMode.STANDARD,
            magnetic_mode=ConstitutiveMode.CLAMP_ZERO,
            electric_decay=1.0,
            electric_drive=dt / EPSILON_0,
            magnetic_decay=0.0,
            magnetic_drive=0.0,
        )
    raise TypeError(f"unsupported non-dispersive medium type {type(normalized)!r}")


def compile_pole_residue_coefficients(
    medium: object,
    *,
    dt: float,
) -> PoleResidueMaterialCoefficients:
    """Compile a PoleResidue medium into auxiliary-state recurrence coefficients."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if not isinstance(normalized, PoleResidue):
        raise TypeError(f"expected PoleResidue medium, got {type(normalized)!r}")

    background = compile_isotropic_medium_coefficients(
        Medium(permittivity=normalized.eps_inf),
        dt=dt,
    )
    terms = []
    try:
        recurrence_terms = normalized.pole_response_factors(dt=dt)
    except OverflowError as exc:
        raise ValueError(
            "dispersive poles are too stiff for the current timestep on the Phase 1 exact-hold path"
        ) from exc

    for (pole_pair, residue_pair), (decay, drive) in zip(
        normalized.poles,
        recurrence_terms,
        strict=True,
    ):
        terms.append(
            PoleResidueTermCoefficients(
                pole=pole_pair,
                residue=residue_pair,
                decay=complex_to_pair(decay),
                drive=complex_to_pair(drive),
            )
        )

    return PoleResidueMaterialCoefficients(
        eps_inf=normalized.eps_inf,
        electric_decay=background.electric_decay,
        electric_drive=background.electric_drive,
        magnetic_decay=background.magnetic_decay,
        magnetic_drive=background.magnetic_drive,
        poles=tuple(terms),
    )


def compile_sellmeier_coefficients(
    medium: object,
    *,
    dt: float,
) -> PoleResidueMaterialCoefficients:
    """Compile a Sellmeier medium onto the shared PoleResidue auxiliary-state path."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if not isinstance(normalized, Sellmeier):
        raise TypeError(f"expected Sellmeier medium, got {type(normalized)!r}")

    compiled = compile_pole_residue_coefficients(normalized.to_pole_residue(), dt=dt)
    return compiled.model_copy(update={"medium_type": normalized.type})


def compile_lorentz_coefficients(
    medium: object,
    *,
    dt: float,
) -> PoleResidueMaterialCoefficients:
    """Compile a Lorentz medium onto the shared PoleResidue auxiliary-state path."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if not isinstance(normalized, Lorentz):
        raise TypeError(f"expected Lorentz medium, got {type(normalized)!r}")

    compiled = compile_pole_residue_coefficients(normalized.to_pole_residue(), dt=dt)
    return compiled.model_copy(update={"medium_type": normalized.type})


def compile_drude_coefficients(
    medium: object,
    *,
    dt: float,
) -> PoleResidueMaterialCoefficients:
    """Compile a Drude medium onto the shared PoleResidue auxiliary-state path."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if not isinstance(normalized, Drude):
        raise TypeError(f"expected Drude medium, got {type(normalized)!r}")

    compiled = compile_pole_residue_coefficients(normalized.to_pole_residue(), dt=dt)
    return compiled.model_copy(update={"medium_type": normalized.type})


def compile_debye_coefficients(
    medium: object,
    *,
    dt: float,
) -> PoleResidueMaterialCoefficients:
    """Compile a Debye medium onto the shared PoleResidue auxiliary-state path."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    normalized = medium_model_from_value(medium)
    if not isinstance(normalized, Debye):
        raise TypeError(f"expected Debye medium, got {type(normalized)!r}")

    compiled = compile_pole_residue_coefficients(normalized.to_pole_residue(), dt=dt)
    return compiled.model_copy(update={"medium_type": normalized.type})


def compile_medium_coefficients(
    medium: object,
    *,
    dt: float,
) -> MaterialCoefficients:
    """Compile a supported Phase 1 medium into runtime-ready coefficients."""

    normalized = medium_model_from_value(medium)
    if isinstance(normalized, PoleResidue):
        return compile_pole_residue_coefficients(normalized, dt=dt)
    if isinstance(normalized, Sellmeier):
        return compile_sellmeier_coefficients(normalized, dt=dt)
    if isinstance(normalized, Lorentz):
        return compile_lorentz_coefficients(normalized, dt=dt)
    if isinstance(normalized, Drude):
        return compile_drude_coefficients(normalized, dt=dt)
    if isinstance(normalized, Debye):
        return compile_debye_coefficients(normalized, dt=dt)
    return compile_isotropic_medium_coefficients(normalized, dt=dt)


def sample_scene_mediums(
    scene: Scene,
    points: Sequence[Sequence[float]],
) -> tuple[SceneMaterialSample, ...]:
    """Resolve the scene material assignment for each sample point."""

    samples: list[SceneMaterialSample] = []
    for point in points:
        normalized_point = normalize_vec3(point, field_name="point")
        stack = scene.resolve_structure_stack_at_point(normalized_point)
        if stack:
            winner = stack[-1]
            medium = winner.structure.medium
            medium_type = medium_model_from_value(medium).type
            samples.append(
                SceneMaterialSample(
                    point=normalized_point,
                    structure_name=winner.structure.name,
                    medium_type=medium_type,
                    medium=medium,
                    precedence_rank=winner.precedence_rank,
                )
            )
            continue

        background = scene.medium
        medium_type = medium_model_from_value(background).type
        samples.append(
            SceneMaterialSample(
                point=normalized_point,
                medium_type=medium_type,
                medium=background,
            )
        )
    return tuple(samples)


def compile_scene_medium_coefficients(
    scene: Scene,
    *,
    points: Sequence[Sequence[float]],
    dt: float,
) -> tuple[SceneMaterialSample, ...]:
    """Resolve scene materials and attach compiled coefficients per sample point."""

    resolved = sample_scene_mediums(scene, points)
    return tuple(
        sample.copy_update(
            medium=compile_medium_coefficients(sample.medium, dt=dt),
        )
        for sample in resolved
    )


def pole_residue_terms_as_complex(
    coefficients: PoleResidueMaterialCoefficients,
) -> tuple[tuple[complex, complex], ...]:
    """Return compiled recurrence terms in complex form for runtime helpers."""

    return tuple(
        (
            complex_from_pair(term.decay),
            complex_from_pair(term.drive),
        )
        for term in coefficients.poles
    )
