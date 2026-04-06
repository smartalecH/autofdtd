"""Constitutive update helpers for Phase 1 material paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autofdtd.compiler.materials import (
    AnisotropicMaterialCoefficients,
    ConstitutiveMode,
    IsotropicMaterialCoefficients,
    PoleResidueMaterialCoefficients,
    pole_residue_terms_as_complex,
)

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


@dataclass(frozen=True)
class PoleResidueAuxiliaryState:
    """Complex per-pole polarization state stored alongside electric fields."""

    polarization: np.ndarray

    @property
    def num_poles(self) -> int:
        return int(self.polarization.shape[0])


@dataclass(frozen=True)
class AnisotropicAuxiliaryState:
    """Per-axis auxiliary state for diagonal anisotropic updates."""

    xx: PoleResidueAuxiliaryState | None = None
    yy: PoleResidueAuxiliaryState | None = None
    zz: PoleResidueAuxiliaryState | None = None


def warp_backend_available() -> bool:
    """Return whether the optional Warp backend is importable."""

    return WARP_AVAILABLE


def electric_constitutive_update(
    electric_field: np.ndarray | list[float] | tuple[float, ...],
    curl_h: np.ndarray | list[float] | tuple[float, ...],
    coefficients: IsotropicMaterialCoefficients,
) -> np.ndarray:
    """Apply the electric-field constitutive update for one homogeneous material."""

    electric = np.asarray(electric_field, dtype=np.float64)
    curl = np.asarray(curl_h, dtype=np.float64)
    if electric.shape != curl.shape:
        raise ValueError("electric_field and curl_h must have matching shapes")
    if coefficients.electric_mode is ConstitutiveMode.CLAMP_ZERO:
        return np.zeros_like(electric)
    return coefficients.electric_decay * electric + coefficients.electric_drive * curl


def magnetic_constitutive_update(
    magnetic_field: np.ndarray | list[float] | tuple[float, ...],
    curl_e: np.ndarray | list[float] | tuple[float, ...],
    coefficients: IsotropicMaterialCoefficients,
) -> np.ndarray:
    """Apply the magnetic-field constitutive update for one homogeneous material."""

    magnetic = np.asarray(magnetic_field, dtype=np.float64)
    curl = np.asarray(curl_e, dtype=np.float64)
    if magnetic.shape != curl.shape:
        raise ValueError("magnetic_field and curl_e must have matching shapes")
    if coefficients.magnetic_mode is ConstitutiveMode.CLAMP_ZERO:
        return np.zeros_like(magnetic)
    return coefficients.magnetic_decay * magnetic - coefficients.magnetic_drive * curl


def allocate_pole_residue_state(
    coefficients: PoleResidueMaterialCoefficients,
    *,
    field_shape: tuple[int, ...],
    dtype: np.dtype[np.complexfloating] = np.complex128,
) -> PoleResidueAuxiliaryState:
    """Allocate zero-initialized auxiliary polarization state for PoleResidue-family media."""

    return PoleResidueAuxiliaryState(
        polarization=np.zeros((coefficients.num_poles, *field_shape), dtype=dtype)
    )


def pole_residue_electric_update(
    electric_field: np.ndarray | list[float] | tuple[float, ...],
    curl_h: np.ndarray | list[float] | tuple[float, ...],
    coefficients: PoleResidueMaterialCoefficients,
    state: PoleResidueAuxiliaryState,
    *,
    dt: float,
) -> tuple[np.ndarray, PoleResidueAuxiliaryState, np.ndarray]:
    """Apply a staged PoleResidue-family update and return the updated field and state.

    The Phase 1 runtime keeps a complex polarization accumulator per pole and subtracts the
    resulting polarization-current estimate from the baseline epsilon-infinity electric update.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field, dtype=np.float64)
    curl = np.asarray(curl_h, dtype=np.float64)
    if electric.shape != curl.shape:
        raise ValueError("electric_field and curl_h must have matching shapes")
    if state.polarization.shape != (coefficients.num_poles, *electric.shape):
        raise ValueError("PoleResidue auxiliary state shape does not match coefficients and field")

    baseline = (
        coefficients.electric_decay * electric + coefficients.electric_drive * curl
        if coefficients.electric_mode is not ConstitutiveMode.CLAMP_ZERO
        else np.zeros_like(electric)
    )
    next_polarization = np.empty_like(state.polarization)
    polarization_current = np.zeros_like(electric, dtype=np.float64)

    for index, (decay, drive) in enumerate(pole_residue_terms_as_complex(coefficients)):
        prev = state.polarization[index]
        updated = decay * prev + drive * electric
        next_polarization[index] = updated
        polarization_current += 2.0 * np.real((updated - prev) / dt)

    updated_electric = baseline - coefficients.electric_drive * polarization_current
    return (
        updated_electric,
        PoleResidueAuxiliaryState(polarization=next_polarization),
        polarization_current,
    )


def allocate_anisotropic_state(
    coefficients: AnisotropicMaterialCoefficients,
    *,
    field_shape: tuple[int, ...],
    dtype: np.dtype[np.complexfloating] = np.complex128,
) -> AnisotropicAuxiliaryState:
    """Allocate per-axis auxiliary state for diagonal anisotropic media."""

    states: list[PoleResidueAuxiliaryState | None] = []
    for component in coefficients.components:
        if isinstance(component, PoleResidueMaterialCoefficients):
            states.append(allocate_pole_residue_state(component, field_shape=field_shape, dtype=dtype))
        else:
            states.append(None)
    return AnisotropicAuxiliaryState(xx=states[0], yy=states[1], zz=states[2])


def anisotropic_electric_update(
    electric_field: np.ndarray | list[float] | tuple[float, ...],
    curl_h: np.ndarray | list[float] | tuple[float, ...],
    coefficients: AnisotropicMaterialCoefficients,
    state: AnisotropicAuxiliaryState | None = None,
    *,
    dt: float,
) -> tuple[np.ndarray, AnisotropicAuxiliaryState, np.ndarray]:
    """Apply a diagonal anisotropic electric update axis-by-axis.

    The input arrays must have a final axis of length 3 corresponding to ``Ex, Ey, Ez``.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field, dtype=np.float64)
    curl = np.asarray(curl_h, dtype=np.float64)
    if electric.shape != curl.shape:
        raise ValueError("electric_field and curl_h must have matching shapes")
    if electric.ndim == 0 or electric.shape[-1] != 3:
        raise ValueError("anisotropic electric updates require a trailing axis of length 3")

    working_state = state or allocate_anisotropic_state(coefficients, field_shape=electric.shape[:-1])
    next_components: list[np.ndarray] = []
    next_states: list[PoleResidueAuxiliaryState | None] = []
    polarization_components: list[np.ndarray] = []

    for axis, (component_coeffs, axis_state) in enumerate(
        zip(coefficients.components, (working_state.xx, working_state.yy, working_state.zz), strict=True)
    ):
        field_axis = electric[..., axis]
        curl_axis = curl[..., axis]
        if isinstance(component_coeffs, PoleResidueMaterialCoefficients):
            if axis_state is None:
                raise ValueError("anisotropic auxiliary state is missing a dispersive axis state")
            updated_axis, next_axis_state, polarization_axis = pole_residue_electric_update(
                field_axis,
                curl_axis,
                component_coeffs,
                axis_state,
                dt=dt,
            )
        else:
            updated_axis = electric_constitutive_update(field_axis, curl_axis, component_coeffs)
            next_axis_state = None
            polarization_axis = np.zeros_like(field_axis)
        next_components.append(updated_axis)
        next_states.append(next_axis_state)
        polarization_components.append(polarization_axis)

    return (
        np.stack(next_components, axis=-1),
        AnisotropicAuxiliaryState(xx=next_states[0], yy=next_states[1], zz=next_states[2]),
        np.stack(polarization_components, axis=-1),
    )


def anisotropic_magnetic_update(
    magnetic_field: np.ndarray | list[float] | tuple[float, ...],
    curl_e: np.ndarray | list[float] | tuple[float, ...],
    coefficients: AnisotropicMaterialCoefficients,
) -> np.ndarray:
    """Apply a diagonal anisotropic magnetic update axis-by-axis."""

    magnetic = np.asarray(magnetic_field, dtype=np.float64)
    curl = np.asarray(curl_e, dtype=np.float64)
    if magnetic.shape != curl.shape:
        raise ValueError("magnetic_field and curl_e must have matching shapes")
    if magnetic.ndim == 0 or magnetic.shape[-1] != 3:
        raise ValueError("anisotropic magnetic updates require a trailing axis of length 3")

    updated_components = [
        magnetic_constitutive_update(magnetic[..., axis], curl[..., axis], component_coeffs)
        for axis, component_coeffs in enumerate(coefficients.components)
    ]
    return np.stack(updated_components, axis=-1)


def constitutive_kernel_metadata() -> dict[str, Any]:
    """Expose the current constitutive backend choices for diagnostics."""

    return {
        "backend": "warp" if WARP_AVAILABLE else "numpy",
        "warp_available": WARP_AVAILABLE,
        "stages": (
            "electric_constitutive_update",
            "magnetic_constitutive_update",
            "pole_residue_electric_update",
            "anisotropic_electric_update",
            "anisotropic_magnetic_update",
        ),
        "module_contents_stable": True,
        "auxiliary_layouts": ("complex_polarization_per_pole", "per_axis_shared_scalar_layout"),
        "dispersive_medium_paths": ("PoleResidue", "Sellmeier", "Lorentz", "Drude", "Debye"),
        "anisotropic_medium_paths": ("AnisotropicMedium",),
    }
