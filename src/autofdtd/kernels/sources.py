"""Source injection helpers and backend conventions for Phase 1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from autofdtd.compiler.sources import (
    CompiledAstigmaticGaussianBeam,
    CompiledCustomCurrentSource,
    CompiledCustomFieldSource,
    CompiledGaussianBeam,
    CompiledModeSource,
    CompiledPlaneWave,
    CompiledPointDipole,
    CompiledTFSF,
    CompiledUniformCurrentSource,
)

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


def uniform_current_source_kernel_metadata() -> dict[str, Any]:
    """Expose the current-source kernel backend choices for diagnostics."""

    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("source_injection",),
    }


def uniform_current_density(
    compiled_source: CompiledUniformCurrentSource,
    *,
    shape: tuple[int, int, int],
    time: float,
) -> np.ndarray:
    """Return the spatial current-density term on the primal-cell grid."""

    density = np.zeros((*shape, 3), dtype=np.complex128)
    amplitude = compiled_source.amplitude_at_time(time)
    axis = compiled_source.component_axis
    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        density[placement][axis] += amplitude * weight
    return density


def inject_uniform_current_source(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledUniformCurrentSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled uniform current source to the supplied field buffers.

    Phase 1 keeps source injection explicit and staged: electric current adds to ``E``,
    magnetic current adds to ``H``. Full constitutive scaling remains the responsibility of
    the future stepper.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)
    target = updated_electric if compiled_source.field_kind == "electric" else updated_magnetic
    axis = compiled_source.component_axis

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        target[placement][axis] += amplitude * weight

    return updated_electric, updated_magnetic


def inject_point_dipole(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledPointDipole,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled point dipole source to the supplied field buffers.

    PointDipole uses density-based amplitude interpretation and interpolates
    across neighboring cells when interpolate=True.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)
    target = updated_electric if compiled_source.field_kind == "electric" else updated_magnetic
    axis = compiled_source.component_axis

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        target[placement][axis] += amplitude * weight

    return updated_electric, updated_magnetic


def inject_custom_current_source(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledCustomCurrentSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled custom current source to the supplied field buffers.

    CustomCurrentSource can inject both electric and magnetic field components
    based on the provided field data arrays.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Apply electric field components
    if compiled_source.has_electric and compiled_source.e_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.e_field_data.items():
                if field_key in ("Ex", "Ey", "Ez"):
                    axis = "xyz".index(field_key[1].lower())
                    # Apply field data - simplified version using index mapping
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        updated_electric[placement][axis] += amplitude * weight * field_values[idx]

    # Apply magnetic field components
    if compiled_source.has_magnetic and compiled_source.h_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.h_field_data.items():
                if field_key in ("Hx", "Hy", "Hz"):
                    axis = "xyz".index(field_key[1].lower())
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        updated_magnetic[placement][axis] += amplitude * weight * field_values[idx]

    return updated_electric, updated_magnetic


def inject_custom_field_source(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledCustomFieldSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled custom field source to the supplied field buffers.

    CustomFieldSource uses the equivalence principle on a planar surface.
    For tangential field components provided:
    - Electric field components (Ex, Ey on xy-plane) contribute via M = -n × E
    - Magnetic field components (Hx, Hy on xy-plane) contribute via J = n × H

    The direction sign determines whether the source injects forward (+)
    or backward (-) relative to the injection axis normal.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward, -1 for backward
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0

    # Injection axis determines the normal direction for equivalence principle
    injection_axis = compiled_source.injection_axis

    # Apply electric field components via equivalence principle: M = -n × E
    # For a planar source, tangential E fields contribute to H update
    if compiled_source.has_electric and compiled_source.e_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.e_field_data.items():
                if field_key in ("Ex", "Ey", "Ez"):
                    axis = "xyz".index(field_key[1].lower())
                    # Skip the injection axis component (normal component is not used in equivalence)
                    if axis == injection_axis:
                        continue
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        # Electric fields contribute to magnetic field update via M = -n × E
                        # This injects into H along the injection axis
                        updated_magnetic[placement][injection_axis] += (
                            amplitude * weight * direction_sign * field_values[idx]
                        )

    # Apply magnetic field components via equivalence principle: J = n × H
    # For a planar source, tangential H fields contribute to E update
    if compiled_source.has_magnetic and compiled_source.h_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.h_field_data.items():
                if field_key in ("Hx", "Hy", "Hz"):
                    axis = "xyz".index(field_key[1].lower())
                    # Skip the injection axis component (normal component is not used in equivalence)
                    if axis == injection_axis:
                        continue
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        # Magnetic fields contribute to electric field update via J = n × H
                        # This injects into E along the injection axis
                        updated_electric[placement][injection_axis] += (
                            amplitude * weight * direction_sign * field_values[idx]
                        )

    return updated_electric, updated_magnetic


def inject_mode_source(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledModeSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled mode source to the supplied field buffers.

    ModeSource uses the equivalence principle: tangential E and H field
    components from the mode profile are converted to J and M currents.
    For direction="+", H fields contribute to E injection and E fields
    contribute to H injection (forward propagation). For direction="-",
    the reverse holds (backward propagation).

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        compiled_source: The compiled mode source with field data
        time: Current simulation time
        dt: Timestep size

    Returns:
        Updated (electric_field, magnetic_field) buffers
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward (+), -1 for backward (-)
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    e_data = compiled_source.e_field_data
    h_data = compiled_source.h_field_data
    x_coords = compiled_source.x_coords
    y_coords = compiled_source.y_coords

    # Get the tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    # Normalize mode fields for injection (power normalization)
    power_norm = 1.0 / math.sqrt(compiled_source.mode_power) if compiled_source.mode_power > 0 else 1.0

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        # Get the local grid position
        idx_x, idx_y = placement[tang_axis_a], placement[tang_axis_b]

        # Get mode field values at this position (interpolated from mode solver grid)
        # The mode fields are defined on the cross-sectional grid and are complex
        ex_val = e_data["Ex"][idx_x, idx_y] if "Ex" in e_data else 0.0
        ey_val = e_data["Ey"][idx_x, idx_y] if "Ey" in e_data else 0.0
        hx_val = h_data["Hx"][idx_x, idx_y] if "Hx" in h_data else 0.0
        hy_val = h_data["Hy"][idx_x, idx_y] if "Hy" in h_data else 0.0

        # Apply equivalence principle for mode injection
        # For + direction: tangential H contributes to E, tangential E contributes to H
        # M = -n × E (electric fields contribute to magnetic current)
        # J = n × H (magnetic fields contribute to electric current)

        # For a planar source in xy-plane (z-injection):
        # Jx = n_y * Hz - n_z * Hy = -Hy (for nz=1, ny=0)
        # Jy = n_z * Hx - n_x * Hz = Hx (for nz=1, nx=0)
        # Mx = -n_y * Ez + n_z * Ey = Ey (for nz=1, ny=0)
        # My = -n_z * Ex + n_x * Ez = -Ex (for nz=1, nx=0)

        if direction_sign > 0:
            # Forward propagation: H fields inject into E, E fields inject into H
            # Jy = Hx, Jx = -Hy for z-injection
            # My = -Ex, Mx = Ey for z-injection
            updated_electric[placement][tang_axis_a] += amplitude * weight * power_norm * (-hy_val)
            updated_electric[placement][tang_axis_b] += amplitude * weight * power_norm * hx_val
            updated_magnetic[placement][tang_axis_a] += amplitude * weight * power_norm * ey_val
            updated_magnetic[placement][tang_axis_b] += amplitude * weight * power_norm * (-ex_val)
        else:
            # Backward propagation: E fields inject into H, H fields inject into E
            updated_electric[placement][tang_axis_a] += amplitude * weight * power_norm * hy_val
            updated_electric[placement][tang_axis_b] += amplitude * weight * power_norm * (-hx_val)
            updated_magnetic[placement][tang_axis_a] += amplitude * weight * power_norm * (-ey_val)
            updated_magnetic[placement][tang_axis_b] += amplitude * weight * power_norm * ex_val

    return updated_electric, updated_magnetic


def inject_plane_wave(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledPlaneWave,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled plane wave source to the supplied field buffers.

    A PlaneWave injects a spatially uniform electromagnetic wave with a defined
    propagation direction and polarization. The source uses the equivalence
    principle: J = n × H and M = -n × E, where n is the source normal.

    For a plane wave, the fields are uniform across the planar injection surface
    and determined by the k-vector and polarization.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        compiled_source: The compiled plane wave source
        time: Current simulation time
        dt: Timestep size
        freq: Frequency for k-vector evaluation (for FixedInPlaneKSpec)

    Returns:
        Updated (electric_field, magnetic_field) buffers
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward (+), -1 for backward (-)
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    # Get tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    # For a plane wave, the tangential field components are uniform across
    # the source plane. We compute the E and H fields based on the
    # polarization and direction vectors.

    # The plane wave fields in the source coordinate system are:
    # E = E0 * pol_vector * exp(i * (k · r - ωt))
    # H = H0 * (k_hat × pol_vector) * exp(i * (k · r - ωt))
    #
    # For injection via equivalence principle:
    # - The tangential H fields contribute to E via J = n × H
    # - The tangential E fields contribute to H via M = -n × E
    #
    # Since the plane wave is spatially uniform on the injection plane,
    # we evaluate at a reference point (center of the source plane).

    # Get the source amplitude at this time
    amplitude = compiled_source.amplitude_at_time(time) * dt

    # Get the direction and polarization vectors
    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    # Normalize direction
    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    # Polarization direction
    pol_hat = pol_vec

    # For plane wave injection on a planar surface, we inject at all
    # placements with uniform weight. The plane wave fields are computed
    # based on the k-vector direction and polarization.

    # The tangential components of E and H that contribute to injection
    # are determined by the cross products with the injection normal.

    # For n = (0, 0, 1) (z-injection):
    # J = n × H = (Hx, Hy, 0) → contributes to (Ez,)
    # M = -n × E = (-Ex, -Ey, 0) → contributes to (Hz,)
    #
    # More generally, we need to compute the tangential components of
    # E and H that are perpendicular to the injection direction.

    # The E field at the source plane is along the polarization direction
    # The H field is along k_hat × pol_hat (perpendicular to both)

    # Compute H direction as cross product of direction and polarization
    # H_dir = k_hat × pol_hat
    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    # The magnitude ratio between E and H in a plane wave is the impedance
    # For vacuum: Z0 = sqrt(μ0/ε0) ≈ 377 ohms
    # In terms of normalized units used in FDTD: E/H = Z0
    z0_normalized = 1.0  # In our normalized units where c=1

    # For a uniform plane wave, the field amplitude is uniform across the plane.
    # The injection injects the equivalent currents at each cell.

    # For each placement, inject the plane wave fields via equivalence principle
    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        # E field at this position (along polarization)
        e_tang_a = pol_hat[tang_axis_a] * amplitude * weight
        e_tang_b = pol_hat[tang_axis_b] * amplitude * weight

        # H field at this position (perpendicular to both k and pol)
        h_tang_a = h_dir[tang_axis_a] * amplitude * weight / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude * weight / z0_normalized

        # Apply via equivalence principle:
        # J = n × H → contributes to E along injection axis
        # M = -n × E → contributes to H along injection axis

        # For + direction: tangential H contributes to E
        # Jy = Hx, Jx = -Hy for z-injection
        # For general injection axis:
        # The current density J = n × H gives components along tang_axes

        if direction_sign > 0:
            # Forward: H tangential contributes to E via J = n × H
            # E is updated along injection axis from H tangential
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            # M = -n × E gives H update from E tangential
            # H is updated along injection axis from E tangential
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            # Backward: E tangential contributes to H via M = -n × E
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            # H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    return updated_electric, updated_magnetic


def inject_gaussian_beam(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledGaussianBeam,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled Gaussian beam source to the supplied field buffers.

    A GaussianBeam injects a spatially Gaussian-shaped electromagnetic wave with
    a defined propagation direction, polarization, and waist parameters.
    The source uses the equivalence principle: J = n × H and M = -n × E.

    The Gaussian envelope determines the amplitude profile across the beam
    cross-section, with peak amplitude at the center and falling off radially.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        compiled_source: The compiled Gaussian beam source
        time: Current simulation time
        dt: Timestep size
        freq: Frequency for k-vector evaluation (for FixedInPlaneKSpec)

    Returns:
        Updated (electric_field, magnetic_field) buffers
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward (+), -1 for backward (-)
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    # Get tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    # Get the source amplitude at this time
    amplitude = compiled_source.amplitude_at_time(time) * dt

    # Get the direction and polarization vectors
    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    # Normalize direction
    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    # Polarization direction
    pol_hat = pol_vec

    # Compute H direction as cross product of direction and polarization
    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    # For vacuum impedance in normalized units
    z0_normalized = 1.0

    # For a Gaussian beam, the field amplitude varies across the beam cross-section
    # according to the Gaussian envelope. The beam_weights already include this
    # spatial variation.

    # For each placement, inject the Gaussian beam fields via equivalence principle
    for placement, placement_weight, beam_weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        compiled_source.beam_weights,
        strict=True,
    ):
        # Combined weight: placement_weight (grid interpolation) * beam_weight (Gaussian envelope)
        combined_weight = placement_weight * beam_weight

        # E field at this position (along polarization)
        e_tang_a = pol_hat[tang_axis_a] * amplitude * combined_weight
        e_tang_b = pol_hat[tang_axis_b] * amplitude * combined_weight

        # H field at this position (perpendicular to both k and pol)
        h_tang_a = h_dir[tang_axis_a] * amplitude * combined_weight / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude * combined_weight / z0_normalized

        # Apply via equivalence principle:
        # J = n × H → contributes to E along injection axis
        # M = -n × E → contributes to H along injection axis

        if direction_sign > 0:
            # Forward: H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            # M = -n × E gives H update from E tangential
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            # Backward: E tangential contributes to H via M = -n × E
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            # H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    return updated_electric, updated_magnetic


def inject_astigmatic_gaussian_beam(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledAstigmaticGaussianBeam,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled astigmatic Gaussian beam source to the supplied field buffers.

    An AstigmaticGaussianBeam injects a spatially Gaussian-shaped electromagnetic wave
    with separate waist radii in x and y directions. The source uses the equivalence
    principle: J = n × H and M = -n × E.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        compiled_source: The compiled astigmatic Gaussian beam source
        time: Current simulation time
        dt: Timestep size
        freq: Frequency for k-vector evaluation (for FixedInPlaneKSpec)

    Returns:
        Updated (electric_field, magnetic_field) buffers
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward (+), -1 for backward (-)
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    # Get tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    # Get the source amplitude at this time
    amplitude = compiled_source.amplitude_at_time(time) * dt

    # Get the direction and polarization vectors
    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    # Normalize direction
    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    # Polarization direction
    pol_hat = pol_vec

    # Compute H direction as cross product of direction and polarization
    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    # For vacuum impedance in normalized units
    z0_normalized = 1.0

    # For an astigmatic Gaussian beam, the field amplitude varies with
    # separate x and y envelope parameters.

    # For each placement, inject the beam fields via equivalence principle
    for placement, placement_weight, beam_weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        compiled_source.beam_weights,
        strict=True,
    ):
        # Combined weight: placement_weight (grid interpolation) * beam_weight (Gaussian envelope)
        combined_weight = placement_weight * beam_weight

        # E field at this position (along polarization)
        e_tang_a = pol_hat[tang_axis_a] * amplitude * combined_weight
        e_tang_b = pol_hat[tang_axis_b] * amplitude * combined_weight

        # H field at this position (perpendicular to both k and pol)
        h_tang_a = h_dir[tang_axis_a] * amplitude * combined_weight / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude * combined_weight / z0_normalized

        # Apply via equivalence principle:
        # J = n × H → contributes to E along injection axis
        # M = -n × E → contributes to H along injection axis

        if direction_sign > 0:
            # Forward: H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            # M = -n × E gives H update from E tangential
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            # Backward: E tangential contributes to H via M = -n × E
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            # H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    return updated_electric, updated_magnetic


def inject_tfsf(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    compiled_source: CompiledTFSF,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compiled TFSF source to the supplied field buffers.

    A TFSF (Total-Field Scattered-Field) source injects a plane wave in a finite
    region of the simulation domain. The incident field is injected at the
    injection plane, and the TFSF boundary interaction subtracts the incident
    field at the box edges to maintain the total-field/scattered-field separation.

    The TFSF source works as follows:
    1. The incident plane wave is injected at the injection plane using the
       equivalence principle (same as a plane wave source)
    2. At the six faces of the TFSF box, the incident field is subtracted
       from the total field to produce the scattered field outside the box

    For normal incidence, the injection is equivalent to a plane wave. For
    oblique incidence, the field is projected onto the injection axis.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        compiled_source: The compiled TFSF source
        time: Current simulation time
        dt: Timestep size

    Returns:
        Updated (electric_field, magnetic_field) buffers
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)
    if electric.shape != magnetic.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric.ndim != 4 or electric.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric.astype(np.result_type(electric.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic.astype(np.result_type(magnetic.dtype, np.complex128), copy=True)

    # Direction sign: +1 for forward (+), -1 for backward (-)
    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    # Get tangential axes
    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    # Get the source amplitude at this time
    amplitude = compiled_source.amplitude_at_time(time) * dt

    # Get the direction and polarization vectors
    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    # Normalize direction
    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    # Polarization direction
    pol_hat = pol_vec

    # Compute H direction as cross product of direction and polarization
    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    # For vacuum impedance in normalized units
    z0_normalized = 1.0

    # TFSF injection has two components:
    # 1. Inject the incident field at the injection plane (forward face)
    # 2. Apply the TF/SF boundary correction at all six faces

    # For Phase 1, we implement the injection component only.
    # The full TFSF boundary correction (subtracting incident field at the
    # box edges) requires additional tracking of the incident field and
    # is deferred to a later task that implements the full TFSF boundary stage.

    # The injection plane is at one face of the TFSF box. For direction="+",
    # the injection plane is at the negative face of the box along the
    # injection axis. For direction="-", it's at the positive face.

    # Get TFSF bounds indices
    (imin, jmin, kmin), (imax, jmax, kmax) = compiled_source.tfsf_bounds_indices

    # For each placement in the TFSF volume, inject the plane wave fields
    # via equivalence principle (same as plane wave injection)
    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        # E field at this position (along polarization)
        e_tang_a = pol_hat[tang_axis_a] * amplitude * weight
        e_tang_b = pol_hat[tang_axis_b] * amplitude * weight

        # H field at this position (perpendicular to both k and pol)
        h_tang_a = h_dir[tang_axis_a] * amplitude * weight / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude * weight / z0_normalized

        # Apply via equivalence principle:
        # J = n × H → contributes to E along tangential axes
        # M = -n × E → contributes to H along tangential axes

        if direction_sign > 0:
            # Forward: H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            # M = -n × E gives H update from E tangential
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            # Backward: E tangential contributes to H via M = -n × E
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            # H tangential contributes to E via J = n × H
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    return updated_electric, updated_magnetic


# ---------------------------------------------------------------------------
# Unified source injection stage kernel
# ---------------------------------------------------------------------------


def inject_sources_stage(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    uniform_current_sources: tuple[CompiledUniformCurrentSource, ...] = (),
    point_dipole_sources: tuple[CompiledPointDipole, ...] = (),
    custom_current_sources: tuple[CompiledCustomCurrentSource, ...] = (),
    custom_field_sources: tuple[CompiledCustomFieldSource, ...] = (),
    mode_sources: tuple[CompiledModeSource, ...] = (),
    plane_wave_sources: tuple[CompiledPlaneWave, ...] = (),
    gaussian_beam_sources: tuple[CompiledGaussianBeam, ...] = (),
    astigmatic_gaussian_beam_sources: tuple[CompiledAstigmaticGaussianBeam, ...] = (),
    tfsf_sources: tuple[CompiledTFSF, ...] = (),
    time: float = 0.0,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject all source contributions in a single kernel call.

    This is the kernel-level unified source injection function that applies
    all source types to the field buffers. It provides the same functionality
    as the runtime-level `apply_source_injection_stage` but is intended for
    use in kernel composition.

    Scheduling
    ---------
    This stage runs between electric_update and magnetic_update::

        electric_update → source_injection → magnetic_update

    Parameters
    ----------
    electric_field : np.ndarray
        E field buffer with shape (nx, ny, nz, 3).
    magnetic_field : np.ndarray
        H field buffer with shape (nx, ny, nz, 3).
    uniform_current_sources : tuple[CompiledUniformCurrentSource, ...]
    point_dipole_sources : tuple[CompiledPointDipole, ...]
    custom_current_sources : tuple[CompiledCustomCurrentSource, ...]
    custom_field_sources : tuple[CompiledCustomFieldSource, ...]
    mode_sources : tuple[CompiledModeSource, ...]
    plane_wave_sources : tuple[CompiledPlaneWave, ...]
    gaussian_beam_sources : tuple[CompiledGaussianBeam, ...]
    astigmatic_gaussian_beam_sources : tuple[CompiledAstigmaticGaussianBeam, ...]
    tfsf_sources : tuple[CompiledTFSF, ...]
    time : float
        Current simulation time.
    dt : float
        Timestep size.
    freq : float, optional
        Frequency for frequency-dependent sources.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (electric_field, magnetic_field) buffers.
    """
    electric = np.asarray(electric_field)
    magnetic = np.asarray(magnetic_field)

    # Uniform current sources
    for source in uniform_current_sources:
        electric, magnetic = inject_uniform_current_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Point dipole sources
    for source in point_dipole_sources:
        electric, magnetic = inject_point_dipole(
            electric, magnetic, source, time=time, dt=dt
        )

    # Custom current sources
    for source in custom_current_sources:
        electric, magnetic = inject_custom_current_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Custom field sources
    for source in custom_field_sources:
        electric, magnetic = inject_custom_field_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Mode sources
    for source in mode_sources:
        electric, magnetic = inject_mode_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Plane wave sources
    for source in plane_wave_sources:
        electric, magnetic = inject_plane_wave(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # Gaussian beam sources
    for source in gaussian_beam_sources:
        electric, magnetic = inject_gaussian_beam(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # Astigmatic Gaussian beam sources
    for source in astigmatic_gaussian_beam_sources:
        electric, magnetic = inject_astigmatic_gaussian_beam(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # TFSF sources
    for source in tfsf_sources:
        electric, magnetic = inject_tfsf(
            electric, magnetic, source, time=time, dt=dt
        )

    return electric, magnetic
