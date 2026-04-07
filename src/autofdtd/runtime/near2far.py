"""Near-to-far (N2F) postprocessing runtime integration.

This module provides runtime integration for near-to-far postprocessing
building on the DFT monitor infrastructure. It connects surface DFT data
from projection monitors to the N2F kernel computations.

The workflow is:
1. DFT fields are accumulated at a surface monitor during simulation
2. After simulation, N2F kernels compute surface currents
3. Far-field radiation patterns are computed via the equivalence principle
4. Diffraction orders are extracted for periodic structures
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from autofdtd.kernels.near2far import (
    C0,
    Z0,
    compute_diffraction_orders_n2f,
    compute_surface_currents,
    compute_surface_positions,
    extract_radiation_pattern,
    far_field_power,
    near2far_kernel_metadata,
    near2far_transform,
    project_to_cartesian_plane,
)


def near2far_runtime_metadata() -> dict[str, object]:
    """Expose N2F runtime metadata for diagnostics."""
    return {
        **near2far_kernel_metadata(),
        "equivalence_principle": "J_s = n × H, M_s = -n × E",
        "greens_function": "far-field spherical wave approximation",
    }


def build_near2far_surface(
    monitor_center: tuple[float, float, float],
    monitor_size: tuple[float, float, float],
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
    cell_sizes: tuple[float, float, float],
) -> dict:
    """Build surface metadata for N2F computation.

    Args:
        monitor_center: Center of the N2F surface
        monitor_size: Size of the N2F surface
        placements: Surface cell indices
        normal_axis: Normal axis (0=x, 1=y, 2=z)
        cell_sizes: Physical cell sizes (dx, dy, dz)

    Returns:
        Dictionary with surface metadata
    """
    # Compute cell area for surface weighting
    # For a surface perpendicular to normal_axis, the cell area is the product
    # of the two tangential cell dimensions
    tang1 = 1 if normal_axis != 0 else 1
    tang2 = 2 if normal_axis != 2 else 2
    tang1_size = cell_sizes[tang1]
    tang2_size = cell_sizes[tang2]
    cell_area = tang1_size * tang2_size

    # Compute cell areas for each cell (same for planar surface)
    n_pts = len(placements)
    cell_areas = np.full(n_pts, cell_area)

    # Compute physical positions
    positions = compute_surface_positions(placements, monitor_center, monitor_size, cell_sizes)

    return {
        "positions": positions,
        "cell_areas": cell_areas,
        "normal_axis": normal_axis,
        "cell_sizes": cell_sizes,
    }


def compute_radiation_from_dft(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    surface_info: dict,
    obs_distance: float,
    phi_vals: tuple[float, ...],
    theta_vals: tuple[float, ...],
    freqs: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Compute far-field radiation pattern from DFT data.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        surface_info: Surface metadata from build_near2far_surface
        obs_distance: Observation distance (far field assumption)
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values

    Returns:
        Dictionary with radiation pattern components
    """
    positions = surface_info["positions"]
    cell_areas = surface_info["cell_areas"]
    normal_axis = surface_info["normal_axis"]

    # Step 1: Compute surface currents from DFT fields
    Js, Ms = compute_surface_currents(dft_e, dft_h, (), normal_axis, surface_info["cell_sizes"])

    # Step 2: Compute far-field radiation via equivalence principle
    phi_arr = np.asarray(phi_vals)
    theta_arr = np.asarray(theta_vals)

    e_theta, e_phi = near2far_transform(
        Js, Ms, positions, obs_distance, phi_arr, theta_arr, freqs, cell_areas
    )

    # Step 3: Extract radiation pattern
    pattern = extract_radiation_pattern(e_theta, e_phi, phi_arr, theta_arr, freqs)

    return pattern


def compute_diffraction_from_dft(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    normal_axis: int,
    freqs: tuple[float, ...],
    *,
    period_y: float | None = None,
    period_z: float | None = None,
    medium_eps: complex = 1.0 + 0.0j,
    num_orders: int = 5,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Compute diffraction orders from DFT surface data.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        normal_axis: Surface normal axis
        freqs: Frequency values
        period_y: Grating period along y direction
        period_z: Grating period along z direction
        medium_eps: Relative permittivity of diffraction medium
        num_orders: Number of diffraction orders

    Returns:
        Tuple of (orders, mx, my)
    """
    # Extract placements from dft_e shape
    n_pts = dft_e.shape[0]

    # Create dummy placements for the computation
    # (actual positions would come from surface_info in full implementation)
    placements = tuple((i, 0, 0) for i in range(n_pts))

    return compute_diffraction_orders_n2f(
        dft_e, dft_h, placements, normal_axis, freqs,
        period_y=period_y, period_z=period_z,
        medium_eps=medium_eps, num_orders=num_orders,
    )


def project_far_field_to_cartesian(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_vals: tuple[float, ...],
    theta_vals: tuple[float, ...],
    freqs: tuple[float, ...],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    obs_distance: float = 1e5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project spherical far-field data to a Cartesian plane.

    Args:
        e_theta: E_theta component from radiation pattern
        e_phi: E_phi component from radiation pattern
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values
        x_range: (x_min, x_max, n_x) for observation plane
        y_range: (y_min, y_max, n_y) for observation plane
        obs_distance: Z distance of observation plane

    Returns:
        Tuple of (Ex, Ey, Ez) arrays on Cartesian grid
    """
    x_min, x_max, n_x = x_range
    y_min, y_max, n_y = y_range

    x_obs = np.linspace(x_min, x_max, n_x)
    y_obs = np.linspace(y_min, y_max, n_y)

    return project_to_cartesian_plane(
        e_theta, e_phi,
        np.asarray(phi_vals),
        np.asarray(theta_vals),
        freqs,
        x_obs, y_obs, obs_distance,
    )


def compute_total_radiated_power(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
) -> np.ndarray:
    """Compute total radiated power by integrating radiation pattern.

    The total power is the integral of the Poynting vector over the
    spherical surface:
        P = ∫ S · dA = ∫ (|E|²/2Z₀) r² sinθ dθ dφ

    Args:
        e_theta: E_theta component
        e_phi: E_phi component
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values

    Returns:
        Total power at each frequency
    """
    power_density = far_field_power(e_theta, e_phi, freqs)

    # Integrate over spherical surface: ∫ f(θ, φ) sinθ dθ dφ
    theta_arr = np.asarray(theta_vals)
    phi_arr = np.asarray(phi_vals)

    d_theta = theta_arr[1] - theta_arr[0] if len(theta_arr) > 1 else 0.0
    d_phi = phi_arr[1] - phi_arr[0] if len(phi_arr) > 1 else 0.0

    # Create sinθ weighting
    sin_theta = np.sin(theta_arr)
    sin_theta = sin_theta[:, np.newaxis]  # Broadcast for phi dimension

    # Integrate
    total_power = np.sum(power_density * sin_theta, axis=(0, 1)) * d_theta * d_phi

    return total_power


def far_field_directivity(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
) -> np.ndarray:
    """Compute directivity pattern D(θ, φ) = 4π * S(θ, φ) / P_total.

    Directivity is the ratio of the radiation intensity to the
    average radiation intensity over a sphere.

    Args:
        e_theta: E_theta component
        e_phi: E_phi component
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values

    Returns:
        Directivity in dBi at each (theta, phi, freq)
    """
    # Compute power density
    power_density = far_field_power(e_theta, e_phi, freqs)

    # Compute total radiated power
    total_power = compute_total_radiated_power(
        e_theta, e_phi, phi_vals, theta_vals, freqs
    )

    # Avoid division by zero
    total_power = np.maximum(total_power, 1e-12)

    # Compute directivity: 4π * S / P_total
    # The factor (r²) cancels in the ratio of intensities at same r
    directivity = 4.0 * np.pi * power_density / total_power[np.newaxis, np.newaxis, :]

    # Convert to dBi
    directivity_db = 10.0 * np.log10(np.maximum(directivity, 1e-12))

    return directivity_db