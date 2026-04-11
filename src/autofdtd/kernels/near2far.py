"""Near-to-far (N2F) postprocessing kernels for Phase 1 FDTD.

This module implements surface-based radiation and diffraction computations
building on the DFT monitor infrastructure. The N2F transformation uses
the equivalence principle to compute far-field radiation from near-field
surface data recorded by projection monitors.

Key concepts:
- Equivalence principle: J_s = n × H, M_s = -n × E (surface currents)
- Green's function: G(r, r') = exp(ik|r-r'|)/(4π|r-r'|)
- Near2Far integral: E_obs = ∫[iωμ J_s G - (M_s × ∇G)/4π] dS
- Diffraction: grating equation for periodic structures

References:
- Clemson University computational EM notes
- Harrington time-domain MA (transmission line快报)
- Taflove/Born & Wolf far-field equivalence principles
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None

# Physical constants
C0 = 299792458.0  # m/s - speed of light in vacuum
MU0 = 4.0 * np.pi * 1e-7  # H/m - permeability of free space
EPS0 = 1.0 / (MU0 * C0**2)  # F/m - permittivity of free space
Z0 = np.sqrt(MU0 / EPS0)  # ohms - characteristic impedance of free space


def near2far_kernel_metadata() -> dict[str, object]:
    """Expose near2far kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("surface_dft", "greens_function", "far_field_projection"),
    }


def compute_surface_currents(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
    cell_sizes: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equivalent surface currents from DFT fields.

    Uses the equivalence principle:
        J_s = n × H   (electric surface current density)
        M_s = -n × E  (magnetic surface current density)

    where n is the outward surface normal and the cross product follows
    the right-hand rule.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        placements: Surface cell indices (x, y, z) - used for shape consistency
        normal_axis: Normal axis for the source surface (0=x, 1=y, 2=z)
        cell_sizes: Physical size of each cell (dx, dy, dz)

    Returns:
        Tuple of (Js, Ms) arrays each with shape (n_pts, n_freqs, 3)
    """
    # Use actual DFT data shape for number of points
    n_pts = dft_e.shape[0]
    n_freqs = dft_e.shape[1]

    Js = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    Ms = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    # Surface normal direction (outward from surface)
    n_x = 1.0 if normal_axis == 0 else 0.0
    n_y = 1.0 if normal_axis == 1 else 0.0
    n_z = 1.0 if normal_axis == 2 else 0.0

    for i_pt in range(n_pts):
        for i_freq in range(n_freqs):
            ex = dft_e[i_pt, i_freq, 0]
            ey = dft_e[i_pt, i_freq, 1]
            ez = dft_e[i_pt, i_freq, 2]
            hx = dft_h[i_pt, i_freq, 0]
            hy = dft_h[i_pt, i_freq, 1]
            hz = dft_h[i_pt, i_freq, 2]

            # J_s = n × H
            Js[i_pt, i_freq, 0] = n_y * hz - n_z * hy
            Js[i_pt, i_freq, 1] = n_z * hx - n_x * hz
            Js[i_pt, i_freq, 2] = n_x * hy - n_y * hx

            # M_s = -n × E
            Ms[i_pt, i_freq, 0] = -(n_y * ez - n_z * ey)
            Ms[i_pt, i_freq, 1] = -(n_z * ex - n_x * ez)
            Ms[i_pt, i_freq, 2] = -(n_x * ey - n_y * ex)

    return Js, Ms


def compute_surface_positions(
    placements: tuple[tuple[int, int, int], ...],
    monitor_center: tuple[float, float, float],
    monitor_size: tuple[float, float, float],
    cell_sizes: tuple[float, float, float],
) -> np.ndarray:
    """Compute physical positions of surface cells.

    Args:
        placements: Surface cell indices (x, y, z)
        monitor_center: Center of the monitor surface
        monitor_size: Size of the monitor surface
        cell_sizes: Physical size of each cell (dx, dy, dz)

    Returns:
        Array of shape (n_pts, 3) with physical positions
    """
    n_pts = len(placements)
    positions = np.zeros((n_pts, 3), dtype=np.float64)

    # Compute lower bounds of the monitor region
    half_size = tuple(s / 2.0 for s in monitor_size)
    lower_bounds = tuple(c - h for c, h in zip(monitor_center, half_size))

    # Determine tangential axes - detect which axis is the normal from placements
    # The normal axis has a single unique index across all placements (planar surface)
    # Tangential axes have multiple unique indices
    idx0_vals = set(idx[0] for idx in placements)
    idx1_vals = set(idx[1] for idx in placements)
    idx2_vals = set(idx[2] for idx in placements)

    # Detect normal axis: it's the one with only 1 unique value (constant index)
    n_unique = [len(idx0_vals), len(idx1_vals), len(idx2_vals)]
    min_unique = min(n_unique)
    max_unique = max(n_unique)

    # If all axes have the same number of unique values (e.g., single cell),
    # fall back to standard formula for all axes
    if min_unique == max_unique:
        for i_pt, idx in enumerate(placements):
            positions[i_pt, 0] = lower_bounds[0] + (idx[0] + 0.5) * cell_sizes[0]
            positions[i_pt, 1] = lower_bounds[1] + (idx[1] + 0.5) * cell_sizes[1]
            positions[i_pt, 2] = lower_bounds[2] + (idx[2] + 0.5) * cell_sizes[2]
    else:
        # Normal axis is the one with only 1 unique value
        if len(idx0_vals) == 1:
            normal_axis = 0
        elif len(idx1_vals) == 1:
            normal_axis = 1
        else:
            normal_axis = 2

        tang1 = (normal_axis + 1) % 3
        tang2 = (normal_axis + 2) % 3

        for i_pt, idx in enumerate(placements):
            # Normal axis: all surface cells have the same coordinate
            positions[i_pt, normal_axis] = lower_bounds[normal_axis] + 0.5 * cell_sizes[normal_axis]
            # Tangential axes: vary across the surface
            positions[i_pt, tang1] = lower_bounds[tang1] + (idx[tang1] + 0.5) * cell_sizes[tang1]
            positions[i_pt, tang2] = lower_bounds[tang2] + (idx[tang2] + 0.5) * cell_sizes[tang2]

    return positions


def far_field_greens_function(
    source_positions: np.ndarray,
    obs_distance: float,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute far-field Green's function contributions.

    For far-field radiation at distance r >> source extent:
        G ≈ exp(ikr) / (4πr)  (spherical wave)
        ∇G ≈ ikG * r̂  (gradient for radiation pattern)

    Args:
        source_positions: Source point positions (n_pts, 3)
        obs_distance: Observation distance r
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values

    Returns:
        Tuple of (phase_factors, r_hat, distances) where:
        - phase_factors: exp(ik*r) for each (obs_direction, freq, source)
        - r_hat: unit vectors from origin to observation directions
        - distances: approximate distances for amplitude scaling
    """
    n_phi = len(phi_vals)
    n_theta = len(theta_vals)
    n_freqs = len(freqs)
    n_sources = source_positions.shape[0]

    phase_factors = np.zeros((n_phi, n_theta, n_freqs, n_sources), dtype=np.complex128)

    # Compute k = 2πf/c0 for each frequency
    k_vals = 2.0 * np.pi * np.array(freqs) / C0

    # Compute observation direction unit vectors
    r_hat = np.zeros((n_phi, n_theta, 3))
    for i_phi, phi in enumerate(phi_vals):
        for i_theta, theta in enumerate(theta_vals):
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            r_hat[i_phi, i_theta, 0] = sin_theta * np.cos(phi)
            r_hat[i_phi, i_theta, 1] = sin_theta * np.sin(phi)
            r_hat[i_phi, i_theta, 2] = cos_theta

    # For each observation direction, compute phase factors
    # The far-field approximation gives: exp(ik * (r + r̂·r'))
    # where r is the observation distance and r' is the source position
    for i_phi in range(n_phi):
        for i_theta in range(n_theta):
            r_hat_vec = r_hat[i_phi, i_theta]

            # Compute r̂ · r' for each source
            projection = source_positions[:, 0] * r_hat_vec[0] + \
                         source_positions[:, 1] * r_hat_vec[1] + \
                         source_positions[:, 2] * r_hat_vec[2]

            for i_freq, k in enumerate(k_vals):
                # Phase: exp(ik * projection)
                phase_factors[i_phi, i_theta, i_freq, :] = np.exp(1j * k * projection)

    return phase_factors, r_hat, np.full((n_phi, n_theta), obs_distance)


def near2far_transform(
    Js: np.ndarray,
    Ms: np.ndarray,
    source_positions: np.ndarray,
    obs_distance: float,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
    cell_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute far-field radiation pattern from surface currents.

    Uses the equivalence principle with full 3D Green's function:
        E_θ = (iωμ/4πr) ∫ J_s · (∂G/∂θ)̂ dS ≈ (ik/r) ∫ J_sθ e^(ikr̂·r') dS
        E_φ = (iωμ/4πr) ∫ J_s · (∂G/∂φ)̂ dS ≈ (ik/r) ∫ J_sφ e^(ikr̂·r') dS

    where the approximation holds for far-field (r >> source extent).

    Args:
        Js: Electric surface current density (n_pts, n_freqs, 3)
        Ms: Magnetic surface current density (n_pts, n_freqs, 3)
        source_positions: Physical positions of surface cells (n_pts, 3)
        obs_distance: Observation distance r (far field assumption)
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values
        cell_areas: Area weighting for each cell (n_pts,)

    Returns:
        Tuple of (E_theta, E_phi) arrays with shape (n_phi, n_theta, n_freqs)
    """
    n_phi = len(phi_vals)
    n_theta = len(theta_vals)
    n_freqs = len(freqs)

    # Far-field amplitude scaling: i*omega*mu/(4πr) for radiation
    # This is the Stratton-Chu far-field radiation formula
    omega_vals = 2.0 * np.pi * np.array(freqs)
    amplitude_scaling = 1j * omega_vals * MU0 / (4.0 * np.pi * obs_distance)

    # Compute observation direction unit vectors
    r_hat = np.zeros((n_phi, n_theta, 3))
    for i_phi, phi in enumerate(phi_vals):
        for i_theta, theta in enumerate(theta_vals):
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            r_hat[i_phi, i_theta, 0] = sin_theta * np.cos(phi)
            r_hat[i_phi, i_theta, 1] = sin_theta * np.sin(phi)
            r_hat[i_phi, i_theta, 2] = cos_theta

    # Compute θ̂ and φ̂ unit vectors for each direction
    theta_hat = np.zeros((n_phi, n_theta, 3))
    phi_hat = np.zeros((n_phi, n_theta, 3))
    for i_phi, phi in enumerate(phi_vals):
        for i_theta, theta in enumerate(theta_vals):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # θ̂ = (cosθ cosφ, cosθ sinφ, -sinθ)
            theta_hat[i_phi, i_theta, 0] = cos_theta * cos_phi
            theta_hat[i_phi, i_theta, 1] = cos_theta * sin_phi
            theta_hat[i_phi, i_theta, 2] = -sin_theta
            # φ̂ = (-sinφ, cosφ, 0)
            phi_hat[i_phi, i_theta, 0] = -sin_phi
            phi_hat[i_phi, i_theta, 1] = cos_phi
            phi_hat[i_phi, i_theta, 2] = 0.0

    E_theta = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)
    E_phi = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)

    n_sources = Js.shape[0]

    # Accumulate far-field contributions
    for i_phi in range(n_phi):
        for i_theta in range(n_theta):
            r_hat_vec = r_hat[i_phi, i_theta]
            theta_hat_vec = theta_hat[i_phi, i_theta]
            phi_hat_vec = phi_hat[i_phi, i_theta]

            # Compute phase factors: exp(ik * r̂ · r')
            projection = source_positions[:, 0] * r_hat_vec[0] + \
                         source_positions[:, 1] * r_hat_vec[1] + \
                         source_positions[:, 2] * r_hat_vec[2]

            for i_freq, omega in enumerate(omega_vals):
                k = omega / C0  # wavenumber
                phase = np.exp(1j * k * projection)
                area_weight = cell_areas if isinstance(cell_areas, np.ndarray) else 1.0

                # E_θ contribution from J_s and M_s
                # E_θ = (i*ω*μ₀/(4πr)) ∫ [J_s · θ̂ - M_s · φ̂ / Z₀] e^(ik·r') dS
                # The Stratton-Chu formula gives E_θ = (i*ω*μ₀/(4πr)) * (N_θ - L_φ/Z₀)
                # where N_θ = J_s · θ̂ and L_φ = M_s · φ̂ (negative because θ̂ × r̂ = -sinθ φ̂)
                js_theta = Js[:, i_freq, 0] * theta_hat_vec[0] + \
                           Js[:, i_freq, 1] * theta_hat_vec[1] + \
                           Js[:, i_freq, 2] * theta_hat_vec[2]
                # M_s · φ̂ (phi component of magnetic surface current)
                ms_phi = Ms[:, i_freq, 0] * phi_hat_vec[0] + \
                         Ms[:, i_freq, 1] * phi_hat_vec[1] + \
                         Ms[:, i_freq, 2] * phi_hat_vec[2]

                E_theta[i_phi, i_theta, i_freq] = np.sum(
                    (js_theta - ms_phi / Z0) * phase * area_weight
                )

                # E_φ contribution from J_s and M_s
                # E_φ = (i*ω*μ₀/(4πr)) ∫ [J_s · φ̂ + M_s · θ̂ / Z₀] e^(ik·r') dS
                # The Stratton-Chu formula gives E_φ = (i*ω*μ₀/(4πr)) * (N_φ + L_θ/Z₀)
                # where N_φ = J_s · φ̂ and L_θ = M_s · θ̂
                js_phi = Js[:, i_freq, 0] * phi_hat_vec[0] + \
                         Js[:, i_freq, 1] * phi_hat_vec[1] + \
                         Js[:, i_freq, 2] * phi_hat_vec[2]
                # M_s · θ̂ (theta component of magnetic surface current)
                ms_theta = Ms[:, i_freq, 0] * theta_hat_vec[0] + \
                           Ms[:, i_freq, 1] * theta_hat_vec[1] + \
                           Ms[:, i_freq, 2] * theta_hat_vec[2]

                E_phi[i_phi, i_theta, i_freq] = np.sum(
                    (js_phi + ms_theta / Z0) * phase * area_weight
                )

            # Apply amplitude scaling
            E_theta[i_phi, i_theta, :] *= amplitude_scaling
            E_phi[i_phi, i_theta, :] *= amplitude_scaling

    return E_theta, E_phi


def far_field_power(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    freqs: tuple[float, ...],
) -> np.ndarray:
    """Compute radiated power from far-field components.

    The radiated power density (Poynting magnitude) is:
        S = |E|² / (2Z₀) = (|E_θ|² + |E_φ|²) / (2Z₀)

    Args:
        e_theta: E_theta far-field component (n_phi, n_theta, n_freqs)
        e_phi: E_phi far-field component (n_phi, n_theta, n_freqs)
        freqs: Frequency values

    Returns:
        Power density array with shape (n_phi, n_theta, n_freqs)
    """
    e_theta_mag = np.abs(e_theta)
    e_phi_mag = np.abs(e_phi)
    power = (e_theta_mag**2 + e_phi_mag**2) / (2.0 * Z0)
    return power


def compute_diffraction_orders_n2f(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
    freqs: tuple[float, ...],
    *,
    period_y: float | None = None,
    period_z: float | None = None,
    medium_eps: complex = 1.0 + 0.0j,
    num_orders: int = 5,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Compute diffraction orders for periodic structures using N2F principles.

    For a 2D grating with periodicity along y and z:
    - k_y = m_y * 2π / d_y
    - k_z = m_z * 2π / d_z
    - k_diff² = ε * k0² - k_y² - k_z²

    Propagating orders satisfy k_diff² > 0 (real k_diff).

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        placements: Surface cell indices (x, y, z)
        normal_axis: Surface normal axis (0=x, 1=y, 2=z)
        freqs: Frequency values for each DFT component
        period_y: Grating period along y direction
        period_z: Grating period along z direction
        medium_eps: Relative permittivity of the diffraction medium
        num_orders: Number of diffraction orders to compute (each side of zero)

    Returns:
        Tuple of (orders, mx, my) where:
        - orders: complex array (num_orders², num_freqs) with diffraction amplitudes
        - mx: integer tuple of m_x indices for each order
        - my: integer tuple of m_y indices for each order
    """
    n_pts = len(placements)
    n_freqs = len(freqs)

    # Get tangential axes and component indices for field extraction
    if normal_axis == 0:
        tang1, tang2 = 1, 2  # y, z
        e_tang1_idx, e_tang2_idx = 1, 2  # Ey, Ez
        h_tang1_idx, h_tang2_idx = 1, 2  # Hy, Hz
        pos_tang1_axis, pos_tang2_axis = 1, 2
    elif normal_axis == 1:
        tang1, tang2 = 0, 2  # x, z
        e_tang1_idx, e_tang2_idx = 0, 2  # Ex, Ez
        h_tang1_idx, h_tang2_idx = 0, 2  # Hx, Hz
        pos_tang1_axis, pos_tang2_axis = 0, 2
    else:  # normal_axis == 2
        tang1, tang2 = 0, 1  # x, y
        e_tang1_idx, e_tang2_idx = 0, 1  # Ex, Ey
        h_tang1_idx, h_tang2_idx = 0, 1  # Hx, Hy
        pos_tang1_axis, pos_tang2_axis = 0, 1

    # If periods not provided, return zero amplitudes
    if period_y is None or period_z is None or period_y <= 0 or period_z <= 0:
        mx_all = tuple(m for m in range(-num_orders, num_orders + 1))
        my_all = tuple(m for m in range(-num_orders, num_orders + 1))
        n_orders = len(mx_all) * len(my_all)
        orders = np.zeros((n_orders, n_freqs), dtype=np.complex128)
        mx_out = []
        my_out = []
        for mx in mx_all:
            for my in my_all:
                mx_out.append(mx)
                my_out.append(my)
        return orders, tuple(mx_out), tuple(my_out)

    # Compute k0 for each frequency
    k0_vals = 2.0 * np.pi * np.array(freqs) / C0

    # Generate diffraction order indices
    mx_all = tuple(m for m in range(-num_orders, num_orders + 1))
    my_all = tuple(m for m in range(-num_orders, num_orders + 1))
    n_orders = len(mx_all) * len(my_all)

    orders = np.zeros((n_orders, n_freqs), dtype=np.complex128)
    mx_out = []
    my_out = []

    # Compute surface normal direction
    n_vec = np.zeros(3)
    n_vec[normal_axis] = 1.0

    order_idx = 0
    for mx in mx_all:
        for my in my_all:
            k_y = mx * 2.0 * np.pi / period_y
            k_z = my * 2.0 * np.pi / period_z
            k_yz_sq = k_y**2 + k_z**2

            mx_out.append(mx)
            my_out.append(my)

            for i_freq in range(n_freqs):
                k0 = k0_vals[i_freq]

                # Check if propagating: k_yz_sq < |ε| * k0²
                eps_real = medium_eps.real
                if k_yz_sq > eps_real * k0**2:
                    orders[order_idx, i_freq] = 0.0 + 0.0j
                else:
                    # Propagating order - compute amplitude via surface integral
                    # The amplitude is the Fourier transform of the tangential field
                    # at the spatial frequency (k_y, k_z)
                    amplitude = 0.0 + 0.0j

                    for i_pt in range(n_pts):
                        idx = placements[i_pt]

                        # Physical position along tangential directions
                        # Use actual cell center coordinates: cell_idx + 0.5
                        pos_tang1 = float(idx[pos_tang1_axis]) + 0.5
                        pos_tang2 = float(idx[pos_tang2_axis]) + 0.5

                        # DFT field values at this point - keep complex components
                        e1 = dft_e[i_pt, i_freq, e_tang1_idx]
                        e2 = dft_e[i_pt, i_freq, e_tang2_idx]

                        # Phase factor for Fourier transform: exp(-i*(k_y*y + k_z*z))
                        # The negative sign is for the forward Fourier transform
                        phase = np.exp(
                            -1j * (k_y * pos_tang1 + k_z * pos_tang2)
                        )

                        # Accumulate using complex tangential E field (not magnitude)
                        amplitude += e1 * phase

                    orders[order_idx, i_freq] = amplitude

            order_idx += 1

    return orders, tuple(mx_out), tuple(my_out)


def extract_radiation_pattern(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Extract radiation pattern data in standard format.

    Args:
        e_theta: E_theta component (n_phi, n_theta, n_freqs)
        e_phi: E_phi component (n_phi, n_theta, n_freqs)
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values

    Returns:
        Dictionary with radiation pattern components
    """
    power = far_field_power(e_theta, e_phi, freqs)

    return {
        "e_theta": e_theta,
        "e_phi": e_phi,
        "power": power,
        "phi_vals": phi_vals,
        "theta_vals": theta_vals,
        "freqs": freqs,
    }


def project_to_cartesian_plane(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    freqs: tuple[float, ...],
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    obs_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project far-field data onto a Cartesian observation plane.

    Converts spherical (theta, phi) far-field data to Cartesian (x, y)
    observation coordinates at a fixed distance.

    Args:
        e_theta: E_theta component (n_phi, n_theta, n_freqs)
        e_phi: E_phi component (n_phi, n_theta, n_freqs)
        phi_vals: Azimuthal angles in radians
        theta_vals: Polar angles in radians
        freqs: Frequency values
        x_obs: X observation coordinates
        y_obs: Y observation coordinates
        obs_distance: Z distance of observation plane

    Returns:
        Tuple of (Ex, Ey, Ez) arrays with shape (n_x, n_y, n_freqs)
    """
    n_x = len(x_obs)
    n_y = len(y_obs)
    n_freqs = len(freqs)

    ex_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)
    ey_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)
    ez_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)

    # Convert Cartesian to spherical for each observation point
    for i_x, x in enumerate(x_obs):
        for i_y, y in enumerate(y_obs):
            # Convert to spherical coordinates
            r = np.sqrt(x**2 + y**2 + obs_distance**2)
            phi = np.arctan2(y, x)
            theta = np.arctan2(np.sqrt(x**2 + y**2), obs_distance)

            # Find nearest phi and theta indices
            phi_idx = np.argmin(np.abs(phi_vals - phi))
            theta_idx = np.argmin(np.abs(theta_vals - theta))

            # Get field components at this direction
            e_t = e_theta[phi_idx, theta_idx, :]
            e_p = e_phi[phi_idx, theta_idx, :]

            # Convert theta/phi components to Cartesian
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            # E_x = E_θ * cos_θ * cos_φ - E_φ * sin_φ
            # E_y = E_θ * cos_θ * sin_φ + E_φ * cos_φ
            # E_z = -E_θ * sin_θ
            ex_out[i_x, i_y, :] = e_t * cos_theta * cos_phi - e_p * sin_phi
            ey_out[i_x, i_y, :] = e_t * cos_theta * sin_phi + e_p * cos_phi
            ez_out[i_x, i_y, :] = -e_t * sin_theta

    return ex_out, ey_out, ez_out