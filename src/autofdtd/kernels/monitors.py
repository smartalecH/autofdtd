"""Monitor recording kernels for Phase 1 field sampling."""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


def field_monitor_kernel_metadata() -> dict[str, object]:
    """Expose the field monitor kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("monitor_recording",),
    }


def record_field_time_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    *,
    time: float,
) -> None:
    """Record field values at monitor cells for a time-domain monitor.

    This function samples the electric and magnetic fields at the monitor's
    placement cells and stores the values in the monitor state for later
    retrieval.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: FieldMonitorState with compiled placement and fields
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_time_domain:
        return

    for field in compiled.fields:
        values = _extract_field_at_monitor(
            electric_field, magnetic_field, field, compiled.placements
        )
        if monitor_state.time_series is not None and field in monitor_state.time_series:
            monitor_state.time_series[field].append(values.copy())

    if monitor_state.time_stamps is not None:
        monitor_state.time_stamps.append(time)


def record_field_frequency_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    *,
    time: float,
) -> None:
    """Record field values for DFT accumulation in a frequency-domain monitor.

    For FieldMonitor with frequency points, this accumulates the DFT sum:
    F(omega) = sum_n f(t_n) * exp(-i * omega * t_n)

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: FieldMonitorState with compiled placement and fields
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_frequency_domain:
        return

    freqs = compiled.freqs
    n_freqs = len(freqs)

    for field in compiled.fields:
        values = _extract_field_at_monitor(
            electric_field, magnetic_field, field, compiled.placements
        )
        if monitor_state.dft_data is not None and field in monitor_state.dft_data:
            omega_t = 2.0 * np.pi * np.asarray(freqs) * time
            for i, f_val in enumerate(values):
                for j, freq in enumerate(freqs):
                    phase = np.exp(-1j * omega_t[j])
                    monitor_state.dft_data[field][i] += f_val * phase

    monitor_state._set_dft_count(monitor_state.dft_count_value + 1)


def _extract_field_at_monitor(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    field: str,
    placements: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Extract field component values at monitor cell indices.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        field: Field component name (Ex, Ey, Ez, Hx, Hy, Hz)
        placements: Monitor cell indices

    Returns:
        Array of field values at monitor cells
    """
    n_pts = len(placements)
    values = np.zeros(n_pts, dtype=np.complex128)

    if field not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        return values

    is_electric = field[0] == "E"
    component_axis = {"x": 0, "y": 1, "z": 2}[field[1]]

    for i, idx in enumerate(placements):
        if is_electric:
            values[i] = electric_field[idx][component_axis]
        else:
            values[i] = magnetic_field[idx][component_axis]

    return values


def sample_field_at_points(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    points: tuple[tuple[int, int, int], ...],
    fields: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Sample field components at a set of grid points.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        points: Grid cell indices
        fields: Field components to sample

    Returns:
        Dictionary mapping field names to value arrays
    """
    result = {}
    for field in fields:
        result[field] = _extract_field_at_monitor(
            electric_field, magnetic_field, field, points
        ).copy()
    return result


def accumulate_flux(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    direction: Literal["+", "-"],
    axis: int,
    placements: tuple[tuple[int, int, int], ...],
) -> float:
    """Compute flux through a surface.

    The flux is computed as the integral of H × E over the surface,
    with the direction determining the sign convention.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        direction: Flux direction (+ or -)
        axis: The normal axis for the flux surface
        placements: Surface cell indices

    Returns:
        Total flux value
    """
    sign = 1.0 if direction == "+" else -1.0

    # For a surface perpendicular to `axis`, flux = integral of
    # the normal component of the Poynting vector S = E × H
    # The normal component depends on the axis:
    # - x-axis: S_x = E_y * H_z - E_z * H_y
    # - y-axis: S_y = E_z * H_x - E_x * H_z
    # - z-axis: S_z = E_x * H_y - E_y * H_x

    tang_axes = tuple(a for a in range(3) if a != axis)

    flux = 0.0
    for idx in placements:
        ex = electric_field[idx][0]
        ey = electric_field[idx][1]
        ez = electric_field[idx][2]
        hx = magnetic_field[idx][0]
        hy = magnetic_field[idx][1]
        hz = magnetic_field[idx][2]

        if axis == 0:
            s_normal = ey * hz - ez * hy
        elif axis == 1:
            s_normal = ez * hx - ex * hz
        else:  # axis == 2
            s_normal = ex * hy - ey * hx

        flux += s_normal.real

    return sign * flux


# --------------------------------------------------------------------
# Flux monitor kernels
# --------------------------------------------------------------------


def flux_monitor_kernel_metadata() -> dict[str, object]:
    """Expose the flux monitor kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("flux_integration",),
    }


def record_flux_time_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    *,
    time: float,
) -> None:
    """Record flux at the monitor surface for time-domain.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: FluxMonitorState with compiled placement and direction
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_time_domain:
        return

    flux_value = accumulate_flux(
        electric_field,
        magnetic_field,
        compiled.direction,
        compiled.normal_axis,
        compiled.placements,
    )

    if monitor_state.flux_series is not None:
        monitor_state.flux_series.append(flux_value)
    if monitor_state.time_stamps is not None:
        monitor_state.time_stamps.append(time)


def record_flux_frequency_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    *,
    time: float,
) -> None:
    """Record flux for DFT accumulation in a frequency-domain monitor.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: FluxMonitorState with compiled placement and direction
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_frequency_domain:
        return

    flux_value = accumulate_flux(
        electric_field,
        magnetic_field,
        compiled.direction,
        compiled.normal_axis,
        compiled.placements,
    )

    if monitor_state.dft_flux is not None:
        omega_t = 2.0 * np.pi * np.asarray(compiled.freqs) * time
        for j, freq in enumerate(compiled.freqs):
            phase = np.exp(-1j * omega_t[j])
            monitor_state.dft_flux[j] += flux_value * phase

    monitor_state._set_dft_count(monitor_state.dft_count_value + 1)


# --------------------------------------------------------------------
# Medium and permittivity monitor kernels
# --------------------------------------------------------------------


def medium_monitor_kernel_metadata() -> dict[str, object]:
    """Expose the medium monitor kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("medium_sampling",),
    }


def sample_medium_at_points(
    placements: tuple[tuple[int, int, int], ...],
    medium_coefficients: np.ndarray,
) -> np.ndarray:
    """Sample medium properties at a set of grid points.

    This function extracts the complex permittivity and permeability values
    at the specified grid indices.

    Args:
        placements: Grid cell indices
        medium_coefficients: Array of shape (n_pts, num_freqs, n_components)
            where n_components is 6 for MediumMonitor (eps_xx, eps_yy, eps_zz,
            mu_xx, mu_yy, mu_zz) or 3 for PermittivityMonitor.

    Returns:
        Array of medium values at the specified points
    """
    n_pts = len(placements)
    values = np.zeros_like(medium_coefficients[:n_pts])
    values[:n_pts] = medium_coefficients[:n_pts]
    return values


def record_medium_monitor(
    monitor_state,
    *,
    step_index: int,
) -> None:
    """Record medium properties for a monitor.

    For medium monitors, the medium properties are static (computed once
    at setup from the scene), so this function just checks whether the
    step index matches the interval/start criteria.

    Args:
        monitor_state: MediumMonitorState with compiled medium data
        step_index: Current step index

    Returns:
        True if the monitor should record at this step, False otherwise
    """
    compiled = monitor_state.compiled

    # Check if this step should be recorded
    if step_index < compiled.start:
        return False
    if (step_index - compiled.start) % compiled.interval != 0:
        return False

    return True


# --------------------------------------------------------------------
# Mode monitor kernels
# --------------------------------------------------------------------


def mode_monitor_kernel_metadata() -> dict[str, object]:
    """Expose the mode monitor kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("mode_overlap",),
    }


def extract_mode_field_components(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
) -> dict[str, np.ndarray]:
    """Extract field components at mode monitor placements for overlap computation.

    For mode monitors, we need the tangential field components on the
    cross-sectional plane.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        placements: Monitor cell indices
        normal_axis: Normal axis for the cross-section

    Returns:
        Dictionary mapping field names to value arrays at monitor placements
    """
    n_pts = len(placements)
    tang_axes = tuple(a for a in range(3) if a != normal_axis)

    # Field component mapping for each normal axis
    if normal_axis == 0:
        e_components = ["Ey", "Ez"]
        h_components = ["Hy", "Hz"]
    elif normal_axis == 1:
        e_components = ["Ex", "Ez"]
        h_components = ["Hx", "Hz"]
    else:  # axis == 2
        e_components = ["Ex", "Ey"]
        h_components = ["Hx", "Hy"]

    component_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 0, "Hy": 1, "Hz": 2}

    result = {}
    for field in e_components + h_components:
        values = np.zeros(n_pts, dtype=np.complex128)
        is_electric = field[0] == "E"
        comp_axis = component_map[field]
        field_buffer = electric_field if is_electric else magnetic_field

        for i, idx in enumerate(placements):
            values[i] = field_buffer[idx][comp_axis]

        result[field] = values

    return result


def compute_mode_overlap(
    field_components: dict[str, np.ndarray],
    mode_solution,
    normal_axis: int,
    placements: tuple[tuple[int, int, int], ...],
) -> complex:
    """Compute overlap integral between field data and a mode profile.

    The overlap integral computes how much of the input field couples
    into the given mode. This is used by mode monitors to record the
    modal power or amplitude as a function of time or frequency.

    The overlap is:
        overlap = Re[integral(E_tangential × H_mode* · n̂) dA]

    where E_tangential is the field at the monitor plane, H_mode* is the
    conjugate of the mode's magnetic field, and n̂ is the normal direction.

    Args:
        field_components: Dict of field arrays at monitor placements
        mode_solution: ModeSolution with Ex, Ey, Ez, Hx, Hy, Hz arrays
        normal_axis: Normal axis for the cross-section
        placements: Monitor cell indices

    Returns:
        Complex overlap amplitude
    """
    n_pts = len(placements)

    # Get mode coordinates
    mode_x = mode_solution.x
    mode_y = mode_solution.y
    mode_nx = len(mode_x)
    mode_ny = len(mode_y)

    def to_complex(c: tuple) -> complex:
        return complex(c[0], c[1]) if isinstance(c, tuple) else complex(c, 0.0)

    def get_mode_field(field_name: str, i: int, j: int) -> complex:
        """Get mode field at grid point (i, j)."""
        if j * mode_nx + i >= len(getattr(mode_solution, field_name)):
            return 0.0 + 0.0j
        return to_complex(getattr(mode_solution, field_name)[j * mode_nx + i])

    overlap = 0.0 + 0.0j

    for i, idx in enumerate(placements):
        # Compute mode grid indices (nearest neighbor interpolation)
        # This is a simplified mapping - real implementation would use proper interpolation
        mx_idx = min(i % mode_nx, mode_nx - 1)
        my_idx = min(i // mode_nx, mode_ny - 1)

        # Get field components at this point
        if normal_axis == 0:
            # x-normal: tangential fields are Ey, Ez, Hy, Hz
            ex_field = 0.0  # No Ex for x-normal
            ey_field = field_components.get("Ey", np.zeros(n_pts))[i]
            ez_field = field_components.get("Ez", np.zeros(n_pts))[i]
            hx_field = 0.0  # No Hx for x-normal
            hy_field = field_components.get("Hy", np.zeros(n_pts))[i]
            hz_field = field_components.get("Hz", np.zeros(n_pts))[i]

            mEy = get_mode_field("Ey", mx_idx, my_idx)
            mEz = get_mode_field("Ez", mx_idx, my_idx)
            mHy = get_mode_field("Hy", mx_idx, my_idx)
            mHz = get_mode_field("Hz", mx_idx, my_idx)

            # Poynting vector S = E × H*
            # Normal component (x): S_x = E_y*H_z* - E_z*H_y
            s_normal = ey_field * np.conj(hz_field) - ez_field * np.conj(hy_field)
            m_s_normal = mEy * np.conj(mHz) - mEz * np.conj(mHy)

        elif normal_axis == 1:
            # y-normal: tangential fields are Ex, Ez, Hx, Hz
            ex_field = field_components.get("Ex", np.zeros(n_pts))[i]
            ey_field = 0.0  # No Ey for y-normal
            ez_field = field_components.get("Ez", np.zeros(n_pts))[i]
            hx_field = field_components.get("Hx", np.zeros(n_pts))[i]
            hy_field = 0.0  # No Hy for y-normal
            hz_field = field_components.get("Hz", np.zeros(n_pts))[i]

            mEx = get_mode_field("Ex", mx_idx, my_idx)
            mEz = get_mode_field("Ez", mx_idx, my_idx)
            mHx = get_mode_field("Hx", mx_idx, my_idx)
            mHz = get_mode_field("Hz", mx_idx, my_idx)

            # Normal component (y): S_y = E_z*H_x* - E_x*H_z*
            s_normal = ez_field * np.conj(hx_field) - ex_field * np.conj(hz_field)
            m_s_normal = mEz * np.conj(mHx) - mEx * np.conj(mHz)

        else:  # axis == 2
            # z-normal: tangential fields are Ex, Ey, Hx, Hy
            ex_field = field_components.get("Ex", np.zeros(n_pts))[i]
            ey_field = field_components.get("Ey", np.zeros(n_pts))[i]
            ez_field = 0.0  # No Ez for z-normal
            hx_field = field_components.get("Hx", np.zeros(n_pts))[i]
            hy_field = field_components.get("Hy", np.zeros(n_pts))[i]
            hz_field = 0.0  # No Hz for z-normal

            mEx = get_mode_field("Ex", mx_idx, my_idx)
            mEy = get_mode_field("Ey", mx_idx, my_idx)
            mHx = get_mode_field("Hx", mx_idx, my_idx)
            mHy = get_mode_field("Hy", mx_idx, my_idx)

            # Normal component (z): S_z = E_x*H_y* - E_y*H_x*
            s_normal = ex_field * np.conj(hy_field) - ey_field * np.conj(hx_field)
            m_s_normal = mEx * np.conj(mHy) - mEy * np.conj(mHx)

        # Add contribution to overlap integral
        overlap += s_normal * np.conj(m_s_normal)

    return overlap


def record_mode_time_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    mode_solutions: tuple,
    *,
    time: float,
) -> None:
    """Record mode overlap amplitudes at the current timestep.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: ModeMonitorState with compiled placement and mode info
        mode_solutions: Tuple of ModeSolution objects for overlap
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_time_domain:
        return

    # Extract field components at monitor placements
    field_components = extract_mode_field_components(
        electric_field, magnetic_field,
        compiled.placements, compiled.normal_axis
    )

    # Compute overlap for each mode
    monitor_state.record_mode_time_domain(field_components, mode_solutions, time)


def record_mode_frequency_domain(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    mode_solutions: tuple,
    *,
    time: float,
) -> None:
    """Record mode overlap for DFT accumulation in a frequency-domain monitor.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: ModeMonitorState with compiled placement and mode info
        mode_solutions: Tuple of ModeSolution objects for overlap
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_frequency_domain:
        return

    # Extract field components at monitor placements
    field_components = extract_mode_field_components(
        electric_field, magnetic_field,
        compiled.placements, compiled.normal_axis
    )

    # Accumulate DFT terms
    monitor_state.accumulate_mode_dft(field_components, mode_solutions, time)


# --------------------------------------------------------------------
# Projection monitor kernels
# --------------------------------------------------------------------


def projection_monitor_kernel_metadata() -> dict[str, object]:
    """Expose the projection monitor kernel backend choices for diagnostics."""
    return {
        "backend": "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("dft_accumulation", "far_field_projection"),
    }


def record_projection_dft(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    monitor_state,
    *,
    time: float,
) -> None:
    """Record DFT terms for a projection monitor.

    This function accumulates the DFT of the E and H fields on the
    source surface for later far-field projection computation.

    Args:
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        monitor_state: ProjectionMonitorState with compiled placement
        time: Current simulation time
    """
    if not hasattr(monitor_state, "compiled"):
        return

    compiled = monitor_state.compiled

    if not compiled.is_frequency_domain:
        return

    monitor_state.accumulate_dft(electric_field, magnetic_field, time)


def far_field_projection_angle(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
    projection_distance: float,
    phi_vals: tuple[float, ...],
    theta_vals: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute far-field projection in angle space.

    Uses the equivalence principle to radiate from the recorded surface
    fields to observation points at a specified distance.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        placements: Surface cell indices
        normal_axis: Normal axis for the source surface
        projection_distance: Distance for far-field observation
        phi_vals: Phi angle values
        theta_vals: Theta angle values

    Returns:
        Tuple of (e_theta, e_phi) arrays with shape (n_phi, n_theta, n_freqs)
    """
    n_pts = len(placements)
    n_freqs = dft_e.shape[1]
    n_phi = len(phi_vals)
    n_theta = len(theta_vals)

    e_theta = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)
    e_phi = np.zeros((n_phi, n_theta, n_freqs), dtype=np.complex128)

    # Get tangential axes
    tang1, tang2 = (1, 2) if normal_axis == 0 else (0, 2) if normal_axis == 1 else (0, 1)

    # For each observation direction, compute far field using equivalence principle
    # Phase 1 uses a simplified Rayleigh approximation:
    # E_obs ~ (k0/r) * exp(-i*k0*r) * integral(Surface_J * exp(i*k0*r') dS')
    # where Surface_J = n × H_surf for electric current
    # and Surface_M = -n × E_surf for magnetic current

    for i_phi, phi in enumerate(phi_vals):
        for i_theta, theta in enumerate(theta_vals):
            # Convert to Cartesian direction
            phi_rad = np.deg2rad(phi)
            theta_rad = np.deg2rad(theta)

            # Direction cosines
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            sin_phi = np.sin(phi_rad)
            cos_phi = np.cos(phi_rad)

            # Observation direction
            obs_x = sin_theta * cos_phi
            obs_y = sin_theta * sin_phi
            obs_z = cos_theta

            # For each source point and frequency, accumulate far field
            for i_pt in range(n_pts):
                idx = placements[i_pt]

                # Get field values at this point
                ex = dft_e[i_pt, :, 0]
                ey = dft_e[i_pt, :, 1]
                ez = dft_e[i_pt, :, 2]
                hx = dft_h[i_pt, :, 0]
                hy = dft_h[i_pt, :, 1]
                hz = dft_h[i_pt, :, 2]

                # Surface normal direction
                n_x = 1.0 if normal_axis == 0 else 0.0
                n_y = 1.0 if normal_axis == 1 else 0.0
                n_z = 1.0 if normal_axis == 2 else 0.0

                # Electric surface current: J_s = n × H
                js_x = n_y * hz - n_z * hy
                js_y = n_z * hx - n_x * hz
                js_z = n_x * hy - n_y * hx

                # Magnetic surface current: M_s = -n × E
                ms_x = -(n_y * ez - n_z * ey)
                ms_y = -(n_z * ex - n_x * ez)
                ms_z = -(n_x * ey - n_y * ex)

                # Far field contribution (simplified, assuming planar surface)
                # E_obs ∝ i*k0*(n × H_surf - Z0*n × E_surf) / (2*r) * exp(i*k0*r)
                # Z0 = 377 ohms for free space
                Z0 = 377.0
                r = projection_distance
                factor = 1j / (2.0 * r)

                for i_freq in range(n_freqs):
                    # Scalar products for this direction
                    js_dot = js_x[i_freq] * obs_x + js_y[i_freq] * obs_y + js_z[i_freq] * obs_z
                    ms_dot = ms_x[i_freq] * obs_x + ms_y[i_freq] * obs_y + ms_z[i_freq] * obs_z

                    # E_theta and E_phi components
                    e_theta[i_phi, i_theta, i_freq] += factor * js_dot
                    e_phi[i_phi, i_theta, i_freq] += factor * (-ms_dot / Z0)

    return e_theta, e_phi


def far_field_projection_cartesian(
    dft_e: np.ndarray,
    dft_h: np.ndarray,
    placements: tuple[tuple[int, int, int], ...],
    normal_axis: int,
    projection_distance: float,
    x_vals: tuple[float, ...],
    y_vals: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute far-field projection on a Cartesian observation plane.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        placements: Surface cell indices
        normal_axis: Normal axis for the source surface
        projection_distance: Distance for far-field observation
        x_vals: X observation coordinates
        y_vals: Y observation coordinates

    Returns:
        Tuple of (Ex, Ey, Ez) arrays with shape (n_x, n_y, n_freqs)
    """
    n_pts = len(placements)
    n_freqs = dft_e.shape[1]
    n_x = len(x_vals)
    n_y = len(y_vals)

    ex_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)
    ey_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)
    ez_out = np.zeros((n_x, n_y, n_freqs), dtype=np.complex128)

    for i_x, x in enumerate(x_vals):
        for i_y, y in enumerate(y_vals):
            # Observation point
            obs_x = x
            obs_y = y
            obs_z = projection_distance

            # Distance from origin
            r = np.sqrt(obs_x**2 + obs_y**2 + obs_z**2)

            # Direction cosines
            ux = obs_x / r
            uy = obs_y / r
            uz = obs_z / r

            for i_pt in range(n_pts):
                idx = placements[i_pt]

                ex = dft_e[i_pt, :, 0]
                ey = dft_e[i_pt, :, 1]
                ez = dft_e[i_pt, :, 2]
                hx = dft_h[i_pt, :, 0]
                hy = dft_h[i_pt, :, 1]
                hz = dft_h[i_pt, :, 2]

                # Phase factor for this point (simplified)
                phase = 1.0  # Would need actual positions for proper phase

                for i_freq in range(n_freqs):
                    # Simplified far-field projection
                    factor = 1j / r
                    ex_out[i_x, i_y, i_freq] += factor * ex[i_freq] * phase
                    ey_out[i_x, i_y, i_freq] += factor * ey[i_freq] * phase
                    ez_out[i_x, i_y, i_freq] += factor * ez[i_freq] * phase

    return ex_out, ey_out, ez_out


def compute_diffraction_orders(
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
) -> tuple[
    np.ndarray,  # orders: (num_orders, num_freqs) complex
    tuple[int, ...],  # mx indices
    tuple[int, ...],  # my indices
]:
    """Compute diffraction orders from surface DFT fields.

    This function applies the grating equation to determine diffraction
    order amplitudes from fields recorded on a 2D surface. The surface
    acts as a 2D Fourier transform element, projecting the field
    distribution into angular spectrum components.

    Physics
    -------
    For a 2D surface with periodicity along y and z directions:
    - k_y = m_y * 2π / d_y
    - k_z = m_z * 2π / d_z
    - k_diff^2 = ε * k0^2 - k_y^2 - k_z^2

    A propagating order requires k_diff^2 > 0 (real k_diff).
    Evanescent orders (k_diff^2 < 0) are suppressed.

    Args:
        dft_e: DFT of E fields at surface points, shape (n_pts, n_freqs, 3)
        dft_h: DFT of H fields at surface points, shape (n_pts, n_freqs, 3)
        placements: Surface cell indices (x, y, z)
        normal_axis: Surface normal axis (0=x, 1=y, 2=z)
        freqs: Frequency values for each DFT component
        period_y: Grating period along y direction (None = use grid spacing)
        period_z: Grating period along z direction (None = use grid spacing)
        medium_eps: Relative permittivity of the diffraction medium
        num_orders: Number of diffraction orders to compute (each side of zero)

    Returns:
        Tuple of (orders, mx, my) where:
        - orders: complex array (num_orders^2, num_freqs) with diffraction amplitudes
        - mx: integer tuple of m_x indices for each order
        - my: integer tuple of m_y indices for each order
    """
    n_pts = len(placements)
    n_freqs = len(freqs)
    c0 = 299792458.0  # speed of light in m/s

    # Get tangential axes (the non-normal directions)
    if normal_axis == 0:
        tang1, tang2 = 1, 2  # y, z
        e_tang1 = 1  # Ey
        e_tang2 = 2  # Ez
        h_tang1 = 1  # Hy
        h_tang2 = 2  # Hz
    elif normal_axis == 1:
        tang1, tang2 = 0, 2  # x, z
        e_tang1 = 0  # Ex
        e_tang2 = 2  # Ez
        h_tang1 = 0  # Hx
        h_tang2 = 2  # Hz
    else:  # normal_axis == 2
        tang1, tang2 = 0, 1  # x, y
        e_tang1 = 0  # Ex
        e_tang2 = 1  # Ey
        h_tang1 = 0  # Hx
        h_tang2 = 1  # Hy

    # If periods not provided, we can't compute proper diffraction orders
    # Return zero amplitudes with a warning
    if period_y is None or period_z is None or period_y <= 0 or period_z <= 0:
        # No valid periodicity - return empty orders
        mx_all = tuple(m for m in range(-num_orders, num_orders + 1))
        my_all = tuple(m for m in range(-num_orders, num_orders + 1))
        n_orders = len(mx_all) * len(my_all)
        orders = np.zeros((n_orders, n_freqs), dtype=np.complex128)
        mx_vals = []
        my_vals = []
        for mx in mx_all:
            for my in my_all:
                mx_vals.append(mx)
                my_vals.append(my)
        return orders, tuple(mx_vals), tuple(my_vals)

    # Compute k0 for each frequency
    k0_vals = 2.0 * np.pi * np.array(freqs) / c0

    # Generate diffraction order indices
    mx_all = tuple(m for m in range(-num_orders, num_orders + 1))
    my_all = tuple(m for m in range(-num_orders, num_orders + 1))
    n_orders = len(mx_all) * len(my_all)

    # Prepare output arrays
    orders = np.zeros((n_orders, n_freqs), dtype=np.complex128)
    mx_out = []
    my_out = []

    # Compute spatial frequencies for each diffraction order
    # k_y = m_y * 2π / period_y
    # k_z = m_z * 2π / period_z
    order_idx = 0
    for mx in mx_all:
        for my in my_all:
            k_y = mx * 2.0 * np.pi / period_y
            k_z = my * 2.0 * np.pi / period_z
            k_yz_sq = k_y**2 + k_z**2

            mx_out.append(mx)
            my_out.append(my)

            # For each frequency, compute amplitude for this order
            for i_freq in range(n_freqs):
                k0 = k0_vals[i_freq]

                # Check if propagating: k_yz_sq < |ε| * k0^2
                # For complex ε, use real part for propagation criterion
                eps_real = medium_eps.real
                if k_yz_sq > eps_real * k0**2:
                    # Evanescent order - amplitude is negligible
                    orders[order_idx, i_freq] = 0.0 + 0.0j
                else:
                    # Propagating order - compute amplitude via Fourier transform
                    # The amplitude is the Fourier transform of the tangential field
                    # at the spatial frequency (k_y, k_z)
                    amplitude = 0.0 + 0.0j

                    for i_pt in range(n_pts):
                        idx = placements[i_pt]
                        # Position of this point along tangential directions
                        # We use the cell index as a proxy for position
                        # In a proper implementation, we'd use actual coordinates
                        pos_tang1 = float(idx[tang1])
                        pos_tang2 = float(idx[tang2])

                        # DFT value at this point
                        e1 = dft_e[i_pt, i_freq, e_tang1]
                        e2 = dft_e[i_pt, i_freq, e_tang2]

                        # Use the total tangential E field for diffraction amplitude
                        e_total = np.sqrt(e1 * np.conj(e1) + e2 * np.conj(e2))

                        # Phase factor for Fourier transform
                        phase = np.exp(
                            -1j * (k_y * pos_tang1 + k_z * pos_tang2)
                        )

                        # Accumulate amplitude (simplified - proper implementation
                        # would use actual physical positions and proper weighting)
                        amplitude += e_total * phase

                    orders[order_idx, i_freq] = amplitude

            order_idx += 1

    return orders, tuple(mx_out), tuple(my_out)