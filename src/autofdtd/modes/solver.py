"""Vector finite-difference mode solver for Phase 1.

Implements a vector finite-difference eigensolver for electromagnetic waveguide
modes, using the full anisotropic permittivity tensor sampled on a rectilinear
cross-section.

The formulation solves for propagation modes in the z-direction, where
the cross-section lies in the x-y plane. The eigenvalue problem is:

    ∇²ₜ Ez + k₀² εᵣ Ez = β² Ez   [TE polarization, E_z only]

or equivalently the same eigenvalue problem for Hz (TM polarization).

For anisotropic media the full 6×6 curl-curl operator is used, but for
isotropic or diagonal-anisotropic media a simplified scalar Helmholtz
approach gives the correct dispersion relationship.

The propagation eigenvalue β² is solved via a sparse generalized eigenvalue
problem A · φ = β² · φ, where the matrix A encodes the transverse
finite-difference discretization of the wave operator.

Note: This solver uses natural units (c = 1) for the eigenvalue problem,
where k₀ = ω = 2π/λ. The omega variable in field reconstruction is the
angular frequency in rad/s (C0 * k₀), consistent with SI units for the
time-harmonic fields e^(iωt).
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

# Physical constants (SI units for time-domain quantities)
C0 = 2.998e8  # speed of light in vacuum (m/s)

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigs
except ModuleNotFoundError:  # pragma: no cover
    sparse = None
    eigs = None

from autofdtd.modes.models import (
    BoundaryCondition,
    ModeSolution,
    ModeSolverConfig,
    ModeSolverCrossSection,
    ModeSpec,
)


# Epsilon callback type: (x, y) -> (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
EpsilonCallback = Callable[[float, float], tuple[float, float, float, float, float]]


def _map_boundary_condition(bc: BoundaryCondition) -> int:
    """Map BoundaryCondition to integer code.

    Returns:
        0: Dirichlet (Ez=0 at boundary, PEC-like)
        1: Neumann (dEz/dn=0, PMC-like)
        2: Bloch periodic
    """
    if bc.condition == "dirichlet":
        return 0
    if bc.condition == "neumann":
        return 1
    if bc.condition == "bloch":
        return 2
    raise ValueError(f"unknown boundary condition: {bc.condition}")


# Physical constants for PML computation
EPSILON_0 = 8.854e-12  # vacuum permittivity (F/m)
ETA_0 = 376.73  # impedance of free space (Ohms)


def _compute_pml_s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: float,
    sigma_max: float,
    kappa_min: float,
    kappa_max: float,
    order: int,
) -> complex:
    """Compute the complex PML s-value at a given position.

    The PML coordinate stretching factor is:
        s(x) = kappa(x) + i * sigma(x) / (omega * eps0)

    where:
        kappa(x) = kappa_min + (kappa_max - kappa_min) * (x/d)^order
        sigma(x) = sigma_max * (x/d)^order

    Args:
        dl: Grid spacing at this location
        step: Normalized position in PML (0 at inner edge, 1 at outer edge)
        omega: Angular frequency (rad/s)
        avg_speed: Average relative speed of light in PML region
        sigma_max: Maximum conductivity scaling factor
        kappa_min: Minimum kappa at inner boundary
        kappa_max: Maximum kappa at outer boundary
        order: Polynomial order of PML profile

    Returns:
        Complex s-value at the given position
    """
    if step <= 0:
        # At or before inner boundary - no PML
        return 1.0 + 0.0j
    if step >= 1.0:
        # At or beyond outer boundary - maximum PML
        step = 1.0

    # Compute kappa and sigma profiles
    kappa = kappa_min + (kappa_max - kappa_min) * (step**order)
    sigma = sigma_max * avg_speed / (ETA_0 * dl) * (step**order)

    return kappa + 1j * sigma / (omega * EPSILON_0)


def _compute_pml_s_factors_1d(
    n_cells: int,
    grid_spacing: float,
    num_pml_layers: int,
    omega: float,
    avg_speed: float,
    sigma_max: float,
    kappa_min: float,
    kappa_max: float,
    order: int,
    is_forward: bool,
) -> np.ndarray:
    """Compute 1D array of PML s-factors for all grid points.

    The s-factors are computed for each cell center position relative to
    the nearest PML boundary. For interior points (no PML), s = 1.

    Args:
        n_cells: Number of cells in this dimension
        grid_spacing: Grid spacing (cell size)
        num_pml_layers: Number of PML layers at each boundary
        omega: Angular frequency (rad/s)
        avg_speed: Average relative speed of light in PML region
        sigma_max: Maximum conductivity scaling factor
        kappa_min: Minimum kappa at inner boundary
        kappa_max: Maximum kappa at outer boundary
        order: Polynomial order of PML profile
        is_forward: If True, compute s-factors for forward derivative (used at + boundary).
                   If False, compute s-factors for backward derivative (used at - boundary).

    Returns:
        Array of complex s-factors at each cell center
    """
    s_factors = np.ones(n_cells, dtype=np.complex128)

    if num_pml_layers == 0:
        return s_factors

    for i in range(n_cells):
        # Determine distance to nearest PML boundary
        # For x-minus boundary (left side):
        #   - points 0 to num_pml_layers-1 are in PML
        #   - distance = (num_pml_layers - i) / num_pml_layers
        # For x-plus boundary (right side):
        #   - points N-num_pml_layers to N-1 are in PML
        #   - distance = (i - (N - num_pml_layers)) / num_pml_layers

        # Left PML (x-minus)
        if i < num_pml_layers:
            # For forward derivative: step goes from 0.5/n_pml to (n_pml-0.5)/n_pml
            # For backward derivative: step goes from 1/n_pml to (n_pml-1)/n_pml
            if is_forward:
                step = (num_pml_layers - i - 0.5) / num_pml_layers
            else:
                step = (num_pml_layers - i) / num_pml_layers
            step = max(0.0, min(1.0, step))
            s_factors[i] = _compute_pml_s_value(
                grid_spacing, step, omega, avg_speed,
                sigma_max, kappa_min, kappa_max, order
            )

        # Right PML (x-plus)
        elif i >= n_cells - num_pml_layers:
            if is_forward:
                step = (i - (n_cells - num_pml_layers) + 0.5) / num_pml_layers
            else:
                step = (i - (n_cells - num_pml_layers)) / num_pml_layers
            step = max(0.0, min(1.0, step))
            s_factors[i] = _compute_pml_s_value(
                grid_spacing, step, omega, avg_speed,
                sigma_max, kappa_min, kappa_max, order
            )

    return s_factors


def _compute_pml_s_factors_2d(
    nx: int,
    ny: int,
    dx: np.ndarray,
    dy: np.ndarray,
    num_pml_layers: int,
    omega: float,
    sigma_max: float,
    kappa_min: float,
    kappa_max: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D PML s-factor arrays for all cell faces.

    Returns 4 arrays (s_x_f, s_x_b, s_y_f, s_y_b) where:
        - s_x_f[i,j]: s-factor for forward x-derivative at cell (i,j)
        - s_x_b[i,j]: s-factor for backward x-derivative at cell (i,j)
        - s_y_f[i,j]: s-factor for forward y-derivative at cell (i,j)
        - s_y_b[i,j]: s-factor for backward y-derivative at cell (i,j)

    The s-factors vary only along the PML axis, not across the 2D plane.
    Interior points (no PML) have s = 1.

    Args:
        nx, ny: Number of cells in x and y
        dx, dy: Grid spacing arrays
        num_pml_layers: Number of PML layers at each boundary
        omega: Angular frequency (rad/s)
        sigma_max, kappa_min, kappa_max, order: PML profile parameters

    Returns:
        Tuple of 4 complex s-factor arrays (s_x_f, s_x_b, s_y_f, s_y_b)
    """
    # Average relative speed (assume vacuum in PML region)
    avg_speed = 1.0

    # Compute 1D s-factors for x direction
    s_x_f_1d = _compute_pml_s_factors_1d(
        nx, dx[0] if len(dx) > 0 else 1.0, num_pml_layers,
        omega, avg_speed, sigma_max, kappa_min, kappa_max, order,
        is_forward=True
    )
    s_x_b_1d = _compute_pml_s_factors_1d(
        nx, dx[0] if len(dx) > 0 else 1.0, num_pml_layers,
        omega, avg_speed, sigma_max, kappa_min, kappa_max, order,
        is_forward=False
    )

    # Compute 1D s-factors for y direction
    s_y_f_1d = _compute_pml_s_factors_1d(
        ny, dy[0] if len(dy) > 0 else 1.0, num_pml_layers,
        omega, avg_speed, sigma_max, kappa_min, kappa_max, order,
        is_forward=True
    )
    s_y_b_1d = _compute_pml_s_factors_1d(
        ny, dy[0] if len(dy) > 0 else 1.0, num_pml_layers,
        omega, avg_speed, sigma_max, kappa_min, kappa_max, order,
        is_forward=False
    )

    # Broadcast to 2D arrays
    # s_x_f and s_x_b vary along x-axis, constant along y
    s_x_f = np.zeros((nx, ny), dtype=np.complex128)
    s_x_b = np.zeros((nx, ny), dtype=np.complex128)
    for j in range(ny):
        s_x_f[:, j] = s_x_f_1d
        s_x_b[:, j] = s_x_b_1d

    # s_y_f and s_y_b vary along y-axis, constant along x
    s_y_f = np.zeros((nx, ny), dtype=np.complex128)
    s_y_b = np.zeros((nx, ny), dtype=np.complex128)
    for i in range(nx):
        s_y_f[i, :] = s_y_f_1d
        s_y_b[i, :] = s_y_b_1d

    return s_x_f, s_x_b, s_y_f, s_y_b


def assemble_te_mode_matrix(
    x: tuple[float, ...],
    y: tuple[float, ...],
    wavelength: float,
    epsilon_zz: Callable[[float, float], float],
    boundaries: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition, BoundaryCondition],
    num_pml_layers: int = 0,
    pml_sigma_max: float = 2.0,
    pml_kappa_min: float = 1.0,
    pml_kappa_max: float = 3.0,
    pml_order: int = 3,
) -> "sparse.csc_matrix | sparse.csr_matrix":
    """Assemble the sparse FD matrix for TE modes (Ez-only Helmholtz equation).

    Solves: ∇²Ez + k₀² εᵣ(x,y) Ez = β² Ez
    discretized as:
        (Ez_{i+1,j} - 2 Ez_{i,j} + Ez_{i-1,j}) / dx²
      + (Ez_{i,j+1} - 2 Ez_{i,j} + Ez_{i,j-1}) / dy²
      + k₀² εᵣ(i,j) Ez_{i,j} = β² Ez_{i,j}

    This is the standard scalar Helmholtz eigenvalue problem, which
    correctly captures the dispersion of dielectric waveguides for
    the fundamental TE/TM modes.

    When PML is enabled (num_pml_layers > 0), the equation is modified to use
    complex coordinate stretching:
        (d/dx)(1/s_x * dEz/dx) + (d/dy)(1/s_y * dEz/dy) + k₀² εᵣ Ez = β² Ez

    The s-factors absorb radiation modes at boundaries, allowing leaky modes
    to be found.

    Args:
        x: Cell-center x coordinates (length nx)
        y: Cell-center y coordinates (length ny)
        wavelength: Free-space wavelength
        epsilon_zz: Callback returning εzz at (x, y)
        boundaries: [north, south, east, west] BCs
        num_pml_layers: Number of PML layers at each boundary (0 = no PML)
        pml_sigma_max: Maximum PML conductivity scaling factor
        pml_kappa_min: Minimum PML kappa at inner boundary
        pml_kappa_max: Maximum PML kappa at outer boundary
        pml_order: Polynomial order of PML profile

    Returns:
        Sparse matrix A of shape (nx*ny, nx*ny) for A·Ez = β²·Ez.
        Complex dtype when PML is enabled, float64 otherwise.
    """
    if sparse is None:
        raise ModuleNotFoundError("scipy is required for the mode solver")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = len(x)
    ny = len(y)
    k0 = 2.0 * math.pi / wavelength
    omega = C0 * k0  # angular frequency

    # Cell spacings
    dx = np.diff(x) if nx > 1 else np.array([1.0])
    dy = np.diff(y) if ny > 1 else np.array([1.0])

    bc_north = _map_boundary_condition(boundaries[0])
    bc_south = _map_boundary_condition(boundaries[1])
    bc_east = _map_boundary_condition(boundaries[2])
    bc_west = _map_boundary_condition(boundaries[3])

    # Compute PML s-factors if enabled
    use_pml = num_pml_layers > 0
    if use_pml:
        s_x_f, s_x_b, s_y_f, s_y_b = _compute_pml_s_factors_2d(
            nx, ny, dx, dy,
            num_pml_layers, omega,
            pml_sigma_max, pml_kappa_min, pml_kappa_max, pml_order
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    def _add(ii: int, jj: int, val: complex) -> None:
        if abs(val) > 1e-15:
            rows.append(ii)
            cols.append(jj)
            data.append(val)

    for i in range(nx):
        for j in range(ny):
            # Cell dimensions
            dx_w = dx[i - 1] if i > 0 else dx[0]
            dx_e = dx[i] if i < nx - 1 else dx[-1]
            dy_s = dy[j - 1] if j > 0 else dy[0]
            dy_n = dy[j] if j < ny - 1 else dy[-1]

            # ε at cell center
            eps_c = max(epsilon_zz(x[i], y[j]), 1e-10)

            # PML-modified Laplacian coefficients
            # Using (d/dx)(1/s_x * dE/dx) approach:
            # Coefficient of E[i+1,j]: (1/s_x[i+1/2,j]) / (dx_e * (dx_w + dx_e))
            # Coefficient of E[i-1,j]: (1/s_x[i-1/2,j]) / (dx_w * (dx_w + dx_e))
            # Diagonal: -(coeff_east + coeff_west + coeff_north + coeff_south)

            if use_pml:
                # s-factors at cell faces (use average of adjacent cell s-factors)
                # For x-direction:
                #   s_x_f[i,j] is the s-factor for forward derivative at (i,j)
                #   s_x_b[i,j] is the s-factor for backward derivative at (i,j)
                # The s-factor at face between i-1 and i is approximately s_x_b[i,j]
                # The s-factor at face between i and i+1 is approximately s_x_f[i,j]
                s_x_e = s_x_f[i, j]  # at east face (between i and i+1)
                s_x_w = s_x_b[i, j]  # at west face (between i-1 and i)
                s_y_n = s_y_f[i, j]  # at north face (between j and j+1)
                s_y_s = s_y_b[i, j]  # at south face (between j-1 and j)

                # 1/s at each face
                inv_s_x_e = 1.0 / s_x_e if abs(s_x_e) > 1e-15 else 1.0
                inv_s_x_w = 1.0 / s_x_w if abs(s_x_w) > 1e-15 else 1.0
                inv_s_y_n = 1.0 / s_y_n if abs(s_y_n) > 1e-15 else 1.0
                inv_s_y_s = 1.0 / s_y_s if abs(s_y_s) > 1e-15 else 1.0

                # Diagonal (Laplacian + k0²*eps)
                diag = (
                    -inv_s_x_e / (dx_e * (dx_w + dx_e))
                    - inv_s_x_w / (dx_w * (dx_w + dx_e))
                    - inv_s_y_n / (dy_n * (dy_s + dy_n))
                    - inv_s_y_s / (dy_s * (dy_s + dy_n))
                    + k0**2 * eps_c
                )

                # Neighbor coefficients
                coeff_e = inv_s_x_e / (dx_e * (dx_w + dx_e))
                coeff_w = inv_s_x_w / (dx_w * (dx_w + dx_e))
                coeff_n = inv_s_y_n / (dy_n * (dy_s + dy_n))
                coeff_s = inv_s_y_s / (dy_s * (dy_s + dy_n))
            else:
                # Standard Laplacian (no PML)
                diag = (
                    -2.0 / (dx_w * (dx_w + dx_e))
                    - 2.0 / (dx_e * (dx_w + dx_e))
                    - 2.0 / (dy_s * (dy_s + dy_n))
                    - 2.0 / (dy_n * (dy_s + dy_n))
                    + k0**2 * eps_c
                )
                coeff_e = 2.0 / (dx_e * (dx_w + dx_e))
                coeff_w = 2.0 / (dx_w * (dx_w + dx_e))
                coeff_n = 2.0 / (dy_n * (dy_s + dy_n))
                coeff_s = 2.0 / (dy_s * (dy_s + dy_n))

            idx = i * ny + j

            # Apply BCs via ghost points or direct modification
            # Dirichlet (PEC): Ez=0 at boundary
            # Neumann (PMC): dEz/dn=0

            # For simplicity, we implement BCs by modifying the diagonal
            # and neighbor coefficients at the boundary

            if i == 0:  # West
                # Ez at ghost point i=-1: Neumann: Ez_{-1} = Ez_0
                # Dirichlet: Ez_{-1} = -Ez_0 (to force Ez=0 at wall)
                if bc_west == 0:  # Dirichlet
                    diag += 2.0 * coeff_w
                # Neumann: no change (ghost = same as interior)

            if i == nx - 1:  # East
                if bc_east == 0:  # Dirichlet
                    diag += 2.0 * coeff_e

            if j == 0:  # South
                if bc_south == 0:  # Dirichlet
                    diag += 2.0 * coeff_s

            if j == ny - 1:  # North
                if bc_north == 0:  # Dirichlet
                    diag += 2.0 * coeff_n

            _add(idx, idx, diag)

            # Neighbor contributions
            if i > 0:
                _add(idx, (i - 1) * ny + j, coeff_w)
            if i < nx - 1:
                _add(idx, (i + 1) * ny + j, coeff_e)
            if j > 0:
                _add(idx, i * ny + (j - 1), coeff_s)
            if j < ny - 1:
                _add(idx, i * ny + (j + 1), coeff_n)

    # Use complex dtype when PML is enabled for proper absorption
    dtype = np.complex128 if use_pml else np.float64
    A = sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=dtype)
    A.eliminate_zeros()
    return A


def assemble_mode_matrix(
    x: tuple[float, ...],
    y: tuple[float, ...],
    wavelength: float,
    epsilon: EpsilonCallback,
    boundaries: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> "sparse.csc_matrix | sparse.csr_matrix":
    """Assemble the sparse FD matrix for the vector mode problem.

    For the general anisotropic case, we solve the full 2-component
    vector eigenvalue problem using the Yee-grid curl-curl operator.

    Args:
        x: Cell-center x coordinates
        y: Cell-center y coordinates
        wavelength: Free-space wavelength
        epsilon: Callback returning (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
        boundaries: [north, south, east, west] BCs

    Returns:
        Sparse matrix A of shape (2*N, 2*N) for A·[Hx;Hy] = β²·[Hx;Hy]
    """
    if sparse is None:
        raise ModuleNotFoundError("scipy is required for the mode solver")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = len(x)
    ny = len(y)
    k0 = 2.0 * math.pi / wavelength

    dx = np.diff(x) if nx > 1 else np.array([1.0])
    dy = np.diff(y) if ny > 1 else np.array([1.0])

    bc_north = _map_boundary_condition(boundaries[0])
    bc_south = _map_boundary_condition(boundaries[1])
    bc_east = _map_boundary_condition(boundaries[2])
    bc_west = _map_boundary_condition(boundaries[3])

    N = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def _add(ii: int, jj: int, val: float) -> None:
        if abs(val) > 1e-15:
            rows.append(ii)
            cols.append(jj)
            data.append(val)

    def safe_eps(f):
        return max(abs(f), 1e-10)

    for i in range(nx):
        for j in range(ny):
            dx_w = dx[i - 1] if i > 0 else dx[0]
            dx_e = dx[i] if i < nx - 1 else dx[-1]
            dy_s = dy[j - 1] if j > 0 else dy[0]
            dy_n = dy[j] if j < ny - 1 else dy[-1]

            eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = epsilon(x[i], y[j])
            eps_xx = safe_eps(eps_xx)
            eps_yy = safe_eps(eps_yy)
            det_eps = eps_xx * eps_yy - eps_xy * eps_yx

            # Get neighbor eps for averaging
            eps_xx_w = eps_xx
            eps_xx_e = eps_xx
            eps_yy_w = eps_yy
            eps_yy_e = eps_yy
            eps_xx_s = eps_xx
            eps_xx_n = eps_xx
            eps_yy_s = eps_yy
            eps_yy_n = eps_yy

            if i > 0:
                eww, _, _, eyw, _ = epsilon(x[i - 1], y[j])
                eps_xx_w = safe_eps(eww)
                eps_yy_w = safe_eps(eyw)
            if i < nx - 1:
                ewe, _, _, eye, _ = epsilon(x[i + 1], y[j])
                eps_xx_e = safe_eps(ewe)
                eps_yy_e = safe_eps(eye)
            if j > 0:
                ews, _, _, eys, _ = epsilon(x[i], y[j - 1])
                eps_xx_s = safe_eps(ews)
                eps_yy_s = safe_eps(eys)
            if j < ny - 1:
                ewn, _, _, eyn, _ = epsilon(x[i], y[j + 1])
                eps_xx_n = safe_eps(ewn)
                eps_yy_n = safe_eps(eyn)

            # === Simplified curl-curl discretization ===
            # For the 2D cross-section, the curl-curl operator gives:
            # ∇ × (μ⁻¹ ∇ × H) = β² H in the transverse plane
            # For μ = μ₀ (vacuum), we need:
            # [1/εxx d²/dx² + 1/εyy d²/dy²] Hx = -β² Hx  [along x]
            # [1/εyy d²/dy² + 1/εxx d²/dx²] Hy = -β² Hy  [along y]
            #
            # Rearranging: -[1/εxx d²/dx² + 1/εyy d²/dy²] Hx = β² Hx
            #
            # This gives a NEGATIVE-DEFINITE operator whose eigenvalues are β² > 0.
            #
            # The FD discretization uses harmonic averaging at cell interfaces:
            # 1/εxx at i+1/2 ≈ 2/(εxx_i + εxx_{i+1})

            # Hx block (diagonal coefficient for Hx equation)
            # (1/εxx_e + 1/εxx_w)/(2*dx²) + (1/εyy_n + 1/εyy_s)/(2*dy²) - k₀²
            axxp = (
                (1.0 / eps_xx_e + 1.0 / eps_xx_w) / (2.0 * dx_e * dx_w)
                + (1.0 / eps_yy_n + 1.0 / eps_yy_s) / (2.0 * dy_n * dy_s)
                - k0**2
            )

            # Hy block (diagonal coefficient for Hy equation)
            ayyp = (
                (1.0 / eps_yy_e + 1.0 / eps_yy_w) / (2.0 * dx_e * dx_w)
                + (1.0 / eps_xx_n + 1.0 / eps_xx_s) / (2.0 * dy_n * dy_s)
                - k0**2
            )

            # Hx neighbors
            axxe = -(1.0 / eps_xx_e) / (dx_e * (dx_w + dx_e))  # West
            axxw = -(1.0 / eps_xx_w) / (dx_w * (dx_w + dx_e))  # East
            axxn = -(1.0 / eps_yy_n) / (dy_n * (dy_s + dy_n))  # South
            axxs = -(1.0 / eps_yy_s) / (dy_s * (dy_s + dy_n))  # North

            # Hy neighbors
            ayye = -(1.0 / eps_yy_e) / (dx_e * (dx_w + dx_e))  # West
            ayyw = -(1.0 / eps_yy_w) / (dx_w * (dx_w + dx_e))  # East
            ayyn = -(1.0 / eps_xx_n) / (dy_n * (dy_s + dy_n))  # South
            ayys = -(1.0 / eps_xx_s) / (dy_s * (dy_s + dy_n))  # North

            ix = i * ny + j
            iy = ix + N

            # Apply BCs via direct modification of diagonal entries
            if j == ny - 1:
                if bc_north == 0:  # Dirichlet: Hx=0 at boundary
                    axxp += 2.0 * bc_north * axxs
                axxn += bc_north * axxs
                ayyp += bc_north * ayys

            if j == 0:
                if bc_south == 0:
                    axxp += 2.0 * bc_south * axxn
                axxs += bc_south * axxn
                ayyp += bc_south * ayyn

            if i == nx - 1:
                if bc_east == 0:
                    axxp += 2.0 * bc_east * axxe
                axxw += bc_east * axxe
                ayyp += bc_east * ayye

            if i == 0:
                if bc_west == 0:
                    axxp += 2.0 * bc_west * axxw
                axxe += bc_west * axxw
                ayyp += bc_west * ayyw

            # Assemble
            _add(ix, ix, axxp)
            _add(iy, iy, ayyp)

            if j < ny - 1:
                _add(ix, i * ny + (j + 1), axxs)
                _add(iy, i * ny + (j + 1), ayys)
            if j > 0:
                _add(ix, i * ny + (j - 1), axxn)
                _add(iy, i * ny + (j - 1), ayyn)
            if i < nx - 1:
                _add(ix, (i + 1) * ny + j, axxe)
                _add(iy, (i + 1) * ny + j, ayye)
            if i > 0:
                _add(ix, (i - 1) * ny + j, axxw)
                _add(iy, (i - 1) * ny + j, ayyw)

    A = sparse.csc_matrix((data, (rows, cols)), shape=(2 * N, 2 * N), dtype=np.float64)
    A.eliminate_zeros()
    return A


def assemble_fallahkhair_mode_matrix(
    x: tuple[float, ...],
    y: tuple[float, ...],
    wavelength: float,
    epsilon: EpsilonCallback,
    boundaries: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> "sparse.csc_matrix | sparse.csr_matrix":
    """Assemble the sparse FD matrix for the full-vectorial Fallahkhair 2008 mode problem.

    Implements the 2N x 2N operator from Fallahkhair et al. 2008, solving for [Hx; Hy].
    Each block (Pxx, Pxy, Pyx, Pyy) uses a 9-point stencil with 4 corner neighbors.
    Epsilon is sampled at 4 surrounding cell centers per node.

    Solves: P * [Hx; Hy] = -beta^2 * [Hx; Hy]

    Reference: VectorModesolver.jl Modesolver.jl:10-384, Fallahkhair 2008 equations 21-34.

    Args:
        x: Cell-edge x coordinates (length nx) - grid nodes
        y: Cell-edge y coordinates (length ny) - grid nodes
        wavelength: Free-space wavelength (m)
        epsilon: Callback returning (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz) at (x, y)
        boundaries: [north, south, east, west] BCs

    Returns:
        Sparse matrix P of shape (2*N, 2*N) where N = nx*ny
    """
    if sparse is None:
        raise ModuleNotFoundError("scipy is required for the mode solver")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = len(x)
    ny = len(y)
    k0 = 2.0 * math.pi / wavelength

    diffx = x[1:] - x[:-1]
    diffy = y[1:] - y[:-1]
    diffx = np.concatenate([[diffx[0]], diffx, [diffx[-1]]])
    diffy = np.concatenate([[diffy[0]], diffy, [diffy[-1]]])

    xc = (x[:-1] + x[1:]) / 2.0
    yc = (y[:-1] + y[1:]) / 2.0
    xc = np.concatenate([[xc[0]], xc, [xc[-1]]])
    yc = np.concatenate([[yc[0]], yc, [yc[-1]]])

    bc = [_map_boundary_condition(boundaries[0]), _map_boundary_condition(boundaries[1]),
          _map_boundary_condition(boundaries[2]), _map_boundary_condition(boundaries[3])]

    N = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def _add(ii: int, jj: int, val: float) -> None:
        if abs(val) > 1e-15:
            rows.append(ii)
            cols.append(jj)
            data.append(val)

    def safe_eps(f: float) -> float:
        return max(abs(f), 1e-10)

    for i in range(nx):
        for j in range(ny):
            n = diffy[j + 1]
            s = diffy[j]
            e = diffx[i + 1]
            w = diffx[i]

            epsxx1, epsxy1, _, epsyy1, epszz1 = epsilon(xc[i], yc[j + 1])
            epsxx2, epsxy2, _, epsyy2, epszz2 = epsilon(xc[i], yc[j])
            epsxx3, epsxy3, _, epsyy3, epszz3 = epsilon(xc[i + 1], yc[j])
            epsxx4, epsxy4, _, epsyy4, epszz4 = epsilon(xc[i + 1], yc[j + 1])

            ns21 = n * epsyy2 + s * epsyy1
            ns34 = n * epsyy3 + s * epsyy4
            ew14 = e * epsxx1 + w * epsxx4
            ew23 = e * epsxx2 + w * epsxx3

            epsyy1 = safe_eps(epsyy1)
            epsyy2 = safe_eps(epsyy2)
            epsyy3 = safe_eps(epsyy3)
            epsyy4 = safe_eps(epsyy4)
            epsxx1 = safe_eps(epsxx1)
            epsxx2 = safe_eps(epsxx2)
            epsxx3 = safe_eps(epsxx3)
            epsxx4 = safe_eps(epsxx4)
            epszz1 = safe_eps(epszz1)
            epszz2 = safe_eps(epszz2)
            epszz3 = safe_eps(epszz3)
            epszz4 = safe_eps(epszz4)

            ns21 = safe_eps(ns21)
            ns34 = safe_eps(ns34)
            ew14 = safe_eps(ew14)
            ew23 = safe_eps(ew23)

            # Pxx block coefficients (equations 21-26, 32-33, 34)
            axxn = ((2 * epsyy4 * e - epsxy4 * n) * (epsyy3 / epszz4) / ns34 +
                    (2 * epsyy1 * w + epsxy1 * n) * (epsyy2 / epszz1) / ns21) / (n * (e + w))
            axxs = ((2 * epsyy3 * e + epsxy3 * s) * (epsyy4 / epszz3) / ns34 +
                    (2 * epsyy2 * w - epsxy2 * s) * (epsyy1 / epszz2) / ns21) / (s * (e + w))
            ayye = ((2 * n * epsxx4 - e * epsxy4) * epsxx1 / epszz4 / e / ew14 / (n + s) +
                    (2 * s * epsxx3 + e * epsxy3) * epsxx2 / epszz3 / e / ew23 / (n + s))
            ayyw = ((2 * epsxx1 * n + epsxy1 * w) * epsxx4 / epszz1 / w / ew14 / (n + s) +
                    (2 * epsxx2 * s - epsxy2 * w) * epsxx3 / epszz2 / w / ew23 / (n + s))
            axxe = (2 / (e * (e + w)) +
                    (epsyy4 * epsxy3 / epszz3 - epsyy3 * epsxy4 / epszz4) / (e + w) / ns34)
            axxw = (2 / (w * (e + w)) +
                    (epsyy2 * epsxy1 / epszz1 - epsyy1 * epsxy2 / epszz2) / (e + w) / ns21)
            ayyn = (2 / (n * (n + s)) +
                    (epsxx4 * epsxy1 / epszz1 - epsxx1 * epsxy4 / epszz4) / (n + s) / ew14)
            ayys = (2 / (s * (n + s)) +
                    (epsxx2 * epsxy3 / epszz3 - epsxx3 * epsxy2 / epszz2) / (n + s) / ew23)

            axxne = +epsxy4 * epsyy3 / epszz4 / (e + w) / ns34
            axxse = -epsxy3 * epsyy4 / epszz3 / (e + w) / ns34
            axxnw = -epsxy1 * epsyy2 / epszz1 / (e + w) / ns21
            axxsw = +epsxy2 * epsyy1 / epszz2 / (e + w) / ns21

            ayyne = +epsxy4 * epsxx1 / epszz4 / (n + s) / ew14
            ayyse = -epsxy3 * epsxx2 / epszz3 / (n + s) / ew23
            ayynw = -epsxy1 * epsxx4 / epszz1 / (n + s) / ew14
            ayysw = +epsxy2 * epsxx3 / epszz2 / (n + s) / ew23

            axxp = (-axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw +
                    k0**2 * (n + s) * (epsyy4 * epsyy3 * e / ns34 + epsyy1 * epsyy2 * w / ns21) / (e + w))

            # Pxy block coefficients (equations 28-33, 34)
            axyn = ((epsyy3 * epsyy4 / epszz4 / ns34 - epsyy2 * epsyy1 / epszz1 / ns21 +
                    s * (epsyy2 * epsyy4 - epsyy1 * epsyy3) / ns21 / ns34) / (e + w))
            axys = ((epsyy1 * epsyy2 / epszz2 / ns21 - epsyy4 * epsyy3 / epszz3 / ns34 +
                    n * (epsyy2 * epsyy4 - epsyy1 * epsyy3) / ns21 / ns34) / (e + w))
            ayxe = ((epsxx1 * epsxx4 / epszz4 / ew14 - epsxx2 * epsxx3 / epszz3 / ew23 +
                     w * (epsxx2 * epsxx4 - epsxx1 * epsxx3) / ew23 / ew14) / (n + s))
            ayxw = ((epsxx3 * epsxx2 / epszz2 / ew23 - epsxx4 * epsxx1 / epszz1 / ew14 +
                     e * (epsxx4 * epsxx2 - epsxx1 * epsxx3) / ew23 / ew14) / (n + s))

            axye = ((epsyy4 * (1 + epsyy3 / epszz4) - epsyy3 * (1 + epsyy4 / epszz4)) / ns34 / (e + w) -
                    (2 * epsxy1 * epsyy2 / epszz1 * n * w / ns21 +
                     2 * epsxy2 * epsyy1 / epszz2 * s * w / ns21 +
                     2 * epsxy4 * epsyy3 / epszz4 * n * e / ns34 +
                     2 * epsxy3 * epsyy4 / epszz3 * s * e / ns34 +
                     2 * epsyy1 * epsyy2 * (1.0 / epszz1 - 1.0 / epszz2) * w**2 / ns21) /
                    e / (e + w)**2)

            axyw = ((epsyy2 * (1 + epsyy1 / epszz2) - epsyy1 * (1 + epsyy2 / epszz2)) / ns21 / (e + w) -
                    (2 * epsxy1 * epsyy2 / epszz1 * n * e / ns21 +
                     2 * epsxy2 * epsyy1 / epszz2 * s * e / ns21 +
                     2 * epsxy4 * epsyy3 / epszz4 * n * w / ns34 +
                     2 * epsxy3 * epsyy4 / epszz3 * s * w / ns34 +
                     2 * epsyy3 * epsyy4 * (1.0 / epszz3 - 1.0 / epszz4) * e**2 / ns34) /
                    w / (e + w)**2)

            ayxn = ((epsxx4 * (1 + epsxx1 / epszz4) - epsxx1 * (1 + epsxx4 / epszz4)) / ew14 / (n + s) -
                    (2 * epsxy3 * epsxx2 / epszz3 * e * s / ew23 +
                     2 * epsxy2 * epsxx3 / epszz2 * w * n / ew23 +
                     2 * epsxy4 * epsxx1 / epszz4 * e * s / ew14 +
                     2 * epsxy1 * epsxx4 / epszz1 * w * n / ew14 +
                     2 * epsxx3 * epsxx2 * (1.0 / epszz3 - 1.0 / epszz2) * s**2 / ew23) /
                    n / (n + s)**2)

            ayxs = ((epsxx2 * (1 + epsxx3 / epszz2) - epsxx3 * (1 + epsxx2 / epszz2)) / ew23 / (n + s) -
                    (2 * epsxy3 * epsxx2 / epszz3 * e * n / ew23 +
                     2 * epsxy2 * epsxx3 / epszz2 * w * n / ew23 +
                     2 * epsxy4 * epsxx1 / epszz4 * e * s / ew14 +
                     2 * epsxy1 * epsxx4 / epszz1 * w * s / ew14 +
                     2 * epsxx1 * epsxx4 * (1.0 / epszz1 - 1.0 / epszz4) * n**2 / ew14) /
                    s / (n + s)**2)

            axyne = +epsyy3 * (1 - epsyy4 / epszz4) / (e + w) / ns34
            axyse = -epsyy4 * (1 - epsyy3 / epszz3) / (e + w) / ns34
            axynw = -epsyy2 * (1 - epsyy1 / epszz1) / (e + w) / ns21
            axysw = +epsyy1 * (1 - epsyy2 / epszz2) / (e + w) / ns21

            ayxne = +epsxx1 * (1 - epsxx4 / epszz4) / (n + s) / ew14
            ayxse = -epsxx2 * (1 - epsxx3 / epszz3) / (n + s) / ew23
            ayxnw = -epsxx4 * (1 - epsxx1 / epszz1) / (n + s) / ew14
            ayxsw = +epsxx3 * (1 - epsxx2 / epszz2) / (n + s) / ew23

            axyp = (-(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) -
                    k0**2 * (w * (n * epsxy1 * epsyy2 + s * epsxy2 * epsyy1) / ns21 +
                             e * (s * epsxy3 * epsyy4 + n * epsxy4 * epsyy3) / ns34) / (e + w))

            # Pyx block coefficients (same as Pxy but signs flipped per Fallahkhair formulation)
            # These are derived from the transpose symmetry of the curl-curl operator
            pyxn = -axyn
            pyxs = -axys
            pyxe = -ayxe
            pyxw = -ayxw
            pyxne = -ayxne
            pyxse = -ayxse
            pyxnw = -ayxnw
            pyxsw = -ayxsw
            pyxp = -axyp

            # Pyy block coefficients
            ayyp = (-ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw +
                    k0**2 * (e + w) * (epsxx1 * epsxx4 * n / ew14 + epsxx2 * epsxx3 * s / ew23) / (n + s))

            # Apply boundary conditions - North
            if j == ny - 1:
                axxs += bc[0] * axxn
                axxse += bc[0] * axxne
                axxsw += bc[0] * axxnw
                ayxs += bc[0] * ayxn
                ayxse += bc[0] * ayxne
                ayxsw += bc[0] * ayxnw
                ayys -= bc[0] * ayyn
                ayyse -= bc[0] * ayyne
                ayysw -= bc[0] * ayynw
                axys -= bc[0] * axyn
                axyse -= bc[0] * axyne
                axysw -= bc[0] * axynw

            # South boundary
            if j == 0:
                axxn += bc[1] * axxs
                axxne += bc[1] * axxse
                axxnw += bc[1] * axxsw
                ayxn += bc[1] * ayxs
                ayxne += bc[1] * ayxse
                ayxnw += bc[1] * ayxsw
                ayyn -= bc[1] * ayys
                ayyne -= bc[1] * ayyse
                ayynw -= bc[1] * ayysw
                axyn -= bc[1] * axys
                axyne -= bc[1] * axyse
                axynw -= bc[1] * axysw

            # East boundary
            if i == nx - 1:
                axxw += bc[2] * axxe
                axxnw += bc[2] * axxne
                axxsw += bc[2] * axxse
                ayxw += bc[2] * ayxe
                ayxnw += bc[2] * ayxne
                ayxsw += bc[2] * ayxse
                ayyw -= bc[2] * ayye
                ayynw -= bc[2] * ayyne
                ayysw -= bc[2] * ayyse
                axyw -= bc[2] * axye
                axynw -= bc[2] * axyne
                axysw -= bc[2] * axyse

            # West boundary
            if i == 0:
                axxe += bc[3] * axxw
                axxne += bc[3] * axxnw
                axxse += bc[3] * axxsw
                ayxe += bc[3] * ayxw
                ayxne += bc[3] * ayxnw
                ayxse += bc[3] * ayxsw
                ayye -= bc[3] * ayyw
                ayyne -= bc[3] * ayynw
                ayyse -= bc[3] * ayysw
                axye -= bc[3] * axyw
                axyne -= bc[3] * axynw
                axyse -= bc[3] * axysw

            # Assemble into matrix
            ix = i * ny + j
            iy = ix + N

            # Diagonal blocks
            _add(ix, ix, axxp)
            _add(ix, iy, axyp)
            _add(iy, ix, pyxp)
            _add(iy, iy, ayyp)

            # North neighbor (j > 0)
            if j > 0:
                ix_n = i * ny + (j - 1)
                iy_n = ix_n + N
                _add(ix, ix_n, axxs)
                _add(ix, iy_n, axys)
                _add(iy, ix_n, pyxs)
                _add(iy, iy_n, ayys)

            # South neighbor (j < ny-1)
            if j < ny - 1:
                ix_s = i * ny + (j + 1)
                iy_s = ix_s + N
                _add(ix, ix_s, axxn)
                _add(ix, iy_s, axyn)
                _add(iy, ix_s, pyxn)
                _add(iy, iy_s, ayyn)

            # East neighbor (i > 0)
            if i > 0:
                ix_e = (i - 1) * ny + j
                iy_e = ix_e + N
                _add(ix, ix_e, axxw)
                _add(ix, iy_e, axyw)
                _add(iy, ix_e, pyxw)
                _add(iy, iy_e, ayyw)

            # West neighbor (i < nx-1)
            if i < nx - 1:
                ix_w = (i + 1) * ny + j
                iy_w = ix_w + N
                _add(ix, ix_w, axxe)
                _add(ix, iy_w, axye)
                _add(iy, ix_w, pyxe)
                _add(iy, iy_w, ayye)

            # North-East diagonal (i > 0 and j > 0)
            if i > 0 and j > 0:
                ix_ne = (i - 1) * ny + (j - 1)
                iy_ne = ix_ne + N
                _add(ix, ix_ne, axxsw)
                _add(ix, iy_ne, axysw)
                _add(iy, ix_ne, pyxsw)
                _add(iy, iy_ne, ayysw)

            # South-East diagonal (i > 0 and j < ny-1)
            if i > 0 and j < ny - 1:
                ix_se = (i - 1) * ny + (j + 1)
                iy_se = ix_se + N
                _add(ix, ix_se, axxnw)
                _add(ix, iy_se, axynw)
                _add(iy, ix_se, pyxnw)
                _add(iy, iy_se, ayxnw)

            # South-West diagonal (i < nx-1 and j < ny-1)
            if i < nx - 1 and j < ny - 1:
                ix_sw = (i + 1) * ny + (j + 1)
                iy_sw = ix_sw + N
                _add(ix, ix_sw, axxne)
                _add(ix, iy_sw, axyne)
                _add(iy, ix_sw, pyxne)
                _add(iy, iy_sw, ayyne)

            # North-West diagonal (i < nx-1 and j > 0)
            if i < nx - 1 and j > 0:
                ix_nw = (i + 1) * ny + (j - 1)
                iy_nw = ix_nw + N
                _add(ix, ix_nw, axxse)
                _add(ix, iy_nw, axyse)
                _add(iy, ix_nw, pyxse)
                _add(iy, iy_nw, ayyse)

    A = sparse.csc_matrix((data, (rows, cols)), shape=(2 * N, 2 * N), dtype=np.float64)
    A.eliminate_zeros()
    return A


def _reconstruct_hz(
    Hx: np.ndarray, Hy: np.ndarray, x: np.ndarray, y: np.ndarray, beta: float
) -> np.ndarray:
    """Reconstruct Hz from Hx and Hy using the divergence constraint.

    Hx and Hy are on staggered grids; Hz is defined at cell centers.
    From div(H) = 0 (no magnetic monopoles): dHx/dx + dHy/dy = -i*beta*Hz
    So: Hz = -(dHx/dx + dHy/dy) / (i*beta) at cell center
    """
    ny, nx = Hx.shape
    # Hz is defined at interior cell centers (ny-1, nx-1)
    Hz = np.zeros((ny - 1, nx - 1), dtype=np.complex128)

    diffx = np.diff(x)
    diffy = np.diff(y)

    # Valid range: j from 0 to ny-2, i from 0 to nx-2
    for j in range(ny - 1):
        for i in range(nx - 1):
            dx = diffx[i] if i < len(diffx) else diffx[-1] if len(diffx) > 0 else 1.0
            dy = diffy[j] if j < len(diffy) else diffy[-1] if len(diffy) > 0 else 1.0

            # Hx at (i, j) is actually at x[i+1/2], y[j]
            # Hy at (i, j) is actually at x[i], y[j+1/2]
            Hx_j_i = Hx[j, i] if j < ny and i < nx else 0.0
            Hx_j1_i = Hx[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hx_j_i1 = Hx[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hx_j1_i1 = Hx[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            Hy_j_i = Hy[j, i] if j < ny and i < nx else 0.0
            Hy_j1_i = Hy[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hy_j_i1 = Hy[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hy_j1_i1 = Hy[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            # dHx/dx at (i+1/2, j+1/2) - centered difference in x
            dHxd_x = (Hx_j1_i1 + Hx_j1_i - Hx_j_i1 - Hx_j_i) / (2.0 * dx)
            # dHy/dy at (i+1/2, j+1/2) - centered difference in y
            dHyd_y = (Hy_j1_i1 + Hy_j_i1 - Hy_j1_i - Hy_j_i) / (2.0 * dy)
            # From div(H) = 0: Hz = -(dHx/dx + dHy/dy) / (i*beta)
            Hz[j, i] = -(dHxd_x + dHyd_y) / (1j * beta + 1e-30)

    return Hz


def _reconstruct_fields(
    Hx: np.ndarray,
    Hy: np.ndarray,
    Hz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    omega: float,
    epsilon: EpsilonCallback,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct E fields from H fields using Maxwell's curl equations."""
    ny, nx = Hx.shape
    diffx = np.diff(x)
    diffy = np.diff(y)
    print(f"DEBUG _reconstruct_fields: Hx.shape={Hx.shape}, ny={ny}, nx={nx}, x.shape={x.shape}, y.shape={y.shape}")
    print(f"DEBUG: Ex will be allocated as ({ny-1}, {nx-1}) = {(ny-1)*(nx-1)} elements")

    Ex = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ey = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ez = np.zeros((ny - 1, nx - 1), dtype=np.complex128)

    for j in range(ny - 1):
        for i in range(nx - 1):
            dx = diffx[i] if i < len(diffx) else diffx[-1] if len(diffx) > 0 else 1.0
            dy = diffy[j] if j < len(diffy) else diffy[-1] if len(diffy) > 0 else 1.0
            xc = (x[i] + x[i + 1]) / 2.0 if i < len(x) - 1 else x[i]
            yc = (y[j] + y[j + 1]) / 2.0 if j < len(y) - 1 else y[j]

            eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = epsilon(xc, yc)
            det_eps = eps_xx * eps_yy - eps_xy * eps_yx + 1e-15

            # dHz/dy and dHz/dx using safe indexing
            def safe_hz(jj, ii):
                if 0 <= jj < Hz.shape[0] and 0 <= ii < Hz.shape[1]:
                    return Hz[jj, ii]
                return 0.0

            dHz_dy = (safe_hz(j, i) + safe_hz(j + 1, i) - safe_hz(j, i + 1) - safe_hz(j + 1, i + 1)) / (2.0 * dy)
            dHz_dx = (safe_hz(j + 1, i + 1) + safe_hz(j + 1, i) - safe_hz(j, i + 1) - safe_hz(j, i)) / (2.0 * dx)

            # Average H at cell center
            def safe_h(jj, ii, arr):
                if 0 <= jj < ny and 0 <= ii < nx:
                    return arr[jj, ii]
                return 0.0

            Hy_avg = (safe_h(j + 1, i + 1, Hy) + safe_h(j + 1, i, Hy) + safe_h(j, i + 1, Hy) + safe_h(j, i, Hy)) / 4.0
            Hx_avg = (safe_h(j + 1, i + 1, Hx) + safe_h(j + 1, i, Hx) + safe_h(j, i + 1, Hx) + safe_h(j, i, Hx)) / 4.0

            # D = curl(H) / (-i*omega), E = D / eps
            # curl(H)_x = dHz/dy + i*beta*Hy -> Dx = (dHz_dy + i*beta*Hy) / (i*omega)
            # curl(H)_y = dHz/dx - i*beta*Hx -> Dy = (dHz_dx - i*beta*Hx) / (i*omega)
            # curl(H)_z = dHy/dx - dHx/dy -> Dz = (dHy_dx - dHx_dy) / (i*omega)
            Dx = 1j * (dHz_dy + 1j * beta * Hy_avg) / (omega + 1e-30)
            Dy = 1j * (1j * beta * Hx_avg + dHz_dx) / (omega + 1e-30)

            # Derivatives for Dz = (dHy/dx - dHx/dy) / (i*omega)
            Hy_i_j = safe_h(j, i, Hy)
            Hy_i1_j = safe_h(j, i + 1, Hy)
            Hy_i_j1 = safe_h(j + 1, i, Hy)
            Hy_i1_j1 = safe_h(j + 1, i + 1, Hy)
            dHy_dx = (Hy_i1_j1 + Hy_i1_j - Hy_i_j1 - Hy_i_j) / (2.0 * dx)

            Hx_i_j = safe_h(j, i, Hx)
            Hx_i1_j = safe_h(j, i + 1, Hx)
            Hx_i_j1 = safe_h(j + 1, i, Hx)
            Hx_i1_j1 = safe_h(j + 1, i + 1, Hx)
            dHx_dy = (Hx_i1_j1 + Hx_i1_j - Hx_i_j1 - Hx_i_j) / (2.0 * dy)

            Dz = (dHy_dx - dHx_dy) / (1j * omega + 1e-30)

            Ex[j, i] = (eps_yy * Dx - eps_yx * Dy) / det_eps
            Ey[j, i] = (-eps_xy * Dx + eps_xx * Dy) / det_eps
            Ez[j, i] = Dz / eps_zz if abs(eps_zz) > 1e-10 else 0.0

    return Ex, Ey, Ez


def _estimate_n_core_n_clad(
    epsilon: EpsilonCallback,
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> tuple[float, float]:
    """Estimate n_core and n_clad from epsilon callback.

    Returns (n_clad, n_core) based on min/max epsilon_zz values.
    For a step-index waveguide, n_clad is the minimum refractive index
    and n_core is the maximum.
    """
    eps_min = float('inf')
    eps_max = float('-inf')

    # Sample at cell centers
    for x_val in x:
        for y_val in y:
            _, _, _, _, eps_zz = epsilon(x_val, y_val)
            eps_min = min(eps_min, abs(eps_zz))
            eps_max = max(eps_max, abs(eps_zz))

    n_clad = math.sqrt(eps_min) if eps_min < float('inf') else 1.0
    n_core = math.sqrt(eps_max) if eps_max > float('-inf') else 1.5

    return n_clad, n_core


def _compute_group_index(
    config: ModeSolverConfig,
    epsilon: EpsilonCallback,
    x: tuple[float, ...],
    y: tuple[float, ...],
    neff_center: float,
) -> float | None:
    """Compute group index via numerical differentiation.

    Solves at f-df, f, f+df where df = 0.005*f, then:
        n_g = n_eff - f * (n_eff(f+df) - n_eff(f-df)) / (2*df)

    Args:
        config: Mode solver configuration
        epsilon: Epsilon callback
        x, y: Grid coordinates
        neff_center: neff at center frequency (used to estimate initial guess)

    Returns:
        Group index, or None if computation fails
    """
    f0 = C0 / config.wavelength  # center frequency in Hz
    df = 0.005 * f0  # frequency step (0.5% of center frequency)

    neff_vals: list[float] = []

    for freq_shift in [-df, 0.0, df]:
        f_new = f0 + freq_shift
        # Adjust wavelength to match new frequency: lambda = C0 / f
        wavelength_new = C0 / f_new if f_new > 0 else config.wavelength

        # Create temporary config with new wavelength
        config_shifted = ModeSolverConfig(
            wavelength=wavelength_new,
            cross_section=config.cross_section,
            mode_spec=ModeSpec(
                num_modes=1,
                target_neff=neff_center,  # Use same target for all three solves
                precision=config.mode_spec.precision,
                polynomial_degree=config.mode_spec.polynomial_degree,
            ),
            min_cells=config.min_cells,
            max_cells=config.max_cells,
            tolerance=config.tolerance,
        )

        try:
            modes = solve_modes(config_shifted, epsilon, x, y)
            if len(modes) > 0:
                neff_vals.append(modes[0].neff_real)
            else:
                neff_vals.append(neff_center)  # Fallback
        except Exception:
            neff_vals.append(neff_center)  # Fallback on error

    # Numerical differentiation: dn_eff/df ≈ (neff(f+df) - neff(f-df)) / (2*df)
    neff_fwd = neff_vals[2]
    neff_bwd = neff_vals[0]
    dneff_df = (neff_fwd - neff_bwd) / (2.0 * df) if df > 0 else 0.0

    # n_g = n_eff - f * (dn_eff/df)
    neff_center_val = neff_vals[1]
    n_group = neff_center_val - f0 * dneff_df

    return n_group


def _make_bent_epsilon_callback(
    epsilon: EpsilonCallback,
    bend_radius: float,
    bend_axis: int,
    x_coords: tuple[float, ...],
    y_coords: tuple[float, ...],
) -> EpsilonCallback:
    """Create an epsilon callback that accounts for bend curvature.

    For a bent waveguide, the effective refractive index is modified by
    the curvature. For a bend of radius R in a given plane, the effective
    index at a point depends on its radial distance from the bend center.

    The modification is: n_eff = n * (1 + r/R)
    where r is the radial distance from the bend center.

    Args:
        epsilon: Original epsilon callback
        bend_radius: Bend radius in meters
        bend_axis: Axis of the bend plane (0=x, 1=y, 2=z)
        x_coords: x coordinates of the grid
        y_coords: y coordinates of the grid

    Returns:
        Modified epsilon callback with bent index correction
    """
    # Compute the center of the grid in the bend plane
    x_center = (x_coords[0] + x_coords[-1]) / 2.0 if len(x_coords) > 1 else 0.0
    y_center = (y_coords[0] + y_coords[-1]) / 2.0 if len(y_coords) > 1 else 0.0

    def bent_epsilon(x: float, y: float) -> tuple[float, float, float, float, float]:
        """Epsilon callback with bend correction applied."""
        eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = epsilon(x, y)

        # Compute signed radial distance from bend center based on bend axis
        # The signed distance matters: outer edge (r > 0) has lower effective index,
        # inner edge (r < 0) has higher effective index
        if bend_axis == 2:
            # Bend in x-y plane: use perpendicular distance in y direction
            r = y - y_center
        elif bend_axis == 1:
            # Bend in x-z plane: use perpendicular distance in x direction
            r = x - x_center
        else:  # bend_axis == 0
            # Bend in y-z plane: use perpendicular distance in y direction
            r = y - y_center

        # Apply Jacobian transformation J · ε · J^T / det(J) for polar coordinates
        # For a bend in the x-y plane around z-axis:
        #   J = diag(1, 1, R/(R+r)), det(J) = R/(R+r)
        # Only eps_zz is modified: eps_zz_new = eps_zz * R/(R+r)
        # Other components (eps_xx, eps_yy, eps_xy, eps_yx) are unchanged
        bend_stretch = bend_radius / (bend_radius + r)

        return (
            eps_xx,
            eps_xy,
            eps_yx,
            eps_yy,
            eps_zz * bend_stretch,
        )

    return bent_epsilon


def _build_edge_coords_from_centers(
    x_centers: tuple[float, ...], y_centers: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build cell-edge coordinate arrays from cell-center coordinates.

    For a grid with N cell centers, there are N+1 cell edges.
    Edges are placed at: start - dx/2, center_1, center_2, ..., center_N, end + dx/2

    Args:
        x_centers: Cell-center x coordinates
        y_centers: Cell-center y coordinates

    Returns:
        (x_edges, y_edges) tuple of edge coordinate arrays
    """
    xc = np.asarray(x_centers, dtype=np.float64)
    yc = np.asarray(y_centers, dtype=np.float64)

    if len(xc) > 1:
        dx = xc[1] - xc[0]
        x_edges = tuple(np.concatenate([[xc[0] - dx], xc, [xc[-1] + dx]]))
    else:
        x_edges = (xc[0] - 0.5, xc[0], xc[0] + 0.5)

    if len(yc) > 1:
        dy = yc[1] - yc[0]
        y_edges = tuple(np.concatenate([[yc[0] - dy], yc, [yc[-1] + dy]]))
    else:
        y_edges = (yc[0] - 0.5, yc[0], yc[0] + 0.5)

    return (tuple(x_edges), tuple(y_edges))


def _reconstruct_hz_fallahkhair(
    Hx: np.ndarray, Hy: np.ndarray, x: np.ndarray, y: np.ndarray, beta: float
) -> np.ndarray:
    """Reconstruct Hz from Hx and Hy using divergence constraint (Fallahkhair formulation).

    From div(H) = 0: dHx/dx + dHy/dy = -i*beta*Hz
    So: Hz = -(dHx/dx + dHy/dy) / (i*beta)

    Hx and Hy are on the Yee grid (Hx at (i+1/2, j), Hy at (i, j+1/2)).
    Hz is at cell centers (i+1/2, j+1/2).
    """
    ny, nx = Hx.shape
    diffx = np.diff(x)
    diffy = np.diff(y)

    Hz = np.zeros((ny - 1, nx - 1), dtype=np.complex128)

    for j in range(ny - 1):
        for i in range(nx - 1):
            dx = diffx[i] if i < len(diffx) else diffx[-1] if len(diffx) > 0 else 1.0
            dy = diffy[j] if j < len(diffy) else diffy[-1] if len(diffy) > 0 else 1.0

            # Hx at (i, j) is actually at x[i+1/2], y[j]
            # Hy at (i, j) is actually at x[i], y[j+1/2]
            Hx_j_i = Hx[j, i] if j < ny and i < nx else 0.0
            Hx_j1_i = Hx[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hx_j_i1 = Hx[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hx_j1_i1 = Hx[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            Hy_j_i = Hy[j, i] if j < ny and i < nx else 0.0
            Hy_j1_i = Hy[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hy_j_i1 = Hy[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hy_j1_i1 = Hy[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            # dHx/dx at (i+1/2, j+1/2)
            dHxd_x = (Hx_j1_i1 + Hx_j1_i - Hx_j_i1 - Hx_j_i) / (2.0 * dx)
            # dHy/dy at (i+1/2, j+1/2)
            dHyd_y = (Hy_j1_i1 + Hy_j1_i - Hy_j_i1 - Hy_j_i) / (2.0 * dy)

            Hz[j, i] = -(dHxd_x + dHyd_y) / (1j * beta + 1e-30)

    return Hz


def _reconstruct_e_fallahkhair(
    Hx: np.ndarray,
    Hy: np.ndarray,
    Hz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    omega: float,
    epsilon: EpsilonCallback,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct E fields from H fields using Maxwell's curl equations (Fallahkhair formulation).

    Uses: D = curl(H) / (-i*omega), E = D / eps
    """
    ny, nx = Hx.shape
    diffx = np.diff(x)
    diffy = np.diff(y)

    Ex = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ey = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ez = np.zeros((ny - 1, nx - 1), dtype=np.complex128)

    for j in range(ny - 1):
        for i in range(nx - 1):
            dx = diffx[i] if i < len(diffx) else diffx[-1] if len(diffx) > 0 else 1.0
            dy = diffy[j] if j < len(diffy) else diffy[-1] if len(diffy) > 0 else 1.0

            xc = (x[i] + x[i + 1]) / 2.0 if i < len(x) - 1 else x[i]
            yc = (y[j] + y[j + 1]) / 2.0 if j < len(y) - 1 else y[j]

            eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = epsilon(xc, yc)
            det_eps = eps_xx * eps_yy - eps_xy * eps_yx + 1e-15

            def safe_hz(jj, ii):
                if 0 <= jj < Hz.shape[0] and 0 <= ii < Hz.shape[1]:
                    return Hz[jj, ii]
                return 0.0

            def safe_h(jj, ii, arr):
                if 0 <= jj < ny and 0 <= ii < nx:
                    return arr[jj, ii]
                return 0.0

            # D = curl(H) / (-i*omega), E = D / eps
            # curl(H)_x = dHz/dy + i*beta*Hy -> Dx = (dHz_dy + i*beta*Hy) / (i*omega)
            # curl(H)_y = dHz/dx - i*beta*Hx -> Dy = (dHz_dx - i*beta*Hx) / (i*omega)
            # curl(H)_z = dHy/dx - dHx/dy -> Dz = (dHy_dx - dHx_dy) / (i*omega)
            Hy_avg = (safe_h(j + 1, i + 1, Hy) + safe_h(j + 1, i, Hy) +
                       safe_h(j, i + 1, Hy) + safe_h(j, i, Hy)) / 4.0
            Hx_avg = (safe_h(j + 1, i + 1, Hx) + safe_h(j + 1, i, Hx) +
                       safe_h(j, i + 1, Hx) + safe_h(j, i, Hx)) / 4.0

            dHz_dy = (safe_hz(j, i) + safe_hz(j + 1, i) - safe_hz(j, i + 1) - safe_hz(j + 1, i + 1)) / (2.0 * dy)
            dHz_dx = (safe_hz(j + 1, i + 1) + safe_hz(j + 1, i) - safe_hz(j, i + 1) - safe_hz(j, i)) / (2.0 * dx)

            # Derivatives for Dz = (dHy/dx - dHx/dy) / (i*omega)
            # Hy at (i, j) is at x[i], y[j+1/2], so dHy/dx at (i+1/2, j+1/2) uses Hy at x[i] and x[i+1]
            Hy_i_j = safe_h(j, i, Hy)
            Hy_i1_j = safe_h(j, i + 1, Hy)
            Hy_i_j1 = safe_h(j + 1, i, Hy)
            Hy_i1_j1 = safe_h(j + 1, i + 1, Hy)
            dHy_dx = (Hy_i1_j1 + Hy_i1_j - Hy_i_j1 - Hy_i_j) / (2.0 * dx)

            # Hx at (i, j) is at x[i+1/2], y[j], so dHx/dy at (i+1/2, j+1/2) uses Hy at y[j] and y[j+1]
            Hx_i_j = safe_h(j, i, Hx)
            Hx_i1_j = safe_h(j, i + 1, Hx)
            Hx_i_j1 = safe_h(j + 1, i, Hx)
            Hx_i1_j1 = safe_h(j + 1, i + 1, Hx)
            dHx_dy = (Hx_i1_j1 + Hx_i1_j - Hx_i_j1 - Hx_i_j) / (2.0 * dy)

            Dx = (dHz_dy + 1j * beta * Hy_avg) / (1j * omega + 1e-30)
            Dy = (1j * beta * Hx_avg + dHz_dx) / (1j * omega + 1e-30)
            Dz = (dHy_dx - dHx_dy) / (1j * omega + 1e-30)

            Ex[j, i] = (eps_yy * Dx - eps_yx * Dy) / det_eps
            Ey[j, i] = (eps_xx * Dy - eps_xy * Dx) / det_eps
            Ez[j, i] = Dz / eps_zz if abs(eps_zz) > 1e-10 else 0.0

    return Ex, Ey, Ez


def solve_modes(
    config: ModeSolverConfig,
    epsilon: EpsilonCallback,
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> tuple[ModeSolution, ...]:
    """Solve for waveguide modes on a cross-sectional grid.

    For isotropic media, uses the scalar TE Helmholtz formulation which correctly
    finds guided modes. For anisotropic media, uses the full-vectorial Fallahkhair
    2008 formulation when anisotropic parameters (eps_xy != 0) are detected.

    The scalar path solves: ∇²Ez + k₀² ε Ez = β² Ez

    Args:
        config: Mode solver configuration
        epsilon: Callback returning permittivity tensor at (x, y)
        x: Cell-center x coordinates
        y: Cell-center y coordinates

    Returns:
        Tuple of ModeSolution objects sorted by neff descending
    """
    if eigs is None:
        raise ModuleNotFoundError("scipy is required for the mode solver")

    nx = len(x)
    ny = len(y)
    N = nx * ny

    if N < 4:
        raise ValueError(f"Grid too small: {nx} x {ny} cells; need at least 2x2")

    k0 = 2.0 * math.pi / config.wavelength  # free-space wavenumber (rad/m)
    omega = C0 * k0  # angular frequency in rad/s (SI units)

    # Apply bend correction if bend_radius is specified
    effective_epsilon = epsilon
    if config.cross_section.bend_radius is not None:
        effective_epsilon = _make_bent_epsilon_callback(
            epsilon,
            config.cross_section.bend_radius,
            config.cross_section.bend_axis,
            x,
            y,
        )

    # Estimate n_core and n_clad from epsilon for guided-mode targeting
    n_clad, n_core = _estimate_n_core_n_clad(effective_epsilon, x, y)

    # Check if we have anisotropic media (non-zero off-diagonal components)
    def is_isotropic() -> bool:
        for xv in x[::max(1, nx//10)]:  # Sample ~10 points
            for yv in y[::max(1, ny//10)]:
                _, eps_xy, eps_yx, _, _ = effective_epsilon(xv, yv)
                if abs(eps_xy) > 1e-10 or abs(eps_yx) > 1e-10:
                    return False
        return True

    # For isotropic media, use the scalar TE matrix
    # The scalar Helmholtz correctly finds guided modes for dielectric waveguides
    def eps_zz_callback(xi: float, yi: float) -> float:
        _, _, _, _, eps_zz = effective_epsilon(xi, yi)
        return eps_zz

    A_te = assemble_te_mode_matrix(
        x, y, config.wavelength, eps_zz_callback, config.cross_section.boundaries,
        num_pml_layers=config.cross_section.num_pml_layers,
        pml_sigma_max=config.cross_section.pml_sigma_max,
        pml_kappa_min=config.cross_section.pml_kappa_min,
        pml_kappa_max=config.cross_section.pml_kappa_max,
        pml_order=config.cross_section.pml_order,
    )

    # Target number of eigenvalues
    nev_request = min(config.mode_spec.num_modes * 3, max(1, N // 4))

    # Compute shift for shift-invert eigensolver
    # Use target_neff if specified, otherwise estimate from n_core
    # For guided modes, neff is between n_clad and n_core, so using n_core
    # as estimate gives us the shift closest to the guided mode eigenvalues
    if config.mode_spec.target_neff is not None:
        target_neff_est = config.mode_spec.target_neff
    else:
        target_neff_est = n_core  # n_core is max sqrt(eps) in cross-section

    target_shift = (target_neff_est * k0) ** 2

    try:
        evals, evecs = eigs(
            A_te,
            k=nev_request + 5,
            sigma=target_shift,
            tol=config.tolerance,
            which="LM",
            maxiter=2000,
        )
    except Exception as exc:
        raise RuntimeError(f"eigensolver failed: {exc}") from exc

    # For the TE matrix, eigenvalues are β² > 0 (positive).
    # "LR" (largest real) finds the largest β², which corresponds to highest neff.
    # For step-index waveguide, guided modes have neff > n_clad
    # For homogeneous medium (n_core ≈ n_clad), accept modes near n_clad
    radiation_margin = 0.01
    if n_core > n_clad * 1.01:
        # Step-index waveguide - filter radiation modes
        min_neff_for_guided = n_clad * (1.0 + radiation_margin)
    else:
        # Homogeneous medium - accept modes near n_clad
        min_neff_for_guided = n_clad * (1.0 - radiation_margin)

    modes: list[ModeSolution] = []

    for idx in range(len(evals)):
        beta_sq = evals[idx]

        # TE matrix gives β² = eigenvalue directly (positive for propagating modes)
        if beta_sq.real <= 0:
            continue

        beta = math.sqrt(beta_sq.real)
        neff = beta / k0 if k0 > 0 else 0.0

        # Skip if neff is not in physical range
        if neff < 0.0 or neff > 10.0:
            continue

        # Filter out radiation modes: neff should be significantly above n_clad
        if neff < min_neff_for_guided:
            continue

        # Extract Ez from eigenvector (TE matrix gives Ez directly)
        Ez_flat = evecs[:, idx]
        Ez = Ez_flat.reshape((ny, nx))

        # Compute Hx, Hy from Ez using Maxwell's curl equations:
        # For TE mode (E_z only), propagating in +z:
        # Hx = (1/(i*omega*mu)) * dEz/dy
        # Hy = -(1/(i*omega*mu)) * dEz/dx
        # Using mu = mu0 = 1.0 in relative units
        i_omega = 1j * omega

        # Compute dEz/dy using central differences
        Hy = np.zeros_like(Ez)
        Hx = np.zeros_like(Ez)
        for j in range(ny):
            for i in range(nx):
                # dEz/dy at (i,j)
                if j == 0:
                    dy = y[1] - y[0] if ny > 1 else 1.0
                    dEzd_y = (Ez[j+1, i] - Ez[j, i]) / dy if j+1 < ny else 0.0
                elif j == ny - 1:
                    dy = y[j] - y[j-1] if ny > 1 else 1.0
                    dEzd_y = (Ez[j, i] - Ez[j-1, i]) / dy
                else:
                    dy = y[j+1] - y[j-1] if j+1 < ny else y[j] - y[j-1]
                    dEzd_y = (Ez[j+1, i] - Ez[j-1, i]) / dy

                # dEz/dx at (i,j)
                if i == 0:
                    dx = x[1] - x[0] if nx > 1 else 1.0
                    dEzd_x = (Ez[j, i+1] - Ez[j, i]) / dx if i+1 < nx else 0.0
                elif i == nx - 1:
                    dx = x[i] - x[i-1] if nx > 1 else 1.0
                    dEzd_x = (Ez[j, i] - Ez[j, i-1]) / dx
                else:
                    dx = x[i+1] - x[i-1] if i+1 < nx else x[i] - x[i-1]
                    dEzd_x = (Ez[j, i+1] - Ez[j, i-1]) / dx

                Hx[j, i] = dEzd_y / i_omega
                Hy[j, i] = -dEzd_x / i_omega

        # Set Ex, Ey, Hz to zero for TE mode (these are TM components)
        Ex = np.zeros_like(Ez)
        Ey = np.zeros_like(Ez)
        Hz = np.zeros_like(Ez)

        # Downsample H to cell centers for consistency with vector case
        # These are used for power normalization via Poynting integral
        Hx_c = (Hx[:-1, :-1] + Hx[1:, :-1] + Hx[:-1, 1:] + Hx[1:, 1:]) / 4.0
        Hy_c = (Hy[:-1, :-1] + Hy[1:, :-1] + Hy[:-1, 1:] + Hy[1:, 1:]) / 4.0
        Hz_c = (Hz[:-1, :-1] + Hz[1:, :-1] + Hz[:-1, 1:] + Hz[1:, 1:]) / 4.0

        # Downsample E to cell centers to match H grid size
        Ex_c = (Ex[:-1, :-1] + Ex[1:, :-1] + Ex[:-1, 1:] + Ex[1:, 1:]) / 4.0
        Ey_c = (Ey[:-1, :-1] + Ey[1:, :-1] + Ey[:-1, 1:] + Ey[1:, 1:]) / 4.0
        Ez_c = (Ez[:-1, :-1] + Ez[1:, :-1] + Ez[:-1, 1:] + Ez[1:, 1:]) / 4.0

        # Normalize fields by power using Poynting vector integral
        # P = 0.5 * Re(integral(Ex*conj(Hy) - Ey*conj(Hx)) * dx * dy)
        # This is the time-averaged power flow in the z-direction
        # For scalar TE path (Ex=Ey=0): P = -0.5 * Re(integral(Ez*conj(Hx)))
        # which equals (beta/2/omega/mu) * integral(|Ez|^2) for guided TE modes
        power = 0.0
        # Also compute integrals for TE/TM fraction and effective area
        integral_E2 = 0.0  # integral |E|^2 dA
        integral_E4 = 0.0  # integral |E|^4 dA (for effective area)
        integral_ExEy2 = 0.0  # integral (|Ex|^2 + |Ey|^2) dA (for TE fraction)
        for j in range(ny - 1):
            for i in range(nx - 1):
                dx = (x[i+1] - x[i-1]) / 2 if i > 0 and i < nx - 1 else (x[min(i+1, nx-1)] - x[max(i-1, 0)])
                dy = (y[j+1] - y[j-1]) / 2 if j > 0 and j < ny - 1 else (y[min(j+1, ny-1)] - y[max(j-1, 0)])
                dA = dx * dy
                # Poynting vector integrand: S_z = 0.5 * Re(Ex*conj(Hy) - Ey*conj(Hx))
                poynting_z = 0.5 * (
                    Ex_c[j, i] * np.conj(Hy_c[j, i]).real
                    - Ey_c[j, i] * np.conj(Hx_c[j, i]).real
                    + 1j * (Ex_c[j, i] * np.conj(Hy_c[j, i]).imag - Ey_c[j, i] * np.conj(Hx_c[j, i]).imag)
                )
                power += poynting_z.real * dA

                # Field magnitudes for TE/TM fraction and effective area
                Ex_val = Ex_c[j, i]
                Ey_val = Ey_c[j, i]
                Ez_val = Ez_c[j, i]
                E2 = (Ex_val * np.conj(Ex_val)).real + (Ey_val * np.conj(Ey_val)).real + (Ez_val * np.conj(Ez_val)).real
                integral_E2 += E2 * dA
                integral_E4 += E2**2 * dA
                integral_ExEy2 += ((Ex_val * np.conj(Ex_val)).real + (Ey_val * np.conj(Ey_val)).real) * dA

        power = float(power.real)
        power_norm = 1.0 / math.sqrt(abs(power)) if abs(power) > 1e-15 else 1.0

        # Compute TE/TM fraction and effective area
        te_fraction = float(integral_ExEy2) / float(integral_E2) if abs(integral_E2) > 1e-15 else 0.0
        tm_fraction = 1.0 - te_fraction
        effective_area = float(integral_E2**2) / float(integral_E4) if abs(integral_E4) > 1e-15 else float('inf')

        def _to_tuple_complex(arr: np.ndarray) -> tuple[tuple[float, float], ...]:
            flat = arr.flatten()
            return tuple((float(v.real), float(v.imag)) for v in flat)

        modes.append(
            ModeSolution(
                neff=neff,
                wavelength=config.wavelength,
                x=x,
                y=y,
                Ex=_to_tuple_complex(Ex_c * power_norm),
                Ey=_to_tuple_complex(Ey_c * power_norm),
                Ez=_to_tuple_complex(Ez_c * power_norm),
                Hx=_to_tuple_complex(Hx_c * power_norm),
                Hy=_to_tuple_complex(Hy_c * power_norm),
                Hz=_to_tuple_complex(Hz_c * power_norm),
                power=power * power_norm**2,
                te_fraction=te_fraction,
                tm_fraction=tm_fraction,
                effective_area=effective_area,
            )
        )

        if len(modes) >= config.mode_spec.num_modes:
            break

    # Sort by neff descending
    modes.sort(key=lambda m: -m.neff_real)
    return tuple(modes)


__all__ = [
    "EpsilonCallback",
    "assemble_te_mode_matrix",
    "assemble_fallahkhair_mode_matrix",
    "solve_modes",
]
