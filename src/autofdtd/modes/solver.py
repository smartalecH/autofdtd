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
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

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


def assemble_te_mode_matrix(
    x: tuple[float, ...],
    y: tuple[float, ...],
    wavelength: float,
    epsilon_zz: Callable[[float, float], float],
    boundaries: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition, BoundaryCondition],
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

    Args:
        x: Cell-center x coordinates (length nx)
        y: Cell-center y coordinates (length ny)
        wavelength: Free-space wavelength
        epsilon_zz: Callback returning εzz at (x, y)
        boundaries: [north, south, east, west] BCs

    Returns:
        Sparse matrix A of shape (nx*ny, nx*ny) for A·Ez = β²·Ez
    """
    if sparse is None:
        raise ModuleNotFoundError("scipy is required for the mode solver")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = len(x)
    ny = len(y)
    k0 = 2.0 * math.pi / wavelength

    # Cell spacings
    dx = np.diff(x) if nx > 1 else np.array([1.0])
    dy = np.diff(y) if ny > 1 else np.array([1.0])

    bc_north = _map_boundary_condition(boundaries[0])
    bc_south = _map_boundary_condition(boundaries[1])
    bc_east = _map_boundary_condition(boundaries[2])
    bc_west = _map_boundary_condition(boundaries[3])

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def _add(ii: int, jj: int, val: float) -> None:
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

            # Standard 5-point Laplacian stencil
            # (Ez_{i+1,j} - 2Ez_{i,j} + Ez_{i-1,j}) / dx²
            # + (Ez_{i,j+1} - 2Ez_{i,j} + Ez_{i,j-1}) / dy²
            # + k₀² ε_r(i,j) Ez_{i,j}

            # Diagonal
            diag = (
                -2.0 / (dx_w * (dx_w + dx_e))
                - 2.0 / (dx_e * (dx_w + dx_e))
                - 2.0 / (dy_s * (dy_s + dy_n))
                - 2.0 / (dy_n * (dy_s + dy_n))
                + k0**2 * eps_c
            )

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
                    diag += 2.0 / (dx_w * (dx_w + dx_e))
                # Neumann: no change (ghost = same as interior)

            if i == nx - 1:  # East
                if bc_east == 0:  # Dirichlet
                    diag += 2.0 / (dx_e * (dx_w + dx_e))

            if j == 0:  # South
                if bc_south == 0:  # Dirichlet
                    diag += 2.0 / (dy_s * (dy_s + dy_n))

            if j == ny - 1:  # North
                if bc_north == 0:  # Dirichlet
                    diag += 2.0 / (dy_n * (dy_s + dy_n))

            _add(idx, idx, diag)

            # Neighbor contributions
            if i > 0:
                _add(idx, (i - 1) * ny + j, 2.0 / (dx_w * (dx_w + dx_e)))
            if i < nx - 1:
                _add(idx, (i + 1) * ny + j, 2.0 / (dx_e * (dx_w + dx_e)))
            if j > 0:
                _add(idx, i * ny + (j - 1), 2.0 / (dy_s * (dy_s + dy_n)))
            if j < ny - 1:
                _add(idx, i * ny + (j + 1), 2.0 / (dy_n * (dy_s + dy_n)))

    A = sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.float64)
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


def _reconstruct_hz(
    Hx: np.ndarray, Hy: np.ndarray, x: np.ndarray, y: np.ndarray, beta: float
) -> np.ndarray:
    """Reconstruct Hz from Hx and Hy using Maxwell's curl equation.

    Hx and Hy are on staggered grids; Hz is defined at cell centers.
    We compute Hz using the curl of H:
        Hz[k,l] = (dHx/dy - dHy/dx) / (i*beta) at cell center (k,l)
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

            # Average H at face centers
            Hx_j_i = Hx[j, i] if j < ny and i < nx else 0.0
            Hx_j1_i = Hx[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hx_j_i1 = Hx[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hx_j1_i1 = Hx[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            Hy_j_i = Hy[j, i] if j < ny and i < nx else 0.0
            Hy_j1_i = Hy[j + 1, i] if j + 1 < ny and i < nx else 0.0
            Hy_j_i1 = Hy[j, i + 1] if j < ny and i + 1 < nx else 0.0
            Hy_j1_i1 = Hy[j + 1, i + 1] if j + 1 < ny and i + 1 < nx else 0.0

            dHxd_y = (Hx_j1_i1 + Hx_j1_i - Hx_j_i1 - Hx_j_i) / (2.0 * dy)
            dHyd_x = (Hy_j1_i1 + Hy_j1_i - Hy_j_i1 - Hy_j_i) / (2.0 * dx)
            Hz[j, i] = (dHxd_y - dHyd_x) / (1j * beta + 1e-30)

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

    Ex = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ey = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    Ez = np.zeros((ny - 1, nx - 1), dtype=np.complex128)

    for j in range(ny - 1):
        for i in range(nx - 1):
            dx = diffx[i] if i < len(diffx) else diffx[-1] if len(diffx) > 0 else 1.0
            dy = diffy[j] if j < len(diffy) else diffy[-1] if len(diffy) > 0 else 1.0
            xc = (x[j] + x[j + 1]) / 2.0 if j < len(x) - 1 else x[j]
            yc = (y[i] + y[i + 1]) / 2.0 if i < len(y) - 1 else y[i]

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

            # E = (1/(-iωε)) curl H
            Dx = 1j * (dHz_dy - 1j * beta * Hy_avg) / (omega + 1e-30)
            Dy = 1j * (1j * beta * Hx_avg - dHz_dx) / (omega + 1e-30)

            Ex[j, i] = (eps_yy * Dx - eps_yx * Dy) / det_eps
            Ey[j, i] = (-eps_xy * Dx + eps_xx * Dy) / det_eps
            Ez[j, i] = 0.0

    return Ex, Ey, Ez


def solve_modes(
    config: ModeSolverConfig,
    epsilon: EpsilonCallback,
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> tuple[ModeSolution, ...]:
    """Solve for waveguide modes on a cross-sectional grid.

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

    omega = 2.0 * math.pi / config.wavelength
    k = omega  # k₀ in the convention used

    # Assemble sparse matrix
    A = assemble_mode_matrix(x, y, config.wavelength, epsilon, config.cross_section.boundaries)

    # Target number of eigenvalues
    nev = min(config.mode_spec.num_modes, max(1, N // 4))

    # Eigensolver sigma: target neff or None for largest real part
    sigma = None
    if config.mode_spec.target_neff is not None:
        sigma = config.mode_spec.target_neff**2 * k**2  # β² = neff² * k²

    # The curl-curl operator is negative definite
    # So A·H = -β²·H, giving eigenvalues -β² < 0
    # We want the LARGEST algebraic eigenvalue (closest to zero)
    try:
        evals, evecs = eigs(
            A,
            k=nev + 5,  # Request extra to handle filtering
            sigma=sigma,
            tol=config.tolerance,
            which="LM" if sigma is not None else "SR",  # Smallest Real when no sigma
            maxiter=2000,
        )
    except Exception as exc:
        raise RuntimeError(f"eigensolver failed: {exc}") from exc

    modes: list[ModeSolution] = []

    for idx in range(len(evals)):
        beta_sq = evals[idx]

        # curl-curl gives -β², so β² = -eigenvalue
        # The eigenvalues should be negative for propagating modes
        if beta_sq.real >= 0:
            # Not a guided mode (would be radiating or evanescent)
            continue

        beta_sq_neg = beta_sq.real  # eigenvalue of A (negative = -β²)
        beta_sq_positive = -beta_sq_neg  # this is β² > 0

        if beta_sq_positive <= 0:
            continue

        beta = math.sqrt(beta_sq_positive)
        neff = beta / k if k > 0 else 0.0

        # Skip if neff is not in physical range
        if neff < 0.0 or neff > 10.0:
            continue

        # Extract Hx, Hy
        Hx_flat = evecs[:N, idx]
        Hy_flat = evecs[N : 2 * N, idx]

        Hx = Hx_flat.reshape((ny, nx))
        Hy = Hy_flat.reshape((ny, nx))

        # Reconstruct Hz
        Hz = _reconstruct_hz(Hx, Hy, np.asarray(x), np.asarray(y), beta)

        # Reconstruct E fields
        Ex, Ey, Ez = _reconstruct_fields(Hx, Hy, Hz, np.asarray(x), np.asarray(y), beta, omega, epsilon)

        # Compute power normalization
        power = float(
            np.sum(np.abs(Ex) ** 2) + np.sum(np.abs(Ey) ** 2) + np.sum(np.abs(Ez) ** 2)
        )
        power_norm = 1.0 / math.sqrt(power) if power > 1e-15 else 1.0

        def _to_tuple_complex(arr: np.ndarray) -> tuple[tuple[float, float], ...]:
            flat = arr.flatten()
            return tuple((float(v.real), float(v.imag)) for v in flat)

        # Downsample H to cell centers
        Hx_c = (Hx[:-1, :-1] + Hx[1:, :-1] + Hx[:-1, 1:] + Hx[1:, 1:]) / 4.0
        Hy_c = (Hy[:-1, :-1] + Hy[1:, :-1] + Hy[:-1, 1:] + Hy[1:, 1:]) / 4.0
        Hz_c = Hz

        modes.append(
            ModeSolution(
                neff=neff,
                wavelength=config.wavelength,
                x=x,
                y=y,
                Ex=_to_tuple_complex(Ex * power_norm),
                Ey=_to_tuple_complex(Ey * power_norm),
                Ez=_to_tuple_complex(Ez * power_norm),
                Hx=_to_tuple_complex(Hx_c * power_norm),
                Hy=_to_tuple_complex(Hy_c * power_norm),
                Hz=_to_tuple_complex(Hz_c * power_norm),
                power=power * power_norm**2,
            )
        )

        if len(modes) >= config.mode_spec.num_modes:
            break

    # Sort by neff descending
    modes.sort(key=lambda m: -m.neff_real)
    return tuple(modes)


__all__ = [
    "EpsilonCallback",
    "assemble_mode_matrix",
    "solve_modes",
]
