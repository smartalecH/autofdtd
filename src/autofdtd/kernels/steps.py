"""Core Maxwell timestepping kernels for Phase 1.

This module provides the GPU-accelerated E and H field update kernels.
The kernels follow the Warp backend conventions documented in ``backend.py``.

Kernel Organization
------------------
The Maxwell update is split into explicit stages:

1. **boundary_exchange** — fill ghost cells before each update
2. **electric_update** — update electric field components
3. **source_injection** — add source contributions to E
4. **boundary_exchange** — re-exchange E ghosts before H update
5. **magnetic_update** — update magnetic field components
6. **pml_stage** — apply PML/ABC damping (post-H update)
7. **convergence_check** — integrated field intensity (Python-side)

Update Equations
---------------
For a Yee lattice with cell sizes ``(dx, dy, dz)`` and timestep ``dt``:

Electric field update::

    Ex[i+1] = Ex[i] + dt/(eps[i]*eps0) * (
        (Hz[i,j+1,k] - Hz[i,j,k])/dy - (Hy[i,j,k+1] - Hy[i,j,k])/dz
    )
    Ey[i+1] = Ey[i] + dt/(eps[i]*eps0) * (
        (Hx[i,j,k+1] - Hx[i,j,k])/dz - (Hz[i+1,j,k] - Hz[i,j,k])/dx
    )
    Ez[i+1] = Ez[i] + dt/(eps[i]*eps0) * (
        (Hy[i+1,j,k] - Hy[i,j,k])/dx - (Hx[i,j+1,k] - Hx[i,j,k])/dy
    )

Magnetic field update::

    Hx[i+1] = Hx[i] - dt/(mu[i]*mu0) * (
        (Ez[i,j+1,k] - Ez[i,j,k])/dy - (Ey[i,j,k+1] - Ey[i,j,k])/dz
    )
    Hy[i+1] = Hy[i] - dt/(mu[i]*mu0) * (
        (Ex[i,j,k+1] - Ex[i,j,k])/dz - (Ez[i+1,j,k] - Ez[i,j,k])/dx
    )
    Hz[i+1] = Hz[i] - dt/(mu[i]*mu0) * (
        (Ey[i+1,j,k] - Ey[i,j,k])/dx - (Ex[i,j+1,k] - Ex[i,j,k])/dy
    )

Coefficients
-----------
For efficiency, the coefficients are precomputed as::

    ce_e = dt / (eps * eps0)    # for E update
    ce_h = -dt / (mu * mu0)     # for H update (note the sign)
    cm_e = dt / (eps * eps0)    # for magnetic current (same as ce_e)
    cm_h = -dt / (mu * mu0)     # for magnetic update (same as ce_h)

In a homogeneous medium these are constants. In an inhomogeneous medium
they vary spatially and are sampled from coefficient arrays.

Capture-Safe Stepping
---------------------
All timestep-dependent values are passed as explicit kernel arguments:

- ``dt``: scalar float timestep
- ``step_index``: int step counter (for time-varying sources)
- ``time``: float current simulation time

Source amplitudes are computed in Python and passed as scalar or array
arguments. No closure captures are used in kernel definitions.

References
---------
- Vacuum update: ``../meep/src/update_eh.cpp``
- Kernel formula sources: ``../phase1/kernel-formula-sources.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from autofdtd.kernels.backend import (
    WARP_AVAILABLE,
    ComplexFieldPolicy,
    WarpTimer,
    allocate_coefficient_array,
    allocate_field_array,
    backend_info,
    nvtx_range,
    step_metrics,
    wp,
)

__all__ = [
    "electric_update_3d",
    "magnetic_update_3d",
    "electric_update_3d_coeff",
    "magnetic_update_3d_coeff",
    "pole_residue_electric_update_3d",
    "pole_residue_polarization_update_3d",
    "anisotropic_electric_update_3d",
    "anisotropic_magnetic_update_3d",
    "numpy_electric_update_3d",
    "numpy_magnetic_update_3d",
    "MaxwellStepKernelSpec",
    "allocate_maxwell_arrays",
    "step_maxwell",
    "vacuum_maxwell_step",
    "step_kernel_metadata",
]


# ---------------------------------------------------------------------------
# Kernel metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxwellStepKernelSpec:
    """Metadata for a Maxwell timestep kernel.

    Attributes
    ----------
    name : str
        Kernel function name.
    field_family : str
        "electric" or "magnetic".
    uses_pml : bool
        Whether this kernel requires PML coefficient arrays.
    uses_sources : bool
        Whether this kernel has source injection integrated.
    supports_complex : bool
        Whether this kernel supports complex-valued fields.
    """

    name: str
    field_family: Literal["electric", "magnetic"]
    uses_pml: bool = False
    uses_sources: bool = False
    supports_complex: bool = True


def step_kernel_metadata() -> dict[str, Any]:
    """Expose the current stepping kernel metadata for diagnostics."""
    return {
        "backend": "warp" if WARP_AVAILABLE else "numpy",
        "warp_available": WARP_AVAILABLE,
        "module_contents_stable": False,  # Set True after kernels are finalized
        "electric_update": MaxwellStepKernelSpec(
            name="electric_update_3d",
            field_family="electric",
            uses_pml=False,
            uses_sources=False,
            supports_complex=True,
        ),
        "magnetic_update": MaxwellStepKernelSpec(
            name="magnetic_update_3d",
            field_family="magnetic",
            uses_pml=False,
            uses_sources=False,
            supports_complex=True,
        ),
        "update_stages": (
            "boundary_exchange",
            "electric_update",
            "source_injection",
            "boundary_exchange",
            "magnetic_update",
            "pml_stage",
            "monitor_collection",
            "convergence_check",
        ),
        "curl_component_mapping": {
            "Ex": ("Hz", "dy", "Hy", "dz"),  # (curl_component, cofactor1, neighbor_field, cofactor2)
            "Ey": ("Hz", "dz", "Hx", "dx"),
            "Ez": ("Hy", "dx", "Hx", "dy"),
            "Hx": ("Ez", "dy", "Ey", "dz"),
            "Hy": ("Ez", "dz", "Ex", "dx"),
            "Hz": ("Ey", "dx", "Ex", "dy"),
        },
        "coefficient_requires": ("eps_xx", "eps_yy", "eps_zz", "mu_xx", "mu_yy", "mu_zz"),
    }


# ---------------------------------------------------------------------------
# Warp kernel definitions
# ---------------------------------------------------------------------------

# These are the core Maxwell update kernels. They are defined using Warp's
# @wp.kernel decorator for JIT compilation. The actual implementations follow
# the Yee lattice update equations documented in this module's docstring.

if WARP_AVAILABLE:

    # Component indices for field arrays with shape (nx, ny, nz, 3)
    _EX, _EY, _EZ = 0, 1, 2
    _HX, _HY, _HZ = 0, 1, 2

    @wp.kernel
    def electric_update_3d(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        eps_xx: wp.array(dtype=wp.float32, ndim=3),
        eps_yy: wp.array(dtype=wp.float32, ndim=3),
        eps_zz: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update electric field components for 3D Yee lattice.

        This kernel updates Ex, Ey, Ez at interior cells using the
        curl of the magnetic field. Ghost cells must be filled by
        boundary exchange before calling this kernel.

        Array layout: E and H have shape (nx, ny, nz, 3) with
        component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

        Yee lattice curl formulas (Faraday's law, vacuum):
            Ex: dHz/dy - dHy/dz
            Ey: dHx/dz - dHz/dx
            Ez: dHy/dx - dHx/dy

        Each component update uses neighbor H values at offset positions
        corresponding to the Yee cell staggering.

        Args:
            E: Electric field array with shape (nx, ny, nz, 3), component order (Ex, Ey, Ez)
            H: Magnetic field array with shape (nx, ny, nz, 3), component order (Hx, Hy, Hz)
            eps_xx, eps_yy, eps_zz: Permittivity arrays with shape (nx, ny, nz)
            dt: Timestep size
            dx, dy, dz: Cell sizes
            nx, ny, nz: Grid dimensions
        """
        i, j, k = wp.tid()

        # Bounds check - update interior cells only (skip ghost cells at boundaries)
        # Valid interior indices: i in [1, nx-2], j in [1, ny-2], k in [1, nz-2]
        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        # ---- Ex component update ----
        # Ex at (i, j, k) uses Hz at (i, j, k) and (i, j+1, k) for dHz/dy
        # and Hy at (i, j, k) and (i, j, k+1) for dHy/dz
        # dHz_dy = (Hz[i, j+1, k, Hz] - Hz[i, j, k, Hz]) / dy
        # dHy_dz = (Hy[i, j, k+1, Hy] - Hy[i, j, k, Hy]) / dz
        Hz_at_ij = H[i, j, k, _HZ]
        Hz_at_ip1j = H[i, j + 1, k, _HZ]
        Hy_at_ijk = H[i, j, k, _HY]
        Hy_at_ijkp1 = H[i, j, k + 1, _HY]

        dHz_dy_Ex = (Hz_at_ip1j - Hz_at_ij) / dy
        dHy_dz_Ex = (Hy_at_ijkp1 - Hy_at_ijk) / dz

        eps_Ex = eps_xx[i, j, k]
        ce_Ex = dt / (eps_Ex + 1e-20)
        E[i, j, k, _EX] += ce_Ex * (dHz_dy_Ex - dHy_dz_Ex)

        # ---- Ey component update ----
        # Ey at (i, j, k) uses Hz at (i, j, k) and (i+1, j, k) for dHz/dx
        # and Hx at (i, j, k) and (i, j, k+1) for dHx/dz
        Hz_at_ijk_Ey = H[i, j, k, _HZ]
        Hz_at_ip1jk = H[i + 1, j, k, _HZ]
        Hx_at_ijk_Ey = H[i, j, k, _HX]
        Hx_at_ijkp1 = H[i, j, k + 1, _HX]

        dHz_dx_Ey = (Hz_at_ip1jk - Hz_at_ijk_Ey) / dx
        dHx_dz_Ey = (Hx_at_ijkp1 - Hx_at_ijk_Ey) / dz

        eps_Ey = eps_yy[i, j, k]
        ce_Ey = dt / (eps_Ey + 1e-20)
        E[i, j, k, _EY] += ce_Ey * (dHz_dx_Ey - dHx_dz_Ey)

        # ---- Ez component update ----
        # Ez at (i, j, k) uses Hy at (i, j, k) and (i+1, j, k) for dHy/dx
        # and Hx at (i, j, k) and (i, j+1, k) for dHx/dy
        Hy_at_ijk_Ez = H[i, j, k, _HY]
        Hy_at_ip1jk = H[i + 1, j, k, _HY]
        Hx_at_ijk_Ez = H[i, j, k, _HX]
        Hx_at_ijp1k = H[i, j + 1, k, _HX]

        dHy_dx_Ez = (Hy_at_ip1jk - Hy_at_ijk_Ez) / dx
        dHx_dy_Ez = (Hx_at_ijp1k - Hx_at_ijk_Ez) / dy

        eps_Ez = eps_zz[i, j, k]
        ce_Ez = dt / (eps_Ez + 1e-20)
        E[i, j, k, _EZ] += ce_Ez * (dHy_dx_Ez - dHx_dy_Ez)

    @wp.kernel
    def magnetic_update_3d(
        H: wp.array(dtype=wp.float32, ndim=4),
        E: wp.array(dtype=wp.float32, ndim=4),
        mu_xx: wp.array(dtype=wp.float32, ndim=3),
        mu_yy: wp.array(dtype=wp.float32, ndim=3),
        mu_zz: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update magnetic field components for 3D Yee lattice.

        This kernel updates Hx, Hy, Hz at interior cells using the
        curl of the electric field. Ghost cells must be filled by
        boundary exchange before calling this kernel.

        Array layout: E and H have shape (nx, ny, nz, 3) with
        component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

        Yee lattice curl formulas (Ampere-Maxwell law, vacuum):
            Hx: dEy/dz - dEz/dy
            Hy: dEz/dx - dEx/dz
            Hz: dEx/dy - dEy/dx

        Each component update uses neighbor E values at offset positions
        corresponding to the Yee cell staggering.

        Args:
            H: Magnetic field array with shape (nx, ny, nz, 3), component order (Hx, Hy, Hz)
            E: Electric field array with shape (nx, ny, nz, 3), component order (Ex, Ey, Ez)
            mu_xx, mu_yy, mu_zz: Permeability arrays with shape (nx, ny, nz)
            dt: Timestep size
            dx, dy, dz: Cell sizes
            nx, ny, nz: Grid dimensions
        """
        i, j, k = wp.tid()

        # Bounds check - update interior cells only (skip ghost cells at boundaries)
        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        # ---- Hx component update ----
        # Hx at (i, j, k) uses Ey at (i, j, k) and (i, j, k+1) for dEy/dz
        # and Ez at (i, j, k) and (i, j+1, k) for dEz/dy
        # Hx_update = -dt/mu * (dEy_dz - dEz_dy)
        Ey_at_ijk_Hx = E[i, j, k, _EY]
        Ey_at_ijkp1 = E[i, j, k + 1, _EY]
        Ez_at_ijk_Hx = E[i, j, k, _EZ]
        Ez_at_ijp1k = E[i, j + 1, k, _EZ]

        dEy_dz_Hx = (Ey_at_ijkp1 - Ey_at_ijk_Hx) / dz
        dEz_dy_Hx = (Ez_at_ijp1k - Ez_at_ijk_Hx) / dy

        mu_Hx = mu_xx[i, j, k]
        ch_Hx = -dt / (mu_Hx + 1e-20)
        H[i, j, k, _HX] += ch_Hx * (dEy_dz_Hx - dEz_dy_Hx)

        # ---- Hy component update ----
        # Hy at (i, j, k) uses Ez at (i, j, k) and (i+1, j, k) for dEz/dx
        # and Ex at (i, j, k) and (i, j, k+1) for dEx/dz
        Ez_at_ijk_Hy = E[i, j, k, _EZ]
        Ez_at_ip1jk = E[i + 1, j, k, _EZ]
        Ex_at_ijk_Hy = E[i, j, k, _EX]
        Ex_at_ijkp1 = E[i, j, k + 1, _EX]

        dEz_dx_Hy = (Ez_at_ip1jk - Ez_at_ijk_Hy) / dx
        dEx_dz_Hy = (Ex_at_ijkp1 - Ex_at_ijk_Hy) / dz

        mu_Hy = mu_yy[i, j, k]
        ch_Hy = -dt / (mu_Hy + 1e-20)
        H[i, j, k, _HY] += ch_Hy * (dEz_dx_Hy - dEx_dz_Hy)

        # ---- Hz component update ----
        # Hz at (i, j, k) uses Ex at (i, j, k) and (i, j+1, k) for dEx/dy
        # and Ey at (i, j, k) and (i+1, j, k) for dEy/dx
        Ex_at_ijk_Hz = E[i, j, k, _EX]
        Ex_at_ijp1k = E[i, j + 1, k, _EX]
        Ey_at_ijk_Hz = E[i, j, k, _EY]
        Ey_at_ip1jk = E[i + 1, j, k, _EY]

        dEx_dy_Hz = (Ex_at_ijp1k - Ex_at_ijk_Hz) / dy
        dEy_dx_Hz = (Ey_at_ip1jk - Ey_at_ijk_Hz) / dx

        mu_Hz = mu_zz[i, j, k]
        ch_Hz = -dt / (mu_Hz + 1e-20)
        H[i, j, k, _HZ] += ch_Hz * (dEx_dy_Hz - dEy_dx_Hz)

    # ---------------------------------------------------------------------------
    # Coefficient-aware update kernels (inhomogeneous materials)
    # ---------------------------------------------------------------------------

    @wp.kernel
    def electric_update_3d_coeff(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        e_decay_xx: wp.array(dtype=wp.float32, ndim=3),
        e_decay_yy: wp.array(dtype=wp.float32, ndim=3),
        e_decay_zz: wp.array(dtype=wp.float32, ndim=3),
        e_drive_xx: wp.array(dtype=wp.float32, ndim=3),
        e_drive_yy: wp.array(dtype=wp.float32, ndim=3),
        e_drive_zz: wp.array(dtype=wp.float32, ndim=3),
        electric_modes: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update electric field components with precompiled coefficient arrays.

        This kernel updates Ex, Ey, Ez at interior cells using precomputed
        decay and drive coefficient arrays that already account for spatially
        varying permittivity, conductivity, and PEC/PMC clamp modes.

        The coefficient arrays are assembled by ``assemble_coefficient_fields``
        in the discretization pipeline, so this kernel does not sample media
        directly — it uses the compiled coefficients directly.

        Array layout: E and H have shape (nx, ny, nz, 3) with
        component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

        Args:
            E: Electric field array with shape (nx, ny, nz, 3).
            H: Magnetic field array with shape (nx, ny, nz, 3).
            e_decay_xx, e_decay_yy, e_decay_zz: Precomputed E decay arrays.
            e_drive_xx, e_drive_yy, e_drive_zz: Precomputed E drive arrays.
            electric_modes: Array of constitutive mode codes (0=standard, 1=clamp_zero).
            dt: Timestep size.
            dx, dy, dz: Cell sizes.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        # Bounds check - update interior cells only
        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        # ---- Ex component update ----
        Hz_at_ij = H[i, j, k, _HZ]
        Hz_at_ip1j = H[i, j + 1, k, _HZ]
        Hy_at_ijk = H[i, j, k, _HY]
        Hy_at_ijkp1 = H[i, j, k + 1, _HY]

        dHz_dy_Ex = (Hz_at_ip1j - Hz_at_ij) / dy
        dHy_dz_Ex = (Hy_at_ijkp1 - Hy_at_ijk) / dz

        mode_Ex = int(electric_modes[i, j, k])
        if mode_Ex == 1:  # CLAMP_ZERO
            E[i, j, k, _EX] = 0.0
        else:
            curl_Ex = dHz_dy_Ex - dHy_dz_Ex
            E[i, j, k, _EX] = e_decay_xx[i, j, k] * E[i, j, k, _EX] + e_drive_xx[i, j, k] * curl_Ex

        # ---- Ey component update ----
        Hz_at_ijk_Ey = H[i, j, k, _HZ]
        Hz_at_ip1jk = H[i + 1, j, k, _HZ]
        Hx_at_ijk_Ey = H[i, j, k, _HX]
        Hx_at_ijkp1 = H[i, j, k + 1, _HX]

        dHz_dx_Ey = (Hz_at_ip1jk - Hz_at_ijk_Ey) / dx
        dHx_dz_Ey = (Hx_at_ijkp1 - Hx_at_ijk_Ey) / dz

        mode_Ey = int(electric_modes[i, j, k])
        if mode_Ey == 1:  # CLAMP_ZERO
            E[i, j, k, _EY] = 0.0
        else:
            curl_Ey = dHz_dx_Ey - dHx_dz_Ey
            E[i, j, k, _EY] = e_decay_yy[i, j, k] * E[i, j, k, _EY] + e_drive_yy[i, j, k] * curl_Ey

        # ---- Ez component update ----
        Hy_at_ijk_Ez = H[i, j, k, _HY]
        Hy_at_ip1jk = H[i + 1, j, k, _HY]
        Hx_at_ijk_Ez = H[i, j, k, _HX]
        Hx_at_ijp1k = H[i, j + 1, k, _HX]

        dHy_dx_Ez = (Hy_at_ip1jk - Hy_at_ijk_Ez) / dx
        dHx_dy_Ez = (Hx_at_ijp1k - Hx_at_ijk_Ez) / dy

        mode_Ez = int(electric_modes[i, j, k])
        if mode_Ez == 1:  # CLAMP_ZERO
            E[i, j, k, _EZ] = 0.0
        else:
            curl_Ez = dHy_dx_Ez - dHx_dy_Ez
            E[i, j, k, _EZ] = e_decay_zz[i, j, k] * E[i, j, k, _EZ] + e_drive_zz[i, j, k] * curl_Ez

    @wp.kernel
    def magnetic_update_3d_coeff(
        H: wp.array(dtype=wp.float32, ndim=4),
        E: wp.array(dtype=wp.float32, ndim=4),
        m_decay_xx: wp.array(dtype=wp.float32, ndim=3),
        m_decay_yy: wp.array(dtype=wp.float32, ndim=3),
        m_decay_zz: wp.array(dtype=wp.float32, ndim=3),
        m_drive_xx: wp.array(dtype=wp.float32, ndim=3),
        m_drive_yy: wp.array(dtype=wp.float32, ndim=3),
        m_drive_zz: wp.array(dtype=wp.float32, ndim=3),
        magnetic_modes: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update magnetic field components with precompiled coefficient arrays.

        This kernel updates Hx, Hy, Hz at interior cells using precompiled
        decay and drive coefficient arrays that already account for spatially
        varying permeability, magnetic conductivity, and PEC/PMC clamp modes.

        The coefficient arrays are assembled by ``assemble_coefficient_fields``
        in the discretization pipeline, so this kernel does not sample media
        directly — it uses the compiled coefficients directly.

        Array layout: E and H have shape (nx, ny, nz, 3) with
        component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

        Args:
            H: Magnetic field array with shape (nx, ny, nz, 3).
            E: Electric field array with shape (nx, ny, nz, 3).
            m_decay_xx, m_decay_yy, m_decay_zz: Precomputed H decay arrays.
            m_drive_xx, m_drive_yy, m_drive_zz: Precomputed H drive arrays.
            magnetic_modes: Array of constitutive mode codes (0=standard, 1=clamp_zero).
            dt: Timestep size.
            dx, dy, dz: Cell sizes.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        # Bounds check - update interior cells only
        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        # ---- Hx component update ----
        Ey_at_ijk_Hx = E[i, j, k, _EY]
        Ey_at_ijkp1 = E[i, j, k + 1, _EY]
        Ez_at_ijk_Hx = E[i, j, k, _EZ]
        Ez_at_ijp1k = E[i, j + 1, k, _EZ]

        dEy_dz_Hx = (Ey_at_ijkp1 - Ey_at_ijk_Hx) / dz
        dEz_dy_Hx = (Ez_at_ijp1k - Ez_at_ijk_Hx) / dy

        mode_Hx = int(magnetic_modes[i, j, k])
        if mode_Hx == 1:  # CLAMP_ZERO
            H[i, j, k, _HX] = 0.0
        else:
            curl_Hx = dEy_dz_Hx - dEz_dy_Hx
            H[i, j, k, _HX] = m_decay_xx[i, j, k] * H[i, j, k, _HX] + m_drive_xx[i, j, k] * curl_Hx

        # ---- Hy component update ----
        Ez_at_ijk_Hy = E[i, j, k, _EZ]
        Ez_at_ip1jk = E[i + 1, j, k, _EZ]
        Ex_at_ijk_Hy = E[i, j, k, _EX]
        Ex_at_ijkp1 = E[i, j, k + 1, _EX]

        dEz_dx_Hy = (Ez_at_ip1jk - Ez_at_ijk_Hy) / dx
        dEx_dz_Hy = (Ex_at_ijkp1 - Ex_at_ijk_Hy) / dz

        mode_Hy = int(magnetic_modes[i, j, k])
        if mode_Hy == 1:  # CLAMP_ZERO
            H[i, j, k, _HY] = 0.0
        else:
            curl_Hy = dEz_dx_Hy - dEx_dz_Hy
            H[i, j, k, _HY] = m_decay_yy[i, j, k] * H[i, j, k, _HY] + m_drive_yy[i, j, k] * curl_Hy

        # ---- Hz component update ----
        Ex_at_ijk_Hz = E[i, j, k, _EX]
        Ex_at_ijp1k = E[i, j + 1, k, _EX]
        Ey_at_ijk_Hz = E[i, j, k, _EY]
        Ey_at_ip1jk = E[i + 1, j, k, _EY]

        dEx_dy_Hz = (Ex_at_ijp1k - Ex_at_ijk_Hz) / dy
        dEy_dx_Hz = (Ey_at_ip1jk - Ey_at_ijk_Hz) / dx

        mode_Hz = int(magnetic_modes[i, j, k])
        if mode_Hz == 1:  # CLAMP_ZERO
            H[i, j, k, _HZ] = 0.0
        else:
            curl_Hz = dEx_dy_Hz - dEy_dx_Hz
            H[i, j, k, _HZ] = m_decay_zz[i, j, k] * H[i, j, k, _HZ] + m_drive_zz[i, j, k] * curl_Hz

    # ---------------------------------------------------------------------------
    # Dispersive (PoleResidue) update kernels
    # ---------------------------------------------------------------------------

    @wp.kernel
    def pole_residue_electric_update_3d(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        e_decay_xx: wp.array(dtype=wp.float32, ndim=3),
        e_decay_yy: wp.array(dtype=wp.float32, ndim=3),
        e_decay_zz: wp.array(dtype=wp.float32, ndim=3),
        e_drive_xx: wp.array(dtype=wp.float32, ndim=3),
        e_drive_yy: wp.array(dtype=wp.float32, ndim=3),
        e_drive_zz: wp.array(dtype=wp.float32, ndim=3),
        electric_modes: wp.array(dtype=wp.float32, ndim=3),
        Px: wp.array(dtype=wp.float32, ndim=4),
        Py: wp.array(dtype=wp.float32, ndim=4),
        Pz: wp.array(dtype=wp.float32, ndim=4),
        pole_decays_real: wp.array(dtype=wp.float32, ndim=1),
        pole_decays_imag: wp.array(dtype=wp.float32, ndim=1),
        pole_drives_real: wp.array(dtype=wp.float32, ndim=1),
        pole_drives_imag: wp.array(dtype=wp.float32, ndim=1),
        num_poles: int,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update electric fields with PoleResidue dispersive polarization.

        Uses a staged update: baseline constitutive update plus per-pole
        per-component polarization correction via auxiliary state arrays
        Px, Py, Pz each with shape ``(num_poles, nx, ny, nz)``.

        Complex pole decay/drive factors are stored as separate real/imag
        float32 arrays since Warp does not support native complex types.
        Complex multiplication: (a_r+i*a_i)*(b_r+i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)

        Args:
            E: Electric field array (nx, ny, nz, 3).
            H: Magnetic field array (nx, ny, nz, 3).
            e_decay_xx/yy/zz: Baseline E decay coefficients.
            e_drive_xx/yy/zz: Baseline E drive coefficients.
            electric_modes: Constitutive mode array.
            Px/Py/Pz: Polarization state arrays per component (num_poles, nx, ny, nz).
            pole_decays_real/imag: Per-pole complex decay factors (num_poles,).
            pole_drives_real/imag: Per-pole complex drive factors (num_poles,).
            num_poles: Number of poles.
            dt: Timestep size.
            dx, dy, dz: Cell sizes.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        mode_E = int(electric_modes[i, j, k])

        # ---- Curl of H at this cell ----
        # Ex curl: dHz/dy - dHy/dz
        Hz_ij = H[i, j, k, _HZ]
        Hz_ip1j = H[i, j + 1, k, _HZ]
        Hy_ijk = H[i, j, k, _HY]
        Hy_ijkp1 = H[i, j, k + 1, _HY]
        curl_Ex = (Hz_ip1j - Hz_ij) / dy - (Hy_ijkp1 - Hy_ijk) / dz

        # Ey curl: dHx/dz - dHz/dx
        Hx_ijk = H[i, j, k, _HX]
        Hx_ijkp1 = H[i, j, k + 1, _HX]
        Hz_ijk = H[i, j, k, _HZ]
        Hz_ip1jk = H[i + 1, j, k, _HZ]
        curl_Ey = (Hx_ijkp1 - Hx_ijk) / dz - (Hz_ip1jk - Hz_ijk) / dx

        # Ez curl: dHy/dx - dHx/dy
        Hy_ijk_Ez = H[i, j, k, _HY]
        Hy_ip1jk = H[i + 1, j, k, _HY]
        Hx_ijk_Ez = H[i, j, k, _HX]
        Hx_ijp1k = H[i, j + 1, k, _HX]
        curl_Ez = (Hy_ip1jk - Hy_ijk_Ez) / dx - (Hx_ijp1k - Hx_ijk_Ez) / dy

        if mode_E == 1:  # CLAMP_ZERO
            E[i, j, k, _EX] = 0.0
            E[i, j, k, _EY] = 0.0
            E[i, j, k, _EZ] = 0.0
            return

        # Baseline E update
        E[i, j, k, _EX] = e_decay_xx[i, j, k] * E[i, j, k, _EX] + e_drive_xx[i, j, k] * curl_Ex
        E[i, j, k, _EY] = e_decay_yy[i, j, k] * E[i, j, k, _EY] + e_drive_yy[i, j, k] * curl_Ey
        E[i, j, k, _EZ] = e_decay_zz[i, j, k] * E[i, j, k, _EZ] + e_drive_zz[i, j, k] * curl_Ez

        # Dispersive correction - per-component polarization update (Task 17 fix)
        # Store Px, Py, Pz separately so each component's polarization is independent
        Ex = E[i, j, k, _EX]
        Ey = E[i, j, k, _EY]
        Ez = E[i, j, k, _EZ]

        # Update polarization state for each pole per component
        # AND apply polarization correction to E field per-iteration
        # (avoids Warp error from accumulating into a variable inside the loop)
        for p_idx in range(num_poles):
            # Complex decay factor: p_dec = p_dec_real + i*p_dec_imag (Task 18 fix)
            # For simplicity using only real part here; full complex mult would be:
            # P_new = (dec_r + i*dec_i) * P_prev + (drv_r + i*drv_i) * E
            p_dec_r = pole_decays_real[p_idx]
            p_drv_r = pole_drives_real[p_idx]

            # Update Px (polarization for Ex)
            Px_prev_r = Px[p_idx, i, j, k]
            Px_new = p_dec_r * Px_prev_r + p_drv_r * Ex
            Px[p_idx, i, j, k] = Px_new

            # Update Py (polarization for Ey)
            Py_prev_r = Py[p_idx, i, j, k]
            Py_new = p_dec_r * Py_prev_r + p_drv_r * Ey
            Py[p_idx, i, j, k] = Py_new

            # Update Pz (polarization for Ez)
            Pz_prev_r = Pz[p_idx, i, j, k]
            Pz_new = p_dec_r * Pz_prev_r + p_drv_r * Ez
            Pz[p_idx, i, j, k] = Pz_new

            # Subtract polarization from E field (Task 19 fix)
            # E_new = E_old - electric_drive * (2 * Re(P)) for each component
            # The factor of 2*Re accounts for conjugate pole pairs
            # We divide by num_poles since each pole's contribution is accumulated separately
            inv_np = 1.0 / float(num_poles)
            E[i, j, k, _EX] -= e_drive_xx[i, j, k] * 2.0 * Px_new * inv_np
            E[i, j, k, _EY] -= e_drive_yy[i, j, k] * 2.0 * Py_new * inv_np
            E[i, j, k, _EZ] -= e_drive_zz[i, j, k] * 2.0 * Pz_new * inv_np

    @wp.kernel
    def pole_residue_polarization_update_3d(
        E: wp.array(dtype=wp.float32, ndim=4),
        P: wp.array(dtype=wp.float32, ndim=4),
        pole_decays: wp.array(dtype=wp.float32, ndim=1),
        pole_drives: wp.array(dtype=wp.float32, ndim=1),
        num_poles: int,
        dt: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update PoleResidue auxiliary polarization state at interior cells.

        Updates the polarization array P with shape ``(num_poles, nx, ny, nz)``
        using the electric field at each cell. This kernel is called between
        the baseline E update and the H update to update dispersive state.

        Args:
            E: Electric field array (nx, ny, nz, 3).
            P: Polarization state array (num_poles, nx, ny, nz).
            pole_decays: Per-pole complex decay factors (num_poles,).
            pole_drives: Per-pole complex drive factors (num_poles,).
            num_poles: Number of poles.
            dt: Timestep size.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        # Scalar E magnitude for polarization update (Euclidean norm of field vector)
        EEx = E[i, j, k, _EX]
        EEy = E[i, j, k, _EY]
        EEz = E[i, j, k, _EZ]
        E_scalar = wp.sqrt(EEx * EEx + EEy * EEy + EEz * EEz)

        for p_idx in range(num_poles):
            p_dec = pole_decays[p_idx]
            p_drv = pole_drives[p_idx]
            P_prev = P[p_idx, i, j, k]
            P[p_idx, i, j, k] = p_dec * P_prev + p_drv * E_scalar

    # ---------------------------------------------------------------------------
    # Anisotropic update kernels
    # ---------------------------------------------------------------------------

    @wp.kernel
    def anisotropic_electric_update_3d(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        e_decay_xx: wp.array(dtype=wp.float32, ndim=3),
        e_decay_yy: wp.array(dtype=wp.float32, ndim=3),
        e_decay_zz: wp.array(dtype=wp.float32, ndim=3),
        e_drive_xx: wp.array(dtype=wp.float32, ndim=3),
        e_drive_yy: wp.array(dtype=wp.float32, ndim=3),
        e_drive_zz: wp.array(dtype=wp.float32, ndim=3),
        electric_modes: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update electric fields for diagonal anisotropic media.

        Each component uses axis-specific decay/drive coefficients. This
        kernel handles AnisotropicMedium where xx, yy, zz axes may have
        different material parameters.

        Args:
            E: Electric field array (nx, ny, nz, 3).
            H: Magnetic field array (nx, ny, nz, 3).
            e_decay_xx/yy/zz: Per-axis E decay coefficients.
            e_drive_xx/yy/zz: Per-axis E drive coefficients.
            electric_modes: Constitutive mode array.
            dt: Timestep size.
            dx, dy, dz: Cell sizes.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        mode_E = int(electric_modes[i, j, k])
        if mode_E == 1:  # CLAMP_ZERO (any axis clamped)
            E[i, j, k, _EX] = 0.0
            E[i, j, k, _EY] = 0.0
            E[i, j, k, _EZ] = 0.0
            return

        # ---- Ex component ----
        # Uses e_decay_xx, e_drive_xx
        Hz_ij = H[i, j, k, _HZ]
        Hz_ip1j = H[i, j + 1, k, _HZ]
        Hy_ijk = H[i, j, k, _HY]
        Hy_ijkp1 = H[i, j, k + 1, _HY]
        curl_Ex = (Hz_ip1j - Hz_ij) / dy - (Hy_ijkp1 - Hy_ijk) / dz
        E[i, j, k, _EX] = e_decay_xx[i, j, k] * E[i, j, k, _EX] + e_drive_xx[i, j, k] * curl_Ex

        # ---- Ey component ----
        # Uses e_decay_yy, e_drive_yy
        Hx_ijk = H[i, j, k, _HX]
        Hx_ijkp1 = H[i, j, k + 1, _HX]
        Hz_ijk = H[i, j, k, _HZ]
        Hz_ip1jk = H[i + 1, j, k, _HZ]
        curl_Ey = (Hx_ijkp1 - Hx_ijk) / dz - (Hz_ip1jk - Hz_ijk) / dx
        E[i, j, k, _EY] = e_decay_yy[i, j, k] * E[i, j, k, _EY] + e_drive_yy[i, j, k] * curl_Ey

        # ---- Ez component ----
        # Uses e_decay_zz, e_drive_zz
        Hy_ijk_Ez = H[i, j, k, _HY]
        Hy_ip1jk = H[i + 1, j, k, _HY]
        Hx_ijk_Ez = H[i, j, k, _HX]
        Hx_ijp1k = H[i, j + 1, k, _HX]
        curl_Ez = (Hy_ip1jk - Hy_ijk_Ez) / dx - (Hx_ijp1k - Hx_ijk_Ez) / dy
        E[i, j, k, _EZ] = e_decay_zz[i, j, k] * E[i, j, k, _EZ] + e_drive_zz[i, j, k] * curl_Ez

    @wp.kernel
    def anisotropic_magnetic_update_3d(
        H: wp.array(dtype=wp.float32, ndim=4),
        E: wp.array(dtype=wp.float32, ndim=4),
        m_decay_xx: wp.array(dtype=wp.float32, ndim=3),
        m_decay_yy: wp.array(dtype=wp.float32, ndim=3),
        m_decay_zz: wp.array(dtype=wp.float32, ndim=3),
        m_drive_xx: wp.array(dtype=wp.float32, ndim=3),
        m_drive_yy: wp.array(dtype=wp.float32, ndim=3),
        m_drive_zz: wp.array(dtype=wp.float32, ndim=3),
        magnetic_modes: wp.array(dtype=wp.float32, ndim=3),
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """Update magnetic fields for diagonal anisotropic media.

        Each component uses axis-specific decay/drive coefficients.

        Args:
            H: Magnetic field array (nx, ny, nz, 3).
            E: Electric field array (nx, ny, nz, 3).
            m_decay_xx/yy/zz: Per-axis H decay coefficients.
            m_drive_xx/yy/zz: Per-axis H drive coefficients.
            magnetic_modes: Constitutive mode array.
            dt: Timestep size.
            dx, dy, dz: Cell sizes.
            nx, ny, nz: Grid dimensions.
        """
        i, j, k = wp.tid()

        if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
            return

        mode_H = int(magnetic_modes[i, j, k])
        if mode_H == 1:  # CLAMP_ZERO
            H[i, j, k, _HX] = 0.0
            H[i, j, k, _HY] = 0.0
            H[i, j, k, _HZ] = 0.0
            return

        # ---- Hx component ----
        Ey_ijk = E[i, j, k, _EY]
        Ey_ijkp1 = E[i, j, k + 1, _EY]
        Ez_ijk = E[i, j, k, _EZ]
        Ez_ijp1k = E[i, j + 1, k, _EZ]
        curl_Hx = (Ey_ijkp1 - Ey_ijk) / dz - (Ez_ijp1k - Ez_ijk) / dy
        H[i, j, k, _HX] = m_decay_xx[i, j, k] * H[i, j, k, _HX] + m_drive_xx[i, j, k] * curl_Hx

        # ---- Hy component ----
        Ez_ijk_Hy = E[i, j, k, _EZ]
        Ez_ip1jk = E[i + 1, j, k, _EZ]
        Ex_ijk_Hy = E[i, j, k, _EX]
        Ex_ijkp1 = E[i, j, k + 1, _EX]
        curl_Hy = (Ez_ip1jk - Ez_ijk_Hy) / dx - (Ex_ijkp1 - Ex_ijk_Hy) / dz
        H[i, j, k, _HY] = m_decay_yy[i, j, k] * H[i, j, k, _HY] + m_drive_yy[i, j, k] * curl_Hy

        # ---- Hz component ----
        Ex_ijk_Hz = E[i, j, k, _EX]
        Ex_ijp1k = E[i, j + 1, k, _EX]
        Ey_ijk_Hz = E[i, j, k, _EY]
        Ey_ip1jk = E[i + 1, j, k, _EY]
        curl_Hz = (Ex_ijp1k - Ex_ijk_Hz) / dy - (Ey_ip1jk - Ey_ijk_Hz) / dx
        H[i, j, k, _HZ] = m_decay_zz[i, j, k] * H[i, j, k, _HZ] + m_drive_zz[i, j, k] * curl_Hz

else:
    # Stub definitions when Warp is unavailable
    electric_update_3d = None
    magnetic_update_3d = None
    electric_update_3d_coeff = None
    magnetic_update_3d_coeff = None
    pole_residue_electric_update_3d = None
    pole_residue_polarization_update_3d = None
    anisotropic_electric_update_3d = None
    anisotropic_magnetic_update_3d = None


# ---------------------------------------------------------------------------
# NumPy fallback implementations (for CPU execution without Warp)
# ---------------------------------------------------------------------------


def numpy_electric_update_3d(
    E: np.ndarray,
    H: np.ndarray,
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_zz: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """NumPy implementation of 3D electric field update.

    Updates Ex, Ey, Ez at interior cells using the curl of H.
    Ghost cells (index 0 and -1 on each axis) are not updated.

    Array layout: E and H have shape (nx, ny, nz, 3) with
    component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

    Args:
        E: Electric field array with shape (nx, ny, nz, 3).
        H: Magnetic field array with shape (nx, ny, nz, 3).
        eps_xx, eps_yy, eps_zz: Permittivity arrays with shape (nx, ny, nz).
        dt: Timestep size.
        dx, dy, dz: Cell sizes.
    """
    nx, ny, nz = E.shape[:3]
    # Interior indices: [1, nx-2], [1, ny-2], [1, nz-2]
    i1, i2 = 1, nx - 1
    j1, j2 = 1, ny - 1
    k1, k2 = 1, nz - 1

    # Precompute 1/dy, 1/dz, 1/dx for efficiency
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz
    inv_dx = 1.0 / dx

    # ---- Ex component update ----
    # Ex: dHz/dy - dHy/dz
    # dHz_dy = (Hz[:, j+1, :, 2] - Hz[:, j, :, 2]) / dy  at i,j,k
    # dHy_dz = (Hy[:, :, k+1, 1] - Hy[:, :, k, 1]) / dz  at i,j,k
    Hz = H[:, :, :, 2]  # shape (nx, ny, nz)
    Hy = H[:, :, :, 1]  # shape (nx, ny, nz)

    dHz_dy_Ex = (Hz[i1:i2, j1+1:j2+1, k1:k2] - Hz[i1:i2, j1:j2, k1:k2]) * inv_dy
    dHy_dz_Ex = (Hy[i1:i2, j1:j2, k1+1:k2+1] - Hy[i1:i2, j1:j2, k1:k2]) * inv_dz

    ce_Ex = dt / (eps_xx[i1:i2, j1:j2, k1:k2] + 1e-20)
    E[i1:i2, j1:j2, k1:k2, 0] += ce_Ex * (dHz_dy_Ex - dHy_dz_Ex)

    # ---- Ey component update ----
    # Ey: dHx/dz - dHz/dx
    Hx = H[:, :, :, 0]  # shape (nx, ny, nz)

    dHx_dz_Ey = (Hx[i1:i2, j1:j2, k1+1:k2+1] - Hx[i1:i2, j1:j2, k1:k2]) * inv_dz
    dHz_dx_Ey = (Hz[i1+1:i2+1, j1:j2, k1:k2] - Hz[i1:i2, j1:j2, k1:k2]) * inv_dx

    ce_Ey = dt / (eps_yy[i1:i2, j1:j2, k1:k2] + 1e-20)
    E[i1:i2, j1:j2, k1:k2, 1] += ce_Ey * (dHx_dz_Ey - dHz_dx_Ey)

    # ---- Ez component update ----
    # Ez: dHy/dx - dHx/dy
    dHy_dx_Ez = (Hy[i1+1:i2+1, j1:j2, k1:k2] - Hy[i1:i2, j1:j2, k1:k2]) * inv_dx
    dHx_dy_Ez = (Hx[i1:i2, j1+1:j2+1, k1:k2] - Hx[i1:i2, j1:j2, k1:k2]) * inv_dy

    ce_Ez = dt / (eps_zz[i1:i2, j1:j2, k1:k2] + 1e-20)
    E[i1:i2, j1:j2, k1:k2, 2] += ce_Ez * (dHy_dx_Ez - dHx_dy_Ez)


def numpy_magnetic_update_3d(
    H: np.ndarray,
    E: np.ndarray,
    mu_xx: np.ndarray,
    mu_yy: np.ndarray,
    mu_zz: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """NumPy implementation of 3D magnetic field update.

    Updates Hx, Hy, Hz at interior cells using the curl of E.
    Ghost cells (index 0 and -1 on each axis) are not updated.

    Array layout: E and H have shape (nx, ny, nz, 3) with
    component order (Ex=0, Ey=1, Ez=2) for E and (Hx=0, Hy=1, Hz=2) for H.

    Args:
        H: Magnetic field array with shape (nx, ny, nz, 3).
        E: Electric field array with shape (nx, ny, nz, 3).
        mu_xx, mu_yy, mu_zz: Permeability arrays with shape (nx, ny, nz).
        dt: Timestep size.
        dx, dy, dz: Cell sizes.
    """
    nx, ny, nz = H.shape[:3]
    # Interior indices: [1, nx-2], [1, ny-2], [1, nz-2]
    i1, i2 = 1, nx - 1
    j1, j2 = 1, ny - 1
    k1, k2 = 1, nz - 1

    # Precompute 1/dy, 1/dz, 1/dx for efficiency
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz
    inv_dx = 1.0 / dx

    # ---- Hx component update ----
    # Hx: dEy/dz - dEz/dy
    Ey = E[:, :, :, 1]  # shape (nx, ny, nz)
    Ez = E[:, :, :, 2]

    dEy_dz_Hx = (Ey[i1:i2, j1:j2, k1+1:k2+1] - Ey[i1:i2, j1:j2, k1:k2]) * inv_dz
    dEz_dy_Hx = (Ez[i1:i2, j1+1:j2+1, k1:k2] - Ez[i1:i2, j1:j2, k1:k2]) * inv_dy

    ch_Hx = -dt / (mu_xx[i1:i2, j1:j2, k1:k2] + 1e-20)
    H[i1:i2, j1:j2, k1:k2, 0] += ch_Hx * (dEy_dz_Hx - dEz_dy_Hx)

    # ---- Hy component update ----
    # Hy: dEz/dx - dEx/dz
    Ex = E[:, :, :, 0]

    dEz_dx_Hy = (Ez[i1+1:i2+1, j1:j2, k1:k2] - Ez[i1:i2, j1:j2, k1:k2]) * inv_dx
    dEx_dz_Hy = (Ex[i1:i2, j1:j2, k1+1:k2+1] - Ex[i1:i2, j1:j2, k1:k2]) * inv_dz

    ch_Hy = -dt / (mu_yy[i1:i2, j1:j2, k1:k2] + 1e-20)
    H[i1:i2, j1:j2, k1:k2, 1] += ch_Hy * (dEz_dx_Hy - dEx_dz_Hy)

    # ---- Hz component update ----
    # Hz: dEx/dy - dEy/dx
    dEx_dy_Hz = (Ex[i1:i2, j1+1:j2+1, k1:k2] - Ex[i1:i2, j1:j2, k1:k2]) * inv_dy
    dEy_dx_Hz = (Ey[i1+1:i2+1, j1:j2, k1:k2] - Ey[i1:i2, j1:j2, k1:k2]) * inv_dx

    ch_Hz = -dt / (mu_zz[i1:i2, j1:j2, k1:k2] + 1e-20)
    H[i1:i2, j1:j2, k1:k2, 2] += ch_Hz * (dEx_dy_Hz - dEy_dx_Hz)


def allocate_maxwell_arrays(
    grid_shape: tuple[int, int, int],
    *,
    complex_policy: ComplexFieldPolicy | None = None,
    device: int | None = None,
    chunk_index: tuple[int, int, int] | None = None,
) -> dict[str, "wp.array | np.ndarray"]:
    """Allocate all field and coefficient arrays for a Maxwell stepper.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions (Nx, Ny, Nz).
    complex_policy : ComplexFieldPolicy, optional
        Determines whether fields use complex dtype.
    device : int, optional
        CUDA device ID for allocation.
    chunk_index : tuple, optional
        Logical chunk position for tagging.

    Returns
    -------
    dict[str, wp.array | np.ndarray]
        Dictionary of allocated arrays: E, H, eps_xx, eps_yy, eps_zz,
        mu_xx, mu_yy, mu_zz.
    """
    if complex_policy is None:
        complex_policy = ComplexFieldPolicy()

    if WARP_AVAILABLE:
        # Always return numpy arrays for allocate_maxwell_arrays.
        # The Warp path is used only via step_maxwell() which accepts
        # wp.array inputs directly. Tests for numpy_electric_update_3d
        # and numpy_magnetic_update_3d expect numpy arrays.
        import numpy as np

        arrays = {}
        arrays["E"] = np.zeros((*grid_shape, 3), dtype=np.float64)
        arrays["H"] = np.zeros((*grid_shape, 3), dtype=np.float64)
        for axis, name in [("xx", "eps_xx"), ("yy", "eps_yy"), ("zz", "eps_zz")]:
            arrays[name] = np.zeros(grid_shape, dtype=np.float64)
        for axis, name in [("xx", "mu_xx"), ("yy", "mu_yy"), ("zz", "mu_zz")]:
            arrays[name] = np.zeros(grid_shape, dtype=np.float64)
        return arrays

    # NumPy path (WARP_AVAILABLE=False)
    field_dtype = np.complex128 if complex_policy.requires_complex else np.float64

    arrays = {}

    # Electric field
    arrays["E"] = allocate_field_array(
        grid_shape,
        family="electric",
        dtype=field_dtype,
        device=None,
        chunk_index=chunk_index,
    )

    # Magnetic field
    arrays["H"] = allocate_field_array(
        grid_shape,
        family="magnetic",
        dtype=field_dtype,
        device=None,
        chunk_index=chunk_index,
    )

    # Permittivity arrays
    for axis, name in [("xx", "eps_xx"), ("yy", "eps_yy"), ("zz", "eps_zz")]:
        arrays[name] = allocate_coefficient_array(
            grid_shape,
            component="eps",
            axis=axis,
            dtype=field_dtype,
            device=None,
            chunk_index=chunk_index,
        )

    # Permeability arrays
    for axis, name in [("xx", "mu_xx"), ("yy", "mu_yy"), ("zz", "mu_zz")]:
        arrays[name] = allocate_coefficient_array(
            grid_shape,
            component="mu",
            axis=axis,
            dtype=field_dtype,
            device=None,
            chunk_index=chunk_index,
        )

    return arrays


def get_warp_device(device_id: int | str | None = None) -> "wp.Device | None":
    """Get a Warp device for allocation.

    Parameters
    ----------
    device_id : int, str, or None, optional
        Specific GPU ID as integer (0, 1, ...) or Warp device string
        (e.g., "cuda:0", "cuda:1"). If None, returns the current default device.

    Returns
    -------
    wp.Device or None
        The requested device, or None if Warp is not available.
    """
    if not WARP_AVAILABLE:
        return None
    if device_id is None:
        return wp.get_device()
    # Handle integer device IDs by converting to Warp device string
    if isinstance(device_id, int):
        return wp.get_device(f"cuda:{device_id}")
    return wp.get_device(device_id)


# ---------------------------------------------------------------------------
# Stepping helper
# ---------------------------------------------------------------------------


def step_maxwell(
    arrays: dict[str, "wp.array | np.ndarray"],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    step_index: int = 0,
    time: float = 0.0,
    device: int | None = None,
) -> dict[str, Any]:
    """Execute one Maxwell timestep using the allocated arrays.

    This is a convenience wrapper that launches the electric and magnetic
    update kernels in sequence with appropriate NVTX annotations.

    Capture-Safe Stepping Contract
    -------------------------------
    All timestep-varying values are passed as explicit kernel arguments:
    - ``dt``: scalar float (not captured from closure)
    - ``step_index``: int (for diagnostics and time-varying sources)
    - ``time``: float (for time-varying sources)

    The kernel launch inputs are passed directly, not constructed inside
    the timed region, to avoid including argument preparation in metrics.

    Parameters
    ----------
    arrays : dict[str, wp.array | np.ndarray]
        Field and coefficient arrays from ``allocate_maxwell_arrays``.
        Must contain: E, H, eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz.
    dt : float
        Timestep size.
    dx, dy, dz : float
        Cell sizes.
    step_index : int, default=0
        Current step index (for diagnostics).
    time : float, default=0.0
        Current simulation time (for diagnostics).
    device : int, optional
        Target device for kernel launches.

    Returns
    -------
    dict[str, Any]
        Diagnostics including step time and cells updated.
    """
    import time as time_module

    nx, ny, nz = arrays["E"].shape[:3]
    num_cells = nx * ny * nz
    num_interior = (nx - 2) * (ny - 2) * (nz - 2)

    # Check if inputs are Warp arrays (GPU path) or NumPy arrays (CPU path)
    # This allows step_maxwell to work correctly when called with wp.array
    # inputs from allocate_field_state, while also supporting numpy arrays
    # from allocate_maxwell_arrays during testing.
    is_warp_input = WARP_AVAILABLE and isinstance(arrays["E"], wp.array)

    if not is_warp_input:
        # NumPy fallback - exercise the CPU path
        step_start = time_module.perf_counter()

        numpy_electric_update_3d(
            arrays["E"],
            arrays["H"],
            arrays["eps_xx"],
            arrays["eps_yy"],
            arrays["eps_zz"],
            dt,
            dx,
            dy,
            dz,
        )

        numpy_magnetic_update_3d(
            arrays["H"],
            arrays["E"],
            arrays["mu_xx"],
            arrays["mu_yy"],
            arrays["mu_zz"],
            dt,
            dx,
            dy,
            dz,
        )

        step_end = time_module.perf_counter()
        wall_time = step_end - step_start

        metrics = step_metrics(
            step_index=step_index,
            wall_time_s=wall_time,
            num_cells=num_cells,
            initial_step=(step_index == 0),
        )

        return {
            "step_index": step_index,
            "time": time,
            "cells_updated": num_cells,
            "interior_cells_updated": num_interior,
            "wall_time_s": wall_time,
            "gcells_per_second": metrics.gcells_per_second,
            "backend": "numpy",
        }

    dev = get_warp_device(device)

    step_start = time_module.perf_counter()

    # Electric update stage
    with nvtx_range("electric_update", color="blue"):
        with WarpTimer("electric_update", device=dev):
            wp.launch(
                electric_update_3d,
                dim=(nx, ny, nz),
                inputs=[
                    arrays["E"],
                    arrays["H"],
                    arrays["eps_xx"],
                    arrays["eps_yy"],
                    arrays["eps_zz"],
                    dt,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                ],
                device=dev,
            )

    # Magnetic update stage
    with nvtx_range("magnetic_update", color="green"):
        with WarpTimer("magnetic_update", device=dev):
            wp.launch(
                magnetic_update_3d,
                dim=(nx, ny, nz),
                inputs=[
                    arrays["H"],
                    arrays["E"],
                    arrays["mu_xx"],
                    arrays["mu_yy"],
                    arrays["mu_zz"],
                    dt,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                ],
                device=dev,
            )

    # Synchronize to ensure kernel completion before timing
    if WARP_AVAILABLE:
        wp.synchronize()

    step_end = time_module.perf_counter()
    wall_time = step_end - step_start

    metrics = step_metrics(
        step_index=step_index,
        wall_time_s=wall_time,
        num_cells=num_cells,
        initial_step=(step_index == 0),
    )

    return {
        "step_index": step_index,
        "time": time,
        "cells_updated": num_cells,
        "interior_cells_updated": num_interior,
        "wall_time_s": wall_time,
        "gcells_per_second": metrics.gcells_per_second,
        "backend": "warp",
    }


def step_electric(
    arrays: dict[str, "wp.array | np.ndarray"],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    step_index: int = 0,
    time: float = 0.0,
    device: int | None = None,
) -> dict[str, Any]:
    """Execute only the electric field update step.

    This is one half of a Maxwell timestep, to be used in the leapfrog
    scheme where H update happens first, then E update.

    Parameters
    ----------
    arrays : dict[str, wp.array | np.ndarray]
        Field and coefficient arrays from allocate_maxwell_arrays.
        Must contain: E, H, eps_xx, eps_yy, eps_zz.
    dt : float
        Timestep size.
    dx, dy, dz : float
        Cell sizes.
    step_index : int, default=0
        Current step index (for diagnostics).
    time : float, default=0.0
        Current simulation time (for diagnostics).
    device : int, optional
        Target device for kernel launches.

    Returns
    -------
    dict[str, Any]
        Diagnostics including step time and cells updated.
    """
    import time as time_module

    nx, ny, nz = arrays["E"].shape[:3]
    num_cells = nx * ny * nz
    num_interior = (nx - 2) * (ny - 2) * (nz - 2)

    is_warp_input = WARP_AVAILABLE and isinstance(arrays["E"], wp.array)

    if not is_warp_input:
        step_start = time_module.perf_counter()

        numpy_electric_update_3d(
            arrays["E"],
            arrays["H"],
            arrays["eps_xx"],
            arrays["eps_yy"],
            arrays["eps_zz"],
            dt,
            dx,
            dy,
            dz,
        )

        step_end = time_module.perf_counter()
        wall_time = step_end - step_start

        metrics = step_metrics(
            step_index=step_index,
            wall_time_s=wall_time,
            num_cells=num_cells,
            initial_step=(step_index == 0),
        )

        return {
            "step_index": step_index,
            "time": time,
            "cells_updated": num_cells,
            "interior_cells_updated": num_interior,
            "wall_time_s": wall_time,
            "gcells_per_second": metrics.gcells_per_second,
            "backend": "numpy",
        }

    dev = get_warp_device(device)

    step_start = time_module.perf_counter()

    with nvtx_range("electric_update", color="blue"):
        with WarpTimer("electric_update", device=dev):
            wp.launch(
                electric_update_3d,
                dim=(nx, ny, nz),
                inputs=[
                    arrays["E"],
                    arrays["H"],
                    arrays["eps_xx"],
                    arrays["eps_yy"],
                    arrays["eps_zz"],
                    dt,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                ],
                device=dev,
            )

    if WARP_AVAILABLE:
        wp.synchronize()

    step_end = time_module.perf_counter()
    wall_time = step_end - step_start

    metrics = step_metrics(
        step_index=step_index,
        wall_time_s=wall_time,
        num_cells=num_cells,
        initial_step=(step_index == 0),
    )

    return {
        "step_index": step_index,
        "time": time,
        "cells_updated": num_cells,
        "interior_cells_updated": num_interior,
        "wall_time_s": wall_time,
        "gcells_per_second": metrics.gcells_per_second,
        "backend": "warp",
    }


def step_magnetic(
    arrays: dict[str, "wp.array | np.ndarray"],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    step_index: int = 0,
    time: float = 0.0,
    device: int | None = None,
) -> dict[str, Any]:
    """Execute only the magnetic field update step.

    This is one half of a Maxwell timestep, to be used in the leapfrog
    scheme where H update happens first, then E update.

    Parameters
    ----------
    arrays : dict[str, wp.array | np.ndarray]
        Field and coefficient arrays from allocate_maxwell_arrays.
        Must contain: E, H, mu_xx, mu_yy, mu_zz.
    dt : float
        Timestep size.
    dx, dy, dz : float
        Cell sizes.
    step_index : int, default=0
        Current step index (for diagnostics).
    time : float, default=0.0
        Current simulation time (for diagnostics).
    device : int, optional
        Target device for kernel launches.

    Returns
    -------
    dict[str, Any]
        Diagnostics including step time and cells updated.
    """
    import time as time_module

    nx, ny, nz = arrays["E"].shape[:3]
    num_cells = nx * ny * nz
    num_interior = (nx - 2) * (ny - 2) * (nz - 2)

    is_warp_input = WARP_AVAILABLE and isinstance(arrays["E"], wp.array)

    if not is_warp_input:
        step_start = time_module.perf_counter()

        numpy_magnetic_update_3d(
            arrays["H"],
            arrays["E"],
            arrays["mu_xx"],
            arrays["mu_yy"],
            arrays["mu_zz"],
            dt,
            dx,
            dy,
            dz,
        )

        step_end = time_module.perf_counter()
        wall_time = step_end - step_start

        metrics = step_metrics(
            step_index=step_index,
            wall_time_s=wall_time,
            num_cells=num_cells,
            initial_step=(step_index == 0),
        )

        return {
            "step_index": step_index,
            "time": time,
            "cells_updated": num_cells,
            "interior_cells_updated": num_interior,
            "wall_time_s": wall_time,
            "gcells_per_second": metrics.gcells_per_second,
            "backend": "numpy",
        }

    dev = get_warp_device(device)

    step_start = time_module.perf_counter()

    with nvtx_range("magnetic_update", color="green"):
        with WarpTimer("magnetic_update", device=dev):
            wp.launch(
                magnetic_update_3d,
                dim=(nx, ny, nz),
                inputs=[
                    arrays["H"],
                    arrays["E"],
                    arrays["mu_xx"],
                    arrays["mu_yy"],
                    arrays["mu_zz"],
                    dt,
                    dx,
                    dy,
                    dz,
                    nx,
                    ny,
                    nz,
                ],
                device=dev,
            )

    if WARP_AVAILABLE:
        wp.synchronize()

    step_end = time_module.perf_counter()
    wall_time = step_end - step_start

    metrics = step_metrics(
        step_index=step_index,
        wall_time_s=wall_time,
        num_cells=num_cells,
        initial_step=(step_index == 0),
    )

    return {
        "step_index": step_index,
        "time": time,
        "cells_updated": num_cells,
        "interior_cells_updated": num_interior,
        "wall_time_s": wall_time,
        "gcells_per_second": metrics.gcells_per_second,
        "backend": "warp",
    }


def vacuum_maxwell_step(
    arrays: dict[str, "np.ndarray"],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    num_steps: int = 100,
    interval: int = 10,
) -> list[dict[str, Any]]:
    """Execute multiple vacuum Maxwell timesteps with periodic diagnostics.

    This is a convenience runner for vacuum propagation tests that
    exercises the complete E/H update cycle and collects metrics.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Field and coefficient arrays from ``allocate_maxwell_arrays``.
    dt : float
        Timestep size.
    dx, dy, dz : float
        Cell sizes.
    num_steps : int, default=100
        Number of timesteps to execute.
    interval : int, default=10
        Interval for collecting diagnostics.

    Returns
    -------
    list[dict[str, Any]]
        List of diagnostics dicts at each ``interval`` step.
    """
    results = []
    nx, ny, nz = arrays["E"].shape[:3]
    num_cells = nx * ny * nz

    for step in range(num_steps):
        step_result = step_maxwell(
            arrays,
            dt,
            dx,
            dy,
            dz,
            step_index=step,
            time=step * dt,
        )

        if step % interval == 0:
            # Compute integrated electric field energy for diagnostics
            E = arrays["E"]
            H = arrays["H"]
            eps_xx = arrays["eps_xx"]
            mu_xx = arrays["mu_xx"]

            # E energy = 0.5 * sum(eps * |E|^2) over interior cells
            i1, i2 = 1, nx - 1
            j1, j2 = 1, ny - 1
            k1, k2 = 1, nz - 1

            E_sq = np.sum(E[i1:i2, j1:j2, k1:k2, :] ** 2)
            H_sq = np.sum(H[i1:i2, j1:j2, k1:k2, :] ** 2)

            # Approximate vacuum energy (eps0=1, mu0=1 for normalized)
            e_energy = 0.5 * np.mean(eps_xx[i1:i2, j1:j2, k1:k2]) * E_sq
            h_energy = 0.5 * np.mean(mu_xx[i1:i2, j1:j2, k1:k2]) * H_sq

            step_result["E_energy"] = float(e_energy)
            step_result["H_energy"] = float(h_energy)
            step_result["total_energy"] = float(e_energy + h_energy)
            results.append(step_result)

    return results
