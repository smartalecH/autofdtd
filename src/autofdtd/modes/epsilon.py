"""Epsilon tensor sampling for mode-solver cross-sections.

Provides utilities to sample the full permittivity tensor from a Scene
onto a rectilinear cross-sectional grid, producing a callback compatible
with the VectorModesolver-inspired eigenvalue formulation.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

from autofdtd.compiler.materials import (
    compile_medium_coefficients,
    sample_scene_mediums,
)
from autofdtd.core.containers import Scene
from autofdtd.core.validation import normalize_vec3
from autofdtd.materials import medium_model_from_value


# Type alias for the epsilon callback used by the mode solver
# Returns (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz) at a given (x, y) point
EpsilonCallback = Callable[[float, float], tuple[float, float, float, float, float]]


def _build_uniform_grid(
    center: float, size: float, num_cells: int
) -> tuple[float, ...]:
    """Build uniform cell-center coordinates along one axis."""
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    half = size / 2.0
    start = center - half + size / (2.0 * num_cells)
    step = size / num_cells
    return tuple(start + step * i for i in range(num_cells))


def _resolve_cell_centers(
    cross_section: "ModeSolverCrossSection",
    sim_center: tuple[float, float, float],
    sim_size: tuple[float, float, float],
    wavelength: float,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Resolve the in-plane grid coordinates for a cross-sectional plane.

    Returns:
        x_coords: cell-center x coordinates
        y_coords: cell-center y coordinates
        cell_size: uniform in-plane cell size
    """
    normal_axis = cross_section.normal_axis
    position = cross_section.position

    # Compute size along each axis
    sx, sy, sz = sim_size

    if normal_axis == 0:
        in_plane_axes = (1, 2)
        in_plane_sizes = (sy, sz)
        offset = (sim_center[0], position)
    elif normal_axis == 1:
        in_plane_axes = (0, 2)
        in_plane_sizes = (sx, sz)
        offset = (position, sim_center[1])
    else:  # normal_axis == 2
        in_plane_axes = (0, 1)
        in_plane_sizes = (sx, sy)
        offset = (sim_center[0], sim_center[1])

    axis_a, axis_b = in_plane_axes
    size_a = in_plane_sizes[0]
    size_b = in_plane_sizes[1]

    # Use the wavelength and a simple resolution target to set grid density
    # Aim for roughly 10 cells per wavelength in the core region
    base_resolution = wavelength / 10.0
    num_cells_a = max(3, min(200, max(3, int(math.ceil(size_a / base_resolution)))))
    num_cells_b = max(3, min(200, max(3, int(math.ceil(size_b / base_resolution)))))

    # Use the coarser resolution to set uniform cell size
    cell_size_a = size_a / num_cells_a
    cell_size_b = size_b / num_cells_b
    cell_size = min(cell_size_a, cell_size_b)

    # Recompute to ensure consistent cell sizes
    num_cells_a = max(3, min(200, int(math.ceil(size_a / cell_size))))
    num_cells_b = max(3, min(200, int(math.ceil(size_b / cell_size))))

    # Cell-center coordinates
    coords_a = _build_uniform_grid(offset[0] if axis_a == 0 else (offset[1] if axis_a == 1 else 0), size_a, num_cells_a)
    coords_b = _build_uniform_grid(offset[0] if axis_b == 0 else (offset[1] if axis_b == 1 else 0), size_b, num_cells_b)

    if normal_axis == 0:
        x_coords = coords_a
        y_coords = coords_b
    elif normal_axis == 1:
        x_coords = coords_a
        y_coords = coords_b
    else:
        x_coords = coords_a
        y_coords = coords_b

    return x_coords, y_coords, cell_size


def sample_scene_epsilon_tensor_2d(
    scene: Scene,
    cross_section: "ModeSolverCrossSection",
    sim_center: tuple[float, float, float],
    sim_size: tuple[float, float, float],
    wavelength: float,
    dt: float,
) -> tuple[EpsilonCallback, tuple[float, ...], tuple[float, ...], float]:
    """Sample a scene's permittivity tensor onto a 2D cross-sectional grid.

    Args:
        scene: The scene containing structures and background medium
        cross_section: Specification of the cross-sectional plane
        sim_center: Simulation domain center (x, y, z)
        sim_size: Simulation domain size (sx, sy, sz)
        wavelength: Wavelength for dispersion evaluation (m)
        dt: Timestep for coefficient compilation (s)

    Returns:
        Tuple of (epsilon_callback, x_coords, y_coords, cell_size) where:
            - epsilon_callback(x, y) returns (eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)
            - x_coords: cell-center x coordinates of the grid
            - y_coords: cell-center y coordinates of the grid
            - cell_size: uniform in-plane cell size
    """
    from autofdtd.modes.models import ModeSolverCrossSection

    if not isinstance(cross_section, ModeSolverCrossSection):
        raise TypeError("cross_section must be a ModeSolverCrossSection")

    x_coords, y_coords, cell_size = _resolve_cell_centers(
        cross_section, sim_center, sim_size, wavelength
    )

    normal_axis = cross_section.normal_axis
    position = cross_section.position

    # Build sample points for the scene material sampler
    sample_points = []
    point_to_idx: dict[tuple[float, float], int] = {}
    for i, xv in enumerate(x_coords):
        for j, yv in enumerate(y_coords):
            # Reconstruct the full 3D point
            if normal_axis == 0:
                pt = (position, xv, yv)
            elif normal_axis == 1:
                pt = (xv, position, yv)
            else:
                pt = (xv, yv, position)
            sample_points.append(pt)
            point_to_idx[(xv, yv)] = i * len(y_coords) + j

    # Sample the scene
    samples = sample_scene_mediums(scene, points=sample_points)

    # Build a lookup from linear index to medium
    medium_at_idx: dict[int, object] = {}
    for sample in samples:
        pt = sample.point
        if normal_axis == 0:
            xv, yv = pt[1], pt[2]
        elif normal_axis == 1:
            xv, yv = pt[0], pt[2]
        else:
            xv, yv = pt[0], pt[1]
        idx = point_to_idx[(xv, yv)]
        medium_at_idx[idx] = sample.medium

    # Pre-compile coefficients for each unique medium
    unique_media: dict[int, tuple[float, float, float, float, float]] = {}
    for idx, medium in medium_at_idx.items():
        try:
            coeff = compile_medium_coefficients(medium, dt=dt)
        except Exception:
            # Fall back to static permittivity for uncompilable media
            normalized = medium_model_from_value(medium)
            eps_val = getattr(normalized, "permittivity", 1.0)
            coeff = None
            unique_media[idx] = (float(eps_val), 0.0, 0.0, float(eps_val), float(eps_val))
            continue

        # Extract the effective scalar permittivity for each component
        if hasattr(coeff, "xx"):
            # Anisotropic - use diagonal components
            xx_coeff = coeff.xx
            yy_coeff = coeff.yy
            zz_coeff = coeff.zz
            eps_xx = getattr(xx_coeff, "permittivity", 1.0) if hasattr(xx_coeff, "permittivity") else 1.0
            eps_yy = getattr(yy_coeff, "permittivity", 1.0) if hasattr(yy_coeff, "permittivity") else 1.0
            eps_zz = getattr(zz_coeff, "permittivity", 1.0) if hasattr(zz_coeff, "permittivity") else 1.0
            unique_media[idx] = (float(eps_xx), 0.0, 0.0, float(eps_yy), float(eps_zz))
        else:
            # Isotropic
            eps_val = getattr(coeff, "permittivity", 1.0)
            unique_media[idx] = (float(eps_val), 0.0, 0.0, float(eps_val), float(eps_val))

    # Pre-sample all grid points into a 2D array
    nx = len(x_coords)
    ny = len(y_coords)
    eps_xx_grid = np.zeros((nx, ny))
    eps_xy_grid = np.zeros((nx, ny))
    eps_yx_grid = np.zeros((nx, ny))
    eps_yy_grid = np.zeros((nx, ny))
    eps_zz_grid = np.zeros((nx, ny))

    for i, xv in enumerate(x_coords):
        for j, yv in enumerate(y_coords):
            idx = i * ny + j
            if idx in unique_media:
                eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = unique_media[idx]
            else:
                # Background
                eps_xx = eps_yy = eps_zz = 1.0
                eps_xy = eps_yx = 0.0
            eps_xx_grid[i, j] = eps_xx
            eps_xy_grid[i, j] = eps_xy
            eps_yx_grid[i, j] = eps_yx
            eps_yy_grid[i, j] = eps_yy
            eps_zz_grid[i, j] = eps_zz

    # Find cell-center coordinates (the epsilon values are at cell centers)
    xc = tuple((x_coords[i] + x_coords[i + 1]) / 2 if i + 1 < len(x_coords) else x_coords[i] for i in range(len(x_coords)))
    yc = tuple((y_coords[j] + y_coords[j + 1]) / 2 if j + 1 < len(y_coords) else y_coords[j] for j in range(len(y_coords)))

    def epsilon_callback(x: float, y: float) -> tuple[float, float, float, float, float]:
        # Find nearest cell center
        i = min(max(0, int(round((x - x_coords[0]) / (x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 0))), nx - 1)
        j = min(max(0, int(round((y - y_coords[0]) / (y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 0))), ny - 1)
        return (
            float(eps_xx_grid[i, j]),
            float(eps_xy_grid[i, j]),
            float(eps_yx_grid[i, j]),
            float(eps_yy_grid[i, j]),
            float(eps_zz_grid[i, j]),
        )

    return epsilon_callback, x_coords, y_coords, cell_size


__all__ = [
    "EpsilonCallback",
    "sample_scene_epsilon_tensor_2d",
]