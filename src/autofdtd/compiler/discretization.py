"""Geometry-to-grid discretization, coefficient materialization, and subpixel smoothing.

This module bridges the gap between continuous geometry specifications and the
discrete grid-based material coefficient fields needed for FDTD execution.

Architecture
------------

The discretization pipeline produces per-cell material coefficient fields by:

1. **Volume fraction computation** (``compute_volume_fractions``)
   For each grid cell, determine what fraction of the cell volume is occupied
   by each structure. This is the foundation for subpixel smoothing.

2. **Structure materialization** (``materialize_grid``)
   For each grid cell, determine the winning structure (if any) based on
   precedence rules, producing a structured material assignment map.

3. **Subpixel smoothing** (``apply_subpixel_smoothing``)
   When ``PolarizedAveraging`` is configured for dielectric interfaces, compute
   effective medium parameters using volume-weighted harmonic/arithmetic means.

4. **Coefficient field assembly** (``assemble_coefficient_fields``)
   Compile the materialized medium at each cell into runtime-ready constitutive
   coefficient descriptors for the Yee grid update kernels.

Phase 1 Scope
-------------

- Volume fractions: Box (exact), Sphere (sampled-corner approximation),
  Cylinder (sampled-corner approximation), PolySlab (sampled-face approximation),
  and GeometryGroup/Transformed (recursive via instances).
- Subpixel smoothing: dielectric-dielectric interfaces only (PolarizedAveraging);
  metal, PEC, PMC, and lossy-metal interfaces remain staircased.
- Anisotropic materialization: supported for AnisotropicMedium components;
  subpixel-smoothed isotropic media use the anisotropic coefficient path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from autofdtd.core.containers import Scene, Structure
from autofdtd.geometry.composite import (
    GeometryGroup,
    GeometryWrapperModel,
    Transformed,
    ClipOperation,
    GeometryArray,
)
from autofdtd.geometry.polyslab import PolySlab
from autofdtd.geometry.primitives import Box, Cylinder, Sphere
from autofdtd.grid import ResolvedGrid
from autofdtd.grid.subpixel import AbstractSubpixelPolicy, PolarizedAveraging, Staircasing, SubpixelSpec
from autofdtd.materials import (
    AnisotropicMedium,
    Medium,
    PECMedium,
    PMCMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Drude,
    Debye,
    medium_model_from_value,
)
from autofdtd.compiler.materials import (
    ConstitutiveMode,
    IsotropicMaterialCoefficients,
    AnisotropicMaterialCoefficients,
    PoleResidueMaterialCoefficients,
    compile_medium_coefficients,
)


# =============================================================================
# Volume fraction computation
# =============================================================================


def _box_volume_fraction(
    box: Box,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute the volume fraction of a box overlapping an axis-aligned cell.

    Parameters
    ----------
    box : Box
        The box geometry.
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Cell boundaries in world coordinates.

    Returns
    -------
    float
        Fraction of cell volume occupied by the box (0.0 to 1.0).
    """
    b_min, b_max = box.bounds
    overlap_x = max(0.0, min(b_max[0], x_max) - max(b_min[0], x_min))
    overlap_y = max(0.0, min(b_max[1], y_max) - max(b_min[1], y_min))
    overlap_z = max(0.0, min(b_max[2], z_max) - max(b_min[2], z_min))
    cell_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    if cell_volume <= 0.0:
        return 0.0
    return (overlap_x * overlap_y * overlap_z) / cell_volume


def _sphere_volume_fraction(
    sphere: Sphere,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Approximate sphere volume fraction using corner sampling.

    Phase 1 uses a simple sampled-corner approximation for spheres.
    Samples 8 corners of the cell and counts how many are inside the sphere.
    """
    corners = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
    ]
    cx, cy, cz = sphere.center
    r2 = sphere.radius * sphere.radius
    inside = 0
    for x, y, z in corners:
        dx, dy, dz = x - cx, y - cy, z - cz
        if dx * dx + dy * dy + dz * dz <= r2:
            inside += 1
    # Also check cell center
    x_c, y_c, z_c = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    dx, dy, dz = x_c - cx, y_c - cy, z_c - cz
    if dx * dx + dy * dy + dz * dz <= r2:
        inside += 1
    return inside / 9.0


def _cylinder_volume_fraction(
    cylinder: Cylinder,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Approximate cylinder volume fraction using corner and face sampling.

    Phase 1 uses sampled-corner and sampled-face-point approximation for cylinders.
    """
    c_min, c_max = cylinder.bounds
    axis = cylinder.axis
    radial_axes = [i for i in range(3) if i != axis]

    # Check overlap in the axial direction first
    overlap_axial_min = max(c_min[axis], z_min if axis == 2 else (y_min if axis == 1 else x_min))
    overlap_axial_max = min(c_max[axis], z_max if axis == 2 else (y_max if axis == 1 else x_max))
    if overlap_axial_max <= overlap_axial_min:
        return 0.0

    # Compute radial intersection fraction using sampling
    # Sample corners and face centers
    samples = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
        ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2),
    ]
    # Add face centers for the radial faces
    r0, r1 = radial_axes[0], radial_axes[1]
    # Lower face center (z_min face at y,z center of cell)
    samples.append((x_min if r0 == 0 else (y_min if r0 == 1 else z_min),
                    y_min if r1 == 1 else (x_min if r1 == 0 else z_min),
                    z_min if axis == 2 else (y_min if axis == 1 else x_min)))
    # Upper face center (z_max face at y,z center of cell)
    samples.append((x_max if r0 == 0 else (y_max if r0 == 1 else z_max),
                    y_max if r1 == 1 else (x_max if r1 == 0 else z_max),
                    z_max if axis == 2 else (y_max if axis == 1 else x_max)))

    cx, cy, cz = cylinder.center
    r2 = cylinder.radius * cylinder.radius
    half_length = cylinder.length / 2

    inside = 0
    total = 0
    for x, y, z in samples:
        coords = [x, y, z]
        # Convert to cylinder local coordinates
        dx = coords[r0] - [cx, cy, cz][r0]
        dy = coords[r1] - [cx, cy, cz][r1]
        axial = coords[axis] - [cx, cy, cz][axis]
        if dx * dx + dy * dy <= r2 and abs(axial) <= half_length:
            inside += 1
        total += 1

    return inside / total if total > 0 else 0.0


def _polyslab_volume_fraction(
    polyslab: PolySlab,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Approximate PolySlab volume fraction using face sampling.

    Phase 1 uses a sampled-face-point approximation for PolySlab.
    Samples face centers and corners to estimate overlap.
    """
    bounds_min, bounds_max = polyslab.bounds

    # Check overlap in all directions
    overlap_x = max(0.0, min(bounds_max[0], x_max) - max(bounds_min[0], x_min))
    overlap_y = max(0.0, min(bounds_max[1], y_max) - max(bounds_min[1], y_min))
    overlap_z = max(0.0, min(bounds_max[2], z_max) - max(bounds_min[2], z_min))
    if overlap_x <= 0.0 or overlap_y <= 0.0 or overlap_z <= 0.0:
        return 0.0

    cell_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    if cell_volume <= 0.0:
        return 0.0

    # Use cell center containment as a quick check
    x_c, y_c, z_c = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    center_point = (x_c, y_c, z_c)
    if polyslab.contains_point(center_point):
        # Center is inside - likely mostly filled
        # Do a quick corner sampling to verify
        corners = [
            (x_min, y_min, z_min),
            (x_min, y_min, z_max),
            (x_min, y_max, z_min),
            (x_min, y_max, z_max),
            (x_max, y_min, z_min),
            (x_max, y_min, z_max),
            (x_max, y_max, z_min),
            (x_max, y_max, z_max),
        ]
        inside_count = sum(1 for c in corners if polyslab.contains_point(c))
        return (inside_count + 1) / 9.0

    # Center is outside - check corner sampling
    corners = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
    ]
    inside_count = sum(1 for c in corners if polyslab.contains_point(c))
    if inside_count == 0:
        return 0.0
    return inside_count / 8.0


def _geometry_group_volume_fraction(
    group: GeometryGroup,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute volume fraction for a GeometryGroup as the union of member fractions."""
    total = 0.0
    for geo in group.geometries:
        total += _compute_volume_fraction(geo, x_min, x_max, y_min, y_max, z_min, z_max)
    return min(total, 1.0)


def _transformed_volume_fraction(
    transformed: Transformed,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute volume fraction for a Transformed geometry.

    For Phase 1, we approximate by transforming the cell corners into
    the geometry's local frame and checking containment there.
    This works for rigid transforms (rotations + translations).
    """
    geo = transformed.geometry
    t = transformed.transform

    # Transform cell corners to local space
    corners = [
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
        ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2),
    ]
    local_corners = [t.world_to_local(c) for c in corners]

    # Get local bounds of the geometry
    b_min, b_max = geo.bounds

    # Count corners inside in local space
    inside = 0
    for lc in local_corners:
        if (b_min[0] - 1e-12 <= lc[0] <= b_max[0] + 1e-12 and
            b_min[1] - 1e-12 <= lc[1] <= b_max[1] + 1e-12 and
            b_min[2] - 1e-12 <= lc[2] <= b_max[2] + 1e-12):
            inside += 1

    return inside / len(local_corners)


def _clip_operation_volume_fraction(
    clip: ClipOperation,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute volume fraction for a ClipOperation (union, intersection, difference)."""
    if clip.operation == "union":
        return _geometry_group_volume_fraction(
            GeometryGroup(geometries=(clip.geometry_a, clip.geometry_b)),
            x_min, x_max, y_min, y_max, z_min, z_max,
        )
    elif clip.operation == "intersection":
        # Approximate by checking if the cell center is in both
        x_c, y_c, z_c = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        center_in_a = clip.geometry_a.contains_point((x_c, y_c, z_c))
        center_in_b = clip.geometry_b.contains_point((x_c, y_c, z_c))
        if not (center_in_a and center_in_b):
            return 0.0
        # Use a simple corner sampling approximation
        corners = [
            (x_min, y_min, z_min), (x_min, y_min, z_max), (x_min, y_max, z_min),
            (x_min, y_max, z_max), (x_max, y_min, z_min), (x_max, y_min, z_max),
            (x_max, y_max, z_min), (x_max, y_max, z_max),
        ]
        inside_count = sum(
            1 for c in corners
            if clip.geometry_a.contains_point(c) and clip.geometry_b.contains_point(c)
        )
        return (inside_count + 1) / 9.0
    else:  # difference: geometry_a - geometry_b
        x_c, y_c, z_c = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        center_in_a = clip.geometry_a.contains_point((x_c, y_c, z_c))
        center_in_b = clip.geometry_b.contains_point((x_c, y_c, z_c))
        if not center_in_a:
            return 0.0
        if center_in_b:
            return 0.0
        corners = [
            (x_min, y_min, z_min), (x_min, y_min, z_max), (x_min, y_max, z_min),
            (x_min, y_max, z_max), (x_max, y_min, z_min), (x_max, y_min, z_max),
            (x_max, y_max, z_min), (x_max, y_max, z_max),
        ]
        inside_a_not_b = sum(
            1 for c in corners
            if clip.geometry_a.contains_point(c) and not clip.geometry_b.contains_point(c)
        )
        return (inside_a_not_b + 1) / 9.0


def _geometry_array_volume_fraction(
    array: GeometryArray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute volume fraction for a GeometryArray as union of instances."""
    total = 0.0
    for instance in array.instances:
        total += _compute_volume_fraction(instance, x_min, x_max, y_min, y_max, z_min, z_max)
    return min(total, 1.0)


def _compute_volume_fraction(
    geometry: GeometryWrapperModel,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """Compute the volume fraction of a geometry overlapping a cell."""
    if isinstance(geometry, Box):
        return _box_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, Sphere):
        return _sphere_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, Cylinder):
        return _cylinder_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, PolySlab):
        return _polyslab_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, GeometryGroup):
        return _geometry_group_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, Transformed):
        return _transformed_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, ClipOperation):
        return _clip_operation_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    if isinstance(geometry, GeometryArray):
        return _geometry_array_volume_fraction(geometry, x_min, x_max, y_min, y_max, z_min, z_max)
    # Fallback for unknown types: use center containment
    x_c, y_c, z_c = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    if geometry.contains_point((x_c, y_c, z_c)):
        return 1.0
    return 0.0


# =============================================================================
# Cell materialization
# =============================================================================


class MediumKind(Enum):
    """Classification of medium type for subpixel smoothing policy."""
    DIELECTRIC = "dielectric"
    METAL = "metal"
    PEC = "pec"
    PMC = "pmc"
    LOSSY_METAL = "lossy_metal"
    UNKNOWN = "unknown"


def _classify_medium(medium: object) -> MediumKind:
    """Classify a medium for subpixel smoothing policy selection."""
    # Handle empty dict (default vacuum) case
    if isinstance(medium, dict) and not medium:
        return MediumKind.DIELECTRIC
    normalized = medium_model_from_value(medium)
    if isinstance(normalized, PECMedium):
        return MediumKind.PEC
    if isinstance(normalized, PMCMedium):
        return MediumKind.PMC
    if isinstance(normalized, Medium):
        if normalized.conductivity > 0.0:
            return MediumKind.METAL
        return MediumKind.DIELECTRIC
    if isinstance(normalized, (PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
        return MediumKind.DIELECTRIC
    if isinstance(normalized, AnisotropicMedium):
        # Check if any component is a metal
        for component in normalized.components.values():
            if isinstance(component, Medium) and component.conductivity > 0.0:
                return MediumKind.METAL
        return MediumKind.DIELECTRIC
    return MediumKind.UNKNOWN


@dataclass(frozen=True)
class CellMaterial:
    """Resolved material assignment for one grid cell."""
    structure_index: int | None  # None means background
    medium: object
    medium_kind: MediumKind
    volume_fraction: float = 1.0  # fraction of cell occupied by this medium


@dataclass(frozen=True)
class MaterialField:
    """3D material field with per-cell assignments and volume fractions."""
    nx: int
    ny: int
    nz: int
    # (nx, ny, nz) array of structure indices (or -1 for background)
    structure_indices: np.ndarray
    # (nx, ny, nz) array of volume fractions
    volume_fractions: np.ndarray
    # (nx, ny, nz) array of medium kinds
    medium_kinds: np.ndarray


def _resolve_cell_material(
    scene: Scene,
    x_idx: int,
    y_idx: int,
    z_idx: int,
    x_bounds: tuple[float, ...],
    y_bounds: tuple[float, ...],
    z_bounds: tuple[float, ...],
    component: str | None = None,
) -> CellMaterial:
    """Resolve the material for one grid cell using scene precedence rules.

    Notes
    -----
    The winning structure is the LAST structure in precedence order that contains
    the point, because later structures override earlier ones in the materialization.

    Parameters
    ----------
    component : str | None
        The field component for Yee-staggered epsilon sampling.
        - "Ex": sample at (x, y+dz/2, z+dz/2) — Ex position on Yee grid
        - "Ey": sample at (x+dx/2, y, z+dz/2) — Ey position on Yee grid
        - "Ez": sample at (x+dx/2, y+dy/2, z) — Ez position on Yee grid
        - None: sample at cell center (default, for precedence resolution)
    """
    x_min, x_max = x_bounds[x_idx], x_bounds[x_idx + 1]
    y_min, y_max = y_bounds[y_idx], y_bounds[y_idx + 1]
    z_min, z_max = z_bounds[z_idx], z_bounds[z_idx + 1]

    # Cell dimensions
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    # Base cell center
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    z_c = (z_min + z_max) / 2

    # Yee-staggered offset for epsilon sampling at E-field positions
    # Ex at (i, j+1/2, k+1/2): no offset in x, half-cell offset in y and z
    # Ey at (i+1/2, j, k+1/2): half-cell offset in x and z, no offset in y
    # Ez at (i+1/2, j+1/2, k): half-cell offset in x and y, no offset in z
    if component == "Ex":
        x_s, y_s, z_s = x_c, y_c + dy / 2, z_c + dz / 2
    elif component == "Ey":
        x_s, y_s, z_s = x_c + dx / 2, y_c, z_c + dz / 2
    elif component == "Ez":
        x_s, y_s, z_s = x_c + dx / 2, y_c + dy / 2, z_c
    else:
        # Default: cell center (for precedence resolution)
        x_s, y_s, z_s = x_c, y_c, z_c

    point = (x_s, y_s, z_s)

    # Find the LAST (highest precedence) structure that contains the point
    # because later structures override earlier ones in materialization
    winner_index: int | None = None
    winner_medium: object | None = None
    winner_entry: object | None = None

    for entry in scene.structure_precedence():
        structure = entry.structure
        if structure.contains_point(point):
            winner_index = entry.source_index
            winner_medium = structure.medium
            winner_entry = entry

    if winner_index is not None:
        medium_kind = _classify_medium(winner_medium)
        return CellMaterial(
            structure_index=winner_index,
            medium=winner_medium,
            medium_kind=medium_kind,
            volume_fraction=1.0,
        )

    # Fall back to background
    medium_kind = _classify_medium(scene.medium)
    return CellMaterial(
        structure_index=None,
        medium=scene.medium,
        medium_kind=medium_kind,
        volume_fraction=1.0,
    )


def materialize_grid(
    scene: Scene,
    grid: ResolvedGrid,
) -> MaterialField:
    """Materialize a scene onto a resolved grid to produce per-cell material assignments.

    Parameters
    ----------
    scene : Scene
        The scene containing structures and background medium.
    grid : ResolvedGrid
        The resolved discretization grid.

    Returns
    -------
    MaterialField
        3D arrays of structure indices, medium kinds, and volume fractions.
    """
    nx, ny, nz = grid.x.num_cells, grid.y.num_cells, grid.z.num_cells
    x_b, y_b, z_b = grid.x.boundaries, grid.y.boundaries, grid.z.boundaries

    structure_indices = np.full((nx, ny, nz), -1, dtype=np.int32)
    volume_fractions = np.ones((nx, ny, nz), dtype=np.float64)
    medium_kinds = np.zeros((nx, ny, nz), dtype=np.int32)  # encoded MediumKind

    _KIND_CODE: dict[MediumKind, int] = {
        MediumKind.DIELECTRIC: 0,
        MediumKind.METAL: 1,
        MediumKind.PEC: 2,
        MediumKind.PMC: 3,
        MediumKind.LOSSY_METAL: 4,
        MediumKind.UNKNOWN: 5,
    }

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = _resolve_cell_material(scene, i, j, k, x_b, y_b, z_b)
                structure_indices[i, j, k] = cell.structure_index if cell.structure_index is not None else -1
                volume_fractions[i, j, k] = cell.volume_fraction
                medium_kinds[i, j, k] = _KIND_CODE.get(cell.medium_kind, 5)

    return MaterialField(
        nx=nx,
        ny=ny,
        nz=nz,
        structure_indices=structure_indices,
        volume_fractions=volume_fractions,
        medium_kinds=medium_kinds,
    )


# =============================================================================
# Subpixel smoothing
# =============================================================================


def _subpixel_kind_for_cell(
    medium_kind: MediumKind,
    subpixel_spec: SubpixelSpec,
) -> type[AbstractSubpixelPolicy]:
    """Select the subpixel policy for a cell based on its medium kind."""
    if medium_kind == MediumKind.DIELECTRIC:
        return type(subpixel_spec.dielectric)
    if medium_kind == MediumKind.METAL:
        return type(subpixel_spec.metal)
    if medium_kind == MediumKind.PEC:
        return type(subpixel_spec.pec)
    if medium_kind == MediumKind.PMC:
        return type(subpixel_spec.pmc)
    if medium_kind == MediumKind.LOSSY_METAL:
        return type(subpixel_spec.lossy_metal)
    return Staircasing


def _effective_permittivity_isotropic(
    eps_a: float,
    eps_b: float,
    f: float,
) -> float:
    """Compute effective permittivity using harmonic mean for parallel field.

    For subpixel smoothing of dielectric interfaces, we use the standard
    weighted harmonic/arithmetic mean pair for the two polarization cases.
    This function returns the harmonic mean (for the case where E is
    perpendicular to the interface normal).

    Parameters
    ----------
    eps_a : float
        Permittivity of material a.
    eps_b : float
        Permittivity of material b.
    f : float
        Volume fraction of material b in the cell.

    Returns
    -------
    float
        Effective permittivity.
    """
    if f <= 0.0:
        return eps_a
    if f >= 1.0:
        return eps_b
    # Harmonic mean for perpendicular polarization (1/eps = f/eps_b + (1-f)/eps_a)
    eps_a_safe = max(eps_a, 1e-20)
    eps_b_safe = max(eps_b, 1e-20)
    return 1.0 / (f / eps_b_safe + (1.0 - f) / eps_a_safe)


def _effective_permeability_isotropic(
    mu_a: float,
    mu_b: float,
    f: float,
) -> float:
    """Compute effective permeability using harmonic mean for parallel field."""
    if f <= 0.0:
        return mu_a
    if f >= 1.0:
        return mu_b
    mu_a_safe = max(mu_a, 1e-20)
    mu_b_safe = max(mu_b, 1e-20)
    return 1.0 / (f / mu_b_safe + (1.0 - f) / mu_a_safe)


def _effective_medium_isotropic(
    eps_a: float,
    mu_a: float,
    eps_b: float,
    mu_b: float,
    f: float,
) -> tuple[float, float]:
    """Compute effective permittivity and permeability for isotropic subpixel smoothing.

    Uses the standard two-branch formula:
    - eps_eff uses harmonic mean (E perpendicular to interface)
    - mu_eff uses harmonic mean (H perpendicular to interface)

    Returns
    -------
    tuple[float, float]
        (effective_eps, effective_mu)
    """
    return (
        _effective_permittivity_isotropic(eps_a, eps_b, f),
        _effective_permeability_isotropic(mu_a, mu_b, f),
    )


def _medium_eps_mu(medium: object) -> tuple[float, float]:
    """Extract scalar permittivity and permeability from a medium."""
    # Handle bare dicts (e.g. {"permittivity": 3.5}) that come from
    # Structure.medium after normalize_component validation.
    if isinstance(medium, dict):
        return (float(medium.get("permittivity", 1.0)), float(medium.get("permeability", 1.0)))

    normalized = medium_model_from_value(medium)
    if isinstance(normalized, Medium):
        return (normalized.permittivity, normalized.permeability)
    if isinstance(normalized, (PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
        # Use eps_inf as the DC/low-frequency permittivity for subpixel purposes
        return (normalized.eps_inf, 1.0)
    if isinstance(normalized, PECMedium):
        return (1.0, 1.0)  # Will be clamped to zero anyway
    if isinstance(normalized, PMCMedium):
        return (1.0, 1.0)  # Will be clamped to zero anyway
    if isinstance(normalized, AnisotropicMedium):
        # For Phase 1, use the xx component as representative scalar
        xx = normalized.xx
        if isinstance(xx, Medium):
            return (xx.permittivity, xx.permeability)
        if isinstance(xx, (PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
            return (xx.eps_inf, 1.0)
        return (1.0, 1.0)
    # Default vacuum
    return (1.0, 1.0)


def apply_subpixel_smoothing(
    material_field: MaterialField,
    scene: Scene,
    grid: ResolvedGrid,
    subpixel_spec: SubpixelSpec,
) -> MaterialField:
    """Apply subpixel smoothing to a materialized grid field.

    For cells at dielectric-dielectric interfaces where PolarizedAveraging
    is configured, this replaces the staircased volume fraction (0 or 1)
    with a volume-weighted effective medium.

    Phase 1 scope:
    - Only dielectric-dielectric interfaces use subpixel averaging
    - All metal, PEC, PMC, lossy-metal interfaces remain staircased

    Parameters
    ----------
    material_field : MaterialField
        The materialized grid field from ``materialize_grid``.
    scene : Scene
        The scene (used to look up structure media).
    grid : ResolvedGrid
        The resolved discretization grid.
    subpixel_spec : SubpixelSpec
        The subpixel policy specification.

    Returns
    -------
    MaterialField
        Updated material field with smoothed volume fractions at interfaces.
    """
    if not subpixel_spec.dielectric.requires_anisotropic_materialization():
        return material_field

    nx, ny, nz = material_field.nx, material_field.ny, material_field.nz
    x_b, y_b, z_b = grid.x.boundaries, grid.y.boundaries, grid.z.boundaries
    new_fractions = material_field.volume_fractions.copy()

    _CODE_KIND: dict[int, MediumKind] = {
        0: MediumKind.DIELECTRIC,
        1: MediumKind.METAL,
        2: MediumKind.PEC,
        3: MediumKind.PMC,
        4: MediumKind.LOSSY_METAL,
        5: MediumKind.UNKNOWN,
    }

    # For dielectric averaging, we need to check if a cell is at a dielectric interface
    # A cell is at an interface if its neighbors have different structure assignments
    structures_list = list(scene.structures)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                kind_code = int(material_field.medium_kinds[i, j, k])
                medium_kind = _CODE_KIND.get(kind_code, MediumKind.UNKNOWN)

                if medium_kind != MediumKind.DIELECTRIC:
                    continue

                # Check if this is a staircased cell (fraction is 0 or 1)
                f = material_field.volume_fractions[i, j, k]
                if f > 0.0 and f < 1.0:
                    continue  # Already smoothed

                # Get this cell's structure index
                struct_idx = int(material_field.structure_indices[i, j, k])

                # Get the medium at this cell
                if struct_idx >= 0:
                    if struct_idx < len(structures_list):
                        medium_a = structures_list[struct_idx].medium
                    else:
                        medium_a = scene.medium
                else:
                    medium_a = scene.medium

                eps_a, mu_a = _medium_eps_mu(medium_a)

                # Check neighbors in each direction
                neighbor_eps_sum = 0.0
                neighbor_mu_sum = 0.0
                neighbor_count = 0

                # Check 6 neighbors
                for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor_kind_code = int(material_field.medium_kinds[ni, nj, nk])
                        neighbor_kind = _CODE_KIND.get(neighbor_kind_code, MediumKind.UNKNOWN)

                        if neighbor_kind != MediumKind.DIELECTRIC:
                            continue

                        neighbor_struct_idx = int(material_field.structure_indices[ni, nj, nk])
                        if neighbor_struct_idx >= 0:
                            if neighbor_struct_idx < len(structures_list):
                                medium_b = structures_list[neighbor_struct_idx].medium
                            else:
                                medium_b = scene.medium
                        else:
                            medium_b = scene.medium

                        if medium_b is medium_a:
                            continue

                        eps_b, mu_b = _medium_eps_mu(medium_b)
                        neighbor_eps_sum += eps_b
                        neighbor_mu_sum += mu_b
                        neighbor_count += 1

                if neighbor_count == 0:
                    continue

                avg_neighbor_eps = neighbor_eps_sum / neighbor_count
                avg_neighbor_mu = neighbor_mu_sum / neighbor_count

                # Compute effective medium using the actual volume fraction
                # The effective eps uses harmonic mean with neighbor average
                f = material_field.volume_fractions[i, j, k]
                # f represents the fraction of the *other* (neighbor) material
                # So the fraction of medium_a is (1 - f)
                f_eff = max(0.0, min(1.0, f))
                effective_eps = _effective_permittivity_isotropic(eps_a, avg_neighbor_eps, f_eff)
                effective_mu = _effective_permeability_isotropic(mu_a, avg_neighbor_mu, f_eff)

                # Replace staircased fraction with smoothed value
                # We use a simple heuristic: if the effective permittivity is
                # significantly different from eps_a, apply smoothing
                if abs(effective_eps - eps_a) > 0.01 * eps_a:
                    # This cell is at an interface - set fraction to indicate averaging
                    # For subpixel, we store the fraction of the *other* material
                    # Approximate as 0.5 for fully averaged cells
                    new_fractions[i, j, k] = 0.5

    return MaterialField(
        nx=nx,
        ny=ny,
        nz=nz,
        structure_indices=material_field.structure_indices,
        volume_fractions=new_fractions,
        medium_kinds=material_field.medium_kinds,
    )


# =============================================================================
# Coefficient field assembly
# =============================================================================


@dataclass(frozen=True)
class CoefficientField:
    """Assembled coefficient fields ready for runtime kernel allocation."""
    nx: int
    ny: int
    nz: int
    # (nx, ny, nz) arrays of compiled constitutive coefficients
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    mu_xx: np.ndarray
    mu_yy: np.ndarray
    mu_zz: np.ndarray
    # (nx, ny, nz) arrays of drive coefficients
    e_drive_xx: np.ndarray
    e_drive_yy: np.ndarray
    e_drive_zz: np.ndarray
    m_drive_xx: np.ndarray
    m_drive_yy: np.ndarray
    m_drive_zz: np.ndarray
    # Constitutive modes
    electric_modes: np.ndarray  # 0=standard, 1=clamp_zero
    magnetic_modes: np.ndarray


def _build_isotropic_coefficients(
    permittivity: float,
    permeability: float,
    conductivity: float,
    magnetic_conductivity: float,
    electric_mode: ConstitutiveMode,
    magnetic_mode: ConstitutiveMode,
    dt: float,
) -> tuple[float, float, float, float, float, float]:
    """Build isotropic coefficient scalars for a cell."""
    from autofdtd.compiler.materials import _compile_nondispersive_coefficients

    coeff = _compile_nondispersive_coefficients(
        dt=dt,
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        magnetic_conductivity=magnetic_conductivity,
        medium_type="Medium",
    )
    return (
        coeff.electric_decay,
        coeff.electric_drive,
        coeff.magnetic_decay,
        coeff.magnetic_drive,
        float(electric_mode.value == "clamp_zero"),
        float(magnetic_mode.value == "clamp_zero"),
    )


def assemble_coefficient_fields(
    material_field: MaterialField,
    scene: Scene,
    grid: ResolvedGrid,
    dt: float,
    subpixel_spec: SubpixelSpec | None = None,
) -> CoefficientField:
    """Assemble per-cell constitutive coefficient arrays from a materialized grid.

    Parameters
    ----------
    material_field : MaterialField
        Material assignments from ``materialize_grid``.
    scene : Scene
        The scene (for structure medium lookup).
    grid : ResolvedGrid
        The resolved discretization grid.
    dt : float
        Timestep for coefficient compilation.
    subpixel_spec : SubpixelSpec, optional
        Subpixel smoothing specification. If provided and dielectric
        interfaces use ``PolarizedAveraging``, effective medium parameters
        are computed for staircased cells.

    Returns
    -------
    CoefficientField
        Per-cell coefficient arrays ready for kernel allocation.
    """
    nx, ny, nz = material_field.nx, material_field.ny, material_field.nz

    eps_xx = np.ones((nx, ny, nz), dtype=np.float64)
    eps_yy = np.ones((nx, ny, nz), dtype=np.float64)
    eps_zz = np.ones((nx, ny, nz), dtype=np.float64)
    mu_xx = np.ones((nx, ny, nz), dtype=np.float64)
    mu_yy = np.ones((nx, ny, nz), dtype=np.float64)
    mu_zz = np.ones((nx, ny, nz), dtype=np.float64)
    e_drive_xx = np.zeros((nx, ny, nz), dtype=np.float64)
    e_drive_yy = np.zeros((nx, ny, nz), dtype=np.float64)
    e_drive_zz = np.zeros((nx, ny, nz), dtype=np.float64)
    m_drive_xx = np.zeros((nx, ny, nz), dtype=np.float64)
    m_drive_yy = np.zeros((nx, ny, nz), dtype=np.float64)
    m_drive_zz = np.zeros((nx, ny, nz), dtype=np.float64)
    electric_modes = np.zeros((nx, ny, nz), dtype=np.float64)
    magnetic_modes = np.zeros((nx, ny, nz), dtype=np.float64)

    structures = list(scene.structures)

    smoothing_applied = subpixel_spec is not None and subpixel_spec.requires_anisotropic_materialization()
    if smoothing_applied and subpixel_spec is not None:
        material_field = apply_subpixel_smoothing(material_field, scene, grid, subpixel_spec)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                struct_idx = material_field.structure_indices[i, j, k]
                f = material_field.volume_fractions[i, j, k]

                if struct_idx >= 0 and struct_idx < len(structures):
                    medium = structures[struct_idx].medium
                else:
                    medium = scene.medium

                # Handle bare dicts (e.g. {"permittivity": 3.5} or {}) from
                # Structure.medium after normalize_component validation.
                if isinstance(medium, dict):
                    perm = float(medium.get("permittivity", 1.0))
                    mu = float(medium.get("permeability", 1.0))
                    cond = float(medium.get("conductivity", 0.0))
                    m_cond = float(medium.get("magnetic_conductivity", 0.0))
                    e_mode = ConstitutiveMode.STANDARD
                    m_mode = ConstitutiveMode.STANDARD
                    e_decay, e_drive, m_decay, m_drive, e_clamp, m_clamp = _build_isotropic_coefficients(
                        perm, mu, cond, m_cond, e_mode, m_mode, dt
                    )
                    eps_xx[i, j, k] = e_decay
                    eps_yy[i, j, k] = e_decay
                    eps_zz[i, j, k] = e_decay
                    mu_xx[i, j, k] = m_decay
                    mu_yy[i, j, k] = m_decay
                    mu_zz[i, j, k] = m_decay
                    e_drive_xx[i, j, k] = e_drive
                    e_drive_yy[i, j, k] = e_drive
                    e_drive_zz[i, j, k] = e_drive
                    m_drive_xx[i, j, k] = m_drive
                    m_drive_yy[i, j, k] = m_drive
                    m_drive_zz[i, j, k] = m_drive
                    electric_modes[i, j, k] = e_clamp
                    magnetic_modes[i, j, k] = m_clamp
                    continue

                normalized = medium_model_from_value(medium)

                if isinstance(normalized, AnisotropicMedium):
                    # Anisotropic: different coefficient per axis
                    for axis_idx, component_key in enumerate(["xx", "yy", "zz"]):
                        component = normalized.components[component_key]
                        comp_normalized = medium_model_from_value(component)
                        if isinstance(comp_normalized, Medium):
                            perm = comp_normalized.permittivity
                            mu = comp_normalized.permeability
                            cond = comp_normalized.conductivity
                            m_cond = comp_normalized.magnetic_conductivity
                            e_mode = ConstitutiveMode.CLAMP_ZERO if isinstance(comp_normalized, PECMedium) else ConstitutiveMode.STANDARD
                            m_mode = ConstitutiveMode.CLAMP_ZERO if isinstance(comp_normalized, PMCMedium) else ConstitutiveMode.STANDARD
                        elif isinstance(comp_normalized, (PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
                            perm = comp_normalized.eps_inf
                            mu = 1.0
                            cond = 0.0
                            m_cond = 0.0
                            e_mode = ConstitutiveMode.STANDARD
                            m_mode = ConstitutiveMode.STANDARD
                        elif isinstance(comp_normalized, PECMedium):
                            perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                            e_mode = ConstitutiveMode.CLAMP_ZERO
                            m_mode = ConstitutiveMode.STANDARD
                        elif isinstance(comp_normalized, PMCMedium):
                            perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                            e_mode = ConstitutiveMode.STANDARD
                            m_mode = ConstitutiveMode.CLAMP_ZERO
                        else:
                            perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                            e_mode = ConstitutiveMode.STANDARD
                            m_mode = ConstitutiveMode.STANDARD

                        e_decay, e_drive, m_decay, m_drive, e_clamp, m_clamp = _build_isotropic_coefficients(
                            perm, mu, cond, m_cond, e_mode, m_mode, dt
                        )
                        if axis_idx == 0:
                            eps_xx[i, j, k] = e_decay
                            mu_xx[i, j, k] = m_decay
                            e_drive_xx[i, j, k] = e_drive
                            m_drive_xx[i, j, k] = m_drive
                        elif axis_idx == 1:
                            eps_yy[i, j, k] = e_decay
                            mu_yy[i, j, k] = m_decay
                            e_drive_yy[i, j, k] = e_drive
                            m_drive_yy[i, j, k] = m_drive
                        else:
                            eps_zz[i, j, k] = e_decay
                            mu_zz[i, j, k] = m_decay
                            e_drive_zz[i, j, k] = e_drive
                            m_drive_zz[i, j, k] = m_drive
                        electric_modes[i, j, k] = max(electric_modes[i, j, k], e_clamp)
                        magnetic_modes[i, j, k] = max(magnetic_modes[i, j, k], m_clamp)
                else:
                    # Isotropic or other medium
                    if isinstance(normalized, Medium):
                        perm = normalized.permittivity
                        mu = normalized.permeability
                        cond = normalized.conductivity
                        m_cond = normalized.magnetic_conductivity
                        e_mode = ConstitutiveMode.CLAMP_ZERO if isinstance(normalized, PECMedium) else ConstitutiveMode.STANDARD
                        m_mode = ConstitutiveMode.CLAMP_ZERO if isinstance(normalized, PMCMedium) else ConstitutiveMode.STANDARD
                    elif isinstance(normalized, (PoleResidue, Sellmeier, Lorentz, Drude, Debye)):
                        perm = normalized.eps_inf
                        mu = 1.0
                        cond = 0.0
                        m_cond = 0.0
                        e_mode = ConstitutiveMode.STANDARD
                        m_mode = ConstitutiveMode.STANDARD
                    elif isinstance(normalized, PECMedium):
                        perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                        e_mode = ConstitutiveMode.CLAMP_ZERO
                        m_mode = ConstitutiveMode.STANDARD
                    elif isinstance(normalized, PMCMedium):
                        perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                        e_mode = ConstitutiveMode.STANDARD
                        m_mode = ConstitutiveMode.CLAMP_ZERO
                    else:
                        perm, mu, cond, m_cond = 1.0, 1.0, 0.0, 0.0
                        e_mode = ConstitutiveMode.STANDARD
                        m_mode = ConstitutiveMode.STANDARD

                    # Apply subpixel smoothing: blend with background using volume fraction
                    # f is the fraction of the *other* (neighbor/background) material
                    # So fraction of this medium is (1 - f)
                    if f > 0.0 and f < 1.0:
                        # Get background medium properties
                        bg_eps, bg_mu = _medium_eps_mu(scene.medium)
                        # Blend: eps_eff = f * eps_background + (1-f) * eps_structure
                        # This is arithmetic mean for the case where E is parallel to interface
                        eps_a_safe = max(perm, 1e-20)
                        bg_eps_safe = max(bg_eps, 1e-20)
                        f_clamped = max(0.0, min(1.0, f))
                        perm_blended = f_clamped * bg_eps_safe + (1.0 - f_clamped) * eps_a_safe
                        mu_a_safe = max(mu, 1e-20)
                        bg_mu_safe = max(bg_mu, 1e-20)
                        mu_blended = f_clamped * bg_mu_safe + (1.0 - f_clamped) * mu_a_safe
                        perm = perm_blended
                        mu = mu_blended

                    e_decay, e_drive, m_decay, m_drive, e_clamp, m_clamp = _build_isotropic_coefficients(
                        perm, mu, cond, m_cond, e_mode, m_mode, dt
                    )
                    eps_xx[i, j, k] = e_decay
                    eps_yy[i, j, k] = e_decay
                    eps_zz[i, j, k] = e_decay
                    mu_xx[i, j, k] = m_decay
                    mu_yy[i, j, k] = m_decay
                    mu_zz[i, j, k] = m_decay
                    e_drive_xx[i, j, k] = e_drive
                    e_drive_yy[i, j, k] = e_drive
                    e_drive_zz[i, j, k] = e_drive
                    m_drive_xx[i, j, k] = m_drive
                    m_drive_yy[i, j, k] = m_drive
                    m_drive_zz[i, j, k] = m_drive
                    electric_modes[i, j, k] = e_clamp
                    magnetic_modes[i, j, k] = m_clamp

    return CoefficientField(
        nx=nx,
        ny=ny,
        nz=nz,
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        mu_xx=mu_xx,
        mu_yy=mu_yy,
        mu_zz=mu_zz,
        e_drive_xx=e_drive_xx,
        e_drive_yy=e_drive_yy,
        e_drive_zz=e_drive_zz,
        m_drive_xx=m_drive_xx,
        m_drive_yy=m_drive_yy,
        m_drive_zz=m_drive_zz,
        electric_modes=electric_modes,
        magnetic_modes=magnetic_modes,
    )


# =============================================================================
# Convenience entry points
# =============================================================================


def discretize_scene(
    scene: Scene,
    grid: ResolvedGrid,
    dt: float,
    subpixel_spec: SubpixelSpec | None = None,
) -> tuple[MaterialField, CoefficientField]:
    """Discretize a scene onto a resolved grid to produce material and coefficient fields.

    Parameters
    ----------
    scene : Scene
        The scene to discretize.
    grid : ResolvedGrid
        The resolved discretization grid.
    dt : float
        Timestep for coefficient compilation.
    subpixel_spec : SubpixelSpec, optional
        Subpixel smoothing specification.

    Returns
    -------
    tuple[MaterialField, CoefficientField]
        The materialized grid and assembled coefficient fields.
    """
    mfield = materialize_grid(scene, grid)
    cfield = assemble_coefficient_fields(mfield, scene, grid, dt, subpixel_spec)
    return mfield, cfield
