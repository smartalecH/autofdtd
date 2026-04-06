"""Source compilation helpers for Phase 1 current injection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from autofdtd.grid import ResolvedGrid, ResolvedGridAxis
from autofdtd.sources import UniformCurrentSource


@dataclass(frozen=True)
class CompiledAxisPlacement:
    """Discrete support metadata for one source axis."""

    indices: tuple[int, ...]
    weights: tuple[float, ...]
    cell_sizes: tuple[float, ...]
    centers: tuple[float, ...]


@dataclass(frozen=True)
class CompiledUniformCurrentSource:
    """Discrete current-source placement and amplitude metadata."""

    source: UniformCurrentSource
    field_kind: Literal["electric", "magnetic"]
    component_axis: int
    placement_kind: Literal["point", "line", "sheet", "volume"]
    placements: tuple[tuple[int, int, int], ...]
    placement_weights: tuple[float, ...]
    axis_placements: tuple[CompiledAxisPlacement, CompiledAxisPlacement, CompiledAxisPlacement]
    support_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    amplitude_scale: float

    @property
    def support_point_count(self) -> int:
        return len(self.placements)

    def amplitude_at_time(self, time: float) -> complex:
        return complex(self.source.source_time.amp_time(time)[0]) * self.amplitude_scale


def _axis_centers(axis: ResolvedGridAxis) -> np.ndarray:
    boundaries = np.asarray(axis.boundaries, dtype=float)
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def _axis_sizes(axis: ResolvedGridAxis) -> np.ndarray:
    return np.asarray(axis.cell_sizes, dtype=float)


def _bracket_indices(values: np.ndarray, position: float) -> tuple[int, int]:
    if values.size == 1:
        return (0, 0)
    if position <= values[0]:
        return (0, 0)
    if position >= values[-1]:
        last = values.size - 1
        return (last, last)
    upper = int(np.searchsorted(values, position, side="right"))
    lower = max(0, upper - 1)
    return (lower, min(values.size - 1, upper))


def _compile_axis_placement(
    axis: ResolvedGridAxis,
    *,
    center: float,
    size: float,
    interpolate: bool,
    confine_to_bounds: bool,
) -> CompiledAxisPlacement:
    if axis.num_cells <= 0:
        raise ValueError("UniformCurrentSource requires a non-collapsed grid axis")

    centers = _axis_centers(axis)
    cell_sizes = _axis_sizes(axis)
    boundaries = np.asarray(axis.boundaries, dtype=float)
    tolerance = max(1e-12, axis.max_step * 1e-9)

    if math.isclose(size, 0.0, abs_tol=tolerance):
        exact = np.nonzero(np.isclose(centers, center, rtol=0.0, atol=tolerance))[0]
        if exact.size:
            index = int(exact[0])
            return CompiledAxisPlacement(
                indices=(index,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[index]),),
                centers=(float(centers[index]),),
            )
        if not interpolate:
            index = int(np.argmin(np.abs(centers - center)))
            return CompiledAxisPlacement(
                indices=(index,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[index]),),
                centers=(float(centers[index]),),
            )

        lower, upper = _bracket_indices(centers, center)
        if lower == upper:
            return CompiledAxisPlacement(
                indices=(lower,),
                weights=(1.0,),
                cell_sizes=(float(cell_sizes[lower]),),
                centers=(float(centers[lower]),),
            )
        left_center = centers[lower]
        right_center = centers[upper]
        fraction = float((center - left_center) / (right_center - left_center))
        return CompiledAxisPlacement(
            indices=(lower, upper),
            weights=(1.0 - fraction, fraction),
            cell_sizes=(float(cell_sizes[lower]), float(cell_sizes[upper])),
            centers=(float(left_center), float(right_center)),
        )

    lower = center - size / 2.0
    upper = center + size / 2.0
    if confine_to_bounds:
        overlaps = np.minimum(boundaries[1:], upper) - np.maximum(boundaries[:-1], lower)
        mask = overlaps > tolerance
        if not np.any(mask):
            raise ValueError("UniformCurrentSource support does not overlap the resolved grid")
        weights = overlaps[mask] / cell_sizes[mask]
        indices = np.nonzero(mask)[0]
    else:
        mask = (centers >= lower - tolerance) & (centers <= upper + tolerance)
        if not np.any(mask):
            raise ValueError("UniformCurrentSource support does not include any resolved grid cells")
        weights = np.ones(int(np.count_nonzero(mask)), dtype=float)
        indices = np.nonzero(mask)[0]
    return CompiledAxisPlacement(
        indices=tuple(int(index) for index in indices),
        weights=tuple(float(weight) for weight in weights),
        cell_sizes=tuple(float(cell_sizes[index]) for index in indices),
        centers=tuple(float(centers[index]) for index in indices),
    )


def _transverse_extent(source: UniformCurrentSource, grid: ResolvedGrid) -> float:
    placements = []
    axes = (grid.x, grid.y, grid.z)
    for axis in range(3):
        if axis == source.component_axis:
            continue
        extent = source.size[axis]
        if extent > 0.0:
            placements.append(extent)
            continue
        centers = _axis_centers(axes[axis])
        cell_sizes = _axis_sizes(axes[axis])
        index = int(np.argmin(np.abs(centers - source.center[axis])))
        placements.append(float(cell_sizes[index]))
    measure = 1.0
    for extent in placements:
        measure *= extent
    return measure


def compile_uniform_current_source(
    source: UniformCurrentSource,
    *,
    grid: ResolvedGrid,
) -> CompiledUniformCurrentSource:
    """Compile a uniform current source onto the resolved primal-cell grid."""

    axis_placements = tuple(
        _compile_axis_placement(
            axis_grid,
            center=source.center[axis],
            size=source.size[axis],
            interpolate=source.interpolate,
            confine_to_bounds=source.confine_to_bounds,
        )
        for axis, axis_grid in enumerate((grid.x, grid.y, grid.z))
    )

    placements: list[tuple[int, int, int]] = []
    placement_weights: list[float] = []
    for x_index, x_weight in zip(
        axis_placements[0].indices,
        axis_placements[0].weights,
        strict=True,
    ):
        for y_index, y_weight in zip(
            axis_placements[1].indices,
            axis_placements[1].weights,
            strict=True,
        ):
            for z_index, z_weight in zip(
                axis_placements[2].indices,
                axis_placements[2].weights,
                strict=True,
            ):
                placements.append((x_index, y_index, z_index))
                placement_weights.append(float(x_weight * y_weight * z_weight))

    amplitude_scale = 1.0
    if source.current_amplitude_definition == "total":
        transverse_extent = _transverse_extent(source, grid)
        if transverse_extent <= 0.0:
            raise ValueError("UniformCurrentSource total-current normalization needs positive extent")
        amplitude_scale = 1.0 / transverse_extent

    return CompiledUniformCurrentSource(
        source=source,
        field_kind=source.field_kind,
        component_axis=source.component_axis,
        placement_kind=source.placement_kind,
        placements=tuple(placements),
        placement_weights=tuple(placement_weights),
        axis_placements=axis_placements,
        support_bounds=source.support_bounds,
        amplitude_scale=amplitude_scale,
    )
