"""Phase 1 grid-specification models and domain-resolution helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, field_validator

from autofdtd.core.models import TaggedModel

Vec3 = tuple[float, float, float]
AxisName = Literal["x", "y", "z"]
_DEFAULT_UNIFORM_DL = 1.0
_DEFAULT_AUTOGRID_STEPS_PER_WVL = 10.0
_DEFAULT_AUTOGRID_STEPS_PER_SIM_SIZE = 10.0
_DEFAULT_AUTOGRID_MAX_SCALE = 1.4
_DEFAULT_REFINEMENT_FACTOR = 2.0


def _finite_float(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _positive_float(value: float, *, field_name: str) -> float:
    numeric = _finite_float(value, field_name=field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    if numeric < 1e-7:
        raise ValueError(f"{field_name} is too small for the Phase 1 unit conventions")
    return numeric


def _positive_int(value: int, *, field_name: str) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return numeric


def _normalize_axis_bounds(*, center: float, size: float) -> tuple[float, float]:
    normalized_center = _finite_float(center, field_name="axis center")
    normalized_size = _finite_float(size, field_name="axis size")
    if normalized_size < 0.0:
        raise ValueError("axis size must be non-negative")
    half = normalized_size / 2.0
    return (normalized_center - half, normalized_center + half)


def _deduplicate_coords(coords: list[float], *, tolerance: float) -> tuple[float, ...]:
    if not coords:
        raise ValueError("grid coordinates must not be empty")
    deduplicated = [coords[0]]
    for value in coords[1:]:
        if math.isclose(value, deduplicated[-1], rel_tol=0.0, abs_tol=tolerance):
            # Keep the first coordinate (earlier value), don't replace with later one
            continue
        deduplicated.append(value)
    return tuple(deduplicated)


def _snap_to_axis(
    point: float, *, axis_min: float, axis_max: float, tolerance: float
) -> float | None:
    if point < axis_min - tolerance or point > axis_max + tolerance:
        return None
    if math.isclose(point, axis_min, rel_tol=0.0, abs_tol=tolerance):
        return axis_min
    if math.isclose(point, axis_max, rel_tol=0.0, abs_tol=tolerance):
        return axis_max
    return _finite_float(point, field_name="snapping point")


def _axis_name_to_index(axis: AxisName | int) -> int:
    if axis in (0, 1, 2):
        return int(axis)
    axis_map = {"x": 0, "y": 1, "z": 2}
    if isinstance(axis, str) and axis.lower() in axis_map:
        return axis_map[axis.lower()]
    raise ValueError("axis must be one of 0, 1, 2, 'x', 'y', or 'z'")


class ResolvedGridAxis(TaggedModel):
    """Resolved axis boundaries suitable for discretization planning."""

    type: Literal["ResolvedGridAxis"] = "ResolvedGridAxis"
    axis: AxisName
    boundaries: tuple[float, ...]

    @field_validator("boundaries")
    @classmethod
    def _validate_boundaries(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("resolved grid boundaries must contain at least two coordinates")
        normalized = tuple(_finite_float(coord, field_name="grid boundary") for coord in value)
        if len(normalized) == 2 and math.isclose(
            normalized[0], normalized[1], rel_tol=0.0, abs_tol=1e-12
        ):
            return normalized
        for previous, current in zip(normalized, normalized[1:], strict=False):
            if current <= previous:
                raise ValueError("resolved grid boundaries must be strictly increasing")
        return normalized

    @property
    def num_cells(self) -> int:
        """Return the number of primal cells on this axis."""
        if len(self.boundaries) == 2 and math.isclose(
            self.boundaries[0], self.boundaries[1], rel_tol=0.0, abs_tol=1e-12
        ):
            return 0
        return len(self.boundaries) - 1

    @property
    def cell_sizes(self) -> tuple[float, ...]:
        """Return per-cell widths for this resolved axis."""
        if self.num_cells == 0:
            return ()
        return tuple(
            upper - lower
            for lower, upper in zip(self.boundaries, self.boundaries[1:], strict=False)
        )

    @property
    def min_step(self) -> float:
        """Return the smallest step on this axis."""
        return min(self.cell_sizes, default=0.0)

    @property
    def max_step(self) -> float:
        """Return the largest step on this axis."""
        return max(self.cell_sizes, default=0.0)


class ResolvedGrid(TaggedModel):
    """Resolved 3D grid ready for later discretization and mode-solver work."""

    type: Literal["ResolvedGrid"] = "ResolvedGrid"
    center: Vec3
    size: Vec3
    x: ResolvedGridAxis
    y: ResolvedGridAxis
    z: ResolvedGridAxis

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the primal-cell shape of the resolved grid."""
        return (self.x.num_cells, self.y.num_cells, self.z.num_cells)

    @property
    def total_cells(self) -> int:
        """Return the total number of primal cells in the resolved grid."""
        nx, ny, nz = self.shape
        # Use max(1, n) for collapsed dimensions to avoid returning 0 for 2D sims
        return max(1, nx) * max(1, ny) * max(1, nz)

    @property
    def min_step(self) -> float:
        """Return the smallest non-zero step across all axes."""
        steps = [step for step in (self.x.min_step, self.y.min_step, self.z.min_step) if step > 0.0]
        return min(steps, default=0.0)

    @property
    def max_step(self) -> float:
        """Return the largest step across all axes."""
        return max((self.x.max_step, self.y.max_step, self.z.max_step), default=0.0)


class UniformGrid(TaggedModel):
    """Uniform one-dimensional grid specification."""

    type: Literal["UniformGrid"] = "UniformGrid"
    dl: float

    @field_validator("dl")
    @classmethod
    def _validate_dl(cls, value: float) -> float:
        return _positive_float(value, field_name="dl")

    def estimated_min_dl(self) -> float:
        """Return the minimum grid spacing implied by this specification."""
        return self.dl

    def make_boundaries(self, *, center: float, size: float) -> tuple[float, ...]:
        """Return axis boundaries snapped to the requested domain extent."""
        axis_min, axis_max = _normalize_axis_bounds(center=center, size=size)
        if math.isclose(axis_min, axis_max, rel_tol=0.0, abs_tol=1e-12):
            return (axis_min, axis_max)
        num_cells = max(1, math.ceil(size / self.dl))
        snapped_dl = size / num_cells
        return tuple(axis_min + snapped_dl * index for index in range(num_cells + 1))


class CustomGridBoundaries(TaggedModel):
    """Explicit one-dimensional boundary coordinates."""

    type: Literal["CustomGridBoundaries"] = "CustomGridBoundaries"
    coords: tuple[float, ...]

    @field_validator("coords")
    @classmethod
    def _validate_coords(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("coords must contain at least two coordinates")
        normalized = tuple(_finite_float(coord, field_name="coords") for coord in value)
        for previous, current in zip(normalized, normalized[1:], strict=False):
            if current <= previous:
                raise ValueError("coords must be strictly increasing")
        return normalized

    def estimated_min_dl(self) -> float:
        """Return the minimum cell width implied by the supplied boundaries."""
        return min(
            upper - lower for lower, upper in zip(self.coords, self.coords[1:], strict=False)
        )

    def make_boundaries(self, *, center: float, size: float) -> tuple[float, ...]:
        """Validate that the explicit boundaries match the simulation axis bounds."""
        axis_min, axis_max = _normalize_axis_bounds(center=center, size=size)
        tolerance = max(1e-12, abs(size) * 1e-9)
        if not math.isclose(
            self.coords[0], axis_min, rel_tol=0.0, abs_tol=tolerance
        ) or not math.isclose(
            self.coords[-1], axis_max, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                "custom grid boundaries must start and end on the simulation axis bounds"
            )
        if math.isclose(axis_min, axis_max, rel_tol=0.0, abs_tol=tolerance):
            return (axis_min, axis_max)
        return self.coords


class CustomGrid(TaggedModel):
    """Explicit one-dimensional cell sizes centered on the simulation axis."""

    type: Literal["CustomGrid"] = "CustomGrid"
    dl: tuple[float, ...]
    custom_offset: float | None = None

    @field_validator("dl")
    @classmethod
    def _validate_dl(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("dl must contain at least one cell size")
        return tuple(_positive_float(step, field_name="dl") for step in value)

    @field_validator("custom_offset")
    @classmethod
    def _validate_custom_offset(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_float(value, field_name="custom_offset")

    def estimated_min_dl(self) -> float:
        """Return the minimum supplied cell width."""
        return min(self.dl)

    def make_boundaries(self, *, center: float, size: float) -> tuple[float, ...]:
        """Return aligned axis boundaries, extending edge cells as needed."""
        axis_min, axis_max = _normalize_axis_bounds(center=center, size=size)
        if math.isclose(axis_min, axis_max, rel_tol=0.0, abs_tol=1e-12):
            return (axis_min, axis_max)

        cumulative = [0.0]
        for step in self.dl:
            cumulative.append(cumulative[-1] + step)

        offset = self.custom_offset
        if offset is None:
            offset = center - cumulative[-1] / 2.0
        boundaries = [offset + value for value in cumulative]
        tolerance = max(1e-12, abs(size) * 1e-9)

        while boundaries[0] > axis_min + tolerance:
            boundaries.insert(0, boundaries[0] - self.dl[0])
        while boundaries[-1] < axis_max - tolerance:
            boundaries.append(boundaries[-1] + self.dl[-1])

        interior = [
            coord for coord in boundaries if axis_min + tolerance < coord < axis_max - tolerance
        ]
        return _deduplicate_coords([axis_min, *interior, axis_max], tolerance=tolerance)


class GridRefinement(TaggedModel):
    """Local refinement metadata for AutoGrid planning."""

    type: Literal["GridRefinement"] = "GridRefinement"
    refinement_factor: float | None = None
    dl: float | None = None
    num_cells: int = 3

    @field_validator("refinement_factor")
    @classmethod
    def _validate_refinement_factor(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _positive_float(value, field_name="refinement_factor")

    @field_validator("dl")
    @classmethod
    def _validate_dl(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _positive_float(value, field_name="dl")

    @field_validator("num_cells")
    @classmethod
    def _validate_num_cells(cls, value: int) -> int:
        return _positive_int(value, field_name="num_cells")

    @property
    def effective_refinement_factor(self) -> float | None:
        """Return the refinement factor that should be applied in Phase 1."""
        if self.refinement_factor is None and self.dl is None:
            return _DEFAULT_REFINEMENT_FACTOR
        return self.refinement_factor

    def effective_dl(self, *, vacuum_dl: float) -> float:
        """Return the smallest cell width requested by this refinement."""
        base = _positive_float(vacuum_dl, field_name="vacuum_dl")
        refined = math.inf
        if self.effective_refinement_factor is not None:
            refined = min(refined, base / self.effective_refinement_factor)
        if self.dl is not None:
            refined = min(refined, self.dl)
        if not math.isfinite(refined):
            raise ValueError("grid refinement could not determine a target dl")
        return refined


class LayerRefinementSpec(TaggedModel):
    """Layer-aware AutoGrid refinement metadata with an explicit Phase 1 subset."""

    type: Literal["LayerRefinementSpec"] = "LayerRefinementSpec"
    axis: int
    center: Vec3
    size: Vec3
    min_steps_along_axis: float | None = None
    bounds_refinement: GridRefinement | None = None
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] | None = "lower"

    @field_validator("axis")
    @classmethod
    def _validate_axis(cls, value: int) -> int:
        return _axis_name_to_index(value)

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Vec3) -> Vec3:
        return tuple(_finite_float(component, field_name="center") for component in value)

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Vec3) -> Vec3:
        normalized = tuple(_finite_float(component, field_name="size") for component in value)
        if any(component < 0.0 for component in normalized):
            raise ValueError("size components must be non-negative")
        return normalized

    @field_validator("min_steps_along_axis")
    @classmethod
    def _validate_min_steps(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _positive_float(value, field_name="min_steps_along_axis")

    @property
    def bounds(self) -> tuple[Vec3, Vec3]:
        """Return axis-aligned layer bounds."""
        lower = tuple(c - s / 2.0 for c, s in zip(self.center, self.size, strict=True))
        upper = tuple(c + s / 2.0 for c, s in zip(self.center, self.size, strict=True))
        return (lower, upper)

    def axis_bounds(self) -> tuple[float, float]:
        """Return the lower and upper coordinates along the normal axis."""
        lower, upper = self.bounds
        return (lower[self.axis], upper[self.axis])

    def snapping_points(self) -> tuple[float, ...]:
        """Return snapping points implied by the current layer configuration."""
        lower, upper = self.axis_bounds()
        center = self.center[self.axis]
        mode = self.bounds_snapping
        if mode is None:
            return ()
        if mode == "bounds":
            return (lower, upper)
        if mode == "lower":
            return (lower,)
        if mode == "upper":
            return (upper,)
        return (center,)

    def target_dl(self, *, base_dl: float) -> float | None:
        """Return the smallest requested dl inside the layer body, if any."""
        target = math.inf
        thickness = self.size[self.axis]
        if self.min_steps_along_axis is not None and thickness > 0.0:
            target = min(target, thickness / math.ceil(self.min_steps_along_axis))
        if self.bounds_refinement is not None:
            target = min(target, self.bounds_refinement.effective_dl(vacuum_dl=base_dl))
        if not math.isfinite(target):
            return None
        return target


class AutoGrid(TaggedModel):
    """Simplified AutoGrid meshing policy with explicit Phase 1 limitations."""

    type: Literal["AutoGrid"] = "AutoGrid"
    min_steps_per_wvl: float = _DEFAULT_AUTOGRID_STEPS_PER_WVL
    min_steps_per_sim_size: float = _DEFAULT_AUTOGRID_STEPS_PER_SIM_SIZE
    max_scale: float = _DEFAULT_AUTOGRID_MAX_SCALE
    dl_min: float | None = None

    @field_validator("min_steps_per_wvl")
    @classmethod
    def _validate_min_steps_per_wvl(cls, value: float) -> float:
        numeric = _positive_float(value, field_name="min_steps_per_wvl")
        if numeric < 6.0:
            raise ValueError("min_steps_per_wvl must be at least 6.0 for Phase 1 AutoGrid")
        return numeric

    @field_validator("min_steps_per_sim_size")
    @classmethod
    def _validate_min_steps_per_sim_size(cls, value: float) -> float:
        if float(value) < 1.0:
            raise ValueError("min_steps_per_sim_size must be at least 1.0")
        return _positive_float(value, field_name="min_steps_per_sim_size")

    @field_validator("max_scale")
    @classmethod
    def _validate_max_scale(cls, value: float) -> float:
        numeric = _positive_float(value, field_name="max_scale")
        if numeric < 1.0:
            raise ValueError("max_scale must be at least 1.0")
        return numeric

    @field_validator("dl_min")
    @classmethod
    def _validate_dl_min(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _positive_float(value, field_name="dl_min")

    def estimated_min_dl(self, *, wavelength: float, sim_size: Vec3) -> float:
        """Return the Phase 1 vacuum-based lower bound for this AutoGrid."""
        wavelength_dl = (
            _positive_float(wavelength, field_name="wavelength") / self.min_steps_per_wvl
        )
        sim_dl = max(
            _positive_float(length, field_name="sim_size")
            for length in sim_size
            if length > 0.0
        )
        sim_dl /= self.min_steps_per_sim_size
        target = min(wavelength_dl, sim_dl)
        if self.dl_min is not None:
            target = max(target, self.dl_min)
        return target

    def _structure_dl(self, wavelength: float, structure: object) -> float | None:
        """Compute grid cell size for a structure based on its refractive index.

        Uses dl = wavelength / (n * min_steps_per_wvl) where n = sqrt(eps_r * mu_r).
        Returns None for PEC/PMC media which don't affect grid refinement.
        """
        medium = getattr(structure, "medium", None)
        if medium is None:
            return None
        medium_type = getattr(medium, "type", None)
        # PEC and PMC don't have a meaningful refractive index for grid purposes
        if medium_type in ("PECMedium", "PMCMedium"):
            return None
        # For dispersive media, fall back to permittivity (frequency-independent)
        eps_r = getattr(medium, "permittivity", 1.0)
        mu_r = getattr(medium, "permeability", 1.0)
        n = math.sqrt(eps_r * mu_r)
        if n <= 0.0:
            return None
        return wavelength / (n * self.min_steps_per_wvl)

    def make_boundaries(
        self,
        *,
        center: float,
        size: float,
        wavelength: float,
        sim_size: Vec3,
        layer_refinement_specs: Iterable[LayerRefinementSpec] = (),
        structures: Iterable[object] = (),
    ) -> tuple[float, ...]:
        """Resolve a simplified AutoGrid axis with optional layer refinement metadata.

        When structures are provided, automatically detects structure bounding box
        boundaries and inserts grid breakpoints there, with cell sizes determined
        by each structure's refractive index (dl = wavelength / (n * min_steps_per_wvl)).
        """
        axis_min, axis_max = _normalize_axis_bounds(center=center, size=size)
        tolerance = max(1e-12, abs(size) * 1e-9)
        if math.isclose(axis_min, axis_max, rel_tol=0.0, abs_tol=tolerance):
            return (axis_min, axis_max)

        base_dl = self.estimated_min_dl(wavelength=wavelength, sim_size=sim_size)
        regions: list[tuple[float, float, float]] = []
        snapping_points = {axis_min, axis_max}

        # Automatic structure-boundary detection: add breakpoints at structure
        # bounding box edges with cell sizes based on refractive index
        for structure in structures:
            # Use structure.bounds() which returns (Vec3, Vec3) = (lower, upper)
            bounds_fn = getattr(structure, "bounds", None)
            if bounds_fn is None:
                continue
            bounds = bounds_fn()
            if bounds is None:
                continue
            lower, upper = bounds
            # Extract axis coordinate from Vec3 bounds
            axis_lower = float(lower[0])
            axis_upper = float(upper[0])

            clipped_lower = max(axis_min, axis_lower)
            clipped_upper = min(axis_max, axis_upper)
            if clipped_upper < axis_min - tolerance or clipped_lower > axis_max + tolerance:
                continue

            # Compute cell size based on refractive index
            structure_dl = self._structure_dl(wavelength, structure)
            if structure_dl is not None and clipped_upper - clipped_lower > tolerance:
                regions.append((clipped_lower, clipped_upper, structure_dl))

            # Add structure boundaries as snapping points
            for point in (clipped_lower, clipped_upper):
                snapped = _snap_to_axis(
                    point, axis_min=axis_min, axis_max=axis_max, tolerance=tolerance
                )
                if snapped is not None:
                    snapping_points.add(snapped)

        for spec in layer_refinement_specs:
            lower, upper = spec.axis_bounds()
            clipped_lower = max(axis_min, lower)
            clipped_upper = min(axis_max, upper)
            if clipped_upper < axis_min - tolerance or clipped_lower > axis_max + tolerance:
                continue

            target_dl = spec.target_dl(base_dl=base_dl)
            if target_dl is not None and clipped_upper - clipped_lower > tolerance:
                regions.append((clipped_lower, clipped_upper, target_dl))

            if spec.bounds_refinement is not None:
                boundary_dl = spec.bounds_refinement.effective_dl(vacuum_dl=base_dl)
                half_width = boundary_dl * spec.bounds_refinement.num_cells / 2.0
                for point in spec.snapping_points():
                    snapped = _snap_to_axis(
                        point, axis_min=axis_min, axis_max=axis_max, tolerance=tolerance
                    )
                    if snapped is None:
                        continue
                    snapping_points.add(snapped)
                    regions.append(
                        (
                            max(axis_min, snapped - half_width),
                            min(axis_max, snapped + half_width),
                            boundary_dl,
                        )
                    )
            else:
                for point in spec.snapping_points():
                    snapped = _snap_to_axis(
                        point, axis_min=axis_min, axis_max=axis_max, tolerance=tolerance
                    )
                    if snapped is not None:
                        snapping_points.add(snapped)

        breakpoints = sorted(
            snapping_points
            | {start for start, _, _ in regions}
            | {end for _, end, _ in regions}
        )
        boundaries = [axis_min]
        for segment_start, segment_end in zip(breakpoints, breakpoints[1:], strict=False):
            if math.isclose(segment_start, segment_end, rel_tol=0.0, abs_tol=tolerance):
                continue
            midpoint = (segment_start + segment_end) / 2.0
            target_dl = min(
                (
                    dl
                    for start, end, dl in regions
                    if start - tolerance <= midpoint <= end + tolerance
                ),
                default=base_dl,
            )
            num_cells = max(1, math.ceil((segment_end - segment_start) / target_dl))
            snapped_dl = (segment_end - segment_start) / num_cells
            for index in range(1, num_cells + 1):
                boundaries.append(segment_start + snapped_dl * index)

        graded = self._enforce_max_scale(boundaries, tolerance=tolerance)
        return _deduplicate_coords(graded, tolerance=tolerance)

    def _enforce_max_scale(self, boundaries: list[float], *, tolerance: float) -> list[float]:
        if len(boundaries) < 3 or self.max_scale <= 1.0:
            return boundaries

        graded = list(boundaries)
        changed = True
        iteration = 0
        max_iterations = 1000
        while changed and iteration < max_iterations:
            iteration += 1
            changed = False
            cell_sizes = [upper - lower for lower, upper in zip(graded, graded[1:], strict=False)]
            for index in range(len(cell_sizes) - 1):
                left = cell_sizes[index]
                right = cell_sizes[index + 1]
                if left <= tolerance or right <= tolerance:
                    continue
                ratio = max(left, right) / min(left, right)
                if ratio <= self.max_scale + 1e-12:
                    continue
                if left > right:
                    graded.insert(index + 1, (graded[index] + graded[index + 1]) / 2.0)
                else:
                    graded.insert(index + 2, (graded[index + 1] + graded[index + 2]) / 2.0)
                changed = True
                break
        return graded


AxisGridSpec = Annotated[
    UniformGrid | CustomGridBoundaries | CustomGrid | AutoGrid,
    Field(discriminator="type"),
]


class GridSpec(TaggedModel):
    """Collective grid specification for the three principal axes."""

    type: Literal["GridSpec"] = "GridSpec"
    grid_x: AxisGridSpec = Field(default_factory=lambda: UniformGrid(dl=_DEFAULT_UNIFORM_DL))
    grid_y: AxisGridSpec = Field(default_factory=lambda: UniformGrid(dl=_DEFAULT_UNIFORM_DL))
    grid_z: AxisGridSpec = Field(default_factory=lambda: UniformGrid(dl=_DEFAULT_UNIFORM_DL))
    wavelength: float | None = None
    layer_refinement_specs: tuple[LayerRefinementSpec, ...] = ()

    @field_validator("wavelength")
    @classmethod
    def _validate_wavelength(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _positive_float(value, field_name="wavelength")

    @classmethod
    def uniform(cls, dl: float) -> GridSpec:
        """Return a uniform grid specification across all three axes."""
        uniform = UniformGrid(dl=dl)
        return cls(grid_x=uniform, grid_y=uniform, grid_z=uniform)

    def axis_spec(self, axis: AxisName) -> AxisGridSpec:
        """Return the axis-local grid specification."""
        return getattr(self, f"grid_{axis}")

    def resolve_axis(
        self,
        axis: AxisName,
        *,
        center: float,
        size: float,
        sim_size: Vec3,
        structures: Iterable[object] = (),
    ) -> ResolvedGridAxis:
        """Resolve one axis against the simulation domain."""
        spec = self.axis_spec(axis)
        if isinstance(spec, AutoGrid):
            if self.wavelength is None:
                raise ValueError("GridSpec.wavelength is required when any axis uses AutoGrid")
            axis_index = _axis_name_to_index(axis)
            layer_specs = tuple(
                item for item in self.layer_refinement_specs if item.axis == axis_index
            )
            boundaries = spec.make_boundaries(
                center=center,
                size=size,
                wavelength=self.wavelength,
                sim_size=sim_size,
                layer_refinement_specs=layer_specs,
                structures=structures,
            )
        else:
            boundaries = spec.make_boundaries(center=center, size=size)
        return ResolvedGridAxis(axis=axis, boundaries=boundaries)

    def make_grid(
        self, *, center: Vec3, size: Vec3, structures: Iterable[object] = ()
    ) -> ResolvedGrid:
        """Resolve all axes into a concrete discretization layout.

        When structures are provided and AutoGrid is used, structure bounding box
        boundaries are automatically detected and used to insert grid breakpoints,
        with cell sizes determined by each structure's refractive index.
        """
        if len(center) != 3 or len(size) != 3:
            raise ValueError("center and size must each contain exactly three components")
        normalized_center = tuple(_finite_float(value, field_name="center") for value in center)
        normalized_size = tuple(_finite_float(value, field_name="size") for value in size)
        if any(component < 0.0 for component in normalized_size):
            raise ValueError("size components must be non-negative")
        return ResolvedGrid(
            center=normalized_center,
            size=normalized_size,
            x=self.resolve_axis(
                "x",
                center=normalized_center[0],
                size=normalized_size[0],
                sim_size=normalized_size,
                structures=structures,
            ),
            y=self.resolve_axis(
                "y",
                center=normalized_center[1],
                size=normalized_size[1],
                sim_size=normalized_size,
                structures=structures,
            ),
            z=self.resolve_axis(
                "z",
                center=normalized_center[2],
                size=normalized_size[2],
                sim_size=normalized_size,
                structures=structures,
            ),
        )


GridModel = (
    GridSpec
    | UniformGrid
    | CustomGridBoundaries
    | CustomGrid
    | AutoGrid
    | GridRefinement
    | LayerRefinementSpec
)


def grid_model_from_value(value: object) -> GridModel:
    """Normalize a grid payload into the corresponding Phase 1 model."""
    if isinstance(
        value,
        (
            GridSpec,
            UniformGrid,
            CustomGridBoundaries,
            CustomGrid,
            AutoGrid,
            GridRefinement,
            LayerRefinementSpec,
        ),
    ):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"grid payload must be mapping-like, got {type(value)!r}")
    type_name = str(value.get("type", ""))
    model_type = {
        "GridSpec": GridSpec,
        "UniformGrid": UniformGrid,
        "CustomGridBoundaries": CustomGridBoundaries,
        "CustomGrid": CustomGrid,
        "AutoGrid": AutoGrid,
        "GridRefinement": GridRefinement,
        "LayerRefinementSpec": LayerRefinementSpec,
    }.get(type_name)
    if model_type is None:
        raise ValueError(f"unsupported grid model type {type_name!r}")
    return model_type.model_validate(value)


def resolve_grid_spec(
    value: object | None, *, center: Vec3, size: Vec3, structures: Iterable[object] = ()
) -> ResolvedGrid | None:
    """Resolve a grid specification payload against a simulation domain.

    When structures are provided and AutoGrid is used, structure bounding box
    boundaries are automatically detected for grid refinement.
    """
    if value is None:
        return None
    grid_spec = grid_model_from_value(value)
    if not isinstance(grid_spec, GridSpec):
        raise TypeError("simulation.grid_spec must use a top-level GridSpec container")
    return grid_spec.make_grid(center=center, size=size, structures=structures)
