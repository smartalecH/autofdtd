"""Scene-compilation pipeline: public API → IR → compiled runtime artifacts.

This module provides the top-level compilation entry point that orchestrates the
full pipeline from validated public API objects (Simulation, Scene, Structure)
into the compiled runtime artifacts needed for execution.

Pipeline Stages
---------------

1. **Grid Resolution**
   - Resolve ``grid_spec`` against simulation domain (center, size)
   - Produces ``ResolvedGrid`` with per-axis cell boundaries and step sizes

2. **IR Lowering**
   - Lower ``Simulation`` → ``SimulationIR`` via ``simulation_to_ir()``
   - Scene structures inherit precedence metadata from ``structure_precedence()``
   - Sources, monitors, boundaries, subpixel policy lower to typed IR payloads

3. **Scene Materialization**
   - Compile scene background medium coefficients
   - Compile each structure's medium into coefficient fields
   - Allocate material auxiliary state descriptors (PoleResidue, Anisotropic)

4. **Source Planning**
   - Compile each source onto the resolved grid
   - Produce discrete placements, weights, and amplitude metadata

5. **Monitor Planning**
   - Compile each monitor onto the resolved grid
   - Produce discrete placements and recording metadata

6. **Boundary Planning**
   - Compile boundary spec into per-face metadata
   - Allocate PML/ABC boundary state descriptors where needed

7. **Runtime Controls**
   - Compile stop policy (run_time, num_time_steps, shutoff)
   - Produce ``CompiledRuntimeControls`` for the stepping loop

Outputs
-------

``CompiledSimulation`` holds all artifacts needed for runtime execution:

- ``simulation_ir``: The tagged IR snapshot (for transport/inspection)
- ``resolved_grid``: The discretized simulation grid
- ``runtime_controls``: Compiled stop-policy metadata
- ``scene_coefficients``: Material coefficient fields and metadata
- ``compiled_sources``: Per-source placement and amplitude metadata
- ``compiled_monitors``: Per-monitor placement and recording metadata
- ``compiled_boundaries``: Per-face boundary metadata and state descriptors
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from autofdtd.core.containers import Scene, Simulation
from autofdtd.grid import ResolvedGrid, resolve_grid_spec
from autofdtd.ir.models import (
    ExecutionPackageIR,
    ExecutionManifestIR,
    SimulationIR,
    simulation_to_ir,
)
from autofdtd.ir import (
    SceneIR,
    StructureIR,
    medium_to_ir,
    geometry_to_ir,
)

# Material compilation
from autofdtd.compiler.materials import (
    ConstitutiveMode,
    IsotropicMaterialCoefficients,
    MaterialCoefficients,
    PoleResidueMaterialCoefficients,
    PoleResidueTermCoefficients,
    SceneMaterialSample,
    compile_isotropic_medium_coefficients,
    compile_medium_coefficients,
    compile_scene_medium_coefficients,
    sample_scene_mediums,
)

# Source compilation
from autofdtd.compiler.sources import (
    CompiledAstigmaticGaussianBeam,
    CompiledCustomCurrentSource,
    CompiledCustomFieldSource,
    CompiledGaussianBeam,
    CompiledModeSource,
    CompiledPlaneWave,
    CompiledPointDipole,
    CompiledTFSF,
    CompiledUniformCurrentSource,
    compile_astigmatic_gaussian_beam,
    compile_custom_current_source,
    compile_custom_field_source,
    compile_gaussian_beam,
    compile_mode_source,
    compile_plane_wave,
    compile_point_dipole,
    compile_tfsf,
    compile_uniform_current_source,
)

# Monitor compilation
from autofdtd.compiler.monitors import (
    CompiledFieldMonitor,
    CompiledFluxMonitor,
    CompiledMediumMonitor,
    CompiledModeMonitor,
    CompiledProjectionMonitor,
    CompiledSurfaceFieldMonitor,
    compile_field_monitor,
    compile_flux_monitor,
    compile_medium_monitor,
    compile_mode_monitor,
)

# Boundary compilation
from autofdtd.compiler.boundaries import (
    CompiledBoundarySpec,
    compile_boundary_spec,
)

# Runtime control compilation
from autofdtd.compiler.runtime import (
    CompiledRuntimeControls,
    compile_runtime_controls,
)

# Chunk layout
from autofdtd.runtime.chunk import ChunkLayout, build_chunk_layout

# Discretization
from autofdtd.compiler.discretization import CoefficientField, discretize_scene


@dataclass(frozen=True)
class CompiledSceneCoefficients:
    """Compiled material coefficients and scene-level metadata.

    Attributes
    ----------
    background : MaterialCoefficients
        Compiled coefficients for the scene background medium.
    structures : tuple[CompiledStructureCoefficients, ...]
        Compiled coefficients for each structure, in precedence order.
    scene_material_samples : tuple[SceneMaterialSample, ...]
        Pointwise material samples at key locations for diagnostics.
    coefficient_field : CoefficientField | None
        Discretized per-cell coefficient arrays (eps, mu, drive, modes).
        Populated by calling ``discretize_scene()`` during compilation.
    """

    background: MaterialCoefficients
    structures: tuple[CompiledStructureCoefficients, ...]
    scene_material_samples: tuple[SceneMaterialSample, ...] = ()
    coefficient_field: CoefficientField | None = None


@dataclass(frozen=True)
class CompiledStructureCoefficients:
    """Compiled material coefficients and metadata for one structure."""

    structure_ir: StructureIR
    material_coefficients: MaterialCoefficients
    has_dispersive_components: bool
    has_anisotropic_components: bool


@dataclass(frozen=True)
class CompiledSources:
    """Compiled source placement and amplitude metadata."""

    uniform_current: tuple[CompiledUniformCurrentSource, ...] = ()
    point_dipole: tuple[CompiledPointDipole, ...] = ()
    custom_current: tuple[CompiledCustomCurrentSource, ...] = ()
    custom_field: tuple[CompiledCustomFieldSource, ...] = ()
    plane_wave: tuple[CompiledPlaneWave, ...] = ()
    gaussian_beam: tuple[CompiledGaussianBeam, ...] = ()
    astigmatic_gaussian_beam: tuple[CompiledAstigmaticGaussianBeam, ...] = ()
    mode_source: tuple[CompiledModeSource, ...] = ()
    tfsf: tuple[CompiledTFSF, ...] = ()


@dataclass(frozen=True)
class CompiledMonitors:
    """Compiled monitor placement and recording metadata."""

    field: tuple[CompiledFieldMonitor, ...] = ()
    flux: tuple[CompiledFluxMonitor, ...] = ()
    medium: tuple[CompiledMediumMonitor, ...] = ()
    mode: tuple[CompiledModeMonitor, ...] = ()
    projection: tuple[CompiledProjectionMonitor, ...] = ()
    surface_field: tuple[CompiledSurfaceFieldMonitor, ...] = ()


@dataclass(frozen=True)
class CompiledBoundaries:
    """Compiled boundary metadata and state descriptors."""

    boundary_spec: CompiledBoundarySpec
    needs_pml_state: bool = False
    needs_abc_state: bool = False


@dataclass(frozen=True)
class CompiledSimulation:
    """Complete compiled simulation ready for runtime execution.

    This container holds all artifacts produced by the compilation pipeline:
    IR snapshot, resolved grid, compiled controls, material coefficients,
    source plans, monitor plans, chunk layout, and boundary metadata.

    Attributes
    ----------
    simulation_ir : SimulationIR
        Tagged IR snapshot of the compiled simulation.
    resolved_grid : ResolvedGrid
        Discretized grid with cell boundaries and step sizes.
    runtime_controls : CompiledRuntimeControls
        Compiled stop-policy and timestep metadata.
    scene_coefficients : CompiledSceneCoefficients
        Compiled material coefficients for the scene.
    compiled_sources : CompiledSources
        Compiled source placement and amplitude metadata.
    compiled_monitors : CompiledMonitors
        Compiled monitor placement and recording metadata.
    compiled_boundaries : CompiledBoundaries
        Compiled boundary metadata and state descriptors.
    chunk_layout : ChunkLayout
        Chunk decomposition plan with per-face halo metadata.

    Notes
    -----
    The compiled simulation is immutable (frozen dataclass). To re-compile with
    different parameters, create a new ``Simulation`` and call ``compile_simulation()``.

    This is a runtime artifact container, not a serializable model. Use the
    ``simulation_ir`` field for serialization and transport.
    """

    simulation_ir: SimulationIR
    resolved_grid: ResolvedGrid
    runtime_controls: CompiledRuntimeControls
    scene_coefficients: CompiledSceneCoefficients
    compiled_sources: CompiledSources
    compiled_monitors: CompiledMonitors
    compiled_boundaries: CompiledBoundaries
    chunk_layout: "ChunkLayout"

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the Yee cell grid shape (nx, ny, nz)."""
        return (
            self.resolved_grid.x.num_cells,
            self.resolved_grid.y.num_cells,
            self.resolved_grid.z.num_cells,
        )

    @property
    def total_cells(self) -> int:
        """Return the total number of Yee cells in the grid."""
        nx, ny, nz = self.grid_shape
        return nx * ny * nz

    @property
    def num_sources(self) -> int:
        """Return the total number of compiled sources."""
        return (
            len(self.compiled_sources.uniform_current)
            + len(self.compiled_sources.point_dipole)
            + len(self.compiled_sources.custom_current)
            + len(self.compiled_sources.custom_field)
            + len(self.compiled_sources.plane_wave)
            + len(self.compiled_sources.gaussian_beam)
            + len(self.compiled_sources.astigmatic_gaussian_beam)
            + len(self.compiled_sources.mode_source)
            + len(self.compiled_sources.tfsf)
        )

    @property
    def num_monitors(self) -> int:
        """Return the total number of compiled monitors."""
        return (
            len(self.compiled_monitors.field)
            + len(self.compiled_monitors.flux)
            + len(self.compiled_monitors.medium)
            + len(self.compiled_monitors.mode)
            + len(self.compiled_monitors.projection)
        )


def _compile_sources_for_simulation(
    simulation: Simulation,
    *,
    grid: ResolvedGrid,
) -> CompiledSources:
    """Compile all sources in a simulation onto the resolved grid."""

    uniform_current: list[CompiledUniformCurrentSource] = []
    point_dipole: list[CompiledPointDipole] = []
    custom_current: list[CompiledCustomCurrentSource] = []
    custom_field: list[CompiledCustomFieldSource] = []
    plane_wave: list[CompiledPlaneWave] = []
    gaussian_beam: list[CompiledGaussianBeam] = []
    astigmatic_gaussian_beam: list[CompiledAstigmaticGaussianBeam] = []
    mode_source: list[CompiledModeSource] = []
    tfsf: list[CompiledTFSF] = []

    # Import source types for isinstance checks
    from autofdtd.sources import (
        UniformCurrentSource,
        PointDipole,
        CustomCurrentSource,
        CustomFieldSource,
        PlaneWave,
        GaussianBeam,
        AstigmaticGaussianBeam,
    )
    from autofdtd.sources.mode import ModeSource
    from autofdtd.sources.tfsf import TFSF

    for source in simulation.sources:
        if isinstance(source, UniformCurrentSource):
            uniform_current.append(compile_uniform_current_source(source, grid=grid))
        elif isinstance(source, PointDipole):
            point_dipole.append(compile_point_dipole(source, grid=grid))
        elif isinstance(source, CustomCurrentSource):
            custom_current.append(compile_custom_current_source(source, grid=grid))
        elif isinstance(source, CustomFieldSource):
            custom_field.append(compile_custom_field_source(source, grid=grid))
        elif isinstance(source, PlaneWave):
            plane_wave.append(compile_plane_wave(source, grid=grid))
        elif isinstance(source, GaussianBeam):
            gaussian_beam.append(compile_gaussian_beam(source, grid=grid))
        elif isinstance(source, AstigmaticGaussianBeam):
            astigmatic_gaussian_beam.append(compile_astigmatic_gaussian_beam(source, grid=grid))
        elif isinstance(source, ModeSource):
            mode_source.append(
                compile_mode_source(
                    source,
                    grid=grid,
                    scene=simulation,
                    sim_center=simulation.center,
                    sim_size=simulation.size,
                    dt=compile_runtime_controls(simulation).dt,
                )
            )
        elif isinstance(source, TFSF):
            tfsf.append(compile_tfsf(source, grid=grid))
        # Unknown source types are silently skipped at compile time;
        # validation during Simulation construction would have already
        # rejected unsupported source types

    return CompiledSources(
        uniform_current=tuple(uniform_current),
        point_dipole=tuple(point_dipole),
        custom_current=tuple(custom_current),
        custom_field=tuple(custom_field),
        plane_wave=tuple(plane_wave),
        gaussian_beam=tuple(gaussian_beam),
        astigmatic_gaussian_beam=tuple(astigmatic_gaussian_beam),
        mode_source=tuple(mode_source),
        tfsf=tuple(tfsf),
    )


def _compile_monitors_for_simulation(
    simulation: Simulation,
    *,
    grid: ResolvedGrid,
    dt: float = 1.0,
) -> CompiledMonitors:
    """Compile all monitors in a simulation onto the resolved grid."""

    field: list[CompiledFieldMonitor] = []
    flux: list[CompiledFluxMonitor] = []
    medium: list[CompiledMediumMonitor] = []
    mode: list[CompiledModeMonitor] = []
    projection: list[CompiledProjectionMonitor] = []
    surface_field: list = []

    # Import monitor types for isinstance checks
    from autofdtd.monitors import (
        FieldMonitor,
        FieldTimeMonitor,
        AuxFieldTimeMonitor,
        FluxMonitor,
        FluxTimeMonitor,
        MediumMonitor,
        PermittivityMonitor,
        ModeMonitor,
        ModeSolverMonitor,
        FieldProjectionAngleMonitor,
        FieldProjectionCartesianMonitor,
        FieldProjectionKSpaceMonitor,
        DiffractionMonitor,
        DirectivityMonitor,
        GaussianOverlapMonitor,
        AstigmaticGaussianOverlapMonitor,
        SurfaceFieldMonitor,
        SurfaceFieldTimeMonitor,
    )

    for monitor in simulation.monitors:
        if isinstance(monitor, (FieldMonitor, FieldTimeMonitor, AuxFieldTimeMonitor)):
            compiled = compile_field_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                fields=monitor.fields,
                interval=monitor.interval,
                start=monitor.start,
                overwrite=getattr(monitor, "overwrite", True),
                freqs=getattr(monitor, "freqs", ()),
                grid=grid,
                dt=dt,
            )
            field.append(compiled)
        elif isinstance(monitor, (FluxMonitor, FluxTimeMonitor)):
            compiled = compile_flux_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                direction=getattr(monitor, "direction", "+"),
                interval=monitor.interval,
                start=monitor.start,
                freqs=getattr(monitor, "freqs", ()),
                grid=grid,
                dt=dt,
            )
            flux.append(compiled)
        elif isinstance(monitor, (MediumMonitor, PermittivityMonitor)):
            compiled = compile_medium_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                interval=monitor.interval,
                start=monitor.start,
                num_freqs=getattr(monitor, "num_freqs", 1),
                freqs=getattr(monitor, "freqs", ()),
                grid=grid,
            )
            medium.append(compiled)
        elif isinstance(monitor, (ModeMonitor, ModeSolverMonitor)):
            compiled = compile_mode_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                interval=monitor.interval,
                start=monitor.start,
                mode_spec=monitor.mode_spec,
                num_freqs=getattr(monitor, "num_freqs", 1),
                freqs=getattr(monitor, "freqs", ()),
                direction=getattr(monitor, "direction", "+"),
                grid=grid,
                dt=dt,
            )
            mode.append(compiled)
        elif isinstance(
            monitor,
            (
                FieldProjectionAngleMonitor,
                FieldProjectionCartesianMonitor,
                FieldProjectionKSpaceMonitor,
                DiffractionMonitor,
                DirectivityMonitor,
            ),
        ):
            # Projection monitors compile to a generic projection monitor type
            # for now - the projection-specific compilation is handled in the
            # projection monitor compiler
            from autofdtd.compiler.monitors import compile_projection_monitor

            compiled = compile_projection_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                normal_vector=getattr(monitor, 'normal_vector', (0.0, 0.0, 1.0)),
                projection_distance=getattr(monitor, 'projection_distance', 1e5),
                interval=monitor.interval,
                start=monitor.start,
                num_freqs=getattr(monitor, 'num_freqs', 1),
                freqs=getattr(monitor, 'freqs', ()),
                phi=getattr(monitor, 'phi', (-90.0, 90.0, 181)),
                theta=getattr(monitor, 'theta', (0.0, 180.0, 181)),
                x=getattr(monitor, 'x', (-50.0, 50.0, 201)),
                y=getattr(monitor, 'y', (-50.0, 50.0, 201)),
                num_k=getattr(monitor, 'num_k', 1),
                kx=getattr(monitor, 'kx', (-10.0, 10.0, 21)),
                ky=getattr(monitor, 'ky', (-10.0, 10.0, 21)),
                grid=grid,
                dt=dt,
            )
            projection.append(compiled)
        elif isinstance(monitor, (SurfaceFieldMonitor, SurfaceFieldTimeMonitor)):
            from autofdtd.compiler.monitors import compile_surface_field_monitor

            compiled = compile_surface_field_monitor(
                name=monitor.name,
                monitor_type=monitor.type,
                center=monitor.center,
                size=monitor.size,
                fields=monitor.fields,
                interval=monitor.interval,
                start=monitor.start,
                freqs=getattr(monitor, "freqs", ()),
                grid=grid,
                dt=dt,
            )
            surface_field.append(compiled)
        elif isinstance(monitor, (GaussianOverlapMonitor, AstigmaticGaussianOverlapMonitor)):
            raise NotImplementedError(
                f"Monitor type {monitor.type} is not yet implemented. "
                "Supported monitors are: FieldMonitor, FieldTimeMonitor, AuxFieldTimeMonitor, "
                "FluxMonitor, FluxTimeMonitor, MediumMonitor, PermittivityMonitor, ModeMonitor, "
                "ModeSolverMonitor, FieldProjectionAngleMonitor, FieldProjectionCartesianMonitor, "
                "FieldProjectionKSpaceMonitor, DiffractionMonitor, DirectivityMonitor, "
                "SurfaceFieldMonitor, SurfaceFieldTimeMonitor."
            )
        # Unknown monitor types are silently skipped at compile time

    return CompiledMonitors(
        field=tuple(field),
        flux=tuple(flux),
        medium=tuple(medium),
        mode=tuple(mode),
        projection=tuple(projection),
        surface_field=tuple(surface_field),
    )


def _compile_scene_coefficients(
    scene: Scene,
    grid: ResolvedGrid,
    dt: float,
) -> CompiledSceneCoefficients:
    """Compile scene materials into coefficient fields."""

    # Compile scene background medium
    background = compile_medium_coefficients(scene.medium, dt=dt)

    # Compile structure media
    structures: list[CompiledStructureCoefficients] = []
    for structure in scene.structures_in_resolution_order():
        material_coefficients = compile_medium_coefficients(structure.medium, dt=dt)

        # Check if medium has dispersive or anisotropic components
        has_dispersive = (
            isinstance(material_coefficients, PoleResidueMaterialCoefficients)
            or (
                hasattr(material_coefficients, "medium_type")
                and material_coefficients.medium_type
                in {"Sellmeier", "Lorentz", "Drude", "Debye", "PoleResidue"}
            )
        )
        has_anisotropic = hasattr(material_coefficients, "medium_type") and (
            material_coefficients.medium_type == "AnisotropicMedium"
        )

        structures.append(
            CompiledStructureCoefficients(
                structure_ir=structure_to_ir_for_compilation(structure, scene),
                material_coefficients=material_coefficients,
                has_dispersive_components=has_dispersive,
                has_anisotropic_components=has_anisotropic,
            )
        )

    # Sample scene at a few key points for diagnostics
    # Use grid cell centers as sample points
    sample_points: list[list[float]] = []
    for ix in range(min(grid.x.num_cells, 5)):
        x_c = 0.5 * (grid.x.boundaries[ix] + grid.x.boundaries[ix + 1])
        for iy in range(min(grid.y.num_cells, 5)):
            y_c = 0.5 * (grid.y.boundaries[iy] + grid.y.boundaries[iy + 1])
            for iz in range(min(grid.z.num_cells, 5)):
                z_c = 0.5 * (grid.z.boundaries[iz] + grid.z.boundaries[iz + 1])
                sample_points.append([x_c, y_c, z_c])

    scene_material_samples = sample_scene_mediums(scene, sample_points)

    # Discretize scene to produce per-cell material and coefficient fields
    _, coefficient_field = discretize_scene(scene, grid, dt)

    return CompiledSceneCoefficients(
        background=background,
        structures=tuple(structures),
        scene_material_samples=scene_material_samples,
        coefficient_field=coefficient_field,
    )


def structure_to_ir_for_compilation(
    structure: Simulation | Scene,
    scene: Scene,
) -> StructureIR:
    """Lower a structure for scene compilation with proper precedence metadata.

    This is a specialized variant of ``ir.models.structure_to_ir()`` that
    uses the scene's precedence order instead of assuming equal priorities.
    """
    from autofdtd.ir.models import geometry_to_ir, medium_to_ir, structure_to_ir

    precedence_by_index = {
        entry.source_index: entry for entry in scene.structure_precedence()
    }

    # Find the source index for this structure
    source_index = None
    for idx, s in enumerate(scene.structures):
        if s is structure:
            source_index = idx
            break
    if source_index is None:
        source_index = 0

    entry = precedence_by_index.get(source_index)
    precedence_rank = entry.precedence_rank if entry else 0
    resolved_priority = entry.resolved_priority if entry else 0

    return StructureIR(
        name=structure.name,
        source_index=source_index,
        priority=structure.priority,
        resolved_priority=resolved_priority,
        precedence_rank=precedence_rank,
        geometry=geometry_to_ir(structure.geometry),
        medium=medium_to_ir(structure.medium),
        background_medium=(
            medium_to_ir(structure.background_medium)
            if structure.background_medium is not None
            else None
        ),
    )


def compile_simulation(
    simulation: Simulation,
    *,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
    num_chunks: tuple[int, int, int] = (1, 1, 1),
) -> CompiledSimulation:
    """Compile a public Simulation into all runtime artifacts needed for execution.

    This is the top-level entry point for the scene-compilation pipeline. It
    orchestrates grid resolution, IR lowering, and compilation of materials,
    sources, monitors, boundaries, and runtime controls.

    Parameters
    ----------
    simulation : Simulation
        The validated public simulation object to compile.
    dt : float, optional
        Override the compiled timestep size. If None, the timestep is
        derived from the grid and Courant factor.
    max_steps : int, optional
        Override the maximum number of timesteps. If None, the step count
        is derived from ``run_time`` and ``dt``.
    shutoff_check_interval : int, default=10
        Interval for convergence checks when shutoff is enabled.
    num_chunks : tuple[int, int, int], default=(1, 1, 1)
        Chunk decomposition for multi-GPU execution. Use (1, 1, 1) for
        single-GPU (monolithic) execution, or e.g. (2, 1, 1) to split
        the domain along the x-axis across 2 GPUs.

    Returns
    -------
    CompiledSimulation
        All compiled artifacts needed for runtime execution.

    Raises
    ------
    ValueError
        If the simulation grid cannot be resolved or compilation fails.

    Notes
    -----
    The compilation pipeline is deterministic for a given simulation
    configuration. Re-compiling the same simulation with the same parameters
    produces equivalent artifacts.

    Examples
    --------
    >>> from autofdtd import Simulation, Box, Medium, UniformGrid, compile_simulation
    >>> sim = Simulation(
    ...     size=(1.0, 1.0, 1.0),
    ...     run_time=1e-12,
    ...     grid_spec=UniformGrid(dl=0.01),
    ... )
    >>> compiled = compile_simulation(sim)
    >>> compiled.grid_shape
    (100, 100, 100)
    >>> compiled.runtime_controls.num_time_steps  # doctest: +SKIP
    1000
    """
    # 1. Resolve grid
    resolved_grid = simulation.resolved_grid()
    if resolved_grid is None:
        raise ValueError(
            "simulation grid_spec could not be resolved; "
            "provide a GridSpec with valid dl or wavelength"
        )

    # 2. Lower to IR
    simulation_ir = simulation_to_ir(simulation)

    # 3. Compile runtime controls
    runtime_controls = compile_runtime_controls(
        simulation,
        dt=dt,
        max_steps=max_steps,
        shutoff_check_interval=shutoff_check_interval,
    )

    # 4. Compile scene materials (needs dt from compiled runtime controls)
    scene_coefficients = _compile_scene_coefficients(
        simulation, resolved_grid, dt=runtime_controls.dt
    )

    # 5. Compile sources
    compiled_sources = _compile_sources_for_simulation(simulation, grid=resolved_grid)

    # 6. Compile monitors
    compiled_monitors = _compile_monitors_for_simulation(
        simulation, grid=resolved_grid, dt=runtime_controls.dt
    )

    # 7. Compile boundaries
    if simulation.boundary_spec is not None:
        compiled_boundaries = CompiledBoundaries(
            boundary_spec=compile_boundary_spec(
                simulation.boundary_spec, symmetry=simulation.symmetry
            ),
            needs_pml_state=any(
                simulation_ir.boundary_spec.pml_faces if simulation_ir.boundary_spec else ()
            ),
            needs_abc_state=any(
                simulation_ir.boundary_spec.abc_faces if simulation_ir.boundary_spec else ()
            ),
        )
    else:
        # No boundary spec - use default (all boundaries open/PML)
        from autofdtd.boundaries import BoundarySpec, Boundary
        from autofdtd.boundaries.models import PML

        default_spec = BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        )
        compiled_boundaries = CompiledBoundaries(
            boundary_spec=compile_boundary_spec(
                default_spec, symmetry=simulation.symmetry
            ),
            needs_pml_state=True,
            needs_abc_state=False,
        )

    # 8. Build chunk layout
    grid_shape = (
        resolved_grid.x.num_cells,
        resolved_grid.y.num_cells,
        resolved_grid.z.num_cells,
    )
    cell_sizes = (
        resolved_grid.x.cell_sizes[0] if resolved_grid.x.cell_sizes else 1.0,
        resolved_grid.y.cell_sizes[0] if resolved_grid.y.cell_sizes else 1.0,
        resolved_grid.z.cell_sizes[0] if resolved_grid.z.cell_sizes else 1.0,
    )
    chunk_layout = build_chunk_layout(
        num_chunks=num_chunks,
        grid_shape=grid_shape,
        cell_sizes=cell_sizes,
        boundary_spec=compiled_boundaries.boundary_spec,
        symmetry=simulation.symmetry,
    )

    return CompiledSimulation(
        simulation_ir=simulation_ir,
        resolved_grid=resolved_grid,
        runtime_controls=runtime_controls,
        scene_coefficients=scene_coefficients,
        compiled_sources=compiled_sources,
        compiled_monitors=compiled_monitors,
        compiled_boundaries=compiled_boundaries,
        chunk_layout=chunk_layout,
    )


def simulation_to_execution_package(
    simulation: Simulation,
    *,
    metadata: dict[str, Any] | None = None,
    dt: float | None = None,
    max_steps: int | None = None,
    shutoff_check_interval: int = 10,
) -> ExecutionPackageIR:
    """Compile a simulation and package it as a transportable execution bundle.

    Parameters
    ----------
    simulation : Simulation
        The validated public simulation to compile and package.
    metadata : dict, optional
        Additional metadata to store in the bundle manifest.
    dt, max_steps, shutoff_check_interval
        Forwarded to ``compile_simulation()``.

    Returns
    -------
    ExecutionPackageIR
        Transportable bundle containing the simulation IR and manifest.

    See Also
    --------
    compile_simulation : The underlying compilation function.
    """
    compiled = compile_simulation(
        simulation,
        dt=dt,
        max_steps=max_steps,
        shutoff_check_interval=shutoff_check_interval,
    )
    return ExecutionPackageIR.from_simulation(simulation, metadata=metadata)


# Export compile_boundary_spec so it can be used directly
from autofdtd.compiler.boundaries import compile_boundary_spec
