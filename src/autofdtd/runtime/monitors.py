"""Monitor runtime helpers for Phase 1 field recording."""

from __future__ import annotations

from typing import Literal

from autofdtd.monitors import (
    FieldData,
    FieldMonitor,
    FieldTimeMonitor,
    AuxFieldTimeMonitor,
    MediumMonitor,
    PermittivityMonitor,
    MediumMonitorData,
    PermittivityData,
    FluxMonitor,
    FluxTimeMonitor,
    FluxData,
    ModeMonitor,
    ModeSolverMonitor,
    ModeData,
)
from autofdtd.compiler.monitors import (
    CompiledFieldMonitor,
    FieldMonitorState,
    compile_field_monitor,
    CompiledMediumMonitor,
    MediumMonitorState,
    compile_medium_monitor,
    CompiledFluxMonitor,
    FluxMonitorState,
    compile_flux_monitor,
    CompiledModeMonitor,
    ModeMonitorState,
    compile_mode_monitor,
)


def build_field_monitor_runtime(
    monitor,
    *,
    grid,
) -> tuple[CompiledFieldMonitor, FieldMonitorState]:
    """Build runtime objects for a field monitor.

    Args:
        monitor: FieldMonitor, FieldTimeMonitor, or AuxFieldTimeMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledFieldMonitor, FieldMonitorState)
    """
    if isinstance(monitor, (FieldMonitor, FieldTimeMonitor, AuxFieldTimeMonitor)):
        monitor_type = monitor.type
        center = monitor.center
        size = monitor.size
        fields = monitor.fields
        interval = monitor.interval
        start = monitor.start
        overwrite = getattr(monitor, "overwrite", True)
    else:
        raise TypeError(f"expected field monitor, got {type(monitor)!r}")

    compiled = compile_field_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        fields=fields,
        interval=interval,
        start=start,
        overwrite=overwrite,
        freqs=getattr(monitor, "freqs", ()),
        grid=grid,
    )

    state = FieldMonitorState(compiled=compiled)

    return compiled, state


def record_monitor_fields(
    monitor_states: list[FieldMonitorState],
    electric_field,
    magnetic_field,
    *,
    time: float,
    step_index: int,
) -> None:
    """Record field values for all active time-domain monitors.

    This function samples the fields at each monitor's placement cells
    and stores the values in the monitor's state for later retrieval.

    Args:
        monitor_states: List of FieldMonitorState objects
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        time: Current simulation time
        step_index: Current step index
    """
    for state in monitor_states:
        compiled = state.compiled

        # Check if this step should be recorded
        if step_index < compiled.start:
            continue
        if (step_index - compiled.start) % compiled.interval != 0:
            continue

        if compiled.is_time_domain:
            state.record_time_domain(electric_field, magnetic_field, time)
        elif compiled.is_frequency_domain:
            state.accumulate_dft(electric_field, magnetic_field, time)


def extract_monitor_data(
    monitor_states: list[FieldMonitorState],
    grid,
) -> dict[str, FieldData]:
    """Extract monitor data from recorded states.

    Args:
        monitor_states: List of FieldMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to FieldData objects
    """
    result = {}

    for state in monitor_states:
        compiled = state.compiled

        field_data_dict = state.to_field_data()

        # Get grid coordinates for the monitor region
        x_coords = tuple(grid.x.boundaries[i] for i in range(len(grid.x.boundaries) - 1))
        y_coords = tuple(grid.y.boundaries[i] for i in range(len(grid.y.boundaries) - 1))
        z_coords = tuple(grid.z.boundaries[i] for i in range(len(grid.z.boundaries) - 1))

        # Build FieldData with stored values
        fd_kwargs = {
            "monitor_name": compiled.name,
            "monitor_type": compiled.monitor_type,
        }

        for field in compiled.fields:
            if field in field_data_dict:
                fd_kwargs[field] = field_data_dict[field]

        if compiled.is_time_domain:
            time_stamps, _ = state.time_data()
            fd_kwargs["t"] = time_stamps

        # Add coordinate info (limited for now)
        # x, y, z would be the coordinates at the monitor placements

        result[compiled.name] = FieldData(**fd_kwargs)

    return result


def build_medium_monitor_runtime(
    monitor,
    *,
    grid,
) -> tuple[CompiledMediumMonitor, MediumMonitorState]:
    """Build runtime objects for a medium or permittivity monitor.

    Args:
        monitor: MediumMonitor or PermittivityMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledMediumMonitor, MediumMonitorState)
    """
    if isinstance(monitor, MediumMonitor):
        monitor_type = "MediumMonitor"
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, PermittivityMonitor):
        monitor_type = "PermittivityMonitor"
        num_freqs = 1
        freqs = ()
    else:
        raise TypeError(f"expected medium or permittivity monitor, got {type(monitor)!r}")

    compiled = compile_medium_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=monitor.center,
        size=monitor.size,
        interval=monitor.interval,
        start=monitor.start,
        num_freqs=num_freqs,
        freqs=freqs,
        grid=grid,
    )

    state = MediumMonitorState(compiled=compiled)

    return compiled, state


def extract_medium_monitor_data(
    monitor_states: list[MediumMonitorState],
    grid,
) -> dict[str, MediumMonitorData | PermittivityData]:
    """Extract monitor data from medium monitor states.

    Args:
        monitor_states: List of MediumMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to MediumMonitorData or PermittivityData
    """
    result = {}

    for state in monitor_states:
        compiled = state.compiled

        if compiled.monitor_type == "PermittivityMonitor":
            data_dict = state.to_permittivity_monitor_data()
            result[compiled.name] = PermittivityData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                eps_xx=data_dict.get("eps_xx"),
                eps_yy=data_dict.get("eps_yy"),
                eps_zz=data_dict.get("eps_zz"),
                x=None,
                y=None,
                z=None,
            )
        else:
            data_dict = state.to_medium_monitor_data()
            result[compiled.name] = MediumMonitorData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                eps_xx=data_dict.get("eps_xx"),
                eps_yy=data_dict.get("eps_yy"),
                eps_zz=data_dict.get("eps_zz"),
                mu_xx=data_dict.get("mu_xx"),
                mu_yy=data_dict.get("mu_yy"),
                mu_zz=data_dict.get("mu_zz"),
                x=None,
                y=None,
                z=None,
            )

    return result


def build_flux_monitor_runtime(
    monitor,
    *,
    grid,
) -> tuple[CompiledFluxMonitor, FluxMonitorState]:
    """Build runtime objects for a flux monitor.

    Args:
        monitor: FluxMonitor or FluxTimeMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledFluxMonitor, FluxMonitorState)
    """
    if isinstance(monitor, FluxMonitor):
        monitor_type = "FluxMonitor"
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, FluxTimeMonitor):
        monitor_type = "FluxTimeMonitor"
        freqs = ()
    else:
        raise TypeError(f"expected flux monitor, got {type(monitor)!r}")

    compiled = compile_flux_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        freqs=freqs,
        grid=grid,
    )

    state = FluxMonitorState(compiled=compiled)

    return compiled, state


def record_monitor_flux(
    monitor_states: list[FluxMonitorState],
    electric_field,
    magnetic_field,
    *,
    time: float,
    step_index: int,
) -> None:
    """Record flux values for all active flux monitors.

    This function computes the flux through each monitor's surface
    and stores the values in the monitor's state for later retrieval.

    Args:
        monitor_states: List of FluxMonitorState objects
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        time: Current simulation time
        step_index: Current step index
    """
    for state in monitor_states:
        compiled = state.compiled

        # Check if this step should be recorded
        if step_index < compiled.start:
            continue
        if (step_index - compiled.start) % compiled.interval != 0:
            continue

        if compiled.is_time_domain:
            state.record_flux_time_domain(electric_field, magnetic_field, time)
        elif compiled.is_frequency_domain:
            state.accumulate_flux_dft(electric_field, magnetic_field, time)


def extract_flux_monitor_data(
    monitor_states: list[FluxMonitorState],
    grid,
) -> dict[str, FluxData]:
    """Extract monitor data from flux monitor states.

    Args:
        monitor_states: List of FluxMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to FluxData objects
    """
    result = {}

    for state in monitor_states:
        compiled = state.compiled

        data_dict = state.to_flux_data()

        result[compiled.name] = FluxData(
            monitor_name=compiled.name,
            monitor_type=compiled.monitor_type,
            flux=data_dict.get("flux"),
            t=data_dict.get("t"),
        )

    return result


def build_mode_monitor_runtime(
    monitor,
    *,
    grid,
) -> tuple[CompiledModeMonitor, ModeMonitorState]:
    """Build runtime objects for a mode monitor.

    Args:
        monitor: ModeMonitor or ModeSolverMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledModeMonitor, ModeMonitorState)
    """
    if isinstance(monitor, ModeMonitor):
        monitor_type = "ModeMonitor"
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
        mode_spec = getattr(monitor, "mode_spec", None)
    elif isinstance(monitor, ModeSolverMonitor):
        monitor_type = "ModeSolverMonitor"
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
        mode_spec = getattr(monitor, "mode_spec", None)
    else:
        raise TypeError(f"expected mode monitor, got {type(monitor)!r}")

    # Convert mode_spec to dict if it's a model
    mode_spec_dict = None
    if mode_spec is not None:
        if hasattr(mode_spec, "model_dump"):
            mode_spec_dict = mode_spec.model_dump(mode="json", exclude_none=True)
        else:
            mode_spec_dict = dict(mode_spec) if isinstance(mode_spec, dict) else None

    compiled = compile_mode_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=monitor.center,
        size=monitor.size,
        direction=monitor.direction,
        interval=monitor.interval,
        start=monitor.start,
        mode_spec=mode_spec_dict,
        num_freqs=num_freqs,
        freqs=freqs,
        grid=grid,
    )

    state = ModeMonitorState(compiled=compiled)

    return compiled, state


def record_monitor_modes(
    monitor_states: list[ModeMonitorState],
    electric_field,
    magnetic_field,
    mode_solutions: tuple,
    *,
    time: float,
    step_index: int,
) -> None:
    """Record mode overlap values for all active mode monitors.

    This function computes the overlap between the current fields and
    the mode profiles and stores the values in the monitor state.

    Args:
        monitor_states: List of ModeMonitorState objects
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        mode_solutions: Tuple of ModeSolution objects for overlap computation
        time: Current simulation time
        step_index: Current step index
    """
    for state in monitor_states:
        compiled = state.compiled

        # Check if this step should be recorded
        if step_index < compiled.start:
            continue
        if (step_index - compiled.start) % compiled.interval != 0:
            continue

        # Set mode solutions if not already set
        if state.mode_solutions is None or len(state.mode_solutions) == 0:
            state.set_mode_solutions(mode_solutions)

        if compiled.is_time_domain:
            from autofdtd.kernels.monitors import record_mode_time_domain
            record_mode_time_domain(
                electric_field, magnetic_field, state, mode_solutions, time=time
            )
        elif compiled.is_frequency_domain:
            from autofdtd.kernels.monitors import record_mode_frequency_domain
            record_mode_frequency_domain(
                electric_field, magnetic_field, state, mode_solutions, time=time
            )


def extract_mode_monitor_data(
    monitor_states: list[ModeMonitorState],
    grid,
) -> dict[str, ModeData]:
    """Extract monitor data from mode monitor states.

    Args:
        monitor_states: List of ModeMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to ModeData objects
    """
    result = {}

    for state in monitor_states:
        compiled = state.compiled

        data_dict = state.to_mode_data()

        result[compiled.name] = ModeData(
            monitor_name=compiled.name,
            monitor_type=compiled.monitor_type,
            amplitudes=data_dict.get("amplitudes"),
            flux=None,  # ModeData flux is computed from amplitudes if needed
            t=data_dict.get("t"),
        )

    return result


# --------------------------------------------------------------------
# Projection monitor runtime
# --------------------------------------------------------------------


def build_projection_monitor_runtime(
    monitor,
    *,
    grid,
):
    """Build runtime objects for a projection monitor.

    Args:
        monitor: FieldProjectionAngleMonitor, FieldProjectionCartesianMonitor,
                 FieldProjectionKSpaceMonitor, DiffractionMonitor, or DirectivityMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledProjectionMonitor, ProjectionMonitorState)
    """
    from autofdtd.monitors import (
        FieldProjectionAngleMonitor,
        FieldProjectionCartesianMonitor,
        FieldProjectionKSpaceMonitor,
        DiffractionMonitor,
        DirectivityMonitor,
    )
    from autofdtd.compiler.monitors import (
        CompiledProjectionMonitor,
        ProjectionMonitorState,
        compile_projection_monitor,
    )

    monitor_type_map = {
        "FieldProjectionAngleMonitor": "FieldProjectionAngleMonitor",
        "FieldProjectionCartesianMonitor": "FieldProjectionCartesianMonitor",
        "FieldProjectionKSpaceMonitor": "FieldProjectionKSpaceMonitor",
        "DiffractionMonitor": "DiffractionMonitor",
        "DirectivityMonitor": "DirectivityMonitor",
    }

    if isinstance(monitor, FieldProjectionAngleMonitor):
        monitor_type = monitor_type_map["FieldProjectionAngleMonitor"]
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, FieldProjectionCartesianMonitor):
        monitor_type = monitor_type_map["FieldProjectionCartesianMonitor"]
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, FieldProjectionKSpaceMonitor):
        monitor_type = monitor_type_map["FieldProjectionKSpaceMonitor"]
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, DiffractionMonitor):
        monitor_type = monitor_type_map["DiffractionMonitor"]
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    elif isinstance(monitor, DirectivityMonitor):
        monitor_type = monitor_type_map["DirectivityMonitor"]
        num_freqs = monitor.num_freqs
        freqs = getattr(monitor, "freqs", ())
    else:
        raise TypeError(f"expected projection monitor, got {type(monitor)!r}")

    compiled = compile_projection_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=monitor.center,
        size=monitor.size,
        normal_vector=getattr(monitor, "normal_vector", (0.0, 0.0, 1.0)),
        projection_distance=getattr(monitor, "projection_distance", 1e5),
        interval=monitor.interval,
        start=monitor.start,
        num_freqs=num_freqs,
        freqs=freqs,
        phi=getattr(monitor, "phi", (-90.0, 90.0, 181)),
        theta=getattr(monitor, "theta", (0.0, 180.0, 181)),
        x=getattr(monitor, "x", (-50.0, 50.0, 201)),
        y=getattr(monitor, "y", (-50.0, 50.0, 201)),
        num_k=getattr(monitor, "num_k", 1),
        kx=getattr(monitor, "kx", (-10.0, 10.0, 21)),
        ky=getattr(monitor, "ky", (-10.0, 10.0, 21)),
        grid=grid,
    )

    state = ProjectionMonitorState(compiled=compiled)

    return compiled, state


def record_projection_monitor_dft(
    monitor_states: list,
    electric_field,
    magnetic_field,
    *,
    time: float,
    step_index: int,
) -> None:
    """Record DFT terms for all active projection monitors.

    Args:
        monitor_states: List of ProjectionMonitorState objects
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        time: Current simulation time
        step_index: Current step index
    """
    for state in monitor_states:
        compiled = state.compiled

        # Check if this step should be recorded
        if step_index < compiled.start:
            continue
        if (step_index - compiled.start) % compiled.interval != 0:
            continue

        state.accumulate_dft(electric_field, magnetic_field, time)


def extract_projection_monitor_data(
    monitor_states: list,
    grid,
) -> dict:
    """Extract monitor data from projection monitor states.

    Args:
        monitor_states: List of ProjectionMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to projection monitor data objects
    """
    from autofdtd.monitors import (
        FieldProjectionAngleData,
        FieldProjectionCartesianData,
        FieldProjectionKSpaceData,
        DiffractionData,
        DirectivityData,
    )

    result = {}

    for state in monitor_states:
        compiled = state.compiled
        data_dict = state.to_projection_data()

        if compiled.monitor_type == "FieldProjectionAngleMonitor":
            # Compute phi and theta arrays from specification
            phi_min, phi_max, phi_count = compiled.phi
            theta_min, theta_max, theta_count = compiled.theta
            phi_vals = tuple(phi_min + i * (phi_max - phi_min) / (phi_count - 1) for i in range(phi_count))
            theta_vals = tuple(theta_min + i * (theta_max - theta_min) / (theta_count - 1) for i in range(theta_count))

            result[compiled.name] = FieldProjectionAngleData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                e_theta=None,  # Phase 1 deferred
                e_phi=None,    # Phase 1 deferred
                phi=phi_vals,
                theta=theta_vals,
                t=(),
            )
        elif compiled.monitor_type == "FieldProjectionCartesianMonitor":
            x_min, x_max, x_count = compiled.x
            y_min, y_max, y_count = compiled.y
            x_vals = tuple(x_min + i * (x_max - x_min) / (x_count - 1) for i in range(x_count))
            y_vals = tuple(y_min + i * (y_max - y_min) / (y_count - 1) for i in range(y_count))

            result[compiled.name] = FieldProjectionCartesianData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                ex=None,  # Phase 1 deferred
                ey=None,
                ez=None,
                x=x_vals,
                y=y_vals,
                t=(),
            )
        elif compiled.monitor_type == "FieldProjectionKSpaceMonitor":
            kx_min, kx_max, kx_count = compiled.kx
            ky_min, ky_max, ky_count = compiled.ky
            kx_vals = tuple(kx_min + i * (kx_max - kx_min) / (kx_count - 1) for i in range(kx_count))
            ky_vals = tuple(ky_min + i * (ky_max - ky_min) / (ky_count - 1) for i in range(ky_count))

            result[compiled.name] = FieldProjectionKSpaceData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                ex=None,  # Phase 1 deferred
                ey=None,
                ez=None,
                kx=kx_vals,
                ky=ky_vals,
                t=(),
            )
        elif compiled.monitor_type == "DiffractionMonitor":
            # Compute diffraction orders from DFT fields
            orders_data = None
            mx_vals = None
            my_vals = None

            # Get grid cell sizes for estimating grating period
            # The period is estimated from the grid spacing along tangential directions
            normal_axis = compiled.normal_axis
            tang1 = 1 if normal_axis == 0 else 0 if normal_axis == 1 else 0
            tang2 = 2 if normal_axis == 2 else 1 if normal_axis == 1 else 2
            if normal_axis == 0:
                tang1, tang2 = 1, 2
            elif normal_axis == 1:
                tang1, tang2 = 0, 2
            else:
                tang1, tang2 = 0, 1

            grid_boundaries = (grid.x.boundaries, grid.y.boundaries, grid.z.boundaries)
            period_tang1 = (grid_boundaries[tang1][1] - grid_boundaries[tang1][0]) / len(grid_boundaries[tang1])
            period_tang2 = (grid_boundaries[tang2][1] - grid_boundaries[tang2][0]) / len(grid_boundaries[tang2])

            # Import here to avoid circular imports
            from autofdtd.kernels.monitors import compute_diffraction_orders

            if state.dft_e is not None and len(state.dft_e) > 0:
                orders_arr, mx_vals, my_vals = compute_diffraction_orders(
                    dft_e=state.dft_e,
                    dft_h=state.dft_h,
                    placements=compiled.placements,
                    normal_axis=normal_axis,
                    freqs=compiled.freqs,
                    period_y=period_tang1,
                    period_z=period_tang2,
                    medium_eps=1.0 + 0.0j,  # Assume vacuum for Phase 1
                    num_orders=5,
                )
                # Convert to JSON-safe tuple format
                orders_data = tuple(
                    tuple(
                        tuple(float(o.real), float(o.imag))
                        for o in orders_arr[i]
                    )
                    for i in range(len(orders_arr))
                )

            result[compiled.name] = DiffractionData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                orders=orders_data,
                mx=mx_vals,
                my=my_vals,
                t=(),
            )
        elif compiled.monitor_type == "DirectivityMonitor":
            # Build angular grid
            phi_min, phi_max, phi_count = compiled.phi
            theta_min, theta_max, theta_count = compiled.theta
            phi_vals = tuple(phi_min + i * (phi_max - phi_min) / (phi_count - 1) for i in range(phi_count))
            theta_vals = tuple(theta_min + i * (theta_max - theta_min) / (theta_count - 1) for i in range(theta_count))

            result[compiled.name] = DirectivityData(
                monitor_name=compiled.name,
                monitor_type=compiled.monitor_type,
                directivity=None,  # Phase 1 deferred
                theta=theta_vals,
                phi=phi_vals,
                t=(),
            )

    return result


# --------------------------------------------------------------------
# Surface field monitor runtime
# --------------------------------------------------------------------


def build_surface_field_monitor_runtime(
    monitor,
    *,
    grid,
) -> tuple:
    """Build runtime objects for a surface field monitor.

    Args:
        monitor: SurfaceFieldMonitor or SurfaceFieldTimeMonitor
        grid: Resolved simulation grid

    Returns:
        Tuple of (CompiledSurfaceFieldMonitor, SurfaceFieldMonitorState)
    """
    from autofdtd.monitors import SurfaceFieldMonitor, SurfaceFieldTimeMonitor
    from autofdtd.compiler.monitors import (
        CompiledSurfaceFieldMonitor,
        SurfaceFieldMonitorState,
        compile_surface_field_monitor,
    )

    if isinstance(monitor, (SurfaceFieldMonitor, SurfaceFieldTimeMonitor)):
        monitor_type = monitor.type
        center = monitor.center
        size = monitor.size
        fields = monitor.fields
        interval = monitor.interval
        start = monitor.start
        freqs = getattr(monitor, "freqs", ())
    else:
        raise TypeError(f"expected surface field monitor, got {type(monitor)!r}")

    compiled = compile_surface_field_monitor(
        name=monitor.name,
        monitor_type=monitor_type,
        center=center,
        size=size,
        fields=fields,
        interval=interval,
        start=start,
        freqs=freqs,
        grid=grid,
    )

    state = SurfaceFieldMonitorState(compiled=compiled)

    return compiled, state


def record_surface_monitor_fields(
    monitor_states: list,
    electric_field,
    magnetic_field,
    *,
    time: float,
    step_index: int,
) -> None:
    """Record field values for all active surface field monitors.

    This function samples the fields at each monitor's surface cells
    and stores the values in the monitor's state for later retrieval.

    Args:
        monitor_states: List of SurfaceFieldMonitorState objects
        electric_field: The E field buffer with shape (nx, ny, nz, 3)
        magnetic_field: The H field buffer with shape (nx, ny, nz, 3)
        time: Current simulation time
        step_index: Current step index
    """
    for state in monitor_states:
        compiled = state.compiled

        # Check if this step should be recorded
        if step_index < compiled.start:
            continue
        if (step_index - compiled.start) % compiled.interval != 0:
            continue

        if compiled.is_time_domain:
            state.record_time_domain(electric_field, magnetic_field, time)
        elif compiled.is_frequency_domain:
            state.accumulate_dft(electric_field, magnetic_field, time)


def extract_surface_field_monitor_data(
    monitor_states: list,
    grid,
) -> dict:
    """Extract monitor data from surface field monitor states.

    Args:
        monitor_states: List of SurfaceFieldMonitorState objects
        grid: Resolved simulation grid (for coordinate info)

    Returns:
        Dictionary mapping monitor names to SurfaceFieldData or SurfaceFieldTimeData objects
    """
    from autofdtd.monitors import SurfaceFieldData, SurfaceFieldTimeData

    result = {}

    for state in monitor_states:
        compiled = state.compiled

        field_data_dict = state.to_surface_field_data()

        # Get grid coordinates for the surface
        # Extract coordinates along tangential axes
        x_boundaries = grid.x.boundaries
        y_boundaries = grid.y.boundaries
        z_boundaries = grid.z.boundaries

        # Get coordinate values for the surface
        x_coords = tuple(
            0.5 * (x_boundaries[i] + x_boundaries[i + 1])
            for i in range(len(x_boundaries) - 1)
        )
        y_coords = tuple(
            0.5 * (y_boundaries[i] + y_boundaries[i + 1])
            for i in range(len(y_boundaries) - 1)
        )
        z_coords = tuple(
            0.5 * (z_boundaries[i] + z_boundaries[i + 1])
            for i in range(len(z_boundaries) - 1)
        )

        # Build data with stored values
        if compiled.is_time_domain:
            time_stamps, _ = state.time_data()
            fd_kwargs = {
                "monitor_name": compiled.name,
                "monitor_type": compiled.monitor_type,
                "t": time_stamps,
            }
        else:
            fd_kwargs = {
                "monitor_name": compiled.name,
                "monitor_type": compiled.monitor_type,
                "t": (),
            }

        for field in compiled.fields:
            if field in field_data_dict:
                fd_kwargs[field] = field_data_dict[field]

        # Add coordinate info based on normal axis
        # For a surface, we provide coordinates along the tangential directions
        if compiled.normal_axis == 0:
            # x-normal surface: tangential are y and z
            fd_kwargs["y"] = y_coords
            fd_kwargs["z"] = z_coords
        elif compiled.normal_axis == 1:
            # y-normal surface: tangential are x and z
            fd_kwargs["x"] = x_coords
            fd_kwargs["z"] = z_coords
        else:
            # z-normal surface: tangential are x and y
            fd_kwargs["x"] = x_coords
            fd_kwargs["y"] = y_coords

        if compiled.is_time_domain:
            result[compiled.name] = SurfaceFieldTimeData(**fd_kwargs)
        else:
            result[compiled.name] = SurfaceFieldData(**fd_kwargs)

    return result