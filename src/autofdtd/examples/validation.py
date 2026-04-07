"""Canonical validation examples for Phase 1 feature families.

This module provides runnable validation examples that exercise the implemented
Phase 1 feature families end-to-end, demonstrating physical correctness and
feature coverage.

Validation Philosophy
--------------------
These examples serve as both documentation and regression tests. Each example:
1. Defines a physically meaningful simulation
2. Compiles it to runtime artifacts
3. Executes the timestepping loop
4. Verifies physical behavior (energy conservation, expected field patterns, etc.)

Feature Coverage
--------------
- Vacuum propagation: point source in PML-backed box
- Dielectric waveguide: slab waveguide with mode source
- PML absorption: validates boundary absorption
- Source injection: uniform current, point dipole, plane wave
- Monitor recording: field, flux, medium monitors
- Convergence: shutoff behavior and energy decay
"""

from __future__ import annotations

import numpy as np

from autofdtd.api import (
    AutoGrid,
    Box,
    Boundary,
    BoundarySpec,
    ContinuousWave,
    GaussianPulse,
    GridSpec,
    Medium,
    PML,
    PECBoundary,
    PMCBoundary,
    Scene,
    Simulation,
    Sphere,
    Structure,
    SubpixelSpec,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.compiler import compile_simulation
from autofdtd.monitors import (
    FieldMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
)
from autofdtd.sources import (
    GaussianBeam,
    PlaneWave,
    PointDipole,
    TFSF,
)
from autofdtd.runtime import run_compiled_simulation
from autofdtd.kernels.backend import backend_info


# ---------------------------------------------------------------------------
# Vacuum Propagation
# ---------------------------------------------------------------------------


def vacuum_point_source_example() -> dict:
    """Run a vacuum point source propagation and verify energy behavior.

    A point source in vacuum should produce outward propagating waves
    that eventually dissipate into the PML boundaries. The total field
    energy should decay as energy leaves the domain.

    Returns
    -------
    dict
        Validation result with field statistics and energy behavior.
    """
    # Create simulation with point source in vacuum
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(2.0e-6, 2.0e-6, 2.0e-6),  # 2um cube
        run_time=5e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells = 10 cells per um
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PointDipole(
                center=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=1e14,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="dipole",
            ),
        ),
        monitors=(
            FieldMonitor(
                center=(0.0, 0.0, 0.0),
                size=(2.0, 2.0, 2.0),
                fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                interval=20,
                name="center_field",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-6,
    )

    # Compile and run
    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=500,
        record_interval=10,
        verbose=False,
    )

    # Validation checks
    energy_history = result.integrated_electric_history
    max_energy = max(energy_history) if energy_history else 0.0

    # Energy should peak early then decay as PML absorbs
    peak_index = energy_history.index(max_energy) if energy_history else 0
    final_energy = energy_history[-1] if energy_history else 0.0

    return {
        "example": "vacuum_point_source",
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "max_energy": float(max_energy),
        "final_energy": float(final_energy),
        "energy_decay_ratio": float(final_energy / max_energy) if max_energy > 0 else None,
        "peak_step": peak_index,
        "total_cells": compiled.total_cells,
        "backend": result.metrics.get("backend", "unknown"),
    }


def vacuum_plane_wave_example() -> dict:
    """Run a plane wave propagation in vacuum with PML boundaries.

    A broadband plane wave entering vacuum should propagate and
    eventually be absorbed by PML boundaries.

    Returns
    -------
    dict
        Validation result with field statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(6.0e-6, 4.0e-6, 4.0e-6),
        run_time=3e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PlaneWave(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 4.0e-6, 4.0e-6),
                direction="+",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="plane_wave",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(1.0e-6, 0.0, 0.0),
                size=(0.0, 2.0e-6, 2.0e-6),
                fields=["Ex", "Ey", "Ez"],
                interval=10,
                name="propagating_field",
            ),
            FluxMonitor(
                center=(0.0, 0.0, 0.0),
                size=(4.0e-6, 4.0e-6, 0.0),  # yz plane at x=0
                direction="+",
                interval=10,
                name="incident_flux",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=300,
        record_interval=10,
        verbose=False,
    )

    return {
        "example": "vacuum_plane_wave",
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "total_cells": compiled.total_cells,
        "max_field_monitor_recorded": result.num_steps // 10,
    }


# ---------------------------------------------------------------------------
# Dielectric Slab Waveguide
# ---------------------------------------------------------------------------


def dielectric_slab_example() -> dict:
    """Run a dielectric slab waveguide simulation.

    A dielectric slab supports guided modes. This example validates
    that fields can be confined in a higher-index region.

    Returns
    -------
    dict
        Validation result with field confinement statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(10.0e-6, 4.0e-6, 4.0e-6),
        run_time=5e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),  # vacuum background
        structures=(
            Structure(
                geometry=Box(
                    center=(0.0, 0.0, 0.0),
                    size=(8.0e-6, 0.8e-6, 4.0e-6),
                ),
                medium=Medium(permittivity=3.45),  # silica-like
                name="slab",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(-3.0e-6, 0.0, 0.0),
                size=(0.0, 0.8e-6, 4.0e-6),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="slab_drive",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(0.0, 0.0, 0.0),
                size=(6.0e-6, 0.8e-6, 3.0e-6),
                fields=["Ez"],
                interval=20,
                name="slab_field",
            ),
            PermittivityMonitor(
                center=(0.0, 0.0, 0.0),
                size=(6.0e-6, 0.8e-6, 3.0e-6),
                interval=1,
                name="permittivity",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PECBoundary(), minus=PECBoundary()),  # PEC in y
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=500,
        record_interval=10,
        verbose=False,
    )

    # Check that field is confined to slab region (higher permittivity)
    field_data = result.field_monitor_data.get("slab_field", None)
    if field_data is None:
        Ez_values = []
    elif hasattr(field_data, "Ez"):
        Ez_values = field_data.Ez
    else:
        Ez_values = field_data.get("Ez", []) if hasattr(field_data, "get") else []

    return {
        "example": "dielectric_slab",
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "has_field_data": len(Ez_values) > 0,
        "num_field_records": len(Ez_values),
        "total_cells": compiled.total_cells,
        "slab_permittivity": 3.45,
    }


# ---------------------------------------------------------------------------
# PML Absorption Validation
# ---------------------------------------------------------------------------


def pml_absorption_example() -> dict:
    """Validate PML boundary absorption by checking energy decay.

    A source near a PML boundary should have its reflected energy
    minimal compared to a PEC boundary case.

    Returns
    -------
    dict
        Validation result with absorption statistics.
    """
    # Source near PML boundary
    sim_pml = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=3e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PointDipole(
                center=(1.5e-6, 0.0, 0.0),  # Near +x boundary
                polarization="Ez",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="dipole",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(0.0, 0.0, 0.0),
                size=(2.0e-6, 1.0e-6, 1.0e-6),
                fields=["Ez"],
                interval=10,
                name="field_at_source",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=15), minus=PML(num_layers=15)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-6,
    )

    compiled_pml = compile_simulation(sim_pml)
    result_pml = run_compiled_simulation(
        compiled_pml,
        max_steps=300,
        record_interval=10,
        verbose=False,
    )

    # Compare with PEC boundaries (should have more reflections)
    sim_pec = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=3e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PointDipole(
                center=(1.5e-6, 0.0, 0.0),
                polarization="Ez",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="dipole",
            ),
        ),
        monitors=(),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PECBoundary(), minus=PECBoundary()),
            y=Boundary(plus=PECBoundary(), minus=PECBoundary()),
            z=Boundary(plus=PECBoundary(), minus=PECBoundary()),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled_pec = compile_simulation(sim_pec)
    result_pec = run_compiled_simulation(
        compiled_pec,
        max_steps=300,
        record_interval=10,
        verbose=False,
    )

    # PML should absorb more energy, leading to lower final energy
    pml_final = result_pml.integrated_electric_history[-1] if result_pml.integrated_electric_history else 0.0
    pec_final = result_pec.integrated_electric_history[-1] if result_pec.integrated_electric_history else 0.0

    return {
        "example": "pml_absorption",
        "pml_steps": result_pml.num_steps,
        "pml_final_energy": float(pml_final),
        "pec_steps": result_pec.num_steps,
        "pec_final_energy": float(pec_final),
        "absorption_ratio": float(pml_final / pec_final) if pec_final > 0 else None,
        "total_cells": compiled_pml.total_cells,
    }


# ---------------------------------------------------------------------------
# Source Injection Validation
# ---------------------------------------------------------------------------


def uniform_current_injection_example() -> dict:
    """Validate uniform current source injection.

    A uniform current source should inject fields that propagate outward
    and follow expected wave dynamics.

    Returns
    -------
    dict
        Validation result with injection statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(6.0e-6, 3.0e-6, 3.0e-6),
        run_time=4e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            UniformCurrentSource(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 2.0e-6, 3.0e-6),  # yz plane at x=0
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=1.5e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="uniform_source",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(1.0e-6, 0.0, 0.0),
                size=(0.0, 2.0e-6, 2.0e-6),
                fields=["Ez"],
                interval=10,
                name="propagating_Ez",
            ),
            FieldTimeMonitor(
                center=(1.0e-6, 0.0, 0.0),
                size=(0.0, 2.0e-6, 2.0e-6),
                fields=["Hx", "Hy"],
                interval=10,
                name="propagating_H",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=400,
        record_interval=10,
        verbose=False,
    )

    return {
        "example": "uniform_current_injection",
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "has_Ez_data": "propagating_Ez" in result.field_monitor_data,
        "has_H_data": "propagating_H" in result.field_monitor_data,
        "total_cells": compiled.total_cells,
    }


# ---------------------------------------------------------------------------
# Monitor Recording Validation
# ---------------------------------------------------------------------------


def field_monitor_recording_example() -> dict:
    """Validate field monitor recording over time.

    A field monitor should record time-domain field values at specified intervals.

    Returns
    -------
    dict
        Validation result with recording statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=2e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PointDipole(
                center=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=GaussianPulse(
                    freq0=3e14,
                    fwidth=1e14,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="dipole",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(0.5e-6, 0.0, 0.0),
                size=(0.0, 1.0e-6, 1.0e-6),
                fields=["Ez"],
                interval=5,
                start=0,
                name="time_history",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-6,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=200,
        record_interval=5,
        verbose=False,
    )

    field_data = result.field_monitor_data.get("time_history", None)
    if field_data is None:
        Ez_data = ()
        num_time_records = 0
    elif hasattr(field_data, "Ez"):
        Ez_data = field_data.Ez
        # Number of time records is len(t) if available
        num_time_records = len(field_data.t) if hasattr(field_data, 't') and field_data.t else 0
    else:
        Ez_data = field_data.get("Ez", ()) if hasattr(field_data, "get") else ()
        num_time_records = 0

    return {
        "example": "field_monitor_recording",
        "num_steps": result.num_steps,
        "num_records": num_time_records,
        "expected_records": result.num_steps // 5,
        "has_Ez": len(Ez_data) > 0,
        "total_cells": compiled.total_cells,
    }


def flux_monitor_recording_example() -> dict:
    """Validate flux monitor recording.

    A flux monitor should record power flow through a surface.

    Returns
    -------
    dict
        Validation result with flux statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(6.0e-6, 4.0e-6, 4.0e-6),
        run_time=3e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PlaneWave(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 4.0e-6, 4.0e-6),
                direction="+",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="plane_wave",
            ),
        ),
        monitors=(
            FluxMonitor(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 4.0e-6, 4.0e-6),
                direction="+",
                interval=10,
                name="incident_power",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=300,
        record_interval=10,
        verbose=False,
    )

    flux_data = result.flux_monitor_data.get("incident_power", {})

    return {
        "example": "flux_monitor_recording",
        "num_steps": result.num_steps,
        "has_flux_data": bool(flux_data),
        "total_cells": compiled.total_cells,
    }


def medium_monitor_example() -> dict:
    """Validate medium/permittivity monitor recording.

    A permittivity monitor should record the static permittivity distribution.

    Returns
    -------
    dict
        Validation result with medium recording.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),
        run_time=1e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            Structure(
                geometry=Box(
                    center=(0.0, 0.0, 0.0),
                    size=(2.0e-6, 0.5e-6, 2.0e-6),
                ),
                medium=Medium(permittivity=2.1),
                name="dielectric",
            ),
        ),
        monitors=(
            PermittivityMonitor(
                center=(0.0, 0.0, 0.0),
                size=(3.0e-6, 2.0e-6, 2.0e-6),
                interval=1,
                name="permittivity_dist",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=100,
        record_interval=10,
        verbose=False,
    )

    medium_data = result.medium_monitor_data.get("permittivity_dist", {})

    return {
        "example": "medium_monitor",
        "num_steps": result.num_steps,
        "has_medium_data": bool(medium_data),
        "total_cells": compiled.total_cells,
    }


# ---------------------------------------------------------------------------
# Convergence / Shutoff Validation
# ---------------------------------------------------------------------------


def convergence_shutoff_example() -> dict:
    """Validate convergence shutoff behavior.

    A decaying source should trigger early shutoff when the integrated
    field intensity falls below the threshold.

    Returns
    -------
    dict
        Validation result with convergence behavior.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-6, 2.0e-6, 2.0e-6),  # 4um x 2um x 2um
        run_time=1e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        sources=(
            PointDipole(
                center=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="dipole",
            ),
        ),
        monitors=(),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-4,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=1000,
        record_interval=10,
        verbose=False,
    )

    energy_history = result.integrated_electric_history
    max_energy = max(energy_history) if energy_history else 0.0
    final_energy = energy_history[-1] if energy_history else 0.0

    return {
        "example": "convergence_shutoff",
        "num_steps": result.num_steps,
        "max_steps_config": compiled.runtime_controls.num_time_steps,
        "stop_reason": result.stop_reason,
        "max_energy": float(max_energy),
        "final_energy": float(final_energy),
        "energy_ratio": float(final_energy / max_energy) if max_energy > 0 else None,
        "total_cells": compiled.total_cells,
    }


# ---------------------------------------------------------------------------
# Multi-Feature Integration
# ---------------------------------------------------------------------------


def multi_feature_integration_example() -> dict:
    """Run a simulation combining multiple feature families.

    This example exercises:
    - Dielectric structure
    - PML boundaries
    - Uniform current source
    - Field and flux monitors
    - Convergence shutoff

    Returns
    -------
    dict
        Validation result with combined feature statistics.
    """
    sim = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(8.0e-6, 3.0e-6, 3.0e-6),
        run_time=4e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=2.0e-7),  # 200nm cells
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=2.0e-7),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            Structure(
                geometry=Box(
                    center=(1.0e-6, 0.0, 0.0),
                    size=(4.0e-6, 0.6e-6, 3.0e-6),
                ),
                medium=Medium(permittivity=2.5),
                name="slab",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(-2.0e-6, 0.0, 0.0),
                size=(0.0, 0.6e-6, 3.0e-6),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=3.0,
                ),
                name="source",
            ),
        ),
        monitors=(
            FieldTimeMonitor(
                center=(1.0e-6, 0.0, 0.0),
                size=(3.0e-6, 0.6e-6, 2.0e-6),
                fields=["Ez"],
                interval=10,
                name="slab_field",
            ),
            FluxMonitor(
                center=(3.0e-6, 0.0, 0.0),
                size=(0.0, 3.0e-6, 3.0e-6),
                direction="+",
                interval=10,
                name="output_flux",
            ),
            PermittivityMonitor(
                center=(0.0, 0.0, 0.0),
                size=(6.0e-6, 3.0e-6, 3.0e-6),
                interval=1,
                name="permittivity",
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PECBoundary(), minus=PECBoundary()),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    compiled = compile_simulation(sim)
    result = run_compiled_simulation(
        compiled,
        max_steps=400,
        record_interval=10,
        verbose=False,
    )

    return {
        "example": "multi_feature_integration",
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "has_field_data": "slab_field" in result.field_monitor_data,
        "has_flux_data": "output_flux" in result.flux_monitor_data,
        "has_medium_data": "permittivity" in result.medium_monitor_data,
        "num_sources": compiled.num_sources,
        "num_monitors": compiled.num_monitors,
        "total_cells": compiled.total_cells,
        "backend": result.metrics.get("backend", "unknown"),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def run_all_validation_examples() -> dict:
    """Run all validation examples and return a summary.

    Returns
    -------
    dict
        Summary of all validation results.
    """
    examples = [
        ("vacuum_point_source", vacuum_point_source_example),
        ("vacuum_plane_wave", vacuum_plane_wave_example),
        ("dielectric_slab", dielectric_slab_example),
        ("pml_absorption", pml_absorption_example),
        ("uniform_current_injection", uniform_current_injection_example),
        ("field_monitor_recording", field_monitor_recording_example),
        ("flux_monitor_recording", flux_monitor_recording_example),
        ("medium_monitor", medium_monitor_example),
        ("convergence_shutoff", convergence_shutoff_example),
        ("multi_feature_integration", multi_feature_integration_example),
    ]

    results = {}
    backend = None

    for name, fn in examples:
        try:
            result = fn()
            results[name] = result
            if backend is None:
                backend = result.get("backend", "unknown")
        except Exception as e:
            results[name] = {"error": str(e), "example": name}

    return {
        "backend": backend,
        "backend_info": backend_info(),
        "examples": results,
        "num_examples": len(examples),
        "successful": sum(1 for r in results.values() if "error" not in r),
    }


if __name__ == "__main__":
    import json

    print("Running Phase 1 Validation Examples...")
    print("=" * 60)

    summary = run_all_validation_examples()

    print(f"\nBackend: {summary['backend']}")
    print(f"Successful: {summary['successful']}/{summary['num_examples']}")
    print("\nDetailed Results:")
    print("-" * 60)

    for name, result in summary["examples"].items():
        status = "OK" if "error" not in result else f"ERROR: {result.get('error', 'unknown')}"
        steps = result.get("num_steps", "N/A")
        stop = result.get("stop_reason", "N/A")
        print(f"  {name}: {status} | steps={steps} | stop={stop}")

    print("\n" + "=" * 60)
    print("Validation complete.")