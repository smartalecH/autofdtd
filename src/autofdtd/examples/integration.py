"""Integration example: public API → normalization → IR → compilation → execution.

This example demonstrates the complete Phase 1 pipeline:
1. Define a Tidy3D-style simulation using the public API
2. Normalize and validate the simulation
3. Lower to tagged execution IR
4. Inspect or serialize the IR
5. Compile to runtime artifacts
6. Execute through backend entrypoints

The key architectural point: execution is driven by the IR, not the original
Python object graph. The CompiledSimulation contains only the IR snapshot and
the concrete compiled artifacts (coefficients, placements, chunk metadata).
"""

from __future__ import annotations

import json

from autofdtd.api import (
    Box,
    Boundary,
    BoundarySpec,
    compile_simulation,
    ContinuousWave,
    GaussianPulse,
    GridSpec,
    Medium,
    PML,
    PECBoundary,
    PMCBoundary,
    Scene,
    Simulation,
    simulation_to_execution_package,
    simulation_to_ir,
    Sphere,
    Structure,
    UniformCurrentSource,
    UniformGrid,
)


def define_tidy3d_style_simulation() -> Simulation:
    """Define a realistic Tidy3D-style simulation with structures, sources, monitors.

    This creates a simulation with:
    - A dielectric slab waveguide
    - A PEC boundary on one side
    - A Gaussian pulse source
    - Field monitors
    """
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(10.0, 4.0, 4.0),
        run_time=1e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=0.05),
            grid_y=UniformGrid(dl=0.05),
            grid_z=UniformGrid(dl=0.05),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            Structure(
                geometry=Box(
                    center=(0.0, 0.0, 0.0),
                    size=(8.0, 1.0, 0.5),
                ),
                medium=Medium(permittivity=3.45),
                name="slab",
            ),
            Structure(
                geometry=Sphere(
                    center=(2.0, 0.0, 0.0),
                    radius=0.25,
                ),
                medium=Medium(permittivity=2.0),
                name="bead",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(-3.0, 0.0, 0.0),
                size=(0.0, 1.0, 0.5),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=2.5,
                ),
                name="slab_drive",
            ),
        ),
        monitors=(),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
            y=Boundary(plus=PMCBoundary(), minus=PECBoundary()),
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )


def normalize_and_lower_to_ir() -> dict:
    """Step 1-2: Normalize and lower simulation to IR."""
    sim = define_tidy3d_style_simulation()

    # Normalization happens during Simulation construction via validators.
    # The result is a validated, normalized Simulation object.

    # Lower to IR - this produces a tagged, versioned IR snapshot
    sim_ir = simulation_to_ir(sim)

    return {
        "ir_type": sim_ir.type,
        "ir_schema_version": sim_ir.schema_version,
        "center": sim_ir.center,
        "size": sim_ir.size,
        "run_time": sim_ir.run_time,
        "num_sources": len(sim_ir.sources),
        "num_monitors": len(sim_ir.monitors),
        "has_boundary_spec": sim_ir.boundary_spec is not None,
        "has_grid_spec": sim_ir.grid_spec is not None,
        "has_runtime_controls": sim_ir.runtime_controls is not None,
    }


def inspect_ir_serialization() -> dict:
    """Step 3: Inspect and serialize the IR."""
    sim = define_tidy3d_style_simulation()
    sim_ir = simulation_to_ir(sim)

    # IR is a frozen, tagged model with stable JSON serialization
    ir_json = sim_ir.to_json_text()

    # Parse it back to verify round-trip
    parsed = json.loads(ir_json)

    return {
        "json_length": len(ir_json),
        "parsed_type": parsed.get("type"),
        "parsed_schema_version": parsed.get("schema_version"),
        "parsed_center": parsed.get("center"),
        "parsed_size": parsed.get("size"),
    }


def compile_and_inspect() -> dict:
    """Step 4: Compile simulation to runtime artifacts."""
    sim = define_tidy3d_style_simulation()

    # compile_simulation is the main entry point for the compilation pipeline
    compiled = compile_simulation(sim)

    return {
        "compiled_type": type(compiled).__name__,
        "grid_shape": compiled.grid_shape,
        "total_cells": compiled.total_cells,
        "num_sources": compiled.num_sources,
        "num_monitors": compiled.num_monitors,
        "has_chunk_layout": compiled.chunk_layout is not None,
        "chunk_count": len(compiled.chunk_layout.chunks),
        "runtime_dt": compiled.runtime_controls.dt,
        "num_time_steps": compiled.runtime_controls.num_time_steps,
        "has_scene_coefficients": compiled.scene_coefficients is not None,
        "scene_background_type": compiled.scene_coefficients.background.medium_type
            if compiled.scene_coefficients else None,
        "num_structures": len(compiled.scene_coefficients.structures)
            if compiled.scene_coefficients else 0,
        "ir_schema_version": compiled.simulation_ir.schema_version,
    }


def execution_package_demo() -> dict:
    """Step 5: Create a transportable execution package."""
    sim = define_tidy3d_style_simulation()

    # simulation_to_execution_package bundles the IR into a transportable format
    package = simulation_to_execution_package(
        sim,
        metadata={"author": "phase1", "purpose": "integration_demo"},
    )

    return {
        "package_format": package.manifest.package_format,
        "package_version": package.manifest.autofdtd_version,
        "entrypoint": package.manifest.entrypoint,
        "has_artifacts": len(package.manifest.artifacts) > 0,
        "simulation_type": package.simulation.type,
        "simulation_schema_version": package.simulation.schema_version,
    }


def full_pipeline_snapshot() -> dict[str, object]:
    """Return a compact snapshot of the complete API → IR → execution pipeline."""
    return {
        "normalization_and_ir": normalize_and_lower_to_ir(),
        "ir_serialization": inspect_ir_serialization(),
        "compilation": compile_and_inspect(),
        "execution_package": execution_package_demo(),
    }


if __name__ == "__main__":
    # Run the full pipeline demonstration
    result = full_pipeline_snapshot()

    print("=== API → IR → Execution Pipeline ===")
    print(f"\n1. Normalization + IR Lowering:")
    for k, v in result["normalization_and_ir"].items():
        print(f"   {k}: {v}")

    print(f"\n2. IR Serialization:")
    for k, v in result["ir_serialization"].items():
        print(f"   {k}: {v}")

    print(f"\n3. Compilation:")
    for k, v in result["compilation"].items():
        print(f"   {k}: {v}")

    print(f"\n4. Execution Package:")
    for k, v in result["execution_package"].items():
        print(f"   {k}: {v}")

    print("\n=== Key Architectural Points ===")
    print("- Execution is driven by CompiledSimulation (IR snapshot + artifacts)")
    print("- The IR is a frozen, tagged, versioned model")
    print("- Compilation pipeline: Simulation → IR → CompiledSimulation")
    print("- CompiledSimulation contains no Python object graph references")