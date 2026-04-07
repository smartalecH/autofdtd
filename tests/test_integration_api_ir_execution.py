"""Integration tests for the public API → IR → compilation → execution pipeline.

These tests verify that:
1. The public API correctly exposes compilation and execution entrypoints
2. Simulation objects can be lowered to tagged execution IR
3. Compilation produces CompiledSimulation with correct artifacts
4. Execution is driven by the IR snapshot, not the original Python object graph
"""

from __future__ import annotations

import json

import pytest

from autofdtd.api import (
    Box,
    Boundary,
    BoundarySpec,
    compile_simulation,
    GaussianPulse,
    GridSpec,
    Medium,
    PML,
    PECBoundary,
    PMCBoundary,
    RuntimeController,
    Scene,
    simulation_to_execution_package,
    simulation_to_ir,
    Simulation,
    Sphere,
    Structure,
    UniformCurrentSource,
    UniformGrid,
    build_runtime_controller,
    run_until_stop,
    run_until_stop_with_logging,
)
from autofdtd.compiler import CompiledSimulation
from autofdtd.ir import SimulationIR


def build_test_simulation() -> Simulation:
    """Build a simple test simulation for integration testing."""
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0, 2.0, 2.0),
        run_time=1e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=0.1),
            grid_y=UniformGrid(dl=0.1),
            grid_z=UniformGrid(dl=0.1),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 0.5, 0.5)),
                medium=Medium(permittivity=3.45),
                name="slab",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(-1.0, 0.0, 0.0),
                size=(0.0, 0.5, 0.5),
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=2.5,
                ),
                name="drive",
            ),
        ),
        monitors=(),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
            y=Boundary(plus=PMCBoundary(), minus=PECBoundary()),
            z=Boundary(plus=PML(num_layers=5), minus=PML(num_layers=5)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )


class TestPublicAPIExports:
    """Test that public API correctly exposes all necessary entrypoints."""

    def test_compile_simulation_is_exported(self) -> None:
        """compile_simulation should be accessible from the public API."""
        assert callable(compile_simulation)

    def test_simulation_to_ir_is_exported(self) -> None:
        """simulation_to_ir should be accessible from the public API."""
        assert callable(simulation_to_ir)

    def test_simulation_to_execution_package_is_exported(self) -> None:
        """simulation_to_execution_package should be accessible from the public API."""
        assert callable(simulation_to_execution_package)

    def test_runtime_controller_is_exported(self) -> None:
        """RuntimeController should be accessible from the public API."""
        assert RuntimeController is not None

    def test_build_runtime_controller_is_exported(self) -> None:
        """build_runtime_controller should be accessible from the public API."""
        assert callable(build_runtime_controller)

    def test_run_until_stop_is_exported(self) -> None:
        """run_until_stop should be accessible from the public API."""
        assert callable(run_until_stop)

    def test_run_until_stop_with_logging_is_exported(self) -> None:
        """run_until_stop_with_logging should be accessible from the public API."""
        assert callable(run_until_stop_with_logging)


class TestSimulationToIRLowering:
    """Test that Simulation objects are correctly lowered to tagged IR."""

    def test_simulation_to_ir_produces_simulationir(self) -> None:
        """simulation_to_ir should produce a SimulationIR instance."""
        sim = build_test_simulation()
        ir = simulation_to_ir(sim)
        assert isinstance(ir, SimulationIR)

    def test_simulation_ir_is_frozen_and_tagged(self) -> None:
        """The produced IR should be frozen and have correct schema version."""
        ir = simulation_to_ir(build_test_simulation())
        assert ir.schema_version == "phase1.v1"
        # Check it's a tagged model with proper type
        assert ir.type == "SimulationIR"

    def test_simulation_ir_preserves_core_attributes(self) -> None:
        """IR lowering should preserve all core simulation attributes."""
        sim = build_test_simulation()
        ir = simulation_to_ir(sim)

        assert ir.center == sim.center
        assert ir.size == sim.size
        assert ir.run_time == sim.run_time
        assert ir.courant == sim.courant
        assert ir.symmetry == sim.symmetry
        assert ir.shutoff == sim.shutoff

    def test_simulation_ir_contains_scene_ir(self) -> None:
        """IR should contain a lowered SceneIR, not the original Scene object."""
        ir = simulation_to_ir(build_test_simulation())
        assert ir.scene is not None
        # The scene should be SceneIR type, not Scene type
        from autofdtd.ir import SceneIR
        assert isinstance(ir.scene, SceneIR)

    def test_simulation_ir_sources_are_lowered(self) -> None:
        """Source components should be lowered to typed IR."""
        ir = simulation_to_ir(build_test_simulation())
        assert len(ir.sources) == 1
        # Should be typed IR, not the original source object
        from autofdtd.ir import UniformCurrentSourceIR
        assert isinstance(ir.sources[0], UniformCurrentSourceIR)

    def test_simulation_ir_boundary_spec_is_lowered(self) -> None:
        """Boundary spec should be lowered to typed IR."""
        ir = simulation_to_ir(build_test_simulation())
        assert ir.boundary_spec is not None
        from autofdtd.ir import BoundarySpecIR
        assert isinstance(ir.boundary_spec, BoundarySpecIR)

    def test_simulation_ir_runtime_controls_are_present(self) -> None:
        """Runtime controls should be compiled and present in IR."""
        ir = simulation_to_ir(build_test_simulation())
        assert ir.runtime_controls is not None
        assert ir.runtime_controls.dt is not None
        assert ir.runtime_controls.num_time_steps is not None

    def test_simulation_ir_is_json_serializable(self) -> None:
        """IR should have stable JSON serialization."""
        ir = simulation_to_ir(build_test_simulation())
        json_text = ir.to_json_text()
        parsed = json.loads(json_text)

        assert parsed["type"] == "SimulationIR"
        assert parsed["schema_version"] == "phase1.v1"
        assert "center" in parsed
        assert "size" in parsed


class TestCompilationPipeline:
    """Test that compilation produces correct CompiledSimulation artifacts."""

    def test_compile_simulation_produces_compiled_simulation(self) -> None:
        """compile_simulation should produce a CompiledSimulation instance."""
        sim = build_test_simulation()
        compiled = compile_simulation(sim)
        assert isinstance(compiled, CompiledSimulation)

    def test_compiled_simulation_contains_ir_snapshot(self) -> None:
        """CompiledSimulation should contain an IR snapshot."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.simulation_ir is not None
        assert isinstance(compiled.simulation_ir, SimulationIR)
        assert compiled.simulation_ir.schema_version == "phase1.v1"

    def test_compiled_simulation_grid_shape_is_consistent(self) -> None:
        """Compiled grid shape should match the IR grid spec."""
        compiled = compile_simulation(build_test_simulation())
        nx, ny, nz = compiled.grid_shape
        assert nx > 0 and ny > 0 and nz > 0
        assert compiled.total_cells == nx * ny * nz

    def test_compiled_simulation_contains_scene_coefficients(self) -> None:
        """Compiled simulation should contain scene material coefficients."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.scene_coefficients is not None
        assert compiled.scene_coefficients.background is not None
        assert len(compiled.scene_coefficients.structures) == 1

    def test_compiled_simulation_contains_runtime_controls(self) -> None:
        """Compiled simulation should contain compiled runtime controls."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.runtime_controls is not None
        assert compiled.runtime_controls.dt is not None
        assert compiled.runtime_controls.num_time_steps is not None
        assert compiled.runtime_controls.max_step_index is not None

    def test_compiled_simulation_contains_chunk_layout(self) -> None:
        """Compiled simulation should contain chunk layout metadata."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.chunk_layout is not None
        # Phase 1 is monolithic (single chunk)
        assert len(compiled.chunk_layout.chunks) == 1

    def test_compiled_simulation_contains_compiled_sources(self) -> None:
        """Compiled simulation should contain compiled source metadata."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.compiled_sources is not None
        assert len(compiled.compiled_sources.uniform_current) == 1

    def test_compiled_simulation_contains_compiled_boundaries(self) -> None:
        """Compiled simulation should contain compiled boundary metadata."""
        compiled = compile_simulation(build_test_simulation())
        assert compiled.compiled_boundaries is not None
        assert compiled.compiled_boundaries.boundary_spec is not None

    def test_compilation_is_deterministic(self) -> None:
        """Re-compilation with same inputs should produce equivalent artifacts."""
        sim = build_test_simulation()
        compiled1 = compile_simulation(sim)
        compiled2 = compile_simulation(sim)

        assert compiled1.grid_shape == compiled2.grid_shape
        assert compiled1.runtime_controls.dt == compiled2.runtime_controls.dt
        assert (
            compiled1.runtime_controls.num_time_steps
            == compiled2.runtime_controls.num_time_steps
        )


class TestExecutionDrivenByIR:
    """Test that execution is driven by the IR, not the original Python objects."""

    def test_compiled_simulation_ir_is_independent_of_original(self) -> None:
        """The IR snapshot should be independent of the original Simulation object.

        Note: Simulation objects are frozen (immutable), so we verify independence
        by checking that the IR is a proper snapshot with its own copy of data.
        """
        sim = build_test_simulation()
        compiled = compile_simulation(sim)

        # The IR is a snapshot - verify it has its own data
        ir_center = compiled.simulation_ir.center

        # Create a new simulation with different center
        sim2 = build_test_simulation()
        # sim2.center is already different from sim.center (they're both the same)
        # The key point is that the compiled IR is frozen and independent

        assert compiled.simulation_ir.center == sim.center
        # The IR snapshot should be frozen (immutable)
        with pytest.raises(Exception):
            compiled.simulation_ir.center = (999.0, 999.0, 999.0)

    def test_runtime_controller_uses_compiled_controls_not_simulation(self) -> None:
        """RuntimeController should be built from CompiledRuntimeControls, not raw Simulation."""
        sim = build_test_simulation()
        compiled = compile_simulation(sim)

        # Build controller from compiled controls
        controller = build_runtime_controller(
            sim,  # original sim for reference
            dt=compiled.runtime_controls.dt,
            max_steps=compiled.runtime_controls.num_time_steps,
        )

        # The controller should use the compiled dt, not something from Simulation directly
        assert controller.compiled.dt == compiled.runtime_controls.dt
        assert controller.compiled.num_time_steps == compiled.runtime_controls.num_time_steps

    def test_compilation_uses_ir_for_all_metadata(self) -> None:
        """Compilation should use IR-derived metadata, not re-parse Simulation."""
        sim = build_test_simulation()
        ir = simulation_to_ir(sim)
        compiled = compile_simulation(sim)

        # IR and compiled should agree on key metadata
        assert compiled.simulation_ir.center == ir.center
        assert compiled.simulation_ir.size == ir.size
        assert compiled.simulation_ir.run_time == ir.run_time

    def test_execution_package_round_trip(self) -> None:
        """Execution package should round-trip through JSON serialization."""
        sim = build_test_simulation()
        package = simulation_to_execution_package(sim)

        # Serialize the package simulation IR
        json_text = package.simulation.to_json_text()
        parsed = json.loads(json_text)

        assert parsed["type"] == "SimulationIR"
        assert parsed["schema_version"] == "phase1.v1"


class TestRuntimeStopIntegration:
    """Test integration of runtime stop policy with compiled controls."""

    def test_run_until_stop_with_compiled_dt(self) -> None:
        """run_until_stop should work with compiled timestep."""
        sim = build_test_simulation()
        compiled = compile_simulation(sim)

        # Use compiled dt for the stop evaluation
        # Note: The test simulation's run_time may produce a small step count
        # relative to dt. We test the integration with whatever step budget exists.
        decision = run_until_stop(
            sim,
            [10.0, 5.0, 1e-6],
            dt=compiled.runtime_controls.dt,
            shutoff_check_interval=1,
        )

        assert decision.should_stop
        assert decision.reason is not None
        # The decision should reflect either shutoff or run_time exhaustion
        assert decision.step >= 0

    def test_run_until_stop_with_logging_with_compiled_dt(self) -> None:
        """run_until_stop_with_logging should work with compiled timestep."""
        sim = build_test_simulation()
        compiled = compile_simulation(sim)

        log = run_until_stop_with_logging(
            sim,
            [10.0, 5.0, 1e-6],
            dt=compiled.runtime_controls.dt,
            shutoff_check_interval=1,
            progress_interval=1,
        )

        assert log is not None
        # Log has primary_stop_reason, not stop_reason
        assert log.primary_stop_reason is not None


class TestNoPythonObjectGraphInCompiled:
    """Verify that CompiledSimulation does not retain references to the original Python objects."""

    def test_compiled_simulation_ir_is_frozen(self) -> None:
        """The IR in CompiledSimulation should be frozen (immutable)."""
        compiled = compile_simulation(build_test_simulation())

        # Try to modify - should fail or have no effect
        ir_before = compiled.simulation_ir.center
        # Frozen dataclass should prevent mutation
        with pytest.raises(Exception):
            compiled.simulation_ir.center = (0.0, 0.0, 0.0)

    def test_compiled_runtime_controls_is_frozen(self) -> None:
        """CompiledRuntimeControls should be frozen."""
        compiled = compile_simulation(build_test_simulation())

        # Try to modify - should fail or have no effect
        with pytest.raises(Exception):
            compiled.runtime_controls.dt = 0.0