# AutoFDTD

GPU-first FDTD package with a Tidy3D-inspired public surface and a chunk-based
multi-node-ready runtime architecture. Phase 1 is complete and runnable.

## Phase 1 Completeness

Phase 1 has produced a full API→IR→execution pipeline:

```
Simulation → simulation_to_ir() → SimulationIR
SimulationIR + compile_simulation() → CompiledSimulation
CompiledSimulation → run_until_stop() → ExecutionResult
```

**751 test functions** cover models, compilation, kernels, boundaries, sources,
monitors, and end-to-end execution. **10 validation examples** demonstrate physical
correctness for vacuum propagation, dielectric waveguides, PML absorption, source
injection, monitor recording, and convergence behavior.

## Feature Coverage

- **73 implemented features** across core, geometry, grid, materials, boundaries,
  sources, monitors, and postprocessing
- **17 deferred features** with explicit deferral rationale (no silent gaps)
- **10 rejected features** with clear `AutoFDTDNotImplementedError` messaging
- **9 canonical validation examples** exercising full compile→execute→validate flows
- **Chunk architecture contract** ready for multi-GPU and MPI scaling
- **Warp backend conventions** with complex field policy, module stability tracking,
  capture-safe stepping, and profiling hooks (NumPy fallback when Warp unavailable)

## Current Status

| Metric | Value |
|--------|-------|
| Tests registered | 751 |
| Feature matrix entries | 100 |
| Implemented features | 73 |
| Deferred features | 17 |
| Rejected features | 10 |
| Validation examples | 10 |
| IR schema version | phase1.v1 |
| Execution pipeline | API → IR → CompiledSimulation → ExecutionResult |

## Quick Start

```bash
python -m pip install -e .[dev]
pytest
autofdtd-feature-matrix --format table

# Run a validation example
python -c "
from autofdtd.examples.validation import vacuum_point_source_example
result = vacuum_point_source_example()
print(f'Stop reason: {result[\"stop_reason\"]}, steps: {result[\"num_steps\"]}')
"
```

## Package Structure

| Module | Responsibility |
|--------|---------------|
| `api` | Public Simulation, Scene, Structure entrypoints |
| `core` | Frozen model base, validation, container semantics |
| `geometry` | Box, Sphere, Cylinder, PolySlab, composite wrappers |
| `materials` | Medium, PEC/PMC, dispersive and anisotropic media |
| `boundaries` | Periodic, PEC/PMC, Bloch, PML, StablePML, Absorber, ABC |
| `sources` | Time profiles, current sources, optical sources, TFSF |
| `monitors` | Field, flux, mode, medium, projection, surface monitors |
| `grid` | UniformGrid, CustomGrid, AutoGrid, GridSpec, SubpixelSpec |
| `modes` | Mode specification, cross-section, mode solver |
| `ir` | Tagged execution IR models and packaging |
| `compiler` | Scene compilation, materialization, chunk layout |
| `runtime` | Timestep loop, convergence, monitor recording |
| `kernels` | Warp kernel conventions, Maxwell update stages |
| `diagnostics` | Runtime logging, progress, metrics |

## Phase 1 Evidence Base

- **Models**: All major Tidy3D feature families have typed public models with frozen,
  tagged, JSON-ready serialization
- **Compilation**: `compile_simulation()` produces frozen `CompiledSimulation` with
  resolved grid, scene coefficients, compiled sources/monitors/boundaries, and chunk layout
- **Execution**: `run_until_stop()` and `run_until_stop_with_logging()` produce
  `ExecutionResult` with field state, metrics (wall_time_s, cells_updated,
  gcells_per_second, backend), and monitor data
- **Validation**: 10 canonical examples with documented expected physical behavior
- **Architecture**: `phase1/architecture/chunk-contract.md` defines chunk data model,
  FaceHalo descriptors, chunk layout, and exchange plan semantics
- **IR**: Versioned (`phase1.v1`) tagged IR models independent from public API

## What's Ready for Hand-Off

1. **Public API**: Full Simulation/Scene/Structure model hierarchy with all geometry,
   material, source, and monitor models
2. **IR lowering**: Full `simulation_to_ir()` pipeline with schema versioning
3. **Compilation**: `compile_simulation()` producing frozen runtime artifacts
4. **Runtime execution**: Timestep loop with convergence control and monitor recording
5. **Validation examples**: 10 canonical physical scenarios
6. **Chunk architecture**: Full contract for multi-GPU and MPI scaling
7. **Feature matrix**: 100-entry code-backed matrix with explicit defer/reject policy

## Known Gaps for Deeper Phases

1. **Warp kernel execution**: Warp not installed; NumPy backend used for tests
2. **GPU performance**: Gcells/s metrics not yet measured on GPU hardware
3. **MPI multi-node**: Chunk layout defined; halo exchange stubbed for single-chunk
4. **Mode source injection**: Mode solver works; timestep-loop integration deferred
5. **Full anisotropy**: `FullyAnisotropicMedium` (dense tensor) deferred; diagonal
   `AnisotropicMedium` implemented
6. **Advanced subpixel**: ContourPath and Volumetric averaging deferred; PolarizedAveraging
   and Staircasing implemented

## Navigation

- [Architecture](architecture.md) — subsystem roles, execution IR, design constraints
- [Geometry Scope](geometry.md) — implemented primitives and composites
- [Grid Scope](grid.md) — discretization and subpixel policy
- [Material Scope](materials.md) — isotropic, dispersive, and anisotropic media
- [Boundary Scope](boundaries.md) — absorbing and non-absorbing boundaries
- [Runtime Controls](runtime-controls.md) — timestep, stop policy, convergence
- [Feature Matrix](feature-matrix.md) — complete feature coverage table
- [Development](development.md) — install, test, lint, docs entrypoints
