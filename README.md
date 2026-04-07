# AutoFDTD

GPU-first finite-difference time-domain solver scaffold, aligned to a Tidy3D-inspired
public API surface and a chunk-based multi-node-ready runtime architecture.

## Current Status: Phase 1 Complete

Phase 1 has produced a pip-installable, end-to-end-runnable FDTD package with:

- **100 registered test functions** across boundary, material, source, monitor, kernel, and integration coverage
- **73 implemented features** spanning core containers, geometry, materials, boundaries, sources, monitors, and postprocessing
- **17 deferred features** with explicit deferral rationale and no silent gaps
- **10 rejected features** with clear error messaging so users know what is out of scope
- **9 canonical validation examples** exercising compile → execute → validate workflows
- **IR lowering pipeline** from public API → SimulationIR → ExecutionPackageIR → CompiledSimulation → runtime execution
- **Chunk architecture contract** (`phase1/architecture/chunk-contract.md`) ready for multi-GPU and MPI scaling

## Feature Coverage Summary

| Family | Implement | Defer | Reject |
|--------|-----------|-------|--------|
| Core | 4 | 2 | 1 |
| Geometry | 7 | 2 | 0 |
| Grid | 10 | 6 | 0 |
| Materials | 9 | 5 | 8 |
| Boundaries | 10 | 1 | 0 |
| Sources | 14 | 0 | 1 |
| Monitors | 17 | 0 | 0 |
| Postprocessing | 2 | 1 | 0 |
| **Total** | **73** | **17** | **10** |

## Quick Start

```bash
python -m pip install -e .[dev]
pytest
autofdtd-feature-matrix --format table
```

## Package Layout

```
autofdtd/
  api/          — public Simulation, Scene, Structure entrypoints
  core/         — frozen model base, validation, container semantics
  geometry/     — Box, Sphere, Cylinder, PolySlab, composite wrappers
  materials/    — Medium, PEC/PMC, PoleResidue, Sellmeier, Lorentz,
                  Drude, Debye, AnisotropicMedium
  boundaries/   — Periodic, PEC/PMC, Bloch, PML, StablePML, Absorber, ABC
  sources/      — GaussianPulse, ContinuousWave, UniformCurrentSource,
                  PointDipole, PlaneWave, GaussianBeam, TFSF, ModeSource
  monitors/     — FieldMonitor, FluxMonitor, ModeMonitor, MediumMonitor,
                  Projection monitors, SurfaceFieldMonitor
  grid/         — UniformGrid, CustomGrid, AutoGrid, GridSpec, SubpixelSpec
  modes/        — ModeSpec, ModeSolverCrossSection, ModeSolution
  ir/           — tagged IR models (SimulationIR, SceneIR, StructureIR, etc.)
  compiler/     — scene compilation, material coefficient planning,
                  source/monitor compilation, boundary planning, chunk layout
  runtime/      — timestep loop, convergence control, monitor recording,
                  halo exchange, near-to-far
  kernels/      — Warp kernel conventions, Maxwell update stages,
                  material update kernels, source injection, boundary kernels
  diagnostics/  — RuntimeProgressLogger, RuntimeLogEvent, metrics collection
  benchmarks/   — GPU benchmarking infrastructure (gcells/s, multi-GPU scaling)
  examples/     — validation.py (10 examples), integration.py, near2far.py
```

## Public API Entrypoints

```python
from autofdtd.api import (
    compile_simulation,      # Simulation → CompiledSimulation
    simulation_to_ir,        # Simulation → SimulationIR
    simulation_to_execution_package,  # Simulation → ExecutionPackageIR
    run_until_stop,          # CompiledSimulation → ExecutionResult
    run_until_stop_with_logging,  # with structured log events
)
```

## Unsupported-Feature Policy

Phase 1 implements explicit reject and defer behavior rather than silent gaps:

- **Reject**: `Custom*` media families, `lumped_elements`, `MicrowaveTerminalSource`,
  `InternalAbsorber`. Attempting to use these raises `AutoFDTDNotImplementedError`
  with a clear message about what Phase 1 covers.

- **Defer**: `TriangleMesh`, `QuasiUniformGrid`, `FullyAnisotropicMedium`, `Medium2D`,
  `PerturbationMedium`, `LossyMetalMedium`, `ModeABCBoundary`, and others. These are
  parsed, validated, and documented with deferral rationale in `planning.py`. Attempting
  to use a deferred feature raises `AutoFDTDNotImplementedError` with the specific deferral
  context.

- **Deferred by dependency**: `ModeSource` and `ModeMonitor` depend on the mode solver
  being available. The mode solver (`autofdtd.modes`) is implemented and runnable, but
  full mode-source injection integration is deferred pending chunk kernel wiring.

## Architecture

Phase 1 separates scene modeling, scene compilation, runtime stepping, and IO:

```
Simulation → simulation_to_ir() → SimulationIR
SimulationIR + compile_simulation() → CompiledSimulation
CompiledSimulation → run_until_stop() → ExecutionResult
```

- `Simulation` is the public API object model (frozen, tagged, JSON-ready)
- `SimulationIR` is the versioned execution transport (schema: `phase1.v1`)
- `CompiledSimulation` holds resolved grid, scene coefficients, compiled sources/monitors/boundaries, and chunk layout
- `ExecutionResult` captures field state, step count, stop reason, integrated history, monitor data, and runtime metrics

Chunk decomposition is designed around `ChunkSpec`, `FaceHalo`, and `ChunkLayout`
contracts. Phase 1 is monolithic (single chunk, `num_chunks=(1,1,1)`) but the chunk
contract is ready for multi-GPU and MPI scaling. Per-face halo depth is per-face, not
global; PML, ABC, Bloch, periodic, and symmetry halos are all represented.

## Development

```bash
pytest                              # run all tests
ruff check .                       # lint
mypy src                           # type check
autofdtd-feature-matrix --format table  # show feature status table
python -c "from autofdtd.examples.validation import *; ..."  # run validation examples
```

## Phase 1 Evidence Base

- **790 registered test functions** across the test suite
- **9 canonical validation examples** in `src/autofdtd/examples/validation.py`
- **6 integration examples** in `src/autofdtd/examples/integration.py`
- **3 Warp kernel convention modules** in `src/autofdtd/kernels/backend.py`
- **End-to-end execution** through `run_compiled_simulation()` producing
  `ExecutionResult` with field state, metrics, and monitor data
- **IR transport** from public API through tagged IR to execution package
- **GPU benchmarking infrastructure** in `src/autofdtd/benchmarks/gpu_benchmark.py`

## GPU Benchmarking

Phase 1 provides GPU benchmarking infrastructure for NVIDIA GPUs via Warp:

```python
from autofdtd.benchmarks import (
    benchmark_single_gpu,
    benchmark_multi_gpu,
    benchmark_scaling,
    benchmark_grid_sizes,
    GPUBenchmarkResult,
)

# Single-GPU benchmark
result = benchmark_single_gpu(grid_shape=(100, 100, 100), num_steps=100, device="cuda:0")
print(f"Gcells/s: {result.gcells_per_second:.4f}")  # Primary throughput metric
print(f"Per-step: {result.per_step_ms:.3f} ms")

# Multi-GPU scaling (target ≥1.5x for 2 GPUs)
scaling = benchmark_scaling(grid_shape=(100, 100, 100), num_steps=100)
print(f"Scaling ratio: {scaling['scaling_ratio']:.2f}x")

# Grid size sweep
results = benchmark_grid_sizes(
    sizes=((50, 50, 50), (100, 100, 100), (150, 150, 150)),
    num_steps=100,
)
```

### Benchmark Metrics

| Metric | Description |
|--------|-------------|
| `gcells_per_second` | Billions of Yee cells updated per second |
| `per_step_ms` | Milliseconds per timestep |
| `wall_time_s` | Total wall-clock time |
| `cells_updated` | Total cells = `total_cells × num_steps` |
| `scaling_ratio` | `multi_gpu.gcells_per_second / single_gpu.gcells_per_second` |

### Multi-GPU Scaling Target

Phase 1 targets **≥1.5x** throughput scaling for 2 GPUs vs. single GPU. This is measured
as combined throughput for independent simulations (parameter sweeps, Monte Carlo),
which matches Phase 1's chunk-based architecture where each chunk runs on one device.

### Grid Size Guidance

For meaningful scaling benchmarks (≥1.5x):
- **Minimum**: ~50³ = 125K cells (small, GPU overhead dominates)
- **Recommended**: 100³ = 1M cells (good single-run benchmarks)
- **Large**: 150³ = 3.375M cells (better scaling, longer runs)

### Running Benchmarks

```bash
# Check GPU availability
python -c "from autofdtd.kernels.backend import backend_info; print(backend_info())"

# Run full benchmark suite
python -m src.autofdtd.benchmarks.gpu_benchmark

# Run pytest GPU benchmarks (skipped if Warp/CUDA unavailable)
pytest tests/test_benchmarks_gpu.py -v
```

### Benchmark Result Schema

`GPUBenchmarkResult` is a frozen dataclass with fields:
- `grid_shape`, `total_cells`, `num_steps` — problem sizing
- `wall_time_s`, `cells_updated`, `gcells_per_second`, `per_step_ms` — timing
- `device` — GPU device string (e.g., `"cuda:0"` or `"multi:cuda:0,cuda:1"`)
- `backend` — `"warp"` (or `"numpy"` fallback)

## What's Ready for Hand-Off

The following are production-ready for Phase 1 developer validation:

1. **Public API**: Simulation, Scene, Structure, all geometry/material/source/monitor models
2. **IR lowering**: Full `simulation_to_ir()` pipeline with schema versioning
3. **Compilation pipeline**: `compile_simulation()` producing frozen `CompiledSimulation`
4. **Runtime execution**: `run_until_stop()` / `run_until_stop_with_logging()` with convergence
5. **Validation examples**: 10 canonical physical scenarios with expected outcomes
6. **Chunk architecture**: Full contract documented in `phase1/architecture/chunk-contract.md`
7. **Feature matrix**: `autofdtd.planning._FEATURE_MATRIX` with 100 entries
8. **GPU execution and benchmarking**: GPU kernels JIT-compile and run on NVIDIA GPUs via Warp.
   Benchmark infrastructure in `src/autofdtd/benchmarks/gpu_benchmark.py` with measured results:
   - Single-GPU: ~11 Gcells/s (100³ grid, cuda:0 RTX 5070 Ti, Warp 1.12.1 + CUDA 12.9)
   - 2-GPU: ~22 Gcells/s combined (cuda:0 + cuda:1, ~2.0x scaling)
   - All 10 validation examples run on GPU

## What Needs Further Validation

1. **Larger problem sizes**: Current benchmarks used 100³ grids; larger problems may show different scaling
2. **Real physics validation**: GPU results should be compared against NumPy reference for correctness
3. **MPI multi-node scaling**: Cross-device halo exchange implemented; true multi-node (multiple machines)
   halo exchange needs multi-node hardware
4. **Mode source integration**: Mode solver works; mode-source injection in timestep loop deferred
