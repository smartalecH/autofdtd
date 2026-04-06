# AutoFDTD

AutoFDTD is being built as a GPU-first FDTD package with a Tidy3D-inspired public
surface and a chunk-oriented runtime. Phase 1 starts from an empty repository, so this
scaffold establishes the package boundaries and development entrypoints that later tasks
will fill with concrete models, compilation logic, and Warp kernels.

## Current Scope

- pip-installable `src/` layout package
- subsystem-oriented namespaces for API, IR, compiler, runtime, kernels, and diagnostics
- feature-matrix metadata anchored to `phase1/feature-checklist.md`
- tagged immutable `Simulation`, `Scene`, and `Structure` container models
- `PolySlab` polygon extrusions for core planar photonics geometry
- concrete `UniformGrid`, `CustomGrid`, `CustomGridBoundaries`, `AutoGrid`, and `GridSpec` models
- `GridRefinement` and `LayerRefinementSpec` planning surfaces for early meshing metadata
- `Simulation.resolved_grid()` for inspectable axis-boundary planning
- `Periodic`, `PECBoundary`, `PMCBoundary`, `Boundary`, and `BoundarySpec` models
- compiled non-absorbing boundary runtime metadata and ghost-update helpers
- structured runtime stop decisions, progress logs, and convergence evidence capture
- versioned `SimulationIR` and `ExecutionPackageIR` transport models for execution handoff
- test, lint, type-check, and docs entrypoints

## Immediate Follow-On Work

The next tasks move from grid planning into subpixel policy, then into concrete
material, boundary, and runtime implementation work.
