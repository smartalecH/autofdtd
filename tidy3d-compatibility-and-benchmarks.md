# Tidy3D Compatibility and Benchmarks

## Purpose

This document extends the solver spec to support a second frontend target:

- Meep-compatible API
- Tidy3D-compatible API

The immediate motivation is to run the notebook corpus in `tidy3d-notebooks` and benchmark:

- correctness
- numerical agreement
- runtime
- memory use

## Main Conclusion

Tidy3D compatibility must be defined in tiers.

The local `tidy3d-notebooks` corpus is not only “core FDTD examples”. It also includes:

- core FDTD
- adjoint/autograd inverse design
- mode solver workflows
- field projection / near-to-far variants
- diffraction and overlap monitors
- EME
- heat/charge / multi-physics
- web/cloud workflow helpers

Therefore “compatible with the Tidy3D API” cannot mean “everything at once” if the first target is a maintainable Meep-compatible GPU FDTD core.

## Observed Tidy3D Surface in Local Repos

The dominant notebook API objects in `tidy3d-notebooks` are:

- `td.Simulation`
- `td.Structure`
- `td.Box`
- `td.Medium`
- `td.GridSpec`
- `td.BoundarySpec`
- `td.FieldMonitor`
- `td.FluxMonitor`
- `td.ModeMonitor`
- `td.ModeSource`
- `td.GaussianPulse`
- `td.PML`
- `td.Cylinder`
- `td.PolySlab`
- `td.PlaneWave`
- `td.PointDipole`
- `td.FieldTimeMonitor`
- `td.TFSF`
- `td.CustomMedium`
- `td.AnisotropicMedium`

This is enough evidence that a substantial fraction of the notebook corpus can be covered by a single core-FDTD compatibility layer.

Relevant local references:

- `tidy3d/tidy3d/components/simulation.py`
- `tidy3d/tidy3d/components/base_sim/simulation.py`
- `tidy3d/tidy3d/components/boundary.py`
- `tidy3d/tidy3d/components/monitor.py`
- `tidy3d/tidy3d/components/source/current.py`
- `tidy3d/tidy3d/components/source/field.py`
- `tidy3d/tidy3d/components/medium.py`

## Compatibility Strategy

## 1. Dual Frontend, Shared Core

The solver should have:

- one shared backend-neutral solver IR
- one Meep-facing frontend adapter
- one Tidy3D-facing frontend adapter

The backend runtime should not know which frontend produced the IR.

That means:

- Meep frontend lowers `mp.Simulation`, sources, monitors, materials, chunk/symmetry settings into the shared IR
- Tidy3D frontend lowers `td.Simulation`, `GridSpec`, `BoundarySpec`, structures, sources, monitors, and autograd design objects into the same IR

## 2. Compatibility Levels

### Level A: Core Tidy3D FDTD Compatibility

This level is enough to run a large fraction of notebooks.

Required objects:

- `td.Simulation`
- `td.Structure`
- geometry primitives:
  - `td.Box`
  - `td.Cylinder`
  - `td.Sphere`
  - `td.PolySlab`
  - `td.GeometryGroup`
- materials:
  - `td.Medium`
  - `td.CustomMedium`
  - `td.AnisotropicMedium`
  - `td.PoleResidue`
  - `td.Lorentz`
  - `td.Drude`
- grid:
  - `td.GridSpec.uniform`
  - `td.GridSpec.auto`
  - enough of `AutoGrid` / override-refinement behavior to reproduce the main examples
- boundaries:
  - `td.Boundary`
  - `td.BoundarySpec`
  - `td.PML`
  - `td.Periodic`
  - `td.BlochBoundary`
  - `td.Absorber`
- sources:
  - `td.ModeSource`
  - `td.PlaneWave`
  - `td.GaussianBeam`
  - `td.TFSF`
  - `td.PointDipole`
  - `td.UniformCurrentSource`
  - `td.CustomCurrentSource`
  - `td.CustomFieldSource`
- source time profiles:
  - `td.GaussianPulse`
  - `td.ContinuousWave`
  - custom time source hooks
- monitors:
  - `td.FieldMonitor`
  - `td.FieldTimeMonitor`
  - `td.FluxMonitor`
  - `td.FluxTimeMonitor`
  - `td.ModeMonitor`
  - `td.ModeSolverMonitor`
  - `td.DiffractionMonitor`
  - field projection monitor family

### Level B: Tidy3D Adjoint / Autograd Compatibility

Required for many inverse-design notebooks.

Needed capabilities:

- differentiable geometry/material parameterization frontend
- custom medium differentiation
- transpose interpolation / reverse interpolation for sources
- monitor-space differentiation and adjoint-source generation
- design-region data extraction
- JVP/VJP plumbing compatible with Tidy3D-style autograd workflows

Important local clue:

- Tidy3D already treats reverse interpolation as a first-class idea in `ReverseInterpolatedSource`
- that is conceptually aligned with the interpolation/restriction operators already required by the Meep-adjoint formulation

This is a useful unification point for the shared IR.

### Level C: Ecosystem Compatibility

These are not part of the core GPU FDTD target and should be treated as separate solvers or auxiliary packages:

- `EMESimulation`
- heat / charge simulations
- TCAD-style material and monitor stack
- web/cloud job submission
- GUI / visualization convenience tools

These should not block the core solver.

## Shared IR Extensions Needed for Tidy3D

Relative to the Meep-first spec, the shared IR needs explicit support for:

### GridSpec IR

Must encode:

- uniform grids
- auto/nonuniform grids
- local refinement regions
- mesh override structures
- snapping and refinement hints

Reason:

Tidy3D examples use `GridSpec.auto(...)` heavily, whereas Meep is more naturally expressed in terms of a global resolution.

### BoundarySpec IR

Must encode per-axis plus/minus boundary conditions:

- PML
- periodic
- Bloch
- PEC/PMC
- absorber

Reason:

Tidy3D exposes boundaries explicitly per side and per axis through `Boundary` and `BoundarySpec`.

### Geometry IR

Must encode explicit geometry trees and override ordering:

- background medium
- ordered structure list
- priority / overlap rules
- grouped geometry
- transformations

Reason:

Tidy3D’s `Structure` and geometry stack is more explicit and object-oriented than Meep’s default geometry flow.

### Monitor IR

Must encode monitor families as first-class operator specs:

- point/volume/surface field monitors
- flux monitors
- mode overlap monitors
- diffraction monitors
- field projection monitors
- time-domain monitors

Each monitor spec must include:

- sampling locus
- Yee or colocated sampling convention
- time/frequency sampling data
- apodization/windowing
- output schema

### Source IR

Must encode both direct and reverse-interpolated source semantics.

This is important because Tidy3D exposes:

- direct current sources
- mode/field sources
- custom field/current datasets
- reverse interpolation on zero-size source dimensions

This maps naturally onto the same interpolation/restriction math already needed for adjoints.

## API Mapping

## Tidy3D Frontend Mapping Rules

### Simulation

`td.Simulation(...)` lowers to:

- domain extents
- background medium
- ordered structure list
- source list
- monitor list
- per-axis boundary spec
- grid spec
- runtime / shutoff / courant parameters
- symmetry flags
- subpixel or interpolation policy

### Structures and Geometry

`td.Structure(geometry=..., medium=...)` lowers to:

- geometry node
- material node
- priority/order metadata

### Mediums

Common mediums lower to material capability descriptors:

- `td.Medium` -> isotropic linear material
- `td.AnisotropicMedium` -> tensor material
- `td.CustomMedium` -> sampled/custom material field
- `td.PoleResidue`, `td.Lorentz`, `td.Drude` -> dispersive susceptibility families

### Sources

Examples:

- `td.ModeSource` -> eigenmode source operator
- `td.PlaneWave` -> extended current source plus angle/Bloch metadata
- `td.GaussianBeam` -> field-profile source operator
- `td.TFSF` -> source operator plus region bookkeeping
- `td.PointDipole` -> zero-volume current source
- `td.CustomCurrentSource` -> dataset-backed source operator

### Monitors

Examples:

- `td.FieldMonitor` -> DFT field monitor
- `td.FieldTimeMonitor` -> time history monitor
- `td.FluxMonitor` -> surface integral operator
- `td.ModeMonitor` -> overlap operator
- `td.DiffractionMonitor` -> periodic/far-field projection operator
- `td.FieldProjection*Monitor` -> near-to-far / Green-function operator family

## Relationship to Meep Compatibility

The core solver should preserve one semantic center and two frontend skins.

The semantic center is:

- Yee FDTD state
- geometry/material distribution
- staged updates
- chunk/halo communication
- source operators
- monitor operators
- adjoint operators

The frontend skins differ mostly in:

- object model
- defaults
- naming
- grid specification style
- monitor/source configuration syntax

This means Meep compatibility and Tidy3D compatibility are not competing goals if the IR boundary is designed correctly.

## Notebook Benchmarking Plan

## 1. Benchmark Tiers

### Tier 1: Core FDTD benchmark set

Include notebooks that mainly require:

- `Simulation`
- `Structure`
- `Medium`
- `GridSpec`
- `BoundarySpec`
- `ModeSource`
- `PlaneWave`
- `FieldMonitor`
- `FluxMonitor`
- `ModeMonitor`

This tier should be the first correctness/performance benchmark target.

### Tier 2: Projection / periodic / advanced source benchmark set

Include notebooks that require:

- TFSF
- Bloch/periodic boundaries
- diffraction monitors
- field projection / near-to-far
- anisotropy
- dispersive media

### Tier 3: Adjoint/autograd benchmark set

Include notebooks that require:

- custom mediums
- design-region parameterization
- gradients
- inverse design

### Tier 4: Out-of-scope for the core solver

Do not use these to judge the core FDTD implementation:

- EME notebooks
- heat/charge notebooks
- cloud/web API notebooks

Those require additional solver families or service layers.

## 2. Benchmark Metrics

For each benchmark case record:

- wall-clock runtime
- GPU memory usage
- number of cells / Yee dofs
- number of time steps
- monitor count and frequency count
- objective metrics:
  - flux / transmission
  - mode overlap
  - resonant frequency / Q where relevant
  - far-field or diffraction efficiencies where relevant
- relative error against notebook reference output

For adjoint notebooks also record:

- gradient agreement against finite-difference spot checks
- total forward+adjoint runtime
- memory overhead of gradient path

## 3. Success Criteria

### API compatibility success

A notebook is considered API-compatible if it can be executed with only:

- import-path substitution
- clearly documented unsupported-argument warnings
- no semantic rewrite of the physical setup

### Numerical success

A notebook is numerically successful if:

- field/monitor outputs agree within predefined tolerances
- convergence trends with mesh/runtime agree
- gradients pass finite-difference spot checks where relevant

### Performance success

The package should report:

- absolute runtime
- relative runtime against the reference Tidy3D setup when reproducible locally
- scaling with resolution/frequency count
- scaling with number of GPUs

## Recommended Deliverable Boundary

The project should promise:

- Meep core-solver compatibility
- Tidy3D core-FDTD compatibility
- notebook benchmarking for Tier 1 and Tier 2 first

The project should not initially promise:

- full Tidy3D ecosystem compatibility
- EME compatibility
- heat/charge compatibility
- cloud/web workflow compatibility

## Final Recommendation

Add a `Tidy3DFrontend` adapter on top of the shared solver IR and treat the notebook corpus as a staged benchmark suite.

This keeps the implementation honest:

- the solver remains one solver
- the frontend compatibility burden stays explicit
- the benchmark goal becomes measurable
