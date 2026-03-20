# Meep-Compatible GPU FDTD Core Spec

## Purpose

This document is the implementation spec for a maintainable GPU-first FDTD solver that is compatible with the core Meep API and supports the hybrid time-/frequency-domain adjoint workflow.

The central rule is:

> Do not encode feature combinations by hand as separate kernels.

Instead, define:

- a compact solver IR
- a state-layout spec
- a stage graph
- feature capability descriptors
- a backend code generator

## Non-Goals

- unstructured meshes
- finite elements
- arbitrary adaptive meshing in the first implementation
- copying Meep's C++ organization literally

## Design Targets

- Meep-compatible Python-facing core solver API
- GPU-first execution on a single GPU and across multiple GPUs
- maintainable specialization strategy
- exact or near-exact reproduction of Meep semantics where compatibility matters
- explicit support for the hybrid adjoint pipeline

## Solver Model

The timestep is a staged DAG, not a monolithic kernel. At minimum:

1. phase material updates
2. refresh conductivity/PML coefficients
3. update `B` from the curl of `E`
4. inject magnetic-side sources
5. boundary/halo exchange for the affected magnetic-side state
6. update `H` from `B`
7. update magnetic polarization/dispersive state
8. update half-step monitor state where required
9. update `D` from the curl of `H`
10. inject electric-side sources
11. boundary/halo exchange for the affected electric-side state
12. update `E` from `D`
13. update electric polarization/dispersive state
14. update monitors and DFT accumulators

This matches the semantic split visible in `meep/src/step.cpp`, `meep/src/step_db.cpp`, `meep/src/update_eh.cpp`, and `meep/src/update_pols.cpp`.

This staging choice is also directly supported by the Meep paper in `papers/meep_paper.pdf`. The paper explains that fusing `D <- curl(H)` with `E <- epsilon^{-1} D` and similarly on the magnetic side is blocked not only by the nonlocal curl stencil, but also by anisotropic/off-diagonal constitutive coupling on the Yee grid. It further notes that Meep already faced a combinatorial explosion in separate update cases, and that naively joining the loops would expand those combinations to `16 x 12 = 192` cases. That argument should be treated as a hard design constraint for any generated GPU implementation.

## Required IR

The LLM-facing spec should generate from the following IR.

### 1. Grid Topology IR

Must define:

- dimensionality: `1d`, `2d`, `3d`, `cylindrical`
- Yee staggering per field component
- component-to-derivative relations
- component-specific owned extents
- component-specific halo extents
- real-field vs complex-field execution mode
- Bloch phase handling
- symmetry transforms
- optional `beta` and cylindrical `m` modifiers

### 2. Material Capability IR

Each region/chunk must declare capabilities rather than requesting prewritten kernels.

Capabilities:

- isotropic linear medium
- anisotropic linear medium
- conductivity
- PML/absorber
- nonlinear `chi2`
- nonlinear `chi3`
- dispersive susceptibility families
- gyrotropy
- material-grid interpolation/projection
- subpixel smoothing

Each capability must specify:

- which state arrays it requires
- which stages it affects
- whether it changes halo requirements
- whether it requires a generated specialized kernel

### 3. State Layout IR

Base state:

- `Ex Ey Ez`
- `Dx Dy Dz`
- `Hx Hy Hz`
- `Bx By Bz`

Optional lazily materialized state:

- PML auxiliary `U`
- PML auxiliary `W`
- conductivity auxiliary state
- polarization state
- previous-step polarization/PML state when required
- `minus_p` or equivalent constitutive correction buffers
- DFT accumulators
- source coefficient arrays
- monitor work buffers
- halo pack/unpack buffers

Rule:

- allocate only what the active feature set requires

This preserves one of the main maintainability and efficiency wins of Meep and avoids Khronos-style early rigidity.

### 4. Stage Graph IR

Each stage must declare:

- inputs
- outputs
- neighbor dependency radius
- required halo components
- applicable feature predicates
- whether it can overlap with communication
- whether it touches monitor state

Examples:

- `curl_B_from_E`
- `curl_D_from_H`
- `constitutive_B_to_H`
- `constitutive_D_to_E`
- `update_polarization_E`
- `update_polarization_H`
- `inject_sources_E`
- `inject_sources_H`
- `accumulate_dft`
- `pack_halo`
- `unpack_halo`

### 5. Observer/Chunkloop IR

Sources and monitors should not be fused into the core solver kernels by default.

Separate observer/operator families:

- source placement
- source injection
- DFT field accumulation
- flux accumulation
- mode monitor accumulation
- near-to-far surface accumulation
- array extraction
- field sampling/integration

This mirrors the useful semantic split in Meep's `loop_in_chunks` and `dft_chunk` machinery without importing its full coupling.

The Meep paper's `loop-in-chunks` abstraction is the right precedent here: output, interpolation, integration, and source projection should operate over chunk intersections and precomputed transforms/interpolation metadata, not by repeatedly querying single points through the full boundary/symmetry/communication stack.

## Kernel Families

The generator should produce kernels from these families.

### Family A: Interior Yee Curl

Purpose:

- update `B` from `E`
- update `D` from `H`

Variants are driven by capabilities, not handwritten kernel names:

- plain interior
- conductivity-corrected
- PML interior
- cylindrical modifier enabled
- `beta` modifier enabled
- fixed-angle/BFAST enabled

### Family B: Constitutive Update

Purpose:

- update `H` from `B`
- update `E` from `D`

Must support:

- scalar inverse coefficients
- tensor inverse coefficients
- nonlinear corrections
- subtraction/addition of polarization terms
- PML `W` evolution where needed

### Family C: Polarization / Dispersive Update

Purpose:

- update susceptibility-specific internal state

Rule:

- treat each susceptibility family as a plugin with a declared contract
- generate the dispatch from capability metadata

### Family D: Source Injection

Purpose:

- inject point, volume, indexed, and mode-derived sources

Must support:

- electric and magnetic injection
- separable spatial profiles
- arbitrary temporal waveforms
- integrated-source semantics where required by Meep

### Family E: Monitor / DFT Accumulation

Purpose:

- accumulate Fourier-domain monitor state during a time run

Must support:

- decimation
- persistence across runs for adjoint workflows
- Yee-grid and voxel-centered sampling
- chunk-local partial accumulation with later reduction

### Family F: Near-to-Far and Mode Postprocessing

Purpose:

- evaluate far fields from stored near-surface Fourier fields
- evaluate eigenmode overlaps and mode coefficients

Rule:

- these are not core timestep kernels
- they are operator kernels over accumulated monitor state

### Family G: Halo Pack/Exchange/Unpack

Purpose:

- move stage-specific boundary data between chunks/devices

Rule:

- halo exchange is stage-specific
- do not assume a single “exchange E/H once per timestep” operation

## Mandatory Feature Surface

The first implementation should be split into tiers.

### Tier 0: Must-Have Core

- 1d/2d/3d Cartesian Yee solver
- real and complex fields
- PML
- isotropic and anisotropic linear materials
- conductivity
- current sources
- DFT field monitors
- flux monitors
- array extraction
- chunked domain decomposition
- single-GPU and multi-GPU execution
- Meep-like Python simulation object

### Tier 1: Required for credible Meep-core compatibility

- Bloch-periodic boundaries
- symmetries
- cylindrical coordinates
- eigenmode source support
- mode monitors/eigenmode overlaps
- near-to-far transforms
- material-grid design regions
- adjoint design-region DFT accumulation

### Tier 2: Important but can follow

- nonlinear materials
- Lorentz/Drude-style dispersive media
- gyrotropy
- subpixel smoothing parity with Meep
- dynamic load-balanced rechunking

## Multi-GPU Contract

### Decomposition

Use explicit chunk decomposition.

Each chunk owns:

- local field state
- local material/coefficient state
- local monitor accumulators
- local optional auxiliary state
- local halo buffers

Each chunk also knows:

- neighboring chunks
- component-specific send regions
- component-specific receive regions
- phase/negation/copy transforms for boundary data

### Communication Rules

- communication is stage-specific
- communication metadata is generated from the stage graph and grid topology
- pack/unpack kernels are backend kernels, not Python loops
- overlap communication with interior compute whenever possible
- use peer access on same-node multi-GPU when available
- keep an MPI-friendly abstraction boundary so a future distributed backend can reuse the same chunk contract

### Load Balancing

Need two modes:

- static chunking
- empirical rechunking for repeated optimization runs

This mirrors why Meep exposes chunk layout and balancing controls in repeated adjoint workflows.

## Code Generation Strategy

### Required Pattern

Generate specialized kernels from:

- dimension
- field precision
- real vs complex
- material capabilities
- PML presence
- coordinate modifiers
- monitor/source needs

### Forbidden Pattern

Do not maintain:

- one handwritten kernel per feature combination
- one class per variant
- one file per small variation of the stencil

### Suggested Specialization Mechanism

1. Build a canonical stage graph from the simulation config.
2. Infer the minimal active capability set.
3. Materialize the minimal state layout.
4. Emit one specialized kernel per stage/capability bundle.
5. Cache compiled kernels by a stable signature.

Example signature fields:

- dimension
- dtype
- complex_mode
- pml_kind
- conductivity_enabled
- anisotropy_kind
- nonlinear_kind
- dispersive_kind
- cylindrical_enabled
- beta_enabled

## Design Patterns

- Keep semantics in the IR and runtime planner, not in ad hoc kernel names.
- Keep field arrays SoA by default.
- Split interior kernels from boundary/halo kernels.
- Treat monitors as first-class operators with their own state.
- Keep source synthesis separate from source injection.
- Make chunk metadata explicit and serializable.
- Preserve Meep-like lazy allocation.
- Keep the Python API surface stable while allowing backend replacement.

## Anti-Patterns

- Mega-kernels that mix timestep, source injection, monitor accumulation, and communication.
- Treating multi-GPU as an afterthought to single-GPU memory layout.
- Hand-coding the cross product of `PML x conductivity x anisotropy x coordinate-system x precision x source-kind`.
- Baking Meep's current implementation quirks directly into the public API.
- Building the solver on top of a framework-specific layout model with no backend-neutral IR.
- Reproducing finite-difference material-gradient hacks where exact analytic kernels are practical.

## Compatibility Rules

Be compatible with Meep at the semantic level:

- same high-level simulation concepts
- same monitor/source meanings
- same forward/adjoint workflow shape
- compatible array and monitor outputs where feasible

Do not require source compatibility with Meep's internal C++ design.

## Implementation Order

1. solver IR and chunk metadata
2. Tier 0 field/material/source/DFT kernels
3. multi-GPU halo protocol
4. mode monitor and near-to-far operators
5. material-grid and adjoint infrastructure
6. higher-order materials and advanced Meep features

## Final Architectural Recommendation

Implement the first production version in Warp.

Keep the spec backend-neutral enough that a future Kokkos/C++ backend can target the same:

- grid topology IR
- stage graph IR
- material capability IR
- chunk/halo contract
- observer/operator contract
