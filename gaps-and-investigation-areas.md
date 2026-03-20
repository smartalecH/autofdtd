# Gaps And Investigation Areas

This document records the current gaps in the `fdtd` design notes that still block a fully reliable from-scratch generation of a high-performance package by an LLM or other code generator.

The existing document set is already strong on architecture, subsystem decomposition, and major operator families. The gaps below are the places where the current specs are still too prose-driven, too implicit, or too incomplete to guarantee a correct and maintainable generated implementation.

## Overall Assessment

Current status:

- coherent at the architecture level
- well grounded in local references
- strong enough to guide a human-led implementation
- not yet strict enough to guarantee correct end-to-end code generation

The core issue is that many concepts are described semantically but not yet pinned down as machine-readable schemas, exact contracts, or acceptance manifests.

## Gap 1: Machine-Readable Solver IR

Current state:

- `meep-gpu-core-spec.md` defines the right conceptual IR families
- the IR is still prose, not a strict schema

Why this matters:

- an LLM will otherwise invent field names, enum values, defaults, optionality, and validation logic
- that is the fastest way for two generated backends to silently diverge

Investigation tasks:

- define a canonical JSON/YAML schema for the solver IR
- specify exact required and optional fields
- define enums for dimension, coordinate system, material capability, stage family, source family, and monitor family
- define versioning and backward-compatibility rules
- define how IR objects are serialized, hashed, diffed, and cached

Deliverable target:

- `solver-ir-schema.yaml`
- `solver-ir-examples/`

## Gap 2: Machine-Readable Geometry IR

Current state:

- `geometry-gpu-core-spec.md` and `geometry-and-domain-initialization.md` define the geometry concepts well
- the geometry layer is not yet encoded as a strict schema

Why this matters:

- geometry is one of the biggest semantic boundaries between Meep, Tidy3D, and the backend
- code generation needs a stable primitive representation

Investigation tasks:

- define the exact schema for primitives, transforms, repetitions, groups, priorities, and subpixel policies
- define a canonical ordering and overlap-resolution contract
- define shape parameter naming and units
- define how differentiable parameters are attached to structures
- define exact outputs for point-sampled and subpixel materialization modes

Deliverable target:

- `geometry-ir-schema.yaml`
- primitive examples for boxes, cylinders, polyslabs, repeated geometry, and design regions

## Gap 3: Source / Monitor / Operator IR

Current state:

- sources, monitors, DFTs, near-to-far, and mode operators are described in multiple docs
- there is no single strict operator schema

Why this matters:

- this is where Meep and Tidy3D compatibility meet the adjoint pipeline
- many subtle bugs come from implicit normalization or sampling conventions

Investigation tasks:

- define a canonical schema for:
  - current sources
  - field-profile sources
  - mode sources
  - TFSF sources
  - field monitors
  - flux monitors
  - mode monitors
  - projection and near-to-far monitors
  - time-history monitors
- define exact sampling conventions and output tensor shapes
- define normalization metadata explicitly
- define transpose or backprop form for every operator family

Deliverable target:

- `operator-ir-schema.yaml`
- operator catalog with forward and adjoint contracts

## Gap 4: Exact Python API Compatibility Surface

Current state:

- Meep and Tidy3D frontend mappings are conceptually documented
- exact user-facing signatures and object behaviors are not

Why this matters:

- “compatible” is underspecified without signature-level behavior
- generated frontends need deterministic lowering rules

Investigation tasks:

- define the target public Python API for the first implementation
- define which Meep APIs are semantic targets versus exact signature targets
- define which Tidy3D APIs are semantic targets versus exact signature targets
- specify default behaviors, warnings, unsupported-argument policy, and output object schemas
- define compatibility tiers at the callable/object level

Deliverable target:

- `meep-api-compatibility-manifest.yaml`
- `tidy3d-api-compatibility-manifest.yaml`

## Gap 5: Rich Material Plugin Contracts

Current state:

- core Yee, conductivity, PML, D-field, adjoint recombination, DFT, and monitor operators are documented mathematically
- richer constitutive families are only partially specified

Missing depth is most obvious for:

- Lorentz/Drude dispersive families
- gyrotropy
- `chi2`
- `chi3`
- any future saturable or gain-like media

Why this matters:

- a plugin architecture is only real if each plugin has an exact state-update contract
- otherwise generated kernels will improvise the hard parts

Investigation tasks:

- define discrete update equations for each supported susceptibility family
- define required state arrays and initialization rules
- define coupling to constitutive stages
- define adjoint/JVP/VJP contract for each family
- define accuracy and stability test cases per family

Deliverable target:

- `material-plugin-spec.md` or one file per family

## Gap 6: Exact Multi-GPU Communication Protocol

Current state:

- chunking and halo semantics are described well at a high level
- the communication protocol is not yet exact

Why this matters:

- distributed correctness depends on ownership and packing details
- this is where generated implementations are most likely to drift or deadlock

Investigation tasks:

- define exact ownership rules for every staggered field component
- define send/receive regions per stage
- define pack order and buffer layout
- define boundary transforms for periodic/Bloch/symmetry exchanges
- define reduction semantics for DFT and monitor accumulators
- define overlap schedule between compute and communication
- define the transport abstraction boundary:
  - peer-to-peer
  - NCCL-like collectives where applicable
  - MPI-friendly point-to-point and reductions

Deliverable target:

- `chunk-halo-protocol.md`
- `chunk-halo-schema.yaml`

## Gap 7: Acceptance Manifests And Golden Outputs

Current state:

- benchmark and compatibility plans are present
- there are not yet fixed manifests and golden data products

Why this matters:

- without explicit acceptance artifacts, generated implementations can claim compatibility while drifting numerically

Investigation tasks:

- define a benchmark manifest format
- define a correctness manifest format
- store golden observables for representative cases
- define acceptable tolerances per observable and per precision level
- separate solver-throughput benchmarks from workflow-throughput benchmarks

Deliverable target:

- `benchmark-manifest.yaml`
- `correctness-manifest.yaml`
- `golden-results/`

## Gap 8: Accuracy Policy By Feature Tier

Current state:

- the docs discuss accuracy and benchmark metrics
- there is no systematic tolerance table by feature class

Why this matters:

- fair performance comparisons require fixed accuracy targets
- adjoint correctness needs stricter expectations than casual forward smoke tests

Investigation tasks:

- define accuracy criteria for:
  - core Yee propagation
  - PML reflection
  - mode overlap
  - near-to-far projection
  - dispersive materials
  - nonlinear materials
  - geometry subpixel smoothing
  - adjoint gradients
- define expected tolerance dependence on precision and resolution

Deliverable target:

- `accuracy-policy.md`

## Gap 9: Code Generation Boundary

Current state:

- the docs say “generate specialized kernels from IR”
- they do not yet define what is generated versus handwritten

Why this matters:

- maintainability depends on a stable generation boundary
- otherwise the implementation can become half-generated, half-ad-hoc

Investigation tasks:

- define which artifacts are generated:
  - stage kernels
  - operator wrappers
  - state layouts
  - communication metadata
- define which artifacts remain handwritten:
  - runtime planner
  - transport layer
  - Python bindings
  - benchmark harness
- define cache keys and regeneration triggers
- define how generated code is tested and formatted

Deliverable target:

- `codegen-boundary-spec.md`

## Gap 10: Design-Parameter IR

Current state:

- inverse design and design-region ideas are discussed in multiple places
- there is no unified design-parameter schema

Why this matters:

- inverse design spans geometry, materials, filtering, projection, and adjoint recombination
- if this boundary is not explicit, gradient semantics will fragment

Investigation tasks:

- define latent parameter grids
- define parameter-to-geometry and parameter-to-material transforms
- define filter and projection operators
- define manufacturability constraints
- define how parameters map into design-region materialization
- define how gradients map back to latent variables

Deliverable target:

- `design-parameter-ir-schema.yaml`

## Gap 11: Mode Solver Completion

Current state:

- `vector-mode-solver-spec.md` is architecturally strong
- several production details are still open

Open areas:

- exact discrete operator variants by boundary condition
- lossy and leaky mode treatment
- PML treatment in eigenmode solves
- preconditioner and shift-invert strategy
- adjoint-through-eigensolve policy

Why this matters:

- the mode solver is a critical dependency for sources, monitors, and adjoint workflows

Investigation tasks:

- define the primary transverse formulations to support
- define exact boundary and normalization contracts
- define eigensolver backend expectations
- define reference validation problems and mode-orthogonality tests

Deliverable target:

- expanded `vector-mode-solver-spec.md` or a companion `mode-solver-math-spec.md`

## Gap 12: Silicon Photonics Device Canon

Current state:

- silicon photonics appears throughout the benchmark and compatibility docs
- there is no dedicated canonical device set

Why this matters:

- a high-performance package for this domain should have a stable set of “reference devices”
- this helps both benchmarking and numerical validation

Investigation tasks:

- define a canonical small set of devices:
  - waveguide bend
  - grating coupler
  - MMI
  - ring resonator
  - metalens
  - AWG or demux
- define for each:
  - geometry
  - grid policy
  - monitors
  - observables
  - acceptance tolerances

Deliverable target:

- `reference-device-suite.md`

## Gap 13: Reference Data Layout And Output Schema

Current state:

- state families are named
- exact in-memory and on-disk output layouts are not yet frozen

Why this matters:

- output compatibility, monitor interoperability, and frontend adapters all depend on predictable data shape contracts

Investigation tasks:

- define array shapes and axis orders for all field outputs
- define complex-data storage conventions
- define chunk-local and global monitor output schemas
- define metadata required for restart, replay, and benchmark logging

Deliverable target:

- `data-layout-and-output-schema.md`

## Priority Order

Highest priority:

1. solver IR schema
2. geometry IR schema
3. operator IR schema
4. chunk/halo protocol
5. code generation boundary
6. benchmark and correctness manifests

Second priority:

7. material plugin contracts
8. design-parameter IR
9. mode-solver completion
10. accuracy policy

Third priority:

11. silicon photonics reference suite
12. data layout and output schema
13. expanded frontend API manifests

## Final Recommendation

The next document wave should turn the current high-quality architectural notes into strict generation targets.

The right sequence is:

1. freeze schemas
2. freeze communication and output contracts
3. freeze benchmark and correctness manifests
4. only then ask an LLM to generate large implementation slices

Until those pieces exist, the current docs are best treated as a strong architectural design package, not yet as a complete code-generation package.
