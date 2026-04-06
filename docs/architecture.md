# Architecture

The package scaffold is split so later Phase 1 work can grow without collapsing public
API, compilation, and execution into one layer.

## Subsystems

| Module | Responsibility |
| --- | --- |
| `autofdtd.api` | Public object-model entrypoints and compatibility-facing names |
| `autofdtd.core` | Shared tagged-model base classes, serialization helpers, validators |
| `autofdtd.geometry` | Analytic geometry families and wrappers |
| `autofdtd.materials` | Isotropic, dispersive, and anisotropic material models |
| `autofdtd.boundaries` | Boundary containers and boundary-family models |
| `autofdtd.sources` | Source-time profiles and source object models |
| `autofdtd.monitors` | Monitor object models and postprocessing metadata |
| `autofdtd.grid` | Grid planning and subpixel policy surfaces |
| `autofdtd.ir` | Versioned execution IR independent from the public API |
| `autofdtd.compiler` | Scene normalization, lowering, and chunk planning |
| `autofdtd.runtime` | Chunk placement, scheduling, stepping, and execution metrics |
| `autofdtd.kernels` | Warp module conventions and staged kernel families |
| `autofdtd.diagnostics` | Structured logs, progress, validation reports, timing data |

## Design Constraints

- Keep scene modeling, scene compilation, runtime stepping, and IO separate.
- Treat Tidy3D compatibility as a tagged typed-object problem, not a loose dict problem.
- Preserve room for chunk-local interior stages, boundary stages, and future overlap-aware scheduling.
- Make unsupported features explicit through validation or feature-matrix policy.

## Current Backbone

The first public models live in `autofdtd.api` and `autofdtd.core`:

- `Structure` carries `geometry`, `medium`, optional `name`, optional explicit `priority`, and
  optional `background_medium`.
- `Scene` carries background `medium`, an ordered tuple of `structures`, and
  `structure_priority_mode`, plus lookup and resolution-order helpers.
- `Simulation` extends `Scene` with domain size and center, runtime controls, sources,
  monitors, boundary and grid references, symmetry, and version tagging.

These models are frozen, tagged, and JSON-ready so later tasks can lower them into a
separate execution IR without redesigning the public container layer.

`autofdtd.core.validation` now provides the first checklist-aware normalization layer for
these containers. It trims names, normalizes basic vector payloads, validates primitive
geometry constraints when enough shape data is present, rejects deferred or rejected
feature tags explicitly, warns when known primitive bounds extend outside the simulation
domain, and coerces zero-thickness absorbing boundaries to `Periodic` so later compiler
work does not inherit ambiguous input states.

The grid namespace now adds a separate manual-discretization seam. `GridSpec` resolves
three axis-local grid models into a `ResolvedGrid` object without entangling that work
with runtime scheduling. That keeps scene modeling, mesh planning, and execution as
distinct subsystems while giving later compiler and mode-solver tasks a concrete domain
discretization contract to consume.

`autofdtd.grid.subpixel` now carries the first explicit interface-materialization policy
surface. `Simulation.subpixel` accepts `bool` values using Tidy3D-style coercion
(`True` -> default `SubpixelSpec()`, `False` -> `SubpixelSpec.staircasing()`) plus an
explicit `SubpixelSpec` object. Phase 1 keeps the policy set narrow: dielectric
interfaces can use `PolarizedAveraging` or `Staircasing`, while metal, PEC, PMC, and
lossy-metal interfaces remain explicit staircasing-only buckets until anisotropic media
and heavier conformal treatments land.

The material namespace now has a real Phase 1 foothold. `Medium`, `PECMedium`, and
`PMCMedium` are typed public models, lower into typed IR, compile into scalar
constitutive coefficients, and feed a staged electric/magnetic constitutive-update path
in `autofdtd.kernels.materials`. That keeps material validation, scene assignment,
coefficient preparation, and runtime updates as distinct seams instead of hiding them in
structure dict payloads.

The boundary namespace now has the first runtime-meaningful boundary subset. `Periodic`,
`PECBoundary`, `PMCBoundary`, `Boundary`, and `BoundarySpec` are typed public models;
`autofdtd.ir` lowers them into typed boundary IR; `autofdtd.compiler.boundaries`
compiles them into per-face halo metadata; and `autofdtd.kernels.boundaries` applies
periodic-copy or conductor-reflection ghost updates without folding boundary behavior
into the constitutive kernels.

The runtime namespace now also has an explicit stop-policy seam separate from the
future Maxwell update loop. `Simulation` exposes derived `scaled_courant`,
`time_step_size()`, and `num_time_steps()` helpers; `autofdtd.compiler.runtime`
compiles those controls into a `CompiledRuntimeControls` record; and
`autofdtd.runtime.controls` evaluates run-time, step-count, and shutoff-based early
termination without coupling that policy to any particular chunk scheduler.

## Execution IR

`autofdtd.ir` now defines a versioned transport layer around the public containers:

- `ComponentIR` is the family-tagged envelope for geometry, medium, source, monitor,
  boundary, grid, and subpixel payloads.
- `StructureIR`, `SceneIR`, and `SimulationIR` preserve ordered scene semantics while
  remaining independent from the public API module layout.
- `ExecutionManifestIR` and `ExecutionPackageIR` provide a bundle contract for local or
  remote execution packaging with a manifest, an entrypoint payload, and hashed artifact
  metadata.

The initial lowering path is snapshot-oriented rather than compilation-oriented. It
captures a normalized, inspectable execution payload early so later compiler tasks can
add chunk plans, coefficient buffers, and runtime schedules without changing the package
transport contract.
