# Boundary Scope

Phase 1 now has an explicit boundary surface for non-absorbing, phase-aware, and
absorbing edges built around `Periodic`, `BlochBoundary`, `PECBoundary`,
`PMCBoundary`, `ABCBoundary`, `PML`, `StablePML`, `Absorber`, `Boundary`, and
`BoundarySpec`.

## Public API

- `Periodic`, `BlochBoundary`, `PECBoundary`, `PMCBoundary`, `ABCBoundary`,
  `PML`, `StablePML`, and `Absorber` are typed boundary-edge models.
- `Boundary` pairs `minus` and `plus` edges along one axis.
- `BoundarySpec` carries `x`, `y`, and `z` axis boundaries and defaults to periodic
  behavior on all sides in the currently implemented subset.
- `ABCBoundary` is implemented as a practical first-order subset. Runtime
  compilation currently requires explicit `permittivity`, and optional
  `conductivity` is only accepted when `permittivity` is also supplied.
- If a periodic edge is placed opposite a PEC or PMC edge, the periodic edge is
  coerced to the conductor edge with a warning so later runtime handling stays
  reflection-based instead of mixing halo semantics on one axis.
- `StablePML` keeps the same parameter surface as `PML` but uses a more conservative
  default preset.
- `Absorber` uses a simpler conductivity-only profile and defaults
  `extrude_structures=False`, matching the local Tidy3D schema surface.
- `ModeABCBoundary`, `BroadbandModeABCSpec`, and
  `BroadbandModeABCFitterParam` are parseable compatibility surfaces, but remain
  deferred until the mode-solver tasks land.

## Lowering And Runtime

- `autofdtd.ir` lowers the supported subset into typed `PeriodicIR`,
  `BlochBoundaryIR`, `PECBoundaryIR`, `PMCBoundaryIR`, `ABCBoundaryIR`,
  `PMLIR`, `StablePMLIR`, `AbsorberIR`, `BoundaryIR`, and `BoundarySpecIR`
  payloads.
- `autofdtd.compiler.boundaries.compile_boundary_spec()` turns the public boundary
  surface into a runtime plan with explicit per-face mode tags, sign transforms,
  halo depth, stage ordering, and per-face absorbing profile metadata.
- `autofdtd.kernels.boundaries.apply_boundary_ghosts()` applies the compiled plan to
  vector fields with one ghost cell on each side.
- Periodic boundaries copy ghost cells from the opposite interior face.
- Bloch boundaries apply explicit complex phase factors on wrapped ghost values.
- PEC and PMC boundaries use reflection-sign conventions so tangential-versus-normal
  field parity is explicit instead of being hidden in a monolithic timestep kernel.
- `ABCBoundary` uses a stateful first-order ghost-cell recurrence with an
  explicit effective-medium coefficient. It is intended as a practical subset
  for simple termination cases, not as a replacement for PML.
- `PML` and `StablePML` share the staged damping-layer backend and differ through
  compiled coefficient presets.
- `Absorber` uses the same staged region-update seam but with conductivity-only
  attenuation coefficients and a reflective outer wall.

## Current Limits

- Runtime compilation currently supports `Periodic`, `BlochBoundary`,
  `PECBoundary`, `PMCBoundary`, `ABCBoundary`, `PML`, `StablePML`, `Absorber`,
  `Boundary`, and `BoundarySpec`.
- `ABCBoundary` without explicit `permittivity` still lowers through the API and
  IR, but runtime compilation fails clearly because automatic medium inference is
  not wired into Phase 1 scene compilation yet.
- `StablePML` is currently implemented as a preset-driven extension of the baseline
  staged PML backend rather than a separate split-field formulation.
- `Absorber` is currently implemented as a conductivity-ramp absorbing subset with a
  reflective outer wall, not a full material-aware adiabatic absorber.
- `ModeABCBoundary` remains outside the runtime compiler path and fails
  explicitly until the mode-solver tasks land.
- The kernel helpers assume vector fields shaped `(nx, ny, nz, 3)` with one ghost
  cell per face.
