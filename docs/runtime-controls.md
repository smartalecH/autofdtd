# Runtime Controls

Phase 1 now has an explicit stop-policy seam for `Simulation` objects instead of
leaving timestep count and early-stop behavior implicit inside the future solver
loop.

## Public Surface

- `Simulation.run_time` remains the compatibility-facing runtime input and is still
  required.
- `Simulation.shutoff` now defaults to `1e-5`, matching the local Tidy3D schema
  surface. Set it to `0` to disable shutoff-based early termination.
- `Simulation.scaled_courant()`, `Simulation.time_step_size()`, and
  `Simulation.num_time_steps()` expose the derived runtime-control values needed by
  downstream compiler and runtime code.
- `normalize_index` is preserved on the simulation shell and carried into the IR and
  compiled runtime-control metadata so later source normalization work can consume a
  stable selector.

## Compilation And Runtime

- `autofdtd.compiler.runtime.compile_runtime_controls()` compiles the public runtime
  settings into a `CompiledRuntimeControls` record.
- The timestep estimate uses the resolved grid and the standard explicit Yee CFL
  form over the active axes. The current Phase 1 path applies subpixel Courant
  scaling but does not yet add material-dependent CFL relaxation.
- `autofdtd.runtime.controls.RuntimeController` evaluates stop decisions against
  either the run-time-derived step budget or an explicit step-count override.
- `autofdtd.runtime.controls.integrated_electric_field_intensity()` implements the
  Phase 1 shutoff metric: the instantaneous integrated electric-field intensity.
- Shutoff convergence follows the local Tidy3D semantics: compare the current
  integrated electric-field intensity against the historical peak and stop when the
  ratio falls below `Simulation.shutoff`.

## Current Limits

- Automatic `RunTimeSpec`-style heuristics are still out of scope. Phase 1 accepts a
  concrete `run_time` only.
- Material-aware CFL relaxation is deferred. The current timestep path assumes the
  resolved grid and effective Courant factor are the dominant control inputs.
- Shutoff convergence is intentionally narrow: it only covers peak-normalized
  integrated electric-field decay. Modal convergence, frequency-domain convergence,
  and resonant heuristics remain future work.
- The runtime controller works with precomputed intensity samples or caller-supplied
  electric fields. It does not yet own a full Maxwell stepping loop.
