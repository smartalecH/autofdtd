# Runtime Logging

Phase 1 now keeps structured execution logging separate from benchmark metrics so
debugging, validation, and stop-behavior inspection have their own stable seam.

## Structured Records

- `autofdtd.diagnostics.RuntimeProgressLogger` collects immutable progress events
  and convergence samples against a compiled runtime-control plan.
- `RuntimeLogEvent` records the emitted progress stream with step index, physical
  time, completed-step count, progress fraction, stop metadata, integrated field
  intensity, peak intensity, and shutoff ratio when available.
- `RuntimeConvergenceEvidence` stores the check history separately from the
  progress stream so callers can inspect every shutoff evaluation even when
  progress emission is downsampled.
- `RuntimeExecutionLog` bundles the compiled stop-policy metadata, final stop
  decision, emitted events, accumulated convergence evidence, and wall-clock
  elapsed time for the control loop itself.

## Execution Helper

- `autofdtd.runtime.run_until_stop_with_logging()` mirrors the existing
  precomputed-intensity control-loop helper but returns a `RuntimeExecutionLog`
  instead of only the last stop decision.
- Progress emission is explicit and deterministic: Phase 1 emits at step `0`, at
  every configured `progress_interval`, and at the final stop event.
- Shutoff evidence is captured whenever the runtime controller performs a
  convergence check, regardless of whether that step also emitted a progress
  record.

## Current Limits

- This logging path currently wraps the runtime-control seam, not a full Maxwell
  field stepper. Later timestep-loop work should call the same logger instead of
  inventing a separate progress format.
- The recorded wall-clock time is for the Python control loop and logging path,
  not a performance benchmark. Initialization, JIT, and throughput reporting stay
  in the benchmark-focused tasks.
