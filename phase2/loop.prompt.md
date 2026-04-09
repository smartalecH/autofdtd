# Phase 2 Accuracy Validation Loop

## Mission

Drive Phase 2 of autofdtd to validate that the Phase 1 implementation produces correct and accurate simulation results. Phase 2 runs 18 curated 3D examples against analytical, semi-analytical, and reference-solver ground truth, measures convergence rates and weak-scale throughput, validates multi-GPU halo correctness, and produces a clean pass/fail report for each example and permutation on a 2×16 GB GPU machine.

## Operator Policies

- Failure policy: `retry`
- Permission mode: `danger-full-access`
- End-of-list behavior: `stop cleanly`
- Codex working directory: `..` (repo root relative to `phase2/`)

## Global Execution Rules

- Read relevant files before editing or running.
- Keep progress notes concise and factual.
- Treat `loop.spec.md` as the queue source of truth.
- Report output that matches `output.schema.json`.
- Use `completed` only when the task's success criteria were actually satisfied.
- Use `needs_retry` when meaningful work was done but the success criteria are not yet fully met.
- Use `blocked` only when an external dependency (e.g., hardware limitation, not a code bug) prevents further progress.
- Use `failed` when the attempt is unusable or the task direction was not successfully carried out.
- Prefer producing concrete artifacts and findings over generic summaries.
- Keep `working-memory.md` concise and deduplicated.
- Do not repeat reference material in `working-memory.md`; point to local files when possible.
- Prefer recording file paths, decisions, and distilled facts instead of copying long passages.
- **Fix Phase 1 bugs when you encounter them**: If an example task hits a Phase 1 bug (see Known Phase 1 Bugs below), fix the bug in-place, then retry the task. Do not mark `blocked` and move on — you are the one tasked with fixing it. Only mark `blocked` for non-code dependencies like missing hardware or external services.
- **Result file precedence**: If `phase2/results/task-XXX-result.json` exists and was written by the agent, its `status` field is authoritative. Fabricate a result only when the agent did not produce one. Never overwrite an honest result with `status: completed`.

## Per-Task Execution Protocol

For each task, you must:

1. **Read the task definition** in `phase2/loop.spec.md` for the specific task ID.
2. **Read the relevant example context** in `phase2/validation-set.md` (Example N).
3. **Implement the reference library** if the task requires it (tasks 201-203): create `phase2/reference/*.py` with the analytical/semi-analytical implementation.
4. **Implement the tool** if the task requires it (tasks 204-206): create `phase2/tools/*.py` with the sweep runner.
5. **Implement the example** if the task requires it: create `phase2/examples/example_*.py` that:
   - Defines the Simulation with correct geometry, sources, monitors, boundaries
   - Calls `compile_simulation(sim)` → `run_compiled_simulation(compiled)`
   - Extracts `ExecutionResult` fields
   - Computes error vs. ground truth reference
   - Reports pass/fail based on success criteria
6. **Run the example** on the actual hardware (2 NVIDIA GPUs, 16 GB each)
7. **If execution fails or produces near-zero fields**:
   a. **Check for source injection bug first**: if `E_max ≈ 0`, `shutoff` triggers early, or FluxMonitor returns zero — this is the Bug 7 source injection bug. Fix `PlaneWave`/`GaussianPulse` amplitude handling in `src/autofdtd/sources/`. Try at least 3 distinct fix approaches (see Bug 7 list) before considering it unfixable.
   b. **If it's a Phase 1 bug** (see Known Phase 1 Bugs below): fix the bug in-place, retry immediately. Do NOT mark `blocked`.
   c. **If it's any other implementation bug**: fix it in-place in the Phase 1 source code or the example script. Retry at least 2 more times with different fix approaches before marking `needs_retry`.
   d. **Only mark `blocked`** when you have genuinely exhausted all reasonable fix attempts AND the blocker is an external dependency (hardware missing) or a Phase 3 architectural gap — NOT a code bug in Phase 1.
   e. **Document every attempted fix** in `key_findings` — what you tried, what failed, what the error was.
8. **Verify the result** against the success criteria in the task
9. **Produce `result.json`** with:
   - `status`: `completed`, `needs_retry`, `blocked`, or `failed`
   - `summary`: concise outcome statement
   - `key_findings`: list of concrete discoveries (error values, convergence rates, pass/fail per variant)
   - `artifacts`: list of `{path, description}` for any produced files
   - `error_summary`: filled for blocked/failed, empty otherwise
   - `next_action`: `retry`, `human_review`, or `none`
10. **Update `Blocked By:` in loop.spec.md**: If this task unblocks other tasks (e.g., you fixed a Phase 1 bug that other tasks were blocked by), update those tasks' `Blocked By:` field to remove this task's ID. The loop auto-unblocks tasks when their blockers are COMPLETED — but only if the `Blocked By:` field is accurate.

## Hardware Context

- Machine has 2 NVIDIA GPUs, each with 16 GB VRAM
- All examples must fit in 16 GB per GPU (or 32 GB total for 2-GPU examples)
- Use `nvidia-smi` to check GPU availability before running
- Use `python -c "from autofdtd.kernels.backend import backend_info; print(backend_info())"` to verify Warp/CUDA availability

## Phase 1 Baseline

The following Phase 1 capabilities are available and should be used as-is:

| Capability | Import | Notes |
|---|---|---|
| Simulation, Scene, Structure | `from autofdtd.api import Simulation, Scene, Structure` | Public API |
| compile_simulation | `from autofdtd.api import compile_simulation` | Compiles to runtime artifacts |
| run_compiled_simulation | `from autofdtd.runtime import run_compiled_simulation` | Executes on GPU |
| GPU execution | `autofdtd.kernels.backend` | Warp backend, CUDA |
| Halo exchange | ChunkHaloExchange, build_chunk_layout | Internal Phase 1 API; multi-GPU via `num_chunks` |
| cells/s metric | `result.metrics.get("cells_per_second")` | From ExecutionResult |
| GaussianBeam | `from autofdtd.sources import GaussianBeam` | Phase 1 task-034 |
| PlaneWave | `from autofdtd.sources import PlaneWave` | Phase 1 task-033 |
| BlochBoundary | `from autofdtd.boundaries import BlochBoundary` | Phase 1 task-020 |
| PML | `from autofofdtd.boundaries import PML` | Phase 1 task-021 |
| Lorentz/Drude | `from autofdtd.materials import LorentzMedium, DrudeMedium` | Phase 1 task-016 |
| ModeSource | `from autofdtd.sources import ModeSource` | Phase 1 task-032 |
| FieldProjectionCartesianMonitor | `from autofdtd.monitors import FieldProjectionCartesianMonitor` | Phase 1 task-042 |

## Known Phase 1 Bugs

When you encounter one of these errors, fix the bug in the Phase 1 source code, then retry — do not mark `blocked`.

### Bug 1: halo_check uses numpy-style item assignment on Warp arrays
**Error pattern**: `"Item indexing is not supported on wp.array"`, `"array object does not support item assignment"`, `TypeError` on `dst_field[:] = ...`
**Files**: `phase2/tools/halo_check.py` (`_run_chunked_simulation`), `src/autofdtd/runtime/boundaries.py` (`cross_device_transfer`, `unpack_halo`)
**Fix**: For Warp arrays, use `.numpy()` to read/write instead of `np.array()` or direct slice assignment. Detect Warp arrays with `hasattr(arr, 'numpy')`. Example: `src_data = packed.numpy()` instead of `np.array(packed)`. For `dst_field[:] = value` on Warp arrays, copy via numpy first then assign.
**Also affects**: `halo_check._exchange_face_halo` — same-device halo copy uses `E_full[dst_slice] = E_full[src_slice]` which fails on Warp arrays.

### Bug 2: compile_simulation hardcodes num_chunks=(1,1,1)
**Error pattern**: `compile_simulation` has no `num_chunks` parameter; multi-GPU examples cannot be tested
**File**: `src/autofdtd/compiler/pipeline.py` line ~732
**Fix**: Add `num_chunks` parameter to `compile_simulation`. Change `num_chunks=(1, 1, 1)` to use the parameter value. Ensure `run_compiled_simulation` accepts and passes through `num_chunks` if needed.

### Bug 3: symmetry_transform() not called in execution loop
**Error pattern**: Symmetry-reduced domains produce identical results to no-symmetry runs (max_error=0)
**File**: `src/autofdtd/runtime/execution.py` — the main timestep loop never calls `symmetry_transform()`
**Fix**: After the H-field update in `run_compiled_simulation`, call `symmetry_transform()` for each active symmetry axis. The function exists at `src/autofdtd/kernels/boundaries.py:224`. Use the `CompiledSymmetryAxis` from `compiled.compiled_boundaries.boundary_spec.symmetry_axes`.

### Bug 4: FluxMonitor.accumulate_flux uses Warp item indexing
**Error pattern**: `"Item indexing is not supported on wp.array objects"` at `electric_field[idx][component]`
**File**: `src/autofdtd/kernels/monitors.py` line ~203
**Fix**: Use flat storage indexing: `electric_field.storage[idx * 3 + component]` instead of `electric_field[idx][component]`. Same for magnetic field.

### Bug 5: ModeSource.solve_modes() returns radiation modes (neff ~ 1.0)
**Error pattern**: ModeSource compilation fails with "found no modes" or neff ~ 0.999 for all found modes
**File**: `src/autofdtd/modes/solver.py` `solve_modes()` function
**Root cause**: Eigensolver uses `sigma="SR"` (smallest real eigenvalue). Radiation modes (neff = n_clad) have eigenvalues closest to zero — they're found first and filtered out as "not guided". Guided modes have more negative eigenvalues.
**Fix**: Either (a) filter out radiation modes by checking `n_clad < neff < n_core` for each solution, or (b) use `target_neff` to aim the eigensolver at the guided-mode region (`sigma = (target_neff * k)**2`), or (c) sort by `beta_sq` ascending and skip the first few radiation modes.

### Bug 6: N2F compile_projection_monitor positional args
**Error pattern**: `TypeError` when compiling FieldProjectionCartesianMonitor
**File**: `src/autofdtd/compiler/pipeline.py` line ~470
**Fix**: Extract individual fields from the monitor object: `compile_projection_monitor(name=monitor.name, monitor_type=monitor.type, center=monitor.center, size=monitor.size, ...)` — do NOT pass the monitor object itself as the first positional arg.

### Bug 7: Source injection produces near-zero fields at optical wavelengths
**Error pattern**: `E_max ≈ 0`, `shutoff` triggers after few steps, FluxMonitor returns zero or near-zero R/T
**Root cause (a)**: `PlaneWave` uses `float(amplitude.real)` — when `GaussianPulse.amp_time` returns a purely imaginary value at t=0 (e.g., `j * exp(-t²)`), `amplitude.real = 0`, so the injected amplitude is zero.
**Root cause (b)**: Float32 precision floor (~1e-7) causes field values below ~1e-7 to underflow to zero before they can propagate.
**Root cause (c)**: `shutoff` criterion triggers at step ~170 before wave fills the domain, freezing the simulation.
**Files**: `src/autofdtd/sources/plane.py` (`PlaneWave` class), `src/autofdtd/sources/source.py` (`GaussianPulse.amp_time`), `src/autofdtd/runtime/execution.py` (shutoff logic)
**Fix attempts to try**:
1. Use `amplitude.imag` instead of `amplitude.real` for purely imaginary source pulses, OR add a π/2 phase offset to the source
2. Increase `shutoff` to `1e-10` or disable it (`shutoff=None`)
3. Use `float(amplitude)` instead of `float(amplitude.real)` — take the magnitude
4. Scale the grid so cells are larger (fewer cells → larger field values per cell)
5. Increase `run_time` to allow wave to fill domain before shutoff triggers
6. Use `amplitude=1.0` instead of `GaussianPulse` with a step or continuous wave source
7. Check: if `PlaneWave` has `amplitude` parameter, ensure it's a real positive number, not a complex whose `.real` is 0
**Impact**: Blocks tasks 101, 106, 107, 110, 112, 114, 115, 118 from producing meaningful field amplitudes. You MUST fix this before reporting ground-truth accuracy results.

---

## Retry-before-Blocking Policy

Before marking a task `blocked` or `needs_retry` for a non-infrastructure reason, you MUST:

1. **Attempt at least 3 distinct fix approaches** for any implementation bug you encounter (not just mark it blocked and move on)
2. **Fix Phase 1 bugs in-place** when you encounter them — this is not optional. If `PlaneWave` injection produces zero fields, fix `PlaneWave`. If `ModeSource` produces wrong modes, fix the mode solver.
3. **Only mark `blocked`** when you have genuinely exhausted all reasonable fix attempts AND the blocker is an external dependency (hardware, external service) or a Phase 3 architectural gap — NOT a code bug that can be fixed in Phase 1.
4. **Document each attempted fix** in `key_findings` so future iterations can see what was tried

If a bug affects multiple example tasks (e.g., source injection bug), fix it once in Phase 1 and then retry all affected tasks — do not file each one as blocked separately.

---

## Reference Library Specifications

### `phase2/reference/tmm.py`

Implement:
- `planar_multilayer_RT(n_layers, d, n, theta, wavelength, pol)` → R, T arrays
- `directional_coupler_C(L_coupling, n_eff_even, n_eff_odd)` → coupling coefficient C
- `directional_coupler_power(P0, C, z)` → P1(z), P2(z) arrays
- `bragg_grating_RT(L, period, n_eff, kappa, wavelength)` → R, T arrays

### `phase2/reference/mie.py`

Implement:
- `mie_efficiencies(n, k, radius, wavelength)` → Q_ext, Q_sca, Q_abs
- `mie_angular_S(n, k, radius, wavelength, theta, phi)` → S1, S2 scattering amplitudes
- `mie_RCS_dB(n, k, radius, wavelength, theta, phi)` → σ(θ, φ) in dB
- `multipole_decomposition(E, H, geometry)` → p, m, Q_e, Q_m coefficients

### `phase2/reference/rayleigh_sommerfeld.py`

Implement:
- `rayleigh_sommerfeld_near2far(E_aperture, wavelength, k, r_obs)` → far-field E, H
- `zone_plate_focal_spot(NA, wavelength)` → focal_spot_size, focal_length
- `zone_plate_fresnel_number(NA, wavelength, z)` → Fresnel number

## Tool Specifications

### `phase2/tools/convergence.py`

```
run_convergence_sweep(simulation_fn, resolutions, ground_truth_fn, error_metric="L2") → dict
  resolutions: list of dl or ppw values
  ground_truth_fn: callable(sim, resolution) → reference value
  error_metric: "L2" | "max" | "R" | "T"
  Returns: {resolutions, errors, rates, pass_fail}
```

### `phase2/tools/weak_scale.py`

```
run_weak_scale_sweep(simulation_fn, domain_sizes, fixed_resolution) → dict
  domain_sizes: list of (Lx, Ly, Lz) tuples
  Returns: {sizes, cells_per_second, pass_fail}
```

### `phase2/tools/halo_check.py`

```
run_halo_check(simulation_fn, single_gpu=True, two_gpu=True) → dict
  Returns: {max_error, mean_error, pass_fail, single_gpu_fields, two_gpu_fields}
```

## Example Naming Convention

All example scripts go in `phase2/examples/`:
- `example_101_base.py` — Example 1 baseline
- `example_101_conv.py` — Example 1 convergence
- `example_101_weak.py` — Example 1 weak-scale
- `example_101_mgpu.py` — Example 1 multi-GPU
- etc.

Each script must be standalone runnable: `python phase2/examples/example_XXX_YYY.py`

## Success Criteria Thresholds

| Metric | Threshold |
|---|---|
| L₂ field error vs. analytical | < 1% at finest grid |
| Reflectance/Transmittance error | < 0.5% vs. Fresnel/TMM |
| Convergence rate | ≥ 1.5th order (2nd-order FDTD) |
| Multi-GPU field error vs. single-GPU | < 1e-6 |
| Weak-scale throughput regression | < 10% drop across domain sizes |
| `cells/s` weak scaling efficiency | > 0.9 across 2 GPUs |

## Output Schema

The `result.json` for each task must conform to:

```json
{
  "task_id": "task-XXX",
  "status": "completed | needs_retry | blocked | failed",
  "summary": "One-line outcome",
  "key_findings": [
    "Error vs. ground truth: X% at finest grid",
    "Convergence rate: X.XX (≥1.5 required)",
    "cells/s: X.XX ± X% across domain sizes",
    "Multi-GPU max error: X.XXe-X (threshold 1e-6)"
  ],
  "ground_truth_error": {
    "metric": "R | T | Q_sca | a1 | b1 | focal_spot | Q_factor | S21 | neff | sigma_theta | <name>",
    "fdtd_value": 0.5695,
    "reference_value": 0.5712,
    "absolute_error": 0.0017,
    "relative_error": 0.003,
    "threshold": 0.005,
    "pass": true,
    "reference_source": "Fresnel equations | Mie series | TMM | Rayleigh-Sommerfeld | Published paper"
  },
  "provenance": {
    "script_sha256": "abc123...",
    "field_data_sha256": "def456...",
    "grid_shape": [80, 40, 40],
    "total_cells": 128000,
    "num_steps": 500,
    "num_sources": 1,
    "num_monitors": 4,
    "E_max": 1.23e-3,
    "E_mean": 4.56e-5,
    "shutoff_reason": "shutoff | max_steps",
    "runtime_seconds": 12.5
  },
  "artifacts": [
    {"path": "phase2/examples/example_XXX_YYY.py", "description": "Example script"},
    {"path": "phase2/results/fields/task-XXX_E.npz", "description": "Raw E-field output"},
    {"path": "phase2/results/metrics/task-XXX_metrics.json", "description": "Execution metrics"}
  ],
  "error_summary": "",
  "follow_up_notes": "",
  "next_action": "none"
}
```

**`ground_truth_error` is REQUIRED** for all example tasks (101-118). Omitting it or filling in self-consistency metrics (e.g., "single-GPU vs 2-GPU error < 1e-6") is not valid and will force `needs_retry`.

**`provenance` is REQUIRED** — it proves the physics actually ran. Required fields:
- `script_sha256`: SHA-256 of the example script that was executed
- `field_data_sha256`: SHA-256 of the raw field data file (`.npz`) — this is the anti-fabrication field
- `E_max`, `E_mean`: Peak and mean field values — if E_max < 1e-10, the source injection failed and the comparison is meaningless
- `runtime_seconds`: Wall-clock time — proves the simulation actually ran, not just a script that printed numbers
- `grid_shape`, `total_cells`, `num_steps`: Grid metadata — must be non-trivial (total_cells > 1000, num_steps > 10)

**Fabrication detection**: If `ground_truth_error.pass=true` but `provenance.E_max < 1e-10`, the result is marked `needs_retry` because the source injection bug produced zero fields — any ground-truth comparison from that run is meaningless.

**On existing results**: The files in `phase2/results/` dated before 2026-04-08 are from the OLD run and contain inflated self-consistency-only results. **Ignore them entirely.** Do not read `phase2/results/task-XXX-result.json` from previous runs. Each task retry starts fresh — the only ground truth that matters is what you compute this run.

## Runtime Context

- The runner injects the current task's title, success criteria, and notes from `loop.spec.md` on every iteration.
- Use `phase2/loop.spec.md` as the authoritative task board.
- Use `phase2/validation-set.md` as the detailed example specifications.
- Use `phase2/runtime-requirements.md` for per-example infrastructure requirements.
- **IGNORE all existing `phase2/results/task-XXX-result.json` files dated before 2026-04-08.** They contain inflated self-consistency-only results from the previous run and must not influence your approach. Each task retry is a fresh execution.
- The Phase 1 GPU infrastructure is assumed working — do not re-implement GPU initialization, Warp kernels, or halo exchange.
- If a Phase 1 feature does not work as expected, it is a bug — fix it, then retry. Only mark `blocked` for non-code dependencies (hardware missing, external service unavailable).
- Each `result.json` must include `provenance` with real execution metadata (grid shape, cell count, E_max, runtime). This is required to prove the physics ran — do not fabricate this data.
