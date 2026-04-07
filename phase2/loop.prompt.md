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
- Use `blocked` only when an external dependency (e.g., Phase 1 feature not yet wired) prevents further progress.
- Use `failed` when the attempt is unusable or the task direction was not successfully carried out.
- Prefer producing concrete artifacts and findings over generic summaries.
- Keep `working-memory.md` concise and deduplicated.
- Do not repeat reference material in `working-memory.md`; point to local files when possible.
- Prefer recording file paths, decisions, and distilled facts instead of copying long passages.

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
   - For multi-GPU tasks: uses `num_chunks=(2,1,1)` to split across 2 GPUs
   - Reports pass/fail based on success criteria
6. **Run the example** on the actual hardware (2 NVIDIA GPUs, 16 GB each)
7. **Verify the result** against the success criteria in the task
8. **Produce `result.json`** with:
   - `status`: `completed`, `needs_retry`, `blocked`, or `failed`
   - `summary`: concise outcome statement
   - `key_findings`: list of concrete discoveries (error values, convergence rates, pass/fail per variant)
   - `artifacts`: list of `{path, description}` for any produced files
   - `error_summary`: filled for blocked/failed, empty otherwise
   - `next_action`: `retry`, `human_review`, or `none`

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
| 2-GPU chunk decomposition | `num_chunks=(2,1,1)` argument | Splits domain across 2 GPUs |
| Halo exchange | Automatic with multi-chunk | Phase 1 task-064 |
| cells/s metric | `result.metrics.get("cells_per_second")` | From ExecutionResult |
| GaussianBeam | `from autofdtd.sources import GaussianBeam` | Phase 1 task-034 |
| PlaneWave | `from autofdtd.sources import PlaneWave` | Phase 1 task-033 |
| BlochBoundary | `from autofdtd.boundaries import BlochBoundary` | Phase 1 task-020 |
| PML | `from autofdtd.boundaries import PML` | Phase 1 task-021 |
| Lorentz/Drude | `from autofdtd.materials import LorentzMedium, DrudeMedium` | Phase 1 task-016 |
| ModeSource | `from autofdtd.sources import ModeSource` | Phase 1 task-032 (API done, wiring may need verification) |
| FieldProjectionCartesianMonitor | `from autofdtd.monitors import FieldProjectionCartesianMonitor` | Phase 1 task-042 |

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
| L₂ field error vs. analytical | < 1% at finest resolution |
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
  "artifacts": [
    {"path": "phase2/examples/example_XXX_YYY.py", "description": "Example script"},
    {"path": "phase2/tools/convergence.py", "description": "Convergence runner"}
  ],
  "error_summary": "",
  "follow_up_notes": "",
  "next_action": "none"
}
```

## Runtime Context

- The runner injects the current task's title, success criteria, and notes from `loop.spec.md` on every iteration.
- Use `phase2/loop.spec.md` as the authoritative task board.
- Use `phase2/validation-set.md` as the detailed example specifications.
- Use `phase2/runtime-requirements.md` for per-example infrastructure requirements.
- The Phase 1 GPU infrastructure is assumed working — do not re-implement GPU initialization, Warp kernels, or halo exchange.
- If a Phase 1 feature (e.g., ModeSource wiring, N2F projection) does not work as expected, report it as a finding and mark `needs_retry` with a clear `error_summary`.
