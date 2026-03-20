# Performance Benchmarking Spec

This document defines how to benchmark a GPU-first FDTD package so that performance claims are interpretable across hardware, solver variants, and workload classes. The goal is not just to report wall-clock time, but to know:

- whether a given result is good relative to hardware limits
- whether regressions have occurred
- how much headroom remains
- which subsystem is limiting performance
- whether a speedup applies to the core timestep, the full solver, or only an end-to-end workflow

The benchmarking strategy should combine:

- hardware-aware throughput modeling from `papers/gpu_benchmarking.pdf`
- realistic large-workload references from `metalens` and `tidy3d-notebooks`
- regression tracking patterns already present in `Khronos.jl/benchmark`

## References

- `papers/gpu_benchmarking.pdf`
- `papers/systolic_fdtd.pdf`
- `metalens/README.md`
- `metalens/Metalens_Optimize.py`
- `Khronos.jl/README.md`
- `Khronos.jl/benchmark/benchmark_utils.jl`
- `Khronos.jl/src/Simulation.jl`
- selected workloads from `tidy3d-notebooks`

## Core Principle

Do not collapse all performance reporting into one number.

At minimum, separate:

1. kernel-level performance
2. core timestep throughput
3. full simulation throughput
4. end-to-end workflow throughput

These are different measurements and should not be mixed.

## Primary Figure Of Merit

The most important primitive benchmark is Yee-cell update throughput:

```math
S_{\mathrm{cells}} = \frac{N_{\mathrm{cells}} \, N_{\mathrm{timesteps}}}{t_{\mathrm{wall}}},
```

reported in:

- `MCells/s`
- `GCells/s`
- optionally `TCUPS` for very large runs

This should be reported for:

- warm steady-state iterations only
- with a clear statement of whether setup time, JIT time, or I/O time is included

## Hardware-Aware Performance Model

The GPU benchmarking paper gives a simple but useful performance model:

```math
S = \min\left(\frac{F}{N_{\mathrm{fl}}}, \frac{B}{N_{\mathrm{bw}}}\right),
```

where:

- `F` is achievable FLOPS
- `B` is achievable memory bandwidth
- `N_fl` is the floating-point work per cell update
- `N_bw` is the bytes transferred per cell update

For the simplest Yee update in a homogeneous nondispersive medium without PML, the paper uses:

- `N_fl = 36`
- `N_bw = 144` bytes for single precision

and emphasizes that FDTD is strongly bandwidth-limited on modern hardware.

This model should be included in the benchmark harness because it gives a first-order answer to:

- whether the measured speed is plausible
- whether the implementation is near the expected memory-bandwidth ceiling
- whether a benchmark is limited by compute, memory, or underutilization

## Required Benchmark Metrics

Every benchmark result row should include:

- hardware id
- backend id
- precision
- solver version / git commit
- benchmark name
- domain dimensions
- effective cell count
- timestep count
- runtime seconds
- cell-update throughput
- peak device memory used
- number of GPUs
- communication volume if multi-GPU
- setup/JIT time
- solve time excluding setup
- monitor/postprocess time

Recommended additional metrics:

- achieved bandwidth estimate
- achieved throughput as fraction of model prediction
- halo exchange fraction of runtime
- monitor/DFT fraction of runtime
- energy or power if available

## Benchmark Tiers

### Tier 0: Microbenchmarks

Purpose:

- isolate single kernels or stage families

Examples:

- plain interior Yee curl
- constitutive `D -> E` and `B -> H`
- PML update stages
- source injection
- DFT accumulation
- halo pack/unpack

Metrics:

- effective cells/s
- bytes/s
- occupancy and launch shape metadata

These catch local regressions and are the fastest to run in CI.

### Tier 1: Synthetic Solver Benchmarks

Purpose:

- measure core timestep throughput without large geometry or workflow overhead

Examples:

- homogeneous box, no PML
- homogeneous box with PML
- anisotropic box
- dispersive box
- monitor-heavy box

This is where the paper's `GCells/s` model applies most cleanly.

### Tier 2: Realistic Single-Run Device Benchmarks

Purpose:

- measure full simulation throughput on representative photonics devices

Examples:

- MMI benchmark
- grating coupler
- metalens
- antenna or metasurface benchmark

Metrics:

- wall-clock time
- cells/s
- monitor/postprocess overhead
- accuracy metric tied to the benchmark

### Tier 3: Workflow Benchmarks

Purpose:

- measure end-to-end design tasks, not just one simulation

Examples:

- brute-force metalens optimization
- adjoint WDM optimization
- parameter sweeps
- batched mode-solver plus FDTD workflows

Metrics:

- simulations per hour
- objective improvement per wall-clock hour
- average time per forward
- average time per adjoint
- queueing/orchestration overhead if relevant

## Baseline Benchmark Suite

The first benchmark suite should contain a small but diverse set of mandatory cases.

### Suite A: Synthetic Throughput Baselines

1. `uniform_box_no_pml`

- purpose: closest match to the simple throughput model from `gpu_benchmarking.pdf`
- use: single-GPU hardware comparison and headroom analysis
- metrics: `GCells/s`, model ratio, memory footprint

2. `uniform_box_with_pml`

- purpose: quantify the cost of realistic absorbing boundaries
- use: compare against `uniform_box_no_pml` to understand PML tax
- metrics: cells/s, PML overhead fraction

3. `anisotropic_box`

- purpose: quantify full-tensor constitutive cost
- use: compare simple versus rich material capability paths
- metrics: cells/s, constitutive-stage fraction

4. `monitor_heavy_box`

- purpose: quantify DFT and monitor overhead
- use: understand cost of realistic monitor configurations
- metrics: cells/s, monitor cost fraction

These cases should be small enough to run often and large enough to saturate the device.

## Realistic Reference Workloads

### 1. `metalens_large_area`

Reference sources:

- `metalens/README.md`
- `metalens/Metalens_Optimize.py`

Why it matters:

- large-area metasurface geometry
- many repeated or parameterized unit cells
- strong geometry/materialization stress
- realistic full-wave focusing workload

Useful details from the local repo:

- Tidy3D-based dielectric metalens
- `metalens/README.md` states a large-area `100 wavelength diameter` metalens with total dimensions roughly `(100 x 100 x 46)` wavelengths
- the example claims a single simulation can be run in under 5 minutes in that environment
- `Metalens_Optimize.py` turns this into a repeated evaluation workflow

Benchmark role:

- single-GPU realistic simulation benchmark
- workflow benchmark for repeated solves
- geometry initialization benchmark

Metrics:

- geometry build time
- materialization time
- timestep throughput
- total runtime
- focal efficiency
- simulations per hour for optimization loops

Important warning:

- this is both a solver benchmark and a workflow benchmark
- benchmark reports must separate one simulation from the optimization loop

### 2. `midir_metalens_extreme_scale`

Reference sources:

- `papers/gpu_benchmarking.pdf`
- `tidy3d-notebooks/MidIRMetalens.ipynb`

Why it matters:

- very large footprint
- many meta-atoms
- strong multi-GPU stress
- realistic metasurface workload with short timestep count but enormous spatial scale

Paper-backed scale reference:

- `10.6 x 10.6 mm` mid-IR metalens
- radius `5.3 mm`
- `574,775` cylindrical meta-atoms
- more than `9 billion` grid points
- about `10,000` timesteps
- reported run time of about `7 minutes` on `32 A100` GPUs

Notebook-backed local relevance:

- `MidIRMetalens.ipynb` describes a mid-IR metalens at `10.6 um`
- a `100 wavelength` diameter design
- approximately `1.060 mm` diameter and `0.707 mm` focal length in the notebook-scale case
- relies on near-to-far projection to avoid simulating the full focal distance directly

Benchmark role:

- many-GPU stress test
- geometry and near-to-far stress test
- communication-light but memory-heavy benchmark

Metrics:

- total cells
- total timesteps
- cells/s aggregated across GPUs
- strong-scaling efficiency
- peak memory per GPU
- near-to-far postprocess cost

### 3. `mmi_meep_accuracy_perf`

Reference source:

- `tidy3d-notebooks/MMIMeepBenchmark.ipynb`

Why it matters:

- direct accuracy/performance comparison style benchmark
- moderate device size
- includes mode monitors and symmetry
- should be portable across Meep-compatible and Tidy3D-compatible APIs

Local notebook hints:

- explicit benchmark framing
- `ModeMonitor`
- symmetry
- auto grid with `min_steps_per_wvl=11`
- fixed `run_time`

Benchmark role:

- medium-size single-GPU regression benchmark
- combined accuracy/performance anchor
- good CI or nightly benchmark

Metrics:

- runtime
- cells/s
- transmitted power / modal coefficients
- error relative to stored reference result

### 4. `edge_feed_patch_antenna`

Reference source:

- `tidy3d-notebooks/EdgeFeedPatchAntennaBenchmark.ipynb`

Why it matters:

- RF or microwave-style large-domain antenna benchmark
- stresses PML, near-to-far, and radiation problems differently from silicon photonics
- useful to avoid overfitting the package only to integrated photonics workloads

Benchmark role:

- single-GPU or multi-GPU radiation benchmark
- far-field and bandwidth-heavy monitor benchmark

Metrics:

- runtime
- cells/s
- far-field or antenna figure of merit
- PML reflection proxy

### 5. `wdm_adjoint_workflow`

Reference source:

- `tidy3d-notebooks/Autograd9WDM.ipynb`

Why it matters:

- repeated forward and adjoint solves
- realistic inverse-design workflow
- mode-source and mode-monitor relevance
- useful for measuring throughput of the hybrid optimization path rather than just one simulation

Benchmark role:

- workflow benchmark
- adjoint benchmark
- mode-coupling benchmark

Metrics:

- forward solve time
- adjoint solve time
- total optimization iteration time
- objective gain per iteration
- total memory use across stored monitor/checkpoint data

### 6. `awg_or_demux_large_device`

Reference sources:

- `papers/gpu_benchmarking.pdf`
- `tidy3d-notebooks/8ChannelDemultiplexer.ipynb`

Why it matters:

- long-propagation photonic integrated circuit benchmark
- multiplexing / demultiplexing workload
- stronger timestep-count stress than the metalens examples

Paper-backed large-scale reference:

- `220 x 220 um` AWG
- `3 billion` grid points
- more than `300,000` timesteps
- run in just under an hour on `32 A100` GPUs

Benchmark role:

- many-GPU long-run stress test
- communication and timestep-count stress test

Metrics:

- aggregated cells/s
- strong scaling
- per-step communication overhead
- wavelength-channel accuracy metrics

## What To Track For Single-GPU Hardware

For each supported GPU family:

- baseline synthetic throughput
- one medium realistic device benchmark
- one heavy realistic device benchmark

Recommended comparison set:

- cells/s versus model prediction
- warm solve time
- peak memory used
- precision sensitivity (`fp32`, `fp64`, mixed if supported)

The GPU paper gives useful anchor numbers for simple no-PML runs:

- up to about `20 GCells/s` on `A100 SXM`
- up to about `33 GCells/s` on `H100 SXM`

These should not be treated as universal targets for full-featured benchmarks, but they are useful sanity checks for the simplest synthetic solver tier.

## What To Track For Multi-GPU

Multi-GPU results must include both absolute performance and scaling efficiency.

Definitions:

```math
E_{\mathrm{strong}}(N) =
\frac{T(1)}{N\,T(N)},
\qquad
E_{\mathrm{weak}}(N) =
\frac{T(1, M)}{T(N, NM)},
```

where:

- `T(1)` is one-GPU time for a fixed problem
- `T(N)` is `N`-GPU time for the same problem
- `M` is a one-GPU problem size

Required multi-GPU metrics:

- strong-scaling efficiency
- weak-scaling efficiency
- halo/communication time fraction
- communication volume per step
- load-balance quality across chunks

The benchmark harness should also record the interconnect type where possible:

- PCIe
- NVLink
- other vendor-specific interconnects

because the GPU paper correctly emphasizes that GPU-to-GPU bandwidth becomes the limiting resource at large scale.

## Accuracy Coupling

Speed benchmarks are only meaningful if accuracy is held constant enough to make comparisons fair.

Each realistic benchmark must define:

- spatial resolution or grid spec
- timestep or Courant factor
- stopping rule
- monitor set
- an accuracy target or reference observable

Examples:

- modal transmission
- focal efficiency
- scattering cross section
- far-field beam shape
- resonance frequency / Q

Benchmark reports should always include the observable and the error tolerance so that "faster because lower accuracy" is not mistaken for genuine progress.

## Benchmark Result Schema

Use a machine-readable manifest inspired by `Khronos.jl/benchmark`.

Suggested schema:

```yaml
benchmark:
  name: metalens_large_area
  category: realistic_single_run
  commit: <git-sha>
  backend: warp
  precision: float32
  hardware:
    vendor: nvidia
    model: H100-SXM
    interconnect: nvlink
    device_count: 8
  problem:
    cells: 9200000000
    timesteps: 10000
    geometry_case: midir_metalens
    monitor_case: focus_near_to_far
  timing:
    setup_s: 12.4
    materialization_s: 8.1
    warm_solve_s: 411.0
    postprocess_s: 17.2
    total_s: 448.7
  performance:
    cell_updates_per_s: 2.24e11
    gcells_per_s: 224.0
    strong_scaling_efficiency: 0.81
    peak_mem_gb_per_gpu: 73.2
    comm_fraction: 0.19
  accuracy:
    metric_name: focal_efficiency
    metric_value: 0.74
    reference_error: 0.012
```

This should be generated automatically and stored under version control or in a benchmark artifact store.

## Regression Policy

Adopt the useful Khronos idea of hardware-specific baseline values, but expand it.

For each benchmark/hardware/backend/precision tuple:

- keep a baseline result
- fail or warn on large regressions
- flag large unexplained improvements for baseline updates

Recommended policy:

- CI: small synthetic benchmarks and one medium realistic benchmark
- nightly: all single-GPU realistic benchmarks
- scheduled or manual: heavy many-GPU benchmarks

Regression thresholds should be benchmark-specific, for example:

- `5-10%` for stable synthetic kernel benchmarks
- `10-20%` for realistic full-run benchmarks
- looser thresholds for shared or cloud hardware

## Benchmark Harness Design

The package should include a benchmark framework with:

- benchmark definitions as code plus metadata
- standard warmup rules
- standard timing windows
- standard result schema
- automatic hardware detection
- optional profiling mode to update baselines

Useful features already present in Khronos:

- backend detection
- precision selection
- hardware-key generation
- profile mode that writes new baseline values
- benchmark result checks against stored baselines

The new package should keep that pattern, but extend it to:

- multiple GPUs
- richer metrics than only `MCells/s`
- realistic workload manifests
- accuracy metrics

## Design Patterns

- benchmark synthetic and realistic workloads separately
- always record both setup and warm solve time
- tie every performance number to an accuracy target
- compare measured cells/s to a simple hardware-aware model
- store machine-readable baseline results per hardware/backend/precision
- track strong and weak scaling explicitly
- include geometry/materialization cost in realistic benchmarks
- include workflow-level benchmarks for adjoint and optimization loops

## Design Anti-Patterns

- reporting only wall time with no cell count or timestep count
- comparing runs with different accuracy settings as if they were equivalent
- mixing JIT/setup time with steady-state throughput without saying so
- using only one benchmark shape and claiming general performance
- using only cloud-vendor timings without local reproducible benchmark manifests
- treating optimization-loop runtime as if it were pure solver throughput

## Recommended First Benchmark Plan

Phase 1:

- implement `uniform_box_no_pml`
- implement `uniform_box_with_pml`
- implement `mmi_meep_accuracy_perf`
- port a `metalens_large_area` benchmark from `metalens`

Phase 2:

- implement `wdm_adjoint_workflow`
- implement `edge_feed_patch_antenna`
- add automated baseline storage and regression checks

Phase 3:

- add `midir_metalens_extreme_scale`
- add `awg_or_demux_large_device`
- add strong and weak scaling studies

This gives a credible path from simple throughput sanity checks to publication-quality many-GPU benchmark evidence.
