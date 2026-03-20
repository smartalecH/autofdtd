# Framework Decision

## Recommendation

Use `Warp` as the primary implementation substrate.

If hard cross-vendor portability becomes a first-order requirement, use a native `Kokkos/C++` backend with thin Python bindings rather than building the solver directly on `PyKokkos`.

Use `Khronos.jl` as an algorithm and data-layout reference, not as the implementation base.

## Why Warp

Warp has the best balance of:

- explicit kernel authoring
- low-level execution control
- Python-native API integration
- practical multi-GPU mechanics
- credible performance headroom on NVIDIA GPUs
- lower maintenance burden than owning a Python-to-C++ compiler frontend

Relevant local references:

- `warp/README.md`
- `warp/docs/user_guide/devices.rst`
- `warp/docs/deep_dive/concurrency.rst`
- `warp/docs/user_guide/tiles.rst`
- `warp/warp/examples/fem/example_diffusion_mgpu.py`

## Comparison

### Warp

Strengths:

- explicit kernels and structs
- streams, events, graph capture, peer access, async copies
- multi-GPU model that is explicit enough for domain decomposition
- tile/block programming support where useful
- clean fit with a Meep-compatible Python surface

Weaknesses:

- strongest path is NVIDIA-first
- distributed execution is manual, not hidden
- portability story is weaker than native Kokkos

Conclusion:

- best overall fit for a production GPU solver

### Taichi

Strengths:

- strong metaprogramming and field-layout system
- good single-GPU productivity
- good support for structured-grid experimentation

Weaknesses:

- no convincing first-class multi-GPU/distributed runtime surface found in the local tree
- more attractive for prototyping than for a Meep-class production solver

Conclusion:

- good research/prototyping option, weaker than Warp for explicit multi-GPU chunked FDTD

### PyKokkos

Strengths:

- real Kokkos semantics
- layout and execution-space control
- theoretically strong portability story

Weaknesses:

- Python AST-to-C++ translation stack is itself a maintenance burden
- multi-GPU support is present but operationally awkward
- too close to “building on a compiler frontend” rather than on a runtime

Conclusion:

- not the right production base for this project

### Khronos.jl

Strengths:

- already expresses FDTD as separable curl/update/source/monitor kernels
- useful field-state and kernel decomposition reference
- demonstrates one path to a composable kernel story

Weaknesses visible in the local repo:

- single GPU only
- linear materials only
- uniform grid only
- PML currently assumed
- halo exchange still marked TODO
- missing large parts of Meep-compatible feature coverage

Relevant local references:

- `Khronos.jl/README.md`
- `Khronos.jl/src/Timestep.jl`
- `Khronos.jl/src/DataStructures.jl`
- `Khronos.jl/benchmark/README.md`

Conclusion:

- reference solver only

## Decision Rules

Use Warp unless one of the following becomes non-negotiable:

- first-class AMD portability
- first-class Intel GPU portability
- long-term policy requirement for a C++/MPI/HPC-native backend

If that happens, do not switch to PyKokkos. Switch to a native Kokkos backend and keep the same solver IR and Python API.

## Architecture Consequence

The framework choice should not leak into the solver semantics. The solver should be specified as:

- a Python control/model layer
- a backend-neutral solver IR
- a backend-specific kernel generator/runtime adapter

That is what allows the same spec to target Warp first and a future Kokkos/CUDA/HIP backend later.

