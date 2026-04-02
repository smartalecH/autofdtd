The goal of this repo is to build a commercial-grade, fully featured, high-performance, GPU-accelerated FDTD Maxwell solver.

## Background

The work is split into three phases:

* Phase 1: implement the full feature set and API on the GPU, including multinode support, using the Tidy3D API as the reference standard.
* Phase 2: validate accuracy against analytic results and benchmark simulations.
* Phase 3: maximize performance on a small set of multinode benchmark problems.

## Current Task

You are working on Phase 1.

Use `$captain-wiggum` to create a comprehensive, multi-task Ralph/Wiggum loop for Phase 1 execution. The resulting loop should be detailed enough to drive sustained autonomous work across architecture, package structure, implementation, testing, documentation, and reference-repo investigation.

At a high level, Phase 1 must accomplish the following:

* Build a pip-installable `autofdtd` package with proper `src/` layout, documentation, and tests.
* Implement the full set of features needed to match the Tidy3D API surface as closely as practical.
* Implement the core kernels on the GPU, with architecture choices that are compatible with multinode execution from the start.
* Define and validate a common IR that can represent simulations independently of the public Python API and can be shipped to remote hardware for execution.

## Required Priorities And Constraints

The loop should preserve the following priorities and assumptions:

* Phase 1 is not primarily about final accuracy or final performance, but neither can be ignored. Architecture decisions made now must not block Phases 2 and 3.
* Multi-node GPU execution is a first-class priority. The base architecture should make efficient chunk-to-chunk and node-to-node communication possible, including approaches such as peer-to-peer exchange, masking communication/computation, compression, CUDA graphs, and related optimizations, even if those are not fully optimized until Phase 3.
* Kernels should be written using NVIDIA Warp.
* The primary target is 3D, but the design must also support 2D, symmetry conditions, and polarization conditions.
* The implementation should support complex materials, boundary layers, subpixel smoothing, variable grids in all dimensions, composable geometries, and important postprocessing routines such as DFT fields and near-to-far transforms.
* Unit tests and integration tests are required.
* Every major task should include testing. The loop should assume the current machine has at least 2 NVIDIA GPUs available for useful testing.
* Intentional logging and performance instrumentation are required, including metrics such as simulation initialization time, JIT time, and steady-state timestep rate after warmup.
* Adjoint solving is out of scope for now.

## Core Architectural Tenets

The generated loop should treat the following two ideas as foundational.

### 1. Domain decomposition and specialized kernels

We expect combinatorial growth in kernel variants as features are added, especially for different materials, boundary conditions, and other localized behaviors. The architecture should explicitly plan for that complexity instead of avoiding it.

The core idea is to decompose the simulation domain into regions or chunks that can be assigned specialized kernels. This is important for memory-bandwidth efficiency, since FDTD is fundamentally memory-bandwidth bound. That implies the need for:

* chunking and partitioning strategies,
* halo or ghost-region updates,
* clean interfaces for inter-chunk communication,
* a path to multinode execution that naturally follows from the same decomposition model.

`meep` is an especially important reference for this architectural direction. Both the repo and the accompanying paper discuss the relevant philosophy and implementation patterns.

### 2. A common IR that supports multiple public APIs

The long-term goal is to support arbitrary front-end APIs by lowering them into a common intermediate representation. In Phase 1, this is not optional. We need to define the IR well enough that Tidy3D-style simulations can be lowered into it reliably and executed without depending on the original user script at solve time.

The initial public API work should target compatibility with Tidy3D first. Meep remains important reference material for architecture and future API support. Getting Meep-style inputs to run seamlessly is not a Phase 1 requirement, but the IR should be designed so that adding a Meep frontend later is a natural extension rather than a redesign.

The IR design should be driven by the Tidy3D feature set, the needs of the execution backend, the operational requirement that a fully specified simulation can be packaged and shipped to a remote cluster for execution without a cumbersome Python script, and the desire to keep a future Meep frontend on the table.

The loop should assume:

* users should ideally be able to swap imports and run existing Tidy3D-style code with minimal changes,
* public API compatibility matters, but the real architectural objective is lowering the public API into a stable execution-oriented IR,
* the IR should decouple simulation specification from execution so it can be serialized, transported, and run remotely,
* conventions such as units, defaults, and object models should be normalized explicitly in the lowering step,
* the IR should be general enough that future frontends, especially a Meep-style frontend, can target it without major backend redesign,
* the simulation should ultimately run from that IR rather than directly from the public API objects.

Tidy3D appears to use a JSON-like intermediate form internally. We want an analogous lowering step, but without copying code, documentation, or any license-restricted material.

Phase 1 should include explicit demonstrations that this works. The loop should plan for examples and tests showing that Tidy3D-style simulations can be lowered into the common IR, serialized or otherwise packaged, and executed through the same backend locally or on remote hardware with consistent behavior.

## Feature Expectations

The Phase 1 plan should cover support for:

* Tidy3D-compatible public APIs wherever feasible, with a rigorous inventory of the required features,
* 3D-first simulation with 2D and symmetry/polarization variants,
* first- and second-order accurate subpixel smoothing,
* variable grids in all dimensions,
* composable geometry definitions,
* complex materials and boundary layers,
* important postprocessing features such as DFT fields and near-to-far transforms.

The loop should explicitly create a Tidy3D feature inventory, including:

* features required for credible API compatibility,
* features that drive IR design,
* features that can be deferred from Phase 1,
* features that should raise explicit errors until they are implemented.

## Important Submodules

A strong FDTD package will need the following major submodules:

* A mode solver. `VectorModeSolver.jl` is a strong reference. We need an efficient, GPU-accelerated, fully vectorial mode solver with support for multiple modes, frequencies, symmetry conditions, and polarization conditions.
* A geometry builder. `libctl` and `GeometryPrimitives.jl` are strong references. We need a parametric geometry construction system that can sample onto discrete grids efficiently, ideally in a GPU-compatible way, while matching the intended Tidy3D-style API without copying implementation code.
* An IR definition and lowering pipeline. This should be treated as a first-class subsystem, not an implementation detail. It needs schema/design work, lowering from the public API, validation, serialization or inspectability as needed, and tests proving that realistic simulations map cleanly into it and can be executed from it.

The IR should be inspectable and preferably serializable from the beginning, because that will simplify debugging, regression testing, reproducibility, and remote execution on clusters.

## Reference Material

All useful references are located relative to this repo (`./autofdtd`):

* `meep` in `../meep`
  Use this to study domain decomposition, postprocessing such as near-to-far transforms, geometry initialization, and the overall architectural philosophy. Also consult `../papers/meep_paper.pdf`.
* `Khronos.jl` in `../Khronos.jl`
  A GPU-accelerated Julia FDTD engine that lacks some Meep/Tidy3D features, but may provide useful ideas for handling kernel combinatorial growth.
* `fdtdx`
  An FDTD engine built on JAX/XLA that relies heavily on compiler graph construction and fusion.
* `tidy3d`
  A commercial FDTD solver and the main API compatibility target. It is important not to copy code, documentation, or other protected material.
* `VectorModeSolver.jl` in `../VectorModeSolver.jl`
  A useful open-source reference for vector mode solving of Maxwell’s equations.
* `GeometryPrimitives.jl` in `../GeometryPrimitives.jl`
  An open-source parametric geometry construction tool with subpixel smoothing and many useful shapes, though it is not GPU-accelerated.
* `libctl` in `../libctl`
  A geometry-construction system written in C/C++ that influenced tools like `GeometryPrimitives.jl`.
* `warp` in `../warp`
  NVIDIA’s kernel-authoring DSL for scientific applications, and the intended kernel implementation framework for this project.
* `papers`
  Additional papers that may be useful for both implementation details and future performance work.

These are reference materials only. Do not violate their licenses, and do not copy code or documentation into this project.

## Loop Guidance

The generated Ralph/Wiggum loop should be comprehensive but practical. It should:

* break the work into concrete tasks,
* use subagents to explore the reference repos efficiently if needed,
* make IR definition, compatibility mapping, and remote-execution demonstrations explicit deliverables for Phase 1,
* keep future Meep frontend support in scope at the architectural level, even though Tidy3D is the only required Phase 1 API target,
* define a policy for unsupported Phase 1 features, including whether they should raise explicit errors, use stubs, or be partially lowered with documented limitations,
* include a small set of canonical validation examples, such as vacuum propagation, a dielectric slab or waveguide, a PML boundary case, a symmetry-reduced case, and a near-to-far example,
* keep the prompt self-contained enough to use tokens efficiently during loop execution,
* maintain a strong focus on Phase 1 delivery while protecting future accuracy and performance work.

The reference repos and materials will also be available when the loop runs, so the loop can point back to them directly. Still, this file should remain comprehensive enough to guide the loop without unnecessary ambiguity.
