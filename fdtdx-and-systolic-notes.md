# FDTDX And Systolic FDTD Notes

This note captures additional core-solver features worth considering after reviewing the local `fdtdx` codebase and the local systolic FDTD paper in `papers/systolic_fdtd.pdf`.

The purpose is not to copy either system literally. The goal is to extract ideas that improve the core solver spec without compromising maintainability, Meep/Tidy3D compatibility, or the backend-neutral IR strategy.

## References Reviewed

- `fdtdx/README.md`
- `fdtdx/src/fdtdx/fdtd/forward.py`
- `fdtdx/src/fdtdx/fdtd/backward.py`
- `fdtdx/src/fdtdx/fdtd/update.py`
- `fdtdx/src/fdtdx/fdtd/container.py`
- `fdtdx/src/fdtdx/core/jax/sharding.py`
- `fdtdx/src/fdtdx/objects/device/device.py`
- `papers/systolic_fdtd.pdf`

The systolic paper is:

- Jesse Lu et al., "A systolic update scheme to overcome memory bandwidth limitations in GPU-accelerated FDTD simulations," arXiv:2502.20610v1, February 28, 2025

## Main Takeaways

- `fdtdx` is most valuable as a reference for solver-state organization, reversible-gradient workflows, detector/source modularity, and design-parameter pipelines.
- `fdtdx` is not a good template for the backend architecture itself because too much of the implementation is coupled to JAX execution and JAX sharding assumptions.
- systolic FDTD is best treated as an optional execution schedule for locality and communication hiding, not as the defining solver abstraction.
- the core solver spec should explicitly reserve space for reversible execution, checkpointing, symmetry constraints, and device-parameter pipelines even if the first implementation uses the Meep-style hybrid adjoint path.

## What To Carry Over From FDTDX

### 1. Explicit Forward And Reverse Runtime Modes

`fdtdx` has a clean conceptual split between forward propagation and reverse-time propagation. Even if the production solver uses the Meep-style hybrid forward/adjoint operator workflow, the runtime contract should still reserve explicit support for:

- forward stepping
- reverse stepping where physically valid
- checkpoint/replay modes
- monitor-recording vs boundary-recording modes

This is worth carrying into the solver IR because it forces the stage graph to declare which data are actually needed for reverse or replay execution.

### 2. Detector And Recorder State As First-Class State Families

`fdtdx` keeps detector state distinct from the core field arrays. That is the right idea.

The core solver spec should make recorder state a first-class optional state family:

- DFT accumulators
- transient recorders
- boundary/interface recording buffers
- mode-overlap work buffers
- adjoint-source reconstruction buffers

This aligns with the existing Meep-style observer/monitor split and avoids contaminating the core Yee kernels with monitor-specific logic.

### 3. Device-Parameter Pipeline Separate From Geometry

One of the more useful ideas in `fdtdx` is that a designable device is not just “geometry.” It is a parameter-to-material transformation pipeline.

That should be reflected in the new solver architecture as a distinct layer:

- geometry primitives define occupancy/support
- design parameter transforms define latent-to-physical parameter maps
- materialization maps those outputs onto Yee coefficients

This is especially important for topology optimization, symmetry-constrained design, filtered/projection design variables, and manufacturing constraints.

### 4. Symmetry And Parameter Constraint Hooks

`fdtdx` has infrastructure for symmetry and parameter transformations. Those features should be promoted into the shared solver/design IR rather than buried in a frontend convenience layer.

Useful core concepts:

- symmetry-constrained geometry or parameter fields
- projection/filter stages for binary or manufacturable designs
- separable latent parameter grids versus simulation grids

This belongs in the spec because it affects geometry materialization, adjoint gradients, and benchmark parity for optimization workflows.

### 5. Capability Detection For Fast Paths

`fdtdx` branches between isotropic/diagonal-anisotropic/full-tensor cases. The architectural lesson is valid even if the exact implementation is not:

- aggressively expose capability metadata
- route simple regions through simpler kernels
- preserve one generic path for full tensor or complex materials

This reinforces the existing capability-driven kernel-generation strategy.

## What Not To Copy From FDTDX

### 1. JAX-Specific Runtime Assumptions

`fdtdx` is deeply shaped by JAX:

- pytree containers
- `jit`/`lax` control flow
- JAX sharding objects
- array donation and recompilation concerns

Those are implementation details of one backend, not good core abstractions. The new solver should keep:

- backend-neutral state/layout IR
- explicit chunk and communication contracts
- explicit stage graph

and let a Warp or future backend map onto those concepts directly.

### 2. Reverse-Time FDTD As The Only Gradient Story

`fdtdx` emphasizes time-reversibility for efficient gradients. That is useful, but it should not become the primary abstraction for this project.

Reasons:

- the project already requires compatibility with Meep’s hybrid time-/frequency-domain adjoint pipeline
- lossy, dispersive, monitor-defined, and source-defined workflows are easier to reason about in the hybrid operator framework
- reverse-time execution can become fragile when many feature plugins are active

The right conclusion is:

- support reversible execution where useful
- do not let it replace the primary hybrid adjoint formulation in the core spec

### 3. Backend-Coupled Sharding Semantics

`fdtdx/src/fdtdx/core/jax/sharding.py` assumes a simple array-axis sharding story that must divide cleanly across devices. That is fine for a JAX implementation, but it is too restrictive for a general multi-GPU FDTD solver.

The production solver needs:

- chunking driven by geometry, halo radius, and stage dependencies
- support for uneven partitions when geometry or memory pressure demands it
- communication contracts that are independent of array-framework sharding APIs

## Additional Core Features Suggested By FDTDX

These should be considered for the core solver spec even if they are not all first-implementation requirements.

### A. Explicit Checkpointing Policy

The runtime should support:

- no checkpointing
- periodic checkpointing
- monitor-only checkpointing
- reversible replay mode

That policy should be explicit in the IR because it affects memory use, adjoint cost, and device communication.

### B. Boundary Recording As A Formal Operator

`fdtdx` records interfaces/boundaries for reverse workflows. Independently of JAX, this suggests a useful abstraction:

- boundary field extraction
- boundary field replay
- boundary compression
- boundary reduction across chunks

This may be useful not just for reverse-time workflows but also for domain-coupling and reduced-order model interfaces.

### C. Parameter-Grid Versus Simulation-Grid Decoupling

`fdtdx` distinguishes design voxels from simulation voxels. That is worth including explicitly in the geometry/design spec:

- latent parameter grid
- materialization grid
- Yee coefficient grid

Those are not always the same resolution, and the spec should say so.

## Systolic FDTD: What It Adds

The paper's main claim is narrower and more concrete than my earlier inference: it keeps the exact same FDTD update equations, but changes the execution schedule so that most work is performed on small subdomains that fit in fast cache and exchange only lower-dimensional boundary data through global memory.

Operationally, systolic FDTD usually means:

- process tiles/subdomains in an ordered sweep
- keep only a narrow working set resident
- hand off boundary data to the next tile/subdomain in a pipelined fashion
- trade simpler global synchronization for more structured local scheduling

The paper motivates this with a compute-to-memory ratio argument. For a representative Yee update it counts:

- 6 memory operations
- 6 floating-point operations
- about `0.25 flop/byte` assuming 4-byte floats

and contrasts that with a much larger GPU compute-to-memory ratio. The conclusion is that naive FDTD is strongly memory-bandwidth bound, and the performance cliff appears when the working set spills out of fast cache and when global synchronization requires multiple thread blocks.

## What We Should Carry Over From Systolic FDTD

### 1. Execution Schedule Is Separate From Discretization

This is the most important lesson.

The Yee discretization and feature semantics should stay fixed, while the runtime may choose among schedules:

- standard staged chunk update
- communication-overlapped chunk update
- pipelined wavefront update
- systolic tile schedule

That means the solver IR should distinguish:

- mathematical stage graph
- execution schedule

instead of fusing them.

### 2. Locality And Communication Hiding Matter As Much As FLOPs

Systolic approaches are appealing because FDTD is memory-bandwidth dominated. The main potential benefit is not fewer floating-point operations; it is:

- less global memory traffic
- smaller active working sets
- better overlap of halo movement with interior work
- reduced synchronization stalls

Those are directly relevant to the multi-GPU chunk contract in our spec.

The paper's practical H100 implementation makes this concrete:

- subdomains of `31 x 15 x 15` Yee cells
- a stored cache block of `32 x 16 x 16` to hold boundary values
- communication between subdomains as 2d faces written to global memory
- an estimated `~5.45x` reduction in global memory operations relative to a one-cell-at-a-time update
- an estimated compute-to-memory ratio increase from `~0.25` to `~4`

It reports about `0.15 TCUPS` for a stripped-down 3d engine on one Nvidia H100.

### 3. Tile-Wavefront Scheduling May Be Useful For Extreme-Scale Cases

For very large runs or strong multi-GPU scaling pressure, a wavefront or systolic schedule may outperform a simpler bulk-synchronous schedule, especially when:

- chunk halos are expensive relative to local work
- GPU memory pressure is tight
- boundary exchange latency dominates

This suggests adding an optional scheduler layer to the implementation plan.

## What Not To Overlearn From Systolic FDTD

### 1. It Should Not Become The Default Semantic Model

The maintainable core abstraction is still:

- chunked staged DAG
- capability-driven kernels
- explicit halo contracts

Systolic execution is one scheduler for that DAG, not the architecture itself.

### 2. GPU-Friendly Does Not Automatically Mean Wavefront-Friendly

On modern GPUs, a wavefront schedule can conflict with:

- coalesced memory access
- regular launch geometry
- large uniform kernels
- simple feature composition

The more feature-rich the solver becomes, the harder it is to preserve a neat systolic pipeline without reintroducing combinatorial special cases.

### 3. Complex Material Plugins Make Systolic Scheduling Harder

PML, dispersive media, nonlinearities, DFT monitors, adjoint recorders, and geometry-derived coefficient lookups all add stage dependencies. A systolic schedule is easiest for a minimal Yee update and progressively harder for a full Meep-class solver.

That means:

- treat systolic scheduling as an optimization tier
- prove correctness first with the standard staged schedule
- only adopt it where profiling shows it is worth the added complexity

The paper itself is also explicit about one of these simplifications: the benchmarked implementation does not use PML, and instead uses the imaginary part of permittivity to form adiabatic absorbers. It also uses 16-bit field values. Both choices are useful performance references, but neither should be confused with the required feature surface of a Meep/Tidy3D-compatible production solver.

## Recommended Spec Changes

The existing solver docs should be interpreted with the following additions:

1. Add an explicit `ExecutionSchedule IR` alongside the existing stage graph.
2. Add checkpoint/replay policy to the state/runtime spec.
3. Add a `DesignParameter IR` that is separate from both geometry and material law.
4. Add boundary-record/replay operators as optional observer/operator families.
5. Treat systolic or wavefront scheduling as an optional backend schedule for large-scale runs.

## Concrete Recommendation

Build the first solver around the current staged chunked DAG and capability-driven kernel generation. Add enough abstraction so that later backends can choose:

- bulk-synchronous staged execution
- communication-overlapped staged execution
- optional systolic/wavefront execution for selected kernels or selected chunk topologies

That captures the useful part of the systolic idea without forcing the entire architecture into a schedule-specific shape.
