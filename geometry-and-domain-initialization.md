# Geometry And Domain Initialization

This document specifies the geometry frontend and the GPU domain-initialization pipeline needed for a maintainable, Meep-compatible, Tidy3D-compatible FDTD solver. The goal is not just to parse shapes, but to convert large shape collections into Yee-grid material tensors quickly, accurately, and in a way that avoids kernel combinatorics.

## Design Goal

The solver must support a high-level geometry API comparable to `libctl`/Meep and Tidy3D while compiling down to a small number of GPU-friendly geometry query and materialization kernels. The critical design split is:

- frontend geometry model: user-visible primitives, transforms, groups, structure priorities, lattice replication, medium attachment
- backend materialization model: chunk-local shape buffers, acceleration structures, point/volume queries, and kernels that populate the solver coefficients on the Yee grid

The frontend may be object-oriented; the backend must not be.

## Compatibility Targets

The shared geometry model must cover the common core of:

- Meep `libctl` geometry: `sphere`, `cylinder`, `cone`, `wedge`, `block`, `ellipsoid`, `prism`, lattice/periodic duplication, material attachment
- `GeometryPrimitives.jl`: primitive membership, nearest-surface normal queries, voxel volume fractions, spatial acceleration, periodic copies
- Tidy3D geometry/structure model: explicit `Geometry` objects, `Structure = geometry + medium`, structure ordering/priority, richer polygonal and swept primitives

Tidy3D appears to use a Python geometry object model with shape-specific `inside(...)`, slicing, and intersection logic rather than a low-level GPU geometry engine. That frontend is useful as an API target, but it should not dictate the backend implementation.

## Required Primitive Families

The solver IR must represent these primitive families directly:

- axis-aligned box / cuboid
- affine-transformed box
- sphere
- cylinder
- cone / truncated cone
- ellipsoid
- prism / polygon extrusion
- polyslab / swept polygon with sidewall angle
- wedge
- triangle mesh or faceted surface set
- geometry group / union container
- repeated geometry / lattice replicated geometry

Optional high-level shapes may lower into the primitive set:

- capsule -> cylinder + hemispheres
- regular polygon prism -> prism
- rotated ellipses / superellipses -> transformed implicit primitive or tessellated prism, depending on accuracy target

## Geometry Semantics

Every structure in the frontend must lower to:

- `primitive_type`
- `medium_id`
- `priority`
- `transform`
- `aabb_world`
- optional `periodicity`
- optional `subpixel_policy`
- optional `shape_params`

The compatibility semantics must include:

- deterministic structure ordering
- background medium fill
- priority override when shapes overlap
- optional boolean composition for groups
- periodic/lattice duplication at the geometry level
- separable distinction between geometric occupancy and material law

That last point matters because adjoint differentiation often needs `d occupancy / d parameter`, `d epsilon / d occupancy`, and `d operator / d epsilon` as separate factors.

## Required Geometry Queries

The backend only needs a small set of geometry predicates and differential queries:

- axis-aligned bounding box `B_s` for each shape `s`
- point containment `chi_s(x) in {0, 1}`
- signed or unsigned surface distance `d_s(x)` when available
- outward normal `n_s(x_surf)` at the closest surface point
- voxel volume fraction `f_s(V) in [0, 1]`
- edge/face area fractions when anisotropic subpixel smoothing is used
- inverse transform `T_s^{-1}(x)` to evaluate canonical-shape predicates

The most important observation from `libctl` and `GeometryPrimitives.jl` is that point membership alone is insufficient for a production solver. Accurate geometry initialization also needs:

- interface normals for effective tensor construction
- local fill fractions for partially cut cells
- robust handling of replicated geometry

## Discrete Materialization Targets

Geometry initialization produces grid coefficients, not just labels. For each Yee location `x_q` associated with field component `q`, the materializer must produce the coefficients required by the time-stepper:

- `epsilon_q^{-1}(x_q)` or effective inverse permittivity tensor entries
- `mu_q^{-1}(x_q)` if magnetic media are supported
- conductivity coefficients
- dispersive-pole coefficients and region masks
- nonlinear/update capability masks
- PML ownership masks, if geometry and boundary regions interact in one pass

For a point-sampled path,

```math
M_q(x_q) = M_bg + \sum_{s \in \mathcal{C}(x_q)} \Pi_s(x_q) \left(M_s - M_{prev(s)}\right),
```

where `\Pi_s` applies the overlap/priority rule and `\mathcal{C}(x_q)` is the candidate shape set for that Yee point.

For a subpixel path over a primal voxel `V`,

```math
\bar{M}(V) = \sum_s f_s(V) M_s,
\qquad
f_s(V) = \frac{1}{|V|} \int_V \chi_s(x)\,dx,
```

followed by a Yee-aware projection from voxel material data to the required edge/face-centered coefficients. The exact projection operators should match the interpolation/restriction conventions already documented in `mathematical-kernel-spec.md`.

## Subpixel And Interface-Aware Materialization

To match Meep/Tidy3D class accuracy, the solver should support at least two initialization modes.

### Mode A: Fast point-sampled initialization

- one material query per Yee sample location
- appropriate for coarse screening, debugging, and homogeneous blocks
- lowest implementation complexity

### Mode B: Interface-aware initialization

- estimate fill fractions and interface normals for cut voxels
- produce effective tensor coefficients for edge/face samples
- required for high-accuracy dielectric interfaces and stable gradients

The Meep paper adds an important constraint here: subpixel smoothing that removes the first-order interface error is anisotropic. Even for interfaces between isotropic materials, the discretized cut-cell representation generally becomes tensor-valued, using arithmetic averaging for field components parallel to the interface and harmonic-style averaging for perpendicular components. That means "subpixel support" is not equivalent to scalar fill fractions; it requires tensor-aware coefficient projection.

The interface-aware path can use:

- analytic voxel fractions for simple shapes
- adaptive quadrature for prisms/polyslabs
- multisample Monte Carlo or stratified sampling only as a fallback

The same paper also notes an implementation boundary that should remain explicit in the spec:

- perfect-metal interface smoothing is a separate treatment
- subpixel smoothing for dispersive media should be treated as its own feature tier, not assumed to come for free with nondispersive subpixel support

The anti-pattern is to hardcode a separate subpixel kernel for every shape and every material law. Instead, the IR should expose a shared query contract:

- `contains(shape, x)`
- `closest_surface(shape, x)`
- `voxel_fraction(shape, voxel_desc)`

Then backend codegen specializes only by primitive family and subpixel mode.

## GPU Domain-Initialization Pipeline

The initialization algorithm must avoid the naive `O(N_voxels N_shapes)` scan. Use a three-stage pipeline.

### Stage 1: Frontend lowering

Convert frontend objects into a flat structure list with:

- canonical primitive type
- packed parameters
- affine transform
- world-space `AABB`
- medium and priority ids
- derivative metadata for adjoint/design parameters

### Stage 2: Chunk-local broad phase

For each simulation chunk `c`, compute its extended bounds `B_c^+` including any stencil/subpixel halo. Build a candidate set

```math
\mathcal{S}_c = \{ s \mid B_s \cap B_c^+ \neq \emptyset \}.
```

Then build a chunk-local acceleration structure over `\mathcal{S}_c`, using one of:

- uniform spatial bins
- BVH over shape `AABB`s
- KD-tree for irregular sparse geometry

For GPU rasterization, uniform bins or Morton-sorted cell lists are usually preferable because they map cleanly to flat arrays and predictable memory access.

### Stage 3: Narrow-phase materialization

For each chunk and each target grid site `x_q`, query only the candidate shapes from the relevant bin/list:

```math
\mathcal{C}(x_q) = \{ s \in \mathcal{S}_c \mid x_q \in B_s^{pad} \}.
```

Then evaluate occupancy and priority resolution:

```math
s^\star(x_q) = \arg\max_{s \in \mathcal{C}(x_q),\, \chi_s(x_q)=1} \mathrm{priority}(s),
```

with a stable tie-break rule matching the frontend API.

For subpixel mode, the same candidate list is reused over voxel quadrature points or analytic volume-fraction calls. This reuse is important; rebuilding candidate lists per quadrature point is too expensive.

## Kernel Families For Geometry Initialization

The geometry system should compile into a small family of kernels:

1. primitive-parameter packing kernels
2. shape `AABB` generation kernels
3. chunk-shape overlap kernels
4. spatial binning / candidate-list construction kernels
5. point-sampled materialization kernels
6. subpixel fraction and normal estimation kernels
7. coefficient projection kernels from voxel data to Yee locations
8. optional derivative/JVP kernels for shape parameters

That is the maintainable decomposition. An anti-pattern is a monolithic "initialize everything" kernel that combines broad phase, containment, subpixel averaging, and coefficient assembly across every primitive type.

## Data Layout

Geometry backend buffers should use structure-of-arrays, grouped by primitive family:

- `sphere.center_x[]`, `sphere.center_y[]`, `sphere.center_z[]`, `sphere.radius[]`
- `box.inv_transform[shape_id]`
- `prism.vertex_offset[]`, `prism.vertex_count[]`, `prism.vertices[]`
- `shape.medium_id[]`, `shape.priority[]`

This layout supports:

- coalesced loads for same-type batches
- type-specialized kernels
- compact candidate lists referencing `(type_id, local_index)`

The backend should avoid storing host-language objects, pointers to heterogeneous structs, or recursive shape trees on the device.

## Avoiding Kernel Combinatorics

Geometry initialization is another place where combinatorial explosion appears. The cross-product is:

- primitive family
- point vs subpixel materialization
- scalar vs tensor medium
- forward-only vs derivative-enabled
- single-GPU vs chunked multi-GPU

The correct strategy is:

- specialize kernels by primitive family and query mode
- keep medium evaluation separate from geometry occupancy where possible
- express overlap resolution, coefficient projection, and derivative accumulation as reusable post-query stages

In other words, do not generate `sphere_x_tensor_x_subpixel_x_derivative_x_multigpu` as an independent handwritten kernel. Generate:

- primitive query kernels
- reduction/projection kernels
- derivative kernels attached to the same query contract

## Multi-GPU Implications

Geometry initialization must use the same chunk decomposition as the solver runtime. Even when the full problem fits on one GPU, this keeps initialization and time stepping aligned.

For each chunk:

- materialize only owned cells plus required halo/support cells
- build local candidate lists from global geometry metadata
- optionally replicate read-only geometry buffers across devices

Large geometry scenes should not be globally expanded into per-device copies of every replicated periodic image. Periodic copies should be generated lazily per chunk by intersecting the chunk bounds with the repetition lattice.

## Adjoint And Shape Differentiation Requirements

The geometry layer must support design derivatives without changing the forward kernels.

For a scalar objective `J`,

```math
\frac{dJ}{dp}
=
\sum_q
\frac{\partial J}{\partial M_q}
\frac{\partial M_q}{\partial p},
```

where `p` is a geometry parameter such as radius, height, vertex position, or transform parameter.

This requires one of:

- analytic `\partial \chi_s / \partial p` and `\partial f_s / \partial p` for supported primitives
- differentiable signed-distance or closest-surface formulas
- stable finite-volume approximations only as a fallback

The important implementation rule is that geometry derivatives should be attached to the same primitive query abstraction used by forward initialization. Otherwise the adjoint path will drift from the forward discretization.

## Shared Geometry IR

The LLM-facing solver spec should include a geometry IR roughly of the form:

```text
GeometryScene
  background_medium
  lattice_spec?
  structures[]

Structure
  id
  medium_id
  priority
  primitive
  transform
  repetition?
  subpixel_policy
  differentiable_params[]

Primitive
  type
  params
  topology_meta?
```

And a backend contract:

```text
build_shape_aabbs(scene) -> AABBBuffer
build_chunk_candidates(scene, chunks) -> CandidateLists
materialize_point(scene, chunk, candidates) -> CoefficientFields
materialize_subpixel(scene, chunk, candidates) -> CoefficientFields
differentiate_geometry(scene, chunk, candidates, adjoint_data) -> ParameterGradients
```

That is the level of abstraction the code generator should target.

## Design Patterns

- lower all frontends to one primitive-and-structure IR
- separate geometry occupancy from material-law evaluation
- use chunk-local broad phase before any voxel query
- use SoA layout and type-batched kernels on device
- keep a single query contract for forward and adjoint geometry operations
- support both fast point sampling and interface-aware materialization
- treat periodic replication as a lazy chunk-local expansion problem

## Design Anti-Patterns

- looping over all shapes for every voxel globally
- keeping frontend object graphs alive inside device kernels
- reparsing geometry every timestep instead of caching packed buffers
- encoding overlap resolution separately in each primitive kernel
- using CPU-only geometry slicing logic as the solver initialization path
- implementing shape derivatives with a different discretization than the forward materialization path
- exploding the kernel surface by mixing shape type, medium law, and execution mode into independent handwritten variants

## Implementation Recommendation

The first production implementation should expose a rich frontend API but compile geometry into a narrow GPU backend:

- frontend adapters for Meep/libctl and Tidy3D
- a shared geometry IR
- chunk-local acceleration structures
- type-specialized primitive query kernels
- reusable overlap, subpixel, and coefficient-projection kernels

This matches the maintainability strategy already chosen for the FDTD update kernels: compact IR plus generated specialization, not manually maintained kernel cross-products.
