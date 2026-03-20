# Multi-GPU Geometry Core Spec

## Purpose

This document is the implementation spec for a maintainable, GPU-first geometry engine that can serve as the frontend-to-backend lowering layer for a Meep-compatible and Tidy3D-compatible FDTD solver.

The geometry subsystem should be treated as a first-class library, not as a thin preprocessing script. It must:

- represent large scenes built from primitive shapes and transformed copies
- support overlap semantics, periodic replication, and differentiable parameters
- initialize Yee-grid material coefficients efficiently on one or many GPUs
- expose the same geometry query contract to forward, adjoint, and postprocessing code

The central rule is the same as for the FDTD solver:

> Do not encode the Cartesian product of shape families, material laws, and execution modes as handwritten kernels.

Instead, define:

- a compact geometry IR
- a query contract
- a chunk-aware stage graph
- a code generator for primitive-specialized kernels and reusable reductions

## Non-Goals

- CAD-grade solid modeling as the primary implementation target
- arbitrary CSG with unconstrained recursive trees in device kernels
- host-language object dispatch inside GPU kernels
- unstructured meshing in the first implementation
- frontend-specific geometry implementations leaking into the backend

## Design Targets

- compatibility with Meep/libctl-style primitives
- compatibility with Tidy3D-style `Geometry` plus `Structure` semantics
- scalability to many-shape scenes and many repeated geometry instances
- single-GPU and multi-GPU execution with the same chunk decomposition used by the solver
- shared forward/adjoint discretization for shape derivatives
- code-generation-friendly representation suitable for LLM-targeted implementation

## Geometry Model

The geometry library should be organized as a staged pipeline rather than a monolithic rasterizer:

1. frontend object lowering
2. primitive canonicalization
3. transform and periodicity normalization
4. world-space `AABB` generation
5. chunk-scene overlap culling
6. chunk-local acceleration structure build
7. point or voxel query execution
8. overlap resolution and structure ordering
9. projection to solver coefficients
10. optional derivative/JVP/VJP accumulation

This stage split keeps the backend small and allows the same geometry scene to serve:

- forward materialization
- adjoint geometry differentiation
- monitor region clipping
- field sampling along geometry-defined manifolds

## Required IR

The LLM-facing implementation spec should generate from the following IR.

### 1. Primitive Topology IR

Must define:

- primitive family: `sphere`, `box`, `cylinder`, `cone`, `ellipsoid`, `prism`, `polyslab`, `wedge`, `triangle_mesh`, `group`, `repetition`
- canonical parameter vector
- canonical local coordinate system
- canonical containment predicate
- canonical closest-surface or normal query support
- canonical `AABB` generation logic
- canonical differentiable parameter list

Rule:

- high-level frontend shapes lower into this set before any backend specialization happens

### 2. Transform IR

Must define:

- translation
- rotation
- scaling if permitted
- full affine transform for imported or advanced geometry
- inverse transform cache
- transform-conditioned `AABB` expansion rule

Rule:

- all primitive queries happen in canonical coordinates through `x_local = T^{-1} x_world`

### 3. Structure Semantics IR

Each structure must declare:

- `primitive_ref`
- `medium_id`
- `priority`
- `boolean_role` if boolean composition is enabled
- `subpixel_policy`
- `materialization_policy`
- `differentiable_params`

Semantics that must be preserved:

- deterministic overlap resolution
- background medium fill
- explicit structure ordering where API requires it
- distinction between geometric occupancy and material constitutive data

### 4. Repetition And Lattice IR

Must define:

- lattice basis vectors
- finite or infinite repetition policy
- symmetry or periodic-image generation rules
- chunk-local image enumeration

Rule:

- repeated geometry should not be globally expanded unless the expansion is proven cheap

### 5. Query Contract IR

All backend kernels should target a small query contract:

- `contains(primitive, x_world) -> bool`
- `distance(primitive, x_world) -> float`
- `closest_normal(primitive, x_world) -> vec3`
- `voxel_fraction(primitive, voxel_desc) -> float`
- `face_fraction(primitive, face_desc) -> float`
- `edge_fraction(primitive, edge_desc) -> float`

Backends may implement exact or approximate variants, but the contract must be stable.

### 6. Scene Decomposition IR

Must define:

- global scene bounds
- chunk bounds
- extended chunk bounds for support and halo
- chunk-local candidate shape lists
- acceleration structure metadata
- local tie-break rules for overlapping geometry

### 7. Output Field IR

The geometry engine should not output only occupancy labels. It must produce one or more of:

- scalar material ids
- region masks
- fill-fraction fields
- interface normals
- effective constitutive coefficients
- derivative metadata and parameter-to-region maps

This allows the geometry backend to serve both a cheap rasterization mode and a high-accuracy coefficient-generation mode.

## Kernel Families

The generator should produce kernels from these families.

### Family A: Canonical Primitive Queries

Purpose:

- evaluate `contains`, `distance`, `closest_normal`, and simple analytic fractions for one primitive family

Examples:

- sphere queries
- transformed box queries
- cylinder queries
- prism/polyslab queries
- triangle mesh BVH leaf queries

Rule:

- specialize by primitive family only

### Family B: Transform And Bounds Kernels

Purpose:

- apply transforms
- generate world-space `AABB`s
- expand bounds for support, smoothing, or derivative stencils

Rule:

- this family must be reusable across all primitive types

### Family C: Chunk-Shape Culling

Purpose:

- determine which structures intersect a chunk or its extended support

Inputs:

- chunk bounds
- shape `AABB`s
- repetition/lattice metadata

Outputs:

- chunk candidate lists
- optional per-bin candidate lists

### Family D: Spatial Binning / Acceleration Build

Purpose:

- build chunk-local acceleration structures for narrow-phase queries

Allowed structures:

- uniform bins
- Morton-sorted bins
- flat BVH
- KD-tree if backend supports it efficiently

Recommended first implementation:

- uniform bins for analytic primitives
- flat BVH for triangle meshes

### Family E: Point Query Materialization

Purpose:

- evaluate structure occupancy and overlap at point samples

Outputs:

- material ids
- medium references
- scalar coefficients for fast initialization paths

### Family F: Subpixel Query Materialization

Purpose:

- evaluate volume and face fractions
- estimate interface normals
- produce cut-cell metadata

Outputs:

- effective voxel tensors
- interface orientation
- fill-fraction fields

### Family G: Overlap Resolution And Projection

Purpose:

- apply ordering/priority rules
- combine overlapping shapes into one discretized material description
- project voxel-centered material descriptions to Yee locations

Rule:

- this family must be independent of primitive family

### Family H: Geometry Derivative Kernels

Purpose:

- compute `d occupancy / d p`, `d fraction / d p`, or JVP/VJP contributions for differentiable shape parameters

Rule:

- derivative kernels must share the same discretization path as the forward materialization mode

### Family I: Communication And Replication Kernels

Purpose:

- replicate geometry metadata to devices
- pack/unpack chunk-local candidate lists
- optionally reduce distributed geometry-derived fields

Rule:

- communication should happen at the metadata and field level, not by moving frontend objects around

## Geometry Stage Graph

Each stage must declare:

- inputs
- outputs
- chunk dependency radius
- whether it is host-side, device-side, or mixed
- whether it is reused by forward and adjoint modes

Minimum stage set:

- `lower_frontend_scene`
- `canonicalize_primitives`
- `build_world_aabbs`
- `enumerate_repetitions_for_chunk`
- `build_chunk_candidates`
- `build_acceleration`
- `query_point_materials`
- `query_subpixel_materials`
- `resolve_overlaps`
- `project_to_yee_coefficients`
- `differentiate_geometry`

This explicit DAG is the maintainability mechanism. It prevents frontend-specific hacks from being welded into one opaque rasterization path.

## Data Layout IR

The backend should use:

- structure-of-arrays by primitive family
- compact indirection from global structure id to `(primitive_family, local_index)`
- packed transform inverses
- packed medium ids and priorities
- chunk-local compressed candidate lists

Example:

- `sphere.cx[]`, `sphere.cy[]`, `sphere.cz[]`, `sphere.r[]`
- `box.invT[]`
- `prism.vertex_offsets[]`, `prism.vertex_counts[]`, `prism.vertices[]`
- `structure.medium_id[]`, `structure.priority[]`, `structure.flags[]`

The backend must avoid:

- Python/C++ object pointers on device
- recursive CSG trees in inner loops
- AoS layouts that mix unrelated primitive types

## Managing Shape Count Scalability

The core scalability problem is:

```math
\text{cost}_{naive} = O(N_{\text{sites}} N_{\text{shapes}}).
```

The geometry engine must reduce this to:

```math
\text{cost}_{practical}
\approx
O(N_{\text{shapes}}^{build})
+
O(N_{\text{sites}} k_{\text{local}}),
```

where `k_local` is the average number of relevant shapes per query site after broad-phase culling.

Required strategies:

- chunk-scene culling before any per-site queries
- per-chunk acceleration structures
- lazy periodic-image generation
- shape-family batching
- reuse of candidate lists across point and quadrature queries

The anti-pattern is a global all-shape scan, even if each containment predicate is cheap.

## Multi-GPU Contract

The geometry engine must use the same spatial chunking contract as the FDTD engine.

For each chunk `c`, define:

- owned spatial region `\Omega_c`
- extended support region `\Omega_c^+`
- candidate structure set `\mathcal{S}_c`
- local acceleration structure `\mathcal{A}_c`

with

```math
\mathcal{S}_c = \{ s \mid AABB(s) \cap \Omega_c^+ \neq \emptyset \}.
```

The geometry backend must support two deployment modes:

- replicated metadata mode: all devices hold the packed scene, each device builds only local candidates
- sharded metadata mode: large scenes are partitioned, and only intersecting metadata is sent to each device

Recommended first implementation:

- replicated metadata for analytic primitives
- sharded metadata only when memory pressure requires it

## Materialization Modes

The geometry library should expose explicit modes rather than hidden heuristics:

- `label_only`
- `point_sampled`
- `subpixel_scalar`
- `subpixel_tensor`
- `derivative_enabled`

The FDTD backend then chooses the cheapest valid mode for the requested physics and accuracy target.

## Adjoint And Differentiation Contract

If geometry parameter `p` affects structure `s`, then the discretized chain should be written as:

```math
\frac{dJ}{dp}
=
\sum_q
\frac{\partial J}{\partial C_q}
\frac{\partial C_q}{\partial G_q}
\frac{\partial G_q}{\partial p},
```

where:

- `G_q` is a geometry-derived discrete quantity such as occupancy, fill fraction, or interface normal
- `C_q` is a solver coefficient such as `\varepsilon_q^{-1}`

This factorization is mandatory. It keeps:

- shape differentiation separate from constitutive differentiation
- the forward and adjoint discretizations aligned
- the code generator from entangling geometry-specific and physics-specific derivatives

## Code Generation Strategy

The geometry engine should be generated from:

- primitive descriptors
- query capability descriptors
- materialization mode descriptors
- chunking and acceleration descriptors

A primitive descriptor should include:

- parameter schema
- canonical query formulas
- exact/approximate fraction support
- derivative support level

The backend generator should then emit:

- primitive query kernels
- bounds kernels
- acceleration-build kernels
- overlap/projection kernels
- optional derivative kernels

This is the geometry analogue of the FDTD kernel-generation strategy.

## Design Patterns

- lower all frontends to one geometry IR
- keep geometry kernels primitive-centric and medium-agnostic
- separate broad phase, narrow phase, overlap resolution, and coefficient projection
- share the same discretization path between forward and adjoint geometry operations
- align geometry chunking with solver chunking
- treat repeated geometry as metadata plus chunk-local image enumeration
- batch by primitive family and query mode

## Design Anti-Patterns

- using frontend object methods directly during device rasterization
- global expansion of all periodic copies before chunking
- per-voxel scans over every scene primitive
- tying overlap semantics to one specific frontend
- implementing shape derivatives with a different discretization than forward initialization
- baking material-law logic into primitive query kernels
- monolithic kernels that do culling, containment, overlap resolution, and coefficient projection all at once

## Recommended First Implementation

Build a standalone geometry library with:

- frontend adapters for Meep/libctl and Tidy3D
- a backend-neutral geometry IR
- Warp as the first GPU backend
- analytic primitive kernels plus a triangle-mesh fallback path
- uniform-bin chunk acceleration for analytic primitives
- explicit materialization modes
- derivative support for the main design primitives first: box, cylinder, sphere, prism/polyslab

That gives a maintainable path to a multi-GPU geometry system that can initialize the solver, support adjoint design, and remain reusable as the rest of the stack evolves.
