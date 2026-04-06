# Phase 1 Maxwell Solver Loop

Drive Phase 1 of autofdtd from an almost-empty repository to a credible, pip-installable, GPU-first FDTD package whose feature plan is anchored explicitly to the local Tidy3D EM API surface. The loop must produce a Tidy3D-inspired public API, a tagged execution IR, a scene-compilation pipeline, a chunk-based multi-node-ready runtime, Warp kernel conventions, concrete implementations for the highest-priority Tidy3D feature families, validation examples, and documentation without copying protected code or docs from any reference project.

Statuses: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `BLOCKED`

## [COMPLETED] task-001 - Create the package, tooling, and docs scaffold around the planned feature surface
Success: Create a pip-installable src-layout package with module namespaces, development tooling, test entrypoints, and docs scaffolding that are aligned with the feature checklist and subsystem boundaries.
Attempts: 1
Last Run: task-001-attempt-001
Last Duration: 335.1s
Completed At: 2026-04-06T02:09:22Z
Notes: The Tidy3D feature checklist is already part of prep and should be consumed here rather than regenerated.

## [COMPLETED] task-002 - Implement the tagged core Simulation, Scene, and Structure data model
Success: Provide the foundational public and IR-facing container models for Simulation, Scene, and Structure, including ordered structure semantics, names, priority behavior, and serialization scaffolding.
Attempts: 1
Last Run: task-002-attempt-001
Last Duration: 318.3s
Completed At: 2026-04-06T02:14:41Z
Notes: This is the container backbone that many later feature tasks depend on.

## [COMPLETED] task-003 - Define the tagged execution IR and serialization contract
Success: Land an inspectable, versioned IR schema and serialization plan that can represent realistic Phase 1 simulations independently of the public API and is suitable for local or remote execution packaging.
Attempts: 1
Last Run: task-003-attempt-001
Last Duration: 249.2s
Completed At: 2026-04-06T02:18:50Z
Notes: The IR should be specific enough to support later feature tasks without redesign.

## [COMPLETED] task-004 - Implement core normalization and validation behavior
Success: Provide validation and normalization passes for names, bounds, basic shape constraints, unsupported-feature errors, and selected warn-and-coerce behaviors that align with the feature checklist.
Attempts: 1
Last Run: task-004-attempt-001
Last Duration: 499.7s
Completed At: 2026-04-06T02:27:10Z
Notes: This task should make later feature work safer instead of treating validation as cleanup at the end.

## [COMPLETED] task-005 - Implement shared model-base and serialization helpers
Success: Provide the shared base-model, tagging, copy-or-update, and serialization helpers needed by the public API and IR layers so later feature tasks build on a common contract.
Attempts: 1
Last Run: task-005-attempt-001
Last Duration: 164.9s
Completed At: 2026-04-06T02:29:54Z
Notes: This keeps tagging and serialization from being reimplemented ad hoc in every feature family.

## [COMPLETED] task-006 - Implement primitive geometry support for Box, Sphere, and Cylinder
Success: Provide public and IR representations plus tests for Box, Sphere, and Cylinder geometries, including bounds, transforms needed for scene compilation, and precedence interaction inside structures.
Attempts: 1
Last Run: task-006-attempt-001
Last Duration: 406.1s
Completed At: 2026-04-06T02:36:41Z
Notes: Start with the simplest Tidy3D geometry primitives first.

## [COMPLETED] task-007 - Implement PolySlab and polygon-driven geometry support
Success: Support PolySlab-style polygon extrusion geometry with explicit scope and limitation documentation, plus tests that cover representative planar photonics cases.
Attempts: 1
Last Run: task-007-attempt-001
Last Duration: 460.5s
Completed At: 2026-04-06T02:44:21Z
Notes: This is one of the most important geometry families for Tidy3D-style compatibility.

## [COMPLETED] task-008 - Implement geometry grouping, transforms, and clip-style wrappers
Success: Provide GeometryGroup-style composition plus the minimum transform and clip operation support needed for credible Phase 1 scene construction.
Attempts: 1
Last Run: task-008-attempt-001
Last Duration: 531.5s
Completed At: 2026-04-06T02:53:13Z
Notes: This task should turn individual primitives into useful scene-building blocks.

## [COMPLETED] task-009 - Implement geometry arrays, precedence, and repeated placement semantics
Success: Support repeated or array-style geometry placement and explicit precedence semantics for ordered structures with tests that verify overlap behavior.
Attempts: 1
Last Run: task-009-attempt-001
Last Duration: 378.3s
Completed At: 2026-04-06T02:59:31Z
Notes: This is where structure ordering becomes concrete.

## [COMPLETED] task-010 - Implement grid controls for UniformGrid, CustomGrid, and GridSpec core paths
Success: Provide the core grid specification surface needed for Phase 1 simulations, including uniform grids, custom boundaries or custom grids, and GridSpec-style composition.
Attempts: 1
Last Run: task-010-attempt-001
Last Duration: 678.8s
Completed At: 2026-04-06T03:10:50Z
Notes: Grid support should be concrete enough to drive later discretization and mode-solver work.

## [COMPLETED] task-011 - Implement AutoGrid and refinement-spec planning surfaces
Success: Support an initial AutoGrid-style and refinement-spec surface with clear limitations and lowering behavior, even if the first meshing policy is simpler than Tidy3D's full behavior.
Attempts: 1
Last Run: task-011-attempt-001
Last Duration: 414.5s
Completed At: 2026-04-06T03:17:44Z
Notes: This task can begin with planning or simplified refinement behavior, but it must not be hand-wavy.

## [COMPLETED] task-012 - Implement SubpixelSpec and core averaging policy models
Success: Provide SubpixelSpec-style controls and the core averaging or staircasing policy representations needed to drive geometry materialization behavior.
Attempts: 1
Last Run: task-012-attempt-001
Last Duration: 629.8s
Completed At: 2026-04-06T03:28:14Z
Notes: This task is about policy modeling before the heavier compiler behavior lands.

## [COMPLETED] task-013 - Implement isotropic Medium, PECMedium, and PMCMedium with coefficient compilation and constitutive kernels
Success: Support the foundational isotropic material family, including plain Medium and perfect-conductor medium variants, with tests for serialization, validation, structure application, coefficient preparation, and the corresponding constitutive-update kernel paths.
Attempts: 1
Last Run: task-013-attempt-001
Last Duration: 589.0s
Completed At: 2026-04-06T03:38:03Z
Notes: These are the baseline material types many later tasks assume exist.

## [COMPLETED] task-014 - Implement PoleResidue medium support with coefficient prep and update kernels
Success: Support PoleResidue-style dispersive media with explicit API and IR models, coefficient preparation, auxiliary-state handling, update-kernel support, and representative tests.
Attempts: 1
Last Run: task-014-attempt-001
Last Duration: 522.2s
Completed At: 2026-04-06T03:46:45Z
Notes: Keep PoleResidue separate because it is the most general dispersive family and often subsumes the others conceptually.

## [COMPLETED] task-015 - Implement Sellmeier medium support with coefficient prep and update kernels
Success: Support Sellmeier media or a clearly documented supported subset, including API and IR models, coefficient preparation, update-kernel support, and representative validation tests.
Attempts: 1
Last Run: task-015-attempt-001
Last Duration: 299.6s
Completed At: 2026-04-06T03:51:45Z
Notes: This stays separate so optical-material support is not hidden inside a large dispersive bucket.

## [COMPLETED] task-016 - Implement Lorentz, Drude, and Debye medium support with update kernels
Success: Support Lorentz, Drude, and Debye medium families or a clearly delimited subset, including API and IR models, coefficient preparation, update-kernel support, and tests that show each family exercises the intended constitutive path.
Attempts: 1
Last Run: task-016-attempt-001
Last Duration: 485.0s
Completed At: 2026-04-06T03:59:50Z
Notes: These families are close enough to share a task without becoming ambiguous if the tests name each family explicitly.

## [COMPLETED] task-017 - Implement anisotropic medium families, Medium2D policy, and anisotropic kernel support
Success: Support the initial anisotropic medium surface, including a clear policy for AnisotropicMedium, FullyAnisotropicMedium, and Medium2D-related behavior, with explicit tests, limitations, coefficient preparation, and anisotropic kernel support.
Attempts: 19
Last Run: task-017-attempt-019
Last Duration: 502.8s
Completed At: 2026-04-06T19:06:42Z
Notes: This task should respect the fact that subpixel smoothing can create effective anisotropy and must land before second-order subpixel materialization is treated as implemented.

## [COMPLETED] task-018 - Implement perturbation, lossy-metal, and advanced-material defer buckets
Success: Provide an explicit Phase 1 handling policy for PerturbationMedium, LossyMetalMedium, custom media, and other advanced material families so the API surface is mapped rather than ignored.
Attempts: 1
Last Run: task-018-attempt-001
Last Duration: 399.5s
Completed At: 2026-04-06T19:13:22Z
Notes: This is intentionally a policy-heavy task if full implementation is not yet realistic.

## [COMPLETED] task-019 - Implement boundary support for Periodic, PECBoundary, PMCBoundary, and Boundary containers
Success: Support the foundational non-absorbing boundary surface, including Boundary and BoundarySpec composition, periodic behavior, conductor boundaries, and the runtime transform or kernel support needed by the timestep loop.
Attempts: 1
Last Run: task-019-attempt-001
Last Duration: 708.3s
Completed At: 2026-04-06T19:25:10Z
Notes: These are the first boundary primitives the compiler and runtime should rely on.

## [COMPLETED] task-020 - Implement BlochBoundary, symmetry-aware metadata, and phase-aware runtime transforms
Success: Provide BlochBoundary support and the related symmetry-aware metadata and runtime transform support needed for chunk planning and execution lowering.
Attempts: 1
Last Run: task-020-attempt-001
Last Duration: 460.9s
Completed At: 2026-04-06T19:32:51Z
Notes: This task is important because periodic-phase behavior leaks into runtime metadata and not just public API models.

## [COMPLETED] task-021 - Implement PML boundary support with coefficient prep and kernel stages
Success: Support PML boundaries with explicit API and IR models, coefficient preparation, boundary-layer state allocation, kernel-stage integration, and representative absorption tests.
Attempts: 1
Last Run: task-021-attempt-001
Last Duration: 632.6s
Completed At: 2026-04-06T19:43:24Z
Notes: PML is important enough to deserve its own task instead of sharing a generic absorbing-boundary bucket.

## [COMPLETED] task-022 - Implement StablePML and Absorber boundary support with coefficient prep and kernel stages
Success: Support StablePML and Absorber boundaries or a clearly delimited subset, including API and IR models, coefficient preparation, kernel-stage integration, and representative tests for practical absorbing-boundary behavior.
Attempts: 1
Last Run: task-022-attempt-001
Last Duration: 524.5s
Completed At: 2026-04-06T19:52:08Z
Notes: StablePML and Absorber are separate enough from baseline PML to justify their own task.

## [COMPLETED] task-023 - Implement ABCBoundary, ModeABCBoundary, and advanced boundary defer policy
Success: Provide an explicit Phase 1 treatment for ABCBoundary, ModeABCBoundary, and other specialized boundary variants, including concrete implementation where practical and clear reject-or-defer behavior otherwise.
Attempts: 1
Last Run: task-023-attempt-001
Last Duration: 823.0s
Completed At: 2026-04-06T20:05:51Z
Notes: This task prevents specialized Tidy3D boundary features from disappearing into ambiguity.

## [COMPLETED] task-024 - Implement Simulation runtime controls for timestep count, shutoff, and convergence policy
Success: Support the core Simulation runtime-control surface needed for practical runs, including run-time or step-count control, shutoff or early-stop criteria, and clearly defined convergence-related behavior.
Attempts: 1
Last Run: task-024-attempt-001
Last Duration: 696.8s
Completed At: 2026-04-06T20:17:28Z
Notes: This task makes convergence and stop behavior explicit instead of leaving it buried inside the simulation loop.

## [COMPLETED] task-025 - Implement detailed simulation logging, progress reporting, and convergence evidence capture
Success: Execution paths emit structured progress logs, stop-reason records, convergence or shutoff evidence, and enough runtime detail to diagnose practical simulation behavior during end-to-end runs.
Attempts: 1
Last Run: task-025-attempt-001
Last Duration: 341.6s
Completed At: 2026-04-06T20:23:10Z
Notes: Keep this distinct from benchmark metrics so debugging and validation remain first-class.

## [COMPLETED] task-026 - Implement GaussianPulse and ContinuousWave source-time profiles
Success: Support GaussianPulse and ContinuousWave source-time profiles with explicit parameter validation, serialization, IR lowering, and representative waveform tests.
Attempts: 1
Last Run: task-026-attempt-001
Last Duration: 752.9s
Completed At: 2026-04-06T20:35:43Z
Notes: These are the two most common source-time profiles and should be available early.

## [COMPLETED] task-027 - Implement BroadbandPulse and CustomSourceTime profiles
Success: Support BroadbandPulse and CustomSourceTime or a clearly delimited subset, including API and IR models, validation behavior, and representative tests.
Attempts: 1
Last Run: task-027-attempt-001
Last Duration: 279.0s
Completed At: 2026-04-06T20:40:22Z
Notes: Keep the more specialized source-time families separate so missing behavior cannot hide behind the common waveforms.

## [COMPLETED] task-028 - Implement UniformCurrentSource with placement semantics and injection kernels
Success: Support UniformCurrentSource with explicit placement semantics, source-time integration, IR lowering, injection-kernel support, and representative end-to-end tests.
Attempts: 1
Last Run: task-028-attempt-001
Last Duration: 547.3s
Completed At: 2026-04-06T20:49:29Z
Notes: This is the cleanest current-source family and should come before the more specialized dipole and custom cases.

## [QUEUED] task-029 - Implement PointDipole and CustomCurrentSource with injection kernels
Success: Support PointDipole and CustomCurrentSource or a clearly supported subset, including validation, IR lowering, injection-kernel support, and representative tests.
Attempts: 88
Last Run: task-029-attempt-088
Last Duration: 3.1s
Completed At: -
Notes: PointDipole is important because it stresses interpolation and restriction behavior directly.

## [QUEUED] task-030 - Implement CustomFieldSource with field-injection runtime support
Success: Support CustomFieldSource or a clearly delimited subset, including validation, IR lowering, field-injection runtime support, and representative tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Keep this separate from ModeSource so the solver-coupled path stays explicit.

## [QUEUED] task-031 - Implement the vector mode solver integration path
Success: Introduce a GPU-capable or GPU-ready mode-solver subsystem whose interface is compatible with sampled epsilon tensors over deterministic cross-sections and includes tests for representative cases.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Use VectorModesolver.jl as a formulation and interface reference, not as a blind dependency surface. This task intentionally lands before ModeSource and mode-monitor work.

## [QUEUED] task-032 - Implement ModeSource with solver-backed mode definitions and injection support
Success: Support ModeSource with explicit integration into the mode-solver or mode-definition surface, including validation, IR lowering, source planning, injection support, and representative tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task comes after the mode solver on purpose so it has a real backend to target.

## [QUEUED] task-033 - Implement PlaneWave plus angular-spec models and injection support
Success: Support PlaneWave along with the angular-spec models it needs, including validation, IR lowering, source planning, injection support, and representative tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: PlaneWave is a core optical source and should not be hidden inside a broader beam-source task.

## [QUEUED] task-034 - Implement GaussianBeam and AstigmaticGaussianBeam with injection support
Success: Support GaussianBeam and AstigmaticGaussianBeam or a clearly delimited subset, including validation, source planning, injection support, and representative tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Beam sources have enough parameter and geometry complexity to deserve their own task.

## [QUEUED] task-035 - Implement TFSF support and source-frame policy
Success: Support the TFSF source family and any required source-frame metadata, boundary interaction logic, and runtime injection support, or provide explicit Phase 1 limitation handling.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: TFSF is important enough to deserve its own task because of its runtime implications.

## [QUEUED] task-036 - Implement the monitor naming and SimulationData access model
Success: Provide the foundational monitor naming, result-registration, and SimulationData-style access pattern that later monitor tasks will populate.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This is the common data contract for all monitor feature families.

## [QUEUED] task-037 - Implement field-monitor families for FieldMonitor, FieldTimeMonitor, and AuxFieldTimeMonitor
Success: Support the core field-monitor families with runtime state, accumulation kernels, and tests for representative field recording cases.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: These are the bread-and-butter monitor types and should be ready before more specialized monitors.

## [QUEUED] task-038 - Implement medium-property monitors for MediumMonitor and PermittivityMonitor
Success: Support the main medium-property monitor families, integrate them with the geometry-material compilation path, and provide the extraction kernels or runtime paths they need.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: These monitors help validate materialization and geometry lowering directly.

## [QUEUED] task-039 - Implement flux-monitor families for FluxMonitor and FluxTimeMonitor
Success: Support the core flux-monitor families with integration behavior, flux-integration kernels, data products, and tests for representative energy-flow setups.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Flux monitoring is foundational for many optical validation examples.

## [QUEUED] task-040 - Implement mode-monitor families for ModeMonitor and ModeSolverMonitor
Success: Support the main mode-monitor families, connect them to the mode-solver or mode-analysis surface expected in Phase 1, and provide the overlap or projection kernels they require.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: These monitors bridge runtime fields and the mode-solver subsystem, so they come after the mode solver intentionally.

## [QUEUED] task-041 - Implement overlap-monitor families for GaussianOverlapMonitor variants
Success: Provide GaussianOverlapMonitor and AstigmaticGaussianOverlapMonitor support or a documented and enforced defer policy consistent with the feature matrix.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task is narrow on purpose so its scope stays clear.

## [QUEUED] task-042 - Implement projection and far-field monitor families
Success: Support FieldProjectionAngleMonitor, FieldProjectionCartesianMonitor, FieldProjectionKSpaceMonitor, and DirectivityMonitor or a documented Phase 1 subset with clear error behavior and explicit projection-kernel support.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Projection monitors have many edge cases, so keep the task focused and explicit.

## [QUEUED] task-043 - Implement DiffractionMonitor and diffraction-specific limitations
Success: Provide DiffractionMonitor support or a narrow supported subset with explicit validation for unsupported order-count or medium constraints and the corresponding diffraction-kernel support.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Diffraction is important enough to split out from the other projection monitors.

## [QUEUED] task-044 - Implement surface-field monitor families
Success: Support SurfaceFieldMonitor and SurfaceFieldTimeMonitor or a documented subset with explicit runtime behavior, surface-sampling kernels, and tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Surface monitors are useful both directly and as building blocks for postprocessing.

## [QUEUED] task-045 - Build the scene-compilation pipeline from public API and IR into compiled runtime artifacts
Success: Document and begin implementing a compiler-style pipeline that lowers validated public API objects into IR and then into coefficient fields, source plans, monitor plans, chunk metadata, and runtime arrays.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task ties the feature-specific surfaces into an executable spine.

## [QUEUED] task-046 - Implement geometry discretization, interpolation, and subpixel materialization
Success: Build the geometry-to-grid compilation path that handles precedence, interpolation or restriction concerns, coefficient sampling, and a scoped first pass at subpixel smoothing with tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This is where the geometry and material feature families become runtime data. Second-order subpixel smoothing depends on the anisotropic material path already existing.

## [QUEUED] task-047 - Define the chunk contract and multi-node decomposition architecture
Success: Land an architecture note that defines what a chunk owns, how chunks expose halo and monitor metadata, how chunk placement maps onto devices or nodes, and how this avoids blocking later phases.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Base this on the Meep findings that chunk is the core architectural unit rather than rank.

## [QUEUED] task-048 - Define Warp backend conventions and instrumentation strategy
Success: Document and implement the initial Warp backend conventions for module organization, dtype or device policy, persistent arrays, capture-safe stepping, and profiling hooks.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task establishes backend guardrails before the heavy kernels land.

## [QUEUED] task-049 - Implement vacuum Yee update kernels and the capture-safe step skeleton
Success: Document and implement the explicit vacuum E and H update kernels, field staggering rules, and capture-safe step skeleton with clear launch ordering and representative tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task isolates the core timestepping spine before material, boundary, and source complications are layered on.

## [QUEUED] task-050 - Implement constitutive and material-update kernel stages
Success: Provide the constitutive and material-update kernel stages needed for isotropic, dispersive, and anisotropic materials that Phase 1 claims to support, with clear staging and tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task ties the material feature tasks into the runtime in an explicit way.

## [QUEUED] task-051 - Implement boundary and absorbing-layer kernel stages
Success: Provide the explicit runtime or kernel stages needed for conductor boundaries, periodic or Bloch remapping, and absorbing-layer updates claimed by Phase 1.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Keep boundary runtime support separate from source and material kernel work.

## [QUEUED] task-052 - Implement source-injection kernel stages and source scheduling
Success: Provide the explicit kernel or runtime stages for current-source, field-source, beam, plane-wave, and TFSF injection claimed by Phase 1, with clear scheduling semantics and tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task keeps source runtime support visible instead of scattering it across API-only tasks.

## [QUEUED] task-053 - Implement monitor accumulation, projection, and extraction kernel stages
Success: Provide the explicit runtime or kernel stages needed for field, flux, mode, projection, diffraction, and surface-monitor accumulation and extraction claimed by Phase 1.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task makes the monitor runtime spine concrete instead of assuming monitor API support implies executable support.

## [QUEUED] task-054 - Implement chunk-aware runtime orchestration and staged halo exchange
Success: Create runtime components that partition compiled scenes into chunks, manage stage-specific halo metadata, and provide the halo pack, unpack, and exchange kernel support needed for chunk-aware control flow.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Keep this bulk-synchronous and credible first; overlap and advanced scheduling can stay as later extensions.

## [QUEUED] task-055 - Wire the public API to normalization, IR lowering, and execution entrypoints
Success: Users can define nontrivial Tidy3D-style simulations, have them normalized and lowered into the IR, inspect or serialize that IR, and execute them through the same backend entrypoints.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task should prove that execution is driven by the IR rather than the original Python object graph.

## [QUEUED] task-056 - Implement runtime metrics, benchmark helpers, and execution evidence capture
Success: Execution paths emit initialization time, JIT time, steady-state timestep metrics, Gcells-per-second style metrics, and other key runtime evidence in a structured and testable form.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task should align with the Warp instrumentation strategy and the GPU benchmarking paper.

## [QUEUED] task-057 - Implement near-to-far and related postprocessing on top of the monitor infrastructure
Success: Provide an initial near-to-far workflow and other key postprocessing routines that build on the DFT monitor foundation and include the postprocessing-kernel support they require, backed by examples or tests.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: Do not implement near-to-far as a disconnected special case.

## [QUEUED] task-058 - Create canonical validation examples and feature-integration tests
Success: Land runnable validation examples and automated tests that exercise the implemented Phase 1 feature families, including vacuum propagation, dielectric slab or waveguide, PML boundaries, symmetry-reduced behavior, source injection, monitor recording, mode-source or mode-monitor workflows, convergence or shutoff behavior, and near-to-far or diffraction-related workflows where supported.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: These examples should become the evidence base for both feature coverage and later validation work.

## [QUEUED] task-059 - Finish README, feature coverage docs, unsupported-feature policy, and Phase 1 readiness summary
Success: Document architecture, current feature coverage against the Tidy3D matrix, unsupported-feature behavior, developer workflows, and the evidence that Phase 1 is ready to hand into deeper validation and optimization phases.
Attempts: 0
Last Run: -
Last Duration: -
Completed At: -
Notes: This task should consolidate the Phase 1 evidence base rather than inventing new architecture.
