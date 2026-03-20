# Full-Vector Mode Solver Spec

This document specifies the mode-solver subsystem needed for a maintainable, GPU-first FDTD package. The target is a full-wave, full-vector eigenmode solver suitable for:

- waveguide mode finding
- mode-source launch profiles
- mode monitor decomposition
- effective index and dispersion extraction
- adjoint overlap operators
- benchmark parity with Meep/Tidy3D mode workflows

The local reference `VectorModeSolver.jl` is a useful proof of the core mathematical structure: assemble a transverse full-vector finite-difference operator, solve for propagating eigenpairs, then reconstruct all field components and modal observables. The production solver should preserve that mathematical split while replacing the CPU sparse-assembly path with a GPU-oriented operator and eigensolver design.

## Purpose

The mode solver is not just a convenience utility. It is a core electromagnetic operator library that feeds several parts of the FDTD stack:

- source injection from guided eigenmodes
- mode decomposition of fields at monitors
- normalization of launched power
- broadband effective index seeding
- adjoint source construction from modal objectives

Because of this, the mode solver must use the same geometry/material conventions as the FDTD core, especially:

- the same geometry materialization pipeline
- the same anisotropic constitutive conventions
- the same interpolation/restriction rules when mapping between Yee fields and transverse slices

## Problem Class

The core problem is a frequency-domain eigenproblem on a 2d transverse cross-section for fields varying as:

```math
F(x, y, z, t) = \tilde{F}(x, y)\, e^{i \beta z - i \omega t},
```

where:

- `\omega` is fixed by the requested wavelength or frequency
- `\beta` is the propagation constant to be solved for
- `n_eff = \beta / k_0`
- the fields are fully vectorial

The production solver should support:

- isotropic dielectric guides
- diagonal and full in-plane anisotropy
- lossy and complex-valued materials
- optional magnetic materials
- bent/effective-index or curvilinear variants as future extensions

The first implementation can prioritize straight waveguides and translationally invariant cross-sections.

## Mathematical Form

The local Julia implementation solves an eigenproblem over transverse magnetic components:

```math
\hat{A}(\omega)\, \phi = \beta^2 \phi,
\qquad
\phi = \begin{bmatrix} H_x \\ H_y \end{bmatrix}.
```

This is a good production form because:

- it reduces the unknown count relative to solving all six field components directly
- longitudinal fields can be reconstructed afterward
- anisotropic permittivity can still be supported

The implementation then reconstructs:

```math
H_z = \frac{1}{i \beta}\left(\partial_x H_x + \partial_y H_y\right)
```

up to the discretization used by the chosen finite-difference stencil, and recovers `E` from discrete Maxwell relations:

```math
D = \frac{1}{i \omega} \nabla \times H,
\qquad
E = \varepsilon^{-1} D.
```

The exact operator algebra can vary by formulation, but the subsystem must preserve:

- a well-defined discrete eigenproblem
- a stable field-reconstruction map
- a consistent normalization and overlap convention

## Required Discretization Support

The mode solver should support:

- nonuniform transverse grids
- tensor permittivity on the cross-section
- boundary conditions on all transverse edges
- complex arithmetic
- interface-aware material sampling

The `VectorModeSolver.jl` reference already shows several of these:

- nonuniform `x` and `y`
- anisotropic `\varepsilon`
- boundary-condition flags on north/south/east/west edges
- sparse coupling stencil over neighboring transverse sites

The production mode solver must use the same geometry/material backends as the FDTD solver so that the cross-section seen by the eigenproblem matches the cross-section seen by time stepping.

## Required Boundary Conditions

At minimum:

- PEC/PMC-like parity boundaries
- mirror-even / mirror-odd symmetry boundaries
- PML or absorbing boundaries for leaky/radiative modes
- periodic/Bloch transverse boundaries for periodic guides

Important rule:

- boundary semantics must match the FDTD frontend semantics

This is required so that a mode source or mode monitor on a symmetric or periodic simulation slice is physically consistent with the time-domain problem.

## Output Contract

Each solved mode must provide:

- `\beta`
- `n_eff`
- `E_x, E_y, E_z`
- `H_x, H_y, H_z`
- complex power normalization
- propagation direction sign
- parity/symmetry metadata
- mode quality metadata such as residual and optional confinement/leakage indicators

Optional but useful:

- group index estimate
- dispersion derivatives
- polarization fraction
- confinement factor by material region

## Normalization And Orthogonality

The mode subsystem must choose and document one canonical normalization convention. A practical default is complex power normalization:

```math
P_m
=
\frac{1}{2}
\int_S
\left(E_m \times H_m^\ast\right)\cdot \hat{z}\, dS.
```

Then normalize modes so that:

- `P_m = +1` for forward modes
- `P_m = -1` for backward modes, or equivalently store the sign separately

For mode decomposition and adjoint use, the solver also needs a biorthogonal or conjugate-overlap convention for non-Hermitian problems:

```math
\langle m, n \rangle
=
\int_S
\left(E_m \times H_n^\ast\right)\cdot \hat{z}\, dS.
```

The important requirement is consistency. The same overlap convention must be used for:

- source amplitude normalization
- monitor mode coefficients
- adjoint source construction

## Kernel Families

The mode solver should compile into a small number of kernel families.

### Family A: Cross-Section Materialization

Purpose:

- materialize `\varepsilon`, `\mu`, conductivity, and region metadata on the transverse slice

This should reuse the geometry engine and coefficient projection logic wherever possible.

### Family B: Discrete Operator Coefficient Kernels

Purpose:

- compute local stencil coefficients for the chosen eigen-operator on the transverse grid

This is the production analogue of the coefficient formulas assembled in `VectorModeSolver.jl`.

### Family C: Sparse/Structured Operator Application

Purpose:

- apply the transverse operator to a vector without requiring CPU-style scalar sparse assembly

Recommended design:

- matrix-free stencil application for structured grids
- optional explicit sparse format for debugging or fallback solvers

### Family D: Eigensolver Kernels

Purpose:

- solve for a small set of eigenpairs nearest a target `\beta`, `n_eff`, or frequency region

Required capabilities:

- shift-invert or target-near search
- block eigensolve for multiple modes
- complex arithmetic
- residual estimation

### Family E: Field Reconstruction Kernels

Purpose:

- reconstruct `H_z`, `E_x`, `E_y`, `E_z`, and derived observables from the eigenvector state

### Family F: Normalization And Overlap Kernels

Purpose:

- compute Poynting flux normalization
- compute modal overlaps with arbitrary field slices
- compute orthogonality matrices when needed

### Family G: Mode-to-FDTD Coupling Kernels

Purpose:

- inject a mode profile as an FDTD source
- project FDTD field data onto solved modes at monitors
- build adjoint sources from modal objectives

## Data Layout

The GPU implementation should avoid CPU sparse-matrix assembly as the primary path.

Preferred layout:

- structured transverse grid metadata
- SoA material tensors and metric arrays
- stencil coefficient arrays by operator family
- dense block vectors for iterative eigensolvers

For structured finite differences, the best default is:

- matrix-free operator apply kernels
- optional CSR/BSR export only for debugging, verification, or third-party solver backends

This is important for performance and maintainability. A handwritten scalar sparse assembly loop like `VectorModeSolver.jl` is a good reference for formulas, but not the right backend design for a GPU production solver.

## Eigensolver Strategy

The solver must support a small number of target modes efficiently. That usually implies iterative methods such as:

- Arnoldi
- Lanczos where applicable
- Jacobi-Davidson
- LOBPCG for suitable formulations
- shift-invert methods when a robust linear solve backend is available

Recommended production strategy:

- matrix-free operator apply
- optional preconditioned iterative linear solve for shift-invert
- target-by-`n_eff` or target-by-`\beta` API

The API should expose:

- requested number of modes
- target `n_eff` or `\beta`
- tolerance
- maximum iterations
- optional orthogonality constraints or symmetry sector

## Multi-GPU Considerations

The mode solver is usually 2d, so it does not need the same decomposition strategy as a full 3d FDTD run. Still, the subsystem should remain GPU-native and should support scale when cross-sections become large or when many solves are batched.

Useful modes:

- single-GPU mode as the default
- batched multi-solve mode across GPUs
- distributed operator apply only if cross-sections become genuinely large

The important point is not to overdesign distributed eigensolving before it is necessary. In most photonics workloads, the high-value win is:

- very fast single-GPU solves
- batched solves for parameter sweeps, source fitting, and monitor libraries

## Integration With FDTD Sources

The mode solver must feed a source operator:

```math
\mathcal{S}_{mode} : (m, a(t), \Sigma) \mapsto J(\Sigma, t),
```

where:

- `m` is the chosen mode
- `a(t)` is the source waveform
- `\Sigma` is the source cross-section or plane

The solver needs:

- field-profile interpolation from mode grid to Yee source locations
- normalization to requested launched power or amplitude
- forward/backward mode selection
- optional phase centering and parity handling

## Integration With Mode Monitors

The mode subsystem must also feed a monitor operator:

```math
\mathcal{M}_{mode} : (E, H)|_{\Sigma} \mapsto \{c_m^+, c_m^-\},
```

where modal coefficients come from overlap integrals between the solved mode basis and the sampled FDTD fields.

This requires:

- a stable overlap convention
- interpolation from Yee locations to the mode-monitor sampling grid
- optional DFT accumulation before mode projection
- support for forward and backward propagating components

## Adjoint Requirements

For objectives defined in terms of mode amplitudes or modal power, the mode solver contributes two operator families:

1. forward modal projection
2. adjoint source reconstruction from the transpose or adjoint of that projection

The mode subsystem therefore must expose:

- differentiable mode normalization metadata
- overlap operators with explicit adjoints
- a clear policy for whether the modes themselves are differentiated with respect to geometry/material parameters

For the first implementation, a practical split is:

- differentiate through mode overlaps and source amplitudes
- treat the solved mode library as fixed during a single forward/adjoint pair unless the workflow explicitly requests differentiating through the eigensolve

Differentiating through the eigensolve itself should be a higher tier because it is substantially more delicate.

## Shared IR

The LLM-facing implementation spec should include a mode-solver IR roughly of the form:

```text
ModeSolveRequest
  wavelength_or_frequency
  cross_section_geometry
  cross_section_grid
  boundary_spec
  target_beta_or_neff
  num_modes
  symmetry_sector?
  solver_options

ModeOperatorIR
  formulation
  field_basis
  material_coefficients
  stencil_family
  operator_apply_contract
  reconstruction_contract
  normalization_contract
```

And coupling operators:

```text
ModeSourceIR
  mode_id
  source_plane
  amplitude_rule
  interpolation_rule

ModeMonitorIR
  mode_basis_ids
  monitor_plane
  overlap_rule
  output_coefficients
```

## Design Patterns

- keep the mode solver as a first-class subsystem, not an afterthought utility
- share geometry/material materialization with the FDTD core
- prefer matrix-free operator application on GPU
- keep normalization and overlap conventions explicit
- separate eigensolve from field reconstruction
- reuse the same mode library for source injection and monitor projection
- reserve differentiating through the eigensolve as a higher feature tier

## Design Anti-Patterns

- CPU scalar sparse assembly as the only implementation path
- using a different geometry discretization than the FDTD solver
- undocumented normalization conventions
- source injection and monitor decomposition using incompatible overlap formulas
- baking backend-specific eigensolver assumptions into the frontend API
- assuming single-mode, lossless, Hermitian behavior everywhere

## Recommended First Implementation

Build the first production mode solver with:

- a structured-grid full-vector transverse formulation
- anisotropic permittivity support
- matrix-free GPU operator apply
- iterative target-near eigensolve
- explicit field reconstruction and power normalization
- mode-source and mode-monitor coupling operators

Use `VectorModeSolver.jl` as a formula reference, especially for the anisotropic transverse stencil and field reconstruction path, but not as the backend architecture. The production implementation should share the same IR/codegen philosophy as the FDTD and geometry subsystems.
