# Hybrid Adjoint and Discrepancy Notes

## Purpose

This note captures what the GPU solver must expose for the hybrid time-/frequency-domain adjoint method and where the current Meep implementation appears to diverge from the dissertation-level theory.

Primary local references:

- `phd_thesis/chapters/chapter3.tex`
- `meep/python/adjoint`
- `meep/src/dft.cpp`
- `meep/src/near2far.cpp`
- `meep/src/meepgeom.cpp`

## Core Adjoint Contract

The solver must support the following pipeline:

1. update design/material state
2. run a forward time-domain simulation
3. accumulate per-frequency DFT state for:
   - objective monitors
   - design regions
4. evaluate scalar objective functions from complex monitor outputs
5. differentiate objective functions with respect to monitor outputs
6. synthesize adjoint sources
7. run one adjoint simulation per scalar objective
8. accumulate adjoint design-region DFT fields
9. combine forward and adjoint DFT fields into `dJ/drho`

This is the important compatibility contract. The exact runtime or framework is secondary.

## Required Adjoint Operators

The solver must provide first-class support for:

- DFT accumulation during time stepping
- direct DFT field sampling
- eigenmode overlap evaluation
- near-to-far evaluation
- adjoint source synthesis from monitor-space gradients
- design-region field capture on Yee-grid offsets
- material-grid gradient recombination

## Required Broadband Adjoint Source Capability

The dissertation uses a compact frequency-basis fit for broadband adjoint sources. Meep implements this with `FilteredSource` and Nuttall-window basis fitting.

The new solver should preserve the concept but avoid making the normalization path an opaque heuristic.

The source-synthesis subsystem should explicitly represent:

- basis function family
- fitting frequencies
- temporal duration
- fitting matrix
- pseudoinverse or stabilized linear solve
- source replay basis coefficients

## Most Important Meep-vs-Theory Discrepancies

### 1. Material-grid differentiation is still too finite-difference-like

The biggest issue is in the material-gradient path.

The dissertation presents the gradient as an operator-level recombination on the Yee grid with timestep correction. In the Meep implementation, the material-grid path still relies heavily on local finite-difference evaluation inside `meep/src/meepgeom.cpp`, including the smoothed/material-grid case.

Consequence:

- do not copy this literally into the GPU solver
- replace it with exact or at least explicit analytic/JVP kernels for interpolation, projection, smoothing, damping, and Yee restriction

### 2. Adjoint source normalization contains empirical corrections

In `meep/python/adjoint/objective.py`, the adjoint-source scale includes:

- a manual DTFT of the source waveform
- a discrete-time derivative correction
- an extra phase correction that is documented as an empirical workaround

Consequence:

- the new solver should make this normalization an explicit, testable operator
- the normalization must be specified in the IR, not hidden in Python helper code

### 3. Reciprocity fixes are not uniformly enforced

`OptimizationProblem` flips cylindrical `m` and Bloch `k_point` during the adjoint run, which is consistent with the reciprocity-based reasoning in the dissertation.

The wrapper path does not uniformly preserve that behavior.

Consequence:

- reciprocity transforms must be part of the core adjoint-run specification
- they cannot live only in a high-level helper path

### 4. Parallelism in practice is narrower than in theory

The dissertation discusses broader simulation-group parallelism for optimization workloads. The current Python layer still executes one adjoint run per scalar objective in sequence.

Consequence:

- the solver runtime should make objective-level and simulation-group parallelism explicit
- this is especially important once multiple GPUs are available

### 5. Near-to-far generality is still limited

The dissertation frames near-to-far backprop as a general transposed Green-function convolution and points toward layered or inhomogeneous extensions.

Meep's current documented implementation remains tied to more restrictive assumptions.

Consequence:

- the new solver should define near-to-far as an operator family
- the homogeneous-medium case should be only one backend/operator instance

## Adjoint-Specific Design Requirements

- Design-region monitors must be persistent and reusable across forward/adjoint phases.
- Monitor state must be chunk-local and reducible across GPUs.
- Adjoint source synthesis must be separable from adjoint source injection.
- All Yee-grid interpolation/restriction operators used in the gradient path must be explicit and testable.
- The gradient path must support analytic kernels for material-grid maps.
- Near-to-far and mode-overlap operators must expose their transpose/backprop forms, not only their forward evaluation forms.

## Recommended Improvement Over Meep

Preserve Meep's high-level workflow:

- forward monitors
- monitor-space AD
- adjoint source placement
- forward/adjoint design-region recombination

Improve the low-level contract:

- analytic Yee-aware differentiation for material grids
- explicit reciprocity transforms
- explicit source normalization operators
- operator-family abstraction for Green-function-based objectives

That yields a solver that is more faithful to the dissertation than the current Meep implementation in the parts that matter most for a new GPU implementation.

