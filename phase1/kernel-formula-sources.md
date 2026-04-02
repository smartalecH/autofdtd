# Kernel Formula Sources

This file is prep context for the Phase 1 loop. It maps feature families to the local references most likely to contain the governing update formulas or implementation patterns.

It is not meant to be the final formula notebook. It is the starting map the loop should use before inventing or guessing update equations.

## Core Yee Timestepping

- Primary purpose:
  Derive the base `E/H` update equations, field staggering, indexing conventions, and the main timestep stage graph.
- First references:
  - `../meep/src/update_eh.cpp`
  - `../Khronos.jl/src/Timestep.jl`
  - `../fdtdx/src/fdtdx/fdtd/update.py`
  - `papers/gpu_benchmarking.pdf`
- Expected outputs:
  - vacuum `E` update formula set
  - vacuum `H` update formula set
  - storage/indexing conventions for the chosen Yee layout
  - kernel family split for `vacuum`, `constitutive`, `sources`, `boundaries`, `diagnostics`

## Isotropic Constitutive Updates

- Primary purpose:
  Map `Medium`, `PECMedium`, and `PMCMedium` into coefficients and runtime update logic.
- First references:
  - `../tidy3d/tidy3d/components/medium.py`
  - `../meep/src/update_eh.cpp`
  - `../Khronos.jl/src/Geometry.jl`
- Expected outputs:
  - coefficient-preparation rules
  - constitutive-update kernel path for isotropic materials
  - explicit PEC/PMC runtime treatment

## Dispersive Material Families

- Primary purpose:
  Settle the update-state formulas and auxiliary-field requirements for `PoleResidue`, `Lorentz`, `Drude`, and `Debye`-style media.
- First references:
  - `../tidy3d/tidy3d/components/medium.py`
  - `../meep/src/update_pols.cpp`
  - `../meep/src/susceptibility.cpp`
  - `../meep/src/material_data.hpp`
- Expected outputs:
  - per-family auxiliary-state definitions
  - coefficient-preparation rules
  - update-kernel formulas
  - supported/deferred list for each dispersive family

## Anisotropic Material Updates

- Primary purpose:
  Determine how anisotropic constitutive updates alter ownership, halo requirements, and kernel specialization.
- First references:
  - `../tidy3d/tidy3d/components/medium.py`
  - `../meep/src/update_eh.cpp`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - tensor-coefficient representation
  - update-kernel formulas for supported anisotropic cases
  - explicit limitations for unsupported tensor structures

## Subpixel Smoothing And Effective Materialization

- Primary purpose:
  Determine how geometry/material interfaces are averaged and when that induces effective anisotropy.
- First references:
  - `../tidy3d/tidy3d/components/subpixel_spec.py`
  - `../GeometryPrimitives.jl/src/util/vxlcut.jl`
  - `../libctl/utils/geom.c`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - staircasing vs averaging policy behavior
  - effective coefficient generation rules
  - clear boundary between implemented and deferred subpixel methods

## Boundary Layers

- Primary purpose:
  Settle runtime formulas and kernel structure for `PML`, `Absorber`, and related boundary families.
- First references:
  - `../tidy3d/tidy3d/components/boundary.py`
  - `../meep/src/boundaries.cpp`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - per-boundary family coefficient/state definitions
  - update-kernel formulas or runtime transforms
  - supported/deferred list for `StablePML`, `ABCBoundary`, `ModeABCBoundary`

## Periodic, Bloch, And Symmetry Transforms

- Primary purpose:
  Clarify what is a kernel update versus what is a runtime/topology transform.
- First references:
  - `../tidy3d/tidy3d/components/boundary.py`
  - `../meep/doc/docs/Chunks_and_Symmetry.md`
  - `../meep/src/boundaries.cpp`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - logical-vs-stored topology rules
  - phase-transform rules for Bloch boundaries
  - ownership/halo implications for symmetry-reduced layouts

## Source Injection

- Primary purpose:
  Determine the interpolation/restriction and injection formulas for current sources, field sources, plane waves, beams, mode sources, and TFSF.
- First references:
  - `../tidy3d/tidy3d/components/source/current.py`
  - `../tidy3d/tidy3d/components/source/field.py`
  - `../meep/src/loop_in_chunks.cpp`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - injection metadata model
  - source-specific injection kernel/runtime rules
  - restrictions for unsupported source/geometry combinations

## Monitor Accumulation And Postprocessing

- Primary purpose:
  Settle the runtime formulas for field sampling, DFT accumulation, flux integration, mode overlap, and near-to-far transforms.
- First references:
  - `../tidy3d/tidy3d/components/monitor.py`
  - `../tidy3d/tidy3d/components/data/monitor_data.py`
  - `../meep/src/near2far.cpp`
  - `papers/meep_paper.pdf`
- Expected outputs:
  - field-sampling kernel rules
  - DFT accumulator formulas
  - flux-integration formulas
  - mode-monitor accumulation formulas
  - near-to-far postprocessing formulas

## Chunk Runtime And Halo Exchange

- Primary purpose:
  Define the pack/unpack/exchange kernel families and the ownership rules that determine what data must move between chunks.
- First references:
  - `../meep/src/structure.cpp`
  - `../meep/src/boundaries.cpp`
  - `papers/systolic_fdtd.pdf`
- Expected outputs:
  - halo metadata model
  - pack/unpack kernel responsibilities
  - per-stage and per-face exchange rules
  - scheduler-hook requirements for later overlap

## End-to-End Example Priority

The first formula-backed end-to-end examples should be:

- vacuum propagation
- dielectric slab or waveguide
- PML absorption
- mode source into a waveguide
- field and flux monitoring
- near-to-far on a small supported case
