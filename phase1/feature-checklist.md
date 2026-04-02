# Phase 1 Tidy3D Feature Checklist

This checklist is part of prep, not a loop task. It exists so the loop can execute against a concrete feature ledger instead of rediscovering the Tidy3D EM surface.

Statuses:
- `Implement`
- `Defer`
- `Reject Clearly`

## Core Containers

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `Simulation` shell (`center`, `size`, `run_time`, `medium`, `structures`, `sources`, `monitors`, `boundary_spec`, `grid_spec`, `symmetry`, `shutoff`, `courant`, `subpixel`) | Implement | Core API and IR entrypoint |
| `Scene` | Implement | Needed for structure grouping and inspectability |
| `Structure` | Implement | Ordered structure semantics required |
| Structure priority / ordered overlap | Implement | Tidy3D-like precedence matters |
| `normalize_index` | Implement | Validation + normalization behavior |
| `relax_courant` | Defer | Keep mapped in feature matrix |
| `low_freq_smoothing` | Defer | Postprocessing-adjacent refinement |
| `lumped_elements` | Reject Clearly | Out of Phase 1 core scope |

## Geometry

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `Box` | Implement | Core primitive |
| `Sphere` | Implement | Core primitive |
| `Cylinder` | Implement | Core primitive |
| `PolySlab` | Implement | High-priority photonics geometry |
| `GeometryGroup` | Implement | Needed for composability |
| `Transformed` | Implement | Minimum transform wrapper |
| `ClipOperation` | Defer | Map explicitly, narrow subset only if needed |
| `GeometryArray` | Implement | Repeated placement / array semantics |
| `TriangleMesh` | Defer | Too much geometry complexity for Phase 1 |
| `ComplexPolySlabBase` | Defer | Advanced geometry bucket |

## Grid And Subpixel

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `UniformGrid` | Implement | Core path |
| `CustomGrid` | Implement | Needed for explicit grid control |
| `CustomGridBoundaries` | Implement | With validation |
| `GridSpec` | Implement | Main grid container |
| `AutoGrid` | Implement | Simplified but real behavior |
| `QuasiUniformGrid` | Defer | Keep mapped |
| `GridRefinement` | Implement | Minimum refinement metadata |
| `LayerRefinementSpec` | Implement | Needed for grid planning surface |
| `SubpixelSpec` | Implement | Policy surface |
| `Staircasing` | Implement | Baseline policy |
| `PolarizedAveraging` | Implement | High-value smoothing policy; requires anisotropic materialization support |
| `ContourPathAveraging` | Defer | Keep mapped |
| `VolumetricAveraging` | Defer | Keep mapped |
| `PECConformal` / `SurfaceImpedance` | Defer | Boundary/material refinement bucket |

## Materials

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `Medium` | Implement | Core isotropic material |
| `PECMedium` / `PMCMedium` | Implement | Boundary-aligned materials |
| `PoleResidue` | Implement | Primary dispersive bucket |
| `Sellmeier` | Implement | Important optical material family |
| `Lorentz` | Implement | Dispersive family |
| `Drude` | Implement | Dispersive family |
| `Debye` | Implement | Dispersive family |
| `AnisotropicMedium` | Implement | Needed explicitly and by second-order subpixel smoothing |
| `FullyAnisotropicMedium` | Defer | Keep mapped |
| `Medium2D` | Defer | Validation + future support |
| `LossyMetalMedium` | Defer | Explicit defer bucket |
| `PerturbationMedium` | Defer | Explicit defer bucket |
| `Custom*` media families | Reject Clearly | Out of Phase 1 |

## Boundaries

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `Boundary` / `BoundarySpec` | Implement | Core surface |
| `Periodic` | Implement | Core |
| `PECBoundary` / `PMCBoundary` | Implement | Core |
| `BlochBoundary` | Implement | Important for periodic phase behavior |
| `PML` | Implement | Must-have |
| `StablePML` | Implement | May land as a supported subset first |
| `Absorber` | Implement | Important practical fallback |
| `ABCBoundary` | Defer | Keep mapped |
| `ModeABCBoundary` | Defer | Keep mapped |
| `InternalAbsorber` | Reject Clearly | Out of Phase 1 |

## Sources

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `GaussianPulse` | Implement | Core time profile |
| `ContinuousWave` | Implement | Core time profile |
| `CustomSourceTime` | Implement | May land as a supported subset first |
| `BroadbandPulse` | Implement | May land as a supported subset first |
| `UniformCurrentSource` | Implement | Core source |
| `PointDipole` | Implement | Core source |
| `ModeSource` | Implement | High-priority photonics source |
| `PlaneWave` | Implement | Core optical source |
| `GaussianBeam` | Implement | Core optical source |
| `AstigmaticGaussianBeam` | Implement | May land as a supported subset first |
| `TFSF` | Implement | Important enough to map and support |
| `CustomFieldSource` | Implement | May land as a supported subset first |
| `CustomCurrentSource` | Implement | May land as a supported subset first |
| `MicrowaveTerminalSource` | Reject Clearly | Out of Phase 1 |

## Monitors And Data

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| `SimulationData` named access model | Implement | Common result contract |
| `FieldMonitor` | Implement | Core |
| `FieldTimeMonitor` | Implement | Core |
| `AuxFieldTimeMonitor` | Implement | Included with field-monitor family task |
| `FluxMonitor` | Implement | Core |
| `FluxTimeMonitor` | Implement | Useful extension |
| `ModeMonitor` | Implement | Important photonics monitor |
| `ModeSolverMonitor` | Implement | Needed for mode workflow |
| `MediumMonitor` | Implement | Useful for validation |
| `PermittivityMonitor` | Implement | Useful for validation |
| `FieldProjectionAngleMonitor` | Implement | May land as a supported subset first |
| `FieldProjectionCartesianMonitor` | Implement | May land as a supported subset first |
| `FieldProjectionKSpaceMonitor` | Implement | May land as a supported subset first |
| `DiffractionMonitor` | Implement | May land as a supported subset first |
| `DirectivityMonitor` | Implement | Via projection-monitor task or supported subset |
| `GaussianOverlapMonitor` variants | Implement | May land as a supported subset first |
| `SurfaceFieldMonitor` / `SurfaceFieldTimeMonitor` | Implement | May land as a supported subset first |

## Postprocessing

| Feature | Phase 1 | Notes |
| --- | --- | --- |
| DFT accumulation for monitors | Implement | Foundation for many outputs |
| Flux integration | Implement | Core postprocessing |
| Mode data extraction | Implement | Needed with `ModeMonitor` |
| Near-to-far transform | Implement | Explicit Phase 1 goal |
| Diffraction postprocessing | Defer | Only if monitor support lands |
| Directivity postprocessing | Defer | Depends on projection stack |

## Runtime / Kernel Support Expectations

Each implemented feature family should map to all relevant layers:

| Feature Family | API / IR | Compiler / Planning | Kernel / Runtime |
| --- | --- | --- | --- |
| Geometry | Yes | Yes | Indirect via materialization |
| Materials | Yes | Yes | Yes |
| Boundaries | Yes | Yes | Yes |
| Sources | Yes | Yes | Yes |
| Monitors | Yes | Yes | Yes |
| Postprocessing | Yes | Yes | Yes |
| API -> IR lowering | Yes | Yes | N/A |
| Chunk runtime | N/A | Yes | Yes |

## Kernel Families Expected In Phase 1

- Yee-grid field storage and indexing
- Vacuum `E/H` timestep kernels
- Isotropic constitutive-update kernels
- Dispersive material update kernels
- Anisotropic material update kernels
- Source injection kernels
- Periodic / Bloch / symmetry boundary handling kernels or runtime transforms
- PML kernels
- Absorber kernels
- Halo pack / unpack / exchange kernels
- Field sampling kernels
- DFT accumulation kernels
- Flux integration kernels
- Mode-monitor accumulation kernels
- Near-to-far postprocessing kernels
