# Feature Matrix

Phase 1 feature coverage is tracked in code at `autofdtd.planning._FEATURE_MATRIX` with
100 entries organized across 8 families. Every feature is tagged as `Implement`,
`Defer`, or `Reject Clearly`.

## Current Coverage

| Family | Implement | Defer | Reject | Total |
|--------|-----------|-------|--------|-------|
| Core | 4 | 2 | 1 | 7 |
| Geometry | 7 | 2 | 0 | 9 |
| Grid | 10 | 6 | 0 | 16 |
| Materials | 9 | 5 | 8 | 22 |
| Boundaries | 10 | 1 | 0 | 11 |
| Sources | 14 | 0 | 1 | 15 |
| Monitors | 17 | 0 | 0 | 17 |
| Postprocessing | 2 | 1 | 0 | 3 |
| **Total** | **73** | **17** | **10** | **100** |

## Key Implemented Features

### Core Containers
- `Simulation`, `Scene`, `Structure` — frozen, tagged, JSON-ready
- `normalize_index` — validation-backed source normalization
- `structure_precedence()` — ordered overlap resolution

### Geometry
- `Box`, `Sphere`, `Cylinder` — core primitives with bounds and containment
- `PolySlab` — simple-polygon extrusions with explicit rejection of advanced features
- `GeometryGroup`, `Transformed`, `GeometryArray` — composite wrappers

### Materials
- `Medium`, `PECMedium`, `PMCMedium` — baseline isotropic materials
- `PoleResidue`, `Sellmeier`, `Lorentz`, `Drude`, `Debye` — dispersive media
- `AnisotropicMedium` — diagonal tensor material with per-axis branches

### Boundaries
- `Periodic`, `PECBoundary`, `PMCBoundary` — non-absorbing boundaries
- `BlochBoundary` — phase-shifted periodic boundaries
- `PML`, `StablePML`, `Absorber` — absorbing boundaries
- `ABCBoundary` — first-order absorbing boundary

### Sources
- `GaussianPulse`, `ContinuousWave` — time profiles
- `UniformCurrentSource`, `PointDipole` — current sources
- `PlaneWave`, `GaussianBeam`, `AstigmaticGaussianBeam` — optical sources
- `TFSF` — total-field/scattered-field source
- `ModeSource` — mode-guided source

### Monitors
- `FieldMonitor`, `FieldTimeMonitor` — field recording
- `FluxMonitor`, `FluxTimeMonitor` — flux integration
- `ModeMonitor`, `ModeSolverMonitor` — mode data extraction
- `MediumMonitor`, `PermittivityMonitor` — material property recording
- `GaussianOverlapMonitor`, `AstigmaticGaussianOverlapMonitor` — overlap monitor
- `FieldProjectionAngleMonitor`, `FieldProjectionCartesianMonitor`,
  `FieldProjectionKSpaceMonitor`, `DiffractionMonitor`, `DirectivityMonitor` —
  projection monitors
- `SurfaceFieldMonitor`, `SurfaceFieldTimeMonitor` — surface field recording

### Postprocessing
- DFT accumulation for frequency-domain monitors
- Flux integration (Poynting vector)
- Near-to-far transform (simplified Rayleigh)
- Mode data extraction

## Deferred Features

The following have explicit deferral rationale in `planning.py`:

| Feature | Reason |
|---------|--------|
| `QuasiUniformGrid` | Alternative grid path; AutoGrid sufficient for Phase 1 |
| `ContourPathAveraging` | Advanced subpixel; PolarizedAveraging covers priority cases |
| `VolumetricAveraging` | Advanced subpixel; PolarizedAveraging covers priority cases |
| `PECConformal` | Metal interface refinement bucket |
| `SurfaceImpedance` | Advanced boundary/material refinement bucket |
| `TriangleMesh` | Too much geometry complexity for Phase 1 |
| `ClipOperation` | Narrow boolean subset implemented; full meshing deferred |
| `FullyAnisotropicMedium` | Dense tensor runtime; diagonal AnisotropicMedium sufficient |
| `Medium2D` | Offline conversion to volumetric only; direct simulation deferred |
| `LossyMetalMedium` | Manual volumetric fallback guidance provided |
| `PerturbationMedium` | Parsed but not executed in Phase 1 |
| `ModeABCBoundary` | Mode-solver dependency; first-order ABC implemented |

## Rejected Features

The following raise `AutoFDTDNotImplementedError` with clear messaging:

| Feature | Reason |
|---------|--------|
| `lumped_elements` | Out of Phase 1 scope |
| `CustomMedium`, `CustomAnisotropicMedium`, `CustomPoleResidue` | Sampled custom media out of scope |
| `CustomSellmeier`, `CustomLorentz`, `CustomDrude`, `CustomDebye` | Sampled custom dispersive out of scope |
| `CustomMedia` (catch-all) | Out of Phase 1 scope |
| `MicrowaveTerminalSource` | Out of Phase 1 scope |
| `InternalAbsorber` | Out of Phase 1 scope |

## CLI

```bash
autofdtd-feature-matrix --format table
autofdtd-feature-matrix --format json
```

The programmatic interface is `autofdtd.planning.feature_matrix()`.
