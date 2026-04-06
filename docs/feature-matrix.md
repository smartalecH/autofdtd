# Feature Matrix

The package ships a planning matrix that mirrors the Phase 1 checklist in code. This
keeps the scaffold anchored to the local Tidy3D EM surface before the solver
implementation lands.

## CLI

```bash
autofdtd-feature-matrix --format table
autofdtd-feature-matrix --format json
```

## Included Categories

- core containers
- geometry
- grid and subpixel controls
- materials
- boundaries
- sources
- monitors
- postprocessing

The code-backed matrix lives in `autofdtd.planning.feature_matrix()`.
