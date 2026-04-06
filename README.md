# AutoFDTD

GPU-first finite-difference time-domain solver scaffolding for Phase 1 Maxwell work.

The repository now provides a pip-installable `src/` layout package, subsystem-oriented
module namespaces, development tooling, test entrypoints, a feature-matrix CLI, and
documentation scaffolding aligned with the local Tidy3D EM feature checklist.

## Package Layout

The package is intentionally split along Phase 1 subsystem boundaries:

- `autofdtd.api`: public object-model entrypoints and compatibility planning surfaces
- `autofdtd.core`: shared model and tagging foundations
- `autofdtd.geometry`, `materials`, `boundaries`, `sources`, `monitors`, `grid`, `modes`
- `autofdtd.ir`: tagged execution-IR namespace
- `autofdtd.compiler`: scene-lowering and planning namespace
- `autofdtd.runtime`: chunk orchestration and execution namespace
- `autofdtd.kernels`: Warp kernel conventions and backend staging namespace
- `autofdtd.diagnostics`: logging, metrics, and validation namespace

The package does not yet claim solver completeness. It now includes the first tagged,
immutable container models for `Simulation`, `Scene`, and `Structure`, with ordered
structure precedence helpers and JSON-ready serialization for later IR lowering. The
`autofdtd.ir` namespace now also exposes a versioned execution package contract that can
snapshot a simulation into `simulation.json` plus a hashed `manifest.json` for local or
remote execution handoff. Shared helpers in `autofdtd.core` now provide the common frozen
model contract, explicit tag access, copy-or-update semantics, and stable serialization
used by both the public API and IR layers. Geometry support now includes `PolySlab`
polygon extrusions for representative planar photonics layouts, with explicit Phase 1
reject behavior for tapering and other advanced polygon operations. Composite wrappers
now also cover grouped, transformed, clipped, and repeated-placement layouts, and
ordered structure precedence is surfaced explicitly for overlap debugging and later
scene compilation.

## Quick Start

```bash
python -m pip install -e .[dev]
pytest
autofdtd-feature-matrix --format table
```

## Development

Common entrypoints are configured in `pyproject.toml`:

- `pytest` for tests
- `ruff check .` for linting
- `mypy src` for static typing
- `mkdocs build` for docs

Documentation sources live under `docs/`, and the MkDocs configuration is in
`mkdocs.yml`.
