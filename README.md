# AutoFDTD

`autofdtd` is a bootstrap repository for an autonomous effort to build a high-performance,
GPU-first, multi-GPU FDTD engine with:

- Meep-level core feature coverage
- Tidy3D-compatible APIs for the overlapping FDTD subset
- high-performance geometry initialization and mode solving
- hybrid time-/frequency-domain adjoint support
- reproducible correctness and performance benchmarking
- maintainable code generation instead of hand-maintained kernel explosions

The design corpus in this repository is the canonical source material. The bootstrap layer
turns that corpus into an executable backlog similar in spirit to `autoresearch` and
`autokernel`, but aimed at a full solver stack: core Yee updates, adjoint methods,
geometry/materialization, distributed chunking, mode solving, near-to-far fields,
API compatibility, correctness tests, and performance benchmarking.

## Installation

```bash
python -m pip install -e .
```

Installed entry points:

```bash
autofdtd-prepare
autofdtd-orchestrate status
autofdtd-orchestrate next
```

Direct checkout usage also works:

```bash
python3 prepare.py
python3 orchestrate.py status
python3 orchestrate.py next
```

## Bootstrap Files

- `program.md`: autonomous operating instructions for the FDTD engineering effort
- `prepare.py`: checkout-friendly wrapper around the packaged bootstrap CLI
- `orchestrate.py`: checkout-friendly wrapper around the packaged backlog CLI
- `pyproject.toml`: package metadata for `pip install -e .`
- `src/autofdtd/`: installable package and CLI implementation
- `work_items.json`: machine-readable backlog of the remaining schema, protocol, benchmark, and scaffolding tasks
- `references.json`: logical external reference catalog using repo names and paper collections instead of machine-specific paths
- `results.tsv`: untracked run log created by the bootstrap
- `workspace/`: untracked generated state, indexes, reports, and run artifacts

## Quick Start

```bash
cd autofdtd
python -m pip install -e .
autofdtd-prepare
autofdtd-orchestrate status
autofdtd-orchestrate next
```

## Running The Agent

Start Codex, Claude Code, or a similar coding agent in this repo and give it this prompt:

```text
Read README.md and program.md, do the setup, then start on the first queued work item.
```

For an autonomous run:

1. Create a branch such as `autofdtd/<tag>`.
2. Read `program.md`.
3. Read the design corpus in the order below.
4. Run `autofdtd-prepare`.
5. Claim and execute work items via `autofdtd-orchestrate`.

## Reading Order

1. `framework-decision.md`
2. `meep-gpu-core-spec.md`
3. `geometry-gpu-core-spec.md`
4. `geometry-and-domain-initialization.md`
5. `fdtdx-and-systolic-notes.md`
6. `vector-mode-solver-spec.md`
7. `performance-benchmarking-spec.md`
8. `gaps-and-investigation-areas.md`
9. `tidy3d-compatibility-and-benchmarks.md`
10. `mathematical-kernel-spec.md`
11. `adjoint-and-discrepancy-notes.md`

## Executive Summary

- Primary implementation substrate: `Warp`
- Primary reference implementations: `meep`, `Khronos.jl`, `VectorModeSolver.jl`, `fdtdx`, the `papers` collection, and the dissertation in `phd_thesis`
- Multi-GPU strategy: explicit chunked domain decomposition with stage-specific halo exchange
- Core maintainability strategy: freeze compact machine-readable IRs and generate specialized kernels from them instead of hand-maintaining the Cartesian product of feature variants
- Adjoint strategy: preserve Meep's high-level forward-monitor/adjoint-source pipeline, but replace fragile low-level pieces with exact Yee-aware analytic kernels where possible
- Geometry strategy: make geometry and materialization a first-class multi-GPU subsystem rather than a frontend afterthought

## External References

The design docs refer to external material by logical names such as `meep`, `papers`,
`phd_thesis`, `tidy3d-notebooks`, `metalens`, `fdtdx`, `warp`, and `VectorModeSolver.jl`.
Those are reference targets, not hard-coded local paths. Optional local checkout hints live
in `references.json`, and generated runtime state stores only those logical IDs and hints.

## Design Corpus

- `framework-decision.md`: decision record for Warp vs Taichi vs PyKokkos vs Khronos.jl
- `meep-gpu-core-spec.md`: the main implementation and code-generation spec intended to be LLM-consumable
- `geometry-gpu-core-spec.md`: standalone geometry-engine spec covering IR, kernel families, chunking, scaling, and differentiation
- `geometry-and-domain-initialization.md`: geometry frontend, GPU rasterization/materialization, subpixel treatment, and scaling strategy for many-shape scenes
- `fdtdx-and-systolic-notes.md`: additional core-solver features and scheduling ideas suggested by `fdtdx` and the local systolic FDTD paper
- `vector-mode-solver-spec.md`: full-vector eigenmode-solver requirements for mode sources, mode monitors, normalization, and adjoint coupling
- `performance-benchmarking-spec.md`: benchmark taxonomy, metrics, realistic workloads, hardware-aware throughput model, and regression-harness guidance
- `gaps-and-investigation-areas.md`: explicit list of remaining blockers between the current design notes and a truly codegen-complete package spec
- `tidy3d-compatibility-and-benchmarks.md`: API-compatibility and notebook benchmark plan for Tidy3D
- `mathematical-kernel-spec.md`: forward/adjoint math spec for the actual kernel updates and recombination operators
- `adjoint-and-discrepancy-notes.md`: hybrid adjoint requirements and theory/code mismatches that should influence the implementation

## Bootstrap Goal

The immediate mission is not to hand-write the final solver in this repository. It is to
freeze the missing machine-readable contracts and package scaffolding that make future
LLM-driven implementation reliable:

- backend-neutral solver, geometry, and operator schemas
- exact Meep and Tidy3D compatibility manifests
- explicit multi-GPU chunk/halo protocol contracts
- benchmark manifests, golden outputs, and tolerance policies
- a clean codegen boundary between generated kernels and handwritten runtime code

`work_items.json` is the current source of truth for that execution backlog.
