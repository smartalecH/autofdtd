# AutoFDTD Bootstrap Program

This repository is the bootstrap layer for an autonomous FDTD engineering effort.
The goal is to turn the research corpus in this repository into a production-quality,
GPU-first, multi-GPU-capable FDTD package with:

- Meep feature parity for the core solver
- Tidy3D-compatible user-facing APIs for the overlapping FDTD subset
- exacting forward and hybrid time-/frequency-domain adjoint support
- high-performance geometry initialization and materialization
- reproducible accuracy and performance benchmarks
- maintainable code generation instead of hand-maintained kernel combinatorics

The source documents in this repository are the canonical design corpus. Do not discard
or casually rewrite them. Extend them only when new implementation facts, formulas,
or measured results justify an update.

## Repo Model

This bootstrap follows the same pattern as `autoresearch` and `autokernel`:

- `program.md`: the operating instructions for the agent
- `prepare.py`: checkout-friendly wrapper for the packaged bootstrap CLI
- `orchestrate.py`: checkout-friendly wrapper for the packaged work-queue CLI
- `pyproject.toml` and `src/autofdtd/`: installable package layer for `pip install -e .`
- `work_items.json`: machine-readable backlog for turning the design corpus into codegen-complete specs and implementation scaffolding
- `references.json`: logical catalog of external repos and paper collections
- `README.md`: human-facing map of the corpus and execution workflow

Unlike `autoresearch`, there is no single mutable source file yet. The near-term mission
is to create the missing schemas, manifests, acceptance criteria, and scaffolding that
allow a future autonomous implementation loop to be robust.

## Success Criteria

The end state of this repo is a package-generation and evaluation framework that can
reliably produce an FDTD engine with all of the following:

- forward Yee FDTD with the core Meep/Tidy3D feature set
- hybrid forward/adjoint pipeline, monitor-to-source maps, and inverse-design support
- geometry primitives, subpixel initialization, and scalable many-shape rasterization
- multi-GPU domain decomposition with explicit halo rules and performance tracking
- mode solvers, near-to-far fields, DFT monitors, and silicon-photonics workloads
- exact benchmark manifests, golden outputs, and hardware-aware performance analysis
- a code generation path that is backend-neutral but optimized first for Warp

## Setup

For a new run:

1. Install the package: `python -m pip install -e .`
2. Pick a tag and create a branch: `git checkout -b autofdtd/<tag>`.
3. Read `README.md`.
4. Read every design document in the reading order given by `README.md`.
5. Read `work_items.json` and `references.json`.
6. Run `autofdtd-prepare` or `python3 prepare.py`.
7. Run `autofdtd-orchestrate status`.
8. Choose the next highest-priority work item with `autofdtd-orchestrate next`.

## Scope Rules

What you should do:

- add machine-readable schemas, manifests, acceptance criteria, and benchmark specs
- add implementation scaffolding, test scaffolding, benchmark harnesses, and codegen tooling
- add compatibility adapters and protocol definitions
- update the design corpus when measurements, source references, or implementation decisions require it
- preserve traceability from implementation artifacts back to the source docs

What you should not do:

- replace precise design decisions with vaguer prose
- silently delete research content because it feels redundant
- introduce framework-specific assumptions into backend-neutral schemas unless the schema explicitly allows backend capabilities
- optimize for one benchmark by weakening correctness guarantees

## Priority Order

The backlog is defined in `work_items.json`, but the broad execution order is:

1. Freeze machine-readable IRs and protocol schemas.
2. Freeze benchmark and correctness manifests.
3. Freeze API compatibility manifests for Meep and Tidy3D.
4. Build implementation scaffolding around the frozen schemas.
5. Add performance-critical kernels and distributed runtime pieces.
6. Expand feature parity to richer materials and advanced workflows.

## Operating Loop

Repeat:

1. Inspect queue state: `autofdtd-orchestrate status`
2. Claim the next task: `autofdtd-orchestrate start <item_id>`
3. Make a narrowly scoped change set
4. Record artifacts, measurements, and unresolved risks
5. Mark progress: `autofdtd-orchestrate record <item_id> <status> --note "<summary>"`
6. If a task is blocked by missing information or a prerequisite, mark it blocked and move on

Valid statuses:

- `pending`
- `active`
- `blocked`
- `complete`

## Deliverable Expectations

For any task that claims completion, leave behind:

- the artifact itself, in a stable path
- a short note in `results.tsv`
- enough instructions that another agent can pick up from there without re-deriving context

## Quality Bar

Prefer artifacts that are directly consumable by code generation:

- schemas with explicit enums, field names, defaults, and validation rules
- manifests with stable IDs and acceptance criteria
- benchmark cases with exact input definitions and tolerances
- protocol specs with pack order, ownership, and reduction semantics

If forced to choose, prefer precise machine-readable contracts over additional prose.
