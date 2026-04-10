# Ralph Loop Prompt: autofdtd Bug Fix Campaign

## Context

You are fixing bugs in **autofdtd**, a Python FDTD electromagnetic simulator.
The codebase is at `src/autofdtd/` (relative to the autofdtd root, which is your working directory).

You have four reference implementations to compare against (all sibling directories to autofdtd):
- **Khronos.jl** (Julia FDTD): `../Khronos.jl/src/`
- **meep** (C++ FDTD): `../meep/src/`
- **tidy3d** (Python FDTD): `../tidy3d/tidy3d/`
- **VectorModesolver.jl** (Julia mode solver): `../VectorModesolver.jl/src/`

Two bug reports contain exact line numbers and detailed descriptions (in the phase2.5 directory):
- Phase 1+2: `phase2.5/bug_report_phase1_2.md`
- Phase 3: `phase2.5/bug_report_phase3.md`

## Shared Context

Read `./context.md` before starting work. It contains:
- Complete directory layout of autofdtd and all reference implementations
- Correct FDTD physics formulas (Yee algorithm, Poynting vector, DFT, etc.)
- Physical constants
- Reference file location table
- Testing and code style conventions

## Instructions

**CRITICAL: You must process EXACTLY ONE task per execution. Not two. Not "while you're at it." ONE.**

1. Read `./context.md` first for shared context
2. Read the spec file at `./spec.md`
3. Find the FIRST task numbered with `PENDING:` status
4. Complete that task — and ONLY that task
5. Update spec.md:
   - Change `PENDING:` to `DONE:` for the completed task
6. Exit immediately — do NOT continue to the next task

**WHY ONE TASK?** Each iteration runs in a fresh context window. This prevents context degradation and ensures consistent quality. The orchestration script handles running the next iteration.

## How To Approach Each Task

Each task in spec.md follows this pattern:
```
N. PENDING: [SUBSYSTEM] Description of what to fix. Exact file and line references.
   What the code currently does. What it SHOULD do. Reference implementation location.
```

### Step-by-step process for each task:

1. **Read the task description carefully.** Note the file paths, line numbers, and bug IDs mentioned.

2. **Read the relevant bug report section.** If the task mentions "M1" or "Bug 5", look up that bug in `phase2.5/bug_report_phase1_2.md` or `phase2.5/bug_report_phase3.md` for the full description.

3. **Read the autofdtd file(s) that need fixing.** Understand the current (broken) code.

4. **Read the reference implementation.** The task tells you which reference file to look at. Read that file to understand the CORRECT approach.

5. **Make the fix.** Edit the autofdtd file(s) to match the correct behavior. Keep changes minimal and focused.

6. **Verify your fix.** If there are existing tests, run them:
   ```
   python -m pytest tests/ -x -v --timeout=60
   ```
   If the task involves creating new tests, run them specifically.

7. **Update spec.md** — change `PENDING:` to `DONE:` for the completed task.

## Constraints

- **Never loosen test tolerances** to make tests pass. Fix the underlying logic.
- **Never invent physics.** Always verify against a reference implementation.
- **Keep changes focused.** Only modify what the task asks for. Don't refactor surrounding code.
- **Preserve backward compatibility** of the Python API where possible.
- **Use consistent constants:** `C0 = 2.998e8`, `EPSILON_0 = 8.854e-12`, `MU_0 = 4*pi*1e-7`.
- **For Warp GPU kernels:** Warp does not support complex numbers natively. Use pairs of float32 arrays (real + imag) and implement complex arithmetic manually.
- **For new functions/classes:** Follow existing naming conventions in the file you're modifying.
- **For the mode solver rewrite (tasks 39-47):** These tasks are larger and may require reading VectorModesolver.jl thoroughly. Take your time to get the math right.

## Completion Criteria

A task is complete when:
- The specific bug(s) listed in the task are fixed
- The fix matches the physics described in the reference implementation
- Existing tests still pass (run `pytest` to verify)
- The spec.md file has been updated (PENDING → DONE)

## Special Instructions

### When reading reference implementations

- **Khronos.jl** uses Julia (1-based indexing, `im` for imaginary, column-major arrays).
  When translating, subtract 1 from indices and be aware of array ordering differences.

- **meep** uses C++ with custom macros. Focus on the mathematical formulas in comments
  and the core computation lines, not the MPI/chunk machinery.

- **tidy3d** is the closest in language (Python) but runs server-side. Focus on the
  `components/` and `plugins/` directories for the computational logic.

- **VectorModesolver.jl** is the primary reference for mode solver tasks (39-47).
  Read `Modesolver.jl` very carefully — it implements the Fallahkhair 2008 formulation.

### When fixing Warp GPU kernels

Warp kernels are in `kernels/steps.py` and use `@wp.kernel` decorators.
They operate on `wp.array` types with `wp.float32`.

For complex numbers, use two arrays (real and imaginary parts):
```python
# Instead of: z = a * b  (where a, b are complex)
z_real = a_real * b_real - a_imag * b_imag
z_imag = a_real * b_imag + a_imag * b_real
```

### When fixing DFT/monitor paths

There are FOUR parallel DFT accumulation paths that all need the same fix:
1. `FieldMonitorState.accumulate_dft` (compiler/monitors.py ~line 270)
2. `FluxMonitorState.accumulate_flux_dft` (compiler/monitors.py ~line 617)
3. `ProjectionMonitorState.accumulate_dft` (compiler/monitors.py ~line 1543)
4. `SurfaceFieldMonitorState.accumulate_dft` (compiler/monitors.py ~line 1849)

When a task says "fix all DFT paths", fix ALL FOUR.

### Task dependency awareness

Tasks are ordered by dependency. Earlier tasks (1-8) establish foundations that later
tasks build on. If you're working on task N and find that task N-1 hasn't been done,
note this but still complete task N as best you can with the current code state.

### Mode solver tasks (39-47)

These are the largest and most complex tasks. For the full-vectorial rewrite (task 39):
- You are replacing the scalar Helmholtz solver with the Fallahkhair formulation
- The key reference is VectorModesolver.jl `Modesolver.jl:48-376`
- The operator is 2N x 2N, solving for [Hx; Hy]
- It has 4 blocks: Pxx (Hx-Hx coupling), Pxy (Hx-Hy coupling), Pyx (Hy-Hx coupling), Pyy (Hy-Hy coupling)
- Each block uses a 9-point stencil (4 cardinal + 4 diagonal + 1 center)
- Epsilon is sampled at 4 surrounding cell centers per node
- The eigenvalue is -beta^2

Keep `assemble_te_mode_matrix` as a fallback for scalar-only mode solving,
but make the default path use the new full-vectorial solver.
