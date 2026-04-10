# Shared Context: autofdtd Bug Fix Campaign

This file provides all the background knowledge needed to fix bugs in autofdtd.
**Read this entire file before starting any task.**

---

## 1. Project Overview

autofdtd is a Python FDTD (Finite-Difference Time-Domain) electromagnetic simulator.
It is buggy. We are systematically fixing all ~185 known bugs by comparing against three
proven reference implementations:

- **Khronos.jl** — Meta's production Julia FDTD solver at `../Khronos.jl/`
- **meep** — MIT's C++ FDTD solver at `../meep/`
- **tidy3d** — Flexcompute's Python FDTD solver at `../tidy3d/`
- **VectorModesolver.jl** — Reference Julia mode solver at `../VectorModesolver.jl/`

All reference paths are relative to the autofdtd root directory (the parent of `src/`).
The layout is: `<parent>/autofdtd/`, `<parent>/Khronos.jl/`, `<parent>/meep/`, etc.

---

## 2. Directory Layout

### autofdtd source tree (`src/autofdtd/`)

```
api/simulation.py          — User-facing Simulation API
boundaries/models.py       — Boundary condition data models (PML, Periodic, Bloch, etc.)
compiler/
  boundaries.py            — Compile boundary specs to runtime coefficients
  discretization.py        — Geometry rasterization and subpixel averaging
  materials.py             — Compile material models to ADE coefficients
  monitors.py              — Compile monitors + DFT state machines
  pipeline.py              — Main compile_simulation() entry point
  runtime.py               — CFL computation, runtime config
  sources.py               — Compile sources to injection data
core/
  containers.py            — Simulation container (holds structures, sources, monitors)
  models.py                — Base Pydantic models
  validation.py            — Simulation validation logic
geometry/
  composite.py             — Boolean geometry operations (union, intersection, difference)
  euler_bend.py            — Euler bend geometry
  polyslab.py              — Polygon extrusion geometry
  primitives.py            — Box, Sphere, Cylinder
grid/
  specs.py                 — Grid specifications (AutoGrid, UniformGrid, CustomGrid)
  subpixel.py              — Subpixel averaging (unused in practice)
ir/models.py               — Intermediate representation data models
kernels/
  backend.py               — Backend selection (NumPy vs Warp)
  boundaries.py            — PML, ghost cell, periodic boundary kernels
  materials.py             — NumPy ADE dispersive update kernels
  monitors.py              — Monitor field extraction, DFT accumulation, flux
  near2far.py              — Near-to-far-field transformation kernels
  sources.py               — Source injection kernels (dipole, plane wave, beam, TFSF)
  steps.py                 — Core FDTD stepping kernels (E/H update, Warp kernels)
materials/
  advanced.py              — Sellmeier, PoleResidue models
  anisotropic.py           — AnisotropicMedium, FullyAnisotropicMedium
  dispersive.py            — Lorentz, Drude, Debye models + PoleResidue
  isotropic.py             — Simple isotropic medium
modes/
  epsilon.py               — Permittivity sampling for mode solver
  ir.py                    — Mode solver IR models
  models.py                — ModeSpec, ModeSolution data models
  solver.py                — Mode solver (eigenvalue problem)
monitors/models.py         — Monitor data models (Field, Flux, Mode, Diffraction, N2F)
runtime/
  boundaries.py            — Runtime boundary application (halo exchange, ghost cells)
  chunk.py                 — Domain decomposition (ChunkSpec, halo descriptors)
  controls.py              — RuntimeController (convergence, shutoff) — currently UNUSED
  execution.py             — Main simulation loop (run_compiled_simulation)
  monitors.py              — Runtime monitor recording
  near2far.py              — Runtime N2F surface setup
  sources.py               — Runtime source injection
sources/
  beam.py                  — GaussianBeam, AstigmaticGaussianBeam
  current.py               — UniformCurrentSource
  mode.py                  — ModeSource
  plane.py                 — PlaneWave
  tfsf.py                  — TFSF (Total-Field/Scattered-Field)
  time.py                  — GaussianPulse, ContinuousWave, BroadbandPulse
```

### Khronos.jl reference tree (`../Khronos.jl/src/`)

```
Khronos.jl                 — Module definition
Simulation.jl              — Main simulation loop (step!)
Kernels/
  Kernels.jl               — Kernel dispatch (step_kernel!)
  ReferenceKernels.jl      — CPU reference kernels (E/H update with B/D formulation)
  CUDAKernels.jl           — GPU CUDA kernels
  Dispersive.jl            — ADE dispersive update (per-component P)
  Helpers.jl               — Kernel helpers
Fields.jl                  — Field data structures (SoA: Ex, Ey, Ez separate arrays)
Boundaries.jl              — PML (CPML with memory), periodic, Bloch
Chunking.jl                — Domain decomposition with halo exchange
Distributed.jl             — Multi-GPU MPI communication
Geometry.jl                — Geometry rasterization with Yee staggering
Mode.jl                    — Mode solver
Susceptibility.jl          — Material susceptibility models (Lorentz, Drude)
Sources/
  Sources.jl               — Source dispatch and update
  SpatialSources.jl        — Plane wave, Gaussian beam, mode source, TFSF
  TimeSources.jl           — Gaussian pulse, CW, broadband
Monitors/
  Monitors.jl              — Monitor base and DFT accumulation
  FluxMonitor.jl           — Flux = Re(DFT_E x conj(DFT_H))
  ModeMonitor.jl           — Mode overlap integral
  DiffractionMonitor.jl    — Diffraction order computation
  Near2Far.jl              — Near-to-far-field transformation
Memory.jl                  — GPU memory management
Visualization.jl           — Plotting utilities
```

### VectorModesolver.jl (`../VectorModesolver.jl/src/`)

```
VectorModesolver.jl        — Module definition
Modesolver.jl              — Full-vectorial Fallahkhair formulation (THE key reference)
Mode.jl                    — Mode data structure
Visualization.jl           — Plotting
```

---

## 3. FDTD Physics Reference

### Correct Yee Algorithm (Leapfrog Time-Stepping)

The correct FDTD loop order (as in Khronos.jl `Kernels.jl:64-88`) is:

```
for each timestep:
    1. Apply H sources at time t
    2. Update H fields:  B_new = B_old - dt * curl(E)
                          H = mu_inv * (B - sum(P_magnetic))
    3. Apply H boundaries (PML, periodic, halo exchange for H)
    4. Record H monitors at time t

    5. Apply E sources at time t + dt/2
    6. Update E fields:  D_new = D_old + dt * curl(H)
                          E = eps_inv * (D - sum(P_electric))
    7. Apply E boundaries (PML, periodic, halo exchange for E)
    8. Record E monitors at time t + dt/2

    9. Update dispersive polarizations P
    10. Advance timestep
```

### Key Physics Formulas

**Maxwell curl equations (SI):**
```
dB/dt = -curl(E)          =>  B_new = B_old - dt * curl(E)
dD/dt = +curl(H)          =>  D_new = D_old + dt * curl(H)
```

**Constitutive relations:**
```
H = (1/mu) * B            (non-dispersive)
E = (1/eps) * D            (non-dispersive)
E = eps_inv * (D - P)      (dispersive, P = sum of polarization currents)
```

**H-field update sign:** `H_new = H_old + (dt/mu) * curl(E)` when curl uses standard
Yee stencil with the `-` absorbed into the curl definition. The key point:
the coefficient multiplying curl(E) in the H update must be `+dt/mu`, NOT `-dt/mu`.

**Courant condition (CFL):**
```
dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
```

**Poynting vector (power flow):**
```
S = (1/2) * Re(E x conj(H))
```

**DFT accumulation (frequency domain):**
```
DFT[f] += field(t) * exp(-i*2*pi*f*t) * dt
```
Note: multiply by `dt`, do NOT divide by count at the end.

**Mode overlap integral:**
```
a_forward  = (1/4P) * integral( E_sim x H_mode* + E_mode* x H_sim ) . n_hat dA
a_backward = (1/4P) * integral( E_sim x H_mode* - E_mode* x H_sim ) . n_hat dA
```

### Yee Grid Staggering

On the Yee grid, field components live at different spatial positions:
```
Ex at (i,   j+½, k+½)     Hx at (i+½, j,   k  )
Ey at (i+½, j,   k+½)     Hy at (i,   j+½, k  )
Ez at (i+½, j+½, k  )     Hz at (i,   j,   k+½)
```

For mode solver epsilon sampling, sample at the field component location:
- eps_xx at Ex position: (i, j+½, k+½)
- eps_yy at Ey position: (i+½, j, k+½)
- eps_zz at Ez position: (i+½, j+½, k)

### Natural Units vs SI

Khronos.jl uses natural units where eps_0 = mu_0 = c = 1.
meep also uses natural units.
autofdtd should use SI units consistently (c = 2.998e8 m/s, eps_0 = 8.854e-12 F/m, mu_0 = 4*pi*1e-7 H/m).

---

## 4. Physical Constants

```python
import math
C0 = 2.998e8          # speed of light (m/s)
EPSILON_0 = 8.854e-12  # vacuum permittivity (F/m)
MU_0 = 4 * math.pi * 1e-7  # vacuum permeability (H/m)
Z0 = 376.73           # impedance of free space (Ohms)
```

---

## 5. Key Reference File Locations

When a task says "compare against Khronos" or "compare against meep", look here:

| Topic | Khronos.jl | meep | tidy3d |
|-------|-----------|------|--------|
| Stepping loop | `Kernels/Kernels.jl:64-88` | `fields.cpp` + `step.cpp` | N/A (server-side) |
| E update | `Kernels/ReferenceKernels.jl:330-370` | `step_generic.cpp` | N/A |
| H update | `Kernels/ReferenceKernels.jl:290-328` | `step_generic.cpp` | N/A |
| PML (CPML) | `Boundaries.jl:100-200` | `boundaries.cpp` | N/A |
| Dispersive ADE | `Kernels/Dispersive.jl` | `susceptibility.cpp` | N/A |
| DFT | `Monitors/Monitors.jl:200-400` | `dft.cpp` | N/A |
| Flux | `Monitors/FluxMonitor.jl:100-160` | `energy_and_flux.cpp` | N/A |
| Mode overlap | `Monitors/ModeMonitor.jl:300-500` | N/A | N/A |
| Diffraction | `Monitors/DiffractionMonitor.jl:100-160` | N/A | N/A |
| Near2Far | `Monitors/Near2Far.jl` | `near2far.cpp` | N/A |
| Sources | `Sources/SpatialSources.jl` | `sources.cpp` | `components/source.py` |
| Time profiles | `Sources/TimeSources.jl` | `sources.cpp` | `components/source.py` |
| Geometry | `Geometry.jl:900-1000` | `anisotropic_averaging.cpp` | `components/grid/` |
| Mode solver | `Mode.jl` | N/A | `components/mode/solver.py` |
| Full-vectorial | N/A | N/A | See VectorModesolver.jl `Modesolver.jl` |

---

## 6. Testing Conventions

- Tests live in `tests/` (relative to autofdtd root)
- Run tests with: `python -m pytest tests/ -x -v`
- For a specific test: `python -m pytest tests/test_file.py::test_name -v`
- **Never loosen test tolerances** to make tests pass. Fix the underlying logic.
- When creating new tests, follow existing patterns in the tests/ directory.

---

## 7. Code Style

- Python 3.9+, type hints encouraged
- Pydantic models for data classes (frozen=True)
- NumPy for CPU kernels, Warp for GPU kernels
- Constants: `EPSILON_0`, `MU_0`, `C0` from materials or constants module
- Use `2.998e8` consistently for speed of light (not `3e8`)

---

## 8. Common Patterns

### Reading a Khronos.jl reference
Julia code uses 1-based indexing. When translating to Python, subtract 1 from indices.
Julia uses `im` for imaginary unit (Python: `1j`).
Julia arrays are column-major; NumPy arrays are row-major by default.

### Warp GPU kernels
Warp kernels use `@wp.kernel` decorator and `wp.float32` types.
They cannot use Python objects — only Warp types.
For complex numbers in Warp, you must use two float arrays (real + imag) since
Warp does not natively support complex types.

### The ADE (Auxiliary Differential Equation) method for dispersive materials
Each Lorentz/Drude pole has auxiliary polarization state P that evolves as:
```
P_new = decay * P_old + drive * E
```
where `decay = exp(-a*dt)` and `drive = -eps0*c*(1-exp(-a*dt))/a` for pole `a`.
The E field is then corrected: `E -= eps_inv * P_current`.
**P must be stored per-component** (Px, Py, Pz), not as a scalar.

---

## 9. Bug Report Locations

The full bug reports with all details, line numbers, and severities are located
in this phase2.5 directory:
- Phase 1+2: `./bug_report_phase1_2.md` (77 bugs)
- Phase 3: `./bug_report_phase3.md` (108 bugs)

When working on a task, read the relevant section of these reports for exact line numbers
and detailed descriptions.
