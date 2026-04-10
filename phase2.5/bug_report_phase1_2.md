# autofdtd Comprehensive Bug Report (Phase 2)

**References**: meep (MIT), Khronos.jl (Meta), Tidy3D (Flexcompute)  
**Date**: 2026-04-09

---

## Summary

Total bugs found: **77** across all subsystems  
- Phase 1 (kernel physics): 16 bugs  
- Phase 2 (wiring, design, dead code): 61 bugs  

**No frequency-domain monitor result, no material assignment, no boundary condition, and no source injection can be trusted.** The only path that partially works is a vacuum NumPy-backend simulation with a point dipole or uniform current source, periodic boundaries, and time-domain field recording — and even that has the H-field sign error.

---

## PHASE 1 BUGS (from initial analysis) — 16 bugs

### Kernel Physics (Bugs 1–16)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| 1 | **P0** | `kernels/steps.py:352` | H-field update sign error — `ch = -dt/μ` should be `+dt/μ`. ALL components, BOTH backends. |
| 2 | **P0** | `kernels/steps.py:276` | Ey curl sign error in Warp kernel — `(dHz_dx - dHx_dz)` should be `(dHx_dz - dHz_dx)`. NumPy correct. |
| 3 | **P0** | `kernels/steps.py:260` | Missing ε₀/μ₀ in vacuum kernels — divides by relative permittivity only. |
| 4 | **P0** | `kernels/steps.py:1273` | `step_maxwell()` never calls coefficient-aware kernels — decay/drive/PEC ignored. |
| 5 | **P1** | `runtime/execution.py:401` | Source injection before both E+H updates — violates leapfrog stagger. |
| 6 | **P1** | `runtime/execution.py:400-486` | PML not applied in single-chunk path — state allocated but unused. |
| 7 | **P1** | `kernels/boundaries.py:153` | PML is multiplicative damping, not CPML — no memory feedback. |
| 8 | **P1** | `kernels/steps.py` (all) | No B/D intermediate fields — prevents CPML and dispersive coupling. |
| 9 | **P1** | `kernels/steps.py:726` | Dispersive polarization uses scalar `|E|` instead of per-component vector. |
| 10 | **P1** | `runtime/execution.py:215` | `populate_material_arrays` ignores geometry — fills entire grid. |
| 11 | **P2** | `kernels/sources.py:1097` | TFSF incomplete — injection only, no boundary correction. |
| 12 | **P2** | `runtime/execution.py:1585` | `_numpy_magnetic_update_chunk` is a stub (`pass`). |
| 13 | **P2** | `kernels/sources.py:1198` | Source injection GPU→CPU→GPU roundtrip every step. |
| 14 | **P2** | `kernels/steps.py:201` | AoS field layout `(nx,ny,nz,3)` — poor GPU coalescing. |
| 15 | **P3** | `kernels/steps.py:248` | Forward-forward stencil — ½-cell offset vs standard Yee. |
| 16 | **P3** | `grid/subpixel.py` | No anisotropic subpixel averaging. |

---

## PHASE 2 BUGS — Wiring, Design, Dead Code

### A. API → Compiler → Runtime Wiring (17 bugs)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| A1 | **CRIT** | `execution.py:215-231` | `populate_material_arrays` fills entire domain with each structure, ignoring geometry. Last structure wins everywhere. |
| A2 | **CRIT** | `execution.py:203,219` | Runtime reads `permittivity`/`permeability` only, ignoring `electric_decay`/`electric_drive`. **Conductivity silently dropped. All dispersive media treated as vacuum.** |
| A3 | **CRIT** | `execution.py:402-412` | 4 source types compiled but never passed to injection: `custom_current`, `custom_field`, `astigmatic_gaussian_beam`, `tfsf`. |
| A4 | **HIGH** | `execution.py:450` | Shutoff check every step, ignoring `shutoff_check_interval`. |
| A5 | **HIGH** | `execution.py:455` | `float(shutoff)` crashes when `shutoff=None`. |
| A6 | **HIGH** | `execution.py:450-458` | `convergence_policy` completely ignored. |
| A7 | **HIGH** | `execution.py:836-907` | Chunked path missing `_apply_periodic_bloch_wrap` — periodic/Bloch broken in multi-chunk. |
| A8 | **HIGH** | `execution.py:565-601` | Projection monitors compiled but never initialized or recorded. |
| A9 | **MED** | `execution.py:542-562` | `_initialize_source_runtimes` populates a dict never read — dead code. |
| A10 | **MED** | `pipeline.py:713` | Fallback PML uses `num_layers=10` instead of standard default 12. |
| A11 | **MED** | `pipeline.py:603-753` | `subpixel` accepted by API but never consumed — entire subpixel pipeline unreachable. |
| A12 | **MED** | `runtime.py:147` | `normalize_index` compiled but never consumed by runtime. |
| A13 | **MED** | `pipeline.py:731` | Non-uniform grids broken — only `cell_sizes[0]` used for all cells. |
| A14 | **MED** | `pipeline.py:502-556` | `_compile_scene_coefficients` never calls `materialize_grid` — spatial discretization is dead code. |
| A15 | **MED** | `discretization.py:526` | `_resolve_cell_material` always returns `volume_fraction=1.0`. |
| A16 | **MED** | `execution.py:400-486` | `RuntimeController` class with proper stop logic exists but is ignored — inline reimplementation has bugs #A4-A6. |
| A17 | **LOW** | `pipeline.py:700` | `any()` on string tuples always True (works by accident). |

### B. Monitor System (16 bugs)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| M1 | **CRIT** | `compiler/monitors.py:288` | E/H field classification uses `field[1]=='x'` instead of `field[0]=='E'` — 4 of 6 components read from wrong buffer. |
| M2 | **CRIT** | `compiler/monitors.py:270` | DFT accumulation missing `dt` scaling — frequency-domain results dimensionally wrong. |
| M3 | **CRIT** | All DFT paths | E/H leapfrog time stagger completely ignored in DFT — phase error in all freq-domain fields. |
| M4 | **CRIT** | `kernels/monitors.py:181` | Flux from time-domain `E×H` instead of `DFT(E)×conj(DFT(H))` — frequency mixing. |
| M5 | **CRIT** | `kernels/near2far.py:264` | `theta_hat` vector is actually `-r_hat`. |
| M6 | **CRIT** | `kernels/near2far.py:293` | Near2far uses simplified scalar formula instead of dyadic Green's function. |
| M7 | **HIGH** | `runtime/near2far.py:65` | Tangential axis selection always returns `tang1=1, tang2=2` regardless of normal axis. |
| M8 | **HIGH** | `kernels/near2far.py:132` | Source positions use global index instead of local — wrong N2F positions. |
| M9 | **CRIT** | `compiler/monitors.py:1209` | Mode overlap computes `|S|²` instead of `∫(E×H_mode* ± E_mode*×H)·n̂ dA`. |
| M10 | **HIGH** | `kernels/monitors.py:1006` | Diffraction uses grid indices as physical positions in Fourier phase. |
| M11 | **HIGH** | `kernels/monitors.py:1014` | Diffraction uses `|E|` magnitude, discarding phase before FFT. |
| M12 | **HIGH** | `kernels/monitors.py:857` | Cartesian far-field projection is a stub with `phase=1.0`. |
| M13 | **MED** | `runtime/near2far.py:243` | `sin_theta` broadcast axes wrong in `compute_total_radiated_power`. |
| M14 | **MED** | `monitors/models.py` | 5+ monitor types defined but never record data (GaussianOverlap, Directivity, FieldProjection*). |
| M15 | **HIGH** | All monitor paths | No Yee-grid colocation/interpolation — E and H sampled at same index from different locations. |
| M16 | **HIGH** | `kernels/monitors.py:787` | Far-field projects J_s onto `r_hat` (radial) instead of `theta_hat/phi_hat`. |

### C. Source System (12 bugs)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| S1 | **CRIT** | `execution.py:402-412` | 4 source types compiled but never injected (= A3). |
| S2 | **CRIT** | `compiler/sources.py:996` | Gaussian beam radial distance from grid origin, not beam center. |
| S3 | **CRIT** | `kernels/sources.py:580` | CustomFieldSource equivalence principle injects into normal component instead of tangential. |
| S4 | **HIGH** | `kernels/sources.py:758` | Impedance `Z0=1.0` hardcoded — H amplitude wrong by factor 376.73 in SI. |
| S5 | **HIGH** | `kernels/sources.py:768` | GPU plane wave takes `abs()` of tangential components — destroys polarization sign. |
| S6 | **HIGH** | `compiler/sources.py:959` | `waist_distance` stored but never used — no z-dependent beam waist. |
| S7 | **MED** | `sources/mode.py:103` | `angle_theta`/`angle_phi` defined but never read by compiler. |
| S8 | **MED** | Multiple files | Inconsistent speed of light: `3e8` vs `2.998e8`. |
| S9 | **MED** | `compiler/sources.py:871` | k-vector computation produces wrong units (missing frequency factor). |
| S10 | **MED** | `kernels/sources.py:1198` | `inject_sources_stage` double-converts GPU→CPU for each source. |
| S11 | **LOW** | `kernels/sources.py:1101` | TFSF only injection, no boundary correction (= Bug 11). |
| S12 | **LOW** | `compiler/sources.py:676` | Mode field reshape dimension naming swap. |

### D. Geometry & Grid System (17 bugs)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| G1 | **CRIT** | `pipeline.py:603-753` | `compile_simulation()` never calls `materialize_grid()`/`discretize_scene()` — no spatial material map produced. |
| G2 | **HIGH** | `discretization.py:864` | Subpixel averaging hardcodes `volume_fraction=0.5`, losing all geometric info. |
| G3 | **HIGH** | `discretization.py:549` | `_resolve_cell_material` always sets `volume_fraction=1.0`. |
| G4 | **HIGH** | `discretization.py:600` | Pure-Python O(N³·S) rasterization — 118s for metalens. |
| G5 | **HIGH** | `composite.py:276` | EulerBend not in `LeafGeometryModel` union — can't be composed. |
| G6 | **MED** | `euler_bend.py:206` | `contains_point()` implements circular arc, not Euler/clothoid spiral. |
| G7 | **MED** | `euler_bend.py:126` | `bounds` mixes up bending vs width extents; `chord_length` computed but unused. |
| G8 | **MED** | `discretization.py:299` | `_transformed_volume_fraction` uses bounding-box, not actual geometry. |
| G9 | **MED** | `pipeline.py:695-724` | PML thickness not accounted for in grid — PML eats into user domain. |
| G10 | **MED** | `solver.py:442` | Mode solver field reconstruction has x/y index swap. |
| G11 | **MED** | `sources.py:679` | Mode field reshape dimension naming swap. |
| G12 | **LOW** | `solver.py:431` | Debug print statements left in production code. |
| G13 | **LOW** | `solver.py:331` | Dead-code vector BC application multiplies by 0 for Dirichlet. |
| G14 | **LOW** | `pipeline.py:559` | Type hint says `Simulation | Scene` but receives `Structure`. |
| G15 | **LOW** | `discretization.py:283` | `_geometry_group_volume_fraction` overcounts overlaps. |
| G16 | **LOW** | `epsilon.py:240` | Epsilon callback assumes uniform grid spacing. |
| G17 | **LOW** | `polyslab.py:176` | `sidewall_angle`/`dilation` hard-rejected (functional limitation). |

### E. Boundary System (15 bugs)

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| B1 | **CRIT** | `kernels/boundaries.py:169-170` | PML memory computed but **never added to field** — memory has zero effect. |
| B2 | **CRIT** | `runtime/boundaries.py:294-320` | `pack_halo`/`unpack_halo` always slice axis 0 — y/z halo exchange broken. |
| B3 | **CRIT** | `runtime/boundaries.py:815-820` | Halo signs: identical if/else branches + hardcoded `"electric"` — magnetic signs never applied. |
| B4 | **CRIT** | `kernels/boundaries.py:123` | `ABSORBER` missing from ghost-cell skip set — gets PEC reflections. |
| B5 | **HIGH** | `models.py:205`, `compiler/boundaries.py:308` | PML sigma not physically scaled — fixed constant 1.5 regardless of grid/dt. |
| B6 | **HIGH** | `compiler/boundaries.py:296` | CFS-PML alpha ramps wrong direction for StablePML. |
| B7 | **HIGH** | `runtime/boundaries.py:727-734` | Warp cross-device transfer is a no-op. |
| B8 | **HIGH** | `runtime/boundaries.py:648` | `cuda_host_to_device` is a no-op for Warp arrays. |
| B9 | **MED** | `models.py:700-708` | `BoundarySpec.periodic(x=False)` ignores the `False` — both branches identical. |
| B10 | **MED** | `models.py:300` + all compiler/runtime | `extrude_structures` accepted but silently ignored. |
| B11 | **MED** | `compiler/boundaries.py:251` | ABC with `permittivity=None` defers error to compile time. |
| B12 | **MED** | `compiler/boundaries.py:307-309` | Inconsistent dt normalization in PML coefficients. |
| B13 | **MED** | `kernels/boundaries.py:355-359` | Ghost boundaries computed 2x, half discarded each time. |
| B14 | **MED** | `kernels/boundaries.py:449-463` | Warp ghost kernel updates only 1 of 3 vector components. |
| B15 | **MED** | `kernels/boundaries.py:435-463` | Warp ghost kernel ignores `mode_code` and `signs`. |

### F. Tests & Dead Code (findings)

| # | Severity | Description |
|---|----------|-------------|
| T1 | **CRIT** | **8 physics accuracy tests deleted** — Mie, Fresnel, Beer-Lambert, Bragg, DBR, PML reflection, TFSF validation, plasmonic nanoparticle. Only `__pycache__` remains. |
| T2 | **HIGH** | Zero tests compare FDTD output to analytical solutions. |
| T3 | **HIGH** | Energy conservation test uses 15% tolerance on 20³ grid for 50 steps — extremely loose. |
| T4 | **MED** | 9 stale `__pycache__`-only directories: `kernel/`, `boundary/`, `domain/`, `medium/`, `monitor/`, `modesolver/`, `simulation/`, `source/`, `tests/`. |
| T5 | **LOW** | `_initialize_source_runtimes` (execution.py:542) is dead code — dict populated but never read. |
| T6 | **LOW** | `planning.py` is documentation-as-code — no runtime effect. |
| T7 | **LOW** | Duplicate `__all__` entry in `compiler/__init__.py:135,137`. |
| T8 | **LOW** | Missing `__all__` exports for `compile_plane_wave`, `compile_point_dipole`, etc. |

---

## Structural Recommendations

### 1. Adopt B/D formulation (like meep and Khronos.jl)
The current E/H-only approach cannot support CPML, proper dispersive coupling, or correct source injection. Refactor to:
```
B_new = B_old + Δt * curl(E)    [with PML cascade: C→U→T]
H = μ⁻¹ * (B + S_B - P_B)      [constitutive]
D_new = D_old + Δt * curl(H)    [with PML cascade: C→U→T]
E = ε⁻¹ * (D + S_D - P_D)      [constitutive]
```

### 2. Adopt SoA field layout (like Khronos.jl)
Replace `E[nx,ny,nz,3]` with separate `Ex[nx,ny,nz], Ey[nx,ny,nz], Ez[nx,ny,nz]`. This improves GPU coalescing by ~30% and simplifies per-component PML/dispersive handling.

### 3. Adopt meep's natural units (ε₀=μ₀=1)
This eliminates the ε₀/μ₀ confusion entirely. Users specify relative permittivity, and the solver uses it directly.

### 4. Wire coefficient-aware kernels into the stepping loop
The `electric_update_3d_coeff` / `magnetic_update_3d_coeff` kernels already correctly handle decay/drive coefficients. Replace `step_maxwell()` with a version that dispatches to coefficient-aware kernels using the compiled coefficients from `_compile_scene_coefficients`.

### 5. Implement proper geometry rasterization in the compilation pipeline
Call `materialize_grid()` from `compile_simulation()` and populate per-cell coefficient arrays. The infrastructure exists in `discretization.py` but is never called.

### 6. Re-enable physics accuracy tests
The deleted test files (Mie, Fresnel, Beer-Lambert, etc.) should be restored and made to pass against analytical solutions. Without them, there is no way to verify solver correctness.

### 7. Separate E and H source injection in the stepping loop
H sources at time t, E sources at time t+Δt/2 — matching meep and Khronos.jl.

### 8. Fix DFT monitors to accumulate with proper `dt` scaling and time stagger
Each field component's DFT should use the correct time at which that component is evaluated (H at t, E at t+Δt/2).
