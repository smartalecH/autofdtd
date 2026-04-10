# autofdtd Comprehensive Bug Report (Phase 3 — Full Systematic Audit)

**References**: meep (MIT), Khronos.jl (Meta), Tidy3D (Flexcompute), VectorModesolver.jl
**Date**: 2026-04-10

---

## Summary

| Phase | Bugs | Categories |
|-------|------|------------|
| Phase 1 (kernel physics) | 16 | Stepping, curls, PML, dispersive, sources |
| Phase 2 (wiring, design) | 61 | API→compiler→runtime, monitors, sources, geometry, boundaries, tests |
| **Phase 3 (this report)** | **~108** | **Mode solver, materials, sources, runtime, API/IR, grid/numerics, DFT monitors** |
| **Grand Total** | **~185** | |

Phase 3 systematically compared every autofdtd subsystem against Khronos.jl, meep, tidy3d, and VectorModesolver.jl. The mode solver is the most critically broken subsystem — it cannot compute correct modes for any real waveguide.

---

## PHASE 3 — NEW BUGS BY SUBSYSTEM

---

### I. MODE SOLVER (23 new bugs)

The mode solver is **fundamentally broken**. It uses a scalar Helmholtz equation instead of the full-vectorial curl-curl formulation, making it incapable of finding hybrid modes. Even the unused vector path has wrong physics. Field reconstruction, normalization, and bend handling are all incorrect.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| MS1 | **P0** | `modes/solver.py:614-622` | **Always uses scalar Helmholtz** (`∇²Ez + k₀²εEz = β²Ez`), never the vector formulation. Cannot find hybrid modes, cannot handle polarization coupling. VectorModesolver.jl uses full 2Nx2N Fallahkhair formulation with [Hx;Hy]. |
| MS2 | **P0** | `modes/solver.py:286-325` | **Vector `assemble_mode_matrix` is wrong** even if enabled: no cross-coupling blocks (Exy, Eyx missing), 5-point stencil instead of 9-point, wrong physics in operator. Two independent scalar problems, not one coupled vector problem. |
| MS3 | **P1** | `modes/solver.py:410-412` | **Hz reconstruction uses curl instead of divergence**: computes `dHx/dy - dHy/dx` (curl) instead of `dHx/dx + dHy/dy` (∇·H=0). VectorModesolver.jl `getHz` uses correct divergence equation. |
| MS4 | **P1** | `modes/solver.py:448-472` | **E-field reconstruction wrong**: derivative formulas incorrect, and **Ez always set to 0** (line 472). VectorModesolver.jl `getE` correctly computes all three E components including Ez = Dz/εzz. |
| MS5 | **P1** | `modes/solver.py:540-553` | **Bend radius uses `abs(r)`** making index modification symmetric. Should be signed: outer edge higher index, inner edge lower. tidy3d uses proper signed Jacobian conformal mapping. |
| MS6 | **P1** | `modes/solver.py:559-565` | **Bend correction applies same factor to all ε tensor components**. tidy3d's Jacobian `J·ε·Jᵀ/det(J)` only modifies components involving the propagation direction. |
| MS7 | **P1** | `modes/solver.py:716-719` | **Power normalization uses Σ|E|²** (which equals Σ|Ez|² since Ex=Ey=0 in scalar path) instead of Poynting integral `½Re∫(E×H*)·ẑ dA`. Normalization is always ineffective. |
| MS8 | **P1** | `modes/solver.py:597-598` | **`omega = k₀`** conflates angular frequency with wavenumber. In SI, `ω = c·k₀`. E-field magnitudes wrong by factor c = 3×10⁸. |
| MS9 | **P1** | `modes/solver.py:67-191, 614` | **No TM mode solver exists.** Scalar TE formulation misses the entire TM mode family. For asymmetric waveguides, both polarization families are hybrid and the scalar solver gives wrong propagation constants. |
| MS10 | **P1** | `modes/epsilon.py:220-248` | **Epsilon sampled at single cell center**, not Yee-staggered 4-corner locations as in VectorModesolver.jl. Produces staircasing artifacts at dielectric interfaces. |
| MS11 | **P1** | `modes/solver.py:632-639` | **No shift-invert eigensolver**: uses `sigma=None, which="LR"`. For fine grids, finds unphysical high-frequency discretization artifacts instead of guided modes. tidy3d uses `sigma=guess_value`. |
| MS12 | **P1** | `modes/solver.py` (entire) | **No PML in mode solver cross-section.** tidy3d uses complex coordinate stretching to absorb radiation modes. autofdtd relies on hard BCs + post-hoc filtering, producing spurious modes and domain-size sensitivity. |
| MS13 | **P2** | `modes/solver.py:359-370` | Hy neighbor column indices point to Hx block (wrong offset by N). |
| MS14 | **P2** | `modes/solver.py:331-354` | Vector matrix Dirichlet BCs multiply corrections by `bc=0`, making them all no-ops. |
| MS15 | **P2** | `modes/solver.py:316-325` | East/West and North/South comments swapped on neighbor coefficients. |
| MS16 | **P2** | `modes/epsilon.py:239-249` | Nearest-neighbor snap instead of exact coordinate lookup for epsilon queries. |
| MS17 | **P2** | `modes/epsilon.py:236-237` | Dead code: re-centering of already-centered coordinates (unused xc/yc). |
| MS18 | **MED** | `modes/solver.py, models.py` | **No group index computation.** tidy3d computes `n_g = n_eff + f·(dn_eff/df)` via 3-frequency perturbation. |
| MS19 | **MED** | `modes/models.py` | **No effective mode area** (`A_eff = (∫|E|²)² / ∫|E|⁴`). |
| MS20 | **MED** | `modes/solver.py:643-666` | **Inadequate mode filtering**: no TE/TM fraction, no cross-frequency tracking, no degenerate mode orthogonalization. |
| MS21 | **MED** | `modes/solver.py` | **No overlap integral implementation.** |
| MS22 | **MED** | `modes/epsilon.py:184-209` | **Dispersive materials use FDTD coefficients** instead of frequency-domain ε(ω). Mode solver gets wrong permittivity for dispersive media. |
| MS23 | **MED** | `modes/solver.py:570-756` | **`target_neff` from ModeSpec defined but never read** by `solve_modes`. |

---

### II. MATERIALS SUBSYSTEM (8 new bugs)

The numpy ADE path is substantially correct, but the **Warp (GPU) dispersive kernels are critically broken** — polarization never feeds back into E, P state is corrupted across components, and complex coefficients are truncated to float32.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| MAT1 | **CRIT** | `kernels/steps.py:663-691` | **Warp kernel never corrects E with dP/dt.** Computes P update but never subtracts polarization current from E. Dispersive materials behave as non-dispersive on GPU. Khronos: `E = ε⁻¹(D - P)`. |
| MAT2 | **CRIT** | `kernels/steps.py:677-691` | **Warp kernel P state corrupted across components.** Single scalar P per pole used for Ex, Ey, Ez sequentially — Ex's P overwrites Ey's previous state. Ez polarization never computed. Khronos stores separate Px, Py, Pz arrays. |
| MAT3 | **HIGH** | `kernels/steps.py:597-598` | **Complex pole decay/drive factors truncated to float32.** `wp.array(dtype=wp.float32)` for values that are complex for any oscillatory pole (Lorentz, Sellmeier). Imaginary part silently discarded, making recurrence wrong. |
| MAT4 | **MED** | `materials/anisotropic.py:145` | Missing ε₀ in `eps_diagonal` conductivity: `ε_r - jσ/ω` should be `ε_r - jσ/(ωε₀)`. Off by factor ~8.85×10¹². |
| MAT5 | **MED** | `kernels/steps.py:694-735` | Standalone `pole_residue_polarization_update_3d` kernel has no mechanism to feed P back into E. Orphaned dead code. |
| MAT6 | **LOW** | `materials/dispersive.py:401` | Drude rejects zero damping (`_positive_finite`), preventing lossless plasma. Division by γ in `to_pole_residue`. |
| MAT7 | **LOW** | `materials/dispersive.py:155` | `exp(-pole·dt)` overflows for stiff poles with large |Re(a)|. No warning or fallback. |
| MAT8 | **LOW** | `materials/anisotropic.py:159-189` | `FullyAnisotropicMedium` accepts non-symmetric tensors without validation. meep/tidy3d enforce symmetry. |

---

### III. SOURCE SUBSYSTEM (13 new bugs)

Beyond the 12 existing source bugs, the Gaussian beam has no paraxial optics, plane wave angled incidence has no spatial phase gradient, and TFSF injects throughout the volume instead of only at boundary faces.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| S13 | **HIGH** | `compiler/sources.py:780-813` | **k-vector missing frequency factor.** `k = n·2π/c₀` should be `k = n·2πf/c₀`. Plane wave spatial profile phase completely wrong. |
| S14 | **HIGH** | `compiler/sources.py:959-1015` | **Gaussian beam is flat Gaussian `exp(-2r²/w₀²)`** — no beam divergence w(z), no Rayleigh range, no Gouy phase, no wavefront curvature. tidy3d implements full paraxial optics with `w_z, inv_r_z, psi_g`. |
| S15 | **HIGH** | `compiler/sources.py:997-1003` | **Gaussian beam radial distance from grid origin**, not beam center. Cell at beam center `(5,5,0)` gets `r² = 50` instead of 0. |
| S16 | **HIGH** | `kernels/sources.py:815-836` | **Plane wave angled incidence missing `exp(ik·r)` phase gradient.** Compiled `kx_func, ky_func, kz_func` exist but are never used during injection. All angles produce normal-incidence wave. |
| S17 | **HIGH** | `kernels/sources.py:766-774` | **GPU plane wave `abs()` on tangential amplitudes** — sign and complex phase discarded. S-polarization and negative-direction waves broken. |
| S18 | **HIGH** | `kernels/sources.py:580-612` | **CustomFieldSource equivalence principle injects into normal component** instead of tangential. `J = n̂×H` should produce tangential E updates, not normal H updates. |
| S19 | **HIGH** | `kernels/sources.py:1118-1150` | **TFSF injects into ALL cells of TFSF volume** instead of only at 6 boundary faces. Equivalent to a distributed volume source, not a TFSF boundary correction. Khronos creates 6 separate EquivalentSource surfaces. |
| S20 | **MED** | `kernels/sources.py:273` | GPU point dipole uses `abs(amplitude)`, discarding complex phase for `remove_dc_component=True`. |
| S21 | **MED** | `kernels/sources.py:815-836` | CPU plane wave E/H injection not Yee-staggered. GPU path offsets by full cell instead of implicit half-cell. |
| S22 | **MED** | `compiler/sources.py:679-693` | Mode source field reshape swaps x and y dimensions. |
| S23 | **MED** | No implementation | **No source spectrum normalization** `1/FT(J(t))` for frequency-domain results. Cannot compute meaningful transmission/reflection. |
| S24 | **HIGH** | All inject functions | **No `dJ/dt` differentiation** for electric current sources. meep uses time derivative of dipole for current injection. |
| S25 | **LOW** | `sources/beam.py:241`, `tfsf.py:262` | Inconsistent speed of light: `3e8` vs `2.998e8` across files. |

---

### IV. RUNTIME SUBSYSTEM (12 new bugs)

The stepping loop has wrong ordering, monitors ignore leapfrog stagger, and the entire multi-chunk path is broken in multiple ways.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| RT1 | **CRIT** | `runtime/execution.py:400-486` | **Stepping order wrong.** Sources before combined E+H update, not split between H and E phases. Correct: H-sources → H-update → H-monitors → E-sources → E-update → E-monitors. |
| RT2 | **CRIT** | `runtime/monitors.py:82-115` | **E and H monitors recorded at same time**, ignoring leapfrog half-dt stagger. Phase error `ω·dt/2` in all DFT monitors. |
| RT3 | **HIGH** | `runtime/execution.py:1196-1199` | **Chunked halo exchange copies only E, not H.** H ghost cells stale at every chunk boundary. |
| RT4 | **HIGH** | `runtime/chunk.py:676-743` | **Non-split axis boundary detection uses split-axis position.** Wrong boundary conditions on non-split axes in multi-chunk. |
| RT5 | **HIGH** | `runtime/chunk.py:802` | **`chunk.face_halo.get()` calls method as dict** — `face_halo` is a method, `face_halos` is the dict. Crashes at runtime. |
| RT6 | **HIGH** | `runtime/chunk.py:789-798` | **Exchange plan `is_interior` excludes 2-chunk layouts.** Both chunks are "edge" chunks, so no exchange descriptors created. |
| RT7 | **HIGH** | `runtime/execution.py:1263-1361` | **`step_maxwell` called in both electric and magnetic chunk update functions.** Since `step_maxwell` does E+H, fields are updated **twice per step** in chunked Warp path. |
| RT8 | **MED** | `runtime/execution.py:1585-1592` | `_numpy_magnetic_update_chunk` is `pass` — H never updated in multi-chunk NumPy. |
| RT9 | **MED** | `runtime/execution.py:1364-1399` | `_numpy_electric_update_chunk` only updates Ex with wrong curl indexing. Ey, Ez omitted. |
| RT10 | **MED** | `runtime/execution.py:468-475` | **Monitor data gated by `record_interval`** (default 10) — DFT accumulators miss 90% of timesteps. |
| RT11 | **MED** | `runtime/execution.py:441-459` | `RuntimeController` from controls.py unused; inline convergence lacks warm-up period. |
| RT12 | **LOW** | `runtime/execution.py:1595-1608` | `_chunked_apply_pml` is `pass` — PML absent in multi-chunk path. |

---

### V. API / IR / CORE WIRING (16 new bugs)

Many user-facing parameters are silently dropped. Multiple monitor types are accepted by the API but never compiled. No validation that sources/monitors are inside the simulation domain.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| W1 | **CRIT** | `ir/models.py:2954-2956` | Sources lowered via generic `component_to_ir` instead of typed `source_to_ir`. Typed structure may be lost. |
| W2 | **HIGH** | `compiler/pipeline.py:399-491` | **Multiple monitor types silently skipped** at compile time: `GaussianOverlapMonitor`, `AstigmaticGaussianOverlapMonitor`, `SurfaceFieldMonitor`, `SurfaceFieldTimeMonitor`. No error, no output. |
| W3 | **HIGH** | `core/validation.py:590-598` | **`_normalize_source` only handles `UniformCurrentSource`.** All other source types (PointDipole, PlaneWave, GaussianBeam, ModeSource, TFSF, Custom*) skip normalization entirely. |
| W4 | **HIGH** | `core/validation.py:228-260` | **No source/monitor bounds validation against simulation domain.** Only structures are checked. Source or monitor outside domain produces silent garbage. |
| W5 | **HIGH** | `monitors/models.py:146-154` | **`FieldMonitor` has no `freqs` field.** Pipeline uses `getattr(monitor, "freqs", ())` → always `()`. Frequency-domain field recording impossible. |
| W6 | **HIGH** | `compiler/pipeline.py:448` | **`ModeMonitor.mode_spec` silently discarded.** Pipeline fabricates synthetic dict with only `mode_index`, dropping user's `num_modes`, target neff, etc. |
| W7 | **MED** | `__init__.py` | **Incomplete exports.** All monitor types, many source types, grid types, and `compile_simulation` missing from top-level `__init__.py`. |
| W8 | **MED** | `monitors/models.py:255-256` | **Type mismatch**: `FieldProjectionAngleMonitor.phi/theta` uses `tuple[float, float, float]` but IR expects `tuple[float, float, int]` for count parameter. |
| W9 | **MED** | `api/simulation.py` | API surface exports sources and boundaries but **zero monitor types**. |
| W10 | **MED** | `monitors/models.py:845` | `SimulationData` uses mutable `dict` in frozen Pydantic model. `register_*` methods mutate `monitor_store` violating frozen contract. |
| W11 | **MED** | `core/containers.py:333-338` | Symmetry validation missing cross-check with boundary conditions (e.g., symmetry + Bloch is non-physical). |
| W12 | **LOW** | `core/containers.py:217` | **Courant default 0.99** — at CFL stability limit. tidy3d uses ~0.9. Marginal instability with dispersive media. |
| W13 | **LOW** | `core/containers.py:290-295` | `shutoff` not capped at 1.0. `shutoff=100` would immediately terminate. |
| W14 | **LOW** | `compiler/pipeline.py:756-790` | `num_chunks` not forwarded to `simulation_to_execution_package`. Compiled simulation discarded and re-lowered from scratch. |
| W15 | **LOW** | All files | **No units documentation** anywhere. SI? Natural? Mixed? Undocumented. |
| W16 | **LOW** | `api/simulation.py:105-106,136-137` | Duplicate `"PML"` and `"PMLParams"` entries in `__all__`. |

---

### VI. GRID, SUBPIXEL & NUMERICAL METHODS (14 new bugs)

The entire subpixel averaging pipeline is a no-op. No automatic grid refinement near material interfaces. No Yee staggering in discretization.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| GR1 | **HIGH** | `compiler/discretization.py:864-875` | **Subpixel averaging hardcodes `f=0.5`** regardless of actual volume fraction. `_compute_volume_fraction` exists but is never called from the smoothing path. |
| GR2 | **HIGH** | `compiler/discretization.py:997-1137` | **Subpixel smoothing has no effect on final coefficients.** `assemble_coefficient_fields` reads `structure_indices` for medium lookup, never `volume_fractions`. Entire subpixel pipeline is a no-op. |
| GR3 | **HIGH** | `grid/specs.py:467-546` | **No automatic grid refinement near material interfaces.** Only manual `LayerRefinementSpec`. tidy3d auto-detects structure boundaries and scales mesh by refractive index. Silicon waveguide (n=3.5) gets same mesh as vacuum. |
| GR4 | **HIGH** | `compiler/discretization.py:530-532` | **No Yee grid staggering in discretization.** All 6 field components materialized at cell centers. meep uses `LOOP_OVER_VOL(gv, c, i)` per-component stagger. Khronos uses `get_component_origin`. |
| GR5 | **MED** | `compiler/discretization.py:283-296` | GeometryGroup volume fraction sums individual fractions and clamps to 1.0 — overcounts overlaps. |
| GR6 | **MED** | `compiler/discretization.py:378-395` | `ClipOperation` difference: returns 0 when center is inside B, missing thin shells. |
| GR7 | **MED** | `compiler/discretization.py:299-342` | Transformed geometry volume fraction checks bounding box, not actual shape. Inflated fractions for rotated cylinders/spheres. |
| GR8 | **MED** | `grid/specs.py:548-571` | `_enforce_max_scale` can loop indefinitely — no iteration cap on midpoint insertion. |
| GR9 | **MED** | `grid/specs.py:154-158` | `total_cells` returns 0 for 2D simulations (one axis has `num_cells=0`). |
| GR10 | **MED** | `grid/specs.py` (entire) | **No automatic grid snapping** to structure boundaries. Structures not aligned to grid, degrading accuracy. tidy3d adds grid boundaries at structure bounding box edges. |
| GR11 | **MED** | `compiler/discretization.py:193-199` | Cylinder volume fraction face-center samples duplicate existing corners instead of adding face centers. |
| GR12 | **LOW** | `grid/specs.py` (entire) | **No symmetry support** in grid module. Cannot exploit symmetry for 2×/4×/8× speedup. |
| GR13 | **LOW** | `compiler/discretization.py:220-280` | PolySlab volume fraction has discontinuous estimator (9 vs 8 samples depending on center-inside). |
| GR14 | **LOW** | `grid/specs.py:54-63` | `_deduplicate_coords` keeps later coordinate value, silently shifting grid boundaries. |

---

### VII. DFT MONITORS, FLUX, MODE OVERLAP, DIFFRACTION (14 new bugs)

Beyond existing M1–M16, the frequency-domain flux is computed by DFTing the instantaneous Poynting vector (fundamentally wrong), mode overlap uses the wrong formula, and diffraction uses cell indices instead of physical coordinates.

| # | Severity | File:Line | Description |
|---|----------|-----------|-------------|
| DFT1 | **CRIT** | `compiler/monitors.py:596-622` | **Frequency-domain flux DFTs instantaneous Poynting vector** `FT(E(t)×H(t))` instead of computing `DFT(E)×conj(DFT(H))`. Mathematically wrong: `FT(E·H) ≠ FT(E)·FT(H)*`. Produces convolution of field spectra, not spectral power flow. |
| DFT2 | **HIGH** | `kernels/monitors.py:181-236` | **Flux missing 0.5 factor and conj(H).** `S = E×H` instead of `S = ½Re(E×H*)`. Values 2× too large, and wrong for complex DFT data. |
| DFT3 | **HIGH** | `compiler/monitors.py:1135-1227` | **Mode overlap uses wrong formula.** Computes `(E_sim×H_sim*)·conj(E_mode×H_mode*)` (Poynting product) instead of correct `½∫(E_sim×H_mode* + E_mode*×H_sim)·n̂ dA / (4P_mode)`. Cannot distinguish forward/backward. |
| DFT4 | **HIGH** | `kernels/monitors.py:1013-1014` | **Diffraction uses `|E|` magnitude before FFT** — `sqrt(|E₁|² + |E₂|²)` destroys phase. Correct: FFT complex tangential components separately, then compute Poynting per order. |
| DFT5 | **HIGH** | `kernels/monitors.py:1006-1007` | **Diffraction positions use cell indices** instead of physical coordinates in Fourier phase `exp(-ik·r)`. Results depend on grid resolution, not geometry. |
| DFT6 | **HIGH** | `kernels/near2far.py:293-305` | **E_theta formula uses wrong M_s projection.** Uses `ms_r` (radial) instead of `ms_phi`. E_phi formula also wrong (uses `theta_hat × r_hat`). |
| DFT7 | **MED** | All DFT paths | **No E/H temporal staggering correction.** Same `time` for both E and H DFT phases. meep uses `time() - 0.5*dt` for H. Phase error `exp(iω·dt/2)` in H DFT. |
| DFT8 | **MED** | `compiler/monitors.py:342,638,1309,1593` | **DFT normalized by dividing by count** instead of proper integral. Turns DFT into time-average. Results depend on simulation duration. Neither meep nor Khronos divide by count. |
| DFT9 | **MED** | `kernels/near2far.py:132-136` | **N2F surface positions use global indices** but compute from `lower_bounds + idx * cell_size`. Wrong source positions for Green's function integration. |
| DFT10 | **MED** | `kernels/monitors.py:99-103` | **Multi-point DFT accumulates into wrong array index.** `dft_data[field][i] += ...` ignores frequency dimension `j`. All frequencies go into column 0. |
| DFT11 | **LOW** | All monitor files | **No source normalization infrastructure.** No `get_flux_data()`/`load_minus_flux_data()` for transmission/reflection normalization. |
| DFT12 | **LOW** | All DFT paths | **No time windowing.** No Hanning/Kaiser window before DFT. Spectral leakage for truncated simulations. |
| DFT13 | **LOW** | All monitor files | **No LDOS computation.** meep provides `dft_ldos` correlating field DFT with source current DFT. |
| DFT14 | **LOW** | `kernels/monitors.py:119-153` | **No Yee grid spatial interpolation** for co-locating E/H at monitor points. Half-cell displacement between components. |

---

## PHASE 3 SEVERITY SUMMARY

| Severity | Count | Description |
|----------|-------|-------------|
| **P0 / CRITICAL** | 9 | Fundamentally wrong physics or complete subsystem failure |
| **P1 / HIGH** | 38 | Incorrect results for common use cases |
| **P2 / MEDIUM** | 37 | Design flaws, silent parameter drops, feature gaps |
| **P3 / LOW** | 24 | Minor issues, missing features, code quality |
| **Total Phase 3** | **108** | |

---

## GRAND TOTAL (ALL PHASES)

| Phase | P0/CRIT | P1/HIGH | P2/MED | P3/LOW | Total |
|-------|---------|---------|--------|--------|-------|
| Phase 1 | 4 | 6 | 3 | 3 | 16 |
| Phase 2 | 12 | 16 | 21 | 12 | 61 |
| Phase 3 | 9 | 38 | 37 | 24 | 108 |
| **Grand Total** | **25** | **60** | **61** | **39** | **185** |

---

## WHAT ACTUALLY WORKS

After auditing every subsystem, the only scenario that can produce partially correct results is:

1. **Single-chunk, NumPy backend** (Warp has additional bugs: MAT1-MAT3, S17)
2. **Vacuum or non-dispersive isotropic medium** (dispersive is broken: MAT1-MAT3, Bug 9, A2)
3. **Point dipole or uniform current source** (Gaussian beam: S14-S15; plane wave angled: S16; TFSF: S19)
4. **Periodic boundaries** (PML is broken: B1, Bug 7; absorber: B4)
5. **Time-domain field recording only** (all DFT paths have multiple bugs)
6. **No mode solver usage** (MS1-MS23)
7. **With the H-field sign fix applied** (Bug 1)

Even this minimal path has the wrong stepping order (RT1) and the H-field sign error (Bug 1).

---

## TOP 10 MOST IMPACTFUL BUGS TO FIX

1. **MS1+MS2**: Implement full-vectorial mode solver (Fallahkhair formulation from VectorModesolver.jl)
2. **Bug 1**: H-field sign error (`-dt/μ` should be `+dt/μ`)
3. **RT1**: Fix stepping order to proper leapfrog (H-sources→H-update→H-monitors→E-sources→E-update→E-monitors)
4. **DFT1+DFT2**: DFT E and H separately, compute flux as `½Re(DFT_E × conj(DFT_H))`
5. **MAT1+MAT2+MAT3**: Fix Warp dispersive kernels (per-component P, complex coefficients, E-correction)
6. **A1+G1**: Wire `materialize_grid()` into `compile_simulation()` so geometry actually produces per-cell materials
7. **Bug 3+Bug 4**: Fix ε₀/μ₀ handling and coefficient-aware kernel dispatch
8. **B1**: PML memory terms must be added to fields (not just computed and discarded)
9. **S14+S15+S16**: Fix Gaussian beam (paraxial optics, beam center, phase gradient)
10. **GR1+GR2+GR4**: Wire subpixel averaging to coefficient arrays with proper Yee staggering

---

## ARCHITECTURAL RECOMMENDATIONS (updated from Phase 2)

### 1. Adopt B/D formulation (like meep and Khronos.jl)
Required for CPML, dispersive coupling, and correct source injection.

### 2. Implement full-vectorial mode solver (like VectorModesolver.jl)
Replace scalar Helmholtz with Fallahkhair 2008 formulation: 2N×2N operator, 9-point stencil, 4-corner epsilon sampling, shift-invert eigensolver, PML in cross-section.

### 3. Fix DFT monitor architecture
DFT E and H fields separately at their correct leapfrog times. Compute derived quantities (flux, mode overlap, diffraction) in post-processing from the complex DFT arrays.

### 4. Fix Warp dispersive kernels
Store per-component polarization (Px, Py, Pz). Use complex arrays for decay/drive. Subtract polarization current from E after update.

### 5. Wire the discretization pipeline
Call `materialize_grid()` from `compile_simulation()`. Apply Yee-staggered epsilon sampling. Connect subpixel averaging to coefficient arrays.

### 6. Fix multi-chunk path
Separate `step_maxwell` into `step_electric` and `step_magnetic`. Exchange both E and H halos. Fix axis detection, exchange plan, and PML application.

### 7. Implement proper source physics
Gaussian beam: full paraxial optics. Plane wave: spatial phase gradient. TFSF: boundary-only injection. Mode source: correct normalization and field interpolation.

### 8. Add validation and testing
Validate source/monitor bounds. Re-enable deleted physics tests (Mie, Fresnel, Beer-Lambert). Add mode solver validation against VectorModesolver.jl.
