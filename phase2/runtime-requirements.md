# Phase 2 Runtime Infrastructure Requirements

Each accuracy validation example needs specific runtime infrastructure beyond the numerical kernels.
This document enumerates per-example requirements and identifies gaps that Phase 2 must close.

## Per-Example Requirements

### Example 1 — Vacuum Plane-Wave Propagation

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| Domain init on GPU (efficient) | Done — Warp persistent arrays | — |
| Yee timestepping on GPU | Done — task-049 | — |
| PML coefficient prep | Done — task-051 | — |
| Field recording to host | Done | — |
| Analytical pulse reference | N/A (internal) | Implement analytical E(x,t) reference for comparison |
| `cells/s` metric | Done — `ExecutionResult.metrics` | — |
| Convergence: L₂ error vs. cell count | N/A in Phase 1 | Add convergence sweep and L₂ computation |

### Example 2 — Planar Multilayer Reflectance (Fresnel)

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| PlaneWave source injection | Done — task-034 | — |
| Subpixel averaging coefficients | Done — task-046 | — |
| FluxMonitor accumulation | Done | — |
| Fresnel equation reference impl | N/A | Implement Fresnel TMM for planar multilayer |
| Grid convergence sweep | N/A | Run at 15/25/40 ppw; compare R,T to Fresnel |
| Oblique incidence support | ? | Verify PlaneWave handles non-normal incidence |

### Example 3 — Mie Scattering from a Sphere

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| TFSF source injection | Done — task-035 | — |
| Sphere geometry intersection | Done — task-006 | — |
| Dispersive material updates | Done — task-050 | — |
| FluxMonitor on spherical surface | Partially done | Verify spherical surface flux monitor works |
| Mie series reference impl | N/A | Implement `mie.py` — efficiencies Q_ext, Q_sca, Q_abs, angular pattern |
| Scattered-field extraction | Done (TFSF separates) | — |
| Convergence: size parameter sweep | N/A | Run a=λ/10, λ/4, λ/2, λ; compare to Mie series |

### Example 4 — Symmetry-Reduced Waveguide

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| Symmetry metadata in Simulation | Done — task-020 | — |
| PEC/PMC plane transforms | Done | — |
| Symmetry-aware chunk planning | ? | Verify chunk planning respects symmetry planes |
| Full-domain reference run | N/A | Run full-domain variant for comparison |
| Weak-scale: speedup measurement | N/A | Measure cells/s reduced vs. full; verify ≈4× or 8× |

### Example 5 — PML and Absorber Reflectance

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| PML coefficient preparation | Done — task-051 | — |
| StablePML / Absorber boundaries | Done — task-022 | — |
| 1D TMM reference impl | N/A | Implement 1D slab TMM for PML reflectance |
| Layer count convergence sweep | N/A | Run PML 12/24/48 layers; measure reflectance |
| Grid-resolution coupling with PML | N/A | Run at 15/25/40 ppw; verify PML behavior independent of grid |
| Absorber vs. PML comparison | N/A | Verify both converge to same result with enough layers |

### Example 6 — Dispersive Medium (Lorentz/Drude)

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| Auxiliary state variable allocation | Done — task-050 | — |
| Lorentz/Drude update kernels | Done | — |
| Causal pulse response reference | N/A | Implement analytical causal impulse response for single-pole Lorentz |
| Convergence: dt refinement | N/A | Run at dt, dt/2, dt/4; measure error in pulse delay |
| Multi-pole dispersive path | Done | Verify 2-pole and 3-pole paths work |
| Causality check | N/A | Verify signal never arrives before t = x/c |

### Example 7 — Waveguide Mode Injection

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| ModeSource injection | Deferred — task-032 integration pending | **Critical**: Wire ModeSource into timestepping loop |
| ModeMonitor decomposition | Deferred — task-040 integration pending | **Critical**: Wire ModeMonitor into accumulation |
| ModeSpec and ModeSolver | Done | Verify end-to-end with ModeSource → ModeMonitor |
| Mode-solver eigenfield reference | Done (mode solver works) | — |
| Grid refinement in cross-section | N/A | Run 10/20/40 cells across waveguide width; compare mode profile |
| Power conservation check | N/A | Verify transmitted power ≈ 1 (no artificial loss) |

### Example 8 — Near-to-Far Field Projection

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| FieldMonitor accumulation | Done | — |
| DFT at monitor planes | Done — task-053 | — |
| FieldProjectionCartesianMonitor | Done — task-042 | — |
| Rayleigh-Sommerfeld reference | N/A | Implement RS diffraction integral for comparison |
| Convergence: focal spot error | N/A | Run at multiple monitor grid spacings; measure focal spot error |
| Projection distance sweep | N/A | Verify far-field pattern independent of projection distance |

### Example 9 — Bloch/Periodic Band Diagram

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| BlochBoundary implementation | Done — task-020 | — |
| Phase-aware halo exchange | Done — task-064 | — |
| k-point sampling | Done (GridSpec k_points) | Verify k-point convergence |
| Published band diagram reference | N/A | Find reference band diagram (Meep/MPB) for comparison |
| Supercell size sweep | N/A | Run 1/4/8 row supercell; verify band diagram stable |
| Mode classification (TE/TM) | N/A | Verify mode classification at band edges |

### Example 10 — Long Directional Coupler (Multi-Node)

| Requirement | Phase 1 status | Phase 2 action |
|---|---|---|
| ModeSource injection | Deferred (see Example 7) | — |
| ModeMonitor decomposition | Deferred (see Example 7) | — |
| Chunk decomposition | Done — task-063 | — |
| Cross-device halo exchange | Done — task-064 | — |
| `cells/s` weak scaling | Done — `gpu_benchmark.py` | Extend with coupler geometry (not just uniform grid) |
| TMM reference for coupling | N/A | Implement directional coupler TMM (even/odd mode analysis) |
| Halo correctness: coupling oscillation | N/A | **Critical**: Verify oscillatory coupling amplitude/period matches TMM |
| Multi-GPU scaling efficiency | Partially done | Measure weak scaling efficiency > 0.9 across 2→4 GPUs |

---

## Cross-Cutting Infrastructure Requirements

The following are needed by multiple examples and should be implemented once:

### 1. TMM Reference Implementation
**Needed by**: Examples 2, 5, 10
**Implementation**: `phase2/reference/tmm.py`
- 1D planar multilayer reflectance/transmittance
- Directional coupler even/odd mode analysis
- Support for complex refractive indices and angled incidence

### 2. Mie Series Reference
**Needed by**: Example 3
**Implementation**: `phase2/reference/mie.py`
- Draine & Pets大招 conventions for efficiency factors
- Angular scattering pattern
- Near/mid/far field computation

### 3. Convergence Sweep Runner
**Needed by**: All examples
**Implementation**: `phase2/tools/convergence.py`
- Runs simulation at N grid resolutions
- Computes L₂ field error against reference
- Fits convergence rate (should be ~2nd order for FDTD)
- Reports error at each resolution and rate

### 4. Weak-Scale Benchmark Runner
**Needed by**: All examples
**Implementation**: `phase2/tools/weak_scale.py`
- Runs simulation at N domain sizes (fixed resolution)
- Extracts `cells/s` from `ExecutionResult.metrics`
- Verifies throughput within 10% across sizes

### 5. Halo Correctness Checker
**Needed by**: Examples 4, 9, 10
**Implementation**: `phase2/tools/halo_check.py`
- Runs multi-GPU simulation
- Runs equivalent single-GPU simulation
- Compares field values at chunk boundaries
- Reports max error at halo interface

### 6. Rayleigh-Sommerfeld Reference
**Needed by**: Example 8
**Implementation**: `phase2/reference/rayleigh_sommerfeld.py`
- Near-to-far field diffraction integral
- For planar aperture fields
- Produces 2D far-field intensity pattern

---

## Critical Path

The following are blocking for specific examples:

```
ModeSource/ModeMonitor wiring (Example 7, 10)
  └─ Blocking: Example 10 cannot use ModeSource injection

Halo correctness checker (Examples 4, 9, 10)
  └─ Blocking: Example 10 multi-GPU validation needs this

TMM reference impl (Examples 2, 5, 10)
  └─ Blocking: Cannot compute error without reference

Mie series reference (Example 3)
  └─ Blocking: Cannot compute scattering cross-section error without reference
```

---

## Gap Summary

| Infrastructure | Status | Files to create/modify |
|---|---|---|
| TMM reference | Not built | `phase2/reference/tmm.py` |
| Mie series reference | Not built | `phase2/reference/mie.py` |
| Rayleigh-Sommerfeld reference | Not built | `phase2/reference/rayleigh_sommerfeld.py` |
| Convergence sweep runner | Not built | `phase2/tools/convergence.py` |
| Weak-scale benchmark runner | Not built | `phase2/tools/weak_scale.py` |
| Halo correctness checker | Not built | `phase2/tools/halo_check.py` |
| ModeSource → timestepping wiring | Deferred in Phase 1 | `autofdtd/kernels/sources.py`, `autofdtd/runtime/stepping.py` |
| ModeMonitor → accumulation wiring | Deferred in Phase 1 | `autofdtd/runtime/monitors.py` |
| Convergence error vs. analytical | Not built | `phase2/examples/convergence.py` |
