# Phase 2 Accuracy Validation Loop

Drive Phase 2 of autofdtd from the Phase 1 implementation baseline to a validated, 3D accuracy-verified FDTD package. Phase 2 runs 18 curated 3D examples against analytical, semi-analytical, and reference-solver ground truth, measures convergence rates and weak-scale throughput, validates multi-GPU halo correctness, and produces a clean pass/fail report for each example and permutation on a 2×16 GB GPU machine.

Hardware target: 2× NVIDIA GPUs, 16 GB each, connected via PCIe. All examples must execute correctly and produce physically accurate results within numerical error bounds.

Status: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `BLOCKED`

## Phase 1 Baseline

Phase 1 provides the following that Phase 2 depends on:

| Capability | Phase 1 task | Status |
|---|---|---|
| GPU execution (Warp, persistent arrays) | tasks 060-062 | ✓ Done |
| 2-GPU chunk decomposition | task-063 | ✓ Done |
| Cross-device halo exchange | task-064 | ✓ Done |
| GaussianBeam source | task-034 | ✓ Done |
| PlaneWave source | task-033 | ✓ Done |
| BlochBoundary | task-020 | ✓ Done |
| PML boundaries | task-021 | ✓ Done |
| Lorentz/Drude dispersive | task-016 | ✓ Done |
| ModeSource (API) | task-032 | ✓ Done (wiring deferred) |
| ModeMonitor (API) | task-040 | ✓ Done (wiring deferred) |
| N2F projection (API) | task-057 | ✓ Done (verification deferred) |
| 9 functional correctness examples | task-058 | ✓ Done |
| GPU benchmarking | task-065 | ✓ Done |

## Phase 2 Task Architecture

Each example has a **core task** plus **permutation tasks**:

- **Core task**: Baseline single-GPU run at default resolution, GPU init verification, error vs. ground truth
- **Convergence task**: Grid refines at 3 points, L₂ error vs. ground truth computed, convergence rate fitted
- **Weak-scale task**: Domain size grows at fixed resolution, `cells/s` tracked, no throughput regression
- **Multi-GPU task**: Same example on 2 GPUs with halo exchange, result matches single-GPU within tolerance
- **Reference task** (some examples): Implement analytical/semi-analytical reference if not yet built

Permutation tasks are numbered as subtasks: e.g., task-101 has subtasks task-101-conv, task-101-weak, task-101-mgpu.

---

## REFERENCE IMPLEMENTATION TASKS

## [QUEUED] task-201 - Implement TMM reference library
Success: Land `phase2/reference/tmm.py` with: (1) planar multilayer reflectance/transmittance for s/p polarization at arbitrary angle, (2) directional coupler even/odd mode analysis producing coupling coefficient C and sin²(C·z) power transfer, (3) ridge waveguide Bragg grating reflection spectrum. All functions accept numpy arrays and return reflectance/transmittance spectra.
Blocked By: —
Attempts: 0

## [QUEUED] task-202 - Implement Mie series reference library
Success: Land `phase2/reference/mie.py` with: (1) Draine & Pets大招 convention Mie efficiency factors Q_ext, Q_sca, Q_abs, (2) angular scattering amplitude matrix elements S1, S2, (3) bistatic RCS σ(θ,φ) in dB scale, (4) multipole decomposition (p, m, Q_e, Q_m contributions). Reference data files from `misc/` directory loaded as needed.
Blocked By: —
Attempts: 0

## [QUEUED] task-203 - Implement Rayleigh-Sommerfeld reference library
Success: Land `phase2/reference/rayleigh_sommerfeld.py` with: (1) near-to-far field diffraction integral for planar apertures, (2) zone plate focal intensity and position from Fresnel zone theory, (3) far-field intensity pattern I(θ,φ) from aperture fields.
Blocked By: —
Attempts: 0

## [QUEUED] task-204 - Implement convergence sweep runner
Success: Land `phase2/tools/convergence.py` with: (1) runs simulation at N grid resolutions, (2) extracts L₂ field error or R/T error vs. ground truth, (3) fits convergence rate via log-log linear regression, (4) reports rate, error at each resolution, and pass/fail vs. 1.5-order minimum threshold. Works with any Simulation → ExecutionResult pipeline.
Blocked By: —
Attempts: 0

## [QUEUED] task-205 - Implement weak-scale benchmark runner
Success: Land `phase2/tools/weak_scale.py` with: (1) runs simulation at N domain sizes at fixed resolution, (2) extracts `cells/s` from `ExecutionResult.metrics`, (3) verifies throughput within 10% across all sizes, (4) reports per-size timing and aggregate pass/fail. Works with any Simulation → ExecutionResult pipeline.
Blocked By: —
Attempts: 0

## [QUEUED] task-206 - Implement halo correctness checker
Success: Land `phase2/tools/halo_check.py` with: (1) runs given simulation on 2 GPUs (chunk decomposition), (2) runs same simulation on 1 GPU (single chunk), (3) compares field values at chunk boundary locations, (4) reports max error, mean error, and pass/fail vs. 1e-6 tolerance. Used to validate halo exchange correctness for Examples 4, 9, 10.
Blocked By: task-205 (uses weak_scale infrastructure)
Attempts: 0

---

## EXAMPLE TASKS — GROUP A (small, no ModeSource/N2D dependency)

## [QUEUED] task-101 - Example 1: Focused Gaussian Beam in Vacuum (3D)
Success: Example 1 baseline, convergence, weak-scale, and multi-GPU all pass. Ground truth: analytical Gaussian beam waist formula. Error < 1% at finest grid for peak intensity and waist position. Multi-GPU result matches single-GPU within 1e-6.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-101-base** - Example 1 baseline run
Success: Gaussian beam, λ=1 µm, waist w₀=2 µm, domain 10×8×8 µm, 20 ppw. Simulation compiles, runs on GPU, produces fields. Field at waist matches analytical E(r=0, z=0) within 2% at 20 ppw.
Blocked By: task-101
Attempts: 0

**[QUEUED] task-101-conv** - Example 1 convergence sweep
Success: Grid refines 20 → 40 → 80 ppw. L₂ field error vs. analytical computed at each resolution. Convergence rate fitted ≥ 1.5. Pass/fail reported.
Blocked By: task-101-base, task-204
Attempts: 0

**[QUEUED] task-101-weak** - Example 1 weak-scale sweep
Success: Domain grows 10³ → 20³ → 40³ µm³ at fixed 20 ppw. `cells/s` measured at each size. No throughput regression (>10% drop). Pass/fail reported.
Blocked By: task-101-base, task-205
Attempts: 0

**[QUEUED] task-101-mgpu** - Example 1 multi-GPU validation
Success: Same geometry on 2 GPUs with chunk decomposition. Field at waist matches single-GPU result within 1e-6. Halo correctness verified.
Blocked By: task-101-base, task-206
Attempts: 0

---

## [QUEUED] task-102 - Example 2: Planar Multilayer Reflectance (Fresnel)
Success: Example 2 baseline, convergence, weak-scale, and angular sweep all pass. Ground truth: Fresnel equations. Reflectance error < 0.5% at finest grid for 0°, 30°, 45° incidence.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-102-base** - Example 2 baseline run
Success: Si slab (εᵣ=6), normal incidence, 20 ppw. R and T from FluxMonitor match Fresnel within 2%.
Blocked By: task-102
Attempts: 0

**[QUEUED] task-102-conv** - Example 2 convergence sweep
Success: Grid refines 15 → 25 → 40 ppw. R(T) error vs. Fresnel computed at each resolution. Rate fitted ≥ 1.5.
Blocked By: task-102-base, task-204
Attempts: 0

**[QUEUED] task-102-weak** - Example 2 weak-scale sweep
Success: Domain expands in y/z at fixed resolution. R, T unchanged within 0.1%. `cells/s` tracked.
Blocked By: task-102-base, task-205
Attempts: 0

**[QUEUED] task-102-angular** - Example 2 angular sweep
Success: Incidence angle = 0°, 15°, 30°, 45°. s and p polarization. R matches Fresnel within 1% at finest grid for all angles.
Blocked By: task-102-base, task-204
Attempts: 0

**[QUEUED] task-102-mgpu** - Example 2 multi-GPU validation
Success: Same geometry on 2 GPUs. R, T match single-GPU within 1e-6.
Blocked By: task-102-base, task-206
Attempts: 0

---

## [QUEUED] task-103 - Example 4: Symmetry-Reduced Waveguide
Success: Example 4 baseline, convergence, symmetry sweep, and multi-GPU all pass. Symmetry-reduced result matches full-domain result at all points within numerical rounding error.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-103-base** - Example 4 baseline run
Success: Si ridge (500×220 nm), TE mode injection, symmetry=(0,-1,1). Reduced-domain result matches full-domain within rounding error.
Blocked By: task-103
Attempts: 0

**[QUEUED] task-103-conv** - Example 4 convergence sweep
Success: Grid refines 10 → 20 → 40 cells across width. Error decreases at ≥ 1.5th order.
Blocked By: task-103-base, task-204
Attempts: 0

**[QUEUED] task-103-sym** - Example 4 symmetry sweep
Success: All four symmetry combinations (PEC-PEC, PEC-PMC, PMC-PEC, PMC-PMC) tested. Each matches full-domain reference within tolerance.
Blocked By: task-103-base
Attempts: 0

**[QUEUED] task-103-weak** - Example 4 weak-scale sweep
Success: Full-domain vs. reduced at increasing sizes. Speedup ≈ 4× or 8× as expected. `cells/s` measured for both.
Blocked By: task-103-base, task-205
Attempts: 0

**[QUEUED] task-103-mgpu** - Example 4 multi-GPU with symmetry
Success: Symmetry-reduced domain chunked across 2 GPUs. Result matches single-GPU symmetry-reduced within 1e-6.
Blocked By: task-103-base, task-206
Attempts: 0

---

## [QUEUED] task-104 - Example 15: Euler Waveguide Bend
Success: Example 15 baseline, convergence, radius sweep, circular comparison, and multi-GPU all pass. Euler bend loss ~0.005 dB, circular ~0.015 dB at R=4 µm, matching Fujisawa 2017. Multi-GPU result correct.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-104-base** - Example 15 baseline run
Success: 90° Euler bend, R=4 µm effective, λ=1.55 µm. Transmitted power ≈ 1.0 (no artificial loss). Loss computed.
Blocked By: task-104
Attempts: 0

**[QUEUED] task-104-conv** - Example 15 convergence sweep
Success: Grid refines in bend region. Transmitted power converges to 1.0 at finest grid. Rate ≥ 1.5.
Blocked By: task-104-base, task-204
Attempts: 0

**[QUEUED] task-104-circular** - Example 15 circular comparison
Success: Same bend geometry with circular arc. Circular loss ~3× Euler loss as expected. Both losses match Fujisawa 2017 within 10%.
Blocked By: task-104-base
Attempts: 0

**[QUEUED] task-104-weak** - Example 15 radius sweep
Success: R = 2, 5, 10, 20 µm. Loss decreases with increasing R. Trend matches expected 1/R scaling.
Blocked By: task-104-base, task-205
Attempts: 0

**[QUEUED] task-104-mgpu** - Example 15 multi-GPU validation
Success: Euler bend chunked across 2 GPUs. Result matches single-GPU within 1e-6.
Blocked By: task-104-base, task-206
Attempts: 0

---

## [QUEUED] task-105 - Example 16: Ridge Waveguide Bragg Grating
Success: Example 16 baseline, convergence, misaligned comparison, and multi-GPU all pass. Reflection notch depth > 20 dB. Misaligned device < 0.1 dB reflection. Results converge to TMM.
Blocked By: task-201, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-105-base** - Example 16 baseline run
Success: Si ridge Bragg grating, 200 periods, aligned corrugation. Reflection notch > 20 dB at Bragg wavelength.
Blocked By: task-105
Attempts: 0

**[QUEUED] task-105-conv** - Example 16 convergence sweep
Success: Grid refines 15 → 25 → 40 ppw. R(λ) compared to TMM at each resolution. Notch depth and width converge. Rate ≥ 1.5.
Blocked By: task-105-base, task-201, task-204
Attempts: 0

**[QUEUED] task-105-misaligned** - Example 16 misaligned comparison
Success: Same grating with π-shifted corrugations. Transmission ≈ 1.0 at design λ (reflection suppressed). Difference from aligned confirms π-shift effect.
Blocked By: task-105-base
Attempts: 0

**[QUEUED] task-105-mgpu** - Example 16 multi-GPU validation
Success: Bragg grating chunked across 2 GPUs. R(λ) matches single-GPU within 1e-6.
Blocked By: task-105-base, task-206
Attempts: 0

---

## [QUEUED] task-106 - Example 17: Bent/Angled Mode Injection
Success: Example 17 baseline, convergence, angle sweep, radius sweep, and multi-GPU all pass. Transmitted power ≈ 1.0 for all bent/angled configurations. Absorber and PML produce identical results.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-106-base** - Example 17 baseline run
Success: Bent waveguide R=5 µm. Transmitted power = 1.0 ± 1e-6 (no artificial junction loss).
Blocked By: task-106
Attempts: 0

**[QUEUED] task-106-conv** - Example 17 convergence sweep
Success: Grid refines in bend region. Transmitted power converges to 1.0 at finest grid. Rate ≥ 1.5.
Blocked By: task-106-base, task-204
Attempts: 0

**[QUEUED] task-106-angle** - Example 17 angle sweep
Success: angle_theta = 0°, 15°, 30°, 45°. Power = 1.0 ± 1e-6 for all angles.
Blocked By: task-106-base
Attempts: 0

**[QUEUED] task-106-radius** - Example 17 radius sweep
Success: R = 2, 5, 10, 20 µm. Loss trend matches expected behavior.
Blocked By: task-106-base, task-205
Attempts: 0

**[QUEUED] task-106-mgpu** - Example 17 multi-GPU validation
Success: Bent waveguide chunked across 2 GPUs. Result matches single-GPU within 1e-6.
Blocked By: task-106-base, task-206
Attempts: 0

---

## [QUEUED] task-107 - Example 18: Scale-Invariant Waveguide
Success: Example 18 baseline, convergence, PML distance study, and multi-GPU all pass. n_eff matches mode solver to 4 decimal places. PML distance study shows < 0.1% loss for planes ≥ 60 µm.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-107-base** - Example 18 baseline run
Success: SiN/SiON/SiO2 scale-invariant waveguide at critical thickness. Plane size ≥ 60 µm. n_eff matches mode solver to 4 decimal places.
Blocked By: task-107
Attempts: 0

**[QUEUED] task-107-conv** - Example 18 convergence sweep
Success: Grid refines in vertical direction. n_eff error vs. mode solver decreases at ≥ 1.5th order.
Blocked By: task-107-base, task-204
Attempts: 0

**[QUEUED] task-107-pml** - Example 18 PML distance study
Success: Plane size = 10, 30, 60, 100 µm. Loss < 0.1% for planes ≥ 60 µm. Confirms LowContrastWaveguide.ipynb finding.
Blocked By: task-107-base, task-205
Attempts: 0

**[QUEUED] task-107-mgpu** - Example 18 multi-GPU validation
Success: Scale-invariant waveguide chunked across 2 GPUs. n_eff matches single-GPU within 1e-6.
Blocked By: task-107-base, task-206
Attempts: 0

---

## EXAMPLE TASKS — GROUP B (medium VRAM, Phase 1 dependencies)

## [QUEUED] task-108 - Example 3: PEC Sphere RCS
Success: Example 3 baseline, convergence, size-parameter sweep, and multi-GPU all pass. Ground truth: Mie series. σ(θ) error < 5% at finest grid for ka=1 (resonant regime).
Blocked By: task-202, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-108-base** - Example 3 baseline run
Success: PEC sphere r=1 µm, λ=1 µm, 20 ppw, domain 6r. N2F projection produces σ(θ) at φ=0. Error vs. Mie series < 5%.
Blocked By: task-108
Attempts: 0

**[QUEUED] task-108-conv** - Example 3 convergence sweep
Success: Grid refines r/10 → r/50 → r/100. σ(θ) error vs. Mie decreases at ≥ 1.5th order.
Blocked By: task-108-base, task-202, task-204
Attempts: 0

**[QUEUED] task-108-ka** - Example 3 size parameter sweep
Success: ka = 0.1, 1, 5 (Rayleigh, resonant, geometric optics). σ_tot converges to Mie at all regimes.
Blocked By: task-108-base, task-202
Attempts: 0

**[QUEUED] task-108-mgpu** - Example 3 multi-GPU validation
Success: PEC sphere with N2F chunked across 2 GPUs. σ(θ) matches single-GPU within 1e-6.
Blocked By: task-108-base, task-206
Attempts: 0

---

## [QUEUED] task-109 - Example 6: Dispersive Block (Lorentz/Oblique Incidence)
Success: Example 6 baseline, convergence (dt and dx), oblique sweep, multi-pole, and multi-GPU all pass. Ground truth: Fresnel with ε(ω). Reflectance error < 1% at finest grid. Causality preserved.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-109-base** - Example 6 baseline run
Success: Lorentz block, ε∞=2, ω₀=2π×200 THz, γ=2π×10 THz, thickness=1 µm, plane wave at 30° incidence. R matches Fresnel with ε(ω) within 2%.
Blocked By: task-109
Attempts: 0

**[QUEUED] task-109-dt** - Example 6 dt convergence sweep
Success: dt refines (Courant stays fixed at 0.99). Pulse delay and ringing error vs. analytical causal response decreases at ≥ 1.5th order.
Blocked By: task-109-base, task-204
Attempts: 0

**[QUEUED] task-109-dx** - Example 6 spatial convergence sweep
Success: Grid refines in all 3 directions. R error vs. Fresnel+ε(ω) decreases at ≥ 1.5th order.
Blocked By: task-109-base, task-204
Attempts: 0

**[QUEUED] task-109-oblique** - Example 6 oblique incidence sweep
Success: Angle = 0°, 15°, 30°, 45°. R matches Fresnel+ε(ω) within 1% at finest grid for all angles.
Blocked By: task-109-base, task-204
Attempts: 0

**[QUEUED] task-109-mgpu** - Example 6 multi-GPU validation
Success: Dispersive block chunked across 2 GPUs. R matches single-GPU within 1e-6. Causality preserved.
Blocked By: task-109-base, task-206
Attempts: 0

---

## [QUEUED] task-110 - Example 11: Multipole Expansion Convergence
Success: Example 11 baseline, convergence, and weak-scale all pass. Ground truth: Mie via PyMieScatt. Error decreases exponentially until ~r/60, then plateaus at roundoff. Convergence behavior is unambiguous.
Blocked By: task-202, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-110-base** - Example 11 baseline run
Success: Si nanosphere r=0.18 µm, λ=0.65 µm, resolution=30. Multipole coefficients p, m, Q_e, Q_m computed. Error vs. Mie < 5%.
Blocked By: task-110
Attempts: 0

**[QUEUED] task-110-conv** - Example 11 resolution convergence
Success: resolution = 10 → 30 → 60 → 100. Error vs. Mie computed at each. Exponential fit shows plateau above r/60.
Blocked By: task-110-base, task-202, task-204
Attempts: 0

**[QUEUED] task-110-mgpu** - Example 11 multi-GPU validation
Success: Nanosphere chunked across 2 GPUs. Multipole coefficients match single-GPU within 1e-6.
Blocked By: task-110-base, task-206
Attempts: 0

---

## [QUEUED] task-111 - Example 14: Nanodisk Directional Scattering
Success: Example 14 baseline, convergence, size sweep, Kerker, and multi-GPU all pass. Ground truth: Staude et al. ACS Nano 2013. Angular scattering pattern error < 5%. Magnetic dipole resonance peak identified.
Blocked By: task-202, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-111-base** - Example 14 baseline run
Success: Si nanodisk d=0.88 µm, h=0.5 µm, λ=0.88 µm. Angular scattering pattern measured. Error vs. Staude < 10%.
Blocked By: task-111
Attempts: 0

**[QUEUED] task-111-conv** - Example 14 convergence sweep
Success: dl = 0.05 → 0.02 → 0.01 µm. Pattern error vs. Staude decreases at ≥ 1.5th order.
Blocked By: task-111-base, task-202, task-204
Attempts: 0

**[QUEUED] task-111-size** - Example 14 size sweep
Success: d = 0.5, 1.0, 1.5 µm. Electric/magnetic dipole resonance crossing observed as expected.
Blocked By: task-111-base, task-202
Attempts: 0

**[QUEUED] task-111-mgpu** - Example 14 multi-GPU validation
Success: Nanodisk chunked across 2 GPUs. Pattern matches single-GPU within 1e-6.
Blocked By: task-111-base, task-206
Attempts: 0

---

## EXAMPLE TASKS — GROUP C (medium VRAM, ModeSource or N2F dependent)

## [QUEUED] task-112 - Example 7: Waveguide Mode Injection (ModeSource)
Success: Example 7 baseline, convergence, mode content, coupler coupling, and multi-GPU all pass. Mode profile matches mode solver. No spurious higher-order modes. Transmitted power ≈ 1.0. Coupling length matches TMM.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-112-base** - Example 7 baseline run
Success: Si ridge 500×220 nm, λ=1.55 µm, 20 ppw. Injected mode profile matches mode-solver field at source. Power = 1.0 ± 1e-6.
Blocked By: task-112
Attempts: 0

**[QUEUED] task-112-conv** - Example 7 cross-section convergence
Success: Grid refines 10 → 20 → 40 cells across width. Mode profile error vs. mode solver decreases. Rate ≥ 1.5.
Blocked By: task-112-base, task-204
Attempts: 0

**[QUEUED] task-112-mode** - Example 7 mode content check
Success: Higher-order modes suppressed. Only fundamental TE mode detectable above -40 dB.
Blocked By: task-112-base
Attempts: 0

**[QUEUED] task-112-coupler** - Example 7 directional coupler coupling
Success: Two parallel waveguides. Power oscillates sin²(C·z). Coupling length L_c matches TMM within 1%.
Blocked By: task-112-base, task-201
Attempts: 0

**[QUEUED] task-112-mgpu** - Example 7 multi-GPU validation
Success: Mode injection chunked across 2 GPUs. Mode profile and power match single-GPU within 1e-6.
Blocked By: task-112-base, task-206
Attempts: 0

---

## [QUEUED] task-113 - Example 8: Near-to-Far Field Projection (Zone Plate)
Success: Example 8 baseline, convergence (near-field and spectral), projection type comparison, and multi-GPU all pass. Ground truth: Rayleigh-Sommerfeld. Focal spot position and peak intensity error < 1%.
Blocked By: task-203, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-113-base** - Example 8 baseline run
Success: Zone plate TiO2/SiO2, λ=1 µm, NA=0.8. Focal spot at predicted position. Intensity error vs. Rayleigh-Sommerfeld < 3%.
Blocked By: task-113
Attempts: 0

**[QUEUED] task-113-near** - Example 8 near-field convergence
Success: Monitor grid refines (more spatial samples). Focal spot error vs. RS decreases at ≥ 1.5th order.
Blocked By: task-113-base, task-203, task-204
Attempts: 0

**[QUEUED] task-113-proj** - Example 8 projection type comparison
Success: FieldProjectionCartesianMonitor vs. FieldProjectionAngleMonitor produce equivalent far-field patterns. Difference < 1%.
Blocked By: task-113-base
Attempts: 0

**[QUEUED] task-113-mgpu** - Example 8 multi-GPU validation
Success: Zone plate with N2F chunked across 2 GPUs. Focal spot matches single-GPU within 1e-6.
Blocked By: task-113-base, task-206
Attempts: 0

---

## [QUEUED] task-114 - Example 12: MMI Power Divider (vs. Meep)
Success: Example 12 baseline, convergence, length sweep, 4-port S-matrix, and multi-GPU all pass. Ground truth: Meep FDTD. S-parameters converge to Meep within 1% at finest grid. Power conservation to within 0.001.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-114-base** - Example 12 baseline run
Success: 2×2 MMI, λ=1.55 µm. S₁₁, S₂₁, S₃₁, S₄₁ computed. Error vs. Meep < 5%.
Blocked By: task-114
Attempts: 0

**[QUEUED] task-114-conv** - Example 12 convergence sweep
Success: Grid refines. S-parameters converge to Meep. Rate ≥ 1.5 for S₂₁, S₃₁.
Blocked By: task-114-base, task-204
Attempts: 0

**[QUEUED] task-114-power** - Example 12 power conservation
Success: Σ|Sᵢⱼ|² = 1.0 ± 0.001 for all ports. Power conserved.
Blocked By: task-114-base
Attempts: 0

**[QUEUED] task-114-mgpu** - Example 12 multi-GPU validation
Success: MMI chunked across 2 GPUs. S-parameters match single-GPU within 1e-6. Power still conserved.
Blocked By: task-114-base, task-206
Attempts: 0

---

## [QUEUED] task-115 - Example 13: High-Q Silicon Metasurface
Success: Example 13 baseline, convergence (grid and run_time), buffer sweep, and multi-GPU all pass. Ground truth: Zhang et al. OL 2018. Q-factor within 5% of published. Resonance wavelength within 0.5%.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-115-base** - Example 13 baseline run
Success: Coupled silicon resonator metasurface. Q-factor and resonance wavelength extracted. Error vs. Zhang < 10%.
Blocked By: task-115
Attempts: 0

**[QUEUED] task-115-conv** - Example 13 grid convergence
Success: dl = P/16 → P/32 → P/64. Q-factor converges. Rate ≥ 1.5. Final Q within 5% of published.
Blocked By: task-115-base, task-204
Attempts: 0

**[QUEUED] task-115-time** - Example 13 run_time convergence
Success: run_time extends. Q-factor converges with longer runtime. Confirms resonance fully resolved.
Blocked By: task-115-base
Attempts: 0

**[QUEUED] task-115-mgpu** - Example 13 multi-GPU validation
Success: Metasurface chunked across 2 GPUs. Q-factor matches single-GPU within 1%.
Blocked By: task-115-base, task-206
Attempts: 0

---

## EXAMPLE TASKS — GROUP D (large VRAM: 8-16 GB, may need 2-GPU by default)

## [QUEUED] task-116 - Example 5: Multilevel Blazed Grating Efficiency
Success: Example 5 baseline, convergence (grid and layers), angle sweep, and multi-GPU all pass. Ground truth: RCWA via `grcwa`. Diffraction efficiencies converge to RCWA within 1% at finest grid.
Blocked By: task-201, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-116-base** - Example 5 baseline run
Success: Blazed grating, λ=1 µm, normal incidence, TE. Diffraction efficiencies per order computed. Error vs. RCWA < 5%.
Blocked By: task-116
Attempts: 0

**[QUEUED] task-116-conv** - Example 5 grid convergence
Success: min_steps_per_wvl = 20 → 40 → 60. Efficiencies converge to RCWA. Rate ≥ 1.5.
Blocked By: task-116-base, task-201, task-204
Attempts: 0

**[QUEUED] task-116-oblique** - Example 5 oblique incidence sweep
Success: Angle = 0°, 15°, 30°. TE/TM efficiencies vs. RCWA within 1% at finest grid.
Blocked By: task-116-base, task-201, task-204
Attempts: 0

**[QUEUED] task-116-mgpu** - Example 5 multi-GPU validation
Success: Grating chunked across 2 GPUs. Efficiencies match single-GPU within 1e-6.
Blocked By: task-116-base, task-206
Attempts: 0

---

## [QUEUED] task-117 - Example 9: Bloch Band Diagram (PhC Slab)
Success: Example 9 baseline, k convergence, grid convergence, supercell sweep, and multi-GPU all pass. Ground truth: Fan & Joannopoulos PRB 65. Band edges converge to within 0.5% of published. Bandgap location and width correct.
Blocked By: task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-117-base** - Example 9 baseline run (12 k-points)
Success: 2D PhC slab, square lattice of holes, TE-like guided mode. 12 k-points along Γ-X-M-Γ. Band edges identified. Error vs. Fan & Joannopoulos < 2%.
Blocked By: task-117
Attempts: 0

**[QUEUED] task-117-kconv** - Example 9 k-point convergence
Success: k-points = 10 → 20 → 40 per irreducible Brillouin zone. Band edges converge at ≥ 20 k-pts. Final error < 0.5%.
Blocked By: task-117-base, task-204
Attempts: 0

**[QUEUED] task-117-mgpu** - Example 9 multi-GPU validation (per k-point)
Success: Each k-point simulation chunked across 2 GPUs. Band edges match single-GPU within 0.1%.
Blocked By: task-117-base, task-206
Attempts: 0

---

## [QUEUED] task-118 - Example 10: Long Directional Coupler (Multi-Node)
Success: Example 10 baseline, convergence, weak-scale (2→4 GPUs), and multi-GPU all pass. Ground truth: TMM. Single-GPU coupling oscillation matches TMM. Multi-GPU coupling oscillation amplitude and period correct (halo correctness). Weak scaling efficiency > 0.9.
Blocked By: task-201, task-204, task-205
Attempts: 0

### Subtasks

**[QUEUED] task-118-base** - Example 10 baseline (single-GPU, short coupler)
Success: Short coupler L_coupling=10 µm. Coupling oscillation P₂(z) = P₀ sin²(C·z). C matches TMM within 1%. Single-GPU, fits on one GPU.
Blocked By: task-118
Attempts: 0

**[QUEUED] task-118-conv** - Example 10 convergence sweep
Success: Grid refines 15 → 25 → 40 ppw. Coupling length error vs. TMM decreases at ≥ 1.5th order.
Blocked By: task-118-base, task-201, task-204
Attempts: 0

**[QUEUED] task-118-weak** - Example 10 weak-scale with multi-GPU
Success: Coupling region grows 10 → 50 → 100 µm. Split across 2 → 4 GPUs. `cells/s` tracked. Weak scaling efficiency > 0.9.
Blocked By: task-118-base, task-205, task-206
Attempts: 0

**[QUEUED] task-118-halo** - Example 10 halo correctness check
Success: Long coupler (100 µm, 4 GPUs) coupling oscillation amplitude and period match single-GPU short coupler reference. Halo exchange does not damp oscillation.
Blocked By: task-118-base, task-206
Attempts: 0

---

## INTEGRATION TASKS

## [QUEUED] task-250 - Full Phase 2 suite execution and report
Success: All 18 examples run all permutations (baseline, convergence, weak-scale, multi-GPU). Clean markdown table produced: example, variant, pass/fail, error vs. ground truth, `cells/s`, multi-GPU speedup. Any failures documented with root cause.
Blocked By: tasks 101-118
Attempts: 0

## [QUEUED] task-251 - Phase 2 README update
Success: `phase2/README.md` documents: the 18-example suite, ground truth references, VRAM requirements, success criteria, hardware setup, how to run the suite, and the pass/fail summary table.
Blocked By: task-250
Attempts: 0
