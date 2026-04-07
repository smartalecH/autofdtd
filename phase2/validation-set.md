# Phase 2 Validation Suite — Accuracy and Feature Completeness

## Relationship to Phase 1 Validation

Phase 1 already provides 9 **functional correctness** examples in `src/autofdtd/examples/validation.py`.
These verify that the code runs without error and produces physically plausible behavior (energy
decays, fields propagate, monitors record). They are regression tests.

Phase 2 adds **accuracy validation** — a separate suite that compares FDTD results against
analytical or semi-analytical ground truth, measures convergence rates, and validates
multi-GPU halo exchange correctness.

## Overview

Phase 2 validates that the autofdtd Phase 1 implementation produces **correct and accurate**
simulation results across the feature surface implemented in Phase 1. The approach is to run
a curated set of 3D examples against reference solutions (analytical, semi-analytical, or
high-quality numerical) and measure error at multiple resolutions and domain sizes.

The suite has three complementary goals:

- **Accuracy validation** — results match ground truth within numerical error
- **Convergence testing** — error decreases at the expected 2nd-order rate with grid refinement
- **Weak-scale / multi-GPU testing** — results are robust to domain-size changes and chunk decomposition

## Design Principles

### Ground Truth Tiers

Not all examples have the same quality of reference solution. The suite is structured in two tiers:

**Tier 1 — Examples with exact analytical ground truth.** Results can be compared against closed-form formulas. Claims of accuracy are absolute, not relative.

**Tier 2 — Examples with semi-analytical or numerical reference ground truth.** Results are compared against transfer-matrix methods, mode-solver eigenfields, or established numerical benchmarks. Claims of accuracy are relative to the reference solver.

| Tier | Ground Truth | Example notebooks |
|---|---|---|
| 1 | Exact closed-form formula | Vacuum propagation, Fresnel equations, Mie series, 1D transfer-matrix |
| 2 | TMM/mode-solver/numerical benchmark | TMM for couplers, mode-solver eigenfields, published band diagrams |

### Per-Example Variants

Each example has three run variants:

| Variant | Purpose | Key metrics |
|---|---|---|
| **Baseline** | Reference correctness at a default resolution | Error vs. analytical/numerical reference |
| **Weak-scale** | Chunk/halo correctness at increased size | `cells/s` across sizes (regression check) |
| **Convergence** | Numerical accuracy vs. grid resolution | L₂ field error vs. cell count; convergence rate |

### Success Criteria

- **Weak-scale**: `cells/s` within 10% of baseline as domain grows 10× (no throughput regression)
- **Convergence**: Error decreases at ≥ 1.5th order for 2nd-order FDTD scheme; within 1% of analytical at finest resolution
- **Feature completeness**: All Phase 1 feature families (geometry, materials, sources, boundaries, monitors, grid, runtime, postprocessing) are exercised across the suite

---

## Example Catalog

### Example 1 — Focused Gaussian Beam in Vacuum (3D)

**Notebook reference**: `../tidy3d-notebooks/Simulation.ipynb` (GaussianBeam section), `../tidy3d-notebooks/GaussianBeam.ipynb`

**Phase 1 feature families tested**: Core timestepping loop (task-049), runtime metrics (task-056), GaussianBeam source, 3D field evolution

**Ground truth**: Exact analytical solution — Maxwell's equations in vacuum with a focused Gaussian beam propagating at speed `c`. The field at any point is given by the Gaussian beam waist formula: `E(r,z) = E₀ · w₀/w(z) · exp(-r²/w(z)²) · exp(-ikr²/2R(z)) · exp(i(kz - ζ(z)))` where `w(z)`, `R(z)`, `ζ(z)` are the standard beam parameters. This has known phase and amplitude variation in all three spatial dimensions.

**What it tests**: Yee update correctness in 3D with realistic field variation, Courant stability (`dt ≤ dx/(c√3)`), Gaussian beam focus and Gouy phase, field decay to machine precision in vacuum, PML absorption of outgoing waves, 3D beam spreading.

**3D status**: Truly 3D — the Gaussian beam varies in all three spatial dimensions: it has a focused waist at the origin with peak intensity on-axis and Gaussian falloff radially, and it accumulates Gouy phase near the focus. A plane wave only varies in one spatial direction; a Gaussian beam varies in all three.

**Variants**:
- Baseline: Gaussian beam, λ=1 µm, waist w₀=2 µm, propagation in x, domain 10×8×8 µm, 20 ppw
- Convergence: Grid refines 20 → 40 → 80 ppw; L₂ field error measured against analytical Gaussian beam at each point
- Weak-scale: Domain grows uniformly 10³ → 20³ → 40³ µm³; `cells/s` tracked across sizes
- Off-focus sweep: Beam waist positioned at z=0, -5, +5 µm; tests Gouy phase accumulation

**Expected convergence rate**: 2nd order (error ∝ `dx²`) away from focus; slightly degraded near focus where field gradients are steepest

---

### Example 2 — Planar Multilayer Reflectance (Fresnel)

**Notebook reference**: `../tidy3d-notebooks/Simulation.ipynb` (dielectric slab section)

**Phase 1 feature families tested**: Isotropic Medium (task-013), PlaneWave source (task-033), FieldMonitor (task-037), GridSpec (task-010), subpixel averaging (task-046)

**Ground truth**: Fresnel equations for a planar multilayer stack. Reflectance and transmittance at any angle and polarization have closed-form expressions.

**What it tests**: Isotropic material parameterization, plane-wave injection, subpixel averaging, flux recording, oblique incidence, and grid resolution sensitivity.

**3D status**: This is 1D physics (plane wave normal to infinite layers) in a 3D simulation domain. The y and z directions are uniform. This is intentional — the Fresnel equations are exact for planar layers and provide a clean numerical accuracy reference even when the simulation is 3D.

**Variants**:
- Baseline: Si slab (εᵣ=6) in vacuum, normal incidence, 20 ppw
- Convergence: Grid refines; compare R(N) and T(N) to Fresnel at each resolution
- Weak-scale: Slab same thickness, domain expands in y/z; flux at fixed monitor position
- Angular: 0°, 30°, 45° incidence; tests polarization and oblique injection

**Expected convergence rate**: 2nd order in grid; subpixel averaging should push effective error lower

---

### Example 3 — PEC Sphere Radar Cross-Section (RCS)

**Notebook reference**: `../tidy3d-notebooks/PECSphereRCS.ipynb`, `../tidy3d-notebooks/Near2FarSphereRCS.ipynb`

**Phase 1 feature families tested**: PlaneWave source, Sphere geometry, FieldMonitor, FieldProjectionCartesianMonitor (task-042), DFT (task-053), near-to-far transformation, curved PEC boundary handling

**Ground truth**: Analytical Mie series solution for scattering from a PEC sphere (exact, closed-form). Provides bistatic RCS σ(θ, φ) at any wavelength and sphere radius. Reference data: `misc/mie_bRCS_phi0_2lambda_epsr4.txt`.

**What it tests**: 3D scattering, near-to-far (N2F) transformation, curved PEC boundary discretization, plane-wave injection, complete scattering cross-section pipeline.

**Why it supersedes the TFSF approach**: The TFSF notebook uses a finite TFSF box with scatterers inside. This example uses a plane wave in a large PML-backed domain — the classic Mie scattering setup. Phase accuracy of the N2F projection is directly testable.

**Variants**:
- Baseline: PEC sphere (r=1 µm), λ=1 µm, 20 ppw, domain 6r
- Convergence: Grid refines r/10 → r/50 → r/100; compare σ(θ) to Mie series at each resolution
- Weak-scale: Sphere fixed, domain expands; N2F projection distance changes; verify far-field pattern independent of domain size
- Size parameter sweep: ka = 0.1, 1, 5 (Rayleigh, resonant, geometric optics regimes)

**Expected convergence rate**: 2nd order; curved boundary treatment is dominant error source

**Estimated VRAM**: 3–6 GB (200³ grid, complex fields)

---

### Example 4 — Symmetry-Reduced Waveguide

**Notebook reference**: `../tidy3d-notebooks/Symposium.ipynb` (full 3D symmetry section)

**Phase 1 feature families tested**: Symmetry metadata (task-020), PEC/PMC boundary behavior, chunk planning, full-domain vs. reduced-domain comparison

**Ground truth**: The full-domain FDTD result is the reference. Symmetry is an optimization, not an independent solution method — the symmetry-reduced result must match the full-domain result exactly.

**What it tests**: Symmetry metadata in Simulation, PEC (odd, -1) and PMC (even, +1) plane transforms, runtime symmetry-aware metadata, chunk decomposition with symmetry planes.

**Variants**:
- Baseline: Si ridge waveguide (500 nm × 220 nm), TE mode injection, symmetry=(0, -1, 1), full-domain reference
- Convergence: Grid refines in all three directions; error decreases at expected rate
- Weak-scale: Full-domain vs. symmetry-reduced at increasing sizes; verify speedup ≈ 4× or 8×
- Mode sweep: Verify all four symmetry combinations (TE, TM, mixed) match full-domain

**Success criterion**: |E_reduced - E_full| < numerical rounding error at all points

---

### Example 5 — Multilevel Blazed Grating Efficiency

**Notebook reference**: `../tidy3d-notebooks/GratingEfficiency.ipynb`

**Phase 1 feature families tested**: BlochBoundary (task-020), periodic geometry (task-009), PlaneWave at oblique incidence, DiffractionMonitor, FieldMonitor, grid resolution for periodic structures

**Ground truth**: Rigorous Coupled-Wave Analysis (RCWA) via `grcwa` library — semi-analytical method specialized for periodic gratings. Produces diffraction efficiencies per order as a function of incidence angle and polarization.

**What it tests**: 3D periodic boundary conditions (Bloch), oblique-incidence plane-wave injection, diffraction order tracking, material discretization at periodic features, convergence with grid refinement.

**Why it replaces the 1D PML test**: The PML/absorber reflectance test is fundamentally 1D. The grating efficiency test is genuinely 3D and exercises periodic boundaries with angled incidence — a more complete test of Bloch boundary handling.

**Variants**:
- Baseline: Multilevel blazed grating, λ=1 µm, normal incidence, TE polarization, min_steps_per_wvl=20
- Convergence (grid): Grid refines min_steps_per_wvl=20 → 40 → 60; compare diffraction efficiencies per order to RCWA
- Convergence (layers): Grating depth discretized with 10 → 20 → 40 cells/period
- Weak-scale: Domain extends in y (unbounded direction), tests PML y-boundary independence
- Oblique incidence: 0°, 15°, 30°; tests polarization and angle-dependent diffraction

**Expected result**: Diffraction efficiencies converge to RCWA reference within 1% at finest grid

**Estimated VRAM**: 8–16 GB (100M cell domain, ~3 µm × 7 µm × 2 µm with 40 layers PML)

---

### Example 6 — Dispersive Medium Block (Lorentz/Drude, 3D)

**Notebook reference**: `../tidy3d-notebooks/TFSF.ipynb` (layered substrate with dispersive media)

**Phase 1 feature families tested**: Pole-residue/dispersive medium (task-014), Lorentz/Drude/Debye (task-016), auxiliary state variables (task-050), causal pulse response, PlaneWave, FluxMonitor

**Ground truth**: Analytical frequency-domain response: `ε(ω) = ε_∞ + Σ_j (A_j·ω_j²)/(ω_j² - ω² - iγ_j·ω)`. For a single-pole Lorentz medium, the transmission coefficient through a finite block at oblique incidence has a closed-form expression in terms of the Fresnel coefficients with the complex ε(ω).

**What it tests**: Auxiliary state variable allocation and update in 3D, constitutive update kernels for dispersive media, causality preservation, material parameterization from dispersive models, oblique-incidence plane wave in dispersive medium.

**3D status**: Truly 3D — the plane wave at oblique incidence has field components in all three directions, the Lorentz update couples all six field components through the auxiliary variables, and the geometry varies in all three spatial dimensions.

**Variants**:
- Baseline: Lorentz medium block (ε∞=2, ω₀=2π×200 THz, γ=2π×10 THz), thickness=1 µm, plane wave at 30° incidence, 20 ppw
- Convergence (temporal): dt refines; error in pulse delay and ringing vs. analytical causal response
- Convergence (spatial): Grid refines in all three directions; combined spatiotemporal error
- Weak-scale: Block same thickness, domain grows in y/z; tracks auxiliary state variable growth
- Oblique incidence: 0°, 15°, 30°, 45°; tests dispersive response at increasing angles
- Multi-pole: Two-pole and three-pole dispersive models; tests general pole-residue path

**Expected result**: Field matches analytical impulse response within 2nd-order error; causality preserved (signal never arrives before t = x/c); reflectance at each angle matches Fresnel with ε(ω) within 1%

---

### Example 7 — Waveguide Mode Injection (ModeSource + ModeMonitor)

**Notebook reference**: `../tidy3d-notebooks/WaveguideCrossing.ipynb`, `../tidy3d-notebooks/DirectionalCoupler.ipynb`, `../tidy3d-notebooks/RingResonator.ipynb`

**Phase 1 feature families tested**: ModeSource (task-032), ModeMonitor (task-040), ModeSpec (task-031), ModeSolver integration, mode decomposition

**Ground truth**: The mode-solver eigenfields (`n_eff`, field profiles) serve as the reference. The FDTD propagation of the injected mode must reproduce the mode-solver profile downstream and conserve power. There is no independent analytical solution for a realistic waveguide cross-section.

**What it tests**: Mode injection from computed eigenfields, mode monitor decomposition, coupling region behavior, mode-dependent loss.

**Variants**:
- Baseline: Si ridge waveguide (500 nm × 220 nm), λ=1.55 µm, 20 ppw, straight waveguide
- Convergence: Grid refines in cross-section (10 → 20 → 40 cells across width); compare transmitted mode profile and power
- Weak-scale: Waveguide length grows 1 → 10 → 100 µm; measure mode decay vs. expected lossless propagation
- Mode content: Verify that only the fundamental TE mode is injected (higher modes suppressed)
- Coupling: Directional coupler geometry — power oscillates between two waveguides at the coupling rate; compare coupling length to TMM

**Success criterion**: Injected mode profile matches mode-solver field at source; no spurious higher-order modes; transmitted power ≈ 1 (no artificial loss)

---

### Example 8 — Near-to-Far Field Projection (Zone Plate)

**Notebook reference**: `../tidy3d-notebooks/ZonePlateFieldProjection.ipynb`

**Phase 1 feature families tested**: FieldMonitor accumulation (task-037), FieldProjectionCartesianMonitor (task-042), DFT-based projection kernels (task-053), postprocessing (task-057)

**Ground truth**: The Rayleigh-Sommerfeld diffraction integral is exact for planar apertures. For a zone plate, focal intensity and position follow from Fresnel zone theory. The `far_field_approx=False` option in Tidy3D's projection gives near-exact results that can be compared directly.

**What it tests**: FieldMonitor accumulation over time, DFT at monitor planes, projection to far-field grid, focal spot intensity and position, k-space reconstruction.

**Variants**:
- Baseline: Zone plate (TiO2 rings on SiO2 substrate), λ=1 µm, NA=0.8, focal length computed from geometry
- Convergence (near-field): Monitor grid refines (more spatial samples); focal spot error decreases
- Convergence (spectral): More frequency points in the GaussianPulse; reduces spectral ringing
- Weak-scale: Near-field monitor same size, domain grows; projection distance changes; verify far-field pattern independent of domain size
- Projection type: Compare FieldProjectionCartesianMonitor vs. FieldProjectionAngleMonitor; verify equivalence

**Expected result**: Focal spot position and peak intensity match Fresnel theory within 1%; projection error decreases with finer monitor sampling

---

### Example 9 — Bloch/Periodic Band Diagram (Photonic Crystal Waveguide)

**Notebook reference**: `../tidy3d-notebooks/BlochPhotonicCrystal.ipynb` (if it exists); otherwise use a 2D PhC slab from `../tidy3d-notebooks/PhotonicCrystalWaveguidePolarizationFilter.ipynb` or similar

**Phase 1 feature families tested**: BlochBoundary (task-020), periodic geometry (task-009), phase-aware halo exchange, k-point sampling

**Ground truth**: Analytical Bloch-Floquet dispersion for 1D multilayers. For 2D/3D photonic crystals, published dispersion data from other FDTD or eigensolver codes (Meep, MPB, COMSOL) serves as reference. The 2D photonic crystal slab is well-studied and has published band diagrams.

**What it tests**: BlochBoundary implementation, phase accumulation across periodic boundaries, k-point sampling convergence, band-edge accuracy, mode classification (TE/TM).

**Variants**:
- Baseline: 2D PhC slab (square lattice of holes in Si, air-bridge), TE-like guided mode, 15 ppw
- Convergence (k): k-point density increases 10 → 20 → 40 points per irreducible Brillouin zone; band edges converge
- Convergence (grid): Grid refines; verify bandgap edges converge to published reference
- Weak-scale: Supercell grows (single row → 4 rows → 8 rows); band diagram stable
- 3D extension: If Phase 1 supports 3D Bloch, test a 3D PhC cavity or waveguide; compare to published dispersion

**Expected result**: Band edges converge to within 0.5% of published reference; bandgap location and width match

---

### Example 10 — Long Directional Coupler (Multi-Node Scaling)

**Notebook reference**: `../tidy3d-notebooks/BroadbandDirectionalCoupler.ipynb`, `../tidy3d-notebooks/DirectionalCoupler.ipynb`

**Phase 1 feature families tested**: ModeSource, ModeMonitor, PolySlab geometry, PML, symmetry, mode decomposition, **chunk decomposition, halo exchange, multi-GPU execution** (task-054)

**Ground truth**: The transfer-matrix method (TMM) gives an exact semi-analytical prediction for the coupling vs. length relationship: `P₂(z) = P₀ sin²(C·z)` where `C` is the coupling coefficient. TMM is the reference — the FDTD must reproduce the oscillatory power transfer.

**Why this is the large/multi-node test**:
- A short coupler (10 µm coupling region) fits on one GPU
- A long coupler (100 µm coupling region) is large enough that it cannot fit on one GPU at adequate resolution — it requires chunking
- The coupling oscillation is a **global observable** — if halos are wrong, the oscillation amplitude and period will be wrong even if individual steps look correct
- The oscillatory output is sensitive to phase errors from incorrect halo exchange

**Variants**:
- Baseline (single-GPU): Short coupler (L_coupling = 10 µm), one chunk, validated against TMM
- Weak-scale (multi-node): Coupling region grows 10 → 50 → 100 µm; split across 2 → 4 GPUs; `cells/s` and time-per-step measured at each scale
- Convergence: Grid refines 15 → 25 → 40 ppw; error vs. TMM decreases at 2nd order
- Angular sweep: Test both TE and TM polarizations; coupling coefficient differs
- Multi-port: Extended to 4-port directional coupler ( BroadbandDirectionalCoupler geometry) — tests multi-port monitoring

**What it specifically tests for multi-node**:
- Chunk decomposition splits the domain at a coupling-region boundary
- Halo exchange must correctly transfer field values across chunk boundaries
- Phase coherence across chunks: coupling oscillation amplitude must not be damped by halo artifacts
- Scaling: `cells/s` should be constant (weak scaling) as domain grows and chunks increase

**Success criteria**:
- Single-GPU result matches TMM at baseline resolution
- Multi-GPU result matches single-GPU result (within numerical tolerance) — validates halo exchange
- `cells/s` constant across chunk counts (weak scaling efficiency > 0.9)

---

### Example 11 — Multipole Expansion Convergence (Si Nanosphere)

**Notebook reference**: `../tidy3d-notebooks/MultipoleExpansion.ipynb`

**Phase 1 feature families tested**: Sphere geometry, dispersive (Lorentz/PoleResidue), PlaneWave, FieldMonitor, volume integration for multipole decomposition, material permittivity handling

**Ground truth**: Analytical Mie theory via PyMieScatt. The multipole decomposition provides electric dipole (p), magnetic dipole (m), electric quadrupole (Q_e), magnetic quadrupole (Q_m) contributions to scattering.

**What it tests**: Convergence study with exponential behavior — the key insight from the notebook is that multipole errors plateau above resolution ~60 pts/radius. This makes it an excellent **convergence validation** because the plateau identifies when discretization error becomes negligible vs. when it's dominant. Tests volume integration accuracy and material averaging at curved interfaces.

**Reference data files**: `misc/mie_electric_dipole`, `misc/mie_magnetic_dipole`, `misc/mie_electric_quadrupole`, `misc/mie_magnetic_quadrupole`

**Variants**:
- Baseline: Si nanosphere (r=0.18 µm), λ=0.65 µm, resolution=30 (r/30)
- Convergence: resolution = 10 → 30 → 60 → 100; measures convergence of each multipole coefficient
- Exponential fit: Fit error vs. resolution; demonstrates exponential until discretization floors, then plateaus
- Weak-scale: Fixed resolution, sphere size grows; measures how scattering efficiency changes with size parameter

**Expected result**: Error decreases exponentially until ~r/60, then plateaus at roundoff. This makes convergence rate measurement unambiguous — no fitting ambiguity about whether 2nd order has been reached.

**Estimated VRAM**: 2–4 GB

---

### Example 12 — MMI Power Divider (vs. Meep)

**Notebook reference**: `../tidy3d-notebooks/MMIMeepBenchmark.ipynb`

**Phase 1 feature families tested**: PolySlab geometry, ModeSource, ModeMonitor, multi-port devices, waveguide mode propagation

**Ground truth**: Meep FDTD — an independent, well-validated open-source FDTD solver. Since no closed-form analytical solution exists for an MMI device, cross-validation against a trusted solver is the gold standard.

**What it tests**: Complex 3D integrated photonics device with multiple waveguide ports, multimode interference physics, mode injection and decomposition accuracy. Tests that autofdtd results are consistent with Meep, not just "physically plausible."

**Why Meep as reference**: Meep has been validated against analytical solutions, published benchmarks, and used by thousands of researchers. Any systematic discrepancy between autofdtd and Meep indicates a bug in autofdtd.

**Variants**:
- Baseline: 2×2 MMI power divider, λ=1.55 µm, min_steps_per_wvl=13, Meep resolution=30 steps/µm
- Convergence: Grid refines; compare S-parameters (S₁₁, S₂₁, S₃₁, S₄₁) vs. Meep at each resolution
- Weak-scale: MMI length sweeps (40 → 80 → 160 µm); measures power splitting convergence
- Multi-port: Full 4-port S-matrix extraction; power conservation check (sum of transmitted powers = 1)

**Expected result**: S-parameters converge to Meep reference within 0.01 (1%) at finest resolution; power conservation to within 0.001

**Estimated VRAM**: 4–8 GB

---

### Example 13 — High-Q Silicon Metasurface (Q-Factor Accuracy)

**Notebook reference**: `../tidy3d-notebooks/HighQSi.ipynb`

**Phase 1 feature families tested**: Resonant cavity physics, broadband pulse → narrowband resonance, transmission spectrum extraction, Q-factor measurement, periodic boundary conditions, high-Q cavity convergence

**Ground truth**: Zhang et al. Optics Letters 43, 1842-1845 (2018) — published Q-factor measurements and resonance positions for coupled silicon resonator metasurfaces.

**What it tests**: Resonance linewidth accuracy, Q-factor extraction from transmission spectrum, high-Q cavity convergence behavior. High-Q resonances require long run times and fine grid resolution; the Q-factor itself is sensitive to discretization error and PML loss.

**Why it matters for Phase 2**: Q-factor error directly maps to energy loss error — a Q that is too high means the FDTD is artificially lossless; a Q that is too low means artificial dissipation. This is a direct physical accuracy metric.

**Variants**:
- Baseline: Coupled silicon resonator metasurface, λ₀ ≈ 1.3–1.6 µm, dl = P/32 (~20 nm), periodic in x,y, PML in z
- Convergence (grid): dl = P/16 → P/32 → P/64; compare extracted Q-factor and resonance wavelength to published
- Convergence (run_time): run_time extends to fully resolve resonance linewidth; Q-factor converges with longer run_time
- Weak-scale: Domain size in z grows (buffer between resonator and PML); verifies Q is independent of buffer thickness

**Expected result**: Q-factor converges to within 5% of published value at finest grid; resonance wavelength to within 0.5%

**Estimated VRAM**: 4–8 GB

---

### Example 14 — Silicon Nanodisk Directional Scattering

**Notebook reference**: `../tidy3d-notebooks/DirectionalScatteringNanodisks.ipynb`

**Phase 1 feature families tested**: Multipole decomposition, disk/sphere geometry, PlaneWave at normal incidence, FieldMonitor for far-field projection, magnetic dipole resonance, back-scattering suppression physics

**Ground truth**: Staude et al., ACS Nano (2013) — multipole decomposition results showing electric and magnetic dipole resonances in silicon nanodisks and their directional (angular) scattering patterns.

**What it tests**: Angular scattering pattern accuracy, magnetic dipole resonance detection, back-scattering suppression (Kerker effect). This goes beyond scalar scattering cross-section to test **directional** accuracy — the angular dependence of scattered light.

**Why it complements the PEC Sphere**: The PEC Sphere tests total scattering efficiency. The nanodisk tests the *angular distribution* of that scattering, which is much more sensitive to geometry discretization and material averaging.

**Variants**:
- Baseline: Silicon nanodisk (diameter=0.88 µm, height=0.5 µm), λ=0.88 µm, min_steps_per_wvl=15, mesh override dl=0.02 µm
- Convergence: dl=0.05 → 0.02 → 0.01 µm; compare angular scattering pattern to published multipole decomposition
- Size sweep: Disk diameter sweeps 0.5 → 1.0 → 1.5 µm; tracks electric/magnetic dipole resonance crossing
- Back-scattering: Kerker suppression condition at specific λ; verifies directional scattering physics

**Expected result**: Angular scattering pattern matches published within 5% at finest resolution; magnetic dipole resonance peak identified correctly

**Estimated VRAM**: 2–4 GB

---

### Example 15 — Euler Waveguide Bend (Bent Mode Injection)

**Notebook reference**: `../tidy3d-notebooks/EulerWaveguideBend.ipynb`

**Phase 1 feature families tested**: Bent waveguide mode injection (`bend_radius`, `bend_axis` parameters), ModeSource, ModeMonitor, Absorber boundary (not PML — needed for bent geometries), clothoid/Euler curve geometry

**Ground truth**: Fujisawa et al. "Efficient and compact athermal laser diode with novel stripe geometry," OE 25, 9170 (2017). The paper gives measured bend loss for Euler bends vs. circular bends at 1550 nm.

**What it tests**: Bent waveguide mode injection accuracy. An Euler (clothoid) bend has continuous curvature — it transitions smoothly from straight to curved with no abrupt curvature at the input/output joints. This tests whether the `bend_radius` mode injection correctly accounts for the bent geometry field profile and whether the bent field propagates with the correct phase accumulation. Also tests that Absorber boundaries work correctly for non-PML-friendly geometries.

**Why bent waveguides matter**: Any integrated photonics chip with routing has bends. Incorrect bend treatment causes artificial radiation loss or phase errors that corrupt downstream devices (couplers, resonators).

**Variants**:
- Baseline: 90° Euler bend vs. 90° circular bend (same bend radius ~4 µm effective), λ=1.55 µm, 20 ppw
- Convergence: Grid refines in bend region (10 → 20 → 40 cells across bend width); compare transmitted power and phase
- Weak-scale: Bend radius grows 2 → 6 → 10 µm; measures bend loss scaling with radius
- Circular comparison: Run same bend with circular geometry; verify Euler has lower loss than circular (as expected from theory)

**Expected result**: Euler bend loss ~0.005 dB, circular bend loss ~0.015 dB at R=4 µm; FDTD reproduces this ratio within 10%

**Estimated VRAM**: <1 GB (small waveguide cross-section, 90° bend footprint ~10 µm × 10 µm)

---

### Example 16 — Ridge Waveguide Bragg Grating (3D Distributed Reflection)

**Notebook reference**: `../tidy3d-notebooks/BraggGratings.ipynb`

**Phase 1 feature families tested**: BlochBoundary (task-020), periodic geometry, distributed feedback, ModeSource, FluxMonitor, reflection/transmission spectrum, PolySlab geometry

**Ground truth**: Xu Wang et al. "Precise control of the coupling coefficient through destructive interference in silicon waveguide Bragg gratings," Opt. Lett. 39, 5519-5522 (2014). The aligned corrugation design gives a strong reflection notch; the misaligned design cancels the reflection. The TMM for a waveguide Bragg grating gives the reflection coefficient as a function of detuning.

**What it tests**: Distributed reflection from a periodic perturbation in a 3D ridge waveguide geometry. The ridge waveguide (Si on SiO₂) has a finite cross-section — the mode profile in y and z is non-uniform, and the Bragg corrugation modulates the effective index along x. This gives genuine 3D field variation: the mode field profile (Ey, Hx, Hz components) varies in y and z while the Bloch wave varies in x. The 3D geometry also introduces coupling between the fundamental mode and higher-order modes at structural discontinuities.

**Why it is truly 3D**: Unlike a planar Bragg grating (uniform in y), a ridge waveguide Bragg grating has a non-uniform cross-section where the mode field varies in y (width) and z (height). The corrugation causes coupling between the forward and backward propagating fundamental modes, but the 3D field profile means the effective coupling coefficient depends on the overlap integral of the mode field with the corrugated region — a genuinely 3D effect.

**Why it complements Example 9 (PhC band diagram)**: Example 9 tests Bloch boundaries in a 2D PhC slab (in-plane variation). Example 16 tests Bloch boundaries in a 3D waveguide geometry where the cross-section matters for the coupling coefficient — it isolates Bloch boundary correctness while testing a richer geometry.

**Variants**:
- Baseline: Si ridge waveguide (w=500 nm, h=220 nm) on SiO₂, period=0.324 µm, 200 periods, λ_B≈1550 nm, aligned corrugation, 20 ppw
- Convergence: Grid refines 15 → 25 → 40 ppw; compare reflection notch depth and width to TMM
- Misaligned comparison: Same grating with π-shifted corrugations; transmission should be near-1 at design λ
- Bandwidth sweep: λ sweeps 1.5–1.6 µm; compare full R(λ), T(λ) spectrum to TMM
- Corrugation depth: Depth = 5 → 10 → 20 nm; coupling coefficient scales with depth², tests coupling strength scaling

**Expected result**: Reflection notch depth > 20 dB at Bragg wavelength for optimal depth; misaligned device has < 0.1 dB reflection; results converge to TMM at finest grid

**Estimated VRAM**: <1 GB (ridge cross-section 0.5×0.22 µm², length ~65 µm with 200 periods)

---

### Example 17 — Bent and Angled Waveguide Mode Injection

**Notebook reference**: `../tidy3d-notebooks/ModesBentAngled.ipynb`

**Phase 1 feature families tested**: Bent mode injection (`bend_radius`, `bend_axis`), angled mode injection (`angle_theta`, `angle_phi`), ModeSource, Absorber boundaries, composition of bend + angle

**Ground truth**: Tidy3D documentation and internal consistency (transmission should equal input power for lossless bent/angled waveguide). No analytical ground truth for complex bent geometries — validates that bent mode injection is correctly implemented.

**What it tests**: The `bend_radius` and `angle_theta`/`angle_phi` parameters in ModeSource. Bent waveguides are ubiquitous in photonics chips. The mode profile in a bent waveguide differs from a straight waveguide (the field is displaced toward the outer wall), and the propagation constant includes a bend-induced correction. Incorrect bent mode injection causes artificial loss at the input junction. Also tests Absorber boundaries as an alternative to PML for geometries where PML is unsuitable (e.g., angled outputs).

**Why it matters**: Example 15 tests loss in Euler vs circular bends. Example 17 tests the more general case: arbitrary `bend_radius`, `bend_axis`, `angle_theta`, `angle_phi`, and their composition. It validates that the ModeSource correctly computes bent/angled eigenmodes and injects them without artificial junction loss.

**Variants**:
- Baseline: Bent waveguide with R=5 µm, straight input/output, λ=1.55 µm, 20 ppw, transmitted power measured
- Convergence: Grid refines in bend region; verify transmitted power converges to 1.0 (no artificial loss)
- Angle sweep: `angle_theta` = 0°, 15°, 30°, 45°; verify power conservation at each angle
- Bend radius sweep: R = 2, 5, 10, 20 µm; compare loss scaling to expected trend
- Bend + angle composition: Bent waveguide at angle; verify both effects compose correctly
- Absorber vs PML: Same geometry with Absorber vs PML boundaries; verify same result

**Expected result**: Transmitted power ≈ 1.0 for all bent/angled configurations (no artificial junction loss); Absorber and PML produce identical results

**Estimated VRAM**: <1 GB

---

### Example 18 — Scale-Invariant Waveguide

**Notebook reference**: `../tidy3d-notebooks/ScaleInvariantWaveguide.ipynb`

**Phase 1 feature families tested**: Low-index-contrast waveguide (SiN/SiON/SiO2), ModeSource, ModeMonitor, mode solver integration, evanescent field sensitivity, dispersive material fitting

**Ground truth**: Rodrigues et al. "All-dielectric scale invariant waveguide," Nat Commun 14, 6675 (2023). The paper demonstrates that at a specific core thickness, the effective index of a three-layer waveguide becomes nearly independent of core thickness — the guiding becomes "scale-invariant."

**What it tests**: Low-index-contrast waveguide physics. Unlike high-contrast Si/SiO2 (n=3.5/1.44), SiN/SiON/SiO2 waveguides have moderate index contrast (n≈2.0/1.76/1.44). The evanescent fields extend further, making them more sensitive to PML proximity and mesh quality. The scale-invariant condition tests whether the mode solver and FDTD correctly handle weak confinement and long evanescent tails. Also tests that dispersive material models (SiON has dispersion) are correctly handled.

**Why it complements Examples 7 and 15**: Examples 7 (straight ridge) and 15 (high-contrast bent) test high-index-contrast waveguides. This tests the opposite regime: low-index-contrast waveguides where evanescent fields dominate and PML interaction with evanescent fields can cause unphysical loss (as demonstrated in `LowContrastWaveguide.ipynb`).

**Important note**: The `LowContrastWaveguide.ipynb` demonstrates that small simulation plane sizes cause artificial loss in low-contrast waveguides because PML absorbs the evanescent tail. This example requires adequate plane size (≥ 60 µm) to avoid this artifact.

**Variants**:
- Baseline: SiN/SiON/SiO2 three-layer waveguide at scale-invariant condition, λ=1.55 µm, plane size 60 µm, 20 ppw
- Convergence: Grid refines (especially in vertical direction where evanescent tails live); compare n_eff to mode solver
- PML distance study: Plane size 10 → 30 → 60 → 100 µm; demonstrates artificial loss at small sizes
- Scale-invariant condition sweep: Core thickness varies around the critical point; n_eff should be nearly constant
- Comparison to regular strip: Same nominal mode area but regular strip waveguide shows strong dispersion (demonstrates the scale-invariant behavior)

**Expected result**: n_eff at scale-invariant condition matches mode solver to 4 decimal places; PML distance study shows <0.1% loss for planes ≥ 60 µm

**Estimated VRAM**: 1–2 GB

---

## Phase 1 Baseline

Phase 1 already provides:

| Capability | Location | Status |
|---|---|---|
| 9 functional correctness examples | `src/autofdtd/examples/validation.py` | Done |
| GPU execution (Warp, persistent arrays) | `autofdtd.kernels`, `autofdtd.runtime` | Done |
| GPU benchmarking (single/multi-GPU, cells/s) | `src/autofdtd/benchmarks/gpu_benchmark.py` | Done |
| 2-GPU chunk decomposition | task-063 | Done |
| Cross-device halo exchange | task-064 | Done |
| Runtime metrics (cells/s, timing) | `ExecutionResult.metrics` | Done |

Phase 2 adds on top of this:

1. **Ground truth comparison** — error quantification vs. analytical/semi-analytical reference for each example
2. **Convergence rate measurement** — L₂ field error vs. cell count, verifying 2nd-order convergence
3. **Weak-scale testing** — throughput (cells/s) vs. domain size, verifying no throughput regression
4. **Multi-GPU halo correctness** — cross-chunk field agreement at chunk boundaries (not just throughput)
5. **TMM/mode-solver reference implementations** — to compute reference reflectance/coupling values

## Feature Coverage Matrix

| Phase 1 Feature | Examples Exercising It |
|---|---|
| Geometry: Box | 2, 7, 10 |
| Geometry: Sphere | 3, 11 |
| Geometry: Cylinder | 3 (via ring), 14 (nanodisk) |
| Geometry: PolySlab | 7, 10, 12 |
| Geometry: GeometryGroup / ClipOperation | 10, 12 |
| Material: Medium (isotropic) | 1, 2, 3, 7, 10, 12, 13, 14, 15, 16, 17, 18 |
| Material: PECMedium / PMCMedium | 4 |
| Material: Dispersive (PoleResidue, Lorentz, Drude) | 6, 11, 18 |
| Source: PlaneWave | 1, 2, 3, 5, 14, 16 |
| Source: GaussianPulse | 1, 2, 3, 6, 7, 8, 10, 11, 13 |
| Source: ModeSource | 7, 10, 12, 15, 16, 17, 18 |
| Source: TFSF | 3 |
| Boundary: PML | 1, 2, 3, 7, 10, 13, 14, 18 |
| Boundary: Absorber/StablePML | 5, 15, 16, 17 |
| Boundary: BlochBoundary | 5, 9, 13, 16 |
| Boundary: Symmetry (PEC/PMC) | 4, 7, 10 |
| Monitor: FieldMonitor | 1, 2, 3, 8, 11, 13, 14 |
| Monitor: FluxMonitor | 2, 3, 10, 12, 16 |
| Monitor: ModeMonitor | 7, 10, 12, 15, 17, 18 |
| Monitor: FieldProjection* | 3, 8, 14 |
| Grid: AutoGrid / GridSpec | 1, 2, 3, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18 |
| Grid: MeshOverrideStructure | 3, 5, 7, 10, 11, 12, 14, 18 |
| Runtime: Chunk decomposition | 4, 9, 10 |
| Runtime: Halo exchange | 9, 10 |
| Runtime: Metrics (cells/s, timing) | 1, 4, 9, 10 |
| Postprocessing: DFT / N2F | 3, 8, 14 |
| Scattering: multipole decomposition | 11, 14 |
| Resonance: Q-factor extraction | 13 |
| Bent/angled mode injection | 15, 17 |
| Bloch distributed reflection | 16 |
| Low-index-contrast / evanescent field handling | 18 |

---

## Cost Estimates (Node-Hours)

Assuming running on a single NVIDIA A100 (≈ 10² cells/s per Wfps for 3D FDTD). Costs assume 2-GPU execution for examples >8GB.

| Example | Baseline | Convergence (3 pts) | Weak-scale (3 pts) | Total runs | Est. node-hrs |
|---|---|---|---|---|---|
| 1. Vacuum | 0.01 | 0.03 | 0.03 | 7 | ~0.1 |
| 2. Fresnel slab | 0.05 | 0.15 | 0.15 | 8 | ~0.4 |
| 3. PEC Sphere RCS | 0.5 | 1.5 | 1.5 | 12 | ~4 |
| 4. Symmetry | 0.05 | 0.15 | 0.15 | 8 | ~0.4 |
| 5. Grating Efficiency | 0.5 | 1.5 | 1.5 | 10 | ~4 |
| 6. Dispersive | 0.5 | 1.5 | 1.5 | 10 | ~4 |
| 7. Waveguide mode | 0.1 | 0.3 | 0.3 | 8 | ~0.7 |
| 8. Near-to-far | 0.5 | 1.5 | 1.5 | 10 | ~4 |
| 9. Bloch band diagram | 1.0 | 3.0 | 3.0 | 12 | ~8 |
| 10. Long coupler (multi-node) | 0.5 | 1.5 | 50+ | 10 | ~55 |
| 11. Multipole expansion | 0.2 | 0.6 | 0.6 | 10 | ~1.5 |
| 12. MMI vs Meep | 0.5 | 1.5 | 1.5 | 10 | ~4 |
| 13. High-Q metasurface | 0.5 | 1.5 | 1.5 | 10 | ~4 |
| 14. Nanodisk scattering | 0.2 | 0.6 | 0.6 | 10 | ~1.5 |
| 15. Euler bend | 0.05 | 0.15 | 0.15 | 8 | ~0.4 |
| 16. Bragg gratings | 0.1 | 0.3 | 0.3 | 10 | ~0.7 |
| 17. Bent/angled mode injection | 0.05 | 0.15 | 0.15 | 10 | ~0.4 |
| 18. Scale-invariant waveguide | 0.1 | 0.3 | 0.3 | 10 | ~0.7 |
| **Total** | | | | **~163 runs** | **~94 node-hrs** |

The long directional coupler (Example 10) dominates the multi-node cost. All other examples are relatively cheap — the entire suite except Example 10 runs in under 40 node-hours combined.

**Note**: These are estimates. Actual cost depends on cell count, resolution, and number of timesteps. Examples with early shutoff (directional coupler, MMI) will be significantly cheaper than estimated.

---

## Tier Summary

| Tier | Examples | Ground Truth Type |
|---|---|---|
| 1 | 1 (vacuum), 2 (Fresnel), 3 (PEC sphere Mie), 5 (grating RCWA), 11 (multipole Mie), 16 (Bragg grating TMM) | Exact closed-form formulas or established semi-analytical (RCWA/TMM) |
| 2 | 4 (symmetry vs. full-domain), 6 (dispersive vs. analytical ε(ω)), 7 (mode vs. mode-solver), 8 (near-to-far vs. Rayleigh-Sommerfeld), 9 (band diagram vs. published), 10 (coupler vs. TMM), 12 (MMI vs. Meep), 13 (Q-factor vs. published), 14 (nanodisk vs. published multipole), 15 (bend loss vs. Fujisawa 2017), 17 (bent/angled mode injection), 18 (scale-invariant vs. Rodrigues 2023) | Semi-analytical or reference-solver comparison |

---

## Notebook Source Index

| Example | Primary notebook | Key cells used |
|---|---|---|
| 1 | `../tidy3d-notebooks/Simulation.ipynb` | Cells 1–15 (basic simulation, plane wave, dielectric slab) |
| 2 | `../tidy3d-notebooks/Simulation.ipynb` | Cells 1–15 (full example) |
| 3 | `../tidy3d-notebooks/PECSphereRCS.ipynb`, `../tidy3d-notebooks/Near2FarSphereRCS.ipynb` | Mie series setup + N2F projection |
| 4 | `../tidy3d-notebooks/Symmetry.ipynb` | Cells 1–55 (waveguide symmetry) |
| 5 | `../tidy3d-notebooks/GratingEfficiency.ipynb` | Full notebook (RCWA comparison) |
| 6 | `../tidy3d-notebooks/TFSF.ipynb` | Cells 28–36 (layered substrate with dispersive media) |
| 7 | `../tidy3d-notebooks/WaveguideCrossing.ipynb` | Full notebook (ModeSource, ModeMonitor, PolySlab) |
| 8 | `../tidy3d-notebooks/ZonePlateFieldProjection.ipynb` | Full notebook (field projection) |
| 9 | `../tidy3d-notebooks/Bandstructure.ipynb` | Full notebook (Bloch boundaries, k-point sweep) |
| 10 | `../tidy3d-notebooks/BroadbandDirectionalCoupler.ipynb` | Full geometry scaled to long coupling region |
| 11 | `../tidy3d-notebooks/MultipoleExpansion.ipynb` | Full notebook (multipole decomposition, convergence) |
| 12 | `../tidy3d-notebooks/MMIMeepBenchmark.ipynb` | Full notebook (Meep cross-validation) |
| 13 | `../tidy3d-notebooks/HighQSi.ipynb` | Full notebook (Q-factor extraction) |
| 14 | `../tidy3d-notebooks/DirectionalScatteringNanodisks.ipynb` | Full notebook (angular scattering pattern) |
| 15 | `../tidy3d-notebooks/EulerWaveguideBend.ipynb` | Full notebook (Euler vs circular bend loss) |
| 16 | `../tidy3d-notebooks/BraggGratings.ipynb` | Full notebook (aligned vs misaligned Bragg gratings) |
| 17 | `../tidy3d-notebooks/ModesBentAngled.ipynb` | Full notebook (bent/angled mode injection) |
| 18 | `../tidy3d-notebooks/ScaleInvariantWaveguide.ipynb` | Full notebook (scale-invariant guiding, evanescent fields) |

---

## Open Questions

- **Example 9 (Bloch band diagram)**: `Bandstructure.ipynb` provides a good 3D PhC slab example. Need to verify it runs in acceptable VRAM (5–10 GB per k-point, 12 simulations total).

- **Example 3 (PEC Sphere RCS)**: The N2F transformation in Phase 1 (task-042, task-053) must be verified to produce correct far-field patterns before this example can be used. The near-to-far projection uses DFT at monitor planes — the Rayleigh-Sommerfeld integration must be correct.

- **Example 10 (multi-node)**: Phase 1 task-064 implements cross-device halo exchange. The long coupler (Example 10) validates halo correctness — the oscillatory coupling output is sensitive to phase errors from incorrect halos. Chunk decomposition along the propagation direction with coupling-region boundaries as chunk seams is the strategy.

- **Example 5 (Grating Efficiency)**: The `grcwa` RCWA library is a Python dependency. Need to verify it is available in the Phase 2 test environment, or implement a minimal RCWA reference.

- **Example 12 (MMI vs Meep)**: Meep must be installed and accessible in the Phase 2 test environment. Without it, this example becomes an MMI TMM comparison (which does not exist as a closed form), limiting it to Meep-only validation.

- **Example 17 (Bent/angled mode injection)**: The `bend_radius`, `bend_axis`, `angle_theta`, `angle_phi` parameters in ModeSource must be wired into the timestepping loop. Bent mode injection requires the ModeSource to compute bent eigenfields — this is a ModeSource integration task beyond what Example 7 already defers.

- **Example 18 (Scale-invariant waveguide)**: Requires adequate plane size (≥60 µm) to avoid artificial PML-evanescent interaction. The `LowContrastWaveguide.ipynb` demonstrates this failure mode. Phase 2 tests must enforce this minimum plane size.

- **VRAM scaling**: All examples should be validated to fit within 16 GB per GPU. Examples 5 (grating), 9 (band structure), and 10 (long coupler at 100 µm) are the highest-risk and may need 2-GPU execution by default. Examples 15–18 are all small (<1–2 GB).

- **Phase 1 existing examples overlap**: `src/autofdtd/examples/validation.py` already covers vacuum point source, plane wave, dielectric slab, PML absorption, source injection, monitor recording, and convergence shutoff. Phase 2 should not re-implement these as ground-truth tests — it extends them with ground-truth comparison, convergence measurement, and weak-scale variants.
