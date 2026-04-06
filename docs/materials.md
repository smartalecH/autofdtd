# Materials

Phase 1 now includes the baseline isotropic material family plus the first analytical
electric-dispersive medium paths required by later compiler and runtime work.

## Supported Types

- `Medium`: homogeneous isotropic material with scalar `permittivity`,
  `conductivity`, `permeability`, and `magnetic_conductivity`
- `PECMedium`: perfect electric conductor marker
- `PMCMedium`: perfect magnetic conductor marker
- `PoleResidue`: homogeneous electric-dispersive medium with `eps_inf` plus explicit
  complex-valued pole-residue pairs stored as JSON-safe `(real, imag)` tuples
- `Sellmeier`: passive homogeneous optical-dispersion model with `(B, C)` coefficient pairs
- `Lorentz`: passive homogeneous oscillator model with `(delta_eps, frequency, damping)` triples
- `Drude`: damped plasma model with `(plasma_frequency, damping)` pairs
- `Debye`: passive homogeneous relaxation model with `(delta_eps, tau)` pairs

These models live in `autofdtd.materials`, are re-exported from `autofdtd.api`, and
lower into typed execution IR (`MediumIR`, `PECMediumIR`, `PMCMediumIR`,
`PoleResidueIR`, `SellmeierIR`, `LorentzIR`, `DrudeIR`, `DebyeIR`) rather than
remaining generic dict payloads.

## Coefficient Preparation

`autofdtd.compiler.materials.compile_isotropic_medium_coefficients()` prepares the
scalar constitutive coefficients used by the staged electric and magnetic update paths.
For a plain `Medium`, the implementation uses the standard lossy-isotropic Yee-form
update with conductivity folded into the decay and drive coefficients. `PECMedium` and
`PMCMedium` compile into explicit clamp modes so unsupported conductor behavior does not
silently look like a dielectric update.

`autofdtd.compiler.materials.compile_pole_residue_coefficients()` prepares a baseline
`eps_inf` electric update plus per-pole exact-hold recurrence terms for a complex
auxiliary polarization state. Phase 1 stores one complex polarization accumulator per
pole and emits explicit coefficient metadata rather than leaving dispersive state
implicit.

`compile_sellmeier_coefficients()`, `compile_lorentz_coefficients()`,
`compile_drude_coefficients()`, and `compile_debye_coefficients()` keep family-specific
public models and tags, but lower onto the same `PoleResidueMaterialCoefficients`
runtime contract after converting the analytical model into an equivalent pole-residue
representation. If a converted pole is too stiff for the selected timestep and would
overflow the current exact-hold recurrence path, compilation fails explicitly instead of
silently producing unusable coefficients.

## Runtime Path

`autofdtd.kernels.materials` provides the first constitutive update helpers:

- `electric_constitutive_update(...)`
- `magnetic_constitutive_update(...)`

The current implementation always exposes a NumPy execution path and reports whether the
optional Warp backend is importable. This keeps the staged constitutive split concrete
without making the package import depend on Warp being installed on every machine.

For `PoleResidue`-family paths, the runtime layer adds:

- `allocate_pole_residue_state(...)`
- `pole_residue_electric_update(...)`

This update is staged separately from the baseline magnetic path and keeps the
auxiliary-state layout explicit so later chunked runtime work can allocate, exchange,
and instrument dispersive state intentionally.
