# Mathematical Kernel Spec

## Purpose

This document defines the discrete math that the generated implementation must realize.

It is intentionally closer to an operator specification than to any one framework.

Primary local references:

- `phd_thesis/chapters/chapter3.tex`
- `meep/src/step.cpp`
- `meep/src/step_generic.cpp`
- `meep/src/step_db.cpp`
- `meep/src/update_eh.cpp`
- `meep/src/dft.cpp`
- `meep/src/near2far.cpp`
- `meep/src/meepgeom.cpp`
- `meep/python/adjoint/objective.py`
- `meep/python/adjoint/filter_source.py`

## Notation

Fields:

- `E, D, H, B` are Yee-grid fields
- hats such as `\hat D(ω)` denote DTFT/frequency-domain quantities
- superscripts `f` and `a` denote forward and adjoint fields

Indices:

- `n` is the discrete timestep
- `i,j,k` are Yee-grid indices
- `c` is a field component
- `ω_m` is one of the requested design frequencies

Operators:

- `[\nabla^+ \times]` is the forward-difference curl
- `[\nabla^- \times]` is the backward-difference curl
- `I_*` denotes interpolation
- `R_* = I_*^T` denotes restriction/backprop to the original Yee nodes

## 1. Core Yee Curl Updates

The plain Cartesian Yee updates are:

```math
B^{n+\frac12} = B^{n-\frac12} - \Delta t \, [\nabla^+ \times] E^n + \Delta t\,K_B^n
```

```math
D^{n+1} = D^{n} + \Delta t \, [\nabla^- \times] H^{n+\frac12} - \Delta t\,J_D^{n+\frac12}
```

with the component curls

```math
(\nabla \times A)_x = \partial_y A_z - \partial_z A_y
```

```math
(\nabla \times A)_y = \partial_z A_x - \partial_x A_z
```

```math
(\nabla \times A)_z = \partial_x A_y - \partial_y A_x
```

and the discrete forward/backward derivatives chosen so that

```math
[\nabla^+ \times]^T = [\nabla^- \times], \qquad
[\nabla^- \times]^T = [\nabla^+ \times].
```

This transpose pairing is part of the adjoint contract and must be preserved exactly.

## 2. Conductivity Update Variant

When a component `f` obeys

```math
\frac{df}{dt} = \operatorname{curl}(g) - \sigma f,
```

Meep uses the centered update

```math
f^{n+1} =
\frac{\left(1-\frac{\Delta t}{2}\sigma\right)f^n + \Delta t\,\operatorname{curl}(g)}
     {1+\frac{\Delta t}{2}\sigma}.
```

This is the exact discrete pattern visible in `meep/src/step_generic.cpp`.

For generation purposes define

```math
c_\sigma^- = 1-\frac{\Delta t}{2}\sigma, \qquad
c_\sigma^+ = 1+\frac{\Delta t}{2}\sigma,
```

then

```math
f^{n+1} = \frac{c_\sigma^- f^n + \Delta t\,\operatorname{curl}(g)}{c_\sigma^+}.
```

## 3. PML / Auxiliary-Field Variant

The kernel family must support three nested auxiliary states:

- conductivity auxiliary `C`
- first PML auxiliary `U`
- constitutive-side PML auxiliary `W`

The operational equations reflected in Meep/Khronos are:

```math
\frac{dC_k}{dt} + \sigma_D C_k = K_k
```

```math
\frac{dU_k}{dt} + \sigma_{k+1} U_k = \frac{dC_k}{dt}
```

```math
\frac{dT_k}{dt} + \sigma_{k-1} T_k = \frac{dU_k}{dt}
```

where `T_k` is the target `B_k` or `D_k` component and `K_k` is the corresponding curl contribution.

The important generation rule is not the variable names but the dependency chain:

- no PML, no conductivity: update target field directly from the curl
- conductivity only: update target via centered conductivity damping
- one PML direction: update target through one auxiliary state
- two PML/conductivity layers: update through `C -> U -> T`

That staged form is the right abstraction to preserve instead of copying the hand-expanded branch tree.

## 4. Cylindrical and Beta Variants

The generator must treat these as operator modifiers, not separate solvers.

### 4.1 2D beta modifier

For 2D simulations with a phase factor `e^{i\beta z}`, additional coupling terms appear in the curl updates. Meep implements them as separate `beta` corrections in `step_db.cpp`.

Generation rule:

- emit the base 2D Yee update
- then emit the `beta` coupling correction as a distinct stage or fused variant

### 4.2 Cylindrical modifier

For cylindrical coordinates, `m/r` terms modify the curl updates and require a special treatment near `r=0`.

Generation rules:

- treat cylindrical coordinates as a modifier on the derivative operator
- preserve the `r=0` special case
- preserve the distinction between the ordinary curl terms and the extra `m/r` terms

## 5. Constitutive Update

Meep advances `D/B` first and then computes `E/H` by applying inverse material operators.

In the simplest isotropic case:

```math
E = \varepsilon^{-1} D, \qquad H = \mu^{-1} B.
```

With conductivity/source/polarization corrections, the effective constitutive input is

```math
\widetilde D = D - P - P_{\text{int-src}},
```

or the magnetic analogue, before the inverse material map is applied.

In PML regions the constitutive-side auxiliary `W` obeys the discrete pattern visible in `Khronos.jl/src/Timestep.jl`:

```math
W^{n+1} = m^{-1}\big(T + S + P\big),
```

```math
A^{n+1} = A^n + (1+\sigma)W^{n+1} - (1-\sigma)W^n,
```

where:

- `A` is `E` or `H`
- `T` is `D` or `B`
- `S` is the explicit source contribution
- `P` is the polarization contribution
- `m^{-1}` is the inverse constitutive coefficient or tensor operator

The implementation must preserve the lazy-allocation semantics:

- if `E == D` or `H == B` is sufficient, do not materialize a distinct array
- only allocate `W` if the corresponding PML direction requires it

## 6. Yee-Grid Tensor Interpolation

This is the key `D`-field trick from the dissertation.

Instead of expressing the solver directly as an `E`-field operator, the theory rewrites the hybrid operator in terms of `D`, because Meep updates `E` via `\varepsilon^{-1}`.

On a colocated continuum:

```math
E(\mathbf r_0) = \varepsilon^{-1}(\mathbf r_0) D(\mathbf r_0).
```

On the Yee grid, off-diagonal tensor terms are not colocated. The discrete update therefore uses interpolation operators. For example:

```math
E_x =
(\varepsilon^{-1})_{xx} D_x +
I_x (\varepsilon^{-1})_{xy} I_y D_y +
I_x (\varepsilon^{-1})_{xz} I_z D_z.
```

Collecting all components:

```math
\begin{bmatrix} E_x \\ E_y \\ E_z \end{bmatrix}
=
\widehat{\varepsilon^{-1}}
\begin{bmatrix} D_x \\ D_y \\ D_z \end{bmatrix},
```

with

```math
\widehat{\varepsilon^{-1}} =
\begin{bmatrix}
(\varepsilon^{-1})_{xx} & I_x(\varepsilon^{-1})_{xy}I_y & I_x(\varepsilon^{-1})_{xz}I_z \\
I_y(\varepsilon^{-1})_{yx}I_x & (\varepsilon^{-1})_{yy} & I_y(\varepsilon^{-1})_{yz}I_z \\
I_z(\varepsilon^{-1})_{zx}I_x & I_z(\varepsilon^{-1})_{zy}I_y & (\varepsilon^{-1})_{zz}
\end{bmatrix}.
```

This operator is the object that the code generator should think about, not just raw tensor entries.

## 7. Restriction Operator for Recombination

The recombination is not just `D^a * d\varepsilon^{-1}/d\rho * D^f` pointwise on a colocated grid.

Because `\widehat{\varepsilon^{-1}}` contains interpolations, its derivative does too:

```math
\frac{\partial \widehat{\varepsilon^{-1}}}{\partial \rho}
```

contains both:

- interpolation of the forward field to the material-tensor node
- restriction of the adjoint field back to the same node

The rule is:

- forward field path: interpolate
- adjoint field path: restrict with the transpose interpolation

If `I` is the interpolation operator, the backpropagated operator is `R = I^T`.

This is the key reason the generator must represent interpolation/restriction operators explicitly.

## 8. Hybrid Time/Frequency-Domain Operator

From the dissertation, the frequency-domain Maxwell system written in terms of `D` is:

```math
\left(
j\omega + \sigma_D + \frac{1}{j\omega}\,\nabla \times \mu^{-1} \nabla \times \varepsilon^{-1}
\right)\hat D
 = \text{source}.
```

With discrete-time substitution, the hybrid time/frequency operator becomes

```math
\widehat A(\rho,\omega)
=
\frac{1-e^{-j\omega\Delta t}}{\Delta t}
+
\frac{1+e^{-j\omega\Delta t}}{2}\,\widehat{\sigma_D}(\rho)
+
\frac{\Delta t}{1-e^{-j\omega\Delta t}}
[\nabla^- \times]\widehat{\mu^{-1}}[\nabla^+ \times]\widehat{\varepsilon^{-1}}(\rho).
```

This is the operator that the adjoint/recombination math must target.

## 9. Discrete Operator Sensitivity and Recombination

Differentiating the hybrid operator with respect to design parameters gives

```math
\frac{\partial \widehat A}{\partial \rho}
=
\frac{1+e^{-j\omega\Delta t}}{2}\,
\frac{\partial \widehat{\sigma_D}}{\partial \rho}
+
\frac{\Delta t}{1-e^{-j\omega\Delta t}}
[\nabla^- \times]\widehat{\mu^{-1}}[\nabla^+ \times]
\frac{\partial \widehat{\varepsilon^{-1}}}{\partial \rho}.
```

Acting on steady-state solutions, the curl-curl part contributes only a phase advance, which yields the discrete recombination factor

```math
\alpha_{\Delta t}(\omega)
=
\frac{\Delta t\,e^{-j\omega\Delta t}}{1-e^{-j\omega\Delta t}}.
```

So the design sensitivity becomes

```math
\frac{\partial \widehat A}{\partial \rho}
=
\frac{1+e^{-j\omega\Delta t}}{2}\,
\frac{\partial \widehat{\sigma_D}}{\partial \rho}
+
\alpha_{\Delta t}(\omega)\,
\frac{\partial \widehat{\varepsilon^{-1}}}{\partial \rho}.
```

and

```math
\lim_{\Delta t\to 0} \alpha_{\Delta t}(\omega) = -1.
```

### 9.1 Generator-Safe Recombination Formula

For each frequency `ω_m`, compute

```math
g_{\rho}(\omega_m)
=
\lambda(\omega_m)^T
\left(
\frac{1+e^{-j\omega_m\Delta t}}{2}
\frac{\partial \widehat{\sigma_D}}{\partial \rho}
+
\alpha_{\Delta t}(\omega_m)
\frac{\partial \widehat{\varepsilon^{-1}}}{\partial \rho}
\right)
x(\omega_m),
```

where:

- `x = \hat D^f`
- `\lambda = \hat D^a` after the required reciprocity transforms

Important convention:

- the dissertation uses the complex-symmetric/reciprocity formulation, so the natural algebra is a transpose form
- if an implementation uses a Hermitian inner product internally, it must also apply the corresponding transformed adjoint convention consistently

### 9.2 Practical Voxel-Level Form

At voxel/material-node level the generator should think in terms of:

```math
g_\rho(\omega)
=
\sum_q
\Big(
\alpha_{\Delta t}(\omega)\,
R_q \hat D^a(\omega)
\Big)^T
\frac{\partial \varepsilon_q^{-1}}{\partial \rho}
\Big(
I_q \hat D^f(\omega)
\Big)
+
\sum_q
\beta_{\sigma}(\omega)\,
\hat D^{a,T}_q
\frac{\partial \sigma_{D,q}}{\partial \rho}
\hat D^f_q,
```

with

```math
\beta_\sigma(\omega)=\frac{1+e^{-j\omega\Delta t}}{2}.
```

Here `q` indexes the Yee/tensor nodes relevant to the material map.

## 10. DFT Accumulation

The forward and adjoint runs both depend on in-run DTFT accumulation. For a sampled time signal `y[n]`:

```math
\hat y(\omega_m)
=
\sum_{n=0}^{N-1}
y[n] e^{+j\omega_m n\Delta t}\,\Delta t / \sqrt{2\pi}.
```

The implementation may decimate in time provided the source/monitor bandwidth makes the decimation Nyquist-safe.

The important invariants are:

- same DTFT normalization in forward and adjoint paths
- same normalization used for source synthesis and monitor evaluation
- persistent DFT buffers for design-region monitors

## 11. Adjoint Sources

## 11.1 Generic monitor-space rule

For any scalar objective `f(monitor_values)`, the adjoint source is built from

```math
\frac{\partial f}{\partial \hat x}.
```

The monitor-side Jacobian may be produced by AD, but the map from monitor-space gradient to Yee-grid current source must be explicitly specified by operator type.

## 11.2 Poynting-flux example

If the figure of merit uses voxel-centered interpolated fields,

```math
f =
\frac14\left[
I_{E,c}\overline{\hat E}\times I_{H,c}\hat H
+
I_{E,c}\hat E \times I_{H,c}\overline{\hat H}
\right]\cdot \hat n,
```

then

```math
\frac{\partial f}{\partial \hat x}
=
\frac14
\begin{bmatrix}
\left(I_{H,c}\hat H \times \hat n\right) I_{E,c} \\
-\left(I_{E,c}\hat E \times \hat n\right) I_{H,c}
\end{bmatrix}.
```

This is the source-side version of the same interpolation/restriction rule:

- evaluate the objective on colocated/interpolated fields
- restrict the resulting source pattern back to the Yee grid

## 11.3 Mode-monitor / eigenmode source

Meep's current implementation uses

```math
\frac{\partial a}{\partial E} \propto \frac12 c_{\text{scale}},
```

with a sign-flipped propagation vector for the adjoint source.

Generation rule:

- treat mode-monitor backprop as a dedicated operator family
- preserve the propagation-direction reversal
- preserve the monitor normalization factor

## 11.4 DFT-field monitor source

Given `dJ/d\hat F`, the adjoint source is constructed by a monitor-specific transpose map from monitor samples back to Yee-grid source entries.

The generator must model this as:

```math
J_{\text{adj}} = \mathcal M_{\text{dft}}^T \left(\frac{\partial f}{\partial \hat F}\right),
```

not as an ad hoc scatter.

## 11.5 Near-to-far source

Near-to-far backprop is the transpose Green-function operator:

```math
J_{\text{adj}} = G_{\text{n2f}}^T \left(\frac{\partial f}{\partial \hat E_{\text{far}}}\right).
```

This must remain an explicit operator family because future layered/inhomogeneous Green functions should slot into the same interface.

## 12. Broadband Adjoint Source Fitting

For frequencies `\omega_m`, the desired source spectrum is approximated by basis functions:

```math
\hat J(\mathbf x,\omega) \approx \sum_{m=0}^{M-1} b_m(\mathbf x)\,\hat W_m(\omega).
```

The weights are found from

```math
\min_{b_m(\mathbf x)}
\left\|
\hat J(\mathbf x,\omega) - \sum_m b_m(\mathbf x)\hat W_m(\omega)
\right\|_2.
```

The time-domain source is then

```math
J(\mathbf x,t) = \sum_m b_m(\mathbf x) W_m(t).
```

### 12.1 Nuttall basis used in the dissertation / Meep code

The frequency-domain basis function is

```math
\hat W_m(\omega) =
\frac12 \sum_{k=0}^3 a_k (-1)^k
\left[
\frac{1-e^{i(N+1)(\omega-\omega_m+k\Delta\omega)\Delta t}}
     {1-e^{i(\omega-\omega_m+k\Delta\omega)\Delta t}}
+
\frac{1-e^{i(N+1)(\omega-\omega_m-k\Delta\omega)\Delta t}}
     {1-e^{i(\omega-\omega_m-k\Delta\omega)\Delta t}}
\right]
```

with

```math
a_0=0.355768,\quad
a_1=0.4873960,\quad
a_2=0.144232,\quad
a_3=0.012604.
```

The corresponding time-domain sequence is

```math
W_m[n] =
e^{-i\omega_m n\Delta t}
\sum_{k=0}^3 a_k (-1)^k
\cos\left(\frac{2\pi k n\Delta t}{N}\right),
\qquad 0 \le n \le N.
```

Implementation note:

- Meep's `FilteredSource` halves `dt` to compensate for the E/H staggering
- that detail must be preserved or replaced by an explicitly equivalent convention

## 13. Adjoint-Source Normalization

Meep currently uses the discrete-time correction

```math
i\omega_{\text{disc}} =
\frac{1-e^{-i\omega \Delta t}}{\Delta t},
```

and computes monitor/source scaling numerically from the DTFT of the forward source waveform.

A generation-quality implementation should make this normalization explicit:

```math
s_{\text{adj}}(\omega)
\propto
\frac{i\omega_{\text{disc}}}{\hat s_{\text{fwd}}(\omega)}
```

plus:

- any frequency-window phase factor
- the real-field factor of `2` when only the real part of the current is injected

## 14. What Must Be Exact

The generator must preserve exactly:

- the forward/backward discrete curl transpose pairing
- the interpolation/restriction transpose pairing
- the discrete recombination factor `\alpha_{\Delta t}`
- the sign conventions for electric vs magnetic source injection
- the reciprocity transforms for Bloch/cylindrical/symmetry cases

## 15. What May Be Improved Relative to Meep

The implementation may improve on Meep by replacing:

- finite-difference material-grid differentiation

with:

- exact analytic differentiation/JVPs for
  - material-grid interpolation
  - projection
  - smoothing
  - conductivity interpolation
  - Yee restriction/interpolation operators

That is the most important mathematical improvement available to a new implementation.

