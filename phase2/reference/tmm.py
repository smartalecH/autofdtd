"""
Transfer Matrix Method (TMM) reference implementations for Phase 2 accuracy validation.

Provides analytical/semi-analytical ground truth for:
- Planar multilayer reflectance/transmittance (Fresnel equations at arbitrary angle/polarization)
- Directional coupler coupling coefficient and power transfer
- Ridge waveguide Bragg grating reflection spectrum

All functions accept numpy arrays and return reflectance/transmittance spectra.
"""

import numpy as np


def _snell_angle(n1, n2, cos_theta1):
    """Compute cos(theta2) from Snell's law. Handles total internal reflection."""
    sin_theta1_sq = 1.0 - cos_theta1**2
    sin_theta2_sq = (n1 / n2) ** 2 * sin_theta1_sq
    # Total internal reflection: cos_theta2 becomes imaginary
    if sin_theta2_sq >= 1.0:
        return 0.0, np.sqrt(sin_theta2_sq - 1.0) * 1j
    return np.sqrt(1.0 - sin_theta2_sq), 0.0


def _fresnel_coeffs(n1, n2, cos_theta1, cos_theta2, pol):
    """
    Compute Fresnel reflection/transmission coefficients.

    Parameters
    ----------
    n1, n2 : complex
        Refractive indices of incident and transmitted media
    cos_theta1, cos_theta2 : complex
        Cosines of incident and transmitted angles
    pol : str
        's' or 'p' polarization

    Returns
    -------
    r, t : complex
        Reflection and transmission coefficients (amplitude)
    """
    if pol == 's':
        r = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        t = 2 * n1 * cos_theta1 / (n1 * cos_theta1 + n2 * cos_theta2)
    else:  # 'p'
        r = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)
        t = 2 * n1 * cos_theta1 / (n2 * cos_theta1 + n1 * cos_theta2)
    return r, t


def _interface_RT(n1, n2, cos_theta1, pol):
    """
    Compute power reflectance and transmittance at a single interface.

    Returns
    -------
    R, T : float
        Power reflectance and transmittance
    """
    cos_theta2, eta = _snell_angle(n1, n2, cos_theta1)
    r, t = _fresnel_coeffs(n1, n2, cos_theta1, cos_theta2, pol)

    # Handle imaginary cos_theta2 (total internal reflection)
    if np.iscomplexobj(cos_theta2) and np.abs(np.imag(cos_theta2)) > 1e-12:
        return 1.0, 0.0

    cos_theta2 = np.real(cos_theta2)

    # Power transmittance includes the cos_theta factor from Poynting vector
    if pol == 's':
        T = np.real(n2 * cos_theta2) / np.real(n1 * cos_theta1) * np.abs(t)**2
    else:
        T = np.real(n2 * cos_theta2) / np.real(n1 * cos_theta1) * np.abs(t)**2

    R = np.abs(r)**2
    return R, T


def planar_multilayer_RT(n_layers, d, n, theta, wavelength, pol):
    """
    Compute reflectance and transmittance of a planar multilayer stack.

    Uses the Transfer Matrix Method (TMM) with the characteristic matrix formalism
    for s and p polarization at arbitrary incidence angle.

    Parameters
    ----------
    n_layers : int
        Number of layers (including surrounding media)
    d : array_like, shape (n_layers,)
        Physical thickness of each layer [length units]. d[0] and d[-1] are
        the surrounding semi-infinite media (unused, can be set to 0).
    n : array_like, shape (n_layers,) of complex
        Refractive index of each layer. n[0] is incident medium, n[-1] is
        transmitted medium.
    theta : float
        Incidence angle in radians (in the incident medium, n[0])
    wavelength : array_like
        Wavelength(s) [length units]. Can be scalar or array.
    pol : str
        Polarization: 's' or 'p'

    Returns
    -------
    R : ndarray
        Reflectance spectrum (same shape as wavelength)
    T : ndarray
        Transmittance spectrum (same shape as wavelength)

    Example
    -------
    >>> # Si slab in vacuum, normal incidence
    >>> n = np.array([1.0, 6.0 + 0.0j, 1.0])  # air, Si, air
    >>> d = np.array([0.0, 1.0, 0.0])  # 1 µm Si slab
    >>> R, T = planar_multilayer_RT(3, d, n, 0.0, np.linspace(0.5, 2.0, 100), 's')
    """
    d = np.asarray(d, dtype=complex)
    n = np.asarray(n, dtype=complex)
    wavelength = np.asarray(wavelength)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Broadcast wavelength if needed
    scalar_input = wavelength.ndim == 0
    if scalar_input:
        wavelength = np.atleast_1d(wavelength)

    n_wl = wavelength.shape[0]
    R = np.zeros(n_wl)
    T = np.zeros(n_wl)

    for i in range(n_wl):
        lam = wavelength[i]

        # Compute k0
        k0 = 2 * np.pi / lam

        # Phase thickness of each layer
        # cos_theta_j for each layer via Snell's law
        cos_thetas = np.zeros(n_layers, dtype=complex)
        cos_thetas[0] = cos_theta
        for j in range(1, n_layers):
            n_j = n[j]
            # cos(theta_j) = sqrt(1 - (n[0]/n_j)^2 * sin^2(theta))
            sin_theta_j_sq = (n[0] / n_j) ** 2 * sin_theta**2
            if np.real(sin_theta_j_sq) >= 1.0:
                cos_thetas[j] = 0.0
            else:
                cos_thetas[j] = np.sqrt(1.0 - sin_theta_j_sq)

        # Build the transfer matrix
        M = np.array([[1.0 + 0.0j, 0.0], [0.0, 1.0 + 0.0j]])

        for j in range(1, n_layers - 1):
            # Fresnel coefficients at interface j-1 -> j
            r_prev, t_prev = _fresnel_coeffs(n[j-1], n[j], cos_thetas[j-1], cos_thetas[j], pol)

            # Propagation matrix through layer j
            phi = k0 * n[j] * cos_thetas[j] * d[j]
            P = np.array([
                [np.exp(1j * phi), 0.0],
                [0.0, np.exp(-1j * phi)]
            ])

            # Interface matrix
            I = np.array([
                [1.0, r_prev],
                [r_prev, 1.0]
            ]) / t_prev

            M = M @ I @ P

        # Final interface: n[n_layers-2] -> n[n_layers-1]
        r_final, t_final = _fresnel_coeffs(
            n[n_layers-2], n[n_layers-1], cos_thetas[n_layers-2], cos_thetas[n_layers-1], pol
        )
        I_final = np.array([[1.0, r_final], [r_final, 1.0]]) / t_final
        M = M @ I_final

        # Overall reflection coefficient
        r_overall = M[1, 0] / M[0, 0]
        R[i] = np.abs(r_overall)**2

        # Overall transmission coefficient
        # T = (n_L * cos_theta_L / n_0 * cos_theta_0) * |t_overall|^2
        t_overall = 1.0 / M[0, 0]
        cos_theta_L = cos_thetas[-1]
        if pol == 's':
            T[i] = np.real(n[-1] * cos_theta_L) / np.real(n[0] * cos_theta) * np.abs(t_overall)**2
        else:
            T[i] = np.real(n[-1] * cos_theta_L) / np.real(n[0] * cos_theta) * np.abs(t_overall)**2

    if scalar_input:
        return float(R[0]), float(T[0])
    return R, T


def directional_coupler_C(L_coupling, n_eff_even, n_eff_odd):
    """
    Compute the coupling coefficient C for a directional coupler.

    Uses even/odd mode effective indices from a mode solver or analytical estimate.

    Parameters
    ----------
    L_coupling : float or array_like
        Coupling length(s) [length units]. Can be scalar or array.
    n_eff_even : complex
        Effective index of the even supermode
    n_eff_odd : complex
        Effective index of the odd supermode

    Returns
    -------
    C : float or ndarray
        Coupling coefficient [1/length units]. If L_coupling is array, returns array.

    Notes
    -----
    The coupling coefficient is:
        C = (n_eff_even - n_eff_odd) * pi / lambda

    This comes from the beat length L_b = lambda / (n_eff_even - n_eff_odd),
    and C = pi / L_b.

    The power evolution is:
        P1(z) = P0 * cos^2(C*z)
        P2(z) = P0 * sin^2(C*z)
    """
    delta_n = n_eff_even - n_eff_odd
    # Return the magnitude of the coupling coefficient
    return np.pi * np.abs(np.real(delta_n))


def directional_coupler_power(P0, C, z):
    """
    Compute power evolution in a directional coupler.

    Parameters
    ----------
    P0 : float
        Input power in waveguide 1 at z=0
    C : float or array_like
        Coupling coefficient [1/length]. If array, must match shape of z.
    z : array_like
        Position(s) along coupler [length units]

    Returns
    -------
    P1 : ndarray
        Power in waveguide 1 at each z
    P2 : ndarray
        Power in waveguide 2 at each z

    Notes
    -----
    For a symmetric 2×2 directional coupler with identical waveguides:
        P1(z) = P0 * cos²(C·z)
        P2(z) = P0 * sin²(C·z)

    Total power is conserved: P1(z) + P2(z) = P0
    """
    z = np.asarray(z)
    P1 = P0 * np.cos(C * z)**2
    P2 = P0 * np.sin(C * z)**2
    return P1, P2


def bragg_grating_RT(L, period, n_eff, kappa, wavelength):
    """
    Compute reflectance and transmittance of a ridge waveguide Bragg grating.

    Uses the coupled-mode theory (CMT) solution for a uniform Bragg grating
    with coupling coefficient kappa. The grating has N = L/period periods.

    Parameters
    ----------
    L : float
        Total grating length [length units]
    period : float
        Grating period [length units]
    n_eff : float
        Effective index of the waveguide mode
    kappa : float
        Coupling coefficient [1/length units]. For a uniform corrugation,
        kappa ≈ (π * Δn * Γ) / λ where Δn is the index modulation depth
        and Γ is the overlap factor.
    wavelength : array_like
        Wavelength(s) [length units]. Can be scalar or array.

    Returns
    -------
    R : ndarray
        Reflectance spectrum (same shape as wavelength)
    T : ndarray
        Transmittance spectrum (same shape as wavelength)

    Notes
    -----
    The Bragg grating is governed by coupled-mode equations:
        dA+/dz = i*delta*A+ + i*kappa*A-
        dA-/dz = -i*delta*A- - i*kappa*A+

    where delta = beta - pi/period is the detuning.

    The reflection coefficient at the input facet (for a uniform grating
    of length L with perfect input/output waveguides) is:

        r = i*kappa * sinh(s*L) / (delta * sinh(s*L) + i*s * cosh(s*L))
        t = s * exp(i*beta*L) / (s * cosh(s*L) + i*delta * sinh(s*L))

    where s = sqrt(kappa² - delta²) and beta = 2*pi*n_eff/lambda.

    The power reflectance is R = |r|² and transmittance is T = |t|².

    For the stopband (near Bragg wavelength lambda_B = 2*n_eff*period),
    delta ≈ 0 and R → tanh²(kappa*L).
    """
    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    if scalar_input:
        wavelength = np.atleast_1d(wavelength)

    n_wl = wavelength.shape[0]
    R = np.zeros(n_wl)
    T = np.zeros(n_wl)

    # Number of periods
    N = int(round(L / period))
    if N <= 0:
        return np.zeros_like(wavelength, dtype=float), np.zeros_like(wavelength, dtype=float)

    beta_arr = 2 * np.pi * n_eff / wavelength  # phase constant

    for i in range(n_wl):
        lam = wavelength[i]
        beta = beta_arr[i]

        # Detuning from Bragg condition
        # delta = beta - pi/period = 2*pi*n_eff/lambda - pi/period
        delta = beta - np.pi / period

        # Solve the coupled-mode equations analytically
        # s^2 = kappa^2 - delta^2
        s_sq = kappa**2 - delta**2

        if np.abs(s_sq) < 1e-20:
            # kappa ≈ delta: middle of stopband or critically coupled
            # Use stable form: for large kappa*L, use 1/cosh approximation
            X = kappa * L
            if X > 50:
                # R ≈ 1 - 4*exp(-2*X), T ≈ 4*exp(-2*X)
                R[i] = 1.0 - 4.0 * np.exp(-2.0 * X)
                T[i] = 4.0 * np.exp(-2.0 * X)
            else:
                tanh_kL = np.tanh(kappa * L)
                cosh_kL = np.cosh(kappa * L)
                R[i] = tanh_kL**2
                T[i] = 1.0 / cosh_kL**2

        elif s_sq.real > 0:
            # Propagating regime (outside stopband): s is real
            s = np.sqrt(s_sq.real)

            # Numerically stable computation for large s*L
            X = s * L
            if X > 50:
                # For large X: sinh(X) ≈ cosh(X) ≈ exp(X)/2
                # r = i*kappa*exp(X)/2 / (delta*exp(X)/2 + i*s*exp(X)/2)
                #   = i*kappa / (delta + i*s)
                # |r|^2 = kappa^2 / (delta^2 + s^2) -> approaches 1 near Bragg
                # But in the propagating regime (delta > kappa), s is imaginary,
                # so this branch shouldn't be hit for large X when we're near Bragg.
                # For large X far from Bragg, R is small anyway.
                # T asymptotic: |t| = |s|/|s*cosh(X) + i*delta*sinh(X)|
                #             ≈ 2*exp(-X) near Bragg (delta ≈ 0)
                #             ≈ |s|/|s| = 1 far from Bragg
                denom = delta**2 + s**2
                R[i] = kappa**2 / denom
                # T = |s|^2 / |delta^2 + s^2| for large X, but this gives T→1 near Bragg
                # which is wrong. The correct asymptotic for T near Bragg (delta≈0) is:
                # T ≈ 4*exp(-2*X)
                if np.abs(delta) < 1e-10:
                    T[i] = 4.0 * np.exp(-2.0 * X)
                else:
                    T[i] = s**2 / denom
            else:
                sinh_sL = np.sinh(s * L)
                cosh_sL = np.cosh(s * L)

                # Reflection coefficient amplitude
                r_num = 1j * kappa * sinh_sL
                r_den = delta * sinh_sL + 1j * s * cosh_sL
                r = r_num / r_den

                # Transmission coefficient amplitude
                t_num = s
                t_den = s * cosh_sL + 1j * delta * sinh_sL
                t = t_num / t_den * np.exp(1j * beta * L)

                R[i] = np.abs(r)**2
                T[i] = np.abs(t)**2
        else:
            # Evanescent regime (inside stopband): s is imaginary
            s_im = np.sqrt(-s_sq.real)

            # Numerically stable: for large s_im*L use asymptotic approximations
            # Inside stopband: s is imaginary, but the effective parameter for
            # the oscillatory form is the imaginary part
            X = s_im * L
            if X > 50:
                # For large X: sin(X) ≈ sign*tan(X) behavior, cos(X) oscillates
                # In this regime the grating is strongly reflective
                # R ≈ 1 - (delta/kappa)^2 * 4*exp(-2*X) (approximately)
                # T ≈ 4*exp(-2*X)
                R[i] = 1.0 - 4.0 * np.exp(-2.0 * X) * (delta / kappa)**2
                T[i] = 4.0 * np.exp(-2.0 * X)
            else:
                sin_sL = np.sin(s_im * L)
                cos_sL = np.cos(s_im * L)

                # Inside stopband: sinh/sin and cosh/cos swap roles
                r_num = 1j * kappa * sin_sL
                r_den = delta * sin_sL + 1j * s_im * cos_sL
                r = r_num / r_den

                t_num = s_im
                t_den = s_im * cos_sL + 1j * delta * sin_sL
                t = t_num / t_den * np.exp(1j * beta * L)

                R[i] = np.abs(r)**2
                T[i] = np.abs(t)**2

    if scalar_input:
        return float(R[0]), float(T[0])
    return R, T


def bragg_grating_RT_full(L, period, n_core, n_clad, w_core, h_core, corrug_depth, wavelength, pol='TE'):
    """
    Full Bragg grating R/T with ridge waveguide mode overlap.

    This version computes the coupling coefficient from the waveguide geometry
    and corrugation depth using the overlap integral approximation.

    Parameters
    ----------
    L : float
        Total grating length [µm]
    period : float
        Grating period [µm]
    n_core : float
        Core refractive index (Si ~ 3.48)
    n_clad : float
        Cladding refractive index (SiO2 ~ 1.44)
    w_core : float
        Core width [µm]
    h_core : float
        Core height [µm]
    corrug_depth : float
        Corrugation depth (half the total index modulation) [µm]
    wavelength : array_like
        Wavelength [µm]
    pol : str
        Polarization: 'TE' or 'TM'

    Returns
    -------
    R, T : ndarray
        Reflectance and transmittance spectra

    Notes
    -----
    For a ridge waveguide Bragg grating, the coupling coefficient is:

        kappa = (omega * eps0 / 2) * integral(E * delta_eps * E_transposed * dA)

    where delta_eps = eps0 * (n_corrug² - n_clad²) ≈ 2*eps0*n_clad*delta_n.

    For a uniform rectangular corrugation of depth delta_n and width w_corrug,
    the overlap with the mode field gives an effective kappa.

    As an approximation, we use:
        kappa ≈ (pi * delta_n * Gamma) / lambda

    where Gamma is the overlap of the modal field with the corrugated region.
    For ridge waveguides, Gamma_TE ≈ 0.5-0.8 for the TE mode in the core region.
    """
    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    if scalar_input:
        wavelength = np.atleast_1d(wavelength)

    # Approximate effective index of the ridge waveguide
    # Using the slab approximation for the core
    # n_eff ≈ n_clad + (n_core - n_clad) * (h_core / lambda) for weakly guiding
    # More accurate: solve characteristic equation for ridge

    # For a ridge waveguide at lambda ~ 1550 nm, use empirical formula
    # This is a simplification; in practice, use a mode solver
    V = (2 * np.pi / wavelength) * w_core * np.sqrt(n_core**2 - n_clad**2)

    # Single-mode approximation for the TE/TM effective index
    with np.errstate(divide='ignore', invalid='ignore'):
        n_eff = np.where(V > 0, n_clad * np.sqrt(1 + (V / (w_core * np.sqrt(n_core**2 - n_clad**2)))**2 * (n_core/n_clad - 1)), n_clad)

    # Clamp to physical range
    n_eff = np.clip(n_eff, n_clad, n_core)

    # Coupling coefficient from corrugation
    # delta_n = n_corrug - n_clad (using approximation n_corrug ≈ n_core for small depth)
    delta_n = corrug_depth * (n_core - n_clad) / (w_core / 2)

    # Overlap factor: fraction of mode field in the corrugated region
    # For ridge, the mode is concentrated in the core
    # Gamma ≈ 0.5-0.8 depending on geometry; use 0.6 as default
    Gamma = 0.6

    # kappa = (pi * delta_n * Gamma) / wavelength
    kappa_arr = np.pi * delta_n * Gamma / wavelength

    R, T = bragg_grating_RT(L, period, n_eff, kappa_arr, wavelength)

    if scalar_input:
        return float(R[0]), float(T[0])
    return R, T


if __name__ == '__main__':
    # Simple self-test
    import numpy as np

    print("Testing planar_multilayer_RT...")
    # Normal incidence on Si slab
    n = np.array([1.0, 3.48 + 0.0j, 1.0])
    d = np.array([0.0, 1.0, 0.0])  # 1 µm Si
    lam = np.array([1.55])
    R, T = planar_multilayer_RT(3, d, n, 0.0, lam, 's')
    print(f"  Si slab (1 µm) at 1.55 µm, s-pol: R={R[0]:.4f}, T={T[0]:.4f}")
    print(f"  (Expected: R ≈ 0.30-0.35, T ≈ 0.65-0.70)")

    # 30 degree incidence
    theta = np.deg2rad(30)
    R, T = planar_multilayer_RT(3, d, n, theta, lam, 's')
    print(f"  Si slab at 30°, s-pol: R={R[0]:.4f}, T={T[0]:.4f}")

    R, T = planar_multilayer_RT(3, d, n, theta, lam, 'p')
    print(f"  Si slab at 30°, p-pol: R={R[0]:.4f}, T={T[0]:.4f}")

    print("\nTesting directional_coupler_C...")
    C = directional_coupler_C(10.0, 2.5, 2.48)
    print(f"  L_coupling=10 µm, delta_n=0.02: C={C:.4f} 1/µm")

    z = np.linspace(0, 50, 200)
    P1, P2 = directional_coupler_power(1.0, C, z)
    print(f"  Power at z=0: P1={P1[0]:.4f}, P2={P2[0]:.4f}")
    print(f"  Power at z=pi/2C: P1={P1[50]:.4f}, P2={P2[50]:.4f}")

    print("\nTesting bragg_grating_RT...")
    lam = np.linspace(1.50, 1.60, 100)
    R, T = bragg_grating_RT(65.0, 0.324, 2.5, 50.0, lam)
    idx_max = np.argmax(R)
    print(f"  Bragg grating (200 periods): peak R at {lam[idx_max]:.4f} µm, R={R[idx_max]:.4f}")
    print(f"  Stopband transmittance at peak: T={T[idx_max]:.6f}")

    # Check energy conservation
    print(f"  R+T at peak: {R[idx_max] + T[idx_max]:.6f} (should be ~1.0)")
    print(f"  R+T at off-peak (1.55 µm): {R[50]+T[50]:.6f}")

    print("\nAll tests completed.")
