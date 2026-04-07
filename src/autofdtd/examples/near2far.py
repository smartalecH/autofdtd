"""Near-to-far (N2F) postprocessing example.

This example demonstrates the N2F workflow:
1. Set up a simulation with a radiating structure
2. Record surface DFT fields using a projection monitor
3. Compute far-field radiation patterns using the equivalence principle
4. Extract diffraction orders for periodic structures

The key physical insight: the equivalence principle converts surface
fields (E, H on a closed surface) to equivalent currents (J_s = n×H, M_s=-n×E)
that radiate to the far field via the Green's function.

References:
- Clemson University computational EM notes on N2F
- Taflove "Computational Electrodynamics" on equivalence principle
- Harrington "Time-Harmonic Electromagnetic Wave" on far-field transformation
"""

from __future__ import annotations

import numpy as np

from autofdtd.api import (
    Box,
    Boundary,
    BoundarySpec,
    GaussianPulse,
    GridSpec,
    Medium,
    PML,
    PECBoundary,
    PMCBoundary,
    Scene,
    Simulation,
    Sphere,
    Structure,
    UniformCurrentSource,
    UniformGrid,
)
from autofdtd.monitors import (
    DiffractionMonitor,
    FieldProjectionAngleMonitor,
)
from autofdtd.kernels.near2far import (
    C0,
    Z0,
    compute_diffraction_orders_n2f,
    compute_surface_currents,
    far_field_greens_function,
    far_field_power,
    near2far_transform,
    extract_radiation_pattern,
)
from autofdtd.runtime.near2far import (
    build_near2far_surface,
    compute_total_radiated_power,
    far_field_directivity,
)


def define_radiator_simulation() -> Simulation:
    """Define a simulation with a simple radiating dipole.

    This creates a simulation with:
    - A vacuum background
    - A z-oriented current source (acts like a Hertzian dipole)
    - Field projection monitors to capture surface fields
    - PML boundaries to absorb radiation
    """
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(8.0, 8.0, 8.0),
        run_time=2e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=0.05),
            grid_y=UniformGrid(dl=0.05),
            grid_z=UniformGrid(dl=0.05),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            Structure(
                geometry=Box(
                    center=(0.0, 0.0, 0.0),
                    size=(0.5, 0.5, 0.5),
                ),
                medium=Medium(permittivity=1.5),
                name="scatterer",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(0.0, 0.0, 0.0),
                size=(0.0, 0.0, 0.2),  # z-oriented dipole
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=3e13,
                    amplitude=1.0,
                    offset=2.5,
                ),
                name="dipole_source",
            ),
        ),
        monitors=(
            FieldProjectionAngleMonitor(
                name="far_field_monitor",
                center=(0.0, 0.0, 0.0),
                size=(0.0, 8.0, 8.0),  # yz-plane surface
                freqs=(1e14, 1.5e14, 2e14),
                phi=(-180.0, 180.0, 181),
                theta=(0.0, 180.0, 181),
            ),
            DiffractionMonitor(
                name="diffraction_monitor",
                center=(0.0, 0.0, 3.0),
                size=(8.0, 8.0, 0.0),
                freqs=(1e14,),
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=15), minus=PML(num_layers=15)),
            y=Boundary(plus=PML(num_layers=15), minus=PML(num_layers=15)),
            z=Boundary(plus=PML(num_layers=15), minus=PML(num_layers=15)),
        ),
    )


def define_grating_simulation() -> Simulation:
    """Define a simulation with a diffraction grating.

    This creates a simulation with:
    - A periodic grating structure
    - A plane wave source
    - Diffraction monitors to capture diffraction orders
    """
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(6.0, 10.0, 6.0),
        run_time=3e-12,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=0.05),
            grid_y=UniformGrid(dl=0.05),
            grid_z=UniformGrid(dl=0.05),
        ),
        medium=Medium(permittivity=1.0),
        structures=(
            # Simple grating: alternating dielectric bars
            Structure(
                geometry=Box(
                    center=(0.0, 0.0, 0.0),
                    size=(6.0, 0.5, 2.0),
                ),
                medium=Medium(permittivity=3.5),
                name="grating_bar",
            ),
            Structure(
                geometry=Box(
                    center=(0.0, 1.0, 0.0),
                    size=(6.0, 0.5, 2.0),
                ),
                medium=Medium(permittivity=1.0),
                name="air_gap",
            ),
            Structure(
                geometry=Box(
                    center=(0.0, 2.0, 0.0),
                    size=(6.0, 0.5, 2.0),
                ),
                medium=Medium(permittivity=3.5),
                name="grating_bar_2",
            ),
        ),
        sources=(
            UniformCurrentSource(
                center=(0.0, -3.0, 0.0),
                size=(6.0, 0.0, 2.0),  # xz-plane plane wave
                polarization="Ez",
                current_amplitude_definition="total",
                source_time=GaussianPulse(
                    freq0=2e14,
                    fwidth=5e13,
                    amplitude=1.0,
                    offset=2.5,
                ),
                name="plane_wave_source",
            ),
        ),
        monitors=(
            DiffractionMonitor(
                name="diffraction_monitor",
                center=(0.0, 4.0, 0.0),
                size=(6.0, 0.0, 6.0),  # y-normal surface
                freqs=(1e14, 2e14, 3e14),
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PECBoundary(), minus=PECBoundary()),
            y=Boundary(plus=PML(num_layers=12), minus=PML(num_layers=12)),
            z=Boundary(plus=PML(num_layers=12), minus=PML(num_layers=12)),
        ),
    )


def demonstrate_surface_currents():
    """Demonstrate surface current computation from DFT fields."""
    print("\n=== Surface Current Computation ===")

    # Simulate DFT data from a z-normal surface
    n_pts = 100
    n_freqs = 3
    freqs = (1e14, 1.5e14, 2e14)

    # Synthetic DFT fields (plane wave at normal incidence)
    dft_e = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    dft_h = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    # E_x and E_y components, H_x and H_y from plane wave
    for i in range(n_pts):
        for j in range(n_freqs):
            k0 = 2 * np.pi * freqs[j] / C0
            phase = k0 * (i * 0.1)
            dft_e[i, j, 0] = np.exp(1j * phase)
            dft_e[i, j, 1] = 0.5 * np.exp(1j * phase)
            dft_h[i, j, 0] = -np.exp(1j * phase) / Z0 * 0.5
            dft_h[i, j, 1] = np.exp(1j * phase) / Z0

    placements = tuple((0, i, 0) for i in range(n_pts))
    cell_sizes = (0.1, 0.1, 0.1)

    Js, Ms = compute_surface_currents(
        dft_e, dft_h, placements,
        normal_axis=2, cell_sizes=cell_sizes,
    )

    print(f"Surface currents computed for {n_pts} points, {n_freqs} frequencies")
    print(f"J_s shape: {Js.shape}, M_s shape: {Ms.shape}")
    print(f"Mean |J_s|: {np.mean(np.abs(Js)):.6e}")
    print(f"Mean |M_s|: {np.mean(np.abs(Ms)):.6e}")


def demonstrate_near2far_transform():
    """Demonstrate far-field transformation from surface currents."""
    print("\n=== Near-to-Far Transformation ===")

    # Create simple surface current distribution (Hertzian dipole)
    n_pts = 20
    n_freqs = 1
    freqs = (2e14,)

    Js = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)
    Ms = np.zeros((n_pts, n_freqs, 3), dtype=np.complex128)

    # z-directed electric current (Hertzian dipole pattern)
    for i in range(n_pts):
        theta = np.pi * i / (n_pts - 1)
        Js[i, 0, 2] = np.sin(theta)  # z-component varies with position

    # Planar positions in xy plane
    positions = np.zeros((n_pts, 3))
    for i in range(n_pts):
        positions[i, 0] = (i % 4) * 0.1
        positions[i, 1] = (i // 4) * 0.1
        positions[i, 2] = 0.0

    cell_areas = np.full(n_pts, 0.01)

    # Far-field angles
    n_phi = 37
    n_theta = 19
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)

    e_theta, e_phi = near2far_transform(
        Js, Ms, positions,
        obs_distance=1e5,
        phi_vals=phi_vals, theta_vals=theta_vals,
        freqs=freqs, cell_areas=cell_areas,
    )

    print(f"Far-field computed: {n_phi} x {n_theta} angles, {n_freqs} frequencies")
    print(f"E_theta shape: {e_theta.shape}, E_phi shape: {e_phi.shape}")

    # Compute power pattern
    power = far_field_power(e_theta, e_phi, freqs)
    print(f"Peak power density: {np.max(power):.6e}")

    # Find peak direction
    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    print(f"Peak direction: phi={phi_vals[peak_idx[0]]:.2f} rad, theta={theta_vals[peak_idx[1]]:.2f} rad")


def demonstrate_diffraction():
    """Demonstrate diffraction order computation."""
    print("\n=== Diffraction Order Computation ===")

    n_pts = 50
    n_freqs = 2
    freqs = (1e14, 2e14)

    # Synthetic DFT data
    dft_e = np.random.randn(n_pts, n_freqs, 3) + 1j * np.random.randn(n_pts, n_freqs, 3)
    dft_h = np.random.randn(n_pts, n_freqs, 3) + 1j * np.random.randn(n_pts, n_freqs, 3)

    # Scale to physically reasonable values
    dft_e *= 0.1
    dft_h *= 0.01

    placements = tuple((0, i, 0) for i in range(n_pts))

    orders, mx, my = compute_diffraction_orders_n2f(
        dft_e, dft_h, placements,
        normal_axis=2,  # z-normal surface
        freqs=freqs,
        period_y=1e-6,  # 1 micron period
        period_z=1e-6,
        medium_eps=1.0 + 0.0j,
        num_orders=3,
    )

    print(f"Diffraction orders: {orders.shape[0]} orders for {n_freqs} frequencies")
    print(f"mx indices: {mx}")
    print(f"my indices: {my}")
    print(f"Order (0,0) amplitude at f={freqs[0]:.2e}: {orders[4, 0]:.4e}")  # center order


def demonstrate_directivity():
    """Demonstrate directivity computation."""
    print("\n=== Far-Field Directivity ===")

    n_phi = 36
    n_theta = 18
    n_freqs = 1
    freqs = (1e14,)

    # Simple isotropic radiator
    e_theta = np.ones((n_phi, n_theta, n_freqs)) * (1.0 + 0.0j)
    e_phi = np.zeros((n_phi, n_theta, n_freqs))

    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)

    # Compute directivity
    directivity_db = far_field_directivity(e_theta, e_phi, phi_vals, theta_vals, freqs)

    print(f"Directivity pattern shape: {directivity_db.shape}")
    print(f"Mean directivity: {np.mean(directivity_db):.2f} dBi")
    print(f"Peak directivity: {np.max(directivity_db):.2f} dBi")

    # For isotropic radiator, directivity should be 0 dBi everywhere
    print(f"Directivity std: {np.std(directivity_db):.4f} dBi (should be ~0 for isotropic)")


def demonstrate_radiation_pattern():
    """Demonstrate radiation pattern extraction."""
    print("\n=== Radiation Pattern Extraction ===")

    n_phi = 20
    n_theta = 10
    n_freqs = 2
    freqs = (1e14, 2e14)

    # Synthetic data with beam pattern
    e_theta = np.zeros((n_phi, n_theta, n_freqs))
    e_phi = np.zeros((n_phi, n_theta, n_freqs))

    # Main beam at phi=0, theta=pi/2
    beam_phi = 0
    beam_theta = n_theta // 2
    e_theta[beam_phi, beam_theta, :] = 1.0
    e_phi[beam_phi, beam_theta, :] = 0.5

    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)

    pattern = extract_radiation_pattern(e_theta, e_phi, phi_vals, theta_vals, freqs)

    print(f"Pattern keys: {list(pattern.keys())}")
    print(f"Peak power at f={freqs[0]}: {np.max(pattern['power'][:,:,0]):.6e}")
    print(f"Peak power at f={freqs[1]}: {np.max(pattern['power'][:,:,1]):.6e}")


def main():
    """Run all N2F demonstrations."""
    print("=" * 60)
    print("Near-to-Far (N2F) Postprocessing Demonstration")
    print("=" * 60)

    demonstrate_surface_currents()
    demonstrate_near2far_transform()
    demonstrate_diffraction()
    demonstrate_directivity()
    demonstrate_radiation_pattern()

    print("\n" + "=" * 60)
    print("N2F demonstration complete")
    print("=" * 60)

    # Show example usage with simulation
    print("\n=== Example: Radiation from Dipole Source ===")

    sim = define_radiator_simulation()
    print(f"Simulation: size={sim.size}, run_time={sim.run_time}")
    print(f"  Grid: {sim.grid_spec}")
    print(f"  Structures: {len(sim.structures)}")
    print(f"  Sources: {len(sim.sources)}")
    print(f"  Monitors: {len(sim.monitors)}")

    print("\nNote: Full simulation + N2F postprocessing requires")
    print("running the timestep loop and accumulating DFT data.")


if __name__ == "__main__":
    main()