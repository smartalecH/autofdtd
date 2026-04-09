"""
Directional Coupler Example - autofdtd implementation

This example simulates a 2x2 directional coupler (4-port device) using autofdtd.
The geometry is based on the gdsfactory generic PDK coupler cell.

The device consists of two parallel waveguides brought close together over
a coupling region. Light injected at port o1 (input) couples to both output
ports o2 (direct throughput) and o4 (cross-over).

Usage:
    python3 examples/directional_coupler.py [--resolution 6]

Arguments:
    --resolution: cells per wavelength (default: 6)

Resolution: 6 cells/wavelength (minimum for Phase 1 accuracy)
Wavelength: 1.55 μm (1550 nm C-band)
Material: SiN (silicon nitride) waveguides on SiO2 substrate

Expected behavior:
- Simulation should run to completion without errors
- Field energy should grow as source injects, then stabilize
- Field monitors should record time-domain fields
"""

import argparse
import numpy as np
import json

# Core simulation
from autofdtd import (
    Simulation,
    Scene,
    Structure,
    Box,
    Medium,
    PoleResidue,
    GaussianPulse,
    UniformCurrentSource,
)
from autofdtd.grid import SubpixelSpec, GridSpec, AutoGrid

# Boundaries
from autofdtd.boundaries import (
    Boundary,
    BoundarySpec,
    Absorber,
)

# Monitors
from autofdtd.monitors import FieldTimeMonitor

from autofdtd.compiler import compile_simulation
from autofdtd.runtime import run_compiled_simulation
from autofdtd.kernels.backend import backend_info


# ============================================================================
# Material Definitions (Pole-Residue models from reference)
# ============================================================================

# SiN (silicon nitride) - refractive index ~ 2.02 at 1550nm
SiN_MATERIAL = PoleResidue(
    eps_inf=1.0,
    poles=(
        ((-2.195613075604907, 0.0), (-0.1481687350520931, 5692186527705453.0)),
        ((-26.27602381601276, -1.391788056240773e+16), (70.84166613734145, 2.1050095837632636e+16)),
    ),
)

# SiO2 (silicon dioxide) - cladding
SiO2_MATERIAL = PoleResidue(
    eps_inf=1.0,
    poles=(
        ((-0.057, 0.0), (-0.057, 9.5e12)),
    ),
)


# ============================================================================
# Geometry Parameters (from coupler.yml)
# ============================================================================

WAVELENGTH = 1.55e-6  # meters
DEFAULT_RESOLUTION = 6  # cells per wavelength

# Waveguide dimensions
WAVEGUIDE_WIDTH = 0.5e-6  # 0.5 μm
COUPLER_LENGTH = 20.0e-6  # 20 μm
GAP = 0.236e-6  # 0.236 μm

# Port positions (from yaml)
WAVEGUIDE1_Y_CENTER = -1.632e-6
WAVEGUIDE2_Y_CENTER = 2.368e-6

# Simulation domain
SOLVER_X_MIN = -21.0e-6
SOLVER_X_MAX = 41.0e-6
SOLVER_Y_MIN = -12.0e-6
SOLVER_Y_MAX = 13.0e-6
SOLVER_Z_MIN = -1.0e-6
SOLVER_Z_MAX = 1.0e-6

SOLVER_SIZE = (SOLVER_X_MAX - SOLVER_X_MIN, SOLVER_Y_MAX - SOLVER_Y_MIN, SOLVER_Z_MAX - SOLVER_Z_MIN)
SOLVER_CENTER = (
    0.5 * (SOLVER_X_MIN + SOLVER_X_MAX),
    0.5 * (SOLVER_Y_MIN + SOLVER_Y_MAX),
    0.5 * (SOLVER_Z_MIN + SOLVER_Z_MAX),
)

# Coupling region
COUPLER_X_MIN = 0.0e-6
COUPLER_X_MAX = 20.0e-6

# Waveguide spans
INPUT_WG_X_MIN = -20.0e-6
INPUT_WG_X_MAX = 10.0e-6
OUTPUT_WG_X_MIN = 20.0e-6
OUTPUT_WG_X_MAX = 40.0e-6

# Port positions
PORT_O1_X = -10.0e-6


# ============================================================================
# Build Structures
# ============================================================================

def build_scene() -> Scene:
    """Build the directional coupler scene with two parallel SiN waveguides."""

    structures = []

    # Input waveguide (lower)
    input_wg = Structure(
        geometry=Box(
            center=(0.5 * (INPUT_WG_X_MIN + INPUT_WG_X_MAX), WAVEGUIDE1_Y_CENTER, 0.0),
            size=(INPUT_WG_X_MAX - INPUT_WG_X_MIN, WAVEGUIDE_WIDTH, SOLVER_Z_MAX - SOLVER_Z_MIN),
        ),
        medium=SiN_MATERIAL,
        name="input_waveguide",
    )
    structures.append(input_wg)

    # Output waveguide (upper)
    output_wg = Structure(
        geometry=Box(
            center=(0.5 * (OUTPUT_WG_X_MIN + OUTPUT_WG_X_MAX), WAVEGUIDE2_Y_CENTER, 0.0),
            size=(OUTPUT_WG_X_MAX - OUTPUT_WG_X_MIN, WAVEGUIDE_WIDTH, SOLVER_Z_MAX - SOLVER_Z_MIN),
        ),
        medium=SiN_MATERIAL,
        name="output_waveguide",
    )
    structures.append(output_wg)

    # Coupling region - two parallel waveguides
    for i, y_center in enumerate([WAVEGUIDE1_Y_CENTER, WAVEGUIDE2_Y_CENTER]):
        coupler_wg = Structure(
            geometry=Box(
                center=(0.5 * (COUPLER_X_MIN + COUPLER_X_MAX), y_center, 0.0),
                size=(COUPLER_X_MAX - COUPLER_X_MIN, WAVEGUIDE_WIDTH, SOLVER_Z_MAX - SOLVER_Z_MIN),
            ),
            medium=SiN_MATERIAL,
            name=f"coupler_wg{i+1}",
        )
        structures.append(coupler_wg)

    return Scene(
        medium=SiO2_MATERIAL,
        structures=structures,
    )


# ============================================================================
# Build Simulation
# ============================================================================

def build_simulation(resolution: int) -> Simulation:
    """Build the complete directional coupler simulation."""

    scene = build_scene()

    c_0 = 299792458.0  # m/s
    freq0 = c_0 / WAVELENGTH
    freq_width = 0.05e-6 / WAVELENGTH * freq0

    pulse = GaussianPulse(
        freq0=freq0,
        fwidth=freq_width,
        amplitude=1.0,
        offset=3.0,
    )

    # Source at input port (x = -10e-6 + small offset into waveguide)
    source_x = PORT_O1_X + 1.0e-6
    source_size_y = WAVEGUIDE_WIDTH + 4.0e-6
    source_size_z = SOLVER_Z_MAX - SOLVER_Z_MIN

    source = UniformCurrentSource(
        center=(source_x, WAVEGUIDE1_Y_CENTER, 0.0),
        size=(0.0, source_size_y, source_size_z),
        polarization="Ez",
        current_amplitude_definition="total",
        source_time=pulse,
        name="input_source",
    )

    # Time-domain field monitor at input
    input_monitor = FieldTimeMonitor(
        center=(source_x, WAVEGUIDE1_Y_CENTER, 0.0),
        size=(0.0, source_size_y, source_size_z),
        fields=["Ez"],
        interval=10,
        name="input_field",
    )

    # Time-domain field monitor for cross-section (xy plane at z=0)
    # This captures the field distribution in the coupling region
    # Use a coarser interval to reduce data size
    cross_section_monitor = FieldTimeMonitor(
        center=(
            0.5 * (SOLVER_X_MIN + SOLVER_X_MAX),
            0.5 * (WAVEGUIDE1_Y_CENTER + WAVEGUIDE2_Y_CENTER),
            0.0,
        ),
        size=(
            SOLVER_X_MAX - SOLVER_X_MIN,
            WAVEGUIDE2_Y_CENTER - WAVEGUIDE1_Y_CENTER + 2 * WAVEGUIDE_WIDTH + 2e-6,
            0.0,
        ),
        fields=["Ez"],  # Only record Ez since source is Ez-polarized
        interval=50,  # Record every 50 steps
        name="cross_section",
    )

    # Run time - enough for fields to propagate through and settle
    run_time = 30.0 * (SOLVER_X_MAX - SOLVER_X_MIN) * 2.0 / c_0

    sim = Simulation(
        center=SOLVER_CENTER,
        size=SOLVER_SIZE,
        run_time=run_time,
        grid_spec=GridSpec(
            grid_x=AutoGrid(min_steps_per_wvl=resolution),
            grid_y=AutoGrid(min_steps_per_wvl=resolution),
            grid_z=AutoGrid(min_steps_per_wvl=resolution),
            wavelength=WAVELENGTH,
        ),
        medium=scene.medium,
        structures=scene.structures,
        sources=(source,),
        monitors=(input_monitor, cross_section_monitor),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=Absorber(), minus=Absorber()),
            y=Boundary(plus=Absorber(), minus=Absorber()),
            z=Boundary(plus=Absorber(), minus=Absorber()),
        ),
        subpixel=SubpixelSpec(),
        symmetry=(0, 0, 0),
        shutoff=1e-6,
    )

    return sim


# ============================================================================
# Main Execution
# ============================================================================

def run_directional_coupler_example(resolution: int = DEFAULT_RESOLUTION) -> dict:
    """Run the directional coupler example and return results."""

    print("=" * 70)
    print("Directional Coupler Example - autofdtd")
    print("=" * 70)
    print(f"Wavelength: {WAVELENGTH * 1e6:.2f} μm")
    print(f"Resolution: {resolution} cells/wavelength")
    print(f"Coupler length: {(COUPLER_X_MAX - COUPLER_X_MIN) * 1e6:.1f} μm")
    print(f"Waveguide gap: {GAP * 1e6:.3f} μm")
    print()

    info = backend_info()
    print(f"Backend: warp_available={info.warp_available}, cuda_available={info.cuda_available}")
    print(f"Devices: {info.num_devices}")
    print()

    # Compute center frequency
    c_0 = 299792458.0
    freq0 = c_0 / WAVELENGTH

    print("Building simulation...")
    sim = build_simulation(resolution)

    print("Compiling simulation...")
    compiled = compile_simulation(sim)
    print(f"  Grid cells: {compiled.total_cells} ({compiled.grid_shape})")
    print(f"  Sources: {compiled.num_sources}")
    print(f"  Monitors: {compiled.num_monitors}")
    print(f"  Runtime dt: {compiled.runtime_controls.dt:.6e} s")
    print(f"  Max steps: {compiled.runtime_controls.num_time_steps}")

    print()
    print("Running simulation...")
    record_interval = 50
    result = run_compiled_simulation(
        compiled,
        max_steps=compiled.runtime_controls.num_time_steps,
        record_interval=record_interval,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Steps run: {result.num_steps}")
    print(f"Final time: {result.final_time:.6e} s")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Backend: {result.metrics.get('backend', 'unknown')}")
    print(f"Wall time: {result.metrics.get('wall_time_s', 'N/A'):.2f} s"
          if isinstance(result.metrics.get('wall_time_s'), (int, float))
          else f"Wall time: {result.metrics.get('wall_time_s', 'N/A')}")

    # Field monitor data
    print()
    print("Field Monitors:")
    for name in ["input_field", "cross_section"]:
        if name in result.field_monitor_data:
            data = result.field_monitor_data[name]
            print(f"  {name}: recorded")
        else:
            print(f"  {name}: no data")

    # Cross-section field intensity
    max_e2 = None
    if "cross_section" in result.field_monitor_data:
        cross_data = result.field_monitor_data["cross_section"]

        # Get time-domain Ez field
        ez = getattr(cross_data, 'Ez', None)

        print()
        print("Cross-section field data:")
        print(f"  Ez type: {type(ez)}, shape: {np.array(ez).shape if ez is not None else 'None'}")

        if ez is not None:
            ez_arr = np.array(ez)
            print(f"  Ez array shape: {ez_arr.shape}")

            # Compute DFT at center frequency to get steady-state field
            # The time samples are at intervals of record_interval * dt
            dt_per_record = compiled.runtime_controls.dt * record_interval
            n_steps = ez_arr.shape[0] if len(ez_arr.shape) > 0 else 0

            if n_steps > 0:
                # The data is stored as (N, 2) pairs (real, imag)
                if len(ez_arr.shape) == 2 and ez_arr.shape[1] == 2:
                    # Combine real and imaginary parts
                    ez_complex = ez_arr[:, 0] + 1j * ez_arr[:, 1]

                    # Determine spatial dimensions from monitor size and resolution
                    dx = WAVELENGTH / resolution
                    dy = WAVELENGTH / resolution
                    nx_monitor = int(round((SOLVER_X_MAX - SOLVER_X_MIN) / dx))
                    ny_monitor = int(round((WAVEGUIDE2_Y_CENTER - WAVEGUIDE1_Y_CENTER + 2 * WAVEGUIDE_WIDTH + 2e-6) / dy))
                    n_spatial = nx_monitor * ny_monitor

                    # Total elements = n_steps * n_spatial
                    total_elements = len(ez_complex)

                    # Determine n_times from the actual recorded data
                    # num_records = floor((max_steps-1)/interval) + 2 = floor(25496/50) + 2 = 510
                    n_times = 510
                    n_spatial = total_elements // n_times
                    print(f"  Total elements: {total_elements}, n_times={n_times}, n_spatial={n_spatial}")

                    # Try to determine ny from nx
                    nx_monitor = int(round((SOLVER_X_MAX - SOLVER_X_MIN) / dx))
                    if n_spatial % nx_monitor == 0:
                        ny_monitor = n_spatial // nx_monitor
                    else:
                        ny_monitor = int(round((WAVEGUIDE2_Y_CENTER - WAVEGUIDE1_Y_CENTER + 2 * WAVEGUIDE_WIDTH + 2e-6) / dy))

                    print(f"  Monitor: nx={nx_monitor}, ny={ny_monitor}, n_spatial={nx_monitor * ny_monitor}")

                    # Reshape to (n_times, ny, nx)
                    ez_reshaped = ez_complex.reshape((n_times, ny_monitor, nx_monitor))
                else:
                    ez_reshaped = ez_arr
                    n_times = n_steps

                print(f"  ez_reshaped shape: {ez_reshaped.shape}")

                omega = 2 * np.pi * freq0
                t = np.arange(n_times) * dt_per_record

                # For each spatial point, compute DFT at freq0
                if len(ez_reshaped.shape) == 3:
                    _, ny, nx = ez_reshaped.shape
                    # DFT: E(w) = sum_t E(t) * exp(-i*w*t) * dt
                    dft_ez = np.zeros((ny, nx), dtype=complex)
                    for i in range(ny):
                        for j in range(nx):
                            e_t = ez_reshaped[:, i, j]
                            dft_ez[i, j] = np.sum(e_t * np.exp(-1j * omega * t)) * dt_per_record

                    # |E|^2 = |Ez|^2 for Ez-polarized source
                    e2 = np.abs(dft_ez)**2
                    max_e2 = float(np.max(e2))
                    print(f"  |E|^2 max (from DFT at {freq0/1e12:.2f} THz): {max_e2:.6e}")

                    # Save cross-section data
                    np.savez(
                        f"cross_section_res{resolution}.npz",
                        x_min=SOLVER_X_MIN, x_max=SOLVER_X_MAX,
                        y_min=WAVEGUIDE1_Y_CENTER - WAVEGUIDE_WIDTH, y_max=WAVEGUIDE2_Y_CENTER + WAVEGUIDE_WIDTH,
                        e2=e2,
                        wavelength=WAVELENGTH,
                        resolution=resolution,
                    )
                    print(f"  Saved to cross_section_res{resolution}.npz")
                elif len(ez_reshaped.shape) == 2:
                    # 2D data (ny, nx)
                    ny, nx = ez_reshaped.shape
                    dft_ez = np.sum(ez_reshaped * np.exp(-1j * omega * t[:, None, None]), axis=0) * dt_per_record
                    e2 = np.abs(dft_ez)**2
                    max_e2 = float(np.max(e2))
                    print(f"  |E|^2 max (from DFT): {max_e2:.6e}")
                else:
                    print(f"  Unexpected array shape: {ez_reshaped.shape}")
            else:
                print("  No time steps recorded")
        else:
            print("  Ez field data is None")

    # Energy
    energy = result.integrated_electric_history
    if energy:
        max_energy = max(energy)
        print()
        print(f"Energy: initial={energy[0]:.6e}, final={energy[-1]:.6e}")
        print(f"Max energy: {max_energy:.6e}")

    # Return results
    return {
        "wavelength_um": WAVELENGTH * 1e6,
        "resolution": resolution,
        "num_steps": result.num_steps,
        "stop_reason": result.stop_reason,
        "total_cells": compiled.total_cells,
        "grid_shape": compiled.grid_shape,
        "backend": result.metrics.get("backend", "unknown"),
        "wall_time_s": result.metrics.get("wall_time_s", None),
        "max_energy": float(max_energy) if energy else None,
        "max_field_intensity": max_e2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directional Coupler Example")
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Resolution in cells per wavelength (default: {DEFAULT_RESOLUTION})"
    )
    args = parser.parse_args()

    results = run_directional_coupler_example(resolution=args.resolution)
    print()
    print("Example completed!")
    print(f"Results: {json.dumps(results, indent=2)}")
