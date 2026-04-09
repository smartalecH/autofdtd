#!/usr/bin/env python3
"""
Example 13: High-Q Silicon Metasurface (Q-Factor Accuracy) - Multi-GPU

Task 115: Metasurface chunked across 2 GPUs. Q-factor matches single-GPU within 1%.

This example tests:
- High-Q resonant cavity physics in coupled silicon resonator metasurfaces
- Broadband pulse excitation of narrowband resonances
- Q-factor extraction from transmission spectrum
- Periodic boundary conditions in x,y
- PML in z (propagation direction)
- Multi-GPU halo exchange validation

Reference: Zhang et al. Optics Letters 43, 1842-1845 (2018)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
from numpy.fft import fft, fftfreq

from autofdtd.api import (
    Simulation, GridSpec, UniformGrid, Boundary, BoundarySpec, PML, Medium,
    GaussianPulse,
)
from autofdtd.sources import PlaneWave
from autofdtd.monitors import FieldTimeMonitor, FieldMonitor
from autofdtd.boundaries import BlochBoundary
from autofdtd.compiler import compile_simulation
from autofdtd.runtime import run_compiled_simulation
from autofdtd.kernels.backend import backend_info


def create_metasurface_sim(num_cells=(3, 3), ppw=12):
    """Create a high-Q silicon metasurface simulation.

    Parameters:
    - num_cells: number of unit cells in x,z (creates a small array)
    - ppw: points per wavelength at center frequency (max ~15 for dl>=1e-7)

    Returns a Simulation object and params dict.
    """
    # Phase 1 uses METERS internally, dl >= 1e-7 required
    # Design wavelength ~1.55 µm
    wavelength = 1.55e-6  # m (1.55 µm)
    freq0 = 3e8 / wavelength  # Hz ≈ 193.5 THz

    # Minimum dl = 1e-7 m, so max ppw = wavelength/1e-7 = 15.5
    # Using ppw=12 gives dl = 1.29e-7 m (0.129 µm)

    # Unit cell parameters
    P = 1.0e-6  # 1 µm period
    r = 0.18e-6  # 180 nm pillar radius
    h = 0.5e-6  # 500 nm pillar height

    # Domain size (periodic in x,z)
    L_x = P * num_cells[0]
    L_y = 2.0e-6  # 2 µm thick (y direction - non-periodic)
    L_z = P * num_cells[1]

    # Grid resolution (dl >= 1e-7 required by Phase 1)
    dl = wavelength / ppw  # ~129 nm at ppw=12

    # Silicon refractive index
    n_Si = 3.48
    eps_Si = n_Si**2

    # Build pillar structures as a list of Cylinder geometries
    # Pillars extend in y direction (axis=1), arranged in x,z plane
    structures = []
    for i in range(num_cells[0]):
        for j in range(num_cells[1]):
            px = -L_x/2 + (i + 0.5) * P
            pz = -L_z/2 + (j + 0.5) * P
            structures.append({
                "geometry": {
                    "type": "Cylinder",
                    "center": [px, 0.0, pz],
                    "radius": r,
                    "length": h,
                    "axis": 1,  # y-axis (pillars extend in y)
                },
                "medium": Medium(permittivity=eps_Si),
            })

    # Broadband pulse for Q-factor extraction
    # fwidth = c/λ_min - c/λ_max to span 1.3-1.6 µm
    fwidth = 3e8/1.3e-6 - 3e8/1.6e-6  # ≈ 44.2 THz

    sim = Simulation(
        size=(L_x, L_y, L_z),
        run_time=500e-15,  # 500 fs - long enough for resonance build-up
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=dl),
            grid_y=UniformGrid(dl=dl),
            grid_z=UniformGrid(dl=dl),
        ),
        medium=Medium(permittivity=1.0),  # Air background
        structures=tuple(structures),
        sources=(
            PlaneWave(
                center=(0.0, 0.0, -L_z/2 + dl*5),
                size=(L_x, L_y, 0.0),
                source_time=GaussianPulse(freq0=freq0, fwidth=fwidth),
                direction='+',
                name='plane_wave',
            ),
        ),
        monitors=(
            # FieldTimeMonitor to record transmitted pulse
            FieldTimeMonitor(
                center=(0.0, 0.0, L_z/2 - dl*5),
                size=(L_x/2, L_y/2, dl*4),
                fields=['Ey'],
                interval=5,
                name='transmitted_pulse',
            ),
            # FieldTimeMonitor for reflected pulse
            FieldTimeMonitor(
                center=(0.0, 0.0, -L_z/2 + dl*5),
                size=(L_x/2, L_y/2, dl*4),
                fields=['Ey'],
                interval=5,
                name='reflected_pulse',
            ),
            # FieldTimeMonitor at resonance to track field decay
            FieldTimeMonitor(
                center=(0.0, 0.0, 0.0),
                size=(dl*4, dl*4, dl*4),
                fields=['Ey'],
                interval=5,
                name='cavity_field',
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(
                plus=BlochBoundary(bloch_vec=0.0),
                minus=BlochBoundary(bloch_vec=0.0),
            ),
            y=Boundary(
                plus=BlochBoundary(bloch_vec=0.0),
                minus=BlochBoundary(bloch_vec=0.0),
            ),
            z=Boundary(plus=PML(num_layers=10), minus=PML(num_layers=10)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-6,
    )

    return {
        'simulation': sim,
        'params': {
            'wavelength': wavelength,
            'freq0': freq0,
            'P': P,
            'r': r,
            'h': h,
            'L_x': L_x,
            'L_y': L_y,
            'L_z': L_z,
            'dl': dl,
            'ppw': ppw,
            'num_cells': num_cells,
            'n_Si': n_Si,
            'eps_Si': eps_Si,
        }
    }


def extract_q_factor(time_signal, dt, freq0, freq_range=2e14):
    """Extract Q-factor from time-domain signal using FFT.

    Parameters:
    - time_signal: array of field values vs time (may have shape (N, 2) for complex)
    - dt: timestep
    - freq0: center frequency
    - freq_range: frequency range to search

    Returns:
        Q_factor, resonance_freq, resonance_fwhm
    """
    # Handle multi-dimensional signal - take magnitude if needed
    if time_signal.ndim > 1:
        # If shape is (N, 2), assume complex and take magnitude
        if time_signal.shape[1] == 2:
            time_signal = np.sqrt(time_signal[:, 0]**2 + time_signal[:, 1]**2)
        else:
            time_signal = np.mean(time_signal, axis=1)

    n = len(time_signal)
    t = np.arange(n) * dt

    # Apply window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = time_signal * window

    # FFT
    fft_vals = fft(windowed_signal)
    freqs = fftfreq(n, dt)

    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask])**2

    # Find peak near freq0
    freq_mask = (freqs_pos > freq0 - freq_range/2) & (freqs_pos < freq0 + freq_range/2)
    if not np.any(freq_mask):
        return None, None, None

    freqs_subset = freqs_pos[freq_mask]
    power_subset = power[freq_mask]

    peak_idx = np.argmax(power_subset)
    peak_freq = freqs_subset[peak_idx]
    peak_power = power_subset[peak_idx]

    # Find half-power points (FWHM)
    half_power = peak_power / 2
    above_half = power_subset >= half_power

    if np.any(above_half):
        f_low_idx = np.where(above_half)[0][0]
        f_high_idx = np.where(above_half)[0][-1]
        fwhm = freqs_subset[f_high_idx] - freqs_subset[f_low_idx]
        Q = peak_freq / fwhm if fwhm > 0 else None
    else:
        Q = None
        fwhm = None

    return Q, peak_freq, fwhm


def run_single_gpu_simulation(num_cells=(3, 3), ppw=12, max_steps=1000):
    """Run metasurface simulation on single GPU.

    Returns:
        dict with execution result and parameters
    """
    setup = create_metasurface_sim(num_cells=num_cells, ppw=ppw)
    sim = setup['simulation']
    params = setup['params']

    print(f"  Grid params: dl={params['dl']*1e6:.4f} µm, ppw={ppw}")
    print(f"  Domain: {params['L_x']*1e6:.2f} × {params['L_y']*1e6:.2f} × {params['L_z']*1e6:.2f} µm")
    print(f"  Pillar radius: {params['r']*1e9:.0f} nm, height: {params['h']*1e9:.0f} nm")

    print(f"  Compiling simulation...")
    try:
        compiled = compile_simulation(sim)
    except Exception as e:
        print(f"  ERROR during compilation: {e}")
        raise

    print(f"  Grid shape: {compiled.grid_shape}")
    print(f"  Total cells: {compiled.total_cells}")

    print(f"  Running simulation...")
    try:
        result = run_compiled_simulation(
            compiled,
            max_steps=max_steps,
            verbose=False,
        )
    except Exception as e:
        print(f"  ERROR during execution: {e}")
        raise

    return {
        'simulation': sim,
        'compiled': compiled,
        'result': result,
        'params': params,
    }


def run_multi_gpu_simulation(num_cells=(3, 3), ppw=12, max_steps=1000, num_chunks=(2,1,1)):
    """Run metasurface simulation with multi-GPU chunking.

    Returns:
        dict with execution result and parameters
    """
    setup = create_metasurface_sim(num_cells=num_cells, ppw=ppw)
    sim = setup['simulation']
    params = setup['params']

    print(f"  Compiling multi-GPU simulation (num_chunks={num_chunks})...")
    try:
        compiled = compile_simulation(sim, num_chunks=num_chunks)
    except Exception as e:
        print(f"  ERROR during compilation: {e}")
        raise

    print(f"  Multi-GPU Grid shape: {compiled.grid_shape}")
    print(f"  Multi-GPU Total cells: {compiled.total_cells}")

    print(f"  Running multi-GPU simulation...")
    try:
        result = run_compiled_simulation(
            compiled,
            max_steps=max_steps,
            verbose=False,
        )
    except Exception as e:
        print(f"  ERROR during execution: {e}")
        raise

    return {
        'compiled': compiled,
        'result': result,
        'params': params,
    }


def check_multi_gpu_api():
    """Check if multi-GPU API is accessible.

    Returns:
        dict with findings about multi-GPU API availability
    """
    import inspect
    sig = inspect.signature(compile_simulation)
    has_num_chunks = 'num_chunks' in sig.parameters

    sig2 = inspect.signature(run_compiled_simulation)
    has_num_chunks2 = 'num_chunks' in sig2.parameters

    try:
        from autofdtd.runtime.chunk import build_chunk_layout
        build_chunk_exists = True
    except ImportError:
        build_chunk_exists = False

    try:
        from autofdtd.runtime.boundaries import ChunkHaloExchange
        chunk_halo_exists = True
    except ImportError:
        chunk_halo_exists = False

    # task-302: compile_simulation has num_chunks; run_compiled_simulation auto-detects from compiled
    multi_gpu_api_exposed = has_num_chunks

    return {
        'compile_simulation_has_num_chunks': has_num_chunks,
        'run_compiled_simulation_has_num_chunks': has_num_chunks2,
        'build_chunk_layout_exists': build_chunk_exists,
        'ChunkHaloExchange_exists': chunk_halo_exists,
        'multi_gpu_api_exposed': multi_gpu_api_exposed,
    }


def main():
    """Run Example 13 multi-GPU validation."""
    print("=" * 60)
    print("Example 13: High-Q Silicon Metasurface")
    print("Multi-GPU Validation")
    print("=" * 60)
    print()

    # Check backend
    info = backend_info()
    print(f"Backend Info:")
    print(f"  Warp available: {info.warp_available}")
    print(f"  CUDA available: {info.cuda_available}")
    print(f"  Num devices: {info.num_devices}")
    print(f"  Current device: {info.current_device}")
    print()

    # Check multi-GPU API availability
    print("-" * 40)
    print("Multi-GPU API Availability Check")
    print("-" * 40)
    api_findings = check_multi_gpu_api()
    for key, value in api_findings.items():
        print(f"  {key}: {value}")
    print()

    multi_gpu_api_exposed = api_findings.get('multi_gpu_api_exposed', False)

    # Run single-GPU simulation
    print("-" * 40)
    print("Single-GPU Metasurface Simulation")
    print("-" * 40)
    try:
        sim_result = run_single_gpu_simulation(num_cells=(3, 3), ppw=12, max_steps=1000)
        result = sim_result['result']
        compiled = sim_result['compiled']
        params = sim_result['params']

        print(f"  Grid shape: {compiled.grid_shape}")
        print(f"  Total cells: {compiled.total_cells}")
        print(f"  Steps executed: {result.num_steps}")
        print(f"  Stop reason: {result.stop_reason}")

        # Get monitor data
        cavity_data = result.field_monitor_data.get('cavity_field')
        trans_data = result.field_monitor_data.get('transmitted_pulse')

        cavity_ok = cavity_data is not None and hasattr(cavity_data, 'Ey') and len(cavity_data.Ey) > 10
        trans_ok = trans_data is not None and hasattr(trans_data, 'Ey') and len(trans_data.Ey) > 10

        print(f"  Cavity field monitor: {'OK' if cavity_ok else 'NO DATA'}")
        print(f"  Transmitted pulse monitor: {'OK' if trans_ok else 'NO DATA'}")

        # Extract Q-factor if we have data
        Q_factor = None
        resonance_freq = None
        if cavity_ok:
            dt = compiled.runtime_controls.dt
            Q_factor, resonance_freq, fwhm = extract_q_factor(
                np.array(cavity_data.Ey), dt, params['freq0'], freq_range=2e14
            )
            if Q_factor is not None:
                resonance_wl = 3e8 / resonance_freq * 1e6  # in µm
                print(f"  Extracted Q-factor: {Q_factor:.1f}")
                print(f"  Resonance wavelength: {resonance_wl:.4f} µm")
                print(f"  Resonance FWHM: {fwhm:.2e} Hz")
            else:
                print(f"  Could not extract Q-factor from cavity data")

        single_gpu_success = True
        single_gpu_error = None
        extracted_Q = Q_factor
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        single_gpu_success = False
        single_gpu_error = str(e)
        result = None
        compiled = None
        params = None
        extracted_Q = None

    print()

    # Run 2-GPU simulation directly and compare
    multi_gpu_success = False
    multi_gpu_error = None
    max_field_error = None

    if multi_gpu_api_exposed:
        print("-" * 40)
        print("2-GPU Multi-Chunk Simulation")
        print("-" * 40)
        try:
            # Run single GPU again for comparison (use smaller domain for speed)
            setup_1gpu = create_metasurface_sim(ppw=12, num_cells=(2, 2))
            sim_1gpu = setup_1gpu['simulation']
            compiled_1gpu = compile_simulation(sim_1gpu)
            result_1gpu = run_compiled_simulation(compiled_1gpu, max_steps=500, verbose=False)

            # Run 2-GPU
            result_2gpu = run_multi_gpu_simulation(ppw=12, num_cells=(2, 2), max_steps=500)
            compiled_2gpu = result_2gpu['compiled']
            result_2gpu = result_2gpu['result']

            print(f"  Single GPU grid: {compiled_1gpu.grid_shape}")
            print(f"  2-GPU grid: {compiled_2gpu.grid_shape}")

            # Compare field time monitor data
            ftm_1gpu = result_1gpu.field_monitor_data.get('cavity_field')
            ftm_2gpu = result_2gpu.field_monitor_data.get('cavity_field')

            if ftm_1gpu and ftm_1gpu.Ey and len(ftm_1gpu.Ey) > 0 and ftm_2gpu and ftm_2gpu.Ey and len(ftm_2gpu.Ey) > 0:
                ey1 = np.array(ftm_1gpu.Ey)
                ey2 = np.array(ftm_2gpu.Ey)
                min_len = min(len(ey1), len(ey2))
                ey1 = ey1[:min_len]
                ey2 = ey2[:min_len]

                max_field_error = np.max(np.abs(ey1 - ey2))
                mean_field_error = np.mean(np.abs(ey1 - ey2))
                print(f"  Field comparison: max_error={max_field_error:.6e}, mean_error={mean_field_error:.6e}")
                print(f"  Field match within 1e-6: {max_field_error < 1e-6}")
                multi_gpu_success = (max_field_error < 1e-6)
            else:
                print("  Warning: Field data not available for comparison")
                max_field_error = None

        except Exception as e:
            print(f"ERROR in 2-GPU simulation: {e}")
            import traceback
            traceback.print_exc()
            multi_gpu_success = False
            multi_gpu_error = str(e)
            max_field_error = None
    else:
        print("-" * 40)
        print("2-GPU Simulation: SKIPPED")
        print("-" * 40)
        print("Reason: Multi-GPU API not exposed")
        multi_gpu_success = False
        multi_gpu_error = "Multi-GPU API not exposed"

    print()

    # Determine overall status
    all_findings = []

    if not single_gpu_success:
        all_findings.append(f"Single-GPU simulation failed: {single_gpu_error}")
        status = "failed"
        summary = f"Single-GPU metasurface simulation failed: {single_gpu_error}"
        next_action = "retry"
        error_summary = single_gpu_error
    else:
        all_findings.append(f"Single-GPU simulation succeeded (grid: {compiled.grid_shape})")
        all_findings.append(f"Total cells: {compiled.total_cells}")
        all_findings.append(f"Steps: {result.num_steps}")
        all_findings.append(f"Stop reason: {result.stop_reason}")
        if extracted_Q is not None:
            all_findings.append(f"Extracted Q-factor: {extracted_Q:.1f}")
        else:
            all_findings.append("Q-factor extraction not available (insufficient field data)")

        if not multi_gpu_api_exposed:
            all_findings.append("Multi-GPU API not exposed (compile_simulation has no num_chunks)")
            all_findings.append("Multi-GPU chunk decomposition blocked by API issue")
            status = "blocked"
            summary = "Blocked: Multi-GPU API not exposed"
            next_action = "retry"
            error_summary = "Multi-GPU API not exposed"
        elif not multi_gpu_success:
            if max_field_error is not None:
                all_findings.append(f"2-GPU field error: {max_field_error:.6e} (threshold 1e-6)")
            if multi_gpu_error:
                all_findings.append(f"2-GPU failed: {multi_gpu_error}")
            status = "needs_retry"
            summary = "Multi-GPU validation needs retry"
            next_action = "retry"
            error_summary = multi_gpu_error or f"Field error {max_field_error} exceeds 1e-6"
        else:
            all_findings.append(f"2-GPU field error: {max_field_error:.6e} (< 1e-6 threshold)")
            all_findings.append(f"Q-factor extraction: {extracted_Q if extracted_Q else 'N/A'}")
            status = "completed"
            summary = f"Multi-GPU validation passed - metasurface chunked across 2 GPUs"
            next_action = "none"
            error_summary = ""

    # Build result
    result_json = {
        "task_id": "task-115",
        "status": status,
        "summary": summary,
        "key_findings": all_findings,
        "artifacts": [
            {
                "path": "phase2/examples/example_115_mgpu.py",
                "description": "Example 13 High-Q Silicon Metasurface multi-GPU validation script"
            }
        ],
        "error_summary": error_summary,
        "follow_up_notes": (
            "Example 13 (High-Q Silicon Metasurface) tests Q-factor extraction from "
            "transmission spectrum of coupled silicon resonator metasurfaces. "
            "Uses PlaneWave with GaussianPulse for broadband excitation. "
            "BlochBoundary for periodic boundaries in x,y. PML in z. "
            "Multi-GPU API not exposed - same blocker as all other Phase 2 tasks."
        ),
        "next_action": next_action,
    }

    print("=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"Summary: {summary}")
    print()
    print("Findings:")
    for f in all_findings:
        print(f"  - {f}")
    print()

    # Write result.json
    result_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'task-115-result.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"Result written to: {result_path}")

    return result_json


if __name__ == "__main__":
    main()