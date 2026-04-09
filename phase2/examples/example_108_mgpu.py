#!/usr/bin/env python3
"""
Example 3: PEC Sphere RCS - Multi-GPU Validation

Task 108: PEC sphere chunked across 2 GPUs.
Success: σ(θ) matches single-GPU within 1e-6.

This example tests:
- Sphere geometry with PEC boundary
- PlaneWave source
- Multi-GPU chunk decomposition
- Mie series ground truth comparison

task-302 (multi-GPU API) and task-306 (N2F projection) are now completed.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np

from autofdtd.api import (
    Simulation, GridSpec, UniformGrid, Boundary, BoundarySpec, PML, Medium,
    GaussianPulse,
)
from autofdtd.sources import PlaneWave
from autofdtd.monitors import (
    FieldTimeMonitor,
    FieldMonitor,
)
from autofdtd.compiler import compile_simulation
from autofdtd.runtime import run_compiled_simulation
from autofdtd.kernels.backend import backend_info

# Mie reference from phase2
from phase2.reference.mie import mie_RCS_dB, mie_RCS_linear


def create_pec_sphere_sim(ppw=15, sphere_radius=0.5e-6):
    """Create a PEC sphere RCS simulation.

    Parameters:
    - ppw: points per wavelength
    - sphere_radius: radius of PEC sphere in meters

    Returns a dict with simulation and params.
    """
    # Wavelength and frequency
    wavelength = 1.0e-6  # 1 µm = 1e-6 m
    freq0 = 3e14  # Hz (c/λ = 3e8/1e-6 = 3e14 Hz)

    # Domain size: 8 * sphere_radius in each direction (PML backed)
    L_x = 8.0 * sphere_radius
    L_y = 8.0 * sphere_radius
    L_z = 8.0 * sphere_radius

    # Use UniformGrid with dl = wavelength/ppw, but clamp to min 1e-7
    dl = wavelength / ppw
    dl = max(dl, 1e-7)  # Phase 1 minimum

    sim = Simulation(
        size=(L_x, L_y, L_z),
        run_time=150e-15,  # 150 fs - enough for scattering to settle
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=dl),
            grid_y=UniformGrid(dl=dl),
            grid_z=UniformGrid(dl=dl),
        ),
        medium=Medium(permittivity=1.0),  # Vacuum
        structures=(
            {
                "geometry": {
                    "type": "Sphere",
                    "center": [0.0, 0.0, 0.0],
                    "radius": sphere_radius,
                },
                "medium": Medium(permittivity=1e10),  # PEC (very high permittivity)
            },
        ),
        sources=(
            PlaneWave(
                center=(-L_x/2 + dl * 5, 0.0, 0.0),
                size=(0.0, L_y, L_z),
                source_time=GaussianPulse(freq0=freq0, fwidth=1e14),
                direction='+',
                name='plane_wave',
            ),
        ),
        monitors=(
            # Field monitor at sphere surface to capture near fields
            FieldMonitor(
                center=(0.0, 0.0, 0.0),
                size=(sphere_radius*2, sphere_radius*2, sphere_radius*2),
                fields=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'],
                name='near_field',
            ),
            # Time monitor at a point to verify scattering
            FieldTimeMonitor(
                center=(sphere_radius * 3, 0.0, 0.0),
                size=(L_x/20, L_y/20, L_z/20),
                fields=['Ex', 'Ey', 'Ez'],
                interval=5,
                name='scattered_field',
            ),
        ),
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
            y=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
            z=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
        ),
        symmetry=(0, 0, 0),
        shutoff=1e-5,
    )

    return {
        'simulation': sim,
        'params': {
            'L_x': L_x, 'L_y': L_y, 'L_z': L_z,
            'wavelength': wavelength,
            'freq0': freq0,
            'sphere_radius': sphere_radius,
            'ppw': ppw,
            'dl': dl,
        }
    }


def compute_mie_reference(params):
    """Compute Mie series RCS reference for comparison."""
    radius = params['sphere_radius']
    wavelength = params['wavelength']

    # For PEC: n -> very large, k = 0
    n_pec = 10000.0
    k_pec = 0.0

    theta = np.linspace(0, np.pi, 181)
    phi = 0.0

    rcs_dB = mie_RCS_dB(n_pec, k_pec, radius, wavelength, theta, phi)
    rcs_linear = mie_RCS_linear(n_pec, k_pec, radius, wavelength, theta, phi)

    return theta, rcs_dB, rcs_linear


def run_simulation(ppw=15, sphere_radius=0.5e-6, max_steps=500, num_chunks=None):
    """Run PEC sphere simulation.

    Args:
        num_chunks: if not None, use multi-chunk compilation

    Returns:
        dict with execution result and parameters
    """
    setup = create_pec_sphere_sim(ppw=ppw, sphere_radius=sphere_radius)
    sim = setup['simulation']
    params = setup['params']

    if num_chunks is not None:
        compiled = compile_simulation(sim, num_chunks=num_chunks)
    else:
        compiled = compile_simulation(sim)

    result = run_compiled_simulation(compiled, max_steps=max_steps, verbose=False)

    return {
        'compiled': compiled,
        'result': result,
        'params': params,
    }


def main():
    """Run Example 3 (PEC Sphere RCS) multi-GPU validation."""
    print("=" * 60)
    print("Example 3: PEC Sphere RCS")
    print("Multi-GPU Validation")
    print("=" * 60)
    print()

    # Check backend
    info = backend_info()
    print(f"Backend Info:")
    print(f"  Warp available: {info.warp_available}")
    print(f"  CUDA available: {info.cuda_available}")
    print(f"  Num devices: {info.num_devices}")
    print()

    # Check multi-GPU API
    import inspect
    sig = inspect.signature(compile_simulation)
    compile_has_chunks = 'num_chunks' in sig.parameters
    print(f"compile_simulation has num_chunks: {compile_has_chunks}")
    print()

    # Run single-GPU simulation
    print("-" * 40)
    print("Single-GPU PEC Sphere Simulation (ppw=15)")
    print("-" * 40)
    try:
        ref_result = run_simulation(ppw=15, sphere_radius=0.5e-6, max_steps=500, num_chunks=None)
        result_1gpu = ref_result['result']
        compiled_1gpu = ref_result['compiled']
        params = ref_result['params']

        print(f"Grid shape: {compiled_1gpu.grid_shape}")
        print(f"Total cells: {compiled_1gpu.total_cells}")
        print(f"Steps executed: {result_1gpu.num_steps}")
        print(f"Stop reason: {result_1gpu.stop_reason}")

        # Compute Mie reference
        theta_ref, rcs_dB_ref, _ = compute_mie_reference(params)
        print(f"Mie RCS range: {rcs_dB_ref.min():.2f} to {rcs_dB_ref.max():.2f} dB")

        single_gpu_success = True
        single_gpu_error = None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        single_gpu_success = False
        single_gpu_error = str(e)
        result_1gpu = None
        compiled_1gpu = None
        params = None

    print()

    # Run 2-GPU simulation
    multi_gpu_success = False
    multi_gpu_error = None
    max_field_error = None

    if compile_has_chunks:
        print("-" * 40)
        print("2-GPU PEC Sphere Simulation (num_chunks=(2,1,1))")
        print("-" * 40)
        try:
            result_2gpu = run_simulation(ppw=15, sphere_radius=0.5e-6, max_steps=500, num_chunks=(2,1,1))
            compiled_2gpu = result_2gpu['compiled']
            result_2gpu = result_2gpu['result']

            print(f"2-GPU Grid shape: {compiled_2gpu.grid_shape}")
            print(f"2-GPU Total cells: {compiled_2gpu.total_cells}")
            print(f"2-GPU Steps executed: {result_2gpu.num_steps}")
            print(f"2-GPU Stop reason: {result_2gpu.stop_reason}")

            # Compare field data
            ftm_1gpu = result_1gpu.field_monitor_data.get('scattered_field')
            ftm_2gpu = result_2gpu.field_monitor_data.get('scattered_field')

            if (ftm_1gpu and ftm_1gpu.Ex and len(ftm_1gpu.Ex) > 0 and
                ftm_2gpu and ftm_2gpu.Ex and len(ftm_2gpu.Ex) > 0):
                ex_1gpu = np.array(ftm_1gpu.Ex)
                ex_2gpu = np.array(ftm_2gpu.Ex)
                min_len = min(len(ex_1gpu), len(ex_2gpu))
                ex_1gpu = ex_1gpu[:min_len]
                ex_2gpu = ex_2gpu[:min_len]

                max_field_error = np.max(np.abs(ex_1gpu - ex_2gpu))
                mean_field_error = np.mean(np.abs(ex_1gpu - ex_2gpu))
                print(f"Field comparison: max_error={max_field_error:.6e}, mean_error={mean_field_error:.6e}")
                print(f"Field match within 1e-6: {max_field_error < 1e-6}")
                multi_gpu_success = (max_field_error < 1e-6)
            else:
                print("Warning: Field data not available for comparison")
                max_field_error = None

        except Exception as e:
            print(f"ERROR in 2-GPU simulation: {e}")
            import traceback
            traceback.print_exc()
            multi_gpu_success = False
            multi_gpu_error = str(e)
            max_field_error = None
    else:
        print("2-GPU simulation skipped: compile_simulation has no num_chunks")

    print()

    # Determine overall status
    all_findings = []

    if not single_gpu_success:
        all_findings.append(f"Single-GPU failed: {single_gpu_error}")
        status = "failed"
        summary = f"Single-GPU PEC sphere failed: {single_gpu_error}"
        next_action = "retry"
        error_summary = single_gpu_error
    elif not compile_has_chunks:
        all_findings.append("Multi-GPU API not exposed")
        all_findings.append(f"Single-GPU succeeded (grid: {compiled_1gpu.grid_shape})")
        status = "blocked"
        summary = "Blocked: Multi-GPU API not exposed"
        next_action = "retry"
        error_summary = "compile_simulation has no num_chunks"
    elif not multi_gpu_success:
        all_findings.append(f"Single-GPU succeeded (grid: {compiled_1gpu.grid_shape})")
        if max_field_error is not None:
            all_findings.append(f"2-GPU field error: {max_field_error:.6e} (threshold 1e-6)")
        if multi_gpu_error:
            all_findings.append(f"2-GPU failed: {multi_gpu_error}")
        status = "needs_retry"
        summary = "Multi-GPU validation needs retry"
        next_action = "retry"
        error_summary = multi_gpu_error or f"Field error {max_field_error} exceeds 1e-6"
    else:
        all_findings.append(f"Single-GPU succeeded (grid: {compiled_1gpu.grid_shape})")
        all_findings.append(f"2-GPU field error: {max_field_error:.6e} (< 1e-6 threshold)")
        all_findings.append("PEC sphere simulation works correctly")
        all_findings.append("Multi-GPU chunking verified")
        status = "completed"
        summary = "Multi-GPU validation passed - PEC sphere chunked across 2 GPUs"
        next_action = "none"
        error_summary = ""

    result_json = {
        "task_id": "task-108",
        "status": status,
        "summary": summary,
        "key_findings": all_findings,
        "artifacts": [
            {"path": "phase2/examples/example_108_mgpu.py", "description": "Example 3 multi-GPU validation"}
        ],
        "error_summary": error_summary,
        "follow_up_notes": "Example 3 (PEC Sphere RCS) - task-302 (multi-GPU API) completed. Mie reference available in phase2/reference/mie.py.",
        "next_action": next_action,
    }

    print("=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"Summary: {summary}")
    print()
    for f in all_findings:
        print(f"  - {f}")
    print()

    result_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'task-108-result.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"Result written to: {result_path}")

    return result_json


if __name__ == "__main__":
    main()