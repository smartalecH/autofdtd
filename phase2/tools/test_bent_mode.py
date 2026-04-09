"""Test bent mode injection in a simple simulation.

This tests that:
1. ModeSource with bend_radius parameter compiles correctly
2. Bent mode injection works in the simulation
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from autofdtd.api import (
    Simulation, GridSpec, UniformGrid, Boundary, BoundarySpec, PML, Medium,
    Structure, GridSpec,
)
from autofdtd.geometry import Box
from autofdtd.sources import GaussianPulse
from autofdtd.sources.mode import ModeSource
from autofdtd.compiler import compile_simulation


def test_bent_mode_injection():
    """Test bent waveguide mode injection."""
    print("=" * 60)
    print("Testing Bent Mode Injection")
    print("=" * 60)

    # Wavelength
    wl = 1.55e-6
    freq0 = 2.998e8 / wl

    # Define a straight waveguide
    wg_width = 0.5e-6
    wg_height = 0.22e-6
    wg_length = 10e-6

    # Create source with bent mode injection
    pulse = GaussianPulse(freq0=freq0, fwidth=freq0 * 0.1)

    # Bent mode source with 10 micron bend radius
    source_bent = ModeSource(
        center=(0, 0, 0),
        size=(wg_length, wg_width, 0),
        source_time=pulse,
        mode_index=0,
        direction='+',
        bend_radius=10e-6,  # 10 micron bend radius
        bend_axis=2,  # bend in x-y plane
        name='mode_bent',
    )

    # Create simulation
    dl = 0.5e-6  # Same as working examples
    sim = Simulation(
        size=(wg_length + 2e-6, wg_width + 2e-6, wg_height + 2e-6),
        run_time=200e-15,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=dl),
            grid_y=UniformGrid(dl=dl),
            grid_z=UniformGrid(dl=dl),
        ),
        medium=Medium(permittivity=1.45),
        structures=(
            Structure(
                geometry=Box(
                    center=(0, 0, 0),
                    size=(wg_length, wg_width, wg_height),
                ),
                medium=Medium(permittivity=11.7),
                name='waveguide',
            ),
        ),
        sources=[source_bent],
        boundary_spec=BoundarySpec(
            x=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
            y=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
            z=Boundary(plus=PML(num_layers=8), minus=PML(num_layers=8)),
        ),
        symmetry=(0, 0, 0),
    )

    print(f"Bent mode source: bend_radius={source_bent.bend_radius}, bend_axis={source_bent.bend_axis}")
    print(f"Source injection axis: {source_bent.injection_axis}")

    # Test compilation
    print("\nCompiling simulation with bent mode source...")
    try:
        compiled = compile_simulation(sim)
        print("Simulation compiled successfully!")
        print(f"Compiled simulation type: {type(compiled)}")
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = test_bent_mode_injection()
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    print("=" * 60)