"""Examples for the Phase 1 boundary subset."""

from __future__ import annotations

import numpy as np

from autofdtd.api import AbsorberParams, Boundary, BoundarySpec
from autofdtd.compiler import compile_boundary_spec
from autofdtd.runtime import allocate_abc_boundary_state, build_boundary_runtime


def boundary_runtime_snapshot() -> dict[str, object]:
    """Return a JSON-ready snapshot of the compiled Phase 1 boundary runtime metadata."""

    compiled = compile_boundary_spec(
        BoundarySpec(
            x=Boundary.periodic(),
            y=Boundary.pec(),
            z=Boundary.pmc(),
        )
    )
    return compiled.to_payload()


def pml_absorption_snapshot() -> dict[str, object]:
    """Return a compact snapshot showing the staged PML attenuation path."""

    runtime = build_boundary_runtime(
        BoundarySpec(x=Boundary.pml(num_layers=4), y=Boundary.periodic(), z=Boundary.periodic()),
        dt=0.5,
        field_shape=(10, 4, 4, 3),
    )
    electric = np.zeros((10, 4, 4, 3), dtype=np.float64)
    magnetic = np.zeros((10, 4, 4, 3), dtype=np.float64)
    electric[5:8, 1:3, 1:3, 0] = 1.0
    before = float(np.sum(np.abs(electric) ** 2))
    for _ in range(8):
        electric, magnetic = runtime.apply(electric, magnetic)
    after = float(np.sum(np.abs(electric) ** 2))
    return {
        "compiled_pml_faces": runtime.compiled.pml_faces,
        "stage_order": runtime.compiled.stage_order,
        "energy_before": before,
        "energy_after": after,
    }


def absorbing_boundary_snapshot() -> dict[str, object]:
    """Return a compact snapshot for StablePML and Absorber compilation."""

    compiled = compile_boundary_spec(
        BoundarySpec(
            x=Boundary.stable_pml(num_layers=8),
            y=Boundary.absorber(
                num_layers=10,
                parameters=AbsorberParams(sigma_max=7.2),
            ),
            z=Boundary.periodic(),
        ),
        dt=0.5,
        grid_spacing=(0.25, 0.25, 0.5),
    )
    return {
        "stage_order": compiled.stage_order,
        "pml_faces": compiled.pml_faces,
        "stable_pml_faces": compiled.stable_pml_faces,
        "absorber_faces": compiled.absorber_faces,
        "y_minus_boundary_type": compiled.y.minus.pml.boundary_type if compiled.y.minus.pml else None,
    }


def abc_boundary_snapshot() -> dict[str, object]:
    """Return a compact snapshot for the implemented first-order ABC subset."""

    compiled = compile_boundary_spec(
        BoundarySpec(x=Boundary.abc(permittivity=2.25, conductivity=0.01)),
        dt=1.0e-12,
        grid_spacing=(5.0e-7, 5.0e-7, 5.0e-7),
    )
    state = allocate_abc_boundary_state(compiled, (8, 4, 4, 3), dtype=np.float64)
    electric = np.zeros((8, 4, 4, 3), dtype=np.float64)
    electric[1, 1:3, 1:3, 0] = 1.0
    runtime = build_boundary_runtime(
        BoundarySpec(x=Boundary.abc(permittivity=2.25, conductivity=0.01)),
        dt=1.0e-12,
        grid_spacing=(5.0e-7, 5.0e-7, 5.0e-7),
        field_shape=electric.shape,
    )
    updated_electric, _ = runtime.apply(electric, np.zeros_like(electric))
    return {
        "abc_faces": compiled.abc_faces,
        "stage_order": compiled.stage_order,
        "reflection_coefficient": compiled.x.minus.abc.reflection_coefficient if compiled.x.minus.abc else None,
        "state_faces": tuple(sorted(state.electric)),
        "ghost_sample": float(updated_electric[0, 1, 1, 0]),
    }
