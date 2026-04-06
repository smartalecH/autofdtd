"""Example subpixel-policy configuration for dielectric photonics scenes."""

from __future__ import annotations

from autofdtd.api import (
    Box,
    GridSpec,
    PolarizedAveraging,
    Simulation,
    Staircasing,
    Structure,
    SubpixelSpec,
)


def build_subpixel_demo() -> Simulation:
    """Return a small simulation using explicit Phase 1 subpixel controls."""
    return Simulation(
        size=(4.0, 2.0, 2.0),
        run_time=20.0,
        structures=(
            Structure(
                name="core",
                geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 0.5, 0.22)),
                medium={"type": "Medium", "permittivity": 12.0},
            ),
        ),
        grid_spec=GridSpec.uniform(0.05),
        subpixel=SubpixelSpec(
            dielectric=PolarizedAveraging(),
            metal=Staircasing(),
            pec=Staircasing(),
            pmc=Staircasing(),
            lossy_metal=Staircasing(),
        ),
    )


if __name__ == "__main__":
    simulation = build_subpixel_demo()
    print(simulation.to_json_text())
