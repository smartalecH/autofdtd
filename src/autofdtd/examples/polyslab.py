"""Representative PolySlab examples for planar photonics cases."""

from __future__ import annotations

from autofdtd.api import PolySlab, Scene, Simulation, Structure


def strip_waveguide_simulation() -> Simulation:
    """Return a minimal straight-waveguide setup using a rectangular PolySlab core."""
    core = PolySlab(
        vertices=((-0.25, -0.11), (0.25, -0.11), (0.25, 0.11), (-0.25, 0.11)),
        slab_bounds=(-5.0, 5.0),
        axis="x",
    )
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(14.0, 4.0, 3.0),
        run_time=200.0,
        medium={"type": "Medium", "permittivity": 2.0736},
        structures=(
            Structure(
                geometry=core,
                medium={"type": "Medium", "permittivity": 12.1104},
                name="core",
            ),
        ),
    )


def directional_coupler_scene() -> Scene:
    """Return two parallel PolySlab waveguides suitable for overlap and precedence tests."""
    upper = PolySlab(
        vertices=((-2.0, -0.2), (2.0, -0.2), (2.0, 0.2), (-2.0, 0.2)),
        slab_bounds=(0.11, 0.33),
        axis="z",
    ).translate((0.0, 0.35, 0.0))
    lower = PolySlab(
        vertices=((-2.0, -0.2), (2.0, -0.2), (2.0, 0.2), (-2.0, 0.2)),
        slab_bounds=(0.11, 0.33),
        axis="z",
    ).translate((0.0, -0.35, 0.0))
    return Scene(
        medium={"type": "Medium", "permittivity": 2.0736},
        structures=(
            Structure(
                geometry=upper,
                medium={"type": "Medium", "permittivity": 12.1104},
                name="upper_core",
            ),
            Structure(
                geometry=lower,
                medium={"type": "Medium", "permittivity": 12.1104},
                name="lower_core",
            ),
        ),
    )


__all__ = ["directional_coupler_scene", "strip_waveguide_simulation"]
