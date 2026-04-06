"""Representative repeated-placement geometry examples."""

from __future__ import annotations

import math

from autofdtd.api import Box, GeometryArray, GeometryTransform, Scene, Structure


def rotated_post_array_scene() -> Scene:
    """Return a small repeated-post layout that exercises overlap precedence."""
    post = Box(center=(0.0, 0.0, 0.0), size=(0.3, 1.0, 0.3))
    transforms = (
        GeometryTransform.identity(),
        GeometryTransform.rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 2.0),
    )
    offsets = ((-0.9, 0.0, 0.0), (0.9, 0.0, 0.0))
    array = GeometryArray(geometry=post, offsets=offsets, transforms=transforms)
    slab = Box(center=(0.0, 0.0, 0.0), size=(3.0, 0.4, 0.3))
    return Scene(
        medium={"type": "Medium", "permittivity": 1.0},
        structures=(
            Structure(
                geometry=slab,
                medium={"type": "Medium", "permittivity": 2.25},
                name="slab",
                priority=0,
            ),
            Structure(
                geometry=array,
                medium={"type": "PECMedium"},
                name="posts",
            ),
        ),
    )


__all__ = ["rotated_post_array_scene"]
