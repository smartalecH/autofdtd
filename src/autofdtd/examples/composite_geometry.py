"""Composite geometry examples for grouped, transformed, and clipped layouts."""

from __future__ import annotations

import math

from autofdtd.api import (
    Box,
    ClipOperation,
    GeometryGroup,
    GeometryTransform,
    Sphere,
    Structure,
    Transformed,
)


def rotated_ring_resonator() -> Structure:
    """Return a simple composite structure that exercises all Phase 1 geometry wrappers."""
    ring = ClipOperation(
        operation="difference",
        geometry_a=Sphere(center=(0.0, 0.0, 0.0), radius=2.0),
        geometry_b=Sphere(center=(0.0, 0.0, 0.0), radius=1.4),
    )
    bus = Transformed(
        geometry=Box(center=(0.0, -2.4, 0.0), size=(4.0, 0.4, 0.3)),
        transform=GeometryTransform.rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 12.0),
    )
    return Structure(
        geometry=GeometryGroup(geometries=(ring, bus)),
        medium={"type": "Medium", "permittivity": 12.1104},
        name="rotated_ring_resonator",
    )
