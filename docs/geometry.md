# Geometry Scope

Phase 1 geometry support currently covers analytic axis-aligned primitives, a
high-priority `PolySlab` path for planar photonics, and the minimum wrapper
layer needed to assemble credible composite scenes.

## PolySlab

`PolySlab` is modeled as a simple 2D polygon extruded between `slab_bounds` on one
principal axis. This covers the common strip, rib, coupler, and etched-layout style
geometries that dominate early EM validation cases.

```python
from autofdtd.api import PolySlab, Structure

core = PolySlab(
    vertices=((-0.25, -0.11), (0.25, -0.11), (0.25, 0.11), (-0.25, 0.11)),
    slab_bounds=(-5.0, 5.0),
    axis="x",
)
structure = Structure(
    geometry=core,
    medium={"type": "Medium", "permittivity": 12.1104},
    name="core",
)
```

### Supported Phase 1 behavior

- simple polygons with three or more vertices
- convex or concave cross-sections
- explicit point containment and axis-aligned bounds
- typed IR lowering through `PolySlabIR`
- translation helpers for repeated planar layouts

### Explicit limitations

- `sidewall_angle` and `dilation` must be zero; tapered or offset extrusions are not yet
  implemented
- curved bulge edges, holes, polygon booleans, and self-intersecting outlines are rejected
- arbitrary transforms beyond rigid translation, rotation, and reflection are not implemented

## Composite Wrappers

Phase 1 now supports four wrapper families:

- `GeometryGroup` for bundling multiple shapes into one structure
- `Transformed` for rigid translation, rotation, or reflection of any supported geometry
- `ClipOperation` for the narrow boolean subset `union`, `intersection`, and `difference`
- `GeometryArray` for repeated placement with per-instance rigid transforms and offsets

```python
import math

from autofdtd.api import (
    Box,
    ClipOperation,
    GeometryArray,
    GeometryGroup,
    GeometryTransform,
    Structure,
    Transformed,
)

cross = GeometryGroup(
    geometries=(
        Box(center=(0, 0, 0), size=(4, 0.5, 0.22)),
        Transformed(
            geometry=Box(center=(0, 0, 0), size=(4, 0.5, 0.22)),
            transform=GeometryTransform.rotation(
                axis=(0, 0, 1),
                angle=math.pi / 2,
            ),
        ),
    )
)
frame = ClipOperation(
    operation="difference",
    geometry_a=Box(center=(0, 0, 0), size=(6, 6, 0.4)),
    geometry_b=Box(center=(0, 0, 0), size=(4, 4, 0.5)),
)
structure = Structure(
    geometry=GeometryGroup(geometries=(cross, frame)),
    medium={"type": "Medium", "permittivity": 3.4},
)
posts = GeometryArray(
    geometry=Box(center=(0, 0, 0), size=(0.3, 1.0, 0.22)),
    offsets=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    transforms=(
        GeometryTransform.identity(),
        GeometryTransform.rotation(axis=(0, 0, 1), angle=math.pi / 2),
    ),
)
```

### Wrapper limitations

- `ClipOperation` intentionally excludes `symmetric_difference`
- transform wrappers use the explicit `GeometryTransform` object-model rather than a free-form 4x4 matrix
- `GeometryArray.transforms` are rigid and linear-only in Phase 1; use `offsets` for per-instance translation
- boolean operations use analytic containment with conservative axis-aligned bounds; exact meshed booleans remain out of scope for Phase 1
- `GeometryArray` lowers into explicit grouped instances during IR compilation so structure precedence stays tied to ordinary ordered geometry materialization

Representative examples live in
[`src/autofdtd/examples/polyslab.py`](/workspace/autofdtd/src/autofdtd/examples/polyslab.py)
and
[`src/autofdtd/examples/composite_geometry.py`](/workspace/autofdtd/src/autofdtd/examples/composite_geometry.py),
and
[`src/autofdtd/examples/geometry_array.py`](/workspace/autofdtd/src/autofdtd/examples/geometry_array.py).
