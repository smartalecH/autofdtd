"""Geometry-family namespace for analytic Phase 1 geometry models."""

from collections.abc import Mapping

from autofdtd.geometry.composite import (
    ClipOperation,
    GeometryArray,
    GeometryGroup,
    Transformed,
    composite_geometry_model_from_value,
)
from autofdtd.geometry.euler_bend import EulerBend, euler_bend_from_value
from autofdtd.geometry.polyslab import PolySlab, polyslab_model_from_value
from autofdtd.geometry.primitives import (
    Box,
    Cylinder,
    GeometryTransform,
    PrimitiveGeometry,
    Sphere,
)
from autofdtd.geometry.primitives import (
    geometry_model_from_value as primitive_geometry_model_from_value,
)

GeometryModel = (
    Box | Sphere | Cylinder | PolySlab | EulerBend | GeometryGroup | Transformed | ClipOperation | GeometryArray
)


def geometry_model_from_value(value: object) -> GeometryModel:
    """Normalize a supported geometry payload into its typed public model."""
    if isinstance(value, PolySlab) or (
        isinstance(value, Mapping) and str(value.get("type")) == "PolySlab"
    ):
        return polyslab_model_from_value(value)
    if isinstance(value, EulerBend) or (
        isinstance(value, Mapping) and str(value.get("type")) == "EulerBend"
    ):
        return euler_bend_from_value(value)
    if isinstance(value, (GeometryGroup, Transformed, ClipOperation, GeometryArray)) or (
        isinstance(value, Mapping)
        and str(value.get("type"))
        in {"GeometryGroup", "Transformed", "ClipOperation", "GeometryArray"}
    ):
        return composite_geometry_model_from_value(value)
    return primitive_geometry_model_from_value(value)

__all__ = [
    "Box",
    "ClipOperation",
    "Cylinder",
    "EulerBend",
    "GeometryArray",
    "GeometryGroup",
    "GeometryModel",
    "GeometryTransform",
    "PolySlab",
    "PrimitiveGeometry",
    "Sphere",
    "Transformed",
    "composite_geometry_model_from_value",
    "euler_bend_from_value",
    "geometry_model_from_value",
    "polyslab_model_from_value",
]
