"""Warp kernel conventions and staged update namespace."""

from autofdtd.kernels.materials import (
    WARP_AVAILABLE,
    PoleResidueAuxiliaryState,
    allocate_pole_residue_state,
    constitutive_kernel_metadata,
    electric_constitutive_update,
    magnetic_constitutive_update,
    pole_residue_electric_update,
    warp_backend_available,
)

__all__ = [
    "PoleResidueAuxiliaryState",
    "WARP_AVAILABLE",
    "allocate_pole_residue_state",
    "constitutive_kernel_metadata",
    "electric_constitutive_update",
    "magnetic_constitutive_update",
    "pole_residue_electric_update",
    "warp_backend_available",
]
