"""Small material-compilation examples."""

from __future__ import annotations

from autofdtd.api import (
    AnisotropicMedium,
    Box,
    Debye,
    Drude,
    Lorentz,
    Medium,
    PECMedium,
    PoleResidue,
    Scene,
    Sellmeier,
    Structure,
    StructurePriorityMode,
)
from autofdtd.compiler import (
    compile_anisotropic_medium_coefficients,
    compile_debye_coefficients,
    compile_drude_coefficients,
    compile_isotropic_medium_coefficients,
    compile_lorentz_coefficients,
    compile_pole_residue_coefficients,
    compile_sellmeier_coefficients,
    sample_scene_mediums,
)


def build_material_scene() -> Scene:
    """Build a minimal scene with dielectric background and a PEC inclusion."""

    return Scene(
        medium=Medium(permittivity=2.25),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)),
                medium=PECMedium(name="shield"),
                name="shield",
            ),
        ),
    )


def compiled_material_example() -> dict[str, object]:
    """Return a JSON-ready snapshot of point sampling and coefficient preparation."""

    scene = build_material_scene()
    samples = sample_scene_mediums(scene, points=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    dielectric_coeffs = compile_isotropic_medium_coefficients(scene.medium, dt=1e-12)
    return {
        "samples": [sample.to_payload() for sample in samples],
        "dielectric_coefficients": dielectric_coeffs.to_payload(),
    }


def compiled_pole_residue_example() -> dict[str, object]:
    """Return a JSON-ready snapshot of PoleResidue coefficient preparation."""

    medium = PoleResidue(
        name="fit",
        eps_inf=2.1,
        poles=(((-1.0e13, 2.0e13), (3.0e11, -4.0e11)),),
    )
    coefficients = compile_pole_residue_coefficients(medium, dt=1e-12)
    return {
        "medium": medium.to_payload(),
        "compiled": coefficients.to_payload(),
    }


def compiled_sellmeier_example() -> dict[str, object]:
    """Return a JSON-ready snapshot of Sellmeier lowering onto the dispersive runtime path."""

    medium = Sellmeier(name="fused-silica-like", coeffs=((0.6961663, 4.67914825849e-15),))
    coefficients = compile_sellmeier_coefficients(medium, dt=1e-12)
    return {
        "medium": medium.to_payload(),
        "pole_residue": medium.to_pole_residue().to_payload(),
        "compiled": coefficients.to_payload(),
    }


def compiled_lorentz_drude_debye_examples() -> dict[str, object]:
    """Return JSON-ready snapshots for the analytical dispersive families."""

    lorentz = Lorentz(name="lorentz-fit", eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),))
    drude = Drude(name="drude-fit", eps_inf=1.0, coeffs=((180e12, 8e12),))
    debye = Debye(name="debye-fit", eps_inf=2.1, coeffs=((1.4, 5.0e-12),))
    return {
        "lorentz": {
            "medium": lorentz.to_payload(),
            "pole_residue": lorentz.to_pole_residue().to_payload(),
            "compiled": compile_lorentz_coefficients(lorentz, dt=1e-12).to_payload(),
        },
        "drude": {
            "medium": drude.to_payload(),
            "pole_residue": drude.to_pole_residue().to_payload(),
            "compiled": compile_drude_coefficients(drude, dt=1e-12).to_payload(),
        },
        "debye": {
            "medium": debye.to_payload(),
            "pole_residue": debye.to_pole_residue().to_payload(),
            "compiled": compile_debye_coefficients(debye, dt=1e-12).to_payload(),
        },
    }


def compiled_anisotropic_example() -> dict[str, object]:
    """Return a JSON-ready snapshot of diagonal anisotropic coefficient preparation."""

    medium = AnisotropicMedium(
        name="uniaxial-like",
        xx=Medium(permittivity=2.0),
        yy=PoleResidue(eps_inf=2.5, poles=(((-1.0e13, 0.0), (2.0e11, 0.0)),)),
        zz=PECMedium(name="cap"),
    )
    coefficients = compile_anisotropic_medium_coefficients(medium, dt=1e-12)
    return {
        "medium": medium.to_payload(),
        "compiled": coefficients.to_payload(),
    }
