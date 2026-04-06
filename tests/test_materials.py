from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    AnisotropicMedium,
    Box,
    CustomAnisotropicMedium,
    CustomMedium,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    Lorentz,
    LossyMetalMedium,
    Medium,
    Medium2D,
    PECMedium,
    PMCMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    PoleResidue,
    Scene,
    Sellmeier,
    Simulation,
    Structure,
    StructurePriorityMode,
)
from autofdtd.compiler import (
    AnisotropicMaterialCoefficients,
    EPSILON_0,
    MU_0,
    ConstitutiveMode,
    PoleResidueMaterialCoefficients,
    compile_anisotropic_medium_coefficients,
    compile_medium_coefficients,
    compile_debye_coefficients,
    compile_drude_coefficients,
    compile_isotropic_medium_coefficients,
    compile_lorentz_coefficients,
    compile_pole_residue_coefficients,
    compile_scene_medium_coefficients,
    compile_sellmeier_coefficients,
    sample_scene_mediums,
)
from autofdtd.materials import medium_model_from_value
from autofdtd.ir import (
    AnisotropicMediumIR,
    DebyeIR,
    DrudeIR,
    LorentzIR,
    PECMediumIR,
    PMCMediumIR,
    PoleResidueIR,
    SellmeierIR,
    simulation_to_ir,
)
from autofdtd.kernels import (
    allocate_anisotropic_state,
    anisotropic_electric_update,
    anisotropic_magnetic_update,
    allocate_pole_residue_state,
    constitutive_kernel_metadata,
    electric_constitutive_update,
    magnetic_constitutive_update,
    pole_residue_electric_update,
)


def test_isotropic_medium_models_validate_and_serialize() -> None:
    medium = Medium(
        name=" core ",
        permittivity=3.45,
        conductivity=0.1,
        permeability=1.2,
        magnetic_conductivity=0.2,
    )
    pec = PECMedium(name=" metal ")
    pmc = PMCMedium(name=" wall ")

    assert medium.name == "core"
    assert medium.to_payload()["permittivity"] == pytest.approx(3.45)
    assert pec.name == "metal"
    assert pmc.name == "wall"

    with pytest.raises(ValueError, match="permittivity must be a positive finite value"):
        Medium(permittivity=0.0)

    with pytest.raises(ValueError, match="conductivity must be a non-negative finite value"):
        Medium(conductivity=-1.0)


def test_anisotropic_medium_models_validate_serialize_and_eval_diagonal_response() -> None:
    medium = AnisotropicMedium(
        name=" crystal ",
        xx=Medium(permittivity=2.0),
        yy=PoleResidue(eps_inf=2.5, poles=(((-1.0e12, 0.0), (1.0e10, 0.0)),)),
        zz=PECMedium(name=" cap "),
    )

    diagonal = medium.eps_diagonal(200e12)

    assert medium.name == "crystal"
    assert medium.component_types == ("Medium", "PoleResidue", "PECMedium")
    assert diagonal[0].real == pytest.approx(2.0)
    assert diagonal[2] == pytest.approx(0.0j)


def test_fully_anisotropic_medium_and_medium2d_expose_explicit_phase1_policy() -> None:
    tensor = FullyAnisotropicMedium(
        name=" tensor ",
        permittivity=((2.0, 0.1, 0.0), (0.1, 2.5, 0.0), (0.0, 0.0, 1.7)),
    )
    sheet = Medium2D(
        name=" sheet ",
        ss=Medium(permittivity=1.5, conductivity=0.2),
        tt=Medium(permittivity=1.8, conductivity=0.1),
    )

    assert tensor.name == "tensor"
    assert "deferred" in tensor.phase1_policy
    assert sheet.name == "sheet"
    assert "not a directly supported simulation medium" in sheet.phase1_policy

    with pytest.raises(ValueError, match="positive"):
        FullyAnisotropicMedium(permittivity=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    with pytest.raises(ValueError, match="both PECMedium or both non-PEC"):
        Medium2D(ss=PECMedium(), tt=Medium())


def test_advanced_material_policy_models_parse_and_expose_explicit_phase1_guidance() -> None:
    lossy = LossyMetalMedium(
        name=" metal ",
        conductivity=5.0e6,
        frequency_range=(8.0e9, 12.0e9),
        roughness=5.0e-9,
    )
    perturbation = PerturbationMedium(
        name=" thermo ",
        permittivity=2.5,
        permittivity_perturbation={"heat": {"coeff": 1.0e-4}},
    )
    dispersive_perturbation = PerturbationPoleResidue(
        name=" tuned ",
        eps_inf=2.0,
        poles=(((-1.0e12, 0.0), (1.0e10, 0.0)),),
        eps_inf_perturbation={"heat": {"coeff": 2.0e-4}},
    )
    custom = CustomMedium(name=" grin ", permittivity={"dataset": "eps"})
    custom_tensor = CustomAnisotropicMedium(
        name=" tensor ",
        xx={"type": "CustomMedium", "permittivity": {"dataset": "xx"}},
        yy={"type": "CustomMedium", "permittivity": {"dataset": "yy"}},
        zz={"type": "CustomMedium", "permittivity": {"dataset": "zz"}},
    )

    assert lossy.name == "metal"
    assert "deferred" in lossy.phase1_policy
    assert lossy.to_medium_approximation().conductivity == pytest.approx(5.0e6)
    assert perturbation.name == "thermo"
    assert "not execute" in perturbation.phase1_policy
    assert perturbation.base_medium().permittivity == pytest.approx(2.5)
    assert "not execute" in dispersive_perturbation.phase1_policy
    assert dispersive_perturbation.base_medium().eps_inf == pytest.approx(2.0)
    assert "compatibility inspection" in custom.phase1_policy
    assert "compatibility inspection" in custom_tensor.phase1_policy

    with pytest.raises(ValueError, match="strictly increasing"):
        LossyMetalMedium(conductivity=5.0e6, frequency_range=(12.0e9, 8.0e9))


def test_medium2d_conversion_helpers_lower_to_phase1_supported_media() -> None:
    sheet = Medium2D(
        ss=Medium(permittivity=1.0, conductivity=0.4),
        tt=Sellmeier(coeffs=((0.5, 3.0e-15),)),
    )

    volumetric = sheet.to_anisotropic_medium(axis="z", thickness=5.0e-9)
    pole_residue = sheet.to_pole_residue(thickness=5.0e-9)

    assert isinstance(volumetric, AnisotropicMedium)
    assert isinstance(volumetric.xx, Medium)
    assert isinstance(volumetric.yy, PoleResidue)
    assert isinstance(volumetric.zz, Medium)
    assert pole_residue.eps_inf >= 1.0
    assert pole_residue.num_poles >= 1


def test_pole_residue_model_validates_and_matches_eps_formula() -> None:
    medium = PoleResidue(
        name=" glass ",
        eps_inf=2.25,
        poles=(((-1.0, 2.0), (3.0, 4.0)),),
    )
    frequency = 200e12
    omega = 2.0 * np.pi * frequency
    pole = complex(-1.0, 2.0)
    residue = complex(3.0, 4.0)
    expected = 2.25 - residue / (1j * omega + pole) - residue.conjugate() / (
        1j * omega + pole.conjugate()
    )

    assert medium.name == "glass"
    assert medium.num_poles == 1
    assert medium.eps_model(frequency) == pytest.approx(expected)

    with pytest.raises(ValueError, match="Re\\(a\\) <= 0"):
        PoleResidue(poles=(((1.0, 0.0), (1.0, 0.0)),))


def test_pole_residue_round_trips_conductivity_only_medium() -> None:
    medium = Medium(name="lossy", permittivity=4.0, conductivity=0.2)
    converted = PoleResidue.from_medium(medium)

    assert converted.eps_inf == pytest.approx(4.0)
    assert converted.to_medium().conductivity == pytest.approx(0.2)
    assert converted.to_medium().permittivity == pytest.approx(4.0)


def test_sellmeier_model_validates_matches_formula_and_lowers_to_pole_residue() -> None:
    medium = Sellmeier(
        name=" silica ",
        coeffs=((0.6961663, 4.67914825849e-15), (0.4079426, 0.0)),
    )
    frequency = 200e12
    wavelength = 299_792_458.0 / frequency
    wavelength_squared = wavelength * wavelength
    expected = 1.0
    for b_coeff, c_coeff in medium.coeffs:
        if c_coeff == 0.0:
            expected += b_coeff
        else:
            expected += b_coeff * wavelength_squared / (wavelength_squared - c_coeff)

    pole_residue = medium.to_pole_residue()

    assert medium.name == "silica"
    assert medium.num_terms == 2
    assert medium.eps_inf == pytest.approx(1.4079426)
    assert medium.eps_model(frequency) == pytest.approx(expected)
    assert pole_residue.eps_inf == pytest.approx(1.4079426)
    assert pole_residue.num_poles == 1
    assert pole_residue.eps_model(frequency).real == pytest.approx(expected)
    assert pole_residue.eps_model(frequency).imag == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="B >= 0"):
        Sellmeier(coeffs=((-1.0, 1.0e-12),))

    with pytest.raises(ValueError, match="C >= 0"):
        Sellmeier(coeffs=((1.0, -1.0e-12),))


def test_sellmeier_from_dispersion_builds_single_pole_passive_fit() -> None:
    medium = Sellmeier.from_dispersion(n=1.45, freq=193.5e12, dn_dwvl=-4.0e4, name="fit")

    assert medium.name == "fit"
    assert medium.num_terms == 1
    assert medium.coeffs[0][0] > 0.0
    assert medium.coeffs[0][1] > 0.0

    with pytest.raises(ValueError, match="dn_dwvl must be negative"):
        Sellmeier.from_dispersion(n=1.45, freq=193.5e12, dn_dwvl=0.0)


def test_lorentz_model_validates_matches_formula_and_lowers_to_pole_residue() -> None:
    medium = Lorentz(name=" lorentz ", eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),))
    frequency = 200e12
    expected = 1.8 + (0.6 * (220e12) ** 2) / (
        (220e12) ** 2 - frequency**2 - 2.0j * frequency * 12e12
    )
    pole_residue = medium.to_pole_residue()

    assert medium.name == "lorentz"
    assert medium.num_terms == 1
    assert medium.eps_model(frequency) == pytest.approx(expected)
    assert pole_residue.eps_model(frequency) == pytest.approx(expected)

    with pytest.raises(ValueError, match="delta_eps >= 0"):
        Lorentz(coeffs=((-1.0, 220e12, 12e12),))

    with pytest.raises(ValueError, match="frequency\\^2 != damping\\^2"):
        Lorentz(coeffs=((0.5, 10.0, 10.0),))


def test_drude_model_validates_matches_formula_and_lowers_to_pole_residue() -> None:
    medium = Drude(name=" drude ", eps_inf=1.2, coeffs=((180e12, 8e12),))
    frequency = 200e12
    expected = 1.2 - (180e12) ** 2 / (frequency**2 + 1.0j * frequency * 8e12)
    pole_residue = medium.to_pole_residue()

    assert medium.name == "drude"
    assert medium.num_terms == 1
    assert medium.eps_model(frequency) == pytest.approx(expected)
    assert pole_residue.eps_model(frequency) == pytest.approx(expected)

    with pytest.raises(ValueError, match="damping must be a positive finite value"):
        Drude(coeffs=((180e12, 0.0),))


def test_debye_model_validates_matches_formula_and_lowers_to_pole_residue() -> None:
    medium = Debye(name=" debye ", eps_inf=2.1, coeffs=((1.4, 5.0e-12),))
    frequency = 200e12
    expected = 2.1 + 1.4 / (1.0 - 1.0j * frequency * 5.0e-12)
    pole_residue = medium.to_pole_residue()

    assert medium.name == "debye"
    assert medium.num_terms == 1
    assert medium.eps_model(frequency) == pytest.approx(expected)
    assert pole_residue.eps_model(frequency) == pytest.approx(expected)

    with pytest.raises(ValueError, match="delta_eps >= 0"):
        Debye(coeffs=((-1.0, 5.0e-12),))


def test_scene_material_sampling_respects_precedence_and_background() -> None:
    scene = Scene(
        medium=Medium(permittivity=2.25),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0)),
                medium=Medium(permittivity=3.4),
                name="core",
            ),
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                medium=PECMedium(name="via"),
                name="via",
            ),
        ),
    )

    samples = sample_scene_mediums(
        scene,
        points=((0.0, 0.0, 0.0), (1.6, 0.0, 0.0), (5.0, 0.0, 0.0)),
    )

    assert [sample.medium_type for sample in samples] == ["PECMedium", "Medium", "Medium"]
    assert samples[0].structure_name == "via"
    assert samples[1].structure_name == "core"
    assert samples[2].structure_name is None


def test_isotropic_coefficient_compilation_matches_fdtd_formulas() -> None:
    dt = 2.5e-12
    medium = Medium(
        permittivity=4.0,
        conductivity=0.05,
        permeability=1.5,
        magnetic_conductivity=0.2,
    )

    coeffs = compile_isotropic_medium_coefficients(medium, dt=dt)

    electric_ratio = medium.conductivity * dt / (2.0 * EPSILON_0 * medium.permittivity)
    magnetic_ratio = medium.magnetic_conductivity * dt / (2.0 * MU_0 * medium.permeability)

    assert coeffs.electric_mode is ConstitutiveMode.STANDARD
    assert coeffs.electric_decay == pytest.approx((1.0 - electric_ratio) / (1.0 + electric_ratio))
    assert coeffs.electric_drive == pytest.approx(
        (dt / (EPSILON_0 * medium.permittivity)) / (1.0 + electric_ratio)
    )
    assert coeffs.magnetic_decay == pytest.approx((1.0 - magnetic_ratio) / (1.0 + magnetic_ratio))
    assert coeffs.magnetic_drive == pytest.approx(
        (dt / (MU_0 * medium.permeability)) / (1.0 + magnetic_ratio)
    )


def test_pole_residue_coefficient_compilation_prepares_auxiliary_terms() -> None:
    dt = 1.5e-12
    medium = PoleResidue(
        eps_inf=2.5,
        poles=(((-1.0e13, 2.0e13), (3.0e11, -4.0e11)), ((0.0, 0.0), (1.0e10, 0.0))),
    )

    coeffs = compile_pole_residue_coefficients(medium, dt=dt)

    assert isinstance(coeffs, PoleResidueMaterialCoefficients)
    assert coeffs.eps_inf == pytest.approx(2.5)
    assert coeffs.num_poles == 2
    assert coeffs.auxiliary_layout == "complex_polarization_per_pole"
    assert coeffs.electric_drive == pytest.approx(dt / (EPSILON_0 * medium.eps_inf))

    first_decay = complex(*coeffs.poles[0].decay)
    first_drive = complex(*coeffs.poles[0].drive)
    pole = complex(-1.0e13, 2.0e13)
    residue = complex(3.0e11, -4.0e11)
    expected_decay = np.exp(-pole * dt)
    expected_drive = -EPSILON_0 * residue * ((1.0 - expected_decay) / pole)
    assert first_decay == pytest.approx(expected_decay)
    assert first_drive == pytest.approx(expected_drive)


def test_anisotropic_coefficient_compilation_prepares_per_axis_branches() -> None:
    dt = 1.5e-12
    medium = AnisotropicMedium(
        xx=Medium(permittivity=2.0, conductivity=0.1),
        yy=PoleResidue(eps_inf=2.5, poles=(((-1.0e13, 0.0), (2.0e11, 0.0)),)),
        zz=PECMedium(),
    )

    coeffs = compile_anisotropic_medium_coefficients(medium, dt=dt)

    assert isinstance(coeffs, AnisotropicMaterialCoefficients)
    assert coeffs.component_medium_types == ("Medium", "PoleResidue", "PECMedium")
    assert coeffs.xx.electric_mode is ConstitutiveMode.STANDARD
    assert coeffs.yy.medium_type == "PoleResidue"
    assert coeffs.zz.electric_mode is ConstitutiveMode.CLAMP_ZERO


def test_sellmeier_compilation_reuses_pole_residue_auxiliary_contract() -> None:
    dt = 1.5e-12
    medium = Sellmeier(coeffs=((0.6961663, 4.67914825849e-15), (0.4079426, 0.0)))

    coeffs = compile_sellmeier_coefficients(medium, dt=dt)
    equivalent = compile_pole_residue_coefficients(medium.to_pole_residue(), dt=dt)

    assert isinstance(coeffs, PoleResidueMaterialCoefficients)
    assert coeffs.medium_type == "Sellmeier"
    assert coeffs.eps_inf == pytest.approx(medium.eps_inf)
    assert coeffs.num_poles == 1
    assert coeffs.electric_drive == pytest.approx(equivalent.electric_drive)
    assert coeffs.poles[0].decay == pytest.approx(equivalent.poles[0].decay)
    assert coeffs.poles[0].drive == pytest.approx(equivalent.poles[0].drive)


@pytest.mark.parametrize(
    ("medium", "compiler"),
    [
        (Lorentz(eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),)), compile_lorentz_coefficients),
        (Drude(eps_inf=1.2, coeffs=((180e12, 8e12),)), compile_drude_coefficients),
        (Debye(eps_inf=2.1, coeffs=((1.4, 5.0e-12),)), compile_debye_coefficients),
    ],
)
def test_analytical_dispersion_compilation_reuses_pole_residue_auxiliary_contract(
    medium: Lorentz | Drude | Debye,
    compiler,
) -> None:
    dt = 1.5e-12
    coeffs = compiler(medium, dt=dt)
    equivalent = compile_pole_residue_coefficients(medium.to_pole_residue(), dt=dt)

    assert isinstance(coeffs, PoleResidueMaterialCoefficients)
    assert coeffs.medium_type == medium.type
    assert coeffs.electric_drive == pytest.approx(equivalent.electric_drive)
    assert coeffs.num_poles == equivalent.num_poles
    for compiled_term, equivalent_term in zip(coeffs.poles, equivalent.poles, strict=True):
        assert compiled_term.decay == pytest.approx(equivalent_term.decay)
        assert compiled_term.drive == pytest.approx(equivalent_term.drive)


def test_debye_compilation_rejects_timestep_stiffer_than_phase1_exact_hold_path() -> None:
    with pytest.raises(ValueError, match="too stiff for the current timestep"):
        compile_debye_coefficients(Debye(eps_inf=2.1, coeffs=((1.4, 2.5e-15),)), dt=1e-12)


def test_pec_and_pmc_compile_to_explicit_clamp_modes() -> None:
    pec = compile_isotropic_medium_coefficients(PECMedium(), dt=1e-12)
    pmc = compile_isotropic_medium_coefficients(PMCMedium(), dt=1e-12)

    assert pec.electric_mode is ConstitutiveMode.CLAMP_ZERO
    assert pec.electric_drive == 0.0
    assert pec.magnetic_mode is ConstitutiveMode.STANDARD
    assert pmc.magnetic_mode is ConstitutiveMode.CLAMP_ZERO
    assert pmc.magnetic_drive == 0.0


def test_scene_coefficient_compilation_attaches_compiled_coefficients() -> None:
    scene = Scene(
        medium=Medium(permittivity=1.5),
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                medium=PoleResidue(eps_inf=2.0, poles=(((-1.0e12, 0.0), (5.0e10, 0.0)),)),
                name="dispersive",
            ),
        ),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
    )

    compiled = compile_scene_medium_coefficients(
        scene,
        points=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        dt=1e-12,
    )

    assert compiled[0].structure_name == "dispersive"
    assert compiled[0].medium.medium_type == "PoleResidue"
    assert compiled[1].medium.permittivity == pytest.approx(1.5)


def test_scene_coefficient_compilation_preserves_sellmeier_family_tag() -> None:
    scene = Scene(
        medium=Medium(permittivity=1.5),
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                medium=Sellmeier(coeffs=((0.6961663, 4.67914825849e-15),)),
                name="glass",
            ),
        ),
    )

    compiled = compile_scene_medium_coefficients(scene, points=((0.0, 0.0, 0.0),), dt=1e-12)

    assert compiled[0].medium.medium_type == "Sellmeier"


def test_medium2d_and_fully_anisotropic_raise_explicit_phase1_runtime_errors() -> None:
    with pytest.raises(ValueError, match="Medium2D is not a directly supported Phase 1 runtime medium"):
        compile_medium_coefficients(Medium2D(ss=Medium(), tt=Medium()), dt=1e-12)

    with pytest.raises(ValueError, match="runtime compilation is deferred"):
        compile_medium_coefficients(FullyAnisotropicMedium(), dt=1e-12)


def test_advanced_material_policy_buckets_raise_explicit_runtime_errors() -> None:
    with pytest.raises(ValueError, match="LossyMetalMedium parsing is available"):
        compile_medium_coefficients(
            LossyMetalMedium(conductivity=5.0e6, frequency_range=(8.0e9, 12.0e9)),
            dt=1e-12,
        )

    with pytest.raises(ValueError, match="PerturbationMedium parsing is available"):
        compile_medium_coefficients(
            PerturbationMedium(permittivity=2.5, permittivity_perturbation={"heat": {"coeff": 1.0e-4}}),
            dt=1e-12,
        )

    with pytest.raises(ValueError, match="custom-media reject bucket"):
        compile_medium_coefficients(
            CustomMedium(permittivity={"dataset": "eps"}),
            dt=1e-12,
        )


def test_typed_medium_ir_is_emitted_for_simulations() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        medium=PoleResidue(eps_inf=2.0, poles=(((-1.0e12, 0.0), (1.0e10, 0.0)),)),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
                medium=PECMedium(),
                name="pec",
                background_medium=PMCMedium(),
            ),
        ),
    )
    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.scene.background_medium, PoleResidueIR)
    assert isinstance(simulation_ir.scene.structures[0].medium, PECMediumIR)
    assert isinstance(simulation_ir.scene.structures[0].background_medium, PMCMediumIR)


def test_anisotropic_medium_ir_is_emitted_for_simulations() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        medium=AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=Sellmeier(coeffs=((0.5, 3.0e-15),)),
            zz=PECMedium(),
        ),
    )

    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.scene.background_medium, AnisotropicMediumIR)
    assert simulation_ir.scene.background_medium.yy.component_type == "Sellmeier"


def test_medium2d_and_fully_anisotropic_are_rejected_by_simulation_validation() -> None:
    with pytest.raises(Exception, match="Medium2D"):
        Simulation(size=(4.0, 4.0, 4.0), run_time=1.0, medium=Medium2D(ss=Medium(), tt=Medium()))

    with pytest.raises(Exception, match="FullyAnisotropicMedium"):
        Simulation(size=(4.0, 4.0, 4.0), run_time=1.0, medium=FullyAnisotropicMedium())


def test_deferred_and_rejected_advanced_media_are_mapped_by_validation_instead_of_unknown() -> None:
    with pytest.raises(Exception, match="deferred feature 'LossyMetalMedium'"):
        Simulation(
            size=(4.0, 4.0, 4.0),
            run_time=1.0,
            medium=LossyMetalMedium(conductivity=5.0e6, frequency_range=(8.0e9, 12.0e9)),
        )

    with pytest.raises(Exception, match="deferred feature 'PerturbationMedium'"):
        Simulation(
            size=(4.0, 4.0, 4.0),
            run_time=1.0,
            medium=PerturbationMedium(permittivity=2.5, permittivity_perturbation={"heat": {"coeff": 1.0e-4}}),
        )

    with pytest.raises(Exception, match="rejected clearly feature 'CustomMedium'"):
        Simulation(
            size=(4.0, 4.0, 4.0),
            run_time=1.0,
            medium=CustomMedium(permittivity={"dataset": "eps"}),
        )

    with pytest.raises(Exception, match="rejected clearly feature 'CustomPoleResidue'"):
        Simulation(
            size=(4.0, 4.0, 4.0),
            run_time=1.0,
            medium={"type": "CustomPoleResidue", "eps_dataset": "eps"},
        )


def test_medium_model_from_value_parses_explicit_advanced_policy_surfaces() -> None:
    assert isinstance(
        medium_model_from_value(
            {"type": "LossyMetalMedium", "conductivity": 5.0e6, "frequency_range": (8.0e9, 12.0e9)}
        ),
        LossyMetalMedium,
    )
    assert isinstance(
        medium_model_from_value(
            {"type": "PerturbationPoleResidue", "eps_inf": 2.0, "poles": (((-1.0e12, 0.0), (1.0e10, 0.0)),)}
        ),
        PerturbationPoleResidue,
    )
    parsed = medium_model_from_value({"type": "CustomPoleResidue", "eps_dataset": "eps"})
    assert parsed.type == "CustomPoleResidue"


def test_sellmeier_ir_is_emitted_for_simulations() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        medium=Sellmeier(coeffs=((0.6961663, 4.67914825849e-15),)),
    )

    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.scene.background_medium, SellmeierIR)


@pytest.mark.parametrize(
    ("medium", "expected_type"),
    [
        (Lorentz(eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),)), LorentzIR),
        (Drude(eps_inf=1.2, coeffs=((180e12, 8e12),)), DrudeIR),
        (Debye(eps_inf=2.1, coeffs=((1.4, 5.0e-12),)), DebyeIR),
    ],
)
def test_analytical_dispersion_ir_is_emitted_for_simulations(
    medium: Lorentz | Drude | Debye,
    expected_type,
) -> None:
    simulation = Simulation(size=(4.0, 4.0, 4.0), run_time=1.0, medium=medium)

    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.scene.background_medium, expected_type)


def test_constitutive_kernel_helpers_apply_standard_and_clamp_paths() -> None:
    coeffs = compile_isotropic_medium_coefficients(Medium(permittivity=2.0), dt=1e-12)
    electric = np.array([1.0, -2.0, 0.5])
    magnetic = np.array([0.2, -0.1, 0.4])
    curl_h = np.array([0.1, 0.3, -0.2])
    curl_e = np.array([-0.5, 0.2, 0.1])

    updated_e = electric_constitutive_update(electric, curl_h, coeffs)
    updated_h = magnetic_constitutive_update(magnetic, curl_e, coeffs)

    assert np.allclose(updated_e, coeffs.electric_decay * electric + coeffs.electric_drive * curl_h)
    assert np.allclose(updated_h, coeffs.magnetic_decay * magnetic - coeffs.magnetic_drive * curl_e)

    pec = compile_isotropic_medium_coefficients(PECMedium(), dt=1e-12)
    pmc = compile_isotropic_medium_coefficients(PMCMedium(), dt=1e-12)
    assert np.allclose(electric_constitutive_update(electric, curl_h, pec), 0.0)
    assert np.allclose(magnetic_constitutive_update(magnetic, curl_e, pmc), 0.0)


def test_pole_residue_kernel_update_tracks_auxiliary_state_and_current() -> None:
    coeffs = compile_pole_residue_coefficients(
        PoleResidue(eps_inf=2.0, poles=(((-2.0e12, 0.0), (8.0e10, 1.0e10)),)),
        dt=1e-12,
    )
    electric = np.array([1.0, -0.5, 0.25])
    curl_h = np.array([0.1, 0.2, -0.1])
    state = allocate_pole_residue_state(coeffs, field_shape=electric.shape)

    updated_e, updated_state, polarization_current = pole_residue_electric_update(
        electric,
        curl_h,
        coeffs,
        state,
        dt=1e-12,
    )

    decay = complex(*coeffs.poles[0].decay)
    drive = complex(*coeffs.poles[0].drive)
    expected_p = decay * 0.0 + drive * electric
    expected_current = 2.0 * np.real(expected_p / 1e-12)
    baseline = coeffs.electric_decay * electric + coeffs.electric_drive * curl_h

    assert np.allclose(updated_state.polarization[0], expected_p)
    assert np.allclose(polarization_current, expected_current)
    assert np.allclose(updated_e, baseline - coeffs.electric_drive * expected_current)


def test_sellmeier_kernel_update_uses_shared_dispersive_path() -> None:
    coeffs = compile_sellmeier_coefficients(
        Sellmeier(coeffs=((0.6961663, 4.67914825849e-15),)),
        dt=1e-12,
    )
    electric = np.array([1.0, -0.5, 0.25])
    curl_h = np.array([0.1, 0.2, -0.1])
    state = allocate_pole_residue_state(coeffs, field_shape=electric.shape)

    updated_e, updated_state, polarization_current = pole_residue_electric_update(
        electric,
        curl_h,
        coeffs,
        state,
        dt=1e-12,
    )

    expected_p = complex(*coeffs.poles[0].drive) * electric
    baseline = coeffs.electric_decay * electric + coeffs.electric_drive * curl_h

    assert coeffs.medium_type == "Sellmeier"
    assert np.allclose(updated_state.polarization[0], expected_p)
    assert np.allclose(updated_e, baseline - coeffs.electric_drive * polarization_current)


def test_anisotropic_kernel_updates_apply_axis_specific_paths() -> None:
    coeffs = compile_anisotropic_medium_coefficients(
        AnisotropicMedium(
            xx=Medium(permittivity=2.0),
            yy=PoleResidue(eps_inf=2.5, poles=(((-2.0e12, 0.0), (8.0e10, 0.0)),)),
            zz=PECMedium(),
        ),
        dt=1e-12,
    )
    electric = np.array([[1.0, -0.5, 0.25], [0.4, 0.2, -0.1]])
    magnetic = np.array([[0.3, -0.1, 0.5], [0.2, 0.4, -0.2]])
    curl_h = np.array([[0.1, 0.2, -0.1], [0.05, -0.3, 0.4]])
    curl_e = np.array([[0.2, -0.4, 0.1], [-0.3, 0.2, 0.5]])
    state = allocate_anisotropic_state(coeffs, field_shape=electric.shape[:-1])

    updated_e, updated_state, polarization_current = anisotropic_electric_update(
        electric,
        curl_h,
        coeffs,
        state,
        dt=1e-12,
    )
    updated_h = anisotropic_magnetic_update(magnetic, curl_e, coeffs)

    baseline_x = coeffs.xx.electric_decay * electric[:, 0] + coeffs.xx.electric_drive * curl_h[:, 0]
    baseline_y = coeffs.yy.electric_decay * electric[:, 1] + coeffs.yy.electric_drive * curl_h[:, 1]
    baseline_z = np.zeros_like(electric[:, 2])

    assert np.allclose(updated_e[:, 0], baseline_x)
    assert np.all(updated_state.yy.polarization[0] != 0.0)
    assert np.allclose(updated_e[:, 1], baseline_y - coeffs.yy.electric_drive * polarization_current[:, 1])
    assert np.allclose(updated_e[:, 2], baseline_z)
    assert updated_h.shape == magnetic.shape


@pytest.mark.parametrize(
    ("medium", "compiler"),
    [
        (Lorentz(eps_inf=1.8, coeffs=((0.6, 220e12, 12e12),)), compile_lorentz_coefficients),
        (Drude(eps_inf=1.2, coeffs=((180e12, 8e12),)), compile_drude_coefficients),
        (Debye(eps_inf=2.1, coeffs=((1.4, 5.0e-12),)), compile_debye_coefficients),
    ],
)
def test_analytical_dispersion_kernel_update_uses_shared_dispersive_path(
    medium: Lorentz | Drude | Debye,
    compiler,
) -> None:
    coeffs = compiler(medium, dt=1e-12)
    electric = np.array([1.0, -0.5, 0.25])
    curl_h = np.array([0.1, 0.2, -0.1])
    state = allocate_pole_residue_state(coeffs, field_shape=electric.shape)

    updated_e, updated_state, polarization_current = pole_residue_electric_update(
        electric,
        curl_h,
        coeffs,
        state,
        dt=1e-12,
    )

    baseline = coeffs.electric_decay * electric + coeffs.electric_drive * curl_h

    assert coeffs.medium_type == medium.type
    assert updated_state.polarization.shape[0] == coeffs.num_poles
    assert np.allclose(updated_e, baseline - coeffs.electric_drive * polarization_current)


def test_constitutive_kernel_metadata_reports_backend_choice() -> None:
    metadata = constitutive_kernel_metadata()

    assert metadata["backend"] in {"numpy", "warp"}
    assert metadata["module_contents_stable"] is True
    assert "pole_residue_electric_update" in metadata["stages"]
    assert "Sellmeier" in metadata["dispersive_medium_paths"]
    assert "Lorentz" in metadata["dispersive_medium_paths"]
    assert "Drude" in metadata["dispersive_medium_paths"]
    assert "Debye" in metadata["dispersive_medium_paths"]
    assert "AnisotropicMedium" in metadata["anisotropic_medium_paths"]
    assert "anisotropic_electric_update" in metadata["stages"]
