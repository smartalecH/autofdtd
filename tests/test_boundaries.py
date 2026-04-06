from __future__ import annotations

import numpy as np
import pytest

from autofdtd.api import (
    ABCBoundary,
    Absorber,
    AbsorberParams,
    BlochBoundary,
    Box,
    BroadbandModeABCSpec,
    Boundary,
    BoundarySpec,
    ModeABCBoundary,
    PECBoundary,
    PML,
    PMLParams,
    PMCBoundary,
    Periodic,
    Simulation,
    StablePML,
)
from autofdtd.compiler import BoundaryMode, compile_boundary_spec
from autofdtd.core.validation import AutoFDTDValidationWarning
from autofdtd.ir import ABCBoundaryIR, AbsorberIR, BoundarySpecIR, PMLIR, StablePMLIR, simulation_to_ir
from autofdtd.kernels import (
    apply_abc_layers,
    apply_boundary_ghosts,
    apply_boundary_stages,
    boundary_edge_transform,
    boundary_kernel_metadata,
    symmetry_transform,
)
from autofdtd.runtime import allocate_abc_boundary_state, allocate_pml_boundary_state, build_boundary_runtime


def test_boundary_coerces_periodic_opposite_conductor_with_warning() -> None:
    with pytest.warns(
        AutoFDTDValidationWarning,
        match="periodic boundary opposite a PEC/PMC boundary",
    ):
        boundary = Boundary(minus=PECBoundary(), plus=Periodic())

    assert boundary.minus.type == "PECBoundary"
    assert boundary.plus.type == "PECBoundary"


def test_simulation_normalizes_supported_boundary_subset() -> None:
    simulation = Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=1.0,
        boundary_spec={
            "type": "BoundarySpec",
            "x": {"type": "Periodic"},
            "y": {
                "type": "Boundary",
                "minus": {"type": "PECBoundary"},
                "plus": {"type": "Periodic"},
            },
            "z": {"type": "PMCBoundary"},
        },
    )

    assert simulation.boundary_spec["x"]["minus"]["type"] == "Periodic"
    assert simulation.boundary_spec["x"]["plus"]["type"] == "Periodic"
    assert simulation.boundary_spec["y"]["minus"]["type"] == "PECBoundary"
    assert simulation.boundary_spec["y"]["plus"]["type"] == "PECBoundary"
    assert simulation.boundary_spec["z"]["minus"]["type"] == "PMCBoundary"
    assert simulation.boundary_spec["z"]["plus"]["type"] == "PMCBoundary"


def test_simulation_normalizes_abc_boundary_subset() -> None:
    simulation = Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=1.0,
        boundary_spec=BoundarySpec(
            x=Boundary.abc(permittivity=2.25, conductivity=0.02),
            y=Boundary.periodic(),
            z=Boundary.pec(),
        ),
    )

    assert simulation.boundary_spec["x"]["minus"].type == "ABCBoundary"
    assert simulation.boundary_spec["x"]["minus"].permittivity == pytest.approx(2.25)
    assert simulation.boundary_spec["x"]["plus"].conductivity == pytest.approx(0.02)


def test_compile_boundary_spec_builds_runtime_modes_for_absorbing_variants() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(
            x=Boundary.bloch(0.25),
            y=Boundary.stable_pml(num_layers=8),
            z=Boundary(minus=Absorber(num_layers=6), plus=PML(num_layers=4)),
        ),
        symmetry=(1, 0, -1),
        dt=0.5,
        grid_spacing=(0.25, 0.5, 0.25),
    )

    assert compiled.x.exchange_kind == "bloch"
    assert compiled.x.minus.mode is BoundaryMode.BLOCH
    assert compiled.x.plus.phase_factor == pytest.approx(1j)
    assert compiled.x.minus.phase_factor == pytest.approx(-1j)
    assert compiled.y.minus.mode is BoundaryMode.STABLE_PML
    assert compiled.z.minus.mode is BoundaryMode.ABSORBER
    assert compiled.z.plus.mode is BoundaryMode.PML
    assert compiled.bloch_axes == ("x",)
    assert compiled.reflective_axes == ("y", "z")
    assert compiled.pml_axes == ("y", "z")
    assert compiled.pml_faces == ("y.minus", "y.plus", "z.minus", "z.plus")
    assert compiled.stable_pml_axes == ("y",)
    assert compiled.stable_pml_faces == ("y.minus", "y.plus")
    assert compiled.absorber_axes == ("z",)
    assert compiled.absorber_faces == ("z.minus",)
    assert compiled.z.plus.pml is not None
    assert compiled.z.plus.pml.num_layers == 4
    assert compiled.z.minus.pml.boundary_type == "Absorber"
    assert compiled.z.minus.pml.terminal_reflection == "pec"
    assert compiled.y.plus.pml.boundary_type == "StablePML"
    assert compiled.z.plus.pml.grid_spacing == pytest.approx(0.25)
    assert compiled.stage_order == (
        "electric_boundary",
        "magnetic_boundary",
        "electric_pml",
        "magnetic_pml",
    )
    assert compiled.requires_complex_fields is True
    assert compiled.active_symmetry_axes == ("x", "z")
    assert compiled.symmetry_axes[0].electric_signs == (-1, 1, 1)
    assert compiled.symmetry_axes[1].magnetic_signs == (1, 1, -1)


def test_compile_boundary_spec_builds_first_order_abc_subset() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(x=Boundary.abc(permittivity=4.0, conductivity=0.01), y=Boundary.periodic()),
        dt=0.25,
        grid_spacing=(0.5, 0.5, 0.5),
    )

    assert compiled.x.exchange_kind == "abc"
    assert compiled.x.minus.mode is BoundaryMode.ABC
    assert compiled.abc_axes == ("x",)
    assert compiled.abc_faces == ("x.minus", "x.plus")
    assert compiled.x.minus.abc is not None
    assert compiled.x.minus.abc.effective_permittivity == pytest.approx(4.0)
    assert compiled.x.minus.abc.effective_conductivity == pytest.approx(0.01)
    assert compiled.stage_order[:4] == (
        "electric_boundary",
        "magnetic_boundary",
        "electric_abc",
        "magnetic_abc",
    )


def test_compile_boundary_spec_requires_explicit_permittivity_for_abc_subset() -> None:
    with pytest.raises(ValueError, match="requires an explicit permittivity"):
        compile_boundary_spec(BoundarySpec(x=Boundary.abc()))


def test_mode_abc_boundary_remains_deferred_in_validation_and_runtime_compilation() -> None:
    plane = Box(center=(0.0, 0.0, 0.0), size=(0.0, 1.0, 1.0))

    with pytest.raises(Exception, match="deferred feature 'ModeABCBoundary'"):
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            boundary_spec=BoundarySpec(x=Boundary.mode_abc(plane=plane)),
        )

    with pytest.raises(ValueError, match="runtime compilation is deferred"):
        compile_boundary_spec(BoundarySpec(x=Boundary.mode_abc(plane=plane)))


def test_boundary_kernels_apply_periodic_and_reflective_ghost_updates() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(x=Boundary.bloch(0.25), y=Boundary.pec(), z=Boundary.pmc())
    )
    electric = np.zeros((4, 4, 4, 3), dtype=np.complex128)
    magnetic = np.zeros((4, 4, 4, 3), dtype=np.complex128)

    electric[1, 1, 1] = (1.0, 2.0, 3.0)
    electric[-2, 1, 1] = (4.0, 5.0, 6.0)
    electric[2, 1, 2] = (7.0, 8.0, 9.0)
    electric[1, -2, 2] = (11.0, 12.0, 13.0)
    electric[2, 2, 1] = (14.0, 15.0, 16.0)
    electric[1, 2, -2] = (17.0, 18.0, 19.0)

    magnetic[1, 1, 1] = (21.0, 22.0, 23.0)
    magnetic[-2, 1, 1] = (24.0, 25.0, 26.0)
    magnetic[2, 1, 2] = (27.0, 28.0, 29.0)
    magnetic[2, -2, 2] = (31.0, 32.0, 33.0)
    magnetic[2, 2, 1] = (34.0, 35.0, 36.0)
    magnetic[2, 2, -2] = (37.0, 38.0, 39.0)

    updated_electric, updated_magnetic = apply_boundary_ghosts(electric, magnetic, compiled)

    assert tuple(updated_electric[0, 1, 1]) == pytest.approx((-4.0j, -5.0j, -6.0j))
    assert tuple(updated_electric[-1, 1, 1]) == pytest.approx((1.0j, 2.0j, 3.0j))
    assert tuple(updated_magnetic[0, 1, 1]) == pytest.approx((-24.0j, -25.0j, -26.0j))
    assert tuple(updated_electric[2, 0, 2]) == pytest.approx((-7.0, 8.0, -9.0))
    assert tuple(updated_magnetic[2, 0, 2]) == pytest.approx((27.0, -28.0, 29.0))
    assert tuple(updated_electric[2, 2, 0]) == pytest.approx((14.0, 15.0, -16.0))
    assert tuple(updated_magnetic[2, 2, -1]) == pytest.approx((-37.0, -38.0, 39.0))


def test_boundary_runtime_wrapper_and_ir_lowering_expose_typed_metadata() -> None:
    boundary_spec = BoundarySpec(x=Boundary.bloch(0.25), y=Boundary.pec(), z=Boundary.pmc())
    runtime = build_boundary_runtime(boundary_spec, symmetry=(1, 0, -1))
    metadata = boundary_kernel_metadata()
    simulation_ir = simulation_to_ir(
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            boundary_spec=boundary_spec,
            symmetry=(1, 0, -1),
        )
    )

    assert runtime.compiled.bloch_axes == ("x",)
    assert metadata["supported_modes"] == (
        "periodic",
        "bloch",
        "pec",
        "pmc",
        "abc",
        "pml",
        "stable_pml",
        "absorber",
    )
    assert isinstance(simulation_ir.boundary_spec, BoundarySpecIR)
    assert simulation_ir.boundary_spec.reflective_axes == ("y", "z")
    assert simulation_ir.boundary_spec.requires_complex_fields is True
    assert simulation_ir.boundary_spec.active_symmetry_axes == ("x", "z")


def test_abc_public_api_and_ir_lowering_are_typed() -> None:
    simulation_ir = simulation_to_ir(
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            boundary_spec=BoundarySpec(x=Boundary.abc(permittivity=2.25, conductivity=0.01)),
        )
    )

    assert isinstance(simulation_ir.boundary_spec, BoundarySpecIR)
    assert isinstance(simulation_ir.boundary_spec.x.minus, ABCBoundaryIR)
    assert simulation_ir.boundary_spec.x.minus.permittivity == pytest.approx(2.25)
    assert simulation_ir.boundary_spec.abc_axes == ("x",)
    assert simulation_ir.boundary_spec.abc_faces == ("x.minus", "x.plus")


def test_pml_public_api_and_ir_lowering_are_typed() -> None:
    simulation_ir = simulation_to_ir(
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            boundary_spec=BoundarySpec(
                x=Boundary.pml(
                    num_layers=8,
                    parameters=PMLParams(
                        sigma_order=2,
                        sigma_max=2.0,
                        kappa_max=4.0,
                        alpha_max=0.1,
                    ),
                )
            ),
        )
    )

    assert isinstance(simulation_ir.boundary_spec, BoundarySpecIR)
    assert isinstance(simulation_ir.boundary_spec.x.minus, PMLIR)
    assert simulation_ir.boundary_spec.x.minus.num_layers == 8
    assert simulation_ir.boundary_spec.x.minus.parameters.kappa_max == pytest.approx(4.0)
    assert simulation_ir.boundary_spec.pml_axes == ("x",)
    assert simulation_ir.boundary_spec.pml_faces == ("x.minus", "x.plus")
    assert simulation_ir.boundary_spec.reflective_axes == ("x",)


def test_stable_pml_and_absorber_public_api_and_ir_lowering_are_typed() -> None:
    simulation_ir = simulation_to_ir(
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            boundary_spec=BoundarySpec(
                x=Boundary.stable_pml(
                    num_layers=10,
                    parameters=PMLParams(sigma_max=1.2, kappa_max=4.5, alpha_max=0.8),
                ),
                z=Boundary.absorber(
                    num_layers=12,
                    parameters=AbsorberParams(sigma_order=2, sigma_max=7.5),
                ),
            ),
        )
    )

    assert isinstance(simulation_ir.boundary_spec, BoundarySpecIR)
    assert isinstance(simulation_ir.boundary_spec.x.minus, StablePMLIR)
    assert isinstance(simulation_ir.boundary_spec.z.minus, AbsorberIR)
    assert simulation_ir.boundary_spec.stable_pml_axes == ("x",)
    assert simulation_ir.boundary_spec.stable_pml_faces == ("x.minus", "x.plus")
    assert simulation_ir.boundary_spec.absorber_axes == ("z",)
    assert simulation_ir.boundary_spec.absorber_faces == ("z.minus", "z.plus")
    assert simulation_ir.boundary_spec.x.minus.parameters.alpha_max == pytest.approx(0.8)
    assert simulation_ir.boundary_spec.z.minus.parameters.sigma_max == pytest.approx(7.5)


def test_boundary_runtime_exposes_phase_and_symmetry_transforms() -> None:
    runtime = build_boundary_runtime(BoundarySpec(x=Boundary.bloch(0.25)), symmetry=(1, 0, 0))
    vector = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)

    boundary_values = runtime.transform_boundary_values(
        vector,
        runtime.compiled.x.plus,
        field_family="electric",
    )
    mirrored = runtime.transform_symmetry_values(
        vector,
        runtime.compiled.symmetry_axes[0],
        field_family="magnetic",
    )

    assert tuple(boundary_values[0]) == pytest.approx((1.0j, 2.0j, 3.0j))
    assert tuple(mirrored[0]) == pytest.approx((1.0, -2.0, -3.0))
    assert tuple(
        boundary_edge_transform(vector, runtime.compiled.x.minus, field_family="electric")[0]
    ) == pytest.approx((-1.0j, -2.0j, -3.0j))
    assert tuple(
        symmetry_transform(
            vector,
            runtime.compiled.symmetry_axes[0],
            field_family="electric",
        )[0]
    ) == pytest.approx((-1.0, 2.0, 3.0))


def test_pml_state_allocation_tracks_active_faces() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(
            x=Boundary.pml(num_layers=3),
            y=Boundary.absorber(num_layers=5),
            z=Boundary(minus=StablePML(num_layers=2), plus=Periodic()),
        )
    )
    state = allocate_pml_boundary_state(compiled, (12, 8, 6, 3), dtype=np.float64)

    assert set(state.electric) == {"x.minus", "x.plus", "y.minus", "y.plus", "z.minus"}
    assert state.electric["x.minus"].memory_shape == (3, 8, 6, 3)
    assert state.electric["y.minus"].memory_shape == (12, 5, 6, 3)
    assert state.electric["z.minus"].memory_shape == (12, 8, 2, 3)
    assert np.all(state.magnetic["x.plus"].memory == 0.0)


def test_abc_state_allocation_and_stage_update_track_active_faces() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(x=Boundary.abc(permittivity=4.0, conductivity=0.02), y=Boundary.periodic()),
        dt=1.0e-12,
    )
    state = allocate_abc_boundary_state(compiled, (8, 6, 4, 3), dtype=np.float64)
    electric = np.zeros((8, 6, 4, 3), dtype=np.float64)
    electric[1, 2:4, 1:3, 0] = 1.0

    updated = apply_abc_layers(electric, compiled, state, field_family="electric")

    assert set(state.electric) == {"x.minus", "x.plus"}
    assert state.electric["x.minus"].value_shape == (6, 4, 3)
    assert np.max(np.abs(updated[0, 2:4, 1:3, 0])) > 0.0
    assert np.max(np.abs(updated[0, 2:4, 1:3, 0])) < 1.0
    assert np.max(np.abs(state.electric["x.minus"].previous_adjacent[..., 0])) == pytest.approx(1.0)


def test_pml_boundary_stages_absorb_boundary_energy() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(x=Boundary.pml(num_layers=4), y=Boundary.periodic(), z=Boundary.periodic()),
        dt=0.5,
        grid_spacing=(0.25, 0.5, 0.5),
    )
    state = allocate_pml_boundary_state(compiled, (12, 4, 4, 3), dtype=np.float64)
    electric = np.zeros((12, 4, 4, 3), dtype=np.float64)
    magnetic = np.zeros((12, 4, 4, 3), dtype=np.float64)
    electric[1:5, 1:3, 1:3, 0] = 1.0
    magnetic[-5:-1, 1:3, 1:3, 1] = 1.0

    before_left = float(np.sum(electric[1:5, ...] ** 2))
    before_center = float(np.sum(electric[5:7, ...] ** 2))
    for _ in range(12):
        electric, magnetic = apply_boundary_stages(electric, magnetic, compiled, state)
    after_left = float(np.sum(electric[1:5, ...] ** 2))
    after_right_h = float(np.sum(magnetic[-5:-1, ...] ** 2))

    assert after_left < before_left * 0.2
    assert after_right_h < 0.2 * 16.0
    assert before_center == pytest.approx(0.0)
    assert float(np.sum(electric[5:7, ...] ** 2)) == pytest.approx(0.0)
    assert np.linalg.norm(state.electric["x.minus"].memory) > 0.0


def test_stable_pml_and_absorber_boundary_stages_absorb_boundary_energy() -> None:
    compiled = compile_boundary_spec(
        BoundarySpec(
            x=Boundary.stable_pml(num_layers=6),
            y=Boundary.absorber(
                num_layers=6,
                parameters=AbsorberParams(sigma_max=8.0),
            ),
            z=Boundary.periodic(),
        ),
        dt=0.5,
        grid_spacing=(0.25, 0.25, 0.5),
    )
    state = allocate_pml_boundary_state(compiled, (12, 12, 4, 3), dtype=np.float64)
    electric = np.zeros((12, 12, 4, 3), dtype=np.float64)
    magnetic = np.zeros((12, 12, 4, 3), dtype=np.float64)
    electric[1:7, 1:7, 1:3, 0] = 1.0
    magnetic[1:7, 1:7, 1:3, 1] = 1.0

    before_x = float(np.sum(electric[1:7, :, :, :] ** 2))
    before_y = float(np.sum(magnetic[:, 1:7, :, :] ** 2))
    for _ in range(10):
        electric, magnetic = apply_boundary_stages(electric, magnetic, compiled, state)
    after_x = float(np.sum(electric[1:7, :, :, :] ** 2))
    after_y = float(np.sum(magnetic[:, 1:7, :, :] ** 2))

    assert after_x < before_x * 0.3
    assert after_y < before_y * 0.3
    assert np.linalg.norm(state.electric["x.minus"].memory) > 0.0
    assert np.linalg.norm(state.electric["y.minus"].memory) == pytest.approx(0.0)


def test_absorber_ghost_updates_use_reflective_outer_wall() -> None:
    compiled = compile_boundary_spec(BoundarySpec(y=Boundary.absorber(num_layers=4)))
    electric = np.zeros((4, 4, 4, 3), dtype=np.float64)
    magnetic = np.zeros((4, 4, 4, 3), dtype=np.float64)

    electric[1, 1, 1] = (1.0, 2.0, 3.0)
    magnetic[1, 1, 1] = (4.0, 5.0, 6.0)

    updated_electric, updated_magnetic = apply_boundary_ghosts(electric, magnetic, compiled)

    assert tuple(updated_electric[1, 0, 1]) == pytest.approx((-1.0, 2.0, -3.0))
    assert tuple(updated_magnetic[1, 0, 1]) == pytest.approx((4.0, -5.0, 6.0))


def test_boundary_rejects_single_sided_bloch_and_normalizes_public_api() -> None:
    boundary = Boundary.bloch(0.125)
    assert isinstance(boundary.plus, BlochBoundary)
    assert boundary.plus.bloch_phase == pytest.approx(
        np.exp(1j * 2.0 * np.pi * 0.125),
    )

    with pytest.raises(ValueError, match="Bloch boundaries must be applied on both sides"):
        Boundary(minus=BlochBoundary(bloch_vec=0.1), plus=Periodic())

    assert isinstance(Boundary.stable_pml().minus, StablePML)
    assert isinstance(Boundary.absorber().plus, Absorber)
    assert isinstance(Boundary.abc(permittivity=2.0).minus, ABCBoundary)
    mode_boundary = Boundary.mode_abc(
        plane=Box(center=(0.0, 0.0, 0.0), size=(0.0, 1.0, 1.0)),
        freq_spec=BroadbandModeABCSpec(frequency_range=(1.0e14, 1.2e14)),
    )
    assert isinstance(mode_boundary.plus, ModeABCBoundary)
    assert "deferred" in mode_boundary.plus.phase1_policy
