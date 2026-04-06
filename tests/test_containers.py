from __future__ import annotations

import json
import math

import pytest
from pydantic import BaseModel, ConfigDict

from autofdtd.api import (
    AutoFDTDModel,
    AutoFDTDValidationWarning,
    Box,
    ClipOperation,
    Cylinder,
    GeometryArray,
    GeometryGroup,
    GeometryTransform,
    GridSpec,
    PolarizedAveraging,
    PolySlab,
    Scene,
    Simulation,
    Sphere,
    Staircasing,
    Structure,
    StructurePrecedence,
    StructurePriorityMode,
    SubpixelSpec,
    Transformed,
    UniformGrid,
)
from autofdtd.compiler import C_0
from autofdtd.core import json_ready


class NamedStub(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None


class TaggedStub(AutoFDTDModel):
    type: str = "TaggedStub"
    value: int = 1


def test_shared_model_helpers_expose_tag_copy_and_json_ready() -> None:
    tagged = TaggedStub(value=2)
    updated = tagged.copy_update(value=5)

    assert tagged.model_tag() == "TaggedStub"
    assert updated.value == 5
    assert tagged.value == 2
    assert json_ready({"payload": tagged, "items": (1, 2)}) == {
        "payload": {"type": "TaggedStub", "value": 2},
        "items": [1, 2],
    }


def test_structure_name_is_trimmed() -> None:
    structure = Structure(
        geometry={"type": "Box"},
        medium={"type": "Medium"},
        name="  core  ",
    )

    assert structure.name == "core"


def test_scene_rejects_duplicate_structure_names() -> None:
    structure_a = Structure(geometry={"type": "Box"}, medium={"type": "Medium"}, name="wg")
    structure_b = Structure(geometry={"type": "Sphere"}, medium={"type": "Medium"}, name="wg")

    with pytest.raises(ValueError, match="structures names must be unique"):
        Scene(structures=(structure_a, structure_b))


def test_scene_resolution_order_uses_priority_then_insertion_order() -> None:
    low = Structure(geometry={"type": "Box"}, medium={"type": "Medium"}, name="low", priority=0)
    high = Structure(
        geometry={"type": "Sphere"},
        medium={"type": "Medium"},
        name="high",
        priority=10,
    )
    equal_late = Structure(geometry={"type": "Cylinder"}, medium={"type": "Medium"}, name="late")

    scene = Scene(structures=(high, low, equal_late))

    assert [item.name for item in scene.structures_in_resolution_order()] == ["low", "late", "high"]


def test_scene_structure_precedence_exposes_explicit_ordering_metadata() -> None:
    substrate = Structure(
        geometry=Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 1.0)),
        medium={"type": "Medium"},
        name="substrate",
        priority=0,
    )
    metal = Structure(
        geometry=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
        medium={"type": "PECMedium"},
        name="metal",
    )
    cap = Structure(
        geometry=Sphere(center=(0.0, 0.0, 0.2), radius=0.25),
        medium={"type": "Medium"},
        name="cap",
        priority=120,
    )

    scene = Scene(
        structures=(cap, substrate, metal),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
    )

    precedence = scene.structure_precedence()

    assert all(isinstance(item, StructurePrecedence) for item in precedence)
    ordered = [
        (item.structure.name, item.source_index, item.precedence_rank) for item in precedence
    ]

    assert ordered == [
        ("substrate", 1, 0),
        ("metal", 2, 1),
        ("cap", 0, 2),
    ]


def test_primitive_geometry_models_expose_bounds_translation_and_containment() -> None:
    box = Box.from_bounds(rmin=(-1.0, -2.0, -3.0), rmax=(3.0, 2.0, 1.0))
    sphere = Sphere(center=(1.0, 2.0, 3.0), radius=2.5)
    cylinder = Cylinder(center=(1.0, 0.0, 0.0), radius=0.5, length=4.0, axis="x")

    assert box.center == (1.0, 0.0, -1.0)
    assert box.size == (4.0, 4.0, 4.0)
    assert sphere.bounds == ((-1.5, -0.5, 0.5), (3.5, 4.5, 5.5))
    assert cylinder.bounds == ((-1.0, -0.5, -0.5), (3.0, 0.5, 0.5))
    assert cylinder.transform.world_to_local((3.0, 0.0, 0.0)) == (0.0, 0.0, 2.0)
    assert box.translate((1.0, 0.0, 0.0)).center == (2.0, 0.0, -1.0)
    assert sphere.contains_point((1.0, 2.0, 5.0))
    assert not cylinder.contains_point((1.0, 0.6, 0.0))


def test_scene_conductor_mode_assigns_default_medium_priorities() -> None:
    dielectric = Structure(geometry={"type": "Box"}, medium={"type": "Medium"}, name="dielectric")
    pec = Structure(geometry={"type": "Box"}, medium={"type": "PECMedium"}, name="pec")

    scene = Scene(
        structures=(pec, dielectric),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
    )

    assert [item.name for item in scene.structures_in_resolution_order()] == ["dielectric", "pec"]


def test_simulation_reports_effective_timestep_and_num_time_steps() -> None:
    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(4.0e-7, 4.0e-7, 4.0e-7),
        run_time=2.5e-15,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=1.0e-7),
            grid_y=UniformGrid(dl=2.0e-7),
            grid_z=UniformGrid(dl=4.0e-7),
        ),
        subpixel=SubpixelSpec(
            dielectric=PolarizedAveraging(),
            metal=Staircasing(),
            pec=Staircasing(),
            pmc=Staircasing(),
            lossy_metal=Staircasing(),
        ),
    )

    expected_dt = 0.99 / (
        C_0 * math.sqrt((1.0 / (1.0e-7**2)) + (1.0 / (2.0e-7**2)) + (1.0 / (4.0e-7**2)))
    )

    assert simulation.shutoff == pytest.approx(1.0e-5)
    assert simulation.scaled_courant() == pytest.approx(0.99)
    assert simulation.time_step_size() == pytest.approx(expected_dt)
    assert simulation.num_time_steps() == math.ceil(simulation.run_time / expected_dt) + 1


def test_polyslab_exposes_bounds_transform_translation_and_containment() -> None:
    polyslab = PolySlab(
        vertices=((-1.0, -0.2), (1.0, -0.2), (1.0, 0.2), (-1.0, 0.2)),
        slab_bounds=(0.1, 0.4),
        axis="z",
    )
    shifted = polyslab.translate((2.0, -1.0, 0.5))

    assert polyslab.bounds == ((-1.0, -0.2, 0.1), (1.0, 0.2, 0.4))
    assert polyslab.transform.origin == (0.0, 0.0, 0.25)
    assert shifted.bounds == ((1.0, -1.2, 0.6), (3.0, -0.8, 0.9))
    assert polyslab.contains_point((0.0, 0.0, 0.25))
    assert polyslab.contains_point((1.0, 0.0, 0.25))
    assert not polyslab.contains_point((0.0, 0.0, 0.45))


def test_polyslab_supports_concave_planar_layout_queries() -> None:
    bend = PolySlab(
        vertices=((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)),
        slab_bounds=(-0.11, 0.11),
        axis="z",
    )
    scene = Scene(
        structures=(
            Structure(geometry=bend, medium={"type": "Medium", "permittivity": 12.0}, name="bend"),
        )
    )

    assert scene.resolve_structure_at_point((0.5, 2.5, 0.0)).name == "bend"
    assert scene.resolve_structure_at_point((2.5, 2.5, 0.0)) is None


def test_geometry_group_and_transformed_wrap_scene_building_queries() -> None:
    ring = ClipOperation(
        operation="difference",
        geometry_a=Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 0.4)),
        geometry_b=Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.5)),
    )
    rotated = Transformed(
        geometry=Box(center=(1.0, 0.0, 0.0), size=(2.0, 0.4, 0.4)),
        transform=GeometryTransform.rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 2.0),
    )
    group = GeometryGroup(geometries=(ring, rotated))
    scene = Scene(
        structures=(
            Structure(
                geometry=group,
                medium={"type": "Medium", "permittivity": 3.4},
                name="composite",
            ),
        )
    )

    assert group.bounds == ((-1.0, -1.0, -0.2), (1.0, 2.0, 0.2))
    assert scene.resolve_structure_at_point((0.9, 0.0, 0.0)).name == "composite"
    assert scene.resolve_structure_at_point((0.3, 0.0, 0.0)) is None
    assert scene.resolve_structure_at_point((0.0, 1.0, 0.0)).name == "composite"


def test_geometry_array_exposes_bounds_instances_translation_and_containment() -> None:
    array = GeometryArray(
        geometry=Box(center=(0.0, 0.0, 0.0), size=(0.5, 1.0, 0.2)),
        offsets=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        transforms=(
            GeometryTransform.identity(),
            GeometryTransform.rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 2.0),
        ),
    )
    shifted = array.translate((0.0, 0.5, 0.0))

    assert array.instance_count == 2
    assert array.bounds == ((-1.25, -0.5, -0.1), (1.5, 0.5, 0.1))
    assert len(array.instances) == 2
    assert isinstance(array.instances[1], Transformed)
    assert array.contains_point((-1.1, 0.4, 0.0))
    assert array.contains_point((1.4, 0.0, 0.0))
    assert not array.contains_point((0.0, 0.0, 0.0))
    assert shifted.bounds == ((-1.25, 0.0, -0.1), (1.5, 1.0, 0.1))


def test_scene_overlap_stack_reports_array_and_priority_winner() -> None:
    substrate = Structure(
        geometry=Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 0.3)),
        medium={"type": "Medium", "permittivity": 2.0},
        name="substrate",
        priority=0,
    )
    array = Structure(
        geometry=GeometryArray(
            geometry=Box(center=(0.0, 0.0, 0.0), size=(0.4, 1.2, 0.3)),
            offsets=((-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)),
            transforms=(
                GeometryTransform.identity(),
                GeometryTransform.rotation(axis=(0.0, 0.0, 1.0), angle=math.pi / 2.0),
            ),
        ),
        medium={"type": "PECMedium"},
        name="posts",
    )
    scene = Scene(
        structures=(substrate, array),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
    )

    left_stack = scene.resolve_structure_stack_at_point((-0.8, 0.0, 0.0))
    right_stack = scene.resolve_structure_stack_at_point((0.8, 0.0, 0.0))

    assert [item.structure.name for item in left_stack] == ["substrate", "posts"]
    assert [item.structure.name for item in right_stack] == ["substrate", "posts"]
    assert scene.resolve_structure_at_point((0.8, 0.0, 0.0)).name == "posts"
    assert scene.resolve_structure_at_point((0.0, 1.5, 0.0)).name == "substrate"


def test_scene_resolves_winning_structure_at_point_from_priority_order() -> None:
    substrate = Structure(
        geometry=Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 1.0)),
        medium={"type": "Medium", "permittivity": 2.0},
        name="substrate",
        priority=0,
    )
    via = Structure(
        geometry=Cylinder(center=(0.0, 0.0, 0.0), radius=0.5, length=1.0, axis="z"),
        medium={"type": "PECMedium"},
        name="via",
    )
    cap = Structure(
        geometry=Sphere(center=(0.0, 0.0, 0.3), radius=0.35),
        medium={"type": "Medium", "permittivity": 5.0},
        name="cap",
        priority=120,
    )

    scene = Scene(
        structures=(cap, substrate, via),
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
    )

    assert scene.resolve_structure_at_point((0.0, 0.0, -0.4)).name == "via"
    assert scene.resolve_structure_at_point((0.0, 0.0, 0.55)).name == "cap"
    assert scene.resolve_structure_at_point((1.5, 0.0, 0.0)).name == "substrate"
    assert scene.resolve_structure_at_point((5.0, 0.0, 0.0)) is None


def test_simulation_requires_unique_source_and_monitor_names() -> None:
    repeated = NamedStub(type="FieldMonitor", name="port")

    with pytest.raises(ValueError, match="sources names must be unique"):
        Simulation(size=(1.0, 1.0, 1.0), run_time=1.0, sources=(repeated, repeated))

    with pytest.raises(ValueError, match="monitors names must be unique"):
        Simulation(size=(1.0, 1.0, 1.0), run_time=1.0, monitors=(repeated, repeated))


def test_simulation_serialization_payload_is_tagged_and_json_ready(tmp_path) -> None:
    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(10.0, 6.0, 2.0),
        run_time=50.0,
        medium={"type": "Medium", "permittivity": 1.0},
        structures=(
            Structure(
                geometry={"type": "Box", "center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]},
                medium={"type": "Medium", "permittivity": 3.4},
                name="core",
            ),
        ),
        sources=(NamedStub(type="PointDipole", name="src"),),
        monitors=(NamedStub(type="FieldMonitor", name="fields"),),
        boundary_spec={"type": "BoundarySpec"},
        grid_spec={"type": "GridSpec"},
        subpixel={"type": "SubpixelSpec"},
    )

    payload = simulation.to_payload()
    assert payload["type"] == "Simulation"
    assert payload["structures"][0]["type"] == "Structure"
    assert payload["sources"][0]["type"] == "PointDipole"

    output_path = tmp_path / "simulation.json"
    simulation.write_json(output_path)
    restored = json.loads(output_path.read_text(encoding="utf-8"))
    assert restored["monitors"][0]["name"] == "fields"
    assert simulation.model_tag() == "Simulation"


def test_scene_named_lookup_preserves_insertion_order() -> None:
    first = Structure(geometry={"type": "Box"}, medium={"type": "Medium"}, name="first")
    second = Structure(geometry={"type": "Box"}, medium={"type": "Medium"}, name="second")
    scene = Scene(structures=(first, second))

    assert list(scene.structures_by_name()) == ["first", "second"]
    assert scene.get_structure("second") is second


def test_geometry_dicts_are_normalized_and_validated() -> None:
    structure = Structure(
        geometry={"type": "Cylinder", "center": [0, 0, 0], "radius": 1, "length": 2, "axis": "z"},
        medium={"type": "Medium"},
        name=" rod ",
    )

    assert structure.name == "rod"
    assert structure.geometry["center"] == (0.0, 0.0, 0.0)
    assert structure.geometry["axis"] == 2


def test_polyslab_dicts_are_normalized_and_phase1_limits_are_enforced() -> None:
    structure = Structure(
        geometry={
            "type": "PolySlab",
            "vertices": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            "slab_bounds": [0.0, 0.22],
            "axis": "z",
        },
        medium={"type": "Medium"},
    )

    assert structure.geometry["vertices"] == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert structure.geometry["axis"] == 2

    with pytest.raises(ValueError, match="constant-cross-section PolySlab extrusions"):
        Structure(
            geometry={
                "type": "PolySlab",
                "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "slab_bounds": [0.0, 0.22],
                "dilation": 0.1,
            },
            medium={"type": "Medium"},
        )


def test_composite_geometry_dicts_are_normalized_and_clip_policy_is_explicit() -> None:
    structure = Structure(
        geometry={
            "type": "Transformed",
            "geometry": {
                "type": "GeometryGroup",
                "geometries": [
                    {"type": "Box", "center": [0, 0, 0], "size": [2, 1, 1]},
                    {
                        "type": "ClipOperation",
                        "operation": "difference",
                        "geometry_a": {"type": "Sphere", "center": [0, 0, 0], "radius": 1.0},
                        "geometry_b": {"type": "Sphere", "center": [0, 0, 0], "radius": 0.4},
                    },
                ],
            },
            "transform": {
                "type": "GeometryTransform",
                "origin": [1, 0, 0],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        },
        medium={"type": "Medium"},
    )

    assert structure.geometry["geometry"]["type"] == "GeometryGroup"
    assert structure.geometry["geometry"]["geometries"][1]["type"] == "ClipOperation"

    with pytest.raises(ValueError, match="only 'union', 'intersection', and 'difference'"):
        Structure(
            geometry={
                "type": "ClipOperation",
                "operation": "symmetric_difference",
                "geometry_a": {"type": "Box", "center": [0, 0, 0], "size": [1, 1, 1]},
                "geometry_b": {"type": "Sphere", "center": [0, 0, 0], "radius": 0.5},
            },
            medium={"type": "Medium"},
        )


def test_geometry_models_work_in_structure_bounds_validation() -> None:
    simulation = Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1.0,
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 1.0, 1.0)),
                medium={"type": "Medium"},
                name="core",
            ),
        ),
    )

    assert simulation.structures[0].bounds() == ((-1.0, -0.5, -0.5), (1.0, 0.5, 0.5))


def test_geometry_rejects_basic_shape_errors() -> None:
    with pytest.raises(ValueError, match="geometry.radius must be a positive finite value"):
        Structure(
            geometry={"type": "Sphere", "center": [0, 0, 0], "radius": 0},
            medium={"type": "Medium"},
        )

    with pytest.raises(ValueError, match="geometry.axis must be one of"):
        Structure(
            geometry={
                "type": "Cylinder",
                "center": [0, 0, 0],
                "radius": 1,
                "length": 1,
                "axis": "q",
            },
            medium={"type": "Medium"},
        )

    with pytest.raises(ValueError, match="simple non-self-intersecting polygon"):
        PolySlab(
            vertices=((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)),
            slab_bounds=(0.0, 0.2),
        )


def test_deferred_or_rejected_features_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="deferred feature 'TriangleMesh'"):
        Structure(geometry={"type": "TriangleMesh"}, medium={"type": "Medium"})

    with pytest.raises(ValueError, match="rejected clearly feature 'MicrowaveTerminalSource'"):
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            sources=({"type": "MicrowaveTerminalSource", "name": "port"},),
        )


def test_zero_dim_absorbing_boundary_is_coerced_to_periodic_with_warning() -> None:
    with pytest.warns(
        AutoFDTDValidationWarning,
        match="zero-sized axis 'x' cannot use absorbing boundaries",
    ):
        simulation = Simulation(
            size=(0.0, 2.0, 2.0),
            run_time=1.0,
            boundary_spec={"type": "BoundarySpec", "x": {"type": "PML"}},
        )

    assert simulation.boundary_spec["x"]["minus"]["type"] == "Periodic"
    assert simulation.boundary_spec["x"]["plus"]["type"] == "Periodic"


def test_simulation_grid_spec_is_normalized_and_resolved() -> None:
    simulation = Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=1.0,
        grid_spec={
            "type": "GridSpec",
            "grid_x": {"type": "UniformGrid", "dl": 0.25},
            "grid_y": {"type": "CustomGrid", "dl": [0.2, 0.1]},
            "grid_z": {"type": "UniformGrid", "dl": 0.5},
        },
    )

    assert simulation.grid_spec["grid_y"]["type"] == "CustomGrid"
    assert simulation.resolved_grid() is not None
    assert simulation.resolved_grid().shape == (4, 8, 2)


def test_simulation_accepts_autogrid_and_layer_refinement_metadata() -> None:
    simulation = Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=1.0,
        grid_spec={
            "type": "GridSpec",
            "wavelength": 1.0,
            "grid_x": {"type": "AutoGrid", "min_steps_per_wvl": 10},
            "layer_refinement_specs": [
                {
                    "type": "LayerRefinementSpec",
                    "axis": 0,
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.5, 1.0, 1.0],
                    "bounds_refinement": {"type": "GridRefinement", "refinement_factor": 2.0},
                    "bounds_snapping": "bounds",
                }
            ],
        },
    )

    assert simulation.grid_spec["grid_x"]["type"] == "AutoGrid"
    assert simulation.grid_spec["layer_refinement_specs"][0]["type"] == "LayerRefinementSpec"
    assert simulation.resolved_grid() is not None


def test_simulation_rejects_still_planned_quasiuniform_grid() -> None:
    with pytest.raises(ValueError, match="deferred feature 'QuasiUniformGrid'"):
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            grid_spec={"type": "GridSpec", "grid_x": {"type": "QuasiUniformGrid"}},
        )


def test_simulation_grid_spec_requires_top_level_gridspec() -> None:
    with pytest.raises(ValueError, match="top-level 'GridSpec' container"):
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            grid_spec={"type": "UniformGrid", "dl": 0.1},
        )


def test_simulation_subpixel_bool_and_spec_inputs_normalize_to_models() -> None:
    enabled = Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1.0,
        subpixel=True,
    )
    disabled = Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1.0,
        subpixel=False,
    )
    explicit = Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1.0,
        subpixel=SubpixelSpec(dielectric=PolarizedAveraging(), metal=Staircasing()),
    )

    assert isinstance(enabled.subpixel, SubpixelSpec)
    assert isinstance(enabled.subpixel.dielectric, PolarizedAveraging)
    assert isinstance(disabled.subpixel, SubpixelSpec)
    assert isinstance(disabled.subpixel.dielectric, Staircasing)
    assert isinstance(explicit.subpixel, SubpixelSpec)


def test_simulation_subpixel_rejects_non_spec_top_level_policy_payload() -> None:
    with pytest.raises(ValueError, match="SubpixelSpec payload"):
        Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1.0,
            subpixel={"type": "Staircasing"},
        )


def test_structure_bounds_outside_domain_warn_or_error() -> None:
    with pytest.warns(
        AutoFDTDValidationWarning,
        match="extends beyond the simulation bounds",
    ):
        Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1.0,
            structures=(
                Structure(
                    geometry={"type": "Box", "center": [0.75, 0.0, 0.0], "size": [2.0, 1.0, 1.0]},
                    medium={"type": "Medium"},
                    name="clip-me",
                ),
            ),
        )

    with pytest.raises(ValueError, match="lies fully outside the simulation bounds"):
        Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1.0,
            structures=(
                Structure(
                    geometry={"type": "Sphere", "center": [5.0, 0.0, 0.0], "radius": 0.5},
                    medium={"type": "Medium"},
                    name="off-domain",
                ),
            ),
        )


def test_simulation_normalize_index_maps_basic_errors() -> None:
    zero_amplitude = NamedStub(
        type="PointDipole",
        name="src",
        source_time=NamedStub(type="GaussianPulse", amplitude=0.0),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        Simulation(size=(1.0, 1.0, 1.0), run_time=1.0, normalize_index=0)

    with pytest.raises(ValueError, match="zero-amplitude source"):
        Simulation(
            size=(1.0, 1.0, 1.0),
            run_time=1.0,
            sources=(zero_amplitude,),
            normalize_index=0,
        )
