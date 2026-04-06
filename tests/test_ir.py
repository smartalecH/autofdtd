from __future__ import annotations

import json
import math

import pytest
from pydantic import BaseModel

from autofdtd.api import (
    AutoGrid,
    Box,
    ClipOperation,
    CustomGrid,
    CustomGridBoundaries,
    Cylinder,
    GeometryArray,
    GeometryGroup,
    GeometryTransform,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    PolarizedAveraging,
    PolySlab,
    Simulation,
    Sphere,
    Staircasing,
    Structure,
    StructurePriorityMode,
    SubpixelSpec,
    Transformed,
    UniformGrid,
)
from autofdtd.ir import (
    EXECUTION_PACKAGE_FORMAT,
    IR_SCHEMA_VERSION,
    AutoGridIR,
    BoxIR,
    ClipOperationIR,
    CustomGridBoundariesIR,
    CustomGridIR,
    CylinderIR,
    ExecutionPackageIR,
    GeometryGroupIR,
    GridRefinementIR,
    GridSpecIR,
    LayerRefinementSpecIR,
    PolarizedAveragingIR,
    PolySlabIR,
    ResolvedGridIR,
    SphereIR,
    StaircasingIR,
    SubpixelSpecIR,
    TransformedIR,
    UniformGridIR,
    execution_package_json_schema,
    simulation_to_ir,
)


class NamedStub(BaseModel):
    type: str
    name: str | None = None
    axis: str | None = None


def build_simulation() -> Simulation:
    return Simulation(
        center=(0.0, 0.0, 0.0),
        size=(12.0, 8.0, 4.0),
        run_time=120.0,
        medium={"type": "Medium", "permittivity": 1.0},
        structure_priority_mode=StructurePriorityMode.CONDUCTOR,
        structures=(
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.0), size=(8.0, 1.0, 0.5)),
                medium={"type": "Medium", "permittivity": 3.45},
                name="core",
            ),
            Structure(
                geometry=Cylinder(center=(0.0, 0.0, 0.0), radius=2.0, length=10.0, axis="x"),
                medium={"type": "PECMedium"},
                name="shield",
            ),
            Structure(
                geometry=Sphere(center=(1.0, 0.0, 0.0), radius=0.25),
                medium={"type": "Medium", "permittivity": 1.2},
                name="bead",
            ),
            Structure(
                geometry=PolySlab(
                    vertices=((-0.5, -0.11), (0.5, -0.11), (0.5, 0.11), (-0.5, 0.11)),
                    slab_bounds=(-4.0, 4.0),
                    axis="x",
                ),
                medium={"type": "Medium", "permittivity": 12.1104},
                name="poly_core",
            ),
            Structure(
                geometry=GeometryGroup(
                    geometries=(
                        Transformed(
                            geometry=Box(center=(0.0, 0.0, 0.0), size=(2.0, 0.5, 0.5)),
                            transform=GeometryTransform.rotation(
                                axis=(0.0, 0.0, 1.0),
                                angle=math.pi / 2.0,
                            ),
                        ),
                        ClipOperation(
                            operation="difference",
                            geometry_a=Sphere(center=(0.0, 0.0, 0.0), radius=0.9),
                            geometry_b=Sphere(center=(0.0, 0.0, 0.0), radius=0.4),
                        ),
                    )
                ),
                medium={"type": "Medium", "permittivity": 2.4},
                name="wrapper_bundle",
            ),
            Structure(
                geometry=GeometryArray(
                    geometry=Box(center=(0.0, 0.0, 0.0), size=(0.4, 1.2, 0.2)),
                    offsets=((-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)),
                    transforms=(
                        GeometryTransform.identity(),
                        GeometryTransform.rotation(
                            axis=(0.0, 0.0, 1.0),
                            angle=math.pi / 2.0,
                        ),
                    ),
                ),
                medium={"type": "Medium", "permittivity": 1.8},
                name="post_array",
            ),
        ),
        sources=(NamedStub(type="ModeSource", name="src", axis="+"),),
        monitors=(NamedStub(type="FieldMonitor", name="fields"),),
        boundary_spec={"type": "BoundarySpec", "x": {"type": "PML"}},
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=0.02),
            grid_y=CustomGrid(dl=(0.03, 0.02, 0.03)),
            grid_z=CustomGridBoundaries(coords=(-2.0, -0.5, 0.5, 2.0)),
        ),
        subpixel=SubpixelSpec(
            dielectric=PolarizedAveraging(),
            metal=Staircasing(),
            pec=Staircasing(),
            pmc=Staircasing(),
            lossy_metal=Staircasing(),
        ),
        symmetry=(0, 1, 0),
        shutoff=1e-6,
        courant=0.85,
    )


def test_simulation_to_ir_preserves_order_and_family_tags() -> None:
    simulation_ir = simulation_to_ir(build_simulation())

    assert simulation_ir.type == "SimulationIR"
    assert simulation_ir.model_tag() == "SimulationIR"
    assert simulation_ir.schema_version == IR_SCHEMA_VERSION
    assert simulation_ir.scene.type == "SceneIR"
    assert [structure.name for structure in simulation_ir.scene.structures] == [
        "core",
        "shield",
        "bead",
        "poly_core",
        "wrapper_bundle",
        "post_array",
    ]
    assert simulation_ir.scene.structures[0].geometry.component_type == "Box"
    assert simulation_ir.scene.structures[1].resolved_priority == 100
    assert isinstance(simulation_ir.scene.structures[1].geometry, CylinderIR)
    assert simulation_ir.scene.structures[1].geometry.transform.axes[2] == (1.0, 0.0, 0.0)
    assert isinstance(simulation_ir.scene.structures[2].geometry, SphereIR)
    assert isinstance(simulation_ir.scene.structures[3].geometry, PolySlabIR)
    assert isinstance(simulation_ir.scene.structures[4].geometry, GeometryGroupIR)
    assert simulation_ir.scene.structures[5].source_index == 5
    assert simulation_ir.scene.structures[5].precedence_rank == 4
    assert simulation_ir.scene.structures[5].geometry.component_type == "GeometryGroup"
    assert simulation_ir.scene.background_medium.family == "medium"
    assert simulation_ir.sources[0].family == "source"
    assert simulation_ir.monitors[0].component_type == "FieldMonitor"
    assert isinstance(simulation_ir.grid_spec, GridSpecIR)
    assert isinstance(simulation_ir.grid_spec.grid_x, UniformGridIR)
    assert isinstance(simulation_ir.grid_spec.grid_y, CustomGridIR)
    assert isinstance(simulation_ir.grid_spec.grid_z, CustomGridBoundariesIR)
    assert isinstance(simulation_ir.grid_spec.resolved, ResolvedGridIR)
    assert isinstance(simulation_ir.subpixel, SubpixelSpecIR)
    assert simulation_ir.subpixel.averaging_targets == ("dielectric",)
    assert isinstance(simulation_ir.subpixel.dielectric, PolarizedAveragingIR)
    assert isinstance(simulation_ir.subpixel.pec, StaircasingIR)


def test_execution_package_bundle_writes_manifest_and_simulation_json(tmp_path) -> None:
    bundle = ExecutionPackageIR.from_simulation(
        build_simulation(),
        metadata={"runtime_policy": "bulk_synchronous"},
    )

    manifest_path = bundle.write_bundle(tmp_path / "bundle")

    assert manifest_path.name == "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    simulation_payload = json.loads(
        (manifest_path.parent / "simulation.json").read_text(encoding="utf-8")
    )

    assert manifest["package_format"] == EXECUTION_PACKAGE_FORMAT
    assert manifest["metadata"]["runtime_policy"] == "bulk_synchronous"
    assert manifest["artifacts"][0]["logical_path"] == "simulation.json"
    assert simulation_payload["scene"]["structures"][0]["geometry"]["type"] == "BoxIR"
    assert simulation_payload["scene"]["structures"][1]["geometry"]["type"] == "CylinderIR"
    assert simulation_payload["scene"]["structures"][3]["geometry"]["type"] == "PolySlabIR"
    assert simulation_payload["scene"]["structures"][4]["geometry"]["type"] == "GeometryGroupIR"
    assert simulation_payload["scene"]["structures"][5]["geometry"]["type"] == "GeometryGroupIR"
    assert simulation_payload["grid_spec"]["type"] == "GridSpecIR"
    assert simulation_payload["grid_spec"]["resolved"]["shape"] == [600, 267, 3]


def test_geometry_ir_carries_bounds_and_transform_metadata() -> None:
    simulation_ir = simulation_to_ir(build_simulation())
    box_geometry = simulation_ir.scene.structures[0].geometry
    polyslab_geometry = simulation_ir.scene.structures[3].geometry

    assert isinstance(box_geometry, BoxIR)
    assert box_geometry.bounds_min == (-4.0, -0.5, -0.25)
    assert box_geometry.bounds_max == (4.0, 0.5, 0.25)
    assert box_geometry.transform.origin == (0.0, 0.0, 0.0)
    assert isinstance(polyslab_geometry, PolySlabIR)
    assert polyslab_geometry.bounds_min == (-4.0, -0.5, -0.11)
    assert polyslab_geometry.bounds_max == (4.0, 0.5, 0.11)
    assert polyslab_geometry.transform.axes[2] == (1.0, 0.0, 0.0)


def test_wrapper_geometry_ir_preserves_nested_types_and_bounds() -> None:
    simulation_ir = simulation_to_ir(build_simulation())
    grouped_geometry = simulation_ir.scene.structures[4].geometry

    assert isinstance(grouped_geometry, GeometryGroupIR)
    assert grouped_geometry.bounds_min == (-0.9, -1.0, -0.9)
    assert grouped_geometry.bounds_max == (0.9, 1.0, 0.9)
    assert isinstance(grouped_geometry.geometries[0], TransformedIR)
    assert grouped_geometry.geometries[0].transform.axes[0] == pytest.approx((0.0, 1.0, 0.0))
    assert isinstance(grouped_geometry.geometries[1], ClipOperationIR)
    assert grouped_geometry.geometries[1].operation == "difference"


def test_geometry_array_lowers_to_explicit_grouped_instances_in_ir() -> None:
    simulation_ir = simulation_to_ir(build_simulation())
    grouped_geometry = simulation_ir.scene.structures[5].geometry

    assert isinstance(grouped_geometry, GeometryGroupIR)
    assert grouped_geometry.bounds_min == (-1.7, -0.6, -0.1)
    assert grouped_geometry.bounds_max == (2.1, 0.6, 0.1)
    assert grouped_geometry.geometries[0].component_type == "Box"
    assert isinstance(grouped_geometry.geometries[1], TransformedIR)
    assert grouped_geometry.geometries[1].transform.origin == (1.5, 0.0, 0.0)


def test_grid_spec_ir_carries_resolved_grid_metadata() -> None:
    simulation_ir = simulation_to_ir(build_simulation())

    assert isinstance(simulation_ir.grid_spec, GridSpecIR)
    assert simulation_ir.grid_spec.resolved.shape == (600, 267, 3)
    assert simulation_ir.grid_spec.resolved.total_cells == 480600
    assert simulation_ir.grid_spec.resolved.min_step == pytest.approx(0.02)
    assert simulation_ir.grid_spec.resolved.z.boundaries == pytest.approx((-2.0, -0.5, 0.5, 2.0))


def test_grid_spec_ir_carries_autogrid_and_layer_refinement_metadata() -> None:
    simulation = Simulation(
        center=(0.0, 0.0, 0.0),
        size=(2.0, 1.0, 1.0),
        run_time=10.0,
        medium={"type": "Medium"},
        grid_spec=GridSpec(
            grid_x=AutoGrid(min_steps_per_wvl=12, max_scale=1.8),
            grid_y=UniformGrid(dl=0.5),
            grid_z=UniformGrid(dl=0.5),
            wavelength=1.2,
            layer_refinement_specs=(
                LayerRefinementSpec(
                    axis=0,
                    center=(0.0, 0.0, 0.0),
                    size=(0.4, 1.0, 1.0),
                    bounds_refinement=GridRefinement(refinement_factor=3.0, num_cells=5),
                    bounds_snapping="bounds",
                ),
            ),
        ),
    )

    simulation_ir = simulation_to_ir(simulation)

    assert isinstance(simulation_ir.grid_spec, GridSpecIR)
    assert isinstance(simulation_ir.grid_spec.grid_x, AutoGridIR)
    assert simulation_ir.grid_spec.grid_x.min_steps_per_wvl == pytest.approx(12.0)
    assert len(simulation_ir.grid_spec.layer_refinement_specs) == 1
    assert isinstance(simulation_ir.grid_spec.layer_refinement_specs[0], LayerRefinementSpecIR)
    assert simulation_ir.grid_spec.layer_refinement_specs[0].bounds_snapping == "bounds"
    assert isinstance(
        simulation_ir.grid_spec.layer_refinement_specs[0].bounds_refinement, GridRefinementIR
    )


def test_ir_models_support_copy_update_without_mutation() -> None:
    bundle = ExecutionPackageIR.from_simulation(build_simulation())
    updated = bundle.copy_update(
        manifest=bundle.manifest.copy_update(metadata={"runtime_policy": "overlap-ready"})
    )

    assert bundle.manifest.metadata == {}
    assert updated.manifest.metadata["runtime_policy"] == "overlap-ready"


def test_execution_package_schema_is_inspectable_and_versioned() -> None:
    schema = execution_package_json_schema()

    assert "$defs" in schema
    assert "ExecutionPackageIR" not in schema["$defs"]
    assert "SimulationIR" in schema["$defs"]
    assert "schema_version" in schema["properties"]
