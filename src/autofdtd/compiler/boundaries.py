"""Boundary compilation helpers for the foundational Phase 1 surface."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from autofdtd.boundaries import (
    ABCBoundary,
    Absorber,
    AbsorberParams,
    BlochBoundary,
    BroadbandModeABCSpec,
    Boundary,
    BoundarySpec,
    DEFAULT_PML_PARAMS,
    DEFAULT_ABSORBER_PARAMS,
    ModeABCBoundary,
    PECBoundary,
    PML,
    PMLParams,
    PMCBoundary,
    Periodic,
    StablePML,
    boundary_edge_model_from_value,
    boundary_model_from_value,
    boundary_spec_model_from_value,
)
from autofdtd.core.models import AutoFDTDModel
from autofdtd.compiler.materials import MU_0


class BoundaryMode(StrEnum):
    """Runtime boundary mode consumed by staged halo handling."""

    PERIODIC = "periodic"
    BLOCH = "bloch"
    PEC = "pec"
    PMC = "pmc"
    ABC = "abc"
    PML = "pml"
    STABLE_PML = "stable_pml"
    ABSORBER = "absorber"


C0 = 299_792_458.0
EPSILON_0 = 8.854_187_812_8e-12


class CompiledABCCoefficients(AutoFDTDModel):
    """Prepared first-order absorbing-boundary coefficients for one face."""

    type: str = "CompiledABCCoefficients"
    boundary_type: Literal["ABCBoundary"] = "ABCBoundary"
    reflection_coefficient: float
    attenuation: float
    effective_permittivity: float
    effective_conductivity: float
    wave_speed: float
    dt: float
    grid_spacing: float
    requires_material_inference: bool = False


class CompiledPMLCoefficients(AutoFDTDModel):
    """Prepared per-layer damping and memory coefficients for one PML face."""

    type: str = "CompiledPMLCoefficients"
    boundary_type: Literal["PML", "StablePML", "Absorber"] = "PML"
    num_layers: int
    sigma: tuple[float, ...]
    kappa: tuple[float, ...]
    alpha: tuple[float, ...]
    attenuation: tuple[float, ...]
    memory_decay: tuple[float, ...]
    memory_drive: tuple[float, ...]
    stretch: tuple[float, ...]
    uses_memory: bool = True
    terminal_reflection: Literal["none", "pec"] = "none"
    dt: float
    grid_spacing: float


class CompiledBoundaryEdge(AutoFDTDModel):
    """Runtime-ready metadata for one boundary face."""

    type: str = "CompiledBoundaryEdge"
    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    mode: BoundaryMode
    electric_signs: tuple[int, int, int]
    magnetic_signs: tuple[int, int, int]
    source_side: Literal["minus", "plus"]
    phase_factor: complex = 1.0 + 0.0j
    abc: CompiledABCCoefficients | None = None
    pml: CompiledPMLCoefficients | None = None


class CompiledBoundaryAxis(AutoFDTDModel):
    """Runtime-ready metadata for one axis boundary pair."""

    type: str = "CompiledBoundaryAxis"
    axis: Literal["x", "y", "z"]
    minus: CompiledBoundaryEdge
    plus: CompiledBoundaryEdge
    exchange_kind: Literal["periodic", "bloch", "reflective", "abc", "pml", "mixed"]
    abc_faces: tuple[Literal["minus", "plus"], ...] = ()
    pml_faces: tuple[Literal["minus", "plus"], ...] = ()


class CompiledSymmetryAxis(AutoFDTDModel):
    """Reflection-sign metadata for one reduced symmetry axis."""

    type: str = "CompiledSymmetryAxis"
    axis: Literal["x", "y", "z"]
    symmetry: Literal[-1, 1]
    electric_signs: tuple[int, int, int]
    magnetic_signs: tuple[int, int, int]


class CompiledBoundarySpec(AutoFDTDModel):
    """Compiled boundary metadata consumed by the Phase 1 timestep loop."""

    type: str = "CompiledBoundarySpec"
    x: CompiledBoundaryAxis
    y: CompiledBoundaryAxis
    z: CompiledBoundaryAxis
    periodic_axes: tuple[str, ...] = ()
    bloch_axes: tuple[str, ...] = ()
    reflective_axes: tuple[str, ...] = ()
    abc_axes: tuple[str, ...] = ()
    abc_faces: tuple[str, ...] = ()
    pml_axes: tuple[str, ...] = ()
    pml_faces: tuple[str, ...] = ()
    stable_pml_axes: tuple[str, ...] = ()
    stable_pml_faces: tuple[str, ...] = ()
    absorber_axes: tuple[str, ...] = ()
    absorber_faces: tuple[str, ...] = ()
    requires_complex_fields: bool = False
    symmetry: tuple[int, int, int] = (0, 0, 0)
    active_symmetry_axes: tuple[str, ...] = ()
    symmetry_axes: tuple[CompiledSymmetryAxis, ...] = ()
    halo_depth: int = 1
    stage_order: tuple[str, ...] = ("electric_boundary", "magnetic_boundary")
    pml_dt: float = 1.0
    pml_grid_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def axes(self) -> tuple[CompiledBoundaryAxis, CompiledBoundaryAxis, CompiledBoundaryAxis]:
        return (self.x, self.y, self.z)


def _mode_for_edge(value: object) -> BoundaryMode:
    edge = boundary_edge_model_from_value(value)
    if isinstance(edge, Periodic):
        return BoundaryMode.PERIODIC
    if isinstance(edge, BlochBoundary):
        return BoundaryMode.BLOCH
    if isinstance(edge, PECBoundary):
        return BoundaryMode.PEC
    if isinstance(edge, PMCBoundary):
        return BoundaryMode.PMC
    if isinstance(edge, ABCBoundary):
        return BoundaryMode.ABC
    if isinstance(edge, ModeABCBoundary):
        raise ValueError(
            "ModeABCBoundary parsing is available, but runtime compilation is deferred in "
            "Phase 1 until mode-solver work lands"
        )
    if isinstance(edge, PML):
        return BoundaryMode.PML
    if isinstance(edge, StablePML):
        return BoundaryMode.STABLE_PML
    if isinstance(edge, Absorber):
        return BoundaryMode.ABSORBER
    raise TypeError(f"unsupported boundary edge type {type(edge)!r}")


def _reflection_signs(
    mode: BoundaryMode, axis: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    normal = [1, 1, 1]
    tangential = [-1, -1, -1]
    normal[axis] = -1
    tangential[axis] = 1
    if mode is BoundaryMode.PEC:
        electric = tuple(tangential)
        magnetic = tuple(normal)
        return electric, magnetic
    if mode is BoundaryMode.PMC:
        electric = tuple(normal)
        magnetic = tuple(tangential)
        return electric, magnetic
    if mode is BoundaryMode.ABSORBER:
        electric = tuple(tangential)
        magnetic = tuple(normal)
        return electric, magnetic
    return (1, 1, 1), (1, 1, 1)


def _phase_factor_for_edge(edge: object, *, side: Literal["minus", "plus"]) -> complex:
    normalized = boundary_edge_model_from_value(edge)
    if not isinstance(normalized, BlochBoundary):
        return 1.0 + 0.0j
    phase = normalized.bloch_phase
    return phase.conjugate() if side == "minus" else phase


def _symmetry_signs(
    symmetry: int,
    axis: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if symmetry not in (-1, 1):
        raise ValueError("symmetry metadata only exists for active axes")
    spatial = [1, 1, 1]
    pseudo = [-1, -1, -1]
    spatial[axis] = -1
    pseudo[axis] = 1
    electric = tuple(sign * symmetry for sign in spatial)
    magnetic = tuple(sign * symmetry for sign in pseudo)
    return electric, magnetic


def _resolve_pml_params(edge: PML | StablePML) -> PMLParams:
    params = edge.parameters
    if isinstance(params, PMLParams):
        return params
    return PMLParams.model_validate(params if params is not None else DEFAULT_PML_PARAMS)


def _resolve_absorber_params(edge: Absorber) -> AbsorberParams:
    params = edge.parameters
    if isinstance(params, AbsorberParams):
        return params
    return AbsorberParams.model_validate(params if params is not None else DEFAULT_ABSORBER_PARAMS)


def _polynomial_profile(minimum: float, maximum: float, order: int, fraction: float) -> float:
    return minimum + (maximum - minimum) * (fraction**order)


def _compile_abc_coefficients(
    edge: ABCBoundary,
    *,
    dt: float,
    grid_spacing: float,
) -> CompiledABCCoefficients:
    if dt <= 0.0:
        raise ValueError("ABC coefficient preparation requires dt > 0")
    if grid_spacing <= 0.0:
        raise ValueError("ABC coefficient preparation requires a positive grid spacing")
    if edge.permittivity is None:
        raise ValueError(
            "ABCBoundary runtime compilation currently requires an explicit permittivity; "
            "automatic boundary-medium inference is deferred in Phase 1"
        )
    effective_permittivity = float(edge.permittivity)
    effective_conductivity = float(edge.conductivity or 0.0)
    wave_speed = C0 / math.sqrt(effective_permittivity)
    courant = wave_speed * dt / grid_spacing
    reflection_coefficient = (courant - 1.0) / (courant + 1.0)
    reflection_coefficient = max(-0.999_999, min(0.999_999, reflection_coefficient))
    attenuation = math.exp(
        -effective_conductivity * dt / max(2.0 * EPSILON_0 * effective_permittivity, 1.0e-30)
    )
    return CompiledABCCoefficients(
        reflection_coefficient=reflection_coefficient,
        attenuation=attenuation,
        effective_permittivity=effective_permittivity,
        effective_conductivity=effective_conductivity,
        wave_speed=wave_speed,
        dt=dt,
        grid_spacing=grid_spacing,
        requires_material_inference=False,
    )


def _compile_pml_coefficients(
    edge: PML | StablePML,
    *,
    dt: float,
    grid_spacing: float,
) -> CompiledPMLCoefficients:
    if dt <= 0.0:
        raise ValueError("PML coefficient preparation requires dt > 0")
    if grid_spacing <= 0.0:
        raise ValueError("PML coefficient preparation requires a positive grid spacing")
    params = _resolve_pml_params(edge)
    sigma: list[float] = []
    kappa: list[float] = []
    alpha: list[float] = []
    attenuation: list[float] = []
    memory_decay: list[float] = []
    memory_drive: list[float] = []
    stretch: list[float] = []
    layer_count = edge.num_layers

    # B5 fix: Compute sigma_max using CFS-PML formula
    # sigma_max = -(m+1)*ln(R_target) / (2*eta*d_pml)
    # R_target = 1e-8 (target reflection coefficient)
    # eta = sqrt(mu/eps) = vacuum impedance since PML is at boundary
    # This replaces the fixed sigma_max=1.5 which ignores grid/dt
    R_target = 1e-8
    d_pml = layer_count * grid_spacing
    eta = math.sqrt(MU_0 / EPSILON_0)  # vacuum impedance
    sigma_max = -(params.sigma_order + 1) * math.log(R_target) / (2.0 * eta * d_pml)

    for layer_index in range(layer_count):
        fraction = float(layer_index + 1) / float(layer_count)
        sigma_value = _polynomial_profile(
            params.sigma_min, sigma_max, params.sigma_order, fraction
        )
        kappa_value = _polynomial_profile(
            params.kappa_min, params.kappa_max, params.kappa_order, fraction
        )
        # B6 fix: For CFS-PML (StablePML), alpha profile should decrease from inner to outer.
        # alpha should be MAX at inner edge (fraction=0), MIN at outer edge (fraction=1).
        # The polynomial uses (1-fraction) to achieve the correct CFS-PML profile.
        if isinstance(edge, StablePML):
            alpha_fraction = 1.0 - fraction
        else:
            alpha_fraction = fraction
        alpha_value = _polynomial_profile(
            params.alpha_min, params.alpha_max, params.alpha_order, alpha_fraction
        )
        damping_rate = max(0.0, sigma_value / kappa_value + alpha_value)
        attenuation_value = math.exp(-damping_rate * dt / grid_spacing)
        memory_decay_value = math.exp(-max(0.0, alpha_value) * dt)
        drive_value = (1.0 - attenuation_value) / max(kappa_value, 1.0)
        stretch_value = 1.0 / kappa_value
        sigma.append(sigma_value)
        kappa.append(kappa_value)
        alpha.append(alpha_value)
        attenuation.append(attenuation_value)
        memory_decay.append(memory_decay_value)
        memory_drive.append(drive_value)
        stretch.append(stretch_value)
    return CompiledPMLCoefficients(
        boundary_type=edge.type,
        num_layers=edge.num_layers,
        sigma=tuple(sigma),
        kappa=tuple(kappa),
        alpha=tuple(alpha),
        attenuation=tuple(attenuation),
        memory_decay=tuple(memory_decay),
        memory_drive=tuple(memory_drive),
        stretch=tuple(stretch),
        uses_memory=True,
        terminal_reflection="none",
        dt=dt,
        grid_spacing=grid_spacing,
    )


def _compile_absorber_coefficients(
    edge: Absorber,
    *,
    dt: float,
    grid_spacing: float,
) -> CompiledPMLCoefficients:
    if dt <= 0.0:
        raise ValueError("Absorber coefficient preparation requires dt > 0")
    if grid_spacing <= 0.0:
        raise ValueError("Absorber coefficient preparation requires a positive grid spacing")
    params = _resolve_absorber_params(edge)
    sigma: list[float] = []
    attenuation: list[float] = []
    zero = [0.0] * edge.num_layers
    ones = [1.0] * edge.num_layers
    for layer_index in range(edge.num_layers):
        fraction = float(layer_index + 1) / float(edge.num_layers)
        sigma_value = _polynomial_profile(
            params.sigma_min, params.sigma_max, params.sigma_order, fraction
        )
        damping_rate = max(0.0, sigma_value)
        sigma.append(sigma_value)
        attenuation.append(math.exp(-damping_rate * dt / grid_spacing))
    return CompiledPMLCoefficients(
        boundary_type=edge.type,
        num_layers=edge.num_layers,
        sigma=tuple(sigma),
        kappa=tuple(ones),
        alpha=tuple(zero),
        attenuation=tuple(attenuation),
        memory_decay=tuple(zero),
        memory_drive=tuple(zero),
        stretch=tuple(ones),
        uses_memory=False,
        terminal_reflection="pec",
        dt=dt,
        grid_spacing=grid_spacing,
    )


def _compile_edge(
    axis_name: Literal["x", "y", "z"],
    side: Literal["minus", "plus"],
    edge: object,
    *,
    dt: float,
    grid_spacing: float,
) -> CompiledBoundaryEdge:
    axis_index = "xyz".index(axis_name)
    mode = _mode_for_edge(edge)
    electric_signs, magnetic_signs = _reflection_signs(mode, axis_index)
    abc = None
    pml = None
    normalized = boundary_edge_model_from_value(edge)
    if isinstance(normalized, ModeABCBoundary):
        raise ValueError(
            "ModeABCBoundary parsing is available, but runtime compilation is deferred in "
            "Phase 1 until mode-solver work lands"
        )
    if isinstance(normalized, ABCBoundary):
        abc = _compile_abc_coefficients(normalized, dt=dt, grid_spacing=grid_spacing)
    if isinstance(normalized, (PML, StablePML)):
        pml = _compile_pml_coefficients(normalized, dt=dt, grid_spacing=grid_spacing)
    elif isinstance(normalized, Absorber):
        pml = _compile_absorber_coefficients(normalized, dt=dt, grid_spacing=grid_spacing)
    return CompiledBoundaryEdge(
        axis=axis_name,
        side=side,
        mode=mode,
        electric_signs=electric_signs,
        magnetic_signs=magnetic_signs,
        source_side="plus" if side == "minus" else "minus",
        phase_factor=_phase_factor_for_edge(edge, side=side),
        abc=abc,
        pml=pml,
    )


def _compile_axis(
    axis_name: Literal["x", "y", "z"],
    boundary: Boundary,
    *,
    dt: float,
    grid_spacing: float,
) -> CompiledBoundaryAxis:
    minus = _compile_edge(axis_name, "minus", boundary.minus, dt=dt, grid_spacing=grid_spacing)
    plus = _compile_edge(axis_name, "plus", boundary.plus, dt=dt, grid_spacing=grid_spacing)
    edge_modes = {minus.mode, plus.mode}
    if edge_modes == {BoundaryMode.PERIODIC}:
        exchange_kind: Literal["periodic", "bloch", "reflective", "abc", "pml", "mixed"] = "periodic"
    elif edge_modes == {BoundaryMode.BLOCH}:
        exchange_kind = "bloch"
    elif edge_modes == {BoundaryMode.ABC}:
        exchange_kind = "abc"
    elif edge_modes <= {BoundaryMode.PML, BoundaryMode.STABLE_PML, BoundaryMode.ABSORBER}:
        exchange_kind = "pml"
    elif edge_modes <= {BoundaryMode.PEC, BoundaryMode.PMC}:
        exchange_kind = "reflective"
    else:
        exchange_kind = "mixed"
    abc_faces = tuple(
        side for side, compiled in (("minus", minus), ("plus", plus)) if compiled.mode is BoundaryMode.ABC
    )
    pml_faces = tuple(
        side
        for side, compiled in (("minus", minus), ("plus", plus))
        if compiled.mode in {BoundaryMode.PML, BoundaryMode.STABLE_PML, BoundaryMode.ABSORBER}
    )
    return CompiledBoundaryAxis(
        axis=axis_name,
        minus=minus,
        plus=plus,
        exchange_kind=exchange_kind,
        abc_faces=abc_faces,
        pml_faces=pml_faces,
    )


def _compile_symmetry_axes(
    symmetry: tuple[int, int, int],
) -> tuple[CompiledSymmetryAxis, ...]:
    compiled: list[CompiledSymmetryAxis] = []
    for axis_name, axis_value in zip("xyz", symmetry, strict=True):
        if axis_value == 0:
            continue
        electric_signs, magnetic_signs = _symmetry_signs(axis_value, "xyz".index(axis_name))
        compiled.append(
            CompiledSymmetryAxis(
                axis=axis_name,
                symmetry=axis_value,
                electric_signs=electric_signs,
                magnetic_signs=magnetic_signs,
            )
        )
    return tuple(compiled)


def compile_boundary_spec(
    value: object | None,
    *,
    symmetry: tuple[int, int, int] = (0, 0, 0),
    dt: float = 1.0,
    grid_spacing: float | tuple[float, float, float] = 1.0,
) -> CompiledBoundarySpec:
    """Compile the supported Phase 1 boundary subset into runtime metadata."""

    if value is None:
        boundary_spec = BoundarySpec()
    else:
        try:
            boundary_spec = boundary_spec_model_from_value(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Phase 1 runtime boundary compilation currently supports only BoundarySpec with "
                "Periodic, BlochBoundary, PECBoundary, PMCBoundary, ABCBoundary, PML, "
                "StablePML, and Absorber entries, while ModeABCBoundary remains deferred"
            ) from exc

    if isinstance(grid_spacing, tuple):
        if len(grid_spacing) != 3 or any(step <= 0.0 for step in grid_spacing):
            raise ValueError("grid_spacing must contain three positive axis spacings")
        axis_spacing = grid_spacing
    else:
        if grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive")
        axis_spacing = (float(grid_spacing), float(grid_spacing), float(grid_spacing))

    compiled_axes = {
        axis_name: _compile_axis(
            axis_name,
            axis_boundary,
            dt=dt,
            grid_spacing=axis_spacing["xyz".index(axis_name)],
        )
        for axis_name, axis_boundary in zip("xyz", boundary_spec.as_tuple(), strict=True)
    }
    periodic_axes = tuple(
        axis.axis for axis in compiled_axes.values() if axis.exchange_kind == "periodic"
    )
    bloch_axes = tuple(axis.axis for axis in compiled_axes.values() if axis.exchange_kind == "bloch")
    reflective_axes = tuple(
        axis.axis
        for axis in compiled_axes.values()
        if axis.exchange_kind in {"reflective", "mixed"} or axis.pml_faces
    )
    abc_axes = tuple(axis.axis for axis in compiled_axes.values() if axis.abc_faces)
    abc_faces = tuple(
        f"{axis.axis}.{side}"
        for axis in compiled_axes.values()
        for side in axis.abc_faces
    )
    pml_axes = tuple(axis.axis for axis in compiled_axes.values() if axis.pml_faces)
    pml_faces = tuple(
        f"{axis.axis}.{side}"
        for axis in compiled_axes.values()
        for side in axis.pml_faces
    )
    stable_pml_axes = tuple(
        axis.axis
        for axis in compiled_axes.values()
        if any(
            getattr(axis, side).mode is BoundaryMode.STABLE_PML for side in ("minus", "plus")
        )
    )
    stable_pml_faces = tuple(
        f"{axis.axis}.{side}"
        for axis in compiled_axes.values()
        for side in ("minus", "plus")
        if getattr(axis, side).mode is BoundaryMode.STABLE_PML
    )
    absorber_axes = tuple(
        axis.axis
        for axis in compiled_axes.values()
        if any(getattr(axis, side).mode is BoundaryMode.ABSORBER for side in ("minus", "plus"))
    )
    absorber_faces = tuple(
        f"{axis.axis}.{side}"
        for axis in compiled_axes.values()
        for side in ("minus", "plus")
        if getattr(axis, side).mode is BoundaryMode.ABSORBER
    )
    symmetry_axes = _compile_symmetry_axes(symmetry)
    stage_order = ["electric_boundary", "magnetic_boundary"]
    if abc_faces:
        stage_order.extend(("electric_abc", "magnetic_abc"))
    if pml_faces:
        stage_order.extend(("electric_pml", "magnetic_pml"))
    return CompiledBoundarySpec(
        x=compiled_axes["x"],
        y=compiled_axes["y"],
        z=compiled_axes["z"],
        periodic_axes=periodic_axes,
        bloch_axes=bloch_axes,
        reflective_axes=reflective_axes,
        abc_axes=abc_axes,
        abc_faces=abc_faces,
        pml_axes=pml_axes,
        pml_faces=pml_faces,
        stable_pml_axes=stable_pml_axes,
        stable_pml_faces=stable_pml_faces,
        absorber_axes=absorber_axes,
        absorber_faces=absorber_faces,
        requires_complex_fields=bool(bloch_axes),
        symmetry=symmetry,
        active_symmetry_axes=tuple(axis.axis for axis in symmetry_axes),
        symmetry_axes=symmetry_axes,
        stage_order=tuple(stage_order),
        pml_dt=dt,
        pml_grid_spacing=axis_spacing,
    )


def normalize_axis_boundary(value: object) -> Boundary:
    """Normalize a single axis boundary using the supported subset."""

    return boundary_model_from_value(value)
