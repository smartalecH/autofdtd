"""Source injection helpers and backend conventions for Phase 1.

This module provides GPU-accelerated source injection via Warp kernels, with
NumPy fallback for CPU execution. All inject functions preserve their wp.array
inputs without silent conversion to NumPy.

Design Principles
----------------
1. **Preserve wp.array inputs**: When given a wp.array, operations are performed
   directly on device without materializing to CPU.
2. **Capture-safe kernels**: All timestep-varying values (time, dt, freq) are
   passed as explicit kernel arguments.
3. **Module-stable kernels**: Kernels are defined early and marked stable to
   avoid JIT reload churn.
4. **Staged injection**: Each source type has a separate kernel for clarity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from autofdtd.compiler.sources import (
    CompiledAstigmaticGaussianBeam,
    CompiledCustomCurrentSource,
    CompiledCustomFieldSource,
    CompiledGaussianBeam,
    CompiledModeSource,
    CompiledPlaneWave,
    CompiledPointDipole,
    CompiledTFSF,
    CompiledUniformCurrentSource,
)

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


# ---------------------------------------------------------------------------
# GPU kernels for source injection (defined early to avoid JIT reload churn)
# ---------------------------------------------------------------------------

if WARP_AVAILABLE:

    @wp.kernel
    def _point_dipole_inject_kernel(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        placements: wp.array(dtype=wp.int32, ndim=2),
        placement_weights: wp.array(dtype=wp.float32, ndim=1),
        num_placements: int,
        amplitude: float,
        field_kind_axis: int,  # 0=E, 1=H
        component_axis: int,
    ):
        """In-place point dipole injection on GPU.

        Adds amplitude * weight to the specified component axis at each placement
        cell. Operates directly on wp.float32 arrays with no CPU roundtrip.
        amplitude is pre-extracted as float32 from the source's complex amplitude
        (imaginary part discarded for Phase 1 real-valued sources).
        For complex sources, precompute complex phasor at initialization time.
        """
        i = wp.tid()
        if i >= num_placements:
            return

        px = placements[i, 0]
        py = placements[i, 1]
        pz = placements[i, 2]
        w = placement_weights[i]

        if field_kind_axis == 0:
            # Electric field
            E[px, py, pz, component_axis] += amplitude * w
        else:
            # Magnetic field
            H[px, py, pz, component_axis] += amplitude * w

    @wp.kernel
    def _uniform_current_inject_kernel(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        placements: wp.array(dtype=wp.int32, ndim=2),
        placement_weights: wp.array(dtype=wp.float32, ndim=1),
        num_placements: int,
        amplitude: float,
        field_kind_axis: int,
        component_axis: int,
    ):
        """In-place uniform current source injection on GPU."""
        i = wp.tid()
        if i >= num_placements:
            return

        px = placements[i, 0]
        py = placements[i, 1]
        pz = placements[i, 2]
        w = placement_weights[i]

        if field_kind_axis == 0:
            E[px, py, pz, component_axis] += amplitude * w
        else:
            H[px, py, pz, component_axis] += amplitude * w

    @wp.kernel
    def _plane_wave_inject_kernel(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        placements: wp.array(dtype=wp.int32, ndim=2),
        placement_weights: wp.array(dtype=wp.float32, ndim=1),
        num_placements: int,
        e_tang_a: float,
        e_tang_b: float,
        h_tang_a: float,
        h_tang_b: float,
        direction_sign: float,
        field_kind_axis: int,  # 0=electric only, 1=magnetic only, 2=both
        tang_axis_a: int,
        tang_axis_b: int,
        injection_axis: int,  # axis along which wave propagates (0=x, 1=y, 2=z)
    ):
        """In-place plane wave injection on GPU.

        For proper FDTD propagation, E and H are staggered by one cell:
        - E is injected at the source plane (placements)
        - H is injected one cell ahead in the propagation direction

        This creates the proper curl relationships for wave propagation.
        """
        i = wp.tid()
        if i >= num_placements:
            return

        px = placements[i, 0]
        py = placements[i, 1]
        pz = placements[i, 2]
        w = placement_weights[i]

        # Compute H placement offset by one cell in propagation direction
        # For +direction, H is at p+1; for -direction, H is at p-1
        h_offset = int(direction_sign)

        if field_kind_axis == 0 or field_kind_axis == 2:
            # Electric field injection at source plane
            E[px, py, pz, tang_axis_a] += (-h_tang_b) * direction_sign * w
            E[px, py, pz, tang_axis_b] += h_tang_a * direction_sign * w
        if field_kind_axis == 1 or field_kind_axis == 2:
            # Magnetic field injection - staggered by one cell in propagation direction
            if injection_axis == 0:
                H[px + h_offset, py, pz, tang_axis_a] += (-e_tang_b) * direction_sign * w
                H[px + h_offset, py, pz, tang_axis_b] += e_tang_a * direction_sign * w
            elif injection_axis == 1:
                H[px, py + h_offset, pz, tang_axis_a] += (-e_tang_b) * direction_sign * w
                H[px, py + h_offset, pz, tang_axis_b] += e_tang_a * direction_sign * w
            else:  # injection_axis == 2 (z)
                H[px, py, pz + h_offset, tang_axis_a] += (-e_tang_b) * direction_sign * w
                H[px, py, pz + h_offset, tang_axis_b] += e_tang_a * direction_sign * w

    @wp.kernel
    def _gaussian_beam_inject_kernel(
        E: wp.array(dtype=wp.float32, ndim=4),
        H: wp.array(dtype=wp.float32, ndim=4),
        placements: wp.array(dtype=wp.int32, ndim=2),
        placement_weights: wp.array(dtype=wp.float32, ndim=1),
        num_placements: int,
        e_tang_a: float,
        e_tang_b: float,
        h_tang_a: float,
        h_tang_b: float,
        direction_sign: float,
        field_kind_axis: int,
        tang_axis_a: int,
        tang_axis_b: int,
    ):
        """In-place Gaussian beam injection on GPU (same as plane wave but with Gaussian envelope)."""
        i = wp.tid()
        if i >= num_placements:
            return

        px = placements[i, 0]
        py = placements[i, 1]
        pz = placements[i, 2]
        w = placement_weights[i]

        if field_kind_axis == 0 or field_kind_axis == 2:
            E[px, py, pz, tang_axis_a] += (-h_tang_b) * direction_sign * w
            E[px, py, pz, tang_axis_b] += h_tang_a * direction_sign * w
        if field_kind_axis == 1 or field_kind_axis == 2:
            H[px, py, pz, tang_axis_a] += (-e_tang_b) * direction_sign * w
            H[px, py, pz, tang_axis_b] += e_tang_a * direction_sign * w


def uniform_current_source_kernel_metadata() -> dict[str, Any]:
    """Expose the current-source kernel backend choices for diagnostics."""

    return {
        "backend": "warp" if WARP_AVAILABLE else "numpy",
        "warp_available": WARP_AVAILABLE,
        "supports_graph_capture": WARP_AVAILABLE,
        "staging": ("source_injection",),
    }


def _is_warp_array(arr: Any) -> bool:
    """Return True if arr is a wp.array."""
    return WARP_AVAILABLE and arr is not None and isinstance(arr, wp.array)


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert to numpy array if needed, otherwise return unchanged."""
    if _is_warp_array(arr):
        return arr.numpy()
    return arr


def _warp_like_array(arr: Any) -> Any:
    """Return arr unchanged if already warp-compatible, else arr."""
    return arr


def uniform_current_density(
    compiled_source: CompiledUniformCurrentSource,
    *,
    shape: tuple[int, int, int],
    time: float,
) -> np.ndarray:
    """Return the spatial current-density term on the primal-cell grid."""

    density = np.zeros((*shape, 3), dtype=np.complex128)
    amplitude = compiled_source.amplitude_at_time(time)
    axis = compiled_source.component_axis
    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        density[placement][axis] += amplitude * weight
    return density


def _launch_gpu_inject_kernel(
    E: "wp.array",
    H: "wp.array",
    placements: tuple[tuple[int, int, int], ...],
    placement_weights: tuple[float, ...],
    amplitude: complex,
    field_kind: str,
    component_axis: int,
    dev: "wp.Device",
):
    """Launch GPU injection kernel with placements converted to device arrays."""
    n = len(placements)
    if n == 0:
        return

    # Convert placements to flat int array on device
    flat = np.array(placements, dtype=np.int32)
    dev_placements = wp.array(data=flat, dtype=wp.int32, ndim=2, device=dev)

    # Convert weights to device array
    w_flat = np.array(placement_weights, dtype=np.float32)
    dev_weights = wp.array(data=w_flat, dtype=wp.float32, ndim=1, device=dev)

    # Real amplitude for float32 injection - use real part to preserve sign.
    # The complex phase from remove_dc_component=True is valid and needed.
    amp_real = float(amplitude.real)

    # field_kind_axis: 0=electric, 1=magnetic
    fk_axis = 0 if field_kind == "electric" else 1

    wp.launch(
        _point_dipole_inject_kernel,
        dim=n,
        inputs=[
            E, H,
            dev_placements, dev_weights,
            n, amp_real, fk_axis, component_axis,
        ],
        device=dev,
    )
    wp.synchronize()


def _launch_plane_wave_gpu_kernel(
    E: "wp.array",
    H: "wp.array",
    placements: tuple[tuple[int, int, int], ...],
    placement_weights: tuple[float, ...],
    e_tang_a: float,
    e_tang_b: float,
    h_tang_a: float,
    h_tang_b: float,
    direction_sign: float,
    field_kind_axis: int,
    tang_axis_a: int,
    tang_axis_b: int,
    injection_axis: int,
    dev: "wp.Device",
):
    """Launch plane wave / Gaussian beam GPU injection kernel."""
    n = len(placements)
    if n == 0:
        return

    flat = np.array(placements, dtype=np.int32)
    dev_placements = wp.array(data=flat, dtype=wp.int32, ndim=2, device=dev)
    w_flat = np.array(placement_weights, dtype=np.float32)
    dev_weights = wp.array(data=w_flat, dtype=wp.float32, ndim=1, device=dev)

    wp.launch(
        _plane_wave_inject_kernel,
        dim=n,
        inputs=[
            E, H, dev_placements, dev_weights, n,
            e_tang_a, e_tang_b, h_tang_a, h_tang_b,
            direction_sign, field_kind_axis, tang_axis_a, tang_axis_b,
            injection_axis,
        ],
        device=dev,
    )
    wp.synchronize()


def inject_uniform_current_source(
    electric_field,
    magnetic_field,
    compiled_source: CompiledUniformCurrentSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled uniform current source to the supplied field buffers.

    GPU path (wp.array input): launches a Warp kernel directly on the GPU
    for in-place injection. No CPU roundtrip, no dtype conversion.

    CPU path (ndarray input): NumPy fallback with complex128 arithmetic.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)

    if is_warp:
        dev = electric_field.device
        amplitude = compiled_source.amplitude_at_time(time) * dt

        _launch_gpu_inject_kernel(
            electric_field,
            magnetic_field,
            compiled_source.placements,
            compiled_source.placement_weights,
            amplitude,
            compiled_source.field_kind,
            compiled_source.component_axis,
            dev,
        )
        # Return same arrays (modified in-place by kernel)
        return electric_field, magnetic_field

    # NumPy fallback path
    electric_np = np.asarray(electric_field)
    magnetic_np = np.asarray(magnetic_field)

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)
    target = updated_electric if compiled_source.field_kind == "electric" else updated_magnetic
    axis = compiled_source.component_axis

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        target[placement][axis] += amplitude * weight

    return updated_electric, updated_magnetic


def inject_point_dipole(
    electric_field,
    magnetic_field,
    compiled_source: CompiledPointDipole,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled point dipole source to the supplied field buffers.

    PointDipole uses density-based amplitude interpretation and interpolates
    across neighboring cells when interpolate=True.

    GPU path (wp.array input): launches a Warp kernel directly on the GPU
    for in-place injection. No CPU roundtrip, no dtype conversion.

    CPU path (ndarray input): NumPy fallback with complex128 arithmetic.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)

    if is_warp:
        dev = electric_field.device
        amplitude = compiled_source.amplitude_at_time(time) * dt

        _launch_gpu_inject_kernel(
            electric_field,
            magnetic_field,
            compiled_source.placements,
            compiled_source.placement_weights,
            amplitude,
            compiled_source.field_kind,
            compiled_source.component_axis,
            dev,
        )
        # Return same arrays (modified in-place by kernel)
        return electric_field, magnetic_field

    # NumPy fallback path
    electric_np = np.asarray(electric_field)
    magnetic_np = np.asarray(magnetic_field)

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)
    target = updated_electric if compiled_source.field_kind == "electric" else updated_magnetic
    axis = compiled_source.component_axis

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        target[placement][axis] += amplitude * weight

    return updated_electric, updated_magnetic


def inject_custom_current_source(
    electric_field,
    magnetic_field,
    compiled_source: CompiledCustomCurrentSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled custom current source to the supplied field buffers.

    CustomCurrentSource can inject both electric and magnetic field components
    based on the provided field data arrays.

    When given a wp.array, preserves device array.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    # Apply electric field components
    if compiled_source.has_electric and compiled_source.e_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.e_field_data.items():
                if field_key in ("Ex", "Ey", "Ez"):
                    axis = "xyz".index(field_key[1].lower())
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        updated_electric[placement][axis] += amplitude * weight * field_values[idx]

    # Apply magnetic field components
    if compiled_source.has_magnetic and compiled_source.h_field_data is not None:
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.h_field_data.items():
                if field_key in ("Hx", "Hy", "Hz"):
                    axis = "xyz".index(field_key[1].lower())
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        updated_magnetic[placement][axis] += amplitude * weight * field_values[idx]

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


def inject_custom_field_source(
    electric_field,
    magnetic_field,
    compiled_source: CompiledCustomFieldSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled custom field source to the supplied field buffers.

    CustomFieldSource uses the equivalence principle on a planar surface.
    For tangential field components provided:
    - Electric field components (Ex, Ey on xy-plane) contribute via M = -n × E
    - Magnetic field components (Hx, Hy on xy-plane) contribute via J = n × H

    The direction sign determines whether the source injects forward (+)
    or backward (-) relative to the injection axis normal.

    When given a wp.array, preserves device array.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    # Equivalence principle:
    # For electric field data: M = -n̂ × E  -> affects H components
    # For magnetic field data: J = n̂ × H  -> affects E components
    # The cross-product determines which field component gets the contribution.
    # n̂ = (nx, ny, nz) where injection_axis determines which component is 1.
    n_vec = [0.0, 0.0, 0.0]
    n_vec[injection_axis] = 1.0

    if compiled_source.has_electric and compiled_source.e_field_data is not None:
        # M = -n̂ × E: magnetic current density from incident E fields
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.e_field_data.items():
                if field_key in ("Ex", "Ey", "Ez"):
                    field_axis = "xyz".index(field_key[1].lower())
                    if field_axis == injection_axis:
                        continue
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        # M_x = ny*Ez - nz*Ey, M_y = nz*Ex - nx*Ez, M_z = nx*Ey - ny*Ex
                        m_x = n_vec[1] * field_values[idx] if field_key == "Ez" else -n_vec[2] * field_values[idx] if field_key == "Ey" else 0.0
                        m_y = n_vec[2] * field_values[idx] if field_key == "Ex" else -n_vec[0] * field_values[idx] if field_key == "Ez" else 0.0
                        m_z = n_vec[0] * field_values[idx] if field_key == "Ey" else -n_vec[1] * field_values[idx] if field_key == "Ex" else 0.0
                        updated_magnetic[placement][0] += amplitude * weight * direction_sign * m_x
                        updated_magnetic[placement][1] += amplitude * weight * direction_sign * m_y
                        updated_magnetic[placement][2] += amplitude * weight * direction_sign * m_z

    if compiled_source.has_magnetic and compiled_source.h_field_data is not None:
        # J = n̂ × H: electric current density from incident H fields
        for placement, weight in zip(
            compiled_source.placements,
            compiled_source.placement_weights,
            strict=True,
        ):
            for field_key, field_values in compiled_source.h_field_data.items():
                if field_key in ("Hx", "Hy", "Hz"):
                    field_axis = "xyz".index(field_key[1].lower())
                    if field_axis == injection_axis:
                        continue
                    if len(field_values) == len(compiled_source.placements):
                        idx = compiled_source.placements.index(placement)
                        # J_x = ny*Hz - nz*Hy, J_y = nz*Hx - nx*Hz, J_z = nx*Hy - ny*Hx
                        j_x = n_vec[1] * field_values[idx] if field_key == "Hz" else -n_vec[2] * field_values[idx] if field_key == "Hy" else 0.0
                        j_y = n_vec[2] * field_values[idx] if field_key == "Hx" else -n_vec[0] * field_values[idx] if field_key == "Hz" else 0.0
                        j_z = n_vec[0] * field_values[idx] if field_key == "Hy" else -n_vec[1] * field_values[idx] if field_key == "Hx" else 0.0
                        updated_electric[placement][0] += amplitude * weight * direction_sign * j_x
                        updated_electric[placement][1] += amplitude * weight * direction_sign * j_y
                        updated_electric[placement][2] += amplitude * weight * direction_sign * j_z

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


def inject_mode_source(
    electric_field,
    magnetic_field,
    compiled_source: CompiledModeSource,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled mode source to the supplied field buffers.

    ModeSource uses the equivalence principle: tangential E and H field
    components from the mode profile are converted to J and M currents.
    For direction="+", H fields contribute to E injection and E fields
    contribute to H injection (forward propagation). For direction="-",
    the reverse holds (backward propagation).

    When given a wp.array, preserves device array.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    amplitude = compiled_source.amplitude_at_time(time) * dt
    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    e_data = compiled_source.e_field_data
    h_data = compiled_source.h_field_data

    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    power_norm = 1.0 / math.sqrt(compiled_source.mode_power) if compiled_source.mode_power > 0 else 1.0

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        # Mode field data is indexed as [y_index, x_index] = [tang_axis_a, tang_axis_b]
        # because it was reshaped to (field_ny, field_nx) = (ny-1, nx-1)
        idx_y, idx_x = placement[tang_axis_a], placement[tang_axis_b]

        # Bounds check: clip to field data shape to avoid crashes
        # This is a fallback; properly the indices should be within bounds
        field_shape = e_data["Ex"].shape
        idx_y_clipped = min(idx_y, field_shape[0] - 1)
        idx_x_clipped = min(idx_x, field_shape[1] - 1)

        ex_val = e_data["Ex"][idx_y_clipped, idx_x_clipped] if "Ex" in e_data else 0.0
        ey_val = e_data["Ey"][idx_y_clipped, idx_x_clipped] if "Ey" in e_data else 0.0
        hx_val = h_data["Hx"][idx_y_clipped, idx_x_clipped] if "Hx" in h_data else 0.0
        hy_val = h_data["Hy"][idx_y_clipped, idx_x_clipped] if "Hy" in h_data else 0.0

        if direction_sign > 0:
            updated_electric[placement][tang_axis_a] += amplitude * weight * power_norm * (-hy_val)
            updated_electric[placement][tang_axis_b] += amplitude * weight * power_norm * hx_val
            updated_magnetic[placement][tang_axis_a] += amplitude * weight * power_norm * ey_val
            updated_magnetic[placement][tang_axis_b] += amplitude * weight * power_norm * (-ex_val)
        else:
            updated_electric[placement][tang_axis_a] += amplitude * weight * power_norm * hy_val
            updated_electric[placement][tang_axis_b] += amplitude * weight * power_norm * (-hx_val)
            updated_magnetic[placement][tang_axis_a] += amplitude * weight * power_norm * (-ey_val)
            updated_magnetic[placement][tang_axis_b] += amplitude * weight * power_norm * ex_val

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


def inject_plane_wave(
    electric_field,
    magnetic_field,
    compiled_source: CompiledPlaneWave,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple:
    """Apply a compiled plane wave source to the supplied field buffers.

    A PlaneWave injects a spatially uniform electromagnetic wave with a defined
    propagation direction and polarization. The source uses the equivalence
    principle: J = n × H and M = -n × E, where n is the source normal.

    GPU path (wp.array input): launches a Warp kernel directly on the GPU
    for in-place injection. No CPU roundtrip, no dtype conversion.

    CPU path (ndarray input): NumPy fallback with complex128 arithmetic.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)

    if is_warp:
        dev = electric_field.device

        direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
        injection_axis = compiled_source.injection_axis
        tang_axes = tuple(a for a in range(3) if a != injection_axis)
        tang_axis_a, tang_axis_b = tang_axes

        amplitude = compiled_source.amplitude_at_time(time) * dt

        dir_vec = compiled_source.dir_vector
        pol_vec = compiled_source.pol_vector
        dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
        dir_hat = tuple(d / dir_mag for d in dir_vec) if dir_mag > 0 else (0.0, 0.0, 1.0)
        pol_hat = pol_vec

        h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
        h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
        h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
        h_dir = (h_dir_x, h_dir_y, h_dir_z)

        z0_normalized = 1.0

        e_tang_a = pol_hat[tang_axis_a] * amplitude
        e_tang_b = pol_hat[tang_axis_b] * amplitude
        h_tang_a = h_dir[tang_axis_a] * amplitude / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude / z0_normalized

        _launch_plane_wave_gpu_kernel(
            electric_field, magnetic_field,
            compiled_source.placements, compiled_source.placement_weights,
            float(e_tang_a), float(e_tang_b),
            float(h_tang_a), float(h_tang_b),
            float(direction_sign), 2,  # field_kind_axis=2 means both E and H
            tang_axis_a, tang_axis_b,
            injection_axis,
            dev,
        )
        return electric_field, magnetic_field

    # NumPy fallback path
    electric_np = np.asarray(electric_field)
    magnetic_np = np.asarray(magnetic_field)

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    amplitude = compiled_source.amplitude_at_time(time) * dt

    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    pol_hat = pol_vec

    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    z0_normalized = 1.0

    # Get k-vector components for spatial phase gradient (angled incidence)
    if freq is None:
        freq = 1e14
    kx = compiled_source.kx_func(freq, 1.0) if compiled_source.kx_func else 0.0
    ky = compiled_source.ky_func(freq, 1.0) if compiled_source.ky_func else 0.0
    kz = compiled_source.kz_func(freq, 1.0) if compiled_source.kz_func else 0.0

    # Get support bounds for position calculation
    (x_min, y_min, z_min), _ = compiled_source.support_bounds

    for placement, weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        strict=True,
    ):
        # Compute position-dependent phase for angled plane wave
        # phase = exp(i * k . r) where k = (kx, ky, kz) and r = (x, y, z)
        px, py, pz = placement
        phase = np.exp(1j * (kx * px + ky * py + kz * pz))

        e_tang_a = pol_hat[tang_axis_a] * amplitude * weight * phase
        e_tang_b = pol_hat[tang_axis_b] * amplitude * weight * phase

        h_tang_a = h_dir[tang_axis_a] * amplitude * weight / z0_normalized * phase
        h_tang_b = h_dir[tang_axis_b] * amplitude * weight / z0_normalized * phase

        if direction_sign > 0:
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    return updated_electric, updated_magnetic


def inject_gaussian_beam(
    electric_field,
    magnetic_field,
    compiled_source: CompiledGaussianBeam,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple:
    """Apply a compiled Gaussian beam source to the supplied field buffers.

    A GaussianBeam injects a spatially Gaussian-shaped electromagnetic wave with
    a defined propagation direction, polarization, and waist parameters.
    The source uses the equivalence principle: J = n × H and M = -n × E.

    When given a wp.array, preserves device array.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    amplitude = compiled_source.amplitude_at_time(time) * dt

    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    pol_hat = pol_vec

    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    z0_normalized = 1.0

    # beam_weights are now complex (w0/w_z)*exp(-r^2/w_z^2)*exp(i*(k*z+k*r^2*inv_r_z/2-psi))
    # amplitude is complex from source time (carrier phase)
    # Use real part of complex product for real-valued FDTD field updates
    for placement, placement_weight, beam_weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        compiled_source.beam_weights,
        strict=True,
    ):
        # Complex field amplitude: amplitude * beam_weight
        # beam_weight is complex from paraxial Gaussian beam formula
        beam_complex = complex(beam_weight)
        field_complex = amplitude * beam_complex
        # Real part goes into the field (imaginary part is reactive energy at injection plane)
        field_real = field_complex.real
        combined = placement_weight * field_real

        e_tang_a = pol_hat[tang_axis_a] * combined
        e_tang_b = pol_hat[tang_axis_b] * combined

        h_tang_a = h_dir[tang_axis_a] * combined / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * combined / z0_normalized

        if direction_sign > 0:
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


def inject_astigmatic_gaussian_beam(
    electric_field,
    magnetic_field,
    compiled_source: CompiledAstigmaticGaussianBeam,
    *,
    time: float,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple:
    """Apply a compiled astigmatic Gaussian beam source to the supplied field buffers.

    An AstigmaticGaussianBeam injects a spatially Gaussian-shaped electromagnetic wave
    with separate waist radii in x and y directions. The source uses the equivalence
    principle: J = n × H and M = -n × E.

    When given a wp.array, preserves device array.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    amplitude = compiled_source.amplitude_at_time(time) * dt

    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    pol_hat = pol_vec

    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    z0_normalized = 1.0

    for placement, placement_weight, beam_weight in zip(
        compiled_source.placements,
        compiled_source.placement_weights,
        compiled_source.beam_weights,
        strict=True,
    ):
        combined_weight = placement_weight * beam_weight

        e_tang_a = pol_hat[tang_axis_a] * amplitude * combined_weight
        e_tang_b = pol_hat[tang_axis_b] * amplitude * combined_weight

        h_tang_a = h_dir[tang_axis_a] * amplitude * combined_weight / z0_normalized
        h_tang_b = h_dir[tang_axis_b] * amplitude * combined_weight / z0_normalized

        if direction_sign > 0:
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
        else:
            updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * direction_sign
            updated_magnetic[placement][tang_axis_b] += e_tang_a * direction_sign
            updated_electric[placement][tang_axis_a] += (-h_tang_b) * direction_sign
            updated_electric[placement][tang_axis_b] += h_tang_a * direction_sign

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


def inject_tfsf(
    electric_field,
    magnetic_field,
    compiled_source: CompiledTFSF,
    *,
    time: float,
    dt: float = 1.0,
) -> tuple:
    """Apply a compiled TFSF source to the supplied field buffers.

    A TFSF (Total-Field Scattered-Field) source injects a plane wave in a finite
    region of the simulation domain. The incident field is injected at the
    injection plane, and the TFSF boundary interaction subtracts the incident
    field at the box edges to maintain the total-field/scattered-field separation.

    When given a wp.array, preserves device array.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        electric_np = electric_field.numpy()
        magnetic_np = magnetic_field.numpy()
    else:
        electric_np = np.asarray(electric_field)
        magnetic_np = np.asarray(magnetic_field)
        dev = None

    if electric_np.shape != magnetic_np.shape:
        raise ValueError("electric_field and magnetic_field must have matching shapes")
    if electric_np.ndim != 4 or electric_np.shape[-1] != 3:
        raise ValueError("field arrays must have shape (nx, ny, nz, 3)")

    updated_electric = electric_np.astype(np.result_type(electric_np.dtype, np.complex128), copy=True)
    updated_magnetic = magnetic_np.astype(np.result_type(magnetic_np.dtype, np.complex128), copy=True)

    direction_sign = 1.0 if compiled_source.direction == "+" else -1.0
    injection_axis = compiled_source.injection_axis

    tang_axes = tuple(a for a in range(3) if a != injection_axis)
    tang_axis_a, tang_axis_b = tang_axes

    amplitude = compiled_source.amplitude_at_time(time) * dt

    dir_vec = compiled_source.dir_vector
    pol_vec = compiled_source.pol_vector

    dir_mag = math.sqrt(sum(d**2 for d in dir_vec))
    if dir_mag > 0:
        dir_hat = tuple(d / dir_mag for d in dir_vec)
    else:
        dir_hat = (0.0, 0.0, 1.0) if injection_axis == 2 else (0.0, 1.0, 0.0)

    # Polarization direction
    pol_hat = pol_vec

    # Compute H direction as cross product of direction and polarization
    h_dir_x = dir_hat[1] * pol_hat[2] - dir_hat[2] * pol_hat[1]
    h_dir_y = dir_hat[2] * pol_hat[0] - dir_hat[0] * pol_hat[2]
    h_dir_z = dir_hat[0] * pol_hat[1] - dir_hat[1] * pol_hat[0]
    h_dir = (h_dir_x, h_dir_y, h_dir_z)

    # For vacuum impedance in normalized units
    z0_normalized = 1.0

    # TFSF injection has two components:
    # 1. Inject the incident field at the injection plane (forward face)
    # 2. Apply the TF/SF boundary correction at all six faces

    # For Phase 1, we implement the injection component only.
    # The full TFSF boundary correction (subtracting incident field at the
    # box edges) requires additional tracking of the incident field and
    # is deferred to a later task that implements the full TFSF boundary stage.

    # The injection plane is at one face of the TFSF box. For direction="+",
    # the injection plane is at the negative face of the box along the
    # injection axis. For direction="-", it's at the positive face.

    # Get TFSF bounds indices
    (imin, jmin, kmin), (imax, jmax, kmax) = compiled_source.tfsf_bounds_indices

    # TFSF boundary: inject only at the 6 faces of the TFSF box
    # The TFSF box separates total-field (inside) from scattered-field (outside).
    # The incident field is injected at the injection plane (one face), and the
    # TF/SF boundary correction is applied at all 6 faces by adding/subtracting
    # the incident field using the equivalence principle.
    #
    # For each face:
    #   - J = n × H_inc → contributes to E along tangential axes
    #   - M = -n × E_inc → contributes to H along tangential axes
    #
    # The 6 faces are: imin (neg), imax (pos) along injection_axis,
    # plus jmin, jmax along tang_axis_a, plus kmin, kmax along tang_axis_b

    # Build list of 6 faces, each face defined by (axis, face_index, direction_sign)
    # The normal direction points OUTWARD from the TFSF box
    faces = [
        (injection_axis, imin, -1.0),   # negative face of injection axis, outward normal = -axis
        (injection_axis, imax, +1.0),   # positive face of injection axis, outward normal = +axis
        (tang_axis_a, jmin, -1.0),      # negative face along tang_axis_a
        (tang_axis_a, jmax, +1.0),      # positive face along tang_axis_a
        (tang_axis_b, kmin, -1.0),      # negative face along tang_axis_b
        (tang_axis_b, kmax, +1.0),      # positive face along tang_axis_b
    ]

    # For each of the 6 faces, apply the equivalence principle correction
    for axis, face_idx, outward_sign in faces:
        # Determine the iteration range for this face
        if axis == injection_axis:
            # Injection axis face: iterate over tang_axis_a × tang_axis_b cross-section
            i_rng = range(imin + 1, imax) if outward_sign < 0 else range(imax - 1, imax)
            j_rng = range(jmin, jmax + 1)
            k_rng = range(kmin, kmax + 1)
        elif axis == tang_axis_a:
            # tang_axis_a face: iterate over injection_axis × tang_axis_b cross-section
            i_rng = range(imin, imax + 1)
            k_rng = range(kmin, kmax + 1)
            # j is fixed at the face layer (inner edge)
            if outward_sign < 0:
                j_face = jmin + 1
            else:
                j_face = jmax - 1
            j_rng = [j_face]
        else:  # axis == tang_axis_b
            # tang_axis_b face: iterate over injection_axis × tang_axis_a cross-section
            i_rng = range(imin, imax + 1)
            j_rng = range(jmin, jmax + 1)
            # k is fixed at the face layer (inner edge)
            if outward_sign < 0:
                k_face = kmin + 1
            else:
                k_face = kmax - 1
            k_rng = [k_face]

        # Iterate over all cells on this face
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    placement = (i, j, k)
                    # Compute position-dependent amplitude (for angled incidence)
                    # The amplitude at this cell includes the spatial phase from the
                    # plane wave k-vector if available, otherwise uniform
                    try:
                        idx = compiled_source.placements.index(placement)
                        weight = compiled_source.placement_weights[idx]
                    except ValueError:
                        weight = 1.0

                    # E field at this position (along polarization)
                    e_tang_a = pol_hat[tang_axis_a] * amplitude * weight
                    e_tang_b = pol_hat[tang_axis_b] * amplitude * weight

                    # H field at this position (perpendicular to both k and pol)
                    h_tang_a = h_dir[tang_axis_a] * amplitude * weight / z0_normalized
                    h_tang_b = h_dir[tang_axis_b] * amplitude * weight / z0_normalized

                    # Apply equivalence principle with outward normal direction
                    # The normal points outward from the TFSF box interior
                    n_tang_a = 1 if tang_axis_a != axis else 0
                    n_tang_b = 1 if tang_axis_b != axis else 0

                    if outward_sign > 0:
                        # Outward normal +: H contributes to E via J = n × H
                        updated_electric[placement][tang_axis_a] += (-h_tang_b) * n_tang_a
                        updated_electric[placement][tang_axis_b] += h_tang_a * n_tang_b
                        # M = -n × E gives H update from E
                        updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * n_tang_a
                        updated_magnetic[placement][tang_axis_b] += e_tang_a * n_tang_b
                    else:
                        # Outward normal -: E contributes to H via M = -n × E
                        updated_magnetic[placement][tang_axis_a] += (-e_tang_b) * n_tang_a
                        updated_magnetic[placement][tang_axis_b] += e_tang_a * n_tang_b
                        # H contributes to E via J = n × H
                        updated_electric[placement][tang_axis_a] += (-h_tang_b) * n_tang_a
                        updated_electric[placement][tang_axis_b] += h_tang_a * n_tang_b

    if is_warp:
        new_E = wp.array(data=updated_electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=updated_magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return updated_electric, updated_magnetic


# ---------------------------------------------------------------------------
# Unified source injection stage kernel
# ---------------------------------------------------------------------------


def inject_sources_stage(
    electric_field,
    magnetic_field,
    *,
    uniform_current_sources: tuple[CompiledUniformCurrentSource, ...] = (),
    point_dipole_sources: tuple[CompiledPointDipole, ...] = (),
    custom_current_sources: tuple[CompiledCustomCurrentSource, ...] = (),
    custom_field_sources: tuple[CompiledCustomFieldSource, ...] = (),
    mode_sources: tuple[CompiledModeSource, ...] = (),
    plane_wave_sources: tuple[CompiledPlaneWave, ...] = (),
    gaussian_beam_sources: tuple[CompiledGaussianBeam, ...] = (),
    astigmatic_gaussian_beam_sources: tuple[CompiledAstigmaticGaussianBeam, ...] = (),
    tfsf_sources: tuple[CompiledTFSF, ...] = (),
    time: float = 0.0,
    dt: float = 1.0,
    freq: float | None = None,
) -> tuple:
    """Inject all source contributions in a single kernel call.

    This is the kernel-level unified source injection function that applies
    all source types to the field buffers. It provides the same functionality
    as the runtime-level `apply_source_injection_stage` but is intended for
    use in kernel composition.

    When given a wp.array, this function preserves the device array by
    passing it through to individual inject functions which each handle
    the wp.array preservation.

    Scheduling
    ---------
    This stage runs between electric_update and magnetic_update::

        electric_update → source_injection → magnetic_update
    """
    is_warp = _is_warp_array(electric_field)
    if is_warp:
        dev = electric_field.device
        # Work on numpy copy for the injection math
        electric = electric_field.numpy()
        magnetic = magnetic_field.numpy()
    else:
        electric = np.asarray(electric_field)
        magnetic = np.asarray(magnetic_field)
        dev = None

    # Uniform current sources
    for source in uniform_current_sources:
        electric, magnetic = inject_uniform_current_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Point dipole sources
    for source in point_dipole_sources:
        electric, magnetic = inject_point_dipole(
            electric, magnetic, source, time=time, dt=dt
        )

    # Custom current sources
    for source in custom_current_sources:
        electric, magnetic = inject_custom_current_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Custom field sources
    for source in custom_field_sources:
        electric, magnetic = inject_custom_field_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Mode sources
    for source in mode_sources:
        electric, magnetic = inject_mode_source(
            electric, magnetic, source, time=time, dt=dt
        )

    # Plane wave sources
    for source in plane_wave_sources:
        electric, magnetic = inject_plane_wave(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # Gaussian beam sources
    for source in gaussian_beam_sources:
        electric, magnetic = inject_gaussian_beam(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # Astigmatic Gaussian beam sources
    for source in astigmatic_gaussian_beam_sources:
        electric, magnetic = inject_astigmatic_gaussian_beam(
            electric, magnetic, source, time=time, dt=dt, freq=freq
        )

    # TFSF sources
    for source in tfsf_sources:
        electric, magnetic = inject_tfsf(
            electric, magnetic, source, time=time, dt=dt
        )

    if is_warp:
        new_E = wp.array(data=electric, dtype=wp.float32, device=dev)
        new_H = wp.array(data=magnetic, dtype=wp.float32, device=dev)
        return new_E, new_H

    return electric, magnetic
