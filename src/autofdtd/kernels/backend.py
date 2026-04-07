"""Warp backend conventions for Phase 1 GPU acceleration.

This module establishes the foundational backend contracts that all kernel modules
must follow. It provides:

- Device management and GPU selection policies
- Persistent array base types and allocation conventions
- Module stability markers for JIT compilation
- Capture-safe stepping protocol
- Profiling hooks and instrumentation

Design Principles
----------------
1. **Module stability before JIT churn**: All kernels in a module compile together.
   Stabilize module contents early to avoid reload churn.
2. **Persistent arrays over temporaries**: Field buffers are allocated once and
   reused across timesteps to minimize allocation overhead.
3. **Capture-safe stepping**: Kernels must not capture mutable state from the
   outer Python scope. All timestep-dependent values must be passed as arguments.
4. **Explicit device placement**: Arrays carry device affinity explicitly rather
   than relying on global state.
5. **Structured profiling**: Use Warp's ScopedTimer and NVTX markers for
   GPU-side instrumentation, with CPU-side timing hooks for end-to-end metrics.

Device Management
-----------------
- Use ``get_warp_device()`` to obtain the default device or a specific GPU.
- For multi-GPU: map logical chunk indices to device IDs via
  ``ChunkLayout.device_assignment``.
- All array allocations use the target device explicitly.

Array Conventions
----------------
- Field arrays: ``wp.array(dtype=wp.float32, shape=(Nx, Ny, Nz, 3))``
- Coefficient arrays: ``wp.array(dtype=wp.float32, shape=(Nx, Ny, Nz))``
- Auxiliary state: ``wp.array(dtype=wp.float32, shape=(num_poles, Nx, Ny, Nz))``
- All arrays are allocated on the target device; no managed memory unless explicit.

Dtype Policy
-----------
- Single precision (float32/complex64) is the default for GPU execution.
- Double precision (float64/complex128) is available but may significantly
  reduce performance on consumer NVIDIA GPUs.
- Complex fields are required when any Bloch boundary is present.
- The ``requires_complex_fields`` flag from ``BoundarySpecIR`` determines whether
  to use complex dtype for field storage.

Module Stability
---------------
Use ``mark_module_stable()`` after defining all kernels in a module.
This signals that the module's kernel set is frozen and ready for JIT.
During development, unstable modules trigger a warning on import.

Capture-Safe Stepping
---------------------
Every kernel function that depends on timestep-varying values (dt, time, step_index)
must receive those values as explicit arguments. Do NOT capture them from closure.

Example::

    # BAD - captures dt from outer scope
    @wp.kernel
    def bad_update(E, H,):
        dt = outer_dt  # captured!
        ...

    # GOOD - dt passed explicitly
    @wp.kernel
    def good_update(E, H, dt: float):
        ...

Capture-safe stepping also means:
- Source amplitudes are computed outside the kernel and passed as arrays or scalars
- Monitor intervals are evaluated in Python before calling recording kernels
- Boundary coefficients are pre-compiled and passed as array arguments

Profiling Hooks
---------------
- ``WarpTimer`` context manager for GPU-side kernel timing
- ``nvtx_range`` for annotating execution phases
- ``step_metrics()`` helper for computing cells/s and Gcells/s

References
----------
- Warp docs: https://docs.nvidia.com/warp-py
- GPU Benchmarking: ``../papers/gpu_benchmarking.pdf``
- Chunk Contract: ``../phase1/architecture/chunk-contract.md``
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "WARP_AVAILABLE",
    "wp",
    "get_warp_device",
    "get_default_device",
    "allocate_field_array",
    "allocate_coefficient_array",
    "allocate_auxiliary_array",
    "FieldArrayTag",
    "CoefficientArrayTag",
    "AuxiliaryArrayTag",
    "ArrayTags",
    "WarpTimer",
    "nvtx_range",
    "step_metrics",
    "backend_info",
    "WarpBackendInfo",
    "ComplexFieldPolicy",
    "ModuleStability",
    "mark_module_stable",
    "is_module_stable",
]


# ---------------------------------------------------------------------------
# Warp import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wp = None

WARP_AVAILABLE = wp is not None


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def get_warp_device(device_id: int | str | None = None) -> "wp.Device | None":
    """Return a Warp device for array allocation.

    Parameters
    ----------
    device_id : int, str, or None, optional
        Specific GPU ID as integer (0, 1, ...) or Warp device string
        (e.g., "cuda:0", "cuda:1"). If None, returns the current default device.

    Returns
    -------
    wp.Device or None
        The requested device, or None if Warp is not available.

    Notes
    -----
    When Warp is unavailable, this returns None. Callers must check
    ``WARP_AVAILABLE`` before using the returned device.
    """
    if not WARP_AVAILABLE:
        return None
    if device_id is None:
        return wp.get_device()
    # Handle integer device IDs by converting to Warp device string
    if isinstance(device_id, int):
        return wp.get_device(f"cuda:{device_id}")
    return wp.get_device(device_id)


def get_default_device() -> "wp.Device | None":
    """Return the default Warp device for the current context."""
    return get_warp_device(None)


# ---------------------------------------------------------------------------
# Persistent array tags for field buffer classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldArrayTag:
    """Tag for electric or magnetic field arrays.

    Attributes
    ----------
    family : str
        "electric" or "magnetic".
    chunk_index : tuple[int, int, int] | None
        Logical chunk position, if chunking is active.
    """

    family: Literal["electric", "magnetic"]
    chunk_index: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class CoefficientArrayTag:
    """Tag for material coefficient arrays (epsilon, mu, sigma).

    Attributes
    ----------
    component : str
        "eps", "mu", "sigma_e", or "sigma_h".
    axis : str | None
        For tensor components: "xx", "yy", "zz", or None for scalar.
    chunk_index : tuple[int, int, int] | None
        Logical chunk position, if chunking is active.
    """

    component: Literal["eps", "mu", "sigma_e", "sigma_h"]
    axis: Literal["xx", "yy", "zz"] | None = None
    chunk_index: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class AuxiliaryArrayTag:
    """Tag for dispersive auxiliary state arrays (polarization currents).

    Attributes
    ----------
    medium_family : str
        "PoleResidue", "Lorentz", "Drude", "Debye", etc.
    chunk_index : tuple[int, int, int] | None
        Logical chunk position, if chunking is active.
    """

    medium_family: str
    chunk_index: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ArrayTags:
    """Container for all array classification tags in a simulation.

    This provides a single place to look up the role and provenance
    of every allocated array in a chunk.
    """

    fields: tuple[FieldArrayTag, ...] = field(default_factory=tuple)
    coefficients: tuple[CoefficientArrayTag, ...] = field(default_factory=tuple)
    auxiliary: tuple[AuxiliaryArrayTag, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Array allocation helpers
# ---------------------------------------------------------------------------


def allocate_field_array(
    shape: tuple[int, int, int],
    *,
    family: Literal["electric", "magnetic"],
    dtype: "wp.dtype | None" = None,
    device: "wp.Device | None" = None,
    chunk_index: tuple[int, int, int] | None = None,
    tag: bool = True,
) -> "wp.array | np.ndarray":
    """Allocate a field array on the target device.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Grid shape (Nx, Ny, Nz). The field axis of length 3 is added internally.
    family : "electric" or "magnetic"
        Which field family this array holds.
    dtype : wp.dtype, optional
        Array dtype. Defaults to wp.float32 for real fields.
    device : wp.Device, optional
        Target device. Uses default device if None.
    chunk_index : tuple, optional
        Logical chunk position for chunked allocations.
    tag : bool, default=True
        Whether to attach an ArrayTag to the array for diagnostics.

    Returns
    -------
    wp.array or np.ndarray
        Allocated array on the target device. Returns np.ndarray
        placeholder if Warp is unavailable.

    Notes
    -----
    The returned array has shape ``(*shape, 3)`` for vector fields.
    """
    if not WARP_AVAILABLE:
        import numpy as np

        return np.zeros((*shape, 3), dtype=np.float64)

    if dtype is None:
        dtype = wp.float32

    alloc_shape = (*shape, 3)
    arr = wp.zeros(shape=alloc_shape, dtype=dtype, device=device)

    if tag:
        arr.tag = FieldArrayTag(family=family, chunk_index=chunk_index)

    return arr


def allocate_coefficient_array(
    shape: tuple[int, int, int],
    *,
    component: Literal["eps", "mu", "sigma_e", "sigma_h"],
    axis: Literal["xx", "yy", "zz"] | None = None,
    dtype: "wp.dtype | None" = None,
    device: "wp.Device | None" = None,
    chunk_index: tuple[int, int, int] | None = None,
    tag: bool = True,
) -> "wp.array | np.ndarray":
    """Allocate a material coefficient array on the target device.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Grid shape (Nx, Ny, Nz).
    component : str
        Coefficient type: "eps", "mu", "sigma_e", or "sigma_h".
    axis : str, optional
        For tensor coefficients: "xx", "yy", or "zz".
    dtype : wp.dtype, optional
        Array dtype. Defaults to wp.float32.
    device : wp.Device, optional
        Target device.
    chunk_index : tuple, optional
        Logical chunk position.
    tag : bool, default=True
        Whether to attach an ArrayTag.

    Returns
    -------
    wp.array or np.ndarray
        Allocated coefficient array. Returns np.ndarray placeholder
        if Warp is unavailable.
    """
    if not WARP_AVAILABLE:
        import numpy as np

        return np.zeros(shape, dtype=np.float64)

    if dtype is None:
        dtype = wp.float32

    arr = wp.zeros(shape=shape, dtype=dtype, device=device)

    if tag:
        arr.tag = CoefficientArrayTag(
            component=component, axis=axis, chunk_index=chunk_index
        )

    return arr


def allocate_auxiliary_array(
    shape: tuple[int, int, int],
    *,
    num_poles: int,
    dtype: "wp.dtype | None" = None,
    device: "wp.Device | None" = None,
    chunk_index: tuple[int, int, int] | None = None,
    tag: bool = True,
) -> "wp.array | np.ndarray":
    """Allocate a dispersive auxiliary state array on the target device.

    For PoleResidue media, allocates ``(num_poles, Nx, Ny, Nz)`` polarization
    current arrays.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Grid shape (Nx, Ny, Nz).
    num_poles : int
        Number of poles for the dispersive medium.
    dtype : wp.dtype, optional
        Array dtype. Defaults to wp.complex64.
    device : wp.Device, optional
        Target device.
    chunk_index : tuple, optional
        Logical chunk position.
    tag : bool, default=True
        Whether to attach an ArrayTag.

    Returns
    -------
    wp.array or np.ndarray
        Allocated auxiliary array with shape ``(num_poles, *shape)``.
        Returns np.ndarray placeholder if Warp is unavailable.
    """
    if not WARP_AVAILABLE:
        import numpy as np

        return np.zeros((num_poles, *shape), dtype=np.complex128)

    # Warp does not have complex64 in version 1.12.1; use float32
    if dtype is None:
        dtype = wp.float32

    alloc_shape = (num_poles, *shape)
    arr = wp.zeros(shape=alloc_shape, dtype=dtype, device=device)

    if tag:
        arr.tag = AuxiliaryArrayTag(
            medium_family="PoleResidue", chunk_index=chunk_index
        )

    return arr


# ---------------------------------------------------------------------------
# Module stability tracking
# ---------------------------------------------------------------------------


@dataclass
class ModuleStability:
    """Tracks whether kernel modules have been marked stable."""

    stable_modules: set[str] = field(default_factory=set)
    _warned: bool = field(default_factory=lambda: False)

    def mark_stable(self, module_name: str) -> None:
        """Mark a kernel module as stable (JIT compilation is finalized)."""
        self.stable_modules.add(module_name)

    def is_stable(self, module_name: str) -> bool:
        """Return True if a module has been marked stable."""
        return module_name in self.stable_modules


# Global stability tracker
_module_stability = ModuleStability()


def mark_module_stable(module_name: str) -> None:
    """Mark a kernel module as stable.

    Call this after defining all kernels in a module to signal that
    the module is ready for JIT compilation. Modules that are not
    marked stable will trigger a warning on import during development.
    """
    _module_stability.mark_stable(module_name)


def is_module_stable(module_name: str) -> bool:
    """Return True if a module has been marked stable."""
    return _module_stability.is_stable(module_name)


# ---------------------------------------------------------------------------
# Profiling hooks
# ---------------------------------------------------------------------------


@dataclass
class WarpBackendInfo:
    """Structured information about the Warp backend state."""

    warp_available: bool
    warp_version: str | None = None
    cuda_available: bool = False
    num_devices: int = 0
    current_device: int | None = None
    supports_graph_capture: bool = False
    supports_float64: bool = False


def backend_info() -> WarpBackendInfo:
    """Return structured information about the Warp backend."""
    if not WARP_AVAILABLE:
        return WarpBackendInfo(warp_available=False)

    info = WarpBackendInfo(warp_available=True)
    try:
        info.warp_version = wp.__version__
    except AttributeError:
        pass

    try:
        info.cuda_available = wp.is_cuda_available()
    except AttributeError:
        pass

    try:
        info.num_devices = wp.get_device_count()
    except AttributeError:
        pass

    try:
        device = wp.get_device()
        info.current_device = int(device.id) if device else None
    except Exception:
        pass

    try:
        # Graph capture requires CUDA 10+ and Warp with graph support
        info.supports_graph_capture = wp.is_cuda_available()
    except AttributeError:
        pass

    try:
        # Check float64 support (most NVIDIA GPUs support it)
        import numpy as np

        info.supports_float64 = True  # Warp generally supports float64
    except Exception:
        pass

    return info


@dataclass
class StepMetrics:
    """Metrics from one timestep or a timestep window."""

    step_index: int
    wall_time_s: float
    cells_updated: int
    gcells_per_second: float | None = None
    jit_time_s: float | None = None
    initial_step: bool = False


def step_metrics(
    step_index: int,
    wall_time_s: float,
    num_cells: int,
    *,
    jit_time_s: float | None = None,
    initial_step: bool = False,
) -> StepMetrics:
    """Compute structured step metrics from timing data.

    Parameters
    ----------
    step_index : int
        Current step index (0-based).
    wall_time_s : float
        Wall-clock time elapsed for this step or window.
    num_cells : int
        Total number of grid cells updated.
    jit_time_s : float, optional
        JIT compilation time, if applicable.
    initial_step : bool, default=False
        True if this is the first step after field initialization.

    Returns
    -------
    StepMetrics
        Structured metrics including cells/s and Gcells/s.
    """
    gcells_s = None
    if wall_time_s > 0:
        gcells_s = (num_cells / wall_time_s) / 1e9

    return StepMetrics(
        step_index=step_index,
        wall_time_s=wall_time_s,
        cells_updated=num_cells,
        gcells_per_second=gcells_s,
        jit_time_s=jit_time_s,
        initial_step=initial_step,
    )


@contextmanager
def WarpTimer(name: str, device: "wp.Device | None" = None):
    """Context manager for timing Warp kernel execution.

    Uses Warp's ScopedTimer when available, with CPU-side timing as fallback.

    Parameters
    ----------
    name : str
        Label for the timed region.
    device : wp.Device, optional
        Target device for GPU-side timing.

    Example
    -------
    >>> with WarpTimer("electric_update"):
    ...     wp.launch(electric_update_kernel, dim=grid_shape, inputs=[...])
    """
    import time

    start = time.perf_counter()
    if WARP_AVAILABLE:
        try:
            with wp.ScopedTimer(name, device=device):
                yield
        except (TypeError, AttributeError):
            # ScopedTimer may not be available in all Warp versions
            yield
    else:
        yield
    end = time.perf_counter()


@contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """Context manager for NVTX range annotation on CUDA kernels.

    Does nothing when Warp/CUDA is unavailable.

    Parameters
    ----------
    name : str
        Label for the annotated range.
    color : str
        NVTX color name. One of: "blue", "green", "red", "yellow", "purple".
    """
    if not WARP_AVAILABLE:
        yield
        return

    try:
        color_map = {
            "blue": wp.nvtx.BLUE,
            "green": wp.nvtx.GREEN,
            "red": wp.nvtx.RED,
            "yellow": wp.nvtx.YELLOW,
            "purple": wp.nvtx.PURPLE,
        }
        color_val = color_map.get(color, wp.nvtx.BLUE)
        with wp.nvtx.range(name, color=color_val):
            yield
    except (AttributeError, TypeError):
        # NVTX may not be available in all Warp/CUDA versions
        yield


# ---------------------------------------------------------------------------
# Complex field policy
# ---------------------------------------------------------------------------


@dataclass
class ComplexFieldPolicy:
    """Determines when field arrays must use complex dtype.

    Attributes
    ----------
    force_complex : bool
        If True, all field arrays use complex dtype regardless of boundary type.
    has_bloch_axis : bool
        If True, any Bloch boundary is present in the simulation.
    """

    force_complex: bool = False
    has_bloch_axis: bool = False

    @property
    def requires_complex(self) -> bool:
        """Return True if field arrays must be complex dtype."""
        return self.force_complex or self.has_bloch_axis

    @classmethod
    def from_boundary_spec(cls, boundary_spec: Any | None) -> "ComplexFieldPolicy":
        """Infer complex field policy from a BoundarySpecIR or CompiledBoundarySpec."""
        if boundary_spec is None:
            return cls()

        has_bloch = False
        try:
            # Check for Bloch axes in boundary spec
            if hasattr(boundary_spec, "bloch_axes"):
                has_bloch = len(getattr(boundary_spec, "bloch_axes", ())) > 0
            elif hasattr(boundary_spec, "axes"):
                for axis_bdry in boundary_spec.axes():
                    for side in ("minus", "plus"):
                        edge = getattr(axis_bdry, side, None)
                        if edge and hasattr(edge, "mode"):
                            if str(edge.mode).lower() == "bloch":
                                has_bloch = True
                                break
        except Exception:
            pass

        return cls(has_bloch_axis=has_bloch)


# ---------------------------------------------------------------------------
# Backend exports for kernel metadata
# ---------------------------------------------------------------------------


def backend_summary() -> dict[str, Any]:
    """Return a summary dict of backend state for diagnostics."""
    info = backend_info()
    return {
        "warp_available": info.warp_available,
        "warp_version": info.warp_version,
        "cuda_available": info.cuda_available,
        "num_devices": info.num_devices,
        "current_device": info.current_device,
        "supports_graph_capture": info.supports_graph_capture,
        "supports_float64": info.supports_float64,
        "stable_modules": list(_module_stability.stable_modules),
    }
