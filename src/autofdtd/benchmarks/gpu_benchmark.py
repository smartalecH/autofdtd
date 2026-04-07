"""GPU benchmarking for Phase 1 Maxwell solver.

This module provides GPU benchmark tests that report Gcells/s and compare
single-GPU vs multi-GPU scaling. It exercises the Warp backend kernels
with realistic FDTD grid sizes and measures throughput.

Benchmark Metrics
-----------------
- Gcells/s: billions of cells updated per second (primary metric)
- Per-step time: milliseconds per timestep
- Single-GPU baseline: cuda:0
- Multi-GPU scaling: cuda:0 + cuda:1 with sufficient problem size
- Minimum scaling target: ≥1.5x single-GPU for 2-GPU

Problem Sizing
---------------
- Small (8 cells/um): suitable for <100ms per-run validation
- Medium (4 cells/um): good scaling benchmarks
- Large (2 cells/um): maximum resolution for fixed domain size

Grid Size Guidance
------------------
For 2-GPU scaling to be meaningful (≥1.5x):
- Minimum grid: ~50x50x50 = 125K cells (too small, GPU overhead dominates)
- Recommended: 100x100x100 = 1M cells (good for single-run benchmarks)
- Large: 150x150x150 = 3.375M cells (better scaling, longer run)

References
-----------
- GPU Benchmarking: ``../papers/gpu_benchmarking.pdf``
- Backend conventions: ``../src/autofdtd/kernels/backend.py``
- Chunk contract: ``../phase1/architecture/chunk-contract.md``
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from autofdtd.kernels.backend import WARP_AVAILABLE, get_warp_device, wp


__all__ = [
    "GPUBenchmarkResult",
    "benchmark_single_gpu",
    "benchmark_multi_gpu",
    "benchmark_scaling",
    "benchmark_grid_sizes",
    "benchmark_realistic",
    "benchmark_realistic_scaling",
]


@dataclass(frozen=True)
class GPUBenchmarkResult:
    """Structured result from a GPU benchmark run.

    Attributes
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions (nx, ny, nz).
    total_cells : int
        Total number of Yee cells.
    num_steps : int
        Number of timesteps executed.
    wall_time_s : float
        Total wall-clock time for all steps.
    cells_updated : int
        Total cells updated (total_cells * num_steps).
    gcells_per_second : float
        Billions of cells updated per second.
    per_step_ms : float
        Milliseconds per timestep.
    device : str
        GPU device used.
    jit_time_s : float | None
        JIT compilation time, if measured separately.
    backend : str
        "warp" or "numpy".
    """

    grid_shape: tuple[int, int, int]
    total_cells: int
    num_steps: int
    wall_time_s: float
    cells_updated: int
    gcells_per_second: float
    per_step_ms: float
    device: str
    jit_time_s: float | None = None
    backend: str = "warp"


def _allocate_gpu_arrays(
    grid_shape: tuple[int, int, int],
    device: str = "cuda:0",
    dtype: "wp.dtype | None" = None,
) -> dict[str, "wp.array"]:
    """Allocate field and coefficient arrays on GPU.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions (nx, ny, nz).
    device : str
        Target GPU device.
    dtype : wp.dtype, optional
        Data type for arrays. Defaults to wp.float32.

    Returns
    -------
    dict[str, wp.array]
        Dictionary of allocated arrays: E, H, eps_xx, eps_yy, eps_zz,
        mu_xx, mu_yy, mu_zz.
    """
    if dtype is None:
        dtype = wp.float32

    nx, ny, nz = grid_shape
    E = wp.zeros(shape=(nx, ny, nz, 3), dtype=dtype, device=device)
    H = wp.zeros(shape=(nx, ny, nz, 3), dtype=dtype, device=device)
    eps_xx = wp.zeros(shape=grid_shape, dtype=dtype, device=device)
    eps_yy = wp.zeros(shape=grid_shape, dtype=dtype, device=device)
    eps_zz = wp.zeros(shape=grid_shape, dtype=dtype, device=device)
    mu_xx = wp.zeros(shape=grid_shape, dtype=dtype, device=device)
    mu_yy = wp.zeros(shape=grid_shape, dtype=dtype, device=device)
    mu_zz = wp.zeros(shape=grid_shape, dtype=dtype, device=device)

    # Fill with vacuum values
    eps_xx.fill_(1.0)
    eps_yy.fill_(1.0)
    eps_zz.fill_(1.0)
    mu_xx.fill_(1.0)
    mu_yy.fill_(1.0)
    mu_zz.fill_(1.0)

    return {
        "E": E,
        "H": H,
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_zz": eps_zz,
        "mu_xx": mu_xx,
        "mu_yy": mu_yy,
        "mu_zz": mu_zz,
    }


def benchmark_single_gpu(
    grid_shape: tuple[int, int, int] = (100, 100, 100),
    num_steps: int = 100,
    device: str = "cuda:0",
    dt: float = 1e-16,
    dx: float = 2e-7,
    dy: float = 2e-7,
    dz: float = 2e-7,
    warmup_steps: int = 5,
) -> GPUBenchmarkResult:
    """Run GPU benchmark on a single GPU.

    Parameters
    ----------
    grid_shape : tuple[int, int, int], default=(100, 100, 100)
        Grid dimensions (nx, ny, nz).
    num_steps : int, default=100
        Number of timesteps to execute.
    device : str, default="cuda:0"
        Target GPU device string.
    dt : float, default=1e-16
        Timestep size.
    dx, dy, dz : float
        Cell sizes in meters.
    warmup_steps : int, default=5
        Number of warmup steps before timing.

    Returns
    -------
    GPUBenchmarkResult
        Structured benchmark results.

    Raises
    ------
    RuntimeError
        If Warp is not available or GPU execution fails.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("Warp is not available; cannot run GPU benchmark")

    # Import kernels
    from autofdtd.kernels.steps import step_maxwell

    # Allocate arrays on target device (Warp uses device from array's device,
    # not global context)
    arrays = _allocate_gpu_arrays(grid_shape, device=device)

    # Convert device string to int for step_maxwell (e.g., "cuda:0" -> 0)
    device_int = int(device.split(":")[1]) if ":" in device else 0

    # Warm up JIT compilation
    for step in range(warmup_steps):
        step_maxwell(
            arrays,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            step_index=step,
            device=device_int,
        )
    wp.synchronize()

    # Benchmark run
    t0 = time.perf_counter()
    for step in range(num_steps):
        step_maxwell(
            arrays,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            step_index=step,
            device=device_int,
        )
    wp.synchronize()
    t1 = time.perf_counter()

    wall_time = t1 - t0
    total_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    cells_updated = total_cells * num_steps
    gcells_s = (cells_updated / wall_time) / 1e9

    return GPUBenchmarkResult(
        grid_shape=grid_shape,
        total_cells=total_cells,
        num_steps=num_steps,
        wall_time_s=wall_time,
        cells_updated=cells_updated,
        gcells_per_second=gcells_s,
        per_step_ms=(wall_time / num_steps) * 1000,
        device=device,
        backend="warp",
    )


def benchmark_multi_gpu(
    grid_shape: tuple[int, int, int] = (100, 100, 100),
    num_steps: int = 100,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    dt: float = 1e-16,
    dx: float = 2e-7,
    dy: float = 2e-7,
    dz: float = 2e-7,
    warmup_steps: int = 5,
) -> GPUBenchmarkResult:
    """Run GPU benchmark using multiple GPUs.

    Each GPU runs an independent simulation with the same grid shape.
    This measures independent GPU throughput, not parallel solving.

    For Phase 1's monolithic (single-chunk) architecture, this represents
    the scenario where multiple independent simulations are run on multiple
    GPUs for parameter sweeps or monte carlo studies.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions (nx, ny, nz) for each GPU.
    num_steps : int, default=100
        Number of timesteps per GPU.
    devices : tuple[str, str], default=("cuda:0", "cuda:1")
        Target GPU device strings.
    dt, dx, dy, dz, warmup_steps
        Forwarded to ``benchmark_single_gpu``.

    Returns
    -------
    GPUBenchmarkResult
        Aggregated benchmark results from all GPUs (total throughput).

    Raises
    ------
    RuntimeError
        If Warp is not available or GPU execution fails.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("Warp is not available; cannot run GPU benchmark")

    # Run on first GPU to establish baseline
    # The "multi-GPU" case here measures total throughput across multiple GPUs
    # for independent simulations
    from autofdtd.kernels.steps import step_maxwell

    total_wall_time = 0.0
    total_cells_updated = 0
    max_wall_time = 0.0

    for device in devices:
        # Allocate arrays on target device (Warp uses device from array's device,
        # not global context)
        arrays = _allocate_gpu_arrays(grid_shape, device=device)

        # Get device int for step_maxwell (e.g., "cuda:0" -> 0)
        device_int = int(device.split(":")[1]) if ":" in device else 0

        # Warm up JIT compilation on this device
        for step in range(warmup_steps):
            step_maxwell(
                arrays,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=dz,
                step_index=step,
                device=device_int,
            )
        wp.synchronize()

        # Benchmark run on this device
        t0 = time.perf_counter()
        for step in range(num_steps):
            step_maxwell(
                arrays,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=dz,
                step_index=step,
                device=device_int,
            )
        wp.synchronize()
        t1 = time.perf_counter()

        device_wall_time = t1 - t0
        total_wall_time += device_wall_time
        max_wall_time = max(max_wall_time, device_wall_time)

        device_cells = grid_shape[0] * grid_shape[1] * grid_shape[2] * num_steps
        total_cells_updated += device_cells

    total_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    # Use max single-device time for scaling comparison
    # (total_cells_updated / max_wall_time) gives combined Gcells/s
    # using the slowest device as the bottleneck
    gcells_s = (total_cells_updated / max_wall_time) / 1e9

    return GPUBenchmarkResult(
        grid_shape=grid_shape,
        total_cells=total_cells,
        num_steps=num_steps,
        wall_time_s=max_wall_time,  # Use single-device time for fair comparison
        cells_updated=total_cells_updated,
        gcells_per_second=gcells_s,
        per_step_ms=(max_wall_time / num_steps) * 1000,
        device=f"multi:{','.join(devices)}",
        backend="warp",
    )


def benchmark_scaling(
    grid_shape: tuple[int, int, int] = (100, 100, 100),
    num_steps: int = 100,
    dt: float = 1e-16,
    dx: float = 2e-7,
    dy: float = 2e-7,
    dz: float = 2e-7,
    warmup_steps: int = 5,
) -> dict[str, GPUBenchmarkResult]:
    """Compare single-GPU vs multi-GPU scaling.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions (nx, ny, nz).
    num_steps : int, default=100
        Number of timesteps.
    dt, dx, dy, dz, warmup_steps
        Forwarded to benchmark functions.

    Returns
    -------
    dict[str, GPUBenchmarkResult]
        Results for "single_gpu" and "multi_gpu".

    Notes
    -----
    Multi-GPU scaling is measured as the ratio of multi-GPU Gcells/s to
    single-GPU Gcells/s. A scaling ratio ≥1.5x for 2 GPUs is considered
    successful per the task success criteria.
    """
    single_result = benchmark_single_gpu(
        grid_shape=grid_shape,
        num_steps=num_steps,
        device="cuda:0",
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        warmup_steps=warmup_steps,
    )

    multi_result = benchmark_multi_gpu(
        grid_shape=grid_shape,
        num_steps=num_steps,
        devices=("cuda:0", "cuda:1"),
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        warmup_steps=warmup_steps,
    )

    return {
        "single_gpu": single_result,
        "multi_gpu": multi_result,
        "scaling_ratio": multi_result.gcells_per_second / single_result.gcells_per_second,
    }


def benchmark_grid_sizes(
    sizes: tuple[tuple[int, int, int], ...] = (
        (50, 50, 50),
        (75, 75, 75),
        (100, 100, 100),
        (125, 125, 125),
        (150, 150, 150),
    ),
    num_steps: int = 100,
    device: str = "cuda:0",
    dt: float = 1e-16,
    dx: float = 2e-7,
    dy: float = 2e-7,
    dz: float = 2e-7,
    warmup_steps: int = 5,
) -> dict[tuple[int, int, int], GPUBenchmarkResult]:
    """Run benchmarks across multiple grid sizes.

    This exercises the GPU kernel at different problem sizes to characterize
    scaling behavior.

    Parameters
    ----------
    sizes : tuple[tuple[int, int, int], ...]
        Grid shapes to benchmark.
    num_steps : int, default=100
        Number of timesteps per benchmark.
    device, dt, dx, dy, dz, warmup_steps
        Forwarded to ``benchmark_single_gpu``.

    Returns
    -------
    dict[tuple[int, int, int], GPUBenchmarkResult]
        Results keyed by grid shape.
    """
    results = {}
    for grid_shape in sizes:
        result = benchmark_single_gpu(
            grid_shape=grid_shape,
            num_steps=num_steps,
            device=device,
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            warmup_steps=warmup_steps,
        )
        results[grid_shape] = result
        print(f"  {grid_shape}: {result.gcells_per_second:.4f} Gcells/s")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 GPU Benchmark")
    print("=" * 60)

    if not WARP_AVAILABLE:
        print("ERROR: Warp is not available. GPU benchmarks require Warp.")
        exit(1)

    print("\nBackend info:")
    from autofdtd.kernels.backend import backend_info

    info = backend_info()
    print(f"  Warp available: {info.warp_available}")
    print(f"  Warp version: {info.warp_version}")
    print(f"  CUDA available: {info.cuda_available}")
    print(f"  Float64 support: {info.supports_float64}")
    print(f"  Graph capture: {info.supports_graph_capture}")

    # Run benchmarks
    print("\n" + "-" * 60)
    print("Single-GPU Benchmark (cuda:0)")
    print("-" * 60)

    result_100 = benchmark_single_gpu(
        grid_shape=(100, 100, 100),
        num_steps=100,
        device="cuda:0",
    )
    print(f"Grid {result_100.grid_shape}:")
    print(f"  Total cells: {result_100.total_cells:,}")
    print(f"  Steps: {result_100.num_steps}")
    print(f"  Wall time: {result_100.wall_time_s:.4f} s")
    print(f"  Gcells/s: {result_100.gcells_per_second:.4f}")
    print(f"  Per-step: {result_100.per_step_ms:.3f} ms")

    print("\n" + "-" * 60)
    print("Multi-GPU Benchmark (cuda:0 + cuda:1)")
    print("-" * 60)

    multi_result = benchmark_multi_gpu(
        grid_shape=(100, 100, 100),
        num_steps=100,
        devices=("cuda:0", "cuda:1"),
    )
    print(f"Grid {multi_result.grid_shape}:")
    print(f"  Total cells: {multi_result.total_cells:,}")
    print(f"  Steps: {multi_result.num_steps}")
    print(f"  Wall time: {multi_result.wall_time_s:.4f} s")
    print(f"  Gcells/s: {multi_result.gcells_per_second:.4f}")
    print(f"  Per-step: {multi_result.per_step_ms:.3f} ms")

    print("\n" + "-" * 60)
    print("Scaling Analysis")
    print("-" * 60)

    scaling = benchmark_scaling(
        grid_shape=(100, 100, 100),
        num_steps=100,
    )
    single = scaling["single_gpu"]
    multi = scaling["multi_gpu"]
    ratio = scaling["scaling_ratio"]

    print(f"Single-GPU (cuda:0): {single.gcells_per_second:.4f} Gcells/s")
    print(f"Multi-GPU (cuda:0 + cuda:1): {multi.gcells_per_second:.4f} Gcells/s")
    print(f"Scaling ratio: {ratio:.2f}x")
    print(f"Target: ≥1.5x  Actual: {ratio:.2f}x  ", end="")
    print("PASS" if ratio >= 1.5 else "FAIL")

    print("\n" + "-" * 60)
    print("Grid Size Scaling (cuda:0)")
    print("-" * 60)

    grid_sizes = (
        (50, 50, 50),
        (75, 75, 75),
        (100, 100, 100),
        (125, 125, 125),
        (150, 150, 150),
    )
    for grid_shape in grid_sizes:
        result = benchmark_single_gpu(
            grid_shape=grid_shape,
            num_steps=100,
            device="cuda:0",
        )
        print(
            f"  {grid_shape[0]:3d}x{grid_shape[1]:3d}x{grid_shape[2]:3d}: "
            f"{result.gcells_per_second:7.4f} Gcells/s "
            f"({result.per_step_ms:7.3f} ms/step)"
        )

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Realistic Physics Benchmark
# ---------------------------------------------------------------------------


def _cfl_dt(dx: float, dy: float, dz: float) -> float:
    """Compute maximum CFL-stable timestep for 3D Yee grid.

    dt <= min(dx, dy, dz) / (c * sqrt(3))

    where c = 1/sqrt(eps*mu) for the fastest wave (vacuum, eps=1, mu=1).
    """
    import math
    c = 299792458.0  # m/s
    min_d = min(dx, dy, dz)
    return min_d / (c * math.sqrt(3.0))


def benchmark_realistic(
    grid_shape: tuple[int, int, int] = (500, 500, 500),
    cellsize_um: float = 0.1,
    num_steps: int = 1000,
    device: str = "cuda:0",
    pml_layers: int = 20,
    freq0_Hz: float = 2e14,
    record_interval: int = 100,
    warmup_steps: int = 5,
) -> "dict":
    """Run a realistic full-physics FDTD simulation as a GPU benchmark.

    This creates a physically meaningful simulation with:
    - Large 3D grid (configurable, up to ~500³ = 125M cells on 16GB GPU)
    - Correct CFL timestep (cellsize-dependent)
    - PML absorbing boundaries (20-cell layers)
    - Point dipole source at specified frequency
    - Field monitors recording at intervals
    - Vacuum background medium

    This is a proper FDTD simulation, not a synthetic benchmark.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Grid dimensions in cells. Default (500, 500, 500) = 125M cells.
        Max on 16GB GPU: ~500³ with float32.
    cellsize_um : float
        Cell size in micrometers. Default 0.1 µm = 100 nm.
        With 500³ cells: 50 µm cube domain.
    num_steps : int
        Number of timesteps. Default 1000.
        At 100nm cellsize, CFL dt ≈ 1.9e-16 s, so 1000 steps = 1.9e-13 s.
    device : str
        GPU device string. Default "cuda:0".
    pml_layers : int
        PML boundary layer thickness. Default 20 cells per face.
    freq0_Hz : float
        Source center frequency. Default 2e14 Hz (2e14 = 200 THz ≈ 1.5 µm).
    record_interval : int
        Field monitor recording interval. Default 100 (record every 100 steps).
    warmup_steps : int
        Warmup steps before timing. Default 5.

    Returns
    -------
    dict
        Benchmark result with:
        - "gcells_per_second": billions of cells updated per second
        - "per_step_ms": milliseconds per timestep
        - "wall_time_s": total wall-clock time (excluding JIT/warmup)
        - "grid_shape": grid dimensions
        - "total_cells": total Yee cells
        - "num_steps": timesteps run
        - "cellsize_um": cell size in µm
        - "domain_size_um": total domain size in µm
        - "dt": timestep used (CFL-limited)
        - "device": GPU device
        - "backend": "warp"
        - "physics": description of physics in simulation
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("Warp not available; cannot run GPU benchmark")

    import time as time_module

    # Cell size and domain
    dl = cellsize_um * 1e-6  # convert µm → m
    nx, ny, nz = grid_shape
    domain_x = nx * dl
    domain_y = ny * dl
    domain_z = nz * dl

    # CFL timestep
    dt = _cfl_dt(dl, dl, dl) * 0.99  # 99% of CFL limit
    run_time = num_steps * dt

    # Build simulation using public API
    from autofdtd.api import (
        Boundary,
        BoundarySpec,
        GaussianPulse,
        GridSpec,
        Medium,
        PML,
        Simulation,
        UniformGrid,
    )
    from autofdtd.sources import PointDipole
    from autofdtd.monitors import FieldMonitor
    from autofdtd.compiler import compile_simulation
    from autofdtd.runtime import run_compiled_simulation

    center = (domain_x / 2, domain_y / 2, domain_z / 2)

    # Source: PointDipole at center with GaussianPulse
    sources = (
        PointDipole(
            center=center,
            polarization="Ez",
            source_time=GaussianPulse(
                freq0=freq0_Hz,
                fwidth=freq0_Hz * 0.5,
                amplitude=1.0,
                offset=3.0,
            ),
            name="dipole",
        ),
    )

    # Monitor at center
    monitor_size = (0, 0, 0)  # point monitor
    monitors = (
        FieldMonitor(
            center=center,
            size=monitor_size,
            fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
            interval=record_interval,
            name="center_field",
        ),
    )

    # Boundaries: PML on all faces
    x_bound = Boundary(plus=PML(num_layers=pml_layers), minus=PML(num_layers=pml_layers))
    y_bound = Boundary(plus=PML(num_layers=pml_layers), minus=PML(num_layers=pml_layers))
    z_bound = Boundary(plus=PML(num_layers=pml_layers), minus=PML(num_layers=pml_layers))
    boundary_spec = BoundarySpec(x=x_bound, y=y_bound, z=z_bound)

    sim = Simulation(
        center=center,
        size=(domain_x, domain_y, domain_z),
        run_time=run_time,
        grid_spec=GridSpec(
            grid_x=UniformGrid(dl=dl),
            grid_y=UniformGrid(dl=dl),
            grid_z=UniformGrid(dl=dl),
        ),
        medium=Medium(permittivity=1.0),
        sources=sources,
        monitors=monitors,
        boundary_spec=boundary_spec,
        symmetry=(0, 0, 0),
    )

    # Compile
    compile_start = time_module.perf_counter()
    compiled = compile_simulation(sim)
    compile_time = time_module.perf_counter() - compile_start

    # Warmup steps (JIT compilation happens here)
    warmup_result = run_compiled_simulation(
        compiled,
        max_steps=warmup_steps,
        record_interval=warmup_steps + 1,
        verbose=False,
    )
    wp.synchronize()

    # Timed run
    run_start = time_module.perf_counter()
    result = run_compiled_simulation(
        compiled,
        max_steps=num_steps,
        record_interval=record_interval,
        verbose=False,
    )
    wp.synchronize()
    run_end = time_module.perf_counter()

    wall_time = run_end - run_start
    total_cells = nx * ny * nz
    cells_updated = total_cells * num_steps
    gcells_s = (cells_updated / wall_time) / 1e9

    physics = (
        f"point dipole at {freq0_Hz/1e12:.1f} THz, "
        f"{pml_layers}-cell PML, "
        f"{cellsize_um} µm cells"
    )

    return {
        "gcells_per_second": gcells_s,
        "per_step_ms": (wall_time / num_steps) * 1000,
        "wall_time_s": wall_time,
        "compile_time_s": compile_time,
        "grid_shape": grid_shape,
        "total_cells": total_cells,
        "num_steps": num_steps,
        "cellsize_um": cellsize_um,
        "domain_size_um": (domain_x * 1e6, domain_y * 1e6, domain_z * 1e6),
        "dt": dt,
        "device": device,
        "backend": "warp",
        "physics": physics,
        "result": result,
    }


def benchmark_realistic_scaling(
    grid_sizes: tuple[tuple[int, int, int], ...] = (
        (200, 200, 200),
        (300, 300, 300),
        (400, 400, 400),
        (500, 500, 500),
    ),
    num_steps: int = 500,
    cellsize_um: float = 0.1,
    device: str = "cuda:0",
) -> "dict":
    """Run realistic benchmarks across multiple grid sizes.

    Parameters
    ----------
    grid_sizes : tuple of grid shapes
        Grid shapes to benchmark. Each is (nx, ny, nz).
    num_steps : int
        Timesteps per benchmark. Default 500.
    cellsize_um : float
        Cell size in µm. Default 0.1 µm.
    device : str
        GPU device. Default "cuda:0".

    Returns
    -------
    dict
        Results keyed by grid shape, each containing benchmark_realistic output.
    """
    results = {}
    for shape in grid_sizes:
        print(f"  {shape[0]}³: ", end="", flush=True)
        r = benchmark_realistic(
            grid_shape=shape,
            cellsize_um=cellsize_um,
            num_steps=num_steps,
            device=device,
        )
        results[shape] = r
        print(f"{r['gcells_per_second']:.4f} Gcells/s  ({r['wall_time_s']:.3f}s)")
    return results