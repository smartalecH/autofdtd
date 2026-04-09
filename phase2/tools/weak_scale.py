"""
Weak-scale benchmark runner for Phase 2 accuracy validation.

Runs a simulation at N domain sizes at fixed grid resolution, tracks cells/s
throughput across each size, and verifies that throughput does not degrade by
more than 10% across the sweep (weak-scale throughput regression check).

This validates that the GPU backend sustains performance as the problem size
grows, catching issues like excessive memory allocation, JIT overhead at scale,
or inefficient halo exchange that would cause per-cell throughput to drop.

Usage
-----
    from phase2.tools.weak_scale import run_weak_scale_sweep

    def sim_fn(domain_size):
        # domain_size is a (Lx, Ly, Lz) tuple in µm
        sim = Simulation(
            center=(Lx/2, Ly/2, Lz/2),
            size=(Lx, Ly, Lz),
            grid_spec=GridSpec(
                grid_x=UniformGrid(dl=0.1),  # fixed resolution
                grid_y=UniformGrid(dl=0.1),
                grid_z=UniformGrid(dl=0.1),
            ),
            ...
        )
        compiled = compile_simulation(sim)
        return run_compiled_simulation(compiled)

    domain_sizes = [
        (10, 10, 10),
        (20, 20, 20),
        (40, 40, 40),
    ]

    result = run_weak_scale_sweep(
        simulation_fn=sim_fn,
        domain_sizes=domain_sizes,
        fixed_resolution=0.1,  # µm per cell
    )
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_weak_scale_sweep(
    simulation_fn: Callable[[tuple[float, float, float]], Any],
    domain_sizes: list[tuple[float, float, float]],
    fixed_resolution: float,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run weak-scale sweep at multiple domain sizes at fixed resolution.

    Parameters
    ----------
    simulation_fn : callable(domain_size) -> ExecutionResult
        Function that builds and runs a simulation at the given domain size.
        The domain_size is a (Lx, Ly, Lz) tuple in µm. Must return an
        ExecutionResult with metrics populated including 'cells_per_second'.
    domain_sizes : list of (Lx, Ly, Lz) tuples
        Domain sizes in µm to sweep. Each tuple is (size_x, size_y, size_z).
    fixed_resolution : float
        Grid resolution in µm (dl). Same resolution is used for all sizes.
    verbose : bool
        If True, print per-size results.

    Returns
    -------
    dict with keys:
        - domain_sizes : list[tuple[float, float, float]] — sorted sizes
        - num_cells : list[int] — total cells at each size
        - cells_per_second : list[float] — cells/s at each size
        - wall_times : list[float] — wall time (s) at each size
        - throughput_variance : float — relative std dev of cells/s across sizes
        - max_throughput_drop : float — max relative drop from max cells/s to min
        - pass_fail : str — "pass" if max_drop < 0.10, else "fail"
        - efficiency_ratio : float — min(cells/s) / max(cells/s)
        - details : list[dict] — per-size detail dicts
    """
    if len(domain_sizes) < 2:
        raise ValueError("Need at least 2 domain sizes for weak-scale validation")

    # Sort by total cell count for consistent ordering
    sorted_sizes = sorted(domain_sizes, key=lambda s: s[0] * s[1] * s[2])

    cells_per_second_list: list[float] = []
    wall_times: list[float] = []
    num_cells_list: list[int] = []
    details: list[dict] = []

    for i, size in enumerate(sorted_sizes):
        Lx, Ly, Lz = size
        num_cells = _estimate_cells(size, fixed_resolution)

        if verbose:
            print(f"  [{i+1}/{len(sorted_sizes)}] size={Lx}x{Ly}x{Lz} µm, "
                  f"resolution={fixed_resolution} µm, ~{num_cells:,} cells")

        # Run simulation
        result = simulation_fn(size)

        # Extract cells/s and wall time from metrics
        cps = result.metrics.get("cells_per_second", None)
        wt = result.metrics.get("wall_time", None)

        # Fallback: try to compute from other metrics
        if cps is None:
            cps = _compute_cells_per_second(result, num_cells)

        cells_per_second_list.append(cps)
        wall_times.append(wt if wt is not None else 0.0)
        num_cells_list.append(num_cells)

        if verbose:
            print(f"      cells/s={cps:.4e}  wall_time={wt:.3f}s" if wt else
                  f"      cells/s={cps:.4e}")

        detail = dict(
            domain_size=size,
            num_cells=num_cells,
            cells_per_second=cps,
            wall_time=wt,
        )
        details.append(detail)

    # Compute throughput statistics
    cps_array = np.array(cells_per_second_list, dtype=np.float64)

    # Mean and std of cells/s
    cps_mean = float(np.mean(cps_array))
    cps_std = float(np.std(cps_array))
    cps_cv = cps_std / cps_mean if cps_mean > 0 else 0.0  # coefficient of variation

    # Throughput drop: max(cells/s) - min(cells/s) relative to max
    cps_max = float(np.max(cps_array))
    cps_min = float(np.min(cps_array))
    max_drop = (cps_max - cps_min) / cps_max if cps_max > 0 else 0.0

    # Efficiency ratio: min / max (1.0 = perfect weak scaling)
    efficiency_ratio = cps_min / cps_max if cps_max > 0 else 0.0

    # Pass/fail: throughput drop < 10%
    pass_fail = "pass" if max_drop < 0.10 else "fail"

    return {
        "domain_sizes": sorted_sizes,
        "num_cells": num_cells_list,
        "cells_per_second": cells_per_second_list,
        "wall_times": wall_times,
        "throughput_mean": cps_mean,
        "throughput_std": cps_std,
        "throughput_variance": float(cps_cv),
        "max_throughput_drop": float(max_drop),
        "efficiency_ratio": float(efficiency_ratio),
        "pass_fail": pass_fail,
        "details": details,
    }


def _estimate_cells(
    domain_size: tuple[float, float, float],
    resolution: float,
) -> int:
    """Estimate total cell count from domain size and resolution.

    Parameters
    ----------
    domain_size : (Lx, Ly, Lz) in µm
    resolution : dl in µm

    Returns
    -------
    int
        Estimated number of cells (rounded up).
    """
    Lx, Ly, Lz = domain_size
    nx = max(1, int(np.ceil(Lx / resolution)))
    ny = max(1, int(np.ceil(Ly / resolution)))
    nz = max(1, int(np.ceil(Lz / resolution)))
    return nx * ny * nz


def _compute_cells_per_second(result: Any, num_cells: int) -> float | None:
    """Attempt to compute cells/s from ExecutionResult if not directly available.

    Checks wall_time and total_cells or cells_per_second metrics.
    """
    metrics = result.metrics

    # Try wall_time with total_cells
    if "wall_time" in metrics and "total_cells" in metrics:
        wt = metrics.get("wall_time")
        tc = metrics.get("total_cells")
        if wt and tc and wt > 0:
            return tc / wt

    # Try wall_time with num_cells computed
    if "wall_time" in metrics:
        wt = metrics.get("wall_time")
        if wt and wt > 0:
            return num_cells / wt

    return None


# ---------------------------------------------------------------------------
# Domain size helpers
# ---------------------------------------------------------------------------

def scale_domain(
    base_size: tuple[float, float, float],
    scale_factors: list[float],
) -> list[tuple[float, float, float]]:
    """Scale a base domain size by geometric factors.

    Parameters
    ----------
    base_size : (Lx, Ly, Lz) in µm
    scale_factors : list of float
        Multiplicative factors. 2.0 doubles each dimension.

    Returns
    -------
    list of (Lx, Ly, Lz) tuples
    """
    Lx, Ly, Lz = base_size
    return [(Lx * s, Ly * s, Lz * s) for s in scale_factors]


def cubic_domains(
    edge_sizes: list[float],
) -> list[tuple[float, float, float]]:
    """Create cubic domain size list.

    Parameters
    ----------
    edge_sizes : list of float
        Edge lengths in µm for each cubic domain.

    Returns
    -------
    list of (Lx, Ly, Lz) tuples (all cubic)
    """
    return [(s, s, s) for s in edge_sizes]


# ---------------------------------------------------------------------------
# Result pretty-printer
# ---------------------------------------------------------------------------

def format_weak_scale_result(result: dict) -> str:
    """Format a weak-scale sweep result as a readable string."""
    lines = [
        "Weak-Scale Sweep Results",
        "=" * 50,
        f"Domain sizes (µm): {[str(s) for s in result['domain_sizes']]}",
        f"Num cells:         {[f'{n:,}' for n in result['num_cells']]}",
        f"cells/s:           {[f'{c:.4e}' if c else 'N/A' for c in result['cells_per_second']]}",
        f"Wall times (s):    {[f'{w:.3f}' if w else 'N/A' for w in result['wall_times']]}",
        f"Throughput mean:   {result['throughput_mean']:.4e}",
        f"Throughput std:   {result['throughput_std']:.4e}",
        f"Variance (CV):    {result['throughput_variance']:.4f}",
        f"Max throughput drop: {result['max_throughput_drop']:.2%}",
        f"Efficiency ratio:  {result['efficiency_ratio']:.4f}",
        f"Pass/Fail:         {result['pass_fail']} (10% threshold)",
    ]
    return "\n".join(lines)
