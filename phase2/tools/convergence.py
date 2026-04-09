"""
Convergence sweep runner for Phase 2 accuracy validation.

Runs a simulation at N grid resolutions, measures error vs. analytical/numerical
ground truth, fits convergence rate via log-log linear regression, and reports
pass/fail against the 1.5-order minimum threshold for a 2nd-order FDTD scheme.

Usage
-----
    from phase2.tools.convergence import run_convergence_sweep

    def sim_fn(resolution):
        sim = Simulation(...)
        sim.grid_spec = GridSpec(
            grid_x=UniformGrid(dl=resolution),
            grid_y=UniformGrid(dl=resolution),
            grid_z=UniformGrid(dl=resolution),
        )
        compiled = compile_simulation(sim)
        return run_compiled_simulation(compiled)

    def gt_fn(resolution):
        return analytical_field(...)  # same geometry at given resolution

    result = run_convergence_sweep(
        simulation_fn=sim_fn,
        resolutions=[0.2, 0.1, 0.05],  # dl values in µm
        ground_truth_fn=gt_fn,
        error_metric="L2",
    )
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_convergence_sweep(
    simulation_fn: Callable[[float], Any],
    resolutions: list[float],
    ground_truth_fn: Callable[[float], Any],
    error_metric: Literal["L2", "max", "R", "T"] = "L2",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run convergence sweep at multiple grid resolutions.

    Parameters
    ----------
    simulation_fn : callable(resolution) -> ExecutionResult
        Function that builds and runs a simulation at the given resolution
        (dl in µm). Must return an ExecutionResult with field_monitor_data
        and metrics populated.
    resolutions : list of float
        Grid resolutions (dl in µm) at which to run. Will be sorted internally.
    ground_truth_fn : callable(resolution) -> Any
        Function that returns the analytical/numerical ground truth at the
        given resolution. Returns a scalar for 'R'/'T' metrics or an array for
        'L2'/'max' field metrics.
    error_metric : "L2" | "max" | "R" | "T"
        - "L2": L2-norm of field error vs. ground_truth array, normalized
        - "max": max absolute field error vs. ground_truth array
        - "R": absolute relative error in reflectance vs. ground_truth scalar
        - "T": absolute relative error in transmittance vs. ground_truth scalar
    verbose : bool
        If True, print per-resolution results.

    Returns
    -------
    dict with keys:
        - resolutions : list[float] — sorted resolutions
        - errors : list[float] — error at each resolution
        - rates : list[float] — pairwise convergence rates (n-1 entries)
        - rate : float — fitted log-log slope (convergence order)
        - rate_std : float — standard error of the fitted rate
        - pass_fail : str — "pass" if rate >= 1.5, else "fail"
        - pass_fail_at_finest : str — "pass" if finest error < 0.01, else "fail"
        - cells_per_second : list[float] — cells/s at each resolution
        - num_cells : list[int] — total cells at each resolution
        - details : list[dict] — per-resolution detail dicts
    """
    if len(resolutions) < 2:
        raise ValueError("Need at least 2 resolutions to measure convergence rate")

    # Sort resolutions low → high
    sorted_pairs = sorted(enumerate(resolutions), key=lambda x: x[1])
    sort_idx = [p[0] for p in sorted_pairs]
    resolutions_sorted = [p[1] for p in sorted_pairs]

    errors: list[float] = []
    rates: list[float] = []
    cells_per_second_list: list[float] = []
    num_cells_list: list[int] = []
    details: list[dict] = []

    prev_error: float | None = None
    prev_resolution: float | None = None

    for i, res in enumerate(resolutions_sorted):
        if verbose:
            print(f"  [{i+1}/{len(resolutions_sorted)}] resolution={res:.4f} µm")

        # Run simulation
        result = simulation_fn(res)

        # Extract ground truth
        gt = ground_truth_fn(res)

        # Compute error based on metric type
        error = _compute_error(result, gt, error_metric)

        # Extract cells/s metric
        cps = result.metrics.get("cells_per_second", None)
        num_cells = _estimate_num_cells(result)

        errors.append(error)
        cells_per_second_list.append(cps)
        num_cells_list.append(num_cells)

        if verbose:
            print(f"      error={error:.6e}  cells/s={cps}  cells={num_cells}")

        # Pairwise rate
        if prev_error is not None and prev_resolution is not None:
            # error ~ resolution^order  →  log error = order * log resolution + const
            # order = (log error2 - log error1) / (log res2 - log res1)
            if prev_resolution > 0 and res > 0 and prev_error > 0 and error > 0:
                rate = (np.log(error) - np.log(prev_error)) / (
                    np.log(res) - np.log(prev_resolution)
                )
                rates.append(rate)
                if verbose:
                    print(f"      pairwise_rate={rate:.3f}")
            else:
                rates.append(np.nan)

        prev_error = error
        prev_resolution = res

        detail = dict(
            resolution=res,
            error=error,
            cells_per_second=cps,
            num_cells=num_cells,
        )
        if len(errors) >= 2:
            detail["pairwise_rate"] = rates[-1] if rates else None
        details.append(detail)

    # Fit log-log linear regression: log(error) = order * log(res) + const
    # Use resolutions as cell size (the independent variable)
    log_res = np.log(np.array(resolutions_sorted, dtype=np.float64))
    log_err = np.log(np.array(errors, dtype=np.float64))

    # Linear regression using numpy polyfit (1st-order polynomial)
    # y = intercept + slope * x  →  log(error) = intercept + slope * log(res)
    slope, intercept = np.polyfit(log_res, log_err, 1)

    # R² coefficient of determination
    y_mean = np.mean(log_err)
    ss_tot = np.sum((log_err - y_mean) ** 2)
    ss_res = np.sum((log_err - (intercept + slope * log_res)) ** 2)
    r_squared = ss_res / ss_tot if ss_tot != 0 else 0.0

    # Standard error of slope estimate
    n = len(log_res)
    if n > 2:
        mse = ss_res / (n - 2)
        var_slope = mse / np.sum((log_res - np.mean(log_res)) ** 2)
        std_err = np.sqrt(var_slope) if var_slope > 0 else 0.0
    else:
        std_err = np.nan

    # P-value is not available without scipy — set to NaN
    p_value = np.nan

    # Fitted convergence rate = slope
    fitted_rate = float(slope)

    # Pass/fail: rate >= 1.5 (1.5-order minimum for 2nd-order FDTD)
    pass_fail = "pass" if fitted_rate >= 1.5 else "fail"

    # Additional pass/fail: finest error < 1%
    finest_error = errors[-1]
    pass_fail_at_finest = "pass" if finest_error < 0.01 else "fail"

    return {
        "resolutions": resolutions_sorted,
        "errors": errors,
        "rates": rates,
        "rate": fitted_rate,
        "rate_std": float(std_err),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "pass_fail": pass_fail,
        "pass_fail_at_finest": pass_fail_at_finest,
        "cells_per_second": cells_per_second_list,
        "num_cells": num_cells_list,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Error computation helpers
# ---------------------------------------------------------------------------

def _compute_error(
    result: Any,
    ground_truth: Any,
    error_metric: Literal["L2", "max", "R", "T"],
) -> float:
    """Compute error between ExecutionResult and ground truth.

    Parameters
    ----------
    result : ExecutionResult
        Simulation result with field_monitor_data, flux_monitor_data.
    ground_truth : array | scalar
        Analytical ground truth (array for field metrics, scalar for R/T).
    error_metric : str
        One of "L2", "max", "R", "T".

    Returns
    -------
    float
        Computed error value.
    """
    if error_metric in ("L2", "max"):
        return _field_error(result, ground_truth, error_metric)
    elif error_metric in ("R", "T"):
        return _rt_error(result, ground_truth, error_metric)
    else:
        raise ValueError(f"Unknown error_metric: {error_metric}")


def _field_error(
    result: Any,
    ground_truth: Any,
    error_metric: Literal["L2", "max"],
) -> float:
    """Compute L2 or max field error vs. ground truth array.

    Parameters
    ----------
    result : ExecutionResult
        Must have field_monitor_data with at least one monitor.
    ground_truth : array-like
        Reference field array (same shape as monitor output).
    error_metric : "L2" or "max"

    Returns
    -------
    float
        Normalized L2 error or max absolute error.
    """
    fmd = result.field_monitor_data
    if not fmd:
        raise ValueError("No field_monitor_data in result")

    # Take the first field monitor
    monitor_name = next(iter(fmd))
    monitor_data = fmd[monitor_name]

    # Extract E field — handle different storage formats
    # FieldMonitor data may be stored as dict with 'Ex', 'Ey', 'Ez' keys
    # or as a single complex array
    if isinstance(monitor_data, dict):
        # Complex field components — try Ex/Ey/Ez first, then Hx/Hy/Hz
        field_keys = list(monitor_data.keys())
        # Check for full Ex/Ey/Ez set
        has_ex = "Ex" in monitor_data
        has_hx = "Hx" in monitor_data

        if has_ex:
            # Build 3-component field from available components
            components = []
            for key in ["Ex", "Ey", "Ez"]:
                if key in monitor_data:
                    components.append(np.asarray(monitor_data[key], dtype=np.complex128))
                else:
                    # Zero-fill missing components
                    ref = next(iter(monitor_data.values()))
                    arr = np.asarray(ref)
                    components.append(np.zeros_like(arr, dtype=np.complex128))
            sim_field = np.stack(components, axis=-1)
        elif has_hx:
            components = []
            for key in ["Hx", "Hy", "Hz"]:
                if key in monitor_data:
                    components.append(np.asarray(monitor_data[key], dtype=np.complex128))
                else:
                    ref = next(iter(monitor_data.values()))
                    arr = np.asarray(ref)
                    components.append(np.zeros_like(arr, dtype=np.complex128))
            sim_field = np.stack(components, axis=-1)
        else:
            # Fallback: use first field in the dict
            first_key = next(iter(monitor_data.values()))
            sim_field = np.asarray(first_key, dtype=np.complex128)
    else:
        sim_field = np.asarray(monitor_data, dtype=np.complex128)

    gt_field = np.asarray(ground_truth, dtype=np.complex128)

    # Broadcast ground truth to match sim field shape if needed
    # Determine original ground truth dimensionality
    gt_original_ndim = np.asarray(ground_truth).ndim

    if sim_field.ndim == 4 and gt_original_ndim == 3:
        # gt_field is (nx, ny, nz), sim_field is (nx, ny, nz, 3)
        # Tile across all 3 field components
        gt_3d = np.asarray(ground_truth, dtype=np.complex128)
        gt_field = np.tile(gt_3d[..., np.newaxis], (1, 1, 1, 3))
    elif sim_field.ndim == 4 and gt_field.ndim == 4 and gt_field.shape[-1] == 1:
        # Single-component gt (4D with 1 component) tiled to match all 3 components
        gt_field = np.tile(gt_field, (1, 1, 1, 3))

    if sim_field.shape != gt_field.shape:
        raise ValueError(
            f"Field shape mismatch: sim={sim_field.shape} gt={gt_field.shape}. "
            "Ensure ground_truth_fn returns an array matching the monitor shape."
        )

    diff = np.abs(sim_field - gt_field)

    if error_metric == "L2":
        # Normalized L2 error: ||sim - gt||_2 / ||gt||_2
        numerator = np.sqrt(np.sum(diff**2))
        denominator = np.sqrt(np.sum(np.abs(gt_field) ** 2))
        if denominator == 0:
            return float("inf")
        return float(numerator / denominator)
    else:  # "max"
        return float(np.max(diff))


def _rt_error(
    result: Any,
    ground_truth: float,
    error_metric: Literal["R", "T"],
) -> float:
    """Compute absolute relative error in reflectance or transmittance.

    Parameters
    ----------
    result : ExecutionResult
        Must have flux_monitor_data with 'R' or 'T' monitor.
    ground_truth : float
        Reference reflectance or transmittance value.
    error_metric : "R" or "T"

    Returns
    -------
    float
        Absolute relative error |R_fdtd - R_gt| / R_gt or |T_fdtd - T_gt| / T_gt.
    """
    fmd = result.flux_monitor_data
    if not fmd:
        raise ValueError("No flux_monitor_data in result")

    # Look for flux monitor matching R or T
    # Convention: monitor names containing 'R' → reflectance, 'T' → transmittance
    monitor_key = None
    for name in fmd:
        if error_metric == "R" and "R" in name.upper():
            monitor_key = name
            break
        elif error_metric == "T" and "T" in name.upper():
            monitor_key = name
            break

    if monitor_key is None:
        # Fallback: use first flux monitor
        monitor_key = next(iter(fmd))

    flux_data = fmd[monitor_key]

    # Flux data may be a scalar or an array with time axis
    # Take the final (steady-state) flux value
    if hasattr(flux_data, "__iter__"):
        flux_val = float(np.mean(np.asarray(flux_data).flat[-10:]))
    else:
        flux_val = float(flux_data)

    # Normalized error
    if ground_truth == 0:
        return abs(flux_val)
    return abs(flux_val - ground_truth) / abs(ground_truth)


def _estimate_num_cells(result: Any) -> int:
    """Estimate total number of cells from ExecutionResult metrics or field shape."""
    metrics = result.metrics

    # Try cells_per_second metric: total_cells = cells_per_second * wall_time
    if "cells_per_second" in metrics and "wall_time" in metrics:
        cps = metrics["cells_per_second"]
        wt = metrics["wall_time"]
        if cps and wt:
            return int(cps * wt)

    if "total_cells" in metrics:
        return int(metrics["total_cells"])

    # Fallback: estimate from field_monitor_data shape
    fmd = result.field_monitor_data
    if fmd:
        monitor_name = next(iter(fmd))
        monitor_data = fmd[monitor_name]
        if isinstance(monitor_data, dict):
            first_key = next(iter(monitor_data.values()))
            arr = np.asarray(first_key)
        else:
            arr = np.asarray(monitor_data)
        if arr.ndim >= 3:
            return int(np.prod(arr.shape[:3]))

    return 0


# ---------------------------------------------------------------------------
# Convenience: build resolution list from ppw
# ---------------------------------------------------------------------------

def resolutions_from_ppw(
    wavelength: float,
    ppw_values: list[float],
    base_um: float = 1.0,
) -> list[float]:
    """Convert points-per-wavelength values to dl (µm) resolutions.

    Parameters
    ----------
    wavelength : float
        Wavelength in µm.
    ppw_values : list of float
        Points per wavelength values.
    base_um : float
        Base unit (wavelength multiplier for domain size).

    Returns
    -------
    list of float
        dl values in µm.
    """
    return [wavelength / ppw * base_um for ppw in ppw_values]


def ppw_from_resolutions(
    wavelength: float,
    resolutions: list[float],
    base_um: float = 1.0,
) -> list[float]:
    """Convert dl (µm) resolutions to points-per-wavelength values.

    Parameters
    ----------
    wavelength : float
        Wavelength in µm.
    resolutions : list of float
        dl values in µm.
    base_um : float
        Base unit.

    Returns
    -------
    list of float
        ppw values.
    """
    return [wavelength / (res / base_um) for res in resolutions]


# ---------------------------------------------------------------------------
# Result pretty-printer
# ---------------------------------------------------------------------------

def format_convergence_result(result: dict) -> str:
    """Format a convergence sweep result as a readable string."""
    lines = [
        "Convergence Sweep Results",
        "=" * 50,
        f"Resolutions (µm): {[f'{r:.4f}' for r in result['resolutions']]}",
        f"Errors:           {[f'{e:.4e}' for e in result['errors']]}",
        f"Pairwise rates:   {[f'{r:.3f}' for r in result['rates']] if result['rates'] else 'N/A'}",
        f"Fitted rate:      {result['rate']:.3f} ± {result['rate_std']:.3f}",
        f"R²:               {result['r_squared']:.4f}",
        f"Pass/Fail (rate): {result['pass_fail']} (threshold 1.5)",
        f"Pass/Fail (error): {result['pass_fail_at_finest']} (finest error < 1%)",
        f"cells/s:          {[f'{c:.2e}' if c else 'N/A' for c in result['cells_per_second']]}",
    ]
    return "\n".join(lines)
