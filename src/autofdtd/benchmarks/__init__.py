"""GPU benchmarking suite for Phase 1 Maxwell solver."""

from autofdtd.benchmarks.gpu_benchmark import (
    GPUBenchmarkResult,
    benchmark_single_gpu,
    benchmark_multi_gpu,
    benchmark_scaling,
    benchmark_grid_sizes,
    benchmark_realistic,
    benchmark_realistic_scaling,
)

__all__ = [
    "GPUBenchmarkResult",
    "benchmark_single_gpu",
    "benchmark_multi_gpu",
    "benchmark_scaling",
    "benchmark_grid_sizes",
    "benchmark_realistic",
    "benchmark_realistic_scaling",
]