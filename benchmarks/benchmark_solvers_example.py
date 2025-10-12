"""
Example benchmark file demonstrating performance measurement conventions.

This is about PERFORMANCE, not correctness.

Run with:
    python benchmarks/benchmark_solvers_example.py

Or with pytest-benchmark:
    pytest benchmarks/benchmark_solvers_example.py --benchmark-only
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


# Note: In real implementation, imports would be:
# from rde_solver.core.solvers import MUSCL1D, MUSCL2D


def benchmark_solver_scaling():
    """
    Benchmark how solver performance scales with problem size.

    This measures PERFORMANCE, not correctness.
    """
    print("=" * 60)
    print("Benchmarking Solver Scaling")
    print("=" * 60)

    # Test different grid sizes
    grid_sizes = [50, 100, 200, 400, 800]
    execution_times = []
    memory_usage = []

    for nx in grid_sizes:
        print(f"\nTesting nx = {nx}...")

        # Create solver
        # solver = MUSCL1D(nx=nx, cfl=0.5)

        # Time execution
        start = time.perf_counter()
        # result = solver.solve_sod_tube(t_end=0.2)
        time.sleep(0.01 * nx / 50)  # Placeholder - simulate scaling
        elapsed = time.perf_counter() - start

        execution_times.append(elapsed)

        # Memory usage (would use memory_profiler or similar)
        # mem = get_memory_usage()
        mem = nx * 1000  # Placeholder
        memory_usage.append(mem)

        print(f"  Time: {elapsed:.4f} s")
        print(f"  Memory: {mem / 1e6:.2f} MB")

    # Analyze scaling
    print("\n" + "=" * 60)
    print("Scaling Analysis")
    print("=" * 60)

    # Compute speedup factors
    for i in range(1, len(grid_sizes)):
        size_ratio = grid_sizes[i] / grid_sizes[i-1]
        time_ratio = execution_times[i] / execution_times[i-1]
        efficiency = size_ratio / time_ratio
        print(f"Grid {grid_sizes[i-1]} → {grid_sizes[i]}: "
              f"{time_ratio:.2f}x slower ({efficiency:.2%} efficient)")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Execution time
    ax1.loglog(grid_sizes, execution_times, 'o-', label='Actual')
    ax1.loglog(grid_sizes, np.array(grid_sizes) / grid_sizes[0] * execution_times[0],
               '--', label='O(N) ideal')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Solver Scaling: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Memory usage
    ax2.loglog(grid_sizes, memory_usage, 's-', label='Actual', color='orange')
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Memory Usage (bytes)')
    ax2.set_title('Solver Scaling: Memory Usage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "solver_scaling.png", dpi=150)
    print(f"\nPlot saved to {output_dir / 'solver_scaling.png'}")

    return {
        'grid_sizes': grid_sizes,
        'execution_times': execution_times,
        'memory_usage': memory_usage
    }


def benchmark_limiter_comparison():
    """Compare performance of different slope limiters"""
    print("\n" + "=" * 60)
    print("Benchmarking Slope Limiters")
    print("=" * 60)

    limiters = ["minmod", "superbee", "mc", "vanleer"]
    nx = 400
    n_iterations = 5

    results = {}

    for limiter in limiters:
        print(f"\nTesting {limiter} limiter...")

        times = []
        for i in range(n_iterations):
            # solver = MUSCL1D(nx=nx, limiter=limiter, cfl=0.5)

            start = time.perf_counter()
            # result = solver.solve_sod_tube(t_end=0.2)
            time.sleep(0.1)  # Placeholder
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        results[limiter] = {'mean': mean_time, 'std': std_time}

        print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f} s")

    # Compare to fastest
    fastest = min(results.values(), key=lambda x: x['mean'])['mean']
    print("\n" + "=" * 60)
    print("Relative Performance (vs fastest)")
    print("=" * 60)
    for limiter, result in results.items():
        relative = result['mean'] / fastest
        print(f"{limiter:15s}: {relative:.2f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    limiters_list = list(results.keys())
    means = [results[lim]['mean'] for lim in limiters_list]
    stds = [results[lim]['std'] for lim in limiters_list]

    ax.bar(limiters_list, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel('Execution Time (s)')
    ax.set_title(f'Slope Limiter Performance Comparison (nx={nx})')
    ax.grid(True, axis='y', alpha=0.3)

    output_dir = Path("benchmarks/results")
    plt.savefig(output_dir / "limiter_comparison.png", dpi=150)
    print(f"\nPlot saved to {output_dir / 'limiter_comparison.png'}")

    return results


def benchmark_1d_vs_2d():
    """Compare performance of 1D vs 2D solvers"""
    print("\n" + "=" * 60)
    print("Benchmarking 1D vs 2D Solvers")
    print("=" * 60)

    # 1D solver
    print("\n1D Solver:")
    nx_1d = 1000
    # solver_1d = MUSCL1D(nx=nx_1d)

    start = time.perf_counter()
    # result_1d = solver_1d.solve_sod_tube(t_end=0.2)
    time.sleep(0.2)  # Placeholder
    time_1d = time.perf_counter() - start

    print(f"  Grid: {nx_1d} cells")
    print(f"  Time: {time_1d:.4f} s")
    print(f"  Time per cell: {time_1d / nx_1d * 1e6:.2f} μs")

    # 2D solver (same total cells)
    print("\n2D Solver:")
    nx_2d = int(np.sqrt(nx_1d))  # ~31x31 for similar total cells
    ny_2d = nx_2d
    total_cells_2d = nx_2d * ny_2d
    # solver_2d = MUSCL2D(nx=nx_2d, ny=ny_2d)

    start = time.perf_counter()
    # result_2d = solver_2d.solve_sod_tube_2d(t_end=0.2)
    time.sleep(0.5)  # Placeholder - 2D is slower
    time_2d = time.perf_counter() - start

    print(f"  Grid: {nx_2d}x{ny_2d} = {total_cells_2d} cells")
    print(f"  Time: {time_2d:.4f} s")
    print(f"  Time per cell: {time_2d / total_cells_2d * 1e6:.2f} μs")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    overhead = (time_2d / total_cells_2d) / (time_1d / nx_1d)
    print(f"2D overhead per cell: {overhead:.2f}x")
    print(f"(2D solvers have more complex flux calculations)")

    return {'time_1d': time_1d, 'time_2d': time_2d}


# ============================================================================
# pytest-benchmark integration examples
# ============================================================================

def test_solver_performance_small(benchmark):
    """
    pytest-benchmark integration: small problem.

    Run with: pytest benchmarks/benchmark_solvers_example.py::test_solver_performance_small
    """
    def run_solver():
        # solver = MUSCL1D(nx=100)
        # return solver.solve_sod_tube(t_end=0.1)
        time.sleep(0.01)  # Placeholder
        return None

    result = benchmark(run_solver)

    # Optional: Assert performance requirements
    # assert benchmark.stats['mean'] < 0.1  # Must complete in < 0.1s


def test_solver_performance_medium(benchmark):
    """pytest-benchmark integration: medium problem"""
    def run_solver():
        # solver = MUSCL1D(nx=500)
        # return solver.solve_sod_tube(t_end=0.2)
        time.sleep(0.05)  # Placeholder
        return None

    benchmark(run_solver)


# ============================================================================
# Main benchmark suite
# ============================================================================

def run_solver_benchmarks(output_dir: Path = Path("benchmarks/results")):
    """
    Run complete solver benchmark suite.

    This is the main entry point when running:
        python benchmarks/benchmark_solvers_example.py

    Or when called from CLI:
        rde-benchmark --suite solvers
    """
    print("\n" + "=" * 60)
    print("RDE Solver Performance Benchmark Suite")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = {}

    results['scaling'] = benchmark_solver_scaling()
    results['limiters'] = benchmark_limiter_comparison()
    results['1d_vs_2d'] = benchmark_1d_vs_2d()

    # Save results
    import json
    results_file = output_dir / "solver_benchmark_results.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    serializable_results = {
        k: {kk: convert_to_serializable(vv) for kk, vv in v.items()}
        for k, v in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")

    return results


if __name__ == "__main__":
    """
    Run benchmarks as standalone script.

    This is DIFFERENT from tests:
    - Tests verify correctness (pytest)
    - Benchmarks measure performance (manual script)
    """
    import sys

    # Optional: Parse command line args
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("benchmarks/results")

    run_solver_benchmarks(output_dir)
