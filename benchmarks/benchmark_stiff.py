import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import cantera as ct
import arrhenius as ar
from functools import partial

def benchmark_cantera_vs_jax(sizes=[2, 4, 6, 8, 10, 12], num_steps=100, dt=1e-6,
                            solvers=["Tsit5", "Kvaerno3", "Kvaerno5"]):
    """
    Compare performance of Cantera vs JAX implementation for different grid sizes.

    Args:
        sizes: List of grid sizes to test (number of cells)
        num_steps: Number of integration steps
        dt: Time step for integration
        solvers: List of diffrax solvers to test

    Returns:
        Dictionary with benchmark results
    """
    print(f"Starting benchmark: grid sizes {sizes}, {num_steps} steps, dt={dt}")
    print(f"Testing solvers: {solvers}")

    # Build mechanism once
    mech = ar.build_mechanism("h2o2.yaml")

    # Initial conditions - H2/O2/N2 mixture
    Y0 = {
        "H2": 0.02,   # 2% hydrogen
        "O2": 0.21,   # 21% oxygen
        "N2": 0.77,   # 77% nitrogen
    }

    # Convert to array format
    Y0_arr = ar.dict_to_array(Y0, mech["names"])

    # Add energy (300K)
    T0 = 300.0  # K
    e0 = ar.temp_to_energy(T0, Y0_arr, mech)
    y0 = jnp.append(Y0_arr, e0)

    # Conditions
    rho = 1.0  # kg/m³

    # Prepare Cantera solution
    gas = ct.Solution("h2o2.yaml")
    species_names = gas.species_names

    # Set up cantera initial state
    gas.TPY = T0, ct.one_atm, Y0

    # Results storage
    results = {
        "sizes": sizes,
        "cantera_times": [],
        "jax_times": {solver: [] for solver in solvers},
        "speedups": {solver: [] for solver in solvers},
    }

    # Test each grid size
    for size in sizes:
        print(f"\nBenchmarking grid size: {size}")

        # Create grid of identical cells
        y0_grid = jnp.tile(y0, (size, 1))

        # Prepare Cantera reactors
        reactors = []
        for _ in range(size):
            gas.TPY = T0, ct.one_atm, Y0
            r = ct.IdealGasConstPressureReactor(contents=gas)
            reactors.append(r)

        # Run Cantera benchmark
        print("Running Cantera benchmark...")
        start_time = time.time()

        for step in range(num_steps):
            t_current = step * dt
            t_next = (step + 1) * dt

            for r in reactors:
                r.advance(t_next)

        cantera_time = time.time() - start_time
        results["cantera_times"].append(cantera_time)
        print(f"Cantera time for {size} cells: {cantera_time:.4f} seconds")

        # Define modified integration functions with different solvers
        def integrate_with_solver(solver_name, y0_grid):
            # Customize the integration function to use a specific solver
            term = dfx.ODETerm(ar.rhs_grid)

            if solver_name == "Tsit5":
                solver = dfx.Tsit5()  # Default, non-stiff solver
            elif solver_name == "Kvaerno3":
                solver = dfx.Kvaerno3()  # 3rd order stiff solver
            elif solver_name == "Kvaerno5":
                solver = dfx.Kvaerno5()  # 5th order stiff solver
            else:
                raise ValueError(f"Unknown solver: {solver_name}")

            steps = dfx.PIDController(rtol=1e-5, atol=1e-8, pcoeff=0.2, dcoeff=0.0)
            saveat = dfx.SaveAt(t1=True)

            sol = dfx.diffeqsolve(
                term, solver, t0=0.0, t1=num_steps*dt, dt0=dt/10,
                y0=y0_grid, args=(mech, rho),
                stepsize_controller=steps, saveat=saveat,
                max_steps=1000*num_steps, throw=True
            )
            return sol

        # JIT compile the integration functions
        for solver_name in solvers:
            print(f"Compiling {solver_name} solver...")
            integrate_jit = eqx.filter_jit(partial(integrate_with_solver, solver_name))

            # Run once to compile (warmup)
            _ = integrate_jit(y0_grid)

            # Run benchmark
            print(f"Running JAX benchmark with {solver_name}...")
            start_time = time.time()
            sol = integrate_jit(y0_grid)
            jax_time = time.time() - start_time

            results["jax_times"][solver_name].append(jax_time)
            speedup = cantera_time / jax_time
            results["speedups"][solver_name].append(speedup)

            print(f"JAX time ({solver_name}) for {size} cells: {jax_time:.4f} seconds")
            print(f"Speedup ({solver_name}): {speedup:.2f}x")

    # Generate plots
    plt.figure(figsize=(12, 10))

    # Plot 1: Execution times
    plt.subplot(2, 1, 1)
    plt.plot(sizes, results["cantera_times"], 'o-', label='Cantera')
    for solver in solvers:
        plt.plot(sizes, results["jax_times"][solver], 'o-', label=f'JAX ({solver})')
    plt.xlabel('Grid Size (Number of Cells)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Cantera vs JAX Performance Comparison')
    plt.legend()
    plt.grid(True)

    # Plot 2: Speedup factor
    plt.subplot(2, 1, 2)
    for solver in solvers:
        plt.plot(sizes, results["speedups"][solver], 'o-', label=f'Speedup ({solver})')
    plt.xlabel('Grid Size (Number of Cells)')
    plt.ylabel('Speedup Factor (Cantera time / JAX time)')
    plt.title('JAX Speedup over Cantera')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()

    return results

if __name__ == "__main__":
    # Run benchmark with default parameters
    results = benchmark_cantera_vs_jax()

    # Print summary
    print("\nBenchmark Summary:")
    print("=================")

    for i, size in enumerate(results["sizes"]):
        print(f"\nGrid Size: {size} cells")
        print(f"Cantera: {results['cantera_times'][i]:.4f} seconds")

        for solver in results["jax_times"].keys():
            jax_time = results["jax_times"][solver][i]
            speedup = results["speedups"][solver][i]
            print(f"JAX ({solver}): {jax_time:.4f} seconds (Speedup: {speedup:.2f}x)")
