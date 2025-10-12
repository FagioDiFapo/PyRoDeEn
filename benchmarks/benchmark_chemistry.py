"""
Cantera vs JAX Chemistry Performance Benchmark

This script compares the performance of chemical kinetics integration using JAX vs Cantera.
It tests different grid sizes to show how each approach scales with problem size.
"""

import numpy as np
import jax
import jax.numpy as jnp
import cantera as ct
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from arrhenius import build_mechanism, T_from_e_newton, integrate_grid

# Set JAX to use GPU
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

def benchmark_chemistry(sizes=None, warmup=True):
    """
    Run benchmark comparing Cantera vs JAX chemistry performance.
    
    Parameters:
    -----------
    sizes : list, optional
        List of grid sizes to test. Default is [2, 4, 6, 8, 10, 12, 14]
    warmup : bool, optional
        Whether to run a warmup iteration to compile JAX functions
    
    Returns:
    --------
    dict
        Dictionary with benchmark results
    """
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK: CANTERA vs JAX CHEMISTRY")
    print("="*80)
    
    # Default sizes to test
    if sizes is None:
        sizes = [2, 4, 6, 8, 10, 12, 14]
    
    # Storage for timing results
    jax_times = []
    cantera_times = []
    
    # Build the mechanism once
    print("Building mechanism...")
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]
    
    # Indices for key species
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    
    # Fixed parameters for all tests
    rho = 1.2        # kg/m³
    T0 = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time
    
    # Helper function to create initial composition
    def initial_Y():
        Y = np.zeros(S, dtype=float)
        Y[iH2] = 0.03
        Y[iO2] = 0.20
        Y[iH2O] = 1.0 - Y[iH2] - Y[iO2]
        # Add tiny seeds to avoid "zero-radical trap"
        eps = 1e-20
        zero_mask = (Y <= 0.0)
        add = zero_mask.sum() * eps
        imax = np.argmax(Y)
        Y = Y + eps * zero_mask
        Y[imax] -= add
        # Renormalize
        Y = np.clip(Y, 0.0, None)
        Y /= Y.sum()
        return Y
    
    # Calculate initial energy using Cantera
    Y0 = jnp.array(initial_Y())
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}
    gas_check.TPY = T0, 101325.0, comp
    e0 = gas_check.int_energy_mass
    
    # Create initial state vector
    y0 = jnp.concatenate([Y0, jnp.array([e0])])
    
    # Create JAX-compatible mechanism
    mech_jax = {k: v for k, v in mech.items() if k not in ["file", "names"]}
    
    # Prepare helper function for Cantera pressure calculation
    def pressure_from_rho_T_Y(rho_val, T_val, Y_val):
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = float(np.sum(Y_val / W_mol))  # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                    # kg/mol
        R_univ = 8.314462618                     # J/(mol*K)
        R_spec = R_univ / Wmix                   # J/(kg*K)
        return rho_val * R_spec * T_val          # Pa
    
    # Define vectorized temperature lookup function to be used after integration
    @jax.jit
    def batch_T_from_e_newton(ys_batch):
        """Vectorized temperature calculation from state vectors"""
        e_batch = ys_batch[:, -1]
        Y_batch = ys_batch[:, :-1]
        return jax.vmap(lambda e, Y: T_from_e_newton(
            e, Y, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"]
        ))(e_batch, Y_batch)
    
    # Standard JIT warmup if requested
    if warmup:
        print("\nPrecompiling JAX functions...")
        # Make a small grid for warmup (2x2)
        y0_cell = np.array(y0)
        y0_grid = np.tile(y0_cell, (2, 2, 1))
        y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (2*2, S+1))
        
        # Warm up the integrator (with short integration time)
        _ = integrate_grid(y0_flat, rho, mech_jax, t0, t1/100)
        
        # Warm up the temperature calculator
        _ = batch_T_from_e_newton(y0_flat)
    
    # Run benchmark for each size
    for n in sizes:
        print(f"\nRunning {n}×{n} grid...")
        
        # ---------- JAX grid integration ----------
        print(f"  Building {n}×{n} JAX grid...")
        y0_cell = np.array(y0)
        y0_grid = np.tile(y0_cell, (n, n, 1))
        y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (n*n, S+1))
        
        # Ensure arrays are copied to device and all previous operations complete
        jax.device_get(jnp.sum(y0_flat))
        
        # Time JAX integration
        print(f"  Running JAX chemistry for {n}×{n} grid...")
        t_jax0 = time.time()
        sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1)
        
        # Ensure computation is complete before stopping timer
        jax.device_get(jnp.sum(sol.ys[-1]))
        t_jax = time.time() - t_jax0
        jax_times.append(t_jax)
        
        # Extract results for validation
        ys_flat = np.array(sol.ys[-1])
        ys_jax = ys_flat.reshape(n, n, S+1)
        Y_jax = ys_jax[:, :, :-1]
        
        # Calculate temperatures using the pre-compiled vectorized function
        T_flat = np.array(batch_T_from_e_newton(sol.ys[-1]))
        T_jax = T_flat.reshape(n, n)
        
        # ---------- Cantera grid integration ----------
        print(f"  Building {n}×{n} Cantera grid...")
        reactors = []
        gases = []
        
        # Build n×n reactors
        for j in range(n):
            row_reactors = []
            row_gases = []
            for i in range(n):
                gas = ct.Solution(mech["file"])
                Y0 = np.array(y0_grid[j, i, :-1], dtype=float)
                P0 = pressure_from_rho_T_Y(rho, T0, Y0)
                comp = {names[k]: float(Y0[k]) for k in range(S)}
                gas.TPY = T0, P0, comp
                r = ct.IdealGasReactor(gas)
                row_gases.append(gas)
                row_reactors.append(r)
            gases.append(row_gases)
            reactors.append(row_reactors)
        
        # Build one network with all reactors
        all_reactors = [r for row in reactors for r in row]
        net = ct.ReactorNet(all_reactors)
        
        # Time Cantera integration
        print(f"  Running Cantera chemistry for {n}×{n} grid...")
        t_ct0 = time.time()
        net.advance(t1)
        t_ct = time.time() - t_ct0
        cantera_times.append(t_ct)
        
        # Collect Cantera results
        Y_ct = np.zeros((n, n, S), dtype=float)
        T_ct = np.zeros((n, n), dtype=float)
        for j in range(n):
            for i in range(n):
                g = gases[j][i]
                Y_ct[j, i, :] = g.Y
                T_ct[j, i] = g.T
        
        # Calculate mean temperature (for validation)
        mean_T_jax = np.mean(T_jax)
        mean_T_ct = np.mean(T_ct)
        
        # Report results for this size
        speedup = t_ct / t_jax
        print(f"  Results for {n}×{n} grid:")
        print(f"    JAX time:      {t_jax:.4f} s")
        print(f"    Cantera time:  {t_ct:.4f} s")
        print(f"    Speedup:       {speedup:.2f}x")
        print(f"    Mean JAX T:    {mean_T_jax:.2f} K")
        print(f"    Mean Cantera T: {mean_T_ct:.2f} K")
    
    # Convert to cell counts for plotting
    cell_counts = [n*n for n in sizes]
    speedups = [ct/jx if jx > 0 else 0 for ct, jx in zip(cantera_times, jax_times)]
    
    # Calculate scaling behavior
    jax_scaling = []
    cantera_scaling = []
    
    for i in range(1, len(sizes)):
        size_ratio = cell_counts[i] / cell_counts[i-1]
        time_ratio_jax = jax_times[i] / jax_times[i-1]
        time_ratio_cantera = cantera_times[i] / cantera_times[i-1]
        jax_scaling.append(time_ratio_jax / size_ratio)
        cantera_scaling.append(time_ratio_cantera / size_ratio)
    
    # Package results
    results = {
        'sizes': sizes,
        'cell_counts': cell_counts,
        'jax_times': jax_times,
        'cantera_times': cantera_times,
        'speedups': speedups,
        'jax_scaling': jax_scaling,
        'cantera_scaling': cantera_scaling
    }
    
    return results

def plot_results(results):
    """
    Create publication-quality plots from benchmark results
    
    Parameters:
    -----------
    results : dict
        Dictionary with benchmark results from benchmark_chemistry()
    """
    sizes = results['sizes']
    cell_counts = results['cell_counts']
    jax_times = results['jax_times']
    cantera_times = results['cantera_times']
    speedups = results['speedups']
    
    # Create detailed plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot: Execution time vs problem size
    ax1.plot(cell_counts, jax_times, 'o-', label='JAX', color='#1f77b4', linewidth=2)
    ax1.plot(cell_counts, cantera_times, 's-', label='Cantera', color='#ff7f0e', linewidth=2)
    ax1.set_xlabel('Number of Cells')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Chemistry Integration Time vs. Problem Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Use log-log scale for better visualization of scaling behavior
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    
    # Second subplot: Speedup vs problem size
    ax2.plot(cell_counts, speedups, 'd-', color='#2ca02c', linewidth=2)
    ax2.set_xlabel('Number of Cells')
    ax2.set_ylabel('Speedup Ratio (Cantera/JAX)')
    ax2.set_title('JAX Speedup vs. Problem Size')
    ax2.grid(True, alpha=0.3)
    
    # Add speedup annotations
    for i, (x, y) in enumerate(zip(cell_counts, speedups)):
        ax2.annotate(f"{y:.1f}x", (x, y), 
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set log scale for x-axis on speedup plot
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig('chemistry_benchmark.png', dpi=300)
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Grid Size':^10s} | {'Cells':^8s} | {'JAX Time (s)':^12s} | {'Cantera Time (s)':^16s} | {'Speedup':^10s}")
    print("-"*10 + "-+-" + "-"*8 + "-+-" + "-"*12 + "-+-" + "-"*16 + "-+-" + "-"*10)
    
    for i, n in enumerate(sizes):
        print(f"{n:^3d}×{n:<6d} | {n*n:^8d} | {jax_times[i]:^12.4f} | {cantera_times[i]:^16.4f} | {speedups[i]:^10.2f}x")
    
    # Print scaling information
    jax_scaling = results['jax_scaling']
    cantera_scaling = results['cantera_scaling']
    
    print("\nSCALING EFFICIENCY (ideal = 1.0):")
    print(f"JAX average scaling: {np.mean(jax_scaling):.2f}")
    print(f"Cantera average scaling: {np.mean(cantera_scaling):.2f}")
    
    print("\n" + "="*80)
    print("Performance analysis complete.")
    print("="*80)

if __name__ == "__main__":
    # Run benchmark with standard sizes
    results = benchmark_chemistry()
    
    # Plot and report results
    plot_results(results)
