import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from scipy import interpolate

# Add the src directory to the path to find modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Adjust the import path as needed for your project structure:
from src.MUSCL_TVD_FagioDiFapo.euler_solvers import EulerExact
from src.rde_solver.cfd_grid import CFDGrid

def blast_ic_2d(x, y, p_high=1.0, p_low=0.1, r0=0.1, center_rel_x=0.5, center_rel_y=0.5):
    """
    2D blast wave: high pressure near a center specified as relative coordinates.
    center_rel_x, center_rel_y are in [0,1] and are multiplied by the domain extents
    derived from the supplied x,y arrays (so 0.5 -> midpoint).
    """
    # domain extents from x,y arrays (works for meshgrid)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    center_x = xmin + center_rel_x * (xmax - xmin)
    center_y = ymin + center_rel_y * (ymax - ymin)

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(r < r0, p_high, p_low)
    return rho, u, v, p

def blast_ic_2d_square(x, y, p_high=1.0, p_low=0.1, halfwidth=0.1, center_rel_x=0.5, center_rel_y=0.5):
    """
    2D blast wave: high pressure in a square around a center specified in relative coords.
    center_rel_x, center_rel_y are in [0,1] and mapped to domain coordinates.
    """
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    center_x = xmin + center_rel_x * (xmax - xmin)
    center_y = ymin + center_rel_y * (ymax - ymin)

    mask = (np.abs(x - center_x) < halfwidth) & (np.abs(y - center_y) < halfwidth)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(mask, p_high, p_low)
    return rho, u, v, p

def sod_ic_2d(x, y):
    # Sod's tube: left state for x < 0.5, right state for x >= 0.5
    # Use Sod's values: rho=1, u=0, p=1 (left); rho=0.125, u=0, p=0.1 (right)
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    return rho, u, v, p

def init_plots(xe, re, ue, pe, Ee):
    plt.ion()
    fig, axs = plt.subplots(2, 2, num=2)
    lines = [
        axs[0, 0].plot([], [], '.b')[0],
        axs[0, 1].plot([], [], '.m')[0],
        axs[1, 0].plot([], [], '.k')[0],
        axs[1, 1].plot([], [], '.r')[0]
    ]
    axs[0, 0].plot(xe, re, '-k', label='Exact')
    axs[0, 1].plot(xe, ue, '-k', label='Exact')
    axs[1, 0].plot(xe, pe, '-k', label='Exact')
    axs[1, 1].plot(xe, Ee, '-k', label='Exact')
    axs[0, 0].set_xlabel('x'); axs[0, 0].set_ylabel(r'$\rho$')
    axs[0, 1].set_xlabel('x'); axs[0, 1].set_ylabel('u')
    axs[1, 0].set_xlabel('x'); axs[1, 0].set_ylabel('p')
    axs[1, 1].set_xlabel('x'); axs[1, 1].set_ylabel('E')
    axs[0, 0].set_title('SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)')
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1.1])
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim([-0.1, 1.1])
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim([0, 1.1])
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim([1.5, 3.5])
    for axrow in axs:
        for ax in axrow:
            ax.legend()
    plt.tight_layout()
    return fig, axs, lines

def update_plots(lines, xc, r, u, p, E):
    lines[0].set_data(xc, r)
    lines[1].set_data(xc, u)
    lines[2].set_data(xc, p)
    lines[3].set_data(xc, E)
    plt.draw()
    plt.pause(0.001)

def calculate_comparison_stats(python_data, matlab_data, variable_name):
    """
    Calculate statistical comparison between Python and MATLAB results.

    Parameters:
    -----------
    python_data : array
        Python simulation results for a specific variable
    matlab_data : array
        MATLAB simulation results for the same variable
    variable_name : str
        Name of the variable being compared

    Returns:
    --------
    dict: Dictionary with absolute and relative differences
    """
    # Interpolate to same grid if necessary (MATLAB might have different resolution)
    if len(python_data) != len(matlab_data):
        # Using simple nearest interpolation for now
        from scipy.interpolate import interp1d
        x_python = np.linspace(0, 1, len(python_data))
        x_matlab = np.linspace(0, 1, len(matlab_data))
        matlab_interp = interp1d(x_matlab, matlab_data, bounds_error=False, fill_value="extrapolate")
        matlab_data_interp = matlab_interp(x_python)
    else:
        matlab_data_interp = matlab_data

    # Calculate absolute differences
    abs_diff = np.abs(python_data - matlab_data_interp)

    # Calculate relative differences (percentage), avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / np.abs(matlab_data_interp) * 100
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)

    # Find locations of maximum differences
    max_abs_diff_idx = np.argmax(abs_diff)
    max_rel_diff_idx = np.argmax(rel_diff)

    # Calculate statistics
    avg_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    avg_rel_diff = np.mean(rel_diff)
    max_rel_diff = np.max(rel_diff)

    # Get values at max difference points for analysis
    value1_at_max_abs = python_data[max_abs_diff_idx]
    value2_at_max_abs = matlab_data_interp[max_abs_diff_idx]
    value1_at_max_rel = python_data[max_rel_diff_idx]
    value2_at_max_rel = matlab_data_interp[max_rel_diff_idx]

    return {
        "variable": variable_name,
        "avg_diff": avg_diff,
        "max_diff": max_diff,
        "avg_rel_diff": avg_rel_diff,
        "max_rel_diff": max_rel_diff,
        "max_abs_diff_idx": max_abs_diff_idx,
        "max_rel_diff_idx": max_rel_diff_idx,
        "value1_at_max_abs": value1_at_max_abs,
        "value2_at_max_abs": value2_at_max_abs,
        "value1_at_max_rel": value1_at_max_rel,
        "value2_at_max_rel": value2_at_max_rel
    }

def print_comparison_table(stats_list):
    """
    Print a comparison table in Excel-pasteable format.

    Parameters:
    -----------
    stats_list : list
        List of dictionaries with comparison statistics
    """
    # Print header line that Excel can directly parse
    headers = ["Variable", "Avg Abs Delta", "Max Abs Delta", "Avg Rel Delta (%)", "Max Rel Delta (%)"]
    print("\t".join(headers))

    # Print data rows
    for stats in stats_list:
        row = [
            stats['variable'],
            f"{stats['avg_diff']:.6f}",
            f"{stats['max_diff']:.6f}",
            f"{stats['avg_rel_diff']:.2f}",
            f"{stats['max_rel_diff']:.2f}"
        ]
        print("\t".join(row))

    print("================================================")

    # Print additional information about maximum differences
    print("\nDetails on maximum differences:")
    for stats in stats_list:
        if 'value1_at_max_rel' in stats:
            var = stats['variable']
            print(f"\n{var}:")
            # For maximum absolute difference
            print(f"  Max abs diff: {stats['max_diff']:.6f} at idx {stats['max_abs_diff_idx']}")
            print(f"  Values at max abs diff: {stats['value1_at_max_abs']:.6f} vs {stats['value2_at_max_abs']:.6f}")

            # For maximum relative difference
            print(f"  Max rel diff: {stats['max_rel_diff']:.2f}% at idx {stats['max_rel_diff_idx']}")
            print(f"  Values at max rel diff: {stats['value1_at_max_rel']:.6f} vs {stats['value2_at_max_rel']:.6f}")

            # Calculate why the relative difference is so high
            if abs(stats['value2_at_max_rel']) < 0.01:
                print(f"  NOTE: Max rel diff is high because reference value is near zero ({stats['value2_at_max_rel']:.8f})")

    print("================================================")

def final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee, matlab_results=None, save_path=None):
    """
    Create final plots comparing numerical solutions with the exact solution.

    Parameters:
    -----------
    xc, r, u, p, E : arrays
        Python numerical solution arrays
    xe, re, ue, pe, Ee : arrays
        Exact solution arrays
    matlab_results : dict or None
        Optional MATLAB results for comparison
    save_path : str or None
        Path to save the output plots as PNG
    """
    # Set font sizes for better readability in a poster
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    if matlab_results is not None:
        # Extract MATLAB data
        xm = matlab_results['x']
        rm = matlab_results['density']
        um = matlab_results['velocity']
        pm = matlab_results['pressure']
        Em = matlab_results['energy']

        # Create a new 2x2 figure with improved layout for better comparison
        # Smaller figure size that scales better for posters
        plt.figure(figsize=(10, 8))

        # Color scheme: blue for Python, orange for MATLAB, black for exact solution
        PYTHON_COLOR = '#1f77b4'  # Blue
        MATLAB_COLOR = '#ff7f0e'  # Orange
        EXACT_COLOR = 'black'

        # 1. Top-left: Density comparison (Python vs Exact)
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(xc, r, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, re, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel(r'$\rho$ (density)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Python vs Exact: Density')
        plt.grid(alpha=0.3)

        # 2. Top-right: Density comparison (MATLAB vs Exact)
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(xm, rm, 'o', color=MATLAB_COLOR, markersize=5, markevery=5, label='MATLAB')
        plt.plot(xe, re, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel(r'$\rho$ (density)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('MATLAB vs Exact: Density')
        plt.grid(alpha=0.3)

        # 3. Bottom-left: Energy comparison (Python vs Exact)
        ax3 = plt.subplot(2, 2, 3)
        plt.plot(xc, E, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, Ee, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('E (energy)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Python vs Exact: Energy')
        plt.grid(alpha=0.3)

        # 4. Bottom-right: Energy comparison (MATLAB vs Exact)
        ax4 = plt.subplot(2, 2, 4)
        plt.plot(xm, Em, 'o', color=MATLAB_COLOR, markersize=5, markevery=5, label='MATLAB')
        plt.plot(xe, Ee, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('E (energy)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('MATLAB vs Exact: Energy')
        plt.grid(alpha=0.3)

        # Add a main title with slightly smaller font to fit better
        plt.suptitle('SSP-RK2 TVD-MUSCL Euler Equations (2D Sod Tube)', fontsize=16)

        # Match the axes limits for direct comparison
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ymin = min(ylim1[0], ylim2[0])
        ymax = max(ylim1[1], ylim2[1])
        ax1.set_ylim([ymin, ymax])
        ax2.set_ylim([ymin, ymax])

        ylim3 = ax3.get_ylim()
        ylim4 = ax4.get_ylim()
        ymin = min(ylim3[0], ylim4[0])
        ymax = max(ylim3[1], ylim4[1])
        ax3.set_ylim([ymin, ymax])
        ax4.set_ylim([ymin, ymax])

        # Create a second figure with velocity and pressure
        plt.figure(figsize=(10, 8))

        # Define colors
        PYTHON_COLOR = '#1f77b4'  # Blue
        MATLAB_COLOR = '#ff7f0e'  # Orange
        EXACT_COLOR = 'black'

        # 1. Top-left: Velocity comparison (Python vs Exact)
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(xc, u, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, ue, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('u (velocity)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Python vs Exact: Velocity')
        plt.grid(alpha=0.3)

        # 2. Top-right: Velocity comparison (MATLAB vs Exact)
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(xm, um, 'o', color=MATLAB_COLOR, markersize=5, markevery=5, label='MATLAB')
        plt.plot(xe, ue, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('u (velocity)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('MATLAB vs Exact: Velocity')
        plt.grid(alpha=0.3)

        # 3. Bottom-left: Pressure comparison (Python vs Exact)
        ax3 = plt.subplot(2, 2, 3)
        plt.plot(xc, p, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, pe, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('p (pressure)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Python vs Exact: Pressure')
        plt.grid(alpha=0.3)

        # 4. Bottom-right: Pressure comparison (MATLAB vs Exact)
        ax4 = plt.subplot(2, 2, 4)
        plt.plot(xm, pm, 'o', color=MATLAB_COLOR, markersize=5, markevery=5, label='MATLAB')
        plt.plot(xe, pe, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('p (pressure)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('MATLAB vs Exact: Pressure')
        plt.grid(alpha=0.3)

        # Add a main title with bold font
        plt.suptitle('SSP-RK2 TVD-MUSCL Euler Equations (2D Sod Tube)', fontsize=16, fontweight='bold')

        # Match the axes limits for direct comparison
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ymin = min(ylim1[0], ylim2[0])
        ymax = max(ylim1[1], ylim2[1])
        ax1.set_ylim([ymin, ymax])
        ax2.set_ylim([ymin, ymax])

        ylim3 = ax3.get_ylim()
        ylim4 = ax4.get_ylim()
        ymin = min(ylim3[0], ylim4[0])
        ymax = max(ylim3[1], ylim4[1])
        ax3.set_ylim([ymin, ymax])
        ax4.set_ylim([ymin, ymax])

    else:
        # Original plots without MATLAB comparison - using same styling for consistency
        # Define colors
        PYTHON_COLOR = '#1f77b4'  # Blue
        EXACT_COLOR = 'black'

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.plot(xc, r, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, re, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel(r'$\rho$ (density)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Density')
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(xc, u, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, ue, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('u (velocity)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Velocity')
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(xc, p, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, pe, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('p (pressure)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Pressure')
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(xc, E, 'o', color=PYTHON_COLOR, markersize=5, markevery=5, label='Python')
        plt.plot(xe, Ee, '-', color=EXACT_COLOR, linewidth=2, label='Exact')
        plt.xlabel('x')
        plt.ylabel('E (energy)')
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.title('Energy')
        plt.grid(alpha=0.3)

        plt.suptitle('SSP-RK2 TVD-MUSCL Euler Equations (2D Sod Tube)', fontsize=16, fontweight='bold')    # Make layout tight before saving/showing
    plt.tight_layout()

    # Calculate and print comparison statistics vs exact solution
    # Interpolate exact solution to match Python's grid points
    from scipy.interpolate import interp1d
    xe_array = np.array(xe)
    re_array = np.array(re)
    ue_array = np.array(ue)
    pe_array = np.array(pe)
    Ee_array = np.array(Ee)

    # Create interpolation functions
    re_interp = interp1d(xe_array, re_array, bounds_error=False, fill_value="extrapolate")
    ue_interp = interp1d(xe_array, ue_array, bounds_error=False, fill_value="extrapolate")
    pe_interp = interp1d(xe_array, pe_array, bounds_error=False, fill_value="extrapolate")
    Ee_interp = interp1d(xe_array, Ee_array, bounds_error=False, fill_value="extrapolate")

    # Interpolate exact solution onto python grid points
    re_on_xc = re_interp(xc)
    ue_on_xc = ue_interp(xc)
    pe_on_xc = pe_interp(xc)
    Ee_on_xc = Ee_interp(xc)

    # Calculate python vs exact solution statistics
    python_exact_stats = [
        calculate_comparison_stats(r, re_on_xc, "Density"),
        calculate_comparison_stats(u, ue_on_xc, "Velocity"),
        calculate_comparison_stats(p, pe_on_xc, "Pressure"),
        calculate_comparison_stats(E, Ee_on_xc, "Energy")
    ]

    # Print comparison table Python vs Exact
    print("\n======== PYTHON vs EXACT SOLUTION Comparison ========")
    print_comparison_table(python_exact_stats)

    # Generate and print comparison statistics if MATLAB results are available
    if matlab_results is not None:
        # Extract MATLAB data
        rm = matlab_results['density']
        um = matlab_results['velocity']
        pm = matlab_results['pressure']
        Em = matlab_results['energy']

        # Calculate comparison statistics between Python and MATLAB
        matlab_python_stats = [
            calculate_comparison_stats(r, rm, "Density"),
            calculate_comparison_stats(u, um, "Velocity"),
            calculate_comparison_stats(p, pm, "Pressure"),
            calculate_comparison_stats(E, Em, "Energy")
        ]

        # Print comparison table between MATLAB and Python
        print("\n======== MATLAB vs PYTHON Comparison ========")
        print_comparison_table(matlab_python_stats)

        # Calculate MATLAB vs Exact statistics
        matlab_x = matlab_results['x']
        re_on_matlab = re_interp(matlab_x)
        ue_on_matlab = ue_interp(matlab_x)
        pe_on_matlab = pe_interp(matlab_x)
        Ee_on_matlab = Ee_interp(matlab_x)

        matlab_exact_stats = [
            calculate_comparison_stats(rm, re_on_matlab, "Density"),
            calculate_comparison_stats(um, ue_on_matlab, "Velocity"),
            calculate_comparison_stats(pm, pe_on_matlab, "Pressure"),
            calculate_comparison_stats(Em, Ee_on_matlab, "Energy")
        ]

        # Print comparison table MATLAB vs Exact
        print("\n======== MATLAB vs EXACT SOLUTION Comparison ========")
        print_comparison_table(matlab_exact_stats)

        # Also save all stats to CSV if a save path is provided
        if save_path:
            import csv

            # Save Python vs MATLAB comparison
            comparison_csv = f"{save_path}_matlab_python_comparison.csv"
            with open(comparison_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Variable", "Average Absolute Delta", "Maximum Absolute Delta",
                                "Average Relative Delta (%)", "Maximum Relative Delta (%)"])
                for stat in matlab_python_stats:
                    writer.writerow([
                        stat['variable'],
                        stat['avg_diff'],
                        stat['max_diff'],
                        stat['avg_rel_diff'],
                        stat['max_rel_diff']
                    ])
            print(f"Saved MATLAB vs Python comparison statistics to {comparison_csv}")

            # Save Python vs Exact comparison
            python_exact_csv = f"{save_path}_python_exact_comparison.csv"
            with open(python_exact_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Variable", "Average Absolute Delta", "Maximum Absolute Delta",
                                "Average Relative Delta (%)", "Maximum Relative Delta (%)"])
                for stat in python_exact_stats:
                    writer.writerow([
                        stat['variable'],
                        stat['avg_diff'],
                        stat['max_diff'],
                        stat['avg_rel_diff'],
                        stat['max_rel_diff']
                    ])
            print(f"Saved Python vs Exact comparison statistics to {python_exact_csv}")

            # Save MATLAB vs Exact comparison
            matlab_exact_csv = f"{save_path}_matlab_exact_comparison.csv"
            with open(matlab_exact_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Variable", "Average Absolute Delta", "Maximum Absolute Delta",
                                "Average Relative Delta (%)", "Maximum Relative Delta (%)"])
                for stat in matlab_exact_stats:
                    writer.writerow([
                        stat['variable'],
                        stat['avg_diff'],
                        stat['max_diff'],
                        stat['avg_rel_diff'],
                        stat['max_rel_diff']
                    ])
            print(f"Saved MATLAB vs Exact comparison statistics to {matlab_exact_csv}")

    # Save figures if a path is provided
    if save_path:
        import os
        if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path))

        # Save only the velocity_pressure figure if MATLAB comparison is enabled
        if matlab_results is not None:
            # Only save the second figure (velocity_pressure)
            plt.figure(2).savefig(f"{save_path}_velocity_pressure.png", dpi=400, bbox_inches='tight',
                              facecolor='white', edgecolor='none')
            print(f"Saved comparison plot to {save_path}_velocity_pressure.png")

            # Also save a combined version for the poster (just density and energy)
            # Create a new figure specifically designed for poster presentation
            plt.figure(figsize=(10, 8))  # Taller figure to accommodate better spacing

            # Create subplots with specific spacing for better readability
            plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase vertical and horizontal spacing

            # Python density plot
            ax1 = plt.subplot(2, 2, 1)
            plt.plot(xc, r, 'o', color='#1f77b4', markersize=4, markevery=5, label='Python')
            plt.plot(xe, re, '-k', linewidth=2, label='Exact')
            plt.xlabel('x', labelpad=7)  # Increase padding between axis and label
            plt.ylabel(r'$\rho$ (density)')
            plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
            plt.grid(alpha=0.3)
            plt.title('Python vs Exact: Density', pad=10)  # Add padding between title and plot

            # MATLAB density plot
            ax2 = plt.subplot(2, 2, 2)
            plt.plot(xm, rm, 'o', color='#ff7f0e', markersize=4, markevery=5, label='MATLAB')
            plt.plot(xe, re, '-k', linewidth=2, label='Exact')
            plt.xlabel('x', labelpad=7)
            plt.ylabel(r'$\rho$ (density)')
            plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
            plt.grid(alpha=0.3)
            plt.title('MATLAB vs Exact: Density', pad=10)

            # Python energy plot
            ax3 = plt.subplot(2, 2, 3)
            plt.plot(xc, E, 'o', color='#1f77b4', markersize=4, markevery=5, label='Python')
            plt.plot(xe, Ee, '-k', linewidth=2, label='Exact')
            plt.xlabel('x', labelpad=7)
            plt.ylabel('E (energy)')
            plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
            plt.grid(alpha=0.3)
            plt.title('Python vs Exact: Energy', pad=10)

            # MATLAB energy plot
            ax4 = plt.subplot(2, 2, 4)
            plt.plot(xm, Em, 'o', color='#ff7f0e', markersize=4, markevery=5, label='MATLAB')
            plt.plot(xe, Ee, '-k', linewidth=2, label='Exact')
            plt.xlabel('x', labelpad=7)
            plt.ylabel('E (energy)')
            plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
            plt.grid(alpha=0.3)
            plt.title('MATLAB vs Exact: Energy', pad=10)

            # Add a main title with some extra space at the top
            plt.suptitle('SSP-RK2 TVD-MUSCL Euler Equations (2D Sod Tube)',
                        fontsize=16, y=0.98, fontweight='bold')

            # Save with high resolution
            plt.savefig(f"{save_path}_poster_format.png", dpi=400, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.25)  # Add extra padding
            print(f"Saved poster-optimized comparison to {save_path}_poster_format.png")
        else:
            # Save the single figure with higher DPI
            plt.savefig(f"{save_path}.png", dpi=400, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved plot to {save_path}.png")

    plt.show()

def run_2d_sod_with_grid(compare_matlab=True, save_plots=False):
    """
    Run 2D Sod shock tube simulation with grid based implementation.

    Parameters:
    -----------
    compare_matlab : bool, default=True
        Whether to include MATLAB results comparison in the final plot.
    save_plots : bool or str, default=False
        If True, saves plots to default location. If a string, saves to that path.
    """
    import os
    nx, ny = 200, 1
    Lx, Ly = 1.0, 0.1
    dx, dy = Lx / nx, Ly / ny
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions
    r0, u0, v0, p0 = sod_ic_2d(x, y)
    n = 5
    gamma = (n + 2) / n
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    CFL = 0.5
    dt = CFL * min(dx, dy) / a0
    tEnd = 0.15

    # Central slice for plotting
    mid_row = ny // 2
    x_slice = xc

    # Compute final exact solution once
    xe, re, ue, pe, ee, *_ = EulerExact(
        1.0, 0.0, 1.0, 0.125, 0.0, 0.1, tEnd, n
    )
    Ee = ee + 0.5 * ue**2

    # Live plot setup (CFD only, exact is static reference)
    plt.ion()
    fig, axs, lines = init_plots(xe, re, ue, pe, Ee)

    # Start timing
    start_time = time.time()

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract central slice
        r = q[mid_row+1, 1:-1, 0]
        u = q[mid_row+1, 1:-1, 1] / r
        E = (q[mid_row+1, 1:-1, 3] / r)
        p = (gamma - 1) * r * (E - 0.5 * u ** 2)

        # Update CFD plots every 10 steps for performance
        if it % 10 == 0:
            update_plots(lines, x_slice, r, u, p, E)
        t += dt
        it += 1

    # End timing
    execution_time = time.time() - start_time
    print(f"Python simulation completed in {execution_time:.6f} seconds")
    print(f"Final time: {t:.3f}, Iterations: {it}")

    plt.ioff()
    plt.show()

    # Load MATLAB results if compare_matlab is True and results file exists
    matlab_results = None
    if compare_matlab:
        matlab_file = os.path.join(os.path.dirname(__file__), 'MUSLC_TVD_Genuine2D', 'matlab_sod_results.txt')
        if os.path.exists(matlab_file):
            try:
                print(f"Loading MATLAB comparison data from {matlab_file}")
                # Skip the header line and load the data
                data = np.loadtxt(matlab_file, delimiter=',', skiprows=2)
                matlab_results = {
                    'x': data[:, 0],
                    'density': data[:, 1],
                    'velocity': data[:, 2],
                    'pressure': data[:, 3],
                    'energy': data[:, 4],
                }
                print("MATLAB comparison data loaded successfully")
            except Exception as e:
                print(f"Error loading MATLAB results: {e}")
                matlab_results = None
        else:
            print(f"MATLAB results file not found at: {matlab_file}")

    # Configure save path for plots if requested
    plot_save_path = None
    if save_plots:
        if isinstance(save_plots, str):
            # Use the provided path
            plot_save_path = save_plots
        else:
            # Use default path
            plot_save_path = os.path.join(os.path.dirname(__file__), 'sod_tube_comparison')

    # Final plot with exact solution and MATLAB results if available
    final_plots(x_slice, r, u, p, E, xe, re, ue, pe, Ee, matlab_results, plot_save_path)

    # Save Python results for later comparison
    result_data = np.column_stack((x_slice, r, u, p, E))
    python_file = os.path.join(os.path.dirname(__file__), 'python_sod_results.txt')
    header = f"Python Sod Shock Tube Results (t = {t:.3f})\nx,density,velocity,pressure,energy"
    np.savetxt(python_file, result_data, delimiter=',', header=header, comments='')
    print(f"Python results saved to {python_file}")

    return execution_time, it
def run_blast_test(tEnd=0.3, nx=100, ny=100, plot=True):
    start = time.time()
    # Parameters
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    CFL = 0.5

    # Grid
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions: blast wave
    r0, u0, v0, p0 = blast_ic_2d(x, y, p_high=1.0, p_low=0.1, r0=0.1)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(dx, dy) / a0

    # Plot setup
    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(p0, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Pressure')
        ax.set_title('2D Blast Wave: Pressure')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)  # Give the GUI event loop time to process

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v0(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v0(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract pressure field (interior)
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

        # Update plot every x steps
        if plot and it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    if plot:
        plt.show()
        plt.close(fig)
        plt.close('all')
    end = time.time()
    print(f"Non-vectorized simulation time: {end - start:.3f} seconds")
    return end - start

def run_blast_test_vectorized(tEnd=0.3, nx=100, ny=100, plot=True):
    start = time.time()
    # Parameters
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    CFL = 0.5

    # Grid
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions: blast wave
    r0, u0, v0, p0 = blast_ic_2d(x, y, p_high=4.0, p_low=0.1, r0=0.1)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(dx, dy) / a0

    # Plot setup
    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(p0, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Pressure')
        ax.set_title('2D Blast Wave: Pressure')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)  # Give the GUI event loop time to process

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract pressure field (interior)
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

        # Update plot every x steps
        if plot and it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    if plot:
        plt.show()
        plt.close(fig)
        plt.close('all')
    end = time.time()
    print(f"Vectorized simulation time: {end - start:.3f} seconds")
    return end - start

def run_tube_test_vectorized(tEnd=0.3, nx=100, ny=100, plot=True):
    start = time.time()
    # Parameters
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    CFL = 0.5

    # Grid
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions: blast wave
    r0, u0, v0, p0 = blast_ic_2d(x, y, p_high=4.0, p_low=0.1, r0=0.2)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(dx, dy) / a0

    # Plot setup
    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(p0, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Pressure')
        ax.set_title('2D Blast Wave: Pressure')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)  # Give the GUI event loop time to process

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res

        # --- IMPORTANT: write provisional state into grid.q BEFORE applying BCs ---
        grid.q = qs.copy()

        # apply a single left reflecting wall, keep other sides transmissive
        grid.apply_reflecting_bc(side='left', ng=1)
        grid.q[:, -1, :] = grid.q[:, -2, :]    # right transmissive
        grid.apply_reflecting_bc(side='bottom', ng=1)
        grid.apply_reflecting_bc(side='top', ng=1)

        # RK2 step 2: compute residuals from the provisional state with correct ghosts
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q_new = 0.5 * (q + grid.q - dt * res2)

        # write final state into grid.q, then enforce BCs again
        grid.q = q_new.copy()
        grid.apply_reflecting_bc(side='left', ng=1)
        grid.q[:, -1, :] = grid.q[:, -2, :]    # right transmissive
        grid.apply_reflecting_bc(side='bottom', ng=1)
        grid.apply_reflecting_bc(side='top', ng=1)

        # update working state
        q = grid.q

        # Extract pressure field (interior)
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

        # Update plot every x steps
        if plot and it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    if plot:
        plt.show()
        plt.close(fig)
        plt.close('all')
    end = time.time()
    print(f"Vectorized simulation time: {end - start:.3f} seconds")
    return end - start

def benchmark_vectorization():
    #sizes = [10, 20, 30, 40, 50, 60]  # You can adjust or extend this list
    sizes = [20, 40, 60, 80, 90, 100, 110, 120, 130, 140]
    t_vec = []
    t_nonvec = []

    for n in sizes:
        print(f"Running for grid size {n}x{n}...")
        t_v = run_blast_test_vectorized(0.3, n, n, False)
        t_vec.append(t_v)
        t_nv = run_blast_test(0.3, n, n, False)
        t_nonvec.append(t_nv)

    # Load MATLAB benchmark data from file
    matlab_sizes = []
    matlab_times = []

    # Try multiple possible paths for the MATLAB results file
    possible_paths = [
        'MUSLC_TVD_Genuine2D/matlab_benchmark_results.txt',
        'matlab_benchmark_results.txt',
        'MUSLC_TVD_Genuine2D\\matlab_benchmark_results.txt',
    ]

    file_found = False
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines[2:]:  # Skip header lines
                    if line.strip():  # Check if line is not empty
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            size, time = parts
                            matlab_sizes.append(int(size))
                            matlab_times.append(float(time))
            file_found = True
            print(f"Loaded MATLAB benchmark data from {path}")
            break  # Exit loop once file is found
        except (FileNotFoundError, IOError):
            continue  # Try next path

    if not file_found:
        print(f"Warning: Could not load MATLAB benchmark results file")
        # If file not found, use empty lists
        matlab_sizes = sizes
        matlab_times = [0] * len(sizes)

    # Compute speedups vs vectorized Python
    speedup_sizes = [nv / v if v > 0 else np.nan for nv, v in zip(t_nonvec, t_vec)]

    # Create plot with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: execution time vs grid size
    axs[0].plot(sizes, t_vec, 'o-', label='Python Vectorized')
    axs[0].plot(sizes, t_nonvec, 's-', label='Python Non-vectorized')
    axs[0].plot(matlab_sizes, matlab_times, '^-', label='MATLAB')
    axs[0].set_xlabel('Grid size (N x N)')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title('Execution Time vs Grid Size')
    axs[0].grid(True)
    axs[0].legend()

    # Right plot: speedup vs grid size (compared to vectorized)
    axs[1].plot(sizes, speedup_sizes, 'd-', color='tab:green', label='Python Non-vec / Python Vec')

    # Add annotations for Python speedup
    for x, y in zip(sizes, speedup_sizes):
        axs[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,6),
                        ha='center', fontsize=8, color='tab:green')

    # Only compute MATLAB speedup if we have data
    if len(matlab_times) == len(t_vec) and not all(x == 0 for x in matlab_times):
        # Calculate speedup of MATLAB vs Python vectorized
        matlab_speedup = [m / v if v > 0 else np.nan for m, v in zip(matlab_times, t_vec)]
        axs[1].plot(sizes, matlab_speedup, 'x-', color='tab:red', label='MATLAB / Python Vec')

        # Add annotations for MATLAB speedup
        for x, y in zip(sizes, matlab_speedup):
            axs[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,-15),
                            ha='center', fontsize=8, color='tab:red')

    axs[1].set_xlabel('Grid size (N x N)')
    axs[1].set_ylabel('Speedup Ratio')
    axs[1].set_title('Speedup vs Grid Size')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300)
    plt.show()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # to run as module
    # python -m src.rde_solver.cfd_grid

    # Run the 2D Sod shock tube test with MATLAB comparison and save plots
    run_2d_sod_with_grid(compare_matlab=True, save_plots=True)

    #t_vec = run_blast_test_vectorized(2,50,50)
    #t_nonvec = run_blast_test()
    #if t_vec > 0:
    #    print(f"Speedup: {t_nonvec / t_vec:.2f}x faster (vectorized vs non-vectorized)")
    #benchmark_vectorization()
    #run_tube_test_vectorized(1.0, 100, 100, True)