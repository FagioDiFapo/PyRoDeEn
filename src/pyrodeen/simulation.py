import numpy as np
from pyrodeen.grid import Grid
from pyrodeen.cfd import Solver, BoundaryConfig, EulerExact
from pyrodeen.solutions import EulerSolution1D
import matplotlib.pyplot as plt
from pyrodeen.draw import *


def run_simulation(
    CFL=0.50,
    t_end=0.15,
    nx=100,
    ny=100,
    lx=1.0,
    ly=1.0,
    nv=4,
    n=5,
    sod_shock_tube=False,
    plot_1d=False,
    video_file_1d=None,
):
    # From gas DOF
    gamma = (n + 2) / n

    # Initialize grid and solver
    grid = Grid(nx, ny, lx, ly, nv)
    solver = Solver(BoundaryConfig.shock_tube(), gamma)

    # Set initial conditions
    if sod_shock_tube:
        solver.sod_tube_ic(grid)
    solver.apply_boundary_conditions(grid)

    # Plot
    xe, re, ue, pe, ee, *_ = EulerExact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, t_end, n)
    Ee = ee + 0.5 * ue**2
    exact_solution = EulerSolution1D(xe, re, ue, pe, Ee)
    if plot_1d:
        plt.ion()
        open = plot_euler_1d(exact_solution)
        plt.ioff()
        plt.tight_layout()

    # Time-stepping loop (simplified)
    a0 = np.sqrt(gamma * grid.values[:, :, 3] / grid.values[:, :, 0])  # Speed of sound
    dt = CFL * np.min(
        [np.min(lx / a0), np.min(ly / a0)]
    )  # Time step based on CFL condition

    r0 = grid.values[:, :, 0]  # Density
    p0 = grid.values[:, :, 3] * ((gamma - 1))  # Pressure
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(grid.values[:, :, 1] ** 2 + grid.values[:, :, 2] ** 2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(grid.dx, grid.dy) / a0
    t = 0.0
    while t < t_end and open:
        # RK2
        solver.RK2_step(grid, dt)
        # Compute flow properties
        r = grid.values[:, :, 0]
        u = grid.values[:, :, 1] / r
        v = grid.values[:, :, 2] / r
        E = grid.values[:, :, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u**2 + v**2))
        c = np.sqrt(gamma * p / r)
        # Update dt and time
        vn = np.sqrt(u**2 + v**2)
        lambda1 = vn + c
        lambda2 = vn - c
        a = np.max(np.abs(np.concatenate([lambda1.flatten(), lambda2.flatten()])))
        dt = CFL * min(grid.dx / a, grid.dy / a)
        x = np.linspace(grid.dx / 2, lx - grid.dx / 2, nx)
        solution = EulerSolution1D(x, r[0,:], u[0,:], p[0,:], E[0,:])
        if plot_1d:
            open = plot_euler_1d(solution)
        if video_file_1d is not None:
            record_euler_1d_frame(solution)
        if t + dt > t_end:
            dt = t_end - t
        t += dt

    if video_file_1d is not None:
        save_1d_animation(exact_solution, video_file_1d)


def main():
    # params = {
    #    "CFL": 0.50,
    #    "tEnd": 0.05,
    #    "nx": 100,
    #    "ny": 100,
    #    "n": 5,
    #    "IC": 5,
    #    "fluxMth": 'HLLE1d',
    #    "method": 1,
    #    "limiter": 'MC',
    #    "plotFig": True,
    # }
    # run_simulation(**params)
    params = {
        "nx": 50,
        "ny": 1,
        "ly": 0.1,
        "sod_shock_tube": True,
        "plot_1d": True,
        "video_file_1d": "simulation.mp4",
    }
    run_simulation(**params)


if __name__ == "__main__":
    main()
