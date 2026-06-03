import numpy as np
from pyrodeen.grid import Grid
from pyrodeen.cfd import Solver, BoundaryConfig
import matplotlib.pyplot as pyplot


def plot_euler_1d(x, r, u, p, E, fig = None):
    if fig is None:
        pyplot.ion()
        fig, axs = pyplot.subplots(2, 2)
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel(r"$\rho$")
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("u")
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("p")
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("E")

    axs = fig.axes
    lines = [
        axs[0].plot([], [], ".b")[0],
        axs[1].plot([], [], ".m")[0],
        axs[2].plot([], [], ".k")[0],
        axs[3].plot([], [], ".r")[0],
    ]
    lines[0].set_data(x, r)
    lines[1].set_data(x, u)
    lines[2].set_data(x, p)
    lines[3].set_data(x, E)

    # draw
    pyplot.draw()
    pyplot.pause(0.001)

    # Return whether the figure still exists (False if closed)
    return fig, pyplot.fignum_exists(fig.number)


def run_simulation(
    CFL=0.50, tEnd=0.05, nx=100, ny=100, lx=1.0, ly=1.0, nv=4, gamma=1.4, plot=False
):
    nx, ny = 100, 1
    lx, ly = 1.0, 0.1
    dx = lx / nx
    dy = ly / ny

    # Initialize grid and solver
    grid = Grid(nx, ny, lx, ly, nv)
    solver = Solver(grid, BoundaryConfig.closed_box(), gamma)

    # Set initial conditions
    solver.sod_tube_ic()
    solver.apply_boundary_conditions()
    # grid.values[:, :, 0] = 1.0  # Density
    # grid.values[:, :, 1] = 0.0  # x-momentum
    # grid.values[:, :, 2] = 0.0  # y-momentum
    # grid.values[:, :, 3] = 1.0 / (gamma - 1)  # Energy

    # Plot
    if plot:
        fig, open = plot_euler_1d(
            np.linspace(0, lx, nx),
            grid.values[:, 0, 0],
            grid.values[:, 0, 1],
            grid.values[:, 0, 2],
            grid.values[:, 0, 3],
        )

    # Time-stepping loop (simplified)
    t_end = 0.15
    a0 = np.sqrt(gamma * grid.values[:, :, 3] / grid.values[:, :, 0])  # Speed of sound
    dt = CFL * np.min([np.min(lx / a0), np.min(ly / a0)])  # Time step based on CFL condition

    r0 = grid.values[:, :, 0]  # Density
    p0 = grid.values[:, :, 3] * ((gamma - 1))  # Pressure
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(grid.values[:, :, 1]**2 + grid.values[:, :, 2]**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(dx, dy) / a0
    t = 0.0
    while t < t_end:
        print("Time: {:.4f}, dt: {:.4e}".format(t, dt))
        if not open:
            break
        # RK2 1st step
        res1 = solver.muscl_euler_res2d()
        values_1 = grid.values - dt * res1
        # Update grid and apply BCs
        grid.values = values_1
        solver.apply_boundary_conditions()
        # RK2 2nd step / update q
        res2 = solver.muscl_euler_res2d()
        grid.values = 0.5 * (grid.values + values_1 - dt * res2)
        # Natural BCs again
        solver.apply_boundary_conditions()
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
        dt = CFL * min(dx / a, dy / a)
        if plot:
            plot_euler_1d(
                np.linspace(0, lx, nx),
                grid.values[:, 0, 0],
                grid.values[:, 0, 1],
                grid.values[:, 0, 2],
                grid.values[:, 0, 3],
                fig,
            )
        if t + dt > tEnd:
            dt = tEnd - t
        t += dt


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
    run_simulation(plot=True)


if __name__ == "__main__":
    main()
