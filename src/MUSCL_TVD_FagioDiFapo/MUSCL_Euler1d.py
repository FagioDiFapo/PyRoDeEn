import numpy as np
import matplotlib.pyplot as plt
import time
from .euler_solvers import Euler_IC1d, EulerExact, MUSCL_EulerRes1d

def init_plots(xe, re, ue, pe, Ee):
    """Initialize interactive plots for the simulation."""
    plt.ion()
    fig, axs = plt.subplots(2, 2, num=2)
    lines = [
        axs[0, 0].plot([], [], '.b')[0],
        axs[0, 1].plot([], [], '.m')[0],
        axs[1, 0].plot([], [], '.k')[0],
        axs[1, 1].plot([], [], '.r')[0]
    ]
    axs[0, 0].plot(xe, re, '-k')
    axs[0, 1].plot(xe, ue, '-k')
    axs[1, 0].plot(xe, pe, '-k')
    axs[1, 1].plot(xe, Ee, '-k')
    axs[0, 0].set_xlabel('x'); axs[0, 0].set_ylabel(r'$\rho$')
    axs[0, 1].set_xlabel('x'); axs[0, 1].set_ylabel('u')
    axs[1, 0].set_xlabel('x'); axs[1, 0].set_ylabel('p')
    axs[1, 1].set_xlabel('x'); axs[1, 1].set_ylabel('E')
    axs[0, 0].set_title('SSP-RK2 TVD-MUSCL Euler Eqns.')
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1.1])
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim([-0.1, 1.1])
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim([0, 1.1])
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim([1.5, 3.5])
    return fig, axs, lines

def update_plots(lines, xc, r, u, p, E):
    """Update plot data during the simulation."""
    lines[0].set_data(xc, r)
    lines[1].set_data(xc, u)
    lines[2].set_data(xc, p)
    lines[3].set_data(xc, E)
    plt.draw()
    plt.pause(0.001)

def final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee, fluxMth):
    """Plot the final results."""
    plt.figure(2)
    plt.subplot(2, 2, 1); plt.plot(xc, r, 'ro', xe, re, '-k'); plt.xlabel('x'); plt.ylabel(r'$\rho$'); plt.legend(['MUSCL-' + fluxMth, 'Exact'])
    plt.title('SSP-RK2 TVD-MUSCL Euler Eqns.')
    plt.subplot(2, 2, 2); plt.plot(xc, u, 'ro', xe, ue, '-k'); plt.xlabel('x'); plt.ylabel('u')
    plt.subplot(2, 2, 3); plt.plot(xc, p, 'ro', xe, pe, '-k'); plt.xlabel('x'); plt.ylabel('p')
    plt.subplot(2, 2, 4); plt.plot(xc, E, 'ro', xe, Ee, '-k'); plt.xlabel('x'); plt.ylabel('E')
    plt.show()

def run_muscl_solver(cfl=0.5, tEnd=0.15, nx=200, n=5, IC=1, limiter='VA', fluxMth='HLLC', plot_fig=1):
    """Run the MUSCL solver for the 1D Euler equations."""
    gamma = (n + 2) / n
    Lx = 1
    dx = Lx / nx
    xc = [dx / 2] + [(i + 1) * dx for i in range(nx)]
    [r0, u0, p0, _, _] = Euler_IC1d(xc, IC)
    E0 = [i[0] / ((gamma - 1) * i[1]) + 0.5 * i[2] ** 2 for i in zip(p0, r0, u0)]
    a0 = [np.sqrt(gamma * i[0] / i[1]) for i in zip(p0, r0)]
    [xe, re, ue, pe, ee, te, Me, se] = EulerExact(
        r0[0], u0[0], p0[0], r0[nx - 1], u0[nx - 1], p0[nx - 1], tEnd, n
    )
    Ee = pe / ((gamma - 1) * re) + 0.5 * ue ** 2

    nx = nx + 2
    r0 = np.array(r0)
    u0 = np.array(u0)
    E0 = np.array(E0)
    q0 = np.vstack([r0, r0 * u0, r0 * E0])
    zero = np.zeros((3, 1))
    q0 = np.hstack([zero, q0, zero])
    q0[:, 0] = q0[:, 1]
    q0[:, -1] = q0[:, -2]
    lambda0 = np.abs(u0) + a0
    dt0 = cfl * dx / np.max(lambda0)
    q = q0
    t = 0
    it = 0
    dt = dt0
    lambda_ = lambda0

    if plot_fig == 1:
        fig, axs, lines = init_plots(xe, re, ue, pe, Ee)

    start_time = time.time()
    while t < tEnd:
        qs = q - dt * MUSCL_EulerRes1d(q, max(lambda_), gamma, dx, q.shape[1], limiter, fluxMth)
        qs[:, 0] = qs[:, 1]
        qs[:, -1] = qs[:, -2]
        q = (q + qs - dt * MUSCL_EulerRes1d(qs, max(lambda_), gamma, dx, q.shape[1], limiter, fluxMth)) / 2
        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]
        r = q[0, :]
        u = q[1, :] / r
        E = q[2, :] / r
        p = (gamma - 1) * r * (E - 0.5 * u ** 2)
        a = np.sqrt(gamma * p / r)
        lambda_ = np.abs(u) + a
        dt = cfl * dx / np.max(lambda_)
        if t + dt > tEnd:
            dt = tEnd - t
        t = t + dt
        it = it + 1
        if it % 10 == 0 and plot_fig == 1:
            update_plots(lines, xc, r[1:-1], u[1:-1], p[1:-1], E[1:-1])
    if plot_fig == 1:
        plt.ioff()
        plt.show()
    cputime = time.time() - start_time

    q = q[:, 1:-1]
    nx = nx - 2
    # xc already has the correct length for plotting
    r = q[0, :]
    u = q[1, :] / r
    E = q[2, :] / r
    p = (gamma - 1) * r * (E - 0.5 * u ** 2)

    print(f'Execution time = {cputime:.2f} seconds')
    final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee, fluxMth)

def main():
    """
    Entry point for the MUSCL Euler 1D solver.

    MUSCL based numerical schemes extend the idea of using a linear
    piecewise approximation to each cell by using slope limited left and
    right extrapolated states. This results in the following high
    resolution, TVD discretisation scheme.

    This code solves the Sod's shock tube problem
    """
    {
    ######################################################################
    #
    #               basic MUSCL solver for Euler system equations
    #                      by Manuel Diaz, NTU, 29.04.2015
    #
    #                             U_t + F(U)_x = 0,
    #
    # MUSCL based numerical schemes extend the idea of using a linear
    # piecewise approximation to each cell by using slope limited left and
    # right extrapolated states. This results in the following high
    # resolution, TVD discretisation scheme.
    #
    # This code solves the Sod's shock tube problem
    #
    # t=0                                 t=tEnd
    # Density                             Density
    #   ****************|                 *********\
    #                   |                           \
    #                   |                            \
    #                   |                             ****|
    #                   |                                 |
    #                   |                                 ****|
    #                   ***************                       ***********
    #######################################################################
    #   Refs:
    #   [1] Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid
    #   Dynamics" Springer-Verlag, Second Edition, 1999.
    #
    #######################################################################
    }
    params = {
        "cfl": 0.5,
        "tEnd": 0.15,
        "nx": 200,
        "n": 5,
        "IC": 1,
        "limiter": "VA",
        "fluxMth": "HLLC",
        "plot_fig": 1,
    }
    run_muscl_solver(**params)

if __name__ == "__main__":
    main()