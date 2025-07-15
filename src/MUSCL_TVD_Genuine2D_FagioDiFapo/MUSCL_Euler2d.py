# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#               basic MUSCL solver for Euler system equations
#                      by Manuel Diaz, NTU, 29.04.2015
#
#                         U_t + F(U)_x + G(U)_y = 0,
#
# MUSCL based numerical schemes extend the idea of using a linear
# piecewise approximation to each cell by using slope limited left and
# right extrapolated states. This results in the following high
# resolution, TVD discretisation scheme.
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Refs:
#   [1] Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid Dynamics" Springer-Verlag, Second Edition, 1999.
#   [2] Balsara, Dinshaw S. "A two-dimensional HLLC Riemann solver for conservation laws: Application to Euler and magnetohydrodynamic flows." Journal of Computational Physics 231.22 (2012): 7476-7503.
#   [3] Einfeldt, Bernd. "On Godunov-type methods for gas dynamics." SIAM Journal on Numerical Analysis 25.2 (1988): 294-318.
#   [4] Kurganov, Alexander, and Eitan Tadmor. "Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers." Numerical Methods for Partial Differential Equations 18.5 (2002): 584-608.
#   [5] Vides, Jeaniffer, Boniface Nkonga, and Edouard Audit. "A simple two-dimensional extension of the HLL Riemann solver for gas dynamics." (2014).
#
# coded by Manuel Diaz, 2015.05.10
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
import time
from .utils import Euler_IC2d
from .MUSCL_EulerRes2d_v0 import MUSCL_EulerRes2d_v0
from .MUSCL_EulerRes2d_v1 import MUSCL_EulerRes2d_v1

def init_plots(x, y, r0, u0, v0, p0):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    h1 = axs[0, 0].contourf(x, y, r0)
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title(r'$\rho$')

    h2 = axs[0, 1].contourf(x, y, u0)
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title(r'$u_x$')

    h3 = axs[1, 0].contourf(x, y, v0)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title(r'$u_y$')

    h4 = axs[1, 1].contourf(x, y, p0)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title('p')

    plt.tight_layout()
    plt.show(block=False)
    return fig, axs

def update_plots(axs, x, y, r, u, v, p):
    axs[0, 0].clear()
    axs[0, 0].contourf(x, y, r)
    axs[0, 0].set_title(r'$\rho$')
    axs[0, 1].clear()
    axs[0, 1].contourf(x, y, u)
    axs[0, 1].set_title(r'$u_x$')
    axs[1, 0].clear()
    axs[1, 0].contourf(x, y, v)
    axs[1, 0].set_title(r'$u_y$')
    axs[1, 1].clear()
    axs[1, 1].contourf(x, y, p)
    axs[1, 1].set_title('p')
    plt.tight_layout()
    plt.pause(0.01)

def final_plots(x, y, r, U, p, ss, M, e):
    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
    ncontours = 22
    axs2[0, 0].contour(x, y, r, ncontours)
    axs2[0, 0].set_title('Density (kg/m^3)')
    axs2[0, 1].contour(x, y, U, ncontours)
    axs2[0, 1].set_title('Velocity Magnitude (m/s)')
    axs2[0, 2].contour(x, y, p, ncontours)
    axs2[0, 2].set_title('Pressure (Pa)')
    axs2[1, 0].contour(x, y, ss, ncontours)
    axs2[1, 0].set_title('Entropy/R gas')
    axs2[1, 1].contour(x, y, M, ncontours)
    axs2[1, 1].set_title('Mach number')
    axs2[1, 2].contour(x, y, e, ncontours)
    axs2[1, 2].set_title('Internal Energy (kg/m^2s)')
    for ax in axs2.flat:
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
    plt.tight_layout()
    axs2[0, 0].set_title('MUSCL with genuinely 2D HLL fluxes')
    plt.show()

def run_muscl_solver(
    CFL=0.50, tEnd=0.05, nx=100, ny=100, n=5, IC=5, fluxMth='HLLE1d',
    method=1, limiter='MC', plotFig=True
):
    gamma = (n + 2) / n

    # Discretize spatial domain
    Lx = 1.0
    dx = Lx / nx
    xc = np.linspace(dx / 2, Lx - dx / 2, nx)
    Ly = 1.0
    dy = Ly / ny
    yc = np.linspace(dy / 2, Ly - dy / 2, ny)
    x, y = np.meshgrid(xc, yc)

    # Set initial conditions
    r0, u0, v0, p0 = Euler_IC2d(x, y, IC)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0 ** 2 + v0 ** 2)
    c0 = np.sqrt(gamma * p0 / r0)
    Q0 = np.stack([r0, r0 * u0, r0 * v0, r0 * E0], axis=2)

    # Set q-array & adjust grid for ghost cells
    nxg = nx + 2
    nyg = ny + 2
    q0 = np.zeros((nyg, nxg, 4))
    q0[1:-1, 1:-1, :] = Q0

    # Boundary Conditions in ghost cells (Natural BCs)
    q0[:, 0, :] = q0[:, 1, :]
    q0[:, -1, :] = q0[:, -2, :]
    q0[0, :, :] = q0[1, :, :]
    q0[-1, :, :] = q0[-2, :, :]

    # Discretize time domain
    vn = np.sqrt(u0 ** 2 + v0 ** 2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt0 = CFL * np.min(np.array([dx / a0, dy / a0]))

    # Select residual function based on method
    if method == 1:
        MUSCL_EulerRes2d = MUSCL_EulerRes2d_v0
    elif method == 2:
        MUSCL_EulerRes2d = MUSCL_EulerRes2d_v1
    else:
        raise Exception('flux assemble not available')

    # Configure figure
    if plotFig:
        fig, axs = init_plots(x, y, r0, u0, v0, p0)

    # Solver Loop
    q = q0.copy()
    t = 0.0
    it = 0
    dt = dt0
    a = a0

    start_time = time.time()
    while t < tEnd:
        # RK2 1st step
        qs = q - dt * MUSCL_EulerRes2d(q, dt, dx, dy, nxg, nyg, limiter, fluxMth)
        # Natural BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 2nd step / update q
        q = 0.5 * (q + qs - dt * MUSCL_EulerRes2d(qs, dt, dx, dy, nxg, nyg, limiter, fluxMth))
        # Natural BCs again
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        # Compute flow properties
        r = q[:, :, 0]
        u = q[:, :, 1] / r
        v = q[:, :, 2] / r
        E = q[:, :, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))
        c = np.sqrt(gamma * p / r)
        # Update dt and time
        vn = np.sqrt(u ** 2 + v ** 2)
        lambda1 = vn + c
        lambda2 = vn - c
        a = np.max(np.abs(np.concatenate([lambda1.flatten(), lambda2.flatten()])))
        dt = CFL * min(dx / a, dy / a)
        if t + dt > tEnd:
            dt = tEnd - t
        t += dt
        it += 1
        # Plot figure
        if plotFig and it % 2 == 0:
            update_plots(axs, x, y, r[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1], p[1:-1, 1:-1])

    cputime = time.time() - start_time

    # Remove ghost cells
    q = q[1:-1, 1:-1, :]
    nx = nxg - 2
    ny = nyg - 2

    # Compute flow properties
    r = q[:, :, 0]
    u = q[:, :, 1] / r
    v = q[:, :, 2] / r
    E = q[:, :, 3] / r
    p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

    # Calculation of flow parameters
    c = np.sqrt(gamma * p / r)
    Mx = u / c
    My = v / c
    U = np.sqrt(u ** 2 + v ** 2)
    M = U / c
    p_ref = 101325.0
    r_ref = 1.225
    s = 1 / (gamma - 1) * (np.log(p / p_ref) + gamma * np.log(r_ref / r))
    ss = np.log(p / r ** gamma)
    r_x = r * u
    r_y = r * v
    e = p / ((gamma - 1) * r)
    # Final plot
    if plotFig:
        final_plots(x, y, r, U, p, ss, M, e)

    print(f'Execution time = {cputime:.2f} seconds')
    return q, r, u, v, p, E, cputime

def main():
    params = {
        "CFL": 0.50,
        "tEnd": 0.05,
        "nx": 100,
        "ny": 100,
        "n": 5,
        "IC": 5,
        "fluxMth": 'HLLE1d',
        "method": 1,
        "limiter": 'MC',
        "plotFig": True,
    }
    run_muscl_solver(**params)

if __name__ == "__main__":
    main()
