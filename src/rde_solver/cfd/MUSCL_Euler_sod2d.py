# run_sod2d.py
import numpy as np
import matplotlib.pyplot as plt
from .MUSCL_EulerRes2d_v0 import MUSCL_EulerRes2d_v0
from MUSCL_TVD_FagioDiFapo.euler_solvers import EulerExact

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

def final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee):
    plt.figure(2)
    plt.subplot(2, 2, 1); plt.plot(xc, r, 'ro', xe, re, '-k'); plt.xlabel('x'); plt.ylabel(r'$\rho$'); plt.legend(['MUSCL-2D', 'Exact'])
    plt.title('SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)')
    plt.subplot(2, 2, 2); plt.plot(xc, u, 'mo', xe, ue, '-k'); plt.xlabel('x'); plt.ylabel('u')
    plt.subplot(2, 2, 3); plt.plot(xc, p, 'ko', xe, pe, '-k'); plt.xlabel('x'); plt.ylabel('p')
    plt.subplot(2, 2, 4); plt.plot(xc, E, 'ro', xe, Ee, '-k'); plt.xlabel('x'); plt.ylabel('E')
    plt.tight_layout()
    plt.show()

def main():
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
    q = np.zeros((nyg, nxg, 4))
    q[1:-1, 1:-1, :] = Q0
    q[:, 0, :] = q[:, 1, :]
    q[:, -1, :] = q[:, -2, :]
    q[0, :, :] = q[1, :, :]
    q[-1, :, :] = q[-2, :, :]

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

    t = 0.0
    it = 0
    while t < tEnd:
        # RK2 step 1
        res = MUSCL_EulerRes2d_v0(q, dt, dx, dy, nxg, nyg, 'MC', 'HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        res2 = MUSCL_EulerRes2d_v0(qs, dt, dx, dy, nxg, nyg, 'MC', 'HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]

        # Extract central slice
        r = q[mid_row, 1:-1, 0]
        u = q[mid_row, 1:-1, 1] / r
        E = (q[mid_row, 1:-1, 3] / r)
        p = (gamma - 1) * r * (E - 0.5 * u ** 2)

        # Update CFD plots every 10 steps for performance
        if it % 10 == 0:
            update_plots(lines, x_slice, r, u, p, E)
        t += dt
        it += 1

    plt.ioff()
    plt.show()

    # Final plot with exact solution (optional, already shown in live plot)
    final_plots(x_slice, r, u, p, E, xe, re, ue, pe, Ee)

if __name__ == "__main__":
    main()