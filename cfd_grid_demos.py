import time
import numpy as np
import matplotlib.pyplot as plt

# Adjust the import path as needed for your project structure:
from MUSCL_TVD_FagioDiFapo.euler_solvers import EulerExact
from src.MUSCL_TVD_Genuine2D_FagioDiFapo.cfd_grid import CFDGrid

def blast_ic_2d(x, y, p_high=1.0, p_low=0.1, r0=0.1):
    """2D blast wave: high pressure in center, low elsewhere."""
    xc, yc = 0.5, 0.5
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(r < r0, p_high, p_low)
    return rho, u, v, p

def blast_ic_2d_square(x, y, p_high=1.0, p_low=0.1, halfwidth=0.1):
    """2D blast wave: high pressure in a square at the center, low elsewhere."""
    xc, yc = 0.5, 0.5
    mask = (np.abs(x - xc) < halfwidth) & (np.abs(y - yc) < halfwidth)
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

def final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee):
    plt.figure(2)
    plt.subplot(2, 2, 1); plt.plot(xc, r, 'ro', xe, re, '-k'); plt.xlabel('x'); plt.ylabel(r'$\rho$'); plt.legend(['MUSCL-2D', 'Exact'])
    plt.title('SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)')
    plt.subplot(2, 2, 2); plt.plot(xc, u, 'mo', xe, ue, '-k'); plt.xlabel('x'); plt.ylabel('u')
    plt.subplot(2, 2, 3); plt.plot(xc, p, 'ko', xe, pe, '-k'); plt.xlabel('x'); plt.ylabel('p')
    plt.subplot(2, 2, 4); plt.plot(xc, E, 'ro', xe, Ee, '-k'); plt.xlabel('x'); plt.ylabel('E')
    plt.tight_layout()
    plt.show()

def run_2d_sod_with_grid():
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

    plt.ioff()
    plt.show()

    # Final plot with exact solution (optional, already shown in live plot)
    final_plots(x_slice, r, u, p, E, xe, re, ue, pe, Ee)


def run_blast_test():
    start = time.time()
    # Parameters
    nx, ny = 100, 100
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    tEnd = 1.0
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
        if it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    plt.show()
    plt.close(fig)
    plt.close('all')
    end = time.time()
    print(f"Non-vectorized simulation time: {end - start:.3f} seconds")
    return end - start

def run_blast_test_vectorized():
    start = time.time()
    # Parameters
    nx, ny = 100, 100
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    tEnd = 1.0
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
        if it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    plt.show()
    plt.close(fig)
    plt.close('all')
    end = time.time()
    print(f"Vectorized simulation time: {end - start:.3f} seconds")
    return end - start

if __name__ == "__main__":
    # to run as module
    # python -m src.MUSCL_TVD_Genuine2D_FagioDiFapo.cfd_grid
    run_2d_sod_with_grid()
    #t_vec = run_blast_test_vectorized()
    #t_nonvec = run_blast_test()
    #if t_vec > 0:
    #    print(f"Speedup: {t_nonvec / t_vec:.2f}x faster (vectorized vs non-vectorized)")