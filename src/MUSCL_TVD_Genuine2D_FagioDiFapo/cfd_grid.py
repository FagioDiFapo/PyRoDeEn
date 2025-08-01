import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from .utils import minmod, vanalbada, vanLeer, HLLE1Dflux
from MUSCL_TVD_FagioDiFapo.euler_solvers import EulerExact

class CFDGrid:
    def __init__(self, Lx, Ly, nx, ny, species=None):
        # Grid geometry
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.nvars = 4  # [rho, rho*u, rho*v, rho*E]

        # Species handling
        if species is None:
            self.species_names = []
            self.nspecies = 0
            self.species_index = {}
            self.species = None
        else:
            self.species_names = list(species)
            self.nspecies = len(species)
            self.species_index = {name: i for i, name in enumerate(species)}
            # Initialize species array (uses mass fractions)
            self.species = np.zeros((ny, nx, self.nspecies))

        # Conserved variables (Euler)
        self.q = np.zeros((ny, nx, self.nvars))

        # Chemistry-related (optional, for convenience)
        self.cantera_reactors = [[None for _ in range(nx)] for _ in range(ny)]  # If using Cantera reactors per cell

    def initialize_chemistry(self, fractions_dict=None, mechanism='gri30.yaml'):
        """
        Set initial species mass fractions for the whole grid, normalized using Cantera.
        fractions_dict: {'H2': 1, 'O2': 1, ...}
        """
        if self.nspecies == 0 or self.species is None:
            raise ValueError("No species defined for this grid.")
        if fractions_dict is None:
            raise ValueError("No fractions_dict provided.")

        # Use Cantera to normalize
        gas = ct.Solution(mechanism)
        # Only include species present in the mechanism and in the grid
        valid_fractions = {k: v for k, v in fractions_dict.items() if k in gas.species_names and k in self.species_names}
        gas.Y = valid_fractions  # Cantera will normalize

        # Fill the grid with normalized values
        for name in self.species_names:
            idx = self.species_index[name]
            if name in gas.species_names:
                self.species[:, :, idx] = gas.Y[gas.species_index(name)]
            else:
                self.species[:, :, idx] = 0.0

    def muscl_euler_res2d(self, limiter='MC', fluxMethod='HLLE1d'):
        """
        A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
        Upstream Centered Scheme for Conservation Laws (MUSCL).
        Vectorized version: uses only numpy arrays for all states and residuals.
        Refactored to use CFDGrid object properties.
        Original code written by Manuel Diaz, NTU, 05.25.2015.
        """
        q = self.q
        dx = self.dx
        dy = self.dy
        N = self.nx
        M = self.ny

        # Allocate arrays for all states
        qN = np.zeros((M, N, 4))
        qS = np.zeros((M, N, 4))
        qE = np.zeros((M, N, 4))
        qW = np.zeros((M, N, 4))
        residual = np.zeros((M, N, 4))

        # Compute and limit slopes at cells (i,j)
        for k in range(4):
            dqw = q[1:-1, 1:-1, k] - q[1:-1, :-2, k]
            dqe = q[1:-1, 2:, k] - q[1:-1, 1:-1, k]
            dqs = q[1:-1, 1:-1, k] - q[:-2, 1:-1, k]
            dqn = q[2:, 1:-1, k] - q[1:-1, 1:-1, k]
            if limiter == 'MC':
                dqc_x = (q[1:-1, 2:, k] - q[1:-1, :-2, k]) / 2
                dqdx = minmod([2*dqw, 2*dqe, dqc_x])
                dqc_y = (q[2:, 1:-1, k] - q[:-2, 1:-1, k]) / 2
                dqdy = minmod([2*dqs, 2*dqn, dqc_y])
            elif limiter == 'MM':
                dqdx = minmod([dqw, dqe])
                dqdy = minmod([dqs, dqn])
            elif limiter == 'VA':
                dqdx = vanalbada(dqw, dqe, dx)
                dqdy = vanalbada(dqs, dqn, dy)
            elif limiter == 'VL':
                dqdx = vanLeer(dqw, dqe)
                dqdy = vanLeer(dqs, dqn)
            else:
                raise ValueError(f"Unknown limiter: {limiter}")

            # Assign slopes to states (interior only)
            qE[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdx / 2
            qW[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdx / 2
            qN[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdy / 2
            qS[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdy / 2

        # Residuals: x-direction
        for i in range(1, M-1):
            for j in range(1, N-2):
                qxL = qE[i, j, :]
                qxR = qW[i, j+1, :]
                if fluxMethod == 'HLLE1d':
                    flux = HLLE1Dflux(qxL, qxR, [1, 0])
                else:
                    raise ValueError("flux option not available")
                residual[i, j, :] += flux / dx
                residual[i, j+1, :] -= flux / dx

        # Residuals: y-direction
        for i in range(1, M-2):
            for j in range(1, N-1):
                qyL = qN[i, j, :]
                qyR = qS[i+1, j, :]
                if fluxMethod == 'HLLE1d':
                    flux = HLLE1Dflux(qyL, qyR, [0, 1])
                else:
                    raise ValueError("flux option not available")
                residual[i, j, :] += flux / dy
                residual[i+1, j, :] -= flux / dy

        # Set BCs: boundary flux contributions
        # North face (i = M-1)
        for j in range(1, N - 1):
            qR = qS[M - 2, j, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [0, 1])
            residual[M - 2, j, :] += flux / dy

        # East face (j = N-2)
        for i in range(1, M - 1):
            qR = qW[i, N - 2, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [1, 0])
            residual[i, N - 2, :] += flux / dx

        # South face (i = 1)
        for j in range(1, N - 1):
            qR = qN[1, j, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [0, -1])
            residual[1, j, :] += flux / dy

        # West face (j = 1)
        for i in range(1, M - 1):
            qR = qE[i, 1, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [-1, 0])
            residual[i, 1, :] += flux / dx

        # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
        res = np.zeros_like(residual)
        res[1:M-1, 1:N-1, :] = residual[1:M-1, 1:N-1, :]

        return res

def test_initialization():
    # Define species for hydrogen-oxygen system with radicals
    species = ['H2', 'O2', 'H', 'O', 'OH', 'HO2', 'H2O2', 'H2O']

    # Create a 1m x 1m grid with 100x100 cells
    grid = CFDGrid(Lx=1.0, Ly=1.0, nx=100, ny=100, species=species)

    print("Grid created with the following species:")
    print(grid.species_names)
    print("Species index mapping:")
    print(grid.species_index)
    print("Shape of species array:", None if grid.species is None else grid.species.shape)
    print("Shape of conserved variables array:", grid.q.shape)

    # Initialize chemistry with mass fractions (example: 100% H2, 100% O2)
    grid.initialize_chemistry({'H2': 1, 'O2': 1, 'H': 0, 'O': 0, 'OH': 0, 'HO2': 0, 'H2O2': 0, 'H2O': 0})

    # Print normalized mass fractions for verification
    print("Normalized mass fractions (from Cantera):")
    for name in grid.species_names:
        idx = grid.species_index[name]
        print(f"{name}: {grid.species[0,0,idx]:.6f}")

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
        res = grid.muscl_euler_res2d(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d(limiter='MC', fluxMethod='HLLE1d')
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

if __name__ == "__main__":
    # python -m src.MUSCL_TVD_Genuine2D_FagioDiFapo.cfd_grid to run as module
    run_2d_sod_with_grid()