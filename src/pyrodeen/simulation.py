import numpy as np
from pyrodeen.grid import Grid
from pyrodeen.cfd import Solver, BoundaryConfig
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def plot_euler_1d(x, r, u, p, E, fig = None, lines = None):
    if lines is None:
        plt.ion()
        fig, axs = plt.subplots(2, 2, num = 2)
        lines = [
            axs[0, 0].plot([], [], ".b")[0],
            axs[0, 1].plot([], [], ".m")[0],
            axs[1, 0].plot([], [], ".k")[0],
            axs[1, 1].plot([], [], ".r")[0],
        ]
        axs[0, 0].plot(x, r, '-k', label='Exact')
        axs[0, 1].plot(x, u, '-k', label='Exact')
        axs[1, 0].plot(x, p, '-k', label='Exact')
        axs[1, 1].plot(x, E, '-k', label='Exact')
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel(r"$\rho$")
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("u")
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("p")
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("E")
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

    lines[0].set_data(x, r)
    lines[1].set_data(x, u)
    lines[2].set_data(x, p)
    lines[3].set_data(x, E)

    # draw
    plt.draw()
    plt.pause(0.1)

    # Return whether the figure still exists (False if closed)
    return fig, plt.fignum_exists(fig.number), lines

def EulerExact(rho1,u1,p1,rho4,u4,p4,tEnd,n):
    ###########################################################################
    # Classical Gas Exact Riemann Solver
    # Coded by Manuel Diaz, IAM, NTU 03.09.2011.
    ###########################################################################
    # Riemann Solver for solving shoc-tube problems
    #
    ###########################################################################
    # This programs was modified by Manuel Diaz, and is based on the code of
    # [1]  P. Wesseling. PRINCIPLES OF COMPUTATIONAL FLUID DYNAMICS
    # Springer-Verlag, Berlin etc., 2001. ISBN 3-540-67853-0
    # See http://dutita0.twi.tudelft.nl/nw/users/wesseling/
    #
    ###########################################################################
    # NOTE:
    # A Cavitation Check is the is incorporated in the code. It further
    # prevents plotting for possible but physically unlikely case of expansion
    # shocks.
    ###########################################################################
    # INPUT VARIABLES:
    # Problem definition: Conditions at time t=0
    #   rho1, u1, p1
    #   rho4, u4, p4
    # 'tEnd' and 'n' are the final solution time and the gas DoFs.
    ###########################################################################

    # Gamma values
    gamma=(n+2)/n; alpha=(gamma+1)/(gamma-1)

    # Assumed structure of exact solution
    #
    #    \         /      |con |       |s|
    #     \   f   /       |tact|       |h|
    # left \  a  /  state |disc| state |o| right
    # state \ n /    2    |cont|   3   |c| state
    #   1    \ /          |tinu|       |k|   4
    #         |           |ity |       | |

    PRL = p4/p1
    cright = np.sqrt(gamma*p4/rho4)
    cleft  = np.sqrt(gamma*p1/rho1)
    CRL = cright/cleft
    MACHLEFT = (u1-u4)/cleft

    # Basic shock tube relation equation (10.51)
    def f(P):
        return (1 + MACHLEFT * (gamma - 1) / 2 - (gamma - 1) * CRL * (P - 1) / np.sqrt(2 * gamma * (gamma - 1 + (gamma + 1) * P))) ** (2 * gamma / (gamma - 1)) / P - PRL

    # solve for P = p34 = p3/p4
    sol = root_scalar(f, bracket=[0.1, 10], method='brentq')
    if not sol.converged:
        raise RuntimeError("Root finding for p34 did not converge")
    p34 = sol.root

    p3 = p34*p4
    rho3 = rho4*(1+alpha*p34)/(alpha+p34)
    rho2 = rho1*(p34*p4/p1)**(1/gamma)
    u2 = u1-u4+(2/(gamma-1))*cleft*(1-(p34*p4/p1)**((gamma-1)/(2*gamma)))
    c2 = np.sqrt(gamma*p3/rho2)
    spos = 0.5+tEnd*cright*np.sqrt((gamma-1)/(2*gamma)+(gamma+1)/(2*gamma)*p34)+tEnd*u4

    x0 = 0.5
    conpos=x0 + u2*tEnd+tEnd*u4	# Position of contact discontinuity
    pos1 = x0 + (u1-cleft)*tEnd	# Start of expansion fan
    pos2 = x0 + (u2+u4-c2)*tEnd	# End of expansion fan

    # Plot structures
    x = np.arange(0, 1, 0.002)
    p = np.zeros_like(x)
    ux= np.zeros_like(x)
    rho = np.zeros_like(x)
    Mach = np.zeros_like(x)
    cexact = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= pos1:
            p[i] = p1
            rho[i] = rho1
            ux[i] = u1
            cexact[i] = np.sqrt(gamma*p[i]/rho[i])
            Mach[i] = ux[i]/cexact[i]
        elif x[i] <= pos2:
            p[i] = p1*(1+(pos1-x[i])/(cleft*alpha*tEnd))**(2*gamma/(gamma-1))
            rho[i] = rho1*(1+(pos1-x[i])/(cleft*alpha*tEnd))**(2/(gamma-1))
            ux[i] = u1 + (2/(gamma+1))*(x[i]-pos1)/tEnd
            cexact[i] = np.sqrt(gamma*p[i]/rho[i])
            Mach[i] = ux[i]/cexact[i]
        elif x[i] <= conpos:
            p[i] = p3
            rho[i] = rho2
            ux[i] = u2+u4
            cexact[i] = np.sqrt(gamma*p[i]/rho[i])
            Mach[i] = ux[i]/cexact[i]
        elif x[i] <= spos:
            p[i] = p3
            rho[i] = rho3
            ux[i] = u2+u4
            cexact[i] = np.sqrt(gamma*p[i]/rho[i])
            Mach[i] = ux[i]/cexact[i]
        else:
            p[i] = p4
            rho[i] = rho4
            ux[i] = u4
            cexact[i] = np.sqrt(gamma*p[i]/rho[i])
            Mach[i] = ux[i]/cexact[i]
    entro = np.log(p/rho**gamma)  # entropy
    e = p/((gamma-1)*rho)          # internal energy
    t = 2/n*e                       # temperature

    return [x,rho,ux,p,e,t,Mach,entro]


def run_simulation(
    CFL=0.50, t_end=0.15, nx=100, ny=100, lx=1.0, ly=1.0, nv=4, gamma=1.4, plot=False
):
    nx, ny = 100, 1
    lx, ly = 1.0, 0.1
    dx = lx / nx
    dy = ly / ny

    n = 5
    gamma = (n + 2) / n

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
    xe, re, ue, pe, ee, *_ = EulerExact(
        1.0, 0.0, 1.0, 0.125, 0.0, 0.1, t_end, n
    )
    Ee = ee + 0.5 * ue**2
    if plot:
        fig, open, lines = plot_euler_1d(
            xe,
            re,
            ue,
            pe,
            Ee,
        )

    # Time-stepping loop (simplified)
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
        #print("Time: {:.4f}, dt: {:.4e}".format(t, dt))
        if not open:
            break
        # RK2 1st step
        res1 = solver.muscl_euler_res2d()
        q_old = grid.values.copy()  # Store original state before update
        values_1 = q_old - dt * res1
        # Update grid and apply BCs
        grid.values = values_1
        solver.apply_boundary_conditions()
        # RK2 2nd step / update q (correct SSP-RK2: both stages weighted by 0.5*dt)
        res2 = solver.muscl_euler_res2d()
        grid.values = 0.5 * (q_old + values_1 - dt * res2)
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
            _, open, _ = plot_euler_1d(
                np.linspace(dx/2, lx-dx/2, nx),
                r[0, :],
                u[0, :],
                p[0, :],
                E[0, :],
                fig, lines
            )
        if t + dt > t_end:
            dt = t_end - t
        t += dt
    print("time:", t)
    print("pressure:", p)

    plt.ioff()
    plt.show()


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
