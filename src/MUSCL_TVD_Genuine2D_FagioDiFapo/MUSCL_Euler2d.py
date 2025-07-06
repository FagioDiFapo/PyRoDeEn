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
from .MUSCL_EulerRes2d_v0 import MUSCL_EulerRes2d_v0

def Euler_IC2d(x, y, input):
    # Load the IC of a 1D Riemann classical schok tube problem configuration.
    # In the notation we take advantage of the matlab array notation as follows
    #
    #   1.0 +-----------+-----------+
    #       |           |           |
    #       |   reg 2   |   reg 1   |
    #       |           |           |
    #   0.5 +-----------+-----------+
    #       |           |           |
    #       |   reg 3   |   reg 4   |
    #       |           |           |
    #   0.0 +-----------+-----------+
    #      0.0         0.5         1.0
    #
    # prop = [prop_reg1 , prop_reg2 , prop_reg3 , prop_reg4]
    #
    #   r = rho/density
    #   u = velocity in x direction
    #   v = velocity in y direction
    #   p = Pressure
    #
    # Manuel Diaz, NTU, 2014.06.27

    ## Initial Physical Properties per case:
    match input:
        case 1:  # Configuration 1
            print('Configuration 1')
            p = [  1.0 ,  0.4    ,  0.0439 ,  0.15   ]
            r = [  1.0 ,  0.5197 ,  0.1072 ,  0.2579 ]
            u = [  0.0 , -0.7259 , -0.7259 ,  0.0    ]
            v = [  0.0 , -0.0    , -1.4045 , -1.4045 ]
        case 2:  # Configuration 2
            print('Configuration 2')
            p = [  1.0 ,  0.4    ,  1.0    ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  1.0    ,  0.5197 ]
            u = [  0.0 , -0.7259 , -0.7259 ,  0.0    ]
            v = [  0.0 ,  0.0    , -0.7259 , -0.7259 ]
        case 3:  # Configuration 3
            print('Configuration 3')
            p = [  1.5 ,  0.3    ,  0.029 ,  0.3    ]
            r = [  1.5 ,  0.5323 ,  0.138 ,  0.5323 ]
            u = [  0.0 ,  1.206  ,  1.206 ,  0.0    ]
            v = [  0.0 ,  0.0    ,  1.206 ,  1.206  ]
        case 4:  # Configuration 4
            print('Configuration 4')
            p = [  1.1,  0.35   ,  1.1    ,  0.35   ]
            r = [  1.1,  0.5065 ,  1.1    ,  0.5065 ]
            u = [  0.0,  0.8939 ,  0.8939 ,  0.0    ]
            v = [  0.0,  0.0    ,  0.8939 ,  0.8939 ]
        case 5:  # Configuration 5
            print('Configuration 5')
            p = [  1.0 ,  1.0   ,  1.0   ,  1.0   ]
            r = [  1.0 ,  2.0   ,  1.0   ,  3.0   ]
            u = [ -0.75, -0.75  ,  0.75  ,  0.75  ]
            v = [ -0.5 ,  0.5   ,  0.5   , -0.5   ]
        case 6:  # Configuration 6
            print('Configuration 6')
            p = [  1.0  ,  1.0  ,  1.0  ,  1.0  ]
            r = [  1.0  ,  2.0  ,  1.0  ,  3.0  ]
            u = [  0.75 ,  0.75 , -0.75 , -0.75 ]
            v = [ -0.5  ,  0.5  ,  0.5  , -0.5  ]
        case 7:  # Configuration 7
            print('Configuration 7')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  0.8 ,  0.5197 ]
            u = [  0.1 , -0.6259 ,  0.1 ,  0.1    ]
            v = [  0.1 ,  0.1    ,  0.1 , -0.6259 ]
        case 8:  # Configuration 8
            print('Configuration 8')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5197 ,  1.0    ,  0.8 ,  1.0    ]
            u = [  0.1    , -0.6259 ,  0.1 ,  0.1    ]
            v = [  0.1    ,  0.1    ,  0.1 , -0.6259 ]
        case 9:  # Configuration 9
            print('Configuration 9')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.039  ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  0.3 , -0.3 , -0.8133 , -0.4259 ]
        case 10:  # Configuration 10
            print('Configuration 10')
            p = [  1.0    ,  1.0    ,  0.3333 ,  0.3333 ]
            r = [  1.0    ,  0.5    ,  0.2281 ,  0.4562 ]
            u = [  0.0    ,  0.0    ,  0.0    ,  0.0    ]
            v = [  0.4297 ,  0.6076 , -0.6076 , -0.4297 ]
        case 11:  # Configuration 11
            print('Configuration 11')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5313 ,  0.8 ,  0.5313 ]
            u = [  0.1 ,  0.8276 ,  0.1 ,  0.1    ]
            v = [  0.0 ,  0.0    ,  0.0 ,  0.7276 ]
        case 12:  # Configuration 12
            print('Configuration 12')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5313 ,  1.0    ,  0.8 ,  1.0    ]
            u = [  0.0    ,  0.7276 ,  0.0 ,  0.0    ]
            v = [  0.0    ,  0.0    ,  0.0 ,  0.7276 ]
        case 13:
            print('Configuration 13')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5313 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [ -0.3 ,  0.3 ,  0.8145 ,  0.4276 ]
        case 14:  # Configuration 14
            print('Configuration 14')
            p = [  8.0    ,  8.0    ,  2.6667 ,  2.6667 ]
            r = [  2.0    ,  1.0    ,  0.4736 ,  0.9474 ]
            u = [  0.0    ,  0.0    ,  0.0    ,  0.0    ]
            v = [ -0.5606 , -1.2172 ,  1.2172 ,  1.1606 ]
        case 15:  # Configuration 15
            print('Configuration 15')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  0.8 ,  0.5313 ]
            u = [  0.1 , -0.6259 ,  0.1 ,  0.1    ]
            v = [ -0.3 , -0.3    , -0.3 ,  0.4276 ]
        case 16:  # Configuration 16
            print('Configuration 16')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5313 ,  1.0222 ,  0.8 ,  1.0    ]
            u = [  0.1    , -0.6179 ,  0.1 ,  0.1    ]
            v = [  0.1    ,  0.1    ,  0.1 ,  0.8276 ]
        case 17:  # Configuration 17
            print('Configuration 17')
            p = [  1.0 ,  1.0 , 0.4    ,  0.4     ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [ -0.4 , -0.3 ,  0.2145 , -1.1259 ]
        case 18:  # Configuration 18
            print('Configuration 18')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  1.0 , -0.3 ,  0.2145 ,  0.2741 ]
        case 19:  # Configuration 19
            print('Configuration 19')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  0.3 , -0.3 ,  0.2145 , -0.4259 ]
        case 'sod_x':  # Sod's shock tube in the x-direction (2-d test)
            print('Sods Shocktube in the x-direction (2-d test)')
            p = [  0.1   ,  1 ,  1 ,  0.1   ]
            r = [  0.125 ,  1 ,  1 ,  0.125 ]
            u = [  0     ,  0 ,  0 ,  0     ]
            v = [  0     ,  0 ,  0 ,  0     ]
        case 'sod_y':  # Sod's shock tube in the y-direction (2-d test)
            print('Sods Shocktube in the y-direction (2-d test)')
            p = [  1 ,  1 ,  0.1   ,  0.1   ]
            r = [  1 ,  1 ,  0.125 ,  0.125 ]
            u = [  0 ,  0 ,  0     ,  0     ]
            v = [  0 ,  0 ,  0     ,  0     ]
        case 'constant':  # Constant state (2-d test)
            print('Constant state (2-d test)')
            p = [  0.1   ,  0.1   ,  0.1   ,  0.1   ]
            r = [  0.125 ,  0.125 ,  0.125 ,  0.125 ]
            u = [  0     ,  0     ,  0     ,  0     ]
            v = [  0     ,  0     ,  0     ,  0     ]
        case _:
            raise Exception(f"specified case '{input}' not available")

    ## Print configuration of selected IC
    print('\n')
    print('          reg 1   reg 2   reg 3   reg 4')
    print(f'density : {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f}')
    print(f'  x-vel : {u[0]:7.4f} {u[1]:7.4f} {u[2]:7.4f} {u[3]:7.4f}')
    print(f'  y-vel : {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} {v[3]:7.4f}')
    print(f'Presure : {p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f} {p[3]:7.4f}')
    print('\n')

    reg1 = (x >= 0.5) & (y >= 0.5);  # region 1
    reg2 = (x < 0.5) & (y >= 0.5); # region 2
    reg3 = (x < 0.5) & (y < 0.5); # region 3
    reg4 = (x >= 0.5) & (y < 0.5); # region 4

    # Initial Condition for our 2D domain
    r_0 = r[0]*reg1 + r[1]*reg2 + r[2]*reg3 + r[3]*reg4; # Density, rho
    u_0 = u[0]*reg1 + u[1]*reg2 + u[2]*reg3 + u[3]*reg4; # velocity in x
    v_0 = v[0]*reg1 + v[1]*reg2 + v[2]*reg3 + v[3]*reg4; # velocity in y
    p_0 = p[0]*reg1 + p[1]*reg2 + p[2]*reg3 + p[3]*reg4; # temperature.

    return r_0, u_0, v_0, p_0

# Parameters
CFL     = 0.50      # CFL number
tEnd    = 0.05      # Final time
nx      = 100       # Number of cells/Elements in x
ny      = 100       # Number of cells/Elements in y
n       = 5         # Degrees of freedom: ideal air=5, monoatomic gas=3
IC      = 5         # 19 IC cases are available
fluxMth = 'HLLE1d'  # HLLE1d, HLLE2d
method  = 1         # 1:Dim by Dim, 2:HLLE2d 1st-order, 3:HLLE2d 2nd-order
limiter = 'MC'      # MM, MC, VA, VL
plotFig = True      # True: visualize evolution

# Ratio of specific heats for ideal di-atomic gas
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
E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0 ** 2 + v0 ** 2)  # Total Energy
c0 = np.sqrt(gamma * p0 / r0)                              # Speed of sound
Q0 = np.stack([r0, r0 * u0, r0 * v0, r0 * E0], axis=2)     # initial state

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

# Initialize parpool
#poolobj = gcp('nocreate'); # If no pool, do not create new one.
#if isempty(poolobj); parpool('local',2); end

# Select residual function based on method
if method == 1:
    MUSCL_EulerRes2d = MUSCL_EulerRes2d_v0  # Do HLLE1d Dim by Dim
elif method == 2:
    MUSCL_EulerRes2d = MUSCL_EulerRes2d_v1  # 1st-order HLLE2d (working on it)
elif method == 3:
    MUSCL_EulerRes2d = MUSCL_EulerRes2d_v2  # 2nd-order HLLE2d (working on it)
else:
    raise Exception('flux assemble not available')

# Internal indexes for plotting (Python is 0-based)
in_idx = slice(1, -1)
jn_idx = slice(1, -1)

# Configure figure
if plotFig:
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

    # Natural BCs (copy boundary values from adjacent internal cells)
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
        for ax in axs.flat:
            ax.clear()
        axs[0, 0].contourf(x, y, r[1:-1, 1:-1])
        axs[0, 0].set_title(r'$\rho$')
        axs[0, 1].contourf(x, y, u[1:-1, 1:-1])
        axs[0, 1].set_title(r'$u_x$')
        axs[1, 0].contourf(x, y, v[1:-1, 1:-1])
        axs[1, 0].set_title(r'$u_y$')
        axs[1, 1].contourf(x, y, p[1:-1, 1:-1])
        axs[1, 1].set_title('p')
        plt.tight_layout()
        plt.pause(0.01)

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
c = np.sqrt(gamma * p / r)   # Speed of sound
Mx = u / c
My = v / c
U = np.sqrt(u ** 2 + v ** 2)
M = U / c
p_ref = 101325.0         # Reference air pressure (N/m^2)
r_ref = 1.225            # Reference air density (kg/m^3)
s = 1 / (gamma - 1) * (np.log(p / p_ref) + gamma * np.log(r_ref / r))  # Entropy w.r.t reference
ss = np.log(p / r ** gamma)  # Dimensionless Entropy
r_x = r * u             # Mass Flow rate per unit area
r_y = r * v             # Mass Flow rate per unit area
e = p / ((gamma - 1) * r)   # Internal Energy

# Final plot
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
