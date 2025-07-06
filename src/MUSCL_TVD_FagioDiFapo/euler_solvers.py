import numpy as np
from scipy.optimize import root_scalar

def Euler_IC1d(x, input):
    # Load the IC of a classical 1D Riemann Problems.
    #
    # By Manuel Diaz 2012.10.24.
    # In the notation we take advantage of the matlab array notation as follows
    #
    # prop = [prop_left , prop_right]
    #
    # Notation:
    # u   = Velocity in x direction
    # p   = Pressure
    # rho = Density
    # r   = Fugacity
    # E   = Enerty
    # t   = temperature
    #
    ## Riemann Problems

    match input:
        case 1:  # Configuration 1, Sod's Problem
            p   = [1,   0.1  ]
            u   = [0,   0    ]
            rho = [1,   0.125]
            tEnd = 0.1; cfl = 0.90
        case _:
            raise Exception(f"specified case '{input}' not available")

    print(f'density (L): {rho[0]}')
    print(f'velocity(L): {u[0]}')
    print(f'Presure (L): {p[0]}')
    print(f'density (R): {rho[1]}')
    print(f'velocity(R): {u[1]}')
    print(f'Presure (R): {p[1]}')

    ## Load Selected case Initial condition:
    # Pre-Allocate variables
    r0 = np.zeros(len(x))
    u0 = np.zeros(len(x))
    p0 = np.zeros(len(x))

    # Parameters of regions dimensions
    x_middle = (x[-1]-x[0])/2

    # Initial Condition for our 2D domain
    # Fugacity
    r0 = [(rho[0] if x[i]<=x_middle else rho[1]) for i in range(len(x))]
    # Velocity in x
    u0 = [(u[0] if x[i]<=x_middle else u[1]) for i in range(len(x))]
    # temperature
    p0 = [(p[0] if x[i]<=x_middle else p[1]) for i in range(len(x))]

    return [r0,u0,p0,tEnd,cfl]

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

def minmod(v):
    # Using Harten's generalized definition
    # minmod: zero if opposite sign, otherwise the one of smaller magnitude.
    v = np.array(v)
    s = np.sum(np.sign(v)) / len(v)
    if abs(s) == 1:
        return s * np.min(np.abs(v))
    else:
        return 0.0

def vanalbada(da, db, h):
    # Van Albada Slope Limiter Function
    # vanAlbada: extend the symmetric formulation of the van Leer limiter
    eps2 = (0.3 * h) ** 3
    numerator = (db ** 2 + eps2) * da + (da ** 2 + eps2) * db
    denominator = da ** 2 + db ** 2 + 2 * eps2
    return 0.5 * (np.sign(da) * np.sign(db) + 1) * numerator / denominator if denominator != 0 else 0.0

def LFflux(qL, qR, gamma, smax):
    # Lax-Friedrichs flux
    rL = qL[0]
    uL = qL[1] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    HL = (qL[2] + pL) / rL

    rR = qR[0]
    uR = qR[1] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    HR = (qR[2] + pR) / rR

    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (rL * HL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (rR * HR)])

    # Lax-Friedrichs Numerical Flux
    return 0.5 * (FR + FL + smax * (qL - qR))

def ROEflux(qL, qR, gamma):
    # Roe flux function
    rL = qL[0]
    uL = qL[1] / rL
    EL = qL[2] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    aL = np.sqrt(gamma * pL / rL)
    HL = (qL[2] + pL) / rL

    rR = qR[0]
    uR = qR[1] / rR
    ER = qR[2] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    aR = np.sqrt(gamma * pR / rR)
    HR = (qR[2] + pR) / rR

    # Roe averages
    RT = np.sqrt(rR / rL)
    r = RT * rL
    u = (uL + RT * uR) / (1 + RT)
    H = (HL + RT * HR) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - u * u / 2))

    # Differences in primitive variables
    dr = rR - rL
    du = uR - uL
    dP = pR - pL

    # Wave strength (Characteristic Variables)
    dV = np.array([
        (dP - r * a * du) / (2 * a ** 2),
        -(dP / (a ** 2) - dr),
        (dP + r * a * du) / (2 * a ** 2)
    ])

    # Absolute values of the wave speeds (Eigenvalues)
    ws = np.array([abs(u - a), abs(u), abs(u + a)])

    # Harten's Entropy Fix
    Da = max(0, 4 * ((uR - aR) - (uL - aL)))
    if ws[0] < Da / 2 and Da != 0:
        ws[0] = ws[0] * ws[0] / Da + Da / 4
    Da = max(0, 4 * ((uR + aR) - (uL + aL)))
    if ws[2] < Da / 2 and Da != 0:
        ws[2] = ws[2] * ws[2] / Da + Da / 4

    # Right eigenvectors
    R = np.array([
        [1, 1, 1],
        [u - a, u, u + a],
        [H - u * a, u ** 2 / 2, H + u * a]
    ])

    # Compute the average flux
    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (rL * EL + pL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (rR * ER + pR)])

    # Add the matrix dissipation term to complete the Roe flux
    return 0.5 * (FL + FR - R @ (ws * dV))

def RUSflux(qL, qR, gamma):
    # Rusanov flux
    rL = qL[0]
    uL = qL[1] / rL
    EL = qL[2] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    HL = (qL[2] + pL) / rL

    rR = qR[0]
    uR = qR[1] / rR
    ER = qR[2] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    HR = (qR[2] + pR) / rR

    # Roe averages
    RT = np.sqrt(rR / rL)
    u = (uL + RT * uR) / (1 + RT)
    H = (HL + RT * HR) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - u * u / 2))

    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (rL * EL + pL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (rR * ER + pR)])

    smax = abs(u) + a
    return 0.5 * (FR + FL + smax * (qL - qR))

def AUSMflux(qL, qR, gamma):
    # AUSM numerical flux
    rL = qL[0]
    uL = qL[1] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    aL = np.sqrt(gamma * pL / rL)
    ML = uL / aL
    HL = (qL[2] + pL) / rL

    rR = qR[0]
    uR = qR[1] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    aR = np.sqrt(gamma * pR / rR)
    MR = uR / aR
    HR = (qR[2] + pR) / rR

    # Positive M and p in the LEFT cell
    if ML <= -1:
        Mp = 0
        Pp = 0
    elif ML < 1:
        Mp = (ML + 1) ** 2 / 4
        Pp = pL * (1 + ML) ** 2 * (2 - ML) / 4
    else:
        Mp = ML
        Pp = pL

    # Negative M and p in the RIGHT cell
    if MR <= -1:
        Mm = MR
        Pm = pR
    elif MR < 1:
        Mm = -(MR - 1) ** 2 / 4
        Pm = pR * (1 - MR) ** 2 * (2 + MR) / 4
    else:
        Mm = 0
        Pm = 0

    # Positive Part of Flux evaluated in the left cell
    Fp = np.zeros(3)
    Fm = np.zeros(3)
    Fp[0] = max(0, Mp + Mm) * aL * rL
    Fp[1] = max(0, Mp + Mm) * aL * rL * uL + Pp
    Fp[2] = max(0, Mp + Mm) * aL * rL * HL

    # Negative Part of Flux evaluated in the right cell
    Fm[0] = min(0, Mp + Mm) * aR * rR
    Fm[1] = min(0, Mp + Mm) * aR * rR * uR + Pm
    Fm[2] = min(0, Mp + Mm) * aR * rR * HR

    # Compute the flux: Fp(uL) + Fm(uR)
    return Fp + Fm

def HLLEflux(qL, qR, gamma):
    # Compute HLLE flux
    rL = qL[0]
    uL = qL[1] / rL
    EL = qL[2] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    aL = np.sqrt(gamma * pL / rL)
    HL = (qL[2] + pL) / rL

    rR = qR[0]
    uR = qR[1] / rR
    ER = qR[2] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    aR = np.sqrt(gamma * pR / rR)
    HR = (qR[2] + pR) / rR

    # Roe averages
    RT = np.sqrt(rR / rL)
    u = (uL + RT * uR) / (1 + RT)
    H = (HL + RT * HR) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - u * u / 2))

    # Wave speed estimates
    SLm = min(uL - aL, u - a)
    SRp = max(uR + aR, u + a)

    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (rL * EL + pL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (rR * ER + pR)])

    # Compute the HLL flux
    if SLm >= 0:
        return FL
    elif SLm <= 0 and SRp >= 0:
        # True HLLE function
        return (SRp * FL - SLm * FR + SLm * SRp * (qR - qL)) / (SRp - SLm)
    elif SRp <= 0:
        return FR
    else:
        return np.zeros(3)

def HLLCflux(qL, qR, gamma):
    # Compute HLLC flux for Euler equations
    # Left state
    rL = qL[0]
    uL = qL[1] / rL
    EL = qL[2] / rL
    pL = (gamma - 1) * (qL[2] - rL * uL * uL / 2)
    aL = np.sqrt(gamma * pL / rL)

    # Right state
    rR = qR[0]
    uR = qR[1] / rR
    ER = qR[2] / rR
    pR = (gamma - 1) * (qR[2] - rR * uR * uR / 2)
    aR = np.sqrt(gamma * pR / rR)

    # Left and Right fluxes
    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (qL[2] + pL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (qR[2] + pR)])

    # Compute guess pressure from PVRS Riemann solver
    PPV = max(0.0, 0.5 * (pL + pR) + 0.5 * (uL - uR) * 0.25 * (rL + rR) * (aL + aR))
    pmin = min(pL, pR)
    pmax = max(pL, pR)
    Qmax = pmax / pmin if pmin != 0 else 1e6
    Quser = 2.0  # parameter manually set

    if (Qmax <= Quser) and (pmin <= PPV) and (PPV <= pmax):
        pM = PPV
    else:
        if PPV < pmin:
            PQ = (pL / pR) ** ((gamma - 1.0) / (2.0 * gamma))
            uM = (PQ * uL / aL + uR / aR + (PQ - 1.0) * 2 / (gamma - 1)) / (PQ / aL + 1.0 / aR)
            PTL = 1 + (gamma - 1) / 2.0 * (uL - uM) / aL
            PTR = 1 + (gamma - 1) / 2.0 * (uM - uR) / aR
            pM = 0.5 * (pL * PTL ** (2 * gamma / (gamma - 1)) + pR * PTR ** (2 * gamma / (gamma - 1)))
        else:
            GEL = np.sqrt((2 / ((gamma + 1) * rL)) / ((gamma - 1) / (gamma + 1) * pL + PPV))
            GER = np.sqrt((2 / ((gamma + 1) * rR)) / ((gamma - 1) / (gamma + 1) * pR + PPV))
            pM = (GEL * pL + GER * pR - (uR - uL)) / (GEL + GER)

    # Estimate wave speeds: SL, SR and SM (Toro, 1994)
    zL = np.sqrt(1 + (gamma + 1) / (2 * gamma) * (pM / pL - 1)) if pM > pL else 1.0
    zR = np.sqrt(1 + (gamma + 1) / (2 * gamma) * (pM / pR - 1)) if pM > pR else 1.0

    SL = uL - aL * zL
    SR = uR + aR * zR
    SM_numer = pL - pR + rR * uR * (SR - uR) - rL * uL * (SL - uL)
    SM_denom = rR * (SR - uR) - rL * (SL - uL)
    SM = SM_numer / SM_denom if SM_denom != 0 else 0.0

    # Compute the HLLC flux
    if 0 <= SL:
        return FL
    elif SL <= 0 <= SM:
        qsL = np.zeros(3)
        coef = rL * (SL - uL) / (SL - SM) if (SL - SM) != 0 else 0.0
        qsL[0] = coef
        qsL[1] = coef * SM
        qsL[2] = coef * (EL + (SM - uL) * (SM + pL / (rL * (SL - uL))))
        return FL + SL * (qsL - qL)
    elif SM <= 0 <= SR:
        qsR = np.zeros(3)
        coef = rR * (SR - uR) / (SR - SM) if (SR - SM) != 0 else 0.0
        qsR[0] = coef
        qsR[1] = coef * SM
        qsR[2] = coef * (ER + (SM - uR) * (SM + pR / (rR * (SR - uR))))
        return FR + SR * (qsR - qR)
    elif 0 >= SR:
        return FR
    else:
        return np.zeros(3)

def MUSCL_EulerRes1d(q, smax, gamma, dx, N, limiter, fluxMethod):
    """
    MUSCL Monotonic Upstream Centered Scheme for Conservation Laws.
    Van Leer's MUSCL reconstruction scheme using piecewise linear reconstruction.
    where: limiter='MC'; fluxMethod='AUSM';
    """

    # Allocate arrays
    res = np.zeros((3, N))
    dq = np.zeros((3, N))
    flux = np.zeros((3, N-1))
    qL = np.zeros((3, N-1))
    qR = np.zeros((3, N-1))

    # Compute and limit slopes
    for i in range(3):
        for j in range(1, N-1):  # Python is 0-based
            if limiter == 'MC':
                dqR = 2 * (q[i, j+1] - q[i, j])
                dqL = 2 * (q[i, j] - q[i, j-1])
                dqC = (q[i, j+1] - q[i, j-1]) / 2
                dq[i, j] = minmod([dqR, dqL, dqC])
            elif limiter == 'MM':
                dqR = (q[i, j+1] - q[i, j])
                dqL = (q[i, j] - q[i, j-1])
                dq[i, j] = minmod([dqR, dqL])
            elif limiter == 'VA':
                dqR = (q[i, j+1] - q[i, j])
                dqL = (q[i, j] - q[i, j-1])
                dq[i, j] = vanalbada(dqR, dqL, dx)
            else:
                dq[i, j] = 0.0

    # Left and Right extrapolated q-values at the boundary j+1/2
    for j in range(1, N-2):
        qL[:, j] = q[:, j] + dq[:, j] / 2
        qR[:, j] = q[:, j+1] - dq[:, j+1] / 2

    # Flux contribution to the residual of every cell
    for j in range(1, N-2):
        # compute flux at j+1/2
        if fluxMethod == 'LF':
            flux[:, j] = LFflux(qL[:, j], qR[:, j], gamma, smax)
        elif fluxMethod == 'ROE':
            flux[:, j] = ROEflux(qL[:, j], qR[:, j], gamma)
        elif fluxMethod == 'RUS':
            flux[:, j] = RUSflux(qL[:, j], qR[:, j], gamma)
        elif fluxMethod == 'HLLE':
            flux[:, j] = HLLEflux(qL[:, j], qR[:, j], gamma)
        elif fluxMethod == 'AUSM':
            flux[:, j] = AUSMflux(qL[:, j], qR[:, j], gamma)
        elif fluxMethod == 'HLLC':
            flux[:, j] = HLLCflux(qL[:, j], qR[:, j], gamma)
        else:
            raise ValueError("Unknown flux method: " + str(fluxMethod))
        res[:, j] += flux[:, j] / dx
        res[:, j+1] -= flux[:, j] / dx

    # Flux contribution of the LEFT MOST FACE: left face of cell j=1
    qR[:, 0] = q[:, 1] - dq[:, 1] * dx / 2
    qL[:, 0] = qR[:, 0]
    if fluxMethod == 'LF':
        flux[:, 0] = LFflux(qL[:, 0], qR[:, 0], gamma, smax)
    elif fluxMethod == 'ROE':
        flux[:, 0] = ROEflux(qL[:, 0], qR[:, 0], gamma)
    elif fluxMethod == 'RUS':
        flux[:, 0] = RUSflux(qL[:, 0], qR[:, 0], gamma)
    elif fluxMethod == 'HLLE':
        flux[:, 0] = HLLEflux(qL[:, 0], qR[:, 0], gamma)
    elif fluxMethod == 'AUSM':
        flux[:, 0] = AUSMflux(qL[:, 0], qR[:, 0], gamma)
    elif fluxMethod == 'HLLC':
        flux[:, 0] = HLLCflux(qL[:, 0], qR[:, 0], gamma)
    else:
        raise ValueError("Unknown flux method: " + str(fluxMethod))
    res[:, 1] -= flux[:, 0] / dx

    # Flux contribution of the RIGHT MOST FACE: right face of cell j=N-2
    qL[:, N-2] = q[:, N-2] + dq[:, N-2] * dx / 2
    qR[:, N-2] = qL[:, N-2]
    if fluxMethod == 'LF':
        flux[:, N-2] = LFflux(qL[:, N-2], qR[:, N-2], gamma, smax)
    elif fluxMethod == 'ROE':
        flux[:, N-2] = ROEflux(qL[:, N-2], qR[:, N-2], gamma)
    elif fluxMethod == 'RUS':
        flux[:, N-2] = RUSflux(qL[:, N-2], qR[:, N-2], gamma)
    elif fluxMethod == 'HLLE':
        flux[:, N-2] = HLLEflux(qL[:, N-2], qR[:, N-2], gamma)
    elif fluxMethod == 'AUSM':
        flux[:, N-2] = AUSMflux(qL[:, N-2], qR[:, N-2], gamma)
    elif fluxMethod == 'HLLC':
        flux[:, N-2] = HLLCflux(qL[:, N-2], qR[:, N-2], gamma)
    else:
        raise ValueError("Unknown flux method: " + str(fluxMethod))
    res[:, N-2] += flux[:, N-2] / dx

    return res