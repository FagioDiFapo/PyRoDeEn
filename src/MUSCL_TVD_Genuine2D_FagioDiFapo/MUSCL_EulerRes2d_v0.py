# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#   A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
#   Upstream Centered Scheme for Conservation Laws (MUSCL).
#
#   e.g. where: limiter='MC'; fluxMethod='HLLE1d';
#
#   Flux at j+1/2
#
#     j+1/2         Cell's grid:
#   | wL|   |
#   |  /|wR |           1   2   3   4        N-2 N-1  N
#   | / |\  |   {x=0} |-o-|-o-|-o-|-o-| ... |-o-|-o-|-o-| {x=L}
#   |/  | \ |             1   2   3   4        N-2 N-1
#   |   |  \|
#   |   |   |       NC: Here cells 1 and N are ghost cells
#     j  j+1            faces 1 and N-1, are the real boundary faces.
#
#   q = np.stack([r, r*u, r*v, r*E], axis=2)
#   F = np.stack([r*u, r*u**2+p, r*u*v, u*(r*E+p)], axis=2)
#   G = np.stack([r*v, r*u*v, r*v**2+p, v*(r*E+p)], axis=2)
#
# Written by Manuel Diaz, NTU, 05.25.2015.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

def MUSCL_EulerRes2d_v0(q, _, dx, dy, N, M, limiter, fluxMethod):
    """
    A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
    Upstream Centered Scheme for Conservation Laws (MUSCL).
    """
    res = np.zeros((M, N, 4))

    # Build cells as a 2D list of dicts for each cell's state and slopes
    cell = [[{
        'q': q[i, j, :].copy(),
        'qN': np.zeros(4),
        'qS': np.zeros(4),
        'qE': np.zeros(4),
        'qW': np.zeros(4),
        'res': np.zeros(4)
    } for j in range(N)] for i in range(M)]

    # Compute and limit slopes at cells (i,j)
    for i in range(1, M-1):       # only internal cells
        for j in range(1, N-1):   # only internal cells
            for k in range(4):
                dqw = cell[i][j]['q'][k] - cell[i][j-1]['q'][k]
                dqe = cell[i][j+1]['q'][k] - cell[i][j]['q'][k]
                dqs = cell[i][j]['q'][k] - cell[i-1][j]['q'][k]
                dqn = cell[i+1][j]['q'][k] - cell[i][j]['q'][k]
                if limiter == 'MC':
                    dqc = (cell[i][j+1]['q'][k] - cell[i][j-1]['q'][k]) / 2
                    dqdx = minmod([2*dqw, 2*dqe, dqc])
                    dqc = (cell[i+1][j]['q'][k] - cell[i-1][j]['q'][k]) / 2
                    dqdy = minmod([2*dqs, 2*dqn, dqc])
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
                cell[i][j]['qE'][k] = cell[i][j]['q'][k] + dqdx / 2
                cell[i][j]['qW'][k] = cell[i][j]['q'][k] - dqdx / 2
                cell[i][j]['qN'][k] = cell[i][j]['q'][k] + dqdy / 2
                cell[i][j]['qS'][k] = cell[i][j]['q'][k] - dqdy / 2

    # Residuals: x-direction
    for i in range(1, M-1):     # internal cells
        for j in range(1, N-2): # internal faces
            qxL = cell[i][j]['qE']  # q_{i,j+1/2}^{-}
            qxR = cell[i][j+1]['qW']  # q_{i,j+1/2}^{+}
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qxL, qxR, [1, 0])  # F_{i,j+1/2}
            else:
                raise ValueError("flux option not available")
            cell[i][j]['res'] += flux / dx
            cell[i][j+1]['res'] -= flux / dx

    # Residuals: y-direction
    for i in range(1, M-2):     # internal faces
        for j in range(1, N-1): # internal cells
            qyL = cell[i][j]['qN']  # q_{i+1/2,j}^{-}
            qyR = cell[i+1][j]['qS']  # q_{i+1/2,j}^{+}
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qyL, qyR, [0, 1])  # F_{i+1/2,j}
            else:
                raise ValueError("flux option not available")
            cell[i][j]['res'] += flux / dy
            cell[i+1][j]['res'] -= flux / dy

    # Set BCs: boundary flux contributions
    # North face (i = M-1)
    for j in range(1, N-1):
        qR = cell[M-1][j]['qS']
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [0, 1])
        cell[M-1][j]['res'] += flux / dy

    # East face (j = N-1)
    for i in range(1, M-1):
        qR = cell[i][N-1]['qW']
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [1, 0])
        cell[i][N-1]['res'] += flux / dx

    # South face (i = 2)
    for j in range(1, N-1):
        qR = cell[2][j]['qN']
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [0, -1])
        cell[2][j]['res'] += flux / dy

    # West face (j = 2)
    for i in range(1, M-1):
        qR = cell[i][2]['qE']
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [-1, 0])
        cell[i][2]['res'] += flux / dx

    # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
    for i in range(1, M-1):
        for j in range(1, N-1):
            res[i][j][:] = cell[i][j]['res']

    return res

#######################
# Auxiliary Functions #
#######################

def minmod(v):
    # Using Harten's generalized definition
    v = np.array(v)
    s = np.sum(np.sign(v)) / len(v)
    if abs(s) == 1:
        return s * np.min(np.abs(v))
    else:
        return 0.0

def vanalbada(da, db, h):
    # Van Albada Slope Limiter Function
    eps2 = (0.3 * h) ** 3
    numerator = (db ** 2 + eps2) * da + (da ** 2 + eps2) * db
    denominator = da ** 2 + db ** 2 + 2 * eps2
    if denominator != 0:
        return 0.5 * (np.sign(da) * np.sign(db) + 1) * numerator / denominator
    else:
        return 0.0

def vanLeer(da, db):
    # Van Leer Slope Limiter Function
    if db != 0:
        r = da / db
        return (r + abs(r)) / (1 + abs(r))
    else:
        return 0.0

def HLLE1Dflux(qL, qR, normal, gamma=1.4):
    # Compute HLLE flux for Euler equations in 2D
    nx, ny = normal

    # Left state
    rL = qL[0]
    uL = qL[1] / rL
    vL = qL[2] / rL
    vnL = uL * nx + vL * ny
    pL = (gamma - 1) * (qL[3] - rL * (uL ** 2 + vL ** 2) / 2)
    aL = np.sqrt(gamma * pL / rL)
    HL = (qL[3] + pL) / rL

    # Right state
    rR = qR[0]
    uR = qR[1] / rR
    vR = qR[2] / rR
    vnR = uR * nx + vR * ny
    pR = (gamma - 1) * (qR[3] - rR * (uR ** 2 + vR ** 2) / 2)
    aR = np.sqrt(gamma * pR / rR)
    HR = (qR[3] + pR) / rR

    # Roe averages
    RT = np.sqrt(rR / rL)
    u = (uL + RT * uR) / (1 + RT)
    v = (vL + RT * vR) / (1 + RT)
    H = (HL + RT * HR) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - (u ** 2 + v ** 2) / 2))
    vn = u * nx + v * ny

    # Wave speed estimates
    SLm = min(vnL - aL, vn - a, 0)
    SRp = max(vnR + aR, vn + a, 0)

    # Left and Right fluxes
    FL = np.array([rL * vnL,
                   rL * vnL * uL + pL * nx,
                   rL * vnL * vL + pL * ny,
                   rL * vnL * HL])
    FR = np.array([rR * vnR,
                   rR * vnR * uR + pR * nx,
                   rR * vnR * vR + pR * ny,
                   rR * vnR * HR])

    # HLLE flux
    if SRp - SLm != 0:
        HLLE = (SRp * FL - SLm * FR + SLm * SRp * (qR - qL)) / (SRp - SLm)
    else:
        HLLE = np.zeros_like(qL)
    return HLLE
