"""
A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
Upstream Centered Scheme for Conservation Laws (MUSCL).

    e.g. where: limiter='MC'; fluxMethod='HLLE1d';

    Flux at j+1/2

      j+1/2         Cell's grid:
    | wL|   |
    |  /|wR |           1   2   3   4        N-2 N-1  N
    | / |\  |   {x=0} |-o-|-o-|-o-|-o-| ... |-o-|-o-|-o-| {x=L}
    |/  | \ |             1   2   3   4        N-2 N-1
    |   |  \|
    |   |   |       NC: Here cells 1 and N are ghost cells
      j  j+1            faces 1 and N-1, are the real boundary faces.

    q = np.stack([r, r*u, r*v, r*E], axis=2)
    F = np.stack([r*u, r*u**2+p, r*u*v, u*(r*E+p)], axis=2)
    G = np.stack([r*v, r*u*v, r*v**2+p, v*(r*E+p)], axis=2)

Original code written by Manuel Diaz, NTU, 05.25.2015.
"""

import numpy as np
from .utils import minmod, vanalbada, vanLeer, HLLE1Dflux

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