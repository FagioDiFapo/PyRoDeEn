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
    Vectorized version: uses only numpy arrays for all states and residuals.
    """

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
    for j in range(1, N - 2):
        qR = qS[M - 2, j, :]
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [0, 1])
        residual[M - 2, j, :] += flux / dy

    # East face (j = N-2)
    for i in range(1, M - 2):
        qR = qW[i, N - 2, :]
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [1, 0])
        residual[i, N - 2, :] += flux / dx

    # South face (i = 1)
    for j in range(1, N - 2):
        qR = qN[1, j, :]
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [0, -1])
        residual[1, j, :] += flux / dy

    # West face (j = 1)
    for i in range(1, M - 2):
        qR = qE[i, 1, :]
        qL = qR
        if fluxMethod == 'HLLE1d':
            flux = HLLE1Dflux(qL, qR, [-1, 0])
        residual[i, 1, :] += flux / dx

    # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
    res = np.zeros_like(residual)
    res[1:M-1, 1:N-1, :] = residual[1:M-1, 1:N-1, :]

    return res