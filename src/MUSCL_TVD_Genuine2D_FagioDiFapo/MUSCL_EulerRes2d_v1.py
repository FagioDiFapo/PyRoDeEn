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
from .utils import HLLE1Dflux, HLLE2Dflux

def MUSCL_EulerRes2d_v1(q, dt, dx, dy, N, M, limiter, fluxMethod):

    """
    A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
    Upstream Centered Scheme for Conservation Laws (MUSCL).
    """

    res = np.zeros((M, N, 4))

    # Normal unitary face vectors: (nx, ny)
    # normals = [[0,1], [1,0], [0,-1], [-1,0]]  # i.e.: [N, E, S, W]

    # Build cells
    cell = np.empty((M, N), dtype=object)
    for i in range(M):
        for j in range(N):
            cell[i, j] = {'q': np.array([q[i, j, 0], q[i, j, 1], q[i, j, 2], q[i, j, 3]]),
                          'res': np.zeros(4)}

    # Build Faces
    face = np.empty((M-1, N-1), dtype=object)
    for i in range(M-1):
        for j in range(N-1):
            face[i, j] = {'HLLE_x': np.zeros(4),
                          'HLLE_y': np.zeros(4),
                          'HLLE2x': np.zeros(4),
                          'HLLE2y': np.zeros(4),
                          'flux_x': np.zeros(4),
                          'flux_y': np.zeros(4)}

    # %%%%%%%%%%%%%
    # Residuals %
    # %%%%%%%%%%%%%

    # Compute fluxes across cells
    for i in range(1, M-2):     # all internal faces
        for j in range(1, N-2): # all internal faces
            qSW = cell[i, j]['q']
            qSE = cell[i, j+1]['q']
            qNW = cell[i+1, j]['q']
            # compute HLLE1d flux
            face[i, j]['HLLE_x'] = HLLE1Dflux(qSW, qSE, [1, 0])   # HLLE1d_{  i  ,j+1/2}
            face[i, j]['HLLE_y'] = HLLE1Dflux(qSW, qNW, [0, 1])   # HLLE1d_{i+1/2,  j  }


    # Compute fluxes at the corners of cells (the stagered grid)
    for i in range(1, M-2):     # all internal faces
        for j in range(1, N-2): # all internal faces
            qSW = cell[i, j]['q']
            qSE = cell[i, j+1]['q']
            qNW = cell[i+1, j]['q']
            qNE = cell[i+1, j+1]['q']
            # compute HLLE2d flux
            face[i, j]['HLLE2x'], face[i, j]['HLLE2y'] = HLLE2Dflux(qSW, qSE, qNW, qNE) # HLLE2d_{i+1/2,j+1/2}


    # Assembling fluxes for HLLE2d with Simpsons Rule
    if fluxMethod == 'HLLE2d':
        for i in range(1, M-1):     # internal nodes
            for j in range(1, N-1): # internal nodes
                face[i, j]['flux_x'] = (
                    face[i, j]['HLLE2x'] + 4 * face[i, j]['HLLE_x'] + face[i, j-1]['HLLE2x']
                ) / 6  # F_{i,j+1/2}
                face[i, j]['flux_y'] = (
                    face[i, j]['HLLE2y'] + 4 * face[i, j]['HLLE_y'] + face[i-1, j]['HLLE2y']
                ) / 6  # F_{i+1/2,j}


    # contributions to the residual of cell (i,j) and cells around it
    for i in range(1, M-2):     # internal faces
        for j in range(1, N-2): # internal faces
            cell[i, j]['res'] += face[i, j]['flux_x'] / dx
            cell[i, j+1]['res'] -= face[i, j]['flux_x'] / dx
            cell[i, j]['res'] += face[i, j]['flux_y'] / dy
            cell[i+1, j]['res'] -= face[i, j]['flux_y'] / dy


    # %%%%%%%%%%
    # set BCs %
    # %%%%%%%%%%

    # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
    for i in range(1, M-1):
        for j in range(1, N-1):
            res[i, j, :] = cell[i, j]['res']


    # Debug
    # Q=[cell(:,:).res]; Q=reshape(Q(1,:),M,N); surf(Q);
    return res