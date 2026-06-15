import numpy as np
from typing import NamedTuple

class EulerSolution1D(NamedTuple):
    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    E: np.ndarray

    # Packing
    # sol = EulerSolution(x, r[0,:], u[0,:], p[0,:], E[0,:])

    # Unpacking in plot — still one liner, but readable
    # axs[0,0].plot(sol.x, sol.rho, "-k", label="Exact")