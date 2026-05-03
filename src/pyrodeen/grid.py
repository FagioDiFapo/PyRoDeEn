import numpy as np

class Grid:
    """
    Data structure for holding the state of a reduced (2D) reactive flow system.

    Uses ghost cells for boundary conditions (grid is internally nx+2*ny+2).
    """

    def __init__(self, nx: int, ny: int, lx: float, ly: float, nv: int):
        """
        Args:
            nx, ny: Grid resolution (cells, excluding ghost cells)
            lx, ly: Physical domain size
            nv: Number of state variables per cell
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.nv = nv
        self._grid = np.zeros([nx+2, ny+2, nv])

    @property
    def values(self) -> np.ndarray:
        return self._grid[1:-1, 1:-1, :]

    @property
    def values_gh(self) -> np.ndarray:
        return self._grid