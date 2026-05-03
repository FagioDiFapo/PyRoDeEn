import pytest
import numpy as np
import pyrodeen.grid

@pytest.mark.parametrize("size",[3, 6, 10],)
def test_grid_instantiation(size):
    grid = pyrodeen.grid.Grid(size, size, .1, .1, 1)
    assert(grid.nx == size)
    assert(grid.ny == size)
    assert(np.size(grid.values) == np.pow(size,2))
    assert(np.size(grid.values_gh) == np.pow(size+2, 2))