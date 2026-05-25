import sys
import pytest
import numpy as np
from pyrodeen import cfd
from pyrodeen.grid import Grid


# Boundary condition tests
def test_transmissive_conditions():
    grid = Grid(3, 2, 0.1, 0.1, 4)
    grid.values[:, :, :] = 1.0
    boundary_config = cfd.BoundaryConfig.all_transmissive()
    gamma = 1.4

    cfd_solver = cfd.Solver(grid, boundary_config, gamma)
    cfd_solver.apply_boundary_conditions()

    assert np.allclose(grid.values_gh[0, 1:-1, :], grid.values[0, :, :])  # Left
    assert np.allclose(grid.values_gh[-1, 1:-1, :], grid.values[-1, :, :])  # Right
    assert np.allclose(grid.values_gh[1:-1, 0, :], grid.values[:, 0, :])  # Bottom
    assert np.allclose(grid.values_gh[1:-1, -1, :], grid.values[:, -1, :])  # Top


@pytest.mark.parametrize(
    "euler_values",
    [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
)
def test_reflective_conditions(euler_values):
    grid = Grid(3, 2, 0.1, 0.1, 4)
    grid.values[:, :] = euler_values
    boundary_config = cfd.BoundaryConfig.closed_box()
    gamma = 1.4

    cfd_solver = cfd.Solver(grid, boundary_config, gamma)
    cfd_solver.apply_boundary_conditions()

    assert np.allclose(
        grid.values_gh[0, 1:-1], euler_values * np.array([1, 1, -1, 1])
    )  # Left
    assert np.allclose(
        grid.values_gh[-1, 1:-1], euler_values * np.array([1, 1, -1, 1])
    )  # Right
    assert np.allclose(
        grid.values_gh[1:-1, 0], euler_values * np.array([1, -1, 1, 1])
    )  # Bottom
    assert np.allclose(
        grid.values_gh[1:-1, -1], euler_values * np.array([1, -1, 1, 1])
    )  # Top


# Solver tests
def test_muscl_euler_res2d():
    grid = Grid(3, 2, 0.1, 0.1, 4)
    grid.values[:, :, :] = 1.0
    boundary_config = cfd.BoundaryConfig.closed_box()
    gamma = 1.4

    cfd_solver = cfd.Solver(grid, boundary_config, gamma)
    res = cfd_solver.muscl_euler_res2d()

    assert res.shape == (2, 3, 4)
    assert np.allclose(res[1:-1, 1:-1, :], 0.0)