"""
Shared pytest fixtures and configuration.

This file is automatically discovered by pytest and makes fixtures available
to all test files in the tests/ directory.

Fixtures defined here can be used by any test without importing.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# Session-scoped fixtures (created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def reference_solutions_dir(test_data_dir):
    """Path to reference solutions for validation"""
    return test_data_dir / "reference_solutions"


# ============================================================================
# Module-scoped fixtures (created once per test module)
# ============================================================================

@pytest.fixture(scope="module")
def sod_tube_exact_solution():
    """
    Exact analytical solution for Sod shock tube.

    Module-scoped because computing exact solution can be expensive.
    """
    # In real implementation:
    # from rde_solver.benchmarks import SodTubeExact
    # return SodTubeExact().solution(t=0.2)

    # Placeholder
    return {
        'x': np.linspace(0, 1, 100),
        'density': np.ones(100),
        'velocity': np.zeros(100),
        'pressure': np.ones(100),
    }


# ============================================================================
# Function-scoped fixtures (created for each test function)
# ============================================================================

@pytest.fixture
def basic_1d_solver():
    """
    Provides a basic 1D MUSCL solver for testing.

    Function-scoped (default) - created fresh for each test.
    """
    # from rde_solver.core.solvers import MUSCL1D
    # return MUSCL1D(nx=100, cfl=0.5)
    return None  # Placeholder


@pytest.fixture
def basic_2d_solver():
    """Provides a basic 2D MUSCL solver for testing"""
    # from rde_solver.core.solvers import MUSCL2D
    # return MUSCL2D(nx=50, ny=50, cfl=0.5)
    return None  # Placeholder


@pytest.fixture
def sod_tube_initial_conditions() -> Dict[str, Dict[str, float]]:
    """
    Standard Sod tube initial conditions.

    Returns:
        Dict with 'left' and 'right' states containing density, pressure, velocity
    """
    return {
        'left': {
            'density': 1.0,
            'pressure': 1.0,
            'velocity': 0.0,
        },
        'right': {
            'density': 0.125,
            'pressure': 0.1,
            'velocity': 0.0,
        }
    }


@pytest.fixture
def shock_tube_grid():
    """Standard grid for shock tube problems"""
    return {
        'x': np.linspace(0, 1, 200),
        'dx': 1.0 / 199,
        'discontinuity': 0.5,  # Location of initial discontinuity
    }


# ============================================================================
# Parametrized fixtures
# ============================================================================

@pytest.fixture(params=["minmod", "superbee", "mc", "vanleer"])
def limiter_name(request):
    """
    Parametrized fixture for testing all limiters.

    Tests using this fixture will run once for each limiter.

    Usage:
        def test_limiter_tvd(limiter_name):
            # This test runs 4 times, once for each limiter
            assert limiter_is_tvd(limiter_name)
    """
    return request.param


@pytest.fixture(params=[50, 100, 200])
def grid_resolution(request):
    """Parametrized fixture for testing different grid resolutions"""
    return request.param


# ============================================================================
# Fixtures for chemistry tests
# ============================================================================

@pytest.fixture
def simple_chemistry_mechanism():
    """Simple chemistry mechanism for testing"""
    return {
        'species': ['H2', 'O2', 'H2O'],
        'reactions': [
            '2 H2 + O2 => 2 H2O'
        ],
        'temperature_range': (300, 3000),
    }


@pytest.fixture
def ethylene_oxygen_mechanism():
    """Full ethylene-oxygen mechanism for integration tests"""
    # In real implementation, would load actual mechanism file
    # from rde_solver.chemistry.mechanisms import load_mechanism
    # return load_mechanism("ethylene_oxygen")
    return None  # Placeholder


# ============================================================================
# Temporary directory fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Provides a temporary directory for test outputs.

    pytest's tmp_path fixture automatically cleans up after test.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# Custom markers configuration
# ============================================================================

def pytest_configure(config):
    """
    Register custom markers.

    This allows using markers like @pytest.mark.slow
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "chemistry: marks tests that require chemistry libraries (Cantera, etc.)"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU/CUDA"
    )


# ============================================================================
# Pytest hooks for custom behavior
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection behavior.

    Automatically adds markers to tests based on their location.
    """
    for item in items:
        # Add 'unit' marker to tests in tests/unit/
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add 'integration' marker to tests in tests/integration/
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add 'slow' marker to integration tests (they tend to be slower)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Fixtures for numerical comparison
# ============================================================================

@pytest.fixture
def assert_arrays_close():
    """
    Fixture providing a function for array comparison with nice error messages.

    Usage:
        def test_something(assert_arrays_close):
            result = compute_something()
            expected = load_expected()
            assert_arrays_close(result, expected, rtol=1e-6)
    """
    def _assert_close(actual, expected, rtol=1e-7, atol=1e-10, name="arrays"):
        """Compare arrays with descriptive error message"""
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} do not match within tolerance"
        )

        # Additional diagnostics if they don't match
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(actual - expected))
            rel_diff = np.max(np.abs((actual - expected) / (expected + 1e-10)))
            print(f"\nMax absolute difference: {max_diff}")
            print(f"Max relative difference: {rel_diff}")

    return _assert_close


# ============================================================================
# Fixtures for performance testing
# ============================================================================

@pytest.fixture
def performance_timer():
    """
    Simple timer fixture for performance assertions.

    Usage:
        def test_solver_fast(performance_timer):
            with performance_timer(max_time=1.0):  # Must complete in 1 second
                solver.solve()
    """
    from contextlib import contextmanager
    import time

    @contextmanager
    def _timer(max_time=None):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        if max_time is not None:
            assert elapsed < max_time, \
                f"Operation took {elapsed:.4f}s, exceeds limit of {max_time}s"

    return _timer


# ============================================================================
# Example: Fixtures with cleanup
# ============================================================================

@pytest.fixture
def solver_with_cleanup():
    """
    Example fixture with explicit cleanup.

    Use yield for fixtures that need cleanup (setup/teardown pattern).
    """
    # Setup
    # solver = MUSCL1D(nx=100)
    solver = None  # Placeholder

    # Provide to test
    yield solver

    # Cleanup (runs after test completes)
    # solver.cleanup()
    # Close files, free memory, etc.
    pass


# ============================================================================
# Fixtures for skip conditions
# ============================================================================

@pytest.fixture
def skip_if_no_cantera():
    """Skip test if Cantera is not installed"""
    pytest.importorskip("cantera", reason="Cantera not installed")


@pytest.fixture
def skip_if_no_jax():
    """Skip test if JAX is not installed"""
    pytest.importorskip("jax", reason="JAX not installed")


# ============================================================================
# Documentation
# ============================================================================

"""
Using these fixtures in tests:
------------------------------

1. Function-scoped (default):
   ```python
   def test_something(basic_1d_solver):
       # basic_1d_solver is created fresh for this test
       result = basic_1d_solver.solve()
   ```

2. Parametrized fixtures:
   ```python
   def test_all_limiters(limiter_name):
       # Test runs once for each limiter
       solver = MUSCL1D(limiter=limiter_name)
   ```

3. Multiple fixtures:
   ```python
   def test_complex(basic_1d_solver, sod_tube_initial_conditions, temp_output_dir):
       # Use multiple fixtures
       result = basic_1d_solver.solve(sod_tube_initial_conditions)
       result.save(temp_output_dir / "output.h5")
   ```

4. Fixture dependencies:
   ```python
   @pytest.fixture
   def configured_solver(basic_1d_solver, sod_tube_initial_conditions):
       # Fixtures can depend on other fixtures
       basic_1d_solver.set_initial_conditions(sod_tube_initial_conditions)
       return basic_1d_solver
   ```

Running tests with markers:
---------------------------
pytest                          # Run all tests
pytest -m unit                  # Run only unit tests
pytest -m "not slow"            # Skip slow tests
pytest -m "unit and not slow"   # Unit tests that aren't slow
pytest tests/unit/              # Run all tests in unit directory
"""
