# Python Project Conventions Guide

This guide explains the standard conventions for modern Python projects, specifically addressing your questions about `pyproject.toml`, testing, and project organization.

## Table of Contents
1. [pyproject.toml Purpose & Standards](#pyprojecttoml-purpose--standards)
2. [What Goes in project.scripts vs Not](#what-goes-in-projectscripts-vs-not)
3. [Testing Strategy](#testing-strategy)
4. [Benchmarking Strategy](#benchmarking-strategy)
5. [Complete Workflow Example](#complete-workflow-example)

---

## pyproject.toml Purpose & Standards

### Official Standards
- **PEP 621**: Project metadata standard
- **PEP 517/518**: Build system specification
- **Source**: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

### What pyproject.toml SHOULD Contain:

```toml
[project]
# Package metadata
name = "your-package"
version = "0.1.0"
dependencies = ["numpy", "scipy"]  # Runtime dependencies

[project.optional-dependencies]
dev = ["pytest", "black"]  # Development tools
docs = ["sphinx"]          # Documentation tools

[project.scripts]
# User-facing command-line tools ONLY
myapp = "package.module:main"

[tool.pytest.ini_options]
# Test configuration (pytest reads this)

[tool.black]
# Formatter configuration

[tool.mypy]
# Type checker configuration
```

### What pyproject.toml SHOULD NOT Contain:
- ❌ Test runner commands (use `pytest` directly)
- ❌ Build scripts (handled by build system)
- ❌ Development workflow scripts (use `Makefile` or `scripts/`)
- ❌ Benchmark runners (separate tools)

---

## What Goes in project.scripts vs Not

### ✅ DO Put in `[project.scripts]`:

**User-facing commands that will be used after installation:**

```toml
[project.scripts]
# Main application entry points
rde-solve = "rde_solver.cli:solve"          # Main solver
rde-visualize = "rde_solver.cli:visualize"  # Visualization tool
rde-validate = "rde_solver.cli:validate"    # Quick validation

# Optional: Convenience commands for users
rde-example = "rde_solver.examples:run_basic_example"
```

**When installed, users can run:**
```bash
$ rde-solve --config my_config.yaml
$ rde-visualize results/output.h5
```

### ❌ DON'T Put in `[project.scripts]`:

**Development/testing commands:**
```toml
# ❌ WRONG - Don't do this
[project.scripts]
run-tests = "pytest:main"           # Just use: pytest
run-benchmarks = "benchmarks:main"  # Use custom script
format-code = "black:main"          # Just use: black src/
```

**Why?** These are development tools with their own interfaces. Users shouldn't need them.

---

## Testing Strategy

### Standard Practice: pytest + pyproject.toml Configuration

#### 1. **Test Structure:**
```
tests/
├── conftest.py                 # Shared fixtures
├── unit/                       # Fast, isolated tests
│   ├── test_solvers.py
│   ├── test_chemistry.py
│   └── test_numerics.py
├── integration/                # Multi-component tests
│   ├── test_sod_tube.py
│   └── test_chemistry_integration.py
└── data/                       # Test data files
    └── reference_solutions.csv
```

#### 2. **Configuration in pyproject.toml:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]           # Where to find tests
addopts = ["-v", "--cov=src"]  # Options always applied

# Define custom markers
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests",
    "unit: unit tests",
]
```

#### 3. **Running Tests:**
```bash
# Standard pytest commands (NO entry point needed)
pytest                          # Run all tests
pytest tests/unit              # Run unit tests only
pytest -k "test_solver"        # Run tests matching pattern
pytest -m "not slow"           # Skip slow tests
pytest --cov=src --cov-report=html  # With coverage

# In CI/CD
pytest --cov=src --cov-report=xml
```

#### 4. **Example Test File:**
```python
# tests/unit/test_solvers.py
import pytest
from rde_solver.core.solvers import MUSCL1D

class TestMUSCL1D:
    """Unit tests for 1D MUSCL solver"""

    def test_initialization(self):
        """Test solver can be initialized"""
        solver = MUSCL1D(nx=100, cfl=0.5)
        assert solver.nx == 100
        assert solver.cfl == 0.5

    @pytest.mark.parametrize("limiter", ["minmod", "superbee", "mc"])
    def test_limiter_selection(self, limiter):
        """Test different limiters can be selected"""
        solver = MUSCL1D(limiter=limiter)
        assert solver.limiter == limiter

    def test_sod_tube_basic(self):
        """Test Sod tube produces physically reasonable results"""
        solver = MUSCL1D()
        result = solver.solve_sod_tube(t_end=0.2)

        # Physical constraints
        assert (result.density > 0).all()
        assert (result.pressure > 0).all()
        assert (result.velocity < 1.0).all()  # Subsonic/sonic
```

#### 5. **Shared Fixtures (conftest.py):**
```python
# tests/conftest.py
import pytest
from rde_solver.core.solvers import MUSCL1D

@pytest.fixture
def basic_solver():
    """Provides a basic MUSCL1D solver for testing"""
    return MUSCL1D(nx=100, cfl=0.5)

@pytest.fixture
def sod_tube_initial_conditions():
    """Standard Sod tube initial conditions"""
    return {
        'left': {'rho': 1.0, 'p': 1.0, 'u': 0.0},
        'right': {'rho': 0.125, 'p': 0.1, 'u': 0.0}
    }
```

---

## Benchmarking Strategy

### Two Approaches:

### Approach 1: pytest-benchmark (Integrated with pytest)

**Installation:**
```toml
[project.optional-dependencies]
dev = ["pytest-benchmark>=4.0"]
```

**Benchmark Tests:**
```python
# tests/benchmarks/test_performance.py
import pytest
from rde_solver.chemistry import JaxChemistry, CanteraChemistry

def test_jax_chemistry_performance(benchmark):
    """Benchmark JAX chemistry integration speed"""
    chemistry = JaxChemistry("ethylene_oxygen")
    initial_state = chemistry.get_initial_state()

    result = benchmark(chemistry.integrate, initial_state, dt=1e-6, steps=1000)

    # Optional: Assert performance requirements
    assert benchmark.stats['mean'] < 0.1  # Must complete in < 0.1s

@pytest.mark.parametrize("n_cells", [100, 1000, 10000])
def test_solver_scaling(benchmark, n_cells):
    """Test how solver scales with problem size"""
    from rde_solver.core.solvers import MUSCL1D

    solver = MUSCL1D(nx=n_cells)
    result = benchmark(solver.solve_sod_tube, t_end=0.2)
```

**Running:**
```bash
pytest tests/benchmarks/                    # Run benchmarks
pytest tests/benchmarks/ --benchmark-only   # Skip regular tests
pytest tests/benchmarks/ --benchmark-compare  # Compare with previous runs
```

### Approach 2: Standalone Benchmark Scripts

**For complex comparisons:**
```python
# benchmarks/benchmark_chemistry.py
"""
Comprehensive chemistry performance comparison.

Run with: python benchmarks/benchmark_chemistry.py
"""
import time
import numpy as np
from rde_solver.chemistry import JaxChemistry, CanteraChemistry
import matplotlib.pyplot as plt

def benchmark_integration_speed():
    """Compare JAX vs Cantera integration speed"""
    n_cells = [10, 100, 1000, 10000]
    jax_times = []
    cantera_times = []

    for n in n_cells:
        # JAX
        jax_chem = JaxChemistry("ethylene_oxygen")
        start = time.perf_counter()
        jax_chem.integrate_grid(n_cells=n, dt=1e-6, steps=100)
        jax_times.append(time.perf_counter() - start)

        # Cantera
        cantera_chem = CanteraChemistry("ethylene_oxygen")
        start = time.perf_counter()
        cantera_chem.integrate_grid(n_cells=n, dt=1e-6, steps=100)
        cantera_times.append(time.perf_counter() - start)

    # Plot results
    plt.loglog(n_cells, jax_times, label='JAX')
    plt.loglog(n_cells, cantera_times, label='Cantera')
    plt.xlabel('Grid Cells')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig('benchmarks/results/chemistry_scaling.png')

    # Print summary
    print(f"JAX speedup: {np.array(cantera_times) / np.array(jax_times)}")

if __name__ == "__main__":
    benchmark_integration_speed()
```

**Running:**
```bash
python benchmarks/benchmark_chemistry.py
```

**Optional: Add convenience command:**
```toml
[project.scripts]
rde-benchmark = "benchmarks.benchmark_chemistry:main"
```

---

## Complete Workflow Example

### Project Structure:
```
PyRoDeEn/
├── pyproject.toml              # Metadata + tool configs
├── src/
│   └── rde_solver/
│       ├── __init__.py
│       ├── cli.py              # Entry points for [project.scripts]
│       └── core/
│           └── solvers.py
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/
│   │   └── test_solvers.py
│   └── integration/
│       └── test_sod_tube.py
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_chemistry.py
├── scripts/
│   └── validate_installation.py
└── examples/
    └── basic_usage.py
```

### Developer Workflow:

```bash
# 1. Clone and install in development mode
git clone https://github.com/FagioDiFapo/ApproximateRiemannSolversPy.git
cd ApproximateRiemannSolversPy
pip install -e ".[dev]"  # Installs with dev dependencies

# 2. Run tests (NO pyproject.toml entry needed)
pytest                          # All tests
pytest -m "not slow"           # Fast tests only
pytest --cov                   # With coverage

# 3. Run benchmarks
python benchmarks/benchmark_chemistry.py  # Standalone script
pytest tests/benchmarks/                  # Or via pytest-benchmark

# 4. Format code
black src/ tests/              # Format code
ruff check src/                # Lint code

# 5. Type check
mypy src/

# 6. Use installed commands (from [project.scripts])
rde-solve --help               # Your CLI tool
rde-validate                   # Validation command
```

### User Workflow (After Installing Package):

```bash
# 1. Install from PyPI (when you publish)
pip install rde-solver

# 2. Use CLI commands (NO need to clone repo)
rde-solve my_config.yaml
rde-visualize results.h5

# 3. Or use as library
python my_script.py  # Imports: from rde_solver import MUSCL1D
```

---

## Summary: When to Use What

| Purpose | Tool | Where | How to Run |
|---------|------|-------|------------|
| **User CLI commands** | `[project.scripts]` | `pyproject.toml` | `rde-solve` (after install) |
| **Unit/Integration tests** | `pytest` | `tests/` | `pytest` |
| **Test configuration** | `[tool.pytest.ini_options]` | `pyproject.toml` | Auto-read by pytest |
| **Performance benchmarks** | Custom scripts or `pytest-benchmark` | `benchmarks/` | `python benchmarks/...` or `pytest benchmarks/` |
| **Code formatting** | `black`, `ruff` | N/A | `black src/` |
| **Type checking** | `mypy` | N/A | `mypy src/` |
| **Documentation** | `sphinx` | `docs/` | `sphinx-build docs/ docs/_build` |
| **Example usage** | Python scripts | `examples/` | `python examples/...` |
| **Dev utilities** | Python scripts | `scripts/` | `python scripts/...` |

---

## Key Takeaways

1. ✅ **Your current `[project.scripts]` usage is CORRECT** - These are for user-facing CLI commands
2. ✅ **Tests don't go in `[project.scripts]`** - Use pytest directly with `[tool.pytest.ini_options]` config
3. ✅ **Benchmarks can be standalone scripts** - Or use pytest-benchmark for integration
4. ✅ **Development tools (black, ruff, mypy) don't need entry points** - They have their own CLIs
5. ✅ **Optional dependencies group related tools** - `pip install package[dev]` for all dev tools

## Further Reading

- **PEP 621 (Project Metadata)**: https://peps.python.org/pep-0621/
- **Python Packaging Guide**: https://packaging.python.org/en/latest/
- **pytest Documentation**: https://docs.pytest.org/
- **pytest-benchmark**: https://pytest-benchmark.readthedocs.io/
- **Python Package Structure**: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
