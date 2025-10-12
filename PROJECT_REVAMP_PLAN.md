# Project Revamp Plan: Approximate Riemann Solvers for RDE Simulation

## Current State Analysis

### Project Structure Overview
Your project currently contains:

**Core Implementation (`src/`)**:
- `rde_solver/`: Main 2D CFD solvers (the heart of your project)
- `MUSCL_TVD_FagioDiFapo/`: 1D solvers
- Mixed Python/MATLAB heritage with inconsistent naming

**Root Directory Issues**:
- Large monolithic files (`arrhenius.py` - 3309 lines!)
- Scattered test/benchmark files
- Mixed outputs and source code
- Legacy MATLAB directories alongside Python code
- Documentation mixed with code

**Key Files Identified**:
- `arrhenius.py`: JAX-based chemistry implementation (MASSIVE - needs splitting)
- `benchmark_chemistry.py`, `benchmark_stiff.py`: Performance testing
- `cantera_test.py`: Validation tests
- Various output files (`.png`, `.txt`, `.csv`) scattered throughout

## Major Issues to Address

### 1. **Code Organization**
- Monolithic files (especially `arrhenius.py`)
- Inconsistent naming conventions
- Mixed responsibilities within modules
- Legacy MATLAB code mixed with Python

### 2. **Package Structure**
- Non-standard Python package layout
- Missing proper module separation
- No clear API boundaries
- Inconsistent import paths

### 3. **Documentation & Testing**
- Minimal docstrings
- No type hints
- Scattered test files with no framework
- No proper documentation structure

### 4. **Development Practices**
- No CI/CD setup
- Mixed development artifacts with source
- No proper dependency management for chemistry libraries

## Proposed New Structure (Following Python Packaging Conventions)

```
PyRoDeEn/                          # Project root
├── pyproject.toml                 # Modern Python packaging (PEP 621)
├── README.md                      # Project documentation
├── LICENSE
├── .gitignore
├── MANIFEST.in                    # Include data files in package
│
├── src/                           # Source layout (PEP 517/518)
│   └── rde_solver/
│       ├── __init__.py
│       ├── core/                  # Core CFD algorithms
│       │   ├── __init__.py
│       │   ├── solvers/
│       │   │   ├── __init__.py
│       │   │   ├── muscl_1d.py
│       │   │   ├── muscl_2d.py
│       │   │   └── riemann.py
│       │   ├── numerics/
│       │   │   ├── __init__.py
│       │   │   ├── limiters.py
│       │   │   ├── fluxes.py
│       │   │   └── reconstruction.py
│       │   └── grid/
│       │       ├── __init__.py
│       │       ├── mesh.py
│       │       └── boundary_conditions.py
│       │
│       ├── chemistry/             # Chemical kinetics
│       │   ├── __init__.py
│       │   ├── jax_chemistry.py   # Split from arrhenius.py
│       │   ├── cantera_interface.py
│       │   ├── mechanisms/
│       │   │   ├── __init__.py
│       │   │   └── ethylene_oxygen.py
│       │   └── thermodynamics/
│       │       ├── __init__.py
│       │       ├── nasa_polynomials.py
│       │       └── properties.py
│       │
│       ├── rde/                   # RDE-specific implementations
│       │   ├── __init__.py
│       │   ├── geometry.py
│       │   ├── detonation.py
│       │   └── rotating_wave.py
│       │
│       ├── visualization/         # Plotting and analysis
│       │   ├── __init__.py
│       │   ├── plots.py
│       │   └── analysis.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── initial_conditions.py
│           ├── file_io.py
│           └── constants.py
│
├── tests/                         # Unit/Integration tests (pytest)
│   ├── __init__.py
│   ├── conftest.py               # pytest configuration & fixtures
│   ├── unit/                     # Unit tests for individual functions
│   │   ├── test_solvers.py
│   │   ├── test_chemistry.py
│   │   └── test_numerics.py
│   ├── integration/              # Integration tests for workflows
│   │   ├── test_sod_tube.py
│   │   └── test_chemistry_integration.py
│   └── data/                     # Test data files
│       ├── sod_tube_exact.csv
│       └── chemistry_reference.json
│
├── benchmarks/                    # Performance benchmarks (separate from tests)
│   ├── __init__.py
│   ├── conftest.py
│   ├── benchmark_chemistry.py    # Performance comparison scripts
│   ├── benchmark_solvers.py
│   └── results/                  # Benchmark output storage
│
├── examples/                      # Usage examples and tutorials
│   ├── basic_sod_tube.py
│   ├── chemistry_validation.py
│   ├── rde_simulation.py
│   └── notebooks/
│       ├── getting_started.ipynb
│       └── advanced_rde.ipynb
│
├── docs/                          # Documentation (Sphinx)
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   ├── build/
│   └── requirements.txt          # Documentation dependencies
│
├── scripts/                       # Development/maintenance scripts
│   ├── setup_dev_env.py         # Development environment setup
│   ├── run_all_benchmarks.py    # Benchmark runner
│   └── validate_installation.py  # Installation validator
│
├── matlab_reference/              # Legacy MATLAB code (for validation & reference)
│   ├── README.md                # Explanation of MATLAB code organization
│   ├── original/                # Original unmodified MATLAB code
│   │   ├── MUSCL_Euler2d.m
│   │   ├── MUSCL_EulerRes2d_v0.m
│   │   └── utils/
│   │       └── Euler_IC2d.m
│   ├── modified/                # Modified MATLAB for validation outputs
│   │   ├── MUSCL_Euler2d_validation.m
│   │   └── generate_reference_data.m
│   └── validation_data/         # MATLAB-generated reference results
│       ├── sod_tube_matlab.csv
│       └── pressure_matlab_iter_*.txt
│
└── data/                        # Static data files (packaged with distribution)
    ├── mechanisms/              # Chemical mechanism files
    │   └── ethylene_oxygen.yaml
    └── reference_solutions/     # Analytical solutions for validation
        └── sod_tube_exact.dat
```

## Key Conventions Explained

### **1. `tests/` vs `benchmarks/` vs `scripts/`**

**`tests/` (Root level)**:
- **Purpose**: Unit and integration tests using pytest
- **Content**: Assert correctness of functionality
- **Run with**: `pytest tests/`
- **Examples**: `assert solver.compute_flux() == expected_flux`

**`benchmarks/` (Root level)**:
- **Purpose**: Performance measurement and comparison
- **Content**: Time/memory usage measurement, not correctness
- **Run with**: Custom benchmark runners or `pytest-benchmark`
- **Examples**: Your current `benchmark_chemistry.py` fits here

**`scripts/` (Root level)**:
- **Purpose**: Development utilities and maintenance tools
- **Content**: Setup scripts, data generators, validators
- **Run manually**: For development/maintenance tasks
- **Examples**: Environment setup, data preprocessing

### **2. `docs/` Location (Root level)**
- **Standard**: Documentation at root level (not in `src/`)
- **Tools**: Sphinx, MkDocs, etc.
- **Content**: API docs, tutorials, mathematical background
- **Build**: `docs/build/` for generated docs

### **3. `src/` Layout**
- **Standard**: PEP 517/518 source layout
- **Benefit**: Prevents import issues during development
- **Install**: `pip install -e .` works correctly

### **4. `examples/` vs `tests/`**
- **`examples/`**: Show how to USE the package (educational/demo)
- **`tests/`**: Verify the package WORKS correctly (automated testing)

## Where Your Current Files Should Go

### **Current → New Location Mapping**:

**`benchmark_chemistry.py`** → `benchmarks/benchmark_chemistry.py`
- **Reason**: Performance measurement, not correctness testing

**`benchmark_stiff.py`** → `benchmarks/benchmark_stiff.py`
- **Reason**: Performance comparison between solvers

**`cantera_test.py`** → `tests/integration/test_cantera_validation.py`
- **Reason**: Validates correctness against reference implementation

**`cfd_grid_demos.py`** → `examples/grid_generation_demo.py`
- **Reason**: Demonstrates how to use grid functionality

**`arrhenius.py`** → Split into:
- `src/rde_solver/chemistry/jax_chemistry.py` (core implementation)
- `src/rde_solver/chemistry/thermodynamics/nasa_polynomials.py` (thermo)
- `benchmarks/chemistry_performance.py` (benchmark functions)

### **Test Categories Clarified**:

**Unit Tests (`tests/unit/`)**:
```python
def test_minmod_limiter():
    """Test that minmod limiter works correctly"""
    result = minmod_limiter([1.0, 2.0, 1.5])
    assert result == 1.0  # Correctness check
```

**Integration Tests (`tests/integration/`)**:
```python
def test_sod_tube_solution():
    """Test complete Sod tube simulation against analytical solution"""
    solver = MUSCLSolver()
    result = solver.solve_sod_tube()
    np.testing.assert_allclose(result.density, analytical_density, rtol=1e-3)
```

**Benchmarks (`benchmarks/`)**:
```python
def benchmark_jax_vs_cantera():
    """Measure performance difference between JAX and Cantera"""
    jax_time = timeit(lambda: jax_chemistry.integrate())
    cantera_time = timeit(lambda: cantera.integrate())
    print(f"JAX is {cantera_time/jax_time:.1f}x faster")
```

**Examples (`examples/`)**:
```python
"""
Complete working example showing how to simulate a Sod tube problem.
This is educational - shows users how to use the package.
"""
from rde_solver import MUSCLSolver

# Set up problem
solver = MUSCLSolver(nx=100, cfl=0.5)
initial_conditions = SodTubeIC()

# Run simulation
result = solver.solve(initial_conditions, t_end=0.2)

# Visualize
result.plot()
```

This structure follows modern Python packaging standards and makes the distinction clear between testing correctness, measuring performance, and demonstrating usage!

## **5. MATLAB Reference Code Organization**

### **Why Keep MATLAB Code:**
- **Validation**: Compare Python implementations against original MATLAB results
- **Reference**: Understand algorithmic details during development
- **Reproducibility**: Enable others to verify the port accuracy
- **Documentation**: Mathematical implementation details often clearer in MATLAB

### **MATLAB Directory Structure:**

**`matlab_reference/original/`**:
- Unmodified MATLAB code from the original sources
- Serves as the definitive reference implementation
- Include attribution and source information

**`matlab_reference/modified/`**:
- MATLAB scripts modified to generate validation data
- Scripts that output CSV/text files for Python comparison
- Document modifications made and why

**`matlab_reference/validation_data/`**:
- Output files generated by MATLAB for validation
- Your current `outputs/pressure_matlab_iter_*.txt` files belong here
- Include metadata about generation conditions

### **Example MATLAB Organization:**
```
matlab_reference/
├── README.md                    # Documents MATLAB code purpose & usage
├── original/
│   ├── MUSCL_Euler2d.m         # Original 2D solver (your Python port source)
│   ├── MUSCL_EulerRes2d_v0.m   # Residual computation
│   └── utils/
│       ├── Euler_IC2d.m        # Initial conditions
│       └── HLLE1Dflux.m        # Flux computations
├── modified/
│   ├── MUSCL_Euler2d_validation.m  # Modified to output validation data
│   ├── generate_sod_reference.m     # Generate Sod tube reference
│   └── run_validation_suite.m       # Run all validation cases
└── validation_data/
    ├── sod_tube_matlab_results.csv
    ├── pressure_matlab_iter_*.txt   # Your existing outputs
    └── chemistry_matlab_reference.json
```

### **MATLAB Validation Workflow:**
1. **Run MATLAB reference** → Generate validation data
2. **Run Python implementation** → Generate Python results
3. **Compare results** → Automated validation tests
4. **Document differences** → Understand port accuracy## Detailed Refactoring Plan

### Phase 1: Core Structure Setup (Week 1)
**Priority: HIGH**

#### 1.1 Create New Package Structure
**Tasks:**
- [ ] Set up modern `pyproject.toml` with proper dependencies
- [ ] Create clean directory structure as outlined above
- [ ] Set up proper Python package with `__init__.py` files
- [ ] Update `.gitignore` to exclude development artifacts

**Notes/Discussion:**
_Add your thoughts on naming, structure modifications, etc._

#### 1.2 Split the Monolithic `arrhenius.py`
**Tasks:**
- [ ] Extract thermodynamics functions → `chemistry/thermodynamics/`
- [ ] Extract JAX chemistry integration → `chemistry/jax_chemistry.py`
- [ ] Extract mechanism parsing → `chemistry/mechanisms/`
- [ ] Extract benchmarking → `scripts/benchmark_chemistry.py`

**Notes/Discussion:**
_Discuss which functions should go where, naming conventions_

#### 1.3 Reorganize CFD Solvers
**Tasks:**
- [ ] Move `rde_solver` content → `core/solvers/`
- [ ] Clean up and standardize naming conventions
- [ ] Separate utility functions → `core/numerics/`

**Notes/Discussion:**
_Discuss solver organization, which components to prioritize_

### Phase 2: Code Quality & Standards (Week 2)
**Priority: HIGH**

#### 2.1 Add Type Hints Throughout
**Tasks:**
- [ ] Use `numpy.typing` for array types
- [ ] Add return type annotations
- [ ] Use `typing.Protocol` for interfaces

**Notes/Discussion:**
_Discuss type annotation strategy, which modules to prioritize_

#### 2.2 Standardize Function Signatures and Docstrings
**Tasks:**
- [ ] Google/NumPy style docstrings
- [ ] Consistent parameter naming
- [ ] Clear function responsibilities

**Notes/Discussion:**
_Choose docstring style, discuss function naming conventions_

#### 2.3 Error Handling and Validation
**Tasks:**
- [ ] Input validation for physical parameters
- [ ] Proper exception hierarchy
- [ ] Graceful degradation for edge cases

**Notes/Discussion:**
_Discuss what error conditions to handle, validation strategies_

### Phase 3: Testing Infrastructure (Week 3)
**Priority: MEDIUM**

#### 3.1 Set Up Pytest Framework
**Tasks:**
- [ ] Unit tests for individual functions
- [ ] Integration tests for complete workflows
- [ ] Parametrized tests for different conditions

**Notes/Discussion:**
_Discuss test coverage goals, which components need testing first_

#### 3.2 Create Validation Test Suite
**Tasks:**
- [ ] Sod tube exact solutions
- [ ] Chemistry vs Cantera validation
- [ ] Performance regression tests

**Notes/Discussion:**
_Discuss validation benchmarks, acceptable tolerances_

#### 3.3 Benchmark Infrastructure
**Tasks:**
- [ ] Automated performance tracking
- [ ] Memory usage profiling
- [ ] Scaling analysis tools

**Notes/Discussion:**
_Discuss performance metrics, benchmarking frequency_

### Phase 4: Documentation & Examples (Week 4)
**Priority: MEDIUM**

#### 4.1 API Documentation
**Tasks:**
- [ ] Sphinx setup with automatic API docs
- [ ] Mathematical background explanations
- [ ] Clear usage examples

**Notes/Discussion:**
_Discuss documentation depth, mathematical detail level_

#### 4.2 Tutorial Notebooks
**Tasks:**
- [ ] Getting started with basic CFD
- [ ] Adding chemistry to simulations
- [ ] RDE-specific workflows

**Notes/Discussion:**
_Discuss tutorial complexity, target audience_

#### 4.3 Developer Documentation
**Tasks:**
- [ ] Contributing guidelines
- [ ] Code style standards
- [ ] Release process

**Notes/Discussion:**
_Discuss contribution workflow, coding standards_

### Phase 5: Advanced Features (Ongoing)
**Priority: LOW-MEDIUM**

#### 5.1 CI/CD Setup
**Tasks:**
- [ ] GitHub Actions for testing
- [ ] Automatic benchmarking on PRs
- [ ] Documentation deployment

**Notes/Discussion:**
_Discuss CI/CD requirements, deployment strategy_

#### 5.2 Performance Optimizations
**Tasks:**
- [ ] Profile bottlenecks
- [ ] JIT compilation strategies
- [ ] Memory optimization

**Notes/Discussion:**
_Discuss performance targets, optimization priorities_

#### 5.3 RDE-Specific Features
**Tasks:**
- [ ] Rotating coordinate systems
- [ ] Detonation wave tracking
- [ ] Multi-physics coupling

**Notes/Discussion:**
_Discuss RDE-specific requirements, physics complexity_

## Files to Remove/Relocate

### Preserve & Reorganize MATLAB Code:
- [ ] **`MUSLC_TVD_Genuine2D/MUSCL_Euler2d.m`** → `matlab_reference/original/MUSCL_Euler2d.m`
- [ ] **Modified MATLAB validation scripts** → `matlab_reference/modified/`
- [ ] **MATLAB-generated outputs** (your `outputs/pressure_matlab_iter_*.txt`) → `matlab_reference/validation_data/`
- [ ] **Other core MATLAB solvers** → `matlab_reference/original/` (keep the ones that Python code is ported from)

### Remove Completely:
- [ ] **Unused MATLAB directories**: `FVmethods_waveFluctuations/`, `MUSCL_THINC_BVD/`, `THINC_BVD/`, `WENO/` (unless they contain reference implementations)
- [ ] Root-level output files: `*.png`, `*.txt`, `*.csv` (move to appropriate locations)
- [ ] `__pycache__/` directories

### Relocate/Reorganize:
- [ ] `benchmark_chemistry.py` → `scripts/benchmark_chemistry.py`
- [ ] `benchmark_stiff.py` → `scripts/benchmark_stiff.py`
- [ ] `cantera_test.py` → `tests/integration/test_cantera_validation.py`
- [ ] All `.png` files → `data/validation_results/` or `examples/outputs/`
- [ ] Documentation files → `docs/`

### Transform:
- [ ] `arrhenius.py` → Split into multiple focused modules
- [ ] `cfd_grid_demos.py` → `examples/grid_generation.py`
- [ ] `setup.py` → Modern `pyproject.toml` only

## Key Refactoring Strategies

### 1. **Modular Design**
```python
# Instead of monolithic arrhenius.py, create focused modules:
from rde_solver.chemistry import JaxChemistry, CanteraInterface
from rde_solver.core.solvers import MUSCL2D
from rde_solver.rde import RotatingDetonationEngine

# Clean, composable API
solver = MUSCL2D(chemistry=JaxChemistry("ethylene_oxygen"))
rde = RotatingDetonationEngine(solver=solver, geometry=annular_geometry)
```

### 2. **Consistent Naming Convention**
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### 3. **Clear Separation of Concerns**
```python
# Physics
class EulerEquations:
    """Handles conservative form, fluxes, eigenvalues"""

# Numerics
class MUSCLReconstruction:
    """Handles slope limiting, reconstruction"""

# Chemistry
class ChemistryIntegrator:
    """Handles species evolution, thermodynamics"""
```

### 4. **Configuration Management**
```python
# Replace scattered parameters with configuration classes
@dataclass
class SolverConfig:
    cfl: float = 0.5
    limiter: str = "minmod"
    flux_method: str = "HLLE"

@dataclass
class ChemistryConfig:
    mechanism: str = "ethylene_oxygen"
    solver: str = "CVODE"
    rtol: float = 1e-9
```

## Success Metrics

### Code Quality:
- [ ] All functions have type hints and docstrings
- [ ] Test coverage > 80%
- [ ] No files > 500 lines
- [ ] Consistent naming throughout

### Usability:
- [ ] Install with `pip install -e .`
- [ ] Simple 5-line example runs Sod tube
- [ ] Chemistry integration in < 10 lines
- [ ] Clear error messages for common mistakes

### Performance:
- [ ] JAX chemistry 10x+ faster than Cantera for grid problems
- [ ] Memory usage scales linearly with grid size
- [ ] JIT compilation overhead < 10% of runtime

### Documentation:
- [ ] API docs auto-generated and complete
- [ ] 3+ tutorial notebooks covering main workflows
- [ ] README explains installation and basic usage