import numpy as np
import time
import matplotlib.pyplot as plt
import cantera as ct
import copy
from .utils import minmod, vanalbada, vanLeer, HLLE1Dflux, HLLE1Dflux_vec
from MUSCL_TVD_FagioDiFapo.euler_solvers import EulerExact

class CFDGrid:
    def __init__(self, Lx, Ly, nx, ny, species=None):
        # Grid geometry
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny

        # Species handling
        if species is None:
            self.species_names = []
            self.nspecies = 0
            self.species_index = {}
        else:
            self.species_names = list(species)
            self.nspecies = len(species)
            self.species_index = {name: i for i, name in enumerate(species)}
        # Conserved variables: [rho, rho*u, rho*v, rho*E, rho*Y_0, ..., rho*Y_N]
        self.nvars = 4 + self.nspecies
        self.q = np.zeros((ny, nx, self.nvars))
        # Cantera reactors (optional)
        self.cantera_reactors = None

    def add_species(self, new_species):
        if new_species not in self.species_names:
            self.species_names.append(new_species)
            self.species_index[new_species] = self.nspecies
            self.nspecies += 1
            # Expand q array to add new species
            ny, nx, nvars = self.q.shape
            new_q = np.zeros((ny, nx, 4 + self.nspecies))
            new_q[:, :, :nvars] = self.q
            self.q = new_q
            self.nvars = 4 + self.nspecies

    def initialize_chemistry(self, mass_fractions, temperature=300.0, mechanism='gri30.yaml'):
        """
        Set initial species mass fractions for the grid interior, normalized using Cantera.
        Stores rho*Y_k in self.q for each species.
        Only creates reactors for interior cells (excluding ghost cells).
        """
        gas_ref = ct.Solution(mechanism)
        for s in mass_fractions:
            if s not in gas_ref.species_names:
                raise ValueError(f"Species '{s}' not found in mechanism '{mechanism}'.")
        for s in mass_fractions:
            self.add_species(s)
        gas_ref.Y = mass_fractions  # normalize

        # Set rho*Y_k in q for each species
        rho = self.q[:, :, 0]
        rho_u = self.q[:, :, 1]
        rho_v = self.q[:, :, 2]
        u = np.zeros_like(rho)
        v = np.zeros_like(rho)
        mask = rho > 0
        u[mask] = rho_u[mask] / rho[mask]
        v[mask] = rho_v[mask] / rho[mask]

        # Set rho*Y_k in q for each species
        for name in self.species_names:
            idx = self.species_index[name]
            Yk = gas_ref.Y[gas_ref.species_index(name)]
            self.q[:, :, 4 + idx] = rho * Yk

        # Variables for updating total internal energy
        E_mat = np.zeros_like(rho)

        # Check if temperature is array or scalar and handle accordingly
        is_scalar_temp = np.isscalar(temperature)
        if is_scalar_temp:
            T = np.full_like(rho, temperature)
        else:
            # For array temperature, we need to map physical indices to grid indices
            T = np.zeros_like(rho)
            T[1:-1, 1:-1] = temperature  # Copy physical array to interior

        # Initialize Cantera reactors only for interior cells
        self.cantera_reactors = [[None for _ in range(self.nx)] for _ in range(self.ny)]
        self._reactor_list = []

        # Only loop over interior cells (excluding ghost cells)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not mask[j, i]:
                    raise ValueError(f"Zero density at interior cell ({j},{i})")

                # Build per-cell mass fraction dict from q
                cell_Y = {name: float(self.q[j, i, 4 + idx] / rho[j, i]) for name, idx in self.species_index.items()}
                gas = ct.Solution(mechanism)
                gas.TDY = T[j, i], rho[j, i], cell_Y
                e = gas.int_energy_mass
                E_mat[j, i] = e + 0.5 * (u[j, i]**2 + v[j, i]**2)

                # Create reactor for this interior cell
                reactor = ct.IdealGasReactor(gas, volume=self.dx * self.dy)
                self.cantera_reactors[j][i] = reactor
                self._reactor_list.append(reactor)

        self.q[:, :, 3] = rho * E_mat

        # Create a single ReactorNet for all reactors (only interior)
        if self._reactor_list:
            self.reactor_net = ct.ReactorNet(self._reactor_list)
        else:
            self.reactor_net = None

    def initialize_euler(self, rho, rho_u, rho_v, rho_E):
        """
        Set initial Euler conserved variables for the interior grid.
        Parameters should be arrays matching the interior grid shape (ny-2, nx-2).
        """
        # Check if input is array or scalar
        if np.isscalar(rho):
            # If scalar, broadcast to entire grid
            self.q[:, :, 0] = rho
            self.q[:, :, 1] = rho_u
            self.q[:, :, 2] = rho_v
            self.q[:, :, 3] = rho_E
        else:
            # If arrays, they should match the interior size
            # and we only set the interior cells (excluding ghost cells)
            self.q[1:-1, 1:-1, 0] = rho
            self.q[1:-1, 1:-1, 1] = rho_u
            self.q[1:-1, 1:-1, 2] = rho_v
            self.q[1:-1, 1:-1, 3] = rho_E

    def create_consistent_initial_state(self, pressure, mass_fractions, velocity_x=0.0, velocity_y=0.0, mechanism='gri30.yaml', temperature=None, energy=None):
        """
        Create thermodynamically consistent initial state variables for Euler equations
        using Cantera for proper equation of state calculations.
        """

        # Validate input mode
        if temperature is None and energy is None:
            raise ValueError("Must provide either temperature or energy")

        # Get interior grid size
        ny_interior, nx_interior = self.ny - 2, self.nx - 2

        # Convert scalar inputs to arrays if needed
        def ensure_array(val, shape):
            if np.isscalar(val):
                return np.full(shape, val)
            return val

        # Create arrays for the interior grid
        P = ensure_array(pressure, (ny_interior, nx_interior))
        ux = ensure_array(velocity_x, (ny_interior, nx_interior))
        uy = ensure_array(velocity_y, (ny_interior, nx_interior))

        if temperature is not None:
            T = ensure_array(temperature, (ny_interior, nx_interior))
        else:
            T = np.zeros((ny_interior, nx_interior))  # Will be computed from energy

        if energy is not None:
            e_in = ensure_array(energy, (ny_interior, nx_interior))
        else:
            e_in = None

        # Initialize output arrays
        rho = np.zeros_like(P)
        e = np.zeros_like(P)
        T_out = np.zeros_like(P)

        # Create reference gas object and normalize mass fractions
        gas = ct.Solution(mechanism)
        gas.Y = mass_fractions
        normalized_Y = {}
        for s in mass_fractions:
            if s in gas.species_names:
                idx = gas.species_index(s)
                normalized_Y[s] = gas.Y[idx]

        # Ensure mass fractions sum to 1
        Y_sum = sum(normalized_Y.values())
        if abs(Y_sum - 1.0) > 1e-6:
            for s in normalized_Y:
                normalized_Y[s] /= Y_sum

        print(f"Computing thermodynamically consistent state for {ny_interior}x{nx_interior} grid...")

        # Loop over interior cells to calculate state variables
        for j in range(ny_interior):
            for i in range(nx_interior):
                # Ensure pressure is physically valid (positive)
                P[j, i] = max(1000.0, P[j, i])  # Minimum pressure of 1000 Pa

                try:
                    if energy is None:
                        # Temperature-driven initialization (intuitive)

                        # Ensure temperature is physically valid (positive)
                        T[j, i] = max(200.0, min(5000.0, T[j, i]))  # Limit T to reasonable range

                        # Set gas state with TPY
                        gas.TPY = T[j, i], P[j, i], normalized_Y

                        # Get density and internal energy
                        rho[j, i] = gas.density
                        e[j, i] = gas.int_energy_mass
                        T_out[j, i] = T[j, i]

                    else:
                        # Energy-driven initialization (numerically stable)

                        # Iterate to find T that produces the desired internal energy
                        # Start with a reasonable guess
                        T_guess = 1000.0 if T[j, i] == 0 else T[j, i]

                        # Simple iterative method to find matching temperature
                        for iter in range(20):  # Max iterations
                            gas.TPY = T_guess, P[j, i], normalized_Y
                            e_current = gas.int_energy_mass
                            residual = e_current - e_in[j, i]

                            if abs(residual) < 1e-4 * abs(e_in[j, i]):
                                break  # Converged

                            # Update guess (simple method)
                            dTde = 1.0 / gas.cv_mass  # Approximate derivative
                            T_guess -= residual * dTde
                            T_guess = max(200.0, min(5000.0, T_guess))  # Ensure valid range

                        # Set final state
                        gas.TPY = T_guess, P[j, i], normalized_Y
                        rho[j, i] = gas.density
                        e[j, i] = gas.int_energy_mass
                        T_out[j, i] = T_guess

                except Exception as ex:
                    print(f"Error in cell ({j},{i}): {ex}")
                    # Use failsafe values
                    T_safe = 300.0 if energy is None else 1000.0
                    P_safe = max(101325.0, P[j, i])

                    try:
                        gas.TPY = T_safe, P_safe, normalized_Y
                        rho[j, i] = gas.density
                        e[j, i] = gas.int_energy_mass
                        T_out[j, i] = T_safe
                    except Exception:
                        # Last resort: hard-coded values
                        rho[j, i] = 1.0
                        e[j, i] = 2.5 * 8314.0 / 0.029  # Approximate air
                        T_out[j, i] = 300.0

        # Calculate conserved variables
        rho_u = rho * ux
        rho_v = rho * uy

        # Total energy = internal + kinetic
        E = e + 0.5 * (ux**2 + uy**2)
        rho_E = rho * E

        print("Initial state created successfully")
        return rho, rho_u, rho_v, rho_E, T_out

    def muscl_euler_res2d_v0(self, limiter='MC', fluxMethod='HLLE1d'):
        """
        A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
        Upstream Centered Scheme for Conservation Laws (MUSCL).
        Vectorized version: uses only numpy arrays for all states and residuals.
        Refactored to use CFDGrid object properties.
        Original code written by Manuel Diaz, NTU, 05.25.2015.
        """
        q = self.q
        dx = self.dx
        dy = self.dy
        N = self.nx
        M = self.ny

        # Allocate arrays for all states
        qN = np.zeros((M, N, self.nvars))
        qS = np.zeros((M, N, self.nvars))
        qE = np.zeros((M, N, self.nvars))
        qW = np.zeros((M, N, self.nvars))
        residual = np.zeros((M, N, self.nvars))

        # Compute and limit slopes at cells (i,j)
        for k in range(self.nvars):
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
        for j in range(1, N - 1):
            qR = qS[M - 2, j, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [0, 1])
            residual[M - 2, j, :] += flux / dy

        # East face (j = N-2)
        for i in range(1, M - 1):
            qR = qW[i, N - 2, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [1, 0])
            residual[i, N - 2, :] += flux / dx

        # South face (i = 1)
        for j in range(1, N - 1):
            qR = qN[1, j, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [0, -1])
            residual[1, j, :] += flux / dy

        # West face (j = 1)
        for i in range(1, M - 1):
            qR = qE[i, 1, :]
            qL = qR
            if fluxMethod == 'HLLE1d':
                flux = HLLE1Dflux(qL, qR, [-1, 0])
            residual[i, 1, :] += flux / dx

        # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
        res = np.zeros_like(residual)
        res[1:M-1, 1:N-1, :] = residual[1:M-1, 1:N-1, :]

        return res

    def muscl_euler_res2d_v1(self, limiter='MC', fluxMethod='HLLE1d'):
        """
        A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
        Upstream Centered Scheme for Conservation Laws (MUSCL).
        Mass vectorized version: uses only numpy arrays and operations states and residuals.
        Original code written by Manuel Diaz, NTU, 05.25.2015.
        """
        q = self.q
        dx = self.dx
        dy = self.dy
        N = self.nx
        M = self.ny

        # Allocate arrays for all states
        qN = np.zeros((M, N, self.nvars))
        qS = np.zeros((M, N, self.nvars))
        qE = np.zeros((M, N, self.nvars))
        qW = np.zeros((M, N, self.nvars))
        residual = np.zeros((M, N, self.nvars))

        # Compute and limit slopes at cells (i,j)
        for k in range(self.nvars):
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

            qE[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdx / 2
            qW[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdx / 2
            qN[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdy / 2
            qS[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdy / 2

        # Residuals: x-direction
        qxL = qE[1:-1, 1:-2, :]   # i = 1..M-2, j = 1..N-3
        qxR = qW[1:-1, 2:-1, :]   # i = 1..M-2, j = 2..N-2
        flux_x = HLLE1Dflux_vec(qxL, qxR, [1, 0])

        residual[1:-1, 1:-2, :] += flux_x / dx
        residual[1:-1, 2:-1, :] -= flux_x / dx

        # Residuals: y-direction
        qyL = qN[1:-2, 1:-1, :]   # lower state at each interface (i=1..M-3, j=1..N-2)
        qyR = qS[2:-1, 1:-1, :]   # upper state at each interface (i+1=2..M-2, j=1..N-2)
        flux_y = HLLE1Dflux_vec(qyL, qyR, [0, 1])

        residual[1:-2, 1:-1, :] += flux_y / dy
        residual[2:-1, 1:-1, :] -= flux_y / dy

        # Set BCs: boundary flux contributions
        # North face (i = M-2, horizontal interface at top boundary)
        qR_N = qS[M-2, 1:-1, :]   # shape (N-2, 4)
        qL_N = qR_N
        flux_N = HLLE1Dflux_vec(qL_N[None, :, :], qR_N[None, :, :], [0, 1])[0]  # shape (N-2, 4)
        residual[M-2, 1:-1, :] += flux_N / dy

        # East face (j = N-2, vertical interface at right boundary)
        qR_E = qW[1:-1, N-2, :]   # shape (M-2, 4)
        qL_E = qR_E
        flux_E = HLLE1Dflux_vec(qL_E[:, None, :], qR_E[:, None, :], [1, 0])[:, 0, :]  # shape (M-2, 4)
        residual[1:-1, N-2, :] += flux_E / dx

        # South face (i = 1, horizontal interface at bottom boundary)
        qR_S = qN[1, 1:-1, :]     # shape (N-2, 4)
        qL_S = qR_S
        flux_S = HLLE1Dflux_vec(qL_S[None, :, :], qR_S[None, :, :], [0, -1])[0]  # shape (N-2, 4)
        residual[1, 1:-1, :] += flux_S / dy

        # West face (j = 1, vertical interface at left boundary)
        qR_W = qE[1:-1, 1, :]     # shape (M-2, 4)
        qL_W = qR_W
        flux_W = HLLE1Dflux_vec(qL_W[:, None, :], qR_W[:, None, :], [-1, 0])[:, 0, :]  # shape (M-2, 4)
        residual[1:-1, 1, :] += flux_W / dx

        # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
        res = np.zeros_like(residual)
        res[1:M-1, 1:N-1, :] = residual[1:M-1, 1:N-1, :]
        return res

    def advance_chemistry(self, dt):
        """
        Advance chemistry in all Cantera reactors by time step dt.
        Only processes interior cells (excluding ghost cells).
        """
        if self.cantera_reactors is None or not hasattr(self, 'reactor_net') or self.reactor_net is None:
            raise ValueError("Chemistry not initialized. Call initialize_chemistry first.")

        # Update all reactors with the latest CFD state before advancing (interior only)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                reactor = self.cantera_reactors[j][i]
                if reactor is not None:
                    gas = reactor.thermo

                    # 1. Extract current Euler values from q
                    rho = self.q[j, i, 0]
                    if not np.isfinite(rho) or rho <= 0:
                        print(f"Skipping cell ({j},{i}): non-physical density {rho}")
                        continue

                    rho_u = self.q[j, i, 1]
                    rho_v = self.q[j, i, 2]
                    u = rho_u / rho
                    v = rho_v / rho

                    # Build mass fraction dict from q
                    Y_dict = {name: float(self.q[j, i, 4 + idx] / rho) for name, idx in self.species_index.items()}
                    Y_sum = sum(Y_dict.values())
                    if Y_sum <= 0 or not np.isfinite(Y_sum):
                        print(f"Skipping cell ({j},{i}): non-physical mass fractions {Y_dict}")
                        continue
                    for k in Y_dict:
                        Y_dict[k] /= Y_sum

                    # 2. Set gas state to match current cell (using density, velocity, Y)
                    E = self.q[j, i, 3] / rho
                    kin = 0.5 * (u**2 + v**2)
                    e = E - kin
                    if not np.isfinite(e):
                        print(f"Skipping cell ({j},{i}): non-physical internal energy {e}")
                        continue
                    try:
                        gas.UVY = e, 1.0 / rho, Y_dict
                    except Exception as ex:
                        print(f"Failed to set UVY for cell ({j},{i}): {ex}")
                        try:
                            gas.TDY = gas.T, max(rho, 1e-8), Y_dict
                        except Exception as ex2:
                            print(f"Failed to set TDY for cell ({j},{i}): {ex2}")
                            continue

        # Advance the entire reactor network by dt
        try:
            self.reactor_net = ct.ReactorNet(self._reactor_list)
            self.reactor_net.advance(dt)
        except Exception as ex:
            print(f"Chemistry failed: {ex}")
            raise

        # Update q with new state from each reactor (interior only)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                reactor = self.cantera_reactors[j][i]
                if reactor is not None:
                    gas = reactor.thermo
                    rho_new = gas.density
                    if not np.isfinite(rho_new) or rho_new <= 0:
                        print(f"Skipping update for cell ({j},{i}): non-physical new density {rho_new}")
                        continue

                    # Velocity is not changed by chemistry
                    u_new = self.q[j, i, 1] / self.q[j, i, 0]
                    v_new = self.q[j, i, 2] / self.q[j, i, 0]
                    e_new = gas.int_energy_mass
                    E_new = e_new + 0.5 * (u_new**2 + v_new**2)

                    # Update conserved variables
                    #print(f"Updating cell ({j},{i}): rho={rho_new}, u={u_new}, v={v_new}, E={E_new}")
                    self.q[j, i, 0] = rho_new
                    self.q[j, i, 1] = rho_new * u_new
                    self.q[j, i, 2] = rho_new * v_new
                    self.q[j, i, 3] = rho_new * E_new

                    # Update species mass fractions
                    for name in self.species_names:
                        idx = self.species_index[name]
                        self.q[j, i, 4 + idx] = rho_new * gas.Y[gas.species_index(name)]

def test_initialization():
    # Define species for hydrogen-oxygen system with radicals
    species = ['H2', 'O2', 'H', 'O', 'OH', 'HO2', 'H2O2', 'H2O']

    # Create a 1m x 1m grid with 100x100 cells
    grid = CFDGrid(Lx=1.0, Ly=1.0, nx=100, ny=100, species=species)

    print("Grid created with the following species:")
    print(grid.species_names)
    print("Species index mapping:")
    print(grid.species_index)
    print("Shape of species array:", None if grid.species is None else grid.species.shape)
    print("Shape of conserved variables array:", grid.q.shape)

    # Initialize chemistry with mass fractions (example: 100% H2, 100% O2)
    grid.initialize_chemistry({'H2': 1, 'O2': 1, 'H': 0, 'O': 0, 'OH': 0, 'HO2': 0, 'H2O2': 0, 'H2O': 0})

    # Print normalized mass fractions for verification
    print("Normalized mass fractions (from Cantera):")
    for name in grid.species_names:
        idx = grid.species_index[name]
        print(f"{name}: {grid.species[0,0,idx]:.6f}")

def blast_ic_2d(x, y, p_high=1.0, p_low=0.1, r0=0.1):
    """2D blast wave: high pressure in center, low elsewhere."""
    xc, yc = 0.5, 0.5
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(r < r0, p_high, p_low)
    return rho, u, v, p

def blast_ic_2d_square(x, y, p_high=1.0, p_low=0.1, halfwidth=0.1):
    """2D blast wave: high pressure in a square at the center, low elsewhere."""
    xc, yc = 0.5, 0.5
    mask = (np.abs(x - xc) < halfwidth) & (np.abs(y - yc) < halfwidth)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(mask, p_high, p_low)
    return rho, u, v, p

def sod_ic_2d(x, y):
    # Sod's tube: left state for x < 0.5, right state for x >= 0.5
    # Use Sod's values: rho=1, u=0, p=1 (left); rho=0.125, u=0, p=0.1 (right)
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    return rho, u, v, p

def init_plots(xe, re, ue, pe, Ee):
    plt.ion()
    fig, axs = plt.subplots(2, 2, num=2)
    lines = [
        axs[0, 0].plot([], [], '.b')[0],
        axs[0, 1].plot([], [], '.m')[0],
        axs[1, 0].plot([], [], '.k')[0],
        axs[1, 1].plot([], [], '.r')[0]
    ]
    axs[0, 0].plot(xe, re, '-k', label='Exact')
    axs[0, 1].plot(xe, ue, '-k', label='Exact')
    axs[1, 0].plot(xe, pe, '-k', label='Exact')
    axs[1, 1].plot(xe, Ee, '-k', label='Exact')
    axs[0, 0].set_xlabel('x'); axs[0, 0].set_ylabel(r'$\rho$')
    axs[0, 1].set_xlabel('x'); axs[0, 1].set_ylabel('u')
    axs[1, 0].set_xlabel('x'); axs[1, 0].set_ylabel('p')
    axs[1, 1].set_xlabel('x'); axs[1, 1].set_ylabel('E')
    axs[0, 0].set_title('SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)')
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1.1])
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim([-0.1, 1.1])
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim([0, 1.1])
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim([1.5, 3.5])
    for axrow in axs:
        for ax in axrow:
            ax.legend()
    plt.tight_layout()
    return fig, axs, lines

def update_plots(lines, xc, r, u, p, E):
    lines[0].set_data(xc, r)
    lines[1].set_data(xc, u)
    lines[2].set_data(xc, p)
    lines[3].set_data(xc, E)
    plt.draw()
    plt.pause(0.001)

def final_plots(xc, r, u, p, E, xe, re, ue, pe, Ee):
    plt.figure(2)
    plt.subplot(2, 2, 1); plt.plot(xc, r, 'ro', xe, re, '-k'); plt.xlabel('x'); plt.ylabel(r'$\rho$'); plt.legend(['MUSCL-2D', 'Exact'])
    plt.title('SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)')
    plt.subplot(2, 2, 2); plt.plot(xc, u, 'mo', xe, ue, '-k'); plt.xlabel('x'); plt.ylabel('u')
    plt.subplot(2, 2, 3); plt.plot(xc, p, 'ko', xe, pe, '-k'); plt.xlabel('x'); plt.ylabel('p')
    plt.subplot(2, 2, 4); plt.plot(xc, E, 'ro', xe, Ee, '-k'); plt.xlabel('x'); plt.ylabel('E')
    plt.tight_layout()
    plt.show()

def run_2d_sod_with_grid():
    nx, ny = 200, 1
    Lx, Ly = 1.0, 0.1
    dx, dy = Lx / nx, Ly / ny
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions
    r0, u0, v0, p0 = sod_ic_2d(x, y)
    n = 5
    gamma = (n + 2) / n
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :4] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    CFL = 0.5
    dt = CFL * min(dx, dy) / a0
    tEnd = 0.15

    # Central slice for plotting
    mid_row = ny // 2
    x_slice = xc

    # Compute final exact solution once
    xe, re, ue, pe, ee, *_ = EulerExact(
        1.0, 0.0, 1.0, 0.125, 0.0, 0.1, tEnd, n
    )
    Ee = ee + 0.5 * ue**2

    # Live plot setup (CFD only, exact is static reference)
    plt.ion()
    fig, axs, lines = init_plots(xe, re, ue, pe, Ee)

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract central slice
        r = q[mid_row+1, 1:-1, 0]
        u = q[mid_row+1, 1:-1, 1] / r
        E = (q[mid_row+1, 1:-1, 3] / r)
        p = (gamma - 1) * r * (E - 0.5 * u ** 2)

        # Update CFD plots every 10 steps for performance
        if it % 1 == 0:
            update_plots(lines, x_slice, r, u, p, E)
        t += dt
        it += 1

    plt.ioff()
    plt.show()

    # Final plot with exact solution (optional, already shown in live plot)
    final_plots(x_slice, r, u, p, E, xe, re, ue, pe, Ee)


def run_blast_test():
    start = time.time()
    # Parameters
    nx, ny = 100, 100
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    tEnd = 1.0
    CFL = 0.5

    # Grid
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions: blast wave
    r0, u0, v0, p0 = blast_ic_2d(x, y, p_high=1.0, p_low=0.1, r0=0.1)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :4] = Q0
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda1 = vn + c0
    lambda2 = vn - c0
    a0 = np.max(np.abs(np.concatenate([lambda1.reshape(-1), lambda2.reshape(-1)])))
    dt = CFL * min(dx, dy) / a0

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(p0, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma', vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Pressure')
    ax.set_title('2D Blast Wave: Pressure')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.1)  # Give the GUI event loop time to process

    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # RK2 step 1
        res = grid.muscl_euler_res2d_v0(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        # BCs
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v0(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        # BCs
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract pressure field (interior)
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

        # Update plot every x steps
        if it % 1 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave: Pressure, t={t:.3f}')
            plt.pause(0.001)
        t += dt
        it += 1

    plt.show()
    plt.close(fig)
    plt.close('all')
    end = time.time()
    print(f"Non-vectorized simulation time: {end - start:.3f} seconds")
    return end - start

def run_blast_test_vectorized(tEnd=0.5, nx=11, ny=11, plot=True):
    """Run a vectorized blast test with chemistry (hydrogen-oxygen reactions)."""
    start = time.time()

    # Parameters
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    CFL = 0.5

    # Chemistry setup
    species = ['H2', 'O2', 'H2O']  # Simplified species list
    fractions = {
        'H2': 0.2,      # Increase hydrogen concentration
        'O2': 0.1,      # Increase oxygen concentration
        'N2': 0.7       # Use N2 as diluent (faster than Ar)
    }
    mechanism = 'h2o2.yaml'  # Much faster than gri30.yaml

    # Grid setup (including ghost cells)
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg, species=species)

    # Create simple temperature field with a hot spot in center
    T0 = np.full((ny, nx), 600.0)  # Base temperature 600K for stability

    # Find center indices
    center_j, center_i = ny // 2, nx // 2

    # Simple hot spot in center (use direct indices rather than calculating distances)
    if nx >= 3 and ny >= 3:
        T0[center_j, center_i] = 1500.0

        # If grid is larger than 3x3, make a slightly larger hot spot
        if nx >= 5 and ny >= 5:
            T0[center_j-1:center_j+2, center_i-1:center_i+2] = 1500.0
    else:
        # For very small grids, just heat the center cell
        T0[center_j, center_i] = 1500.0

    # Create uniform pressure field (higher pressure for better stability)
    P0 = np.full((ny, nx), 3.0 * 101325.0)

    # Generate thermodynamically consistent initial state
    rho0, rho_u0, rho_v0, rho_E0, T_actual = grid.create_consistent_initial_state(
        temperature=T0,
        pressure=P0,
        mass_fractions=fractions,
        mechanism=mechanism
    )

    # Initialize Euler variables
    grid.initialize_euler(rho=rho0, rho_u=rho_u0, rho_v=rho_v0, rho_E=rho_E0)

    # Initialize chemistry using the same temperature field
    grid.initialize_chemistry(fractions, temperature=T_actual, mechanism=mechanism)

    # Set boundary conditions
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time stepping parameters - use Cantera for accurate gamma
    import cantera as ct
    gas = ct.Solution('gri30.yaml')
    gas.TPY = 600.0, 2.0 * 101325.0, fractions
    gamma = gas.cp / gas.cv

    # Get a representative sound speed
    c_sound = np.sqrt(gamma * 2.0 * 101325.0 / rho0.mean())
    dt = 0.5 * CFL * min(dx, dy) / c_sound  # Conservative time step

    print(f"Using dt = {dt:.6f} seconds")

    # Plot setup
    if plot:
        plt.ion()
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Pressure plot (atm)
        p_plot = P0 / 101325.0
        im1 = axs[0].imshow(p_plot, origin='lower', extent=[0, Lx, 0, Ly],
                          cmap='plasma', vmin=0, vmax=10.0)
        plt.colorbar(im1, ax=axs[0], label='Pressure [atm]')
        axs[0].set_title('Pressure')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')

        # Temperature plot - IMPROVED VISUALIZATION
        # Use 'inferno' or 'viridis' colormap for better temperature gradient visibility
        # Set vmin closer to base temperature to see smaller variations
        temp_min = 550  # Just below base temperature
        temp_max = 2000  # Maximum expected temperature

        im2 = axs[1].imshow(T_actual, origin='lower', extent=[0, Lx, 0, Ly],
                         cmap='viridis', vmin=temp_min, vmax=temp_max)
        cbar = plt.colorbar(im2, ax=axs[1], label='Temperature [K]')

        # Add tick marks at relevant temperatures
        cbar.set_ticks([600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
        axs[1].set_title('Temperature')
        axs[1].set_xlabel('x')

        # H2O mass fraction plot
        h2o_idx = grid.species_index.get('H2O', -1)
        h2o_data = np.zeros((ny, nx))
        if h2o_idx >= 0:
            im3 = axs[2].imshow(h2o_data, origin='lower', extent=[0, Lx, 0, Ly],
                              cmap='Blues', vmin=0, vmax=0.15)  # Increased max to show more detail
            plt.colorbar(im3, ax=axs[2], label='H2O Mass Fraction')
            axs[2].set_title('H2O (Water)')
            axs[2].set_xlabel('x')

        plt.tight_layout()
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)

    # Time stepping loop
    t = 0.0
    it = 0
    q = grid.q
    while t < tEnd:
        # Split scheme: advance chemistry first
        if it % 1 == 0:  # Chemistry step less frequently for stability
            try:
                print(f"Chemistry step at t={t:.3f}")
                grid.advance_chemistry(dt)
            except Exception as ex:
                print(f"Chemistry failed at t={t:.3f}: {ex}")

        # RK2 step 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res

        # BCs
        for s in range(grid.nvars):
            qs[:, 0, s] = qs[:, 1, s]
            qs[:, -1, s] = qs[:, -2, s]
            qs[0, :, s] = qs[1, :, s]
            qs[-1, :, s] = qs[-2, :, s]

        # RK2 step 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)

        # BCs
        for s in range(grid.nvars):
            q[:, 0, s] = q[:, 1, s]
            q[:, -1, s] = q[:, -2, s]
            q[0, :, s] = q[1, :, s]
            q[-1, :, s] = q[-2, :, s]

        grid.q = q

        # Extract fields (interior)
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u**2 + v**2))

        # Update plot every few steps
        if plot and it % 10 == 0:
            im1.set_data(p / 101325.0)  # Convert to atm for display
            axs[0].set_title(f'Pressure [atm], t={t:.3f}')

            # Extract temperature from reactors
            temps = np.zeros((ny, nx))
            h2o_vals = np.zeros((ny, nx))
            for j in range(ny):
                for i in range(nx):
                    reactor = grid.cantera_reactors[j+1][i+1]
                    if reactor is not None:
                        temps[j, i] = reactor.thermo.T
                        if h2o_idx >= 0:
                            h2o_vals[j, i] = q[j+1, i+1, 4 + h2o_idx] / r[j, i]

            im2.set_data(temps)
            axs[1].set_title(f'Temperature [K], t={t:.3f}')

            if h2o_idx >= 0:
                im3.set_data(h2o_vals)
                axs[2].set_title(f'H2O Mass Fraction, t={t:.3f}')

            plt.pause(0.001)

        t += dt
        it += 1
        if it % 10 == 0:
            print(f"Step {it}, t={t:.3f}")

    if plot:
        plt.show()
        plt.close(fig)
        plt.close('all')

    end = time.time()
    print(f"Vectorized reactive simulation time: {end - start:.3f} seconds")
    return end - start

def initialization_test():
    import cantera as ct

    # Define species and initial mass fractions (including AR, which is in gri30.yaml)
    species = ['H2', 'O2', 'H', 'O', 'OH', 'HO2', 'H2O2', 'H2O']
    fractions = {'H2': 2, 'O2': 1, 'H': 0, 'O': 0, 'OH': 0, 'HO2': 0, 'H2O2': 0, 'H2O': 0, 'AR': 0.5}

    # Create grid
    g = CFDGrid(Lx=1.0, Ly=1.0, nx=10, ny=10, species=species)

    # Use Cantera to get consistent initial thermodynamic state
    gas = ct.Solution('gri30.yaml')
    gas.TPY = 320.0, 101325.0, fractions  # Set T [K], P [Pa], and composition

    rho0 = gas.density
    u0 = 0.0
    v0 = 0.0
    e0 = gas.int_energy_mass
    E0 = e0 + 0.5 * (u0**2 + v0**2)

    # Initialize Euler variables with physical values
    g.initialize_euler(rho=rho0, rho_u=u0, rho_v=v0, rho_E=E0 * rho0)

    print("Species before chemistry initialization:", g.species_names)
    g.initialize_chemistry(fractions)
    print("Species after chemistry initialization:", g.species_names)
    print("Species index mapping:", g.species_index)
    print("Shape of species array:", g.q[:, :, 4:].shape)
    print("Shape of cantera_reactors array:", len(g.cantera_reactors), "x", len(g.cantera_reactors[0]))

    # Print normalized mass fractions for the first cell
    print("Normalized mass fractions in cell (0,0):")
    rho = g.q[0, 0, 0]
    for name in g.species_names:
        idx = g.species_index[name]
        Yk = g.q[0, 0, 4 + idx] / rho
        print(f"  {name}: {Yk:.6f}")

if __name__ == "__main__":
    run_blast_test_vectorized()