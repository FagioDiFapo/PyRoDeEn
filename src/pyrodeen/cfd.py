import numpy as np
from enum import Enum
from dataclasses import dataclass


class SlopeLimiters:
    """
    Class of sape-agnostic slope limiters.
    """

    @staticmethod
    def minmod(vectors: list[np.ndarray]) -> np.ndarray:
        """Minmod slope limiter (most dissipative TVD limiter).

        Returns minimum absolute slope when all signs agree, zero otherwise.
        """
        stacked = np.stack(vectors, axis=0)  # shape (num_slopes, M-2, N-2)
        s = np.sum(np.sign(stacked), axis=0) / stacked.shape[0]
        mm = np.zeros_like(stacked[0])
        mask = np.abs(s) == 1
        mm[mask] = s[mask] * np.min(np.abs(stacked[:, mask]), axis=0)
        return mm

    @staticmethod
    def van_albada(da: np.ndarray, db: np.ndarray, h: float) -> np.ndarray:
        """Van Albada slope limiter with smooth epsilon regularization."""
        eps2 = (0.3 * h) ** 3
        numerator = (db**2 + eps2) * da + (da**2 + eps2) * db
        denominator = da**2 + db**2 + 2 * eps2
        return np.where(
            denominator != 0,
            0.5 * (np.sign(da) * np.sign(db) + 1) * numerator / denominator,
            0.0,
        )

    @staticmethod
    def van_leer(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        """Van Leer slope limiter: phi(r) = (r + |r|) / (1 + |r|)."""
        # Use np.divide with where to avoid division by zero warnings
        r = np.divide(da, db, out=np.zeros_like(da), where=db != 0)
        return (r + np.abs(r)) / (1 + np.abs(r))


class BoundaryCondition(Enum):
    """Boundary condition types."""

    TRANSMISSIVE = 1
    """TRANSMISSIVE: Zero-gradient (outflow) boundary condition."""
    REFLECTING = 2
    """REFLECTING: Wall boundary condition with momentum reflection."""


@dataclass
class BoundaryConfig:
    """Boundary condition configuration for all four sides.

    Attributes:
        left (BoundaryCondition): Left boundary (x=0).
        right (BoundaryCondition): Right boundary (x=Lx).
        bottom (BoundaryCondition): Bottom boundary (y=0, typically bottom of plot).
        top (BoundaryCondition): Top boundary (y=Ly, typically top of plot).
    """

    left: "BoundaryCondition"
    right: "BoundaryCondition"
    bottom: "BoundaryCondition"
    top: "BoundaryCondition"

    @staticmethod
    def all_transmissive() -> "BoundaryConfig":
        """Create configuration with transmissive boundaries on all sides."""
        BC = BoundaryCondition
        return BoundaryConfig(
            BC.TRANSMISSIVE, BC.TRANSMISSIVE, BC.TRANSMISSIVE, BC.TRANSMISSIVE
        )

    @staticmethod
    def shock_tube() -> "BoundaryConfig":
        """Create configuration for 1D shock tube (reflecting top/bottom walls)."""
        BC = BoundaryCondition
        return BoundaryConfig(
            BC.TRANSMISSIVE, BC.TRANSMISSIVE, BC.REFLECTING, BC.REFLECTING
        )

    @staticmethod
    def closed_box() -> "BoundaryConfig":
        """Create configuration for closed box (all reflecting walls)."""
        BC = BoundaryCondition
        return BoundaryConfig(
            BC.REFLECTING, BC.REFLECTING, BC.REFLECTING, BC.REFLECTING
        )


class Solver:

    def __init__(self, boundary_config: BoundaryConfig, gamma: float):
        """Initialize the CFD solver with grid, boundary conditions, and gas properties.
        Args:
            grid: Grid object containing state variables and geometry.
            boundary_config: BoundaryConfig object specifying BC types for each side.
            gamma: Ratio of specific heats for the gas (e.g., 1.4 for air).
        """
        self.boundary_config = boundary_config
        self.gamma = gamma

    @staticmethod
    def _compute_state_variables(q, normal: list[float], gamma: float):
        """Compute primitive variables, normal velocity, pressure, sound speed, and enthalpy from conserved variables."""
        n_x, n_y = normal
        rho = q[..., 0]
        u, v = q[..., 1] / rho, q[..., 2] / rho
        nor_vel = u * n_x + v * n_y
        p = (gamma - 1) * (q[..., 3] - rho * (u**2 + v**2) / 2)
        c = np.sqrt(gamma * p / rho)
        enthalpy = (q[..., 3] + p) / rho
        return rho, u, v, nor_vel, p, c, enthalpy

    @staticmethod
    def _roe_average(
        rho_L,
        rho_R,
        u_L,
        u_R,
        v_L,
        v_R,
        ent_L,
        ent_R,
        normal: list[float],
        gamma: float,
    ):
        """Compute Roe-averaged velocity, enthalpy, and sound speed."""
        n_x, n_y = normal
        weight = np.sqrt(rho_R / rho_L)
        u_avg = (u_L + weight * u_R) / (1 + weight)
        v_avg = (v_L + weight * v_R) / (1 + weight)
        ent_avg = (ent_L + weight * ent_R) / (1 + weight)
        c_roe = np.sqrt((gamma - 1) * (ent_avg - (u_avg**2 + v_avg**2) / 2))
        nor_vel_roe = u_avg * n_x + v_avg * n_y
        return nor_vel_roe, c_roe

    @staticmethod
    def _euler_flux(rho, u, v, nor_vel, p, ent, normal: list[float]):
        """Physical Euler flux projected onto the face normal."""
        n_x, n_y = normal
        flux = np.stack(
            [
                rho * nor_vel,
                rho * nor_vel * u + p * n_x,
                rho * nor_vel * v + p * n_y,
                rho * nor_vel * ent,
            ],
            axis=-1,
        )
        return flux

    @staticmethod
    def _species_flux(mass_flux, qL, qR, rho_L, rho_R):
        """Transport of chemical species by the resolved mass flux."""
        y_L = qL[..., 4:] / rho_L[..., None]
        y_R = qR[..., 4:] / rho_R[..., None]
        y_upwind = np.where(mass_flux[..., None] >= 0, y_L, y_R)
        return mass_flux[..., None] * y_upwind

    def HLLE1Dflux_vec(
        self, qL: np.ndarray, qR: np.ndarray, normal: list[float]
    ) -> np.ndarray:
        """Vectorized Harten-Lax-van Leer-Einfeldt approximate Riemann flux with Roe wave speed estimates.

        Supports Euler variables [rho, rho*u, rho*v, rho*E] with optional species transport.

        Convention note:
            The conserved energy slot is the total energy density `rho*E` (energy per
            unit volume). In literature you will sometimes see `E` used as the
            specific total energy (per unit mass). In this code `q[..., 3]` is
            `rho*E` and specific quantities are obtained by dividing by `rho`.

            Enthalpy is computed as (rho*E + p) / rho which yields the specific
            total enthalpy consistent with the conserved-variable convention used here.

        Args:
            qL (np.ndarray): Left state conserved variables.
            qR (np.ndarray): Right state conserved variables.
            normal (list[float]): Normal vector [nx, ny] for interface.

        Returns:
            np.ndarray: HLLE flux at the interface.

        Note:
            Original code by Manuel Diaz, NTU, 05.25.2015.
        """
        n_x, n_y = normal
        num_species = qL.shape[-1] - 4

        # Decode left and right primitive states
        rho_L, u_L, v_L, nor_vel_L, p_L, c_L, ent_L = self._compute_state_variables(
            qL, normal, self.gamma
        )
        rho_R, u_R, v_R, nor_vel_R, p_R, c_R, ent_R = self._compute_state_variables(
            qR, normal, self.gamma
        )

        # Roe-averaged intermediate state for wave speed estimates.
        nor_vel_roe, c_roe = self._roe_average(
            rho_L, rho_R, u_L, u_R, v_L, v_R, ent_L, ent_R, normal, self.gamma
        )

        # Einfeldt wave speed bounds, clamped so left speed <= 0 <= right speed.
        zeros = np.zeros_like(nor_vel_roe)
        wave_speed_left = np.minimum.reduce(
            [nor_vel_L - c_L, nor_vel_roe - c_roe, zeros]
        )
        wave_speed_right = np.maximum.reduce(
            [nor_vel_R + c_R, nor_vel_roe + c_roe, zeros]
        )

        # Physical fluxes on each side.
        flux_L = self._euler_flux(rho_L, u_L, v_L, nor_vel_L, p_L, ent_L, normal)
        flux_R = self._euler_flux(rho_R, u_R, v_R, nor_vel_R, p_R, ent_R, normal)

        # HLL blend (safe against the degenerate SR == SL case).
        denom = wave_speed_right - wave_speed_left
        denom_safe = np.where(denom == 0, 1.0, denom)
        numerator = (
            wave_speed_right[..., None] * flux_L
            - wave_speed_left[..., None] * flux_R
            + wave_speed_left[..., None]
            * wave_speed_right[..., None]
            * (qR[..., :4] - qL[..., :4])
        ) / denom_safe[..., None]
        euler_flux = np.where(denom[..., None] == 0, 0.0, numerator)

        if num_species == 0:
            return euler_flux

        species_flux = self._species_flux(euler_flux[..., 0], qL, qR, rho_L, rho_R)
        return np.concatenate([euler_flux, species_flux], axis=-1)

    def muscl_euler_res2d(self, grid, limiter: str = "MC") -> np.ndarray:
        """Compute residuals for 2D Euler equations using MUSCL-HLLE.

        A genuine 2D HLLE Riemann solver for Euler equations using Monotonic
        Upstream Centered Scheme for Conservation Laws (MUSCL). Mass vectorized
        version using only NumPy array operations.

        Args:
            limiter (str): Slope limiter ('MC', 'MM', 'VA', 'VL').

        Returns:
            np.ndarray: Residuals of shape (ny, nx, nvars) for physical domain.

        Note:
            Original code by Manuel Diaz, NTU, 05.25.2015.
        """
        q = grid.values_gh  # Need ghost cells for MUSCL reconstruction
        dx = grid.lx / grid.nx
        dy = grid.ly / grid.ny
        nvars = grid.nv
        N = (
            grid.nx + 2
        )  # Total cells including ghosts (matches CFDGrid convention where nx=total array size)
        M = grid.ny + 2  # Total cells including ghosts

        # Allocate arrays for all states
        qN = np.zeros((M, N, nvars))
        qS = np.zeros((M, N, nvars))
        qE = np.zeros((M, N, nvars))
        qW = np.zeros((M, N, nvars))
        residual = np.zeros((M, N, nvars))

        # Compute and limit slopes at interior cells (i,j)
        # q[1:-1, 1:-1] are interior cells (indices 2:M-1, 2:N-1 in MATLAB 1-based indexing)
        for k in range(nvars):
            dqw = q[1:-1, 1:-1, k] - q[1:-1, :-2, k]
            dqe = q[1:-1, 2:, k] - q[1:-1, 1:-1, k]
            dqs = q[1:-1, 1:-1, k] - q[:-2, 1:-1, k]
            dqn = q[2:, 1:-1, k] - q[1:-1, 1:-1, k]
            if limiter == "MC":
                dqc_x = (q[1:-1, 2:, k] - q[1:-1, :-2, k]) / 2
                dqdx = SlopeLimiters.minmod([2 * dqw, 2 * dqe, dqc_x])
                dqc_y = (q[2:, 1:-1, k] - q[:-2, 1:-1, k]) / 2
                dqdy = SlopeLimiters.minmod([2 * dqs, 2 * dqn, dqc_y])
            elif limiter == "MM":
                dqdx = SlopeLimiters.minmod([dqw, dqe])
                dqdy = SlopeLimiters.minmod([dqs, dqn])
            elif limiter == "VA":
                dqdx = SlopeLimiters.van_albada(dqw, dqe, dx)
                dqdy = SlopeLimiters.van_albada(dqs, dqn, dy)
            elif limiter == "VL":
                dqdx = SlopeLimiters.van_leer(dqw, dqe)
                dqdy = SlopeLimiters.van_leer(dqs, dqn)
            else:
                raise ValueError(f"Unknown limiter: {limiter}")

            qE[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdx / 2
            qW[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdx / 2
            qN[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdy / 2
            qS[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdy / 2

        # Residuals: x-direction
        qxL = qE[1:-1, 1:-2, :]  # i = 1..M-2, j = 1..N-3
        qxR = qW[1:-1, 2:-1, :]  # i = 1..M-2, j = 2..N-2
        flux_x = self.HLLE1Dflux_vec(qxL, qxR, [1, 0])

        residual[1:-1, 1:-2, :] += flux_x / dx
        residual[1:-1, 2:-1, :] -= flux_x / dx

        # Residuals: y-direction
        qyL = qN[1:-2, 1:-1, :]  # lower state at each interface (i=1..M-3, j=1..N-2)
        qyR = qS[2:-1, 1:-1, :]  # upper state at each interface (i+1=2..M-2, j=1..N-2)
        flux_y = self.HLLE1Dflux_vec(qyL, qyR, [0, 1])

        residual[1:-2, 1:-1, :] += flux_y / dy
        residual[2:-1, 1:-1, :] -= flux_y / dy

        # Set BCs: boundary flux contributions
        # North face (i = M-2, horizontal interface at top boundary)
        qR_N = qS[M - 2, 1:-1, :]  # shape (N-2, nvars)
        qL_N = qR_N
        flux_N = self.HLLE1Dflux_vec(qL_N[None, :, :], qR_N[None, :, :], [0, 1])[
            0
        ]  # shape (N-2, nvars)
        residual[M - 2, 1:-1, :] += flux_N / dy

        # East face (j = N-2, vertical interface at right boundary)
        qR_E = qW[1:-1, N - 2, :]  # shape (M-2, nvars)
        qL_E = qR_E
        flux_E = self.HLLE1Dflux_vec(qL_E[:, None, :], qR_E[:, None, :], [1, 0])[
            :, 0, :
        ]  # shape (M-2, nvars)
        residual[1:-1, N - 2, :] += flux_E / dx

        # South face (i = 1, horizontal interface at bottom boundary)
        qR_S = qN[1, 1:-1, :]  # shape (N-2, nvars)
        qL_S = qR_S
        flux_S = self.HLLE1Dflux_vec(qL_S[None, :, :], qR_S[None, :, :], [0, -1])[
            0
        ]  # shape (N-2, nvars)
        residual[1, 1:-1, :] += flux_S / dy

        # West face (j = 1, vertical interface at left boundary)
        qR_W = qE[1:-1, 1, :]  # shape (M-2, nvars)
        qL_W = qR_W
        flux_W = self.HLLE1Dflux_vec(qL_W[:, None, :], qR_W[:, None, :], [-1, 0])[
            :, 0, :
        ]  # shape (M-2, nvars)
        residual[1:-1, 1, :] += flux_W / dx

        # Return only interior residuals (excluding ghost cells) to match Grid's physical domain
        return residual[1:-1, 1:-1, :]

    def apply_boundary_conditions(self, grid) -> None:
        """Apply boundary conditions to ghost cells based on boundary_config.

        Uses the per-side boundary condition configuration to set ghost cell values.
        TRANSMISSIVE: Zero-gradient (copy from interior).
        REFLECTING: Wall boundary with normal momentum flip.
        """
        q = grid.values_gh
        BC = BoundaryCondition
        bc = self.boundary_config

        # Left boundary (x-direction, affects u-momentum at index 1)
        if bc.left == BC.TRANSMISSIVE:
            q[:, 0, :] = q[:, 1, :]
        elif bc.left == BC.REFLECTING:
            q[:, 0, :] = q[:, 1, :]
            q[:, 0, 1] *= -1  # Flip u-momentum

        # Right boundary
        if bc.right == BC.TRANSMISSIVE:
            q[:, -1, :] = q[:, -2, :]
        elif bc.right == BC.REFLECTING:
            q[:, -1, :] = q[:, -2, :]
            q[:, -1, 1] *= -1  # Flip u-momentum

        # Bottom boundary (y-direction, affects v-momentum at index 2)
        if bc.bottom == BC.TRANSMISSIVE:
            q[0, :, :] = q[1, :, :]
        elif bc.bottom == BC.REFLECTING:
            q[0, :, :] = q[1, :, :]
            q[0, :, 2] *= -1  # Flip v-momentum

        # Top boundary
        if bc.top == BC.TRANSMISSIVE:
            q[-1, :, :] = q[-2, :, :]
        elif bc.top == BC.REFLECTING:
            q[-1, :, :] = q[-2, :, :]
            q[-1, :, 2] *= -1  # Flip v-momentum


    def RK2_step(self, grid, dt: float) -> None:
        """Perform a single RK2 (Heun's method) time step.
        """
        # RK2 1st step
        res1 = self.muscl_euler_res2d(grid)
        q_old = grid.values.copy()  # Store original state before update
        values_1 = q_old - dt * res1
        # Update grid and apply BCs
        grid.values = values_1
        self.apply_boundary_conditions(grid)
        # RK2 2nd step / update q (correct SSP-RK2: both stages weighted by 0.5*dt)
        res2 = self.muscl_euler_res2d(grid)
        grid.values = 0.5 * (q_old + values_1 - dt * res2)
        # Natural BCs again
        self.apply_boundary_conditions(grid)

    def sod_tube_ic(
        self,
        grid,
        gamma: float = 1.4,
        p_high: float = 1.0,
        p_low: float = 0.1,
        rho_high: float = 1.0,
        rho_low: float = 0.125,
    ):
        """1D Sod shock tube initial condition."""
        grid.values[:, :, :] = 0.0  # Clear all values
        grid.values[:, : grid.nx // 2, :] = [
            rho_high,
            0.0,
            0.0,
            p_high / ((gamma - 1)),
        ]  # Left half
        grid.values[:, grid.nx // 2 :, :] = [
            rho_low,
            0.0,
            0.0,
            p_low / ((gamma - 1)),
        ]  # Right half
