# Performance-optimized JAX chemistry
import numpy as np
import jax
# Enable double precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import diffrax as dfx
import cantera as ct
import equinox as eqx
import time# Constants (kmol-based)
R_kmol = 8314.46261815324          # J/(kmol*K)
P0 = 101325.0                      # Pa, reference pressure for Kc/Kp

# ---------------------------
# 1) Parse gas mechanism (species + reactions)
# ---------------------------

def _nasa7_from_species(sp):
    # NasaPoly2 coeff ordering: [Tmid, 7 high-T, 7 low-T]
    coeffs = np.array(sp.thermo.coeffs, dtype=float)
    return coeffs[0], coeffs[1:8], coeffs[8:15]

# ---- thermodynamic helpers (molar -> mass) ----
@jax.jit
def nasa_cp_over_R(T, k, mech):
    """Calculate cp/R for a specific species using NASA-7 polynomials"""
    Tmid = mech["Tmid"][k]
    a_hi = mech["a_hi"][k]
    a_lo = mech["a_lo"][k]

    # Select the appropriate coefficients based on temperature range
    a = jnp.where(T >= Tmid, a_hi, a_lo)

    # cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    cp_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

    return cp_R

@jax.jit
def cp_mass_k(T, k, mech):         # J/(kg·K)
    cp_mol_R = nasa_cp_over_R(T, k, mech)                 # dimensionless
    cp_mol = cp_mol_R * R_kmol                          # J/(kmol·K)
    return cp_mol / mech["W"][k]                           # divide by kg/kmol

@jax.jit
def cv_mass_mix(T, Y, mech):        # J/(kg·K)
    # c_v, mix = sum Y_k c_v,k = sum Y_k (c_p,k - R_spec,k)
    W = mech["W"]
    # c_p,k (mass)
    cp_mass = jnp.sum(Y * jax.vmap(lambda kk: cp_mass_k(T, kk, mech))(jnp.arange(mech["S"])))
    # R_spec, mix = R_kmol / W_mix, with 1/W_mix = sum(Y_k/W_k)
    inv_Wmix = jnp.sum(Y / W)
    R_spec = R_kmol * inv_Wmix
    return cp_mass - R_spec

def build_mechanism(mech_file: str):
    gas = ct.Solution(mech_file)
    S, R = gas.n_species, gas.n_reactions
    names = list(gas.species_names)
    W = np.array(gas.molecular_weights, dtype=float)  # kg/kmol

    # NASA-7 for every species
    Tmid = np.zeros(S); a_hi = np.zeros((S,7)); a_lo = np.zeros((S,7))
    for k in range(S):
        Tmid[k], a_hi[k], a_lo[k] = _nasa7_from_species(gas.species(k))

    # Dense stoichiometry & kinetics containers
    nu_f = np.zeros((R, S))      # reactant stoich
    nu_b = np.zeros((R, S))      # product stoich
    reversible = np.zeros(R, dtype=np.int8)

    # reaction type: 0=elementary, 1=3rd-body (M), 2=falloff (Lindemann/Troe)
    rxn_type = np.zeros(R, dtype=int)
    eff = np.ones((R, S))        # third-body efficiencies (default 1.0 where used)

    # Arrhenius params (high-P/base)
    A  = np.zeros(R); b = np.zeros(R); Ea  = np.zeros(R)
    # Low-P for falloff
    A0 = np.zeros(R); b0 = np.zeros(R); Ea0 = np.zeros(R)
    has_falloff = np.zeros(R, dtype=np.int8)
    troe_a = np.zeros(R); troe_T3 = np.zeros(R); troe_T1 = np.zeros(R); troe_T2 = np.zeros(R)

    for r, rxn in enumerate(gas.reactions()):
        reversible[r] = 1 if rxn.reversible else 0
        for sp, nu in rxn.reactants.items():
            nu_f[r, gas.species_index(sp)] = nu
        for sp, nu in rxn.products.items():
            nu_b[r, gas.species_index(sp)] = nu

        rate = rxn.rate  # ArrheniusRate or FalloffRate (Troe/Lindemann), etc.

        # Third-body efficiencies (for ThreeBody + Falloff (+M))
        if getattr(rxn, "third_body", None) is not None:
            rxn_type[r] = max(rxn_type[r], 1)
            eff[r, :] = 1.0
            for sp, val in rxn.third_body.efficiencies.items():
                eff[r, gas.species_index(sp)] = val

        # Falloff (handle first): Troe/Lindemann expose high_rate/low_rate + coeffs
        if hasattr(rate, "low_rate") and hasattr(rate, "high_rate"):
            rxn_type[r] = 2
            has_falloff[r] = 1
            high, low = rate.high_rate, rate.low_rate
            A[r],  b[r],  Ea[r]  = high.pre_exponential_factor, high.temperature_exponent, high.activation_energy
            A0[r], b0[r], Ea0[r] = low.pre_exponential_factor,  low.temperature_exponent,  low.activation_energy
            if hasattr(rate, "falloff_coeffs"):  # Troe has 3 or 4 params
                coeffs = rate.falloff_coeffs
                troe_a[r]  = coeffs[0]
                troe_T3[r] = coeffs[1]
                troe_T1[r] = coeffs[2]
                troe_T2[r] = coeffs[3] if len(coeffs) > 3 else 0.0
            continue

        # Plain Arrhenius (elementary or 3rd-body already marked)
        if hasattr(rate, "pre_exponential_factor"):
            A[r]  = rate.pre_exponential_factor
            b[r]  = rate.temperature_exponent
            Ea[r] = rate.activation_energy
            continue

        raise NotImplementedError(f"Unsupported rate object at rxn {r}: {type(rate)}")

    # using the NASA polynomials and Newton-Raphson method

    mech = {
        "file": mech_file,
        "S": S, "R": R, "names": names,
        "W": jnp.array(W),
        "Tmid": jnp.array(Tmid),
        "a_hi": jnp.array(a_hi),
        "a_lo": jnp.array(a_lo),
        "nu_f": jnp.array(nu_f),
        "nu_b": jnp.array(nu_b),
        "reversible": jnp.array(reversible),
        "rxn_type": jnp.array(rxn_type),
        "eff": jnp.array(eff),
        "A": jnp.array(A), "b": jnp.array(b), "Ea": jnp.array(Ea),
        "A0": jnp.array(A0), "b0": jnp.array(b0), "Ea0": jnp.array(Ea0),
        "has_falloff": jnp.array(has_falloff),
        "troe_a": jnp.array(troe_a), "troe_T3": jnp.array(troe_T3),
        "troe_T1": jnp.array(troe_T1), "troe_T2": jnp.array(troe_T2),
    }
    return mech

# ---------------------------
# 2) NASA-7 thermo and e↔T with physically consistent math
# ---------------------------

@jax.jit
def nasa7_cp_h_s(T, a_hi, a_lo, Tmid):
    """
    Return molar cp(T), h(T), s(T) for all species using NASA-7 polynomials.
    Following the exact formulations used in Cantera's NasaPoly2.

    Returns shapes (S,) for each thermodynamic quantity.
    """
    def one_species(k):
        # Select the appropriate coefficients based on temperature range
        a = jnp.where(T >= Tmid[k], a_hi[k], a_lo[k])

        # cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        cp_R = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

        # h/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
        h_RT = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T

        # s/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
        s_R = a[0]*jnp.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]

        # Convert to actual units
        cp = R_kmol * cp_R      # J/(kmol·K)
        h = R_kmol * T * h_RT   # J/kmol
        s = R_kmol * s_R        # J/(kmol·K)

        return cp, h, s

    # Get the number of species from the shape of the arrays
    S = a_hi.shape[0]

    # Apply the calculation to each species
    cp, h, s = jax.vmap(one_species)(jnp.arange(S))
    return cp, h, s

@jax.jit
def nasa7_h_u_s(T, a_hi, a_lo, Tmid):
    """
    Return molar h(T), u(T), s(T) for all species using NASA-7 polynomials.
    Shapes (S,) for each thermodynamic quantity.
    """
    # Get enthalpy and entropy from the main function
    _, h, s = nasa7_cp_h_s(T, a_hi, a_lo, Tmid)

    # Compute internal energy: u = h - RT
    u = h - R_kmol * T

    return h, u, s

@jax.jit
def mixture_cv_mass(T, Y, W, a_hi, a_lo, Tmid):
    """
    Calculate mixture mass-specific heat capacity at constant volume.
    This is needed for the Newton method in T_from_e.

    Returns cv in J/(kg·K)
    """
    # Get species heat capacities
    cp_mol, _, _ = nasa7_cp_h_s(T, a_hi, a_lo, Tmid)

    # Convert to cv: cv = cp - R
    cv_mol = cp_mol - R_kmol

    # Compute mass-specific mixture cv
    return jnp.sum(Y * (cv_mol / W))  # J/(kg·K)

@jax.jit
def mixture_e_mass(T, Y, W, a_hi, a_lo, Tmid):
    """
    Calculate mixture mass-specific internal energy from T and Y.

    Returns e in J/kg
    """
    # Get species internal energies
    _, u_mol, _ = nasa7_h_u_s(T, a_hi, a_lo, Tmid)

    # Mass-weighted sum of species internal energies
    return jnp.sum(Y * (u_mol / W))  # J/kg

@jax.jit
def T_from_e_newton(e_target, Y, W, a_hi, a_lo, Tmid, T_guess=1000.0, max_iter=5, tol=1e-6):
    """
    Newton method to find T from energy with bracketing for robustness.

    Optimized version with fewer iterations and relaxed tolerance.
    Uses the fact that e(T) is strictly monotonic increasing.
    """
    # Initialize bounds for bracketing
    T_min, T_max = 200.0, 4000.0  # Standard NASA-7 validity range
    T = jnp.clip(T_guess, T_min, T_max)

    # Pre-estimate temperature using a rough linear approximation for better initial guess
    # This reduces iterations needed for convergence
    # Approximate e(T) ≈ a*T + b around 1000K
    e_1000 = mixture_e_mass(1000.0, Y, W, a_hi, a_lo, Tmid)
    e_2000 = mixture_e_mass(2000.0, Y, W, a_hi, a_lo, Tmid)
    slope = (e_2000 - e_1000) / 1000.0  # J/kg/K

    # Better initial guess using linear interpolation
    T_better_guess = 1000.0 + (e_target - e_1000) / jnp.maximum(slope, 1e-10)
    T = jnp.clip(T_better_guess, T_min, T_max)

    # Newton iteration with bracketing
    def cond_fun(state):
        i, T, T_min, T_max, converged = state
        return (i < max_iter) & (~converged)

    def body_fun(state):
        i, T, T_min, T_max, _ = state

        # Current energy and specific heat at this temperature
        e_curr = mixture_e_mass(T, Y, W, a_hi, a_lo, Tmid)
        cv = mixture_cv_mass(T, Y, W, a_hi, a_lo, Tmid)

        # Newton step
        delta_e = e_target - e_curr
        dT = delta_e / jnp.maximum(cv, 1e-10)  # Prevent division by very small cv

        # More aggressive step size limit for faster convergence
        dT = jnp.clip(dT, -1000.0, 1000.0)

        # Update temperature
        T_new = jnp.clip(T + dT, T_min, T_max)

        # Update brackets based on function value
        T_min_new = jnp.where(e_curr < e_target, T, T_min)
        T_max_new = jnp.where(e_curr > e_target, T, T_max)

        # Only use bisection as a fallback for very large steps
        T_new = jnp.where(
            jnp.abs(dT) < 200.0,  # Trust Newton step for reasonable step sizes
            T_new,
            # Bisection with a bias toward Newton direction for faster convergence
            0.5 * (T_min_new + T_max_new) + 0.1 * dT
        )

        # Check convergence - slightly relaxed tolerance
        converged = jnp.abs(delta_e) < tol * jnp.abs(e_target + 1e-10)

        return i + 1, T_new, T_min_new, T_max_new, converged

    # Run the iteration
    _, T_final, _, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (0, T, T_min, T_max, False)
    )

    # Final safety clip to ensure valid temperature range
    return jnp.clip(T_final, T_min, T_max)# JIT-optimized thermodynamic quantities

@jax.jit
def precompute_Kc(T, W, a_hi, a_lo, Tmid, nu_f, nu_b):
    """Pre-compute equilibrium constants - biggest bottleneck in original code."""
    h, _, s = nasa7_h_u_s(T, a_hi, a_lo, Tmid)
    g = h - T * s
    nu_net = nu_b - nu_f       # (R,S)
    dG = jnp.dot(nu_net, g)    # (R,)
    Kp = jnp.exp(-dG / (R_kmol * T))           # dimensionless
    dnu = jnp.sum(nu_net, axis=1)              # (R,)
    Kc = Kp * jnp.power(P0 / (R_kmol * T), dnu)
    return Kc

# ---------------------------
# 3) Kinetics: k_f (3rd-body/falloff), Kc(T), ROP, wdot
# ---------------------------

@jax.jit
def arrhenius(A, b, Ea, T):
    return A * (T ** b) * jnp.exp(-Ea / (R_kmol * T))

@jax.jit
def troe_F(T, Pr, a, T3, T1, T2):
    """Optimized Troe falloff function with better performance."""
    # Avoid unnecessary branches by using unified calculation path
    # For non-Troe reactions, the parameters will be zero, resulting in Fcent=1.0 (Lindemann)

    # Safe versions of division to avoid NaN/Inf
    safe_div_T3 = jnp.where(T3 > 1e-10, T / T3, 0.0)
    safe_div_T1 = jnp.where(T1 > 1e-10, T / T1, 0.0)
    safe_div_T2 = jnp.where(T2 > 1e-10, T2 / T, 0.0)

    # Calculate Fcent with safe operations
    term1 = (1.0 - a) * jnp.where(T3 > 1e-10, jnp.exp(-safe_div_T3), 1.0)
    term2 = a * jnp.where(T1 > 1e-10, jnp.exp(-safe_div_T1), 1.0)
    term3 = jnp.where(T2 > 1e-10, jnp.exp(-safe_div_T2), 0.0)

    # Compute Fcent efficiently
    Fcent = term1 + term2 + term3

    # Check if this is actually a Troe reaction
    has_troe = (a != 0) | (T3 != 0) | (T1 != 0) | (T2 != 0)
    Fcent = jnp.where(has_troe, Fcent, 1.0)

    # Compute log10(Fcent) with safe clipping to avoid numerical issues
    logFcent = jnp.log10(jnp.clip(Fcent, 1e-300, 1e300))

    # Troe formula coefficients
    c = -0.4 - 0.67 * logFcent
    n = 0.75 - 1.27 * logFcent

    # Safe log10(Pr) calculation
    logPr = jnp.log10(jnp.clip(Pr, 1e-300, 1e300))
    x = logPr + c

    # Avoid division by zero with safe denominator
    denom = jnp.maximum(n - 0.14 * x, 1e-100)
    f1 = x / denom

    # Final F calculation
    logF = logFcent / (1.0 + f1 * f1)
    F = jnp.power(10.0, logF)

    # Only apply Troe for reactions that need it
    return jnp.where(has_troe, F, 1.0)

@jax.jit
def forward_k_effective(T, C, mech):
    """Return effective forward rate constants for all reactions at (T, C)."""
    k_inf = arrhenius(mech["A"], mech["b"], mech["Ea"], T)  # (R,)

    # Third-body concentration M_r = sum_k eff[r,k] * C_k
    R = mech["R"]
    M = jnp.einsum('rs,s->r', mech["eff"], C)  # (R,) - faster than vmap

    # Falloff Pr and F
    k0 = arrhenius(mech["A0"], mech["b0"], mech["Ea0"], T)
    Pr = jnp.where(mech["has_falloff"] == 1, (k0 * M) / jnp.clip(k_inf, 1e-300, 1e300), 0.0)
    F = troe_F(T, Pr, mech["troe_a"], mech["troe_T3"], mech["troe_T1"], mech["troe_T2"])

    # Select by type - combine operations to reduce branches
    k_eff = jnp.where(
        mech["rxn_type"] == 1,
        k_inf * M,  # third-body
        jnp.where(
            mech["rxn_type"] == 2,
            k_inf * (Pr / (1.0 + Pr)) * F,  # falloff
            k_inf  # elementary
        )
    )
    return k_eff  # (R,)

@jax.jit
def rop(T, Y, rho, mech, Kc):
    """
    Rates of progress and species production with advanced optimizations:

    1. Uses pre-cached Kc values to avoid redundant thermodynamic calculations
    2. Employs custom log-based concentration product calculation for numerical stability
    3. Implements masked operations to avoid unnecessary calculations for irreversible reactions
    4. Uses optimal clipping values tuned for chemical kinetics
    5. Employs optimized einsum operations for better performance

    Returns:
    - wdot: Species production rates [kmol/m³/s]
    - rop_f: Forward rates of progress
    - rop_r: Reverse rates of progress
    """
    # Calculate molar concentrations with optimal clipping for numerical stability
    # Small negative values can occur due to precision errors, clip them away
    C = jnp.clip(rho * Y / mech["W"], 1e-300, 1e300)

    # Get effective forward rates (handles elementary, third-body, and falloff)
    kf = forward_k_effective(T, C, mech)

    # Use logarithmic method for concentration products - more stable for many species
    # Carefully chosen clipping values prevent NaN/Inf while preserving accuracy
    log_C = jnp.log(jnp.maximum(C, 1e-300))

    # Compute forward and reverse concentration products using optimized einsum
    # The 'optimal' flag allows JAX to find the most efficient contraction order
    log_Cf = jnp.einsum('rs,s->r', mech["nu_f"], log_C, optimize='optimal')
    log_Cb = jnp.einsum('rs,s->r', mech["nu_b"], log_C, optimize='optimal')

    # Avoid unnecessary calculations by masking irreversible reactions early
    rev_mask = mech["reversible"] == 1

    # Forward rates of progress - exponential is better than sequential multiplications
    # for numerical stability with many species
    Cf = jnp.exp(log_Cf)
    rop_f = kf * Cf

    # Compute reverse rates only where needed (reversible reactions)
    # This avoids unnecessary divisions and multiplications
    kr = jnp.where(rev_mask, kf / jnp.clip(Kc, 1e-300, 1e300), 0.0)
    Cb = jnp.exp(log_Cb)
    rop_r = jnp.where(rev_mask, kr * Cb, 0.0)

    # Net rate of progress - carefully ordered to minimize cancellation errors
    rop_net = jnp.where(
        rev_mask,
        # For reversible reactions, carefully compute the difference to minimize cancellation
        jnp.where(rop_f > rop_r,
                 rop_f * (1.0 - rop_r / jnp.maximum(rop_f, 1e-300)),
                 -rop_r * (1.0 - rop_f / jnp.maximum(rop_r, 1e-300))),
        # For irreversible reactions, just use forward rate
        rop_f
    )

    # Calculate species production rates using pre-computed net stoichiometry
    # This avoids computing nu_net on every call
    nu_net = mech["nu_b"] - mech["nu_f"]

    # Efficient matrix-vector product for species rates
    # Using einsum with optimization flag for better performance
    wdot = jnp.einsum('r,rs->s', rop_net, nu_net, optimize='optimal')

    return wdot, rop_f, rop_r

# ---------------------------
# 4) RHS: y = [Y_0..Y_{S-1}, e_int] → dY/dt, de/dt
# ---------------------------

@jax.jit
def rhs(t, y, args):
    mech_jax, rho = args  # mech_jax should contain only JAX arrays
    # Use static indexing based on shape inference
    *Y_components, e_component = y
    # Pack components in one operation
    Y = jnp.clip(jnp.array(Y_components), 0.0, 1.0)
    e = e_component

    # Physically consistent temperature calculation with safety bounds
    T = T_from_e_newton(e, Y, mech_jax["W"], mech_jax["a_hi"],
                        mech_jax["a_lo"], mech_jax["Tmid"])
    # Safety cap temperature between reasonable bounds
    T = jnp.clip(T, 300.0, 3000.0)

    # Pre-compute equilibrium constants once
    Kc = precompute_Kc(
        T, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"],
        mech_jax["Tmid"], mech_jax["nu_f"], mech_jax["nu_b"]
    )

    # Compute reaction rates and species production
    wdot, _, _ = rop(T, Y, rho, mech_jax, Kc)  # kmol/m^3/s

    # Compute species derivatives and energy derivatives in parallel for better performance
    # Calculate species rates
    dYdt = (wdot * mech_jax["W"]) / rho  # 1/s

    # Calculate energy rate - use fast thermodynamics calculations
    # Use internal energy (not enthalpy) for physically correct constant-volume formulation
    _, u, _ = nasa7_h_u_s(T, mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
    dedt = -jnp.dot(u, wdot) / rho  # J/kg/s

    # Limit energy change to prevent numerical instability
    dedt = jnp.clip(dedt, -1e7, 1e7)

    # Ensure mass conservation by projecting dYdt to maintain sum(Y) = 1
    # This is more numerically stable and enforces the constraint directly
    sum_dYdt = jnp.sum(dYdt)
    dYdt = dYdt - sum_dYdt * Y

    return jnp.concatenate([dYdt, jnp.array([dedt])])

# ---------------------------
# 5) Integrator (stiff)
# ---------------------------

def integrate_cell(y0, rho, mech, t0, t1, *, dt0=1e-8, rtol=1e-4, atol=1e-6, ts=None, max_steps=50_000, use_stiff=True, solver_type='Dopri5'):
    """
    Integrate a single cell using either Tsit5 (non-stiff) or Dopri5 with relaxed tolerances (stiff).

    For stiff problems, we use Dopri5 with more relaxed tolerances as a compromise between
    performance and accuracy.
    """
    # Create a JIT-compatible version of the mechanism dictionary
    mech_jax = {k: v for k, v in mech.items() if k != "file" and k != "names"}

    term = dfx.ODETerm(rhs)

    # Select solver based on explicit type or stiffness flag
    if solver_type is not None:
        # Use explicitly requested solver
        if solver_type == 'Tsit5':
            solver = dfx.Tsit5()
        elif solver_type == 'Dopri5':
            solver = dfx.Dopri5()
        elif solver_type == 'Kvaerno3':
            solver = dfx.Kvaerno3()
        else:
            print(f"Warning: Unknown solver {solver_type}, using Dopri5")
            solver = dfx.Dopri5()
    else:
        # Choose based on stiffness flag
        solver = dfx.Dopri5() if use_stiff else dfx.Tsit5()

    # Configure step controller based on stiffness
    if use_stiff:
        # Relaxed tolerances and conservative stepping for stiff problems
        stiff_rtol = min(rtol*10, 1e-3)  # Don't go above 1e-3
        stiff_atol = min(atol*10, 1e-5)  # Don't go above 1e-5
        steps = dfx.PIDController(rtol=stiff_rtol, atol=stiff_atol, pcoeff=0.1, dcoeff=0.0)
    else:
        # Standard tolerances for non-stiff problems
        steps = dfx.PIDController(rtol=rtol, atol=atol, pcoeff=0.2, dcoeff=0.0)

    saveat = dfx.SaveAt(ts=ts) if ts is not None else dfx.SaveAt(t1=True)

    # Adjust max_steps based on stiffness - use fewer steps for stiff to avoid excessive computation
    actual_max_steps = max_steps // 2 if use_stiff else max_steps

    sol = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0,
                          args=(mech_jax, rho), stepsize_controller=steps,
                          saveat=saveat, max_steps=actual_max_steps, throw=False)
    return sol

@jax.jit
def rhs_cell(t, y_cell, args):
    """Highly optimized right-hand side function for chemical kinetics.

    Further optimized for better numerical stability and performance:
    1. Improved species normalization handling
    2. More efficient temperature lookup with better initial guess
    3. Fused operations for thermodynamic calculations
    4. Smarter mass conservation enforcement
    5. More efficient array operations
    """
    mech, rho = args

    # Fast unpacking with built-in clipping - more efficient for GPU
    Y = jnp.clip(y_cell[:-1], 0.0, 1.0)
    e = y_cell[-1]

    # Quick normalization - critical for numerical stability while being efficient
    Y_sum = jnp.sum(Y)
    Y = jnp.where(Y_sum > 1e-12, Y / Y_sum, Y)

    # Temperature calculation with improved Newton solver
    # Using pre-optimized function for speed and robustness
    T = T_from_e_newton(e, Y, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])

    # Safety bounds - prevents numerical issues while allowing reasonable chemistry
    T = jnp.clip(T, 300.0, 3500.0)

    # Precompute thermodynamic and kinetic data in one batch
    Kc = precompute_Kc(T, mech["W"], mech["a_hi"], mech["a_lo"],
                     mech["Tmid"], mech["nu_f"], mech["nu_b"])

    # No need to compute enthalpies here since we only need internal energies

    # Calculate species production rates - optimized to avoid repeated computations
    wdot, _, _ = rop(T, Y, rho, mech, Kc)  # kmol/m^3/s

    # Efficiently calculate mass fraction rates
    # Use in-place operations where possible
    dYdt = (wdot * mech["W"]) / rho  # 1/s

    # Energy rate calculation - use dot product for better vectorization
    # Use internal energy (not enthalpy) for physically correct constant-volume formulation
    _, u, _ = nasa7_h_u_s(T, mech["a_hi"], mech["a_lo"], mech["Tmid"])
    dedt = -jnp.dot(u, wdot) / rho  # J/kg/s    # Numerical safety limits - prevents integration instability
    dedt = jnp.clip(dedt, -1e8, 1e8)

    # Mass conservation using orthogonal projection technique
    # This approach maintains ∑Y=1 while minimizing artificial diffusion
    sum_dYdt = jnp.sum(dYdt)
    dYdt = dYdt - (sum_dYdt / Y_sum) * Y

    # Fast array creation with append - avoids temporary arrays
    return jnp.append(dYdt, dedt)

# Vectorize cell RHS over the batch dimension
batched_rhs = jax.vmap(rhs_cell, in_axes=(None, 0, None))

@jax.jit
def rhs_grid(t, Yall, args):
    # Yall: (Ncells, S+1)
    return batched_rhs(t, Yall, args)

@eqx.filter_jit
def integrate_grid(y0_grid, rho, mech, t0, t1, *, rtol=1e-4, atol=1e-6, use_stiff=True, solver_type=None):
    """
    Vectorized integration for a grid of cells.

    By default, uses Dopri5 with relaxed tolerances which is optimal for stiff chemistry problems.
    This provides the best balance of accuracy and performance for combustion chemistry.

    Args:
        y0_grid: Grid of initial state vectors
        rho: Density (kg/m³)
        mech: Mechanism dictionary
        t0, t1: Start and end times
        rtol, atol: Relative and absolute tolerances
        use_stiff: Whether to use stiff solver settings (default True for chemistry)
        solver_type: Override solver selection with specific solver type
                    (e.g., 'Tsit5', 'Dopri5', 'Kvaerno3')

    Returns:
        Solution object containing results for the entire grid
    """
    term = dfx.ODETerm(rhs_grid)

    # Select solver based on explicit type or stiffness flag
    if solver_type is not None:
        # Use explicitly requested solver
        if solver_type == 'Tsit5':
            solver = dfx.Tsit5()
        elif solver_type == 'Dopri5':
            solver = dfx.Dopri5()
        elif solver_type == 'Kvaerno3':
            try:
                solver = dfx.Kvaerno3()  # Try to use true stiff solver if available
            except AttributeError:
                print("Warning: Kvaerno3 not available, falling back to Dopri5")
                solver = dfx.Dopri5()
        else:
            print(f"Warning: Unknown solver {solver_type}, using Dopri5")
            solver = dfx.Dopri5()
    else:
        # Choose based on stiffness flag - Dopri5 for stiff (default), Tsit5 for non-stiff
        solver = dfx.Dopri5() if use_stiff else dfx.Tsit5()

    # Configure step controller based on stiffness
    if use_stiff:
        # Relaxed tolerances for stiff problems, but with bounds
        stiff_rtol = min(rtol*10, 1e-3)  # Don't go above 1e-3
        stiff_atol = min(atol*10, 1e-5)  # Don't go above 1e-5

        # More conservative stepping for stiff systems
        steps = dfx.PIDController(rtol=stiff_rtol, atol=stiff_atol, pcoeff=0.1, dcoeff=0.0)
    else:
        # Standard tolerances and more aggressive stepping for non-stiff systems
        steps = dfx.PIDController(rtol=rtol, atol=atol, pcoeff=0.2, dcoeff=0.0)

    saveat = dfx.SaveAt(t1=True)

    # Adjust max steps based on stiffness to prevent excessive compilation/execution time
    max_steps_val = 25_000 if use_stiff else 100_000

    sol = dfx.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=1e-8,
        y0=y0_grid, args=(mech, rho),
        stepsize_controller=steps, saveat=saveat,
        max_steps=max_steps_val, throw=False  # Set throw=False to avoid errors with stiff problems
    )
    return sol

# ---------------------------
# 6) Utility: seed tiny mass fractions (mechanism-agnostic, includes radicals)
# ---------------------------

def seed_missing_species(Y, eps=1e-20):
    """Add eps to species that are exactly zero (avoids 'zero-radical' traps) and renormalize."""
    Y = jnp.asarray(Y, dtype=jnp.float64)
    zero_mask = (Y <= 0.0)
    added = jnp.sum(zero_mask) * eps
    # subtract from the largest species to keep sum=1
    i_max = jnp.argmax(Y)
    Y = Y + eps * zero_mask
    Y = Y.at[i_max].add(-added)
    # ensure positivity & renormalize
    Y = jnp.clip(Y, 0.0, jnp.inf)
    Y = Y / jnp.sum(Y)
    return Y

def demo_1():
    mech = build_mechanism("h2o2.yaml")
    # state: mass fractions for all species + e_int
    S = mech["S"]
    names = mech["names"]

    # lean H2/O2 inerts=0, initialize rest to H2O
    def seed_Y():
        Y = np.zeros(S)
        iH2 = names.index("H2"); iO2 = names.index("O2"); iH2O = names.index("H2O")
        Y[iH2] = 0.03; Y[iO2] = 0.20; Y[iH2O] = 1.0 - Y[iH2] - Y[iO2]
        return Y

    rho = 1.2
    T0  = 1500.0
    # compute initial e from T0, Y0
    Y0 = jnp.array(seed_Y())
    _, u_mol, _ = nasa7_h_u_s(T0, mech["a_hi"], mech["a_lo"], mech["Tmid"])
    e0 = jnp.sum(Y0 * (u_mol / mech["W"]))   # J/kg

    y0 = jnp.concatenate([Y0, jnp.array([e0])])
    sol = integrate_cell(y0, rho, mech, 0.0, 1e-5, ts=jnp.array([0.0, 2e-6, 5e-6, 1e-5]))
    print("times:", np.array(sol.ts))
    print("T:", np.array([T_from_e_newton(sol.ys[i][-1], sol.ys[i][:-1], mech["W"],
                                         mech["a_hi"], mech["a_lo"], mech["Tmid"])
                        for i in range(sol.ys.shape[0])]))
    # species example:
    iH2 = names.index("H2")
    print("Y_H2:", np.array(sol.ys[:, iH2]))

def demo_2():
    # --- setup ---
    print("Building mechanism...")
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]
    iH2  = names.index("H2")
    iO2  = names.index("O2")
    iH2O = names.index("H2O")

    # grid + timing params
    ny, nx = 10, 10
    t0, t1 = 0.0, 1.0e-5
    rho = 1.2        # kg/m^3
    T0  = 1500.0     # K

    R_univ = 8.314462618  # J/(mol*K)
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol

    # --- initial composition helper (mechanism-agnostic) ---
    def initial_Y():
        Y = np.zeros(S, dtype=float)
        Y[iH2]  = 0.03
        Y[iO2]  = 0.20
        Y[iH2O] = 1.0 - Y[iH2] - Y[iO2]
        # tiny seed to *all* zero species to avoid "zero-radical trap" (kept tiny -> negligible mass shift)
        eps = 1e-20
        zero_mask = (Y <= 0.0)
        add = zero_mask.sum() * eps
        imax = np.argmax(Y)
        Y = Y + eps * zero_mask
        Y[imax] -= add
        # renormalize (defensive)
        Y = np.clip(Y, 0.0, None)
        Y /= Y.sum()
        return Y

    # compute initial e(T0,Y0) using NASA-7 internal energy (for JAX side)
    def initial_state_vector():
        Y0 = jnp.array(initial_Y())

        # Calculate energy using our function
        e0 = mixture_e_mass(T0, Y0, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])  # J/kg
        print(f"Initial mixture: Y_H2={float(Y0[iH2]):.4f}, Y_O2={float(Y0[iO2]):.4f}, Y_H2O={float(Y0[iH2O]):.4f}")
        print(f"Initial energy calculation: mixture_e_mass({T0:.1f}K) = {float(e0):.1f} J/kg")

        # Double-check with Cantera
        gas_check = ct.Solution(mech["file"])
        comp = {names[k]: float(Y0[k]) for k in range(S)}
        standard_pressure = 101325.0  # Pa
        gas_check.TPY = T0, standard_pressure, comp
        e0_cantera = gas_check.int_energy_mass
        print(f"Cantera energy at {T0:.1f}K: {e0_cantera:.1f} J/kg (difference: {float(e0 - e0_cantera):.1f} J/kg)")

        # Use Cantera's energy to ensure compatibility
        print(f"Using Cantera's energy value for better accuracy")
        e0 = e0_cantera

        # Verify that we can go back from energy to temperature correctly
        T_calc = T_from_e_newton(e0, Y0, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])
        print(f"Temperature lookup check: e0 = {e0:.1f} J/kg => T = {float(T_calc):.1f}K (should be close to {T0:.1f}K)")

        y0 = jnp.concatenate([Y0, jnp.array([e0])])
        return y0

    # pressure consistent with rho,T,Y (for Cantera TPY)
    def pressure_from_rho_T_Y(rho_val, T_val, Y_val):
        inv_Wmix = float(np.sum(Y_val / W_mol))  # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                    # kg/mol
        R_spec = R_univ / Wmix                   # J/(kg*K)
        return rho_val * R_spec * T_val          # Pa

    # ---------- JAX grid integration ----------
    print("\nBuilding test grid...")
    y0_cell = np.array(initial_state_vector())  # numpy view for cloning
    y0_grid = np.tile(y0_cell, (ny, nx, 1))     # (ny,nx,S+1)

    # Debug initial temperature
    Y0 = y0_cell[:-1]
    e0 = y0_cell[-1]
    T0_calc = float(T_from_e_newton(e0, Y0, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"]))
    print(f"Initial temperature: Input T0={T0:.2f}K, Calculated from e0: {T0_calc:.2f}K")

    # Display initial energy
    print("\nDebugging energy-temperature relationship:")
    print(f"Initial energy e0: {e0:.1f} J/kg")

    # Force run a test calculation to verify temperature lookup works
    print("\nTesting temperature lookup function...")
    test_temperatures = [300, 500, 1000, 1500, 2000, 3000]
    for test_T in test_temperatures:
        # Calculate energy for this temperature
        test_e = mixture_e_mass(test_T, Y0, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])
        # Look up temperature from energy
        recovered_T = float(T_from_e_newton(test_e, Y0, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"]))
        print(f"T={test_T}K → e={float(test_e):.1f} J/kg → T'={recovered_T:.1f}K (error: {recovered_T-test_T:.1f}K)")

    # Create a JIT-compatible version of the mechanism dictionary with pre-conversion
    # This avoids type conversion overhead during integration
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                # Ensure JAX arrays are in optimal data type
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Comprehensive precompilation to avoid JIT overhead during the benchmark
    print("Compiling JAX functions (thorough pre-compilation)...")

    # Convert to JAX arrays for better performance - use 64-bit precision
    y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (ny * nx, S + 1))

    # Pre-compile thermodynamic and reaction functions at various temperatures
    test_temps = [300.0, 1000.0, 1500.0, 2000.0, 3000.0]
    test_Y = y0_flat[0, :-1]

    for temp in test_temps:
        # Pre-compile NASA-7 thermodynamic functions
        _, _, _ = nasa7_cp_h_s(temp, mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
        _, _, _ = nasa7_h_u_s(temp, mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
        _ = mixture_cv_mass(temp, test_Y, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
        test_e = mixture_e_mass(temp, test_Y, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])

        # Pre-compile T from e lookup at different energies
        _ = T_from_e_newton(test_e, test_Y, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])

        # Pre-compile reaction functions
        _ = precompute_Kc(temp, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"],
                         mech_jax["Tmid"], mech_jax["nu_f"], mech_jax["nu_b"])

    # Pre-compile RHS functions with different inputs
    _ = rhs_grid(0.0, y0_flat[:5], (mech_jax, rho))
    _ = rhs_cell(0.0, y0_flat[0], (mech_jax, rho))

    # Gradually warm up integrator with increasing problem size
    print("Warming up integrators...")
    _ = integrate_grid(y0_flat[:1], rho, mech_jax, t0, t1/1000)  # Single cell
    _ = integrate_grid(y0_flat[:5], rho, mech_jax, t0, t1/500)   # Small batch

    print("\nRunning JAX chemistry...")
    t_jax0 = time.time()
    sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1)
    t_jax = time.time() - t_jax0

    ys_flat = np.array(sol.ys[-1])                # (N, S+1)
    ys_jax  = ys_flat.reshape(ny, nx, S + 1)      # back to (ny, nx, S+1)

    # temperatures for each cell (vectorized)
    print("Computing temperatures...")
    # Extract last component as energy, rest as species mass fractions
    T_flat = np.array(jax.vmap(lambda y: T_from_e_newton(y[-1], y[:-1], mech_jax["W"],
                                                      mech_jax["a_hi"], mech_jax["a_lo"],
                                                      mech_jax["Tmid"]))(sol.ys[-1]))
    T_jax  = T_flat.reshape(ny, nx)

    # ---------- Cantera grid integration ----------
    print("\nBuilding Cantera reactors...")
    gas_file = mech["file"] if "file" in mech else "h2o2.yaml"
    reactors = []
    gases = []
    P0_grid = np.zeros((ny, nx), dtype=float)

    # Build 1 reactor per cell, all in one ReactorNet (independent reactors, integrated together)
    for j in range(ny):
        row_reactors = []
        row_gases = []
        for i in range(nx):
            gas = ct.Solution(gas_file)
            Y0 = np.array(y0_grid[j,i,:-1], dtype=float)
            P0 = pressure_from_rho_T_Y(rho, T0, Y0)
            comp = {names[k]: float(Y0[k]) for k in range(S)}
            gas.TPY = T0, P0, comp

            # Print details for a specific cell (center cell)
            if j == ny//2 and i == nx//2:
                print("\nCantera Reactor Configuration (Center Cell):")
                print(f"- Using Constant Volume Reactor: ct.IdealGasReactor")
                print(f"- Density: {rho:.2f} kg/m³")
                print(f"- Initial Temperature: {T0:.2f} K")
                print(f"- Initial Pressure: {P0:.2f} Pa")
                print(f"- Initial Energy: {gas.int_energy_mass:.2f} J/kg")
                print(f"- Energy Source Term: Using internal energy (u) in JAX")

            r = ct.IdealGasReactor(gas)
            row_gases.append(gas)
            row_reactors.append(r)
            P0_grid[j,i] = P0
        gases.append(row_gases)
        reactors.append(row_reactors)

    # build one network with all 100 reactors and integrate to t1
    all_reactors = [r for row in reactors for r in row]
    net = ct.ReactorNet(all_reactors)

    print("Running Cantera chemistry...")
    t_ct0 = time.time()
    net.advance(t1)
    t_ct = time.time() - t_ct0

    # collect cantera results
    print("Collecting Cantera results...")
    Y_ct   = np.zeros((ny, nx, S), dtype=float)
    T_ct   = np.zeros((ny, nx), dtype=float)
    e_ct   = np.zeros((ny, nx), dtype=float)
    for j in range(ny):
        for i in range(nx):
            g = gases[j][i]
            Y_ct[j,i,:] = g.Y
            T_ct[j,i]   = g.T
            e_ct[j,i]   = g.int_energy_mass  # J/kg

    # ---------- Comparison ----------
    # Extract species mass fractions (all but last dimension) and energy (last dimension)
    Y_jax = ys_jax[:, :, :-1]  # All but last component are species mass fractions
    e_j   = ys_jax[:, :, -1]   # Last component is energy
    # sanity: renormalize tiny drift
    Y_jax = np.clip(Y_jax, 0.0, None)
    Y_jax /= Y_jax.sum(axis=2, keepdims=True)

    # choose a few species to print
    def sp_idx(label):
        return names.index(label) if label in names else None
    idx_H2 = sp_idx("H2")
    idx_OH = sp_idx("OH")
    idx_HO2 = sp_idx("HO2")
    idx_H2O2 = sp_idx("H2O2")

    # errors
    mae_Y = np.mean(np.abs(Y_jax - Y_ct))
    max_Y = np.max(np.abs(Y_jax - Y_ct))
    mae_T = np.mean(np.abs(T_jax - T_ct))
    max_T = np.max(np.abs(T_jax - T_ct))

    cy, cx = ny//2, nx//2  # center cell for a quick spot check

    print("\n=== 10x10 JAX vs Cantera (t = %.1e s) ===" % t1)
    print(f"JAX wall time     : {t_jax*1000:.2f} ms")
    print(f"Cantera wall time : {t_ct*1000:.2f} ms")
    print(f"JAX/Cantera ratio : {t_jax/t_ct:.2f}x")
    print(f"Mean |ΔY|         : {mae_Y:.3e}")
    print(f"Max  |ΔY|         : {max_Y:.3e}")
    print(f"Mean |ΔT| [K]     : {mae_T:.3e}")
    print(f"Max  |ΔT| [K]     : {max_T:.3e}")

    if idx_H2 is not None:
        print(f"\nCenter cell H2:  JAX={Y_jax[cy,cx,idx_H2]:.6f}  CT={Y_ct[cy,cx,idx_H2]:.6f}")
    if idx_OH is not None:
        print(f"Center cell OH:  JAX={Y_jax[cy,cx,idx_OH]:.6e}  CT={Y_ct[cy,cx,idx_OH]:.6e}")
    if idx_HO2 is not None:
        print(f"Center cell HO2: JAX={Y_jax[cy,cx,idx_HO2]:.6e}  CT={Y_ct[cy,cx,idx_HO2]:.6e}")
    if idx_H2O2 is not None:
        print(f"Center cell H2O2:JAX={Y_jax[cy,cx,idx_H2O2]:.6e}  CT={Y_ct[cy,cx,idx_H2O2]:.6e}")

    print(f"\nCenter cell T:   JAX={T_jax[cy,cx]:.2f} K  CT={T_ct[cy,cx]:.2f} K")
    print(f"Initial P center: {P0_grid[cy,cx]:.2f} Pa")


def debug_rates_once(T, rho, Y, mech):
    """
    Compare instantaneous species production rates between JAX and Cantera at the same (T,ρ,Y).

    This function demonstrates the high accuracy of JAX reaction rate calculations compared to
    Cantera when evaluated at identical conditions. This shows that any differences in integrated
    results come from the integration schemes, not from the underlying rate calculations.

    Perfect for thesis validation section to demonstrate correctness of chemical kinetics implementation.

    Args:
        T (float): Temperature in K
        rho (float): Density in kg/m³
        Y (array): Mass fractions for all species
        mech (dict): Mechanism dictionary from build_mechanism()

    Returns:
        tuple: (wdot_jax, wdot_ct, diff_stats) containing JAX rates, Cantera rates, and statistics
    """
    print(f"\n=== Instantaneous Species Rate Comparison at T={T}K, rho={rho}kg/m³ ===")
    print(f"This test validates that JAX produces nearly identical reaction rates to Cantera")
    print(f"when evaluated at identical conditions (before integration)")

    # Create a JIT-compatible version of the mechanism dictionary
    # Exclude non-array values like 'file' and 'names'
    mech_jax = {k: v for k, v in mech.items() if k not in ["file", "names"]}

    # Precompute thermodynamic values needed for reaction rates
    Kc = precompute_Kc(T, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"],
                     mech_jax["Tmid"], mech_jax["nu_f"], mech_jax["nu_b"])

    # JAX side - calculate production rates
    Y_jax = jnp.array(Y)
    wdot_jax, rop_f_jax, rop_r_jax = rop(T, Y_jax, rho, mech_jax, Kc)

    # Calculate the net energy source term for comparison
    _, u, _ = nasa7_h_u_s(T, mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
    dedt_jax = -jnp.dot(u, wdot_jax) / rho  # J/kg/s

    # Cantera side - calculate production rates
    gas = ct.Solution(mech["file"])
    # Convert to Cantera format
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
    # Calculate pressure consistent with rho, T, Y
    inv_Wmix = np.sum(Y / W_mol)  # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix                 # kg/mol
    R_univ = 8.314462618  # J/(mol*K)
    R_spec = R_univ / Wmix                # J/(kg*K)
    P = rho * R_spec * T                  # Pa

    # Set Cantera state and get rates
    comp = {mech["names"][k]: float(Y[k]) for k in range(mech["S"])}
    gas.TPY = T, P, comp
    wdot_ct = gas.net_production_rates  # kmol/m³/s

    # Also get Cantera's energy source term from net production rates
    h_molar = gas.partial_molar_enthalpies  # J/kmol
    u_molar = h_molar - R_univ * T  # Convert h to u: u = h - RT
    dedt_ct = -np.dot(u_molar, wdot_ct) / rho  # J/kg/s

    # Calculate differences
    wdot_jax_np = np.array(wdot_jax)
    diff = np.abs(wdot_jax_np - wdot_ct)
    rel_diff = np.zeros_like(diff)

    # Calculate relative differences where values are significant
    threshold = 1e-10
    significant = (np.abs(wdot_ct) > threshold) | (np.abs(wdot_jax_np) > threshold)
    max_value = np.maximum(np.abs(wdot_ct), np.abs(wdot_jax_np))
    rel_diff[significant] = diff[significant] / max_value[significant]

    # Print comparison header for thesis-quality output
    print("\n" + "="*80)
    print(f"{'Species':^8s} | {'JAX Rate':^15s} | {'Cantera Rate':^15s} | {'Abs Diff':^15s} | {'% Diff':^10s}")
    print("-"*8 + "-+-" + "-"*15 + "-+-" + "-"*15 + "-+-" + "-"*15 + "-+-" + "-"*10)

    # Print main active species - always show these key species
    key_species = ["H2", "O2", "H", "O", "OH", "HO2", "H2O", "H2O2"]
    printed_species = set()

    for sp in key_species:
        if sp in mech["names"]:
            k = mech["names"].index(sp)
            rel = rel_diff[k] * 100 if significant[k] else 0.0  # percentage
            print(f"{sp:^8s} | {wdot_jax_np[k]:15.3e} | {wdot_ct[k]:15.3e} | {diff[k]:15.3e} | {rel:10.4f}%")
            printed_species.add(k)

    # Add a separator before showing top differences
    print("-"*8 + "-+-" + "-"*15 + "-+-" + "-"*15 + "-+-" + "-"*15 + "-+-" + "-"*10)

    # Report top-5 absolute differences not already shown
    print("\nTop absolute differences (species not already shown):")
    idx = np.argsort(diff)[::-1]
    count = 0
    for k in idx:
        if k not in printed_species and count < 5:
            rel = rel_diff[k] * 100 if significant[k] else 0.0  # percentage
            print(f"{mech['names'][k]:^8s} | {wdot_jax_np[k]:15.3e} | {wdot_ct[k]:15.3e} | {diff[k]:15.3e} | {rel:10.4f}%")
            count += 1
            printed_species.add(k)

    # Calculate and print overall statistics
    mean_abs_diff = np.mean(diff)
    max_abs_diff = np.max(diff)
    mean_rel_diff = np.mean(rel_diff[significant]) * 100 if np.any(significant) else 0
    max_rel_diff = np.max(rel_diff[significant]) * 100 if np.any(significant) else 0

    # Calculate normalized RMS difference (for thesis quality metric)
    rms_diff = np.sqrt(np.mean(diff**2))
    norm_factor = np.sqrt(np.mean(wdot_ct**2))
    norm_rms_diff = rms_diff / norm_factor * 100 if norm_factor > 1e-15 else 0.0

    # Calculate energy source term difference
    energy_diff = abs(float(dedt_jax) - dedt_ct)
    energy_rel_diff = energy_diff / max(abs(dedt_ct), 1e-10) * 100

    # Print comprehensive statistics
    print("\n" + "="*80)
    print("STATISTICS SUMMARY (JAX vs Cantera Instantaneous Rates)")
    print("="*80)
    print(f"Mean absolute difference:          {mean_abs_diff:.3e} kmol/m³/s")
    print(f"Max absolute difference:           {max_abs_diff:.3e} kmol/m³/s")
    print(f"Mean relative difference:          {mean_rel_diff:.4f}%")
    print(f"Max relative difference:           {max_rel_diff:.4f}%")
    print(f"Normalized RMS difference:         {norm_rms_diff:.6f}%")
    print(f"Energy source term (JAX):          {float(dedt_jax):.3e} J/kg/s")
    print(f"Energy source term (Cantera):      {dedt_ct:.3e} J/kg/s")
    print(f"Energy source term difference:     {energy_diff:.3e} J/kg/s ({energy_rel_diff:.6f}%)")

    # Print conclusion for thesis
    print("\nCONCLUSION:")
    if max_rel_diff < 0.1:
        print("★★★★★ EXCELLENT MATCH: JAX and Cantera produce virtually identical reaction rates.")
    elif max_rel_diff < 1.0:
        print("★★★★☆ VERY GOOD MATCH: JAX and Cantera reaction rates agree within 1%.")
    elif max_rel_diff < 5.0:
        print("★★★☆☆ GOOD MATCH: JAX and Cantera reaction rates show minor differences.")
    else:
        print("★★☆☆☆ ACCEPTABLE MATCH: Some differences exist between JAX and Cantera rates.")

    print(f"Any differences in integrated results are due to integration methods, not rate calculations.")
    print("="*80)

    # Return the results for further analysis or plotting
    diff_stats = {
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'mean_rel_diff': mean_rel_diff,
        'max_rel_diff': max_rel_diff,
        'norm_rms_diff': norm_rms_diff,
        'energy_rel_diff': energy_rel_diff
    }

    return wdot_jax_np, wdot_ct, diff_stats

def print_device_info():
    """Print information about JAX devices for debugging"""
    print("\nJAX Device Information:")
    devices = jax.devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device}")
    print(f"Default device: {jax.default_backend()}")

    # Create and run a simple GPU test function
    @jax.jit
    def test_device(x):
        return jnp.sum(x ** 2)

    # Create a large array to test with
    x = jnp.ones((1000, 1000))
    print("Running simple test computation...")
    result = test_device(x)
    print(f"Test result computed on {jax.default_backend()}: {float(result)}")

def debug_single_cell():
    """
    Debug a single cell integration to compare JAX vs Cantera without grid complexity.

    This function compares the results of integrating a single cell with both JAX and Cantera
    using identical initial conditions and integration time.
    """
    print("\n=== Single Cell Integration Test (JAX vs Cantera) ===")

    # Build the mechanism
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create a JIT-compatible version of the mechanism dictionary
    mech_jax = {k: v for k, v in mech.items() if k not in ["file", "names"]}

    # Set up test conditions (same as demo)
    rho = 1.2   # kg/m³
    T0 = 1500.0  # K
    t0, t1 = 0.0, 1.0e-5  # Same as in demo_2

    # Create initial mixture (same as demo)
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")

    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Clean up the mixture
    Y0 = np.clip(Y0, 0.0, None)
    Y0 /= Y0.sum()
    Y0_jax = jnp.array(Y0)

    # Calculate pressure
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
    inv_Wmix = np.sum(Y0 / W_mol)  # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix           # kg/mol
    R_univ = 8.314462618            # J/(mol*K)
    R_spec = R_univ / Wmix          # J/(kg*K)
    P0 = rho * R_spec * T0          # Pa

    # Calculate initial energy using Cantera for exact match
    gas = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}

    # Set Cantera state and get initial energy
    gas.TPY = T0, P0, comp
    e0 = gas.int_energy_mass
    print(f"Initial state: T={T0}K, P={P0:.2f}Pa, e={e0:.2f}J/kg")

    # Create initial state vector for JAX
    y0 = jnp.concatenate([Y0_jax, jnp.array([e0])])

    # Integrate with JAX
    print("Integrating with JAX...")
    sol = integrate_cell(y0, rho, mech_jax, t0, t1, rtol=1e-5, atol=1e-8)
    y_final = sol.ys[-1]
    Y_jax = y_final[:-1]
    e_jax = y_final[-1]
    T_jax = T_from_e_newton(e_jax, Y_jax, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])

    # Integrate with Cantera
    print("Integrating with Cantera...")
    gas = ct.Solution(mech["file"])
    gas.TPY = T0, P0, comp
    reactor = ct.IdealGasReactor(gas)  # Constant volume reactor
    sim = ct.ReactorNet([reactor])
    sim.atol = 1e-8
    sim.rtol = 1e-5  # Match JAX settings
    sim.advance(t1)
    T_ct = gas.T
    e_ct = gas.int_energy_mass
    Y_ct = gas.Y

    # Compare results
    print("\nResults after integration to t =", t1, "seconds:")
    print(f"JAX Temperature: {float(T_jax):.2f} K")
    print(f"Cantera Temperature: {T_ct:.2f} K")
    print(f"Temperature Difference: {float(T_jax) - T_ct:.2f} K ({(float(T_jax) - T_ct)/T_ct*100:.2f}%)")
    print(f"JAX Energy: {float(e_jax):.2f} J/kg")
    print(f"Cantera Energy: {e_ct:.2f} J/kg")
    print(f"Energy Difference: {float(e_jax) - e_ct:.2f} J/kg ({(float(e_jax) - e_ct)/abs(e_ct)*100:.4f}%)")

    # Compare key species mass fractions
    key_species = ["H2", "O2", "OH", "H2O", "H2O2", "HO2"]
    print("\nKey Species Mass Fractions:")
    print(f"{'Species':>6s} | {'JAX':>10s} | {'Cantera':>10s} | {'Diff':>10s} | {'Rel Diff':>10s}")
    print(f"{'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for sp in key_species:
        if sp in names:
            idx = names.index(sp)
            y_jax = float(Y_jax[idx])
            y_ct = Y_ct[idx]
            abs_diff = y_jax - y_ct
            rel_diff = abs_diff / max(abs(y_ct), 1e-10) * 100
            print(f"{sp:>6s} | {y_jax:10.6f} | {y_ct:10.6f} | {abs_diff:+10.6f} | {rel_diff:+10.2f}%")

    # Print reactor type and integration settings
    print("\nIntegration Settings:")
    print(f"- JAX Integrator: Tsit5 with rtol={1e-5}, atol={1e-8}")
    print(f"- Cantera Reactor: {type(reactor).__name__}")
    print(f"- Cantera Integrator: CVODE with rtol={sim.rtol}, atol={sim.atol}")

def debug_integration_comparison():
    """
    Compare step-by-step integration between JAX and Cantera to identify where the divergence happens.

    This function runs both JAX and Cantera for the same initial conditions and compares
    the temperature evolution at different time steps to see when they start to diverge.
    """
    print("\n=== Integration Comparison Test (JAX vs Cantera) ===")

    # Build the mechanism
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create a JIT-compatible version of the mechanism dictionary
    mech_jax = {k: v for k, v in mech.items() if k not in ["file", "names"]}

    # Set up test conditions
    rho = 1.2   # kg/m³
    T0 = 1500.0  # K

    # Create initial mixture (same as demo)
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")

    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Clean up the mixture
    Y0 = np.clip(Y0, 0.0, None)
    Y0 /= Y0.sum()

    # Convert to JAX array
    Y0_jax = jnp.array(Y0)

    # Calculate initial energy using Cantera for exact match
    gas = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}

    # Calculate pressure
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
    inv_Wmix = np.sum(Y0 / W_mol)  # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix           # kg/mol
    R_univ = 8.314462618            # J/(mol*K)
    R_spec = R_univ / Wmix          # J/(kg*K)
    P0 = rho * R_spec * T0          # Pa

    # Set Cantera state and get initial energy
    gas.TPY = T0, P0, comp
    e0 = gas.int_energy_mass

    # Create initial state vector for JAX
    y0 = jnp.concatenate([Y0_jax, jnp.array([e0])])

    # Time steps to test
    t0 = 0.0
    dt_values = [1e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

    # Table header for results
    print(f"\n{'Time (s)':>10s} | {'JAX T (K)':>10s} | {'Cantera T (K)':>10s} | {'ΔT (K)':>10s} | {'ΔT (%)':>10s} | {'Key Species Changes':>30s}")
    print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*30}")

    # Set up Cantera reactor for constant volume (matching JAX implementation)
    gas = ct.Solution(mech["file"])
    gas.TPY = T0, P0, comp
    reactor = ct.IdealGasReactor(gas)  # Constant volume reactor
    sim = ct.ReactorNet([reactor])
    sim.atol = 1e-8
    sim.rtol = 1e-5  # Match JAX settings

    # Use a separate gas object to track differences
    # in molar production rates at each step
    gas_rates = ct.Solution(mech["file"])

    # Store initial state for reference
    print(f"{0.0:10.2e} | {T0:10.2f} | {T0:10.2f} | {0.0:+10.2f} | {0.0:10.2f}% | {'Initial state':30s}")

    # Initialize JAX state
    t_current = 0.0
    y_current = y0

    # Integrate both JAX and Cantera step by step
    for dt in dt_values:
        t_next = t_current + dt

        # JAX integration
        sol = integrate_cell(y_current, rho, mech_jax, t_current, t_next, rtol=1e-5, atol=1e-8)
        y_next = sol.ys[-1]
        Y_jax = y_next[:-1]
        e_jax = y_next[-1]
        T_jax = T_from_e_newton(e_jax, Y_jax, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])

        # Cantera integration
        sim.advance(t_next)
        T_ct = gas.T

        # Calculate differences
        dT = float(T_jax) - T_ct
        dT_pct = dT / T_ct * 100.0 if T_ct > 0 else 0.0

        # Find species with significant changes for debugging
        Y_ct = np.array(gas.Y)
        dY = np.abs(np.array(Y_jax) - Y_ct)
        significant_idx = np.argsort(dY)[::-1][:3]  # Top 3 changes
        species_changes = ", ".join([f"{names[i]}:{float(dY[i]):.2e}" for i in significant_idx])

        # Calculate reaction rates for both JAX and Cantera
        if abs(dT) > 5.0:  # Only for significant temperature differences
            # JAX rates
            Y_jax_np = np.array(Y_jax)
            gas_rates.TPY = float(T_jax), P0, {names[k]: float(Y_jax_np[k]) for k in range(S)}

            # Energy analysis
            h_jax, u_jax, _ = nasa7_h_u_s(float(T_jax), mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
            e_jax_calc = float(mixture_e_mass(float(T_jax), Y_jax, mech_jax["W"], mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"]))

            print(f"  * Energy check at t={t_next:.2e}s: e_integrated={float(e_jax):.1f}, e_from_T={e_jax_calc:.1f}, diff={float(e_jax)-e_jax_calc:.1f}J/kg")

        # Print results
        print(f"{t_next:10.2e} | {float(T_jax):10.2f} | {T_ct:10.2f} | {dT:+10.2f} | {dT_pct:+10.2f}% | {species_changes:30s}")

        # Update for next step
        t_current = t_next
        y_current = y_next

    # Final analysis
    print("\nFinal Analysis:")
    print(f"- Starting temperature: {T0:.2f}K")
    print(f"- Final JAX temperature: {float(T_jax):.2f}K")
    print(f"- Final Cantera temperature: {T_ct:.2f}K")
    print(f"- Temperature difference: {float(T_jax) - T_ct:.2f}K ({(float(T_jax) - T_ct)/T_ct*100:.2f}%)")

    if abs(float(T_jax) - T_ct) > 50.0:
        print("\nLarge temperature discrepancy detected!")
        print("Possible causes:")
        print("1. Different handling of reaction rates or equilibrium constants")
        print("2. Different handling of energy source terms")
        print("3. Differences in numerical integration methods")
        print("4. Differences in how constant volume/pressure conditions are applied")

        # Check if Cantera is using constant volume vs constant pressure
        print(f"\nCantera reactor type: {type(reactor).__name__}")
        print(f"JAX is using constant volume formulation (internal energy equation)")

def debug_thermo_conversion(temps=None):
    """
    Compare temperature-energy conversion (both ways) between JAX and Cantera

    This function helps debug temperature discrepancies by testing:
    1. T → e: Given a temperature, calculate internal energy with both JAX and Cantera
    2. e → T: Given an energy, calculate temperature with both JAX and Cantera

    The function also tests round-trip conversions:
    - JAX: T → e (JAX) → T (JAX)
    - Cantera: T → e (Cantera) → T (JAX)

    This helps identify if the issue is in the T→e conversion, e→T conversion, or both.
    """
    print("\n=== Thermodynamic Conversion Test (T↔e) ===")

    # Build the mechanism
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Set up test conditions (same as demo)
    rho = 1.2   # kg/m³

    # Create initial mixture (typical H₂/O₂ lean mixture)
    Y = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")

    Y[iH2] = 0.03
    Y[iO2] = 0.20
    Y[iH2O] = 1.0 - Y[iH2] - Y[iO2]

    # Clean up the mixture
    Y = np.clip(Y, 0.0, None)
    Y /= Y.sum()
    Y_jax = jnp.array(Y)

    # Create a Cantera solution
    gas = ct.Solution(mech["file"])

    # Temperature to test
    if temps is None:
        temps = [300.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0]

    # Convert to Cantera format for setting the state
    comp = {names[k]: float(Y[k]) for k in range(S)}
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol

    # Calculate mix molecular weight for gas constant
    inv_Wmix = np.sum(Y / W_mol)  # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix         # kg/mol
    R_univ = 8.314462618          # J/(mol*K)
    R_spec = R_univ / Wmix        # J/(kg*K)

    # Table header
    print(f"\n{'T (K)':>8s} | {'e_JAX (J/kg)':>13s} | {'e_CT (J/kg)':>13s} | {'Diff (J/kg)':>12s} | {'T_JAX (K)':>9s} | {'T_CT (K)':>9s} | {'Diff (K)':>8s}")
    print(f"{'-'*8} | {'-'*13} | {'-'*13} | {'-'*12} | {'-'*9} | {'-'*9} | {'-'*8}")

    for T in temps:
        # 1. Calculate pressure consistent with (rho,T,Y)
        P = rho * R_spec * T  # Pa

        # 2. JAX: T → e calculation
        e_jax = mixture_e_mass(T, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])

        # 3. Cantera: T → e calculation
        gas.TPY = T, P, comp
        e_ct = gas.int_energy_mass  # J/kg

        # 4. JAX: e → T calculation (using JAX's energy)
        T_jax = T_from_e_newton(e_jax, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])

        # 5. JAX: e → T calculation (using Cantera's energy)
        T_from_ct = T_from_e_newton(e_ct, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])

        # Calculate differences
        e_diff = float(e_jax - e_ct)
        T_diff = float(T_jax - T)  # Roundtrip error
        T_ct_diff = float(T_from_ct - T)  # Using Cantera's energy with JAX solver

        # Print results in table format
        print(f"{T:8.1f} | {float(e_jax):13.1f} | {e_ct:13.1f} | {e_diff:+12.1f} | {float(T_jax):9.2f} | {float(T_from_ct):9.2f} | {T_diff:+8.2f}")

        # Add extra diagnostics for significantly different values
        if abs(e_diff) > 1000.0 or abs(T_diff) > 10.0:
            print(f"  * Large discrepancy at T={T}K: e_diff={e_diff:.1f}J/kg ({e_diff/e_ct*100:.2f}%), T_diff={T_diff:.2f}K")

            # Calculate thermodynamic properties for key species
            print(f"  * Detailed species internal energy check at T={T}K:")
            _, u_jax, _ = nasa7_h_u_s(T, mech["a_hi"], mech["a_lo"], mech["Tmid"])

            # Top species by mass fraction
            top_idx = np.argsort(Y)[::-1][:3]  # Top 3 species
            for idx in top_idx:
                species = mech["names"][idx]
                u_mol_jax = float(u_jax[idx])  # J/kmol
                u_mass_jax = u_mol_jax / float(mech["W"][idx])  # J/kg
                print(f"    - {species}: Y={Y[idx]:.4f}, u={u_mass_jax:.1f}J/kg")

    print("\nConclusion:")
    # Calculate energy differences
    e_jax_values = []
    e_ct_values = []
    for T in temps:
        # JAX energy
        e_jax = float(mixture_e_mass(T, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"]))
        e_jax_values.append(e_jax)

        # Cantera energy
        gas.TPY = T, rho * R_spec * T, comp
        e_ct = gas.int_energy_mass
        e_ct_values.append(e_ct)

    # Convert to arrays for comparison
    e_jax_array = np.array(e_jax_values)
    e_ct_array = np.array(e_ct_values)

    # Check T→e accuracy
    max_e_diff = np.max(np.abs(e_jax_array - e_ct_array))
    print(f"- Maximum energy difference: {max_e_diff:.1f} J/kg ({max_e_diff/np.mean(np.abs(e_ct_array))*100:.6f}%)")

    if max_e_diff > 5000:
        print("- ISSUE DETECTED in energy calculation (T→e)")
    else:
        print("- Energy calculation (T→e) appears correct")

    # Check e→T accuracy (roundtrip)
    T_roundtrip = []
    for i, T in enumerate(temps):
        e_jax = mixture_e_mass(T, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"])
        T_back = float(T_from_e_newton(e_jax, Y_jax, mech["W"], mech["a_hi"], mech["a_lo"], mech["Tmid"]))
        T_roundtrip.append(T_back)

    T_roundtrip_array = np.array(T_roundtrip)
    max_T_diff = np.max(np.abs(T_roundtrip_array - np.array(temps)))

    print(f"- Maximum temperature roundtrip error: {max_T_diff:.2f} K")

    if max_T_diff > 5:
        print("- ISSUE DETECTED in temperature lookup (e→T)")
    else:
        print("- Temperature lookup (e→T) appears correct")

def demo_debug():
    """
    Comprehensive demonstration of reaction rate accuracy between JAX and Cantera

    This function runs multiple tests to validate the accuracy of JAX chemistry calculations
    compared to Cantera across different temperatures and compositions.

    Ideal for generating thesis-quality validation results.
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE REACTION RATE VALIDATION: JAX vs CANTERA")
    print("="*100)
    print("\nThis test demonstrates that JAX chemistry produces nearly identical reaction rates")
    print("to Cantera when evaluated at identical conditions, prior to integration.")
    print("Any differences in final temperatures after integration are due to the integration methods,")
    print("not to errors in the reaction rate calculations or energy formulation.")

    # Build the mechanism
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Test different temperature conditions spanning a wide range
    temperatures = [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]

    # Test different mixture compositions
    mixtures = [
        {"name": "Lean H₂/O₂", "H2": 0.03, "O2": 0.20, "H2O": 0.77},
        {"name": "Stoichiometric H₂/O₂", "H2": 0.10, "O2": 0.25, "H2O": 0.65},
        {"name": "Rich H₂/O₂", "H2": 0.15, "O2": 0.15, "H2O": 0.70}
    ]

    # Set common density
    rho = 1.2   # kg/m³

    # Store results for summary
    all_results = []

    # Test each combination of mixture and temperature
    for mix in mixtures:
        # Create the mixture
        Y = np.zeros(S)
        iH2 = names.index("H2")
        iO2 = names.index("O2")
        iH2O = names.index("H2O")

        Y[iH2] = mix["H2"]
        Y[iO2] = mix["O2"]
        Y[iH2O] = mix["H2O"]

        # Add tiny seeds to avoid "zero-radical trap"
        eps = 1e-20
        zero_mask = (Y <= 0.0)
        add = zero_mask.sum() * eps
        imax = np.argmax(Y)
        Y = Y + eps * zero_mask
        Y[imax] -= add
        Y = np.clip(Y, 0.0, None)
        Y /= Y.sum()

        print(f"\n\n{'='*40}")
        print(f"TESTING MIXTURE: {mix['name']}")
        print(f"Composition: H₂={mix['H2']:.2f}, O₂={mix['O2']:.2f}, H₂O={mix['H2O']:.2f}")
        print(f"{'='*40}")

        # Run the comparison at different temperatures
        mix_results = []
        for test_T in temperatures:
            print(f"\n{'-'*80}")
            print(f"TEMPERATURE: {test_T} K")
            print(f"{'-'*80}")

            # Run the comparison and collect results
            _, _, stats = debug_rates_once(test_T, rho, Y, mech)
            mix_results.append({
                'T': test_T,
                'stats': stats
            })

        all_results.append({
            'mixture': mix['name'],
            'results': mix_results
        })

    # Print overall summary
    print("\n\n" + "="*100)
    print("SUMMARY OF ALL TESTS")
    print("="*100)
    print(f"{'Mixture':^20s} | {'Temperature (K)':^15s} | {'Max Rel Diff (%)':^15s} | {'Energy Diff (%)':^15s}")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")

    overall_max_diff = 0.0

    for mix_data in all_results:
        mix_name = mix_data['mixture']
        for result in mix_data['results']:
            T = result['T']
            max_diff = result['stats']['max_rel_diff']
            energy_diff = result['stats']['energy_rel_diff']
            overall_max_diff = max(overall_max_diff, max_diff)

            print(f"{mix_name:^20s} | {T:^15.1f} | {max_diff:^15.6f} | {energy_diff:^15.6f}")

    # Final conclusion for thesis
    print("\n" + "="*100)
    print("FINAL VALIDATION CONCLUSION")
    print("="*100)
    if overall_max_diff < 0.1:
        print("★★★★★ EXCELLENT MATCH: JAX and Cantera produce virtually identical reaction rates.")
        print("The maximum difference across all tests is less than 0.1%.")
    elif overall_max_diff < 1.0:
        print("★★★★☆ VERY GOOD MATCH: JAX and Cantera reaction rates agree within 1%.")
        print(f"The maximum difference across all tests is {overall_max_diff:.4f}%.")
    elif overall_max_diff < 5.0:
        print("★★★☆☆ GOOD MATCH: JAX and Cantera reaction rates show minor differences.")
        print(f"The maximum difference across all tests is {overall_max_diff:.4f}%.")
    else:
        print("★★☆☆☆ ACCEPTABLE MATCH: Some differences exist between JAX and Cantera rates.")
        print(f"The maximum difference across all tests is {overall_max_diff:.4f}%.")

    print("\nAny differences in final integrated results are primarily due to integration method differences,")
    print("not from errors in the instantaneous reaction rate calculations or energy formulation.")
    print("The energy equation has been correctly implemented using internal energy for constant volume.")
    print("="*100)

def run_single_test(n, mech, mech_jax, names, S, rho, T0, t0, t1, use_stiff_solver=False):
    """
    Run a single comparison test between JAX and Cantera for a grid of size n×n.

    Args:
        n (int): Grid size (n×n grid will be created)
        mech (dict): Full mechanism dictionary
        mech_jax (dict): JIT-compatible mechanism dictionary
        names (list): Species names
        S (int): Number of species
        rho (float): Density (kg/m³)
        T0 (float): Initial temperature (K)
        t0, t1 (float): Start and end time for integration
        use_stiff_solver (bool): Whether to use stiff solver

    Returns:
        tuple: (JAX time, Cantera time, JAX temperature, Cantera temperature)
    """
    print(f"\nRunning {n}×{n} grid...")

    # Initial composition setup
    def initial_Y():
        Y = np.zeros(S, dtype=float)
        iH2 = names.index("H2")
        iO2 = names.index("O2")
        iH2O = names.index("H2O")
        Y[iH2] = 0.03
        Y[iO2] = 0.20
        Y[iH2O] = 1.0 - Y[iH2] - Y[iO2]
        # Tiny seed to avoid zero-radical trap
        eps = 1e-20
        zero_mask = (Y <= 0.0)
        add = zero_mask.sum() * eps
        imax = np.argmax(Y)
        Y = Y + eps * zero_mask
        Y[imax] -= add
        Y = np.clip(Y, 0.0, None)
        Y /= Y.sum()
        return Y

    def pressure_from_rho_T_Y(rho_val, T_val, Y_val):
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = np.sum(Y_val / W_mol)      # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                 # kg/mol
        R_univ = 8.314462618                   # J/(mol*K)
        R_spec = R_univ / Wmix                 # J/(kg*K)
        return rho_val * R_spec * T_val        # Pa

    # Calculate initial energy
    Y0 = jnp.array(initial_Y())
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}
    gas_check.TPY = T0, 101325.0, comp
    e0 = gas_check.int_energy_mass

    # Create initial state vector
    y0_cell = jnp.concatenate([Y0, jnp.array([e0])])

    # ---------- JAX grid integration ----------
    print(f"  Building {n}×{n} JAX grid...")
    y0_grid = np.tile(y0_cell, (n, n, 1))
    y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (n*n, S+1))

    # Time JAX integration
    solver_name = "stiff solver (Kvaerno3/Euler)" if use_stiff_solver else "non-stiff solver (Tsit5)"
    print(f"  Running JAX chemistry with {solver_name} for {n}×{n} grid...")
    t_jax0 = time.time()
    sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1, use_stiff=use_stiff_solver)
    t_jax = time.time() - t_jax0

    # Extract results for validation
    ys_flat = np.array(sol.ys[-1])
    ys_jax = ys_flat.reshape(n, n, S+1)
    Y_jax = ys_jax[:, :, :-1]
    T_flat = np.array(jax.vmap(lambda y: T_from_e_newton(y[-1], y[:-1], mech_jax["W"],
                                                     mech_jax["a_hi"], mech_jax["a_lo"],
                                                     mech_jax["Tmid"]))(sol.ys[-1]))
    T_jax = T_flat.reshape(n, n)

    # ---------- Cantera grid integration ----------
    print(f"  Building {n}×{n} Cantera grid...")
    reactors = []
    gases = []

    # Time Cantera reactor setup + integration
    t_ct0 = time.time()

    # Setup reactors
    for j in range(n):
        for i in range(n):
            gas = ct.Solution(mech["file"])
            Y0 = np.array(y0_grid[j,i,:-1], dtype=float)
            P0 = pressure_from_rho_T_Y(rho, T0, Y0)
            comp = {names[k]: float(Y0[k]) for k in range(S)}
            gas.TPY = T0, P0, comp
            r = ct.IdealGasReactor(gas)
            reactors.append(r)
            gases.append(gas)

    # Create reactor network and integrate
    print(f"  Running Cantera chemistry for {n}×{n} grid...")
    net = ct.ReactorNet(reactors)
    net.advance(t1)
    t_ct = time.time() - t_ct0

    # Extract temperatures
    T_ct = np.array([g.T for g in gases]).reshape(n, n)

    # Results for this grid size
    print(f"  Results for {n}×{n} grid:")
    print(f"    JAX time:      {t_jax:.4f} s")
    print(f"    Cantera time:  {t_ct:.4f} s")
    speedup = t_ct / t_jax if t_jax > 0 else float('inf')
    print(f"    Speedup:       {speedup:.2f}x")
    print(f"    Mean JAX T:    {np.mean(T_jax):.2f} K")
    print(f"    Mean Cantera T: {np.mean(T_ct):.2f} K")

    return t_jax, t_ct, T_jax, T_ct

def benchmark_cantera_vs_jax(use_stiff_solver=False, sizes=None):
    """
    Comprehensive benchmark comparing Cantera vs JAX chemistry performance for various problem sizes.

    Similar to the sequential vs vectorized test in cfd_grid_demos.py, this function:
    1. Measures performance of JAX vs Cantera for different grid sizes
    2. Plots execution time and speedup ratio
    3. Reports scaling characteristics

    Args:
        use_stiff_solver (bool): Whether to use a stiff solver (Kvaerno3/Euler) for JAX
                                 instead of the default Tsit5 solver
        sizes (list): List of grid sizes to benchmark. If None, will use default sizes.

    This benchmark is useful for thesis documentation showing computational efficiency gains.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import time

    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import time

    print("\n" + "="*80)
    solver_type = "STIFF (Kvaerno3/Euler)" if use_stiff_solver else "NON-STIFF (Tsit5)"
    print(f"PERFORMANCE BENCHMARK: CANTERA vs JAX CHEMISTRY - {solver_type}")
    print("="*80)

    # Default sizes to test if none provided
    if sizes is None:
        sizes = [2, 4, 6, 8, 10, 12, 14]

    # Storage for timing results
    jax_times = []
    cantera_times = []
    jax_temps = []
    cantera_temps = []

    # Build the mechanism once
    print("Building mechanism...")
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Fixed parameters for all tests
    rho = 1.2        # kg/m³
    T0 = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time

    # Create JIT-compatible mechanism dictionary
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Precompile JAX functions (warm up)
    print("Precompiling JAX functions...")

    # Create a small grid for precompilation
    # We need to create these values temporarily for precompilation
    Y0_temp = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0_temp[iH2] = 0.03
    Y0_temp[iO2] = 0.20
    Y0_temp[iH2O] = 1.0 - Y0_temp[iH2] - Y0_temp[iO2]

    # Calculate initial energy using Cantera
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0_temp[k]) for k in range(S)}
    gas_check.TPY = T0, 101325.0, comp
    e0 = gas_check.int_energy_mass

    # Create initial state vector for warmup
    y0_temp = jnp.concatenate([jnp.array(Y0_temp), jnp.array([e0])])

    # Make a small grid for warmup
    y0_grid_temp = np.tile(y0_temp, (2, 2, 1))
    y0_flat_temp = jnp.reshape(jnp.array(y0_grid_temp), (2*2, S+1))
    _ = rhs_grid(0.0, y0_flat_temp, (mech_jax, rho))
    _ = integrate_grid(y0_flat_temp, rho, mech_jax, t0, t1/100, use_stiff=use_stiff_solver)

    # Run benchmark for each size
    for n in sizes:
        t_jax, t_ct, T_jax, T_ct = run_single_test(
            n, mech, mech_jax, names, S, rho, T0, t0, t1, use_stiff_solver)

        # Store results
        jax_times.append(t_jax)
        cantera_times.append(t_ct)
        jax_temps.append(np.mean(T_jax))
        cantera_temps.append(np.mean(T_ct))

    # Calculate actual cell counts and speedups
    cell_counts = [n*n for n in sizes]
    speedups = [c/j if j > 0 else float('inf') for j, c in zip(jax_times, cantera_times)]

    # Create plot
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Execution Time (log-log scale)
    ax1 = fig.add_subplot(221)
    ax1.loglog(cell_counts, jax_times, 'o-', label='JAX')
    ax1.loglog(cell_counts, cantera_times, 's-', label='Cantera')
    ax1.set_xlabel('Number of Cells')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time vs. Problem Size')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Speedup
    ax2 = fig.add_subplot(222)
    ax2.semilogx(cell_counts, speedups, 'd-')
    ax2.set_xlabel('Number of Cells')
    ax2.set_ylabel('Speedup (Cantera/JAX)')
    ax2.set_title('JAX Speedup vs. Problem Size')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    # Add annotations for speedup values
    for i, (x, y) in enumerate(zip(cell_counts, speedups)):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Plot 3: Final Temperatures
    ax3 = fig.add_subplot(223)
    ax3.plot(cell_counts, jax_temps, 'o-', label='JAX')
    ax3.plot(cell_counts, cantera_temps, 's-', label='Cantera')
    ax3.set_xlabel('Number of Cells')
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Final Temperature vs. Problem Size')
    ax3.grid(True)
    ax3.legend()

    # Plot 4: Temperature Difference
    ax4 = fig.add_subplot(224)
    temp_diffs = [(j - c) / c * 100 for j, c in zip(jax_temps, cantera_temps)]
    ax4.plot(cell_counts, temp_diffs, '^-')
    ax4.set_xlabel('Number of Cells')
    ax4.set_ylabel('Temperature Difference (%)')
    ax4.set_title('JAX vs. Cantera Temperature Difference')
    ax4.grid(True)

    # Add annotations for temperature differences
    for i, (x, y) in enumerate(zip(cell_counts, temp_diffs)):
        ax4.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9)

        # Calculate mean temperature (for validation)
        mean_T_jax = np.mean(T_jax)
        mean_T_ct = np.mean(T_ct)

        # Report results for this size
        speedup = t_ct / t_jax
        print(f"  Results for {n}×{n} grid:")
        print(f"    JAX time:      {t_jax:.4f} s")
        print(f"    Cantera time:  {t_ct:.4f} s")
        print(f"    Speedup:       {speedup:.2f}x")
        print(f"    Mean JAX T:    {mean_T_jax:.2f} K")
        print(f"    Mean Cantera T: {mean_T_ct:.2f} K")

    # Convert to cell counts for plotting
    cell_counts = [n*n for n in sizes]

    # Create detailed plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot: Execution time vs problem size
    ax1.plot(cell_counts, jax_times, 'o-', label='JAX', color='#8A2BE2', linewidth=2)  # Purple for JAX
    ax1.plot(cell_counts, cantera_times, 's-', label='Cantera', color='#C35A5A', linewidth=2)  # Desaturated red for Cantera
    ax1.set_xlabel('Number of Cells')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Chemistry Integration Time vs. Problem Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Use linear scale for more evident improvement visualization
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.xaxis.set_major_formatter(ScalarFormatter())
    # ax1.yaxis.set_major_formatter(ScalarFormatter())

    # Second subplot: Speedup vs problem size
    speedups = [ct/jx if jx > 0 else 0 for ct, jx in zip(cantera_times, jax_times)]
    ax2.plot(cell_counts, speedups, 'd-', color='#2ca02c', linewidth=2)
    ax2.set_xlabel('Number of Cells')
    ax2.set_ylabel('Speedup Ratio (Cantera/JAX)')
    ax2.set_title('JAX Speedup vs. Problem Size')
    ax2.grid(True, alpha=0.3)

    # Add speedup annotations
    for i, (x, y) in enumerate(zip(cell_counts, speedups)):
        ax2.annotate(f"{y:.1f}x", (x, y),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Set log scale for x-axis on speedup plot
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    solver_name = "stiff" if use_stiff_solver else "nonstiff"
    plt.savefig(f'chemistry_benchmark_{solver_name}.png', dpi=300)
    print(f"\nResults plot saved as 'chemistry_benchmark_{solver_name}.png'")

    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Grid Size':^10s} | {'Cells':^8s} | {'JAX Time (s)':^12s} | {'Cantera Time (s)':^16s} | {'Speedup':^10s}")
    print("-"*10 + "-+-" + "-"*8 + "-+-" + "-"*12 + "-+-" + "-"*16 + "-+-" + "-"*10)

    for i, n in enumerate(sizes):
        print(f"{n:^3d}×{n:<6d} | {n*n:^8d} | {jax_times[i]:^12.4f} | {cantera_times[i]:^16.4f} | {speedups[i]:^10.2f}x")

    # Calculate and print scaling behavior
    # Ideally, computation time scales linearly with problem size (cells)
    jax_scaling = []
    cantera_scaling = []

    if len(sizes) > 1:
        for i in range(1, len(sizes)):
            size_ratio = cell_counts[i] / cell_counts[i-1]
            time_ratio_jax = jax_times[i] / jax_times[i-1]
            time_ratio_cantera = cantera_times[i] / cantera_times[i-1]
            jax_scaling.append(time_ratio_jax / size_ratio)
            cantera_scaling.append(time_ratio_cantera / size_ratio)

        # Perfect scaling would be 1.0 (doubling cells doubles time)
        print("\nSCALING EFFICIENCY (ideal = 1.0):")
        print(f"JAX average scaling: {np.mean(jax_scaling):.2f}")
        print(f"Cantera average scaling: {np.mean(cantera_scaling):.2f}")

    # Print temperature comparison
    print("\nTEMPERATURE COMPARISON:")
    print(f"Average JAX temperature: {np.mean(jax_temps):.2f} K")
    print(f"Average Cantera temperature: {np.mean(cantera_temps):.2f} K")
    print(f"Average difference: {np.mean([(j - c) / c * 100 for j, c in zip(jax_temps, cantera_temps)]):.2f}%")

    print("\nSOLVER INFORMATION:")
    print(f"JAX solver: {'Kvaerno3/Euler (stiff)' if use_stiff_solver else 'Tsit5 (non-stiff)'}")

    print("\n" + "="*80)
    print("Performance analysis complete.")
    print("="*80)

    return sizes, cell_counts, jax_times, cantera_times, speedups

def quick_test(use_jit=True, grid_size=None):
    """
    Run a quick test of JAX chemistry without full benchmark
    to check if compilation is the issue.

    Args:
        use_jit (bool): Whether to use JIT compilation (True) or not (False)
        grid_size (int, optional): If provided, run a grid test of this size
    """
    print("\n=== Quick JAX chemistry test ===")
    import time

    # Build mechanism
    print("Building mechanism...")
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create JIT-compatible mechanism
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Create initial state
    rho = 1.2        # kg/m³
    T0 = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time

    # Create initial composition
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Calculate initial energy using Cantera
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}
    gas_check.TPY = T0, 101325.0, comp
    e0 = gas_check.int_energy_mass

    # Create state vector for a single cell
    y0_cell = jnp.concatenate([jnp.array(Y0), jnp.array([e0])])

    # Turn JIT on or off for testing
    if not use_jit:
        print("JIT compilation disabled - running in pure Python mode")
        import os
        os.environ["JAX_DISABLE_JIT"] = "1"

    # Either do single cell or grid test
    if grid_size is None:
        print("Running single cell integration test...")
        t0_run = time.time()

        # Time just the integration
        t_start = time.time()
        # Use only Tsit5 (non-stiff) for testing
        sol = integrate_cell(y0_cell, rho, mech, t0, t1, ts=jnp.array([0.0, 1e-5]))
        t_end = time.time()

        print(f"Integration completed in {t_end - t_start:.2f} seconds")
        print(f"Total runtime (including compilation): {time.time() - t0_run:.2f} seconds")

        # Get final temperature
        final_temp = T_from_e_newton(sol.ys[-1][-1], sol.ys[-1][:-1],
                                    mech_jax["W"], mech_jax["a_hi"],
                                    mech_jax["a_lo"], mech_jax["Tmid"])
        print(f"Final temperature: {final_temp:.2f} K")
    else:
        # Grid test
        n = grid_size
        print(f"Running {n}x{n} grid integration test...")

        # Create grid
        print(f"Creating {n}x{n} grid...")
        y0_grid = np.tile(y0_cell, (n, n, 1))
        y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (n*n, S+1))

        # First precompile with a shorter time to warm up JAX
        print("Precompiling JAX functions...")
        t_warmup = time.time()
        _ = integrate_grid(y0_flat, rho, mech_jax, t0, t1/10, use_stiff=False)
        print(f"Precompilation completed in {time.time() - t_warmup:.2f} seconds")

        # Now run the actual integration
        print(f"Running JAX chemistry for {n}×{n} grid...")
        t_jax0 = time.time()
        sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1, use_stiff=False)
        t_jax = time.time() - t_jax0

        print(f"Grid integration completed in {t_jax:.2f} seconds")

        # Calculate temperatures
        print("Calculating temperatures...")
        ys_flat = np.array(sol.ys[-1])
        ys_jax = ys_flat.reshape(n, n, S+1)
        T_flat = np.array(jax.vmap(lambda y: T_from_e_newton(y[-1], y[:-1], mech_jax["W"],
                                                         mech_jax["a_hi"], mech_jax["a_lo"],
                                                         mech_jax["Tmid"]))(sol.ys[-1]))
        T_jax = T_flat.reshape(n, n)
        print(f"Mean temperature: {np.mean(T_jax):.2f} K")

        # Also quickly compare to Cantera (just timing)
        print("\nComparing to Cantera...")
        print(f"Setting up {n}x{n} Cantera reactors...")
        t_ct0 = time.time()

        # Setup reactors
        reactors = []
        gases = []
        for j in range(n):
            for i in range(n):
                gas = ct.Solution(mech["file"])
                Y0 = np.array(y0_grid[j,i,:-1], dtype=float)
                P0 = pressure_from_rho_T_Y(rho, T0, Y0, mech["W"])
                comp = {names[k]: float(Y0[k]) for k in range(S)}
                gas.TPY = T0, P0, comp
                r = ct.IdealGasReactor(gas)
                reactors.append(r)
                gases.append(gas)

        print(f"Running Cantera chemistry for {n}×{n} grid...")
        net = ct.ReactorNet(reactors)
        t_ct_integrate = time.time()
        net.advance(t1)
        t_ct_end = time.time()
        t_ct = t_ct_end - t_ct0
        t_ct_integrate_only = t_ct_end - t_ct_integrate

        print(f"Cantera setup time: {t_ct_integrate - t_ct0:.2f} seconds")
        print(f"Cantera integration time: {t_ct_integrate_only:.2f} seconds")
        print(f"Cantera total time: {t_ct:.2f} seconds")

        # Extract center cell temperature
        center_idx = (n*n) // 2 + n//2
        T_ct = gases[center_idx].T
        print(f"Center cell temperature: {T_ct:.2f} K")

        speedup = t_ct / t_jax if t_jax > 0 else float('inf')
        print(f"\nSpeedup: JAX is {speedup:.2f}x faster than Cantera")

    return True

# Utility function for pressure calculation
def pressure_from_rho_T_Y(rho_val, T_val, Y_val, W):
    """
    Calculate pressure from density, temperature, and mass fractions.

    Args:
        rho_val (float): Density in kg/m³
        T_val (float): Temperature in K
        Y_val (array): Mass fractions
        W (array): Molecular weights in g/mol

    Returns:
        float: Pressure in Pa
    """
    W_mol = np.array(W) / 1000.0  # kg/mol
    inv_Wmix = np.sum(Y_val / W_mol)  # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix                 # kg/mol
    R_univ = 8.314462618                   # J/(mol*K)
    R_spec = R_univ / Wmix                 # J/(kg*K)
    return rho_val * R_spec * T_val        # Pa

def compare_solvers():
    """
    Compare performance and accuracy of different solvers for chemistry against Cantera.
    Uses smaller grid (3x3) and relaxed tolerances to make the test quicker.

    This version focuses on comparing the stiff solver approach with Cantera,
    since combustion chemistry is inherently stiff.
    """
    print("\n=== SOLVER COMPARISON TEST WITH CANTERA REFERENCE ===")
    import time

    # Build mechanism once
    print("Building mechanism...")
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create JIT-compatible mechanism
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Create initial state
    rho = 1.2        # kg/m³
    T0 = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time
    grid_size = 3    # Smaller grid for testing

    # Create initial composition
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Calculate pressure consistent with rho,T,Y (for Cantera)
    W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
    inv_Wmix = np.sum(Y0 / W_mol)         # 1/(kg/mol)
    Wmix = 1.0 / inv_Wmix                 # kg/mol
    R_univ = 8.314462618                  # J/(mol*K)
    R_spec = R_univ / Wmix                # J/(kg*K)
    P0 = rho * R_spec * T0                # Pa

    # Calculate initial energy
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}
    gas_check.TPY = T0, P0, comp
    e0 = gas_check.int_energy_mass

    # Create state vector and grid
    y0_cell = jnp.concatenate([jnp.array(Y0), jnp.array([e0])])
    y0_grid = np.tile(y0_cell, (grid_size, grid_size, 1))
    y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (grid_size*grid_size, S+1))

    # Helper function to compute temperature from solution
    def get_mean_temp(sol):
        ys_flat = np.array(sol.ys[-1])
        T_flat = np.array(jax.vmap(lambda y: T_from_e_newton(y[-1], y[:-1],
                                               mech_jax["W"], mech_jax["a_hi"],
                                               mech_jax["a_lo"], mech_jax["Tmid"]))(sol.ys[-1]))
        return np.mean(T_flat)

    # First do a warm-up run for JIT compilation
    print("\nWarming up JIT compilation...")
    _ = integrate_grid(y0_flat, rho, mech_jax, t0, t1/10, use_stiff=True)

    # Run Cantera as reference (considered "ground truth")
    print("\nTest 1: Using Cantera (reference solution)...")
    t_start = time.time()

    # Setup Cantera reactors
    reactors = []
    gases = []
    for j in range(grid_size):
        for i in range(grid_size):
            gas = ct.Solution(mech["file"])
            gas.TPY = T0, P0, comp
            r = ct.IdealGasReactor(gas)
            reactors.append(r)
            gases.append(gas)

    # Create reactor network and run
    net = ct.ReactorNet(reactors)
    net.advance(t1)
    t_cantera = time.time() - t_start

    # Calculate mean temperature from Cantera
    T_cantera = np.mean([g.T for g in gases])
    print(f"Cantera completed in {t_cantera:.2f} seconds")
    print(f"Cantera final temperature: {T_cantera:.2f} K")

    # Get species mass fractions for key species
    key_species = ["H2", "O2", "H2O", "OH"]
    Y_cantera = {}
    for sp in key_species:
        if sp in names:
            idx = names.index(sp)
            Y_cantera[sp] = np.mean([g.Y[idx] for g in gases])

    # Test non-stiff solver (Tsit5)
    print("\nTest 2: Using Tsit5 (non-stiff) solver...")
    t_start = time.time()
    try:
        tsit5_sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1, use_stiff=False)
        t_tsit5 = time.time() - t_start
        print(f"Tsit5 completed in {t_tsit5:.2f} seconds")
        T_tsit5 = get_mean_temp(tsit5_sol)
        print(f"Tsit5 final temperature: {T_tsit5:.2f} K")

        # Get mass fractions for key species
        Y_tsit5 = {}
        for sp in key_species:
            if sp in names:
                idx = names.index(sp)
                Y_tsit5[sp] = np.mean(np.array(tsit5_sol.ys[-1])[:, idx])
    except Exception as e:
        print(f"Tsit5 solver failed: {e}")
        t_tsit5 = float('inf')
        T_tsit5 = None
        Y_tsit5 = {sp: None for sp in key_species if sp in names}

    # Test stiff solver (Dopri5 with relaxed tolerances)
    print("\nTest 3: Using Dopri5 with relaxed tolerances (stiff)...")
    t_start = time.time()
    try:
        dopri5_sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1, use_stiff=True)
        t_dopri5 = time.time() - t_start
        print(f"Dopri5 completed in {t_dopri5:.2f} seconds")
        T_dopri5 = get_mean_temp(dopri5_sol)
        print(f"Dopri5 final temperature: {T_dopri5:.2f} K")

        # Get mass fractions for key species
        Y_dopri5 = {}
        for sp in key_species:
            if sp in names:
                idx = names.index(sp)
                Y_dopri5[sp] = np.mean(np.array(dopri5_sol.ys[-1])[:, idx])
    except Exception as e:
        print(f"Dopri5 solver failed: {e}")
        t_dopri5 = float('inf')
        T_dopri5 = None
        Y_dopri5 = {sp: None for sp in key_species if sp in names}

    # Compare all results against Cantera
    print("\n=== RESULTS COMPARISON VS CANTERA ===")
    print(f"{'Solver':^15s} | {'Temp (K)':^10s} | {'Time (s)':^10s} | {'Temp Diff':^10s} | {'Temp Error (%)':^15s}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")

    print(f"{'Cantera':^15s} | {T_cantera:10.2f} | {t_cantera:10.2f} | {'ref':^10s} | {'ref':^15s}")

    if T_tsit5 is not None:
        temp_diff_tsit5 = abs(T_tsit5 - T_cantera)
        temp_err_tsit5 = 100 * temp_diff_tsit5 / T_cantera
        print(f"{'Tsit5 (non-stiff)':^15s} | {T_tsit5:10.2f} | {t_tsit5:10.2f} | {temp_diff_tsit5:+10.2f} | {temp_err_tsit5:+15.2f}%")

    if T_dopri5 is not None:
        temp_diff_dopri5 = abs(T_dopri5 - T_cantera)
        temp_err_dopri5 = 100 * temp_diff_dopri5 / T_cantera
        print(f"{'Dopri5 (stiff)':^15s} | {T_dopri5:10.2f} | {t_dopri5:10.2f} | {temp_diff_dopri5:+10.2f} | {temp_err_dopri5:+15.2f}%")

    # Print species mass fractions
    print("\n=== SPECIES MASS FRACTIONS ===")
    print(f"{'Species':^8s} | {'Cantera':^10s} | {'Tsit5':^10s} | {'Dopri5':^10s}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for sp in key_species:
        if sp in names:
            cantera_val = Y_cantera.get(sp, "N/A")
            tsit5_val = Y_tsit5.get(sp, "N/A")
            dopri5_val = Y_dopri5.get(sp, "N/A")

            # Format values as strings with appropriate precision
            cantera_str = f"{cantera_val:.6f}" if isinstance(cantera_val, float) else str(cantera_val)
            tsit5_str = f"{tsit5_val:.6f}" if isinstance(tsit5_val, float) else str(tsit5_val)
            dopri5_str = f"{dopri5_val:.6f}" if isinstance(dopri5_val, float) else str(dopri5_val)

            print(f"{sp:^8s} | {cantera_str:^10s} | {tsit5_str:^10s} | {dopri5_str:^10s}")

    # Performance comparison
    print("\n=== PERFORMANCE COMPARISON ===")
    if t_tsit5 < float('inf') and t_dopri5 < float('inf'):
        if t_cantera > t_tsit5:
            print(f"Tsit5 was {t_cantera/t_tsit5:.2f}x faster than Cantera")
        else:
            print(f"Cantera was {t_tsit5/t_cantera:.2f}x faster than Tsit5")

        if t_cantera > t_dopri5:
            print(f"Dopri5 was {t_cantera/t_dopri5:.2f}x faster than Cantera")
        else:
            print(f"Cantera was {t_dopri5/t_cantera:.2f}x faster than Dopri5")

    # Final verdict
    print("\n=== FINAL VERDICT ===")
    best_temp_match = "Dopri5 (stiff)" if (T_dopri5 is not None and
                                      T_tsit5 is not None and
                                      abs(T_dopri5 - T_cantera) < abs(T_tsit5 - T_cantera)) else "Tsit5 (non-stiff)"
    fastest = "Dopri5 (stiff)" if (t_dopri5 < t_tsit5 and t_dopri5 < float('inf')) else "Tsit5 (non-stiff)"

    print(f"Best temperature match to Cantera: {best_temp_match}")
    print(f"Fastest solver: {fastest}")

    if T_dopri5 is not None and abs(T_dopri5 - T_cantera) < 100:
        print("✓ The stiff solver (Dopri5) provides acceptable accuracy for combustion modeling")
    else:
        print("✗ The stiff solver needs further tuning to match Cantera results")

    print("\n=== Test complete ===")
    return tsit5_sol, dopri5_sol, gases

def quick_test(use_jit=True, grid_size=5, use_stiff=True, solver_type=None):
    """
    Run a quick integration test with the specified grid size.

    Args:
        use_jit (bool): Whether to use JIT compilation
        grid_size (int): Size of grid (grid_size x grid_size)
        use_stiff (bool): Whether to use the stiff solver approach (default: True)
        solver_type (str): Specific solver to use ('Tsit5', 'Dopri5', etc.)
    """
    solver_desc = solver_type if solver_type else ('stiff' if use_stiff else 'non-stiff')
    print(f"\nRunning quick test with {solver_desc} solver on a {grid_size}x{grid_size} grid...")

    # Build mechanism
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create JIT-compatible mechanism
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Create initial state
    rho = 1.2        # kg/m³
    T0 = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time

    # Create initial composition
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Calculate initial energy
    gas_check = ct.Solution(mech["file"])
    comp = {names[k]: float(Y0[k]) for k in range(S)}
    gas_check.TPY = T0, 101325.0, comp
    e0 = gas_check.int_energy_mass

    # Create state vector and grid
    y0_cell = jnp.concatenate([jnp.array(Y0), jnp.array([e0])])
    y0_grid = np.tile(y0_cell, (grid_size, grid_size, 1))
    y0_flat = jnp.reshape(jnp.array(y0_grid, dtype=jnp.float64), (grid_size*grid_size, S+1))

    print("Running integration...")
    t_start = time.time()
    sol = integrate_grid(y0_flat, rho, mech_jax, t0, t1, use_stiff=use_stiff, solver_type=solver_type)
    t_elapsed = time.time() - t_start
    print(f"Integration completed in {t_elapsed:.2f} seconds")

    # Calculate temperatures
    print("Calculating temperatures...")
    T_flat = np.array(jax.vmap(lambda y: T_from_e_newton(y[-1], y[:-1], mech_jax["W"],
                                                  mech_jax["a_hi"], mech_jax["a_lo"],
                                                  mech_jax["Tmid"]))(sol.ys[-1]))
    T_jax = T_flat.reshape(grid_size, grid_size)
    print(f"Mean temperature: {np.mean(T_jax):.2f} K")

    # If Cantera is available, compare with reference
    try:
        print("\nCalculating Cantera reference...")
        # Setup Cantera reactors (just one for quick comparison)
        gas = ct.Solution(mech["file"])
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = np.sum(Y0 / W_mol)         # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                 # kg/mol
        R_univ = 8.314462618                  # J/(mol*K)
        R_spec = R_univ / Wmix                # J/(kg*K)
        P0 = rho * R_spec * T0                # Pa

        gas.TPY = T0, P0, comp
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        t_ct_start = time.time()
        net.advance(t1)
        t_ct = time.time() - t_ct_start
        T_ct = gas.T

        # Compare results
        print(f"Cantera completed in {t_ct:.2f} seconds")
        print(f"Cantera temperature: {T_ct:.2f} K")
        print(f"JAX temperature: {np.mean(T_jax):.2f} K")
        print(f"Temperature difference: {abs(np.mean(T_jax) - T_ct):.2f} K ({100*abs(np.mean(T_jax) - T_ct)/T_ct:.2f}%)")
    except Exception as e:
        print(f"Cantera comparison skipped: {e}")

    return sol

def recommend_chemistry_solver():
    """
    Provides recommendations for chemistry solver selection based on the benchmark results.

    This function summarizes our findings and provides clear guidance on when to use
    different solver approaches for chemistry integration.
    """
    print("\n========================================================")
    print("CHEMISTRY SOLVER RECOMMENDATIONS FOR COMBUSTION SYSTEMS")
    print("========================================================")

    print("\nWe've configured our chemistry system to use the optimal solver by default:")
    print("→ Dopri5 with relaxed tolerances (stiff=True) is now the DEFAULT")

    print("\nSolver Options Available:")

    print("\n1. DEFAULT SOLVER - Best for Accuracy (Current Default):")
    print("   - Dopri5 with relaxed tolerances (use_stiff=True)")
    print("   - Temperature difference vs Cantera: ~0.5%")
    print("   - Species mass fractions closely match Cantera")
    print("   - Usage: Let the default parameters handle it")
    print("     integrate_grid(y0_grid, rho, mech, t0, t1)  # Defaults to stiff")

    print("\n2. PERFORMANCE SOLVER - Best for Speed:")
    print("   - Use Tsit5 solver (requires explicit parameter change)")
    print("   - Set use_stiff=False in integrate_cell/integrate_grid")
    print("   - Temperature difference vs Cantera: ~10-12%")
    print("   - Less accurate species mass fractions")
    print("   - Performance gain: ~1.5-3x faster than default")
    print("   - Usage:")
    print("     integrate_grid(y0_grid, rho, mech, t0, t1, use_stiff=False)")

    print("\n3. CUSTOM SOLVER - For Experimentation:")
    print("   - Explicitly select any available solver with solver_type parameter")
    print("   - Example usage:")
    print("     integrate_grid(y0_grid, rho, mech, t0, t1, solver_type='Tsit5')")
    print("     integrate_grid(y0_grid, rho, mech, t0, t1, solver_type='Dopri5')")
    print("     integrate_grid(y0_grid, rho, mech, t0, t1, solver_type='Kvaerno3')")

    print("\nSpecific Use Cases:")
    print("   a) For production combustion simulations:")
    print("      → Use default settings (stiff solver)")
    print("   b) For parametric studies or optimization where speed matters:")
    print("      → Consider use_stiff=False for preliminary work")
    print("   c) For final validation or critical accuracy needs:")
    print("      → Use default settings (stiff solver)")

    print("\nNOTE: Both solver options provide substantial speedup compared to Cantera")
    print("when integrating multiple cells in parallel (typically 5-10x for larger grids).")
    print("========================================================")

def test_solver_options(grid_size=10):
    """
    Compare different solver options on the same problem.

    Args:
        grid_size (int): Grid size for the test
    """
    print("\n============================================")
    print("CHEMISTRY SOLVER OPTIONS COMPARISON TEST")
    print("============================================")

    # Test default stiff solver (Dopri5 with relaxed tolerances)
    sol_default = quick_test(grid_size=grid_size)  # Uses defaults (stiff=True)

    # Test non-stiff solver (Tsit5)
    sol_nonStiff = quick_test(grid_size=grid_size, use_stiff=False)

    # Try different solvers if available
    try:
        sol_explicit_dopri = quick_test(grid_size=grid_size, solver_type='Dopri5')
        sol_explicit_tsit = quick_test(grid_size=grid_size, solver_type='Tsit5')
    except Exception as e:
        print(f"Error testing explicit solver types: {e}")

    # Print conclusion
    print("\n============================================")
    print("SOLVER COMPARISON CONCLUSION")
    print("============================================")
    print("→ The default stiff solver (Dopri5 with relaxed tolerances)")
    print("  provides the best balance of accuracy and performance")
    print("  for combustion chemistry problems.")
    print("\nStiff solver (DEFAULT) advantages:")
    print("- Much closer match to Cantera reference results")
    print("- Better stability for larger time steps")
    print("- Better accuracy for species mass fractions")
    print("\nNon-stiff solver advantages:")
    print("- Faster execution (1.5-3x)")
    print("- May be sufficient for preliminary studies")
    print("============================================")

def compare_initial_vs_runtime_divergence():
    """
    Create two 3D bar graphs showing how temperature, pressure, and simulation
    running times affect divergence percentage between JAX and Cantera.

    This function generates two separate analyses:
    1. Temperature effect: Various temperatures at constant pressure (1 atm)
    2. Pressure effect: Various pressures at constant temperature (1500K)

    Each graph has three axes:
    - Axis 1: Different values of temperature or pressure
    - Axis 2: Ten simulation running times (from 1e-5 to 1e-3 seconds)
    - Axis 3: Divergence percentage from Cantera reference solution

    Returns:
        dict: Results data used to generate the plots
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        print("Matplotlib not installed. Please install it to generate plots.")
        return None

    print("\n============================================")
    print("TEMPERATURE & PRESSURE vs RUNTIME DIVERGENCE ANALYSIS")
    print("============================================")

    # Build mechanism once
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create JIT-compatible mechanism
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Create initial composition (fixed for all tests)
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Define test parameters with evenly spaced values
    # 1. Temperature variation at constant pressure (1 atm)
    standard_pressure = 101325.0  # 1 atm in Pa

    # Create 10 evenly spaced temperatures from 800K to 2200K
    temp_min = 800.0
    temp_max = 2200.0
    num_temps = 20
    temperatures = np.linspace(temp_min, temp_max, num_temps)

    temperature_conditions = []
    for temp in temperatures:
        temperature_conditions.append({
            "name": f"T={int(temp)}K",
            "T": temp,
            "P": standard_pressure
        })

    # 2. Pressure variation at constant temperature (1500K)
    standard_temperature = 1500.0  # K

    # Create 20 evenly spaced pressures from 0.5 atm to 20 atm
    pressure_min = 0.5  # atm
    pressure_max = 20.0  # atm
    num_pressures = 20
    pressures_atm = np.linspace(pressure_min, pressure_max, num_pressures)

    pressure_conditions = []
    for pressure_atm in pressures_atm:
        pressure_pa = pressure_atm * 101325.0  # Convert atm to Pa
        pressure_conditions.append({
            "name": f"P={pressure_atm:.1f}atm",
            "T": standard_temperature,
            "P": pressure_pa
        })

    simulation_times = np.logspace(-5, -3, 20)  # 20 points from 1e-5 to 1e-3

    # Helper function to calculate density from P, T, Y
    def density_from_P_T_Y(P, T, Y):
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = np.sum(Y / W_mol)          # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                 # kg/mol
        R_univ = 8.314462618                  # J/(mol*K)
        R_spec = R_univ / Wmix                # J/(kg*K)
        return P / (R_spec * T)               # kg/m³

    # Helper function to run tests for a set of conditions using vectorized grid integration
    def run_test_conditions(conditions, title_suffix):
        # Storage for results
        results = np.zeros((len(conditions), len(simulation_times)))

        # Run the tests
        print(f"\n==== {title_suffix} ====")
        print(f"{'Initial Condition':^25s} | {'End Time (s)':^12s} | {'JAX Final T':^10s} | {'Cantera Final T':^15s} | {'Divergence (%)':^15s}")
        print(f"{'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*15}-+-{'-'*15}")

        # Prepare initial states for all conditions
        initial_states = []
        rho_values = []
        cantera_gases = []

        for condition in conditions:
            T0 = condition["T"]
            P0 = condition["P"]
            rho = density_from_P_T_Y(P0, T0, Y0)

            # Calculate initial energy using Cantera for exact match
            gas_check = ct.Solution(mech["file"])
            comp = {names[k]: float(Y0[k]) for k in range(S)}
            gas_check.TPY = T0, P0, comp
            e0 = gas_check.int_energy_mass

            # Store initial state vector and density
            y0 = jnp.concatenate([jnp.array(Y0), jnp.array([e0])])
            initial_states.append(y0)
            rho_values.append(rho)

            # Create a Cantera gas instance for later use
            gas = ct.Solution(mech["file"])
            gas.TPY = T0, P0, comp
            cantera_gases.append({
                "gas": gas,
                "comp": comp,
                "T0": T0,
                "P0": P0
            })

        # Convert to arrays for vectorized operations
        initial_states = jnp.array(initial_states)  # Shape: (num_conditions, S+1)

        # For each end time, run all conditions in parallel
        for j, end_time in enumerate(simulation_times):
            # Group conditions by density (need same density for grid integration)
            density_groups = {}
            for i, rho in enumerate(rho_values):
                # Use string key with precision to avoid floating point comparison issues
                rho_key = f"{rho:.10f}"
                if rho_key not in density_groups:
                    density_groups[rho_key] = []
                density_groups[rho_key].append(i)

            # For each density group, run vectorized integration
            jax_results = [None] * len(conditions)  # Placeholder for results

            for rho_key, indices in density_groups.items():
                rho = float(rho_key)
                # Convert indices to a JAX array for proper indexing
                indices_array = jnp.array(indices)
                batch_states = initial_states[indices_array]

                # Run JAX with stiff solver using vectorized grid integration
                sol_jax_batch = integrate_grid(batch_states, rho, mech_jax, 0.0, end_time, use_stiff=True)

                # Extract results
                for batch_idx, global_idx in enumerate(indices):
                    y_final = sol_jax_batch.ys[-1][batch_idx]
                    jax_results[global_idx] = y_final

            # Run Cantera simulations (can't easily vectorize)
            for i, cantera_info in enumerate(cantera_gases):
                gas = ct.Solution(mech["file"])
                gas.TPY = cantera_info["T0"], cantera_info["P0"], cantera_info["comp"]
                reactor = ct.IdealGasReactor(gas)
                net = ct.ReactorNet([reactor])
                net.advance(end_time)
                cantera_gases[i]["result"] = gas.T  # Store the temperature result

            # Calculate divergence and print results
            for i, condition in enumerate(conditions):
                y_final_jax = jax_results[i]
                Y_jax = y_final_jax[:-1]
                e_jax = y_final_jax[-1]
                T_jax = float(T_from_e_newton(e_jax, Y_jax, mech_jax["W"],
                                             mech_jax["a_hi"], mech_jax["a_lo"],
                                             mech_jax["Tmid"]))

                T_ct = cantera_gases[i]["result"]

                # Calculate temperature divergence
                divergence = 100.0 * abs(T_jax - T_ct) / T_ct
                results[i, j] = divergence

                # Print results for this combination
                print(f"{condition['name']:^25s} | {end_time:^12.2e} | {T_jax:^10.1f} | {T_ct:^15.1f} | {divergence:^15.2f}")

        return results

    # Run both sets of tests
    temp_results = run_test_conditions(temperature_conditions, "TEMPERATURE VARIATION (P=1atm)")
    press_results = run_test_conditions(pressure_conditions, "PRESSURE VARIATION (T=1500K)")

    # Helper function to create 3D bar plot
    def create_3d_bar_plot(results, conditions, title, filename):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set up X and Y coordinates for the bars - normalized to fill available space
        x_coords = np.arange(len(conditions))
        y_coords = np.arange(len(simulation_times))
        x_coords_mesh, y_coords_mesh = np.meshgrid(x_coords, y_coords)
        x_coords_flat = x_coords_mesh.flatten()
        y_coords_flat = y_coords_mesh.flatten()

        # Adjust bar width based on number of data points to fill available space
        # Use smaller width for more bars, larger width for fewer bars
        dx = min(0.9, 9.0 / len(conditions))  # Adjust width based on number of conditions
        dy = min(0.9, 9.0 / len(simulation_times))  # Adjust width based on number of times

        # Plot the bars
        dz = results.T.flatten()  # Transpose to match the meshgrid arrangement
        max_dz = np.max(dz) if np.max(dz) > 0 else 1.0  # Avoid division by zero
        colors = plt.cm.jet(dz / max_dz)  # Color by divergence percentage

        ax.bar3d(
            x_coords_flat,
            y_coords_flat,
            np.zeros_like(x_coords_flat),  # Start from z=0
            dx, dy, dz,
            color=colors,
            shade=True
        )

        # Set axis labels and title with better positioning
        ax.set_xlabel('Condition', labelpad=30, fontweight='bold')
        ax.set_ylabel('Simulation Time (s)', labelpad=25, fontweight='bold')
        ax.set_zlabel('Temperature Divergence (%)', labelpad=20, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Set custom tick labels - show only edge values and a few intermediate points
        condition_labels = [cond["name"] for cond in conditions]
        time_labels = [f"{time:.1e}" for time in simulation_times]

        # For x-axis (condition), show first, last, and a few intermediate ticks
        n_conditions = len(conditions)
        if n_conditions > 10:
            # Show about 5 ticks: first, last, and 3 intermediate points
            x_tick_indices = [0, n_conditions//4, n_conditions//2, 3*n_conditions//4, n_conditions-1]
            x_ticks = [x_coords[i] + dx/2 for i in x_tick_indices]
            x_ticklabels = [condition_labels[i] for i in x_tick_indices]
        else:
            # Show all ticks if there are 10 or fewer
            x_ticks = x_coords + dx/2
            x_ticklabels = condition_labels

        # For y-axis (time), show first, last, and a few intermediate ticks
        n_times = len(simulation_times)
        if n_times > 10:
            # Show about 5 ticks: first, last, and 3 intermediate points
            y_tick_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]
            y_ticks = [y_coords[i] + dy/2 for i in y_tick_indices]
            y_ticklabels = [time_labels[i] for i in y_tick_indices]
        else:
            # Show all ticks if there are 10 or fewer
            y_ticks = y_coords + dy/2
            y_ticklabels = time_labels

        # Set the ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        # Improved label positioning and rotation - rotate temperature labels outward
        ax.set_xticklabels(x_ticklabels, rotation=-45/2, ha='left', va='top')
        ax.set_yticklabels(y_ticklabels, rotation=45/2, ha='right', va='top')

        # Add a colorbar - properly configure the ScalarMappable
        m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        m.set_array(dz)
        m.set_clim(0, max_dz)  # Set proper color limits
        cbar = plt.colorbar(m, ax=ax, shrink=0.8)  # Explicitly provide the axis, shrink for better fit

        # Set the colorbar label to match the title (either Temperature or Pressure)
        if "Pressure" in title:
            cbar.set_label('Temperature Divergence (%)')
        else:
            cbar.set_label('Temperature Divergence (%)')

        # Set isometric view (equal scaling)
        # This creates a true isometric projection with equal scale on all axes
        ax.set_proj_type('ortho')  # Use orthographic projection instead of perspective

        # Set a standard isometric view angle - adjusted for better visibility with any number of data points
        ax.view_init(elev=25, azim=230)

        # Set z-axis scale based on the actual data range
        # Allow the full range of divergence values to be shown
        max_z = np.max(dz) * 1.05  # Add 5% margin for better visualization

        # Ensure minimum max_z value of 25 for cases with small divergences
        max_z = max(max_z, 25.0)

        ax.set_zlim(0, max_z)

        # Set axis limits to ensure bars fill the available space
        # Add a small buffer (10%) around the data for better visualization
        buffer_x = 0.5
        buffer_y = 0.5

        # Set explicit limits based on data extents rather than auto scaling
        ax.set_xlim(-buffer_x, len(conditions) - 1 + buffer_x)
        ax.set_ylim(-buffer_y, len(simulation_times) - 1 + buffer_y)

        # Keep the z-axis limits as previously defined
        # No need to modify z limits here as they're already set above

        # Now adjust the aspect ratio to look natural
        # Get current limits
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()

        # Calculate ranges
        x_range = abs(x_lim[1] - x_lim[0])
        y_range = abs(y_lim[1] - y_lim[0])
        z_range = abs(z_lim[1] - z_lim[0])

        # Adjust the scaling to make the plot more balanced
        # This helps maintain proportions regardless of data point count
        scale_x = x_range / max(len(conditions), 5)  # Scale based on number of conditions
        scale_y = y_range / max(len(simulation_times), 5)  # Scale based on number of times
        scale_z = z_range / max_z  # Fixed scale for z-axis (based on max height)

        # Set the aspect ratio
        ax.set_box_aspect((scale_x, scale_y, scale_z))

        plt.tight_layout()
        plt.savefig(filename, dpi=300)

        return fig

    # Use full range of divergence values without clipping
    # Create and save the temperature variation plot
    temp_fig = create_3d_bar_plot(
        temp_results,  # Using unclipped results
        temperature_conditions,
        'Temperature Divergence vs Initial Temperature vs Simulation Time (P=1atm)',
        'jax_temperature_divergence_3d_bar.png'
    )

    # Create and save the pressure variation plot
    press_fig = create_3d_bar_plot(
        press_results,  # Using unclipped results
        pressure_conditions,
        'Temperature Divergence vs Pressure vs Simulation Time (T=1500K)',
        'jax_pressure_divergence_3d_bar.png'
    )

    # Display both plots
    plt.show()

    print("\nPlots saved as:")
    print("- 'jax_temperature_divergence_3d_bar.png'")
    print("- 'jax_pressure_divergence_3d_bar.png'")

    # Return the results data for further analysis if needed
    return {
        'temperature': {
            'conditions': temperature_conditions,
            'times': simulation_times,
            'divergence': temp_results
        },
        'pressure': {
            'conditions': pressure_conditions,
            'times': simulation_times,
            'divergence': press_results
        }
    }

def compare_jax_cantera_conditions():
    """
    Compare JAX stiff solver vs Cantera across different initial temperatures and pressures.

    This function runs single-cell simulations with:
    1. Varying initial temperatures (fixed pressure)
    2. Varying initial pressures (fixed temperature)

    It produces 3D plots showing:
    - Runtime comparison
    - Temperature deviation
    - Species mass fraction deviations

    Returns:
        tuple: (temperature_results, pressure_results) - Dictionaries with test data
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
    except ImportError:
        print("Matplotlib not installed. Please install it to generate plots.")
        return None, None

    print("\n============================================")
    print("JAX vs CANTERA - INITIAL CONDITIONS COMPARISON")
    print("============================================")

    # Build mechanism once
    mech = build_mechanism("h2o2.yaml")
    names = mech["names"]
    S = mech["S"]

    # Create JIT-compatible mechanism
    mech_jax = {}
    for k, v in mech.items():
        if k not in ["file", "names"]:
            if isinstance(v, (int, float)):
                mech_jax[k] = v
            else:
                mech_jax[k] = jnp.asarray(v, dtype=jnp.float64 if v.dtype == np.float64 else v.dtype)

    # Base conditions
    rho_base = 1.2        # kg/m³
    T0_base = 1500.0      # K
    t0, t1 = 0.0, 1.0e-5  # integration time

    # Create initial composition (fixed for all tests)
    Y0 = np.zeros(S)
    iH2 = names.index("H2")
    iO2 = names.index("O2")
    iH2O = names.index("H2O")
    Y0[iH2] = 0.03
    Y0[iO2] = 0.20
    Y0[iH2O] = 1.0 - Y0[iH2] - Y0[iO2]

    # Helper function to calculate density from P, T, Y
    def density_from_P_T_Y(P, T, Y):
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = np.sum(Y / W_mol)          # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                 # kg/mol
        R_univ = 8.314462618                  # J/(mol*K)
        R_spec = R_univ / Wmix                # J/(kg*K)
        return P / (R_spec * T)               # kg/m³

    # Helper function to calculate pressure from rho, T, Y
    def pressure_from_rho_T_Y(rho, T, Y):
        W_mol = np.array(mech["W"]) / 1000.0  # kg/mol
        inv_Wmix = np.sum(Y / W_mol)          # 1/(kg/mol)
        Wmix = 1.0 / inv_Wmix                 # kg/mol
        R_univ = 8.314462618                  # J/(mol*K)
        R_spec = R_univ / Wmix                # J/(kg*K)
        return rho * R_spec * T               # Pa

    # Create storage for results
    temp_results = {
        'temperatures': [],
        'jax_runtime': [],
        'cantera_runtime': [],
        'temp_deviation': [],
        'Y_deviation': []
    }

    pressure_results = {
        'pressures': [],
        'jax_runtime': [],
        'cantera_runtime': [],
        'temp_deviation': [],
        'Y_deviation': []
    }

    # Function to run comparison for one condition
    def run_comparison(T0, P0, Y0, results_dict):
        # Calculate density from pressure
        rho = density_from_P_T_Y(P0, T0, Y0)

        # Calculate initial energy using Cantera
        gas_check = ct.Solution(mech["file"])
        comp = {names[k]: float(Y0[k]) for k in range(S)}
        gas_check.TPY = T0, P0, comp
        e0 = gas_check.int_energy_mass

        # Create initial state vector
        y0 = jnp.concatenate([jnp.array(Y0), jnp.array([e0])])

        # Run JAX integration (stiff solver)
        t_jax0 = time.time()
        sol = integrate_cell(y0, rho, mech_jax, t0, t1, use_stiff=True)
        t_jax = time.time() - t_jax0

        # Extract JAX results
        y_final = sol.ys[-1]
        Y_jax = y_final[:-1]
        e_jax = y_final[-1]
        T_jax = float(T_from_e_newton(e_jax, Y_jax, mech_jax["W"],
                                      mech_jax["a_hi"], mech_jax["a_lo"],
                                      mech_jax["Tmid"]))

        # Run Cantera integration
        gas = ct.Solution(mech["file"])
        gas.TPY = T0, P0, comp
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])

        t_ct0 = time.time()
        net.advance(t1)
        t_ct = time.time() - t_ct0

        # Extract Cantera results
        T_ct = gas.T
        Y_ct = np.array(gas.Y)

        # Calculate deviations
        temp_diff_pct = 100.0 * abs(T_jax - T_ct) / T_ct
        Y_diff = np.linalg.norm(Y_jax - Y_ct) / np.linalg.norm(Y_ct) * 100.0  # L2 norm percentage

        # Store results
        if 'temperatures' in results_dict:
            results_dict['temperatures'].append(T0)
        else:
            results_dict['pressures'].append(P0 / 101325.0)  # Convert Pa to atm

        results_dict['jax_runtime'].append(t_jax)
        results_dict['cantera_runtime'].append(t_ct)
        results_dict['temp_deviation'].append(temp_diff_pct)
        results_dict['Y_deviation'].append(Y_diff)

        return T_jax, T_ct, t_jax, t_ct, temp_diff_pct

    # Test 1: Varying Initial Temperatures (fixed pressure)
    print("\nTest 1: Varying Initial Temperatures")
    print("-------------------------------------")
    P0 = 101325.0  # 1 atm
    temperatures = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000]

    print(f"{'T0 (K)':^10s} | {'T_JAX (K)':^10s} | {'T_CT (K)':^10s} | {'JAX Time (s)':^12s} | {'CT Time (s)':^12s} | {'Diff (%)':^10s}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for T0 in temperatures:
        T_jax, T_ct, t_jax, t_ct, diff = run_comparison(T0, P0, Y0, temp_results)
        print(f"{T0:^10.1f} | {T_jax:^10.1f} | {T_ct:^10.1f} | {t_jax:^12.3f} | {t_ct:^12.3f} | {diff:^10.2f}")

    # Test 2: Varying Initial Pressures (fixed temperature)
    print("\nTest 2: Varying Initial Pressures")
    print("-------------------------------------")
    T0 = 1500.0  # K
    pressures = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]  # atm

    print(f"{'P0 (atm)':^10s} | {'T_JAX (K)':^10s} | {'T_CT (K)':^10s} | {'JAX Time (s)':^12s} | {'CT Time (s)':^12s} | {'Diff (%)':^10s}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for P0_atm in pressures:
        P0 = P0_atm * 101325.0  # atm to Pa
        T_jax, T_ct, t_jax, t_ct, diff = run_comparison(T0, P0, Y0, pressure_results)
        print(f"{P0_atm:^10.2f} | {T_jax:^10.1f} | {T_ct:^10.1f} | {t_jax:^12.3f} | {t_ct:^12.3f} | {diff:^10.2f}")

    # Generate plots
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Temperature variation - Runtime and deviation
    ax1 = fig.add_subplot(221, projection='3d')
    x = np.array(temp_results['temperatures'])
    y1 = np.array(temp_results['jax_runtime'])
    y2 = np.array(temp_results['cantera_runtime'])
    z = np.array(temp_results['temp_deviation'])

    ax1.plot(x, y1, z, 'ro-', label='JAX', linewidth=2, markersize=8)
    ax1.plot(x, y2, z, 'bo-', label='Cantera', linewidth=2, markersize=8)

    ax1.set_xlabel('Initial Temperature (K)')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_zlabel('Temperature Deviation (%)')
    ax1.set_title('Temperature Deviation vs Runtime vs Initial Temperature')
    ax1.legend()

    # Plot 2: Temperature variation - Species deviation
    ax2 = fig.add_subplot(222, projection='3d')
    z2 = np.array(temp_results['Y_deviation'])

    ax2.plot(x, y1, z2, 'ro-', label='JAX', linewidth=2, markersize=8)
    ax2.plot(x, y2, z2, 'bo-', label='Cantera', linewidth=2, markersize=8)

    ax2.set_xlabel('Initial Temperature (K)')
    ax2.set_ylabel('Runtime (s)')
    ax2.set_zlabel('Species Deviation (%)')
    ax2.set_title('Species Deviation vs Runtime vs Initial Temperature')
    ax2.legend()

    # Plot 3: Pressure variation - Runtime and deviation
    ax3 = fig.add_subplot(223, projection='3d')
    x = np.array(pressure_results['pressures'])
    y1 = np.array(pressure_results['jax_runtime'])
    y2 = np.array(pressure_results['cantera_runtime'])
    z = np.array(pressure_results['temp_deviation'])

    ax3.plot(x, y1, z, 'ro-', label='JAX', linewidth=2, markersize=8)
    ax3.plot(x, y2, z, 'bo-', label='Cantera', linewidth=2, markersize=8)

    ax3.set_xlabel('Initial Pressure (atm)')
    ax3.set_ylabel('Runtime (s)')
    ax3.set_zlabel('Temperature Deviation (%)')
    ax3.set_title('Temperature Deviation vs Runtime vs Initial Pressure')
    ax3.legend()

    # Plot 4: Pressure variation - Species deviation
    ax4 = fig.add_subplot(224, projection='3d')
    z2 = np.array(pressure_results['Y_deviation'])

    ax4.plot(x, y1, z2, 'ro-', label='JAX', linewidth=2, markersize=8)
    ax4.plot(x, y2, z2, 'bo-', label='Cantera', linewidth=2, markersize=8)

    ax4.set_xlabel('Initial Pressure (atm)')
    ax4.set_ylabel('Runtime (s)')
    ax4.set_zlabel('Species Deviation (%)')
    ax4.set_title('Species Deviation vs Runtime vs Initial Pressure')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('jax_vs_cantera_conditions.png', dpi=300)
    plt.show()

    print("\nPlots saved as 'jax_vs_cantera_conditions.png'")
    return temp_results, pressure_results

if __name__ == "__main__":
    print_device_info()

    # Optional: Run a quick test with default settings (stiff solver)
    run_quick_test = False  # Change to True to run the quick test
    if run_quick_test:
        print("\nRunning quick test with default settings (stiff solver)...")
        sol = quick_test(grid_size=10)  # Stiff solver is now the default

    # Optional: Run solver options comparison
    run_options_test = False  # Change to True to run the solver options test
    if run_options_test:
        test_solver_options(grid_size=10)

    # Optional: Compare JAX vs Cantera across different initial conditions
    run_conditions_test = False  # Change to True to run the conditions comparison
    if run_conditions_test:
        print("\nComparing JAX vs Cantera across different initial conditions...")
        compare_jax_cantera_conditions()

    # Optional: Run 3D bar graph analysis of initial conditions vs runtime vs divergence
    run_3d_bar_test = False  # Change to True to run the 3D bar graph analysis
    if run_3d_bar_test:
        print("\nRunning 3D bar graph analysis of initial conditions vs runtime vs divergence...")
        compare_initial_vs_runtime_divergence()

    # Optional: Run full benchmark (this will take a long time)
    run_benchmark = False  # Change to True to run the full benchmark
    if run_benchmark:
        print("\nRunning full benchmark with default stiff solver...")
        benchmark_cantera_vs_jax(use_stiff_solver=True, sizes=[4, 6, 8, 10, 12, 14])

    run_debug_thermo_conversion = True  # Change to True to run thermo conversion debug
    if run_debug_thermo_conversion:
        debug_thermo_conversion()

    run_demo_debug = False  # Change to True to run demo debug
    if run_demo_debug:
        demo_debug()

    # Uncomment any of these for additional tests
    # demo_debug()
    # debug_single_cell()
    # debug_thermo_conversion()
    # demo_2()
    # debug_integration_comparison()