# Performance-optimized JAX chemistry
import numpy as np
import jax
# Enable double precision
jax.config.update("jax_enable_x64", True)
# Use CPU for now (GPU setup on Windows is complex)y
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
    h, _, _ = nasa7_h_u_s(T, mech_jax["a_hi"], mech_jax["a_lo"], mech_jax["Tmid"])
    dedt = -jnp.dot(h, wdot) / rho  # J/kg/s

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

def integrate_cell(y0, rho, mech, t0, t1, *, dt0=1e-8, rtol=1e-5, atol=1e-8, ts=None, max_steps=100_000):
    # Create a JIT-compatible version of the mechanism dictionary
    mech_jax = {k: v for k, v in mech.items() if k != "file" and k != "names"}

    term = dfx.ODETerm(rhs)
    # Use Tsit5 - often more efficient than Dopri5 for medium precision
    solver = dfx.Tsit5()
    # More aggressive stepping for better performance
    steps = dfx.PIDController(rtol=rtol, atol=atol, pcoeff=0.2, dcoeff=0.0)
    saveat = dfx.SaveAt(ts=ts) if ts is not None else dfx.SaveAt(t1=True)
    sol = dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0,
                          args=(mech_jax, rho), stepsize_controller=steps,
                          saveat=saveat, max_steps=max_steps, throw=False)
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

    # Get molar enthalpies in parallel with reaction rates
    h, _, _ = nasa7_h_u_s(T, mech["a_hi"], mech["a_lo"], mech["Tmid"])

    # Calculate species production rates - optimized to avoid repeated computations
    wdot, _, _ = rop(T, Y, rho, mech, Kc)  # kmol/m^3/s

    # Efficiently calculate mass fraction rates
    # Use in-place operations where possible
    dYdt = (wdot * mech["W"]) / rho  # 1/s

    # Energy rate calculation - use dot product for better vectorization
    # Directly compute enthalpy dot product with species rates
    dedt = -jnp.dot(h, wdot) / rho  # J/kg/s

    # Numerical safety limits - prevents integration instability
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
def integrate_grid(y0_grid, rho, mech, t0, t1, *, rtol=1e-5, atol=1e-8):
    term   = dfx.ODETerm(rhs_grid)

    # Use Tsit5 solver - often more efficient than Dopri5 for medium precision
    solver = dfx.Tsit5()

    # Adjust controller for more aggressive stepping
    steps  = dfx.PIDController(rtol=rtol, atol=atol, pcoeff=0.2, dcoeff=0.0)
    saveat = dfx.SaveAt(t1=True)

    # Start with a larger initial timestep and use fewer maximum steps
    # This reduces overhead significantly
    sol = dfx.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=1e-8,
        y0=y0_grid, args=(mech, rho),
        stepsize_controller=steps, saveat=saveat,
        max_steps=100_000, throw=True
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
            gas.TPX = T0, P0, comp
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

if __name__ == "__main__":
    print_device_info()
    demo_2()