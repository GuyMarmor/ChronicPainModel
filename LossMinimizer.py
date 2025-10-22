"""
Chronic-pain model – parameter fitting
-------------------------------------
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def data_rescale(x, y):
    """
    Map (months, VAS) to simulation steps / scaled value.

    Args:
        x: Time scale adjusted from Apkarian paper
        y: Data values extracted from Apkarian paper

    Returns:
        tuple: (x_steps, y_value) - rescaled time steps and value
    """
    x_steps = round((51840 / 5.2) * x)
    y_value = (5.405 / 1.09) * y
    return x_steps, y_value


# ==================================================================
# TIME CONFIGURATION
# ==================================================================

# Base time step and simulation duration
dt = 1.0  # Base time step in seconds
steps = 64800  # Total simulation steps (~16 months at 1s resolution)
SEC_PER_MONTH = 4320  # Seconds per month conversion factor

# Time boundaries
T0 = 0.0  # Initial time
T1 = (steps - 1) * dt  # Final time in seconds

# Analysis window (months 2-14, corresponding to steps 12960-64800)
window = slice(12960, 64800)

# Segment boundaries
seg1_end_time = (steps) * dt  # End time for first segment in seconds

# Full resolution time array (matches target data structure)
t_full = np.arange(steps, dtype=float) * dt

# Coarse output grid for ODE solver (improves performance)
OUTPUT_STEP = 1000.0  # Output every 1000 seconds
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)


# ==================================================================
# TARGET DATA CONSTRUCTION
# ==================================================================

def build_targets():
    """
    Build target curves P_d, S_d, Ps_d from anchor points.
    These represent the desired behavior for Pain (P), Sensitization (S),
    and Pain Stimulus (Ps) over time.

    Returns:
        tuple: (P_d, S_d, Ps_d) - target arrays for each variable
    """
    # Anchor points for pain stimulus (Ps)
    ps0x, ps0y = 0, 5.34745
    ps1x, ps1y = 6809, 3.9993179
    ps2x, ps2y = 27515, 3.04170
    psFx, psFy = 51840, 0.0435

    # Anchor points for sensitization (S) - rescaled from months/VAS
    s0x, s0y = data_rescale(0, 0.245)
    s1x, s1y = data_rescale(0.594, 0.165)
    s2x, s2y = data_rescale(2.647, 0.47)
    sFx, sFy = data_rescale(5.2, 0.93)

    # Anchor points for pain (P)
    p0x, p0y = 0, 2 * 2.93
    p1x, p1y = round((51840 / 5.2) * 0.743), 2 * 2.815
    p2x, p2y = round((51840 / 5.2) * 2.763), 2 * 2.695
    pFx, pFy = 51840, 2 * 2.991

    # Construct piecewise-linear target curves
    # All curves: 12960 zeros (first 3 months), then linear segments
    Ps_d = np.concatenate([
        np.zeros(12960),
        np.linspace(ps0y, ps1y, ps1x),
        np.linspace(ps1y, ps2y, ps2x - ps1x),
        np.linspace(ps2y, psFy, psFx - ps2x)
    ])

    S_d = np.concatenate([
        np.zeros(12960),
        np.linspace(s0y, s1y, s1x),
        np.linspace(s1y, s2y, s2x - s1x),
        np.linspace(s2y, sFy, sFx - s2x)
    ])

    P_d = np.concatenate([
        np.zeros(12960),
        np.linspace(p0y, p1y, p1x),
        np.linspace(p1y, p2y, p2x - p1x),
        np.linspace(p2y, pFy, pFx - p2x)
    ])

    return P_d, S_d, Ps_d


# Build target data once at module level
P_d, S_d, Ps_d = build_targets()


# ==================================================================
# FORWARD SIMULATION
# ==================================================================

def simulate(params):
    """
    Run forward simulation of the chronic pain model using given parameters.

    Args:
        params: tuple of (c1, ps_amp, cs_ampf, tcp, tcc)
            c1: Coupling coefficient from pain to sensitization
            ps_amp: Amplitude of pain stimulus
            cs_ampf: Final amplitude of calming stimulus
            tcp: Time constant for pain stimulus decay
            tcc: Time constant for calming stimulus decay

    Returns:
        tuple: (P_full, S_full, ps_full) - simulated pain, sensitization, and stimulus
    """
    c1, ps_amp, cs_ampf, tcp, tcc = params

    # -------------------------
    # Stimulus Functions
    # -------------------------

    def ps_of_t(t):
        """Pain stimulus: exponential decay over time."""
        return ps_amp * np.exp(-t / tcp)

    def cs_of_t(t):
        """
        Calming stimulus: piecewise exponential.
        Single segment decay from initial value 10.
        """
        if np.isscalar(t):
            t = float(t)
            if t < seg1_end_time:
                return (10 - cs_ampf) * np.exp(-t / tcc) + cs_ampf
        else:
            t = np.asarray(t)
            out = np.empty_like(t, dtype=float)
            mask1 = t < seg1_end_time
            out[mask1] = (10 - cs_ampf) * np.exp(-t[mask1] / tcc) + cs_ampf
            return out

    # -------------------------
    # ODE System Definition
    # -------------------------

    def rhs(t, y):
        """
        Right-hand side of the ODE system.

        System equations:
        dP/dt = (-P + 10/(1 + exp(-(S + ps(t) - 5)))) / tau_P
        dS/dt = (-S + 10/(1 + exp(-(c1*P - cs(t) + 5)))) / tau_S

        Args:
            t: Current time
            y: State vector [P, S]

        Returns:
            list: [P_dot, S_dot] - time derivatives
        """
        P_val, S_val = y
        tau_P = 0.000167  # Time constant for pain dynamics ~ 1 second relative to 4320 steps/month
        tau_S = 1254.4483  # Time constant for sensitization dynamics ~ 1.2 weeks relative to 4320 steps/month

        # Pain dynamics (fast timescale)
        P_dot = (-P_val + 10.0 / (1.0 + np.exp(-(S_val + ps_of_t(t) - 5.0)))) / tau_P

        # Sensitization dynamics (slow timescale)
        S_dot = (1.0 / tau_S) * (-S_val + 10.0 / (1.0 + np.exp(-(c1 * P_val - cs_of_t(t) + 5.0))))

        return [P_dot, S_dot]

    # -------------------------
    # Initial Conditions
    # -------------------------

    # Pain starts at steady state given initial stimulus
    P0 = 10.0 / (1.0 + np.exp(-(ps_amp - 5.0)))
    S0 = 0.0  # Sensitization starts at zero
    y0 = [P0, S0]

    # -------------------------
    # Numerical Integration
    # -------------------------

    csi_values = [10]  # Initial calming stimulus value

    for i, cs_initial in enumerate(csi_values):
        # Solve ODE system using LSODA method on coarse grid
        sol = solve_ivp(
            rhs,
            (T0, T1),
            y0,
            method="LSODA",
            t_eval=t_eval,  # Coarse output grid for efficiency
            dense_output=True,  # Enable interpolation to full resolution
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        # Interpolate solution to full resolution (matches target data structure)
        P_full = sol.sol(t_full)[0]
        S_full = sol.sol(t_full)[1]
        ps_full = ps_of_t(t_full)

        # Debug output
        print(f"cs_initial: {cs_ampf}, P_final: {P_full[-1]:.4f}, "
              f"S_final: {S_full[-1]:.4f}, c1: {c1}")

    return P_full, S_full, ps_full


# ==================================================================
# LOSS FUNCTION
# ==================================================================

def true_loss(p):
    """
    Compute mean squared error between simulation and target data.
    Loss is calculated only over the analysis window (months 3-15).

    Args:
        p: Parameter vector [c1, ps_amp, cs_ampf, tcp, tcc]

    Returns:
        float: Combined MSE across P, S, and ps
    """
    P, S, ps = simulate(p)

    # MSE for each variable over the analysis window
    mse_P = np.mean((P[window] - P_d[window]) ** 2)
    mse_S = np.mean((S[window] - S_d[window]) ** 2)
    mse_ps = np.mean((ps[window] - Ps_d[window]) ** 2)

    return mse_P + mse_S + mse_ps


# ==================================================================
# PARAMETER TRANSFORMATION (LOG-SPACE FOR TIME CONSTANTS)
# ==================================================================

def pack(p):
    """
    Transform parameters to optimization space.
    Time constants tcp and tcc are converted to log10 space for better
    numerical behavior during optimization.

    Args:
        p: [c1, ps_amp, cs_ampf, tcp, tcc] in natural units

    Returns:
        array: [c1, ps_amp, cs_ampf, log10(tcp), log10(tcc)]
    """
    c1, ps_amp, cs_ampf, tcp, tcc = p
    return np.array([c1, ps_amp, cs_ampf, np.log10(tcp), np.log10(tcc)])


def unpack(z):
    """
    Transform parameters from optimization space back to natural units.

    Args:
        z: [c1, ps_amp, cs_ampf, log10(tcp), log10(tcc)]

    Returns:
        array: [c1, ps_amp, cs_ampf, tcp, tcc] in natural units
    """
    c1, ps_amp, cs_ampf, lg_tcp, lg_tcc = z
    return np.array([c1, ps_amp, cs_ampf, 10 ** lg_tcp, 10 ** lg_tcc])


def loss(z):
    """
    Loss function wrapper for optimizer (operates in log-space).

    Args:
        z: Parameters in optimization space

    Returns:
        float: MSE loss
    """
    return true_loss(unpack(z))


# ==================================================================
# OPTIMIZATION PROCEDURE
# ==================================================================

# Parameter bounds in optimization space
# c1: [0, 0.15], ps_amp: [0, 10], cs_ampf: [0, 10]
# log10(tcp): [3, 5] → tcp: [1000, 100000] seconds
# log10(tcc): [3, 5] → tcc: [1000, 100000] seconds
bounds_log = [(0, 0.15), (0, 10), (0, 10), (3, 5), (3, 5)]

# Step 1: Global optimization using differential evolution
print("Running global search (differential evolution)…")
pop = differential_evolution(
    loss,
    bounds_log,
    strategy='best1bin',
    maxiter=600,
    polish=False,
    disp=False
)

# Step 2: Local refinement using L-BFGS-B
print("Refining with L-BFGS-B…")
local = minimize(
    loss,
    pop.x,
    method='L-BFGS-B',
    bounds=bounds_log,
    options={'maxiter': 400, 'eps': 1e-3, 'disp': True}
)

# Extract best parameters in natural units
best = unpack(local.x)

# Display results
print("\nOptimal parameters:")
print(f"  c1       = {best[0]:.4f}")
print(f"  ps_amp   = {best[1]:.4f}")
print(f"  cs_ampf  = {best[2]:.4f}")
print(f"  tcp      = {best[3]:.4f}")
print(f"  tcc      = {best[4]:.4f}")
print(f"\nFinal windowed MSE = {true_loss(best):.6f}")

# ==================================================================
# VISUALIZATION
# ==================================================================

if __name__ == "__main__":
    # Generate fitted curves using optimal parameters
    P, S, ps = simulate(best)

    # Convert time steps to months for plotting
    months = np.arange(steps) * dt / SEC_PER_MONTH

    # Create comparison plot
    plt.figure(figsize=(9, 4))

    # Target curves (dashed)
    plt.plot(months, P_d, 'salmon', lw=2, label='P_d  (target)')
    plt.plot(months, S_d, 'aqua', lw=2, label='S_d  (target)')
    plt.plot(months, Ps_d, 'pink', lw=2, label='Ps_d (target)')

    # Fitted curves (solid)
    plt.plot(months, P, 'r-', label='P  (fit)')
    plt.plot(months, S, 'b-', label='S  (fit)')
    plt.plot(months, ps, 'm--', label='ps (fit)')

    # Formatting
    plt.xlim(0, 14)
    plt.ylim(-0.2, 10.5)
    plt.xlabel("Time (months)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()