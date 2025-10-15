import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp


# -----------------------------
# Helper Functions
# -----------------------------
def fixed_point(vars, ps, cs):
    """Calculate fixed point equations for the system."""
    P, S = vars
    eq1 = P - 10 / (1 + np.exp(-0.5 * (S + 1 * ps - 8.5)))
    eq2 = S - 10 / (1 + np.exp(-0.05 * (P - 3 * cs - 10)))
    return [eq1, eq2]


def P_(P):
    """Scale P value."""
    return P * (0.973 - 0.751) + 0.751


def S_(S):
    """Scale S value."""
    return S * (3.08 - 1.605) + 1.605


def sat(x):
    """Saturate value between 0 and 1."""
    return max(min(x, 1), 0)


def data_rescale(x, y):
    """Rescale data points for SAF."""
    x_steps = round((51840 / 5.2) * x)
    y_value = (5.405 / 1.09) * y
    return x_steps, y_value


def data_ps_rescale(x, y):
    """Rescale data points for pain stimulus."""
    x_steps = round((51840 / 5.2) * x)
    y_value = -np.log((10 / ((5.86 / 1.09) * y) - 1)) + 5
    return x_steps, y_value


# -----------------------------
# Global Timing and Scaling Parameters
# -----------------------------
dt = 1.0  # Time step in seconds
steps = 64800  # Total simulation steps (~15 months at 1s resolution)
T0 = 0.0
T1 = (steps - 1) * dt  # Final time in seconds
SEC_PER_MONTH = 4320  # Seconds per month in simulation time

# Output grid for solver (60-second intervals)
OUTPUT_STEP = 1.0
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)

# -----------------------------
# Reference Data Points
# -----------------------------
# Pain stimulus reference points
ps0x, ps0y = data_ps_rescale(0, 1.09)
ps1x, ps1y = data_ps_rescale(0.683, 0.5)
ps2x, ps2y = data_ps_rescale(2.76, 0.23)
psFx, psFy = data_ps_rescale(5.2, 0.013)

print("Pain Stimulus Reference Points:")
print(f"  ps0: ({ps0x}, {ps0y:.4f})")
print(f"  ps1: ({ps1x}, {ps1y:.4f})")
print(f"  ps2: ({ps2x}, {ps2y:.4f})")
print(f"  psF: ({psFx}, {psFy:.4f})")

# SAF reference points
s0x, s0y = data_rescale(0, 0.245)
s1x, s1y = data_rescale(0.594, 0.165)
s2x, s2y = data_rescale(2.647, 0.47)
sFx, sFy = data_rescale(5.2, 0.93)

print("\nSAF Reference Points:")
print(f"  s0: ({s0x}, {s0y:.4f})")
print(f"  s1: ({s1x}, {s1y:.4f})")
print(f"  s2: ({s2x}, {s2y:.4f})")
print(f"  sF: ({sFx}, {sFy:.4f})")

# Pain reference points
p0x, p0y = 0, 2 * 2.93
p1x, p1y = round((51840 / 5.2) * (0.743)), 2 * 2.815
p2x, p2y = round((51840 / 5.2) * (2.763)), 2 * 2.695
pFx, pFy = 51840, 2 * 2.991

# -----------------------------
# Model Parameters
# -----------------------------
ps_amp = 7.3833  # Pain stimulus amplitude
cs_ampf = 5.1070  # Calming stimulus amplitude (baseline)
tcp = 37066.3060  # Time constant for pain stimulus decay
tcc = 23853.3621  # Time constant for calming stimulus decay

# Simulation boundary
seg1_end_time = steps * dt  # End of first segment in seconds


# -----------------------------
# Time-Varying Input Functions
# -----------------------------
def ps_of_t(t):
    """Pain stimulus as a function of time (exponential decay)."""
    return ps_amp * np.exp(-t / tcp)


def cs_of_t(t):
    """Calming stimulus as a function of time (piecewise exponential)."""
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


# -----------------------------
# ODE System Definition
# -----------------------------
def rhs(t, y):
    """Right-hand side of the ODE system for pain (P) and SAF (S)."""
    P_val, S_val = y
    tau_P = 0.000166

    # Pain dynamics
    P_dot = (-P_val + 10.0 / (1.0 + np.exp(-(S_val + ps_of_t(t) - 5.0)))) / tau_P

    # SAF dynamics
    S_dot = (1.0 / 1254.4483) * (-S_val + 10.0 / (1.0 + np.exp(-(0.0118 * P_val - cs_of_t(t) + 5.0))))

    return [P_dot, S_dot]


# -----------------------------
# Initial Conditions
# -----------------------------
P0 = 10.0 / (1.0 + np.exp(-(ps_amp - 5.0)))
S0 = 0.0
y0 = [P0, S0]

# -----------------------------
# Simulation Loop: Family of Curves
# -----------------------------
plt.figure(figsize=(5, 4))

csi_values = [6.80, 5.8, 5.1070, 4.7]  # Different calming stimulus initial values
red_colors = ['#ff6666', '#ff3333', 'red', '#b30000']  # Color gradient for pain curves

for i, cs_initial in enumerate(tqdm(csi_values, desc="Simulating curves")):
    cs_ampf = csi_values[i]

    # Solve ODE system using LSODA method
    sol = solve_ivp(
        rhs,
        (T0, T1),
        y0,
        method="LSODA",
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-10,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    P = sol.y[0]  # Pain trajectory
    S = sol.y[1]  # SAF trajectory

    print(f"\ncs_initial = {cs_initial:.4f}: P_final = {P[-1]:.4f}, S_final = {S[-1]:.4f}")

    # Prepare inputs on the same grid for plotting
    ps_coarse = ps_of_t(sol.t)
    cs_coarse = cs_of_t(sol.t)
    t_month = sol.t / SEC_PER_MONTH

    # Plot pain stimulus (only label once)
    plt.plot(t_month, ps_coarse, color=(1.0, 0.2137, 0.9843), linestyle='--',
             label='Pain Stimulus' if i == 0 else None)

    # Plot pain trajectory
    plt.plot(t_month, P, color=red_colors[i],
             label='Pain' if i == 0 else None)

# Add reference markers for pain data points
plt.plot(p0x / SEC_PER_MONTH + 3, p0y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p1x / SEC_PER_MONTH + 3, p1y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p2x / SEC_PER_MONTH + 3, p2y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(pFx / SEC_PER_MONTH + 3, pFy, marker='s', color=(0.85, 0.1137, 0.293))

# Configure plot appearance
plt.ylim(-0.05, 10.5)
plt.xlabel('Time (months)', fontsize=14)
plt.ylabel('Value (0-10)', fontsize=14)
plt.title('Development of Chronic Pain for Different Calming Stimulus Values', pad=15)
plt.legend(fontsize='x-small')
plt.tight_layout()
plt.show()