import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp


# -----------------------------
# Helper Functions
# -----------------------------
def data_rescale(x, y):
    """Rescale data points for SAF."""
    x_steps = round((51840 / 5.2) * x)
    y_value = (5.405 / 1.09) * y
    return x_steps, y_value


# -----------------------------
# Global Timing and Scaling Parameters
# -----------------------------
dt = 1.0  # Time step in seconds
steps = 4320  # Total simulation steps (~1 month at 1s resolution)
T0 = 0.0
T1 = (steps - 1) * dt  # Final time in seconds
SEC_PER_MONTH = 4320  # Seconds per month in simulation time

# Output grid for solver (100-second intervals)
OUTPUT_STEP = 100.0
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)

# -----------------------------
# Reference Data Points
# -----------------------------
# Pain stimulus reference points
ps0x, ps0y = data_rescale(0, 1.09)
ps1x, ps1y = data_rescale(0.683, 0.5)
ps2x, ps2y = data_rescale(2.76, 0.23)
psFx, psFy = data_rescale(5.2, 0)

# SAF reference points
s0x, s0y = data_rescale(0, 0.245)
s1x, s1y = data_rescale(0.594, 0.165)
s2x, s2y = data_rescale(2.647, 0.47)
sFx, sFy = data_rescale(5.2, 0.93)

print("SAF Reference Points:")
print(f"  s0: ({s0x}, {s0y:.4f})")
print(f"  s1: ({s1x}, {s1y:.4f})")
print(f"  s2: ({s2x}, {s2y:.4f})")
print(f"  sF: ({sFx}, {sFy:.4f})")

# Pain reference points for markers
p0x, p0y = 0, 5
p1x, p1y = 4320, 2

# -----------------------------
# Model Parameters
# -----------------------------
ps_amp = 0  # Pain stimulus amplitude (constant zero for this patient)
cs_ampf = 10  # Calming stimulus amplitude
tcp = 26793.5765  # Time constant for pain stimulus decay (unused here)
tcc = 25254.4483  # Time constant for calming stimulus decay
tcs = 1254.4483  # Time constant for SAF dynamics

# Initial conditions
P0 = 5  # Initial pain level
S0 = 5  # Initial SAF level
y0 = [P0, S0]

# Calming stimulus initial value
csi_values = [5.052]

# Simulation boundary
seg1_end_time = steps * dt  # End of first segment in seconds


# -----------------------------
# Time-Varying Input Functions
# -----------------------------
def ps_of_t(t):
    """Pain stimulus as a function of time (constant at ps_amp)."""
    if np.isscalar(t):
        return ps_amp
    else:
        return np.full_like(t, ps_amp, dtype=float)


def cs_of_t(t):
    """Calming stimulus as a function of time (exponential decay)."""
    if np.isscalar(t):
        t = float(t)
        if t < seg1_end_time:
            return (csi_values[0] - cs_ampf) * np.exp(-t / tcc) + cs_ampf
    else:
        t = np.asarray(t)
        out = np.empty_like(t, dtype=float)
        mask1 = t < seg1_end_time
        out[mask1] = (csi_values[0] - cs_ampf) * np.exp(-t[mask1] / tcc) + cs_ampf
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
    S_dot = (1.0 / tcs) * (-S_val + 10.0 / (1.0 + np.exp(-(0.0118 * P_val - cs_of_t(t) + 5.0))))

    return [P_dot, S_dot]


# -----------------------------
# Simulation and Plotting
# -----------------------------
plt.figure(figsize=(1.5, 6))

for i, cs_initial in enumerate(tqdm(csi_values, desc="Simulating patient")):
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

    # Plot trajectories
    plt.plot(t_month, ps_coarse, color=(1.0, 0.2137, 0.9843), linestyle='--')
    plt.plot(t_month, P, color='red' if i == 0 else None)
    plt.plot(t_month, S, color='blue' if i == 0 else None)
    plt.plot(t_month, cs_coarse, color='lightgreen', linestyle='--')

# Add reference markers for pain data points
plt.plot(p0x / SEC_PER_MONTH, p0y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p1x / SEC_PER_MONTH, p1y, marker='s', color=(0.85, 0.1137, 0.293))

# Configure plot appearance
plt.ylim(-0.05, 10.5)
plt.xlabel('Time (months)', fontsize=10)
plt.ylabel('', fontsize=14)
plt.tick_params(axis='y', length=0)
plt.title('#INSERT INJURY#', pad=15)
plt.tight_layout()
plt.show()