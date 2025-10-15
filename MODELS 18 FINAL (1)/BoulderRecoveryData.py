import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp


# -----------------------------
# Helper functions
# -----------------------------
def data_rescale(x, y):
    """Rescale data points to model coordinates"""
    x_steps = round((51840 / 5.2) * x)
    y_value = (5.405 / 1.09) * y
    return x_steps, y_value


# -----------------------------
# Global timing / scaling
# -----------------------------
dt = 1.0
steps = 4320  # ~ 1 month at 1s resolution
T0 = 0.0
T1 = (steps - 1) * dt
SEC_PER_MONTH = 4320

# Output grid for solver
OUTPUT_STEP = 100.0
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)

# -----------------------------
# Data points for validation
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
print(s0x, s0y)
print(s1x, s1y)
print(s2x, s2y)
print(sFx, sFy)

# Pain reference points
p0x, p0y = 0, 4.10
p1x, p1y = 4320, 1.18

# -----------------------------
# Model parameters
# -----------------------------
ps_amp = 0
cs_ampf = 10
tcp = 26793.5765
tcc = 20054.4483

seg1_end_time = (steps) * dt

csi_values = [5.2]


# -----------------------------
# Time-varying stimulus functions
# -----------------------------
def ps_of_t(t):
    """Pain stimulus: exponential decay"""
    return ps_amp * np.exp(-t / tcp)


def cs_of_t(t):
    """Calming stimulus: exponential decay from initial sensitivity"""
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
# ODE system
# -----------------------------
def rhs(t, y):
    """Right-hand side of differential equations for pain recovery"""
    P_val, S_val = y
    tau_P = 0.000166
    P_dot = (-P_val + 10.0 / (1.0 + np.exp(-(S_val + ps_of_t(t) - 5.0)))) / tau_P
    S_dot = (1.0 / 1254.4483) * (-S_val + 10.0 / (1.0 + np.exp(-(0.0118 * P_val - cs_of_t(t) + 5.0))))
    return [P_dot, S_dot]


# -----------------------------
# Initial conditions
# -----------------------------
P0 = 4.1
S0 = 4.63604
y0 = [P0, S0]

# -----------------------------
# Run simulation and plot
# -----------------------------
plt.figure(figsize=(1, 4))

for i, cs_initial in enumerate(tqdm(csi_values)):
    # Solve ODE
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

    P = sol.y[0]
    S = sol.y[1]
    print(cs_initial, P[-1], S[-1])

    # Prepare data for plotting
    ps_coarse = ps_of_t(sol.t)
    cs_coarse = cs_of_t(sol.t)
    t_month = sol.t / SEC_PER_MONTH

    # Plot time series
    plt.plot(t_month, ps_coarse, color=(1.0, 0.2137, 0.9843), linestyle='--')
    plt.plot(t_month, P, color='red' if i == 0 else None)
    plt.plot(t_month, S, color='blue' if i == 0 else None)
    plt.plot(t_month, cs_coarse, color='lightgreen', linestyle='--')

# Plot reference data points (Pain markers)
plt.plot(p0x / SEC_PER_MONTH, p0y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p1x / SEC_PER_MONTH, p1y, marker='s', color=(0.85, 0.1137, 0.293))

# Format plot
plt.ylim(-0.05, 10.5)
plt.xlabel('Time (month)', fontsize=12)
plt.ylabel('', fontsize=14)
plt.tick_params(axis='y', length=0)
plt.tick_params(axis='x')
plt.title('Recovery', pad=15)
plt.tight_layout()
plt.show()