import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

# -----------------------------
# Global timing / scaling
# -----------------------------
dt = 1.0
steps = 64800  # ~ 16 months at 1s resolution
T0 = 0.0
T1 = (steps - 1) * dt
SEC_PER_MONTH = 4320

# 60-second output grid (solver still adapts internally)
OUTPUT_STEP = 1.0
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)

# The values that affect Sensitivity
s_values = [0, 4]

# -----------------------------
# Time-varying linearly increasing stimulus function
# -----------------------------
def ps_of_t(t):
    """Pain stimulus as a function of time"""
    return (t / 4320 - 5)

# -----------------------------
# ODE system
# -----------------------------
def rhs(t, y):
    """Right-hand side of differential equations for pain dynamics"""
    P_val, S_val = y
    tau_P = 0.000166
    P_dot = (-P_val + 10.0 / (1.0 + np.exp(-(S_val + ps_of_t(t) - 5.0)))) / tau_P
    S_dot = 0
    return [P_dot, S_dot]


# -----------------------------
# Run simulations and plot
# -----------------------------
plt.figure(figsize=(4,4))

for i, cs_initial in enumerate(tqdm(s_values)):
    # Initial conditions
    S0 = s_values[i]
    P0 = 10.0 / (1.0 + np.exp(-(S0 + -5 - 5.0)))
    y0 = [P0, S0]

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

    # Time axis in months
    t_month = np.arange(steps) / SEC_PER_MONTH

    # Plot pain response
    if i == 0:
        # Normal sensitivity: plot from month 5 onwards
        plt.plot(t_month[21600:], P[21600:], color='red', label='Normal', linewidth=3)
    else:
        # Hyperalgesia: plot full curve with shaded region
        mask = (t_month <= 5) & (P >= 0)
        plt.plot(t_month, P, color='red', linestyle='--', label='Hyperalgesia', linewidth=3)
        plt.fill_between(t_month, P, 0.0, where=mask, color='red', alpha=0.5, interpolate=True)

# Format plot
ax = plt.gca()
ax.set_xticks([])
plt.yticks([0, 5, 10], fontsize=13)
plt.ylim(-0.00, 10.5)
plt.xlabel('Stimulus intensity', fontsize=14, labelpad=17)
plt.ylabel('Pain', fontsize=14, labelpad=10)
plt.title('Pain vs. Stimulus', pad=14)
plt.tight_layout()
plt.show()
