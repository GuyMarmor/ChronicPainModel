import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp


# -----------------------------
# Helper functions
# -----------------------------
def fixed_point(vars, ps, cs):
    """Calculate fixed point equations for the system"""
    P, S = vars
    eq1 = P - 10 / (1 + np.exp(-0.5 * (S + 1 * ps - 8.5)))
    eq2 = S - 10 / (1 + np.exp(-0.05 * (P - 3 * cs - 10)))
    return [eq1, eq2]


def P_(P):
    """Pain rescaling function"""
    return P * (0.973 - 0.751) + 0.751


def S_(S):
    """SAF rescaling function"""
    return S * (3.08 - 1.605) + 1.605


def sat(x):
    """Saturation function to bound values between 0 and 1"""
    return max(min(x, 1), 0)


def data_rescale(x, y):
    """Rescale data points to model coordinates"""
    x_steps = round((51840 / 5.2) * x)
    y_value = (5.405 / 1.09) * y
    return x_steps, y_value


def data_ps_rescale(x, y):
    """Rescale pain stimulus data points to model coordinates"""
    x_steps = round((51840 / 5.2) * x)
    y_value = -np.log((10 / ((5.86 / 1.09) * y) - 1)) + 5
    return x_steps, y_value


# -----------------------------
# Global timing / scaling
# -----------------------------
dt = 1.0
steps = 64800  # ~ 16 months at 1s resolution
T0 = 0.0
T1 = (steps - 1) * dt
SEC_PER_MONTH = 4320

# Output grid for solver
OUTPUT_STEP = 1.0
n_out = int(np.floor((T1 - T0) / OUTPUT_STEP))
t_eval = T0 + OUTPUT_STEP * np.arange(n_out + 1, dtype=float)

# -----------------------------
# Data points for validation
# -----------------------------
# Pain stimulus reference points
ps0x, ps0y = data_ps_rescale(0, 1.09)
ps1x, ps1y = data_ps_rescale(0.683, 0.5)
ps2x, ps2y = data_ps_rescale(2.76, 0.23)
psFx, psFy = data_ps_rescale(5.2, 0.013)
print(ps0x, ps0y)
print(ps1x, ps1y)
print(ps2x, ps2y)
print(psFx, psFy)

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
p0x, p0y = 0, 2 * 2.93
p1x, p1y = round((51840 / 5.2) * (0.743)), 2 * 2.815
p2x, p2y = round((51840 / 5.2) * (2.763)), 2 * 2.695
pFx, pFy = 51840, 2 * 2.991

# -----------------------------
# Model parameters acquired from Loss Minimizer
# -----------------------------
ps_amp = 7.3833
cs_ampf = 5.1070
tcp = 37066.3060
tcc = 23853.3621

seg1_end_time = (steps) * dt


# -----------------------------
# Time-varying stimulus functions
# -----------------------------
def ps_of_t(t):
    """Pain stimulus: exponential decay"""
    return ps_amp * np.exp(-t / tcp)


def cs_of_t(t):
    """Calming stimulus: exponential decay"""
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
# ODE system
# -----------------------------
def rhs(t, y):
    """Right-hand side of differential equations for chronic pain dynamics"""
    P_val, S_val = y
    tau_P = 0.000166
    P_dot = (-P_val + 10.0 / (1.0 + np.exp(-(S_val + ps_of_t(t) - 5.0)))) / tau_P
    S_dot = (1.0 / 1254.4483) * (-S_val + 10.0 / (1.0 + np.exp(-(0.0118 * P_val - cs_of_t(t) + 5.0))))
    return [P_dot, S_dot]


# -----------------------------
# Initial conditions
# -----------------------------
P0 = 10.0 / (1.0 + np.exp(-(ps_amp - 5.0)))
S0 = 0.0
y0 = [P0, S0]

# -----------------------------
# Run simulation and plot
# -----------------------------
plt.figure(figsize=(9, 4))

csi_values = [10]
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
    plt.plot(t_month, ps_coarse, color=(1.0, 0.2137, 0.9843), linestyle='--', label='pain stimulus' if i == 0 else None)
    plt.plot(t_month, P, color='red', label='Pain' if i == 0 else None)
    plt.plot(t_month, S, color='blue', label='SAF' if i == 0 else None)
    plt.plot(t_month, cs_coarse, color='lightgreen', linestyle='--', label='calming stimulus' if i == 0 else None)

# Plot reference data points
# Pain stimulus markers
plt.plot(ps0x / SEC_PER_MONTH + 3, ps0y, marker='D', color=(0.85, 0.1137, 0.893))
plt.plot(ps1x / SEC_PER_MONTH + 3, ps1y, marker='D', color=(0.85, 0.1137, 0.893))
plt.plot(ps2x / SEC_PER_MONTH + 3, ps2y, marker='D', color=(0.85, 0.1137, 0.893))
plt.plot(psFx / SEC_PER_MONTH + 3, psFy, marker='D', color=(0.85, 0.1137, 0.893))

# SAF markers
plt.plot(s0x / SEC_PER_MONTH + 3, s0y, marker='o', color=(0.4, 0.4137, 0.893))
plt.plot(s1x / SEC_PER_MONTH + 3, s1y, marker='o', color=(0.4, 0.4137, 0.893))
plt.plot(s2x / SEC_PER_MONTH + 3, s2y, marker='o', color=(0.4, 0.4137, 0.893))
plt.plot(sFx / SEC_PER_MONTH + 3, sFy, marker='o', color=(0.4, 0.4137, 0.893))

# Pain markers
plt.plot(p0x / SEC_PER_MONTH + 3, p0y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p1x / SEC_PER_MONTH + 3, p1y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(p2x / SEC_PER_MONTH + 3, p2y, marker='s', color=(0.85, 0.1137, 0.293))
plt.plot(pFx / SEC_PER_MONTH + 3, pFy, marker='s', color=(0.85, 0.1137, 0.293))

# Format plot
plt.ylim(-0.05, 10.5)
plt.xlabel('Time (month)', fontsize=14)
plt.ylabel('Value (0-10)', fontsize=14)
plt.title('Development of Chronic Pain', pad=15)
plt.legend()
plt.tight_layout()
plt.show()