import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
gr_value = 10
ks = 0.5
ki = 0.05
s_in = 2

# Specific Growth rate function
def gr(s):
    return gr_value * s / (s**2 / ki + ks + s)

# ODE system Functions
def bioreactor(t, y, q_in, q_out):
    b, s, V = y
    db_dt = -q_in / V * b + gr(s) * b
    ds_dt = q_in / V * (s_in - s) - gr(s) * b
    dV_dt = q_in - q_out
    return [db_dt, ds_dt, dV_dt]

# Numerical Simulation function
def reactor_simulation(initial_conditions, q_in, q_out, time_span):
    time_cal = np.linspace(time_span[0], time_span[1], 1000)
    sol = solve_ivp(bioreactor, time_span, initial_conditions, args=(q_in, q_out), time_cal=time_cal)
    return sol.t, sol.y

# Initial conditions and flow rates
condition = [
    ((0.01, 0.5, 0.5), 0.5, 0.4),
    ((0.5, 1.4, 1.0), 0.4, 0.4),
    ((5.0, 1.4, 1.0), 0.4, 0.4), 
    ((5.0, 1.4, 1.0), 1.4, 1.4)]

# Time span
time_span = (0, 10)

# Legend Position
legend_pos= ['best', 'upper left', 'best', 'best']


# Plot results
plt.figure(figsize=(12, 8))

for i, (initial_conditions, q_in, q_out) in enumerate(condition):
    t, y = reactor_simulation(initial_conditions, q_in, q_out, time_span)
    b, s, V = y
    plt.subplot(2, 2, i + 1)
    plt.plot(t, b, label='Biomass', color='green')
    plt.plot(t, s, label='Substrate',  color='red')
    plt.plot(t, V, label='Volume', color='blue')
    plt.xlabel('Time') 
    plt.ylabel('Concentration')
    plt.legend(loc=legend_pos[i]) 
    plt.title(f'Condition {i + 1}: q_in = {q_in}, q_out = {q_out}')
    plt.grid(True)

plt.tight_layout()
plt.show()