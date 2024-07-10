import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# parameters
s_in = 2
gr_value = 10
k_s = 0.5
k_i = 0.05

# Specific growth rate function
def gr(s):
    return gr_value * s / (s**2 / k_i + k_s + s)

# Equations for steady state
def steady_state_equations(vars, q_in):
    b, s = vars
    eq1 = -q_in * b + gr(s) * b
    eq2 = q_in * (s_in - s) - gr(s) * b
    return [eq1, eq2]

# Function to find steady states
def loc_steady_state(q_in):
    initial_guess = [1, 1]
    solution = fsolve(steady_state_equations, initial_guess, args=(q_in,))
    return solution

# Bifurcation analysis
q_in_range = np.linspace(0.01, 2, 1000)
steady_states = [loc_steady_state(q) for q in q_in_range]

# biomass and substrate steady states seperation function
biomass = [state[0] for state in steady_states]
sub_steady = [state[1] for state in steady_states]

# Plot  for bifurcation
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(q_in_range, biomass)
plt.xlabel('Inflow rate (q_in)')
plt.ylabel('Biomass Concentration(Steady-State)')
plt.title('Bifurcation Diagram For Biomass')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(q_in_range, sub_steady)
plt.xlabel('Inflow rate (q_in)')
plt.ylabel('Substrate Concentration(Steady-State)')
plt.title('Bifurcation Diagram For Substrate')
plt.grid(True)

plt.tight_layout()
plt.show()

# Analyze stability
def stability_exam(q_in, b, s):
    J = np.array([
        [-q_in + gr(s), b * gr_value * (k_i * k_s - s**2) / (k_i * (k_s + s) + s**2)**2],
        [-gr(s), -q_in - b * gr_value * (k_i * k_s - s**2) / (k_i * (k_s + s) + s**2)**2]
    ])
    eigenvalues = np.linalg.eigvals(J)
    return np.all(np.real(eigenvalues) < 0)

# Stability of steady states analysis
stability = [stability_exam(q, b, s) for q, b, s in zip(q_in_range, biomass, sub_steady)]

# Bifurcation graph 
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.scatter(q_in_range, biomass, c=stability, cmap='coolwarm', s=10)
plt.xlabel('Inflow rate (q_in)')
plt.ylabel('Biomass Concentration(Steady-State)')
plt.title('Bifurcation Diagram For Biomass')
plt.colorbar(label='Stable (1) / Unstable (0)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(q_in_range, sub_steady, c=stability, cmap='coolwarm', s=10)
plt.xlabel('Inflow rate (q_in)')
plt.ylabel('Substrate Concentration(Steady-State)')
plt.title('Bifurcation Diagram for Substrate')
plt.colorbar(label='Stable (1) / Unstable (0)')
plt.grid(True)

plt.tight_layout()
plt.show()
