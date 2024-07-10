import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
gr_value = 10
k_s = 0.5
k_i = 0.05
s_in = 2

# Specific growth rate function
def sgr(s):
    return gr_value * s / (s**2 / k_i + k_s + s)

# Control law
def feedback_ctrl_rule(b, s, V, desired_P_rate):
    sgr_s = sgr(s)
    u = (desired_P_rate / V + sgr_s * b) / b
    return u

# Closed-loop system of differential equations 
def closed_loop(y, t, desired_P_rate):
    b, s, V = y
    u = feedback_ctrl_rule(b, s, V, desired_P_rate)
    
    dbdt = -u * b + sgr(s) * b
    dsdt = u * (s_in - s) - sgr(s) * b
    dVdt = 0
    
    return [dbdt, dsdt, dVdt]

# Sisgrlate and plot results
def reactor_plot_controller(initial_conditions, desired_P_rate, t_span):
    t = np.linspace(0, t_span, 1000)
    solution = odeint(closed_loop, initial_conditions, t, args=(desired_P_rate,))
    
    b, s, V = solution.T
    
    # Estimate control input and production rate
    u = [feedback_ctrl_rule(b[i], s[i], V[i], desired_P_rate) for i in range(len(t))]
    P = [u[i] * V[i] * b[i] for i in range(len(t))]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    ax1.plot(t, b, label='Biomass', color='green')
    ax1.plot(t, s, label='Substrate',color='red')
    ax1.set_ylabel('Concentration')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, u, label='Control input',color='blue')
    ax2.set_ylabel('Dilution rate')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(t, P, label='Production rate', color='green')
    ax3.axhline(y=desired_P_rate, color='r', linestyle='--', label='Desired production rate')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Production rate')
    ax3.legend()
    ax3.grid(True)
    
    plt.suptitle('Feedback_Control Analysis')
    plt.tight_layout()
    plt.show()

# Controlled system for different initial conditions
initial_conditions_list = [
    [0.01, 0.5, 0.5],
    [0.5, 1.4, 1],
    [5, 1.4, 1]
]
# Desired K biomass production rate
desired_P_rate = 2

for i, initial_conditions in enumerate(initial_conditions_list, 1):
    print(f"Sisgrlation {i} with initial conditions: {initial_conditions}")
    reactor_plot_controller(initial_conditions, desired_P_rate, 10)
