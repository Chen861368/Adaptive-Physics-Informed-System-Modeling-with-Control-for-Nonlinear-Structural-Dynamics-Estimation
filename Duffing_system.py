import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Define the Duffing equation
def duffing_eq(x, t, delta, alpha, beta, gamma, omega):
    """
    Duffing system equations (nonlinear second-order differential equation).
    Converts to a system of first-order ODEs for numerical solving.
    
    Parameters:
    - x: list of state variables [x1 (displacement), x2 (velocity)]
    - t: time
    - delta, alpha, beta, gamma, omega: parameters of the Duffing system
    
    Returns:
    - dx/dt: list containing first derivatives [dx1/dt, dx2/dt]
    """
    x1, x2 = x  # x1: displacement, x2: velocity
    dx1dt = x2
    dx2dt = -delta * x2 + alpha * x1 - beta * x1**3 + gamma * np.cos(omega * t)
    return [dx1dt, dx2dt]

# Solve the Duffing system
def solve_duffing_system(initial_conditions, time_span, params):
    """
    Numerically solve the Duffing system using scipy's odeint.
    
    Parameters:
    - initial_conditions: initial [x1, x2] values
    - time_span: array of time points to solve over
    - params: tuple/list of parameters (delta, alpha, beta, gamma, omega)
    
    Returns:
    - solution: 2D array of [x1, x2] over time
    """
    delta, alpha, beta, gamma, omega = params
    solution = odeint(duffing_eq, initial_conditions, time_span, args=(delta, alpha, beta, gamma, omega))
    return solution

# Save state and load separately in NumPy format
def save_state_and_load(time_span, solution, gamma, omega, save_path):
    """
    Save displacement/velocity (state) and external forcing (load) as .npy files.
    
    Parameters:
    - time_span: array of time points
    - solution: 2D array of state [x1, x2] over time
    - gamma, omega: parameters of the external forcing
    - save_path: directory where files will be saved
    """
    u = gamma * np.cos(omega * time_span)  # External load (forcing function)

    # Create a 2D array: first column as zeros (unused), second as load values
    load_data = np.vstack((np.zeros_like(time_span), u)).T

    # Ensure target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save state (x1, x2) and load data to .npy files
    np.save(os.path.join(save_path, "state_data.npy"), solution)
    np.save(os.path.join(save_path, "load_data.npy"), load_data)

    print(f"State data saved to {save_path}/state_data.npy")
    print(f"Load data saved to {save_path}/load_data.npy")

# Plot displacement vs time
def plot_displacement(time_span, solution, save_path=None):
    """
    Plot the displacement x1 over time and optionally save the plot as a PDF.
    
    Parameters:
    - time_span: array of time points
    - solution: 2D array of [x1, x2] over time
    - save_path: if provided, path to save the plot PDF
    """
    x1 = solution[:, 0]  # Extract displacement (x1) from solution array
    
    plt.figure(figsize=(8, 6))  # Create a figure with specified size
    plt.plot(time_span, x1, 'b-', linewidth=1.5)  # Plot in blue with line width
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)  # Light dashed grid
    plt.tight_layout()  # Optimize layout to prevent clipping

    # If save_path is given, save the figure as a PDF
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if not exist
        file_name = os.path.join(save_path, "displacement_time_plot.pdf")
        plt.savefig(file_name)

    plt.show()

# Plot velocity vs time
def plot_velocity(time_span, solution, save_path=None):
    """
    Plot the velocity (x2) over time and optionally save the plot as a PDF.
    
    Parameters:
    - time_span: array of time points
    - solution: 2D array where column 1 (solution[:, 1]) is the velocity
    - save_path: optional path to save the plot as a PDF
    """
    x2 = solution[:, 1]  # Extract velocity (x2) from solution array
    
    plt.figure(figsize=(8, 6))  # Create a figure with specified size
    plt.plot(time_span, x2, 'b-', linewidth=1.5)  # Plot velocity in blue with defined width
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Velocity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)  # Light dashed grid lines
    plt.tight_layout()  # Automatically adjust subplot parameters for a clean layout

    # If save_path is provided, save the plot as a PDF
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        file_name = os.path.join(save_path, "velocity_time_plot.pdf")
        plt.savefig(file_name)  # Save the figure as a PDF file

    plt.show()


# Plot phase portrait (displacement vs velocity)
def plot_phase_portrait(solution, save_path=None):
    """
    Plot the phase portrait (displacement x1 vs. velocity x2) and optionally save as a PDF.
    
    Parameters:
    - solution: 2D array where column 0 is x1 (displacement), column 1 is x2 (velocity)
    - save_path: optional path to save the phase portrait plot
    """
    x1 = solution[:, 0]  # Extract displacement (x1)
    x2 = solution[:, 1]  # Extract velocity (x2)

    plt.figure(figsize=(8, 6))  # Create a figure with consistent size
    plt.plot(x1, x2, 'b-', linewidth=1.5)  # Plot phase trajectory in blue
    plt.xlabel('Displacement', fontsize=20)
    plt.ylabel('Velocity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid lines
    plt.tight_layout()  # Optimize spacing for a neat layout

    # If save_path is given, save the plot to a file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if not already present
        file_name = os.path.join(save_path, "phase_portrait_plot.pdf")
        plt.savefig(file_name)  # Save the plot as a PDF

    plt.show()


# Set parameters for the Duffing system
c = 0.1        # Damping coefficient (delta)
alpha = 1      # Linear stiffness parameter
beta = 1       # Nonlinear stiffness parameter
F = 10         # Amplitude of the external periodic force (gamma)
omega = 1      # Frequency of the external force
params = (c, alpha, beta, F, omega)  # Tuple of parameters

# Set directory path to save data and plots
save_path = r"D:\博士课题\小论文\基于APSM方法的最优估计\论文图片\Duffing"

# Initial conditions for simulation
x1_0 = 0       # Initial displacement
x2_0 = 0       # Initial velocity
initial_conditions = [x1_0, x2_0]

# Time domain configuration
t_start = 0
t_end = 300
dt = 0.001  # Time step size
time_span = np.arange(t_start, t_end, dt)  # Time vector for simulation

# Solve the Duffing system using ODE integration
solution = solve_duffing_system(initial_conditions, time_span, params)

# Save simulation state (displacement and velocity) and external force as .npy files
save_state_and_load(time_span, solution, c, omega, save_path)

# Plot and optionally save displacement vs. time curve
plot_displacement(time_span, solution, save_path)

# Plot and optionally save velocity vs. time curve
plot_velocity(time_span, solution, save_path)

# Plot and optionally save phase portrait (displacement vs. velocity)
plot_phase_portrait(solution, save_path)
