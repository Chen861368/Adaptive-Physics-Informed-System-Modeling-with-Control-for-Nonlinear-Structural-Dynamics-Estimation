# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm


def gradient_descent_update(A_k, B_k, x_k, y_k, u_k, learning_rate):
    """
    Gradient descent update for the matrix A_k, using a single column vector x_k,
    its corresponding target y_k, and external input u_k.
    
    Parameters:
    A_k (numpy.ndarray): Matrix to be updated (system matrix)
    B_k (numpy.ndarray): Input matrix (assumed fixed in this update)
    x_k (numpy.ndarray): State vector (column vector)
    y_k (numpy.ndarray): Target output value
    u_k (numpy.ndarray): External input vector
    learning_rate (float): Learning rate for gradient descent
    
    Returns:
    numpy.ndarray: Updated A_k matrix
    """
    # Compute the gradient of the loss function with respect to A_k
    # Assuming a quadratic loss and that input u_k affects the output
    grad = -2 * (y_k - A_k.dot(x_k) - B_k @ u_k) * x_k.T

    # Update A_k using the computed gradient
    A_k -= learning_rate * grad
    return A_k


def gradient_descent_update_unknown_B(A_k, B_k, x_k, y_k, u_k, learning_rate):
    """
    Gradient descent update for both A_k and B_k matrices, using a single state vector x_k,
    target y_k, and input u_k.
    
    Parameters:
    A_k (numpy.ndarray): Matrix A to be updated (system matrix)
    B_k (numpy.ndarray): Matrix B to be updated (input influence matrix)
    x_k (numpy.ndarray): State vector (column vector)
    y_k (numpy.ndarray): Target output value
    u_k (numpy.ndarray): External input vector
    learning_rate (float): Learning rate for gradient descent
    
    Returns:
    numpy.ndarray, numpy.ndarray: Updated matrices A_k and B_k
    """
    # Compute the prediction error (residual)
    error = y_k - A_k.dot(x_k) - B_k @ u_k

    # Compute gradients for A_k and B_k based on the residual
    grad_A_k = -2 * error * x_k.T  # Gradient w.r.t. A_k
    grad_B_k = -2 * error * u_k.T  # Gradient w.r.t. B_k

    # Perform the gradient descent updates
    A_k -= learning_rate * grad_A_k
    B_k -= learning_rate * grad_B_k

    return A_k, B_k


def bilinear_continuous_to_discrete_A(A_c, dt):
    """
    Convert a continuous-time system matrix A_c to its discrete-time equivalent A_d using the bilinear transformation.
    
    Parameters:
    A_c (numpy.ndarray): Continuous-time system matrix
    dt (float): Time step for the discrete system
    
    Returns:
    numpy.ndarray: Discrete-time system matrix A_d
    """
    # Bilinear transformation (Tustin's method)
    # A_d = (I - A_c * dt / 2) ^ (-1) * (I + A_c * dt / 2)
    
    I = np.eye(A_c.shape[0])  # Identity matrix
    A_d = np.linalg.inv(I - (dt/2)*A_c).dot(I + (dt/2)*A_c)  # 状态矩阵
    return A_d


def gradient_descent_update_unknown_B_physics(A_k, B_k, x_k, y_k, u_k, learning_rate):
    """
    Gradient descent update for A_k and B_k matrices, using a single x_k and its corresponding y_k,
    while incorporating physics-inspired projection of A_k.

    Parameters:
    A_k (numpy.ndarray): Matrix A to be updated (system dynamics matrix)
    B_k (numpy.ndarray): Matrix B to be updated (input effect matrix)
    x_k (numpy.ndarray): State vector (column vector)
    y_k (numpy.ndarray): Target output
    u_k (numpy.ndarray): External input vector
    learning_rate (float): Learning rate for gradient descent

    Returns:
    numpy.ndarray, numpy.ndarray: Updated matrices A_k and B_k
    """
    # Compute the residual error
    error = y_k - A_k.dot(x_k) - B_k @ u_k

    # Compute gradients for A_k and B_k
    grad_A_k = -2 * error * x_k.T  # Gradient with respect to A_k
    grad_B_k = -2 * error * u_k.T  # Gradient with respect to B_k

    # Update A_k using gradient descent
    A_k -= learning_rate * grad_A_k

    # Convert A_k from discrete to continuous domain using bilinear transform
    A_con = bilinear_discrete_to_continuous_A(A_k, dt)

    # Apply physics-based constraints to the continuous A matrix (e.g., projection)
    A_re = project_matrix_elementwise(A_con)
    # Optionally use a model-based Jacobian projection:
    # A_re = jacobian(x_k, c=0.1, alpha=1, beta=1)

    # Convert A back to discrete-time domain
    A_k = bilinear_continuous_to_discrete_A(A_re, dt)
    # Alternative method using matrix logarithm:
    # A_k = continuous_to_discrete_log(A_re, dt)

    # Update B_k using gradient descent
    B_k -= learning_rate * grad_B_k

    return A_k, B_k




def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE, NMSE, and R^2 metrics.
    Handles complex-valued inputs by using only their real parts.

    Parameters:
    - actual (np.ndarray): The ground truth data, potentially complex-valued.
    - predicted (np.ndarray): The predicted output data, potentially complex-valued.

    Returns:
    - mse (float): Mean Squared Error between actual and predicted values.
    - nmse (float): Normalized Mean Squared Error (MSE divided by variance of actual values).
    - r2 (float): Coefficient of Determination (R^2 score).
    """
    # Convert to real-valued arrays (if complex)
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(actual_real, predicted_real)

    # Compute Normalized MSE (NMSE)
    variance_actual = np.var(actual_real)
    nmse = mse / variance_actual

    # Compute R^2 score
    r2 = r2_score(actual_real, predicted_real)

    return mse, nmse, r2




def bilinear_discrete_to_continuous_A(A_d, dt):
    """
    Recovers the continuous dynamic system matrices from the discrete ones using the Bilinear Transformation method.
    
    Args:
    Ad (ndarray): Discrete system matrix.
    Bd (ndarray): Discrete input matrix.
    T (float): Sampling time used for discretization.
    
    Returns:
    A (ndarray): Continuous system matrix.
    B (ndarray): Continuous input matrix.
    """
    n = A_d.shape[0]  # Assuming Ad is square and represents the state dimension
    I = np.eye(n)
    
    # Bilinear transformation inverse
    A = (2/dt) * np.linalg.inv(A_d + I) @ (A_d-I)
    return A



def jacobian(x_k, c=0.1, alpha=1, beta=1):
    """
    Compute the Jacobian matrix for the given nonlinear system.
    
    Parameters:
    x_k (numpy.ndarray): State vector [x1, x2] (can be a column vector)
    c (float): Damping coefficient (default 0.1)
    alpha (float): Parameter alpha (default -1)
    beta (float): Parameter beta (default 1)
    F (float): Amplitude of the external forcing (default 10)
    omega (float): Frequency of the external forcing (default 1)
    
    Returns:
    numpy.ndarray: Jacobian matrix of the system
    """
    # Flatten the input vector x_k to ensure it's a 1D array
    x1, x2 = x_k.flatten()  # Ensure x_k is a 1D array (flatten if it's a column vector)
    
    # Define the Jacobian matrix elements
    J = np.array([
        [0, 1],
        [alpha - 3 * beta * x1**2, -c]
    ])
    
    return J

def frobenius_norm(A, B=None):
    """
    Calculate the Frobenius norm of one or two matrices.
    
    Parameters:
    A (numpy.ndarray): First matrix
    B (numpy.ndarray, optional): Second matrix (default is None)
    
    Returns:
    tuple: Frobenius norm of A, and if B is provided, Frobenius norm of A + B
    """
    # Compute Frobenius norm of A
    frobenius_norm_A = np.linalg.norm(A, 'fro')
    
    # If B is provided, compute Frobenius norm of A + B
    if B is not None:
        frobenius_norm_A_plus_B = np.linalg.norm(A + B, 'fro')
        return frobenius_norm_A, frobenius_norm_A_plus_B
    else:
        return frobenius_norm_A

def bilinear_continuous_to_discrete(A_cont, B_cont, delta_t):
    """
    Convert continuous-time state-space matrices to discrete-time using the bilinear (Tustin) transform.

    Parameters:
    - A_cont (np.ndarray): Continuous-time system matrix A
    - B_cont (np.ndarray): Continuous-time input matrix B
    - delta_t (float): Sampling time interval

    Returns:
    - A_d (np.ndarray): Discrete-time system matrix (Phi)
    - B_d (np.ndarray): Discrete-time input matrix (Gamma)
    """
    I = np.eye(A_cont.shape[0])  # Identity matrix of the same size as A_cont

    # Discrete-time A matrix using the bilinear (Tustin) transformation
    A_d = np.linalg.inv(I - (delta_t / 2) * A_cont) @ (I + (delta_t / 2) * A_cont)

    # Discrete-time B matrix
    B_d = np.linalg.inv(I - (delta_t / 2) * A_cont) @ (delta_t * B_cont)

    return A_d, B_d



def APSMC_grad(combined_matrix_noise, A_initial, U, B, learning_rate, iterations):
    """
    Run APSMC (Adaptive Physics-based Structured Model Calibration) using noisy state data 
    to update matrix A_k via gradient descent (assuming B_k is known).

    Parameters:
    - combined_matrix_noise (np.ndarray): Noisy state matrix, each column is a state at a time step.
    - A_initial (np.ndarray): Initial guess for matrix A_k.
    - U (np.ndarray): Input matrix (columns correspond to time steps).
    - B (np.ndarray): Known input matrix B_k (fixed during update).
    - learning_rate (float): Gradient descent learning rate.
    - iterations (int): Number of time steps / gradient updates.

    Returns:
    - A_k (np.ndarray): Updated system matrix A_k after gradient descent.
    - y_estimates (np.ndarray): Predicted outputs over time.
    - Frobenius_norm (list): List of Frobenius norms of (J_k_discrete - A_k) at each iteration.
    """
    n = combined_matrix_noise.shape[0]
    A_k = A_initial

    y_estimates = np.zeros((n, iterations))
    Frobenius_norm = []

    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]        # Current state
        u_k = U[:, idx+1:idx+2]                          # Next input
        y_k = combined_matrix_noise[:, idx+1:idx+2]      # Next state (target output)

        J_k = jacobian(x_k, c=0.1, alpha=1, beta=1)      # Compute Jacobian from physics model
        J_k_discrete, B_k = bilinear_continuous_to_discrete(J_k, B, dt)  # Discretize J_k and B

        A_k = gradient_descent_update(A_k, B_k, x_k, y_k, u_k, learning_rate)  # Update A_k

        frobenius_k = frobenius_norm(J_k_discrete - A_k)
        y_k_pred = A_k @ x_k + B_k @ u_k               # Predict next state

        y_estimates[:, idx] = y_k_pred.flatten()
        Frobenius_norm.append(frobenius_k)

    return A_k, y_estimates, Frobenius_norm


def APSMC(combined_matrix_noise, A_initial, B_initial, U, B, learning_rate, iterations):
    """
    Run APSMC with physics-informed projection to iteratively update both A_k and B_k.
    
    Parameters:
    - combined_matrix_noise (np.ndarray): Noisy state matrix, each column is a state at a time step.
    - A_initial (np.ndarray): Initial estimate of A_k.
    - B_initial (np.ndarray): Initial estimate of B_k.
    - U (np.ndarray): Input matrix, each column corresponds to a control input at a time step.
    - B (np.ndarray): Reference B matrix for discretization of Jacobian.
    - learning_rate (float): Gradient descent learning rate.
    - iterations (int): Number of iterations to perform updates.

    Returns:
    - A_k (np.ndarray): Final updated A_k matrix.
    - B_k (np.ndarray): Final updated B_k matrix.
    - y_estimates (np.ndarray): Predicted outputs over time.
    - Frobenius_norm_A (list): Frobenius norm of (J_k_discrete - A_k) at each iteration.
    - Frobenius_norm_B (list): Frobenius norm of (J_b - B_k) at each iteration.
    """
    n = combined_matrix_noise.shape[0]
    A_k = A_initial
    B_k = B_initial

    y_estimates = np.zeros((n, iterations))
    Frobenius_norm_A = []
    Frobenius_norm_B = []

    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]        # Current state
        u_k = U[:, idx+1:idx+2]                          # Next input
        y_k = combined_matrix_noise[:, idx+1:idx+2]      # Target next state

        J_k = jacobian(x_k, c=0.1, alpha=1, beta=1)      # Compute physics-based Jacobian
        J_k_discrete, J_b = bilinear_continuous_to_discrete(J_k, B, dt)  # Discretize

        y_k_pred = A_k @ x_k + B_k @ u_k                 # Predict next state

        # Gradient-based update without physics projection
        A_k, B_k = gradient_descent_update_unknown_B(A_k, B_k, x_k, y_k, u_k, learning_rate)

        # Gradient-based update with physics-informed projection
        # A_k, B_k = gradient_descent_update_unknown_B_physics(A_k, B_k, x_k, y_k, u_k, learning_rate)

        frobenius_A_k = frobenius_norm(J_k_discrete - A_k)
        frobenius_B_k = frobenius_norm(J_b - B_k)

        y_estimates[:, idx] = y_k_pred.flatten()
        Frobenius_norm_A.append(frobenius_A_k)
        Frobenius_norm_B.append(frobenius_B_k)

    return A_k, B_k, y_estimates, Frobenius_norm_A, Frobenius_norm_B




def add_nonstationary_gaussian_noise(signal, noise_ratio):
    """
    Add non-stationary Gaussian noise to a signal. The noise added to each sample is proportional
    to the magnitude of the signal at that point.

    Parameters:
    - signal (np.ndarray): The original signal.
    - noise_ratio (float): The ratio of the noise amplitude to the signal amplitude.

    Returns:
    - noisy_signal (np.ndarray): Signal with added non-stationary Gaussian noise.
    """
    # Calculate noise standard deviation for each sample
    noise_std_per_sample = np.abs(signal) * noise_ratio

    # Generate non-stationary Gaussian noise
    noise = noise_std_per_sample * np.random.normal(0, 1, signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal



def visualize_matrix(A, title, save_path=None):
    """
    Visualize a matrix A as a heatmap using a diverging colormap centered at zero.
    
    Parameters:
    - A (np.ndarray): The matrix to be visualized (can be complex; only real part is shown)
    - title (str): Title of the plot (currently not displayed)
    - save_path (str, optional): If provided, saves the plot as a PDF to this path
    
    Returns:
    - None
    """
    # Determine the maximum absolute value for symmetric color scaling
    vmax = np.abs(A).max()

    # Create the figure and axis with high resolution
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Create grid for pcolormesh (one more than shape in both directions)
    X, Y = np.meshgrid(np.arange(A.shape[1] + 1), np.arange(A.shape[0] + 1))

    # Invert y-axis to display matrix top-to-bottom
    ax.invert_yaxis()

    # Create pseudocolor plot of the real part of the matrix
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)

    # Optional title (commented out for now)
    # plt.title(f"{title}")

    # Add a colorbar and adjust its tick label size
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    # Adjust subplot layout for tight margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save figure as PDF if a path is provided
    if save_path:
        plt.savefig(f"{save_path}", format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()




def predict_dmd_with_input(A, B, x_0, U, k):
    """
    Predict the future states from the 1st step to the k-th step using the Dynamic Mode Decomposition (DMD) method, considering input u.

    Parameters:
        A (numpy.ndarray): The state transition matrix.
        B (numpy.ndarray): The matrix relating the input to the state.
        x_0 (numpy.ndarray): The current state vector.
        U (numpy.ndarray): The input matrix, where each column represents the input for a time step.
        k (int): The number of steps to predict.

    Returns:
        numpy.ndarray: A matrix containing the predicted states from the 1st step to the k-th step, 
                        with each column representing the state at a specific time step.
    """
    # Perform eigenvalue decomposition to get the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Project the initial state onto the eigenvector space, solving for the coefficients (b) in the eigenvector basis
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute the eigenvalues raised to the powers from 1 to k (for time steps 1 to k)
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Predict the states without input first
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    # Incorporate the effect of the input u at each time step
    for step in range(k):
        input_effect = B @ U[:, step]  # Assuming U is a matrix where each column is the input at a time step
        predicted_states[:, step] += input_effect
    
    return predicted_states


def project_matrix_elementwise(A):
    """
    Project a 2x2 matrix elementwise into a specific structure:
    [
        [0, 1],
        [?, ?]
    ]
    
    This function enforces the top row to be [0, 1] and retains or overrides
    the lower row based on desired structure.

    Parameters:
    - A (np.ndarray): Input 2x2 matrix to be projected

    Returns:
    - np.ndarray: Projected 2x2 matrix
    """
    # Initialize a 2x2 zero matrix
    projected_matrix = np.zeros((2, 2))

    # Set the first row to [0, 1] (fixed structure)
    projected_matrix[0, 0] = 0
    projected_matrix[0, 1] = 1

    # Set the second row based on specific values or logic
    projected_matrix[1, 0] = A[1, 0]         # Preserve original A[1, 0]
    projected_matrix[1, 1] = -0.1            # Set A[1, 1] to a fixed value

    return projected_matrix



def plot_comparison_separate(X_test, predicted_states_grad, k_steps):
    """
    Plot the comparison of true data and predicted states for x0 and x1 on separate figures.
    
    Parameters:
        X_test (numpy.ndarray): True data for testing (state matrix).
        predicted_states_grad (numpy.ndarray): Predicted states using Online DMD.
        k_steps (int): The number of prediction steps (time steps).
    """
    # Plot for x0 (first state dimension)
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(1, k_steps, k_steps), X_test[0, :], label='True Data (x0)', color='blue', linestyle='-', linewidth=2)
    plt.plot(np.linspace(1, k_steps, k_steps), predicted_states_grad[0, :], label='Predicted (x0) (Online DMD)', color='green', linestyle=':', linewidth=2)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('State Value', fontsize=14)
    plt.title('Comparison of True Data and Predicted States for x0', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot for x1 (second state dimension)
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(1, k_steps, k_steps), X_test[1, :], label='True Data (x1)', color='orange', linestyle='-', linewidth=2)
    plt.plot(np.linspace(1, k_steps, k_steps), predicted_states_grad[1, :], label='Predicted (x1) (Online DMD)', color='red', linestyle=':', linewidth=2)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('State Value', fontsize=14)
    plt.title('Comparison of True Data and Predicted States for x1', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_frobenius_norm_vs_time(Frobenius_norms, sample_interval=0.001, save_path=None):
    # Calculate time for each iteration (assuming sampling interval is 0.001)
    time = np.arange(0, len(Frobenius_norms) * sample_interval, sample_interval)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(time, Frobenius_norms, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=14, family='Times New Roman')
    plt.ylabel('Frobenius Norm', fontsize=14, family='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # If a save path is provided, save the plot as a high-quality PDF
    if save_path:
        filename = f"{save_path}/Frobenius_norm_plot.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved at {filename}")

    # Display the plot
    plt.show()



# Set random seed for reproducibility
np.random.seed(0)

# Load the data
state_load_path = 'D:\\博士课题\\小论文\\基于APSM方法的最优估计\\论文图片\\Duffing\\state_data.npy'
input_load_path = 'D:\\博士课题\\小论文\\基于APSM方法的最优估计\\论文图片\\Duffing\\load_data.npy'


# Define input matrix B (can be modified based on prior knowledge)
B = np.eye(2)  # Identity input matrix for simplicity

# Load state and input data
X = np.load(state_load_path).T
U = np.load(state_load_path).T

# Learning rate for optimization
# learning_rate_grad = 0.01  # When B_k is unknown
learning_rate_grad = 0.005  # When B_k is known


dt = 0.001  # Time step

# Add non-stationary Gaussian noise to the training data
noise_ratio = 0  # Set to > 0 to add noise
X_noisy = add_nonstationary_gaussian_noise(X, noise_ratio)

# Set prediction steps
k_steps = 1000  # Number of prediction steps
x_0 = X[:, -1 - k_steps]  # Use the (k+1)-th last time point as the initial state


# Split the dataset into training and testing segments
X_train = X_noisy[:, :-1 - k_steps]      # Training states (noisy)
U_train = U[:, :-1 - k_steps]            # Training inputs
X_test = X[:, -k_steps:]                 # Ground-truth test states
X_test_noisy = X_noisy[:, -k_steps:]     # Noisy test states (optional)
U_test = U[:, -k_steps:]                 # Test inputs

# Run OPIDMD to optimize the A matrix
iterations = X_train.shape[1]-1

# Case 1: Initialize A with a zero matrix (when B is known)
A_initial_grad_0 = np.zeros((2, 2))
A_k_pure_grad_0, X_pure_pred_0, Frobenius_norms_grad_0 = APSMC_grad(
    X_train, A_initial_grad_0, U_train, B, learning_rate_grad, iterations
)
# Plot Frobenius norm between A_k and discretized Jacobian over time
plot_frobenius_norm_vs_time(Frobenius_norms_grad_0)


# Case 2: Initialize A with the Jacobian matrix at the first state (when B is known)
x_0 = X_train[:, 0:1]
J_k = jacobian(x_0, c=0.1, alpha=1, beta=1)
A_initial_grad_J = bilinear_continuous_to_discrete_A(J_k, dt)
A_k_pure_grad_J, X_pure_pred_J, Frobenius_norms_grad_J = APSMC_grad(
    X_train, A_initial_grad_J, U_train, B, learning_rate_grad, iterations
)
plot_frobenius_norm_vs_time(Frobenius_norms_grad_J)


# Case 3: Initialize both A and B with identity matrices (when B is unknown)
A_initial = np.eye(2)
B_initial = np.eye(2)
A_k_physics_I, B_k_physics_I, X_physics_I, Frobenius_norms_A_physics_I, Frobenius_norms_B_physics_I = APSMC(
    X_train, A_initial, B_initial, U_train, B, learning_rate_grad, iterations
)
# Plot Frobenius norms over time for A_k and B_k updates
plot_frobenius_norm_vs_time(Frobenius_norms_A_physics_I)
plot_frobenius_norm_vs_time(Frobenius_norms_B_physics_I)

# Note: By default, APSMC applies the physics-constrained update. To run without physics constraints,
# modify APSMC to use `gradient_descent_update_unknown_B` instead of `gradient_descent_update_unknown_B_physics`.


# Case 4: Initialize A and B with values from the Jacobian and bilinear transform (when B is unknown)
J_k = jacobian(x_0, c=0.1, alpha=1, beta=1)
A_initial, B_initial = bilinear_continuous_to_discrete(J_k, B, dt)
A_k_physics_J, B_k_physics_J, X_physics_J, Frobenius_norms_A_physics_J, Frobenius_norms_B_physics_J = APSMC(
    X_train, A_initial, B_initial, U_train, B, learning_rate_grad, iterations
)
plot_frobenius_norm_vs_time(Frobenius_norms_A_physics_J)
plot_frobenius_norm_vs_time(Frobenius_norms_B_physics_J)






































