# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
import pandas as pd
import os


def gradient_descent_update_unknown_B_physics(A_k, B_k, M, x_k, y_k, u_k, learning_rate):
    """
    Perform gradient descent update for both A_k and B_k using state x_k, 
    target y_k, and input u_k, with additional physics-informed projection.

    Parameters:
    - A_k (np.ndarray): State transition matrix to be updated
    - B_k (np.ndarray): Input matrix to be updated
    - M (np.ndarray): Optional prior or model constraint (currently unused)
    - x_k (np.ndarray): State vector (column)
    - y_k (np.ndarray): Target output (column)
    - u_k (np.ndarray): Input vector (column)
    - learning_rate (float): Learning rate for gradient descent

    Returns:
    - A_k (np.ndarray): Updated A_k matrix after projection
    - B_k (np.ndarray): Updated B_k matrix
    """
    # Compute prediction error
    error = y_k - A_k.dot(x_k) - B_k @ u_k

    # Compute gradients of the loss with respect to A_k and B_k
    grad_A_k = -2 * error * x_k.T
    grad_B_k = -2 * error * u_k.T

    # Update A_k and B_k using gradient descent
    A_k -= learning_rate * grad_A_k
    B_k -= learning_rate * grad_B_k

    # Convert A_k to continuous domain
    A_con = bilinear_discrete_to_continuous_A(A_k, 0.02)

    # Project A_con onto a physically constrained subspace
    A_re = projection_onto_system_matrices_matrices(A_con)

    # Convert back to discrete domain
    A_k = bilinear_continuous_to_discrete_A(A_re, 0.02)

    return A_k, B_k



def gradient_descent_update_unknown_B(A_k, B_k, M, x_k, y_k, u_k, learning_rate):
    """
    Perform standard gradient descent update for both A_k and B_k using state x_k, 
    target y_k, and input u_k (no physics constraints).

    Parameters:
    - A_k (np.ndarray): State transition matrix to be updated
    - B_k (np.ndarray): Input matrix to be updated
    - M (np.ndarray): Placeholder (not used here)
    - x_k (np.ndarray): State vector (column)
    - y_k (np.ndarray): Target output (column)
    - u_k (np.ndarray): Input vector (column)
    - learning_rate (float): Learning rate for gradient descent

    Returns:
    - A_k (np.ndarray): Updated A_k matrix
    - B_k (np.ndarray): Updated B_k matrix
    """
    # Compute prediction error
    error = y_k - A_k.dot(x_k) - B_k @ u_k

    # Compute gradients of the loss with respect to A_k and B_k
    grad_A_k = -2 * error * x_k.T
    grad_B_k = -2 * error * u_k.T

    # Apply gradient updates
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



def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE, NMSE, and R^2 metrics.
    Adjusts for complex data by using only the real part of the arrays.

    Parameters:
    - actual (np.ndarray): The ground truth values (can be complex).
    - predicted (np.ndarray): The predicted values (can be complex).

    Returns:
    - mse (float): Mean Squared Error between actual and predicted values.
    - nmse (float): Normalized Mean Squared Error (MSE divided by variance of actual values).
    - r2 (float): Coefficient of Determination (R^2 score).
    """
    # Extract real part if data is complex
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(actual_real, predicted_real)

    # Compute Normalized Mean Squared Error (NMSE)
    variance_actual = np.var(actual_real)
    nmse = mse / variance_actual

    # Compute R^2 (coefficient of determination)
    r2 = r2_score(actual_real, predicted_real)

    return mse, nmse, r2




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
    Convert a continuous-time state-space system to discrete-time using the bilinear (Tustin) transform.

    Parameters:
    - A_cont (np.ndarray): Continuous-time system matrix A
    - B_cont (np.ndarray): Continuous-time input matrix B
    - delta_t (float): Sampling time interval

    Returns:
    - A_d (np.ndarray): Discrete-time system matrix (Phi)
    - B_d (np.ndarray): Discrete-time input matrix (Gamma)
    """
    I = np.eye(A_cont.shape[0])  # Identity matrix of the same shape as A_cont

    # Compute the discrete-time A matrix using the bilinear transform
    A_d = np.linalg.inv(I - (delta_t / 2) * A_cont).dot(I + (delta_t / 2) * A_cont)

    # Compute the discrete-time B matrix
    B_d = np.linalg.inv(I - (delta_t / 2) * A_cont).dot(delta_t * B_cont)

    return A_d, B_d


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
    vmax = np.abs(A).max()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    ax.invert_yaxis()
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)
    # plt.title(f"{title}")

    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    

    if save_path:
        plt.savefig(f"{save_path}", format='pdf', bbox_inches='tight')
    
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



def plot_comparison_multi_method(X_test, 
                                  predicted_states_grad, 
                                  predicted_states_APSMC, 
                                  predicted_states_DMDC, 
                                  k_steps,
                                  save_path=None):
    """
    Plot comparison of true and predicted states across 12 dimensions (6 displacement, 6 velocity).

    Parameters:
    - X_test: Ground truth, shape (12, k_steps)
    - predicted_states_grad: Predicted states by Online DMD
    - predicted_states_APSMC: Predicted states by APSMC
    - predicted_states_DMDC: Predicted states by DMDc
    - k_steps: Number of time steps
    - save_path: Optional path to save the plots; if None, only display
    """
    time_steps = np.linspace(1, k_steps, k_steps)

    for i in range(12):
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, X_test[i, :], label='True', color='black', linestyle='-', linewidth=2)
        plt.plot(time_steps, predicted_states_grad[i, :], label='APSMC Unconstrained', color='green', linestyle=':', linewidth=2)
        plt.plot(time_steps, predicted_states_APSMC[i, :], label='APSMC Constrained', color='blue', linestyle='--', linewidth=2)
        plt.plot(time_steps, predicted_states_DMDC[i, :], label='DMDc', color='red', linestyle='-.', linewidth=2)

        plt.xlabel('Time Step', fontsize=14)

        # Set Y label
        ylabel = 'Displacement' if i < 6 else 'Velocity'
        plt.ylabel(ylabel, fontsize=14)

        # Fixed legend at upper right
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # Save or show
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f'state_{i+1}_plot.pdf')
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.show()
        else:
            plt.show()



def plot_frobenius_norm_vs_time(Frobenius_norms, sample_interval=0.001, save_path="C:\\Users\\HIT\\Desktop"):
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




def plot_input_signal(time, U, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot the ground motion signal and save it as a PDF file.

    Parameters:
    - time: 1D array, time vector
    - U: 1xN array, ground motion signal
    - save_path: The directory where the plot will be saved
    """
    sns.set(style="white", context="talk")  # White background for better readability

    plt.figure(figsize=(12, 8), dpi=300)

    # Set font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    # Plot the ground motion
    plt.plot(time, U.flatten(), label='Ground Motion', linestyle='-', color='blue', linewidth=2.5)

    # Axis labels
    plt.xlabel('Time (seconds)', fontsize=30)
    plt.ylabel('Acceleration (g)', fontsize=30)

    # Legend customization
    legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)

    # Tick customization
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Dashed grid lines
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "input_signal_plot.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()
    
from scipy.signal import welch


def plot_input_psd(time, U, fs=100.0):
    """
    Plot the Power Spectral Density (PSD) of the input signal.

    Parameters:
    - time (np.ndarray): 1D array representing the time vector (e.g., t[:-1])
    - U (np.ndarray): 1xN or Nx1 array representing the input signal (e.g., ground motion)
    - fs (float): Sampling frequency in Hz (default: 100.0)
    """

    sns.set(style="white", context="talk")

    plt.figure(figsize=(12, 8), dpi=300)

    # Set global font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    # Compute Power Spectral Density using Welch's method
    f, Pxx = welch(U.flatten(), fs=fs, nperseg=1024)

    # Plot PSD on a semilog-y scale
    plt.semilogy(f, Pxx, label='Ground Motion', linestyle='-', color='blue', linewidth=2.5)

    # Axis labels
    plt.xlabel('Frequency (Hz)', fontsize=30)
    plt.ylabel('Power Spectral Density (g²/Hz)', fontsize=30)

    # Legend settings
    legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)

    # Tick settings
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Add dashed grid for better readability
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Display the plot
    plt.show()


    
def projection_onto_symmetric(A):
    """
    Project a given matrix onto the space of symmetric matrices.

    This operation ensures that the output matrix satisfies A_sym = A_sym.T.

    Parameters:
    - A (np.ndarray): Input matrix (not necessarily symmetric)

    Returns:
    - np.ndarray: Symmetric matrix obtained by averaging A and its transpose
    """
    return (A + A.T) / 2


def projection_onto_system_matrices_matrices(A_general):
    """
    Project a general system matrix A_general onto a structured form representing 
    a second-order mechanical system (without using a mass matrix M).

    The assumed structure of A_general is:
        [ 0     I  ]
        [ K_sym C_sym ]

    Where:
        - K_sym is the symmetric stiffness-like component
        - C_sym is the symmetric damping-like component

    This projection extracts the lower half blocks (K and C), symmetrizes them,
    and reconstructs A with fixed identity and zero blocks in the upper half.

    Parameters:
    - A_general (np.ndarray): General (unstructured) system matrix, expected shape (2n x 2n)

    Returns:
    - A_reconstructed (np.ndarray): Structured matrix with projected symmetric K and C blocks
    """
    n = 6  # Number of degrees of freedom (assumed fixed to 6 for now)

    # Extract K and C blocks from lower half
    K = A_general[n:, :n]
    C = A_general[n:, n:]

    # Project K and C onto the space of symmetric matrices
    K = (K + K.T) / 2
    C = (C + C.T) / 2

    # Reconstruct A in block form: [0 I; K C]
    A_reconstructed = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [K, C]
    ])

    return A_reconstructed

    
def kalman_filter_update(A, B, C, Q, R, x_hat, P, u, y):
    def predict(x_hat, P, A, B, u, Q):
        x_hat_pred = A @ x_hat + B @ u
        P_pred = A @ P @ A.T + Q
        return x_hat_pred, P_pred

    def update(x_hat_pred, P_pred, y, C, R):
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_hat = x_hat_pred + K @ (y - C @ x_hat_pred)
        P = (np.eye(len(P_pred)) - K @ C) @ P_pred
        return x_hat, P

    x_hat_pred, P_pred = predict(x_hat, P, A, B, u, Q)
    x_hat, P = update(x_hat_pred, P_pred, y, C, R)
    
    return x_hat, P, C @ x_hat_pred

def APSMC_algorithm(combined_matrix, U_matrix, M, learning_rate, iterations, A, B, C_matrix, P, Q, R):
    """
    APSMC algorithm implementation with physics-constrained gradient descent and Kalman filtering.

    Parameters:
    - combined_matrix (np.ndarray): Output data matrix (each column is a time step)
    - U_matrix (np.ndarray): Input data matrix (each column is a control input)
    - M (np.ndarray): Mass matrix used in physics-based projection
    - learning_rate (float): Learning rate for gradient descent
    - iterations (int): Number of time steps for the update loop
    - A (np.ndarray): Initial state transition matrix
    - B (np.ndarray): Initial control input matrix
    - C_matrix (np.ndarray): Observation matrix
    - P (np.ndarray): Initial error covariance matrix
    - Q (np.ndarray): Process noise covariance matrix
    - R (np.ndarray): Measurement noise covariance matrix

    Returns:
    - x_estimates (np.ndarray): Estimated state trajectory over time
    - A (np.ndarray): Updated state transition matrix after training
    - B (np.ndarray): Updated input matrix after training
    - y_estimates (np.ndarray): Predicted outputs (observations) over time
    """
    n = A.shape[0]
    m = C_matrix.shape[0]

    x_hat = np.zeros((n, 1))                # Initial state estimate
    x_estimates = np.zeros((n, iterations)) # To store state estimates at each step
    y_estimates = np.zeros((m, iterations)) # To store predicted outputs at each step

    for k in range(iterations):
        x_hat_old = x_hat
        y_k = combined_matrix[:, k].reshape(-1, 1)   # Current observation
        u_k = U_matrix[:, k].reshape(-1, 1)          # Current control input

        # Kalman filter update using current model
        x_hat, P, y_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)

        # Update A and B using gradient descent with physics-informed projection
        A, B = gradient_descent_update_unknown_B_physics(A, B, M, x_hat_old, x_hat, u_k, learning_rate)

        # Store current estimates
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()

    return x_estimates, A, B, y_estimates



def APSMC_algorithm_grad(combined_matrix, U_matrix, M, learning_rate, iterations, A, B, C_matrix, P, Q, R):
    """
    APSMC algorithm with gradient-based A/B matrix updates embedded in a Kalman filtering loop.

    Parameters:
    - combined_matrix (np.ndarray): Combined measurement/output matrix (each column is a time step).
    - U_matrix (np.ndarray): Input matrix (each column is a control input at a time step).
    - M (np.ndarray): Placeholder or reference signal used for gradient update.
    - learning_rate (float): Learning rate for gradient descent updates.
    - iterations (int): Number of time steps (iterations) to run the algorithm.
    - A (np.ndarray): Initial state transition matrix (to be updated).
    - B (np.ndarray): Initial input matrix (to be updated).
    - C_matrix (np.ndarray): Observation matrix (maps states to outputs).
    - P (np.ndarray): Initial error covariance matrix.
    - Q (np.ndarray): Process noise covariance matrix.
    - R (np.ndarray): Measurement noise covariance matrix.

    Returns:
    - x_estimates (np.ndarray): Estimated states over time (n x iterations).
    - A (np.ndarray): Final updated state transition matrix.
    - B (np.ndarray): Final updated input matrix.
    - y_estimates (np.ndarray): Predicted measurements over time (m x iterations).
    """
    n = A.shape[0]  # State dimension
    m = C_matrix.shape[0]  # Measurement dimension

    x_hat = np.zeros((n, 1))  # Initial state estimate
    x_estimates = np.zeros((n, iterations))  # Storage for state estimates
    y_estimates = np.zeros((m, iterations))  # Storage for measurement predictions

    for k in range(iterations):
        x_hat_old = x_hat  # Previous state estimate
        y_k = combined_matrix[:, k].reshape(-1, 1)  # Current measurement
        u_k = U_matrix[:, k].reshape(-1, 1)  # Current input

        # Kalman filter update using current A matrix
        x_hat, P, y_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)

        # Gradient-based update of A and B using the Kalman-estimated state
        A, B = gradient_descent_update_unknown_B(A, B, M, x_hat_old, x_hat, u_k, learning_rate)

        # Store current estimates
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()

    return x_estimates, A, B, y_estimates


def compute_dmdc_matrices(X, U):
    """
    Compute the system matrices A and B using Dynamic Mode Decomposition with Control (DMDc).

    Parameters:
    X (numpy.ndarray): The matrix of all responses (outputs) with shape (time_steps, n_outputs).
    U (numpy.ndarray): The matrix of all inputs with shape (time_steps, n_inputs).

    Returns:
    A (numpy.ndarray): The system matrix A.
    B (numpy.ndarray): The input matrix B.
    """
    # Ensure the inputs are numpy arrays
    X = np.array(X)
    U = np.array(U)
    # Construct the data matrix Ω
    Ω = np.hstack((X[:-1, :], U[:-1, :])).T

    # Create the shifted data matrices X and X'
    X_prime = X[1:, :]
    # Compute the system matrix G = [A B]
    G = X_prime.T @ np.linalg.pinv(Ω)
    
    # Extract A and B from G
    n_outputs = X.shape[1]
    A = G[:, :n_outputs]
    B = G[:, n_outputs:]
    
    return A, B

def plot_all_clean_vs_noisy(time, clean_data, noisy_data, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot comparison figures between clean and noisy signals for all 12 dimensions.
    Optionally save each figure as a PDF if save_path is provided.

    Parameters:
    - time (np.ndarray): 1D array, time vector (seconds)
    - clean_data (np.ndarray): 2D array of shape (12, time_steps), clean/original data
    - noisy_data (np.ndarray): 2D array of shape (12, time_steps), noisy data
    - save_path (str or None): Directory to save plots. If None, plots are not saved.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white", context="talk")

    # Set font globally
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    # Create directory if saving is enabled
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for i in range(12):
        plt.figure(figsize=(12, 8), dpi=300)

        # Plot clean signal
        plt.plot(time, clean_data[i, :], label='Original Data', linestyle='-', color='blue', linewidth=2.5)

        # Plot noisy signal
        plt.plot(time, noisy_data[i, :], label='Noisy Data', linestyle=':', color='orange', linewidth=2.5)

        # Axis labels
        plt.xlabel('Time (seconds)', fontsize=25)
        ylabel = 'Displacement' if i < 6 else 'Velocity'
        plt.ylabel(ylabel, fontsize=25)

        # Legend configuration
        legend = plt.legend(loc='upper right', fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
        legend.get_frame().set_linewidth(1.5)

        # Tick and grid settings
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path is not None:
            filename = os.path.join(save_path, f'clean_vs_noisy_mass_{i+1}.pdf')
            plt.savefig(filename, format='pdf', bbox_inches='tight')

        # Show the plot
        plt.show()










# The paper presents the results with added noise by default.
# To obtain noise-free results, you can control the ratio of non-stationary Gaussian noise accordingly.







# Load the data
accl_load_path = r'D:\博士课题\小论文\基于APSM方法的最优估计\代码\数据\完整数据\RSN968_NORTHR_DWN360.AT2\absAccel.xlsx'
relaDisp_load_path = r'D:\博士课题\小论文\基于APSM方法的最优估计\代码\数据\完整数据\RSN968_NORTHR_DWN360.AT2\relaDisp.xlsx'
relaVel_load_path = r'D:\博士课题\小论文\基于APSM方法的最优估计\代码\数据\完整数据\RSN968_NORTHR_DWN360.AT2\relaVel.xlsx'
input_load_path = r'D:\博士课题\小论文\基于APSM方法的最优估计\代码\数据\完整数据\RSN968_NORTHR_DWN360.AT2\ground_motion.xlsx'

# Read Excel files and convert to NumPy arrays
accl = pd.read_excel(accl_load_path).to_numpy()
relaDisp = pd.read_excel(relaDisp_load_path).to_numpy()
relaVel = pd.read_excel(relaVel_load_path).to_numpy()
U = pd.read_excel(input_load_path).to_numpy().T  # Transpose to shape (1, time_steps)

# Extract time vector and acceleration data
t = accl[:, 0]           # Time vector, shape (3000,)
accl_data = accl[:, 1:]  # Absolute acceleration, shape (3000, 6)

# Use t[:-1] for plotting input signal and PSD
t_plot = t[:-1]
plot_input_signal(t_plot, U)
plot_input_psd(t_plot, U, fs=50.0)

# Stack displacement and velocity, then transpose to shape (12, time_steps)
X = np.hstack([relaDisp, relaVel]).T  # Shape: (12, 3000)

# Add non-stationary Gaussian noise with 30% intensity
X_noisy = add_nonstationary_gaussian_noise(X, 0.3)

# Visualize clean vs noisy signals (12 dimensions), no saving
plot_all_clean_vs_noisy(t, X, X_noisy, save_path=None)

# Trim data to time range [800, 2000]
X = X[:, 800:2000]  # Shape: (12, 1200)
U = U[:, 800:2000]  # Shape: (1, 1200)

# Expand input signal U to shape (12, time_steps), placing ground motion in last row
U_new = np.zeros((12, U.shape[1]))  # Shape: (12, 1200)
U_new[-1, :] = U  # Assign U to the last row

# Set random seed for reproducibility
np.random.seed(0)

# Learning rates for gradient-based and APSMC-based methods
learning_rate_grad = 3.3     # For noisy data (0.3 noise)
learning_rate_APSMC = 5      # For noisy data (0.3 noise)
# learning_rate_grad = 3.3     # For noise-free data (default override)
# learning_rate_APSMC = 10     # For noise-free data (default override)


# Add non-stationary Gaussian noise to the state data
noise_ratio = 0.3  # Set to 0 for noise-free, >0 for noisy data
X_noisy = add_nonstationary_gaussian_noise(X, noise_ratio)

# Set number of prediction steps
k_steps = 35  # How many steps to forecast into the future

# Initial state for prediction (taken from (k+1)-th last time point)
x_0 = X[:, -1 - k_steps]

# Split data into training and testing sets
X_train = X_noisy[:, :-1 - k_steps]       # Noisy training data
X_train_free = X[:, :-1 - k_steps]        # Clean version of training data (optional use)
U_train = U_new[:, :-1 - k_steps]         # Training input
X_test = X[:, -k_steps:]                  # Ground truth test output
X_test_noisy = X_noisy[:, -k_steps:]      # Noisy test output (optional use)
U_test = U_new[:, -k_steps:]              # Test input

# Dimensions of system
n = 12  # State dimension
m = 12  # Output dimension

# System matrices initialization
C_matrix = np.eye(n)         # Observation matrix (identity: full state observed)
P = np.eye(n) * 0.01         # Initial error covariance
Q = np.eye(n) * 0.01         # Process noise covariance
R = np.eye(m) * 0.1          # Measurement noise covariance (adjust based on noise level)
# R = np.eye(m) * 0.0        # Use this if there is no observation noise

A = np.eye(n)  # Initial A matrix (identity)
B = np.eye(n)  # Initial B matrix (identity)
M = np.eye(n)  # Placeholder matrix used in APSMC (e.g., mass matrix)

# Number of iterations = number of training time steps - 1
iterations = X_train.shape[1] - 1

# Compute DMDc matrices as baseline
A_DMDc, B_DMDc = compute_dmdc_matrices(X_train.T, U_train.T)

# Run APSMC with physics-based constraints
x_estimates_APSMC, A_updated, B_updated, y_estimates_APSMC = APSMC_algorithm(
    X_train, U_train, M, learning_rate_APSMC, iterations, A, B, C_matrix, P, Q, R
)

# Run APSMC with purely gradient-based updates
x_estimates_APSMC_grad, A_updated_grad, B_updated_grad, y_estimates_APSMC_grad = APSMC_algorithm_grad(
    X_train, U_train, M, learning_rate_grad, iterations, A, B, C_matrix, P, Q, R
)

# Generate predictions using each method
predicted_states_grad = predict_dmd_with_input(A_updated_grad, B_updated_grad, x_0, U_test, k_steps)
predicted_states_APSMC = predict_dmd_with_input(A_updated, B_updated, x_0, U_test, k_steps)
predicted_states_DMDC = predict_dmd_with_input(A_DMDc, B_DMDc, x_0, U_test, k_steps)

# Evaluate performance with standard metrics
mse_optimized_grad, nmse_grad, r2_grad = evaluate_predictions(X_test, predicted_states_grad)
mse_optimized_APSMC, nmse_APSMC, r2_grad_APSMC = evaluate_predictions(X_test, predicted_states_APSMC)
mse_optimized_DMDC, nmse_DMDC, r2_grad_DMDC = evaluate_predictions(X_test, predicted_states_DMDC)

# Print normalized mean square error for each method
print("NMSE for APSMC     :", nmse_APSMC)
print("NMSE for DMDc      :", nmse_DMDC)
print("NMSE for Grad-only :", nmse_grad)

# Plot multi-method comparison of test predictions
plot_comparison_multi_method(
    X_test,
    predicted_states_grad,
    predicted_states_APSMC,
    predicted_states_DMDC,
    k_steps,
    save_path=None # Set to None to only display without saving
)


A_dmdc = bilinear_discrete_to_continuous_A(A, 0.02)
A_gradc = bilinear_discrete_to_continuous_A(A_updated_grad, 0.02)
A_APSMC = bilinear_discrete_to_continuous_A(A_updated, 0.02)
A_DMDc_c = bilinear_discrete_to_continuous_A(A_DMDc, 0.02)



visualize_matrix(A_dmdc , 'A_DMDc')
visualize_matrix(A_gradc, 'A_gradc' )
visualize_matrix(A_APSMC, 'A_APSMC' )
visualize_matrix(A_DMDc_c , 'A_DMDc_c')






