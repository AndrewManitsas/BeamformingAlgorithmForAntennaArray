import numpy as np

def steering_vector(theta_deg, M=24, d_lambda=0.5):
    """Calculates the steering vector for a Uniform Linear Array."""
    theta_rad = np.radians(theta_deg)
    m = np.arange(M)
    phase_shifts = np.exp(1j * 2 * np.pi * d_lambda * m * np.cos(theta_rad))
    return phase_shifts.reshape(-1, 1)

def calculate_nsb_weights(theta_desired, theta_interferers, M=24, d_lambda=0.5):
    """Calculates the Null Steering Beamformer (NSB) complex weights."""
    v_0 = steering_vector(theta_desired, M, d_lambda)
    v_int = [steering_vector(theta, M, d_lambda) for theta in theta_interferers]
    
    C = np.hstack([v_0] + v_int)
    
    num_constraints = C.shape[1]
    g = np.zeros((num_constraints, 1))
    g[0, 0] = 1.0  
    
    C_H = C.conj().T
    w = np.linalg.pinv(C_H) @ g
    return w

def compute_pattern(w, M=24, d_lambda=0.5):
    """Computes the normalized radiation pattern in dB across 0 to 180 degrees."""
    theta_scan = np.arange(0, 180.1, 0.1)
    theta_rad = np.radians(theta_scan)
    
    m = np.arange(M).reshape(-1, 1)
    phase_matrix = np.exp(1j * 2 * np.pi * d_lambda * m @ np.cos(theta_rad).reshape(1, -1))
    
    AF = (w.conj().T @ phase_matrix).flatten()
    
    AF_mag = np.abs(AF)
    AF_mag = np.maximum(AF_mag, 1e-12)
    P_dB = 20 * np.log10(AF_mag / np.max(AF_mag))
    
    return theta_scan, P_dB