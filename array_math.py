# ==============================================================================
# Title:       Simulation of a Null Steering (NSB) and the Minimum Variance Distortionless Response (MVDR) beamforming algorithms,
#              developed for the Advanced RF Aspects of UAVs MSc course (UAV15)
#
# Author:      Andreas Manitsas
# Email:       amanitsb@ece.auth.gr
# 
# Course:      UAV15 Advanced RF Aspects of UAVs
# Program:     MSc Aerial Autonomous Systems
# Institution: Aristotle University of Thessaloniki, Faculty of Polytechnics
# Term:        2025-2026
#
# Description: The software simulates a 24-element uniform linear antenna array.
#              Depending on the selected algorithm, it calculates complex weights to
#              steer the main lobe toward a desired signal while suppressing incoming
#              interference signals (either through deterministic mathematical nulls or
#              statistical spatial covariance). To meet strict Side Lobe Level (SLL) constraints -20 dB,
#              it utilizes an iterative peak-nulling algorithm to strategically place artificial "dummy" interferers.
#
# Disclaimer:  AI assistance may have been used during development
# ==============================================================================

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

def calculate_mvdr_weights(theta_desired, theta_interferers, snr_db, M=24, d_lambda=0.5):
    """Calculates the Minimum Variance Distortionless Response (MVDR) complex weights."""
    v_0 = steering_vector(theta_desired, M, d_lambda)
    
    # 1. Calculate Noise Power Variance
    sigma_n2 = 10 ** (-snr_db / 10)
    
    # 2. Construct the spatial covariance matrix (R)
    # CRITICAL FIX: Explicitly set dtype to complex so numpy can add the steering vectors
    R = sigma_n2 * np.eye(M, dtype=complex)
    
    # Add the desired signal covariance
    R += v_0 @ v_0.conj().T
    
    # Add the interference signal covariances
    for theta_i in theta_interferers:
        v_i = steering_vector(theta_i, M, d_lambda)
        R += v_i @ v_i.conj().T
        
    # 3. Calculate MVDR weights: w = (R^-1 * v_0) / (v_0^H * R^-1 * v_0)
    R_inv = np.linalg.pinv(R)
    
    numerator = R_inv @ v_0
    denominator = v_0.conj().T @ R_inv @ v_0
    
    w = numerator / denominator
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