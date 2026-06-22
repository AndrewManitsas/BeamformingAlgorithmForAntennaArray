# ==============================================================================
# Title:       Simulation of a Null Steering Beamformer (NSB), developed for the Advanced RF Aspects of UAVs MSc course
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
#              It calculates complex weights to steer the main lobe toward a desired
#              signal while forcing mathematical nulls at the angles of incoming
#              interference signals. To meet strict Side Lobe Level (SLL) constraints,
#              it utilizes an iterative peak-nulling algorithm to strategically place
#              artificial "dummy" interferers.
#
# Disclaimer:  AI assistance may have been used during development
# ==============================================================================

import numpy as np
from scipy.signal import find_peaks
from array_math import steering_vector

def calculate_sinr(w, theta_desired, theta_interferers, snr_db, M=24, d_lambda=0.5):
    """Calculates the Signal-to-Interference-and-Noise Ratio (SINR) in dB."""
    v_0 = steering_vector(theta_desired, M, d_lambda)
    P_S = np.abs((w.conj().T @ v_0)[0, 0])**2
    
    P_I = 0
    for theta_i in theta_interferers:
        v_i = steering_vector(theta_i, M, d_lambda)
        P_I += np.abs((w.conj().T @ v_i)[0, 0])**2
        
    sigma_n2 = 10 ** (-snr_db / 10)
    w_norm_sq = np.sum(np.abs(w)**2)
    P_N = sigma_n2 * w_norm_sq
    
    sinr_linear = P_S / (P_I + P_N)
    return 10 * np.log10(sinr_linear)

def extract_deviations_and_sll(theta_scan, P_dB, theta_desired, theta_interferers):
    """Extracts the deviations (Delta theta) and the maximum Side Lobe Level (SLL)."""
    idx_desired = int(round(theta_desired * 10))
    window = 30 
    
    search_start = max(0, idx_desired - window)
    search_end = min(len(P_dB), idx_desired + window)
    true_peak_idx = search_start + np.argmax(P_dB[search_start:search_end])
    delta_theta_0 = theta_scan[true_peak_idx] - theta_desired
    
    delta_theta_nulls = []
    for theta_i in theta_interferers:
        idx_i = int(round(theta_i * 10))
        s_start = max(0, idx_i - window)
        s_end = min(len(P_dB), idx_i + window)
        true_null_idx = s_start + np.argmin(P_dB[s_start:s_end])
        delta_theta_nulls.append(theta_scan[true_null_idx] - theta_i)
        
    peaks, _ = find_peaks(P_dB)
    main_lobe_mask = (theta_scan[peaks] > theta_desired - 10) & (theta_scan[peaks] < theta_desired + 10)
    side_lobe_peaks = peaks[~main_lobe_mask]
    
    sll_db = np.max(P_dB[side_lobe_peaks]) if len(side_lobe_peaks) > 0 else -100
        
    return delta_theta_0, delta_theta_nulls, sll_db