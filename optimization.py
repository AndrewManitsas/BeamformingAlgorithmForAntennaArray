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
from scipy.signal import find_peaks
from array_math import calculate_nsb_weights, calculate_mvdr_weights, compute_pattern

def optimize_dummy_interferers(theta_desired, theta_interferers, beamformer_type='nsb', snr_db=10, max_dummies=18, target_sll=-20.0):
    """Iteratively places dummy interferers to meet SLL target using the selected beamformer."""
    current_dummies = []
    
    for _ in range(max_dummies):
        all_interferers = theta_interferers + current_dummies
        
        # --- BEAMFORMER SELECTION TOGGLE ---
        if beamformer_type.lower() == 'mvdr':
            w = calculate_mvdr_weights(theta_desired, all_interferers, snr_db)
        else:
            w = calculate_nsb_weights(theta_desired, all_interferers)
            
        angles, P_dB = compute_pattern(w)
        
        peaks, _ = find_peaks(P_dB)
        main_lobe_mask = (angles[peaks] > theta_desired - 10) & (angles[peaks] < theta_desired + 10)
        side_lobe_peaks = peaks[~main_lobe_mask]
        
        if len(side_lobe_peaks) == 0:
            break
            
        highest_peak_idx = side_lobe_peaks[np.argmax(P_dB[side_lobe_peaks])]
        worst_sll_val = P_dB[highest_peak_idx]
        
        if worst_sll_val <= target_sll:
            break
            
        current_dummies.append(angles[highest_peak_idx])
        
    # Final weight calculation with all accumulated dummies
    final_interferers = theta_interferers + current_dummies
    if beamformer_type.lower() == 'mvdr':
        final_w = calculate_mvdr_weights(theta_desired, final_interferers, snr_db)
    else:
        final_w = calculate_nsb_weights(theta_desired, final_interferers)
    
    return final_w, current_dummies