import numpy as np
from scipy.signal import find_peaks
from array_math import calculate_nsb_weights, compute_pattern

def optimize_dummy_interferers(theta_desired, theta_interferers, max_dummies=18, target_sll=-20.0):
    """Iteratively places dummy interferers at highest side lobes to meet SLL target."""
    current_dummies = []
    
    for _ in range(max_dummies):
        all_interferers = theta_interferers + current_dummies
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
        
    final_interferers = theta_interferers + current_dummies
    final_w = calculate_nsb_weights(theta_desired, final_interferers)
    
    return final_w, current_dummies