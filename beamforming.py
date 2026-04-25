import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def steering_vector(theta_deg, M=24, d_lambda=0.5):
    """
    Calculates the steering vector for a Uniform Linear Array.
    
    Parameters:
    theta_deg : Angle of arrival in degrees
    M         : Number of antenna elements
    d_lambda  : Element spacing in wavelengths
    
    Returns:
    A complex numpy column vector of shape (M, 1)
    """
    # 1. Convert the angle from degrees to radians
    theta_rad = np.radians(theta_deg)
    
    # 2. Create an array of element indices: [0, 1, 2, ..., M-1]
    m = np.arange(M)
    
    # 3. Calculate the phase shift for the entire array at once
    # Note: In Python, '1j' represents the imaginary unit (sqrt(-1))
    phase_shifts = np.exp(1j * 2 * np.pi * d_lambda * m * np.cos(theta_rad))
    
    # 4. Reshape the flat array into a strict column vector (M rows, 1 column)
    # This is critical for the matrix math later.
    v = phase_shifts.reshape(-1, 1)
    
    return v

def calculate_nsb_weights(theta_desired, theta_interferers, M=24, d_lambda=0.5):
    """
    Calculates the Null Steering Beamformer (NSB) complex weights.
    
    Parameters:
    theta_desired     : Angle of arrival for the desired signal in degrees
    theta_interferers : List or array of angles for interference signals in degrees
    M                 : Number of antenna elements
    d_lambda          : Element spacing in wavelengths
    
    Returns:
    Complex weight vector of shape (M, 1)
    """
    # 1. Generate the steering vector for the desired signal
    v_0 = steering_vector(theta_desired, M, d_lambda)
    
    # 2. Generate steering vectors for all interference signals
    # This uses a list comprehension, a very clean Python feature
    v_int = [steering_vector(theta, M, d_lambda) for theta in theta_interferers]
    
    # 3. Construct the constraint matrix C
    # np.hstack concatenates all the column vectors horizontally
    C = np.hstack([v_0] + v_int)
    
    # 4. Construct the desired response vector g
    # Size matches the number of columns in C. First element is 1, rest are 0.
    num_constraints = C.shape[1]
    g = np.zeros((num_constraints, 1))
    g[0, 0] = 1.0  
    
    # 5. Solve for the weight vector w
    # Equation: C^H * w = g 
    # w = pseudo_inverse(C^H) * g
    C_H = C.conj().T  # Hermitian transpose
    
    # The '@' operator in Python performs matrix multiplication
    w = np.linalg.pinv(C_H) @ g
    
    return w

def compute_pattern(w, M=24, d_lambda=0.5):
    """
    Computes the normalized radiation pattern in dB across 0 to 180 degrees.
    """
    # 1. Create the scanning array from 0 to 180 in 0.1 degree steps (1801 points)
    theta_scan = np.arange(0, 180.1, 0.1)
    theta_rad = np.radians(theta_scan)
    
    # 2. Vectorized phase matrix calculation
    # m is shaped as a column (24, 1), cos(theta) as a row (1, 1801)
    # Their matrix multiplication gives a (24, 1801) grid of phase values
    m = np.arange(M).reshape(-1, 1)
    phase_matrix = np.exp(1j * 2 * np.pi * d_lambda * m @ np.cos(theta_rad).reshape(1, -1))
    
    # 3. Calculate the Array Factor (AF) for all angles simultaneously
    # w^H * phase_matrix -> (1, 24) @ (24, 1801) = (1, 1801)
    AF = (w.conj().T @ phase_matrix).flatten()
    
    # 4. Convert to normalized dB
    AF_mag = np.abs(AF)
    AF_mag = np.maximum(AF_mag, 1e-12) # Prevent log10(0) errors
    P_dB = 20 * np.log10(AF_mag / np.max(AF_mag))
    
    return theta_scan, P_dB

def plot_pattern(theta_scan, P_dB, theta_desired, theta_interferers):
    """
    Plots the radiation pattern and overlays the target angles.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(theta_scan, P_dB, 'b-', linewidth=1.5, label='Radiation Pattern')
    
    # Mark the desired signal direction
    plt.axvline(x=theta_desired, color='g', linestyle='--', label='Desired Signal')
    
    # Mark the interference signal directions
    for i, th in enumerate(theta_interferers):
        label = 'Interferer' if i == 0 else "" # Only label the first one to avoid clutter
        plt.axvline(x=th, color='r', linestyle=':', label=label)
        
    # Draw the -20 dB threshold line for reference
    plt.axhline(y=-20, color='k', linestyle='--', alpha=0.5, label='-20 dB SLL Target')
    
    plt.ylim([-80, 5])
    plt.xlim([0, 180])
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('NSB Radiation Pattern')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def calculate_sinr(w, theta_desired, theta_interferers, snr_db, M=24, d_lambda=0.5):
    """
    Calculates the Signal-to-Interference-and-Noise Ratio (SINR) in dB.
    """
    # 1. Signal Power
    v_0 = steering_vector(theta_desired, M, d_lambda)
    P_S = np.abs((w.conj().T @ v_0)[0, 0])**2
    
    # 2. Interference Power
    P_I = 0
    for theta_i in theta_interferers:
        v_i = steering_vector(theta_i, M, d_lambda)
        P_I += np.abs((w.conj().T @ v_i)[0, 0])**2
        
    # 3. Noise Power
    # Input signal power is 1W (0 dBW). SNR = 10*log10(1 / sigma_n^2)
    sigma_n2 = 10 ** (-snr_db / 10)
    w_norm_sq = np.sum(np.abs(w)**2) # ||w||^2
    P_N = sigma_n2 * w_norm_sq
    
    # 4. SINR
    sinr_linear = P_S / (P_I + P_N)
    sinr_db = 10 * np.log10(sinr_linear)
    
    return sinr_db

def extract_deviations_and_sll(theta_scan, P_dB, theta_desired, theta_interferers):
    """
    Extracts the deviations (Delta theta) and the maximum Side Lobe Level (SLL).
    """
    # --- Deviations ---
    # Convert angles to array indices (0.1 degree steps means index = angle * 10)
    idx_desired = int(round(theta_desired * 10))
    
    # Search for the true peak within a +/- 3 degree window of the desired angle
    window = 30 # 3.0 degrees / 0.1 step
    search_start = max(0, idx_desired - window)
    search_end = min(len(P_dB), idx_desired + window)
    
    true_peak_idx = search_start + np.argmax(P_dB[search_start:search_end])
    true_peak_angle = theta_scan[true_peak_idx]
    delta_theta_0 = true_peak_angle - theta_desired
    
    # Search for true nulls within a +/- 3 degree window of interference angles
    delta_theta_nulls = []
    for theta_i in theta_interferers:
        idx_i = int(round(theta_i * 10))
        s_start = max(0, idx_i - window)
        s_end = min(len(P_dB), idx_i + window)
        
        true_null_idx = s_start + np.argmin(P_dB[s_start:s_end])
        true_null_angle = theta_scan[true_null_idx]
        delta_theta_nulls.append(true_null_angle - theta_i)
        
    # --- Side Lobe Level (SLL) ---
    # Find all peaks in the pattern
    peaks, _ = find_peaks(P_dB)
    
    # Mask out the main lobe to find the highest side lobe.
    # We estimate the main lobe bounds by finding where the pattern drops
    # significantly (e.g., -10 dB) around the peak, or just exclude a fixed window.
    # Let's exclude a 10-degree window around the desired signal for safety.
    main_lobe_mask = (theta_scan[peaks] > theta_desired - 10) & (theta_scan[peaks] < theta_desired + 10)
    
    side_lobe_peaks = peaks[~main_lobe_mask]
    
    if len(side_lobe_peaks) > 0:
        sll_db = np.max(P_dB[side_lobe_peaks])
    else:
        sll_db = -100 # Fallback if no side lobes are found
        
    return delta_theta_0, delta_theta_nulls, sll_db

def optimize_dummy_interferers(theta_desired, theta_interferers, max_dummies=18, target_sll=-20.0):
    """
    Iteratively places dummy interferers at the highest side lobes 
    until the SLL drops below the target, or max_dummies is reached.
    """
    current_dummies = []
    
    for iteration in range(max_dummies):
        # 1. Combine real interferers with current dummies
        all_interferers = theta_interferers + current_dummies
        
        # 2. Calculate current weights and pattern
        w = calculate_nsb_weights(theta_desired, all_interferers)
        angles, P_dB = compute_pattern(w)
        
        # 3. Find all peaks in the current pattern
        peaks, _ = find_peaks(P_dB)
        
        # Mask out the main lobe (roughly +/- 10 degrees from the desired signal)
        main_lobe_mask = (angles[peaks] > theta_desired - 10) & (angles[peaks] < theta_desired + 10)
        side_lobe_peaks = peaks[~main_lobe_mask]
        
        if len(side_lobe_peaks) == 0:
            break # No side lobes left to target
            
        # 4. Identify the worst offender (highest side lobe)
        highest_peak_idx = side_lobe_peaks[np.argmax(P_dB[side_lobe_peaks])]
        worst_sll_val = P_dB[highest_peak_idx]
        worst_sll_angle = angles[highest_peak_idx]
        
        # 5. Check if we met the target constraint
        if worst_sll_val <= target_sll:
            # Target met! Exit the control loop early.
            break
            
        # 6. Target not met: Add a dummy at the exact angle of the worst side lobe
        current_dummies.append(worst_sll_angle)
        
    # --- Final Polish ---
    # Calculate the definitive weight vector using all accumulated dummies
    final_interferers = theta_interferers + current_dummies
    final_w = calculate_nsb_weights(theta_desired, final_interferers)
    
    return final_w, current_dummies

def generate_scenarios(delta):
    """
    Generates the list of (theta_0, interferers) tuples per the assignment rules.
    """
    scenarios = []
    base_start = 30
    
    # Continue until the highest angle in the set exceeds 150 degrees
    while base_start + 5 * delta <= 150:
        # Create the base set of 6 angles
        base_angles = [base_start + i * delta for i in range(6)]
        
        # Rotate each angle to act as the desired signal
        for i in range(6):
            theta_0 = base_angles[i]
            # The other 5 angles are the interferers
            interferers = base_angles[:i] + base_angles[i+1:]
            scenarios.append((theta_0, interferers))
            
        base_start += 1
        
    return scenarios

def run_master_simulation():
    """
    Executes the full batch simulation for all SNRs and Deltas,
    and computes the required statistical metrics.
    """
    snr_values = [-10, 0, 10, 20]
    delta_values = [6, 8, 10, 12, 14, 16]
    
    print("Starting Master Simulation...")
    print("This will take a few minutes depending on your CPU. Grab a coffee.\n")
    
    start_time = time.time()
    
    for snr in snr_values:
        print(f"================ SNR = {snr} dB ================")
        
        for delta in delta_values:
            scenarios = generate_scenarios(delta)
            
            # Arrays to hold the results for this specific (SNR, Delta) combo
            all_delta_theta_0 = []
            all_delta_theta_nulls = []
            all_sinr = []
            all_sll = []
            
            for theta_0, interferers in scenarios:
                # 1. Optimize dummy placement to meet SLL target
                w_opt, active_dummies = optimize_dummy_interferers(
                    theta_0, interferers, max_dummies=18, target_sll=-20.0
                )
                
                # 2. Compute pattern
                all_int = interferers + active_dummies
                angles, P_dB = compute_pattern(w_opt)
                
                # 3. Extract metrics
                dt0, dtnulls, sll = extract_deviations_and_sll(angles, P_dB, theta_0, all_int)
                sinr_val = calculate_sinr(w_opt, theta_0, interferers, snr)
                
                # 4. Store metrics
                all_delta_theta_0.append(dt0)
                all_delta_theta_nulls.extend(dtnulls) # extend flattens the list of 5 nulls
                all_sinr.append(sinr_val)
                all_sll.append(sll)
            
            # --- Compute Statistics for this combination ---
            # Using numpy to easily calculate min, max, mean, std
            dt0_stats = [np.min(all_delta_theta_0), np.max(all_delta_theta_0), 
                         np.mean(all_delta_theta_0), np.std(all_delta_theta_0)]
                         
            dtn_stats = [np.min(all_delta_theta_nulls), np.max(all_delta_theta_nulls), 
                         np.mean(all_delta_theta_nulls), np.std(all_delta_theta_nulls)]
                         
            sinr_stats = [np.min(all_sinr), np.max(all_sinr), 
                          np.mean(all_sinr), np.std(all_sinr)]
                          
            sll_stats = [np.min(all_sll), np.max(all_sll), 
                         np.mean(all_sll), np.std(all_sll)]
            
            # --- Print formatted output for easy copy-pasting to Word ---
            print(f"--- Delta = {delta}° ---")
            print(f"Δθ0 [deg] : Min: {dt0_stats[0]:.3f} | Max: {dt0_stats[1]:.3f} | Mean: {dt0_stats[2]:.3f} | Std: {dt0_stats[3]:.3f}")
            print(f"Δθn [deg] : Min: {dtn_stats[0]:.3f} | Max: {dtn_stats[1]:.3f} | Mean: {dtn_stats[2]:.3f} | Std: {dtn_stats[3]:.3f}")
            print(f"SINR [dB] : Min: {sinr_stats[0]:.3f} | Max: {sinr_stats[1]:.3f} | Mean: {sinr_stats[2]:.3f} | Std: {sinr_stats[3]:.3f}")
            print(f"SLL  [dB] : Min: {sll_stats[0]:.3f} | Max: {sll_stats[1]:.3f} | Mean: {sll_stats[2]:.3f} | Std: {sll_stats[3]:.3f}\n")
            
    total_time = time.time() - start_time
    print(f"Simulation Complete! Total execution time: {total_time/60:.1f} minutes.")

# --- Execute ---
if __name__ == "__main__":
    run_master_simulation()