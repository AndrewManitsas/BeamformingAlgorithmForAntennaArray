import time
import numpy as np

# Import our modularized functions
from array_math import compute_pattern
from metrics import calculate_sinr, extract_deviations_and_sll
from optimization import optimize_dummy_interferers

def generate_scenarios(delta):
    """Generates the list of (theta_0, interferers) tuples per the assignment rules."""
    scenarios = []
    base_start = 30
    
    while base_start + 5 * delta <= 150:
        base_angles = [base_start + i * delta for i in range(6)]
        for i in range(6):
            theta_0 = base_angles[i]
            interferers = base_angles[:i] + base_angles[i+1:]
            scenarios.append((theta_0, interferers))
        base_start += 1
        
    return scenarios

def run_master_simulation():
    """Executes the full batch simulation and computes required statistical metrics."""
    snr_values = [-10, 0, 10, 20]
    delta_values = [6, 8, 10, 12, 14, 16]
    
    print("Starting Master Simulation...")
    start_time = time.time()
    
    for snr in snr_values:
        print(f"================ SNR = {snr} dB ================")
        for delta in delta_values:
            scenarios = generate_scenarios(delta)
            
            all_delta_theta_0 = []
            all_delta_theta_nulls = []
            all_sinr = []
            all_sll = []
            
            for theta_0, interferers in scenarios:
                w_opt, active_dummies = optimize_dummy_interferers(
                    theta_0, interferers, max_dummies=18, target_sll=-20.0
                )
                
                all_int = interferers + active_dummies
                angles, P_dB = compute_pattern(w_opt)
                
                dt0, dtnulls, sll = extract_deviations_and_sll(angles, P_dB, theta_0, all_int)
                sinr_val = calculate_sinr(w_opt, theta_0, interferers, snr)
                
                all_delta_theta_0.append(dt0)
                all_delta_theta_nulls.extend(dtnulls) 
                all_sinr.append(sinr_val)
                all_sll.append(sll)
            
            dt0_stats = [np.min(all_delta_theta_0), np.max(all_delta_theta_0), np.mean(all_delta_theta_0), np.std(all_delta_theta_0)]
            dtn_stats = [np.min(all_delta_theta_nulls), np.max(all_delta_theta_nulls), np.mean(all_delta_theta_nulls), np.std(all_delta_theta_nulls)]
            sinr_stats = [np.min(all_sinr), np.max(all_sinr), np.mean(all_sinr), np.std(all_sinr)]
            sll_stats = [np.min(all_sll), np.max(all_sll), np.mean(all_sll), np.std(all_sll)]
            
            print(f"--- Delta = {delta}° ---")
            print(f"Δθ0 [deg] : Min: {dt0_stats[0]:.3f} | Max: {dt0_stats[1]:.3f} | Mean: {dt0_stats[2]:.3f} | Std: {dt0_stats[3]:.3f}")
            print(f"Δθn [deg] : Min: {dtn_stats[0]:.3f} | Max: {dtn_stats[1]:.3f} | Mean: {dtn_stats[2]:.3f} | Std: {dtn_stats[3]:.3f}")
            print(f"SINR [dB] : Min: {sinr_stats[0]:.3f} | Max: {sinr_stats[1]:.3f} | Mean: {sinr_stats[2]:.3f} | Std: {sinr_stats[3]:.3f}")
            print(f"SLL  [dB] : Min: {sll_stats[0]:.3f} | Max: {sll_stats[1]:.3f} | Mean: {sll_stats[2]:.3f} | Std: {sll_stats[3]:.3f}\n")
            
    print(f"Simulation Complete! Total execution time: {(time.time() - start_time)/60:.1f} minutes.")

if __name__ == "__main__":
    run_master_simulation()