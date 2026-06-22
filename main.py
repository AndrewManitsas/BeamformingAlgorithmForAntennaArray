import time
import numpy as np
import os

# CRITICAL: Set matplotlib to 'Agg' (headless mode) BEFORE importing visualization
# This prevents it from requiring a GUI or crashing during massive batch loops
import matplotlib
matplotlib.use('Agg')

from array_math import compute_pattern
from metrics import calculate_sinr, extract_deviations_and_sll
from optimization import optimize_dummy_interferers
from visualization import save_pattern

# --- CONFIGURATION ---
# Set this to False if you just want the math fast without generating thousands of images
SAVE_PLOTS = False
# ---------------------

def generate_scenarios(delta):
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
    snr_values = [-10, 0, 10, 20]
    delta_values = [6, 8, 10, 12, 14, 16]
    
    print("Starting Master Simulation...")
    if SAVE_PLOTS:
        print("WARNING: Plot saving is ENABLED. This will generate thousands of images.")
        print("Execution will take significantly longer due to disk I/O.\n")
        
    start_time = time.time()
    
    with open("simulation_results.txt", "w", encoding="utf-8") as file:
        file.write("================================================\n\n")
        
        for snr in snr_values:
            header = f"================ SNR = {snr} dB ================"
            print(header)
            file.write(header + "\n")
            
            for delta in delta_values:
                scenarios = generate_scenarios(delta)
                
                all_delta_theta_0 = []
                all_delta_theta_nulls = []
                all_sinr = []
                all_sll = []
                
                # Use enumerate to get a unique index (idx) for our filenames
                for idx, (theta_0, interferers) in enumerate(scenarios):
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
                    
                    # --- PLOT SAVING LOGIC ---
                    if SAVE_PLOTS:
                        # Create a clean folder hierarchy: plots/SNR_10/Delta_6/
                        folder_path = os.path.join("plots", f"SNR_{snr}", f"Delta_{delta}")
                        # Name file logically: e.g., scenario_0042_th0_90.png
                        filename = os.path.join(folder_path, f"scenario_{idx:04d}_th0_{theta_0}.png")
                        
                        save_pattern(angles, P_dB, theta_0, all_int, filename)
                
                # --- Statistics Calculation ---
                dt0_stats = [np.min(all_delta_theta_0), np.max(all_delta_theta_0), np.mean(all_delta_theta_0), np.std(all_delta_theta_0)]
                dtn_stats = [np.min(all_delta_theta_nulls), np.max(all_delta_theta_nulls), np.mean(all_delta_theta_nulls), np.std(all_delta_theta_nulls)]
                sinr_stats = [np.min(all_sinr), np.max(all_sinr), np.mean(all_sinr), np.std(all_sinr)]
                sll_stats = [np.min(all_sll), np.max(all_sll), np.mean(all_sll), np.std(all_sll)]
                
                output_block = (
                    f"--- Delta = {delta}° ---\n"
                    f"Δθ0 [deg] : Min: {dt0_stats[0]:.3f} | Max: {dt0_stats[1]:.3f} | Mean: {dt0_stats[2]:.3f} | Std: {dt0_stats[3]:.3f}\n"
                    f"Δθn [deg] : Min: {dtn_stats[0]:.3f} | Max: {dtn_stats[1]:.3f} | Mean: {dtn_stats[2]:.3f} | Std: {dtn_stats[3]:.3f}\n"
                    f"SINR [dB] : Min: {sinr_stats[0]:.3f} | Max: {sinr_stats[1]:.3f} | Mean: {sinr_stats[2]:.3f} | Std: {sinr_stats[3]:.3f}\n"
                    f"SLL  [dB] : Min: {sll_stats[0]:.3f} | Max: {sll_stats[1]:.3f} | Mean: {sll_stats[2]:.3f} | Std: {sll_stats[3]:.3f}\n"
                )
                
                print(output_block)
                file.write(output_block + "\n")
                
            file.write("\n")
                
    total_time_msg = f"Simulation Complete! Total execution time: {(time.time() - start_time)/60:.1f} minutes."
    print(total_time_msg)
    
    with open("simulation_results.txt", "a", encoding="utf-8") as file:
        file.write(total_time_msg + "\n")

if __name__ == "__main__":
    run_master_simulation()