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

import time
import numpy as np
import os
import csv

# CRITICAL: Set matplotlib to 'Agg' (headless mode) BEFORE importing visualization
import matplotlib
matplotlib.use('Agg')

from array_math import compute_pattern
from metrics import calculate_sinr, extract_deviations_and_sll
from optimization import optimize_dummy_interferers
from visualization import save_pattern

# --- VERSION & CONFIGURATION ---
__version__ = "1.0.0"

# Set this to False to bypass generating thousands of images
SAVE_PLOTS = False 
# ---------------------

def load_config(filename="config.csv"):
    """Reads the master simulation parameters from a CSV file."""
    config = {}
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                val = row[1].strip()
                config[key] = val
                
    snr_list = [float(x) for x in config['snr_values'].replace('"', '').split(',')]
    delta_list = [int(x) for x in config['delta_values'].replace('"', '').split(',')]
    
    # Fallback to 'nsb' if the user forgets to add it to the CSV
    beamformer = config.get('beamformer_type', 'nsb').strip().lower()
    
    return {
        'snr_values': snr_list,
        'delta_values': delta_list,
        'base_start': int(config['base_start']),
        'max_angle': int(config['max_angle']),
        'target_sll': float(config['target_sll']),
        'max_dummies': int(config['max_dummies']),
        'beamformer_type': beamformer
    }

def generate_scenarios(delta, base_start, max_angle):
    """Generates scenario tuples based on dynamically provided boundaries."""
    scenarios = []
    current_start = base_start
    
    while current_start + 5 * delta <= max_angle:
        base_angles = [current_start + i * delta for i in range(6)]
        for i in range(6):
            theta_0 = base_angles[i]
            interferers = base_angles[:i] + base_angles[i+1:]
            scenarios.append((theta_0, interferers))
        current_start += 1
        
    return scenarios

def run_master_simulation():
    print("Loading parameters from config.csv...")
    try:
        cfg = load_config("config.csv")
    except FileNotFoundError:
        print("Error: config.csv not found! Please create it in the same directory.")
        return
        
    snr_values = cfg['snr_values']
    delta_values = cfg['delta_values']
    bf_type = cfg['beamformer_type'].upper()
    
    print(f"Beamforming Algorithm Calculator for Antenna Array -- (v{__version__})\n")
    print(f"Beamforming Algorithm Used: {bf_type} --> Simulation Results\n")
    if SAVE_PLOTS:
        print("WARNING: Plot saving is ENABLED. This will generate thousands of images.")
        print("Execution will take significantly longer due to disk I/O.\n")
        
    start_time = time.time()
    
    with open("simulation_results.txt", "w", encoding="utf-8") as file:
        file.write(f"Beamforming Algorithm Calculator for Antenna Array -- (v{__version__})\n")
        file.write(f"Beamforming Algorithm Used: {bf_type} --> Simulation Results\n")
        file.write("================================================\n\n")
        
        for snr in snr_values:
            header = f"================ SNR = {snr} dB ================"
            print(header)
            file.write(header + "\n")
            
            for delta in delta_values:
                scenarios = generate_scenarios(delta, cfg['base_start'], cfg['max_angle'])
                
                all_delta_theta_0 = []
                all_delta_theta_nulls = []
                all_sinr = []
                all_sll = []
                
                for idx, (theta_0, interferers) in enumerate(scenarios):
                    # Pass the beamformer type and SNR to the optimization loop
                    w_opt, active_dummies = optimize_dummy_interferers(
                        theta_0, interferers, 
                        beamformer_type=cfg['beamformer_type'],
                        snr_db=snr,
                        max_dummies=cfg['max_dummies'], 
                        target_sll=cfg['target_sll']
                    )
                    
                    all_int = interferers + active_dummies
                    angles, P_dB = compute_pattern(w_opt)
                    
                    dt0, dtnulls, sll = extract_deviations_and_sll(angles, P_dB, theta_0, all_int)
                    sinr_val = calculate_sinr(w_opt, theta_0, interferers, snr)
                    
                    all_delta_theta_0.append(dt0)
                    all_delta_theta_nulls.extend(dtnulls) 
                    all_sinr.append(sinr_val)
                    all_sll.append(sll)
                    
                    if SAVE_PLOTS:
                        folder_path = os.path.join("plots", bf_type, f"SNR_{snr}", f"Delta_{delta}")
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