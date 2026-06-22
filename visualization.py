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

import os
import matplotlib.pyplot as plt

def plot_pattern(theta_scan, P_dB, theta_desired, theta_interferers):
    """Plots the radiation pattern and overlays the target angles."""
    plt.figure(figsize=(10, 6))
    plt.plot(theta_scan, P_dB, 'b-', linewidth=1.5, label='Radiation Pattern')
    
    plt.axvline(x=theta_desired, color='g', linestyle='--', label='Desired Signal')
    
    for i, th in enumerate(theta_interferers):
        label = 'Interferer' if i == 0 else "" 
        plt.axvline(x=th, color='r', linestyle=':', label=label)
        
    plt.axhline(y=-20, color='k', linestyle='--', alpha=0.5, label='-20 dB SLL Target')
    
    plt.ylim([-80, 5])
    plt.xlim([0, 180])
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('NSB Radiation Pattern')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_pattern(theta_scan, P_dB, theta_desired, theta_interferers, filename):
    """
    Saves the radiation pattern to a file without displaying it.
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(theta_scan, P_dB, 'b-', linewidth=1.5, label='Radiation Pattern')
    
    plt.axvline(x=theta_desired, color='g', linestyle='--', label='Desired Signal')
    
    for i, th in enumerate(theta_interferers):
        label = 'Interferer' if i == 0 else "" 
        plt.axvline(x=th, color='r', linestyle=':', label=label)
        
    plt.axhline(y=-20, color='k', linestyle='--', alpha=0.5, label='-20 dB SLL Target')
    
    plt.ylim([-80, 5])
    plt.xlim([0, 180])
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title(f'NSB Pattern | Desired: {theta_desired}°')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the file at a crisp resolution
    plt.savefig(filename, dpi=150)
    
    # CRITICAL: Close the figure to prevent catastrophic memory leaks
    plt.close('all')