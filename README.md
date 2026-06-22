# Advanced RF Aspects of UAVs - Beamforming Simulation

This repository contains a Python-based simulation of a **Null Steering Beamformer (NSB)**, developed for the *Advanced RF Aspects of UAVs* MSc course. 

The software simulates a 24-element uniform linear antenna array. It calculates complex weights to steer the main lobe toward a desired signal while forcing mathematical nulls at the angles of incoming interference signals. To meet strict Side Lobe Level (SLL) constraints ($\leq -20$ dB), it utilizes an iterative peak-nulling algorithm to strategically place artificial "dummy" interferers.

## File Architecture

To maintain a clean codebase, it is modularized into five distinct Python files:

* **`array_math.py`**: *The core RF physics engine.* Handles the generation of steering vectors, construction of the constraint matrix, calculation of the complex weight vector via pseudo-inversion, and computation of the complete 180° radiation pattern.

* **`metrics.py`**: *The validation and extraction layer.* Uses `scipy` to systematically find local peaks and nulls in the radiation pattern to calculate exact directional deviations ($\Delta\theta$) and the maximum Side Lobe Level (SLL). Also computes the Signal-to-Interference-and-Noise Ratio (SINR).

* **`optimization.py`**: *The SLL control algorithm.* Implements an iterative, greedy approach that scans the array pattern for the highest side lobe and dynamically drops "dummy" interferers at those exact angles until the overall SLL floor falls below the -20 dB target.

* **`visualization.py`**: *The headless plotting logic.* Configured to bypass GUI rendering. Generates high-resolution 2D plots of the normalized radiation pattern (in dB) and safely saves them directly to disk to prevent memory leaks during massive batch loops.

* **`main.py`**: *The master execution loop.* The entry point for the simulation. It generates the required matrix of test scenarios (sweeping through various SNRs and angular steps $\delta$), runs the optimization algorithm across all permutations, manages the `plots/` directory structure, and outputs the final statistical metrics to both the console and a text file.

## Requirements

To run this simulation, you will need Python installed along with the following scientific libraries:
```bash
pip install numpy scipy matplotlib
```

## How to Run the Simulation

```bash
python main.py
```