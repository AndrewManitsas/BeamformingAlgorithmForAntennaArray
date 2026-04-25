import numpy as np

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

# --- Quick Test ---
if __name__ == "__main__":
    # Let's test it for a desired signal at 90 degrees (broadside)
    v_90 = steering_vector(90)
    
    print("Shape of vector:", v_90.shape)
    print("First 3 elements of v(90°):\n", np.round(v_90, len(v_90)))