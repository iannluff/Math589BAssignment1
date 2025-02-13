import ctypes
import numpy as np
import subprocess

# Load the shared library
lib = ctypes.CDLL('./energy.so')  # Use 'energy.dll' on Windows

# Define function argument and return types
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_int,                    # n_beads
    ctypes.c_double,                 # epsilon
    ctypes.c_double,                 # sigma
    ctypes.c_double,                 # b
    ctypes.c_double                  # k_b
]
lib.total_energy.restype = ctypes.c_double

def compute_total_energy(positions, gradient=None, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Wrapper function to compute total energy using the C++ library.
    """
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    if(gradient):
        grad_array = np.zeros(n_beads*3, dtype=np.float64)
        grad_ptr = grad_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        f_val = lib.total_energy(positions_ptr, grad_ptr, n_beads, epsilon, sigma, b, k_b)
        return f_val, grad_array
    else:
        f_val = lib.total_energy(positions_ptr, None, n_beads, epsilon, sigma, b, k_b)
    
    return f_val, None

# Example usage
if __name__ == "__main__":
    n_beads = 10
    positions = np.random.rand(n_beads * 3)  # Random 3D positions for each bead
    energy, grad = compute_total_energy(positions, True)
    print(f"Total Energy: {energy}")
    print(f"Gradient: {grad}")
