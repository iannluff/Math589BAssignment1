import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./energy.so')  # Use 'energy.dll' on Windows

# Define argument and return types for the BFGS function
lib.bfgs.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions (x)
    ctypes.c_int,                     # n (total dimension = n_beads * 3)
    ctypes.c_int,                     # max_iters
    ctypes.c_double,                   # tol
    ctypes.c_double,                   # k_b
    ctypes.c_double,                   # b
    ctypes.c_double,                   # epsilon
    ctypes.c_double,                   # sigma
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # trajectory (output)
    ctypes.POINTER(ctypes.c_int)       # num_iters_out
]
lib.bfgs.restype = None  # Void function

# Define argument and return types for total_energy
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.POINTER(ctypes.c_double),  # gradient
    ctypes.c_int,                     # n_beads
    ctypes.c_int,                     # d
    ctypes.c_double,                  # k_b
    ctypes.c_double,                  # b
    ctypes.c_double,                  # epsilon
    ctypes.c_double                   # sigma
]
lib.total_energy.restype = ctypes.c_double  # Returns energy value

def compute_total_energy(positions, n_beads, d=3, k_b=100.0, b=1.0, epsilon=1.0, sigma=1.0):
    positions_array = np.array(positions, dtype=np.float64)
    n_elements = n_beads * d

    # Allocate gradient array in Python and initialize to zero
    gradient_array = np.zeros(n_elements, dtype=np.float64)

    # Convert arrays to C pointers
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    gradient_ptr = gradient_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call the C function
    energy = lib.total_energy(positions_ptr, gradient_ptr, n_beads, d, k_b, b, epsilon, sigma)

    return energy

def run_bfgs(positions, n_beads, max_iters, tol, k_b = 100.0, b = 1.0, epsilon = 1.0, sigma = 1.0):
    n = n_beads * 3  # Each bead has 3 coordinates

    # Convert positions to a contiguous NumPy array
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Prepare pointers for outputs
    trajectory_ptr = ctypes.POINTER(ctypes.c_double)()
    num_iters = ctypes.c_int()

    # Call the BFGS function from the C library
    lib.bfgs(
        positions_ptr, n, max_iters, tol, k_b, b, epsilon, sigma,
        ctypes.byref(trajectory_ptr), ctypes.byref(num_iters)
    )

    # Convert results to NumPy arrays
    num_iters = num_iters.value
    optimized_positions = np.ctypeslib.as_array(positions_ptr, shape=(n,))  # Final positions
    trajectory = np.ctypeslib.as_array(trajectory_ptr, shape=(num_iters, n))  # Optimization path

    return optimized_positions, trajectory