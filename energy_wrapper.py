import ctypes
import numpy as np

# Load the compiled shared C library
lib = ctypes.CDLL('./energy.so')  # Update path if needed

# Function prototypes
lib.bfgs.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x (positions)
    ctypes.c_int,                     # n (total number of elements)
    ctypes.c_int,                     # max_iters
    ctypes.c_double,                   # tol
    ctypes.c_double,                   # h (step size)
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # epsilon, sigma, b, k_b
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # trajectory (array pointer)
    ctypes.POINTER(ctypes.c_int)  # num_iters_out
]
lib.bfgs.restype = None  # Function returns results via output parameters

def bfgs_optimize(x, n, max_iters, tol, h, epsilon, sigma, b, k_b):
    """
    Wrapper for the C BFGS optimizer.
    """
    x_array = np.array(x, dtype=np.float64)
    x_ptr = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Output: Trajectory storage
    trajectory_ptr = ctypes.POINTER(ctypes.c_double)()
    num_iters = ctypes.c_int()

    # Call C BFGS function
    lib.bfgs(x_ptr, n, max_iters, tol, h, epsilon, sigma, b, k_b, ctypes.byref(trajectory_ptr), ctypes.byref(num_iters))

    # Retrieve trajectory
    num_steps = num_iters.value
    trajectory = np.ctypeslib.as_array(trajectory_ptr, shape=(num_steps, n))

    return x_array, trajectory