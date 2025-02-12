import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./energy.so')  # Use 'energy.dll' on Windows

# Define function argument and return types
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double), # positions
    ctypes.c_int,                    # n_beads
    ctypes.c_double,                 # epsilon
    ctypes.c_double,                 # sigma
    ctypes.c_double,                 # b
    ctypes.c_double                  # k_b
]
lib.total_energy.restype = ctypes.c_double

# Define function argument and return types
lib.find_grad.argtypes = [
    ctypes.POINTER(ctypes.c_double), # positions
    ctypes.c_int,                    # n_beads
    ctypes.POINTER(ctypes.c_double), # gradient
    ctypes.c_double,                 # step size
    ctypes.c_double,                 # epsilon
    ctypes.c_double,                 # sigma
    ctypes.c_double,                 # b
    ctypes.c_double                  # k_b
]
lib.find_grad.restype = None

lib.bfgs.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # Trajectory output
    ctypes.POINTER(ctypes.c_int)  # Number of iterations output
]
lib.bfgs.restype = None

def compute_total_energy(positions, h=1e-6, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    return lib.total_energy(positions_ptr, n_beads, epsilon, sigma, b, k_b)

def compute_grad(positions, h=1e-6, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    gradient = np.zeros(n_beads * 3, dtype=np.float64)
    gradient_ptr = gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.find_grad(positions_ptr, n_beads, gradient_ptr, h, epsilon, sigma, b, k_b)
    
    return gradient

def find_optimal_protein(positions, max_iters=1000, tol=1e-6, c=1e-4, h=1e-6, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    positions_ptr = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    trajectory_ptr = ctypes.POINTER(ctypes.c_double)()
    num_iters = ctypes.c_int()
    
    lib.bfgs(positions_ptr, n_beads, max_iters, tol, c, h, epsilon, sigma, b, k_b, ctypes.byref(trajectory_ptr), ctypes.byref(num_iters))
    
    trajectory = np.ctypeslib.as_array(trajectory_ptr, shape=(num_iters.value, n_beads)).copy()
    
    lib.free(trajectory_ptr)
    
    return positions_array, trajectory

if __name__ == "__main__":
    n_beads = 10
    positions = np.random.rand(n_beads * 3)  # Random 3D positions for each bead
    energy = compute_total_energy(positions)
    grad = compute_grad(positions)
    print(f"Total Energy: {energy}")
    print(f"Gradient: {grad}")