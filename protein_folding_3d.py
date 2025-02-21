import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from energy_wrapper import run_bfgs, compute_total_energy

def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)  # Fixed bond length of 1 unit
        positions[i, 2] = fudge * np.sin(i*i)  # Fixed bond length of 1 unit                
    return positions


def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    optimized_positions, trajectory = run_bfgs(positions.flatten(), n_beads, maxiter, tol)
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")

    return optimized_positions, trajectory