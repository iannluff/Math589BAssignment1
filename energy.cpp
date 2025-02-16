#include <cmath>
#include <vector>
#include <cstring>
#include "energy.hpp"

double lennard_jones_potential(double r, double epsilon, double sigma) {
  if (r < 1e-12) return 1e12; // Avoid division by zero or extremely small r
  double sr6 = pow(sigma / r, 6);
  return 4 * epsilon * (sr6 * sr6 - sr6);
}

double bond_potential(double r, double b, double k_b) {
  return k_b * pow(r - b, 2);
}

/*double total_energy(double* positions, double* gradient, int n_beads, double epsilon, double sigma, double b, double k_b) {
  double energy = 0.0;

  // Bond potential
  for (int i = 0; i < n_beads - 1; ++i) {
    int idx1 = i * 3;
    int idx2 = (i + 1) * 3;
    double dx = positions[idx2] - positions[idx1];
    double dy = positions[idx2 + 1] - positions[idx1 + 1];
    double dz = positions[idx2 + 2] - positions[idx1 + 2];
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    energy += bond_potential(r, b, k_b);
  }

  // Lennard-Jones potential
  for (int i = 0; i < n_beads; ++i) {
    for (int j = i + 1; j < n_beads; ++j) {
      int idx1 = i * 3;
      int idx2 = j * 3;
      double dx = positions[idx2] - positions[idx1];
      double dy = positions[idx2 + 1] - positions[idx1 + 1];
      double dz = positions[idx2 + 2] - positions[idx1 + 2];
      double r = sqrt(dx * dx + dy * dy + dz * dz);
      energy += lennard_jones_potential(r, epsilon, sigma);
    }
  }
  
  if(gradient){
    std::memset(gradient, 0, n_beads * 3 * sizeof(double)); // Initialize gradient to zero

    for (int i = 0; i < n_beads * 3; ++i) {
        double original_pos = positions[i];

        // Forward step
        positions[i] = original_pos + 1e-6;
        double E_plus = total_energy(positions, NULL, n_beads, epsilon, sigma, b, k_b);

        // Backward step
        positions[i] = original_pos - 1e-6;
        double E_minus = total_energy(positions, NULL, n_beads, epsilon, sigma, b, k_b);

        // Compute central difference gradient
        gradient[i] = (E_plus - E_minus) / (2 * 1e-6);

        // Restore original position
        positions[i] = original_pos;
    }
  }
  return energy;
}*/

double total_energy(double* positions, double* gradient, int n_beads, double epsilon, double sigma, double b, double k_b) {
    double energy = 0.0;
    std::memset(gradient, 0, n_beads * 3 * sizeof(double)); // Zero out gradients

    // Bond potential and its gradient
    for (int i = 0; i < n_beads - 1; ++i) {
        int idx1 = i * 3;
        int idx2 = (i + 1) * 3;

        double dx = positions[idx2] - positions[idx1];
        double dy = positions[idx2 + 1] - positions[idx1 + 1];
        double dz = positions[idx2 + 2] - positions[idx1 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);

        double bond_force = 2 * k_b * (r - b) / r;

        energy += k_b * pow(r - b, 2);

        // Compute force contribution to gradient
        gradient[idx1] -= bond_force * dx;
        gradient[idx1 + 1] -= bond_force * dy;
        gradient[idx1 + 2] -= bond_force * dz;

        gradient[idx2] += bond_force * dx;
        gradient[idx2 + 1] += bond_force * dy;
        gradient[idx2 + 2] += bond_force * dz;
    }

    // Lennard-Jones potential and its gradient
    for (int i = 0; i < n_beads; ++i) {
        for (int j = i + 1; j < n_beads; ++j) {
            int idx1 = i * 3;
            int idx2 = j * 3;

            double dx = positions[idx2] - positions[idx1];
            double dy = positions[idx2 + 1] - positions[idx1 + 1];
            double dz = positions[idx2 + 2] - positions[idx1 + 2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r < 1e-2) continue; // Avoid singularity

            double sr6 = pow(sigma / r, 6);
            double sr12 = sr6 * sr6;
            double lj_force = 24 * epsilon * (2 * sr12 - sr6) / (r * r);

            energy += 4 * epsilon * (sr12 - sr6);

            // Compute force contribution to gradient
            gradient[idx1] -= lj_force * dx;
            gradient[idx1 + 1] -= lj_force * dy;
            gradient[idx1 + 2] -= lj_force * dz;

            gradient[idx2] += lj_force * dx;
            gradient[idx2 + 1] += lj_force * dy;
            gradient[idx2 + 2] += lj_force * dz;
        }
    }
    return energy;
}
