#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to compute the Euclidean distance squared
double distance_squared(const double *a, const double *b, int d) {
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;  
}

// Lennard-Jones potential function
double lennard_jones_potential(double r2, double epsilon, double sigma) {
    if(r2 < 1e-12){
        return 1e12;
    }
    double sigma_r2 = (sigma * sigma) / r2;
    double sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;
    return 4.0 * epsilon * sigma_r6 * (sigma_r6 - 1.0);
}

// Harmonic bond potential function
double bond_potential(double r, double b, double k_b) {
    double diff = r - b;
    return k_b * diff * diff;
}

// Compute total energy
double total_energy(const double *positions, int n_beads, double epsilon, double sigma, double b, double k_b) {
    double energy = 0.0;
    int d = 3;

    // Compute bond energy
    for (int i = 0; i < n_beads - 1; i++) {
        double r = sqrt(distance_squared(&positions[i * d], &positions[(i + 1) * d], d));
        energy += bond_potential(r, b, k_b);
    }

    // Compute Lennard-Jones potential
    for (int i = 0; i < n_beads; i++) {
        for (int j = i + 1; j < n_beads; j++) {
            double r2 = distance_squared(&positions[i * d], &positions[j * d], d);
            energy += lennard_jones_potential(r2, epsilon, sigma);
        }
    }

    return energy;
}

// Compute gradient using Richardson central difference
void compute_gradient(const double *positions, int n_beads, double *grad, double h, double epsilon, double sigma, double b, double k_b) {
    int d = 3;
    double *x = (double *)malloc(n_beads * d * sizeof(double));

    // Copy positions
    for (int i = 0; i < n_beads * d; i++) {
        x[i] = positions[i];
    }

    // Compute gradient
    for (int i = 0; i < n_beads * d; i++) {
        x[i] += 2 * h;
        double f_plus_two = total_energy(x, n_beads, epsilon, sigma, b, k_b);
        x[i] -= h;
        double f_plus = total_energy(x, n_beads, epsilon, sigma, b, k_b);
        x[i] -= 2 * h;
        double f_minus = total_energy(x, n_beads, epsilon, sigma, b, k_b);
        x[i] -= h;
        double f_minus_two = total_energy(x, n_beads, epsilon, sigma, b, k_b);
        x[i] += 2 * h;

        grad[i] = (-f_plus_two + 8 * f_plus - 8 * f_minus + f_minus_two) / (12 * h);
    }

    free(x);
}

// **BFGS Optimization**
void bfgs(double *x, int n, int max_iters, double tol, double h, double epsilon, double sigma, double b, double k_b, double **trajectory_out, int *num_iters_out) {
    double *grad = malloc(n * sizeof(double));
    double *p = malloc(n * sizeof(double));
    double *H = malloc(n * n * sizeof(double));
    double *s = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));

    double *trajectory = malloc(max_iters * n * sizeof(double));

    // Initialize Hessian approximation to identity
    for (int i = 0; i < n * n; i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    compute_gradient(x, n / 3, grad, h, epsilon, sigma, b, k_b);

    for (int iter = 0; iter < max_iters; iter++) {

        printf("%d", iter);
        // Compute gradient norm
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) {
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tol) {
            printf("Converged after %d iterations\n", iter);
            break;
        }

        // Compute step direction p = -H * grad
        for (int i = 0; i < n; i++) {
            p[i] = 0;
            for (int j = 0; j < n; j++) {
                p[i] -= H[i * n + j] * grad[j];
            }
        }

        // Allocate temporary position array
        double *x_temp = (double *)malloc(n * sizeof(double));
        double alpha = 1.0;

        // Update x_temp using alpha * p
        for (int i = 0; i < n; i++) {
            x_temp[i] = x[i] + alpha * p[i];
        }

        // Use x_temp in total_energy function
        while (total_energy(x, n / 3, epsilon, sigma, b, k_b) >
            total_energy(x_temp, n / 3, epsilon, sigma, b, k_b) + 1e-4 * alpha * grad_norm) {
            alpha *= 0.9;

            // Update x_temp again
            for (int i = 0; i < n; i++) {
                x_temp[i] = x[i] + alpha * p[i];
            }
        }

        // Update positions
        for (int i = 0; i < n; i++) {
            s[i] = alpha * p[i];
            x[i] += s[i];
        }

        // Compute new gradient
        double *grad_new = malloc(n * sizeof(double));
        compute_gradient(x, n / 3, grad_new, h, epsilon, sigma, b, k_b);

        for (int i = 0; i < n; i++) {
            y[i] = grad_new[i] - grad[i];
        }

        // Hessian update
        double sy = 0.0;
        for (int i = 0; i < n; i++) {
            sy += s[i] * y[i];
        }
        if (sy > 1e-10) {
            double rho = 1.0 / sy;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    H[i * n + j] += rho * (s[i] * s[j] - H[i * n + j] * y[j]);
                }
            }
        }

        // Update gradient
        for (int i = 0; i < n; i++) {
            grad[i] = grad_new[i];
        }

        // Store trajectory
        for (int i = 0; i < n; i++) {
            trajectory[iter * n + i] = x[i];
        }

        free(grad_new);
        free(x_temp);
    }

    *trajectory_out = trajectory;
    *num_iters_out = max_iters;

    free(grad);
    free(p);
    free(H);
    free(s);
    free(y);
}