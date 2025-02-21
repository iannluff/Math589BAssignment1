#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Function to compute dot product of two vectors
double dot_product(const double *a, const double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Function to compute outer product of two vectors (matrix)
void outer_product(const double *a, const double *b, double *result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = a[i] * b[j];
        }
    }
}

// Function to perform matrix-vector multiplication
void matrix_vector_mult(const double *matrix, const double *vector, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

void vector_matrix_mult(const double *vector, const double *matrix, double *result, int n) {
    for (int j = 0; j < n; j++) {  // Loop over columns
        result[j] = 0.0;
        for (int i = 0; i < n; i++) {  // Loop over rows
            result[j] += vector[i] * matrix[i * n + j]; // Multiply vector with column
        }
    }
}

// Function to compute Euclidean distance
double compute_distance(const double *a, const double *b, int d) {
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

// Function to compute Lennard-Jones potential
double lennard_jones_potential(double r, double epsilon, double sigma) {
    return 4 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
}

// Function to compute the bond potential
double bond_potential(double r, double b, double k_b) {
    return k_b * pow(r - b, 2);
}

// Function to compute total energy and its gradient
double total_energy(const double *positions, double *grad, int n_beads, int d, double k_b, double b, double epsilon, double sigma) {
    double energy = 0.0;

    for (int i = 0; i < n_beads * d; i++) {
        grad[i] = 0.0;
    }

    for (int i = 0; i < n_beads - 1; i++) {
        double r = compute_distance(&positions[i * d], &positions[(i + 1) * d], d);
        double bond_energy = bond_potential(r, b, k_b);
        energy += bond_energy;

        double factor = 2 * k_b * (r - b) / r;
        for (int k = 0; k < d; k++) {
            double diff = positions[i * d + k] - positions[(i + 1) * d + k];
            grad[i * d + k] += factor * diff;
            grad[(i + 1) * d + k] -= factor * diff;
        }
    }

    for (int i = 0; i < n_beads; i++) {
        for (int j = i + 1; j < n_beads; j++) {
            double r = compute_distance(&positions[i * d], &positions[j * d], d);
            if (r > 1e-10){
                double lj_energy = lennard_jones_potential(r, epsilon, sigma);
                energy += lj_energy;
            }
            double r6 = pow(sigma / r, 6);
            double r12 = r6 * r6;
            double factor = 4 * epsilon * (-12 * r12 / (r * r) + 6 * r6 / (r * r));
            for (int k = 0; k < d; k++) {
                double diff = positions[i * d + k] - positions[j * d + k];
                grad[i * d + k] += factor * diff;
                grad[j * d + k] -= factor * diff;
            }
        }
    }

    return energy;
}

// Function to update inverse Hessian matrix using your specified formula
void update_inverse_hessian(double *H, const double *s, const double *y, int n) {
    double ys = dot_product(s, y, n);
    if (ys <= 1e-10) {
        for (int i = 0; i < n * n; i++) {
            H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
        }
        return;  // Avoid division by zero
    }

    double rho = 1.0 / ys;
    double *Hy = malloc(n * sizeof(double));
    double *yH = malloc(n * sizeof(double));
    double *ssT = malloc(n * n * sizeof(double));
    double *HysT = malloc(n * n * sizeof(double));
    double *syT_H = malloc(n * n * sizeof(double));

    matrix_vector_mult(H, y, Hy, n);
    vector_matrix_mult(y, H, yH, n);
    double yHy_scalar = dot_product(y, Hy, n);

    outer_product(s, s, ssT, n);
    outer_product(Hy, s, HysT, n);
    outer_product(s, yH, syT_H, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i * n + j] += ((ys + yHy_scalar) / (ys * ys)) * ssT[i * n + j] 
                            - (HysT[i * n + j] + syT_H[i * n + j]) / ys;
        }
    }

    free(Hy);
    free(yH);
    free(ssT);
    free(HysT);
    free(syT_H);
}

// BFGS Optimization Function
void bfgs(double *x, int n, int max_iters, double tol, double k_b, double b, double epsilon, double sigma, double **trajectory_out, int *num_iters_out) {
    double *grad = malloc(n * sizeof(double));
    double *p = malloc(n * sizeof(double));
    double *H = malloc(n * n * sizeof(double));
    double *s = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *trajectory = malloc(max_iters * n * sizeof(double));

    double energy;

    for (int i = 0; i < n * n; i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    for (int iter = 0; iter < max_iters; iter++) {
        energy = total_energy(x, grad, n / 3, 3, k_b, b, epsilon, sigma);
        double grad_norm = sqrt(dot_product(grad, grad, n));
        if (grad_norm < tol) {
            printf("Converged after %d iterations\n", iter);
            break;
        }

        matrix_vector_mult(H, grad, p, n);
        for (int i = 0; i < n; i++) {
            p[i] = -p[i];
        }

        double grad_p = dot_product(grad, p, n);
        double alpha = 1.0;
        double *x_new = malloc(n * sizeof(double));
        double *grad_new = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            x_new[i] = x[i] + p[i];
        }
        while (total_energy(x_new, grad_new, n / 3, 3, k_b, b, epsilon, sigma) > energy + 1e-4 * alpha * grad_p) {
            alpha *= 0.9;
            for (int i = 0; i < n; i++) {
                x_new[i] = x[i] + alpha * p[i];
            }
        }

        for (int i = 0; i < n; i++) {
            s[i] = alpha * p[i];
            x[i] += s[i];
        }

        for (int i = 0; i < n; i++) {
            y[i] = grad_new[i] - grad[i];
        }

        update_inverse_hessian(H, s, y, n);

        for (int i = 0; i < n; i++) {
            grad[i] = grad_new[i];
        }

        for (int i = 0; i < n; i++) {
            trajectory[iter * n + i] = x[i];
        }

        free(grad_new);
        free(x_new);
    }

    *trajectory_out = trajectory;
    *num_iters_out = max_iters;

    free(grad);
    free(p);
    free(H);
    free(s);
    free(y);
}