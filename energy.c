#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Function to compute the Euclidean distance between two beads
double compute_distance(double *a, double *b, int d) {
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

// Function to compute the total energy of the system
double total_energy(const double *positions, int n_beads, double epsilon, double sigma, double b, double k_b) {
    double energy = 0.0;
    int d = 3;
    // Bond energy
    for (int i = 0; i < n_beads - 1; i++) {
        double r = compute_distance(&positions[i * d], &positions[(i + 1) * d], d);
        energy += bond_potential(r, b, k_b);
    }

    // Lennard-Jones potential for non-bonded interactions
    for (int i = 0; i < n_beads; i++) {
        for (int j = i + 1; j < n_beads; j++) {
            double r = compute_distance(&positions[i * d], &positions[j * d], d);
            if (r > 1e-2) {  // Avoid division by zero
                energy += lennard_jones_potential(r, epsilon, sigma);
            }
        }
    }

    return energy;
}

/*void find_grad(const double *positions, int n_beads, double *grad, double h, double epsilon, double sigma, double b, double k_b) {
    int d = 3;  // Assuming 3-dimensional positions (change if necessary)
    double *X = malloc(n_beads * d * sizeof(double));
    double *E_ij = malloc(n_beads * d * sizeof(double));

    // Reshape positions into a 2D array
    for (int i = 0; i < n_beads; i++) {
        for (int j = 0; j < d; j++) {
            X[i * d + j] = positions[i * d + j];
        }
    }

    // Initialize gradient array
    for (int i = 0; i < n_beads * d; i++) {
        grad[i] = 0.0;
    }

    // Compute gradient using central difference
    for (int i = 0; i < n_beads; i++) {
        for (int j = 0; j < d; j++) {
            // Perturb only the (i, j) entry
            for (int k = 0; k < n_beads * d; k++) {
                E_ij[k] = 0.0;
            }
            E_ij[i * d + j] = h;

            // Compute function values at f_plus, f_minus, f_plus_two, and f_minus_two
            double *positions_plus = malloc(n_beads * d * sizeof(double));
            double *positions_minus = malloc(n_beads * d * sizeof(double));
            double *positions_plus_two = malloc(n_beads * d * sizeof(double));
            double *positions_minus_two = malloc(n_beads * d * sizeof(double));

            for (int k = 0; k < n_beads * d; k++) {
                positions_plus[k] = positions[k] + E_ij[k];
                positions_minus[k] = positions[k] - E_ij[k];
                positions_plus_two[k] = positions[k] + 2 * E_ij[k];
                positions_minus_two[k] = positions[k] - 2 * E_ij[k];
            }

            // Compute the total energy at the perturbed positions
            
            double f_plus = total_energy(positions_plus, n_beads, epsilon, sigma, b, k_b);
            double f_minus = total_energy(positions_minus, n_beads, epsilon, sigma, b, k_b);
            double f_plus_two = total_energy(positions_plus_two, n_beads, epsilon, sigma, b, k_b);
            double f_minus_two = total_energy(positions_minus_two, n_beads, epsilon, sigma, b, k_b);

            // Central difference formula for gradient
            grad[i * d + j] = (-f_plus_two + 8 * f_plus - 8 * f_minus + f_minus_two) / (12 * h);

            // Free allocated memory for each perturbed position
            free(positions_plus);
            free(positions_minus);
            free(positions_plus_two);
            free(positions_minus_two);
        }
    }

    free(X);
    free(E_ij);
}*/

void find_grad(const double *positions, int n_beads, double *grad, double h, double epsilon, double sigma, double b, double k_b){
    double *x = (double *)malloc(n_beads * 3 * sizeof(double));

    // Copy positions into x
    for (int i = 0; i < n_beads * 3; i++) {
        x[i] = positions[i];
    }

    for (int i = 0; i < n_beads * 3; i++) {
        //x[i] += 2 * h;
        //double f_plus_two = compute_total_energy(x, n_beads * 3);
        //x[i] -= h;
        x[i] += h;
        double f_plus = compute_total_energy(x, n_beads * 3);
        x[i] -= 2 * h;
        double f_minus = compute_total_energy(x, n_beads * 3);
        //x[i] -= h;
        //double f_minus_two = compute_total_energy(x, n_beads * 3);
        //x[i] += 2 * h;
        x[i] += h;

        //grad[i] = (-1 * f_plus_two + 8 * f_plus - 8 * f_minus + f_minus_two) / (12 * h);
        grad[i] = (f_plus - f_minus) / (2*h);
    }

    free(x);
}

void outer_product(const double *a, const double *b, double *result, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i * n + j] = a[i] * b[j];
    }
  }
}

double dot_product(const double *vector_1, const double *vector_2, int n)
{
    double dot_prod = 0.0;
    for (int i = 0; i < n; i++){
        dot_prod += vector_1[i] * vector_2[i];
    }
    return dot_prod;
}

void matrix_vector_mult(const double *matrix, const double *vector, double *result, int n)
{
  for (int i = 0; i < n; i++) {
    result[i] = 0.0;
    for (int j = 0; j < n; j++) {
      result[i] += matrix[i * n + j] * vector[j];
    }
  }
}

void matrix_update(double *H, const double *s, const double *y, int n) {  
  double ys = dot_product(y, s, n*3);
  if (ys <= 1e-10) {
    for (int i = 0; i < (n *3) * (n * 3); i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }
    return; // Avoid division by zero or negative curvature
  }

  double rho = 1.0 / ys;
  double *Hy = malloc(n * 3 * sizeof(double));
  double *ssT = malloc((n * 3) * (n * 3) * sizeof(double));
  double *HysT = malloc((n * 3) * (n * 3) * sizeof(double));
  double *syT_H = malloc((n * 3) * (n * 3) * sizeof(double));

  // Compute Hy = H * y
  matrix_vector_mult(H, y, Hy, n*3);
  double yHy_scalar = dot_product(y, Hy, n*3);

  // Compute outer products
  outer_product(s, s, ssT, n*3);
  outer_product(Hy, s, HysT, n*3);
  outer_product(s, Hy, syT_H, n*3);

  // BFGS Update: H = H + ((ys + yHy) / ys^2) * ssT - (HysT + syT_H) / ys
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      H[i * n + j] += ((ys + yHy_scalar) / (ys * ys)) * ssT[i * n + j] 
                      - (HysT[i * n + j] + syT_H[i * n + j]) / ys;
    }
  }

  // Free allocated memory
  free(Hy);
  free(ssT);
  free(HysT);
  free(syT_H);
}

/*void matrix_update(double *H, const double *s, const double *y, int n) {
  double sy = 0.0;
  sy = dot_product(s, y, n);
  if (sy <= 0.0) {
    return; // Avoid division by zero or negative curvature
  }

  double rho = 1.0 / sy;
  double *Hy = malloc(n * sizeof(double));
  double *yH = malloc(n * sizeof(double));

  matrix_vector_mult(H, y, Hy, n);
  matrix_vector_mult(y, H, yH, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double correction = rho * ((1+rho)*s[i] * s[j] - Hy[i] * s[j] + s[i]*yH[j]);
      H[i * n + j] += correction;
    }
  }

  free(Hy);
}*/

int armijo_condition(const double *x, const double *grad, const double *p, double alpha, int n,
                     double c1, double epsilon, double sigma, double b, double k_b)
{
  double *x_new = malloc(n * 3 * sizeof(double));

  for (int i = 0; i < n * 3; i++) {
    x_new[i] = x[i] + alpha * p[i];
  }

  double f_x = total_energy(x, n, epsilon, sigma, b, k_b);

  double f_x_new = total_energy(x_new, n, epsilon, sigma, b, k_b);

  double dot_prod = 0.0;
  dot_prod = dot_product(grad, p, n*3);

  free(x_new);

  return f_x_new <= f_x + c1 * alpha * dot_prod;
}

void bfgs(double *x, int n, int max_iters, double tol, double c1, double h, double epsilon, double sigma, double b, double k_b, double **trajectory_out, int *num_iters_out)
{
  double *grad = malloc(n * 3 * sizeof(double));
  double *p = malloc(n * 3 * sizeof(double));
  double *H = malloc((n * 3) * (n * 3) * sizeof(double));
  double *s = malloc(n * 3 * sizeof(double));
  double *y = malloc(n * 3 * sizeof(double));

  double *trajectory = malloc(max_iters * n * 3 * sizeof(double));

  // Initialize H to identity
  for (int i = 0; i < (n * 3) * (n * 3); i++) {
    H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
  }

  find_grad(x, n, grad, h, epsilon, sigma, b, k_b);

  int iter;
  for (iter = 0; iter < max_iters; iter++) {
    double grad_norm = 0.0;
    for (int i = 0; i < n*3; i++) {
      grad_norm += grad[i] * grad[i];
    }
    grad_norm = sqrt(grad_norm);
    if (grad_norm < tol) {
      printf("Converged after %d iterations\n", iter);
      break;
    }

    // Compute p = -H * grad
    matrix_vector_mult(H, grad, p, n*3);
    for (int i = 0; i < n; i++) {
      p[i] = -p[i];
    }

    // Armijo line search
    double alpha = 1.0;

    while (1) {
      int status = armijo_condition(x, grad, p, alpha, n, c1, epsilon, sigma, b, k_b);
      if(status) {
	    break;
      }
      alpha *= 0.9;
    }

    // Update x, s, and y
    for (int i = 0; i < n * 3; i++) {
      s[i] = alpha * p[i];
      x[i] += s[i];
    }

    double *grad_new = malloc(n * 3 * sizeof(double));

    find_grad(x, n, grad_new, h, epsilon, sigma, b, k_b);
    
    for (int i = 0; i < n * 3; i++) {
      y[i] = grad_new[i] - grad[i];
    }

    // Update H
    matrix_update(H, s, y, n);

    // Update grad
    for (int i = 0; i < n * 3; i++) {
      grad[i] = grad_new[i];
    }

    for (int i = 0; i < n * 3; i++) {
      trajectory[iter * n * 3 + i] = x[i];  
    }

    free(grad_new);
  }

  *trajectory_out = trajectory;
  *num_iters_out = iter;

  free(grad);
  free(p);
  free(H);
  free(s);
  free(y);
}
