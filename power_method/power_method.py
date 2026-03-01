"""
The Power Iteration Method to determine the largest eigenvalue and eigenvector 
in implemented from the following source code:

https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/ 
"""

import numpy as np

def pow_method(A, tol=1e-6, max_iter=100):
    """
    A        - matrix 
    tol      - The tolerance for the eigenvalue and eigenvector approximations
               (i.e. the maximum allowed difference between
               the approximations and the actual values)
    max_iter - The maximum number of iterations
    """
    # Choose the initial vector x
    x = np.array([[1, 2]]).T

    # Define the variable lam_prev to store the
    # previous approximation for the largest eigenvalue
    lam_prev = 0

    # Iteratively improve the approximations
    # for the largest eigenvalue and eigenvector
    # using the power method
    for _ in range(max_iter):
        # Compute the updated approximation for the eigenvector
        x = A @ x / np.linalg.norm(A @ x)

        # Compute the updated approximation for the largest eigenvalue
        lam = (x.T @ A @ x) / (x.T @ x)

        # Check if the approximations have converged
        if np.abs(lam - lam_prev) < tol:
            break

        # Store the current approximation for the largest eigenvalue
        lam_prev = lam

    # Return the approximations for the
    # largest eigenvalue and eigenvector
    return (lam, x)