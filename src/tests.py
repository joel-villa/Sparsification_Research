"""
Some testing code
"""

"""
Some functions which have return type: 

xs - x-axis
ys - y-axis
"""

from .MDSparsifier import MDSparsifier
from .Sparsifier import Sparsifier
from .SGenerator import SGenerator
from scipy.sparse.linalg import eigs
import numpy as np

def norm_of_diff(v1, v2):
    """
    Return the 2 norm of the difference of the two vectors
    """
    # Simpler approach: take the minimum of flipping one of th vectors
    v2_neg = -v2

    difference_pos = v1 - v2
    difference_neg = v1 - v2_neg

    # Test the difference of both the eigenvector and the eigenvector flipped
    norm_of_diff_pos = np.linalg.norm(difference_pos, ord=2)
    norm_of_diff_neg = np.linalg.norm(difference_neg, ord=2)

    # Take minimum
    return min(norm_of_diff_pos, norm_of_diff_neg)

def top_eig_difference(A, A_sparse, seed):
    '''
    Return the 2 norm of the difference of the top eigenvectors of the 
    matrix A and it's sparsified counterpart 

    A        - original matrix
    A_sparse - sparsified matrix
    seed     - for generating initial guess consistently

    Return: None if eigs does not converge, 2 norm of difference of top
            eigenvectors otw
    '''

    # Random number generator:
    r_gen = np.random.default_rng(seed=seed)

    n, _ = A.shape

    # Choose the initial vector x, 1 by n
    # Initial guess is close to zero 
    v0 = r_gen.normal(loc=0.0, scale=0.01, size=n) 

    try:
        _, e = eigs(A, k=1, v0=v0) #k = 1 -> only get top eigenvector
        _, e_sparse = eigs(A_sparse, k=1, v0=v0)
        
        return norm_of_diff(e, e_sparse)

    except Exception as e:
        # Print exception
        print(f"Error getting top eigenvector of matrix with dimension {A.shape}")
        return None

def test_eig_pres_of_md_sparsifier(A, expected_zeroes, seed):
    """
    A               - matrix to test eigenvector preservation of 
    expected_zeroes - expected number of new zeros from sparsificiation
    seed            - for reproducible results

    RETURN: xs - s-values
            ys - error in top eigenvector preservation
    """

    sparsifier = MDSparsifier(seed)
    s_generator = SGenerator(A.shape[0], A.nnz)

    ys = np.zeros(np.shape(expected_zeroes))
    xs = np.zeros(np.shape(expected_zeroes))

    for i, x in enumerate(expected_zeroes):
        A_sparse = A.copy()

        s = s_generator.get_min_s(x=x)

        sparsifier.sparsify(A_sparse, s)

        eig_diff = top_eig_difference(A, A_sparse, seed=seed)

        xs[i] = s
        ys[i] = eig_diff
    
    return xs, ys


def test_eig_pres_of_sparsifier(A, expected_zeroes, seed):
    """
    A               - matrix to test eigenvector preservation of 
    expected_zeroes - expected number of new zeros from sparsificiation
    seed            - for reproducible results

    RETURN: xs - s-values
            ys - error in top eigenvector preservation
    """

    sparsifier = Sparsifier(seed)
    s_generator = SGenerator(A.shape[0], A.nnz)

    ys = np.zeros(np.shape(expected_zeroes))
    xs = np.zeros(np.shape(expected_zeroes))

    for i, x in enumerate(expected_zeroes):
        A_sparse = A.copy()

        s = s_generator.get_min_s(x=x)

        sparsifier.sparsify(A_sparse, s)

        eig_diff = top_eig_difference(A, A_sparse, seed=seed)

        xs[i] = s
        ys[i] = eig_diff
    
    return xs, ys





