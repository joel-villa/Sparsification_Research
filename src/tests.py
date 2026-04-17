"""
Some functions which have return type: 

xs - x-axis
ys - y-axis
"""

from .MDSparsifier import MDSparsifier
from .Sparsifier import Sparsifier
from .SGenerator import SGenerator
from .util import *
import numpy as np

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





