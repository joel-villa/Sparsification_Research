import random
from math import log2
from numpy import linspace

"""
The following sparsification algorithms are based on 
the Sparsification Algorithms Paper:
https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf
"""

def s_upper_bound(num_rows, num_cols):
    """
    Get the upper bound for the valid s values of a matrix

    num_rows - the first dimension of a matrix 
    num_cols - second dimension of a matrix
    """
    numerator = (num_rows + num_cols)
    # TODO: assuming log base 2, not entirely sure this is correct
    denominator = 4 * log2(num_rows + num_cols) ** 6
    return numerator / denominator

def n_valid_ss(num_rows, num_cols, n):
    """
    Get an array of n equally spaced valid s values for a matrix with the given 

    num_rows - the first dimension of the matrix to sparsify
    num_cols - the second dimension of the matrix to sparsify
    n        - the number of valid s's to generate
    """
    return linspace(1, s_upper_bound(num_rows, num_cols), n)

def s_valid(s, num_rows, num_cols):
    """
    Check if s is valid for a matrix with given dimensions

    s        - sparsification factor
    num_rows - first dimension of the  matrix A
    num_cols - second dimensions of the matrix A

    return - True if valid, False if not
    """

    if s < 1:
        # s too small
        print(f"INVALID s, {s} < 1")
        return False
    
    if (s > s_upper_bound(num_rows, num_cols)):
        # s too big
        print(f"INVALID s, {s} > {s_upper_bound(num_rows, num_cols)} = (n + d) / (4 * log(n + d)^6))")
    return True

def sparsify(A, s=2):
    '''
    Direct implementation of 8.2.2 in Sparsification Algorithms Paper

    Increasing s => make more sparse 

    Sparsify a matrix A given some value s
    A - the matrix to sparsify
    s - the factor of sparsification
    '''

    # Number of nonzeroes in A
    nnz = A.nnz

    n, d = A.shape

    # Check for invalid s's
    if (not s_valid(s, n, d)):
        return A

    for i in range(nnz):
        r = random.random() # r in range [0.0, 1.0)
        if r < 1/s:
            # With probability 1/s, scale A_{i,j}
            A.data[i] *= s
        else:
            A.data[i] = 0.0
    A.eliminate_zeros()


"""
A per row sparsification
"""
def per_row_sparse(A, s = 2):
    return A