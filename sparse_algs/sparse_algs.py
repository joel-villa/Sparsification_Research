import random
from math import log2

"""
The following sparsification algorithms are based on 
the Sparsification Algorithms Paper:
https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf
"""

def s_valid(s, n, d):
    """
    Check if the s is valid

    s - sparsification factor
    n - first dimension of the matrix A
    d - second dimensions of the matrix A

    return - True if valid, False if not
    """

    if s < 1:
        # s too small
        print("INVALID s, s must be greater than or equal to one")
        return False
    
    # TODO: assuming log base 2, not entirely sure this is correct 
    numerator = (n + d) 
    denominator = 4 * log2(n + d) ** 6
    if (s > numerator / denominator):
        # s too big
        print("INVALID s, s must be less than or equal to (n + d) / (4 * log(n + d)^6))")
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