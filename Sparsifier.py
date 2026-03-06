import random
from math import log
from numpy import linspace

"""
The following sparsification algorithms are based on 
the Sparsification Algorithms Paper:
https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf

Which is a summary of the following paper:
https://dl.acm.org/doi/pdf/10.1145/1219092.1219097 
"""
class Sparsifier():
    """
    A sparsifier object used for sparsifying scipy.sparse matrices in csr format
    """
    def s_upper_bound(self, num_rows, num_cols, log_base=10):
        """
        Get the upper bound for the valid s values of a matrix

        num_rows - the first dimension of a matrix 
        num_cols - second dimension of a matrix
        log_base - base of the log used to generate the upperbound

        TODO: assuming log base 10, not entirely sure this is correct, the following
        bounds are given by base 10 and 2 respectively (assumed square matrix):
        base 10: num_rows > 17,755 -> max valid s value > 1
        base 2: num_rows > 2,150,000,000 -> max valid s value > 1   
        """
        numerator = (num_rows + num_cols)

        denominator = 4 *(log(num_rows + num_cols, log_base) ** 6)
        return numerator / denominator

    def n_valid_ss(self, num_rows, num_cols, n):
        """
        Get an array of n equally spaced valid s values for a matrix with the given 

        num_rows - the first dimension of the matrix to sparsify
        num_cols - the second dimension of the matrix to sparsify
        n        - the number of valid s's to generate
        """

        upper = self.s_upper_bound(num_cols, num_rows)
        if (upper < 1):
            print(f"NO VALID S VALUES: upperbound = {upper} < 1")
            return [1]
        return linspace(1, upper, n)

    def s_valid(self, s, num_rows, num_cols):
        """
        Check if s is valid for a matrix with given dimensions

        s        - sparsification factor
        num_rows - first dimension of the  matrix A
        num_cols - second dimensions of the matrix A

        return - True if valid, False if not
        """

        if s < 1:
            # s too small
            # print(f"INVALID s, {s} < 1")
            return False

        if (s > self.s_upper_bound(num_rows, num_cols)):
            # s too big
            # print(f"INVALID s, {s} > {s_upper_bound(num_rows, num_cols)} = (n + d) / (4 * log(n + d)^6))")
            return False 
        return True

    def sparsify(self, A, s=2):
        '''
        Direct implementation of 8.2.2 in Sparsification Algorithms Paper

        Increasing s => make more sparse 

        Sparsify a matrix A given some value s
        A - the matrix to sparsify
        s - the factor of sparsification
        '''

        # Number of nonzeroes in A
        nnz = A.nnz
    
        # Check for s too small
        if (s <= 1):
            return A

        for i in range(nnz):
            r = random.random() # r in range [0.0, 1.0)
            if r <= 1/s:
                # With probability 1/s, scale A_{i,j}
                A.data[i] *= s
            else:
                A.data[i] = 0.0
        A.eliminate_zeros()