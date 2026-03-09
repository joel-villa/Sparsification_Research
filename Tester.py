"""
A class for testing the behavior of a Sparsifier
"""
import numpy as np
from Sparsifier import Sparsifier
from MatrixChecker import MatrixChecker

class Tester:
    """
    A class for testing a sparsifier
    """
    def __init__(self,
                 sparsifier=Sparsifier(),
                 num_iter=10, 
                 num_ss=50, 
                 max_s=5):
        self.sparsifier = sparsifier
        self.num_iter   = num_iter
        self.num_ss     = num_ss
        self.max_s      = max_s

    def test_s_behavior(self, A):
        """
        Run sparsifier.sparsify() algorithm on the matrix A given various s 
        values

        RETURN: 
        ss    - the s values run
        nnzs  - the number of nonzeros on the sparsified A's
        diffs - the difference in the two norm of the sparsified A's
        """

        diffs = np.zeros(self.num_ss)
        nnzs  = np.zeros(self.num_ss)

        cols, rows = A.shape

        sparsifier = self.sparsifier
        s_max = sparsifier.s_upper_bound(rows, cols, log_base=10)

        if s_max < 1:
            ## the upper bound is below 1, no valid s's
            print(f"s_max = {s_max} < 1, no valid s's")
            s_max = self.max_s

        if s_max > self.max_s:
            s_max = self.max_s

        ss = np.linspace(1, s_max, self.num_ss)
        mc = MatrixChecker()

        num_misses = 0 # For handling eigenvector not converging cases
        for i, s in zip(range(self.num_ss), ss):
            diff = []
            for j in range(self.num_iter):
                A_prime = A.copy()
                sparsifier.sparsify(A_prime, s)
                difference = mc.difference(A, A_prime)
                if difference is not None:
                    # Default behavior
                    diff.append(difference)
                else:
                    # Track non converging calls to eigs
                    num_misses += 1


            average_diff = np.mean(diff)
            diffs[i] = average_diff

        if (num_misses > 0):
            print(f"""WARNING: matrix with dimensions {A.shape}, and {A.nnz} 
                  nonzeroes had {num_misses} instances where the eigenvector 
                  estimator did not converge""")
        return (ss, diffs)