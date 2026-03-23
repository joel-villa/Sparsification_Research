"""
A class for testing the behavior of a Sparsifier
"""
import numpy as np
from Sparsifier import Sparsifier
from MatrixChecker import MatrixChecker
from scipy.linalg import norm # 2 norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html

class Tester:
    """
    A class for testing a sparsifier

    sparsifier - the sparsifier which can be used to sparsify a scipy sparse
                 matrix
    num_iter   - the number of iterations the tester will do
    num_ss     - only relevant for test_s_behavior
    max_s      - only relevant for test_s_behavior
    """
    def __init__(self,
                 sparsifier=Sparsifier(),
                 num_iter=1, 
                 num_ss=50, 
                 max_s=5):
        self.sparsifier = sparsifier
        self.num_iter   = num_iter
        self.num_ss     = num_ss
        self.max_s      = max_s

    def test(self, A, ss):
        """
        Given a scipy sparse matrix A, and some factor of sparsification s, 
        test how close Ax is to ~Ax, where ~A is the sparsified version of A.
        Do this for randomly generated x values, num_iter times

        A  - sparse matrix 
        ss - the s value(s) to sparsify A with
        """

        residuals = []
        xs = np.random.rand(A.shape[0], self.num_iter) # randomly generated x vectors
        
        for s_val in ss:
            s_res = []
            for i in range(xs.shape[1]):
                x = xs[:, i]
                
                # Sparsify A
                A_sparse = A.copy()
                self.sparsifier.sparsify(A_sparse, s_val)

                # Calculate b
                b = A @ x

                # Calculate sparse b
                b_sparse = A_sparse @ x

                # Measure and save difference (residual)
                res = (norm(b - b_sparse) / norm(b))
                s_res.append(res)
            residuals.append(s_res)

        # Return residuals
        return residuals
        #TODO: test this


    def test_s_behavior(self, A):
        """
        Run sparsifier.sparsify() algorithm on the matrix A given various s 
        values

        RETURN: 
        ss    - the s values run
        diffs - the difference in the two norm of the sparsified A's
        """

        diffs = np.zeros(self.num_ss)

        sparsifier = self.sparsifier

        cols, rows = A.shape

        ss = self.get_ss(rows=rows, cols=cols)

        mc = MatrixChecker()

        num_misses = 0 # For handling eigenvector not converging cases
        for i, s in zip(range(self.num_ss), ss):
            diff = []
            for _ in range(self.num_iter):
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
        return diffs
    

    def get_ss(self, rows, cols):
        """
        Get the s values for a particular matrix A
        rows - number of rows in A 
        cols - number of columns in A

        return - a set of self.num_ss points from 1 to some upperbound
        """
       
        s_max = self.sparsifier.s_upper_bound(num_rows=rows, num_cols=cols, log_base=10)

        if s_max < 1:
            ## the upper bound is below 1, no valid s's
            print(f"s_max = {s_max} < 1, no valid s's")
            s_max = self.max_s

        if s_max > self.max_s:
            s_max = self.max_s

        return np.linspace(1, s_max, self.num_ss)