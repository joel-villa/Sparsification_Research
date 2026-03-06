from scipy.sparse.linalg import eigs
import numpy as np


class MatrixChecker:
    """
    Class for checking the difference between two scipy.sparse matrices
    """
    def norm_of_diff(self, v1, v2):
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

        # print(f"norm_of_diff_pos = {norm_of_diff_pos}, norm_of_diff_neg = {norm_of_diff_neg}")

        # Take minimum
        return min(norm_of_diff_pos, norm_of_diff_neg)

    def difference(self, A, A_sparse):
        '''
        Return the 2 norm of the difference of the top eigenvectors of the 
        matrix A and it's sparsified counterpart 

        A        - original matrix
        A_sparse - sparsified matrix
        '''

        # Random number generator:
        r_gen = np.random.default_rng()

        n, _ = A.shape

        # Choose the initial vector x, 1 by n
        # Initial guess is close to zero 
        v0 = r_gen.normal(loc=0.0, scale=0.01, size=n) 

        _, e = eigs(A, k=1, v0=v0) #k = 1 -> only get top eigenvector
        _, e_sparse = eigs(A_sparse, k=1, v0=v0) 

        # print(f"e: {e}")
        # print(f"e_sparse: {e_sparse}")
        # for i in range(len(e)):
        #     if e[i] - e_sparse[i] > 0.001:
        #         print(f"e[{i}]: {e[i]}, e_sparse[{i}]: {e_sparse[i]}")
        # e = e.real
        # e_sparse = e_sparse.real

        return self.norm_of_diff(e, e_sparse)

if __name__ == '__main__': 
    mp = MatrixComparer()
    mp.difference()