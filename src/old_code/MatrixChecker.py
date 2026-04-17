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

        Return: None if eigs does not converge, 2 norm of difference of top
                eigenvectors otw
        '''

        # Random number generator:
        r_gen = np.random.default_rng()

        n, _ = A.shape

        # Choose the initial vector x, 1 by n
        # Initial guess is close to zero 
        v0 = r_gen.normal(loc=0.0, scale=0.01, size=n) 

        try:
            _, e = eigs(A, k=1, v0=v0) #k = 1 -> only get top eigenvector
            _, e_sparse = eigs(A_sparse, k=1, v0=v0)
            
            return self.norm_of_diff(e, e_sparse)

        except Exception as e:
            # Print exception
            print(f"Error getting top eigenvector of matrix with dimension {A.shape}")
            return None
