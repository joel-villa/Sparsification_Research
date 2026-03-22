"""
This class is used for checking the validity of a matrix and two vectors that
supposedly solve Ax = b
"""

from scipy.linalg import norm

class ResCalculator:
    def __init__(self, A=None, x=None, b=None):
        """
        Initialize the Calculator given some combination of A, x, and b 
        that is likely unchanging
        A - a nxm sparse matrix
        x - a vector of length m
        b - a vector of length n
        """
        if (A is not None):
            self.A = A.copy()
        if (x is not None):
            self.x = x.copy()
        if (b is not None):
            self.b = b
        
    def get_res(self, A=None, x=None):
        """
        Given a matrix, and two vectors that supposedly solve Ax = b, 
        calculate the residual and return it

        A - an nxm sparse matrix
        x - a vector of length m

        RETURN: residual of Ax = b
        """

        if (A is None):
            # Set A to initialization A
            if (self.A is None):
                print(f"ERROR: A never set in ResCalculator")
            A = self.A

        if (x is None):
            # Set x to initialization x
            if (self.x is None):
                print(f"ERROR: x never set in ResCalculator")
            x = self.x
        
        res = self.b - (A @ x)

        return (norm(res) / norm(self.b))