"""
A class for generating apt s-values,

PAPER: https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf

Goal: what factor of sparsification best converges with the Jacobi itterative 
method
"""
import numpy as np

class SGenerator:
    def __init__(self, n, nnz, log_base=2):
        """
        Some information about the dimensions of the matrix to pick s-values for
        n        - number of rows and number of columns of the matrix (assuming square)
        nnz      - number of nonzeros of matrix (assuming sparse)
        log_base - performing log function in s generation, what base? 

        BEHAVED BEST EMPIRICALLY (so far):
        Sqrt(nnz) 
        Log base 2

        """
        self.n = n
        self.nnz = nnz
        #The number of potential sparsified values (when not sparsifying diagonal)
        self.off_diags = self.nnz - self.n
        self.log = self.get_log_func(log_base)

    def get_log_func(self, log_base):
        """
        log_base = 2 -> log2()
        log_base = anything else -> natural log

        RETURN: a numpy log function
        """
        match log_base:
            # Get log function
            case 2:
                return np.log2
            case _:
                return np.log
    
    def get_min_s(self, x=1):
        """
        The minimum s-value s.t. the expected number of nonzeros is 1 or more

        x - the desired expected amount of new nonzeros after sparsification

        This values was found mathematically:
        E(X) = # of new nonzeros
        E(X) = n(1-s)
        E(X) >= x --> s >= n / (n-x)
            where n is the candidate nonzeros (off diagonal count)
        """
        if (self.off_diags - x <= 0):
            raise ValueError(f"nnz candidates is {self.off_diags}, can't reduce by {x}")

        s = self.off_diags / (self.off_diags - x)

        return s
    
    def proportion_sparse_s(self, p=0.5, include_diags=False):
        """ Get the s value s.t. the expected percent of sparsification is p

        Args:
            p: the expected proportion of reduction
            include_diags: True -> based on nnzs, 
                           False -> based on nnzs - diagonal
        
        Return:
            The appropriate s-value
        """        
        if (include_diags):
            # Diagonal is being sparsified, take it into account
            x = self.nnz * p
        else:
            # Diagonal is not being sparsified, don't account for it
            x = self.off_diags * p

        return self.get_min_s(x)
    
    def get_safe_s(self, type="nnz"):
        """
        Get the maximum possible s-value (i.e. expected number of zeros greater
        than one, and based on upper bound of sparsification algorithm)

        type - "nnz"        -> square root of the number of zeros in matrix is n
               "candidates" -> square root of non-diagonals in matrix is n
               _            -> matrix dimension is n
        """
        return max(self.get_min_s(), self.get_s(type))

    def get_s(self, type="nnz"):
        """
        Based on upper bound of s in the paper, generate an s

        type - "nnz"        -> square root of the number of zeros in matrix is n
               "candidates" -> square root of non-diagonals in matrix is n
               _            -> matrix dimension is n
        """
        match type:
            # Get n value
            case "nnz":
                n = np.sqrt(self.nnz)
            case "candidates":
                n = np.sqrt(self.off_diags)
            case _:
                n = self.n
        
        two_n = 2*n
        
        #  log2() -> log base 2
        denominator = 4 * ((self.log(two_n)) ** 6)

        s = (two_n / denominator)

        # Adding one to force to be in valid s-range
        return s + 1