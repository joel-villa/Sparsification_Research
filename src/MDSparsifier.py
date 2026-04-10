"""
An algorithm for sparsifying matrices which lends itself to Jacobi

Note: Jacobi gaurunteed to converge if it is strictly diagonally dominant
      Symmetric, positive definite matrices are likely, but not guaranteed to converge
      
"""

import random
from .Sparsifier import Sparsifier
import numpy as np

class MDSparsifier(Sparsifier):
    """
    Mantain Diagonal Sparsifier, i.e. sparsify as usual, except garuntee 
    diagonal is kept and scaled
    """
    def __init__(self, seed=42):
        super().__init__(seed)

    def only_diag(self, A):
        """
        For testing purposes, keeps only the diagonal of a matrix
        """    
        for row, col, i in zip(A.row, A.col, range(A.nnz)):
            if row != col:
                # Get rid of off diagonal entries
                A.data[i] *= 0
        A.eliminate_zeros()
    
    def sparsify(self, A, s):
        """
        Based on Theorem 8.2.2 in Sparsification Algorithms Paper:
        https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf

        Sparsify a matrix A given some value s
        A - the matrix to sparsify (assumed to be in coo format)
        s - factor of sparsification (Increasing s => make more sparse)
        
        Sparsifying for Jacobi preconditioning -> always keep diagonal entries

        Psuedocode: 
        (1) sparsify entries of A' with probability 1 - 1/s (except diags)
        (2) if not sparsified, scale up by factor of s
            Notice: all diags scaled up -> Jacobi should still converge
        """
        for row, col, i in zip(A.row, A.col, range(A.nnz)):
            if row == col:
                #Gaurunteed to keep and scale up diagonals
                A.data[i] *= s
            else:
                # Normal sparsification algorithm otherwise
                A.data[i] = self.sparse_entry(x=A.data[i], s=s)
        A.eliminate_zeros()
