import numpy as np
from scipy import io
import simple_sparsify 

s = 10

# Read in matrix in CSR format
A = io.mmread("matrices/1138_bus.mtx").tocsr() 



if __name__ == '__main__': 

    A_nnz = A.nnz

    simple_sparsify.sparsifyCSR(A, s)

    A_prime_nnz = A.nnz

    print(f"type(A): {A_nnz}")
    print(f"A.shape: {A_prime_nnz}")

    io.mmwrite("matrices/sparse_1138_bus.mtx", A)


