import numpy as np
from scipy import io
import simple_sparsify 

s = 10


if __name__ == '__main__': 
    A = io.mmread("matrices/1138_bus.mtx") 

    # TODO o
    vectorized_sparsify = np.vectorize(simple_sparsify.sparsify)

    A_prime = vectorized_sparsify(A)

    print(type(A_prime))
    print(A_prime.shape)

    # TODO: 
    io.mmwrite("matrices/sparse_1138_bus.mtx", A_prime)


