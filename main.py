import numpy as np
from scipy import io
from scipy.sparse.linalg import eigs
from sparse_algs.simple_sparsify import sparsify 
import matplotlib.pyplot as plt
import os

# Get path to matrices 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(BASE_DIR, "matrices")

NUM_ITERATIONS = 10

# Read in matrices in CSR format
As = [io.mmread(os.path.join(matrix_path, "494_bus.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "662_bus.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "685_bus.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "1138_bus.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "arc130.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "ash85.mtx")).tocsr(),
      io.mmread(os.path.join(matrix_path, "ash292.mtx")).tocsr()
      ]

'''
Return the percent difference of the 2 norm of the sparsified matrices
top eigenvector
A       - original matrix
A_tilda - sparsified matrix
'''
def difference(A, A_tilda):
    _, eigenvectors_A = eigs(A, k=1) #k = 1 -> only get top eigenvector
    _, eigenvectors_A_tilda = eigs(A_tilda, k=1) 

    # 2 norm of both top eigenvectors
    norm_A       = np.linalg.norm(eigenvectors_A, ord=2)
    norm_A_tilda = np.linalg.norm(eigenvectors_A_tilda, ord=2)

    return abs(norm_A - norm_A_tilda)/ norm_A

"""

"""
def test(A):
    start = 0
    stop = 100
    step = 2
    n = (stop - start) // step
    diffs = np.zeros(n)
    nnzs = np.zeros(n)
    ss = np.zeros(n)

    for i, s in zip(range(n), range(start, stop, step)):
        diff = 0
        nnz = 0
        for _ in range(NUM_ITERATIONS):
            A_prime = A.copy()
            sparsify(A_prime, s)
            diff += difference(A, A_prime)
            nnz += A_prime.nnz

        average_diff = diff / NUM_ITERATIONS
        average_nnz = nnz / NUM_ITERATIONS

        ss[i] = s
        diffs[i] = average_diff
        nnzs[i] = average_nnz

    return (ss, nnzs, diffs)

"""
Generate plots demonstrating sparsification behavior

ss       - the factor of sparsification
diff     - the percent difference in the 2 norm
p_sparse - the pecent the matrix was sparsified

"""
def plot(ss, diff, p_sparse):
    # TODO fix plots
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    ax[0].plot(ss, diff, marker='')
    # ax.plot(ss, p_sparse, marker='', label="Percent Sparsified")
    ax[0].set_xlabel('s')
    # ax[0].set_ylabel('s')
    ax[0].set_title("Sparsification Behavior")

    # ax[1].plot(ss, diff, marker='', label='Accuracy (Based on 2 Norm)')
    ax[1].plot(ss, p_sparse, marker='')
    ax[1].set_xlabel('s')

    plt.legend()

if __name__ == '__main__': 
    for A, i in zip(As, range(len(As))):
        print(i)
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz
        plot(ss, diff, p_sparse)
    plt.show
    plt.savefig("plots/2_norm_preservation.png")
