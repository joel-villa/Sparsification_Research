import numpy as np
from scipy import io
from scipy.sparse.linalg import eigs
from sparse_algs.simple_sparsify import sparsify 
import matplotlib.pyplot as plt
import os

# Get path to matrices 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(BASE_DIR, "matrices")

NUM_ITERATIONS = 1

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
def difference(A, A_sparse, s):
    _, e = eigs(A, k=1) #k = 1 -> only get top eigenvector
    _, e_sparse = eigs(A_sparse, k=1) 
    # print(f"e: {e}")
    # print(f"e_sparse: {e_sparse}")
    difference = abs(e) - abs(e_sparse)
    norm_of_diff = np.linalg.norm(difference, ord=2)

    if (s == 1):
        print(difference)
        print(f"norm(|e - e_sparse|) = {norm_of_diff}")
    # print()

    return norm_of_diff
    # norm_e_tilda = np.linalg.norm(, ord=1)
    # # print(f"norm_e: {norm_e}")
    # # print(f"norm_e_tilda: {norm_e_tilda}")
    # return abs()

"""

"""
def test(A):
    start = 1
    stop = 2
    num = 100
    
    diffs = np.zeros(num)
    nnzs  = np.zeros(num)
    ss    = np.zeros(num)

    for i, s in zip(range(num), np.linspace(start, stop, num)):
        diff = []
        nnz = []
        for _ in range(NUM_ITERATIONS):
            A_prime = A.copy()
            sparsify(A_prime, s)
            diff.append(difference(A, A_prime, s))
            nnz.append(A_prime.nnz) 

        # print(f"diff: {diff}")
        average_diff = np.mean(diff)
        average_nnz = np.mean(nnz)

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
    plt.show()
    plt.savefig("plots/2_norm_preservation.png")



if __name__ == '__main__': 
    for A, i in zip(As, range(1)):
        print(i)
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        print(diff)
        p_sparse = nnzs / nnz
        plot(ss, diff, p_sparse)
    # plt.show
