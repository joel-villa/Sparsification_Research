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

START = 1
STOP = 4
NUM = 100

Ms = ["494_bus.mtx" ,
      "662_bus.mtx" ,
      "685_bus.mtx" ,
      "1138_bus.mtx",
      "arc130.mtx"  ,
      "ash85.mtx"   ,
      "ash292.mtx"  
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

    e = e.real
    e_sparse = e_sparse.real

    if np.vdot(e, e_sparse) < 0:
        #Force both eigenvectors to point in similar directions
        e_sparse = -e_sparse

    difference = e - e_sparse
    norm_of_diff = np.linalg.norm(difference, ord=2)

    return norm_of_diff

"""

"""
def test(A):
    
    diffs = np.zeros(NUM)
    nnzs  = np.zeros(NUM)
    ss    = np.zeros(NUM)

    for i, s in zip(range(NUM), np.linspace(START, STOP, NUM)):
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

X - A 2D array of x values
Y - A 2D array of y values

"""
def plot(X, Y, title):
    i = 0
    for x, y in zip(X, Y):
        plt.plot(X[i], Y[i])
        plt.
        i += 1
    ax[0].plot(ss, diff, marker='')
    ax[0].set_title("2 Norm Preservation")
    ax[0].set_xlabel('s')

    ax[1].plot(ss, p_sparse, marker='')
    ax[1].set_title("Percent Sparsified")
    ax[1].set_xlabel('s')
    plt.plot()
    plt.legend()
    plt.show()
    plt.savefig("plots/" + title + ".svg")

def load_A():
    #TODO: load this properly
    # Read in matrices in CSR format
    As = [io.mmread(os.path.join(matrix_path, "494_bus.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "662_bus.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "685_bus.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "1138_bus.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "arc130.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "ash85.mtx")).tocsr(),
          io.mmread(os.path.join(matrix_path, "ash292.mtx")).tocsr()
          ]

if __name__ == '__main__': 

    S = []
    P = []
    for A, i in zip(As, range(1)):
        print(i)
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz
        plot(ss, diff, p_sparse)
