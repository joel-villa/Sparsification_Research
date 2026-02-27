import numpy as np
from scipy.io import mmread
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

MTX_FILES = ["494_bus.mtx" ,
             "662_bus.mtx" ,
             "685_bus.mtx" ,
             "1138_bus.mtx",
             "arc130.mtx"  ,
             "ash85.mtx"   ,
             "ash292.mtx"  
]

def difference(A, A_sparse, s):
    '''
    Return the percent difference of the 2 norm of the sparsified matrices
    top eigenvector
    A       - original matrix
    A_tilda - sparsified matrix
    '''
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

def plot(X, Y, labels, x_label, y_label, title):
    """
    Generate plots

    X - A 2D array of x values
    Y - A 2D array of y values
    """
    # ax[0].plot(ss, diff, marker='')
    # ax[0].set_title("2 Norm Preservation")
    # ax[0].set_xlabel('s')

    # ax[1].plot(ss, p_sparse, marker='')
    # ax[1].set_title("Percent Sparsified")
    # ax[1].set_xlabel('s')

    for x, y, lbl in zip(X, Y, labels):
        plt.plot(x, y, label=lbl)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    file_name = title.replace(" ", "_")
    plt.savefig("plots/" + title + ".svg")

    plt.show()

def load_A():
    """
    Load in the .mtx files as a scipy csr sparse matrix
    """
    A = []
    for mtx_file in MTX_FILES:
        # Read in matrices in CSR format
        A.append(mmread(os.path.join(matrix_path, mtx_file)).tocsr())
    return A

if __name__ == '__main__': 
    As = load_A()
    S = []
    P = []
    D = []
    for A , i in zip(As, range(2)):
        # print(i)
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz

        D.append(diff)
        S.append(ss)
        P.append(p_sparse)
    
    plot(S, D, MTX_FILES, "s", rf"$||e - \tilde e||$", "Sparsification Behavior")