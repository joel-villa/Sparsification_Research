from scipy.io import mmread
from scipy.sparse.linalg import eigs
import numpy as np
import sparse_algs.sparse_algs as spa
import matplotlib.pyplot as plt
import os
from MatGetter import get_mats

# Get path to matrices 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
matrix_path = os.path.join(BASE_DIR, "matrices")

NUM_ITERATIONS = 5
NUM_SS = 50
MAX_S = 5

MTX_FILES = ["494_bus.mtx" ,
             "662_bus.mtx" ,
             "685_bus.mtx" ,
             "1138_bus.mtx",
             "arc130.mtx"  ,
             "ash85.mtx"   ,
             "ash292.mtx"  
]

def difference(A, A_sparse):
    '''
    Return the 2 norm the difference of the top eigenvectors of the 
    matrix A and it's sparsified counterpart 

    A        - original matrix
    A_sparse - sparsified matrix
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

def test(A):
    """
    Run sparsification algorithm on the matrix A given various s values
    
    RETURN:
    ss    - the s values run
    nnzs  - the number of nonzeros on the sparsified A's
    diffs - the difference in the two norm of the sparsified A's
    """
    
    diffs = np.zeros(NUM_SS)
    nnzs  = np.zeros(NUM_SS)

    cols, rows = A.shape

    
    s_max = spa.s_upper_bound(rows, cols, log_base=10)

    if s_max < 1:
        ## the upper bound is below 1, no valid s's
        print(f"s_max = {s_max} < 1, no valid s's")
        s_max = MAX_S

    if s_max > MAX_S:
        s_max = MAX_S

    ss = np.linspace(1, s_max, NUM_SS)

    for i, s in zip(range(NUM_SS), ss):
        diff = np.zeros(NUM_ITERATIONS)
        nnz = np.zeros(NUM_ITERATIONS)
        for j in range(NUM_ITERATIONS):
            A_prime = A.copy()
            spa.sparsify(A_prime, s)
            diff[j] = difference(A, A_prime)
            nnz[j] = A_prime.nnz

        average_diff = np.mean(diff)
        average_nnz = np.mean(nnz)
        diffs[i] = average_diff
        nnzs[i] = average_nnz

    return (ss, nnzs, diffs)

def plot(X, Y, labels, x_label, y_label, title):
    """
    Generate plots

    X - A 2D array of x values
    Y - A 2D array of y values
    """

    for x, y, lbl in zip(X, Y, labels):
        plt.plot(x, y, label=lbl)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    file_name = title.replace(" ", "_")
    plt.savefig("plots/" + file_name  + ".svg")
    plt.show()

def load_A():
    """
    Load in the .mtx files as a scipy csr sparse matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html 
    """
    A = []
    for mtx_file in MTX_FILES:
        # Read in matrices in CSR format
        A.append(mmread(os.path.join(matrix_path, mtx_file)).tocsr())
    return A

def load_big():
    """
    Loading large matrices
    """
    return get_mats()
  
if __name__ == '__main__': 
    #TODO: MSE
    #As = load_A()\
    As = load_big()
    S = []
    P = []
    D = []

    num_mats = len(As)
    for A , i in zip(As, range(num_mats)):
        print(MTX_FILES[i])
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz

        D.append(diff)
        S.append(ss)
        P.append(p_sparse)
    
    
    plot(S, D, MTX_FILES, "s", rf"$||e - \tilde e||$", "Sparsification Behavior")
