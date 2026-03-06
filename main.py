from scipy.io import mmread
from scipy.sparse.linalg import eigs
import numpy as np
import sparse_algs.sparse_algs as spa
import matplotlib.pyplot as plt
import os
from SSGetter import get_mats
from SSGetter import SSGetter

"""
TODO: 
(1) use SSGetter rather than get_mats
(2) Make Utitility class for checking difference of two matrices: MatChecker? 
(3) Clean code up
"""
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

def norm_of_diff(v1, v2, s):
    """
    Return the 2 norm of the difference of the two vectors
    """
    # Simpler approach: take the minimum of flipping one of th vectors
    v2_neg = -v2
    
    
    difference_pos = v1 - v2
    difference_neg = v1 - v2_neg

    # Test the difference of both the eigenvector and the eigenvector flipped
    # Take minimum
    norm_of_diff_pos = np.linalg.norm(difference_pos, ord=2)
    norm_of_diff_neg = np.linalg.norm(difference_neg, ord=2)

    return min(norm_of_diff_pos, norm_of_diff_neg)

def difference(A, A_sparse, s):
    '''
    Return the 2 norm of the difference of the top eigenvectors of the 
    matrix A and it's sparsified counterpart 

    A        - original matrix
    A_sparse - sparsified matrix
    '''
    _, e = eigs(A, k=1) #k = 1 -> only get top eigenvector
    _, e_sparse = eigs(A_sparse, k=1) 

    e = e.real
    e_sparse = e_sparse.real

    return norm_of_diff(e, e_sparse, s)

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
            diff[j] = difference(A, A_prime, s)
            if (s == 1):
                print(f"s = {s}, diff[j] = {diff[j]}, A.nnz = {A.nnz}, A_prime.nnz = {A_prime.nnz}")
            nnz[j] = A_prime.nnz

        average_diff = np.mean(diff)
        average_nnz = np.mean(nnz)
        diffs[i] = average_diff
        nnzs[i] = average_nnz

    return (ss, nnzs, diffs)

def plot(X, Y, labels, ns):
    """
    Generate plots

    X - A 2D array of x values
    Y - A 2D array of y values
    """
    title = "Sparsification Behavior"
    x_label = "s"
    y_label = rf"$||e - \tilde e||$"
    for x, y, lbl, n in zip(X, Y, labels, ns):
        t = title + " of " + lbl
        plt.plot(x, y)
        plt.title(t + f" (n = {n})")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend()
        file_name = t.replace(" ", "_")
        plt.savefig("plots/" + lbl  + ".svg")
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

def iter(A, S, D, ns):
    n, _ = A.shape
    ss, _, diff = test(A)
    D.append(diff)
    S.append(ss)
    ns.append(n)

def test_smalls():
    As = load_A()
    S = []
    D = []
    ns = []

    num_mats = len(MTX_FILES)
    for A , i in zip(As, range(num_mats)):
        print(MTX_FILES[i])
        iter(A, S, D, ns)

    plot(S, D, MTX_FILES, ns)


def test_bigs():
    """
    Doing tests for large matrices 
    """
    A_dict = SSGetter(True, row_bounds=(17755,100000))
    A_dict = SSGetter.get_next(5)
    S = []
    D = []
    names = []
    ns = []
    for name, A in A_dict.items():
        print(name)
        iter(A, S, D, ns)
        names.append(name)

    plot(S, D, names, ns)
  
if __name__ == '__main__': 
    #TODO: MSE
    # test_smalls()
    test_bigs()