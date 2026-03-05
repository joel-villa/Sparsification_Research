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

def norm_of_diff(v1, v2, s):
    """
    Return the 2 norm of the difference of the two vectors
    """

    dot_prod = np.vdot(v1, v2)
    if dot_prod < 0:
        #Force both eigenvectors to point in similar directions
        v2 = -v2

    difference = v1 - v2
    if (s == 1):
        print(f"np.vdot(e, e_sparse) = {dot_prod}")
    norm_of_diff = np.linalg.norm(difference, ord=2)

    return norm_of_diff

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
        plt.title(t + f" n = {n}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        file_name = t.replace(" ", "_")
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

def test_smalls():
    As = load_A()
    S = []
    P = []
    D = []
    ns = []

    num_mats = len(MTX_FILES)
    for A , i in zip(As, range(num_mats)):
        print(MTX_FILES[i])
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz

        D.append(diff)
        S.append(ss)
        P.append(p_sparse)
        ns.append(nnz)

    plot(S, D, MTX_FILES, ns)


def test_bigs():
    """
    Doing tests for large matrices 
    """
    A_dict = get_mats()
    S = []
    P = []
    D = []
    names = []
    ns = []
    for name, A in A_dict.items():
        # if (name == "bcsstm39" or name == "crystm03"):
        print(name)
        nnz = A.nnz
        ss, nnzs, diff = test(A)
        p_sparse = nnzs / nnz
        D.append(diff)
        S.append(ss)
        P.append(p_sparse)
        names.append(name)
        ns.append(nnz)

    plot(S, D, names, ns)
  
if __name__ == '__main__': 
    #TODO: MSE
    test_smalls()