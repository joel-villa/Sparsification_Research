import numpy as np
import matplotlib.pyplot as plt
import os
from SSGetter import SSGetter
from MatrixChecker import MatrixChecker
from Sparsifier import Sparsifier

NUM_ITERATIONS = 5
NUM_SS = 50
MAX_S = 5

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

    sparsifier = Sparsifier()
    s_max = sparsifier.s_upper_bound(rows, cols, log_base=10)

    if s_max < 1:
        ## the upper bound is below 1, no valid s's
        print(f"s_max = {s_max} < 1, no valid s's")
        s_max = MAX_S

    if s_max > MAX_S:
        s_max = MAX_S

    ss = np.linspace(1, s_max, NUM_SS)
    mc = MatrixChecker()

    for i, s in zip(range(NUM_SS), ss):
        diff = np.zeros(NUM_ITERATIONS)
        nnz = np.zeros(NUM_ITERATIONS)
        for j in range(NUM_ITERATIONS):
            A_prime = A.copy()
            sparsifier.sparsify(A_prime, s)
            diff[j] = mc.difference(A, A_prime)
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

if __name__ == '__main__':
    # big_ssgetter = SSGetter(True, row_bounds=(17755,100000))
    small_ssgetter = SSGetter(True, row_bounds=(100,10000))

    # big_mats = big_ssgetter.get_next(5)
    small_mats = small_ssgetter.get_next(5)

    S = []
    D = []
    names = []
    ns = []

    

    for name, A in small_mats.items():
        n, _ = A.shape
        ss, _, diff = test(A)
        D.append(diff)
        S.append(ss)
        ns.append(n)
        names.append(name)

    plot(S, D, names, ns)
