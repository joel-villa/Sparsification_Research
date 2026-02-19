import numpy as np
from scipy import io
from scipy.sparse.linalg import eigs
import simple_sparsify 
import matplotlib.pyplot as plt

NUM_ITERATIONS = 25


# Read in matrix in CSR format
A = io.mmread("matrices/1138_bus.mtx").tocsr() 

# plot

'''
Return the factor by which the spectral radius is "off", i.e. the c s.t.
c * rho = rho'
Where rho is the spectral radius of A
And rho' is the spectral radius of A' (or B)
'''
def diffSpecRad(A, B):
    eigenvalues_A, eigenvectors_A = eigs(A, k=1) #k = 1 -> only get top eigenvalue (spectral radius)
    eigenvalues_B, eigenvectors_B = eigs(B, k=1) 
    # print(f"eigenvalues_A: {eigenvalues_A}\n eignevalues_B: {eigenvalues_B}")
    return abs(eigenvalues_A[0] / eigenvalues_B[0])


def test():
    diff_spec = 0
    specs = []
    nnzs = []
    ss = []
    for s in range(1, 100, 5):
        for i in range(NUM_ITERATIONS):
            A_prime = A.copy()
            simple_sparsify.sparsifyCSR(A_prime, s)
            diff_spec += diffSpecRad(A, A_prime)
        diff_spec /= NUM_ITERATIONS
        ss.append(s)
        specs.append(diff_spec)
        nnzs.append(A_prime.nnz)
    return (ss, nnzs, specs)

def plot(ps, specs, nnzs, title_1, title_2, save_name):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    ax[0].plot(ps, specs, marker='')
    ax[0].set_xlabel('Percent Sparsified')
    # ax[0].xscale('log', base=2) # Set x-axis to logarithmic scale
    ax[0].set_title(title_1)

    ax[1].plot(ps, nnzs, marker='')
    ax[1].set_xlabel('Percent Sparsified')
    # ax[1].xscale('log', base=2) # Set x-axis to logarithmic scale
    ax[1].set_title(title_2)

    plt.show()
    plt.savefig("./plots/" + save_name)

if __name__ == '__main__': 
    (ss, nnzs, specs)= test()
    # ps = 
    plot(ss, specs, nnzs, "Accuracy (Based on Spectral Radius)",  "Number of Nonzeroes", "spectral_radius_preservation.png")

