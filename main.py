import numpy as np
from scipy import io
from scipy.sparse.linalg import eigs
import simple_sparsify 
import matplotlib.pyplot as plt

NUM_ITERATIONS = 10


# Read in matrix in CSR format
A = io.mmread("matrices/1138_bus.mtx").tocsr() 

# plot

'''
Return the factor by which the spectral radius is "off", i.e. the c s.t.
c * rho = rho'
Where rho is the spectral radius of A
And rho' is the spectral radius of A' (or B)
'''
def difference(A, A_tilda, s):
    #TODO: 2 norm of the difference of top eigenvectors
    eigenvalues_A, eigenvectors_A = eigs(A, k=1) #k = 1 -> only get top eigenvalue (spectral radius)
    eigenvalues_B, eigenvectors_A_tilda = eigs(A_tilda, k=1) 

    # 2 norm of both top eigenvectors
    norm_A       = np.linalg.norm(eigenvectors_A, ord=1)
    norm_A_tilda = np.linalg.norm(eigenvectors_A_tilda, ord=1)
    # if (s == 90):
    #     # print(A.shape)
    #     # print(f"eig_a: {eigenvalues_A[0]}")
    #     # print(f"eig_a_tilda: {eigenvalues_B[0]}")
    #     # print(f"eigv_a: {eigenvectors_A}")
    #     # print(f"eigv_a_tilda: {eigenvectors_A_tilda}")
    #     print(f"norm_a: {norm_A}")
    #     print(f"norm_a_tilda: {norm_A_tilda}")
    #     # print(f"min(abs(norm_A - norm_A_tilda) / norm_A, 1) = {min(abs(norm_A - norm_A_tilda) / norm_A, 1)}")
    return min(abs(norm_A - norm_A_tilda)/ norm_A, 1)


def test():
    d = []
    diffs = []
    nnzs = []
    ss = []

    for s in range(0, 100, 5):
        # print(s)
        for i in range(NUM_ITERATIONS):
            A_prime = A.copy()
            simple_sparsify.sparsifyCSR(A_prime, s)
            d.append(difference(A, A_prime, s))
        average_diff = np.average(d)
        # print(average_diff)
        d = []
        ss.append(s)
        diffs.append(average_diff)
        nnzs.append(A_prime.nnz)

    return (ss, nnzs, diffs)

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
    plot(ss, specs, nnzs, "Accuracy (Based on 2 Norm)",  "Number of Nonzeroes", "spectral_radius_preservation.png")

