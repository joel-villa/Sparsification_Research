import random

'''
Based on Theorem 8.2.2 in Sparsification Algorithms Paper:
https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf

Increasing s => make more sparse 
'''
def sparsifyCSR(A, s=5):
    # Number of nonzeroes in A
    nnz = A.nnz

    if s == 0:
        return A

    for i in range(nnz):
        r = random.random() # r in range [0.0, 1.0)
        if r < 1/s:
            # With probability 1/s, scale A_{i,j}
            A.data[i] *= s
        else:
            # With probability 1 - 1/s, sparsify A_{i,j}
            A.data[i] = 0.0
    A.eliminate_zeros()
    
    A_prime_nnz = A.nnz
    e = nnz / s
    # print(f"Expected number of nonzeroes: {e}")

    #TODO: is this a silly check
    if (A_prime_nnz >= (nnz / s) ** 2):
        print("Warning: number of nonzeroes in A' is greater than expectation squared")