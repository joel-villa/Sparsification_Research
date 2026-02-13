import random

def sparsify(A, s):
    # TODO Figure out API
    for r, c, v in zip(A.row, A.col, A.data):
        if random.random() - (1 / s) >= 0:
            # Sparsify with probability 1 - 1/s
            A.data = 0
        return x * s