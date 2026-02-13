'''
Joel Villarreal II
2/2/26
A file for testing parsing Suite Sparse matrices
'''
# Note to self on activating venv environment: source path/to/venv/bin/activate
# Once venv environment active, run `pip install scipy` and `pip install numpy`

# Import the numpy library with the alias np
import numpy as np
# import matrix

# Suite Sparse Matrix Collection Getter: https://nbviewer.org/github/drdarshan/ssgetpy/blob/master/demo.ipynb
import ssgetpy
# Downloads go to ~/.ssgetpy/MM/HB

# CSR - compressed sparse row format
# (see: https://en.wikipedia.org/wiki/Sparse_matrix, 
# and https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
from scipy.sparse import csr_matrix
from scipy import io

# main 
if __name__ == '__main__':
    #  
    A = io.mmread("matrices/1138_bus.mtx") #.tocsr() for csr format
    
