"""
A file for interfacing with the suite sparse matrix collection
https://nbviewer.org/github/drdarshan/ssgetpy/blob/master/demo.ipynb

To use the ssgetpy library: 
pip install ssgetpy
"""
import tarfile
from scipy.io import mmread

import ssgetpy
import os

class SSGetter:
    """
    A wrapper for the ssgetpy library

    variables:
    in_csr       - FALSE -> return matrices in coo format
                   TRUE  -> return matrices in csr format
    row_bounds   - upper and lower bound on dimensions of the matrix 
                   (assuming symmetric -> no need for col_bounds)
    isspd        - matrix assumed to be symetric positive definite
    mats         - A queue of matrices fetched but not yet loaded in
    num_gotten   - tracking how many sparse matrices have been fetched via 
                   ssgetpy
    """
    def __init__(self, 
                 in_csr=True, 
                 row_bounds=(17755,100000), 
                 col_bounds=(17755,100000),
                 isspd=True):
        self.in_csr     = in_csr
        self.row_bounds = row_bounds
        self.col_bounds = col_bounds
        self.isspd      = isspd
        self.dtype      = "real" #TODO: is this okay?
        self.num_gotten = 0

    def get_by_name(self, names):
        """
        Generate a dictionary of form (key, value) = ("name", matrix)
        """
        mat_dict = {}

        for name in names:
            try:
                mat = ssgetpy.search(name)[0]
            except: 
                print(f"Matrix with name \"{name}\" is not in suitesparse collection")

            loaded_mat = self._load_mat(mat)
            if self.in_csr:
                loaded_mat = loaded_mat.tocsr()
            mat_dict[mat.name] = loaded_mat
        
        return mat_dict

    def get_next(self, num_mats=4):
        """
        Gets the next matrices from the suite sparse matrix collection
        
        - num_mats - number of matrices to return 

        RETURN: dictionary with matrix names for keys and the csr or coo 
                matrices for values

        Psuedocode:
        (1) Fetch mats with ssgetpy and save metadata
            ssgetpy always fetches same matrices in same order -> must fetch 
            num_gotten + num_to_fetch and use list slicing to exclude the 
            duplicates
        (2) Download and load in num_mats matrices, necessitates 
            interracting with tar files using the OS API
        (3) Return dictionary
        """

        # (1) Fetch mats with ssgetpy and save metadata
        mats = ssgetpy.search(rowbounds=self.row_bounds,
                              colbounds=self.col_bounds, 
                              isspd=self.isspd, 
                              dtype=self.dtype,
                              limit=(self.num_gotten + num_mats))
        
        mats = mats[self.num_gotten:]

        mat_dict = {}

        # (2) Download and load in num_mats matrices in specified format
        for m in mats: 
            loaded_mat = self._load_mat(m)
            if self.in_csr:
                loaded_mat = loaded_mat.tocsr()
            mat_dict[m.name] = loaded_mat

        if (len(mat_dict) < num_mats):
            # When not enough matrices read in, print a warning
            print(f"Warning, {num_mats} matrices requested, returning {len(mat_dict)}")

        # Increment num_gotten
        self.num_gotten += num_mats
        
        return mat_dict

    def _load_mat(self, m):
        """
        Load the matrix m in from memory and return it in coo format

        m - matrix to load from memory
        RETURN: a scipy.sparse.coo_matrix 
        """
        m.download()  # Download the matrix
        
        # Get the local path of the .tar.gz file
        path_tuple = m.localpath() # This returns a tuple (two identical paths)
        tar_path = path_tuple[0]   # Use the first path (both are the same)
        
        try:
            # Extract the .tar.gz file to get the matrix file inside
            if tar_path.endswith('.tar.gz'):
                with tarfile.open(tar_path, 'r:gz') as tar:
                    # Extract files in the same folder as the .tar.gz file
                    extract_dir = os.path.dirname(tar_path)
                    tar.extractall(path=extract_dir)  # Extract to the same folder
                
                # After extraction, find the matrix file (.mtx) in the extracted files
                matrix_file = None
                extract_dir = os.path.join(extract_dir, m.name)
                for extracted_file in os.listdir(extract_dir):
                    if extracted_file.endswith('.mtx'):
                        matrix_file = extracted_file
                        break
                
                if matrix_file is None:
                    raise FileNotFoundError(f"No .mtx file found in {extract_dir}")
                
                # Full path to the extracted matrix file
                extracted_path = os.path.join(extract_dir, matrix_file)

                # Now, read the extracted matrix file using mmread and return it in coo
                A = mmread(extracted_path)
                return A
            else:
                print(f"Matrix {m.name} is not in .tar.gz format.")
        
        except Exception as e:
            # Print exception
            print(f"Error loading matrix {m.name}: {e}")
