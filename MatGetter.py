"""
A file for interfacing with the suite sparse matrix collection

To use the ssgetpy library: 
pip install ssgetpy
"""
import tarfile
from scipy.io import mmread
import ssgetpy
import os

# Matrix must be at least 17,755 by 17,755 to have valid s-values for sparse-alg
mats = ssgetpy.search(rowbounds=(17755,100000), isspd=True, limit=2)


def get_mats():
    matrices = []

    for m in mats:
        m.download()  # Download the matrix
        
        # Get the local path of the .tar.gz file
        path_tuple = m.localpath()  # This returns a tuple (two identical paths)
        path = path_tuple[0]  # Use the first path (both are the same)

        # Print the path for debugging
        print(f"Local path of matrix {m.name}: {path}")
        
        try:
            # Extract the .tar.gz file to get the matrix file inside
            if path.endswith('.tar.gz'):
                with tarfile.open(path, 'r:gz') as tar:
                    # Extract files in the same folder as the .tar.gz file
                    extract_dir = os.path.dirname(path)
                    extract_dir = os.path.join(extract_dir, m.name)
                    tar.extractall(path=extract_dir)  # Extract to the same folder
                
                # After extraction, find the matrix file (.mtx) in the extracted files
                matrix_file = None
                for extracted_file in os.listdir(extract_dir):
                    if extracted_file.endswith('.mtx'):
                        matrix_file = extracted_file
                        break
                
                if matrix_file is None:
                    raise FileNotFoundError(f"No .mtx file found in {extract_dir}")
                
                # Full path to the extracted matrix file
                extracted_path = os.path.join(extract_dir, matrix_file)

                # Print extracted matrix path for debugging
                print(f"Extracted matrix path: {extracted_path}")

                # Now, read the extracted matrix file using mmread
                A = mmread(extracted_path).tocsr()
                matrices.append(A)
            else:
                print(f"Matrix {m.name} is not in .tar.gz format.")
        
        except Exception as e:
            print(f"Error loading matrix {m.name}: {e}")

    return matrices

# def get_mats():
#     """
#     A getter
#     TODO: convert to Class
#     """
#     As = []
#     for m in mats:
#         m.download()  # downloads the matrix

#         # path = m.localpath  # path to the downloaded .mtx file
#         A = m.load().tocsr()

#         As.append(A)
#         mmread(m).tocsr()
#     return As
