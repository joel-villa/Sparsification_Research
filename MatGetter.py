"""
A file for interfacing with the suite sparse matrix collection

To use the ssgetpy library: 
pip install ssgetpy
"""

import ssgetpy

# Matrix must be at least 17,755 by 17,755 to have valid s-values for sparse-alg
mats = ssgetpy.search(rowbounds=(17755,100000), isspd=True, limit=2)

def get_mats():
	"""
	A getter
	TODO: convert to Class
	"""
	return mats
