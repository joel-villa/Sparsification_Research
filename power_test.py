from power_method.power_method import pow_method
from scipy.sparse.linalg import eigs
from MatGetter import get_mats
from main import norm_of_diff

mat_dict = get_mats()
for name, mat in mat_dict.items():
    _, e = eigs(mat, k=1) #k = 1 -> only get top eigenvector
    e_pow = pow_method(mat)

    print(f"{name}: norm_of_diff(e, e_pow) = {norm_of_diff(e, e_pow)}")


