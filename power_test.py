from power_method.power_method import pow_method
from time import perf_counter
from scipy.sparse.linalg import eigs
from SSGetter import get_mats
from main import norm_of_diff
from main import load_A
from numpy import mean

mat_dict = get_mats()

tot_time_eigs = []
tot_time_pow = []

for name, mat in mat_dict.items():
    """
    Calculate top eigenvector with scipy.sparse.linalg.eigs and with python 
    implementation of the power method: time performance, compare estimates
    """
    print(f"{name}")
    # time eigs
    start_time_eigs = perf_counter()
    _, e = eigs(mat, k=1) #k = 1 -> only get top eigenvector
    end_time_eigs = perf_counter()

    # time power method
    start_time_pow = perf_counter()
    _, e_pow = pow_method(mat)
    end_time_pow = perf_counter()

    # Elapsed times
    tot_time_eigs.append(end_time_eigs - start_time_eigs)
    tot_time_pow.append(end_time_pow - start_time_pow)

    print(f"{name}: norm_of_diff(e, e_pow) = {norm_of_diff(e, e_pow, 1)}")

print(f"average time eigs: {mean(tot_time_eigs)}")
print(f"average time eigs: {mean(tot_time_pow)}")

