# import matplotlib.pyplot as plt
from .SSGetter import SSGetter
from .old_code.Tester import Tester
from .old_code.Loader import Loader
from .old_code.PlotGenerator import PlotGenerator
from .Plotter import Plotter
from .tests import *
import numpy as np

NUM_ITER = 10
NUM_SS = 32
MAX_S = 1.5

def weighted_avg(x1, x2, w1, w2):
    """
    Calculate the weighted average of the given data

    x1 - first data (can be a numpy array)
    x2 - second data (can be numpy array)
    w1 - weight of first data
    w2 - weight of second data

    RETURN: an array of the new weighted average data
    """
    total_weight = w1 + w2
        
    # Weighted average
    return (w1 * x1+ w2 * x2) / total_weight
def test_one():
    tester = Tester(num_iter=NUM_ITER, 
                    num_ss=NUM_SS, 
                    max_s=MAX_S)
    plt = PlotGenerator()

    loader = Loader()

    ssgetter = SSGetter(True, row_bounds=(150000,200000))
    #ssgetter = SSGetter(True, row_bounds=(100,10000))

    # mats = ssgetter.get_next(2)
    mats = ssgetter.get_by_name(names=["bmw7st_1"])


    for name, A in mats.items():

        #Load in the stored data
        data_dict = loader.load(name)

        # Number of rows & columns (assumed symmetric)
        n, m = A.shape

        ss = tester.get_ss(rows=n, cols=m)

        if (data_dict is  None):
            # Initialize data dict if not already in system
            data_dict = loader.default_dict(n=n, ss=ss)

        ss0, diffs0, num_iter0, n0 = data_dict.values()

        print(name)
        
        if (n != n0 or (not np.array_equal(ss, ss0))):
            #Sanity check
            print(f"Error with {name}\nn = {n} != {n0} = n0, or s arrays unequal")
            break

        # Generate new data
        diffs = tester.test_s_behavior(A)

        # Update data
        diffs = weighted_avg(diffs0, diffs, num_iter0, NUM_ITER)

        data_dict['diffs'] = diffs
        data_dict['num_runs'] = num_iter0 + NUM_ITER

        # Save new data
        loader.save(name, data_dict)


        #plt.plot(name=name)
    print("DONE")

def test_two():
    """
    Checking how accurate the sparser matrix is at approximating Ax
    """
    MATS = ["494_bus"]
    # MATS = ["494_bus", "662_bus", "685_bus", "1138_bus", "bcsstk21", 
    #         "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"]
    NUM_XS = 5

    tester = Tester(num_iter=NUM_ITER, num_ss=NUM_SS, max_s=MAX_S)
    ssgetter = SSGetter()
    mat_dict = ssgetter.get_by_name(MATS)
    plt_gen = PlotGenerator()

    for name, A in mat_dict.items():
        print(f"matrix name: {name}")
        n, m = A.shape

        ss = tester.get_ss(rows=n, cols=m)
        residuals = tester.test(A, ss, NUM_XS)

        # axis=0 computes the mean along the first dimension (down the columns)
        average_res = np.mean(residuals, axis=0) 

        plt_gen.plot_residuals(name = name,
                               n = n,
                               nnz = A.nnz,
                               ss = ss,
                               residuals = residuals,
                               avg_res = average_res,
                               download = True)

def test_md():
    """
    For testing preservation of top eigenvector w/ tests from tests.py
    """
    ss_getter = SSGetter(in_csr=False)
    mats = ss_getter.get_by_name(names=["494_bus", "662_bus", "685_bus"])
    # mats = ss_getter.get_by_name(names=["494_bus"])
    # mats = ss_getter.get_by_name(names=["662_bus"])
    # mats = ss_getter.get_by_name(names=["685_bus"])

    xs = range(0, 100, 5)
    seed = 10
    num_avg = 25

    plotter = Plotter(save_fig=False, show_fig=True)
    plotter.init_plot(title="top eigenvector preservation of Sparsification",
                      x_label="expected number of new zeros",
                      y_label="norm of difference in top eigenvectors",
                      save_name="new_plot")

    # MD Sparsifier test
    print("Starting MD Sparsifier test")
    for name, A in mats.items():
        ys = np.zeros(np.shape(xs))

        for i in range(num_avg):
            seed_i = seed + i 
            ss, ys_i = test_eig_pres_of_md_sparsifier(A, xs, seed=seed_i)

            ys += ys_i

        ys = ys / num_avg

        plotter.add_to_plot(xs, ys, label=f"{name} (nnz = {A.nnz}) (MD)")
    print("Finished MD Sparsifier test")
    
    # # Generic Sparsifier test
    # print("Starting Generic Sparsifier test")
    # for name, A in mats.items():
    #     ys = np.zeros(np.shape(xs))

    #     for i in range(num_avg):
    #         seed_i = seed + i 
    #         ss, ys_i = test_eig_pres_of_sparsifier(A, xs, seed=seed_i)

    #         ys += ys_i

    #     ys = ys / num_avg

    #     plotter.add_to_plot(xs, ys, label=f"{name} (nnz = {A.nnz})")
    # print("Finished Generic Sparsifier test")

    plotter.finish()



if __name__ == '__main__':
    test_md()