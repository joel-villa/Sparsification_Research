# import matplotlib.pyplot as plt
from SSGetter import SSGetter
from Tester import Tester
from Loader import Loader
from PlotGenerator import PlotGenerator
import numpy as np

NUM_ITER = 10
NUM_SS = 50
MAX_S = 5

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


if __name__ == '__main__':
    tester = Tester(num_iter=NUM_ITER, 
                    num_ss=NUM_SS, 
                    max_s=MAX_S)
    plt = PlotGenerator()

    loader = Loader()

    # big_ssgetter = SSGetter(True, row_bounds=(17755,100000))
    small_ssgetter = SSGetter(True, row_bounds=(100,10000))

    # big_mats = big_ssgetter.get_next(5)
    small_mats = small_ssgetter.get_next(5)

    for name, A in small_mats.items():

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


        plt.plot(name=name)
