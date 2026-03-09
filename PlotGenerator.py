"""
A class for loading the plots utilizing Loader class
"""
from Loader import Loader
import matplotlib.pyplot as plt

class PlotGenerator:
    def plot(self, name):
        """
        Generate a plot for the given file name
        """

        loader = Loader()
        data_dict = loader.load(filename=name)

        title = "Sparsification Behavior of " + name + f" (n = {data_dict['mat_size']}, number of runs = {data_dict['num_runs']})"

        x_label = "s"
        y_label = rf"$||e - \tilde e||$"

        plt.plot(data_dict['ss'], data_dict['diffs'])
        plt.title(title)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # plt.savefig("plots/" + name  + ".svg")

        plt.show()
    

generator = PlotGenerator()
generator.plot("bcsstk03")