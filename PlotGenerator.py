"""
A class for loading the plots utilizing Loader class
"""
from Loader import Loader
import matplotlib.pyplot as plt

class PlotGenerator:
    def plot(self, name, download=False):
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

        if download:
            plt.savefig("plots/" + name  + ".svg")

        plt.show()
    
    def plot_residuals(self, name, n, nnz, ss, residuals, avg_res=None, download=False):
        """
        Generate the plot for the residuals of a matrix with the provided name

        name      - name of the matrix 
        n         - number of columns & rows in A
        nnz       - number of nonzeroes in A
        ss        - s values used
        residuals - residuals from sparse matrix vector multiplication, 2d array
                    rows are the sparsification behavior of some random x 
                    vector
        avg_res   - the average residuals
        download  - download the figure? 
        """
        # Residual plotting
        for x_res, i in zip(residuals, range(len(residuals))):
            lbl = rf"$x_{{{i}}}$"
            plt.plot(ss, x_res)
            # plt.plot(ss, x_res, label=lbl)

        # Average Case
        plt.plot(ss, avg_res, label="average_behavior")

        # Labels
        title = f"SpMV Residual Behavior of {name} (n = {n} , nnz = {nnz})"
        x_label = "s"
        y_label = r"$\frac{||Ax - \tilde{A} x||_2 } {||Ax||_2} $"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Legend
        plt.legend(loc='best')

        if download:
            plt.savefig(f"plots/{name}.svg")

        plt.show()
    
if __name__ == '__main__':
    generator = PlotGenerator()
    generator.plot("bmw7st_1"  , True)

