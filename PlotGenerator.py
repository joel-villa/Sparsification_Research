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
    
    def plot_residuals(self, name, n, nnz, num_iter, residuals, download=False):
        """
        Generate the plot for the residuals of a matrix with the provided name

        name      - name of the matrix 
        residuals - residuals from sparse matrix vector multiplication
        download  - download the figure? 
        """
        # COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # if num_iter > len(COLORS):
        #     raise ValueError(f"Can only plot {len(COLORS)}, iterations, {num_iter} too large")

        
        title = f"SpMv Residual Behavior of {name} (n = {n} , nnz = {nnz})"
        x_label = "s"
        y_label = r"$\frac{||Ax - \tilde{A} x||_2 } {||Ax||_2} $"

        #TODO: residual plotting

        if download:
            plt.savefig(f"plots/{name}.svg")

        plt.show()
    
if __name__ == '__main__':
    generator = PlotGenerator()
    generator.plot("bmw7st_1"  , True)

