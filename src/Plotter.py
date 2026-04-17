"""
For plotting thangs
"""
import matplotlib.pyplot as plt

COLORS = ["black", 
        "#FF6B6B",  
        "#FFA94DDD", 
        "#EEF132", 
        "#6BCB77", 
        "#54FCD8",
        "#4D96FF",
        "#8C77FF",  
        "#FF7DF0",
        "#F72B2B"]
class Plotter():
    def __init__(self, save_fig, show_fig):
        self.save_fig = save_fig
        self.show_fig = show_fig
        plt.rcParams['figure.figsize'] = (16, 8)
        
    def init_plot(self, title, x_label, y_label, save_name, grid_on=False, desceneding_x=False):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        self.save_path = f"plots/{save_name}.jpg"
        plt.grid(visible=grid_on)

        if (desceneding_x):
            # flipping vertical axis
            ax=plt.gca()
            ax.invert_xaxis()

    def add_to_plot(self, xs, ys, label):
        """
        Add given info to the plot
        """
        plt.plot(xs, ys, label=label)

    def finish(self):
        plt.legend()
        plt.tight_layout()
        
        if self.save_fig:
            plt.savefig(self.save_path)
        if self.show_fig:
            plt.show()

        plt.clf()