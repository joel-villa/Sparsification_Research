"""
A class for saving and loading in files 
"""
import numpy as np


class Loader:
    def __init__(self):
        pass

    def load(filename, names):
        """
        Load in the data from the specified file and return it
        If none exists, return None

        filename - file to read from
        names    - list of the names of the attributes to get

        RETURN: a dictionary (key, value) = ("name", data) 
                where "name" comes dirrectly from names
        """

        path = "./data/" + filename + ".npz"
        
        if not os.path.exists(path):
        # No such path, return None
            return None
        
        # Load the data
        loaded_data = np.load(path, allow_pickle=True)a

        # the dictionary to return
        data_dict = {}

        for name in names:
            data_dict[name] = loaded_data[name]

        return data_dict


    def save(filename, data):
        """
        Save the data to the specified filename

        filename - where to save the data 
        data     - a python dictionary (key, value) = ("name", data)
        """
        np.savez(
            "./data/" + filename + ".npz",
            **data # Note: this is called dictionary unpacking
        )
