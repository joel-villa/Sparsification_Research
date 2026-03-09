"""
A class for saving and loading in files 
"""
import numpy as np
import os

DICT_KEYS = ["ss", "diffs", "num_runs", "mat_size"]

class Loader:
    def __init__(self):
        pass

    def load(self, filename, names=DICT_KEYS):
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
        loaded_data = np.load(path)

        # the dictionary to return
        data_dict = {}

        for name in names:
            data_dict[name] = loaded_data[name]

        return data_dict


    def save(self, filename, data):
        """
        Save the data to the specified filename

        filename - where to save the data 
        data     - a python dictionary (key, value) = ("name", data)
        """
        np.savez(
            "./data/" + filename + ".npz",
            **data # Note: this is called dictionary unpacking
        )

    def get_keys(self):
        """
        Return default data dictionary keys
        """
        return DICT_KEYS
    
    def default_dict(self, n, ss):
        """
        Return a default dictionary, with everything initialized to zero, except
        num_rows

        n   - number of rows in the matrix
        ss  - the s values which will be used for testing

        RETURN: 0 dictionary
        """

        data_dict = {}
        data_dict['ss'] = ss
        # print(f"ss.")
        data_dict['diffs'] = np.zeros(shape=np.shape(ss))
        data_dict[DICT_KEYS[2]] = 0
        data_dict[DICT_KEYS[3]] = n

        return data_dict


l = Loader()
dict = l.load("662_bus")
print(dict)