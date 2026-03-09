"""
For associating data with it's weight
i.e. the number 0.2 was generated from 26 test runs
"""

class Data_Weight:
    def __init__(self, data=0, weight=0):
        """
        """
        self.data=data
        self.weight=weight

    def update_weight(self, new_d, new_w):
        """
        Update the weight of this data with some new data and its weight
        (taking the weighted average)
        """
        total_weight = self.weight + new_w

        # Weighted average
        self.data = (self.weight * self.data + new_w * new_d) / total_weight
        self.weight = total_weight

    def get(self):
        """
        A getter
        """
        return self.data, self.weight
    