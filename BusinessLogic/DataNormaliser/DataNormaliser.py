import numpy as np


class DataNormaliser:
    def __init__(self, data):
        self.data = data

    def log_scale_percentage_normalisation(self):
        return self.percentage_normalisation().apply(np.log1p)

    def percentage_normalisation(self):
        return self.data.apply(lambda col: col / (col.sum() or 1))
