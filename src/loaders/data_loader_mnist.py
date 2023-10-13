from loaders import DataLoader
import pandas as pd
import numpy as np


class DataLoaderMNIST(DataLoader):

    def __init__(self, filename, n_samples=None):
        self.filename = filename
        self.n_samples = n_samples

    def load_data(self):
        # loads data from a CSV file hosted in our repository
        data = pd.read_csv(self.filename)
        data = np.array(data)  # cast pandas dataframe to numpy array
        if self.n_samples is not None:
            data = data[:self.n_samples, :]
        y = data[:, 0]
        x = data[:, 1:] / 255
        return x, y
