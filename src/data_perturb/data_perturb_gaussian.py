from data_perturb import DataPerturb
import numpy as np


class DataPerturbGaussian(DataPerturb):

    def __init__(self, min_value=0, max_value=255, sigma=100):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = int(value)

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        self._max_value = int(value)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_perturbation(self, x):
        xp = x + np.round(self.sigma * np.random.randn(x.ravel().size))
        xp[xp < self.min_value] = self.min_value  # projections on box [0, 255]
        xp[xp > self.max_value] = self.max_value
        return xp
