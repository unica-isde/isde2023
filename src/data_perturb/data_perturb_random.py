from data_perturb import DataPerturb
import numpy as np


class DataPerturbRandom(DataPerturb):

    def __init__(self, min_value=0, max_value=255, K=100):
        self.min_value = min_value
        self.max_value = max_value
        self.K = K

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
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = int(value)

    def data_perturbation(self, x):
        idx = np.array(list(range(0, x.size)))
        np.random.shuffle(idx)
        idx = idx[:self._K]
        xp = x.copy()
        xp[idx] = np.random.randint(
            low=self._min_value,
            high=self._max_value + 1, size=self._K)
        return xp
