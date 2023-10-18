from abc import ABC, abstractmethod

import numpy as np


class DataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        raise NotImplementedError("data_perturbation not implemented!")

    def perturb_dataset(self, data):
        """
        Dataset perturbation...

        Parameters
        ----------
        data: input dataset

        Returns
        -------
        data_pert: perturbed version of dataset
        """
        if len(data.shape) != 2:
            raise TypeError("Input data has not the right format")

        n_samples = data.shape[0]
        data_prt = np.zeros(shape=data.shape)
        for i in range(n_samples):
            data_prt[i, :] = self.data_perturbation(data[i, :])
        return data_prt
