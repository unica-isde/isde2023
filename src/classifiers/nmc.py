import numpy as np


class NMC:
    """
    Nearest Mean Centroid (NMC) Classifier
    ...
    """

    def __init__(self):
        """This is the class constructor"""
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    # @centroids.setter
    # def centroids(self, value):
    #    self._centroids = value

    def fit(self, xtr, ytr):

        if not isinstance(xtr, np.ndarray):
            raise TypeError("inputs should be ndarrays")

        if xtr.shape[0] != ytr.size:
            raise ValueError("input sizes are not consistent")

        n_classes = np.unique(ytr).size
        n_features = xtr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(0, n_classes):
            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
        return self

    def predict(self, xts):
        n_samples = xts.shape[0]
        n_classes = self._centroids.shape[0]
        dist = np.zeros(shape=(n_samples, n_classes))
        for k in range(0, n_classes):
            dist[:, k] = np.sum((xts - self._centroids[k, :]) ** 2,
                                axis=1)  # brdcast
        ypred = np.argmin(dist, axis=1)
        return ypred
