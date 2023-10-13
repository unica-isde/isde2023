# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from splitters import split_data
from loaders import DataLoaderMNIST

from classifiers import NMC
from sklearn.svm import SVC


def plot_ten_digits(x, y=None, shape=(28, 28)):
    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i, :].reshape(shape[0], shape[1]), cmap='gray')
        if y is not None:
            plt.title('Label: ' + str(y[i]))


def fit(xtr, ytr):
    n_classes = np.unique(ytr).size
    n_features = xtr.shape[1]
    centroids = np.zeros(shape=(n_classes, n_features))
    for k in range(0, n_classes):
        centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
    return centroids


def predict(xts, centroids):
    """Computes ..."""
    n_samples = xts.shape[0]
    n_classes = centroids.shape[0]
    dist = np.zeros(shape=(n_samples, n_classes))
    for k in range(0, n_classes):
        dist[:, k] = np.sum((xts - centroids[k, :]) ** 2, axis=1)  # brdcast
    ypred = np.argmin(dist, axis=1)
    return ypred


filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"
# filename = "sample_data/mnist_test.csv"
# data = pd.read_csv(filename)
#  data = np.array(data)  # cast pandas dataframe to numpy array

data_loader = DataLoaderMNIST(filename=filename, n_samples=10000)
clf = SVC()

x, y = data_loader.load_data()
plot_ten_digits(x, y)
plt.show()

xtr, ytr, xts, yts = split_data(x, y, fraction_tr=0.5)

clf.fit(xtr, ytr)
ypred = clf.predict(xts)
# plot_ten_digits(clf.centroids)
# plt.show()

# compute the test error (fraction of samples that are misclassified)
print("Test error: " + str(np.mean(yts != ypred)))
