# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from splitters import split_data
from loaders import DataLoaderMNIST
from data_perturb import DataPerturbRandom, DataPerturbGaussian

from classifiers import NMC
from sklearn.svm import SVC


def plot_ten_digits(x, y=None, shape=(28, 28)):
    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i, :].reshape(shape[0], shape[1]), cmap='gray')
        if y is not None:
            plt.title('Label: ' + str(y[i]))


filename = "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv"
# filename = "sample_data/mnist_test.csv"
# data = pd.read_csv(filename)
#  data = np.array(data)  # cast pandas dataframe to numpy array

data_loader = DataLoaderMNIST(
    filename=filename, n_samples=100, normalize=False)
x, y = data_loader.load_data()

# xtr, ytr, xts, yts = split_data(x, y, fraction_tr=0.5)

img = x[0, :]
plt.figure()
plt.imshow(img.reshape(28, 28), cmap="gray")
plt.show()

# prt = DataPerturbRandom(K=200)
prt = DataPerturbGaussian(sigma=100)
xp = prt.data_perturbation(img)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.imshow(xp.reshape(28, 28))
plt.show()

plt.figure()
plot_ten_digits(x, y)
plt.show()

xp = prt.perturb_dataset(x)
plt.figure()
plot_ten_digits(xp, y)
plt.show()
