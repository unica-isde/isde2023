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

data_loader = DataLoaderMNIST(
    filename=filename, n_samples=2000, normalize=False)
x, y = data_loader.load_data()

img = x[0, :]
plt.figure()
plt.imshow(img.reshape(28, 28), cmap="gray")
plt.show()

# prt = DataPerturbRandom(K=200)
prt = DataPerturbGaussian(sigma=1)
xp = prt.data_perturbation(img)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.imshow(xp.reshape(28, 28))
plt.show()

K = [0, 10, 20, 50, 100, 200, 500]
sigma = [10, 20, 200, 200, 500]

acc_K = np.zeros(shape=len(K))
acc_sigma = np.zeros(shape=len(sigma))

xtr, ytr, xts, yts = split_data(x, y, fraction_tr=0.6)
clf = NMC()
clf.fit(xtr, ytr)

rnd = DataPerturbRandom()
gsn = DataPerturbGaussian()

plt.figure()
plot_ten_digits(xts, yts)
plt.show()

for i, k in enumerate(K):
    rnd.K = k
    xts_pert = rnd.perturb_dataset(xts)
    ypred = clf.predict(xts_pert)
    acc_K[i] = np.mean(yts == ypred)

    if i == 2:
        plt.figure()
        plot_ten_digits(xts_pert, yts)
        plt.show()

for i, s in enumerate(sigma):
    gsn.sigma = s
    xts_pert = gsn.perturb_dataset(xts)
    ypred = clf.predict(xts_pert)
    acc_sigma[i] = np.mean(yts == ypred)
    if i == 2:
        plt.figure()
        plot_ten_digits(xts_pert, yts)
        plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(K, acc_K)
plt.xlabel("K")
plt.ylabel("Acc")
plt.subplot(1, 2, 2)
plt.plot(sigma, acc_sigma)
plt.xlabel("sigma")
plt.ylabel("Acc")
plt.show()