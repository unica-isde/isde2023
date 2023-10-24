# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from splitters import split_data
from loaders import DataLoaderMNIST
from data_perturb import DataPerturbRandom, DataPerturbGaussian


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

kernel_size = 11
mask = np.ones(shape=(kernel_size,))
mask /= np.sum(mask)
print(mask)

# kernel is the function computing the convolution (in the abstract class)
img_conv = np.zeros(shape=img.shape)
offset = (kernel_size - 1) // 2
num_iter = img.size - kernel_size + 1  # excluding elements not involved in the conv
for i in range(num_iter):
    img_conv[offset + i] = np.dot(img[i:i+kernel_size], mask)

plt.figure()
plt.imshow(img_conv.reshape(28, 28), cmap="gray")
plt.show()
