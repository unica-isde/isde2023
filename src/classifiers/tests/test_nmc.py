import unittest
import numpy as np
from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        self.x0 = np.zeros(shape=(100, 10))
        self.y0 = np.zeros(shape=(50,))
        self.clf = NMC()

    def test_fit(self):
        self.assertRaises(TypeError, self.clf.fit, xtr=None, ytr=None)
        self.assertRaises(ValueError, self.clf.fit,
                          xtr=np.zeros(shape=(100, 10)),
                          ytr=np.zeros(shape=(50,)))
        # self.clf.fit(xtr = self.x, ytr = self.y)