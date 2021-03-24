from pandas import DataFrame
import numpy as np

np.random.seed(4242)


class LSHBias:
    def __init__(self, feature_dim: int, bits: int):
        self.feature_dim = feature_dim
        self.bits = bits
        self.create_hyperplanes()

    def create_hyperplanes(self):
        # W = (D+1)xK matrix where k^th column is the normal vector for kth hyperplane
        # D+1 because of bias term and considering all features are 0-1 scaled
        W = np.zeros(shape=(self.feature_dim + 1, self.bits), dtype=np.float)
        for k in range(self.bits):
            for i in range(self.feature_dim):
                W[i, k] = np.random.uniform(-1, 1)
            W[self.feature_dim, k] = np.random.uniform(0, 1)
        self.W = W

    def hash_many(self, X):
        # add 1 for the bias term
        ones = np.ones(shape=(X.shape[0], 1))
        X = np.concatenate((X, ones), axis=1)
        mul = np.matmul(self.W.transpose(), X.transpose()).transpose()
        signs = np.sign(mul)

        # convert to 0,1 instead of +1,-1
        def func(x):
            if x == -1:
                return 0
            else:
                return 1
        _01 = np.vectorize(func)(signs)
        return _01

    def hash_to_categories(self, _01):
        out = []
        for hashed_value in _01:
            out.append([''.join(str(e) for e in hashed_value)])
        return np.array(out)
