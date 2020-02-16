# MV
import numpy as np


class HopfieldNetwork():
    def __init__(self, max_iter=150):
        self.w = 0
        self._sign = np.vectorize(self._sign)
        self.max_iter = max_iter

    def fit(self, data):
        '''
        Expects data as rows of patters
        '''
        if len(data.shape) < 2:
            data = np.array([data])

        n_data = data.shape[0]

        for pattern in data:
            pattern = np.array([pattern])
            self.w += np.dot(pattern.T, pattern)/n_data
        np.fill_diagonal(self.w, 0)

    def predict(self, pattern, method="batch"):
        c_pattern = pattern.copy()
        tried = False

        is_single_dim_pattern = len(pattern.shape) < 2
        if is_single_dim_pattern:
            self.n_data = len(pattern)
        else:
            self.n_data = pattern.shape[1]
        n_iter = 0
        self.iterations = np.array([n_iter])

        while (pattern != c_pattern).any() and n_iter < self.max_iter or not tried:
            n_iter += 1

            if method == "batch":
                c_pattern = self._sign(np.dot(c_pattern, self.w))

            if method == "sequential":
                order = np.arange(self.n_data)
                np.random.shuffle(order)
                for i in order:
                    if is_single_dim_pattern:
                        c_pattern[i] = np.sign(
                            np.dot(self.w[i, :], c_pattern))
                    else:
                        for row in range(c_pattern.shape[0]):
                            c_pattern[row, i] = self._sign(
                                np.dot(self.w[i, :], c_pattern[row, :]))
            tried = True
            if n_iter == self.max_iter:
                print("Exceeded max number of iterations", self.max_iter)
        return self.iterations, c_pattern

    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1
