import numpy as np


class HopfieldNetwork():
    def __init__(self):
        self.w = 0
        self._sign = np.vectorize(self._sign)

    def fit(self, data):
        '''
        Expects data as rows of patters
        '''
        self.data = data
        self.n_data = self.data.shape[0]
        self.n_patterns = self.data.shape[1]

        if self.n_patterns <= 1:
            self.data = np.array([data])
            self.n_data = self.data.shape[0]
            self.n_patterns = self.data.shape[1]

        for pattern_index in range(self.n_patterns):
            self.w += np.dot(data.T, data)/self.n_data
            np.fill_diagonal(self.w, 0)

    def predict(self, pattern, method="batch"):
        self.pattern = pattern
        self.c_pattern = pattern.copy()
        tried = False

        is_single_dim_pattern = len(self.pattern.shape) < 2
        if is_single_dim_pattern:
            # self.c_pattern = np.ones(len(self.pattern))*-999
            self.n_data = len(self.pattern)
        else:
            # self.c_pattern = np.ones(
                # (self.pattern.shape[0], self.pattern.shape[1]))*-999
            self.n_data = self.pattern.shape[1]
        self.n_iter = 0
        self.iterations = np.array([self.n_iter])

        while (self.pattern != self.c_pattern).any() or not tried:
            self.n_iter += 1
            self.iterations = np.append(self.iterations, self.n_iter)

            if method == "batch":
                self.c_pattern = self._sign(np.dot(self.c_pattern, self.w))

            if method == "sequential":
                self.order = np.arange(self.n_data)
                np.random.shuffle(self.order)
                for i in self.order:
                    if is_single_dim_pattern:
                        self.c_pattern[i] = self._sign(
                            np.dot(self.w[i, :], self.c_pattern))
                    else:
                        for row in range(self.c_pattern.shape[0]):
                            self.c_pattern[row, i] = self._sign(
                                np.dot(self.w[i, :], self.c_pattern[row, :]))
            tried = True

        return self.iterations, self.c_pattern

    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1
