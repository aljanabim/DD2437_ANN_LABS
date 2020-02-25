# MV
import numpy as np


class HopfieldNetwork():
    def __init__(self, max_iter=150, bias=0, sparse=False):
        self.w = 0
        self._sign = np.vectorize(self._sign)
        self.max_iter = max_iter
        self.bias = bias
        self.sparse = sparse

    def fit(self, data, zero_diag=False, random_weights=False, random_symmetric=False, sparsity=0):
        '''
        Expects data as rows of patters
        '''
        if len(data.shape) < 2:
            data = np.array([data])

        n_patterns = data.shape[0]
        n_data = data.shape[1]

        if random_weights:
            weights = np.random.normal(size=(n_data, n_data))
            if random_symmetric:
                self.w = 0.5*(weights+weights.T)/n_data
            else:
                self.w = weights/n_data
        else:
            for pattern in data:
                pattern = np.array([pattern])
                if self.sparse:
                    self.w += np.dot(pattern.T-sparsity,
                                     pattern-sparsity)/n_data
                else:
                    self.w += np.dot(pattern.T, pattern)/n_data

            if zero_diag:
                np.fill_diagonal(self.w, 0)

    def predict(self, pattern, method="batch", calc_energy=False):
        c_pattern = pattern.copy()
        self.energy = np.array([[0, 0]])
        tried = False

        is_single_dim_pattern = len(pattern.shape) < 2
        if is_single_dim_pattern:
            self.n_data = len(pattern)
        else:
            self.n_data = pattern.shape[1]
        n_iter = 0
        energy_stable = False
        while ((pattern != c_pattern).any() and n_iter < self.max_iter) or not tried:
            n_iter += 1

            if method == "batch":
                if self.sparse:
                    c_pattern = 0.5+0.5 * \
                        self._sign(np.dot(c_pattern, self.w)-self.bias)
                else:
                    c_pattern = self._sign(np.dot(c_pattern, self.w))
                if calc_energy:
                    self.get_energy(c_pattern, n_iter)

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
                    if calc_energy:
                        self.get_energy(c_pattern, n_iter)
                        # if round(np.sum(self.energy[-50:-1, 1])/self.energy[-50:-1].shape[0]) == round(self.energy[-1, 1]):
                        #     energy_stable = True
                        #     break
            if energy_stable:
                break
            tried = True
            if n_iter == self.max_iter:
                # print("Exceeded max number of iterations", self.max_iter)
                pass
        return c_pattern
        """
        The weight matrices where still normailized by the numer of nodes in the network
        """

    def get_energy(self, c_pattern, n_iter):
        c_pattern = np.array([c_pattern])
        E = -np.dot(c_pattern, np.dot(self.w, c_pattern.T))
        self.energy = np.vstack((self.energy, np.array([n_iter, E])))

    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1
