import numpy as np

class HopfieldNet:
    def __init__(self, zero_diagonal=False, min_iter=1, max_iter=5):
        self.w = None
        self.n_inputs = None
        self.zero_diagonal = zero_diagonal
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.sequential_learning_snapshots = None

    def fit(self, patterns):
        patterns = np.array(patterns)
        if len(patterns.shape) == 1:
            patterns = np.reshape(patterns, (1, -1))

        n_patterns = patterns.shape[0]
        self.n_inputs = patterns.shape[1]

        self.w = np.zeros((self.n_inputs, self.n_inputs))

        for pattern in patterns:
            pattern = np.reshape(pattern, (-1, 1))
            self.w += (pattern @ pattern.T)/n_patterns

        if self.zero_diagonal:
            np.fill_diagonal(self.w, 0)

    def predict(self, pattern, method='batch'):
        input_pattern = np.array(pattern)
        current_pattern = input_pattern.copy()
        iter = 0
        self.sequential_learning_snapshots = [] # Used to plot progress

        while (((current_pattern != input_pattern).any() and (iter < self.max_iter))
               or (iter < self.min_iter)):
            if method == 'batch':
                current_pattern = self._batch_update(current_pattern)
            elif method == 'sequential':
                current_pattern = self._sequential_update(current_pattern)
            iter += 1

        return current_pattern

    def _batch_update(self, pattern):
        return np.sign(pattern @ self.w)

    def _sequential_update(self, pattern):
        current_pattern = pattern.copy()
        node_indexes = np.array(range(0, self.n_inputs))
        np.random.shuffle(node_indexes)
        for i in node_indexes:
            current_pattern[i] = np.sign(self.w[i, :].dot(current_pattern))
            if (i%100) == 0:
                self.sequential_learning_snapshots.append(current_pattern.copy())
        return current_pattern


def test_basic_hopfield_net():
    patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
                         [-1, -1, -1, -1, -1,  1, -1, -1],
                         [-1,  1,  1, -1, -1,  1, -1,  1]])

    net = HopfieldNet()
    net.fit(patterns)

    assert (patterns[0] == net.predict(patterns[0])).all()
    assert (patterns[1] == net.predict(patterns[1])).all()
    assert (patterns[2] == net.predict(patterns[2])).all()
    assert (patterns == net.predict(patterns)).all()



if __name__ == '__main__':
    test_basic_hopfield_net()
