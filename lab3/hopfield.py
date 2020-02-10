import numpy as np

class HopfieldNet:
    def __init__(self, zero_diagonal=False):
        self.w = None
        self.zero_diagonal = zero_diagonal

    def fit(self, patterns):
        n_patterns = patterns.shape[0]
        n_inputs = patterns.shape[1]

        self.w = np.zeros((n_inputs, n_inputs))
        for pattern in patterns:
            pattern = np.reshape(pattern, (-1, 1))
            self.w += (pattern @ pattern.T)/n_patterns

        if self.zero_diagonal:
            np.fill_diagonal(self.w, 0)

    def predict(self, pattern):
        x = np.sign(pattern @ self.w)
        return x


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
