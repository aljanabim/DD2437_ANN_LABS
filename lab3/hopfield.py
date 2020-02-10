import numpy as np

class HopfieldNet:
    def __init__(self):
        self.w = None

    def fit(self, patterns):
        n_patterns = patterns.shape[0]
        n_inputs = patterns.shape[1]

        self.w = np.zeros((n_inputs, n_inputs))
        for pattern in patterns:
            pattern = np.reshape(pattern, (-1, 1))
            self.w += (pattern @ pattern.T)/n_patterns

    def predict(self, pattern):
        x = np.sign((self.w @ pattern))
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


if __name__ == '__main__':
    test_basic_hopfield_net()
