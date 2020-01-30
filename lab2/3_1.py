# %%
import numpy as np
from matplotlib import pyplot as plt

VAR = 0.1

np.random.seed(3)

class RBFNetwork():
    def __init__(self, n_inputs, n_rbf, n_outputs, learning_rate=0.1, min_val=0.05, max_val=2*np.pi):
        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.rbf_centers = np.array([np.linspace(min_val, max_val, n_rbf)])
        self.w = np.array([np.random.normal(0, 1, n_rbf)])
        self.RBF = np.vectorize(self._base_func)

    def _base_func(self, x, center):
        return np.exp(-np.linalg.norm(x-center)**2/(2*VAR**2))

    def fit(self, data, f, method='batch'):
        self.data = np.array([data]).T
        if method == 'batch':
            phi = self.RBF(self.data, self.rbf_centers)
            self.w = np.dot(
                np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)

            # print(w)

    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T)


def plot_prediction():
    network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y[i] = network.predict(x_i)
        return np.dot(self.w, self.RBF(x, self.rbf_centers))


def sin2(x):
    return np.sin(2*x)


def square(x):
    if np.sin(x) >= 0:
        return 1
    else:
        return -1


def generate_input(start, n_points=10):
    patterns = np.linspace(start, 2*np.pi, n_points)
    np.random.shuffle(patterns)
    return patterns


def gen_func_data(n_train, n_test, func):
    patterns = np.linspace(0, 2*np.pi, n_train+n_test)
    targets = np.array([func(x) for x in patterns])

    data = np.column_stack((patterns, targets))
    np.random.shuffle(data)
    train_data = data[:n_train]
    test_data = data[n_train:]

    train_patterns = train_data[:,0]
    train_targets = train_data[:,1]

    test_patterns = test_data[:,0]
    test_targets = test_data[:,1]

    return train_patterns, train_targets, test_patterns, test_targets


def plot_prediction():
    n_train = 64
    n_test = 63

    network = RBFNetwork(n_inputs=1, n_rbf=100, n_outputs=1)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(n_train, n_test, sin2)

    network.fit(train_patterns, train_targets)
    train_preds = network.predict(train_patterns)
    plt.plot(train_patterns, train_preds, 'o')
    plt.show()


plot_prediction()
