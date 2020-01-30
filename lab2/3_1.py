# %%
import numpy as np
from matplotlib import pyplot as plt

VAR = 0.1


class RBFNetwork():
    def __init__(self, n_inputs, n_rbf, n_outputs, learning_rate=0.1, min_val=0.05, max_val=2*np.pi):
        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.rbf_centers = np.linspace(min_val, max_val, n_rbf)
        self.w = np.array([np.random.normal(0, 1, n_rbf)])
        self.RBF = np.vectorize(self._base_func)

    def _base_func(self, x, center):
        return np.exp(-np.linalg.norm(x-center)**2/(2*VAR**2))

    def fit(self, data, f, method='batch'):
        self.data = np.array([data]).T
        if method == 'batch':
            phi = self.RBF(self.data, self.rbf_centers)
            w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)

            print(w)

    def predict(self, x):
        return np.dot(self.w, self.RBF(x, self.rbf_centers))


def sin2(x):
    return np.sin(2*x)


def square(x):
    if np.sin(x) >= 0:
        return 1
    else:
        return -1


def generate_input(start):
    patterns = np.arange(start, 2*np.pi, 0.1)
    np.random.shuffle(patterns)
    return patterns


def gen_sin_data():
    x_train = generate_input(0)
    x_test = generate_input(0.05)
    sin2_train = [sin2(val) for val in generate_input(0)]
    sin2_test = [sin2(val) for val in generate_input(0.05)]
    return sin2_train, sin2_test

def gen_square_data():
    square_train =  [square(val) for val in generate_input(0)]
    square_test =  [square(val) for val in generate_input(0.05)]
    return square_train, square_test


def plot_prediction():
    network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)
    sin2_train, sin2_test = gen_sin_data()
    square_train, square_test = gen_square()
    network.fit(sin2_train, sin2_test)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y[i] = network.predict(x_i)

    plt.plot(x, y)
    plt.show()


plot_prediction()


x_train = generate_input(0)
x_test = generate_input(0.05)
sin2_train = list(map(sin2, generate_input(0)))
sin2_test = list(map(sin2, generate_input(0.05)))

square_train = list(map(square, generate_input(0)))
square_test = list(map(square, generate_input(0.05)))

network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)
network.fit(sin2_train, sin2_train)
network.predict([0.5])
# network.RBF(0.5, 0.45)
# print(network.w)
