# %%
import numpy as np
from matplotlib import pyplot as plt

VAR = 1

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

    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T)


def plot_prediction():
    network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.zeros(x.shape)
    y_target = list(map(sin2, x))
    for i, x_i in enumerate(x):
        y[i] = network.predict(x_i)


def sin2(x):
    return np.sin(2*x)


def square(x):
    if np.sin(x) >= 0:
        return 1
    else:
        return -1


<<<<<<< HEAD
def generate_input(start, n_points=10):
    patterns = np.linspace(start, 2*np.pi, n_points)
    np.random.shuffle(patterns)
    return patterns


def gen_func_data(n_train, n_test, func):
    patterns = np.linspace(0, 2*np.pi, n_train+n_test)
    targets = np.array([func(x) for x in patterns])
=======
def generate_input(start, end=2*np.pi):
    patterns = np.arange(start, end, 0.1)
    # np.random.shuffle(patterns)
    return patterns


def gen_sin_data():
    sin2_train = [sin2(val) for val in generate_input(0)]
    sin2_test = [sin2(val) for val in generate_input(0.05)]
    return sin2_train, sin2_test
>>>>>>> ac6807c0d69b524f2a726cf5dcbb94a02a3698ae

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
<<<<<<< HEAD
    n_train = 64
    n_test = 63

    network = RBFNetwork(n_inputs=1, n_rbf=100, n_outputs=1)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(n_train, n_test, sin2)

    network.fit(train_patterns, train_targets)
    train_preds = network.predict(train_patterns)
    plt.plot(train_patterns, train_preds, 'o')
    plt.show()


plot_prediction()
<<<<<<< HEAD
=======
=======
    network = RBFNetwork(n_inputs=1, n_rbf=63, n_outputs=1)
    sin2_train, sin2_test = gen_sin_data()
    square_train, square_test = gen_square_data()
    network.fit(square_train, square_train)

    x = np.linspace(0, 2*np.pi, 100)
    y = list(map(network.predict, x))
    sin2_target = list(map(sin2, x))
    square_target = list(map(square, x))

    plt.plot(x, y, label='Fit')
    plt.plot(x, sin2_target, label='Target')
    plt.legend()
    plt.show()


def plot_error():
    # plot error as function of number of rbfs
    rbf_ = np.arange(0, 63)
>>>>>>> ac6807c0d69b524f2a726cf5dcbb94a02a3698ae


x_train = generate_input(0)
x_test = generate_input(0.05)
sin2_train = list(map(sin2, generate_input(0)))
sin2_test = list(map(sin2, generate_input(0.05)))

square_train = list(map(square, generate_input(0)))
square_test = list(map(square, generate_input(0.05)))

network = RBFNetwork(n_inputs=1, n_rbf=62, n_outputs=1)
network.fit(x_train, sin2_train)
print(network.predict([0.5, 0]))
print(sin2(np.array([0.5, 0])))
# network.RBF(0.5, 0.45)
# print(network.w)
>>>>>>> 9bc39c6292520f816afb9124526702169c017220
