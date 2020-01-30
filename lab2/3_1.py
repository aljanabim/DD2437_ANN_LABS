# %%
import numpy as np
from matplotlib import pyplot as plt

VAR = 0.1


class RBFNetwork():
    def __init__(self, n_inputs, n_rbf, n_outputs, min_val=0.05, max_val=2*np.pi):
        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.rbf_centers = np.linspace(min_val, max_val, n_rbf)
        self.w = np.random.normal(0, 1, n_rbf)

    def RBF(self, x, center):
        return np.exp(-np.linalg.norm(x-center)**2/(2*VAR**2))

    def fit(self, data, method='batch'):
        self.data = data
        if method == 'batch':
            phi =


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


sin2_train = list(map(sin2, generate_input(0)))
sin2_test = list(map(sin2, generate_input(0.05)))

square_train = list(map(square, generate_input(0)))
square_test = list(map(square, generate_input(0.05)))

network = RBFNetwork(n_inputs=1, n_rbf=4, n_outputs=1)
network.RBF(0.5, 0.45)
print(network.w)
