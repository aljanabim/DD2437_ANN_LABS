import numpy as np
from matplotlib import pyplot as plt

from generate_data import generate_data
plt.style.use('ggplot')


NUMBER_OF_LAYERS = 1
NUMPER_OF_PERCEPTRONS = 2


class Perceptron():
    def __init__(self, learning_method="perceptron", learning_rate=0.001, n_epochs=20, n_data=None):
        # Learning parameters
        self.learning_method = learning_method
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # Model variables
        self.weights = None
        self.n_inputs = None
        self.n_outputs = None
        self.n_data = n_data
        # Bookkeeping
        self.squared_errors = None
        self.n_errors = None


class Network():
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self._init_network()
        print(self.v)

    def _init_network(self):
        self.x = np.zeros(self.n_inputs)
        self.h = np.zeros(self.n_hidden)
        self.y = np.zeros(self.n_outputs)
        self.v = np.zeros((self.n_hidden, self.n_inputs))
        self.w = np.zeros((self.n_outputs, self.n_hidden))
        self.activation = np.tanh()

    def _activation(self, x):
        return np.tanh(x)

    def _activation_deriv(self, x):
        return 1-np.tanh(x)**2

    def _forward_pass(self):
        pass

    def _backward_pass(self):
        pass

    def _weight_updating(self):
        pass


N = 100
network = Network(2, 5, 1)

data = generate_data(N, plot=False, meanA=[1, 0.5], meanB=[
                     0, -1], sigmaA=0.2, sigmaB=0.5)
