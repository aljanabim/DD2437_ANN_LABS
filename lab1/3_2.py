# %%
import numpy as np
from matplotlib import pyplot as plt

from generate_data import generate_data
plt.style.use('ggplot')

np.random.seed(17)

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


class NeuralNetwork():
    def __init__(self, method='batch', learning_rate=0.001, n_inputs=2, n_hidden=5, n_outputs=1):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.method = method
        self.learning_rate = learning_rate
        self._init_network()

    def _init_network(self):
        self.x = np.zeros(self.n_inputs)
        self.h = np.zeros(self.n_hidden)
        self.y = np.zeros(self.n_outputs)
        cov_v = np.array([[1, 0],
                          [0, 1]])
        mean_v = np.array([0, 0])
        self.v = np.random.multivariate_normal(
            mean_v, cov_v, self.n_hidden)
        cov_w = np.eye(self.n_hidden)
        mean_w = np.zeros(self.n_hidden)
        self.w = np.random.multivariate_normal(
            mean_w, cov_w, self.n_outputs)

    def _activation(self, x):
        return np.tanh(x)

    def _activation_deriv(self, x):
        return 1-np.tanh(x)**2

    def _forward_pass(self):
        self.h = self._activation(np.dot(self.v, self.data.T))
        self.y = self._activation(np.dot(self.w, self.h))

    def _backward_pass(self, labels):
        # calc delta_k
        y_in = np.dot(self.w, self.h)
        self.delta_k = (labels - self.y) * self._activation_deriv(y_in)
        # calc delta_j
        h_in = np.dot(self.v, self.data.T)
        self.delta_j = np.zeros((self.n_hidden, self.n_data))

        delta_k_w = np.dot(np.array(self.delta_k).T, self.w)
        h_in_deriv = self._activation_deriv((h_in))
        self.delta_j = np.multiply(delta_k_w.T, h_in_deriv)

        # Stupid way using for loops
        # for j in range(self.n_data):
        #     h_in_deriv = self._activation_deriv((h_in[:, j]))
        #     self.delta_j[:, j] = np.multiply(
        #         np.dot(self.delta_k[:, j], self.w), h_in_deriv)

    def _weight_updating(self, index):
        self.w = self.w + self.learning_rate * \
            self.delta_k[:, index] * self.h[:, index]
        self.v = self.v + self.learning_rate * \
            np.dot(np.array([self.delta_j[:, index]]).T,
                   np.array([self.data[index, :]]))

        # print(self.v)
        # self.w = self.w + self.learning_rate * np.dot()

    def fit(self, data, labels, n_epochs):
        '''
        Expects data in the form
        data = [[x1,y1],
                [x2,y2],
                [x3,y3],
                ...
                [xN,yN]]
        '''
        self.labels = labels
        self.n_data = len(labels)
        self.data = data
        self.mse = np.zeros(n_epochs+1)
        self.miss_ratio = np.zeros(n_epochs+1)
        # self._forward_pass()
        # self._backward_pass(labels)
        for i in range(n_epochs):
            if self.method == 'sequential':
                for index in range(self.n_data):
                    self._forward_pass()
                    self._backward_pass(labels)
                    self._weight_updating(index)

            if self.method == 'batch':
                self._forward_pass()
                self._backward_pass(labels)
                for index in range(self.n_data):
                    self._weight_updating(index)
            self.result = list(map(classifier, self.y[0, :]))
            self.mse[i] = np.sum(
                np.square(self.labels-self.y[0, :]))/self.n_data
            self.miss_ratio[i] = 1 - \
                np.count_nonzero(self.result == self.labels)/self.n_data

    def predict(self):
        self._forward_pass()

    def metrics(self):
        self.predict()
        self.mse[-1] = np.sum(
            np.square(self.labels-self.y[0, :]))/self.n_data
        self.miss_ratio[-1] = 1 - \
            np.count_nonzero(self.result == self.labels)/self.n_data
        print(self.mse, self.miss_ratio)

        # for i in range(self.n_data):
        #     print(labels[i] == self.result[i])


def classifier(val):
    if val >= 0:
        return 1
    else:
        return -1


N = 100
network = NeuralNetwork(method='batch', n_inputs=2,
                        n_hidden=3, n_outputs=1)

data = generate_data(N, plot=True, meanA=[1, 0.5], meanB=[
    0, -1], sigmaA=0.2, sigmaB=0.3)

network.fit(data[:, 0:2], data[:, 2], n_epochs=3)
network.metrics()
