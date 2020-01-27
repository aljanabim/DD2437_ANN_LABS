# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')


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

    def _add_bias(self):
        # To data
        self.data = np.column_stack((self.data, np.ones(self.data.shape[0])))
        # To hidden layers weights
        self.v = np.column_stack((self.v, np.ones(self.n_hidden)))
        # print(self.v)

    def _activation(self, x):
        return np.tanh(x)

    def _activation_deriv(self, x):
        return 1-np.tanh(x)**2

    def _forward_pass(self):
        self.h = self._activation(np.dot(self.v, self.data.T))
        self.y = np.dot(self.w, self.h)

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
        self.misses = np.zeros(n_epochs+1)
        self._add_bias()
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
            self.misses[i] = self.n_data - \
                np.count_nonzero(self.result == self.labels)

    def predict(self, data):
        data = np.column_stack((data, np.ones(data.shape[0])))
        h = self._activation(np.dot(self.v, data.T))
        y = np.dot(self.w, h)
        return y

    def metrics(self):
        self._forward_pass()
        self.mse[-1] = np.sum(
            np.square(self.labels-self.y[0, :]))/self.n_data
        self.misses[-1] = self.n_data - \
            np.count_nonzero(self.result == self.labels)
        return self.mse, self.misses

        # for i in range(self.n_data):
        #     print(labels[i] == self.result[i])


def classifier(val):
    if val >= 0:
        return 1
    else:
        return -1

def surface_plot(xx, yy, zz, title=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    if title:
        plt.title(title)
    plt.show()


def generate_gaussian(n_x, n_y, plot=True):
    x = np.reshape(np.linspace(-5, 5, n_x), (-1, 1))
    y = np.reshape(np.linspace(-5, 5, n_y), (-1, 1))
    zz = np.exp(-x*x*0.1) @ np.exp(-y*y*0.1).T - 0.5
    xx, yy = np.meshgrid(x, y)

    if plot:
        surface_plot(xx, yy, zz, title="Original Gaussian")

    patterns = np.column_stack((np.reshape(xx, (-1, 1)),
                                np.reshape(yy, (-1, 1))))
    targets = np.reshape(zz, (-1))

    return patterns, targets


def mse_vs_n_hidden():
    n_x = 20
    n_y = 20
    n_trials = 10
    frac_train = 0.6

    patterns, targets = generate_gaussian(n_x, n_y, plot=False)

    data = np.column_stack((patterns, targets))
    validation_patterns = data[:, :2].copy()
    validation_targets = data[:, 2].copy()
    np.random.shuffle(data)

    n_train = int(frac_train*len(targets))
    patterns = data[:n_train, :2]
    targets = data[:n_train, 2]

    n_hidden_list = list(range(1, 26, 1))
    mse_avgs = []
    mse_stds = []
    for n_hidden in n_hidden_list:
        mse_list = []
        for trial in range(n_trials):
            net = NeuralNetwork(method='batch', n_inputs=2,
                                n_hidden=n_hidden, n_outputs=1)
            net.fit(patterns, targets, n_epochs=1000)

            preds = net.predict(validation_patterns)
            mse = np.sum(np.square(validation_targets-preds)) / len(validation_targets)
            mse_list.append(mse)
        mse_avgs.append(np.mean(mse_list))
        mse_stds.append(np.std(mse_list))

    mse_sem = mse_stds / np.sqrt(n_trials)

    # Plot mse by neurons
    plt.errorbar(n_hidden_list, mse_avgs, mse_sem, capsize=2, label='Average MSE +- SEM')
    plt.title('MSE as function of neurons in hidden layer')
    plt.xlabel('Number of neurons')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

def mse_vs_frac_train():
    n_x = 20
    n_y = 20
    n_trials = 10
    n_hidden = 14

    patterns, targets = generate_gaussian(n_x, n_y, plot=False)
    n_total_samples = len(targets)

    data = np.column_stack((patterns, targets))
    validation_patterns = data[:, :2].copy()
    validation_targets = data[:, 2].copy()
    np.random.shuffle(data)

    frac_train_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mse_avgs = []
    mse_stds = []
    for frac_train in frac_train_list:
        n_train = int(frac_train*n_total_samples)
        print(n_train)
        np.random.shuffle(data)
        patterns = data[:n_train, :2]
        targets = data[:n_train, 2]

        mse_list = []
        for trial in range(n_trials):
            net = NeuralNetwork(method='batch', n_inputs=2,
                                n_hidden=n_hidden, n_outputs=1)
            net.fit(patterns, targets, n_epochs=1000)
            preds = net.predict(validation_patterns)
            mse = np.sum(np.square(validation_targets-preds)) / len(validation_targets)
            mse_list.append(mse)
        mse_avgs.append(np.mean(mse_list))
        mse_stds.append(np.std(mse_list))
    mse_sem = mse_stds / np.sqrt(n_trials)

    # Plot mse by neurons
    plt.errorbar(frac_train_list, mse_avgs, mse_sem, capsize=2, label='Average MSE +- SEM')
    plt.title('MSE as function of training fraction')
    plt.xlabel('Fraction of samples used for training')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()


def function_approximation():
    n_x = 20
    n_y = 20
    frac_train = 0.7
    n_hidden = 3

    patterns, targets = generate_gaussian(n_x, n_y)

    data = np.column_stack((patterns, targets))
    validation_patterns = data[:, :2].copy()
    validation_targets = data[:, 2].copy()

    np.random.shuffle(data)
    n_train = int(frac_train*len(targets))
    patterns = data[:n_train, :2]
    targets = data[:n_train, 2]

    net = NeuralNetwork(method='batch', n_inputs=2,
                        n_hidden=n_hidden, n_outputs=1)

    net.fit(patterns, targets, n_epochs=10000)

    # Plot surface
    preds = net.predict(validation_patterns)
    xx = np.reshape(validation_patterns[:,0], (n_x, n_y))
    yy = np.reshape(validation_patterns[:,1], (n_x, n_y))
    zz_approx = np.reshape(preds, (n_x, n_y))
    surface_plot(xx, yy, zz_approx, title="Approximated Gaussian - {} hidden neurons".format(n_hidden))


# mse_vs_n_hidden()
# mse_vs_frac_train()
function_approximation()
