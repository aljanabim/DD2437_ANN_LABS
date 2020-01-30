import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


class NeuralNetwork():
    def __init__(self, method='batch', learning_rate=0.01, n_inputs=2, n_hidden=5, n_outputs=1):
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

        cov_v = np.eye(self.n_inputs+1)
        mean_v = np.zeros(self.n_inputs+1)
        self.v = np.random.multivariate_normal(
            mean_v, cov_v, self.n_hidden)
        # print(self.v)

        cov_w = np.eye(self.n_hidden+1)
        mean_w = np.zeros(self.n_hidden+1)
        self.w = np.random.multivariate_normal(
            mean_w, cov_w, self.n_outputs)

    def _add_bias(self):
        self.data = np.column_stack((self.data, np.ones(self.data.shape[0])))

    def _activation(self, x):
        return np.tanh(x)

    def _activation_deriv(self, x):
        return 1-np.tanh(x)**2

    def _forward_pass(self):
        self.h = self._activation(np.dot(self.v, self.data.T))
        self.h = np.row_stack((self.h, np.ones(self.n_data)))  # add bias
        self.y = self._activation(np.dot(self.w, self.h))

    def _backward_pass(self, labels):
        # calc delta_k
        y_in = np.dot(self.w, self.h)
        self.delta_k = (labels.T - self.y) * self._activation_deriv(y_in)
        # print(self.h.shape)
        # calc delta_j
        h_in = np.dot(self.v, self.data.T)
        h_in = np.row_stack((h_in, np.ones(self.n_data)))  # add bias
        delta_k_w = np.dot(np.array(self.delta_k).T, self.w)
        h_in_deriv = self._activation_deriv((h_in))
        self.delta_j = np.multiply(delta_k_w.T, h_in_deriv)
        # print(h_in_deriv.shape, self.delta_j.shape)

    def _weight_updating(self, index):
        self.w = self.w + self.learning_rate * \
            np.dot(np.array([self.delta_k[:, index]]).T,
                   np.array([self.h[:, index]]))

        self.v = self.v + self.learning_rate * \
            np.dot(np.array([self.delta_j[0:-1, index]]).T,
                   np.array([self.data[index, :]]))

    def fit(self, data, labels, n_epochs, validate=False, data_valid=None, labels_valid=None):
        '''
        Expects data in the form
        data = [[x1,y1],
                [x2,y2],
                [x3,y3],
                ...
                [xN,yN]]
        '''
        self.labels = labels
        self.n_data = 100
        self.data = data
        self.mse = np.zeros(n_epochs+1)
        self.miss_ratio = np.zeros(n_epochs+1)
        self.misses = np.zeros(n_epochs+1)
        self._add_bias()
        self.validate = False
        if validate:
            self.validate = True
            self.labels_valid = labels_valid
            self.n_data_valid = len(self.labels_valid)
            self.mse_valid = np.zeros(n_epochs+1)
            self.misses_valid = np.zeros(n_epochs+1)

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
            # self.result = classifier(self.y)
            # print(self.result)
            # print(self.labels.shape, self.y.shape)
            self.mse[i] = np.sum(
                np.square(self.labels-self.y.T))/self.n_data
            self.misses[i] = (self.n_data -
                              np.count_nonzero(self.y.T == self.labels))/self.n_data

            if self.validate:
                self.y_valid = list(map(self.predict, data_valid))
                self.result_valid = list(map(classifier, self.y_valid))
                self.mse_valid[i] = np.sum(
                    np.square(self.labels_valid-self.result_valid))/self.n_data_valid
                self.misses_valid[i] = (self.n_data_valid -
                                        np.count_nonzero(self.result_valid == self.labels_valid))/self.n_data_valid

    def predict_2(self, data_point):
        data_point = np.concatenate((data_point, np.ones(1)))
        h = self._activation(np.dot(self.v, data_point))
        # print('The v', np.dot(self.v, data_point))
        # h = np.concatenate((h, np.ones(1)))  # add bias
        # y = self._activation(np.dot(self.w, h))
        return h

    def predict(self, data_point):
        data_point = np.concatenate((data_point, np.ones(1)))
        h = self._activation(np.dot(self.v, data_point))
        h = np.concatenate((h, np.ones(1)))  # add bias
        y = self._activation(np.dot(self.w, h))
        return y

    def metrics(self):
        self._forward_pass()
        self.mse[-1] = np.sum(
            np.square(self.labels-self.y.T))/self.n_data
        self.misses[-1] = (self.n_data - np.count_nonzero(self.y.T ==
                                                          self.labels))/self.n_data
        if self.validate:
            self._forward_pass()
            self.mse_valid[-1] = np.sum(
                np.square(self.labels_valid-self.result_valid))/self.n_data_valid
            self.misses_valid[-1] = (self.n_data_valid -
                                     np.count_nonzero(self.result_valid == self.labels_valid))/self.n_data_valid

            # print('Misses', self.misses)
            # print('Misses valid', self.misses_valid)

            return self.mse, self.misses, self.mse_valid, self.misses_valid

        return self.mse, self.misses

        # for i in range(self.n_data):
        #     print(labels[i] == self.result[i])


def classifier(val):
    if val >= 0:
        return 1
    else:
        return -1


def generate_data(N):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    data = np.ones((N, 8))*(-1)

    for i in range(N):
        data[i, np.random.randint(0, 8)] = 1

    # if plot:
    #     plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
    #     plt.scatter(classB[:, 0], classB[:, 1], label="Class B")

    #     plt.plot()
    return data


def test_num_nodes():
    N = 100

    epochs = np.arange(101)
    mse = np.zeros(101)
    miss = np.zeros(101)
    iters = 1
    learning_rates = [0.9, 0.1, 0.001, 0.0001]
    for i, learning_rate in enumerate(learning_rates):
        for j in range(iters):
            data = generate_data(100)
            network = NeuralNetwork(n_inputs=8, n_hidden=2, n_outputs=8)
            network.fit(data, data, n_epochs=100)
            temp_mse, temp_miss = network.metrics()
            mse += temp_mse
            miss += temp_miss
        mse = mse/iters
        miss = miss/iters
        # plt.subplot(1, 2, 1)
        plt.plot(epochs, mse, label=r'$\eta=$'+str(learning_rate))
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title("Mean squared error")
        plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, miss)
    # plt.xlabel('Epochs')
    # plt.ylabel('# of missclassifications')
    # plt.title("Missclassifications")
    plt.show()
    # plt.legend()


def test_net():
    data = generate_data(100)
    network = NeuralNetwork(n_inputs=8, n_hidden=3, n_outputs=8)
    network.fit(data, data, n_epochs=100)
    mse, miss_ratio = network.metrics()
    test_in = [1, -1, -1, -1, -1, -1, -1, -1]

    h = network.predict_2(test_in)
    print(test_in)
    print(list(map(classifier, h)))
    # print(network.predict_2(test_in))

    # print('mse', mse[-1], 'miss_ratio', miss_ratio[-1])


# test_net()
test_num_nodes()
