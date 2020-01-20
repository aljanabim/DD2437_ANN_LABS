# %%
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(3)
# %%


class Perceptron():
    def __init__(self, learning_method="perceptron", learning_rate=0.1, n_epochs=20):
        # Learning parameters
        self.learning_method = learning_method
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # Model variables
        self.weights = None
        self.n_inputs = None
        self.n_outputs = None

    def predict(self, x):
        x = self.extend_data_with_bias(x)
        if int(self.weights.dot(x)) >= 0:
            return 1
        else:
            return -1

    def fit(self, data, labels):
        self.n_inputs = len(data)
        self.n_outputs = len(labels)

        if self.learning_method == "perceptron":
            self.perceptron_fit(data, labels)
        elif self.learning_method == "delta":
            self.delta_fit(data, labels)

    def perceptron_fit(self, data, labels):
        '''
        Fit classifier using perceptron learning
        '''
        self.weights = np.random.normal(0, 0.5, self.n_inputs+1)
        data = self.extend_data_with_bias(data)
        inputs = data[:, 0]
        outputs = data[:, 1]
        for i in range(self.n_epochs):
            pass
        print(data)

    def delta_fit(self, data, labels, n_epochs=None):
        if not n_epochs:
            n_epochs = self.n_epochs
        bias_ones = np.ones((self.n_inputs, 1))
        data = np.column_stack([data, bias_ones])
        data = data.transpose()

        self.weights = np.random.normal(0, 0.5, (1, 3))

        for _ in range(self.n_epochs):
            delta_weights = np.zeros((1, self.n_inputs))
            delta_weights = -self.learning_rate * \
                (self.weights @ data  - labels) @ (data.transpose())
            self.weights += delta_weights

    def extend_data_with_bias(self, data):
        '''
        Extend data with a row of ones for the bias weight
        '''
        dim = data.shape[1] if len(data.shape) > 1 else 1
        data = np.row_stack([data, np.ones(dim)])
        return data


# %%


def generate_data(N, plot=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    meanA = [1, 3]
    covA = np.array([[0.2, 0],
                     [0, 0.8]])
    meanB = [-1, -2]
    covB = np.array([[0.8, 0],
                     [0, 0.5]])
    classA = np.random.multivariate_normal(meanA, covA, N)
    classB = np.random.multivariate_normal(meanB, covB, N)
    classA_extended = np.column_stack([classA, np.ones(N)])
    classB_extended = np.column_stack([classB, -np.ones(N)])
    data = np.row_stack([classA_extended, classB_extended])
    np.random.shuffle(data)
    # data = np.array([[classA, np.zeros(N)],
    #                  [classB, np.ones(N)]])
    # print(data)
    if plot:
        plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
        plt.scatter(classB[:, 0], classB[:, 1], label="Class B")
        plt.legend()
        plt.plot()

    return data



def test_delta_learning():
    # Test delta learning
    n_data = 10
    n_train_samples = 5
    n_test_samples = n_data - n_train_samples

    data = generate_data(n_data)

    patterns_train = data[:n_train_samples, :2]
    targets_train = data[:n_train_samples, 2]
    patterns_test = data[-n_test_samples:, :2]
    targets_test = data[-n_test_samples:, 2]

    perceptron = Perceptron(learning_method="delta")
    perceptron.fit(patterns_train, targets_train)

    n_correct = 0
    n_incorrect = 0
    for test_sample in zip(patterns_test, targets_test):
        pattern = np.reshape(test_sample[0], (-1, 1))
        target = test_sample[1]
        prediction = perceptron.predict(pattern)

        print(target, prediction)

        if prediction == target:
            n_correct += 1
        else:
            n_incorrect += 1


test_delta_learning()
