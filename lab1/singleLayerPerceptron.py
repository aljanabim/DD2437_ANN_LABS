# %%
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

# np.random.seed(3)
# %%


class Perceptron():
    def __init__(self, learning_method="perceptron", learning_rate=0.1, n_epochs=20, n_data=None):
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

    def predict(self, x):
        if not self.learning_method == "delta_no_bias":
            x = self.extend_data_with_bias(x)
        if float(self.weights.dot(x)) > 0:
            return 1
        else:
            return -1

    def fit(self, data, labels):
        self.squared_errors = [] # Clear error bookkeeping
        self.n_inputs = len(data[0:2])
        self.n_outputs = 1

        if self.learning_method == "perceptron":
            self.perceptron_fit(data, labels)
        elif self.learning_method == "delta":
            self._delta_fit(data, labels)
        elif self.learning_method == "delta_no_bias":
            self._delta_fit_no_bias(data, labels)

    def perceptron_fit(self, data, labels):
        '''
        Fit classifier using perceptron learning
        '''
        self.weights = np.random.normal(0, 0.5, (self.n_inputs+1))
        # Plot the DATA
        plot_data(data.T[0:-1, :])

        data = self.extend_data_with_bias(data[0:2])
        labels = labels > 0
        output = np.dot(self.weights, data) > 0
        for i in range(self.n_epochs):
            output = np.dot(self.weights, data) > 0
            check = output == labels
            self.update_weights(data, check, labels)

        plot_decision_boundary(self.weights)
        print("Correct out of", self.n_data*2,
              "are", sum(labels == output))


    def update_weights(self, data, check, labels):
        """Helper function to perceptron learning."""
        for i in range(len(labels)):
            if not check[i]:
                if labels[i] > 0:  # -1 but should have been 1
                    self.weights = self.weights + \
                        self.learning_rate * data[:, i]
                else:
                    self.weights = self.weights - \
                        self.learning_rate * data[:, i]


    def _delta_fit(self, data, labels, n_epochs=None):
        """Fit classifier using delta rule learning."""
        if not n_epochs:
            n_epochs = self.n_epochs
        bias_ones = np.ones((len(data), 1))
        # Add column of ones, representing bias
        data = np.column_stack([data, bias_ones])
        # Tranpose so as to match assignment instruction dimensions
        data = data.transpose()
        # Randomizes initial weights
        self.weights = np.random.normal(0, 0.5, (1, data.shape[0]))
        # Run delta rule iterations
        self._delta_iterate(data, labels, n_epochs)


    def _delta_fit_no_bias(self, data, labels, n_epochs=None):
        """Fit classifier using delta rule learning, without using any bias."""
        if not n_epochs:
            n_epochs = self.n_epochs
        # Tranpose so as to match assignment instruction dimensions
        data = data.transpose()
        # Randomizes initial weights
        self.weights = np.random.normal(0, 0.5, (1, data.shape[0]))
        # Run delta rule iterations
        self._delta_iterate(data, labels, n_epochs)


    def _delta_iterate(self, data, labels, n_epochs):
        for _ in range(self.n_epochs):
            # Calculate error vector
            error = self.weights @ data - labels
            # Save squared errors
            error_square_sum = float(error @ error.T)
            self.squared_errors.append(error_square_sum)
            # Delta learning rule taken from assignment instructions
            delta_weights = -(self.learning_rate *
                error @ (data.transpose()))
            # Update weights
            self.weights += delta_weights

    def extend_data_with_bias(self, data):
        '''
        Extend data with a row of ones for the bias weight
        '''
        dim = data.shape[1] if len(data.shape) > 1 else 1
        data = np.row_stack([data, np.ones(dim)])
        return data


def generate_data(N, plot=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    meanA = [4, 2]
    covA = np.array([[0.2, 0],
                     [0, 0.8]])
    meanB = [-2, 2]
    covB = np.array([[0.5, 0],
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

        plt.plot()

    return data

def split_data(data, n_train_samples):
    n_test_samples = len(data) - n_train_samples
    patterns_train = data[:n_train_samples, :2]
    targets_train = data[:n_train_samples, 2]
    patterns_test = data[-n_test_samples:, :2]
    targets_test = data[-n_test_samples:, 2]
    return patterns_train, targets_train, patterns_test, targets_test

def test_accuracy(model, patterns_test, targets_test):
    n_correct = 0
    n_incorrect = 0
    for test_sample in zip(patterns_test, targets_test):
        # Reshape pattern array into 2-d array
        pattern = np.reshape(test_sample[0], (-1, 1))
        target = test_sample[1]
        prediction = model.predict(pattern)
        if prediction == target:
            n_correct += 1
        else:
            n_incorrect += 1
        return n_correct/(n_correct + n_incorrect)


def plot_decision_boundary(data, *weights, title=None):
    """Take weights and plot corresponding decision boundary (2d).

    Weights should be an (1,N) shape 2-d array, and not a (N,) 1-d array.

    Data is used for scatter plotting, and should contain train data or test data, but probably
    not both. An arbitrary number of sets of weights can be given.

    Decision boundary is defined by the line wv = 0. In other words,
    it is given by the line that is perpendicular to w. This is given by
    wv = w_0*x_0 + w_1*x_1 + w_2 = 0 => x_2 = -(w_0/w_1)*x_0 - (w_2/w_1)
    which is the equation for a straight line.
    """

    # Scatterplot data points
    classA = data[data[:, 2] == 1]
    classB = data[data[:, 2] == -1]
    plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
    plt.scatter(classB[:, 0], classB[:, 1], label="Class B")

    # Plot decision boundaries
    for i, weight_set in enumerate(weights):
        v_x = np.linspace(-5, 5, 100)
        if len(weight_set[0]) == 3:
            # with bias
            v_y = -(weight_set[0,0]/weight_set[0,1])*v_x - weight_set[0,2]/weight_set[0,1]
        else:
            # without bias
            v_y = -(weight_set[0,0]/weight_set[0,1])*v_x
        plt.plot(v_x, v_y, label='Decision boundary {}'.format(i))

    # Show plot
    x_max = np.max(data[:, 0])
    x_min = np.min(data[:, 0])
    y_max = np.max(data[:, 1])
    y_min = np.min(data[:, 1])
    plt.xlim(x_min-2, x_max+2)
    plt.ylim(y_min-2, y_max+2)
    plt.legend()
    if title:
        plt.title(title)
    plt.show()

def plot_squared_errors(squared_errors_list, plot_label_list=None):
    """Take list of lists squared errors and plot it."
    Keyword arg plot_labels_list allows custom labels to be set.
    """
    if not plot_label_list:
        plot_label_list = ["Squared errors {}".format(i) for i in range(len(squared_errors_list))]
    for squared_errors, plot_label in zip(squared_errors_list, plot_label_list):
        plt.plot(range(len(squared_errors)), squared_errors, label=plot_label)
    plt.title('Squared errors vs. epochs')
    plt.xlabel("Epoch")
    plt.ylabel("Squared error")
    plt.legend()
    plt.show()


def test_perceptron_learning():
    # Test perceptron learning
    print("DOING PERCEPTRON LEARNING")
    n_data = 50
    data = generate_data(n_data).T
    node = Perceptron(learning_method='perceptron',
                      learning_rate=0.1, n_epochs=600, n_data=n_data)

    node.fit(data, data[2])
    print("FINISHED PERCEPTRON LEARNING")


def test_delta_learning():
    """Script for testing delta learning implementation and plotting decision boundaries."""

    # Set training and testing parameters
    n_epochs = 20
    learning_rate = 0.001
    n_data = 50
    n_train_samples = 25
    n_test_samples = n_data - n_train_samples
    n_trials = 5
    # learning_rule = "delta"
    learning_rule = "delta_no_bias"

    # Generate data
    data = generate_data(n_data)
    # Split data
    patterns_train, targets_train, patterns_test, targets_test = split_data(data, n_train_samples)


    # Initialize percepptron
    perceptron = Perceptron(learning_method=learning_rule, learning_rate=learning_rate, n_epochs=n_epochs)

    # Run training and testing n_trials times, save weights in list.
    weights_list = []
    accuracy = 0
    for trial in range(n_trials):
        perceptron.fit(patterns_train, targets_train)
        # Test accuracy
        accuracy += test_accuracy(perceptron, patterns_test, targets_test) / n_trials
        # Save final weights for plotting of decision boundaries
        weights_list.append(perceptron.weights)

    print("Example weights: {}".format(weights_list))
    print("Total testing accuracy: {}".format(accuracy))

    # Plot decision boundaries together with training data
    plot_decision_boundary(data[:n_train_samples], *weights_list,
                           title="Decision boundaries and training data")
    # Plot decision boundaries together with testing data
    plot_decision_boundary(data[-n_test_samples:], *weights_list,
                           title="Decision boundaries and testing data")

    # Plot squared errors
    plot_squared_errors([perceptron.squared_errors], ["Delta rule learning"])


def no_bias_comparison():
    # Set training and testing parameters
    n_epochs = 20
    learning_rate = 0.001
    n_data = 50
    n_train_samples = 25
    n_test_samples = n_data - n_train_samples
    n_trials = 5

    # Generate data
    data = generate_data(n_data)
    # Split data
    patterns_train, targets_train, patterns_test, targets_test = split_data(data, n_train_samples)

    delta_perceptron = Perceptron(learning_method="delta")
    delta_no_bias_perceptron = Perceptron(learning_method="delta_no_bias")

    delta_perceptron.fit(patterns_train, targets_train)
    delta_no_bias_perceptron.fit(patterns_train, targets_train)





if __name__ == "__main__":
    # test_perceptron_learning()
    test_delta_learning()
