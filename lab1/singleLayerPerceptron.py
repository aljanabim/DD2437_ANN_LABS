# %%
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

# np.random.seed(3)
# %%


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

    def predict(self, x):
        if not self.learning_method == "delta_no_bias":
            x = self.extend_data_with_bias(x)

        if float(self.weights.dot(x)) > 0:
            return 1
        else:
            return -1

    def predict_array(self, X):
        pred = self.weights @ X
        pred[pred > 0] = 1
        pred[pred < 0] = -1
        return pred


    def fit(self, data, labels):
        self.squared_errors = [] # Clear error bookkeeping
        self.n_errors = []
        self.n_inputs = len(data[0:2])
        self.n_outputs = 1

        if self.learning_method == "perceptron":
            self.perceptron_fit(data, labels)
        elif self.learning_method == "delta":
            self._delta_fit(data, labels)
        elif self.learning_method == "delta_no_bias":
            self._delta_fit_no_bias(data, labels)
        elif self.learning_method == "delta_seq":
            self._delta_fit_seq(data, labels)
        
    def perceptron_fit(self, data, labels):
        '''
        Fit classifier using perceptron learning
        '''
        self.weights = np.random.normal(0, 0.5, (self.n_inputs+1))
        # Plot the DATA
        # plot_data(data.T[0:-1, :])
        data = data.T
        data = self.extend_data_with_bias(data)
        symmetric_labels = labels.copy()
        labels = labels > 0
        output = np.dot(self.weights, data) > 0
        for i in range(self.n_epochs):
            preds = self.predict_array(data)
            n_errors = np.sum(preds != symmetric_labels)
            self.n_errors.append(n_errors)

            output = np.dot(self.weights, data) > 0
            check = output == labels
            self.update_weights(data, check, labels)


    def _delta_fit_seq(self, data, labels, n_epochs=None):
        
        """Fit classifier using delta rule learning and sequential update"""
        if not n_epochs:
            n_epochs = self.n_epochs
        
        bias_ones = np.ones((len(data), 1))
        # Add column of ones, representing bias
        data = np.column_stack([data, bias_ones])

        # Tranpose so as to match assignment instruction dimensions
        data = data.transpose()
        
        # Randomizes initial weights
        self.weights = np.random.normal(0, 0.5, (1, 3))
        for _ in range(self.n_epochs):
            
            # Calculate squared errors
            error = self.weights @ data - labels # Calculate error across all the samples using weight matrix (which changes using the sequential learning)
            error_square_sum = float(error @ error.T)
            self.squared_errors.append(error_square_sum)
            # Calculate number of errors
            preds = self.predict_array(data)
            n_errors = np.sum(preds != labels)
            self.n_errors.append(n_errors)
            
            print(error_square_sum)
            print(self.squared_errors)


            for i in range(len(data)):

                x_bar = data[:,i] # extracting each input sequence from the data
                x_bar = np.array([x_bar])
                x_bar = x_bar.transpose()
                t_bar = labels[i] # Extracting label for each input sample
                t_bar= np.array([t_bar])


                # x_bar = data[:,i] # extracting each input sequence from the data
                # t_bar = labels[:,i] # Extracting label for each input sample
                # t_bar= np.array([t_bar])
                # print(data.shape)
                # print(x_bar.shape)
                # print (labels.shape)
                # print(t_bar.shape)
                # print(self.weights.shape)
                # Delta learning rule with sequential update taken from assignment instructions
                delta_weights = -(self.learning_rate *
                    (self.weights @x_bar  - t_bar) @ (x_bar.transpose()))
                # Update weights
                self.weights += delta_weights


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
            # Calculate squared errors
            error = self.weights @ data - labels
            error_square_sum = float(error @ error.T)
            self.squared_errors.append(error_square_sum)
            # Calculate number of errors
            preds = self.predict_array(data)
            n_errors = np.sum(preds != labels)
            self.n_errors.append(n_errors)
            print(error_square_sum)
            print(self.squared_errors)

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


def generate_data(N, plot=False, meanA=None, meanB=None, sigmaA=None, sigmaB=None):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    # Set up gaussian distribution parameters
    if not meanA:
        meanA = [4, 2]
    if not sigmaA:
        covA = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covA = np.array([[sigmaA, 0],
                         [0, sigmaA]])
    if not meanB:
        meanB = [-2, 2]
    if not sigmaB:
        covB = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covB = np.array([[sigmaB, 0],
                         [0, sigmaB]])

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


def plot_decision_boundary(data, *weights, title=None, labels=None):
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

    if not labels:
        labels = ['Decision boundary {}'.format(i) for i in range(len(weights))]

    # Plot decision boundaries
    for i, weight_set in enumerate(weights):
        weight_set = weight_set.flatten()
        v_x = np.linspace(-5, 5, 100)
        if len(weight_set) == 3:
            # with bias
            v_y = -(weight_set[0]/weight_set[1])*v_x - weight_set[2]/weight_set[1]
        else:
            # without bias
            v_y = -(weight_set[0]/weight_set[1])*v_x
        plt.plot(v_x, v_y, label=labels[i])

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

def plot_squared_errors(*squared_errors_list, labels=None):
    """Take list of lists squared errors and plot it."
    Keyword arg plot_labels_list allows custom labels to be set.
    """
    if not labels:
        labels = ["Preceptron {}".format(i) for i in range(len(squared_errors_list))]
    for squared_errors, plot_label in zip(squared_errors_list, labels):
        plt.plot(range(len(squared_errors)), squared_errors, label=plot_label)
    plt.title('Squared errors vs. training epochs')
    plt.xlabel("Epoch")
    plt.ylabel("Squared error")
    plt.legend()
    plt.show()

def plot_n_errors(*n_errors_list, labels=None):
    """Take list of lists of number of errors and plot it."
    Keyword arg plot_labels_list allows custom labels to be set.
    """
    if not labels:
        labels = ["Preceptron {}".format(i) for i in range(len(n_errors_list))]
    for n_errors, plot_label in zip(n_errors_list, labels):
        plt.plot(range(len(n_errors)), n_errors, label=plot_label)
    plt.title('Number of errors vs. training epochs')
    plt.xlabel("Epoch")
    plt.ylabel("Number of errors")
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
    n_epochs = 200
    learning_rate = 0.001
    n_data = 100
    n_train_samples = 25
    n_test_samples = n_data - n_train_samples
    n_trials = 5
    # learning_rule = "delta"
    learning_rule = "delta_seq"

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
    """Generates plots needed for 3.1.2.3."""
    # Set training and testing parameters
    n_epochs = 2000
    learning_rate = 0.001
    n_data = 10
    n_train_samples = 5
    n_test_samples = n_data - n_train_samples

    # Generate data
    data = generate_data(n_data, meanA = [1.5, 2], meanB = [4.5, 2])
    # data = generate_data(n_data, meanA = [-1.5, 2], meanB = [1.5, 2])
    # Split data
    patterns_train, targets_train, patterns_test, targets_test = split_data(data, n_train_samples)

    delta_perceptron = Perceptron(learning_method="delta",
                                  learning_rate=learning_rate,
                                  n_epochs=n_epochs)
    delta_no_bias_perceptron = Perceptron(learning_method="delta_no_bias",
                                          learning_rate=learning_rate,
                                          n_epochs=n_epochs)

    delta_perceptron.fit(patterns_train, targets_train)
    delta_no_bias_perceptron.fit(patterns_train, targets_train)

    plot_squared_errors(delta_perceptron.squared_errors,
                        delta_no_bias_perceptron.squared_errors,
                        labels=["Delta learning with bias", "Delta learning without bias"])

    plot_decision_boundary(data[:n_train_samples],
                           delta_perceptron.weights,
                           delta_no_bias_perceptron.weights,
                           title="Effect of bias on delta learning",
                           labels=["Delta learning with bias", "Delta learning without bias"])

def non_linearly_separable():
    """Script that generates plots needed for 3.1.3."""
    n_epochs = 100
    learning_rate = 0.001
    n_data = 100
    n_train_samples = 100
    n_test_samples = n_data - n_train_samples

    data = generate_data(n_data, meanA = [-1, 2], meanB = [1, 2])
    patterns_train, targets_train, patterns_test, targets_test = split_data(data, n_train_samples)

    delta_perceptron = Perceptron(learning_method="delta",
                                  learning_rate=learning_rate,
                                  n_epochs=n_epochs)

    perceptron_perceptron = Perceptron(learning_method="perceptron",
                                       learning_rate=learning_rate,
                                       n_epochs=n_epochs)

    delta_perceptron.fit(patterns_train, targets_train)
    perceptron_perceptron.fit(patterns_train, targets_train)

    plot_decision_boundary(data[:n_train_samples],
                           delta_perceptron.weights,
                           perceptron_perceptron.weights,
                           title="Perceptron and delta learning on non-linearly separable set",
                           labels=["Delta learning", "Perceptron learning"])

    plot_n_errors(delta_perceptron.n_errors,
                  perceptron_perceptron.n_errors,
                  labels=["Delta learning", "Perceptron learning"])

    print(delta_perceptron.n_errors)

if __name__ == "__main__":
    # test_perceptron_learning()
     test_delta_learning()
    # no_bias_comparison()
    #non_linearly_separable()