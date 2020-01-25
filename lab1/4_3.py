import numpy as np
import matplotlib.pyplot as plt
import torch

plt.style.use('ggplot')

def mackey_glass_time_series(n_steps, step_size=1, x_0=1.5, beta=0.2, gamma=0.1, n=10, tau=25):
    x = np.zeros((n_steps))
    x[0] = x_0
    for i in range(n_steps-1):
        x_tau = x[i - tau] if i >= tau else 0
        x[i+1] = x[i] + step_size*(beta*x_tau/(1 + x_tau**n) - gamma*x[i])
    return x

def generate_data(n_samples, offset=300):
    series = mackey_glass_time_series(n_samples+offset+5)
    patterns = []
    targets = []
    for t in range(offset, n_samples+offset):
        patterns.append([series[t-20], series[t-15], series[t-10], series[t-5], series[t]])
        targets.append(series[t+5])

    return np.array(patterns), np.array(targets)


def split_data(patterns, targets, validation_fraction=0.15, test_fraction=0.25):
    n_samples = len(patterns)
    n_train = int(n_samples*(1 - validation_fraction - test_fraction))
    n_validation = int(n_samples*validation_fraction)
    n_test = int(n_samples*test_fraction)

    train_patterns = torch.tensor(patterns[:n_train], dtype=torch.float)
    train_targets = torch.tensor(targets[:n_train], dtype=torch.float)
    validation_patterns = torch.tensor(patterns[n_train:(n_train+n_validation)], dtype=torch.float)
    validation_targets = torch.tensor(targets[n_train:(n_train+n_validation)], dtype=torch.float)
    test_patterns = torch.tensor(patterns[-n_test:], dtype=torch.float)
    test_targets = torch.tensor(targets[-n_test:], dtype=torch.float)

    return train_patterns, train_targets, validation_patterns, validation_targets, test_patterns, test_targets


class FullyConnectedNet:
    def __init__(self, network_dims, learning_rate=1e-6, convergence_threshold=1e-9, max_iter=100000):
        # Set up network dimensions
        self.network_dims = network_dims
        self.dim_in = network_dims[0]
        self.dim_hidden = network_dims[1:-1]
        self.dim_out = network_dims[-1]

        # Set up weights
        self.weights_in = torch.randn(self.dim_in, self.dim_hidden[0], requires_grad=True)
        self.weights_hidden = []
        for i in range(len(self.dim_hidden)-1):
            self.weights_hidden.append(
                torch.randn(self.dim_hidden[i], self.dim_hidden[i+1], requires_grad=True))
        self.weights_out = torch.randn(self.dim_hidden[-1], self.dim_out, requires_grad=True)
        self.weights_all = [self.weights_in, *self.weights_hidden, self.weights_out]

        # Set up learning hyperparameters
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter

        # Set up bookkeeping
        self.validation_loss_record = None

    def forward(self, input):
        # Input layer
        x = torch.sigmoid(torch.mm(input, self.weights_in))
        # Hidden layers
        for weights in self.weights_hidden:
            x = torch.sigmoid(torch.mm(x, weights))
        # Output layer
        output = torch.mm(x, self.weights_out)

        return output

    def loss(self, predictions, targets):
        return (predictions - targets).pow(2).sum()

    def fit(self, train_patterns, train_targets, validation_patterns, validation_targets):
        validation_loss = np.infty
        delta_validation_loss = np.infty
        iter = 0
        self.validation_loss_record = []

        while delta_validation_loss > self.convergence_threshold and iter < self.max_iter:
            old_validation_loss = validation_loss

            train_predictions = self.forward(train_patterns)
            train_loss = self.loss(train_predictions, train_targets)
            train_loss.backward()

            with torch.no_grad():
                for weights in self.weights_all:
                    weights -= learning_rate*weights.grad
                    weights.grad.zero_()

                validation_predictions = self.forward(validation_patterns)
                validation_loss = self.loss(validation_predictions, validation_targets)
                self.validation_loss_record.append(validation_loss/len(validation_targets))
                delta_validation_loss = abs(old_validation_loss - validation_loss)

            # Progress bar
            if iter % int(max_iter/10) == 0:
                print("Training progress: {}".format(iter/max_iter))
                print("Iteration number {}. Average validation loss per sample: {:.2f}".format(
                    iter, validation_loss/len(validation_targets)))

            iter += 1



# Hyperparameters
hidden_layer_dims = [5, 5]
learning_rate = 1e-9
convergence_threshold = 1e-12
max_iter = 50000
n_samples = 1200

# Generate and process data
patterns, targets = generate_data(n_samples)
train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
    patterns, targets, validation_fraction=200/1200, test_fraction=200/1200)

# Set up network
dim_in = train_patterns.shape[1]
dim_out = 1
layer_dims = [dim_in, *hidden_layer_dims, dim_out]
net = FullyConnectedNet(layer_dims,
                        learning_rate=learning_rate,
                        convergence_threshold=convergence_threshold,
                        max_iter=max_iter)

# Train networks
net.fit(train_patterns, train_targets, validation_patterns, validation_targets)

# Plot convergence of validation loss
plt.plot(net.validation_loss_record[20:])
plt.show()

# Test
test_predictions = net.forward(test_patterns)
test_loss = net.loss(test_predictions, test_targets)
print("Average test loss per sample: {:.2f}".format(test_loss/len(test_predictions)))
plt.plot(test_predictions.detach().numpy(), label='Prediction')
plt.plot(test_targets.detach().numpy(), label='Target')
plt.legend()
plt.show()
