import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from sklearn_net import SklearnNet
from fully_connected_net import FullyConnectedNet
from sequential_net import SequentialNet

plt.style.use('ggplot')


# ================================ DATA HANDLING ===============================

def mackey_glass_time_series(n_steps, step_size=1, x_0=1.5, beta=0.2, gamma=0.1, n=10, tau=25, plot=False):
    """Mackey-Glass time series as defined in the instructions."""
    x = np.zeros((n_steps))
    x[0] = x_0
    for i in range(n_steps-1):
        x_tau = x[i - tau] if i >= tau else 0
        x[i+1] = x[i] + step_size*(beta*x_tau/(1 + x_tau**n) - gamma*x[i])

    if plot:
        plt.plot(x)
        plt.title("Mackey-Glass Time Series")
        plt.show()

    return x


def generate_data(n_samples, offset=300, plot=False):
    """Sample from Mackey-Glass time series and produce pattern, target pairs according to instructions."""
    series = mackey_glass_time_series(n_samples+offset+5, plot=plot)
    patterns = []
    targets = []
    for t in range(offset, n_samples+offset):
        patterns.append([series[t-20], series[t-15], series[t-10], series[t-5], series[t]])
        targets.append(series[t+5])

    data = np.column_stack((np.array(patterns), np.array(targets)))
    # data = (data - np.mean(data))/np.std(data)
    return data[:, :-1], data[:, -1]


def extend_with_bias(patterns):
    """Add column of ones to pattern, in order to accomodate bias weights."""
    n_patterns = patterns.shape[0]
    ones = torch.ones((n_patterns, 1))
    return torch.cat((patterns, ones), 1)


def split_data(patterns, targets, validation_fraction=0.15, test_fraction=0.25):
    """Split data into training, validation and testing set, and add bias to patterns."""
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


# ================================ PLOTTING FUNCTIONS ===============================

def plot_weights_histogram(weights_list):
    """Takes list of weights and plots a histogram."""
    for weights in weights_list:
        print(weights.shape)
    flat_weights = [w.detach().numpy().flatten() for w in weights_list]
    all_weights = np.concatenate(flat_weights)

    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(all_weights, bins=20)
    plt.show()


# ================================ MODEL SELECTION ============================

def loss_function(predictions, targets):
    return (predictions - targets).pow(2).sum().item()


def model_selection(models, train_patterns, train_targets, validation_patterns, validation_targets):
    """Trains and evaluates given models on given data."""
    loss_list = []
    for i, model in enumerate(models):
        model.fit(train_patterns, train_targets, validation_patterns, validation_targets)
        with torch.no_grad():
            validation_predictions = model.forward(validation_patterns)
            loss = loss_function(validation_predictions, validation_targets)
            loss_list.append(loss)

    for i in range(len(models)):
        print("Model {} loss: {:.2f}".format(i, loss_list[i]))

    return models[loss_list.index(min(loss_list))]


# ================================ BENCHMARKING ===============================

def average_loss(n_trials, net, train_patterns, train_targets, validation_patterns,
                 validation_targets, test_patterns, test_targets):
    """Takes model and data and evaluates model on data."""
    losses = []
    for trial in range(n_trials):
        net.reset_weights()
        net.fit(train_patterns, train_targets, validation_patterns, validation_targets)

        test_predictions = net.forward(test_patterns)
        test_loss = net.loss(test_predictions, test_targets)
        losses.append(test_loss.item())

    return np.mean(losses), np.std(losses)


# ================================ TESTING ===============================

def test_average():
    """Example illustrating the use of the average loss benchmark."""
    hidden_layer_dims = [6]
    learning_rate = 1e-8     #1e-6 seems to be the largest usable learning rate
    convergence_threshold = 1e-13
    max_iter = 100
    reg_factor = 0.0005
    print(average_loss(100, hidden_layer_dims, learning_rate, convergence_threshold, max_iter, reg_factor))


def test_model_selection():
    """Example illustrating the use of model selection."""
    # Hyperparameters
    learning_rate = 1e-10     #1e-6 seems to be the largest usable learning rate
    convergence_threshold = 1e-13
    max_iter = 100
    reg_factor = 0.0005
    n_samples = 1200

    # Generate and process data
    patterns, targets = generate_data(n_samples, plot=False)
    train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
        patterns, targets, validation_fraction=200/1200, test_fraction=200/1200)

    # Build models
    hidden_layer_dims_list = [[4], [8]]
    models = []
    for hidden_layer_dims in hidden_layer_dims_list:
        dim_in = train_patterns.shape[1]
        dim_out = 1
        layer_dims = [dim_in, *hidden_layer_dims, dim_out]
        net = FullyConnectedNet(layer_dims,
                                learning_rate=learning_rate,
                                convergence_threshold=convergence_threshold,
                                max_iter=max_iter,
                                reg_factor=reg_factor)
        models.append(net)

    # Run model selection
    best_net = model_selection(
        models, train_patterns, train_targets, validation_patterns, validation_targets)


def test_sequential():
    # Hyperparameters
    hidden_layer_dims = [3,3]
    learning_rate = 1e-4    #1e-6 seems to be the largest usable learning rate
    convergence_threshold = 1e-13
    max_iter = 50
    reg_factor = 0.64
    n_samples = 1200

    # Generate and process data
    patterns, targets = generate_data(n_samples, plot=True)
    train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
        patterns, targets, validation_fraction=200/1200, test_fraction=200/1200)


    # Set up network
    dim_in = train_patterns.shape[1]
    print(train_patterns[0])
    dim_out = 1
    layer_dims = [dim_in, *hidden_layer_dims, dim_out]
    net = SequentialNet(layer_dims,
                        learning_rate=learning_rate,
                        convergence_threshold=convergence_threshold,
                        max_iter=max_iter,
                        reg_factor=reg_factor)
    # Train networks
    net.fit(train_patterns, train_targets, validation_patterns, validation_targets)

    plt.plot(net.validation_loss_record[:])
    plt.show()

    # Test
    test_predictions = net.model(test_patterns)
    test_loss = net.loss(test_predictions, test_targets)
    print("Average test loss per sample: {:.2f}".format(test_loss/len(test_predictions)))
    plt.plot(test_predictions.detach().numpy(), label='Prediction')
    plt.plot(test_targets.detach().numpy(), label='Target')
    plt.legend()
    plt.show()

    print("end")

def test_sklearn():
    # Hyperparameters
    hidden_layer_dims = [2,3]
    learning_rate = 0.001   #1e-6 seems to be the largest usable learning rate
    convergence_threshold = 1e-13
    max_iter = 100
    reg_factor = 0
    n_samples = 1200

    # Generate and process data
    patterns, targets = generate_data(n_samples, plot=True)
    train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
        patterns, targets, validation_fraction=200/1200, test_fraction=200/1200)


    train_patterns = train_patterns.detach().numpy()
    train_targets = train_targets.detach().numpy()
    validation_patterns = validation_patterns.detach().numpy()
    validation_targets = validation_targets.detach().numpy()
    test_patterns = test_patterns.detach().numpy()
    test_targets = test_targets.detach().numpy()

    # Set up network
    dim_in = train_patterns.shape[1]
    print(train_patterns[0])
    dim_out = 1
    layer_dims = [dim_in, *hidden_layer_dims, dim_out]
    net = SklearnNet(layer_dims,
                        learning_rate=learning_rate,
                        convergence_threshold=convergence_threshold,
                        max_iter=max_iter,
                        reg_factor=reg_factor)
    # Train networks
    net.fit(train_patterns, train_targets, validation_patterns, validation_targets)

    plt.plot(net.validation_loss_record[:])
    plt.show()

    # Test
    test_predictions = net.model.predict(test_patterns)
    test_loss = net.loss(test_predictions, test_targets)
    print("Average test loss per sample: {:.2f}".format(test_loss/len(test_predictions)))
    plt.plot(test_predictions, label='Prediction')
    plt.plot(test_targets, label='Target')
    plt.legend()
    plt.show()


def main():
    # Hyperparameters
    hidden_layer_dims = [2, 2]
    learning_rate = 0.001     #1e-6 seems to be the largest usable learning rate
    convergence_threshold = 1e-13
    max_iter = 10
    reg_factor = 0.3
    n_samples = 1200

    # Generate and process data
    patterns, targets = generate_data(n_samples, plot=True)
    train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
        patterns, targets, validation_fraction=200/1200, test_fraction=200/1200)

    # Set up network
    dim_in = train_patterns.shape[1]
    print(train_patterns[0])
    dim_out = 1
    layer_dims = [dim_in, *hidden_layer_dims, dim_out]
    net = FullyConnectedNet(layer_dims,
                            learning_rate=learning_rate,
                            convergence_threshold=convergence_threshold,
                            max_iter=max_iter,
                            reg_factor=reg_factor)

    # Train networks
    # net.fit(train_patterns, train_targets, validation_patterns, validation_targets)
    net.fit_with_torch_optimizer(train_patterns, train_targets, validation_patterns, validation_targets)

    plot_weights_histogram(net.weights_all)

    # Plot convergence of validation loss
    plt.plot(net.validation_loss_record[:])
    plt.show()

    # Test
    test_predictions = net.forward(test_patterns)
    test_loss = net.loss(test_predictions, test_targets)
    print("Average test loss per sample: {:.2f}".format(test_loss/len(test_predictions)))
    plt.plot(test_predictions.detach().numpy(), label='Prediction')
    plt.plot(test_targets.detach().numpy(), label='Target')
    plt.legend()
    plt.show()

    print("end")

# main()
# test_model_selection()
# test_sequential()

test_sklearn()
