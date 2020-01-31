import numpy as np
from matplotlib import pyplot as plt
from rbf_net import RBFNetwork
plt.style.use('ggplot')

np.random.seed(3)
VARS = [10, 1, 0.1, 0.01, 0.001]
VAR = 0.1


def sin2(x):
    return np.sin(2 * x)


def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1


def gen_func_data(n_train, n_test, func, noise_var=0):
    patterns = np.linspace(0, 2 * np.pi, n_train + n_test)
    noise = np.random.randn((n_train + n_test))*np.sqrt(noise_var)
    targets = np.array([func(x) for x in patterns]) + noise

    data = np.column_stack((patterns, targets))
    np.random.shuffle(data)
    train_data = data[:n_train]
    test_data = data[n_train:]

    train_patterns = train_data[:, 0]
    train_targets = train_data[:, 1]

    test_patterns = test_data[:, 0]
    test_targets = test_data[:, 1]

    return train_patterns, train_targets, test_patterns, test_targets


def plot_prediction(func):
    n_train = 64
    n_test = 63

    network = RBFNetwork(n_inputs=1, n_rbf=60, n_outputs=1, n_epochs=100)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=0.1)

    network.fit(train_patterns, train_targets, method='sequential')
    train_preds = network.predict(train_patterns)
    plt.plot(train_patterns, train_preds, 'o', color='m', label='Estimated')
    plt.plot(train_patterns, train_targets, '+', color='c', label='Target')
    plt.legend()
    plt.show()


def plot_error(func):
    n_train = 64
    n_test = 63

    rbfs = np.arange(1, 100)

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=0)

    n_train = len(train_patterns)
    n_test = len(test_patterns)
    mse_train = np.zeros(len(rbfs))
    mse_test = np.zeros(len(rbfs))

    for i, n in enumerate(rbfs):
        network = RBFNetwork(n_inputs=1, n_rbf=n, n_outputs=1, n_epochs=100)
        network.fit(train_patterns, train_targets, method='sequential')

        train_preds = network.predict(train_patterns)
        mse_train[i] = np.linalg.norm(train_preds - train_targets)**2 / n_train
        test_preds = network.predict(test_patterns)
        mse_test[i] = np.linalg.norm(test_preds - test_targets)**2 / n_test

        plt.xlabel('#RBF nodes')
        plt.ylabel('Residual error')
    plt.plot(rbfs, mse_train, 'r', label='Training set')
    plt.plot(rbfs, mse_test, 'b', label='Test set')
    plt.title('Residual error over the number of RBFs, #data-points=' +
              str(n_train)+', RBF-variance=' + str(VAR))
    plt.legend()
    plt.show()


# plot_prediction(square)
plot_error(sin2)
