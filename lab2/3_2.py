import numpy as np
from matplotlib import pyplot as plt
from rbf_net import RBFNetwork
plt.style.use('ggplot')

np.random.seed(3)


def gen_func_data(n_train, n_test, func, noise_std=0):
    patterns = np.linspace(0, 2 * np.pi, n_train + n_test)
    noise = np.random.randn(n_train + n_test)*noise_std
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

    network = RBFNetwork(n_inputs=1, n_rbf=30, n_outputs=1)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_std=0.1)

    network.fit(train_patterns, train_targets, method='sequential')
    train_preds = network.predict(train_patterns)
    targets = list(map(func, train_patterns))
    print(train_preds)
    plt.plot(train_patterns, train_preds, 'o', color='m', label='Estimated')
    plt.plot(train_patterns, targets, '+', color='c', label='Target')
    plt.legend()
    plt.show()

def sin2(x):
    return np.sin(2 * x)


def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1


plot_prediction(sin2)
