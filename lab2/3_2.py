import numpy as np
from matplotlib import pyplot as plt
from rbf_net import RBFNetwork
plt.style.use('ggplot')

np.random.seed(3)

VAR = 0.1


def sin2(x):
    return np.sin(2 * x)


def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1


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
    func = sin2

    network = RBFNetwork(n_inputs=1, n_rbf=64, n_outputs=1, n_epochs=200,
                         learning_rate_start=0.1, learning_rate_end=0.001)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=0.001)

    print(network.learning_rate)
    print(network.learning_rate_decay)
    print(np.exp(-network.learning_rate_decay))
    network.fit(train_patterns, train_targets, method='sequential')
    train_preds = network.predict(train_patterns)
    print(network.learning_rate)
    plt.plot(train_patterns, train_preds, 'o', color='m', label='Estimated')
    plt.plot(train_patterns, train_targets, '+', color='c', label='Target')
    plt.legend()
    plt.show()

    plt.plot(network.learning_rate_record)
    plt.show()


def calc_mse(predictions, targets):
    return np.linalg.norm(predictions - targets)**2 / len(targets)


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


def plot_convergence_comparison(func):
    n_nodes = 5
    n_trials = 10
    n_epochs = 1000
    learning_rates = [(0.1, 0.1),
                      (0.05, 0.05),
                      (0.01, 0.01),
                      (0.005, 0.005),
                      (0.001, 0.001),
                      (0.1, 0.001)]

    mses = np.zeros((len(learning_rates), n_trials, n_epochs))
    for i, rate in enumerate(learning_rates):
        n_train = 64
        n_test = 63
        for j in range(n_trials):
            train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
                n_train, n_test, func, noise_var=0.1)

            net = RBFNetwork(n_inputs=1, n_rbf=n_nodes, n_outputs=1, n_epochs=n_epochs,
                             learning_rate_start=rate[0], learning_rate_end=rate[1])

            net.fit(train_patterns, train_targets, method='sequential')
            mses[i, j, :] = net.mse_record

    average_mses = np.mean(mses, 1)
    for i, rate in enumerate(learning_rates):
        plt.semilogx(average_mses[i, :],
                     label="Learning rate: {}".format(rate[0]))

    plt.title("Comparison of convergence speeds for different learning rates")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()


def final_test_error_comparison(func):
    n_train = 64
    n_test = 63
    n_nodes = 5
    n_trials = 10
    n_epochs = 1000

    mses = np.zeros((n_trials))

    for i in range(n_trials):
        train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
            n_train, n_test, func, noise_var=0)

        net = RBFNetwork(n_inputs=1, n_rbf=n_nodes, n_outputs=1, n_epochs=n_epochs,
                         learning_rate_start=0.1, learning_rate_end=0.001)

        net.fit(train_patterns, train_targets, method='sequential')
        test_preds = net.predict(test_patterns)
        mses[i] = calc_mse(test_preds, test_targets)

    mse_mean = np.mean(mses)
    mse_std = np.std(mses)/len(mses)

    print("Mean Square Error: {:.2f} +/- {:.2f}".format(mse_mean, mse_std))


def plot_variance(func):
    VARS = [10, 1, 0.1, 0.01, 0.001]
    n_train = 64
    n_test = 63
    rbfs = np.arange(1, 100)
    noise_var = 0.1

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=noise_var)

    for var in VARS:
        error = np.zeros(len(rbfs))
        for i, n_rbf in enumerate(rbfs):
            network = RBFNetwork(n_inputs=1, n_rbf=n_rbf,
                                 n_outputs=1, rbf_var=var)
            network.fit(train_patterns, train_targets, 'batch')
            preds = network.predict(train_patterns)
            mse = network.calc_mse(preds, train_targets)
            error[i] = mse
        plt.plot(rbfs, error, label='RBF variance: '+str(var))
        plt.legend()
    plt.xlabel('#RBF nodes')
    plt.ylabel('MSE')
    plt.title('Variance comparison for ' +
              func.__name__ + ' with data noise of '+str(int(noise_var*100)) + '%')
    plt.show()


def plot_centers(func):
    VARS = [10, 1, 0.1, 0.01, 0.001]
    n_train = 64
    n_test = 63
    rbfs = np.arange(1, 100)
    noise_var = 0.1

    centering = ['linspace', 'random']

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=noise_var)

    for center in centering:
        error = np.zeros(len(rbfs))
        for i, n_rbf in enumerate(rbfs):
            network = RBFNetwork(n_inputs=1, n_rbf=n_rbf,
                                 n_outputs=1, centering=center)
            network.fit(train_patterns, train_targets, 'batch')
            preds = network.predict(train_patterns)
            mse = network.calc_mse(preds, train_targets)
            error[i] = mse
        plt.plot(rbfs, error, label='Centering using: '+str(center))
        plt.legend()
    plt.xlabel('#RBF nodes')
    plt.ylabel('MSE')
    plt.title('Centering comparison for ' +
              func.__name__ + ' with data noise of '+str(int(noise_var*100)) + '%')
    plt.show()


# plot_prediction(square)
# plot_error(sin2)
plot_convergence_comparison(sin2)
# final_test_error_comparison(sin2)
# plot_variance(sin2)
# plot_variance(square)
# plot_centers(sin2)
# plot_centers(square)
