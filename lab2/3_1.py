# %%
import numpy as np
from matplotlib import pyplot as plt
from rbf_net import RBFNetwork
plt.style.use('ggplot')
VAR = 0.1

np.random.seed(3)


def plot_prediction():
    network = RBFNetwork(n_inputs=1, n_rbf=50, n_outputs=1)

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.zeros(x.shape)
    y_target = list(map(sin2, x))
    for i, x_i in enumerate(x):
        y[i] = network.predict(x_i)


def sin2(x):
    return np.sin(2 * x)


def square(x):
    if np.sin(2 * x) >= 0:
        return 1
    else:
        return -1


def classifier(x):
    if x >= 0:
        return 1
    else:
        return -1


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
    n_train = 128
    n_test = 128

    network = RBFNetwork(n_inputs=1, n_rbf=100, n_outputs=1)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_std=0.5)

    network.fit(train_patterns, train_targets, method='sequential')
    train_preds = network.predict(train_patterns)
    targets = list(map(func, train_patterns))
    print(train_preds)
    plt.plot(train_patterns, train_preds, 'o', color='m', label='Estimated')
    plt.plot(train_patterns, targets, '+', color='c', label='Target')
    plt.legend()
    plt.show()


def plot_error(func, MAKE_SQAURE_GREAT=False):
    n_train = 64
    n_test = 63

    rbfs = np.arange(1, 100)

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func)

    n_train = len(train_patterns)
    n_test = len(test_patterns)
    error_train = np.zeros(len(rbfs))
    error_train_improved = np.zeros(len(rbfs))
    error_test = np.zeros(len(rbfs))
    error_test_improved = np.zeros(len(rbfs))

    for i, n in enumerate(rbfs):
        network = RBFNetwork(n_inputs=1, n_rbf=n, n_outputs=1, n_epochs=10)
        network.fit(train_patterns, train_targets)

        train_preds_imporved = list(
            map(classifier, network.predict(train_patterns)))
        error_train_improved[i] = np.sum(
            np.abs(train_preds_imporved - train_targets)) / n_train

        train_preds = network.predict(train_patterns)
        error_train[i] = np.sum(np.abs(train_preds - train_targets)) / n_train

        test_preds_improved = list(
            map(classifier, network.predict(test_patterns)))
        error_test_improved[i] = np.sum(
            np.abs(test_preds_improved - test_targets)) / n_test

        test_preds = network.predict(test_patterns)
        error_test[i] = np.sum(np.abs(test_preds - test_targets)) / n_test

        plt.xlabel('#RBF nodes')
        plt.ylabel('Residual error')
    if MAKE_SQAURE_GREAT:
        plt.plot(rbfs, error_train, 'r', label='Training set')
        plt.plot(rbfs, error_train_improved, 'r--',
                 label='Training set (improved)')
        # plt.plot(rbfs, error_test_improved, 'b--', label='Test set (improved)')
    else:
        plt.plot(rbfs, error_train, 'r', label='Training set')
        plt.plot(rbfs, error_test, 'b', label='Test set')
    plt.title('Residual error over the number of RBFs, #data-points=' +
              str(n_train)+', RBF-variance=' + str(VAR))
    plt.legend()
    plt.show()


def calc_absolute_error(func, n_rbfs, n_trials=10):
    n_train = 64
    n_test = 63

    error_record = []
    for i in range(n_trials):
        train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
            n_train, n_test, func)
        network = RBFNetwork(n_inputs=1, n_rbf=n_rbfs, n_outputs=1, n_epochs=10)
        network.fit(train_patterns, train_targets)
        test_preds = network.predict(test_patterns)
        mean_abs_error = np.mean(np.abs(test_preds - test_targets))
        error_record.append(mean_abs_error)

    errors = np.array(error_record)
    return np.mean(errors), np.std(errors)/np.sqrt(len(errors))


def plot_mean_abs_error(func, max_n_rbfs, n_trials=10):
    n_rbfs_list = np.arange(1, max_n_rbfs+1)
    error_list = []
    sem_list = []
    for n_rbfs in n_rbfs_list:
        error, sem = calc_absolute_error(func, n_rbfs, n_trials)
        error_list.append(error)
        sem_list.append(sem)

    ax = plt.subplot(111)
    ax.set_yscale("log", nonposy='clip')
    ax.errorbar(n_rbfs_list, error_list, sem_list, label="Mean Absolute Error +- SEM")
    plt.title("Decay of mean absolute error on sin2.\nAverage over {} trials.".format(n_trials))
    plt.xlabel("Number of RBF units")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.show()



def perfect_square(func):
    n_train = 64
    n_test = 63
    n_rbfs = 4

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func)

    network = RBFNetwork(n_inputs=1, n_rbf=n_rbfs, n_outputs=1, n_epochs=10)
    network.rbf_centers = np.array([(1/4)*np.pi,
                                    (3/4)*np.pi,
                                    (5/4)*np.pi,
                                    (7/4)*np.pi])
    network.fit(train_patterns, train_targets)

    test_preds = network.predict(test_patterns)
    test_preds_improved = list(map(classifier, network.predict(test_patterns)))

    plt.plot(network.rbf_centers, np.zeros(network.rbf_centers.shape), 'x', color='black', label='RBF centers')
    plt.plot(test_patterns, test_targets, 's', markersize=8, color='grey', label='Targets')
    plt.plot(test_patterns, test_preds, 'o', markersize=4, color='steelblue', label='Prediction')
    plt.plot(test_patterns, test_preds_improved, 'o', markersize=4, color='firebrick', label='Step-transformed prediction')
    plt.title("Transformed vs non-transformed RBF output")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


plot_mean_abs_error(square, 80, n_trials=100)
# perfect_square(square)

# ALL NEEDED PLOTS
# plot_error(func=sin2, MAKE_SQAURE_GREAT=False)

# plot_error(func=square, MAKE_SQAURE_GREAT=False)
# plot_error(func=square, MAKE_SQAURE_GREAT=True)





# plot_prediction(func=sin2)


# root = 'plots/3_1/'

# filenames = [root+'residual_error_sin2.png',
#              root+'residual_error_square_improved.png',
#              root+'residual_error_square.png']

# args = {0: {'func': sin2,
#             'mk_great': False},
#         1: {'func': square,
#             'mk_great': False},
#         2: {'func': sin2,
#             'mk_great': True}}

# for i, filename in enumerate(filenames):
#     plot_error(filename, func=args[i]['func'],
#                MAKE_SQAURE_GREAT=args[i]['mk_great'])
