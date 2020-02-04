import numpy as np
from matplotlib import pyplot as plt
from rbf_net import RBFNetwork
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.mlab import griddata
from scipy import interpolate
plt.style.use('ggplot')

np.random.seed(3)
VARS = [10, 1, 0.1, 0.01, 0.001]
VAR = 0.5


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


def get_ballistic_data():
    train_data = np.loadtxt('lab2/plots/3_3/ballist.dat')
    test_data = np.loadtxt('lab2/plots/3_3/balltest.dat')

    train_patterns, train_targets = train_data[:, :2], train_data[:, 2:]
    test_patterns, test_targets = test_data[:, :2], test_data[:, 2:]

    return train_patterns, train_targets, test_patterns, test_targets


def ballistic_plot():
    n_rbf = 4
    n_train = 120
    n_test = 120
    func = sin2

    network = RBFNetwork(n_inputs=1, n_rbf=4, n_outputs=1, n_epochs=100,
                         learning_rate_start=0.1, learning_rate_end=0.1,
                         rbf_var=0.5, cl_learning_rate=0.01, cl_leak_rate = 0.0001,
                         min_val=-2, max_val=9)

    train_patterns, train_targets, test_patterns, test_targets = get_ballistic_data()
    # network.fit(train_patterns, train_targets)

    x = test_patterns[:, 0]
    y = test_patterns[:, 1]
    z = test_targets[:, 1]


    grid_x, grid_y = np.mgrid[0:1:0.001, 0:1:0.001]
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid_z = interpolate.griddata((x, y), z, (grid_x, grid_y), method='linear')
    print(grid_z)

    CS = plt.contour(grid_x, grid_y, grid_z, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(grid_x, grid_y, grid_z, 15,
                      vmax=abs(grid_z).max(), vmin=-abs(grid_z).max())

    plt.colorbar()  # draw colorbar
    plt.show()


def test_cl():

    n_rbf = 4
    n_train = 120
    n_test = 120
    func = sin2

    network = RBFNetwork(n_inputs=1, n_rbf=n_rbf, n_outputs=1, n_epochs=600,
                         learning_rate_start=0.1, learning_rate_end=0.1,
                         rbf_var=0.5, cl_learning_rate=0.01, cl_leak_rate = 0.0001,
                         min_val=-2, max_val=9)
    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=0.01)



    # Plot initials
    px, py = zip(*sorted(zip(test_patterns, test_targets)))
    plt.plot(px, py, '--x', markersize=2, label='Test targets')
    plt.plot(network.rbf_centers.flatten(), np.zeros(n_rbf), 'o', color='grey', label='Initial centers', fillstyle='none')

    # Train network, make predictions
    network.fit(train_patterns, train_targets, method='sequential', cl_method='leaky')
    test_preds = network.predict(test_patterns)

    # Plot trained
    px, py = zip(*sorted(zip(test_patterns, test_preds)))
    plt.plot(px, py, '-x', markersize=2, label='Test predictions')
    plt.plot(network.rbf_centers.flatten(), np.zeros(n_rbf), 'o', color='grey', label='Trained centers')

    plt.legend()
    plt.show()


# test_cl()
ballistic_plot()
