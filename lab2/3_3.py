import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
# from rbf_net import RBFNetwork
from rbf_net_2d import RBFNetwork
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
    train_data = np.loadtxt('lab2/data/ballist.dat')
    test_data = np.loadtxt('lab2/data/balltest.dat')

    train_patterns, train_targets = train_data[:, :2], train_data[:, 2:]
    test_patterns, test_targets = test_data[:, :2], test_data[:, 2:]

    return train_patterns, train_targets, test_patterns, test_targets


def ballistic():

    # Hyperparams
    n_train = 120
    n_test = 120
    func = sin2
    n_rbf_x = 5
    n_rbf_y = 5

    # Get data
    train_patterns, train_targets, test_patterns, test_targets = get_ballistic_data()
    train_patterns = train_patterns[:]
    train_targets = train_targets[:]
    test_patterns = test_patterns[:]
    test_targets = test_targets[:]

    # Set up rbf layout
    n_rbf = n_rbf_x*n_rbf_y
    rbf_layout = [(0, 1, n_rbf_x),
                  (0, 1, n_rbf_y)]

    # Init model
    network = RBFNetwork(n_inputs=2, n_rbf=n_rbf, n_outputs=2, n_epochs=5000,
                         learning_rate_start=0.01, learning_rate_end=0.01,
                         rbf_var=0.2, cl_learning_rate=0.01, cl_leak_rate = 0.000001,
                         centering='linspace2d', rbf_layout=rbf_layout,
                         validation_patterns=test_patterns, validation_targets=test_targets)

    # Plotting prep
    x = train_patterns[:,0]
    y = train_patterns[:,1]
    z = train_targets[:,0]
    xi = np.linspace(-0.1, 1.1, 100)
    yi = np.linspace(-0.1, 1.1, 100)

    # Plot target
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.magma)
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.title("Target distance")
    cb = plt.colorbar()
    cb.set_label("Distance", rotation=270, labelpad=12)
    plt.plot(network.rbf_centers[:,0], network.rbf_centers[:,1], 'o', markersize=8, color='white', markeredgewidth=1,
             markeredgecolor='black', label='Initial RBF Centers')
    plt.legend()
    plt.show()

    # Train network
    network.fit(train_patterns, train_targets, method='sequential', cl_method='leaky')

    # Predict train data
    train_preds = network.predict(train_patterns)

    # Plot prediction
    z = train_preds[:,0]
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.magma)
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.title("Predicted distance")
    cb = plt.colorbar()
    cb.set_label("Distance", rotation=270, labelpad=12)
    plt.plot(network.rbf_centers[:,0], network.rbf_centers[:,1], 'o', markersize=8, color='white', markeredgewidth=1,
             markeredgecolor='black', label='Trained RBF Centers')
    plt.legend()
    plt.show()

    # Plot MSE plots
    plt.plot(network.mse_record, label='Training MSE')
    plt.plot(network.validation_mse_record, label='Validation MSE')
    plt.title("Convergence of MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()

    # Compare different rbf layouts
    # rbf_sides = [1, 2, 3, 4, 5, 6, 7, 8]
    rbf_sides = []
    for rbf_side in rbf_sides:
        n_rbf = n_rbf_x*n_rbf_y
        rbf_layout = [(0, 1, rbf_side),
                      (0, 1, rbf_side)]
        network = RBFNetwork(n_inputs=2, n_rbf=n_rbf, n_outputs=2, n_epochs=1000,
                             learning_rate_start=0.01, learning_rate_end=0.01,
                             rbf_var=0.1, cl_learning_rate=0.01, cl_leak_rate = 0.0001,
                             centering='linspace2d', rbf_layout=rbf_layout,
                             validation_patterns=test_patterns, validation_targets=test_targets)
        network.fit(train_patterns, train_targets, method='sequential', cl_method='leaky')
        plt.semilogy(network.validation_mse_record, label='RBF grid side size: {}'.format(rbf_side))
    plt.title("Comparison of MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()





def test_cl():

    n_rbf = 4
    n_train = 120
    n_test = 120
    func = sin2

    train_patterns, train_targets, test_patterns, test_targets = gen_func_data(
        n_train, n_test, func, noise_var=0.01)

    network = RBFNetwork(n_inputs=1, n_rbf=n_rbf, n_outputs=1, n_epochs=600,
                         learning_rate_start=0.1, learning_rate_end=0.1,
                         rbf_var=0.5, cl_learning_rate=0.01, cl_leak_rate = 1,
                         min_val=-2, max_val=9)




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
ballistic()
