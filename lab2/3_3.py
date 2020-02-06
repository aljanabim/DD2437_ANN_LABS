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

    n_train = 120
    n_test = 120
    func = sin2
    n_rbf_x = 8
    n_rbf_y = 8

    n_rbf = n_rbf_x*n_rbf_y
    rbf_layout = [(0, 1, n_rbf_x),
                  (0, 1, n_rbf_y)]

    train_patterns, train_targets, test_patterns, test_targets = get_ballistic_data()

    network = RBFNetwork(n_inputs=2, n_rbf=n_rbf, n_outputs=2, n_epochs=1000,
                         learning_rate_start=0.01, learning_rate_end=0.01,
                         rbf_var=0.1, cl_learning_rate=0.01, cl_leak_rate = 0.0001,
                         centering='linspace2d', rbf_layout=rbf_layout,
                         validation_patterns=test_patterns, validation_targets=test_targets)


    train_patterns = train_patterns[:]
    train_targets = train_targets[:]
    test_patterns = test_patterns[:]
    test_targets = test_targets[:]

    network.fit(train_patterns, train_targets, method='sequential', cl_method='basic')
    train_preds = network.predict(train_patterns)

    # plt.plot(network.mse_record, label='Training MSE')
    # plt.plot(network.validation_mse_record, label='Validation MSE')
    # plt.legend()
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_patterns[:,0], train_patterns[:,1], train_targets[:,0])
    # ax.scatter(train_patterns[:,0], train_patterns[:,1], train_preds[:,0])
    # ax.scatter(network.rbf_centers[:,0], network.rbf_centers[:,1], np.zeros(network.rbf_centers[:,1].shape), color='black')
    # plt.show()


    x = train_patterns[:,0]
    y = train_patterns[:,1]
    z = train_targets[:,0]

    xi = np.linspace(-0.1, 1.1, 100)
    yi = np.linspace(-0.1, 1.1, 100)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.magma)
    plt.xlabel("Velocity")
    plt.ylabel("Angle")
    plt.title("Height as function of velocity and angle")
    cb = plt.colorbar()
    cb.set_label("Height", rotation=270, labelpad=12)

    plt.plot(network.rbf_centers[:,0], network.rbf_centers[:,1], 'o', markersize=8, color='white', markeredgewidth=1,
             markeredgecolor='black', label='RBF Centers')
    plt.legend()
    plt.show()


    x = train_patterns[:,0]
    y = train_patterns[:,1]
    z = train_preds[:,0]

    xi = np.linspace(-0.1, 1.1, 100)
    yi = np.linspace(-0.1, 1.1, 100)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.magma)
    plt.xlabel("Velocity")
    plt.ylabel("Angle")
    plt.title("Height as function of velocity and angle")
    cb = plt.colorbar()
    cb.set_label("Height", rotation=270, labelpad=12)

    plt.plot(network.rbf_centers[:,0], network.rbf_centers[:,1], 'o', markersize=8, color='white', markeredgewidth=1,
             markeredgecolor='black', label='RBF Centers')
    plt.legend()
    plt.show()


    # plt.plot(train_patterns[:,1], train_preds[:,0], 'o', label="Prediction")
    # plt.plot(train_patterns[:,1], train_targets[:,0], 'o', label="Target")
    # plt.legend()
    # plt.show()
    #
    #
    # test_preds = network.predict(test_patterns)
    # plt.plot(test_patterns[:,0], test_preds[:,0], 'o', label="Prediction")
    # plt.plot(test_patterns[:,0], test_targets[:,0], 'o', label="Target")
    # plt.legend()
    # plt.show()


    # x = test_patterns[:, 0]
    # y = test_patterns[:, 1]
    # z = test_targets[:, 1]
    #
    #
    # grid_x, grid_y = np.mgrid[0:1:0.001, 0:1:0.001]
    # grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    # grid_z = interpolate.griddata((x, y), z, (grid_x, grid_y), method='nearest')
    # print(grid_z)
    #
    # CS = plt.contour(grid_x, grid_y, grid_z, 15, linewidths=0.5, colors='k')
    # CS = plt.contourf(grid_x, grid_y, grid_z, 15,
    #                   vmax=abs(grid_z).max(), vmin=-abs(grid_z).max())
    #
    # plt.colorbar()  # draw colorbar
    # plt.show()


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
