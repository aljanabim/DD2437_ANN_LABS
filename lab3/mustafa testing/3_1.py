import numpy as np
from hopfield_net import HopfieldNetwork

data = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                 [-1, -1, -1, -1, -1, 1, -1, -1],
                 [-1, 1, 1, -1, -1, 1, -1, 1]])

data_distorted = np.array([[1, - 1, 1, - 1, 1, - 1, - 1, 1],
                           [1, 1, - 1, - 1, - 1, 1, - 1, - 1],
                           [1, 1, 1, - 1, 1, 1, - 1, 1]])

network = HopfieldNetwork()
network.fit(data)
print(data[2, :])
print(network.predict(data[2, :], method="sequential"))
