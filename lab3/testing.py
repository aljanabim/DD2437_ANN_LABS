import numpy as np
from hopfield_net import HopfieldNetwork

data = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                 [-1, -1, -1, -1, -1, 1, -1, -1],
                 [-1, 1, 1, -1, -1, 1, -1, 1]])

data_distorted = np.array([[1, - 1, 1, - 1, 1, - 1, - 1, 1],
                           [1, 1, - 1, - 1, - 1, 1, - 1, - 1],
                           [1, 1, 1, - 1, 1, 1, - 1, 1],
                           [-1, -1, 1, -1, -1, 1, -1, 1]])
# print(data_distorted)
network = HopfieldNetwork()
network.fit(data)
# print(data[2, :])
print(data_distorted[3, :])
iterations, result = network.predict(data_distorted[3, :], method="batch")
print(iterations)
print(result)
