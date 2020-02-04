import sys
import numpy as np
from matplotlib import pyplot as plt
from som_net import SOMNetwork
import re
# np.set_printoptions(threshold=sys.maxsize)
# plt.style.use('ggplot')
plt.grid(False)
plt.axis('off')


def cities_data():
    f_data = open('./data/cities.dat', 'r').readlines()[3:]
    cities = np.array([re.split(', |;', i)[0:2]
                       for i in f_data]).astype('float64')
    return cities


def plot_travel_path():
    cities = cities_data()
    n_inputs = cities.shape[1]
    n_nodes = 10
    net = SOMNetwork(n_inputs=n_inputs, n_nodes=n_nodes, topology='circular')
    result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    travel_order = {start: end for start,
                    end in enumerate(result)}  # net.fit(cities)
    print(travel_order)
    for i in range(cities.shape[0]):
        plt.scatter(cities[i, 0], cities[i, 1], linewidths=35)
    for start, end in travel_order.items():
        # print(cities[i, 0])
        print(start, end)
        plt.arrow(cities[start, 0], cities[start, 1],
                  cities[end, 0]-cities[start, 0], cities[end, 1]-cities[start, 1], length_includes_head=True,
                  head_width=0.035, head_length=0.03, linewidth=4)
    plt.show()


plot_travel_path()
