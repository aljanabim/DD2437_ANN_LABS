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


def contains_duplicates(X):
    return len(np.unique(X)) != len(X)


def plot_travel_path():
    cities = cities_data()
    n_inputs = cities.shape[1]
    n_nodes = 10
    repeat = True
    n_epochs = 200
    attempts = 1
    seed = False  # also works 2
    # Make sure no city visits itself and no city is visited more than onces
    while repeat:
        if n_epochs < 1 and not seed:
            attempts += 1
            n_epochs = 200
        net = SOMNetwork(n_inputs=n_inputs,
                         n_nodes=n_nodes,
                         step_size=0.1,
                         topology='circular',
                         neighbourhood_start=2,
                         neighbourhood_end=1,
                         n_epochs=n_epochs,
                         seed=seed)
        result = net.fit(cities)
        holder = np.arange(cities.shape[0])
        n_epochs -= 1
        print("Attempt #" + str(attempts) + " Epoch "+str(n_epochs))
        if not (result == holder).any() and not contains_duplicates(result):
            repeat = False

    travel_order = {start: end for start,
                    end in enumerate(result)}

    print(travel_order)
    for i in range(cities.shape[0]):
        plt.scatter(cities[i, 0], cities[i, 1], linewidths=35)
    for start, end in travel_order.items():
        start = int(start)
        end = int(end)
        # print(start, end)
        plt.arrow(cities[start, 0], cities[start, 1],
                  cities[end, 0]-cities[start, 0], cities[end, 1]-cities[start, 1], length_includes_head=True,
                  head_width=0.035, head_length=0.03, linewidth=4)
    plt.show()


plot_travel_path()
