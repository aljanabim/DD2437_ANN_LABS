import sys
import numpy as np
from matplotlib import pyplot as plt
from som_net import SOMNetwork
np.set_printoptions(threshold=sys.maxsize)
# plt.style.use('ggplot')
plt.grid(False)
plt.axis('off')


def animal_data():
    f_props = open('./data/animals.dat', 'r').readlines()[0].split(',')
    f_names = open('./data/animalnames.txt').readlines()

    animal_props = np.array(f_props).reshape((32, 84)).astype('int')
    animal_names = np.array([animal.split("'")[1].capitalize()
                             for animal in f_names])

    return animal_props, animal_names


def plot_animal_clusters():
    animal_props, animal_names = animal_data()
    n_inputs = animal_props.shape[1]
    n_nodes = 100
    net = SOMNetwork(n_inputs=n_inputs, n_nodes=n_nodes, step_size=0.2,
                     topology='linear', neighbourhood_start=50, neighbourhood_end=1, n_epochs=120, seed=15)
    pos = net.fit(animal_props)

    animal_dict = dict()
    for i, p in enumerate(pos):
        animal_dict[p] = []
    for i, p in enumerate(pos):
        animal_dict[p].append(animal_names[i])

    y = np.zeros(len(pos))
    plt.scatter(pos, y)

    for index, key in enumerate(sorted(animal_dict.keys())):
        if index % 2 == 0:
            sign = -1
        else:
            sign = 1
        for i, name in enumerate(animal_dict[key]):
            # Hardcoded styling
            if index == 0 or index == 11 or index == 10:
                plt.annotate(name, (key, y[i]),
                             (key-0.25, sign*(0.00065*i+0.0013)))
            else:
                plt.annotate(name, (key, y[i]),
                             (key-0.25, sign*(0.00065*i+0.0007)))
    plt.show()


plot_animal_clusters()
