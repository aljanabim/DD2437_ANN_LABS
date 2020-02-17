from helpers import load_data, show_image
from hopfield_net import HopfieldNetwork
from hopfield import HopfieldNet
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
plt.style.use('ggplot')


def show_all_images(images, size):
    for i in range(images.shape[0]):
        sp = plt.subplot(size[0], size[1], i+1)
        sp.axis('off')
        if i == 0:
            sp.axis('on')
            plt.yticks([])
            plt.xticks([])
            plt.ylabel("Input pattern")
        elif i % size[1] == 0:
            sp.axis('on')
            plt.yticks([])
            plt.xticks([])
            plt.ylabel("Predicted pattern")
        show_image(images[i, :])
    plt.show()


def study_energy(data):
    network = HopfieldNetwork(max_iter=10)
    network.fit(data[:3], zero_diag=True)
    results = []
    for i in [8, 9, 10]:
        result = network.predict(
            data[i], method='sequential', calc_energy=True)
        results.append(result)
        energy = network.energy[1:, 1]
        print(energy[-1])
        plt.plot(energy, label="P"+str(i+1))
    energy_P1 = np.ones(len(energy))*(-1439.390625)
    energy_P2 = np.ones(len(energy))*(-1365.640625)
    energy_P3 = np.ones(len(energy))*(-1462.25)
    i = np.arange(0, len(energy_P1), 100)
    plt.plot(i, energy_P1[i], '_', label="P1")
    plt.plot(i, energy_P2[i], '_', label="P2")
    plt.plot(i, energy_P3[i], '_', label="P3")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

    show_all_images(np.vstack((data[[8, 9, 10], :], results)), (3, 3))
    """
    Sometimes P11 converges to the lower attractor meaning the randomness in the order
    makes a difference
    """


def study_energy_rnd_weights(data):
    network = HopfieldNetwork(max_iter=10)
    results = []
    for j in range(2):
        if j == 0:
            network.fit(data[:3], random_weights=True, random_symmetric=False)
        if j == 1:
            network.fit(data[:3], random_weights=True, random_symmetric=True)

        for i in [0, 1]:
            plt.subplot(1, 2, j+1)
            result = network.predict(
                data[i], method='sequential', calc_energy=True)
            results.append(result)
            energy = network.energy[1:, 1]
            print(energy[-1])
            plt.plot(energy, label="P"+str(i+1))
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        if j == 0:
            plt.title("Random weights")
        if j == 1:
            plt.title("Random symmetric weights")
        plt.legend()
    plt.show()

    show_all_images(
        np.vstack((data[[0, 1], :], results)), (3, 2))


if __name__ == "__main__":
    images = load_data()
    # show_all_images(images, (3, 4))
    print("There are {} images".format(images.shape[0]))

    # study_energy(images)
    study_energy_rnd_weights(images)
