from helpers import load_data, show_image
from hopfield_net import HopfieldNetwork
from hopfield import HopfieldNet
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


def show_all_images(images, size):
    print("There are {} images".format(images.shape[0]))
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


def train_model(data):
    network = HopfieldNetwork(max_iter=150)
    # network = HopfieldNet(max_iter=150)

    network.fit(data[:3])
    iterations, result = network.predict(data[[1, 10], :], method='sequential')
    # result = network.predict(data[9, :], method='sequential')

    show_all_images(np.vstack((data[[1, 10], :], result)), (2, 2))
    # print(iterations)
    # print(result == data[0, :])


if __name__ == "__main__":
    images = load_data()
    show_all_images(images, (3, 4))
    train_model(images)
