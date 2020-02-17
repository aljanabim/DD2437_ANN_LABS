from helpers import load_data, show_image, calc_sample_accuracy
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


def generate_sparse_patterns(n_data, n_patterns, sparsity):
    patterns = np.zeros((n_patterns, n_data))
    patterns[:, 0:int(round(sparsity*n_data))] = 1
    for row in patterns:
        np.random.shuffle(row)
    return patterns


def study_sparsity():
    n_patterns = 250
    n_data = 100
    sparsity = 0.5
    patterns = generate_sparse_patterns(n_data, n_patterns, sparsity)
    number_of_patterns = np.arange(1, n_patterns)

    for bias in [0, 0.5, 1, 5, 10]:
        print("Going through", bias)
        accuracy = []
        network = HopfieldNetwork(max_iter=10, bias=bias, sparse=True)
        for i in range(n_patterns-1):
            i = i+1
            network.fit(patterns[0:i], sparsity=sparsity)
            result = network.predict(patterns[0:i])
            acc = calc_sample_accuracy(patterns[0:i], result)
            accuracy.append(acc)
        plt.plot(number_of_patterns, accuracy, label=r"$\Theta=$"+str(bias))
        plt.title("Storage of sparse patterns, " +
                  str(int(sparsity*100))+"% sparsity")
        plt.xlabel("Patterns learned")
        plt.ylabel("Accuracy")
        plt.legend()
    plt.show()


if __name__ == "__main__":
    study_sparsity()
