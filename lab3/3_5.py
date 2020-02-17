import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet
import helpers as hl

plt.style.use('ggplot')


def benchmark_on_images(net, images, flip_fraction=0.2):
    image_accuracy_list = []
    max_n_images = len(images)
    for n_images in range(1, max_n_images+1):
        selected_images = images[:n_images]
        net.fit(selected_images)
        preds = []
        for image in selected_images:
            distorted_image = hl.add_image_noise(image, flip_fraction=flip_fraction)
            preds.append(net.predict(distorted_image, 'sequential'))
        preds = np.array(preds)

        image_accuracy = hl.calc_sample_accuracy(preds, selected_images)
        image_accuracy_list.append(image_accuracy)
    return image_accuracy_list


def test_storage_capacity():
    """How many patterns could safely be stored? Was the drop in performance
       gradual or abrupt?"""
    max_n_images = 7
    flip_fraction = 0.1


    net = HopfieldNet(min_iter=1, max_iter=5)

    images = hl.load_data()
    image_shape = images[0].shape

    image_accuracy_list = benchmark_on_images(net, images[0:max_n_images], flip_fraction=flip_fraction)

    """Try to repeat this with learning a few random patterns instead of the
       pictures and see if you can store more."""

    randoms =  [hl.random_image_pattern(image_shape, 0.5) for _ in range(max_n_images)]
    random_accuracy_list = benchmark_on_images(net, randoms, flip_fraction=flip_fraction)

    plt.plot(image_accuracy_list, label="Images")
    plt.plot(random_accuracy_list, label="Random patterns, unbiased")
    plt.xlabel("Number of stored images")
    plt.ylabel("Imagewise accuracy")
    plt.title("Storage capacity test, flip fraction = {}".format(flip_fraction))
    plt.legend()
    plt.show()

    """It has been shown that the capacity of a Hopeld network is around
       0:138N. How do you explain the difference between random patterns
       and the pictures? Answer: If randomized image are about equally
       positive and negative, the performance is good. However, if they
       are not even, then the performance quickly drops. The images
       have one dominant background color, and so suffer from this as well."""


def memory_limit_test():
    pattern_shape = (100,)
    max_n_patterns = 300
    force_zero_diagonal = True

    net = HopfieldNet(zero_diagonal=force_zero_diagonal, min_iter=1, max_iter=5)

    n_patterns_list = []
    n_stable_list = []
    n_noise_stable_list = []
    patterns = []
    for n_patterns in range(max_n_patterns):
        patterns.append(hl.random_image_pattern(pattern_shape, 0.3))
        net.fit(patterns)
        n_stable = 0
        n_noise_stable = 0
        for pattern in patterns:
            pred = net.predict(pattern)
            if (pred == pattern).all():
                n_stable += 1

            noisy_pattern = hl.add_image_noise(pattern, 0.05)
            noisy_pred = net.predict(noisy_pattern)
            if (noisy_pred == pattern).all():
                n_noise_stable += 1

        n_patterns_list.append(n_patterns+1)
        n_stable_list.append(n_stable)
        n_noise_stable_list.append(n_noise_stable)

    plt.plot(n_patterns_list, n_stable_list, label="Clean patterns")
    plt.plot(n_patterns_list, n_noise_stable_list, label="Noisy patterns (10 iterations)")
    plt.legend()
    if net.zero_diagonal:
        plt.title("Stability and convergence analysis with zero-diagonal\n(biased patterns)")
    else:
        plt.title("Stability and convergence analysis")
    plt.xlabel("Number of patterns")
    plt.ylabel("Correct convergence")
    plt.show()

    """Plot indicates that the network has many critical points, but these are very unstable."""





if __name__ == '__main__':
    # test_storage_capacity()
    memory_limit_test()
