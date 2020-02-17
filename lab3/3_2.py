import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet
import helpers as hl

# ============= HELPER FUNCTIONS ===============

# Moved to helpers.py

# ============= TESTING AND PLOTTING ===============


def stability_check():
    """Check that the three patterns are stable."""
    images = hl.load_data()
    base_images = images[:3]

    net = HopfieldNet(max_iter=100)
    net.fit(base_images)
    preds = net.predict(base_images)
    accuracy = hl.calc_element_accuracy(base_images, preds)
    print("Accuracy on training data: {}".format(accuracy))


def test_image_recovery():
    images = hl.load_data()
    base_images = images[0:3]
    degraded_image = images[9]

    net = HopfieldNet(min_iter=1, max_iter=2)
    net.fit(base_images[0])
    recovered_image = net.predict(degraded_image, method='batch')
    accuracy = hl.calc_element_accuracy(base_images[0], recovered_image)

    print("Accuracy on degraded data: {}".format(accuracy))

    plt.subplot(131)
    hl.show_image(base_images[0])
    plt.subplot(132)
    hl.show_image(degraded_image)
    plt.subplot(133)
    hl.show_image(recovered_image)
    plt.show()


def test_sequential_updates():
    images = hl.load_data()
    base_images = images[0:3]
    degraded_image = images[9]

    net = HopfieldNet(min_iter=1, max_iter=2)
    net.fit(base_images[0])
    recovered_image = net.predict(degraded_image, method='sequential')
    accuracy = hl.calc_element_accuracy(base_images[0], recovered_image)

    snapshots = net.sequential_learning_snapshots
    print(len(snapshots))
    n_subplot_cols = 8
    n_subplot_rows = len(snapshots)//n_subplot_cols + 1
    for i, img in enumerate(snapshots):
        plt.subplot(n_subplot_rows, n_subplot_cols, i+1)
        plt.title('Iteration {}'.format((i+1)*100))
        hl.show_image(snapshots[i])
    plt.show()

    print("Accuracy on degraded data: {}".format(accuracy))


def test_show_image():
    images = load_data()
    show_image(images[10])


if __name__ == '__main__':
    stability_check()
    # All stable
    test_image_recovery()
    # All recovered
    test_sequential_updates()
    # See plot
