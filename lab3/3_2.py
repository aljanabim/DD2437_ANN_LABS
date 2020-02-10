import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet

# ============= HELPER FUNCTIONS ===============

def load_data():
    with  open('lab3/data/pict.dat', 'r') as f:
        text = str(f.read())
        value_list = np.array([int(val) for val in text.split(',')])
        images = []
        for n in range(11):
            start_index = 1024*n
            end_index = 1024*(n+1)
            images.append(value_list[start_index:end_index])
        return np.array(images)


def show_image(image):
        image = np.reshape(image, (32, 32)).T
        plt.imshow(image)


def calc_element_accuracy(patterns, preds):
    if len(patterns.shape) == 1:
        patterns = np.reshape(patterns, (-1, 1))

    n_total = patterns.shape[0] * patterns.shape[1]
    n_correct = np.sum(patterns == preds)
    return n_correct / n_total


def calc_sample_accuracy(patterns, preds):
    n_total = patterns.shape[0]
    n_correct = 0
    for pattern, pred in zip(patterns, preds):
        if (pattern == pred).all():
            n_correct += 1
    return n_correct / n_total


# ============= TESTING AND PLOTTING ===============


def stability_check():
    """Check that the three patterns are stable."""
    images = load_data()
    base_images = images[:3]

    net = HopfieldNet(max_iter=100)
    net.fit(base_images)
    preds = net.predict(base_images)
    accuracy = calc_element_accuracy(base_images, preds)
    print("Accuracy on training data: ".format(accuracy))


def test_image_recovery():
    images = load_data()
    base_image = images[0]
    degraded_image = images[9]

    net = HopfieldNet(max_iter=1)
    net.fit(base_image)
    recovered_image = net.predict(degraded_image)
    accuracy = calc_element_accuracy(base_image, recovered_image)

    plt.subplot(131)
    show_image(base_image)
    plt.subplot(132)
    show_image(degraded_image)
    plt.subplot(133)
    show_image(recovered_image)
    plt.show()

    print("Accuracy on degraded data: ".format(accuracy))











def test_show_image():
    images = load_data()
    show_image(images[10])


if __name__ == '__main__':
    stability_check()
    test_image_recovery()
