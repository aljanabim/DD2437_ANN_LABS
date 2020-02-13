import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet

# ============= HELPER FUNCTIONS ===============


def load_data():
    with open('./data/pict.dat', 'r') as f:
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
    n_total = len(patterns.flatten())
    n_correct = np.sum(patterns.flatten() == preds.flatten())
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
    base_images = images[0:3]
    degraded_image = images[9]

    net = HopfieldNet(min_iter=1, max_iter=2)
    net.fit(base_images[0])
    recovered_image = net.predict(degraded_image, method='batch')
    accuracy = calc_element_accuracy(base_images[0], recovered_image)

    print("Accuracy on degraded data: {}".format(accuracy))


    plt.subplot(131)
    show_image(base_images[0])
    plt.subplot(132)
    show_image(degraded_image)
    plt.subplot(133)
    show_image(recovered_image)
    plt.show()


def test_sequential_updates():
    images = load_data()
    base_images = images[0:3]
    degraded_image = images[9]

    net = HopfieldNet(min_iter=1, max_iter=2)
    net.fit(base_images[0])
    recovered_image = net.predict(degraded_image, method='sequential')
    accuracy = calc_element_accuracy(base_images[0], recovered_image)

    snapshots = net.sequential_learning_snapshots
    print(len(snapshots))
    n_subplot_cols = 8
    n_subplot_rows = len(snapshots)//n_subplot_cols + 1
    for i, img in enumerate(snapshots):
        plt.subplot(n_subplot_rows, n_subplot_cols, i+1)
        plt.title('Iteration {}'.format((i+1)*100))
        show_image(snapshots[i])
    plt.tight_layout(pad=3)
    plt.show()

    print("Accuracy on degraded data: {}".format(accuracy))



def test_show_image():
    images = load_data()
    show_image(images[10])


if __name__ == '__main__':
    # stability_check()
    # test_image_recovery()
    test_sequential_updates()
