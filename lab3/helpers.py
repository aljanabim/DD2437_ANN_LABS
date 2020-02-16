import numpy as np
import matplotlib.pyplot as plt


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


def add_image_noise(image, flip_fraction):
    """Takes image and return copy with specified fraction of pixels flipped."""
    image_shape = image.shape
    new_image = image.copy().flatten()
    n_pixels = len(new_image)
    n_noise_pixels = int(flip_fraction*n_pixels)
    mask = np.zeros(new_image.shape).astype(np.bool)
    mask[:n_noise_pixels] = True
    np.random.shuffle(mask)
    new_image[mask] *= -1
    new_image = np.reshape(new_image, image_shape)
    return new_image


def random_image_pattern(shape, negative_fraction=0.5):
    return add_image_noise(np.ones(shape), negative_fraction)


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
