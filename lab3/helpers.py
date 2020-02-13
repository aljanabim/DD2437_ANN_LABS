import numpy as np
import matplotlib.pyplot as plt


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
