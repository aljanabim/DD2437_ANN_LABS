import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet


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
    image = np.reshape(image, (32, 32))
    plt.imshow(image)
    plt.show()


def main():
    images = load_data()
    show_image(images[9])


if __name__ == '__main__':
    main()
