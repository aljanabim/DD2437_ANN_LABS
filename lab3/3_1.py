import numpy as np
from hopfield import HopfieldNet


def get_clean_data():
    patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
                         [-1, -1, -1, -1, -1,  1, -1, -1],
                         [-1,  1,  1, -1, -1,  1, -1,  1]])
    return patterns


def get_noisy_data():
    patterns = np.array([[ 1, -1,  1, -1,  1, -1, -1,  1],
                         [ 1,  1, -1, -1, -1,  1, -1, -1],
                         [ 1,  1,  1, -1,  1,  1, -1,  1]])
    return patterns


def test_noise_reduction():
    clean_patterns = get_clean_data()
    noisy_patterns = get_noisy_data()

    net = HopfieldNet()
    net.fit(clean_patterns)
    noisy_preds = net.predict(noisy_patterns)

    print(noisy_preds == clean_patterns)



if __name__ == '__main__':
    test_noise_reduction()
