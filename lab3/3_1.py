import numpy as np
from hopfield import HopfieldNet

# ======= DATA HANDLING ===========

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


# ======= TESTING AND PLOTTING ===========


def test_noise_reduction():
    """Apply the update rule repeatedly until you reach a stable xed point. Did
       all the patterns converge towards stored patterns?"""

    clean_patterns = get_clean_data()
    noisy_patterns = get_noisy_data()

    net = HopfieldNet()
    net.fit(clean_patterns)
    noisy_preds = net.predict(noisy_patterns)

    print(noisy_preds == clean_patterns)


def find_attractors():
    """How many attractors are there in this network?
       Hint: automate the searching."""

    # TODO: Implement
    pass


def test_more_noisy():
    """What happens when you make the starting pattern even more dissimilar
       to the stored ones (e.g. more than half is wrong)?"""
       
    # TODO: Implement
    pass



if __name__ == '__main__':
    test_noise_reduction()
