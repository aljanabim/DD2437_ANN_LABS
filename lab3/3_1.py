import numpy as np
from hopfield import HopfieldNet

# ======= HELPER FUNCTIONS ===========

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


def calc_element_accuracy(patterns, preds):
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
    print('Element accuracy: {}'.format(calc_element_accuracy(clean_patterns, noisy_preds)))
    print('Sample accuracy: {}'.format(calc_sample_accuracy(clean_patterns, noisy_preds)))


def find_attractors():
    """How many attractors are there in this network?
       Hint: automate the searching."""

    # TODO: Implement
    pass


def test_more_noisy():
    """What happens when you make the starting pattern even more dissimilar
       to the stored ones (e.g. more than half is wrong)?"""

    clean_patterns = get_clean_data()
    noisy_patterns = np.array([[1,  1, -1,  1,  1, -1, -1,  1],
                               [1,  1,  1,  1, -1,  1, -1, -1],
                               [1, -1, -1,  1, -1,  1, -1,  1]])

    net = HopfieldNet()
    net.fit(clean_patterns)
    noisy_preds = net.predict(noisy_patterns)

    print(noisy_preds == clean_patterns)
    print('Element accuracy: {}'.format(calc_element_accuracy(clean_patterns, noisy_preds)))
    print('Sample accuracy: {}'.format(calc_sample_accuracy(clean_patterns, noisy_preds)))



if __name__ == '__main__':
    test_noise_reduction()
    test_more_noisy()
