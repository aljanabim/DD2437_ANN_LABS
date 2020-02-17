from hopfield import HopfieldNet
import helpers as hl
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def test_total_distortion_resistance():
    n_trials = 10
    n_fracs = 100
    max_iter = 5

    images = hl.load_data()
    train_imgs = images[:3]

    net = HopfieldNet(min_iter=1, max_iter=max_iter)
    net.fit(train_imgs)

    noise_fracs = np.linspace(0, 1, n_fracs)
    accuracy_array = np.zeros((n_trials, n_fracs))
    for trial in range(n_trials):
        accuracy_list = []
        for frac in noise_fracs:
            noisy_imgs = [hl.add_image_noise(img, flip_fraction=frac) for img in images]
            pred_imgs = net.predict(noisy_imgs, method='batch')
            accuracy = hl.calc_sample_accuracy(train_imgs, pred_imgs)
            accuracy_list.append(accuracy)
        accuracy_array[trial, :] = np.array(accuracy_list)

    accuracy_means = np.mean(accuracy_array, axis=0)
    accuracy_sems = np.std(accuracy_array, axis=0) / np.sqrt(accuracy_array.shape[1])

    plt.errorbar(noise_fracs, accuracy_means, yerr=accuracy_sems, fmt='-o', capsize=2, label="All images")
    plt.title("Total distortion resistance")
    plt.xlabel("Flip fraction")
    plt.ylabel("Recovery rate")
    plt.legend()
    plt.show()


def test_individual_distortion_resistance():
    n_trials = 10
    n_fracs = 100
    max_iter = 5

    images = hl.load_data()
    train_imgs = images[:3]
    # plot_formats = ['-o', '--s', ':^']

    net = HopfieldNet(min_iter=1, max_iter=max_iter)
    net.fit(train_imgs)

    noise_fracs = np.linspace(0, 1, n_fracs)
    accuracy_array = np.zeros((n_trials, n_fracs))

    for i, train_img in enumerate(train_imgs):
        for trial in range(n_trials):
            accuracy_list = []
            for frac in noise_fracs:
                noisy_imgs = [hl.add_image_noise(img, flip_fraction=frac) for img in images]
                pred_imgs = net.predict(noisy_imgs, method='batch')
                accuracy = hl.calc_sample_accuracy(train_imgs, pred_imgs)
                accuracy_list.append(accuracy)
            accuracy_array[trial, :] = np.array(accuracy_list)

        accuracy_means = np.mean(accuracy_array, axis=0)
        accuracy_sems = np.std(accuracy_array, axis=0) / np.sqrt(accuracy_array.shape[1])
        plt.plot(noise_fracs, accuracy_means, label="Image p{}".format(i+1))

    plt.title("Individiual image distortion resistance")
    plt.xlabel("Flip fraction")
    plt.ylabel("Recovery rate")
    plt.legend()
    plt.show()


def look_for_other_attractors():
    max_iter = 1
    noise_fracs = [0.45, 0.49, 0.5, 0.5, 0.51, 0.55]
    n_examples = 6

    images = hl.load_data()
    train_imgs = images[:3]

    net = HopfieldNet(min_iter=1, max_iter=max_iter)
    net.fit(train_imgs)

    for i in range(n_examples):
        test_img = images[0]
        noisy_img = hl.add_image_noise(test_img, flip_fraction=noise_fracs[i])
        pred_img = net.predict(noisy_img, method='batch')
        plt.subplot(2, n_examples, i+1)
        hl.show_image(noisy_img)
        plt.subplot(2, n_examples, i+n_examples+1)
        hl.show_image(pred_img)

    plt.show()



if __name__ == '__main__':
    test_total_distortion_resistance()
    test_individual_distortion_resistance()
    # look_for_other_attractors()
