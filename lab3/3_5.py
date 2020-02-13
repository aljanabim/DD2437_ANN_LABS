import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNet
import helpers as hl



def test_storage_capacity():
    """How many patterns could safely be stored? Was the drop in performance
       gradual or abrupt?"""
    max_n_images = 7

    images = hl.load_data()


    noisy_image = hl.add_image_noise(images[0], 0.1)
    # hl.show_image(noisy_image)
    # plt.show()


    net = HopfieldNet(min_iter=1, max_iter=5)
    for n_images in range(1, max_n_images+1):
        selected_images = images[:n_images]
        net.fit(selected_images)

        preds = []
        for image in selected_images:
            preds.append(net.predict(image))
        preds = np.array(preds)

        accuracy = hl.calc_sample_accuracy(preds, selected_images)
        print(accuracy)



if __name__ == '__main__':
    test_storage_capacity()
