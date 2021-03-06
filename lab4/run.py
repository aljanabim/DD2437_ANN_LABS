from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import time

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=600, n_test=100)

    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine..")
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
                                     )

    rbm.cd1(visible_trainset=train_imgs, n_iterations=20)

    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=15
                        )

    ''' greedy layer-wise training '''
    train_start_time = time.time()
    dbn.train_greedylayerwise(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=2000)
    train_end_time = time.time()
    print("Train time: {}s".format(train_end_time - train_start_time))

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    generate_start_time = time.time()
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    generate_end_time = time.time()
    print("Generate time: {}s".format(generate_end_time - generate_start_time))

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(
    #     vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    # dbn.recognize(train_imgs, train_lbls)

    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
