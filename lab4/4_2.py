from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

'''
Results:
Batch_size 15 | iterations 800 | n_tran 600 | n_test 100 | Accu train 92.17 | Accu test 84
Batch_size 15 | iterations 50 | n_tran 6000 | n_test 1000 | Accu train 73.30 | Accu test 70.9

'''

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    print(train_lbls)

    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")



    ''' greedy layer-wise training '''
    train_accuracy_list = []
    test_accuracy_list = []
    # iter_values = [20, 100, 400, 800, 1200]
    iter_values = [2000]
    for n_iter in iter_values:
        dbn = DeepBeliefNet(sizes={"vis": image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                            image_size=image_size,
                            n_labels=10,
                            batch_size=15)

        train_start_time = time.time()
        dbn.train_greedylayerwise(vis_trainset=train_imgs,
                                  lbl_trainset=train_lbls, n_iterations=n_iter, make_plots=False)
        train_end_time = time.time()
        print("Train time: {}s".format(train_end_time - train_start_time))

        train_accuracy = dbn.recognize(train_imgs, train_lbls)
        test_accuracy = dbn.recognize(test_imgs, test_lbls)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    print(train_accuracy_list)
    print(test_accuracy_list)
    plt.plot(iter_values, train_accuracy_list, label="Train accuracy")
    plt.plot(iter_values, test_accuracy_list, label="Test accuracy")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


    generate_start_time = time.time()
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    generate_end_time = time.time()
    print("Generate time: {}s".format(generate_end_time - generate_start_time))
