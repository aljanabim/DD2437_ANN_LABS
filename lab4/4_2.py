from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import time

'''
Results:
Batch_size 15 | iterations 800 | n_tran 600 | n_test 100 | Accu train 92.17 | Accu test 84
Batch_size 15 | iterations 50 | n_tran 6000 | n_test 1000 | Accu train 73.30 | Accu test 70.9
Batch_size 15 | iterations 100 | n_tran 6000 | n_test 1000 | Accu train 80.20 | Accu test 73.30
Batch_size 15 | iterations 300 | n_tran 6000 | n_test 1000 | Accu train 82.70 | Accu test 76.50
Batch_size 15 | iterations 500 | n_tran 6000 | n_test 1000 | Accu train 83.20 | Accu test 77.70
Batch_size 15 | iterations 800 | n_tran 6000 | n_test 1000 | Accu train 83.90 | Accu test 80.30
Batch_size 15 | iterations 2000 | n_tran 6000 | n_test 1000 | Accu train 83.20 | Accu test 76.90
'''

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=6000, n_test=1000)

    ''' deep- belief net '''

    print("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=15)

    ''' greedy layer-wise training '''
    train_start_time = time.time()
    dbn.train_greedylayerwise(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=800)
    train_end_time = time.time()
    print("Train time: {}s".format(train_end_time - train_start_time))

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    generate_start_time = time.time()
    all_last = []
    for digit in range(10):
        print("Generating number",digit)
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        all_last.append(dbn.generate(digit_1hot, name="rbms"))
    
    plt.close('all')
    # all_last=[np.random.randint(0,2,(5,5)) for i in range(10)]
    for i, img in enumerate(all_last):
        plt.subplot(2,5,i+1)
        plt.imshow(img, cmap='Greys', interpolation=None)
        plt.xticks([]);
        plt.yticks([]);
        plt.xlabel(i)
    plt.savefig('plots/gen_4_2_samples6000iter800.pdf')
    generate_end_time = time.time()
    print("Generate time: {}s".format(generate_end_time - generate_start_time))