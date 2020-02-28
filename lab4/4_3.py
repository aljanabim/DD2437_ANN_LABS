from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import time

'''
Results:
Batch_size 15 | iterations 2 | n_tran 6000 | n_test 1000  | Accu train 83.55% | Accu test 81.60%
Batch_size 15 | iterations 2 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 2 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 2 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 2 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %

Batch_size 15 | iterations 5 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 5 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 5 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 5 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %
Batch_size 15 | iterations 5 | n_tran 6000 | n_test 1000  | Accu train % | Accu test %

Batch_size 15 | iterations 10 | n_tran 6000 | n_test 1000  | Accu train 84.60% | Accu test 81.10%
Batch_size 15 | iterations 10 | n_tran 6000 | n_test 1000  | Accu train 84.60% | Accu test 81.10%
Batch_size 15 | iterations 10 | n_tran 6000 | n_test 1000  | Accu train 84.60% | Accu test 81.10%
Batch_size 15 | iterations 10 | n_tran 6000 | n_test 1000  | Accu train 84.60% | Accu test 81.10%
Batch_size 15 | iterations 10 | n_tran 6000 | n_test 1000  | Accu train 84.60% | Accu test 81.10%
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
                        batch_size=15
                        )

    ''' fine-tune wake-sleep training '''
    train_start_time = time.time()
    dbn.train_greedylayerwise(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=800)
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=2)
    train_end_time = time.time()
    print("Train time: {}s".format(train_end_time - train_start_time))

    # dbn.recognize(train_imgs, train_lbls)

    # dbn.recognize(test_imgs, test_lbls)

    generate_start_time = time.time()
    all_last = []
    for digit in range(10):
        print("Generating number",digit)
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        all_last.append(dbn.generate(digit_1hot, name="dbn"))
    
    plt.close('all')
    # all_last=[np.random.randint(0,2,(5,5)) for i in range(10)]
    for i, img in enumerate(all_last):
        plt.subplot(2,5,i+1)
        plt.imshow(img, cmap='Greys', interpolation=None)
        plt.xticks([]);
        plt.yticks([]);
        plt.xlabel(i)
    plt.savefig('plots/ep_5_lr_0_01_.pdf')
    generate_end_time = time.time()
    print("Generate time: {}s".format(generate_end_time - generate_start_time))