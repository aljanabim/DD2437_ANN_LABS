from util import *
from rbm import RestrictedBoltzmannMachine
from dbn_small import DeepBeliefNet
import time
# plt.style.use('ggplot')
'''
Results:
EP 2 | Learning Rate 0.01 | Accu train 75.10% | Accu test 74.10%
EP 2 | Learning Rate 0.01 | Accu train 75.62% | Accu test 72.10%
EP 2 | Learning Rate 0.01 | Accu train 76.22% | Accu test 72.70%

EP 2 NORM | Learning Rate 0.001 | Accu train 85.02% | Accu test 81.60%
EP 2 NORM | Learning Rate 0.001 | Accu train 85.47% | Accu test 80.70%
EP 2 NORM | Learning Rate 0.001 | Accu train 83.72% | Accu test 80.50%

EP 5 | Learning Rate 0.01 | Accu train 55.55% | Accu test 49.90%
EP 5 | Learning Rate 0.01 | Accu train 55.37% | Accu test 52.50%
EP 5 | Learning Rate 0.01 | Accu train 54.28% | Accu test 54.60%

EP 5 NORM | Learning Rate 0.001 | Accu train 85.32% | Accu test 78.40%
EP 5 NORM | Learning Rate 0.001 | Accu train 83.97% | Accu test 79.90%
EP 5 NORM | Learning Rate 0.001 | Accu train 84.08% | Accu test 79.80%

EP 10 | Learning Rate 0.01 | Accu train 48.17% | Accu test 44.80%
EP 10 | Learning Rate 0.01 | Accu train 47.47% | Accu test 42.40%
EP 10 | Learning Rate 0.01 | Accu train 47.18% | Accu test 43.50%

EP 10 NORM | Learning Rate 0.001 | Accu train 84.13% | Accu test 80.50%
EP 10 NORM | Learning Rate 0.001 | Accu train 84.53% | Accu test 81.50%
EP 10 NORM | Learning Rate 0.001 | Accu train 85.40% | Accu test 77.60%

'''

def plot_results():
    '''
    Avg accuracy of three runs
    '''
    x = [2, 5, 10]

    EP_2_gen = np.mean([75.10, 75.62, 76.22])
    EP_5_gen = np.mean([55.55, 55.37, 54.28])
    EP_10_gen = np.mean([48.17, 47.47, 47.18])
    y_gen = [EP_2_gen, EP_5_gen, EP_10_gen]
    
    EP_2_gen_test = np.mean([74.10,72.10,72.70])
    EP_5_gen_test = np.mean([49.90,52.50,54.60])
    EP_10_gen_test = np.mean([44.80,42.40,43.50])
    y_gen_test = [EP_2_gen_test, EP_5_gen_test, EP_10_gen_test]

    EP_2_rec = np.mean([85.02,85.47,83.72])
    EP_5_rec = np.mean([85.32,83.97,84.08])
    EP_10_rec = np.mean([84.13,84.53,85.40])
    y_rec = [EP_2_rec, EP_5_rec, EP_10_rec]
    
    EP_2_rec_test = np.mean([81.60,80.70,80.50])
    EP_5_rec_test = np.mean([78.40,79.90,79.80])
    EP_10_rec_test = np.mean([80.50,81.50,77.60])
    y_rec_test = [EP_2_rec_test, EP_5_rec_test, EP_10_rec_test]

    plt.close('all')
    plt.plot(x, y_rec, label="Recognition specialised (training)")
    plt.plot(x, y_rec_test, label="Recognition specialised (testing)")
    plt.plot(x, y_gen, label="Generation specialised (training)")
    plt.plot(x, y_gen_test, label="Generation specialised (testing)")
    plt.xlabel('Epochs')
    plt.ylabel('Avg accuracy in %')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/4_3/compare_models.pdf')
    

if __name__ == "__main__":
    # plot_results()
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=600, n_test=100)

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
                              lbl_trainset=train_lbls, n_iterations=8)
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=10)
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
        all_last.append(dbn.generate(digit_1hot, name="dbn"))
    
    plt.close('all')
    # all_last=[np.random.randint(0,2,(5,5)) for i in range(10)]
    for i, img in enumerate(all_last):
        plt.subplot(2,5,i+1)
        plt.imshow(img, cmap='Greys', interpolation=None)
        plt.xticks([]);
        plt.yticks([]);
        plt.xlabel(i)
    plt.savefig('plots/small_gen_ep_10_lr_0_001.pdf')
    generate_end_time = time.time()
    print("Generate time: {}s".format(generate_end_time - generate_start_time))